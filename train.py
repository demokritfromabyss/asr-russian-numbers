"""
Training script for the Russian spoken-numbers Conformer-CTC model.

Usage:
    python train.py --config config.yaml [--resume checkpoints/last.pt]
"""

import argparse
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.text.vocabulary import Vocabulary
from src.data.dataset import AudioDataset, collate_fn
from src.model.ctc_model import build_model
from src.utils.decoder import batch_decode
from src.utils.metrics import compute_cer_batch
from src.text.normalization import words_to_num


# ── helpers ──────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['global_step'], ckpt.get('best_cer', float('inf'))


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device, cfg, epoch, global_step=0):
    model.train()
    t_cfg = cfg['training']
    total_loss = 0.0
    log_interval = t_cfg.get('log_interval', 50)

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for step, (mels, mel_lengths, tokens, token_lengths, _) in enumerate(pbar):
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)

        log_probs, out_lengths = model(mels, mel_lengths)
        # log_probs: (T', B, vocab)
        loss = criterion(log_probs, tokens, out_lengths, token_lengths)

        # Entropy regularization (label smoothing equivalent for CTC)
        smooth_w = t_cfg.get('label_smoothing', 0.0)
        if smooth_w > 0:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
            loss = loss - smooth_w * entropy

        if torch.isnan(loss) or torch.isinf(loss):
            continue  # skip bad batches

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get('grad_clip', 1.0))
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_loss += loss.item()
        if (step + 1) % log_interval == 0:
            avg = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{avg:.4f}', 'lr': f'{lr:.2e}', 'step': global_step})

    return total_loss / max(len(loader), 1), global_step


@torch.no_grad()
def validate(model, loader, criterion, vocab, device, ind_spks, ood_spks):
    model.eval()
    total_loss = 0.0
    all_hyps, all_refs, all_spks = [], [], []

    for mels, mel_lengths, tokens, token_lengths, spk_ids in loader:
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)

        log_probs, out_lengths = model(mels, mel_lengths)
        loss = criterion(log_probs, tokens, out_lengths, token_lengths)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()

        hyps_words = batch_decode(log_probs, out_lengths, vocab)
        # decode reference tokens back to words for CER on word level
        for i, tok in enumerate(tokens):
            T = int(token_lengths[i].item())
            ref_words = vocab.decode(tok[:T].tolist())
            all_refs.append(ref_words)

        all_hyps.extend(hyps_words)
        all_spks.extend(spk_ids)

    stats = compute_cer_batch(all_hyps, all_refs, all_spks)
    per_spk = stats['per_speaker']

    ind_cers = [v for k, v in per_spk.items() if k in ind_spks]
    ood_cers = [v for k, v in per_spk.items() if k in ood_spks]

    ind_cer = sum(ind_cers) / len(ind_cers) if ind_cers else stats['overall']
    ood_cer = sum(ood_cers) / len(ood_cers) if ood_cers else stats['overall']
    hm_cer = 2 * ind_cer * ood_cer / max(ind_cer + ood_cer, 1e-9)

    return {
        'loss': total_loss / max(len(loader), 1),
        'overall_cer': stats['overall'],
        'ind_cer': ind_cer,
        'ood_cer': ood_cer,
        'hm_cer': hm_cer,
        'per_speaker': per_spk,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    t_cfg = cfg['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    vocab = Vocabulary()

    # Datasets
    train_ds = AudioDataset(
        cfg['data']['train_csv'], cfg['data']['train_audio_dir'], vocab, cfg, is_train=True
    )
    dev_ds = AudioDataset(
        cfg['data']['dev_csv'], cfg['data']['dev_audio_dir'], vocab, cfg, is_train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg['batch_size'], shuffle=True,
        num_workers=t_cfg.get('num_workers', 4), collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=t_cfg['batch_size'] // 2, shuffle=False,
        num_workers=t_cfg.get('num_workers', 4), collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(cfg, vocab.size).to(device)
    print(f'Parameters: {model.count_parameters():,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg['learning_rate'],
        weight_decay=t_cfg.get('weight_decay', 1e-6),
    )

    total_steps = t_cfg['max_epochs'] * len(train_loader)
    scheduler = make_scheduler(optimizer, t_cfg.get('warmup_steps', 5000), total_steps)

    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    start_epoch = 1
    global_step = 0
    best_cer = float('inf')
    ckpt_dir = t_cfg.get('checkpoint_dir', 'checkpoints/')

    if args.resume:
        start_epoch, global_step, best_cer = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f'Resumed from epoch {start_epoch}, best CER {best_cer:.4f}')

    # Train/dev speaker split (6 train speakers are inD, remaining in dev are ooD)
    train_spks = set(train_ds.df['spk_id'].unique())
    dev_spks = set(dev_ds.df['spk_id'].unique())
    ind_spks = train_spks & dev_spks
    ood_spks = dev_spks - train_spks

    print(f'inD speakers in dev: {sorted(ind_spks)}')
    print(f'ooD speakers in dev: {sorted(ood_spks)}')

    for epoch in range(start_epoch, t_cfg['max_epochs'] + 1):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg, epoch, global_step
        )

        torch.cuda.empty_cache()
        val = validate(model, dev_loader, criterion, vocab, device, ind_spks, ood_spks)

        print(
            f'Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val["loss"]:.4f} '
            f'| CER={val["overall_cer"]:.4f} | inD={val["ind_cer"]:.4f} '
            f'| ooD={val["ood_cer"]:.4f} | HM={val["hm_cer"]:.4f}'
        )

        state = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_cer': best_cer,
            'val': val,
        }
        save_checkpoint(state, os.path.join(ckpt_dir, 'last.pt'))

        # Save per-epoch checkpoint for later averaging (every 5 epochs after warmup)
        if epoch >= t_cfg.get('warmup_steps', 5000) // max(len(train_loader), 1) and epoch % 5 == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))

        if val['hm_cer'] < best_cer:
            best_cer = val['hm_cer']
            state['best_cer'] = best_cer
            save_checkpoint(state, os.path.join(ckpt_dir, 'best.pt'))
            print(f'  ✓ New best HM-CER: {best_cer:.4f}')


if __name__ == '__main__':
    main()
