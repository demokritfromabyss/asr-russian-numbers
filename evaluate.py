"""
Evaluate a trained model on the dev set, printing per-speaker CER.

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
import yaml

from src.text.vocabulary import Vocabulary
from src.data.dataset import AudioDataset, collate_fn
from src.model.ctc_model import build_model
from src.utils.decoder import batch_decode
from src.utils.metrics import compute_cer_batch, harmonic_mean_cer
from src.text.normalization import words_to_num


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, vocab, device, train_spks, beam_width: int = 1):
    model.eval()
    all_hyps, all_refs, all_spks = [], [], []

    for mels, mel_lengths, tokens, token_lengths, spk_ids in loader:
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        log_probs, out_lengths = model(mels, mel_lengths)
        hyps_words = batch_decode(log_probs, out_lengths, vocab, beam_width=beam_width)

        for i, tok in enumerate(tokens):
            T = int(token_lengths[i].item())
            ref_words = vocab.decode(tok[:T].tolist())
            all_refs.append(ref_words)

        all_hyps.extend(hyps_words)
        all_spks.extend(spk_ids)

    stats = compute_cer_batch(all_hyps, all_refs, all_spks)
    per_spk = stats['per_speaker']

    ind_spks = [s for s in per_spk if s in train_spks]
    ood_spks = [s for s in per_spk if s not in train_spks]

    ind_cer = sum(per_spk[s] for s in ind_spks) / len(ind_spks) if ind_spks else stats['overall']
    ood_cer = sum(per_spk[s] for s in ood_spks) / len(ood_spks) if ood_spks else stats['overall']
    hm = harmonic_mean_cer(ind_cer, ood_cer)

    print(f'\n{"Speaker":<12} {"CER":>8}  {"Type":>5}')
    print('-' * 32)
    for spk in sorted(per_spk):
        tag = 'inD' if spk in train_spks else 'ooD'
        print(f'{spk:<12} {per_spk[spk]:>8.4f}  {tag:>5}')
    print('-' * 32)
    print(f'{"Overall":<12} {stats["overall"]:>8.4f}')
    print(f'{"inD avg":<12} {ind_cer:>8.4f}')
    print(f'{"ooD avg":<12} {ood_cer:>8.4f}')
    print(f'{"HM-CER":<12} {hm:>8.4f}  ← primary metric')

    # Also report digit-level CER (after denormalization)
    hyp_nums = [words_to_num(h) for h in all_hyps]
    ref_nums = [words_to_num(r) for r in all_refs]
    digit_stats = compute_cer_batch(hyp_nums, ref_nums, all_spks)
    print(f'\nDigit-level CER (after denorm): {digit_stats["overall"]:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary()

    train_df = __import__('pandas').read_csv(cfg['data']['train_csv'])
    train_spks = set(train_df['spk_id'].unique())

    dev_ds = AudioDataset(cfg['data']['dev_csv'], cfg['data']['dev_audio_dir'], vocab, cfg, is_train=False)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    model = build_model(cfg, vocab.size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint (epoch {ckpt.get("epoch", "?")})')

    beam_width = cfg.get('decoding', {}).get('beam_width', 1)
    print(f'Decoding with beam_width={beam_width}')
    torch.cuda.empty_cache()
    evaluate(model, dev_loader, vocab, device, train_spks, beam_width=beam_width)


if __name__ == '__main__':
    main()
