"""
Run inference on the test set and produce a Kaggle submission CSV.

Usage:
    python inference.py \\
        --config config.yaml \\
        --checkpoint checkpoints/best.pt \\
        --test_csv data/test.csv \\
        --audio_dir data/ \\
        --output submission.csv
"""

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.text.vocabulary import Vocabulary
from src.data.dataset import TestDataset, collate_fn_test
from src.model.ctc_model import build_model
from src.utils.decoder import batch_decode
from src.text.normalization import words_to_num


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, vocab, device, beam_width: int = 1):
    model.eval()
    filenames_all, predictions_all = [], []

    for mels, mel_lengths, filenames in tqdm(loader, desc='Inference'):
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        log_probs, out_lengths = model(mels, mel_lengths)
        hyps_words = batch_decode(log_probs, out_lengths, vocab, beam_width=beam_width)

        for fname, words in zip(filenames, hyps_words):
            digit_str = words_to_num(words)
            filenames_all.append(fname)
            predictions_all.append(digit_str)

    return filenames_all, predictions_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--output', default='submission.csv')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary()

    test_ds = TestDataset(args.test_csv, args.audio_dir, cfg)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=2, collate_fn=collate_fn_test)

    model = build_model(cfg, vocab.size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint from {args.checkpoint}')

    beam_width = cfg.get('decoding', {}).get('beam_width', 10)
    print(f'Decoding with beam_width={beam_width}')
    filenames, predictions = run_inference(model, test_loader, vocab, device, beam_width=beam_width)

    submission = pd.DataFrame({'filename': filenames, 'transcription': predictions})
    submission.to_csv(args.output, index=False)
    print(f'Saved {len(submission)} predictions → {args.output}')


if __name__ == '__main__':
    main()
