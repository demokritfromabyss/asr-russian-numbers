"""
Data preparation and validation helper.

Usage:
    python prepare_data.py --csv data/train.csv --audio_dir data/
    python prepare_data.py --csv data/train.csv --audio_dir data/ --stats
"""

import argparse
import os
import sys
import pandas as pd


REQUIRED_COLS = {'filename', 'transcription'}


def validate(csv_path: str, audio_dir: str, verbose: bool = False) -> bool:
    if not os.path.exists(csv_path):
        print(f'ERROR: CSV not found: {csv_path}')
        return False

    df = pd.read_csv(csv_path)

    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        print(f'ERROR: Missing columns: {missing_cols}')
        print(f'       Found: {list(df.columns)}')
        return False

    # Check transcription range
    try:
        nums = df['transcription'].astype(int)
    except ValueError as e:
        print(f'ERROR: transcription column must be integers: {e}')
        return False

    out_of_range = nums[(nums < 1000) | (nums > 999999)]
    if not out_of_range.empty:
        print(f'WARNING: {len(out_of_range)} transcriptions outside [1000, 999999]')
        if verbose:
            print(out_of_range.to_string())

    # Check audio files
    missing = []
    for fname in df['filename']:
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)

    if missing:
        print(f'ERROR: {len(missing)} audio files not found (showing first 10):')
        for f in missing[:10]:
            print(f'  {os.path.join(audio_dir, f)}')
        return False

    print(f'OK  {csv_path}: {len(df)} samples, all audio files present')
    return True


def print_stats(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f'\n--- Stats: {csv_path} ---')
    print(f'  Rows          : {len(df)}')

    if 'spk_id' in df.columns:
        print(f'  Speakers      : {df["spk_id"].nunique()} — {sorted(df["spk_id"].unique())}')

    if 'transcription' in df.columns:
        nums = df['transcription'].astype(int)
        print(f'  Transcription : min={nums.min()}, max={nums.max()}, unique={nums.nunique()}')

    if 'gender' in df.columns:
        print(f'  Gender split  : {df["gender"].value_counts().to_dict()}')

    if 'sample_rate' in df.columns:
        print(f'  Sample rates  : {sorted(df["sample_rate"].unique())}')

    if 'format' in df.columns:
        print(f'  File formats  : {sorted(df["format"].unique())}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--audio_dir', required=True, help='Root directory for audio files')
    parser.add_argument('--stats', action='store_true', help='Print dataset statistics')
    args = parser.parse_args()

    ok = validate(args.csv, args.audio_dir, verbose=True)
    if args.stats:
        print_stats(args.csv)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
