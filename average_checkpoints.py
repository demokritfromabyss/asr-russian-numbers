"""
Average weights of multiple checkpoints to improve generalization.

Usage:
    python average_checkpoints.py \
        --checkpoints checkpoints/epoch_60.pt checkpoints/epoch_70.pt checkpoints/epoch_80.pt \
        --output checkpoints/averaged.pt

    # Or average all epoch_*.pt files automatically:
    python average_checkpoints.py --checkpoint_dir checkpoints/ --top_k 5 --output checkpoints/averaged.pt
"""

import argparse
import os
import glob
import torch


def average_checkpoints(paths: list[str], output: str):
    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        cer = torch.load(p, map_location='cpu').get('val', {}).get('hm_cer', '?')
        print(f"  {p}  HM-CER={cer}")

    # Load and average model weights
    avg_state = None
    for path in paths:
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['model']
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    n = len(paths)
    for k in avg_state:
        avg_state[k] /= n
        # Cast back to original dtype
        avg_state[k] = avg_state[k].to(
            torch.load(paths[0], map_location='cpu')['model'][k].dtype
        )

    # Save averaged checkpoint (reuse metadata from best individual)
    best_path = min(
        paths,
        key=lambda p: torch.load(p, map_location='cpu').get('val', {}).get('hm_cer', float('inf'))
    )
    ref_ckpt = torch.load(best_path, map_location='cpu')
    ref_ckpt['model'] = avg_state
    ref_ckpt['averaged_from'] = paths

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(ref_ckpt, output)
    print(f"\nSaved averaged checkpoint → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', default=None,
                        help='Explicit list of checkpoint paths')
    parser.add_argument('--checkpoint_dir', default=None,
                        help='Directory to search for epoch_*.pt files')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Use top-k checkpoints by HM-CER (with --checkpoint_dir)')
    parser.add_argument('--output', required=True, help='Output path for averaged checkpoint')
    args = parser.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
    elif args.checkpoint_dir:
        all_ckpts = glob.glob(os.path.join(args.checkpoint_dir, 'epoch_*.pt'))
        if not all_ckpts:
            all_ckpts = glob.glob(os.path.join(args.checkpoint_dir, '*.pt'))
            all_ckpts = [p for p in all_ckpts if 'averaged' not in p]

        # Sort by HM-CER ascending (lower = better)
        def get_hm_cer(p):
            try:
                return torch.load(p, map_location='cpu').get('val', {}).get('hm_cer', float('inf'))
            except Exception:
                return float('inf')

        all_ckpts.sort(key=get_hm_cer)
        paths = all_ckpts[:args.top_k]
        print(f"Found {len(all_ckpts)} checkpoints, using top-{args.top_k} by HM-CER")
    else:
        raise ValueError("Provide --checkpoints or --checkpoint_dir")

    if not paths:
        raise ValueError("No checkpoints found")

    average_checkpoints(paths, args.output)


if __name__ == '__main__':
    main()
