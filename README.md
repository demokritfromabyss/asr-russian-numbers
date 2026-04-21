# ASR Russian Numbers

Automatic Speech Recognition system for Russian spoken numbers (1,000–999,999).
Built from scratch using a Conformer-CTC architecture (~3.3 M parameters).

## Architecture

```
Log-mel features (80 bands)
  → Conv2d subsampling (4×)
  → Sinusoidal positional encoding
  → 6 × Conformer blocks (d_model=144, ff_dim=576, 4 heads, kernel=31)
  → Linear CTC head → 21-token character vocabulary
```

The vocabulary covers all 19 unique Russian characters that appear in
`num2words(n, lang='ru')` for n ∈ [1 000, 999 999], plus space and blank.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the competition data from Kaggle and place it under `data/`:

```
data/
├── train.csv          # columns: filename, transcription, spk_id
├── dev.csv
├── test.csv
└── <audio files>      # .wav or .flac referenced by filename column
```

CSV format:

| filename          | transcription | spk_id |
|-------------------|---------------|--------|
| train/spk1_001.wav | 12500         | spk_1  |

Validate your data layout before training:

```bash
python prepare_data.py --csv data/train.csv --audio_dir data/
```

## Training

```bash
python train.py --config config.yaml
```

Resume from a checkpoint:

```bash
python train.py --config config.yaml --resume checkpoints/last.pt
```

Key config options (`config.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `training.max_epochs` | 120 | Total training epochs |
| `training.batch_size` | 32 | Batch size per GPU |
| `training.learning_rate` | 5e-4 | Peak LR (linear warmup + cosine decay) |
| `training.warmup_steps` | 5000 | Warmup steps |
| `augmentation.speed_perturbation` | true | Speed perturbation (0.9/1.0/1.1) |
| `augmentation.spec_augment` | true | SpecAugment (time + freq masking) |

## Evaluation

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt
```

Outputs per-speaker CER, inD/ooD split, and the primary **HM-CER** metric:

```
Speaker      CER   Type
--------------------------------
spk_01    0.0123   inD
spk_07    0.0451   ooD
...
HM-CER    0.0234   ← primary metric
```

## Inference / Kaggle submission

```bash
python inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best.pt \
    --test_csv data/test.csv \
    --audio_dir data/ \
    --output submission.csv
```

The output CSV has two columns: `filename` and `transcription` (digit string).

## Metrics

- **Primary**: Harmonic mean of inD-CER and ooD-CER (per-speaker averages).
  Balances seen and unseen speakers.
- **Secondary**: ooD-CER alone (tie-breaker).

## Project structure

```
├── config.yaml          # all hyperparameters
├── train.py             # training loop
├── evaluate.py          # dev-set evaluation
├── inference.py         # test-set inference → submission CSV
├── prepare_data.py      # data validation helper
├── requirements.txt
└── src/
    ├── data/
    │   ├── dataset.py       # AudioDataset, TestDataset, collate_fn
    │   └── augmentation.py  # SpeedPerturbation, SpecAugment, GaussianNoise
    ├── model/
    │   ├── conformer.py     # ConformerBlock, attention, conv module
    │   ├── ctc_model.py     # ConformerCTC end-to-end model
    │   └── subsampling.py   # Conv2d 4× subsampling
    ├── text/
    │   ├── vocabulary.py    # 21-token Russian char vocabulary
    │   └── normalization.py # num_to_words / words_to_num
    └── utils/
        ├── decoder.py       # CTC greedy decoder
        └── metrics.py       # CER, per-speaker stats
```
