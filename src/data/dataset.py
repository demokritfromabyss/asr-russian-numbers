"""PyTorch Dataset for the Russian spoken-numbers ASR task."""

import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from ..text.normalization import num_to_words
from ..text.vocabulary import Vocabulary
from .augmentation import SpecAugment, WaveformAugment


class AudioDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        vocab: Vocabulary,
        cfg: dict,
        is_train: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.is_train = is_train
        self.target_sr = cfg['data']['sample_rate']

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=cfg['data']['n_fft'],
            hop_length=cfg['data']['hop_length'],
            win_length=cfg['data']['win_length'],
            n_mels=cfg['data']['n_mels'],
            f_min=cfg['data'].get('f_min', 80.0),
            f_max=cfg['data'].get('f_max', 7600.0),
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

        aug_cfg = cfg.get('augmentation', {})
        self.wave_aug = WaveformAugment(aug_cfg) if is_train else None
        self.spec_aug = SpecAugment(
            time_mask_param=aug_cfg.get('time_mask_param', 80),
            freq_mask_param=aug_cfg.get('freq_mask_param', 20),
            num_time_masks=aug_cfg.get('num_time_masks', 2),
            num_freq_masks=aug_cfg.get('num_freq_masks', 2),
        ) if is_train and aug_cfg.get('spec_augment', True) else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load audio
        path = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = torchaudio.load(path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Resample to 16 kHz
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Waveform augmentation
        if self.wave_aug is not None:
            waveform = self.wave_aug(waveform, self.target_sr)

        # Log-mel spectrogram: (1, F, T) → (T, F)
        mel = self.mel_transform(waveform)      # (1, n_mels, T)
        mel = self.amp_to_db(mel)
        mel = mel.squeeze(0).T                  # (T, n_mels)

        # Per-utterance CMVN
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Spec augmentation
        if self.spec_aug is not None:
            mel = self.spec_aug(mel)

        # Encode label
        number = int(row['transcription'])
        text = num_to_words(number)
        token_ids = self.vocab.encode(text)

        spk_id = row.get('spk_id', 'unknown') if 'spk_id' in row else 'unknown'
        return mel, torch.tensor(token_ids, dtype=torch.long), str(spk_id)


class TestDataset(Dataset):
    """Dataset for inference — no labels required."""

    def __init__(self, csv_path: str, audio_dir: str, cfg: dict):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sr = cfg['data']['sample_rate']

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=cfg['data']['n_fft'],
            hop_length=cfg['data']['hop_length'],
            win_length=cfg['data']['win_length'],
            n_mels=cfg['data']['n_mels'],
            f_min=cfg['data'].get('f_min', 80.0),
            f_max=cfg['data'].get('f_max', 7600.0),
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        mel = self.mel_transform(waveform)
        mel = self.amp_to_db(mel)
        mel = mel.squeeze(0).T
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel, str(row['filename'])


def collate_fn(batch):
    """Pad mel spectrograms and token sequences to the longest in the batch."""
    mels, tokens, spk_ids = zip(*batch)

    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    max_T = mel_lengths.max().item()
    n_mels = mels[0].shape[1]
    mels_padded = torch.zeros(len(mels), max_T, n_mels)
    for i, m in enumerate(mels):
        mels_padded[i, :m.shape[0]] = m

    token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    max_L = token_lengths.max().item()
    tokens_padded = torch.zeros(len(tokens), max_L, dtype=torch.long)
    for i, t in enumerate(tokens):
        tokens_padded[i, :len(t)] = t

    return mels_padded, mel_lengths, tokens_padded, token_lengths, list(spk_ids)


def collate_fn_test(batch):
    mels, filenames = zip(*batch)
    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    max_T = mel_lengths.max().item()
    n_mels = mels[0].shape[1]
    mels_padded = torch.zeros(len(mels), max_T, n_mels)
    for i, m in enumerate(mels):
        mels_padded[i, :m.shape[0]] = m
    return mels_padded, mel_lengths, list(filenames)
