"""Audio and spectrogram augmentations for training robustness."""

import glob
import os
import random
import torch
import torchaudio
import torchaudio.transforms as T


class SpeedPerturbation:
    """Randomly changes playback speed without pitch change via resampling."""

    def __init__(self, rates: tuple = (0.9, 1.0, 1.1)):
        self.rates = rates

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        rate = random.choice(self.rates)
        if rate == 1.0:
            return waveform
        new_sr = int(sample_rate * rate)
        perturbed = torchaudio.functional.resample(waveform, sample_rate, new_sr)
        perturbed = torchaudio.functional.resample(perturbed, new_sr, sample_rate)
        return perturbed


class AddBackgroundNoise:
    """Mixes speech with a random noise file from MUSAN at a random SNR."""

    def __init__(self, noise_dir: str, snr_db_range: tuple = (5, 20)):
        self.snr_min, self.snr_max = snr_db_range
        self.noise_files = (
            glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True) +
            glob.glob(os.path.join(noise_dir, "**/*.flac"), recursive=True)
        )
        if self.noise_files:
            print(f"AddBackgroundNoise: loaded {len(self.noise_files)} noise files from {noise_dir}")

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.noise_files:
            return waveform

        noise, noise_sr = torchaudio.load(random.choice(self.noise_files))
        if noise_sr != sample_rate:
            noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        if noise.shape[0] > 1:
            noise = noise.mean(0, keepdim=True)

        # Tile noise to match speech length
        speech_len = waveform.shape[-1]
        if noise.shape[-1] < speech_len:
            repeats = (speech_len // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[..., :speech_len]

        snr_db = random.uniform(self.snr_min, self.snr_max)
        snr_linear = 10 ** (snr_db / 20)
        speech_power = waveform.norm(p=2)
        noise_power = noise.norm(p=2)
        noise = noise * (speech_power / (noise_power * snr_linear + 1e-9))
        return waveform + noise


class AddReverb:
    """Convolves speech with a random room impulse response (RIR)."""

    def __init__(self, rir_dir: str):
        self.rir_files = (
            glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True) +
            glob.glob(os.path.join(rir_dir, "**/*.flac"), recursive=True)
        )
        if self.rir_files:
            print(f"AddReverb: loaded {len(self.rir_files)} RIR files from {rir_dir}")

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.rir_files:
            return waveform

        rir, rir_sr = torchaudio.load(random.choice(self.rir_files))
        if rir_sr != sample_rate:
            rir = torchaudio.functional.resample(rir, rir_sr, sample_rate)
        if rir.shape[0] > 1:
            rir = rir.mean(0, keepdim=True)

        rir = rir / (rir.norm(p=2) + 1e-9)
        reverbed = torchaudio.functional.fftconvolve(waveform, rir)
        return reverbed[..., :waveform.shape[-1]]


class VolumePerturb:
    """Randomly scales waveform amplitude by ±6 dB."""

    def __init__(self, gain_db_range: tuple = (-6, 6)):
        self.min_db, self.max_db = gain_db_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.min_db, self.max_db)
        gain = 10 ** (gain_db / 20)
        return waveform * gain


class AddGaussianNoise:
    """Adds Gaussian noise at a random SNR from the given range."""

    def __init__(self, snr_db_range: tuple = (10, 40)):
        self.snr_min, self.snr_max = snr_db_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        snr_linear = 10 ** (snr_db / 20)
        signal_power = waveform.norm(p=2)
        noise = torch.randn_like(waveform)
        noise_power = noise.norm(p=2)
        noise = noise * (signal_power / (noise_power * snr_linear + 1e-9))
        return waveform + noise


class SpecAugment:
    """
    SpecAugment: time and frequency masking on the log-mel spectrogram.
    Expects input shape (T, F) — i.e. time-first.
    """

    def __init__(
        self,
        time_mask_param: int = 80,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self.time_masks = [T.TimeMasking(time_mask_param, iid_masks=True) for _ in range(num_time_masks)]
        self.freq_masks = [T.FrequencyMasking(freq_mask_param, iid_masks=True) for _ in range(num_freq_masks)]

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.T.unsqueeze(0)          # (1, F, T)
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x.squeeze(0).T           # (T, F)


class WaveformAugment:
    """Chains waveform-level augmentations applied during training."""

    def __init__(self, cfg: dict):
        self.speed = SpeedPerturbation(tuple(cfg.get('speed_rates', [0.9, 1.0, 1.1]))) \
            if cfg.get('speed_perturbation', True) else None

        noise_dir = cfg.get('musan_noise_dir', '')
        self.bg_noise = AddBackgroundNoise(noise_dir) if noise_dir else None

        rir_dir = cfg.get('rir_dir', '')
        self.reverb = AddReverb(rir_dir) if rir_dir else None

        self.volume = VolumePerturb()
        self.gaussian = AddGaussianNoise()

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.speed is not None and random.random() < 0.5:
            waveform = self.speed(waveform, sample_rate)
        if self.reverb is not None and random.random() < 0.5:
            waveform = self.reverb(waveform, sample_rate)
        if self.bg_noise is not None and random.random() < 0.5:
            waveform = self.bg_noise(waveform, sample_rate)
        elif random.random() < 0.3:
            waveform = self.gaussian(waveform)
        if random.random() < 0.5:
            waveform = self.volume(waveform)
        return waveform
