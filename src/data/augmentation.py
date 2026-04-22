"""Audio and spectrogram augmentations for training robustness."""

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


class PitchPerturbation:
    """Shifts pitch by a random number of semitones via resampling trick."""

    def __init__(self, steps: list = (-2, -1, 0, 1, 2)):
        self.steps = steps

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        step = random.choice(self.steps)
        if step == 0:
            return waveform
        # Pitch shift via resampling: change SR without changing duration
        factor = 2 ** (step / 12.0)
        new_sr = int(sample_rate * factor)
        shifted = torchaudio.functional.resample(waveform, sample_rate, new_sr)
        # Restore original length to keep time alignment
        shifted = torchaudio.functional.resample(shifted, new_sr, sample_rate)
        orig_len = waveform.shape[-1]
        if shifted.shape[-1] > orig_len:
            shifted = shifted[..., :orig_len]
        elif shifted.shape[-1] < orig_len:
            shifted = torch.nn.functional.pad(shifted, (0, orig_len - shifted.shape[-1]))
        return shifted


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
        # mel: (T, F) → convert to (1, F, T) for torchaudio masking
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
        self.pitch = PitchPerturbation(cfg.get('pitch_shift_steps', [-2, -1, 0, 1, 2])) \
            if cfg.get('pitch_perturbation', False) else None
        self.noise = AddGaussianNoise()

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.speed is not None and random.random() < 0.5:
            waveform = self.speed(waveform, sample_rate)
        if self.pitch is not None and random.random() < 0.4:
            waveform = self.pitch(waveform, sample_rate)
        if random.random() < 0.3:
            waveform = self.noise(waveform)
        return waveform
