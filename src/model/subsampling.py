"""Conv2d subsampling layer: reduces time by 4× before the Conformer encoder."""

import torch
import torch.nn as nn


def _conv_output_len(length: torch.Tensor, kernel: int = 3, stride: int = 2, padding: int = 1) -> torch.Tensor:
    return (length + 2 * padding - kernel) // stride + 1


class Conv2dSubsampling(nn.Module):
    """
    Two stride-2 conv layers reduce time by 4× and frequency by 4×.
    Input : (B, T, n_mels)
    Output: (B, T//4, d_model)
    """

    def __init__(self, n_mels: int = 80, d_model: int = 144, conv_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = n_mels // 4          # 80 → 20
        self.proj = nn.Linear(conv_channels * freq_out, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T, F)
        x = x.unsqueeze(1)              # (B, 1, T, F)
        x = self.conv(x)                # (B, C, T//4, F//4)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T//4, C*F)
        x = self.proj(x)                # (B, T//4, d_model)

        lengths = _conv_output_len(_conv_output_len(lengths))
        lengths = lengths.clamp(min=1)
        return x, lengths
