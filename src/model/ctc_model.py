"""Full Conformer-CTC model."""

import torch
import torch.nn as nn

from .subsampling import Conv2dSubsampling
from .conformer import ConformerBlock, PositionalEncoding


class ConformerCTC(nn.Module):
    """
    Conformer encoder with a CTC output head.

    Architecture (default ~3.3 M params, well within the 5 M limit):
      Conv2d subsampling (4×) → sinusoidal PE → 6 × Conformer blocks → Linear CTC head
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 144,
        ff_dim: int = 576,
        n_heads: int = 4,
        n_layers: int = 6,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(n_mels, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, ff_dim, n_heads, conv_kernel, dropout)
            for _ in range(n_layers)
        ])
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            x       : (B, T, n_mels) log-mel features
            lengths : (B,)  frame counts before padding

        Returns:
            log_probs : (T', B, vocab_size)  log-softmax output for CTCLoss
            out_lengths: (B,)  frame counts after subsampling
        """
        x, lengths = self.subsampling(x, lengths)   # (B, T//4, d_model)
        x = self.pos_enc(x)

        T = x.size(1)
        mask = torch.arange(T, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

        for block in self.encoder:
            x = block(x, key_padding_mask=mask)

        logits = self.ctc_head(x)                   # (B, T', vocab)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.permute(1, 0, 2), lengths  # (T', B, vocab)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict, vocab_size: int) -> ConformerCTC:
    m = cfg['model']
    model = ConformerCTC(
        vocab_size=vocab_size,
        n_mels=m.get('n_mels', 80),
        d_model=m.get('d_model', 144),
        ff_dim=m.get('ff_dim', 576),
        n_heads=m.get('n_heads', 4),
        n_layers=m.get('n_layers', 6),
        conv_kernel=m.get('conv_kernel', 31),
        dropout=m.get('dropout', 0.1),
    )
    return model
