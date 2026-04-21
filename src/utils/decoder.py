"""CTC decoding utilities."""

import torch
from ..text.vocabulary import Vocabulary, BLANK_ID


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary) -> list[str]:
    """
    Greedy CTC decoding: argmax → collapse repeats → remove blanks → decode.

    Args:
        log_probs : (T, B, vocab_size)
        lengths   : (B,) encoder output lengths
        vocab     : Vocabulary instance

    Returns:
        List of decoded strings (Russian words).
    """
    log_probs = log_probs.permute(1, 0, 2)      # (B, T, vocab)
    predictions = log_probs.argmax(dim=-1)       # (B, T)
    results = []

    for b in range(predictions.size(0)):
        T = int(lengths[b].item())
        ids = predictions[b, :T].tolist()

        # Collapse consecutive duplicates and remove blank
        collapsed = []
        prev = None
        for idx in ids:
            if idx != prev:
                if idx != BLANK_ID:
                    collapsed.append(idx)
                prev = idx

        results.append(vocab.decode(collapsed))

    return results


def batch_decode(log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary) -> list[str]:
    """Wrapper that moves tensors to CPU before decoding."""
    return ctc_greedy_decode(log_probs.cpu(), lengths.cpu(), vocab)
