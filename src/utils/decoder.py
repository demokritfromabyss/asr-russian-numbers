"""CTC decoding utilities: greedy and prefix beam search."""

import numpy as np
import torch
from ..text.vocabulary import Vocabulary, BLANK_ID

NEG_INF = float('-inf')


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary) -> list[str]:
    """
    Greedy CTC decoding: argmax → collapse repeats → remove blanks → decode.

    Args:
        log_probs : (T, B, vocab_size)
        lengths   : (B,) encoder output lengths
        vocab     : Vocabulary instance
    """
    log_probs = log_probs.permute(1, 0, 2)      # (B, T, vocab)
    predictions = log_probs.argmax(dim=-1)       # (B, T)
    results = []

    for b in range(predictions.size(0)):
        T = int(lengths[b].item())
        ids = predictions[b, :T].tolist()

        collapsed = []
        prev = None
        for idx in ids:
            if idx != prev:
                if idx != BLANK_ID:
                    collapsed.append(idx)
                prev = idx

        results.append(vocab.decode(collapsed))

    return results


def _ctc_beam_search_single(log_probs_np: np.ndarray, beam_width: int, vocab: Vocabulary) -> str:
    """
    CTC prefix beam search for a single utterance.

    Args:
        log_probs_np : (T, vocab_size) log-softmax probabilities
        beam_width   : number of beams to keep
        vocab        : Vocabulary instance
    """
    T = log_probs_np.shape[0]

    # beam: prefix_tuple -> [log_prob_ending_blank, log_prob_ending_nonblank]
    beam = {(): [0.0, NEG_INF]}

    for t in range(T):
        probs = log_probs_np[t]   # (vocab_size,)
        new_beam = {}

        for prefix, (log_pb, log_pnb) in beam.items():
            log_p_total = np.logaddexp(log_pb, log_pnb)

            # Extend with blank — prefix stays the same
            new_log_pb = log_p_total + probs[BLANK_ID]
            if prefix not in new_beam:
                new_beam[prefix] = [NEG_INF, NEG_INF]
            new_beam[prefix][0] = np.logaddexp(new_beam[prefix][0], new_log_pb)

            # Extend with each non-blank token
            for c in range(1, probs.shape[0]):
                new_prefix = prefix + (c,)

                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeated char: can only extend from a blank-ending prefix
                    new_log_pnb = log_pb + probs[c]
                else:
                    new_log_pnb = log_p_total + probs[c]

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = [NEG_INF, NEG_INF]
                new_beam[new_prefix][1] = np.logaddexp(new_beam[new_prefix][1], new_log_pnb)

        # Prune to beam_width
        beam = dict(sorted(
            new_beam.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True,
        )[:beam_width])

    best_prefix = max(beam, key=lambda p: np.logaddexp(beam[p][0], beam[p][1]))
    return vocab.decode(list(best_prefix))


def ctc_beam_decode(log_probs: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary, beam_width: int = 10) -> list[str]:
    """
    CTC beam search decoding for a batch.

    Args:
        log_probs  : (T, B, vocab_size)
        lengths    : (B,) encoder output lengths
        beam_width : number of beams (higher = better quality, slower)
    """
    log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()   # (B, T, vocab)
    results = []
    for b in range(log_probs_np.shape[0]):
        T = int(lengths[b].item())
        text = _ctc_beam_search_single(log_probs_np[b, :T], beam_width, vocab)
        results.append(text)
    return results


def batch_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 1,
) -> list[str]:
    """
    Unified decode entry point.
    beam_width=1  → greedy (fast)
    beam_width>1  → beam search (better accuracy)
    """
    log_probs = log_probs.cpu()
    lengths = lengths.cpu()
    if beam_width <= 1:
        return ctc_greedy_decode(log_probs, lengths, vocab)
    return ctc_beam_decode(log_probs, lengths, vocab, beam_width)
