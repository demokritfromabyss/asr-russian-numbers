"""CER computation per speaker and overall."""

from collections import defaultdict
import editdistance


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate between two strings."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(hypothesis, reference) / len(reference)


def compute_cer_batch(
    hypotheses: list[str],
    references: list[str],
    spk_ids: list[str],
) -> dict:
    """
    Compute overall CER and per-speaker CER.

    Returns a dict with keys:
        'overall'           : float
        'per_speaker'       : {spk_id: float}
        'harmonic_mean_ind_ood': float  (requires 'ind_spk_ids' and 'ood_spk_ids' in the call)
    """
    total_dist, total_len = 0, 0
    spk_dist: dict[str, int] = defaultdict(int)
    spk_len: dict[str, int] = defaultdict(int)

    for hyp, ref, spk in zip(hypotheses, references, spk_ids):
        d = editdistance.eval(hyp, ref)
        total_dist += d
        total_len += len(ref)
        spk_dist[spk] += d
        spk_len[spk] += len(ref)

    overall = total_dist / max(total_len, 1)
    per_speaker = {spk: spk_dist[spk] / max(spk_len[spk], 1) for spk in spk_dist}
    return {'overall': overall, 'per_speaker': per_speaker}


def harmonic_mean_cer(ind_cer: float, ood_cer: float) -> float:
    """Primary Kaggle metric: harmonic mean of inD and ooD CER."""
    denom = ind_cer + ood_cer
    if denom == 0:
        return 0.0
    return 2 * ind_cer * ood_cer / denom
