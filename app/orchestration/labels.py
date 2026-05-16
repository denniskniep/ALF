from __future__ import annotations

INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NORMAL = "NORMAL"
SLIGHTLY_ELEVATED = "SLIGHTLY_ELEVATED"
ELEVATED = "ELEVATED"
ANOMALOUS = "ANOMALOUS"
HIGHLY_ANOMALOUS = "HIGHLY_ANOMALOUS"


def score_label(score: float | None) -> str:
    """Map a linear 0–100 anomaly score to a label constant.

    All models must return scores on this linear scale so thresholds are
    applied consistently regardless of algorithm.
    """
    if score is None:
        return INSUFFICIENT_DATA
    if score < 30:
        return NORMAL
    if score < 50:
        return SLIGHTLY_ELEVATED
    if score < 70:
        return ELEVATED
    if score < 85:
        return ANOMALOUS
    return HIGHLY_ANOMALOUS
