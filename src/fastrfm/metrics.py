"""Ranking and error metrics. No sklearn dependency."""

from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Average ranks of the positive class, midrank ties.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        # ranks are 1-based
        mid = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = mid
        i = j
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - p[mask])))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of exact matches. ``y_pred`` uses -1 for 'no prediction'."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(y == p))
