"""Paired comparisons of classifiers on the same test images, and mean ± sd."""

from __future__ import annotations

import math

import numpy as np


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    from scipy.stats import binomtest

    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": min(p, 1.0)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [1.0] * m, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvalues[i]))
        adj[i] = running
    return adj


def majority_vote(preds: list[np.ndarray], k: int) -> np.ndarray:
    votes = np.zeros((len(preds[0]), k), dtype=int)
    for p in preds:
        votes[np.arange(len(p)), p] += 1
    return votes.argmax(1)


def mean_sd(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def fmt(values, digits=2, scale=1.0) -> str:
    if not values:
        return "--"
    m, s = mean_sd([x * scale for x in values])
    if math.isnan(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"
