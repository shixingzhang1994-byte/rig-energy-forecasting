from __future__ import annotations

import itertools

import numpy as np


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided paired randomization p-value for a mean difference."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("differences 必须是一维有限非空数组")
    observed = abs(float(values.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_block_bootstrap_ci(
    differences: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 100_000,
    seed: int = 20260902,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("differences 必须是一维有限非空数组")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于0和1之间")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)

