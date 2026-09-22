from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_v22_confirmatory import (  # noqa: E402
    bootstrap_mean_interval,
    exact_sign_flip_pvalue,
    holm_adjust,
)


def test_exact_sign_flip_enumerates_the_seed_level_null() -> None:
    assert exact_sign_flip_pvalue(np.array([1.0, 1.0])) == pytest.approx(0.5)
    assert exact_sign_flip_pvalue(np.zeros(12)) == 1.0


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    assert holm_adjust([0.01, 0.03, 0.20]) == pytest.approx([0.03, 0.06, 0.20])


def test_seed_bootstrap_is_reproducible_and_brackets_mean() -> None:
    values = np.array([-4.0, -2.0, -1.0, 0.0, 1.0])
    first = bootstrap_mean_interval(values, draws=2_000, seed=7)
    second = bootstrap_mean_interval(values, draws=2_000, seed=7)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
