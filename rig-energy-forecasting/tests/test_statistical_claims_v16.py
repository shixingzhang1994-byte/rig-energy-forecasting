from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.validation.statistical_claims import (  # noqa: E402
    exact_sign_flip_pvalue,
    paired_block_bootstrap_ci,
)


def test_exact_sign_flip_detects_insufficient_small_sample_evidence():
    differences = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 1.3])
    assert exact_sign_flip_pvalue(differences) >= 0.25


def test_bootstrap_interval_is_reproducible_and_contains_zero_with_many_ties():
    differences = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 1.3])
    first = paired_block_bootstrap_ci(differences, samples=10_000, seed=7)
    second = paired_block_bootstrap_ci(differences, samples=10_000, seed=7)
    assert first == second
    assert first[0] == 0.0
    assert first[1] > 0.0

