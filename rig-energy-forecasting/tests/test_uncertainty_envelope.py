from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.experiment import _one_sided_horizon_correction  # noqa: E402


def test_horizon_correction_uses_finite_sample_upper_residual_order_statistic():
    prediction = np.full((5, 2), 100.0)
    actual = prediction + np.array(
        [
            [0.0, -4.0],
            [1.0, -3.0],
            [2.0, -2.0],
            [3.0, -1.0],
            [4.0, 0.0],
        ]
    )
    correction = _one_sided_horizon_correction(
        actual, prediction, coverage=0.50
    )
    # k=ceil((n+1)*0.5)=3: third ordered residual at each horizon;
    # negative upper corrections are clipped because this is a safety envelope.
    assert np.allclose(correction, [2.0, 0.0])


def test_horizon_correction_rejects_mismatched_or_empty_arrays():
    with np.testing.assert_raises(ValueError):
        _one_sided_horizon_correction(
            np.ones((2, 3)), np.ones((2, 2)), coverage=0.90
        )
    with np.testing.assert_raises(ValueError):
        _one_sided_horizon_correction(
            np.empty((0, 2)), np.empty((0, 2)), coverage=0.90
        )
