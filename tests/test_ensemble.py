from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.models import (  # noqa: E402
    apply_causal_error_feedback,
    apply_weighted_ensemble,
    fit_validation_weighted_ensemble,
)


def test_validation_weighted_ensemble_prefers_better_member():
    target = np.arange(24, dtype=np.float32).reshape(6, 4)
    predictions = {
        "good": target + 0.1,
        "bad": target + 8.0,
        "medium": target - 2.0,
    }
    weights, metadata = fit_validation_weighted_ensemble(target, predictions)
    blended = apply_weighted_ensemble(predictions, weights)
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(weight >= 0.0 for weight in weights.values())
    assert weights["good"] > weights["bad"]
    assert np.mean(np.abs(blended - target)) < np.mean(
        np.abs(predictions["medium"] - target)
    )
    assert "fit_method" in metadata


def test_causal_error_feedback_rejects_short_delay():
    target = np.ones((20, 4), dtype=np.float32)
    with np.testing.assert_raises(ValueError):
        apply_causal_error_feedback(
            target,
            target,
            feedback_delay_steps=3,
            smoothing=0.1,
            correction_clip_kw=10.0,
        )


def test_causal_error_feedback_does_not_use_unreleased_future():
    base = np.full((12, 4), 100.0, dtype=np.float32)
    actual_a = base.copy()
    actual_b = base.copy()
    actual_b[5:] += 1000.0
    corrected_a, _ = apply_causal_error_feedback(
        base,
        actual_a,
        feedback_delay_steps=4,
        smoothing=1.0,
        correction_clip_kw=2000.0,
    )
    corrected_b, _ = apply_causal_error_feedback(
        base,
        actual_b,
        feedback_delay_steps=4,
        smoothing=1.0,
        correction_clip_kw=2000.0,
    )
    # actual_b[5] 只能在样本9及以后被释放。
    assert np.array_equal(corrected_a[:9], corrected_b[:9])
    assert not np.array_equal(corrected_a[9:], corrected_b[9:])
