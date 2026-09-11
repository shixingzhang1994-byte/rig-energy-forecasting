from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory import (
    HIGH_RISK,
    UNIVERSE,
    belief,
    discount_mass,
    fuse_risk_frame,
    margin_interval_focal,
    pignistic_probability,
    plausibility,
    yager_combine,
)


CONFIG = {
    "discounts": {
        "learned_probability": 0.70,
        "margin_interval": 0.80,
        "low_soc": 0.45,
        "healthy_soc": 0.25,
        "grid_derating": 0.35,
        "rapid_ramp": 0.30,
    },
    "margin_thresholds": {
        "normal_margin_ratio": 0.15,
        "watch_margin_ratio": 0.05,
        "warning_margin_ratio": 0.00,
    },
    "uncertainty_interval_scale": 1.00,
    "soc_warning_pct": 35.00,
    "soc_healthy_pct": 65.00,
    "grid_derating_threshold_kw": 700.00,
    "ramp_threshold_kw_per_s": 25.00,
    "decision": {
        "high_score_threshold": 0.50,
        "severe_score_threshold": 0.45,
        "conflict_weight": 0.25,
    },
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_peak_kw": [1000.0, 1000.0],
            "forecast_margin_ratio": [0.20, -0.10],
            "grid_available_capacity_kw": [1200.0, 300.0],
            "storage_soc_pct": [75.0, 20.0],
            "forecast_uncertainty_kw": [10.0, 30.0],
            "forecast_ramp_kw_per_s": [2.0, 40.0],
            "prob_normal": [0.80, 0.05],
            "prob_watch": [0.10, 0.05],
            "prob_warning": [0.05, 0.20],
            "prob_severe": [0.05, 0.70],
            "true_risk_level": [3, 0],
            "actual_peak_kw": [9999.0, 0.0],
        }
    )


def test_discount_transfers_mass_to_ignorance() -> None:
    discounted = discount_mass({1: 0.75, 2: 0.25}, 0.60)
    assert discounted[1] == pytest.approx(0.45)
    assert discounted[2] == pytest.approx(0.15)
    assert discounted[UNIVERSE] == pytest.approx(0.40)


def test_yager_rule_preserves_total_conflict_as_ignorance() -> None:
    combined, conflict = yager_combine([{1: 1.0}, {8: 1.0}])
    assert conflict == pytest.approx(1.0)
    assert combined == {UNIVERSE: pytest.approx(1.0)}
    assert np.allclose(pignistic_probability(combined), np.full(4, 0.25))


def test_belief_plausibility_bound_high_risk_support() -> None:
    mass = {1: 0.30, HIGH_RISK: 0.40, UNIVERSE: 0.30}
    assert belief(mass, HIGH_RISK) == pytest.approx(0.40)
    assert plausibility(mass, HIGH_RISK) == pytest.approx(0.70)


def test_margin_interval_expands_to_adjacent_classes_under_uncertainty() -> None:
    thresholds = CONFIG["margin_thresholds"]
    assert margin_interval_focal(0.10, 0.00, thresholds) == 1 << 1
    assert margin_interval_focal(0.10, 0.06, thresholds) == (1 << 0) | (1 << 1) | (1 << 2)


def test_fusion_is_causal_with_respect_to_future_truth_columns() -> None:
    frame = _frame()
    original = fuse_risk_frame(frame, CONFIG)
    changed = frame.copy()
    changed["true_risk_level"] = [0, 3]
    changed["actual_peak_kw"] = [-1.0, 1e9]
    repeated = fuse_risk_frame(changed, CONFIG)
    pd.testing.assert_frame_equal(original, repeated)


def test_fusion_escalates_coherent_high_risk_evidence() -> None:
    output = fuse_risk_frame(_frame(), CONFIG)
    assert output.loc[0, "evidential_risk_level"] <= 1
    assert output.loc[1, "evidential_dispatch_risk_level"] >= 2
    assert output.loc[1, "evidential_high_belief"] > output.loc[0, "evidential_high_belief"]
