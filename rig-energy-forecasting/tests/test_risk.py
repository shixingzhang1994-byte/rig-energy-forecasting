from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.labels import assign_risk_levels, build_risk_evidence  # noqa: E402
from rig_energy.risk.experiment import (  # noqa: E402
    _apply_severe_sentinel,
    _causal_probability_filter,
    _classification_metrics,
    _fit_minimum_recall_threshold,
    _one_sided_conformal_correction,
    _apply_ordinal_cumulative_threshold,
    _simulate_supply_context,
    _causal_transition_feature,
)
from rig_energy.risk.models import (  # noqa: E402
    apply_probability_ensemble,
    fit_probability_ensemble,
)


def test_four_level_risk_definition_follows_margin_thresholds():
    actual = np.full((4, 12), 1000.0)
    # Required power is 1050 kW. These margins map to normal/watch/warning/severe.
    supply = np.array([1260.0, 1155.0, 1071.0, 1000.0])
    levels, margin_kw, margin_ratio = assign_risk_levels(actual, supply)
    assert levels.tolist() == [0, 1, 2, 3]
    assert np.allclose(margin_kw, [210.0, 105.0, 21.0, -50.0])
    assert np.all(np.diff(margin_ratio) < 0.0)


def test_probability_ensemble_weights_are_valid_and_apply():
    labels = np.array([0, 1, 2, 3] * 10)
    good = np.eye(4)[labels] * 0.85 + 0.15 / 4.0
    weak = np.full((len(labels), 4), 0.25)
    probabilities = {"good": good, "weak": weak}
    weights = fit_probability_ensemble(probabilities, labels)
    combined = apply_probability_ensemble(probabilities, weights)
    assert all(value >= 0.0 for value in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    assert combined.shape == (len(labels), 4)
    assert np.allclose(combined.sum(axis=1), 1.0)
    assert weights["good"] > weights["weak"]


def test_evidence_exposes_multiple_auditable_triggers():
    evidence = build_risk_evidence(
        np.array([-0.1]),
        np.array([500.0]),
        np.array([25.0]),
        np.array([100.0]),
        np.array([35.0]),
        grid_derating_threshold_kw=700.0,
        low_soc_threshold_pct=35.0,
        uncertainty_threshold_kw=80.0,
        ramp_threshold_kw_per_s=25.0,
    )[0]
    assert "预测供电缺口" in evidence
    assert "网电受限" in evidence
    assert "储能SOC偏低" in evidence
    assert "模型分歧较大" in evidence
    assert "负荷快速上升" in evidence


def test_severe_recall_is_not_hidden_by_merged_high_risk_recall():
    labels = np.array([2, 3, 3], dtype=np.int64)
    # 三个样本均被判为高风险，但严重级全部被低估为警告级。
    probability = np.array(
        [
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.97, 0.01],
        ]
    )
    metrics = _classification_metrics(labels, probability)
    assert metrics["high_risk_recall"] == 1.0
    assert metrics["severe_recall"] == 0.0
    assert metrics["severe_underclassification_rate"] == 1.0


def test_validation_calibrated_sentinel_only_escalates_qualified_severe_cases():
    labels = np.array([3, 3, 3, 3, 2, 1], dtype=np.int64)
    severe_scores = np.array([0.90, 0.80, 0.70, 0.10, 0.65, 0.05])
    threshold = _fit_minimum_recall_threshold(
        severe_scores, labels, minimum_recall=0.75
    )
    base = np.tile(np.array([0.05, 0.10, 0.80, 0.05]), (len(labels), 1))
    calibrated = _apply_severe_sentinel(base, severe_scores, threshold)
    prediction = calibrated.argmax(axis=1)
    assert recall_score(labels == 3, prediction == 3, zero_division=0) >= 0.75
    assert prediction[-1] == 2
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_one_sided_conformal_correction_uses_only_underprediction_tail():
    actual_peak = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    forecast_peak = np.full(5, 100.0)
    correction = _one_sided_conformal_correction(
        actual_peak, forecast_peak, coverage=0.50
    )
    assert correction == 3.0

    # 系统性高估不应产生负的“安全修正”。
    assert _one_sided_conformal_correction(
        np.array([90.0, 95.0]), np.array([100.0, 100.0]), coverage=0.50
    ) == 0.0


def test_causal_risk_filter_cannot_be_changed_by_future_probabilities():
    probability = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.2, 0.7, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    changed_future = probability.copy()
    changed_future[2] = [0.9, 0.05, 0.03, 0.02]
    filtered = _causal_probability_filter(probability, smoothing=0.5)
    changed = _causal_probability_filter(changed_future, smoothing=0.5)
    assert np.allclose(filtered[:2], changed[:2])
    assert np.allclose(filtered.sum(axis=1), 1.0)
    safety_filtered = _causal_probability_filter(probability, smoothing=0.90)
    assert safety_filtered[2].argmax() == 3


def test_risk_transition_feature_uses_history_not_future_truth():
    arrays = {
        "history_transition_flags": np.array([0, 1, 0], dtype=np.int8),
        "future_transition_flags": np.array([1, 0, 1], dtype=np.int8),
        "transition_flags": np.array([1, 0, 1], dtype=np.int8),
    }
    feature, source = _causal_transition_feature(arrays, expected_length=3)
    assert feature.tolist() == [0.0, 1.0, 0.0]
    assert source == "history_transition_flags"

    arrays["future_transition_flags"][:] = 0
    arrays["transition_flags"][:] = 0
    changed, _ = _causal_transition_feature(arrays, expected_length=3)
    assert np.array_equal(feature, changed)


def test_risk_transition_feature_can_fall_back_to_observed_state_changes():
    feature, source = _causal_transition_feature(
        {"operation_state_codes": np.array([2, 2, 3, 3], dtype=np.int16)},
        expected_length=4,
    )
    assert feature.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert source == "observed_operation_state_change"


def test_ordinal_threshold_uses_monotone_cumulative_boundaries():
    probability = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.60, 0.20, 0.15, 0.05],
        ]
    )
    calibrated = _apply_ordinal_cumulative_threshold(probability, threshold=0.50)
    # Row 1 crosses Y>=1 and Y>=2 but not Y>=3; row 2 crosses none.
    assert calibrated.argmax(axis=1).tolist() == [2, 0]
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_independent_supply_process_does_not_change_with_load_forecast():
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=80, freq="5s"))
    config = {
        "split": {"train_fraction": 0.60, "val_fraction": 0.20},
        "synthetic_supply": {
            "generation_mode": "independent_process",
            "grid_rated_capacity_kw": 1300.0,
            "generator_rated_power_kw": 1200.0,
            "storage_rated_power_kw": 500.0,
            "soc_min_pct": 15.0,
            "soc_max_pct": 90.0,
            "soc_full_power_above_pct": 30.0,
            "block_length_steps": [8, 8],
            "regime_cycle": ["normal", "constrained", "weak", "emergency"],
            "regimes": {
                "normal": {
                    "grid_capacity_kw": 1200.0,
                    "generator_capacity_kw": 900.0,
                    "storage_soc_pct": 72.0,
                },
                "constrained": {
                    "grid_capacity_kw": 850.0,
                    "generator_capacity_kw": 650.0,
                    "storage_soc_pct": 55.0,
                },
                "weak": {
                    "grid_capacity_kw": 500.0,
                    "generator_capacity_kw": 480.0,
                    "storage_soc_pct": 38.0,
                },
                "emergency": {
                    "grid_capacity_kw": 180.0,
                    "generator_capacity_kw": 300.0,
                    "storage_soc_pct": 24.0,
                },
            },
        },
    }
    low_forecast = np.full(len(timestamps), 300.0)
    high_forecast = np.full(len(timestamps), 2000.0)
    low = _simulate_supply_context(timestamps, low_forecast, config, seed=17)
    high = _simulate_supply_context(timestamps, high_forecast, config, seed=17)
    columns = [
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_soc_pct",
        "storage_available_discharge_power_kw",
    ]
    assert np.allclose(low[columns], high[columns])
    assert set(low["supply_source_type"]) == {"synthetic_independent_supply"}
