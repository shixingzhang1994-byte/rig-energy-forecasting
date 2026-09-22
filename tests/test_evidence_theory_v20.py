from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory import UNIVERSE  # noqa: E402
from rig_energy.risk.evidence_theory_v20 import (  # noqa: E402
    DEMPSTER_RULE,
    YAGER_RULE,
    dempster_combine,
    fuse_risk_frame_v20,
    temperature_scale,
)


def test_dempster_normalizes_nonconflicting_mass_and_reports_conflict() -> None:
    mass, conflict, fallback = dempster_combine([{1: 0.6, UNIVERSE: 0.4}, {1: 0.5, 2: 0.5}])
    assert conflict == pytest.approx(0.3)
    assert sum(mass.values()) == pytest.approx(1.0)
    assert mass[1] == pytest.approx(0.5 / 0.7)
    assert not fallback


def test_dempster_total_conflict_falls_back_to_explicit_ignorance() -> None:
    mass, conflict, fallback = dempster_combine([{1: 1.0}, {2: 1.0}])
    assert mass == {UNIVERSE: 1.0}
    assert conflict == pytest.approx(1.0)
    assert fallback


def test_temperature_scale_preserves_normalization_and_argmax() -> None:
    probability = np.array([[0.10, 0.20, 0.60, 0.10], [0.70, 0.10, 0.10, 0.10]])
    calibrated = temperature_scale(probability, 0.25)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.array_equal(calibrated.argmax(axis=1), probability.argmax(axis=1))
    assert np.all(calibrated.max(axis=1) > probability.max(axis=1))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_peak_kw": [1000.0, 1200.0],
            "forecast_margin_ratio": [0.10, -0.05],
            "grid_available_capacity_kw": [1300.0, 650.0],
            "storage_soc_pct": [70.0, 25.0],
            "forecast_uncertainty_kw": [50.0, 100.0],
            "forecast_ramp_kw_per_s": [5.0, 30.0],
            "prob_normal": [0.70, 0.05],
            "prob_watch": [0.20, 0.10],
            "prob_warning": [0.08, 0.25],
            "prob_severe": [0.02, 0.60],
        }
    )


def _config(rule: str, temperature: float) -> dict:
    config = yaml.safe_load(
        (PROJECT_DIR / "artifacts/v19_evidential_risk/development_search/selected_evidential_config.yaml").read_text(encoding="utf-8")
    )
    config["combination_rule"] = rule
    config["calibration"] = {
        "method": "scalar_temperature_power",
        "temperature": temperature,
        "dispatch_uses": "raw_pignistic_probability",
    }
    return config


@pytest.mark.parametrize("rule", [YAGER_RULE, DEMPSTER_RULE])
def test_calibration_cannot_change_dispatch_for_fixed_rule(rule: str) -> None:
    cold = fuse_risk_frame_v20(_frame(), _config(rule, 0.20))
    unit = fuse_risk_frame_v20(_frame(), _config(rule, 1.00))
    assert cold["v20_dispatch_risk_level"].equals(unit["v20_dispatch_risk_level"])
    assert not cold["v20_calibration_changed_argmax"].any()


def test_v20_frame_does_not_require_or_read_truth_column() -> None:
    result = fuse_risk_frame_v20(_frame(), _config(YAGER_RULE, 0.25))
    assert len(result) == 2
    assert set(result["v20_combination_rule"]) == {YAGER_RULE}
