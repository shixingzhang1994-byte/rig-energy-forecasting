from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.data.public_calibration import (
    COMPONENT_COLUMNS,
    calibrate_synthetic_frame,
    load_public_calibration,
    validate_public_calibration,
)
from run_v14_dispatch_benchmark import _v14_fuel_lph  # noqa: E402
from run_v15_generator_first_diagnostic import (  # noqa: E402
    _generator_first_for_supervisory,
)
from rig_energy.optimization.dispatch import _load_config  # noqa: E402


CALIBRATION_PATH = PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml"


def _example_frame() -> pd.DataFrame:
    states = [
        "idle",
        "circulation",
        "drilling",
        "connection",
        "tripping",
        "maintenance",
    ]
    rows = []
    for state_index, state in enumerate(states):
        for index in range(20):
            total = 500.0 + 50.0 * state_index + 4.0 * index
            rows.append(
                {
                    "operation_state": state,
                    "auxiliary_power_kw": total * 0.25,
                    "mud_pump_power_kw": total * 0.30,
                    "topdrive_power_kw": total * 0.15,
                    "drawworks_power_kw": total * 0.20,
                    "other_power_kw": total * 0.10,
                    "physical_total_power_kw": total,
                    "total_active_power_kw": total,
                    "total_active_power_kw_raw": total,
                    "transient_event_kw": 80.0 if index == 10 else 0.0,
                    "quality_flag": "imputed" if index == 3 else "observed",
                    "source_type": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def test_public_calibration_evidence_schema_is_complete() -> None:
    config = load_public_calibration(CALIBRATION_PATH)
    audit = validate_public_calibration(config)
    assert audit.passed, audit.issues
    assert len(config["sources"]) >= 18


def test_calibration_preserves_balance_and_declares_synthetic_scope() -> None:
    config = load_public_calibration(CALIBRATION_PATH)
    calibrated, report = calibrate_synthetic_frame(
        _example_frame(), config, seed=20261001
    )
    component_total = calibrated.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1)
    assert np.allclose(component_total, calibrated["physical_total_power_kw"])
    assert set(calibrated["source_type"]) == {
        "public_evidence_calibrated_synthetic_v14"
    }
    assert report["outcome_data_used_for_calibration"] is False
    assert report["field_scada_claimed"] is False


def test_calibration_hits_state_targets_within_declared_ranges() -> None:
    config = load_public_calibration(CALIBRATION_PATH)
    calibrated, _ = calibrate_synthetic_frame(_example_frame(), config, seed=7)
    for state, target in config["load_calibration"]["state_targets"].items():
        values = calibrated.loc[
            calibrated["operation_state"] == state, "physical_total_power_kw"
        ]
        low, high = target["acceptance_mean_kw"]
        assert low <= values.mean() <= high
        assert values.max() <= target["hard_cap_kw"] + 1e-6


def test_calibration_keeps_raw_missingness_visible() -> None:
    config = load_public_calibration(CALIBRATION_PATH)
    calibrated, report = calibrate_synthetic_frame(_example_frame(), config, seed=11)
    assert calibrated["total_active_power_kw_raw"].isna().sum() == 6
    assert calibrated["total_active_power_kw"].notna().all()
    assert report["imputed_rows"] == 6


def test_v14_fuel_curve_matches_cat_prime_points() -> None:
    power = np.array([0.0, 120.0, 180.0, 240.0])
    assert np.allclose(_v14_fuel_lph(power, 240.0), [0.0, 33.8, 47.3, 62.5])
    assert np.all(np.diff(_v14_fuel_lph(power, 240.0)) >= 0.0)


def test_v15_generator_first_guard_is_explicit_in_trajectory_record() -> None:
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    command = {"grid_kw": 100.0, "generator_kw": 0.0, "charge_kw": 0.0, "discharge_kw": 0.0}
    executed, _ = _generator_first_for_supervisory(
        100.0,
        command,
        1300.0,
        0.65,
        0.0,
        plant,
        None,
        None,
        None,
        (),
        (),
        False,
        0,
        0.75,
        100.0,
    )
    assert executed["v15_generator_first_guard_active"] is True
