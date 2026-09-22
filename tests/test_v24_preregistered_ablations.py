from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_v24_preregistered_ablations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v24_preregistered_ablations", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_retains_null_and_adverse_component_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "PROJECT_DIR", tmp_path)
    contrasts = {
        "residual_scenarios": {
            "full_minus_ablation_eens_kwh": -2.0,
            "full_minus_ablation_operating_cost_yuan": 3.0,
        },
        "risk_adaptive_cvar": {
            "full_minus_ablation_eens_kwh": 0.0,
            "full_minus_ablation_operating_cost_yuan": -1.0,
        },
        "generator_first_execution": {
            "full_minus_ablation_eens_kwh": 1.5,
            "full_minus_ablation_operating_cost_yuan": 0.5,
        },
    }
    results = [
        {"seed": 1, "technical_gate_pass": True, "contrasts": contrasts},
        {"seed": 2, "technical_gate_pass": True, "contrasts": contrasts},
    ]

    manifest = MODULE.aggregate(results, tmp_path, "test-protocol")
    effects = pd.read_csv(tmp_path / "seed_component_effects.csv")

    assert manifest["all_null_and_adverse_results_retained"] is True
    assert manifest["seed_count"] == 2
    assert set(effects["component"]) == {
        "residual_scenarios",
        "risk_adaptive_cvar",
        "generator_first_execution",
    }
    assert effects.loc[
        effects["component"] == "risk_adaptive_cvar",
        "full_minus_ablation_eens_kwh",
    ].eq(0.0).all()
    assert effects.loc[
        effects["component"] == "generator_first_execution",
        "full_minus_ablation_eens_kwh",
    ].eq(1.5).all()
