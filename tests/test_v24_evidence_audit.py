from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_v24_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_v24_evidence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_trajectory_path_does_not_use_ambiguous_glob(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "dispatch_matched"
        / "trajectory_A_stable_drilling_residual_cvar_milp.csv"
    )
    assert MODULE.trajectory_path(
        tmp_path, "A_stable_drilling", "Residual-CVaR-MILP"
    ) == expected
    assert "risk_adaptive" not in expected.name


def test_paired_summary_retains_adverse_seeds() -> None:
    summary = MODULE.paired_summary(np.asarray([-3.0, -1.0, 2.0, 0.0]))
    assert summary["improve_count"] == 2
    assert summary["tie_count"] == 1
    assert summary["worsen_count"] == 1
    assert summary["seed_count"] == 4
