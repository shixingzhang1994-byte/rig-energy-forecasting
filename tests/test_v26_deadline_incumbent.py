from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts/run_v26_dispatch_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_v26_dispatch_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_incumbent_accepts_feasible_integer_solution() -> None:
    audited = MODULE.audit_incumbent(
        np.array([1.0, 0.25]),
        integrality=np.array([1, 0]),
        bounds=Bounds([0.0, 0.0], [2.0, 1.0]),
        constraints=[LinearConstraint([[1.0, 1.0]], [1.0], [1.5])],
    )
    assert audited["feasible"] is True
    assert audited["maximum_violation"] == 0.0


def test_audit_incumbent_rejects_fractional_integer_solution() -> None:
    audited = MODULE.audit_incumbent(
        np.array([1.2, 0.25]),
        integrality=np.array([1, 0]),
        bounds=Bounds([0.0, 0.0], [2.0, 1.0]),
        constraints=[LinearConstraint([[1.0, 1.0]], [0.0], [2.0])],
    )
    assert audited["feasible"] is False
    assert audited["integrality_violation"] > 0.1


def test_audit_incumbent_rejects_constraint_violation() -> None:
    audited = MODULE.audit_incumbent(
        np.array([1.0, 0.75]),
        integrality=np.array([1, 0]),
        bounds=Bounds([0.0, 0.0], [2.0, 1.0]),
        constraints=[LinearConstraint([[1.0, 1.0]], [0.0], [1.5])],
    )
    assert audited["feasible"] is False
    assert audited["constraint_upper_violation"] > 0.2


def test_audit_incumbent_rejects_missing_solution() -> None:
    audited = MODULE.audit_incumbent(
        None,
        integrality=np.array([1]),
        bounds=Bounds([0.0], [1.0]),
        constraints=[LinearConstraint([[1.0]], [0.0], [1.0])],
    )
    assert audited == {
        "available": False,
        "feasible": False,
        "reason": "missing_incumbent",
    }
