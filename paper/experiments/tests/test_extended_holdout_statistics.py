from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_extended_holdouts.py"
SPEC = importlib.util.spec_from_file_location("holdout_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_sign_flip_resolution_for_consistent_nonzero_effects() -> None:
    summary = MODULE.summarize([-1.0] * 10, np.random.default_rng(1))
    assert summary["exact_two_sided_sign_flip_p"] == 2.0 / (2**10)
    assert summary["nonpositive_count"] == 10


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = MODULE.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_solver_scale_residuals_are_reported_as_zero() -> None:
    assert MODULE.clean_numerical_zero(2.7e-15) == 0.0
    assert MODULE.clean_numerical_zero(-7.1e-15) == 0.0
    assert MODULE.clean_numerical_zero(1e-6) == 1e-6
