from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1]
MODULE_PATH = EXPERIMENTS / "summarize_v19_evidential_risk.py"
SPEC = importlib.util.spec_from_file_location("v19_evidential_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_sign_flip_has_expected_six_seed_resolution() -> None:
    assert MODULE.exact_sign_flip(np.ones(6)) == 2.0 / (2**6)
    assert MODULE.exact_sign_flip(np.zeros(6)) == 1.0


def test_frozen_manifest_hashes_match() -> None:
    audit = MODULE.verify_manifest()
    assert audit["declared_file_count"] == 20
    assert audit["hash_failures"] == []
    assert audit["pass"] is True


def test_archived_paper_audit_closes_all_predeclared_checks() -> None:
    path = EXPERIMENTS / "results/v19_evidential_risk/v19_paper_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["seeds"] == MODULE.EXPECTED_SEEDS
    assert all(audit["checks"].values())


def test_probability_audit_uses_documented_float_tolerance() -> None:
    audits = [MODULE.probability_audit(seed) for seed in MODULE.EXPECTED_SEEDS]
    assert all(audit["sum_tolerance"] == 1e-6 for audit in audits)
    assert all(audit["pass"] is True for audit in audits)
    assert max(audit["evidential_max_abs_sum_error"] for audit in audits) < 1e-12
