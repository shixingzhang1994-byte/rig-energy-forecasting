from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PAPER_DIR / "experiments/results/v21_probability_calibration"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v21_paper_audit_reconstructs_frozen_seed_set() -> None:
    audit = json.loads((RESULT_DIR / "v21_paper_audit.json").read_text(encoding="utf-8"))
    assert audit["pass"] is True
    assert audit["eligible_seeds"] == [
        20261040,
        20261041,
        20261042,
        20261045,
        20261046,
        20261047,
    ]
    assert audit["runner_reporting_issue"]["effect_on_gate_or_status"].startswith("none")


def test_v21_effect_directions_and_control_boundary() -> None:
    frame = pd.read_csv(RESULT_DIR / "v21_seed_metrics.csv")
    assert len(frame) == 6
    assert (frame["calibration_brier_delta"] < 0.0).all()
    assert (frame["rule_brier_delta"] < 0.0).all()
    assert (frame["candidate_macro_f1_delta_vs_rss"] > 0.0).all()
    assert abs(
        frame["DC_RSS_unserved_kwh"].sum() - frame["RSS_unserved_kwh"].sum()
    ) <= 1e-12
    assert frame["candidate_action_changed_rows_vs_rss"].sum() == 57
    assert frame["candidate_action_changed_rows_vs_yager"].sum() == 9


def test_v21_paper_manifest_hashes_match() -> None:
    manifest = json.loads((RESULT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = Path(item["path"])
        assert path.exists()
        assert path.stat().st_size == item["size_bytes"]
        assert sha256(path) == item["sha256"]
