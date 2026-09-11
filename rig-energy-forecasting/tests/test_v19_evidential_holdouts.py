from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v19_evidential_holdouts import (  # noqa: E402
    ACTION_COLUMNS,
    HOLDOUT_CONFIG_PATH,
    PROTOCOL_PATH,
    bootstrap_mean_interval,
    compare_actions,
    exact_sign_flip_p,
    scenario_selection_matches,
)


def test_holdout_execution_queue_matches_declared_v19_protocol() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    holdout = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert holdout["protocol"]["holdout_seed_order"] == protocol["protocol"][
        "prospective_holdout_seed_order"
    ]
    assert holdout["protocol"]["required_eligible_seeds_per_phase"] == protocol[
        "protocol"
    ]["required_eligible_holdouts"]


def test_exact_sign_flip_preserves_ties_and_is_two_sided() -> None:
    assert exact_sign_flip_p(np.zeros(6)) == pytest.approx(1.0)
    assert exact_sign_flip_p(np.array([-1.0, -1.0])) == pytest.approx(0.5)


def test_seed_bootstrap_is_deterministic() -> None:
    values = np.array([-2.0, 0.0, 1.0, -1.0])
    first = bootstrap_mean_interval(values, resamples=2000, seed=20260908)
    second = bootstrap_mean_interval(values, resamples=2000, seed=20260908)
    assert first == second
    assert first[0] <= float(values.mean()) <= first[1]


def _trajectory(value: float) -> pd.DataFrame:
    data = {column: np.full(4, value) for column in ACTION_COLUMNS}
    return pd.DataFrame(data)


def test_action_comparison_counts_changed_physical_rows(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    evidential = tmp_path / "evidential"
    baseline.mkdir()
    evidential.mkdir()
    for index in range(3):
        name = f"trajectory_{index}_risk_soc_supervisory_mpc.csv"
        base = _trajectory(0.0)
        changed = base.copy()
        if index == 1:
            changed.loc[2, "generator_kw"] = 10.0
        base.to_csv(baseline / name, index=False)
        changed.to_csv(evidential / name, index=False)
    result = compare_actions(baseline, evidential)
    assert result["trajectory_count"] == 3
    assert result["row_count"] == 12
    assert result["physical_action_changed_rows"] == 1
    assert result["physical_action_change_rate"] == pytest.approx(1.0 / 12.0)


def test_scenario_selection_requires_identical_windows(tmp_path: Path) -> None:
    payload = {
        "scenarios": {
            "A": {"start_index": 10, "stop_index": 20, "warmup_steps": 2}
        }
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(payload), encoding="utf-8")
    right.write_text(json.dumps(payload), encoding="utf-8")
    assert scenario_selection_matches(left, right)
    changed = json.loads(right.read_text(encoding="utf-8"))
    changed["scenarios"]["A"]["start_index"] = 11
    right.write_text(json.dumps(changed), encoding="utf-8")
    assert not scenario_selection_matches(left, right)
