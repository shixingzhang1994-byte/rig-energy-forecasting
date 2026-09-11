from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v21_probability_holdouts import (  # noqa: E402
    PROTOCOL_PATH,
    SOURCE_CONFIG_PATH,
    bootstrap_mean_interval,
    exact_sign_flip_p,
)


def test_v21_source_queue_matches_scientific_protocol() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["protocol"]
    source = yaml.safe_load(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))["protocol"]
    assert source["holdout_seed_order"] == protocol["prospective_holdout_seed_order"]
    assert source["required_eligible_seeds_per_phase"] == protocol[
        "required_eligible_holdouts"
    ]


def test_v21_selected_candidate_matches_v20_descriptive_best() -> None:
    config = yaml.safe_load(
        (PROJECT_DIR / "configs/v21_selected_probability_candidate.yaml").read_text(encoding="utf-8")
    )
    assert config["source_v20_candidate_index"] == 9
    assert config["combination_rule"] == "dempster_normalized"
    assert config["calibration"]["temperature"] == pytest.approx(0.50)
    assert config["calibration"]["dispatch_uses"] == "raw_pignistic_probability"


def test_v21_statistics_are_deterministic_and_two_sided() -> None:
    assert exact_sign_flip_p(np.array([-1.0, -1.0])) == pytest.approx(0.5)
    assert exact_sign_flip_p(np.zeros(6)) == pytest.approx(1.0)
    first = bootstrap_mean_interval(np.array([-1.0, 0.0, 1.0]), resamples=1000, seed=9)
    second = bootstrap_mean_interval(np.array([-1.0, 0.0, 1.0]), resamples=1000, seed=9)
    assert first == second
