from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_v20_seed_sets_are_disjoint_and_candidate_count_is_fixed() -> None:
    document = yaml.safe_load(
        (PROJECT_DIR / "configs/v20_evidential_calibration_protocol.yaml").read_text(encoding="utf-8")
    )
    protocol = document["protocol"]
    development = set(protocol["development_seed_order"])
    holdout = set(protocol["prospective_holdout_seed_order"])
    prohibited = set(protocol["overlap_prohibited_with_v19_seeds"])
    assert development.isdisjoint(holdout)
    assert development.isdisjoint(prohibited)
    assert holdout.isdisjoint(prohibited)
    space = document["candidate_space"]
    assert space["candidate_count"] == len(space["combination_rules"]) * len(
        space["scalar_temperature_grid"]
    )


def test_development_source_queue_matches_scientific_protocol() -> None:
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v20_evidential_calibration_protocol.yaml").read_text(encoding="utf-8")
    )["protocol"]
    source = yaml.safe_load(
        (PROJECT_DIR / "configs/v20_evidential_calibration_development_source.yaml").read_text(encoding="utf-8")
    )["protocol"]
    assert source["holdout_seed_order"] == protocol["development_seed_order"]
    assert source["required_eligible_seeds_per_phase"] == protocol[
        "required_eligible_development_seeds"
    ]


def test_temperature_grid_contains_identity_and_predeclared_sharpening_values() -> None:
    document = yaml.safe_load(
        (PROJECT_DIR / "configs/v20_evidential_calibration_protocol.yaml").read_text(encoding="utf-8")
    )
    grid = document["candidate_space"]["scalar_temperature_grid"]
    assert grid == [0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
    assert document["candidate_space"]["dispatch_uses"] == "raw_pignistic_probability"
