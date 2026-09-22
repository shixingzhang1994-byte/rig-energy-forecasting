from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_v26_factorial_assignment_is_complete_and_unique() -> None:
    family = yaml.safe_load(
        (PROJECT_DIR / "configs/v26_heterogeneous_family.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v26_heterogeneous_validation.yaml").read_text(
            encoding="utf-8"
        )
    )["protocol"]
    units = family["prospective_units"]
    combinations = {
        (unit["load"], unit["transition"], unit["plant"]) for unit in units
    }

    assert len(units) == 12
    assert len(combinations) == 12
    assert [unit["seed"] for unit in units] == protocol[
        "prospective_holdout_seed_order"
    ]


def test_materialized_parameter_sets_were_created_without_outcomes() -> None:
    manifest = json.loads(
        (
            PROJECT_DIR
            / "artifacts/V26_heterogeneous_validation/parameter_sets/manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["prospective_unit_count"] == 12
    assert manifest["outcomes_used"] is False
    assert all(record["outcomes_used"] is False for record in manifest["records"])


def test_v26_deadline_policy_reserves_overhead_and_forbids_optimality_claim() -> None:
    document = yaml.safe_load(
        (PROJECT_DIR / "configs/v26_heterogeneous_validation.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy = document["real_time_milp_policy"]

    assert policy["solver_deadline_seconds"] < policy["controller_period_seconds"]
    assert (
        policy["solver_deadline_seconds"]
        + policy["reserved_python_and_safety_overhead_seconds"]
        == policy["controller_period_seconds"]
    )
    assert policy["missing_or_infeasible_incumbent_action"] == "hard technical failure"
    assert policy["optimality_claim_for_deadline_incumbent"] == "prohibited"
