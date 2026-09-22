from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_v25_factorial_assignment_is_complete_and_unique() -> None:
    family = yaml.safe_load(
        (PROJECT_DIR / "configs/v25_heterogeneous_family.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v25_heterogeneous_validation.yaml").read_text(
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
            / "artifacts/V25_heterogeneous_validation/parameter_sets/manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["prospective_unit_count"] == 12
    assert manifest["outcomes_used"] is False
    assert all(record["outcomes_used"] is False for record in manifest["records"])

