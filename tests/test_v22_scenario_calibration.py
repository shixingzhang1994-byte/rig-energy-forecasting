from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import json
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.optimization.scenario_calibration import (  # noqa: E402
    audit_cvar_discretization,
    build_residual_scenarios,
    build_validation_residual_library,
    write_validation_residual_artifact,
)
from run_v22_scenario_calibration import build_calibration_manifest  # noqa: E402


def test_seven_equal_scenarios_reject_high_confidence_cvar_collapse() -> None:
    probabilities = np.full(7, 1.0 / 7.0)

    with pytest.raises(ValueError, match="effective tail"):
        audit_cvar_discretization(
            probabilities,
            alpha=0.90,
            minimum_effective_tail_scenarios=2.0,
        )


def test_one_hundred_scenarios_resolve_the_95_percent_tail() -> None:
    audit = audit_cvar_discretization(
        np.full(100, 0.01),
        alpha=0.95,
        minimum_effective_tail_scenarios=5.0,
    )

    assert audit["scenario_count"] == 100
    assert audit["tail_probability"] == pytest.approx(0.05)
    assert audit["effective_tail_scenarios"] == pytest.approx(5.0)
    assert audit["collapses_to_single_worst_scenario"] is False


def test_residual_library_refuses_test_targets() -> None:
    actual = np.arange(12, dtype=float).reshape(3, 4)
    prediction = actual - 1.0

    with pytest.raises(ValueError, match="validation"):
        build_validation_residual_library(
            actual,
            prediction,
            split_name="test",
            warmup_samples=0,
        )


def test_residual_library_applies_causal_warmup_and_records_provenance() -> None:
    actual = np.arange(20, dtype=float).reshape(5, 4)
    prediction = actual - np.array([1.0, 2.0, 3.0, 4.0])

    residuals, metadata = build_validation_residual_library(
        actual,
        prediction,
        split_name="validation",
        warmup_samples=2,
        source_artifact="validation_predictions.npz",
    )

    assert residuals.shape == (3, 4)
    assert np.allclose(residuals, [1.0, 2.0, 3.0, 4.0])
    assert metadata["split_name"] == "validation"
    assert metadata["test_targets_used"] is False
    assert metadata["excluded_warmup_samples"] == 2


def test_block_residual_scenarios_are_deterministic_and_preserve_horizon_vectors() -> None:
    point = np.full((3, 4), 100.0)
    residual_library = np.array(
        [
            [-4.0, -3.0, -2.0, -1.0],
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 6.0, 8.0, 10.0],
            [12.0, 9.0, 6.0, 3.0],
            [20.0, 10.0, 0.0, -10.0],
            [30.0, 20.0, 10.0, 0.0],
        ]
    )

    first, first_probabilities, first_indices = build_residual_scenarios(
        point,
        residual_library,
        scenario_count=5,
        seed=20260914,
    )
    second, second_probabilities, second_indices = build_residual_scenarios(
        point,
        residual_library,
        scenario_count=5,
        seed=20260914,
    )

    assert first.shape == (3, 5, 4)
    assert np.array_equal(first, second)
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first_probabilities, second_probabilities)
    assert first_probabilities.sum() == pytest.approx(1.0)
    for origin in range(len(point)):
        for scenario in range(5):
            sampled_residual = first[origin, scenario] - point[origin]
            assert np.array_equal(
                sampled_residual,
                residual_library[first_indices[origin, scenario]],
            )


def test_residual_scenarios_reject_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="horizon"):
        build_residual_scenarios(
            np.ones((2, 4)),
            np.ones((8, 3)),
            scenario_count=5,
            seed=1,
        )


def test_validation_artifact_contains_residuals_and_explicit_split_audit(
    tmp_path: Path,
) -> None:
    actual = np.arange(20, dtype=float).reshape(5, 4)
    prediction = actual - 2.5

    npz_path, metadata_path = write_validation_residual_artifact(
        tmp_path,
        actual,
        prediction,
        warmup_samples=1,
        source_model="Causal-ErrorFeedback-Ensemble",
        source_config="configs/v22_submission_revision.yaml",
    )

    arrays = np.load(npz_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert arrays.files == ["validation_residuals_kw"]
    assert arrays["validation_residuals_kw"].shape == (4, 4)
    assert metadata["split_name"] == "validation"
    assert metadata["test_targets_used"] is False
    assert metadata["source_model"] == "Causal-ErrorFeedback-Ensemble"


def test_v22_protocol_selected_cvar_passes_resolution_gate() -> None:
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v22_submission_revision.yaml").read_text(
            encoding="utf-8"
        )
    )
    scenario = protocol["scenario_generation"]
    cvar = protocol["cvar"]
    audit = audit_cvar_discretization(
        np.full(
            int(scenario["scenario_count"]),
            1.0 / int(scenario["scenario_count"]),
        ),
        alpha=float(cvar["primary_alpha"]),
        minimum_effective_tail_scenarios=float(
            cvar["minimum_effective_tail_scenarios"]
        ),
    )
    assert audit["effective_tail_scenarios"] >= 5.0 - 1e-12


def test_v22_calibration_manifest_passes_only_audited_validation_artifact(
    tmp_path: Path,
) -> None:
    actual = np.arange(2400, dtype=float).reshape(200, 12) + 1000.0
    prediction = actual - np.linspace(-5.0, 15.0, 12)
    residual_path, metadata_path = write_validation_residual_artifact(
        tmp_path,
        actual,
        prediction,
        warmup_samples=12,
        source_model="Causal-ErrorFeedback-Ensemble",
        source_config="configs/v22_submission_revision.yaml",
    )

    manifest = build_calibration_manifest(
        residual_path,
        metadata_path,
        PROJECT_DIR / "configs/v22_submission_revision.yaml",
    )

    assert manifest["leakage_gate_pass"] is True
    assert manifest["scenario_generation"]["scenario_count"] == 50
    assert manifest["cvar_discretization_audit"][
        "effective_tail_scenarios"
    ] == pytest.approx(5.0)
    coverage = manifest["internal_validation_coverage_audit"]
    assert coverage["test_targets_used"] is False
    assert coverage["calibration_sample_count"] == 94
    assert coverage["evaluation_sample_count"] == 94
    assert 0.0 <= coverage["central_90_percent"]["overall"] <= 1.0


def test_v22_dispatch_config_matches_protocol_tail_and_terminal_policy() -> None:
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v22_submission_revision.yaml").read_text(
            encoding="utf-8"
        )
    )
    dispatch = yaml.safe_load(
        (PROJECT_DIR / "configs/v22_dispatch_matched.yaml").read_text(
            encoding="utf-8"
        )
    )
    scenario_count = dispatch["scenario_cvar_mpc"]["residual_scenario_count"]
    assert scenario_count == protocol["scenario_generation"]["scenario_count"]
    assert (
        dispatch["robust_mpc"]["matched_terminal_soc"][
            "terminal_soc_drop_by_level"
        ]
        == protocol["matched_comparison"]["terminal_soc_drop_by_risk_level"]
    )
    alphas = {
        dispatch["scenario_cvar_mpc"]["alpha"],
        *dispatch["scenario_cvar_mpc"]["risk_adaptive"][
            "alpha_by_level"
        ].values(),
    }
    for alpha in alphas:
        audit_cvar_discretization(
            np.full(scenario_count, 1.0 / scenario_count),
            alpha=float(alpha),
            minimum_effective_tail_scenarios=5.0,
        )
