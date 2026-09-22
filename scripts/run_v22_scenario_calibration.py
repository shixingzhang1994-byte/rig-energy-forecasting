from __future__ import annotations

"""Audit and freeze a validation-only residual scenario calibration bundle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.optimization.scenario_calibration import (  # noqa: E402
    audit_cvar_discretization,
    build_residual_scenarios,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_calibration_manifest(
    residual_path: Path,
    metadata_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    residual_file = Path(residual_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_file.read_text(encoding="utf-8"))
    if metadata.get("split_name") != "validation":
        raise ValueError("residual source must be explicitly labeled validation")
    if metadata.get("test_targets_used") is not False:
        raise ValueError("residual source must explicitly exclude test targets")

    arrays = np.load(residual_file)
    if arrays.files != ["validation_residuals_kw"]:
        raise ValueError(
            "residual artifact must contain only validation_residuals_kw"
        )
    residuals = np.asarray(arrays["validation_residuals_kw"], dtype=float)
    if residuals.ndim != 2 or min(residuals.shape) < 1:
        raise ValueError("validation residual library must be a non-empty matrix")
    if not np.isfinite(residuals).all():
        raise ValueError("validation residual library contains non-finite values")
    if int(metadata.get("residual_sample_count", -1)) != len(residuals):
        raise ValueError("residual sample count differs from metadata")
    if int(metadata.get("horizon_steps", -1)) != residuals.shape[1]:
        raise ValueError("residual horizon differs from metadata")

    scenario_cfg = protocol["scenario_generation"]
    cvar_cfg = protocol["cvar"]
    scenario_count = int(scenario_cfg["scenario_count"])
    probabilities = np.full(scenario_count, 1.0 / scenario_count)
    cvar_audit = audit_cvar_discretization(
        probabilities,
        alpha=float(cvar_cfg["primary_alpha"]),
        minimum_effective_tail_scenarios=float(
            cvar_cfg["minimum_effective_tail_scenarios"]
        ),
    )

    # Small deterministic construction audit; full scenarios are generated only
    # for selected dispatch windows to avoid a redundant hundreds-of-MB file.
    smoke_point = np.full((3, residuals.shape[1]), np.median(residuals) + 2000.0)
    _, smoke_probabilities, smoke_indices = build_residual_scenarios(
        smoke_point,
        residuals,
        scenario_count=scenario_count,
        seed=int(scenario_cfg["random_seed"]),
        minimum_load_kw=float(scenario_cfg["nonnegative_load_floor_kw"]),
    )
    adjacent_correlation = None
    if residuals.shape[1] > 1:
        left = residuals[:, :-1].ravel()
        right = residuals[:, 1:].ravel()
        if left.std() > 0.0 and right.std() > 0.0:
            adjacent_correlation = float(np.corrcoef(left, right)[0, 1])

    quantiles = np.quantile(residuals, [0.01, 0.05, 0.50, 0.95, 0.99])
    internal_cut = len(residuals) // 2
    if internal_cut < 2 or len(residuals) - internal_cut < 1:
        raise ValueError("validation residual library is too short for coverage audit")
    calibration_residuals = residuals[:internal_cut]
    evaluation_residuals = residuals[internal_cut:]
    lead_quantiles = {
        probability: np.quantile(calibration_residuals, probability, axis=0)
        for probability in (0.05, 0.10, 0.90, 0.95)
    }

    def central_coverage(lower: float, upper: float) -> dict[str, object]:
        covered = (
            (evaluation_residuals >= lead_quantiles[lower])
            & (evaluation_residuals <= lead_quantiles[upper])
        )
        per_lead = covered.mean(axis=0)
        return {
            "nominal": float(upper - lower),
            "overall": float(covered.mean()),
            "minimum_across_leads": float(per_lead.min()),
            "maximum_across_leads": float(per_lead.max()),
        }

    def upper_exceedance(probability: float) -> dict[str, object]:
        exceeded = evaluation_residuals > lead_quantiles[probability]
        per_lead = exceeded.mean(axis=0)
        return {
            "nominal": float(1.0 - probability),
            "overall": float(exceeded.mean()),
            "minimum_across_leads": float(per_lead.min()),
            "maximum_across_leads": float(per_lead.max()),
        }
    return {
        "protocol_id": protocol["protocol"]["id"],
        "status": (
            "validation_calibration_audited_before_"
            + str(protocol["protocol"]["id"]).split("-")[0]
            + "_controller_outcomes"
        ),
        "residual_source": {
            "path": str(residual_file),
            "sha256": _sha256(residual_file),
            "metadata_path": str(metadata_file),
            "metadata_sha256": _sha256(metadata_file),
            "split_name": "validation",
            "test_targets_used": False,
            "source_model": metadata.get("source_model"),
            "sample_count": int(len(residuals)),
            "horizon_steps": int(residuals.shape[1]),
        },
        "protocol": {
            "path": str(protocol_file),
            "sha256": _sha256(protocol_file),
        },
        "scenario_generation": {
            "method": scenario_cfg["method"],
            "scenario_count": scenario_count,
            "random_seed": int(scenario_cfg["random_seed"]),
            "complete_horizon_vector_sampling": True,
            "full_scenario_array_persisted": False,
            "generation_scope": scenario_cfg["output_mode"],
            "probability_sum": float(smoke_probabilities.sum()),
            "smoke_unique_residual_rows": int(len(np.unique(smoke_indices))),
        },
        "cvar_discretization_audit": cvar_audit,
        "residual_diagnostics_kw": {
            "mean": float(residuals.mean()),
            "standard_deviation": float(residuals.std()),
            "minimum": float(residuals.min()),
            "maximum": float(residuals.max()),
            "quantiles": {
                key: float(value)
                for key, value in zip(
                    ["q01", "q05", "q50", "q95", "q99"], quantiles
                )
            },
            "adjacent_horizon_correlation": adjacent_correlation,
        },
        "internal_validation_coverage_audit": {
            "scope": "chronological_first_half_calibration_second_half_evaluation_within_validation_only",
            "test_targets_used": False,
            "calibration_sample_count": int(len(calibration_residuals)),
            "evaluation_sample_count": int(len(evaluation_residuals)),
            "central_80_percent": central_coverage(0.10, 0.90),
            "central_90_percent": central_coverage(0.05, 0.95),
            "upper_90_percent_quantile_exceedance": upper_exceedance(0.90),
            "upper_95_percent_quantile_exceedance": upper_exceedance(0.95),
        },
        "leakage_gate_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/v22_submission_revision.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_calibration_manifest(
        args.residuals,
        args.metadata,
        args.protocol,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "existing V22 calibration manifest differs; use a new output namespace"
            )
    else:
        output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
