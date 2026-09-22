from __future__ import annotations

"""Freeze the V23 implementation after development and before holdout replay."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402
from run_v23_protocol_seed import audit_startup_delay  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs/v23_submission_revision.yaml"
DISPATCH_PATH = PROJECT_DIR / "configs/v23_dispatch_matched.yaml"
RUNNER_PATHS = [PROJECT_DIR / "scripts/run_v23_protocol_seed.py"]
ANALYSIS_PATHS = [PROJECT_DIR / "scripts/analyze_v23_confirmatory.py"]
FREEZE_ENTRY_PATH = Path(__file__).resolve()
DEVELOPMENT_ROOT = (
    PROJECT_DIR / "artifacts/V23_submission_revision/development/seed_20261008"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_DIR)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    version = str(protocol["id"]).split("-", 1)[0]
    metrics_path = DEVELOPMENT_ROOT / "dispatch_matched/dispatch_metrics.csv"
    calibration_path = DEVELOPMENT_ROOT / "scenario_calibration_manifest.json"
    risk_metadata_path = DEVELOPMENT_ROOT / "risk/run_metadata.json"
    if not metrics_path.exists() or not calibration_path.exists() or not risk_metadata_path.exists():
        raise RuntimeError(f"{version} development dispatch/calibration is incomplete")

    metrics = pd.read_csv(metrics_path)
    expected_methods = set(document["matched_comparison"]["methods"])
    if set(metrics["method"]) != expected_methods:
        raise RuntimeError(f"{version} development method set differs from protocol")
    if metrics.groupby("scenario")["method"].nunique().min() != len(expected_methods):
        raise RuntimeError(f"{version} development method-by-scenario matrix is incomplete")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
    physical = _audit_trajectories(metrics_path.parent)
    dispatch = yaml.safe_load(
        DISPATCH_PATH.read_text(
            encoding="utf-8"
        )
    )
    startup_delay = audit_startup_delay(metrics_path.parent, dispatch)
    proposed = metrics[metrics["method"] == "Full-Risk-SOC-Supervisory-MILP"]
    development_checks = {
        "validation_only_residuals": bool(calibration["leakage_gate_pass"]),
        "cvar_not_single_scenario": not bool(
            calibration["cvar_discretization_audit"][
                "collapses_to_single_worst_scenario"
            ]
        ),
        "complete_method_scenario_matrix": int(len(metrics))
        == len(expected_methods) * int(metrics["scenario"].nunique()),
        "constraint_audit": bool(physical["physical_pass"]),
        "startup_delay_audit": bool(startup_delay["pass"]),
        "real_time_latency": float(proposed["max_decision_seconds"].max())
        < float(document["hard_gates"]["maximum_decision_seconds"]),
        "risk_validation_policy_resolved": (
            not bool(risk_metadata.get("validation_gate_fallback_active"))
            or document.get("risk_validation_fallback", {}).get("action")
            == "set every dispatch-facing risk level to severe for that seed"
        ),
    }
    development_seed = int(DEVELOPMENT_ROOT.name.split("_")[-1])
    development_result = {
        "seed": development_seed,
        "role": "development_only_previously_revealed_seed",
        "confirmatory_evidence": False,
        "checks": development_checks,
        "technical_gate_pass": all(development_checks.values()),
        "proposed_eens_kwh": float(proposed["unserved_energy_kwh"].sum()),
        "proposed_operating_cost_yuan": float(
            proposed["realized_operating_cost_yuan"].sum()
        ),
        "proposed_max_decision_seconds": float(
            proposed["max_decision_seconds"].max()
        ),
        "physical_audit": physical,
        "startup_delay_audit": startup_delay,
        "risk_validation_fallback_active": bool(
            risk_metadata.get("validation_gate_fallback_active")
        ),
        "effective_dispatch_guard_model": risk_metadata.get("dispatch_guard_model"),
    }
    development_result_path = DEVELOPMENT_ROOT / "development_result.json"
    development_result_path.write_text(
        json.dumps(development_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not development_result["technical_gate_pass"]:
        raise RuntimeError(f"{version} development technical gate failed; refusing freeze")

    frozen_paths = [
        PROTOCOL_PATH,
        DISPATCH_PATH,
        PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml",
        PROJECT_DIR / "configs/v5_acceptance.yaml",
        PROJECT_DIR / "configs/v6_dispatch_development.yaml",
        PROJECT_DIR / "configs/synthetic_default.yaml",
        PROJECT_DIR / "configs/risk_default.yaml",
        PROJECT_DIR / "configs/dispatch_default.yaml",
        PROJECT_DIR / "environment.yml",
        PROJECT_DIR / "scripts/generate_v14_calibrated.py",
        PROJECT_DIR / "scripts/audit_v14_calibration.py",
        PROJECT_DIR / "scripts/run_benchmark.py",
        PROJECT_DIR / "scripts/run_risk_benchmark.py",
        PROJECT_DIR / "scripts/run_v14_dispatch_benchmark.py",
        PROJECT_DIR / "scripts/run_v22_scenario_calibration.py",
        PROJECT_DIR / "scripts/run_v22_dispatch_benchmark.py",
        *RUNNER_PATHS,
        *ANALYSIS_PATHS,
        Path(__file__).resolve(),
        FREEZE_ENTRY_PATH,
        *sorted((PROJECT_DIR / "src/rig_energy").rglob("*.py")),
    ]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    freeze_path = artifact_root / "freeze_manifest.json"
    if freeze_path.exists():
        raise RuntimeError(f"{version} freeze manifest already exists; refusing overwrite")
    manifest = {
        "protocol_id": protocol["id"],
        "status": f"frozen_before_any_prospective_{version}_outcome",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "prospective_seed_queue": [
            int(value) for value in protocol["prospective_holdout_seed_order"]
        ],
        "required_eligible_holdouts": int(protocol["required_eligible_holdouts"]),
        "stopping_rule": protocol["stopping_rule"],
        "primary_contrasts": document["matched_comparison"]["primary_contrasts"],
        "development_record": _record(development_result_path),
        "development_calibration": _record(calibration_path),
        "development_risk_metadata": _record(risk_metadata_path),
        "development_metrics": _record(metrics_path),
        "frozen_files": [_record(path) for path in frozen_paths],
        "post_freeze_rule": (
            "Any frozen-file hash change invalidates prospective status. Retain "
            "every upstream-eligible seed in queue order regardless of outcome."
        ),
    }
    freeze_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
