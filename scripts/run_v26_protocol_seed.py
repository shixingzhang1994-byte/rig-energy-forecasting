from __future__ import annotations

"""Run one frozen V26 heterogeneous parameter-set-by-seed unit."""

import json
import subprocess
import sys
import traceback
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v23_protocol_seed as implementation  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs/v26_heterogeneous_validation.yaml"
PARAMETER_ROOT = PROJECT_DIR / "artifacts/V26_heterogeneous_validation/parameter_sets"
DEADLINE_RUNNER = "scripts/run_v26_dispatch_benchmark.py"
_ORIGINAL_SUBPROCESS_RUN = subprocess.run


def argument_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def parameter_record(seed: int) -> tuple[dict, Path]:
    manifest = json.loads((PARAMETER_ROOT / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["records"] if int(item["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} has no unique V26 parameter assignment")
    record = matches[0]
    calibration_path = PROJECT_DIR / record["calibration_path"]
    dispatch_path = PROJECT_DIR / record["dispatch_path"]
    if implementation._sha256(calibration_path) != record["calibration_sha256"]:
        raise RuntimeError("V26 materialized calibration hash mismatch")
    if implementation._sha256(dispatch_path) != record["dispatch_sha256"]:
        raise RuntimeError("V26 materialized dispatch hash mismatch")
    return record, calibration_path


def prepare_inputs(
    seed: int,
    days: int,
    root: Path,
    reuse: bool,
    protocol_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    record, calibration_path = parameter_record(seed)
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    v5 = yaml.safe_load((PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8"))
    v6 = yaml.safe_load((PROJECT_DIR / "configs/v6_dispatch_development.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8"))
    forecast = implementation.legacy_runner._deep_merge(base, calibration["synthetic_overrides"])
    forecast = implementation.legacy_runner._deep_merge(forecast, v5["forecast_overrides"])
    forecast["project"].update({"seed": int(seed), "days": int(days)})
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": v6["protocol"]["coverage_candidates"],
    }
    forecast["metadata"] = {"v26_parameter_unit": record["unit_id"]}
    forecast_config = root / "source/resolved_forecast_config.yaml"
    implementation.legacy_runner._write_yaml(forecast, forecast_config)

    risk = implementation.legacy_runner._deep_merge(
        yaml.safe_load((PROJECT_DIR / "configs/risk_default.yaml").read_text(encoding="utf-8")),
        v5["risk_overrides"],
    )
    risk = implementation.legacy_runner._deep_merge(
        risk, calibration.get("v26_risk_overrides", {})
    )
    risk["project"]["seed"] = int(seed)
    risk["split"]["train_fraction"] = float(v6["protocol"]["risk_split"]["train_fraction"])
    risk["split"]["val_fraction"] = float(v6["protocol"]["risk_split"]["val_fraction"])
    risk["metadata"] = {"v26_parameter_unit": record["unit_id"]}
    risk_config = root / "source/resolved_risk_config.yaml"
    implementation.legacy_runner._write_yaml(risk, risk_config)

    data_path = PROJECT_DIR / f"data/processed/rig_load_v26_{seed}.parquet"
    data_dir = root / "source/data"
    forecast_dir = root / "source/forecast"
    predictions = forecast_dir / "forecast_predictions.npz"
    keys = forecast_dir / "forecast_prediction_keys.json"
    residuals = forecast_dir / "validation_residual_library.npz"
    residual_metadata = forecast_dir / "validation_residual_library.json"
    if not (reuse and data_path.exists()):
        implementation._run(
            "scripts/generate_v14_calibrated.py", "--calibration", str(calibration_path),
            "--seed", str(seed), "--days", str(days), "--output", str(data_path),
            "--artifact-dir", str(data_dir),
        )
        implementation._run(
            "scripts/audit_v14_calibration.py", "--data", str(data_path),
            "--calibration", str(calibration_path), "--artifact-dir", str(data_dir / "audit"),
        )
    if not (reuse and predictions.exists() and keys.exists() and residuals.exists() and residual_metadata.exists()):
        implementation._run(
            "scripts/run_benchmark.py", "--data", str(data_path), "--config", str(forecast_config),
            "--artifact-dir", str(forecast_dir), "--export-validation-residuals",
        )

    calibration_manifest = root / "scenario_calibration_manifest.json"
    implementation._run(
        "scripts/run_v22_scenario_calibration.py", "--residuals", str(residuals),
        "--metadata", str(residual_metadata), "--protocol", str(protocol_path),
        "--output", str(calibration_manifest),
    )
    risk_dir = root / "risk"
    risk_signals = risk_dir / "risk_predictions.csv"
    risk_metadata = risk_dir / "run_metadata.json"
    if not (reuse and risk_signals.exists() and risk_metadata.exists()):
        implementation._run(
            "scripts/run_risk_benchmark.py", "--predictions", str(predictions),
            "--prediction-keys", str(keys), "--data", str(data_path),
            "--forecast-config", str(forecast_config), "--risk-config", str(risk_config),
            "--artifact-dir", str(risk_dir),
        )
    (root / "parameter_unit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return forecast_dir, risk_signals, risk_metadata, residuals, calibration_manifest


def write_preparation_failure(exc: BaseException) -> None:
    seed_text = argument_value("--seed")
    if seed_text is None:
        return
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    root = PROJECT_DIR / "artifacts" / document["protocol"]["artifact_namespace"] / f"seed_{int(seed_text)}"
    root.mkdir(parents=True, exist_ok=True)
    failure = {
        "seed": int(seed_text),
        "phase": "heterogeneous_confirmatory",
        "retained_regardless_of_outcome_direction": True,
        "technical_gate_pass": False,
        "failure_stage": "input_preparation",
        "failure_type": type(exc).__name__,
        "controller_outcomes_generated": False,
        "traceback": "".join(traceback.format_exception(exc)),
    }
    (root / "result.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)


def _augment_result_with_deadline_audit(seed: int, root: Path) -> None:
    result_path = root / "result.json"
    audit_path = root / "dispatch_matched/deadline_audit.json"
    if not result_path.exists() or not audit_path.exists():
        raise RuntimeError("V26 result or deadline audit is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    policy_checks = {
        "deadline_audit_completed": bool(audit["completed_without_exception"]),
        "no_rejected_deadline_incumbent": int(audit["rejected_deadline_count"]) == 0,
        "accepted_incumbents_independently_feasible": float(
            audit["maximum_accepted_incumbent_violation"]
        )
        <= 1.0e-6,
    }
    result["phase"] = "heterogeneous_confirmatory"
    result["technical_checks"].update(policy_checks)
    result["technical_gate_pass"] = all(result["technical_checks"].values())
    result["deadline_policy_audit"] = {
        key: audit[key]
        for key in (
            "call_count",
            "deadline_count",
            "accepted_feasible_incumbent_count",
            "rejected_deadline_count",
            "maximum_solver_elapsed_seconds",
            "maximum_accepted_incumbent_violation",
        )
    }
    result["deadline_audit_path"] = str(audit_path.relative_to(PROJECT_DIR))
    result["reproduction_command"] = result["reproduction_command"].replace(
        "scripts/run_v22_dispatch_benchmark.py", DEADLINE_RUNNER
    ) + f" --deadline-audit {audit_path}"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_path = root / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [PROJECT_DIR / item["path"] for item in manifest["files"]]
    if audit_path not in paths:
        paths.append(audit_path)
    manifest["files"] = [
        {
            "path": str(path.relative_to(PROJECT_DIR)),
            "sha256": implementation._sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["technical_gate_pass"]:
        raise SystemExit(2)


def main() -> None:
    seed_text = argument_value("--seed")
    if seed_text is None:
        raise SystemExit("--seed is required")
    seed = int(seed_text)
    record, _ = parameter_record(seed)
    implementation.DEFAULT_PROTOCOL = PROTOCOL_PATH
    implementation.DISPATCH_CONFIG = PROJECT_DIR / record["dispatch_path"]
    implementation._prepare_inputs = prepare_inputs
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    root = (
        PROJECT_DIR
        / "artifacts"
        / document["protocol"]["artifact_namespace"]
        / f"seed_{seed}"
    )
    deadline_audit = root / "dispatch_matched/deadline_audit.json"

    def run_with_deadline_policy(command, *args, **kwargs):
        rewritten = list(command) if isinstance(command, (list, tuple)) else command
        if (
            isinstance(rewritten, list)
            and len(rewritten) > 1
            and rewritten[1] == "scripts/run_v22_dispatch_benchmark.py"
        ):
            rewritten[1] = DEADLINE_RUNNER
            rewritten.extend(["--deadline-audit", str(deadline_audit)])
        return _ORIGINAL_SUBPROCESS_RUN(rewritten, *args, **kwargs)

    implementation.subprocess.run = run_with_deadline_policy
    try:
        implementation.main()
        _augment_result_with_deadline_audit(seed, root)
    except SystemExit:
        raise
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        write_preparation_failure(exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
