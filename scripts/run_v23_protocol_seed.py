from __future__ import annotations

"""Run one prospective V23 seed in the prospectively frozen queue.

Eligibility is written before dispatch.  Controller outcomes never determine
whether a seed is retained, and failed/null/adverse completed results remain in
the confirmatory namespace.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v10_protocol_seed as legacy_runner  # noqa: E402
import run_v14_protocol_seed as v14_runner  # noqa: E402
from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v23_submission_revision.yaml"
DISPATCH_CONFIG = PROJECT_DIR / "configs/v23_dispatch_matched.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def audit_startup_delay(dispatch_dir: Path, dispatch: dict) -> dict[str, object]:
    delay = int(dispatch["robust_mpc"].get("generator_startup_delay_steps", 0))
    selection = json.loads(
        (dispatch_dir / "scenario_selection.json").read_text(encoding="utf-8")
    )["scenarios"]
    violations = 0
    audited_rows = 0
    for scenario, metadata in selection.items():
        initial = [
            int(value)
            for value in metadata.get(
                "comparison_initial_start_command_history", []
            )
        ]
        if delay and len(initial) != delay:
            raise RuntimeError("startup-delay audit lacks complete initial queue")
        for path in dispatch_dir.glob(f"trajectory_{scenario}_*.csv"):
            frame = pd.read_csv(path)
            commands = frame["generator_start_command_units"].to_numpy(dtype=int)
            synchronized = frame["generator_synchronized_units"].to_numpy(dtype=int)
            matured = (
                pd.Series(initial + commands.tolist()).iloc[: len(frame)].to_numpy(int)
                if delay
                else commands
            )
            violations += int((synchronized > matured).sum())
            audited_rows += int(len(frame))
    return {
        "startup_delay_steps": delay,
        "audited_rows": audited_rows,
        "early_synchronization_violations": violations,
        "pass": violations == 0,
    }


def _verify_queue(seed: int, protocol: dict, artifact_root: Path) -> None:
    queue = [int(value) for value in protocol["prospective_holdout_seed_order"]]
    if seed not in queue:
        raise ValueError(f"seed {seed} is not in the frozen V23 holdout queue")
    retained = 0
    required = int(protocol["required_eligible_holdouts"])
    for earlier in queue[: queue.index(seed)]:
        earlier_root = artifact_root / f"seed_{earlier}"
        eligibility_path = earlier_root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"process earlier V23 seed {earlier} first")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if eligibility.get("eligible"):
            retained += 1
    if retained >= required:
        raise RuntimeError(
            f"the first {required} eligible V23 seeds are already complete"
        )


def _verify_freeze(seed: int, protocol_path: Path, artifact_root: Path) -> dict:
    freeze_path = artifact_root / "freeze_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("V23 is not frozen; refusing prospective execution")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise RuntimeError("V23 protocol changed after freeze")
    if seed not in [int(value) for value in manifest["prospective_seed_queue"]]:
        raise RuntimeError("seed is absent from the frozen V23 queue")
    changed = []
    for item in manifest["frozen_files"]:
        path = PROJECT_DIR / item["path"]
        if not path.exists() or _sha256(path) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("V23 frozen files changed: " + ", ".join(changed))
    return manifest


def _prepare_inputs(
    seed: int,
    days: int,
    root: Path,
    reuse: bool,
    protocol_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    v6 = yaml.safe_load(
        (PROJECT_DIR / "configs/v6_dispatch_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    calibration_path = PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(
        (PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8")
    )
    forecast = legacy_runner._deep_merge(base, calibration["synthetic_overrides"])
    forecast = legacy_runner._deep_merge(forecast, v5["forecast_overrides"])
    forecast["project"]["seed"] = int(seed)
    forecast["project"]["days"] = int(days)
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": v6["protocol"]["coverage_candidates"],
    }
    forecast_config = root / "source/resolved_forecast_config.yaml"
    legacy_runner._write_yaml(forecast, forecast_config)

    risk = legacy_runner._deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/risk_default.yaml").read_text(encoding="utf-8")
        ),
        v5["risk_overrides"],
    )
    risk["project"]["seed"] = int(seed)
    risk["split"]["train_fraction"] = float(
        v6["protocol"]["risk_split"]["train_fraction"]
    )
    risk["split"]["val_fraction"] = float(
        v6["protocol"]["risk_split"]["val_fraction"]
    )
    risk_config = root / "source/resolved_risk_config.yaml"
    legacy_runner._write_yaml(risk, risk_config)

    protocol_document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_version = str(protocol_document["protocol"]["id"]).split("-", 1)[0].lower()
    data_path = PROJECT_DIR / f"data/processed/rig_load_{protocol_version}_{seed}.parquet"
    data_dir = root / "source/data"
    forecast_dir = root / "source/forecast"
    predictions = forecast_dir / "forecast_predictions.npz"
    keys = forecast_dir / "forecast_prediction_keys.json"
    residuals = forecast_dir / "validation_residual_library.npz"
    residual_metadata = forecast_dir / "validation_residual_library.json"
    if not (reuse and data_path.exists()):
        _run(
            "scripts/generate_v14_calibrated.py",
            "--calibration",
            str(calibration_path),
            "--seed",
            str(seed),
            "--days",
            str(days),
            "--output",
            str(data_path),
            "--artifact-dir",
            str(data_dir),
        )
        _run(
            "scripts/audit_v14_calibration.py",
            "--data",
            str(data_path),
            "--calibration",
            str(calibration_path),
            "--artifact-dir",
            str(data_dir / "audit"),
        )
    if not (
        reuse
        and predictions.exists()
        and keys.exists()
        and residuals.exists()
        and residual_metadata.exists()
    ):
        _run(
            "scripts/run_benchmark.py",
            "--data",
            str(data_path),
            "--config",
            str(forecast_config),
            "--artifact-dir",
            str(forecast_dir),
            "--export-validation-residuals",
        )

    calibration_manifest = root / "scenario_calibration_manifest.json"
    _run(
        "scripts/run_v22_scenario_calibration.py",
        "--residuals",
        str(residuals),
        "--metadata",
        str(residual_metadata),
        "--protocol",
        str(protocol_path),
        "--output",
        str(calibration_manifest),
    )

    risk_dir = root / "risk"
    risk_signals = risk_dir / "risk_predictions.csv"
    risk_metadata = risk_dir / "run_metadata.json"
    if not (reuse and risk_signals.exists() and risk_metadata.exists()):
        _run(
            "scripts/run_risk_benchmark.py",
            "--predictions",
            str(predictions),
            "--prediction-keys",
            str(keys),
            "--data",
            str(data_path),
            "--forecast-config",
            str(forecast_config),
            "--risk-config",
            str(risk_config),
            "--artifact-dir",
            str(risk_dir),
        )
    return forecast_dir, risk_signals, risk_metadata, residuals, calibration_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen V23 holdout seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    _verify_queue(args.seed, protocol, artifact_root)
    _verify_freeze(args.seed, protocol_path, artifact_root)

    # Hold this advisory lock for the entire seed, including forecasting. This
    # makes the wall-clock dispatch gate reproducible and prevents the V22
    # contention failure from recurring if several V23 commands are launched.
    artifact_root.mkdir(parents=True, exist_ok=True)
    serial_lock = (artifact_root / ".serial_execution.lock").open("a+")
    fcntl.flock(serial_lock.fileno(), fcntl.LOCK_EX)

    root = artifact_root / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    forecast_dir, risk_signals, risk_metadata_path, residuals, calibration_manifest = (
        _prepare_inputs(
            args.seed,
            int(protocol["simulation_days"]),
            root,
            args.reuse_existing,
            protocol_path,
        )
    )
    risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
    transition = risk_metadata.get("transition_feature", {})
    if transition.get("source") != "history_transition_flags" or not bool(
        transition.get("future_transition_flags_excluded")
    ):
        raise RuntimeError("V23 risk features failed the causal-information gate")

    dispatch = yaml.safe_load(DISPATCH_CONFIG.read_text(encoding="utf-8"))
    eligibility = legacy_runner._eligibility(
        forecast_dir / "forecast_predictions.npz",
        risk_signals,
        dispatch,
        args.seed,
        "confirmatory",
    )
    eligibility["outcomes_inspected_before_eligibility"] = False
    eligibility_path = root / "eligibility.json"
    eligibility_path.write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(eligibility, ensure_ascii=False, indent=2), flush=True)
    if not eligibility["eligible"]:
        raise SystemExit(3)

    dispatch_dir = root / "dispatch_matched"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    command = [
        sys.executable,
        "scripts/run_v22_dispatch_benchmark.py",
        "--predictions",
        str(forecast_dir / "forecast_predictions.npz"),
        "--prediction-keys",
        str(forecast_dir / "forecast_prediction_keys.json"),
        "--config",
        str(DISPATCH_CONFIG),
        "--risk-signals",
        str(risk_signals),
        "--scenario-residuals",
        str(residuals),
        "--scenario-seed",
        str(args.seed),
        "--artifact-dir",
        str(dispatch_dir),
    ]
    if args.reuse_existing and metrics_path.exists():
        print("REUSE", metrics_path, flush=True)
    else:
        print("RUN", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            failure = {
                "seed": int(args.seed),
                "phase": "confirmatory",
                "retained_regardless_of_outcome_direction": True,
                "technical_gate_pass": False,
                "performance_gate_applied": False,
                "failure_stage": "dispatch_benchmark",
                "failure_type": "subprocess_nonzero_exit",
                "returncode": int(exc.returncode),
                "partial_trajectory_count": len(list(dispatch_dir.glob("trajectory_*.csv"))),
                "reproduction_command": " ".join(command),
            }
            (root / "result.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
            raise SystemExit(2) from exc

    metrics = pd.read_csv(metrics_path)
    sum_columns = [
        "unserved_energy_kwh",
        "loss_of_load_duration_hours",
        "realized_operating_cost_yuan",
        "terminal_energy_adjustment_yuan",
        "reliability_penalty_yuan",
        "total_social_cost_yuan",
        "diesel_fuel_l",
        "generator_startups",
        "solve_seconds",
    ]
    summary = metrics.groupby("method")[sum_columns].sum()
    summary["minimum_soc_pct"] = metrics.groupby("method")["minimum_soc_pct"].min()
    summary["max_decision_seconds"] = metrics.groupby("method")[
        "max_decision_seconds"
    ].max()
    summary.to_csv(root / "method_summary.csv")

    physical = _audit_trajectories(dispatch_dir)
    startup_delay_audit = audit_startup_delay(dispatch_dir, dispatch)
    calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
    proposed_name = "Full-Risk-SOC-Supervisory-MILP"
    checks = {
        "eligible_before_outcomes": bool(eligibility["eligible"]),
        "validation_only_residuals": bool(calibration["leakage_gate_pass"]),
        "cvar_discretization": not bool(
            calibration["cvar_discretization_audit"][
                "collapses_to_single_worst_scenario"
            ]
        ),
        "constraint_audit": bool(physical["physical_pass"]),
        "startup_delay_audit": bool(startup_delay_audit["pass"]),
        "real_time_latency": float(summary.loc[proposed_name, "max_decision_seconds"])
        < float(document["hard_gates"]["maximum_decision_seconds"]),
    }
    result = {
        "seed": int(args.seed),
        "phase": "confirmatory",
        "retained_regardless_of_outcome_direction": True,
        "proposed_method": proposed_name,
        "technical_checks": checks,
        "technical_gate_pass": all(checks.values()),
        "performance_gate_applied": False,
        "performance_direction": {
            baseline: {
                "delta_eens_kwh": float(
                    summary.loc[proposed_name, "unserved_energy_kwh"]
                    - summary.loc[baseline, "unserved_energy_kwh"]
                ),
                "delta_operating_cost_yuan": float(
                    summary.loc[proposed_name, "realized_operating_cost_yuan"]
                    - summary.loc[baseline, "realized_operating_cost_yuan"]
                ),
            }
            for baseline in [
                "Rule-Based",
                "Point-Forecast-MILP",
                "Risk-Adaptive-Residual-CVaR-MILP",
            ]
        },
        "physical_audit": physical,
        "startup_delay_audit": startup_delay_audit,
        "reproduction_command": " ".join(command),
    }
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_paths = [
        protocol_path,
        DISPATCH_CONFIG,
        eligibility_path,
        risk_metadata_path,
        risk_signals,
        residuals,
        calibration_manifest,
        metrics_path,
        root / "method_summary.csv",
        result_path,
    ]
    (root / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "files": [
                    {
                        "path": str(path.relative_to(PROJECT_DIR)),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in manifest_paths
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["technical_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
