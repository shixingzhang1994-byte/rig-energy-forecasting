from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


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


def _write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _verify_freeze(path: Path, seed: int) -> None:
    if not path.exists():
        raise RuntimeError("V9未冻结，拒绝运行留出种子")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if seed not in [int(value) for value in manifest.get("holdout_seeds", [])]:
        raise RuntimeError(f"留出种子 {seed} 不在冻结清单中")
    changed = []
    for item in manifest.get("frozen_files", []):
        target = PROJECT_DIR / item["path"]
        if not target.exists() or _sha256(target) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("冻结后文件发生变化，拒绝留出运行: " + ", ".join(changed))


def _prepare_independent_inputs(
    *, seed: int, source: Path, causal_root: Path, reuse_existing: bool
) -> tuple[Path, Path, Path]:
    """Prepare a seed-specific forecast and causal risk model without dispatch search."""
    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    v6 = yaml.safe_load(
        (PROJECT_DIR / "configs/v6_dispatch_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    forecast = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/synthetic_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5["forecast_overrides"],
    )
    forecast["project"]["seed"] = seed
    forecast["project"]["days"] = int(v6["protocol"]["simulation_days"])
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": v6["protocol"]["coverage_candidates"],
    }
    forecast_config = source / "resolved_forecast_config.yaml"
    _write_yaml(forecast, forecast_config)

    risk = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/risk_default.yaml").read_text(encoding="utf-8")
        ),
        v5["risk_overrides"],
    )
    risk["project"]["seed"] = seed
    risk["split"]["train_fraction"] = float(
        v6["protocol"]["risk_split"]["train_fraction"]
    )
    risk["split"]["val_fraction"] = float(
        v6["protocol"]["risk_split"]["val_fraction"]
    )
    risk_config = source / "resolved_risk_config.yaml"
    _write_yaml(risk, risk_config)

    data_path = PROJECT_DIR / f"data/processed/rig_load_v6_dev_{seed}.parquet"
    forecast_dir = source / "forecast"
    prediction_path = forecast_dir / "forecast_predictions.npz"
    key_path = forecast_dir / "forecast_prediction_keys.json"
    if not (reuse_existing and data_path.exists()):
        _run(
            "scripts/generate_synthetic.py",
            "--config",
            str(forecast_config),
            "--output",
            str(data_path),
            "--artifact-dir",
            str(source / "data"),
        )
    if not (reuse_existing and prediction_path.exists() and key_path.exists()):
        _run(
            "scripts/run_benchmark.py",
            "--data",
            str(data_path),
            "--config",
            str(forecast_config),
            "--artifact-dir",
            str(forecast_dir),
        )

    risk_dir = causal_root / "risk"
    risk_signal_path = risk_dir / "risk_predictions.csv"
    risk_metadata_path = risk_dir / "run_metadata.json"
    if not (
        reuse_existing and risk_signal_path.exists() and risk_metadata_path.exists()
    ):
        _run(
            "scripts/run_risk_benchmark.py",
            "--predictions",
            str(prediction_path),
            "--prediction-keys",
            str(key_path),
            "--data",
            str(data_path),
            "--forecast-config",
            str(forecast_config),
            "--risk-config",
            str(risk_config),
            "--artifact-dir",
            str(risk_dir),
        )
    return forecast_dir, risk_metadata_path, risk_signal_path


def _audit_trajectories(root: Path) -> dict:
    files = sorted(root.glob("trajectory_*.csv"))
    if not files:
        raise FileNotFoundError("V9调度没有生成轨迹文件")
    worst = {
        "trajectory_count": len(files),
        "max_power_balance_error_kw": 0.0,
        "max_grid_capacity_violation_kw": 0.0,
        "max_generator_capacity_violation_kw": 0.0,
        "max_storage_capacity_violation_kw": 0.0,
        "soc_violation_count": 0,
        "simultaneous_charge_discharge_count": 0,
        "noninteger_commitment_count": 0,
    }
    for path in files:
        frame = pd.read_csv(path)
        balance = (
            frame["grid_kw"]
            + frame["generator_kw"]
            + frame["storage_kw"]
            + frame["unserved_kw"]
            - frame["spill_kw"]
            - frame["load_kw"]
        )
        worst["max_power_balance_error_kw"] = max(
            worst["max_power_balance_error_kw"], float(balance.abs().max())
        )
        worst["max_grid_capacity_violation_kw"] = max(
            worst["max_grid_capacity_violation_kw"],
            float(
                np.maximum(
                    frame["grid_kw"] - frame["grid_available_capacity_kw"], 0.0
                ).max()
            ),
        )
        worst["max_generator_capacity_violation_kw"] = max(
            worst["max_generator_capacity_violation_kw"],
            float(
                np.maximum(
                    frame["generator_kw"]
                    - frame["generator_available_capacity_kw"],
                    0.0,
                ).max()
            ),
        )
        worst["max_storage_capacity_violation_kw"] = max(
            worst["max_storage_capacity_violation_kw"],
            float(
                np.maximum(
                    frame["storage_kw"].abs()
                    - frame["storage_available_power_kw"],
                    0.0,
                ).max()
            ),
        )
        worst["soc_violation_count"] += int(
            ((frame["soc"] < 0.15 - 1e-9) | (frame["soc"] > 0.90 + 1e-9)).sum()
        )
        worst["simultaneous_charge_discharge_count"] += int(
            ((frame["charge_kw"] > 1e-8) & (frame["discharge_kw"] > 1e-8)).sum()
        )
        worst["noninteger_commitment_count"] += int(
            (
                np.abs(
                    frame["generator_units_on"]
                    - np.rint(frame["generator_units_on"])
                )
                > 1e-9
            ).sum()
        )
    worst["physical_pass"] = bool(
        worst["max_power_balance_error_kw"] <= 1e-8
        and worst["max_grid_capacity_violation_kw"] <= 1e-8
        and worst["max_generator_capacity_violation_kw"] <= 1e-8
        and worst["max_storage_capacity_violation_kw"] <= 1e-8
        and worst["soc_violation_count"] == 0
        and worst["simultaneous_charge_discharge_count"] == 0
        and worst["noninteger_commitment_count"] == 0
    )
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description="运行V9唯一监督器的开发种子硬门槛")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    protocol_path = PROJECT_DIR / "configs/v9_causal_supervisor_development.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    first_seed = int(protocol["protocol"]["first_seed_only_until_gate_passes"])
    development_seeds = [
        int(seed) for seed in protocol["protocol"]["development_seeds"]
    ]
    holdout_seeds = [int(seed) for seed in protocol["protocol"]["holdout_seeds"]]
    if args.seed not in development_seeds + holdout_seeds:
        raise ValueError(
            f"只允许协议种子 {development_seeds + holdout_seeds}"
        )
    is_holdout = args.seed in holdout_seeds
    if args.seed != first_seed:
        first_gate_path = (
            PROJECT_DIR
            / f"artifacts/v9_causal_supervisor/seed_{first_seed}/first_seed_gate.json"
        )
        if not first_gate_path.exists() or not bool(
            json.loads(first_gate_path.read_text(encoding="utf-8")).get(
                "overall_pass"
            )
        ):
            raise RuntimeError("首开发种子未过硬门，拒绝运行第二开发种子")
    if is_holdout:
        second_seed = development_seeds[1]
        second_gate_path = (
            PROJECT_DIR
            / f"artifacts/v9_causal_supervisor/seed_{second_seed}/second_seed_gate.json"
        )
        if not second_gate_path.exists() or not bool(
            json.loads(second_gate_path.read_text(encoding="utf-8")).get(
                "overall_pass"
            )
        ):
            raise RuntimeError("第二开发种子未过硬门，拒绝运行留出种子")
        _verify_freeze(
            PROJECT_DIR / "artifacts/v9_causal_supervisor/freeze_manifest.json",
            args.seed,
        )

    causal_root = PROJECT_DIR / f"artifacts/v9_causal_supervisor/seed_{args.seed}"
    source = (
        causal_root / "source"
        if is_holdout
        else PROJECT_DIR / f"artifacts/v6_dispatch_development/seed_{args.seed}"
    )
    forecast_dir, risk_metadata_path, risk_signal_path = _prepare_independent_inputs(
        seed=args.seed,
        source=source,
        causal_root=causal_root,
        reuse_existing=args.reuse_existing,
    )
    required = [
        forecast_dir / "forecast_predictions.npz",
        forecast_dir / "forecast_prediction_keys.json",
        risk_metadata_path,
        risk_signal_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("V9开发种子输入缺失: " + ", ".join(missing))
    risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
    transition = risk_metadata.get("transition_feature", {})
    if transition.get("source") != "history_transition_flags" or not bool(
        transition.get("future_transition_flags_excluded")
    ):
        raise RuntimeError("风险特征未通过未来工况泄漏门槛")

    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    dispatch = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/dispatch_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5["dispatch_overrides"],
    )
    fixed = protocol["fixed_unit_commitment"]
    generator = dispatch["plant"]["generator"]
    for key in (
        "unit_count",
        "unit_rated_power_kw",
        "unit_minimum_stable_power_kw",
        "unit_ramp_kw_per_step",
        "minimum_up_steps",
        "minimum_down_steps",
    ):
        generator[key] = fixed[key]
    dispatch["cost"]["generator_startup_cost_yuan_per_unit"] = fixed[
        "startup_cost_yuan_per_unit"
    ]
    dispatch["supervisory_mpc"] = deepcopy(protocol["supervisor"])
    # V9只评估监督切换本身；不继承V8从泄漏风险结果观察到的预启映射。
    dispatch["scenario_cvar_mpc"]["risk_adaptive"][
        "minimum_committed_units_by_level"
    ] = {"0": 0, "1": 0, "2": 0, "3": 0}
    resolved_path = causal_root / "resolved_dispatch_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(dispatch, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dispatch_dir = causal_root / "dispatch"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    command = [
        sys.executable,
        "scripts/run_dispatch_benchmark.py",
        "--predictions",
        str(forecast_dir / "forecast_predictions.npz"),
        "--prediction-keys",
        str(forecast_dir / "forecast_prediction_keys.json"),
        "--config",
        str(resolved_path),
        "--risk-signals",
        str(risk_signal_path),
        "--artifact-dir",
        str(dispatch_dir),
    ]
    if args.reuse_existing and metrics_path.exists():
        print("REUSE", metrics_path, flush=True)
    else:
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_DIR, check=True)

    metrics = pd.read_csv(metrics_path)
    sum_columns = [
        "equivalent_cost_yuan",
        "risk_adjusted_cost_yuan",
        "unserved_energy_kwh",
        "generator_startups",
        "diesel_fuel_l",
        "solve_seconds",
    ]
    summary = metrics.groupby("method")[sum_columns].sum()
    summary["max_decision_seconds"] = metrics.groupby("method")[
        "max_decision_seconds"
    ].max()
    summary["p95_decision_seconds_max_scenario"] = metrics.groupby("method")[
        "p95_decision_seconds"
    ].max()
    summary["supervisory_protection_fraction_mean"] = metrics.groupby("method")[
        "supervisory_protection_fraction"
    ].mean()
    summary["supervisory_mode_switches"] = metrics.groupby("method")[
        "supervisory_mode_switches"
    ].sum()
    summary.to_csv(causal_root / "method_summary.csv")

    proposed_name = "Risk-SOC-Supervisory-MPC"
    proposed = summary.loc[proposed_name]
    rule = summary.loc["Rule-Based"]
    robust = summary.loc["ML-Robust-MPC"]
    gate_cfg = protocol["first_seed_gate"]
    physical = _audit_trajectories(dispatch_dir)
    safety_limit = min(
        float(rule["unserved_energy_kwh"]),
        float(robust["unserved_energy_kwh"]),
    ) + float(gate_cfg["unserved_tolerance_kwh"])
    safety_pass = float(proposed["unserved_energy_kwh"]) <= safety_limit
    cost_ratio = float(proposed["risk_adjusted_cost_yuan"]) / float(
        robust["risk_adjusted_cost_yuan"]
    )
    cost_pass = cost_ratio <= float(
        gate_cfg["maximum_risk_adjusted_cost_ratio_vs_ml_robust"]
    )
    latency_pass = float(proposed["max_decision_seconds"]) < float(
        gate_cfg["maximum_decision_seconds"]
    )
    dominated = bool(
        float(proposed["unserved_energy_kwh"])
        >= float(robust["unserved_energy_kwh"]) - 1e-9
        and float(proposed["risk_adjusted_cost_yuan"])
        >= float(robust["risk_adjusted_cost_yuan"]) - 1e-9
        and (
            float(proposed["unserved_energy_kwh"])
            > float(robust["unserved_energy_kwh"]) + 1e-9
            or float(proposed["risk_adjusted_cost_yuan"])
            > float(robust["risk_adjusted_cost_yuan"]) + 1e-9
        )
    )
    nondominance_pass = not dominated
    overall = bool(
        physical["physical_pass"]
        and safety_pass
        and cost_pass
        and latency_pass
        and nondominance_pass
    )
    gate = {
        "seed": args.seed,
        "role": (
            "sealed_holdout"
            if is_holdout
            else (
                "first_development_seed_only"
                if args.seed == first_seed
                else "second_development_confirmation_seed"
            )
        ),
        "risk_transition_feature": transition,
        "proposed_method": proposed_name,
        "safety_limit_kwh": safety_limit,
        "proposed_unserved_energy_kwh": float(proposed["unserved_energy_kwh"]),
        "rule_unserved_energy_kwh": float(rule["unserved_energy_kwh"]),
        "ml_robust_unserved_energy_kwh": float(robust["unserved_energy_kwh"]),
        "risk_adjusted_cost_ratio_vs_ml_robust": cost_ratio,
        "max_decision_seconds": float(proposed["max_decision_seconds"]),
        "physical_audit": physical,
        "checks": {
            "safety_noninferiority": safety_pass,
            "risk_adjusted_cost_cap": cost_pass,
            "real_time_latency": latency_pass,
            "not_dominated_by_ml_robust": nondominance_pass,
        },
        "overall_pass": overall,
        "next_action": (
            "report_all_holdouts_without_parameter_changes"
            if is_holdout
            else (
                "allow_generation_of_development_seed_20260902"
                if args.seed == first_seed
                else "allow_freeze_before_holdouts"
            )
            if overall
            else (
                "stop_and_archive_route; do_not_run_seed_20260902_or_holdouts"
                if args.seed == first_seed
                else "stop_and_revise_on_development_only; do_not_run_holdouts"
            )
        ),
        "reproduction_command": " ".join(command),
    }
    gate_path = causal_root / (
        "holdout_result.json"
        if is_holdout
        else (
            "first_seed_gate.json"
            if args.seed == first_seed
            else "second_seed_gate.json"
        )
    )
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_files = [
        protocol_path,
        PROJECT_DIR / "src/rig_energy/risk/experiment.py",
        PROJECT_DIR / "src/rig_energy/optimization/dispatch.py",
        forecast_dir / "forecast_prediction_keys.json",
        risk_metadata_path,
        risk_signal_path,
        resolved_path,
        metrics_path,
        causal_root / "method_summary.csv",
        gate_path,
    ]
    manifest = {
        "seed": args.seed,
        "files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_files
        ],
    }
    (causal_root / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
