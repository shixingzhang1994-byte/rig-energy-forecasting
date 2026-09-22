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
sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_v9_supervisor_first_seed import _audit_trajectories, _deep_merge
from rig_energy.data.schema import STATE_TO_CODE  # noqa: E402


DEFAULT_PROTOCOL_PATH = PROJECT_DIR / "configs/v10_evaluable_holdout.yaml"


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


def _verify_phase_order(seed: int, protocol: dict, artifact_root: Path) -> str:
    development = [int(v) for v in protocol["development_seed_order"]]
    holdout = [int(v) for v in protocol["holdout_seed_order"]]
    if seed in development:
        phase, order = "development", development
    elif seed in holdout:
        phase, order = "holdout", holdout
    else:
        raise ValueError(f"种子 {seed} 不在V10预声明队列中")
    required = int(protocol["required_eligible_seeds_per_phase"])
    eligible_before = 0
    for earlier in order[: order.index(seed)]:
        root = artifact_root / f"seed_{earlier}"
        eligibility_path = root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"必须先按顺序处理种子 {earlier}")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if eligibility.get("eligible"):
            eligible_before += 1
            if not (root / "result.json").exists():
                raise RuntimeError(f"前序可评估种子 {earlier} 尚未完成")
    if eligible_before >= required:
        raise RuntimeError(f"{phase}已取得{required}个最先可评估种子，拒绝继续挑选")
    return phase


def _verify_protocol_freeze(seed: int, artifact_root: Path) -> None:
    path = artifact_root / "freeze_manifest.json"
    if not path.exists():
        raise RuntimeError("开发阶段尚未冻结，拒绝留出运行")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if seed not in [int(v) for v in manifest["holdout_seed_order"]]:
        raise RuntimeError("种子不在冻结留出队列")
    changed = []
    for item in manifest["frozen_files"]:
        target = PROJECT_DIR / item["path"]
        if not target.exists() or _sha256(target) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("冻结文件变化: " + ", ".join(changed))


def _prepare_inputs(seed: int, days: int, root: Path, reuse: bool) -> tuple[Path, Path, Path]:
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
    forecast["project"]["days"] = days
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": v6["protocol"]["coverage_candidates"],
    }
    forecast_config = root / "source/resolved_forecast_config.yaml"
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
    risk_config = root / "source/resolved_risk_config.yaml"
    _write_yaml(risk, risk_config)

    data_path = PROJECT_DIR / f"data/processed/rig_load_v10_{seed}.parquet"
    forecast_dir = root / "source/forecast"
    predictions = forecast_dir / "forecast_predictions.npz"
    keys = forecast_dir / "forecast_prediction_keys.json"
    if not (reuse and data_path.exists()):
        _run(
            "scripts/generate_synthetic.py",
            "--config",
            str(forecast_config),
            "--output",
            str(data_path),
            "--artifact-dir",
            str(root / "source/data"),
        )
    if not (reuse and predictions.exists() and keys.exists()):
        _run(
            "scripts/run_benchmark.py",
            "--data",
            str(data_path),
            "--config",
            str(forecast_config),
            "--artifact-dir",
            str(forecast_dir),
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
    return forecast_dir, risk_signals, risk_metadata


def _build_dispatch(protocol: dict) -> dict:
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
    dispatch["scenario_cvar_mpc"]["risk_adaptive"][
        "minimum_committed_units_by_level"
    ] = {"0": 0, "1": 0, "2": 0, "3": 0}
    return dispatch


def _eligibility(
    predictions: Path, risk_signals: Path, dispatch: dict, seed: int, phase: str
) -> dict:
    arrays = np.load(predictions)
    risk = pd.read_csv(risk_signals)
    indices = risk["sample_index"].to_numpy(dtype=int)
    states = arrays["operation_state_codes"][indices]
    transitions = arrays["transition_flags"][indices].astype(bool)
    supply = risk["supply_regime"].astype(str).to_numpy()
    length = int(dispatch["dispatch"]["scenario_steps"])
    stride = max(1, length // 20)
    scenario_results = {}
    for scenario, spec in dispatch["scenario_selection"]["scenarios"].items():
        primary = [STATE_TO_CODE[str(v)] for v in spec["primary_states"]]
        anchors = [STATE_TO_CODE[str(v)] for v in spec.get("anchor_states", [])]
        acceptable = [str(v) for v in spec.get("acceptable_supply_regimes", [])]
        rows = []
        for start in range(0, len(states) - length + 1, stride):
            window_states = states[start : start + length]
            window_supply = supply[start : start + length]
            rows.append(
                {
                    "start": start,
                    "primary_fraction": float(np.isin(window_states, primary).mean()),
                    "anchor_fraction": (
                        float(np.isin(window_states, anchors).mean()) if anchors else 0.0
                    ),
                    "acceptable_supply_fraction": float(
                        np.isin(window_supply, acceptable).mean()
                    ),
                    "emergency_supply_fraction": float(
                        (window_supply == "emergency").mean()
                    ),
                    "transition_count": int(
                        transitions[start : start + length].sum()
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        passed = frame[
            (frame["primary_fraction"] + 1e-12 >= float(spec["minimum_primary_fraction"]))
            & (
                (not anchors)
                | (frame["anchor_fraction"] + 1e-12 >= float(spec.get("minimum_anchor_fraction", 0.0)))
            )
            & (frame["acceptable_supply_fraction"] + 1e-12 >= float(spec["minimum_supply_fraction"]))
            & (frame["emergency_supply_fraction"] <= float(spec["maximum_emergency_fraction"]) + 1e-12)
        ]
        scenario_results[scenario] = {
            "candidate_window_count": int(len(frame)),
            "passing_window_count": int(len(passed)),
            "maximum_primary_fraction": float(frame["primary_fraction"].max()),
            "maximum_anchor_fraction": float(frame["anchor_fraction"].max()),
            "maximum_acceptable_supply_fraction": float(
                frame["acceptable_supply_fraction"].max()
            ),
            "minimum_emergency_supply_fraction": float(
                frame["emergency_supply_fraction"].min()
            ),
        }
    return {
        "seed": seed,
        "phase": phase,
        "timing": "before_dispatch_and_before_any_controller_outcome",
        "effective_dispatch_samples": int(len(states)),
        "effective_dispatch_hours": float(len(states) * 5.0 / 3600.0),
        "scenarios": scenario_results,
        "eligible": all(v["passing_window_count"] > 0 for v in scenario_results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按预声明顺序运行V10单种子")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    full_protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = full_protocol["protocol"]
    artifact_namespace = str(
        protocol.get("artifact_namespace", "v10_evaluable_holdout")
    )
    artifact_root = PROJECT_DIR / "artifacts" / artifact_namespace
    phase = _verify_phase_order(args.seed, protocol, artifact_root)
    if phase == "holdout":
        _verify_protocol_freeze(args.seed, artifact_root)

    root = artifact_root / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    forecast_dir, risk_signals, risk_metadata_path = _prepare_inputs(
        args.seed,
        int(protocol["simulation_days"]),
        root,
        args.reuse_existing,
    )
    risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
    transition = risk_metadata.get("transition_feature", {})
    if transition.get("source") != "history_transition_flags" or not bool(
        transition.get("future_transition_flags_excluded")
    ):
        raise RuntimeError("V10风险特征未通过因果门")

    dispatch = _build_dispatch(full_protocol)
    dispatch_path = root / "resolved_dispatch_config.yaml"
    _write_yaml(dispatch, dispatch_path)
    eligibility = _eligibility(
        forecast_dir / "forecast_predictions.npz",
        risk_signals,
        dispatch,
        args.seed,
        phase,
    )
    eligibility_path = root / "eligibility.json"
    eligibility_path.write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(eligibility, ensure_ascii=False, indent=2), flush=True)
    if not eligibility["eligible"]:
        raise SystemExit(3)

    dispatch_dir = root / "dispatch"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    command = [
        sys.executable,
        "scripts/run_dispatch_benchmark.py",
        "--predictions",
        str(forecast_dir / "forecast_predictions.npz"),
        "--prediction-keys",
        str(forecast_dir / "forecast_prediction_keys.json"),
        "--config",
        str(dispatch_path),
        "--risk-signals",
        str(risk_signals),
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
    summary.to_csv(root / "method_summary.csv")
    proposed_name = "Risk-SOC-Supervisory-MPC"
    proposed = summary.loc[proposed_name]
    rule = summary.loc["Rule-Based"]
    robust = summary.loc["ML-Robust-MPC"]
    gate = full_protocol["performance_gate"]
    tolerance = float(gate["aggregate_unserved_tolerance_kwh"])
    safety_pass = float(proposed["unserved_energy_kwh"]) <= min(
        float(rule["unserved_energy_kwh"]), float(robust["unserved_energy_kwh"])
    ) + tolerance
    scenario_table = metrics.pivot(
        index="scenario", columns="method", values="unserved_energy_kwh"
    )
    regret_references = [
        str(value)
        for value in gate.get(
            "scenario_regret_reference_methods", ["ML-Robust-MPC"]
        )
    ]
    reference_by_scenario = scenario_table[regret_references].min(axis=1)
    regrets = (scenario_table[proposed_name] - reference_by_scenario).to_dict()
    worst_regret = max(float(v) for v in regrets.values())
    regret_cap = float(
        gate.get(
            "maximum_single_scenario_unserved_regret_kwh",
            gate.get("maximum_single_scenario_unserved_regret_kwh_vs_ml_robust"),
        )
    )
    scenario_regret_pass = worst_regret <= regret_cap
    cost_ratio = float(proposed["risk_adjusted_cost_yuan"]) / float(
        robust["risk_adjusted_cost_yuan"]
    )
    cost_pass = cost_ratio <= float(
        gate["maximum_risk_adjusted_cost_ratio_vs_ml_robust"]
    )
    latency_pass = float(proposed["max_decision_seconds"]) < float(
        gate["maximum_decision_seconds"]
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
    physical = _audit_trajectories(dispatch_dir)
    checks = {
        "aggregate_safety_noninferiority": safety_pass,
        "single_scenario_regret_cap": scenario_regret_pass,
        "risk_adjusted_cost_cap": cost_pass,
        "real_time_latency": latency_pass,
        "not_dominated_by_ml_robust": not dominated,
        "physical_feasibility": bool(physical["physical_pass"]),
    }
    overall = all(checks.values())
    result = {
        "seed": args.seed,
        "phase": phase,
        "eligible": True,
        "risk_transition_feature": transition,
        "proposed_method": proposed_name,
        "proposed_unserved_energy_kwh": float(proposed["unserved_energy_kwh"]),
        "rule_unserved_energy_kwh": float(rule["unserved_energy_kwh"]),
        "ml_robust_unserved_energy_kwh": float(robust["unserved_energy_kwh"]),
        "risk_adjusted_cost_ratio_vs_ml_robust": cost_ratio,
        "scenario_regret_reference_methods": regret_references,
        "scenario_unserved_regret_kwh_vs_best_reference": {
            key: float(value) for key, value in regrets.items()
        },
        "worst_scenario_unserved_regret_kwh_vs_ml_robust": worst_regret,
        "max_decision_seconds": float(proposed["max_decision_seconds"]),
        "physical_audit": physical,
        "checks": checks,
        "overall_pass": overall,
        "reproduction_command": " ".join(command),
    }
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_paths = [
        protocol_path,
        eligibility_path,
        dispatch_path,
        risk_metadata_path,
        risk_signals,
        metrics_path,
        root / "method_summary.csv",
        result_path,
    ]
    (root / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
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
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
