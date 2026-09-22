from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="运行V8风险条件机组可达性开发实验")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v8_reachability_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    allowed = set(protocol["protocol"]["development_seeds"])
    if args.seed not in allowed:
        raise ValueError(f"本脚本只允许开发种子 {sorted(allowed)}")
    candidates = protocol["candidates"]
    if args.candidate:
        requested = set(args.candidate)
        known = {item["name"] for item in candidates}
        if requested - known:
            raise ValueError(f"未知候选: {sorted(requested - known)}")
        candidates = [item for item in candidates if item["name"] in requested]

    source = PROJECT_DIR / f"artifacts/v6_dispatch_development/seed_{args.seed}"
    forecast = source / "forecast"
    risk = source / "risk"
    required = [
        forecast / "forecast_predictions.npz",
        forecast / "forecast_prediction_keys.json",
        risk / "risk_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("冻结预测/风险输出缺失: " + ", ".join(missing))

    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    config_base = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/dispatch_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5["dispatch_overrides"],
    )
    fixed = protocol["protocol"]["fixed_unit_commitment"]
    root = PROJECT_DIR / f"artifacts/v8_reachability/seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        config = deepcopy(config_base)
        generator = config["plant"]["generator"]
        for key in (
            "unit_count",
            "unit_rated_power_kw",
            "unit_minimum_stable_power_kw",
            "unit_ramp_kw_per_step",
            "minimum_up_steps",
            "minimum_down_steps",
        ):
            generator[key] = fixed[key]
        config["cost"]["generator_startup_cost_yuan_per_unit"] = fixed[
            "startup_cost_yuan_per_unit"
        ]
        config["scenario_cvar_mpc"]["risk_adaptive"][
            "minimum_committed_units_by_level"
        ] = candidate["minimum_committed_units_by_level"]
        config_path = root / f"resolved_{candidate['name']}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        artifact_dir = root / candidate["name"]
        metrics_path = artifact_dir / "dispatch_metrics.csv"
        command = [
            sys.executable,
            "scripts/run_dispatch_benchmark.py",
            "--predictions",
            str(forecast / "forecast_predictions.npz"),
            "--prediction-keys",
            str(forecast / "forecast_prediction_keys.json"),
            "--config",
            str(config_path),
            "--risk-signals",
            str(risk / "risk_predictions.csv"),
            "--artifact-dir",
            str(artifact_dir),
        ]
        if args.reuse_existing and metrics_path.exists():
            print("REUSE", metrics_path, flush=True)
        else:
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
        metrics = pd.read_csv(metrics_path)
        aggregate = metrics.groupby("method")[
            [
                "equivalent_cost_yuan",
                "risk_adjusted_cost_yuan",
                "unserved_energy_kwh",
                "generator_startups",
                "diesel_fuel_l",
                "solve_seconds",
            ]
        ].sum()
        proposed = aggregate.loc["Risk-Adaptive-CVaR-MPC"]
        rule = aggregate.loc["Rule-Based"]
        rows.append(
            {
                "seed": args.seed,
                "name": candidate["name"],
                "minimum_committed_units_by_level": json.dumps(
                    candidate["minimum_committed_units_by_level"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **{f"proposed_{key}": float(value) for key, value in proposed.items()},
                **{f"rule_{key}": float(value) for key, value in rule.items()},
                "unserved_noninferior": float(proposed["unserved_energy_kwh"])
                <= float(rule["unserved_energy_kwh"]) + 1e-9,
                "risk_adjusted_improvement_pct": 100.0
                * (
                    float(rule["risk_adjusted_cost_yuan"])
                    - float(proposed["risk_adjusted_cost_yuan"])
                )
                / float(rule["risk_adjusted_cost_yuan"]),
            }
        )
    result = pd.DataFrame(rows)
    summary_path = root / "candidate_metrics.csv"
    if summary_path.exists():
        result = pd.concat([pd.read_csv(summary_path), result], ignore_index=True)
        result = result.drop_duplicates(subset=["seed", "name"], keep="last")
    result.to_csv(summary_path, index=False)
    (root / "development_metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "role": "development_only",
                "fixed_unit_commitment": fixed,
                "selection": protocol["selection"],
                "holdout_seeds_not_run": protocol["protocol"]["holdout_seeds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.sort_values("name").to_string(index=False))


if __name__ == "__main__":
    main()
