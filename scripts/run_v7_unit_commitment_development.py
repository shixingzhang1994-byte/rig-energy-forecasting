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


def _write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行V7多机组启停开发实验")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        help="只运行指定候选；可重复提供。默认运行全部预声明候选。",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="若候选已有完整dispatch_metrics.csv则只重新汇总。",
    )
    args = parser.parse_args()

    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v7_unit_commitment_development.yaml").read_text(
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
        unknown = requested - known
        if unknown:
            raise ValueError(f"未知候选: {sorted(unknown)}")
        candidates = [item for item in candidates if item["name"] in requested]

    source = PROJECT_DIR / f"artifacts/v6_dispatch_development/seed_{args.seed}"
    forecast_dir = source / "forecast"
    risk_dir = source / "risk"
    required = [
        forecast_dir / "forecast_predictions.npz",
        forecast_dir / "forecast_prediction_keys.json",
        risk_dir / "risk_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "V7复用同开发种子的冻结预测/风险输出，以下文件缺失: "
            + ", ".join(missing)
        )

    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    base = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/dispatch_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5["dispatch_overrides"],
    )
    unit_model = protocol["protocol"]["unit_model"]
    root = PROJECT_DIR / f"artifacts/v7_unit_commitment/seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        config = deepcopy(base)
        generator = config["plant"]["generator"]
        generator.update(unit_model)
        generator["minimum_up_steps"] = int(candidate["minimum_up_steps"])
        generator["minimum_down_steps"] = int(candidate["minimum_down_steps"])
        config["cost"]["generator_startup_cost_yuan_per_unit"] = float(
            candidate["startup_cost_yuan_per_unit"]
        )
        config_path = root / f"resolved_{candidate['name']}.yaml"
        _write_yaml(config, config_path)
        artifact_dir = root / candidate["name"]
        command = [
            sys.executable,
            "scripts/run_dispatch_benchmark.py",
            "--predictions",
            str(forecast_dir / "forecast_predictions.npz"),
            "--prediction-keys",
            str(forecast_dir / "forecast_prediction_keys.json"),
            "--config",
            str(config_path),
            "--risk-signals",
            str(risk_dir / "risk_predictions.csv"),
            "--artifact-dir",
            str(artifact_dir),
        ]
        metrics_path = artifact_dir / "dispatch_metrics.csv"
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
                **candidate,
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
    new_rows = pd.DataFrame(rows)
    summary_path = root / "candidate_metrics.csv"
    if summary_path.exists():
        old_rows = pd.read_csv(summary_path)
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
        new_rows = new_rows.drop_duplicates(
            subset=["seed", "name"], keep="last"
        )
    new_rows.to_csv(summary_path, index=False)
    (root / "development_metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "role": "development_only",
                "unit_model": unit_model,
                "selection": protocol["selection"],
                "holdout_seeds_not_run": protocol["protocol"]["holdout_seeds"],
                "validation_envelope_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(new_rows.sort_values("name").to_string(index=False))


if __name__ == "__main__":
    main()
