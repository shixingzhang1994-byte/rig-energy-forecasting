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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行V6验证残差上包络调度开发种子"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reuse-forecast", action="store_true")
    parser.add_argument("--reuse-risk", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v6_dispatch_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    allowed = set(protocol["protocol"]["development_seeds"])
    if args.seed not in allowed:
        raise ValueError(
            f"本脚本只允许开发种子 {sorted(allowed)}，拒绝运行留出种子"
        )
    v5_overrides = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    root = PROJECT_DIR / f"artifacts/v6_dispatch_development/seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)

    forecast = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/synthetic_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5_overrides["forecast_overrides"],
    )
    forecast["project"]["seed"] = args.seed
    forecast["project"]["days"] = int(
        protocol["protocol"]["simulation_days"]
    )
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": protocol["protocol"]["coverage_candidates"],
    }
    forecast_path = root / "resolved_forecast_config.yaml"
    _write_yaml(forecast, forecast_path)

    risk = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/risk_default.yaml").read_text(encoding="utf-8")
        ),
        v5_overrides["risk_overrides"],
    )
    risk["project"]["seed"] = args.seed
    risk["split"]["train_fraction"] = float(
        protocol["protocol"]["risk_split"]["train_fraction"]
    )
    risk["split"]["val_fraction"] = float(
        protocol["protocol"]["risk_split"]["val_fraction"]
    )
    risk_path = root / "resolved_risk_config.yaml"
    _write_yaml(risk, risk_path)

    data_path = PROJECT_DIR / f"data/processed/rig_load_v6_dev_{args.seed}.parquet"
    forecast_dir = root / "forecast"
    risk_dir = root / "risk"
    if not args.reuse_forecast:
        _run(
            "scripts/generate_synthetic.py",
            "--config",
            str(forecast_path),
            "--output",
            str(data_path),
            "--artifact-dir",
            str(root / "data"),
        )
        _run(
            "scripts/run_benchmark.py",
            "--data",
            str(data_path),
            "--config",
            str(forecast_path),
            "--artifact-dir",
            str(forecast_dir),
        )
    if not args.reuse_risk:
        _run(
            "scripts/run_risk_benchmark.py",
            "--predictions",
            str(forecast_dir / "forecast_predictions.npz"),
            "--prediction-keys",
            str(forecast_dir / "forecast_prediction_keys.json"),
            "--data",
            str(data_path),
            "--forecast-config",
            str(forecast_path),
            "--risk-config",
            str(risk_path),
            "--artifact-dir",
            str(risk_dir),
        )

    base_dispatch = _deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/dispatch_default.yaml").read_text(
                encoding="utf-8"
            )
        ),
        v5_overrides["dispatch_overrides"],
    )
    rows = []
    for coverage in protocol["protocol"]["coverage_candidates"]:
        token = int(round(float(coverage) * 1000.0))
        envelope_name = f"Validation-Calibrated-Upper-Envelope-q{token:03d}"
        dispatch = deepcopy(base_dispatch)
        dispatch["scenario_cvar_mpc"]["forecast_members"] = [
            *dispatch["scenario_cvar_mpc"]["forecast_members"],
            envelope_name,
        ]
        dispatch_path = root / f"resolved_dispatch_q{token:03d}.yaml"
        _write_yaml(dispatch, dispatch_path)
        dispatch_dir = root / f"dispatch_q{token:03d}"
        _run(
            "scripts/run_dispatch_benchmark.py",
            "--predictions",
            str(forecast_dir / "forecast_predictions.npz"),
            "--prediction-keys",
            str(forecast_dir / "forecast_prediction_keys.json"),
            "--config",
            str(dispatch_path),
            "--risk-signals",
            str(risk_dir / "risk_predictions.csv"),
            "--artifact-dir",
            str(dispatch_dir),
        )
        metrics = pd.read_csv(dispatch_dir / "dispatch_metrics.csv")
        aggregate = metrics.groupby("method")[[
            "equivalent_cost_yuan",
            "risk_adjusted_cost_yuan",
            "unserved_energy_kwh",
            "solve_seconds",
        ]].sum()
        proposed = aggregate.loc["Risk-Adaptive-CVaR-MPC"]
        rule = aggregate.loc["Rule-Based"]
        rows.append(
            {
                "seed": args.seed,
                "coverage": float(coverage),
                "envelope_name": envelope_name,
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
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "coverage_candidate_metrics.csv", index=False)
    (root / "development_metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "role": "development_only",
                "holdout_seeds_not_run": protocol["protocol"]["holdout_seeds"],
                "selection_rule": protocol["selection"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
