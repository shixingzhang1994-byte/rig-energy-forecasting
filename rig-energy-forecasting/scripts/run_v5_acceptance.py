from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

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


def _write_resolved(base_path: Path, overrides: dict, output: Path) -> None:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    resolved = _deep_merge(base, overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="一键重建V5物理与作业场景验收链")
    parser.add_argument("--reuse-forecast", action="store_true")
    parser.add_argument("--reuse-risk", action="store_true")
    args = parser.parse_args()

    root = PROJECT_DIR / "artifacts/v5_acceptance_rework"
    protocol = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    forecast_config = root / "resolved_forecast_config.yaml"
    risk_config = root / "resolved_risk_config.yaml"
    dispatch_config = root / "resolved_dispatch_config.yaml"
    _write_resolved(
        PROJECT_DIR / "configs/synthetic_default.yaml",
        protocol["forecast_overrides"],
        forecast_config,
    )
    _write_resolved(
        PROJECT_DIR / "configs/risk_default.yaml",
        protocol["risk_overrides"],
        risk_config,
    )
    _write_resolved(
        PROJECT_DIR / "configs/dispatch_default.yaml",
        protocol["dispatch_overrides"],
        dispatch_config,
    )

    data_path = PROJECT_DIR / "data/processed/rig_load_synthetic_v5.parquet"
    forecast_dir = root / "forecast"
    risk_dir = root / "risk"
    dispatch_dir = root / "dispatch"
    if not args.reuse_forecast:
        _run(
            "scripts/generate_synthetic.py",
            "--config",
            str(forecast_config),
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
            str(forecast_config),
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
            str(forecast_config),
            "--risk-config",
            str(risk_config),
            "--artifact-dir",
            str(risk_dir),
        )
    _run(
        "scripts/run_dispatch_benchmark.py",
        "--predictions",
        str(forecast_dir / "forecast_predictions.npz"),
        "--prediction-keys",
        str(forecast_dir / "forecast_prediction_keys.json"),
        "--config",
        str(dispatch_config),
        "--risk-signals",
        str(risk_dir / "risk_predictions.csv"),
        "--artifact-dir",
        str(dispatch_dir),
    )
    _run("scripts/audit_v5_acceptance.py")


if __name__ == "__main__":
    main()
