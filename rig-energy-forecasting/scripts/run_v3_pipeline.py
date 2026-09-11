from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键重建V3竞争性仿真证据链")
    parser.add_argument(
        "--reuse-forecasts",
        action="store_true",
        help="复用已有主种子和稳健性预测，只重建风险/调度/审计",
    )
    args = parser.parse_args()

    if not args.reuse_forecasts:
        _run(
            "scripts/generate_synthetic.py",
            "--seed",
            "20260825",
            "--output",
            "data/processed/rig_load_synthetic_v3.parquet",
            "--artifact-dir",
            "artifacts/v3_data",
        )
        _run(
            "scripts/run_benchmark.py",
            "--data",
            "data/processed/rig_load_synthetic_v3.parquet",
            "--seed",
            "20260825",
            "--artifact-dir",
            "artifacts/v3_competitive_forecast",
        )
        for seed in (20260826, 20260827):
            _run(
                "scripts/generate_synthetic.py",
                "--seed",
                str(seed),
                "--output",
                f"data/robustness/rig_load_seed_{seed}.parquet",
                "--artifact-dir",
                f"artifacts/v3_robustness/seed_{seed}/data",
            )
            _run(
                "scripts/run_benchmark.py",
                "--data",
                f"data/robustness/rig_load_seed_{seed}.parquet",
                "--seed",
                str(seed),
                "--artifact-dir",
                f"artifacts/v3_robustness/seed_{seed}/forecast",
            )

    _run("scripts/summarize_forecast_robustness.py")
    _run(
        "scripts/run_risk_benchmark.py",
        "--predictions",
        "artifacts/v3_competitive_forecast/forecast_predictions.npz",
        "--prediction-keys",
        "artifacts/v3_competitive_forecast/forecast_prediction_keys.json",
        "--data",
        "data/processed/rig_load_synthetic_v3.parquet",
        "--artifact-dir",
        "artifacts/v3_competitive_risk",
    )
    _run(
        "scripts/run_dispatch_benchmark.py",
        "--predictions",
        "artifacts/v3_competitive_forecast/forecast_predictions.npz",
        "--prediction-keys",
        "artifacts/v3_competitive_forecast/forecast_prediction_keys.json",
        "--risk-signals",
        "artifacts/v3_competitive_risk/risk_predictions.csv",
        "--artifact-dir",
        "artifacts/v3_competitive_dispatch",
    )
    _run("scripts/audit_acceptance.py")
    _run("scripts/audit_competitive_v3.py")
    _run("scripts/build_v3_evidence_manifest.py")


if __name__ == "__main__":
    main()
