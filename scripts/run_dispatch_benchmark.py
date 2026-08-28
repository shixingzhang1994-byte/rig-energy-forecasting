from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.optimization import run_dispatch_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行网电-柴油机-储能三方法协同优化对比")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "artifacts/v2_forecast_acceptance/forecast_predictions.npz",
    )
    parser.add_argument(
        "--prediction-keys",
        type=Path,
        default=PROJECT_DIR / "artifacts/v2_forecast_acceptance/forecast_prediction_keys.json",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/dispatch_default.yaml")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v2_dispatch_acceptance",
    )
    parser.add_argument(
        "--risk-signals",
        type=Path,
        default=PROJECT_DIR / "artifacts/v2_risk_acceptance/risk_predictions.csv",
        help="风险预测CSV；默认使用V2验收版以生成风险感知MPC对比",
    )
    args = parser.parse_args()
    metrics = run_dispatch_benchmark(
        args.predictions,
        args.prediction_keys,
        args.config,
        args.artifact_dir,
        args.risk_signals,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
