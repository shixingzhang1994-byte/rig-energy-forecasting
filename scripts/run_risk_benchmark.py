from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk import run_risk_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行四级供需风险分类、证据与集成对比实验")
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
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data/processed/rig_load_synthetic_v2.parquet",
    )
    parser.add_argument(
        "--forecast-config", type=Path, default=PROJECT_DIR / "configs/synthetic_default.yaml"
    )
    parser.add_argument(
        "--risk-config", type=Path, default=PROJECT_DIR / "configs/risk_default.yaml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/v2_risk_acceptance"
    )
    parser.add_argument(
        "--supply-data",
        type=Path,
        default=None,
        help="可选的真实供能CSV/Parquet；不提供时优先读取标准数据中的供能字段，否则仿真",
    )
    parser.add_argument("--cpu", action="store_true", help="强制CPU运行")
    args = parser.parse_args()
    metrics = run_risk_benchmark(
        args.predictions,
        args.prediction_keys,
        args.data,
        args.forecast_config,
        args.risk_config,
        args.artifact_dir,
        supply_path=args.supply_data,
        force_cpu=args.cpu,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
