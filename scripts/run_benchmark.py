from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.experiment import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行钻机短时负荷预测多模型对比")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data/processed/rig_load_synthetic_v2.parquet")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/synthetic_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/v2_forecast_acceptance")
    parser.add_argument("--quick", action="store_true", help="缩短轮数，用于接口冒烟测试")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--seed", type=int, default=None, help="可选的训练随机种子覆盖")
    parser.add_argument(
        "--export-validation-residuals",
        action="store_true",
        help="导出V22验证集因果残差库；不包含测试目标",
    )
    args = parser.parse_args()
    metrics = run_benchmark(
        args.data,
        args.config,
        args.artifact_dir,
        quick=args.quick,
        force_cpu=args.cpu,
        seed_override=args.seed,
        export_validation_residuals=args.export_validation_residuals,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
