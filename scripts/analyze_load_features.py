from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.analysis import analyze_load_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="生成钻机负荷特征和验收指标表")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data/processed/rig_load_synthetic_v2.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/feature_analysis.yaml",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v2_load_features",
    )
    args = parser.parse_args()
    summary = analyze_load_features(args.data, args.config, args.artifact_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
