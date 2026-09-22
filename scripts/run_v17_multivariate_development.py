from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.validation.multivariate_causal import (  # noqa: E402
    run_multivariate_causal_development,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="V17 因果多变量预测开发迭代")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v17_multivariate_causal_development.yaml",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v17_multivariate_causal_development",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    metrics = run_multivariate_causal_development(
        args.config, args.artifact_dir, force_cpu=args.cpu
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
