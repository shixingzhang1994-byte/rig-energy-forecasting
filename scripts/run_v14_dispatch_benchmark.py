from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.optimization import dispatch as dispatch_module  # noqa: E402


def _v14_fuel_lph(power_kw: np.ndarray | float, rated_power_kw: float) -> np.ndarray:
    """CAT XQP300 50-Hz prime curve, scaled only for declared sensitivity runs."""

    power = np.asarray(power_kw, dtype=float)
    scale = float(rated_power_kw) / 240.0
    points_kw = np.array([0.0, 120.0, 180.0, 240.0]) * scale
    points_lph = np.array([0.0, 33.8, 47.3, 62.5]) * scale
    return np.interp(np.clip(power, 0.0, rated_power_kw), points_kw, points_lph)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用CAT XQP300公开prime燃油曲线运行V14调度对比"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--risk-signals", type=Path, required=True)
    args = parser.parse_args()

    dispatch_module._fuel_lph = _v14_fuel_lph
    metrics = dispatch_module.run_dispatch_benchmark(
        args.predictions,
        args.prediction_keys,
        args.config,
        args.artifact_dir,
        args.risk_signals,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

