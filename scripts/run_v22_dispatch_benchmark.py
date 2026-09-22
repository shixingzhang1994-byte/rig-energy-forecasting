from __future__ import annotations

"""Run the V22 matched-path dispatch comparison with residual scenarios."""

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.optimization import dispatch as dispatch_module  # noqa: E402
from run_v14_dispatch_benchmark import _v14_fuel_lph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V22 validation-residual CVaR matched dispatch benchmark"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--risk-signals", type=Path, required=True)
    parser.add_argument("--scenario-residuals", type=Path, required=True)
    parser.add_argument("--scenario-seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Override comparison methods; repeat once per method.",
    )
    args = parser.parse_args()

    dispatch_module._fuel_lph = _v14_fuel_lph
    metrics = dispatch_module.run_dispatch_benchmark(
        args.predictions,
        args.prediction_keys,
        args.config,
        args.artifact_dir,
        args.risk_signals,
        args.scenario_residuals,
        args.scenario_seed,
        args.methods,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
