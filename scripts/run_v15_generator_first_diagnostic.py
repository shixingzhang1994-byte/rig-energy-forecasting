from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.optimization import dispatch as dispatch_module  # noqa: E402
from run_v14_dispatch_benchmark import _v14_fuel_lph  # noqa: E402


_ORIGINAL_EXECUTE = dispatch_module._execute_with_safety_layer


def _generator_first_for_supervisory(*args, **kwargs):
    positional = list(args)
    reserve_target = (
        positional[13]
        if len(positional) > 13
        else kwargs.get("grid_headroom_reserve_target_soc")
    )
    original_generator_first = (
        bool(positional[11])
        if len(positional) > 11
        else bool(kwargs.get("generator_before_storage", False))
    )
    guard_active = reserve_target is not None and not original_generator_first
    if reserve_target is not None:
        if len(positional) > 11:
            positional[11] = True
        else:
            kwargs["generator_before_storage"] = True
    executed, soc_next = _ORIGINAL_EXECUTE(*positional, **kwargs)
    executed["v15_generator_first_guard_active"] = bool(guard_active)
    return executed, soc_next


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V15单变量诊断：监督方案始终优先使用已承诺机组裕量"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--risk-signals", type=Path, required=True)
    args = parser.parse_args()

    dispatch_module._fuel_lph = _v14_fuel_lph
    dispatch_module._execute_with_safety_layer = _generator_first_for_supervisory
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
