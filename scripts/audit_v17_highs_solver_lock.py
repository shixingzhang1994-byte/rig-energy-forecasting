from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize._highspy import _core


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dispatch_dir = (
        PROJECT_DIR
        / "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned"
    )
    metrics = pd.read_csv(dispatch_dir / "dispatch_metrics.csv")
    numeric = metrics.select_dtypes(include="number").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("调度指标含非有限数")
    result = {
        "solver_backend": "HiGHS",
        "highs_version": (
            f"{_core.HIGHS_VERSION_MAJOR}.{_core.HIGHS_VERSION_MINOR}."
            f"{_core.HIGHS_VERSION_PATCH}"
        ),
        "interface": "scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "solver_options": {"time_limit_seconds": 5.0, "mip_rel_gap": 1e-6},
        "multi_period_problem_features": [
            "clustered_integer_unit_commitment",
            "generator_minimum_stable_power",
            "generator_ramp_limits",
            "minimum_up_and_down_time",
            "storage_soc_dynamics_and_terminal_bound",
            "dynamic_grid_generator_and_storage_availability",
            "scenario_cvar_nonanticipativity",
        ],
        "evaluated_methods": sorted(metrics["method"].unique().tolist()),
        "evaluated_scenarios": sorted(metrics["scenario"].unique().tolist()),
        "metric_rows": int(len(metrics)),
        "all_metric_values_finite": True,
        "dispatch_source_sha256": _sha256(
            PROJECT_DIR / "src/rig_energy/optimization/dispatch.py"
        ),
        "evaluation_config_sha256": _sha256(
            PROJECT_DIR / "configs/v17_external_dispatch_data_aligned.yaml"
        ),
        "dispatch_metrics_sha256": _sha256(dispatch_dir / "dispatch_metrics.csv"),
        "claim_boundary": (
            "locks the independent open-source MILP solver and same encoded "
            "multi-period problems; it is not an independent reimplementation "
            "of the project mathematical formulation"
        ),
    }
    (dispatch_dir / "multi_period_highs_solver_lock.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
