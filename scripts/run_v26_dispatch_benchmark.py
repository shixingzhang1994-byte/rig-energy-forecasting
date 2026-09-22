from __future__ import annotations

"""V26 dispatch entry point with a bounded, audited MILP incumbent policy.

The controller reserves 0.5 s of a five-second decision period for Python and
safety-layer overhead.  If HiGHS reaches the 4.5 s optimization deadline, an
incumbent is accepted only after an independent numerical check of bounds,
integrality, and every linear constraint.  Missing or infeasible incumbents
retain the original hard failure.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.optimization import dispatch as dispatch_module  # noqa: E402
from run_v14_dispatch_benchmark import _v14_fuel_lph  # noqa: E402


SOLVER_DEADLINE_SECONDS = 4.5
FEASIBILITY_TOLERANCE = 1e-6
_ORIGINAL_MILP = dispatch_module.milp
_ORIGINAL_SIMULATE_METHOD = dispatch_module._simulate_method
_CURRENT_METHOD = "unknown"
_CALL_AUDIT: list[dict[str, Any]] = []


def _maximum_positive_violation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(max(0.0, finite.max(initial=0.0)))


def audit_incumbent(
    x: Any,
    *,
    integrality: Any,
    bounds: Any,
    constraints: Any,
    tolerance: float = FEASIBILITY_TOLERANCE,
) -> dict[str, Any]:
    """Independently audit a SciPy MILP incumbent without using its status."""

    if x is None:
        return {
            "available": False,
            "feasible": False,
            "reason": "missing_incumbent",
        }
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        return {
            "available": True,
            "feasible": False,
            "reason": "nonfinite_or_nondimensional_incumbent",
        }

    lower = np.broadcast_to(np.asarray(bounds.lb, dtype=float), vector.shape)
    upper = np.broadcast_to(np.asarray(bounds.ub, dtype=float), vector.shape)
    lower_violation = _maximum_positive_violation(lower - vector)
    upper_violation = _maximum_positive_violation(vector - upper)

    integral = np.broadcast_to(np.asarray(integrality, dtype=np.int8), vector.shape)
    integer_mask = np.isin(integral, [1, 3])
    integrality_violation = (
        float(np.max(np.abs(vector[integer_mask] - np.rint(vector[integer_mask]))))
        if np.any(integer_mask)
        else 0.0
    )

    constraint_lower_violation = 0.0
    constraint_upper_violation = 0.0
    constraint_list = constraints if isinstance(constraints, (list, tuple)) else [constraints]
    for constraint in constraint_list:
        activity = np.asarray(constraint.A @ vector, dtype=float)
        constraint_lower = np.broadcast_to(
            np.asarray(constraint.lb, dtype=float), activity.shape
        )
        constraint_upper = np.broadcast_to(
            np.asarray(constraint.ub, dtype=float), activity.shape
        )
        constraint_lower_violation = max(
            constraint_lower_violation,
            _maximum_positive_violation(constraint_lower - activity),
        )
        constraint_upper_violation = max(
            constraint_upper_violation,
            _maximum_positive_violation(activity - constraint_upper),
        )

    maximum_violation = max(
        lower_violation,
        upper_violation,
        integrality_violation,
        constraint_lower_violation,
        constraint_upper_violation,
    )
    return {
        "available": True,
        "feasible": bool(maximum_violation <= tolerance),
        "reason": "pass" if maximum_violation <= tolerance else "violation",
        "tolerance": float(tolerance),
        "maximum_violation": maximum_violation,
        "bound_lower_violation": lower_violation,
        "bound_upper_violation": upper_violation,
        "integrality_violation": integrality_violation,
        "constraint_lower_violation": constraint_lower_violation,
        "constraint_upper_violation": constraint_upper_violation,
    }


def _deadline_milp(
    c: Any,
    *,
    integrality: Any = None,
    bounds: Any = None,
    constraints: Any = None,
    options: dict[str, Any] | None = None,
) -> Any:
    configured = dict(options or {})
    requested_limit = float(configured.get("time_limit", np.inf))
    configured["time_limit"] = min(requested_limit, SOLVER_DEADLINE_SECONDS)
    started = time.perf_counter()
    result = _ORIGINAL_MILP(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=configured,
    )
    elapsed = time.perf_counter() - started
    incumbent = audit_incumbent(
        result.x,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    original_success = bool(result.success)
    message = str(result.message)
    deadline_reached = (not original_success) and "time limit" in message.lower()
    accepted = bool(deadline_reached and incumbent["feasible"])
    if accepted:
        result.success = True
        result.message = f"{message}; feasible incumbent independently accepted by V26"
    _CALL_AUDIT.append(
        {
            "call_index": len(_CALL_AUDIT),
            "method": _CURRENT_METHOD,
            "requested_time_limit_seconds": requested_limit,
            "applied_solver_deadline_seconds": float(configured["time_limit"]),
            "solver_elapsed_seconds": elapsed,
            "original_success": original_success,
            "solver_status": int(result.status),
            "solver_message": message,
            "deadline_reached": deadline_reached,
            "feasible_incumbent_accepted": accepted,
            "objective_value": (
                None if getattr(result, "fun", None) is None else float(result.fun)
            ),
            "mip_gap": (
                None
                if getattr(result, "mip_gap", None) is None
                else float(result.mip_gap)
            ),
            "incumbent_audit": incumbent,
        }
    )
    return result


def _simulate_method_with_context(method: str, *args: Any, **kwargs: Any) -> Any:
    global _CURRENT_METHOD
    previous = _CURRENT_METHOD
    _CURRENT_METHOD = method
    try:
        return _ORIGINAL_SIMULATE_METHOD(method, *args, **kwargs)
    finally:
        _CURRENT_METHOD = previous


def write_deadline_audit(path: Path, failure: BaseException | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted = [item for item in _CALL_AUDIT if item["feasible_incumbent_accepted"]]
    rejected_deadlines = [
        item
        for item in _CALL_AUDIT
        if item["deadline_reached"] and not item["feasible_incumbent_accepted"]
    ]
    document = {
        "policy": {
            "controller_period_seconds": 5.0,
            "solver_deadline_seconds": SOLVER_DEADLINE_SECONDS,
            "reserved_overhead_seconds": 0.5,
            "feasibility_tolerance": FEASIBILITY_TOLERANCE,
            "acceptance_rule": "deadline incumbent must independently pass bounds, integrality, and all linear constraints",
            "missing_or_infeasible_incumbent_policy": "hard failure",
        },
        "completed_without_exception": failure is None,
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        "call_count": len(_CALL_AUDIT),
        "deadline_count": sum(bool(item["deadline_reached"]) for item in _CALL_AUDIT),
        "accepted_feasible_incumbent_count": len(accepted),
        "rejected_deadline_count": len(rejected_deadlines),
        "maximum_solver_elapsed_seconds": max(
            (float(item["solver_elapsed_seconds"]) for item in _CALL_AUDIT),
            default=0.0,
        ),
        "maximum_accepted_incumbent_violation": max(
            (
                float(item["incumbent_audit"]["maximum_violation"])
                for item in accepted
            ),
            default=0.0,
        ),
        "calls": _CALL_AUDIT,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V26 deadline-audited dispatch benchmark")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-keys", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--risk-signals", type=Path, required=True)
    parser.add_argument("--scenario-residuals", type=Path, required=True)
    parser.add_argument("--scenario-seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--deadline-audit", type=Path, required=True)
    parser.add_argument("--method", action="append", dest="methods")
    args = parser.parse_args()

    dispatch_module._fuel_lph = _v14_fuel_lph
    dispatch_module.milp = _deadline_milp
    dispatch_module._simulate_method = _simulate_method_with_context
    failure: BaseException | None = None
    try:
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
    except BaseException as exc:
        failure = exc
        raise
    finally:
        write_deadline_audit(args.deadline_audit, failure)


if __name__ == "__main__":
    main()
