from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog
from scipy.sparse import kron, eye


TIER_NAMES = ("critical", "essential", "interruptible")


@dataclass(frozen=True)
class PriorityAllocation:
    served_kw: np.ndarray
    shed_kw: np.ndarray
    method: str
    solver_metadata: dict[str, str | int | float | bool]


def _validate_inputs(
    demand_by_tier_kw: np.ndarray, available_supply_kw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    demand = np.asarray(demand_by_tier_kw, dtype=float)
    available = np.asarray(available_supply_kw, dtype=float)
    if demand.ndim != 2 or demand.shape[1] != len(TIER_NAMES):
        raise ValueError("demand_by_tier_kw 必须是 N×3 数组")
    if available.shape != (len(demand),):
        raise ValueError("available_supply_kw 长度必须与需求行数一致")
    if not np.isfinite(demand).all() or not np.isfinite(available).all():
        raise ValueError("需求和可用供能必须为有限数")
    if (demand < -1e-12).any() or (available < -1e-12).any():
        raise ValueError("需求和可用供能不得为负")
    return np.maximum(demand, 0.0), np.maximum(available, 0.0)


def allocate_priority_greedy(
    demand_by_tier_kw: np.ndarray, available_supply_kw: np.ndarray
) -> PriorityAllocation:
    """Serve critical, then essential, then interruptible demand."""

    demand, available = _validate_inputs(demand_by_tier_kw, available_supply_kw)
    served = np.zeros_like(demand)
    remaining = available.copy()
    for tier in range(demand.shape[1]):
        served[:, tier] = np.minimum(demand[:, tier], remaining)
        remaining = np.maximum(0.0, remaining - served[:, tier])
    return PriorityAllocation(
        served_kw=served,
        shed_kw=demand - served,
        method="priority-greedy",
        solver_metadata={"lexicographic_priority": True},
    )


def allocate_proportional(
    demand_by_tier_kw: np.ndarray, available_supply_kw: np.ndarray
) -> PriorityAllocation:
    """Unprotected comparator: curtail all tiers by the same fraction."""

    demand, available = _validate_inputs(demand_by_tier_kw, available_supply_kw)
    total = demand.sum(axis=1)
    fraction = np.ones_like(total)
    positive = total > 0.0
    fraction[positive] = np.minimum(1.0, available[positive] / total[positive])
    served = demand * fraction[:, None]
    return PriorityAllocation(
        served_kw=served,
        shed_kw=demand - served,
        method="proportional-curtailment",
        solver_metadata={"critical_load_protection": False},
    )


def _highs_version() -> str:
    try:
        from scipy.optimize._highspy import _core

        return (
            f"{_core.HIGHS_VERSION_MAJOR}."
            f"{_core.HIGHS_VERSION_MINOR}."
            f"{_core.HIGHS_VERSION_PATCH}"
        )
    except Exception:
        return "embedded-version-unavailable"


def allocate_priority_highs(
    demand_by_tier_kw: np.ndarray,
    available_supply_kw: np.ndarray,
    *,
    tier_values: tuple[float, float, float] = (1_000_000.0, 1_000.0, 1.0),
) -> PriorityAllocation:
    """Independent HiGHS LP reference for priority load preservation.

    The large, separated tier values encode the declared lexicographic order.
    This is an external solver reference, not a field-approved protection relay.
    """

    demand, available = _validate_inputs(demand_by_tier_kw, available_supply_kw)
    n_rows, n_tiers = demand.shape
    values = np.asarray(tier_values, dtype=float)
    if values.shape != (n_tiers,) or not np.all(np.diff(values) < 0):
        raise ValueError("tier_values 必须按 critical>essential>interruptible 严格递减")

    objective = -np.tile(values, n_rows)
    a_ub = kron(eye(n_rows, format="csr"), np.ones((1, n_tiers)), format="csr")
    bounds = [(0.0, float(limit)) for limit in demand.reshape(-1)]
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=available,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"HiGHS优先负荷分配失败: {result.message}")
    served = np.asarray(result.x, dtype=float).reshape(n_rows, n_tiers)
    return PriorityAllocation(
        served_kw=served,
        shed_kw=np.maximum(0.0, demand - served),
        method="HiGHS-priority-LP",
        solver_metadata={
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "scipy_version": scipy_version,
            "highs_version": _highs_version(),
            "method": "scipy.optimize.linprog(method='highs')",
        },
    )

