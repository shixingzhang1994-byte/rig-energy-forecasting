#!/usr/bin/env python3
"""Independent full-horizon MILP reference for the frozen V15 dispatch data.

This file intentionally does not import the project dispatch implementation.  It
re-encodes a deterministic, perfect-foresight full-horizon MILP with clustered
integer generator commitment, minimum up/down time, ramping, storage SOC, and
scenario-specific capacity limits.  It is an oracle/reference bound, not an
online causal controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


DT_SECONDS = 5.0
EPS = 1e-8


class ModelBuilder:
    def __init__(self) -> None:
        self.names: List[str] = []
        self.lower: List[float] = []
        self.upper: List[float] = []
        self.integrality: List[int] = []
        self.rows: List[Dict[int, float]] = []
        self.row_lb: List[float] = []
        self.row_ub: List[float] = []

    def var(self, name: str, lb: float, ub: float, integer: bool = False) -> int:
        idx = len(self.names)
        self.names.append(name)
        self.lower.append(lb)
        self.upper.append(ub)
        self.integrality.append(1 if integer else 0)
        return idx

    def row(self, coeffs: Dict[int, float], lb: float, ub: float) -> None:
        self.rows.append({i: v for i, v in coeffs.items() if abs(v) > EPS})
        self.row_lb.append(lb)
        self.row_ub.append(ub)

    def equality(self, coeffs: Dict[int, float], value: float) -> None:
        self.row(coeffs, value, value)

    def build(self):
        rr: List[int] = []
        cc: List[int] = []
        dd: List[float] = []
        for r, coeffs in enumerate(self.rows):
            for c, value in coeffs.items():
                rr.append(r)
                cc.append(c)
                dd.append(value)
        matrix = coo_matrix((dd, (rr, cc)), shape=(len(self.rows), len(self.names))).tocsr()
        return (
            matrix,
            np.asarray(self.row_lb, dtype=float),
            np.asarray(self.row_ub, dtype=float),
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
            np.asarray(self.integrality, dtype=int),
        )


def read_rows(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty trajectory: {path}")
    required = {
        "load_kw",
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_available_power_kw",
    }
    missing = required - set(raw[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return [{key: float(row[key]) for key in required} for row in raw]


def solve_window(rows: Sequence[Dict[str, float]], config: Dict[str, float]) -> Dict[str, object]:
    t_count = len(rows)
    dt_h = config["interval_seconds"] / 3600.0
    unit_count = int(config["unit_count"])
    unit_rated = config["unit_rated_power_kw"]
    unit_min = config["unit_minimum_stable_power_kw"]
    ramp = config["ramp_kw_per_step"]
    min_up = int(config["minimum_up_steps"])
    min_down = int(config["minimum_down_steps"])
    storage_energy = config["energy_capacity_kwh"]
    eta_c = config["charge_efficiency"]
    eta_d = config["discharge_efficiency"]
    soc_initial = config["soc_initial"]
    soc_min = config["soc_min"]
    soc_max = config["soc_max"]

    b = ModelBuilder()
    grid = [b.var(f"grid[{t}]", 0.0, float(rows[t]["grid_available_capacity_kw"])) for t in range(t_count)]
    gen = [b.var(f"gen[{t}]", 0.0, float(rows[t]["generator_available_capacity_kw"])) for t in range(t_count)]
    charge = [b.var(f"charge[{t}]", 0.0, float(rows[t]["storage_available_power_kw"])) for t in range(t_count)]
    discharge = [b.var(f"discharge[{t}]", 0.0, float(rows[t]["storage_available_power_kw"])) for t in range(t_count)]
    unserved = [b.var(f"unserved[{t}]", 0.0, max(float(rows[t]["load_kw"]), 0.0)) for t in range(t_count)]
    spill = [b.var(f"spill[{t}]", 0.0, 3000.0) for t in range(t_count)]
    soc = [b.var(f"soc[{t}]", soc_min, soc_max) for t in range(t_count + 1)]
    committed = [b.var(f"committed[{t}]", 0.0, float(unit_count), integer=True) for t in range(t_count)]
    startup = [b.var(f"startup[{t}]", 0.0, float(unit_count), integer=True) for t in range(t_count)]
    shutdown = [b.var(f"shutdown[{t}]", 0.0, float(unit_count), integer=True) for t in range(t_count)]
    storage_mode = [b.var(f"storage_mode[{t}]", 0.0, 1.0, integer=True) for t in range(t_count)]

    b.equality({soc[0]: 1.0}, soc_initial)
    for t, row in enumerate(rows):
        # grid + generator + discharge + unserved - charge - spill = load
        b.equality(
            {grid[t]: 1.0, gen[t]: 1.0, discharge[t]: 1.0, unserved[t]: 1.0, charge[t]: -1.0, spill[t]: -1.0},
            float(row["load_kw"]),
        )
        b.row({gen[t]: 1.0, committed[t]: -unit_rated}, -math.inf, 0.0)
        b.row({gen[t]: 1.0, committed[t]: -unit_min}, 0.0, math.inf)
        b.row({charge[t]: 1.0, storage_mode[t]: -float(row["storage_available_power_kw"])}, -math.inf, 0.0)
        b.row({discharge[t]: 1.0, storage_mode[t]: float(row["storage_available_power_kw"])}, -math.inf, float(row["storage_available_power_kw"]))
        b.equality(
            {
                soc[t + 1]: 1.0,
                soc[t]: -1.0,
                charge[t]: -eta_c * dt_h / storage_energy,
                discharge[t]: dt_h / (eta_d * storage_energy),
            },
            0.0,
        )
        if t == 0:
            b.equality({committed[t]: 1.0, startup[t]: -1.0, shutdown[t]: 1.0}, 0.0)
            b.row({gen[t]: 1.0}, -math.inf, ramp)
        else:
            b.equality(
                {committed[t]: 1.0, committed[t - 1]: -1.0, startup[t]: -1.0, shutdown[t]: 1.0},
                0.0,
            )
            b.row({gen[t]: 1.0, gen[t - 1]: -1.0}, -math.inf, ramp)
            b.row({gen[t - 1]: 1.0, gen[t]: -1.0}, -math.inf, ramp)
        b.row({startup[t]: 1.0, committed[t]: -1.0}, -math.inf, 0.0)
        b.row({shutdown[t]: 1.0, committed[t]: 1.0}, -math.inf, float(unit_count))

    # Clustered minimum up/down constraints, starting from all units off.
    for t in range(t_count):
        if t >= min_up - 1:
            b.row({startup[k]: 1.0 for k in range(t - min_up + 1, t + 1)} | {committed[t]: -1.0}, -math.inf, 0.0)
        if t >= min_down - 1:
            b.row({shutdown[k]: 1.0 for k in range(t - min_down + 1, t + 1)} | {committed[t]: 1.0}, -math.inf, float(unit_count))

    # Retain a small terminal SOC reserve, matching the frozen risk-MPC lock.
    b.row({soc[-1]: 1.0}, soc_initial - config["terminal_soc_drop"], math.inf)

    # Linear fuel approximation through the rated-point slope. This is an
    # intentionally conservative, transparent linearization of the frozen
    # CAT-XQP300-50Hz-prime curve and keeps the independent model MILP-sized.
    fuel_slope_lph_per_kw = config["fuel_lph_at_rated"] / config["rated_power_kw"]
    c = np.zeros(len(b.names), dtype=float)
    for t in range(t_count):
        c[grid[t]] = dt_h * config["grid_price"]
        c[gen[t]] = dt_h * config["diesel_price"] * fuel_slope_lph_per_kw
        c[charge[t]] = dt_h * config["storage_degradation"]
        c[discharge[t]] = dt_h * config["storage_degradation"]
        c[unserved[t]] = dt_h * config["unserved_penalty"]
        c[spill[t]] = dt_h * config["spill_penalty"]
        c[startup[t]] = config["startup_cost"]

    matrix, row_lb, row_ub, lb, ub, integrality = b.build()
    started = time.perf_counter()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, row_lb, row_ub),
        options={"time_limit": config["time_limit_seconds"], "mip_rel_gap": config["mip_rel_gap"]},
    )
    solve_seconds = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"independent MILP failed: status={result.status}, message={result.message}")
    x = result.x

    def values(indices: Iterable[int]) -> np.ndarray:
        return np.asarray([x[i] for i in indices], dtype=float)

    grid_v = values(grid)
    gen_v = values(gen)
    charge_v = values(charge)
    discharge_v = values(discharge)
    unserved_v = values(unserved)
    spill_v = values(spill)
    soc_v = values(soc)
    committed_v = values(committed)
    startup_v = values(startup)
    balance_error = grid_v + gen_v + discharge_v + unserved_v - charge_v - spill_v - np.asarray([r["load_kw"] for r in rows])
    return {
        "status": int(result.status),
        "message": str(result.message),
        "objective_yuan": float(result.fun),
        "solve_seconds": float(solve_seconds),
        "unserved_energy_kwh": float(unserved_v.sum() * dt_h),
        "grid_energy_kwh": float(grid_v.sum() * dt_h),
        "generator_energy_kwh": float(gen_v.sum() * dt_h),
        "storage_throughput_kwh": float((charge_v + discharge_v).sum() * dt_h),
        "diesel_fuel_l": float(gen_v.sum() * dt_h * fuel_slope_lph_per_kw),
        "generator_startups": int(round(startup_v.sum())),
        "max_power_balance_error_kw": float(np.max(np.abs(balance_error))),
        "min_soc": float(np.min(soc_v)),
        "terminal_soc": float(soc_v[-1]),
        "max_committed_units": int(round(np.max(committed_v))),
        "max_spill_kw": float(np.max(spill_v)),
        "max_charge_discharge_product": float(np.max(charge_v * discharge_v)),
        "load_steps": t_count,
    }


def parse_config(path: Path) -> Dict[str, float]:
    # The V15 YAML is intentionally simple; parse only the scalar fields needed
    # by this independent implementation without importing project utilities.
    values: Dict[str, float] = {}
    stack: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped.endswith(":"):
            section_name = stripped[:-1]
            while stack and indent <= stack[-1][0]:
                stack.pop()
            stack.append((indent, section_name))
            continue
        if ":" not in stripped:
            continue
        key, value = [part.strip() for part in stripped.split(":", 1)]
        if value in {"", "true", "false"} or value.startswith("[") or value.startswith("{"):
            continue
        try:
            path_key = ".".join(item[1] for item in stack) + "." + key
            values[path_key] = float(value.strip("'\""))
        except ValueError:
            continue
    return {
        "interval_seconds": values["dispatch.interval_seconds"],
        "rated_power_kw": values["plant.generator.rated_power_kw"],
        "unit_count": values["plant.generator.unit_count"],
        "unit_rated_power_kw": values["plant.generator.unit_rated_power_kw"],
        "unit_minimum_stable_power_kw": values["plant.generator.unit_minimum_stable_power_kw"],
        "ramp_kw_per_step": values["plant.generator.ramp_kw_per_step"],
        "minimum_up_steps": values["plant.generator.minimum_up_steps"],
        "minimum_down_steps": values["plant.generator.minimum_down_steps"],
        "energy_capacity_kwh": values["plant.storage.energy_capacity_kwh"],
        "charge_efficiency": values["plant.storage.charge_efficiency"],
        "discharge_efficiency": values["plant.storage.discharge_efficiency"],
        "soc_initial": values["plant.storage.soc_initial"],
        "soc_min": values["plant.storage.soc_min"],
        "soc_max": values["plant.storage.soc_max"],
        "grid_price": values["cost.grid_price_yuan_per_kwh"],
        "diesel_price": values["cost.diesel_price_yuan_per_l"],
        "storage_degradation": values["cost.storage_degradation_yuan_per_kwh"],
        "unserved_penalty": values["cost.unserved_energy_penalty_yuan_per_kwh"],
        "startup_cost": values["cost.generator_startup_cost_yuan_per_unit"],
        "fuel_lph_at_rated": 62.5,
        "terminal_soc_drop": 0.005,
        "spill_penalty": 0.01,
        "time_limit_seconds": 30.0,
        "mip_rel_gap": 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = parse_config(args.config)
    dispatch_dir = args.artifact_root / "dispatch"
    # Each V15 seed stores the same realized load/capacity window once per
    # controller.  Use the proposed-controller copy as the canonical input so
    # that the oracle aggregate counts the three operating scenarios once.
    trajectories = sorted(dispatch_dir.glob("trajectory_*_risk_soc_supervisory_mpc.csv"))
    if not trajectories:
        raise SystemExit(f"no trajectory CSVs found under {dispatch_dir}")

    results = []
    for trajectory in trajectories:
        rows = read_rows(trajectory)
        solved = solve_window(rows, config)
        solved["trajectory"] = trajectory.name
        results.append(solved)
        print(json.dumps(solved, ensure_ascii=False, sort_keys=True))

    aggregate = {
        "trajectory_count": len(results),
        "total_unserved_energy_kwh": float(sum(r["unserved_energy_kwh"] for r in results)),
        "total_objective_yuan": float(sum(r["objective_yuan"] for r in results)),
        "mean_solve_seconds": float(np.mean([r["solve_seconds"] for r in results])),
        "max_power_balance_error_kw": float(max(r["max_power_balance_error_kw"] for r in results)),
        "min_soc": float(min(r["min_soc"] for r in results)),
        "max_charge_discharge_product": float(max(r["max_charge_discharge_product"] for r in results)),
    }
    payload = {
        "baseline": "Independent deterministic full-MILP oracle",
        "claim_boundary": "Perfect-foresight reference using actual load trajectory; not an online causal competitor.",
        "model_independence": "Standalone scipy.optimize.milp re-encoding; does not import rig_energy.optimization.dispatch.",
        "solver": {
            "backend": "HiGHS via scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "time_limit_seconds": config["time_limit_seconds"],
            "mip_rel_gap": config["mip_rel_gap"],
        },
        "config": config,
        "trajectories": results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
