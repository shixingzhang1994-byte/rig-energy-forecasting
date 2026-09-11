#!/usr/bin/env python3
"""Independent causal rolling full-MILP baseline for the frozen V15 windows.

The baseline uses the same realized plant/capacity traces as the online
controllers, but only the causal 12-step Causal-ErrorFeedback-Ensemble
forecast available at the current source index.  It solves a complete
clustered unit-commitment/storage MILP at every step and applies only the first
control action.  The implementation is deliberately separate from
``rig_energy.optimization.dispatch``; the local oracle module supplies only
the small sparse-model builder and scalar configuration parser.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_independent_full_milp import ModelBuilder, parse_config


DT_SECONDS = 5.0
FORECAST_HORIZON = 12
TERMINAL_SOC_DROP = 0.005
EPS = 1e-8


def _history_still_locked(history: Sequence[int], minimum_steps: int, future_step: int) -> int:
    """Count past clustered transitions whose minimum lock is still active."""

    if minimum_steps <= 1:
        return 0
    return int(
        sum(
            int(count)
            for age, count in enumerate(reversed(tuple(history)), start=1)
            if age + int(future_step) < minimum_steps
        )
    )


def read_trajectory(path: Path) -> Dict[str, np.ndarray]:
    required = {
        "load_kw",
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_available_power_kw",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in sorted(required)
    }


def solve_causal_window(
    forecast: Sequence[float],
    grid_capacity: Sequence[float],
    generator_capacity: Sequence[float],
    storage_capacity: Sequence[float],
    config: Dict[str, float],
    *,
    soc_initial: float,
    previous_generator_kw: float,
    previous_generator_units: int,
    startup_history: Sequence[int] = (),
    shutdown_history: Sequence[int] = (),
) -> Dict[str, float]:
    """Solve one causal forecast window and return its first action."""

    demand = np.maximum(np.asarray(forecast, dtype=float), 0.0)
    grid_cap = np.asarray(grid_capacity, dtype=float)
    gen_cap = np.asarray(generator_capacity, dtype=float)
    storage_cap = np.asarray(storage_capacity, dtype=float)
    horizon = len(demand)
    if horizon == 0 or not (len(grid_cap) == len(gen_cap) == len(storage_cap) == horizon):
        raise ValueError("causal MILP window arrays have inconsistent lengths")

    unit_count = int(config["unit_count"])
    unit_rated = config["unit_rated_power_kw"]
    unit_min = config["unit_minimum_stable_power_kw"]
    unit_ramp = config["ramp_kw_per_step"] / unit_count
    min_up = int(config["minimum_up_steps"])
    min_down = int(config["minimum_down_steps"])
    energy_capacity = config["energy_capacity_kwh"]
    eta_c = config["charge_efficiency"]
    eta_d = config["discharge_efficiency"]
    dt_h = config["interval_seconds"] / 3600.0

    b = ModelBuilder()
    grid = [b.var(f"grid[{t}]", 0.0, max(0.0, grid_cap[t])) for t in range(horizon)]
    gen = [b.var(f"gen[{t}]", 0.0, max(0.0, gen_cap[t])) for t in range(horizon)]
    charge = [b.var(f"charge[{t}]", 0.0, max(0.0, storage_cap[t])) for t in range(horizon)]
    discharge = [b.var(f"discharge[{t}]", 0.0, max(0.0, storage_cap[t])) for t in range(horizon)]
    unserved = [b.var(f"unserved[{t}]", 0.0, demand[t]) for t in range(horizon)]
    spill = [b.var(f"spill[{t}]", 0.0, 3000.0) for t in range(horizon)]
    soc = [b.var(f"soc[{t}]", config["soc_min"], config["soc_max"]) for t in range(horizon + 1)]
    available_units = [
        min(unit_count, math.floor((max(0.0, gen_cap[t]) + 1e-9) / unit_min))
        for t in range(horizon)
    ]
    committed = [
        b.var(
            f"committed[{t}]",
            0.0,
            available_units[t],
            integer=True,
        )
        for t in range(horizon)
    ]
    startup = [b.var(f"startup[{t}]", 0.0, float(unit_count), integer=True) for t in range(horizon)]
    shutdown = [b.var(f"shutdown[{t}]", 0.0, float(unit_count), integer=True) for t in range(horizon)]
    storage_mode = [b.var(f"storage_mode[{t}]", 0.0, 1.0, integer=True) for t in range(horizon)]

    previous_units = int(np.clip(previous_generator_units, 0, unit_count))
    previous_generator = max(0.0, float(previous_generator_kw))
    previous_units = min(previous_units, available_units[0])
    previous_generator = min(previous_generator, max(0.0, gen_cap[0]))

    b.equality({soc[0]: 1.0}, float(np.clip(soc_initial, config["soc_min"], config["soc_max"])))
    for t in range(horizon):
        b.equality(
            {
                grid[t]: 1.0,
                gen[t]: 1.0,
                discharge[t]: 1.0,
                unserved[t]: 1.0,
                charge[t]: -1.0,
                spill[t]: -1.0,
            },
            float(demand[t]),
        )
        b.row({gen[t]: 1.0, committed[t]: -unit_rated}, -math.inf, 0.0)
        b.row({gen[t]: 1.0, committed[t]: -unit_min}, 0.0, math.inf)
        b.row({charge[t]: 1.0, storage_mode[t]: -max(0.0, storage_cap[t])}, -math.inf, 0.0)
        b.row(
            {discharge[t]: 1.0, storage_mode[t]: max(0.0, storage_cap[t])},
            -math.inf,
            max(0.0, storage_cap[t]),
        )
        b.equality(
            {
                soc[t + 1]: 1.0,
                soc[t]: -1.0,
                charge[t]: -eta_c * dt_h / energy_capacity,
                discharge[t]: dt_h / (eta_d * energy_capacity),
            },
            0.0,
        )
        if t == 0:
            b.equality({committed[t]: 1.0, startup[t]: -1.0, shutdown[t]: 1.0}, float(previous_units))
            b.row({gen[t]: 1.0, startup[t]: -unit_min}, -math.inf, previous_generator + unit_ramp * previous_units)
            b.row({gen[t]: -1.0, shutdown[t]: -unit_rated}, -math.inf, unit_ramp * previous_units - previous_generator)
        else:
            b.equality(
                {
                    committed[t]: 1.0,
                    committed[t - 1]: -1.0,
                    startup[t]: -1.0,
                    shutdown[t]: 1.0,
                },
                0.0,
            )
            b.row(
                {gen[t]: 1.0, gen[t - 1]: -1.0, committed[t - 1]: -unit_ramp, startup[t]: -unit_min},
                -math.inf,
                0.0,
            )
            b.row(
                {gen[t - 1]: 1.0, gen[t]: -1.0, committed[t]: -unit_ramp, shutdown[t]: -unit_rated},
                -math.inf,
                0.0,
            )

    for t in range(horizon):
        starts = {startup[k]: 1.0 for k in range(max(0, t - min_up + 1), t + 1)}
        starts[committed[t]] = -1.0
        # A forced dynamic-capacity derating makes unavailable units physically
        # impossible to keep online. Match the plant rule by retaining only the
        # subset of historical minimum-up locks that current capacity can host.
        locked_from_history = min(
            _history_still_locked(startup_history, min_up, t),
            previous_units,
            available_units[t],
        )
        b.row(starts, -math.inf, -float(locked_from_history))
        stops = {shutdown[k]: 1.0 for k in range(max(0, t - min_down + 1), t + 1)}
        stops[committed[t]] = 1.0
        b.row(stops, -math.inf, float(unit_count - _history_still_locked(shutdown_history, min_down, t)))

    terminal_min = max(config["soc_min"], float(soc_initial) - TERMINAL_SOC_DROP)
    b.row({soc[-1]: 1.0}, terminal_min, math.inf)

    fuel_slope_lph_per_kw = config["fuel_lph_at_rated"] / config["rated_power_kw"]
    objective = np.zeros(len(b.names), dtype=float)
    for t in range(horizon):
        objective[grid[t]] = dt_h * config["grid_price"]
        objective[gen[t]] = dt_h * config["diesel_price"] * fuel_slope_lph_per_kw
        objective[charge[t]] = dt_h * config["storage_degradation"]
        objective[discharge[t]] = dt_h * config["storage_degradation"]
        objective[unserved[t]] = dt_h * config["unserved_penalty"]
        objective[spill[t]] = dt_h * config["spill_penalty"]
        objective[committed[t]] = 1e-8
        objective[startup[t]] = config["startup_cost"] + 1e-7

    matrix, row_lb, row_ub, lower, upper, integrality = b.build()
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, row_lb, row_ub),
        options={"time_limit": 5.0, "mip_rel_gap": 1e-6},
    )
    solve_seconds = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"causal full-MILP failed: status={result.status}, message={result.message}")
    x = result.x
    return {
        "grid_kw": float(x[grid[0]]),
        "generator_kw": float(x[gen[0]]),
        "generator_units_on": float(round(x[committed[0]])),
        "charge_kw": float(x[charge[0]]),
        "discharge_kw": float(x[discharge[0]]),
        "solve_seconds": float(solve_seconds),
    }


def run_scenario(
    trajectory: Dict[str, np.ndarray],
    forecasts: np.ndarray,
    source_start: int,
    config: Dict[str, float],
    initial_soc: float,
    initial_generator_kw: float,
    initial_generator_units: int,
    initial_startup_history: Sequence[int] = (),
    initial_shutdown_history: Sequence[int] = (),
) -> Dict[str, object]:
    actual = trajectory["load_kw"]
    if source_start < 0 or source_start + len(actual) + FORECAST_HORIZON > len(forecasts):
        raise ValueError("causal forecast array does not cover the selected scenario plus horizon")
    dt_h = config["interval_seconds"] / 3600.0
    soc = float(initial_soc)
    previous_generator = float(initial_generator_kw)
    previous_units = int(initial_generator_units)
    startup_history: List[int] = [int(v) for v in initial_startup_history]
    shutdown_history: List[int] = [int(v) for v in initial_shutdown_history]
    records: List[Dict[str, float]] = []
    for t in range(len(actual)):
        start = source_start + t
        try:
            plan = solve_causal_window(
                forecasts[start, :FORECAST_HORIZON],
                np.full(FORECAST_HORIZON, trajectory["grid_available_capacity_kw"][t]),
                np.full(FORECAST_HORIZON, trajectory["generator_available_capacity_kw"][t]),
                np.full(FORECAST_HORIZON, trajectory["storage_available_power_kw"][t]),
                config,
                soc_initial=soc,
                previous_generator_kw=previous_generator,
                previous_generator_units=previous_units,
                startup_history=startup_history,
                shutdown_history=shutdown_history,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "causal full-MILP failed at "
                f"step={t}, source_index={start}, soc={soc}, "
                f"previous_generator_kw={previous_generator}, previous_units={previous_units}, "
                f"generator_capacity_kw={trajectory['generator_available_capacity_kw'][t]}, "
                f"startup_history={startup_history}, shutdown_history={shutdown_history}: {error}"
            ) from error
        grid = min(plan["grid_kw"], trajectory["grid_available_capacity_kw"][t])
        generator = min(plan["generator_kw"], trajectory["generator_available_capacity_kw"][t])
        charge = min(plan["charge_kw"], trajectory["storage_available_power_kw"][t])
        discharge = min(plan["discharge_kw"], trajectory["storage_available_power_kw"][t])
        balance_without_slack = grid + generator + discharge - charge - actual[t]
        unserved = max(0.0, -balance_without_slack)
        spill = max(0.0, balance_without_slack)
        soc = float(
            np.clip(
                soc + (config["charge_efficiency"] * charge - discharge / config["discharge_efficiency"])
                * dt_h
                / config["energy_capacity_kwh"],
                config["soc_min"],
                config["soc_max"],
            )
        )
        units = int(round(plan["generator_units_on"]))
        starts = max(0, units - previous_units)
        stops = max(0, previous_units - units)
        records.append(
            {
                "load_kw": float(actual[t]),
                "grid_kw": float(grid),
                "generator_kw": float(generator),
                "charge_kw": float(charge),
                "discharge_kw": float(discharge),
                "unserved_kw": float(unserved),
                "spill_kw": float(spill),
                "soc": soc,
                "generator_units_on": float(units),
                "generator_startup_units": float(starts),
                "generator_shutdown_units": float(stops),
                "solve_seconds": plan["solve_seconds"],
            }
        )
        previous_generator = generator
        previous_units = units
        startup_history.append(starts)
        shutdown_history.append(stops)
        startup_history = startup_history[-max(0, int(config["minimum_up_steps"]) - 1) :]
        shutdown_history = shutdown_history[-max(0, int(config["minimum_down_steps"]) - 1) :]

    frame = {key: np.asarray([row[key] for row in records], dtype=float) for key in records[0]}
    fuel_slope = config["fuel_lph_at_rated"] / config["rated_power_kw"]
    generator_fuel_l = frame["generator_kw"].sum() * dt_h * fuel_slope
    objective = (
        frame["grid_kw"].sum() * dt_h * config["grid_price"]
        + generator_fuel_l * config["diesel_price"]
        + (frame["charge_kw"] + frame["discharge_kw"]).sum() * dt_h * config["storage_degradation"]
        + frame["unserved_kw"].sum() * dt_h * config["unserved_penalty"]
        + frame["spill_kw"].sum() * dt_h * config["spill_penalty"]
        + frame["generator_startup_units"].sum() * config["startup_cost"]
    )
    residual = (
        frame["grid_kw"]
        + frame["generator_kw"]
        + frame["discharge_kw"]
        + frame["unserved_kw"]
        - frame["charge_kw"]
        - frame["spill_kw"]
        - frame["load_kw"]
    )
    return {
        "unserved_energy_kwh": float(frame["unserved_kw"].sum() * dt_h),
        "objective_yuan": float(objective),
        "grid_energy_kwh": float(frame["grid_kw"].sum() * dt_h),
        "generator_energy_kwh": float(frame["generator_kw"].sum() * dt_h),
        "generator_startups": int(round(frame["generator_startup_units"].sum())),
        "mean_solve_seconds": float(frame["solve_seconds"].mean()),
        "max_solve_seconds": float(frame["solve_seconds"].max()),
        "terminal_soc": float(frame["soc"][-1]),
        "min_soc": float(frame["soc"].min()),
        "max_power_balance_error_kw": float(np.max(np.abs(residual))),
        "max_charge_discharge_product": float(np.max(frame["charge_kw"] * frame["discharge_kw"])),
        "steps": int(len(records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = parse_config(args.config)
    seed_name = args.artifact_root.name
    dispatch_dir = args.artifact_root / "dispatch"
    forecast_dir = args.artifact_root / "source" / "forecast"
    arrays = np.load(forecast_dir / "forecast_predictions.npz")
    key_map = json.loads((forecast_dir / "forecast_prediction_keys.json").read_text(encoding="utf-8"))
    name_to_key = {name: key for key, name in key_map.items()}
    forecast_key = name_to_key["Causal-ErrorFeedback-Ensemble"]
    causal_forecasts = np.asarray(arrays[forecast_key], dtype=float)
    actual_series = np.asarray(arrays["y_true"][:, 0], dtype=float)
    selection = json.loads((dispatch_dir / "scenario_selection.json").read_text(encoding="utf-8"))
    initial_histories = None
    if args.initial_history is not None:
        initial_histories = json.loads(
            args.initial_history.read_text(encoding="utf-8")
        )

    results = []
    for scenario, metadata in selection["scenarios"].items():
        history = (
            initial_histories["scenarios"][scenario]
            if initial_histories is not None
            else {"startup_history": [], "shutdown_history": []}
        )
        trajectory = read_trajectory(dispatch_dir / f"trajectory_{scenario}_risk_soc_supervisory_mpc.csv")
        source_start = int(metadata["start_index"])
        realized = actual_series[source_start : source_start + len(trajectory["load_kw"])]
        if not np.allclose(realized, trajectory["load_kw"], atol=1e-9):
            raise ValueError(f"trajectory/forecast source alignment failed for {seed_name}/{scenario}")
        result = run_scenario(
            trajectory,
            causal_forecasts,
            source_start,
            config,
            float(metadata["comparison_initial_soc_pct"]) / 100.0,
            float(metadata["comparison_initial_generator_kw"]),
            int(metadata["comparison_initial_generator_units"]),
            history["startup_history"],
            history["shutdown_history"],
        )
        result.update(
            {
                "seed": seed_name,
                "scenario": scenario,
                "source_start": source_start,
                "initial_startup_history": history["startup_history"],
                "initial_shutdown_history": history["shutdown_history"],
            }
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    aggregate = {
        "scenario_count": len(results),
        "total_unserved_energy_kwh": float(sum(r["unserved_energy_kwh"] for r in results)),
        "total_objective_yuan": float(sum(r["objective_yuan"] for r in results)),
        "mean_solve_seconds": float(np.mean([r["mean_solve_seconds"] for r in results])),
        "max_solve_seconds": float(max(r["max_solve_seconds"] for r in results)),
        "max_power_balance_error_kw": float(max(r["max_power_balance_error_kw"] for r in results)),
        "max_charge_discharge_product": float(max(r["max_charge_discharge_product"] for r in results)),
    }
    payload = {
        "baseline": "Independent causal rolling full-MILP-MPC",
        "claim_boundary": "Causal 12-step point-forecast comparator using the same plant/capacity traces, cost terms, and shared pre-window unit-transition histories; no risk signal, uncertainty envelope, or SOC supervisory layer.",
        "model_independence": "Standalone scipy.optimize.milp re-encoding; does not import rig_energy.optimization.dispatch.",
        "forecast": {
            "model": "Causal-ErrorFeedback-Ensemble",
            "source": str(forecast_dir / "forecast_predictions.npz"),
            "horizon_steps": FORECAST_HORIZON,
            "decision_rule": "solve at every 5 s and apply the first action",
            "terminal_soc_drop": TERMINAL_SOC_DROP,
        },
        "initial_history": {
            "source": str(args.initial_history) if args.initial_history else None,
            "matched_shared_warmup": bool(args.initial_history),
        },
        "solver": {
            "backend": "HiGHS via scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "time_limit_seconds": 5.0,
            "mip_rel_gap": 1e-6,
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
