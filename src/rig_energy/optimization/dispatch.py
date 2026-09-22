from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp

from ..data.schema import CODE_TO_STATE, STATE_TO_CODE
from .scenario_calibration import (
    audit_cvar_discretization,
    build_residual_scenarios,
)


@dataclass
class Plant:
    step_hours: float
    grid_price: float
    diesel_price: float
    storage_degradation: float
    shortage_penalty: float
    generator_min_kw: float
    generator_max_kw: float
    generator_ramp_kw_per_step: float
    storage_energy_kwh: float
    storage_power_kw: float
    soc_min: float
    soc_max: float
    soc_initial: float
    charge_efficiency: float
    discharge_efficiency: float
    generator_unit_count: int = 1
    generator_unit_rated_kw: float = 1200.0
    generator_unit_min_kw: float = 240.0
    generator_unit_ramp_kw_per_step: float = 120.0
    generator_startup_cost: float = 0.0
    generator_min_up_steps: int = 1
    generator_min_down_steps: int = 1
    storage_low_soc_derating_band: float = 0.0
    storage_high_soc_derating_band: float = 0.0


def _load_config(path: Path) -> tuple[dict, Plant]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    interval_seconds = float(config["dispatch"]["interval_seconds"])
    grid = config["plant"]["grid"]
    gen = config["plant"]["generator"]
    storage = config["plant"]["storage"]
    cost = config["cost"]
    unit_count = int(gen.get("unit_count", 1))
    if unit_count < 1:
        raise ValueError("generator.unit_count 必须至少为1")
    rated_power = float(gen["rated_power_kw"])
    minimum_power = float(gen["minimum_stable_power_kw"])
    ramp_power = float(gen["ramp_kw_per_step"])
    plant = Plant(
        step_hours=interval_seconds / 3600.0,
        grid_price=float(cost["grid_price_yuan_per_kwh"]),
        diesel_price=float(cost["diesel_price_yuan_per_l"]),
        storage_degradation=float(cost["storage_degradation_yuan_per_kwh"]),
        shortage_penalty=float(cost["unserved_energy_penalty_yuan_per_kwh"]),
        generator_min_kw=minimum_power,
        generator_max_kw=rated_power,
        generator_ramp_kw_per_step=ramp_power,
        storage_energy_kwh=float(storage["energy_capacity_kwh"]),
        storage_power_kw=float(storage["rated_power_kw"]),
        soc_min=float(storage["soc_min"]),
        soc_max=float(storage["soc_max"]),
        soc_initial=float(storage["soc_initial"]),
        charge_efficiency=float(storage["charge_efficiency"]),
        discharge_efficiency=float(storage["discharge_efficiency"]),
        generator_unit_count=unit_count,
        generator_unit_rated_kw=float(
            gen.get("unit_rated_power_kw", rated_power / unit_count)
        ),
        generator_unit_min_kw=float(
            gen.get("unit_minimum_stable_power_kw", minimum_power / unit_count)
        ),
        generator_unit_ramp_kw_per_step=float(
            gen.get("unit_ramp_kw_per_step", ramp_power / unit_count)
        ),
        generator_startup_cost=float(
            cost.get("generator_startup_cost_yuan_per_unit", 0.0)
        ),
        generator_min_up_steps=int(gen.get("minimum_up_steps", 1)),
        generator_min_down_steps=int(gen.get("minimum_down_steps", 1)),
        storage_low_soc_derating_band=float(
            storage.get("low_soc_power_derating_band", 0.0)
        ),
        storage_high_soc_derating_band=float(
            storage.get("high_soc_power_derating_band", 0.0)
        ),
    )
    if min(
        plant.generator_unit_rated_kw,
        plant.generator_unit_min_kw,
        plant.generator_unit_ramp_kw_per_step,
    ) <= 0.0:
        raise ValueError("单机额定、最小稳定和爬坡功率必须为正")
    if plant.generator_unit_min_kw > plant.generator_unit_rated_kw:
        raise ValueError("单机最小稳定功率不得高于单机额定功率")
    if min(plant.generator_min_up_steps, plant.generator_min_down_steps) < 1:
        raise ValueError("最小开停时间步数必须至少为1")
    if min(
        plant.storage_low_soc_derating_band,
        plant.storage_high_soc_derating_band,
    ) < 0.0:
        raise ValueError("储能SOC功率降额区间不能为负")
    return config, plant


def _history_still_locked(
    history: tuple[int, ...] | list[int],
    minimum_steps: int,
    future_step: int,
) -> int:
    """Return past unit transitions still locked at a future horizon step.

    History is ordered oldest to newest and contains only already executed
    transitions.  A transition executed one step ago has age one.
    """

    if minimum_steps <= 1:
        return 0
    locked = 0
    for age, count in enumerate(reversed(tuple(history)), start=1):
        if age + int(future_step) < minimum_steps:
            locked += int(count)
    return locked


def _cancel_recent_transition_locks(history: list[int], count: int) -> None:
    """Cancel newest clustered transition locks after a forced outage/recovery."""

    remaining = max(0, int(count))
    for index in range(len(history) - 1, -1, -1):
        if remaining == 0:
            break
        released = min(int(history[index]), remaining)
        history[index] -= released
        remaining -= released


def _convex_generator_fuel_coefficients(
    plant: Plant,
) -> tuple[float, float, float, float]:
    """Return an exact two-segment fuel model over the operating range.

    The tuple is ``(first_slope, on_intercept, breakpoint_kw,
    second_slope_increment)`` for one committed unit.  If the configured curve
    is not convex over the feasible range, the historical rated-point linear
    approximation is retained rather than introducing invalid epigraph logic.
    V22 uses the CAT XQP300 curve with a 50% minimum load and passes this gate.
    """

    rated = float(plant.generator_unit_rated_kw)
    minimum = float(plant.generator_unit_min_kw)
    breakpoint = 0.75 * rated
    if not 0.0 < minimum < breakpoint < rated:
        slope = float(_fuel_lph(rated, rated)) / rated
        return slope, 0.0, rated, 0.0
    fuel_minimum, fuel_breakpoint, fuel_rated = np.asarray(
        _fuel_lph(np.asarray([minimum, breakpoint, rated]), rated),
        dtype=float,
    )
    first_slope = (fuel_breakpoint - fuel_minimum) / (breakpoint - minimum)
    second_slope = (fuel_rated - fuel_breakpoint) / (rated - breakpoint)
    intercept = fuel_minimum - first_slope * minimum
    if (
        first_slope < -1e-12
        or second_slope + 1e-12 < first_slope
        or intercept < -1e-9
    ):
        slope = float(_fuel_lph(rated, rated)) / rated
        return slope, 0.0, rated, 0.0
    return (
        float(first_slope),
        float(max(0.0, intercept)),
        float(breakpoint),
        float(max(0.0, second_slope - first_slope)),
    )


def _solve_clustered_linear_mpc(
    forecast_kw: np.ndarray,
    grid_capacity_kw: float,
    soc: float,
    previous_generator_kw: float,
    plant: Plant,
    *,
    terminal_soc_drop: float,
    generator_available_kw: float | None,
    storage_available_kw: float | None,
    previous_generator_units: int,
    startup_history: tuple[int, ...] | list[int],
    shutdown_history: tuple[int, ...] | list[int],
    minimum_generator_units: int,
) -> dict[str, float]:
    """Clustered unit-commitment MPC for identical generator sets."""

    demand = np.asarray(forecast_kw, dtype=float)
    horizon = len(demand)
    unit_count = int(plant.generator_unit_count)
    previous_units = int(np.clip(previous_generator_units, 0, unit_count))
    g0, d0, c0, b0, u0, w0, s0 = (i * horizon for i in range(7))
    on0 = 7 * horizon + 1
    start0 = on0 + horizon
    stop0 = start0 + horizon
    high0 = stop0 + horizon
    size = high0 + horizon
    objective = np.zeros(size)
    fuel_slope, fuel_intercept, fuel_breakpoint, fuel_high_increment = (
        _convex_generator_fuel_coefficients(plant)
    )
    objective[g0 : g0 + horizon] = plant.grid_price * plant.step_hours
    objective[d0 : d0 + horizon] = (
        plant.diesel_price * fuel_slope * plant.step_hours
    )
    objective[c0 : c0 + horizon] = plant.storage_degradation * plant.step_hours
    objective[b0 : b0 + horizon] = plant.storage_degradation * plant.step_hours
    objective[u0 : u0 + horizon] = plant.shortage_penalty * plant.step_hours
    objective[w0 : w0 + horizon] = 0.5 * plant.step_hours
    objective[on0 : on0 + horizon] = (
        plant.diesel_price * fuel_intercept * plant.step_hours + 1e-8
    )
    objective[start0 : start0 + horizon] = plant.generator_startup_cost + 1e-7
    objective[stop0 : stop0 + horizon] = 1e-7
    objective[high0 : high0 + horizon] = (
        plant.diesel_price * fuel_high_increment * plant.step_hours
    )

    generator_limit = min(
        plant.generator_max_kw,
        plant.generator_max_kw
        if generator_available_kw is None
        else float(generator_available_kw),
    )
    storage_limit = min(
        plant.storage_power_kw,
        plant.storage_power_kw
        if storage_available_kw is None
        else float(storage_available_kw),
    )
    terminal_min = max(plant.soc_min, soc - terminal_soc_drop)
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0.0, grid_capacity_kw)] * horizon)
    bounds.extend([(0.0, generator_limit)] * horizon)
    bounds.extend([(0.0, storage_limit)] * horizon)
    bounds.extend([(0.0, storage_limit)] * horizon)
    bounds.extend([(0.0, None)] * horizon)
    bounds.extend([(0.0, None)] * horizon)
    bounds.extend(
        [(plant.soc_min, plant.soc_max)] * horizon
        + [(terminal_min, plant.soc_max)]
    )
    bounds.extend([(0.0, float(unit_count))] * (3 * horizon))
    bounds.extend(
        [(0.0, unit_count * max(0.0, plant.generator_unit_rated_kw - fuel_breakpoint))]
        * horizon
    )

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for step in range(horizon):
        row = np.zeros(size)
        row[g0 + step] = 1.0
        row[d0 + step] = 1.0
        row[b0 + step] = 1.0
        row[u0 + step] = 1.0
        row[c0 + step] = -1.0
        row[w0 + step] = -1.0
        a_eq.append(row)
        b_eq.append(float(demand[step]))
    row = np.zeros(size)
    row[s0] = 1.0
    a_eq.append(row)
    b_eq.append(float(soc))
    for step in range(horizon):
        row = np.zeros(size)
        row[s0 + step + 1] = 1.0
        row[s0 + step] = -1.0
        row[c0 + step] = (
            -plant.charge_efficiency * plant.step_hours / plant.storage_energy_kwh
        )
        row[b0 + step] = plant.step_hours / (
            plant.discharge_efficiency * plant.storage_energy_kwh
        )
        a_eq.append(row)
        b_eq.append(0.0)
    for step in range(horizon):
        row = np.zeros(size)
        row[on0 + step] = 1.0
        row[start0 + step] = -1.0
        row[stop0 + step] = 1.0
        if step == 0:
            b = float(previous_units)
        else:
            row[on0 + step - 1] = -1.0
            b = 0.0
        a_eq.append(row)
        b_eq.append(b)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for step in range(horizon):
        upper = np.zeros(size)
        upper[d0 + step] = 1.0
        upper[on0 + step] = -plant.generator_unit_rated_kw
        lower = np.zeros(size)
        lower[d0 + step] = -1.0
        lower[on0 + step] = plant.generator_unit_min_kw
        a_ub.extend([upper, lower])
        b_ub.extend([0.0, 0.0])

        high_lower = np.zeros(size)
        high_lower[d0 + step] = 1.0
        high_lower[on0 + step] = -fuel_breakpoint
        high_lower[high0 + step] = -1.0
        high_upper = np.zeros(size)
        high_upper[high0 + step] = 1.0
        high_upper[on0 + step] = -max(
            0.0, plant.generator_unit_rated_kw - fuel_breakpoint
        )
        a_ub.extend([high_lower, high_upper])
        b_ub.extend([0.0, 0.0])

        ramp_up = np.zeros(size)
        ramp_up[d0 + step] = 1.0
        ramp_up[start0 + step] = -plant.generator_unit_min_kw
        ramp_down = np.zeros(size)
        ramp_down[d0 + step] = -1.0
        ramp_down[stop0 + step] = -plant.generator_unit_rated_kw
        if step == 0:
            b_up = (
                previous_generator_kw
                + plant.generator_unit_ramp_kw_per_step * previous_units
            )
            b_down = (
                plant.generator_unit_ramp_kw_per_step * previous_units
                - previous_generator_kw
            )
        else:
            ramp_up[d0 + step - 1] = -1.0
            ramp_up[on0 + step - 1] = -plant.generator_unit_ramp_kw_per_step
            ramp_down[d0 + step - 1] = 1.0
            ramp_down[on0 + step] = -plant.generator_unit_ramp_kw_per_step
            b_up = 0.0
            b_down = 0.0
        a_ub.extend([ramp_up, ramp_down])
        b_ub.extend([b_up, b_down])

        recent_start = np.zeros(size)
        recent_start[on0 + step] = -1.0
        for candidate in range(max(0, step - plant.generator_min_up_steps + 1), step + 1):
            recent_start[start0 + candidate] = 1.0
        past_start = _history_still_locked(
            startup_history, plant.generator_min_up_steps, step
        )
        a_ub.append(recent_start)
        b_ub.append(float(-past_start))

        recent_stop = np.zeros(size)
        recent_stop[on0 + step] = 1.0
        for candidate in range(max(0, step - plant.generator_min_down_steps + 1), step + 1):
            recent_stop[stop0 + candidate] = 1.0
        past_stop = _history_still_locked(
            shutdown_history, plant.generator_min_down_steps, step
        )
        a_ub.append(recent_stop)
        b_ub.append(float(unit_count - past_stop))

    available_units = min(
        unit_count,
        int(np.floor((generator_limit + 1e-9) / plant.generator_unit_min_kw)),
    )
    locked_off_at_first_step = _history_still_locked(
        shutdown_history, plant.generator_min_down_steps, 0
    )
    reachable_units = min(
        available_units, max(0, unit_count - locked_off_at_first_step)
    )
    required_units = int(np.clip(minimum_generator_units, 0, reachable_units))
    if required_units > 0:
        commitment_floor = np.zeros(size)
        commitment_floor[on0] = -1.0
        a_ub.append(commitment_floor)
        b_ub.append(float(-required_units))

    constraints = [
        LinearConstraint(np.asarray(a_eq), np.asarray(b_eq), np.asarray(b_eq)),
        LinearConstraint(np.asarray(a_ub), -np.inf, np.asarray(b_ub)),
    ]
    lower_bounds = np.asarray([item[0] for item in bounds], dtype=float)
    upper_bounds = np.asarray(
        [np.inf if item[1] is None else item[1] for item in bounds], dtype=float
    )
    integrality = np.zeros(size, dtype=np.int8)
    integrality[on0 : stop0 + horizon] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": 5.0, "mip_rel_gap": 1e-4},
    )
    if not result.success:
        raise RuntimeError(
            "多机组滚动优化求解失败: "
            f"{result.message}; previous_units={previous_units}, "
            f"required_units={required_units}, available_units={available_units}, "
            f"reachable_units={reachable_units}, "
            f"locked_off_units={locked_off_at_first_step}, "
            f"previous_generator_kw={previous_generator_kw:.6f}, "
            f"generator_limit_kw={generator_limit:.6f}"
        )
    x = result.x
    return {
        "grid_kw": float(x[g0]),
        "generator_kw": float(x[d0]),
        "generator_units_on": int(round(float(x[on0]))),
        "charge_kw": float(x[c0]),
        "discharge_kw": float(x[b0]),
    }


def _fuel_lph(power_kw: np.ndarray | float, rated_power_kw: float) -> np.ndarray:
    """CAT 300 ekW公开数据按容量比例缩放的分段线性燃油曲线。"""
    power = np.asarray(power_kw, dtype=float)
    scale = rated_power_kw / 300.0
    points_kw = np.array([0.0, 150.0, 225.0, 300.0]) * scale
    points_lph = np.array([0.0, 51.3, 66.7, 86.1]) * scale
    return np.interp(np.clip(power, 0.0, rated_power_kw), points_kw, points_lph)


def _generator_fuel_lph(frame: pd.DataFrame, plant: Plant) -> np.ndarray:
    """Fuel of committed identical units under equal load sharing."""

    power = frame["generator_kw"].to_numpy(dtype=float)
    if plant.generator_unit_count <= 1 or "generator_units_on" not in frame:
        return _fuel_lph(power, plant.generator_max_kw)
    units = frame["generator_units_on"].to_numpy(dtype=int)
    result = np.zeros(len(frame), dtype=float)
    on = units > 0
    result[on] = units[on] * _fuel_lph(
        power[on] / units[on], plant.generator_unit_rated_kw
    )
    return result


def _available_storage_power(soc: float, plant: Plant) -> tuple[float, float]:
    discharge_energy_limited = (
        max(0.0, soc - plant.soc_min)
        * plant.storage_energy_kwh
        * plant.discharge_efficiency
        / plant.step_hours
    )
    charge_energy_limited = (
        max(0.0, plant.soc_max - soc)
        * plant.storage_energy_kwh
        / (plant.charge_efficiency * plant.step_hours)
    )
    discharge_soc_factor = 1.0
    if plant.storage_low_soc_derating_band > 0.0:
        discharge_soc_factor = float(
            np.clip(
                (soc - plant.soc_min) / plant.storage_low_soc_derating_band,
                0.0,
                1.0,
            )
        )
    charge_soc_factor = 1.0
    if plant.storage_high_soc_derating_band > 0.0:
        charge_soc_factor = float(
            np.clip(
                (plant.soc_max - soc) / plant.storage_high_soc_derating_band,
                0.0,
                1.0,
            )
        )
    return (
        min(
            plant.storage_power_kw * discharge_soc_factor,
            discharge_energy_limited,
        ),
        min(
            plant.storage_power_kw * charge_soc_factor,
            charge_energy_limited,
        ),
    )


def _solve_linear_mpc(
    forecast_kw: np.ndarray,
    grid_capacity_kw: float,
    soc: float,
    previous_generator_kw: float,
    plant: Plant,
    *,
    terminal_soc_drop: float,
    generator_available_kw: float | None = None,
    storage_available_kw: float | None = None,
    previous_generator_units: int | None = None,
    startup_history: tuple[int, ...] | list[int] = (),
    shutdown_history: tuple[int, ...] | list[int] = (),
    minimum_generator_units: int = 0,
) -> dict[str, float]:
    if plant.generator_unit_count > 1:
        inferred_units = int(
            np.ceil(max(previous_generator_kw, 0.0) / plant.generator_unit_rated_kw)
        )
        return _solve_clustered_linear_mpc(
            forecast_kw,
            grid_capacity_kw,
            soc,
            previous_generator_kw,
            plant,
            terminal_soc_drop=terminal_soc_drop,
            generator_available_kw=generator_available_kw,
            storage_available_kw=storage_available_kw,
            previous_generator_units=(
                inferred_units
                if previous_generator_units is None
                else previous_generator_units
            ),
            startup_history=startup_history,
            shutdown_history=shutdown_history,
            minimum_generator_units=minimum_generator_units,
        )
    demand = np.asarray(forecast_kw, dtype=float)
    horizon = len(demand)
    # Variable blocks: grid, generator, charge, discharge, unserved, spill,
    # SOC[0:H+1], generator-on[0:H].
    g0, d0, c0, b0, u0, w0, s0 = (i * horizon for i in range(7))
    on0 = 7 * horizon + 1
    size = on0 + horizon
    objective = np.zeros(size)
    objective[g0 : g0 + horizon] = plant.grid_price * plant.step_hours
    # Linear dispatch coefficient approximates the public CAT curve; final fuel is recomputed piecewise.
    generator_cost = plant.diesel_price * _fuel_lph(plant.generator_max_kw, plant.generator_max_kw)
    generator_cost /= plant.generator_max_kw
    objective[d0 : d0 + horizon] = generator_cost * plant.step_hours
    objective[c0 : c0 + horizon] = plant.storage_degradation * plant.step_hours
    objective[b0 : b0 + horizon] = plant.storage_degradation * plant.step_hours
    objective[u0 : u0 + horizon] = plant.shortage_penalty * plant.step_hours
    objective[w0 : w0 + horizon] = 0.5 * plant.step_hours
    objective[on0 : on0 + horizon] = 1e-7

    generator_limit = min(
        plant.generator_max_kw,
        plant.generator_max_kw if generator_available_kw is None else float(generator_available_kw),
    )
    storage_limit = min(
        plant.storage_power_kw,
        plant.storage_power_kw if storage_available_kw is None else float(storage_available_kw),
    )
    bounds = []
    bounds.extend([(0.0, grid_capacity_kw)] * horizon)
    bounds.extend([(0.0, generator_limit)] * horizon)
    bounds.extend([(0.0, storage_limit)] * horizon)
    bounds.extend([(0.0, storage_limit)] * horizon)
    bounds.extend([(0.0, None)] * horizon)
    bounds.extend([(0.0, None)] * horizon)
    terminal_min = max(plant.soc_min, soc - terminal_soc_drop)
    bounds.extend([(plant.soc_min, plant.soc_max)] * horizon + [(terminal_min, plant.soc_max)])
    if generator_limit >= plant.generator_min_kw:
        bounds.extend([(0.0, 1.0)] * horizon)
    else:
        bounds.extend([(0.0, 0.0)] * horizon)

    a_eq = []
    b_eq = []
    for step in range(horizon):
        row = np.zeros(size)
        row[g0 + step] = 1.0
        row[d0 + step] = 1.0
        row[b0 + step] = 1.0
        row[u0 + step] = 1.0
        row[c0 + step] = -1.0
        row[w0 + step] = -1.0
        a_eq.append(row)
        b_eq.append(demand[step])

    row = np.zeros(size)
    row[s0] = 1.0
    a_eq.append(row)
    b_eq.append(soc)
    for step in range(horizon):
        row = np.zeros(size)
        row[s0 + step + 1] = 1.0
        row[s0 + step] = -1.0
        row[c0 + step] = -plant.charge_efficiency * plant.step_hours / plant.storage_energy_kwh
        row[b0 + step] = plant.step_hours / (
            plant.discharge_efficiency * plant.storage_energy_kwh
        )
        a_eq.append(row)
        b_eq.append(0.0)

    a_ub = []
    b_ub = []
    for step in range(horizon):
        # 柴油机只能停机或运行在最小稳定功率以上。
        upper_link = np.zeros(size)
        upper_link[d0 + step] = 1.0
        upper_link[on0 + step] = -generator_limit
        lower_link = np.zeros(size)
        lower_link[d0 + step] = -1.0
        lower_link[on0 + step] = plant.generator_min_kw
        a_ub.extend([upper_link, lower_link])
        b_ub.extend([0.0, 0.0])

        up = np.zeros(size)
        up[d0 + step] = 1.0
        down = np.zeros(size)
        down[d0 + step] = -1.0
        if step == 0:
            previous_on = previous_generator_kw >= plant.generator_min_kw - 1e-6
            # 已运行时执行常规爬坡；停机到启机或启机到停机允许跨越不连续边界。
            up_limit = (
                plant.generator_ramp_kw_per_step + previous_generator_kw
                if previous_on
                else generator_limit
            )
            down[on0 + step] = generator_limit
            a_ub.extend([up, down])
            b_ub.extend(
                [
                    up_limit,
                    plant.generator_ramp_kw_per_step
                    + generator_limit
                    - previous_generator_kw,
                ]
            )
        else:
            up[d0 + step - 1] = -1.0
            up[on0 + step - 1] = generator_limit
            down[d0 + step - 1] = 1.0
            down[on0 + step] = generator_limit
            a_ub.extend([up, down])
            b_ub.extend(
                [
                    plant.generator_ramp_kw_per_step + generator_limit,
                    plant.generator_ramp_kw_per_step + generator_limit,
                ]
            )

    constraints = [
        LinearConstraint(np.asarray(a_eq), np.asarray(b_eq), np.asarray(b_eq)),
        LinearConstraint(np.asarray(a_ub), -np.inf, np.asarray(b_ub)),
    ]
    lower_bounds = np.asarray([item[0] for item in bounds], dtype=float)
    upper_bounds = np.asarray(
        [np.inf if item[1] is None else item[1] for item in bounds], dtype=float
    )
    integrality = np.zeros(size, dtype=np.int8)
    integrality[on0 : on0 + horizon] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": 5.0, "mip_rel_gap": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"滚动优化求解失败: {result.message}")
    x = result.x
    return {
        "grid_kw": float(x[g0]),
        "generator_kw": float(x[d0]),
        "generator_units_on": int(float(x[d0]) >= plant.generator_min_kw - 1e-6),
        "charge_kw": float(x[c0]),
        "discharge_kw": float(x[b0]),
    }


def _solve_clustered_scenario_cvar_mpc(
    forecast_scenarios_kw: np.ndarray,
    grid_capacity_kw: float,
    soc: float,
    previous_generator_kw: float,
    plant: Plant,
    *,
    terminal_soc_drop: float,
    cvar_alpha: float,
    cvar_weight: float,
    generator_available_kw: float | None,
    storage_available_kw: float | None,
    previous_generator_units: int,
    startup_history: tuple[int, ...] | list[int],
    shutdown_history: tuple[int, ...] | list[int],
    minimum_generator_units: int,
) -> dict[str, float]:
    """Scenario-CVaR MPC with shared clustered unit commitment."""

    scenarios = np.maximum(np.asarray(forecast_scenarios_kw, dtype=float), 0.0)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2 or scenarios.shape[1] < 1:
        raise ValueError("场景-CVaR MPC 需要 [场景数,时域] 二维预测，且至少2个场景")
    alpha = float(cvar_alpha)
    weight = float(cvar_weight)
    if not 0.0 < alpha < 1.0:
        raise ValueError("cvar_alpha 必须在 (0, 1) 内")
    if weight < 0.0:
        raise ValueError("cvar_weight 不能为负")

    scenario_count, horizon = scenarios.shape
    unit_count = int(plant.generator_unit_count)
    previous_units = int(np.clip(previous_generator_units, 0, unit_count))
    block_size = 8 * horizon + 1
    on_start = scenario_count * block_size
    startup_start = on_start + horizon
    shutdown_start = startup_start + horizon
    eta_index = shutdown_start + horizon
    z_start = eta_index + 1
    size = z_start + scenario_count
    objective = np.zeros(size)
    probability = 1.0 / scenario_count
    fuel_slope, fuel_intercept, fuel_breakpoint, fuel_high_increment = (
        _convex_generator_fuel_coefficients(plant)
    )
    generator_limit = min(
        plant.generator_max_kw,
        plant.generator_max_kw
        if generator_available_kw is None
        else float(generator_available_kw),
    )
    storage_limit = min(
        plant.storage_power_kw,
        plant.storage_power_kw
        if storage_available_kw is None
        else float(storage_available_kw),
    )
    terminal_min = max(plant.soc_min, soc - terminal_soc_drop)
    bounds: list[tuple[float | None, float | None]] = []
    for scenario in range(scenario_count):
        base = scenario * block_size
        g0, d0, c0, b0, u0, w0, _ = (
            base + offset * horizon for offset in range(7)
        )
        high0 = base + 7 * horizon + 1
        objective[g0 : g0 + horizon] = probability * plant.grid_price * plant.step_hours
        objective[d0 : d0 + horizon] = (
            probability * plant.diesel_price * fuel_slope * plant.step_hours
        )
        objective[c0 : c0 + horizon] = probability * plant.storage_degradation * plant.step_hours
        objective[b0 : b0 + horizon] = probability * plant.storage_degradation * plant.step_hours
        objective[u0 : u0 + horizon] = probability * plant.shortage_penalty * plant.step_hours
        objective[w0 : w0 + horizon] = probability * 0.5 * plant.step_hours
        objective[high0 : high0 + horizon] = (
            probability
            * plant.diesel_price
            * fuel_high_increment
            * plant.step_hours
        )
        bounds.extend([(0.0, grid_capacity_kw)] * horizon)
        bounds.extend([(0.0, generator_limit)] * horizon)
        bounds.extend([(0.0, storage_limit)] * horizon)
        bounds.extend([(0.0, storage_limit)] * horizon)
        bounds.extend([(0.0, None)] * horizon)
        bounds.extend([(0.0, None)] * horizon)
        bounds.extend(
            [(plant.soc_min, plant.soc_max)] * horizon
            + [(terminal_min, plant.soc_max)]
        )
        bounds.extend(
            [
                (
                    0.0,
                    unit_count
                    * max(
                        0.0,
                        plant.generator_unit_rated_kw - fuel_breakpoint,
                    ),
                )
            ]
            * horizon
        )
    objective[on_start : on_start + horizon] = (
        plant.diesel_price * fuel_intercept * plant.step_hours + 1e-8
    )
    objective[startup_start : startup_start + horizon] = (
        plant.generator_startup_cost + 1e-7
    )
    objective[shutdown_start : shutdown_start + horizon] = 1e-7
    bounds.extend([(0.0, float(unit_count))] * (3 * horizon))
    objective[eta_index] = weight
    objective[z_start : z_start + scenario_count] = weight / (
        (1.0 - alpha) * scenario_count
    )
    bounds.append((0.0, None))
    bounds.extend([(0.0, None)] * scenario_count)

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for scenario in range(scenario_count):
        base = scenario * block_size
        g0, d0, c0, b0, u0, w0, s0 = (
            base + offset * horizon for offset in range(7)
        )
        for step in range(horizon):
            row = np.zeros(size)
            row[g0 + step] = 1.0
            row[d0 + step] = 1.0
            row[b0 + step] = 1.0
            row[u0 + step] = 1.0
            row[c0 + step] = -1.0
            row[w0 + step] = -1.0
            a_eq.append(row)
            b_eq.append(float(scenarios[scenario, step]))
        row = np.zeros(size)
        row[s0] = 1.0
        a_eq.append(row)
        b_eq.append(float(soc))
        for step in range(horizon):
            row = np.zeros(size)
            row[s0 + step + 1] = 1.0
            row[s0 + step] = -1.0
            row[c0 + step] = (
                -plant.charge_efficiency
                * plant.step_hours
                / plant.storage_energy_kwh
            )
            row[b0 + step] = plant.step_hours / (
                plant.discharge_efficiency * plant.storage_energy_kwh
            )
            a_eq.append(row)
            b_eq.append(0.0)
    for scenario in range(1, scenario_count):
        for offset in (0, 1, 2, 3):
            row = np.zeros(size)
            row[scenario * block_size + offset * horizon] = 1.0
            row[offset * horizon] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)
    for step in range(horizon):
        row = np.zeros(size)
        row[on_start + step] = 1.0
        row[startup_start + step] = -1.0
        row[shutdown_start + step] = 1.0
        if step == 0:
            b = float(previous_units)
        else:
            row[on_start + step - 1] = -1.0
            b = 0.0
        a_eq.append(row)
        b_eq.append(b)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for scenario in range(scenario_count):
        base = scenario * block_size
        d0 = base + horizon
        u0 = base + 4 * horizon
        high0 = base + 7 * horizon + 1
        for step in range(horizon):
            upper = np.zeros(size)
            upper[d0 + step] = 1.0
            upper[on_start + step] = -plant.generator_unit_rated_kw
            lower = np.zeros(size)
            lower[d0 + step] = -1.0
            lower[on_start + step] = plant.generator_unit_min_kw
            a_ub.extend([upper, lower])
            b_ub.extend([0.0, 0.0])

            high_lower = np.zeros(size)
            high_lower[d0 + step] = 1.0
            high_lower[on_start + step] = -fuel_breakpoint
            high_lower[high0 + step] = -1.0
            high_upper = np.zeros(size)
            high_upper[high0 + step] = 1.0
            high_upper[on_start + step] = -max(
                0.0, plant.generator_unit_rated_kw - fuel_breakpoint
            )
            a_ub.extend([high_lower, high_upper])
            b_ub.extend([0.0, 0.0])

            ramp_up = np.zeros(size)
            ramp_up[d0 + step] = 1.0
            ramp_up[startup_start + step] = -plant.generator_unit_min_kw
            ramp_down = np.zeros(size)
            ramp_down[d0 + step] = -1.0
            ramp_down[shutdown_start + step] = -plant.generator_unit_rated_kw
            if step == 0:
                b_up = (
                    previous_generator_kw
                    + plant.generator_unit_ramp_kw_per_step * previous_units
                )
                b_down = (
                    plant.generator_unit_ramp_kw_per_step * previous_units
                    - previous_generator_kw
                )
            else:
                ramp_up[d0 + step - 1] = -1.0
                ramp_up[on_start + step - 1] = -plant.generator_unit_ramp_kw_per_step
                ramp_down[d0 + step - 1] = 1.0
                ramp_down[on_start + step] = -plant.generator_unit_ramp_kw_per_step
                b_up = 0.0
                b_down = 0.0
            a_ub.extend([ramp_up, ramp_down])
            b_ub.extend([b_up, b_down])
        row = np.zeros(size)
        row[u0 : u0 + horizon] = plant.shortage_penalty * plant.step_hours
        row[eta_index] = -1.0
        row[z_start + scenario] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    for step in range(horizon):
        recent_start = np.zeros(size)
        recent_start[on_start + step] = -1.0
        for candidate in range(max(0, step - plant.generator_min_up_steps + 1), step + 1):
            recent_start[startup_start + candidate] = 1.0
        a_ub.append(recent_start)
        b_ub.append(
            float(
                -_history_still_locked(
                    startup_history, plant.generator_min_up_steps, step
                )
            )
        )
        recent_stop = np.zeros(size)
        recent_stop[on_start + step] = 1.0
        for candidate in range(max(0, step - plant.generator_min_down_steps + 1), step + 1):
            recent_stop[shutdown_start + candidate] = 1.0
        a_ub.append(recent_stop)
        b_ub.append(
            float(
                unit_count
                - _history_still_locked(
                    shutdown_history, plant.generator_min_down_steps, step
                )
            )
        )

    available_units = min(
        unit_count,
        int(np.floor((generator_limit + 1e-9) / plant.generator_unit_min_kw)),
    )
    locked_off_at_first_step = _history_still_locked(
        shutdown_history, plant.generator_min_down_steps, 0
    )
    reachable_units = min(
        available_units, max(0, unit_count - locked_off_at_first_step)
    )
    required_units = int(np.clip(minimum_generator_units, 0, reachable_units))
    if required_units > 0:
        commitment_floor = np.zeros(size)
        commitment_floor[on_start] = -1.0
        a_ub.append(commitment_floor)
        b_ub.append(float(-required_units))

    constraints = [
        LinearConstraint(np.asarray(a_eq), np.asarray(b_eq), np.asarray(b_eq)),
        LinearConstraint(np.asarray(a_ub), -np.inf, np.asarray(b_ub)),
    ]
    lower_bounds = np.asarray([item[0] for item in bounds], dtype=float)
    upper_bounds = np.asarray(
        [np.inf if item[1] is None else item[1] for item in bounds], dtype=float
    )
    integrality = np.zeros(size, dtype=np.int8)
    integrality[on_start : shutdown_start + horizon] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": 5.0, "mip_rel_gap": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"多机组场景-CVaR滚动优化求解失败: {result.message}")
    x = result.x
    predicted_shortage_costs = np.asarray(
        [
            plant.shortage_penalty
            * plant.step_hours
            * x[scenario * block_size + 4 * horizon : scenario * block_size + 5 * horizon].sum()
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    predicted_cvar = float(
        x[eta_index]
        + x[z_start : z_start + scenario_count].sum()
        / ((1.0 - alpha) * scenario_count)
    )
    return {
        "grid_kw": float(x[0]),
        "generator_kw": float(x[horizon]),
        "generator_units_on": int(round(float(x[on_start]))),
        "charge_kw": float(x[2 * horizon]),
        "discharge_kw": float(x[3 * horizon]),
        "predicted_expected_shortage_cost_yuan": float(
            predicted_shortage_costs.mean()
        ),
        "predicted_cvar_shortage_cost_yuan": predicted_cvar,
        "predicted_weighted_cvar_term_yuan": weight * predicted_cvar,
    }


def _solve_scenario_cvar_mpc(
    forecast_scenarios_kw: np.ndarray,
    grid_capacity_kw: float,
    soc: float,
    previous_generator_kw: float,
    plant: Plant,
    *,
    terminal_soc_drop: float,
    cvar_alpha: float,
    cvar_weight: float,
    generator_available_kw: float | None = None,
    storage_available_kw: float | None = None,
    previous_generator_units: int | None = None,
    startup_history: tuple[int, ...] | list[int] = (),
    shutdown_history: tuple[int, ...] | list[int] = (),
    minimum_generator_units: int = 0,
) -> dict[str, float]:
    """Two-stage scenario MPC with CVaR on unserved-energy cost.

    The first power-control action is non-anticipative across all forecast
    members.  A shared generator commitment schedule avoids pretending that a
    physical unit can be on in one forecast scenario and off in another, while
    later power levels remain scenario recourse.  This is materially different
    from adding a fixed reserve to one point forecast.
    """

    if plant.generator_unit_count > 1:
        inferred_units = int(
            np.ceil(max(previous_generator_kw, 0.0) / plant.generator_unit_rated_kw)
        )
        return _solve_clustered_scenario_cvar_mpc(
            forecast_scenarios_kw,
            grid_capacity_kw,
            soc,
            previous_generator_kw,
            plant,
            terminal_soc_drop=terminal_soc_drop,
            cvar_alpha=cvar_alpha,
            cvar_weight=cvar_weight,
            generator_available_kw=generator_available_kw,
            storage_available_kw=storage_available_kw,
            previous_generator_units=(
                inferred_units
                if previous_generator_units is None
                else previous_generator_units
            ),
            startup_history=startup_history,
            shutdown_history=shutdown_history,
            minimum_generator_units=minimum_generator_units,
        )
    scenarios = np.maximum(np.asarray(forecast_scenarios_kw, dtype=float), 0.0)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2 or scenarios.shape[1] < 1:
        raise ValueError("场景-CVaR MPC 需要 [场景数,时域] 二维预测，且至少2个场景")
    alpha = float(cvar_alpha)
    weight = float(cvar_weight)
    if not 0.0 < alpha < 1.0:
        raise ValueError("cvar_alpha 必须在 (0, 1) 内")
    if weight < 0.0:
        raise ValueError("cvar_weight 不能为负")

    scenario_count, horizon = scenarios.shape
    # Per-scenario continuous blocks: grid, generator, charge, discharge,
    # unserved, spill, SOC[0:H+1]. One shared generator commitment schedule
    # reduces the integer dimension from scenarios*horizon to horizon.
    block_size = 7 * horizon + 1
    on_start = scenario_count * block_size
    eta_index = on_start + horizon
    z_start = eta_index + 1
    size = z_start + scenario_count
    objective = np.zeros(size)
    probability = 1.0 / scenario_count
    generator_cost = (
        plant.diesel_price
        * _fuel_lph(plant.generator_max_kw, plant.generator_max_kw)
        / plant.generator_max_kw
    )
    generator_limit = min(
        plant.generator_max_kw,
        plant.generator_max_kw
        if generator_available_kw is None
        else float(generator_available_kw),
    )
    storage_limit = min(
        plant.storage_power_kw,
        plant.storage_power_kw
        if storage_available_kw is None
        else float(storage_available_kw),
    )

    bounds: list[tuple[float | None, float | None]] = []
    terminal_min = max(plant.soc_min, soc - terminal_soc_drop)
    for scenario in range(scenario_count):
        base = scenario * block_size
        g0, d0, c0, b0, u0, w0, s0 = (
            base + offset * horizon for offset in range(7)
        )
        objective[g0 : g0 + horizon] = (
            probability * plant.grid_price * plant.step_hours
        )
        objective[d0 : d0 + horizon] = (
            probability * generator_cost * plant.step_hours
        )
        objective[c0 : c0 + horizon] = (
            probability * plant.storage_degradation * plant.step_hours
        )
        objective[b0 : b0 + horizon] = (
            probability * plant.storage_degradation * plant.step_hours
        )
        objective[u0 : u0 + horizon] = (
            probability * plant.shortage_penalty * plant.step_hours
        )
        objective[w0 : w0 + horizon] = probability * 0.5 * plant.step_hours
        bounds.extend([(0.0, grid_capacity_kw)] * horizon)
        bounds.extend([(0.0, generator_limit)] * horizon)
        bounds.extend([(0.0, storage_limit)] * horizon)
        bounds.extend([(0.0, storage_limit)] * horizon)
        bounds.extend([(0.0, None)] * horizon)
        bounds.extend([(0.0, None)] * horizon)
        bounds.extend(
            [(plant.soc_min, plant.soc_max)] * horizon
            + [(terminal_min, plant.soc_max)]
        )
    objective[on_start : on_start + horizon] = 1e-7
    if generator_limit >= plant.generator_min_kw:
        bounds.extend([(0.0, 1.0)] * horizon)
    else:
        bounds.extend([(0.0, 0.0)] * horizon)
    objective[eta_index] = weight
    objective[z_start : z_start + scenario_count] = weight / (
        (1.0 - alpha) * scenario_count
    )
    bounds.append((0.0, None))
    bounds.extend([(0.0, None)] * scenario_count)

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for scenario in range(scenario_count):
        base = scenario * block_size
        g0, d0, c0, b0, u0, w0, s0 = (
            base + offset * horizon for offset in range(7)
        )
        for step in range(horizon):
            row = np.zeros(size)
            row[g0 + step] = 1.0
            row[d0 + step] = 1.0
            row[b0 + step] = 1.0
            row[u0 + step] = 1.0
            row[c0 + step] = -1.0
            row[w0 + step] = -1.0
            a_eq.append(row)
            b_eq.append(float(scenarios[scenario, step]))
        row = np.zeros(size)
        row[s0] = 1.0
        a_eq.append(row)
        b_eq.append(float(soc))
        for step in range(horizon):
            row = np.zeros(size)
            row[s0 + step + 1] = 1.0
            row[s0 + step] = -1.0
            row[c0 + step] = (
                -plant.charge_efficiency
                * plant.step_hours
                / plant.storage_energy_kwh
            )
            row[b0 + step] = plant.step_hours / (
                plant.discharge_efficiency * plant.storage_energy_kwh
            )
            a_eq.append(row)
            b_eq.append(0.0)

    # Only the first implementable control is shared; future controls are
    # scenario-dependent recourse and are re-optimized at the next 5 s step.
    reference = 0
    for scenario in range(1, scenario_count):
        for offset in (0, 1, 2, 3):
            row = np.zeros(size)
            row[scenario * block_size + offset * horizon] = 1.0
            row[reference * block_size + offset * horizon] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for scenario in range(scenario_count):
        base = scenario * block_size
        d0 = base + horizon
        u0 = base + 4 * horizon
        for step in range(horizon):
            upper_link = np.zeros(size)
            upper_link[d0 + step] = 1.0
            upper_link[on_start + step] = -generator_limit
            lower_link = np.zeros(size)
            lower_link[d0 + step] = -1.0
            lower_link[on_start + step] = plant.generator_min_kw
            a_ub.extend([upper_link, lower_link])
            b_ub.extend([0.0, 0.0])

            up = np.zeros(size)
            down = np.zeros(size)
            up[d0 + step] = 1.0
            down[d0 + step] = -1.0
            if step == 0:
                previous_on = (
                    previous_generator_kw >= plant.generator_min_kw - 1e-6
                )
                up_limit = (
                    plant.generator_ramp_kw_per_step + previous_generator_kw
                    if previous_on
                    else generator_limit
                )
                down[on_start + step] = generator_limit
                b_ub.extend(
                    [
                        up_limit,
                        plant.generator_ramp_kw_per_step
                        + generator_limit
                        - previous_generator_kw,
                    ]
                )
            else:
                up[d0 + step - 1] = -1.0
                up[on_start + step - 1] = generator_limit
                down[d0 + step - 1] = 1.0
                down[on_start + step] = generator_limit
                b_ub.extend(
                    [
                        plant.generator_ramp_kw_per_step + generator_limit,
                        plant.generator_ramp_kw_per_step + generator_limit,
                    ]
                )
            a_ub.extend([up, down])
        # shortage_cost_s - eta - z_s <= 0
        row = np.zeros(size)
        row[u0 : u0 + horizon] = plant.shortage_penalty * plant.step_hours
        row[eta_index] = -1.0
        row[z_start + scenario] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    required_on = int(
        minimum_generator_units > 0
        and generator_limit >= plant.generator_min_kw
    )
    if required_on:
        commitment_floor = np.zeros(size)
        commitment_floor[on_start] = -1.0
        a_ub.append(commitment_floor)
        b_ub.append(-1.0)

    constraints = [
        LinearConstraint(np.asarray(a_eq), np.asarray(b_eq), np.asarray(b_eq)),
        LinearConstraint(np.asarray(a_ub), -np.inf, np.asarray(b_ub)),
    ]
    lower_bounds = np.asarray([item[0] for item in bounds], dtype=float)
    upper_bounds = np.asarray(
        [np.inf if item[1] is None else item[1] for item in bounds], dtype=float
    )
    integrality = np.zeros(size, dtype=np.int8)
    integrality[on_start : on_start + horizon] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": 5.0, "mip_rel_gap": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"场景-CVaR滚动优化求解失败: {result.message}")
    x = result.x
    predicted_shortage_costs = np.asarray(
        [
            plant.shortage_penalty
            * plant.step_hours
            * x[scenario * block_size + 4 * horizon : scenario * block_size + 5 * horizon].sum()
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    predicted_cvar = float(
        x[eta_index]
        + x[z_start : z_start + scenario_count].sum()
        / ((1.0 - alpha) * scenario_count)
    )
    return {
        "grid_kw": float(x[0]),
        "generator_kw": float(x[horizon]),
        "generator_units_on": int(
            float(x[horizon]) >= plant.generator_min_kw - 1e-6
        ),
        "charge_kw": float(x[2 * horizon]),
        "discharge_kw": float(x[3 * horizon]),
        "predicted_expected_shortage_cost_yuan": float(
            predicted_shortage_costs.mean()
        ),
        "predicted_cvar_shortage_cost_yuan": predicted_cvar,
        "predicted_weighted_cvar_term_yuan": weight * predicted_cvar,
    }


def _execute_with_safety_layer(
    actual_load_kw: float,
    command: dict[str, float],
    grid_capacity_kw: float,
    soc: float,
    previous_generator_kw: float,
    plant: Plant,
    generator_available_kw: float | None = None,
    storage_available_kw: float | None = None,
    previous_generator_units: int | None = None,
    startup_history: tuple[int, ...] | list[int] = (),
    shutdown_history: tuple[int, ...] | list[int] = (),
    generator_before_storage: bool = False,
    minimum_generator_units: int = 0,
    grid_headroom_reserve_target_soc: float | None = None,
    grid_headroom_reserve_max_charge_kw: float = 0.0,
    maximum_synchronized_generator_units: int | None = None,
    newly_synchronized_generator_units: int = 0,
) -> tuple[dict[str, float], float]:
    max_discharge, max_charge = _available_storage_power(soc, plant)
    if storage_available_kw is not None:
        max_discharge = min(max_discharge, float(storage_available_kw))
        max_charge = min(max_charge, float(storage_available_kw))
    grid = float(np.clip(command["grid_kw"], 0.0, grid_capacity_kw))
    generator_capacity = min(
        plant.generator_max_kw,
        plant.generator_max_kw if generator_available_kw is None else float(generator_available_kw),
    )
    minimum_available_generator = (
        plant.generator_unit_min_kw
        if plant.generator_unit_count > 1
        else plant.generator_min_kw
    )
    if generator_capacity < minimum_available_generator:
        generator_capacity = 0.0
    # The 5 s execution layer sees the current metered load. If grid plus the
    # physically available battery cannot serve it, emergency commitment can
    # override a forecast command that left generator sets off.
    emergency_generator_needed = (
        actual_load_kw > grid_capacity_kw + max_discharge + 1e-6
    )
    if plant.generator_unit_count > 1:
        inferred_previous_units = int(
            np.ceil(max(previous_generator_kw, 0.0) / plant.generator_unit_rated_kw)
        )
        previous_units = int(
            np.clip(
                inferred_previous_units
                if previous_generator_units is None
                else previous_generator_units,
                0,
                plant.generator_unit_count,
            )
        )
        requested_units = int(
            command.get(
                "generator_units_on",
                np.ceil(
                    max(float(command["generator_kw"]), 0.0)
                    / plant.generator_unit_rated_kw
                ),
            )
        )
        if emergency_generator_needed:
            immediate_deficit = max(
                0.0, actual_load_kw - grid_capacity_kw - max_discharge
            )
            requested_units = max(
                requested_units,
                int(np.ceil(immediate_deficit / plant.generator_unit_min_kw)),
            )
        capacity_units = min(
            plant.generator_unit_count,
            int(np.floor((generator_capacity + 1e-9) / plant.generator_unit_min_kw)),
        )
        requested_units = max(
            requested_units,
            int(np.clip(minimum_generator_units, 0, capacity_units)),
        )
        locked_on = _history_still_locked(
            startup_history, plant.generator_min_up_steps, 0
        )
        locked_off = _history_still_locked(
            shutdown_history, plant.generator_min_down_steps, 0
        )
        minimum_units = min(capacity_units, locked_on)
        maximum_units = min(capacity_units, plant.generator_unit_count - locked_off)
        if maximum_synchronized_generator_units is not None:
            maximum_units = min(
                maximum_units,
                max(0, int(maximum_synchronized_generator_units)),
            )
            minimum_units = min(
                maximum_units,
                max(
                    minimum_units,
                    min(
                        capacity_units,
                        locked_on + int(newly_synchronized_generator_units),
                    ),
                ),
            )
        committed_units = int(
            np.clip(requested_units, minimum_units, max(minimum_units, maximum_units))
        )
        started_units = max(0, committed_units - previous_units)
        stopped_units = max(0, previous_units - committed_units)
        overlap_units = min(previous_units, committed_units)
        if committed_units == 0 or generator_capacity == 0.0:
            generator_lower = 0.0
            generator_upper = 0.0
        else:
            generator_lower = max(
                plant.generator_unit_min_kw * committed_units,
                previous_generator_kw
                - plant.generator_unit_ramp_kw_per_step * overlap_units
                - plant.generator_unit_rated_kw * stopped_units,
            )
            generator_upper = min(
                generator_capacity,
                plant.generator_unit_rated_kw * committed_units,
                previous_generator_kw
                + plant.generator_unit_ramp_kw_per_step * overlap_units
                + plant.generator_unit_min_kw * started_units,
            )
    else:
        requested_on = (
            float(command["generator_kw"]) > 1e-6 or emergency_generator_needed
        )
        previously_on = previous_generator_kw >= plant.generator_min_kw - 1e-6
        if not requested_on or generator_capacity == 0.0:
            generator_lower = 0.0
            generator_upper = 0.0
        elif previously_on:
            generator_lower = max(
                plant.generator_min_kw,
                previous_generator_kw - plant.generator_ramp_kw_per_step,
            )
            generator_upper = min(
                generator_capacity,
                previous_generator_kw + plant.generator_ramp_kw_per_step,
            )
        else:
            # 启机后直接进入最小稳定运行区；常规爬坡约束只作用于已运行机组。
            generator_lower = plant.generator_min_kw
            generator_upper = min(
                generator_capacity,
                max(plant.generator_min_kw, plant.generator_ramp_kw_per_step),
            )
        committed_units = int(generator_upper > 0.0)
    generator = float(np.clip(command["generator_kw"], generator_lower, generator_upper))
    discharge = float(np.clip(command["discharge_kw"], 0.0, max_discharge))
    charge = float(np.clip(command["charge_kw"], 0.0, max_charge))
    # 正成本、损耗小于1时，同时充放电始终可由等价净功率支配。
    # 先净额化，再执行实时功率平衡，避免安全层制造物理伪动作。
    storage_net = discharge - charge
    discharge = max(storage_net, 0.0)
    charge = max(-storage_net, 0.0)
    if generator_before_storage and discharge > 0.0:
        replacement = min(discharge, max(0.0, generator_upper - generator))
        generator += replacement
        discharge -= replacement

    net_supply = grid + generator + discharge - charge
    deficit = actual_load_kw - net_supply
    if deficit > 0.0:
        reduce = min(deficit, charge)
        charge -= reduce
        deficit -= reduce
        add = min(deficit, grid_capacity_kw - grid)
        grid += add
        deficit -= add
        if generator_before_storage:
            add = min(deficit, generator_upper - generator)
            generator += add
            deficit -= add
            add = min(deficit, max_discharge - discharge)
            discharge += add
            deficit -= add
        else:
            add = min(deficit, max_discharge - discharge)
            discharge += add
            deficit -= add
            add = min(deficit, generator_upper - generator)
            generator += add
            deficit -= add
        unserved = max(0.0, deficit)
        spill = 0.0
    else:
        surplus = -deficit
        reduce = min(surplus, discharge)
        discharge -= reduce
        surplus -= reduce
        reduce = min(surplus, grid)
        grid -= reduce
        surplus -= reduce
        add = min(surplus, max_charge - charge)
        charge += add
        surplus -= add
        reduce = min(surplus, max(0.0, generator - generator_lower))
        generator -= reduce
        surplus -= reduce
        unserved = 0.0
        spill = max(0.0, surplus)

    grid_headroom_storage_preservation_kw = 0.0
    grid_headroom_storage_charge_kw = 0.0
    if grid_headroom_reserve_target_soc is not None:
        reserve_target = float(grid_headroom_reserve_target_soc)
        reserve_max_charge = float(grid_headroom_reserve_max_charge_kw)
        if not plant.soc_min <= reserve_target <= plant.soc_max:
            raise ValueError("网电余量储能储备目标超出SOC物理范围")
        if reserve_max_charge < 0.0:
            raise ValueError("网电余量储能储备充电功率不得为负")
        if soc < reserve_target - 1e-12 and unserved <= 1e-9:
            grid_headroom = max(0.0, grid_capacity_kw - grid)
            grid_headroom_storage_preservation_kw = min(
                discharge, grid_headroom, reserve_max_charge
            )
            grid += grid_headroom_storage_preservation_kw
            discharge -= grid_headroom_storage_preservation_kw
            grid_headroom -= grid_headroom_storage_preservation_kw

            target_limited_charge_kw = (
                (reserve_target - soc)
                * plant.storage_energy_kwh
                / (plant.charge_efficiency * plant.step_hours)
            )
            remaining_reserve_power = max(
                0.0,
                reserve_max_charge - grid_headroom_storage_preservation_kw,
            )
            grid_headroom_storage_charge_kw = min(
                max_charge - charge,
                grid_headroom,
                remaining_reserve_power,
                target_limited_charge_kw,
            )
            grid += grid_headroom_storage_charge_kw
            charge += grid_headroom_storage_charge_kw

    soc_next = soc + (
        plant.charge_efficiency * charge
        - discharge / plant.discharge_efficiency
    ) * plant.step_hours / plant.storage_energy_kwh
    soc_next = float(np.clip(soc_next, plant.soc_min, plant.soc_max))
    return (
        {
            "load_kw": actual_load_kw,
            "grid_kw": grid,
            "generator_kw": generator,
            "generator_units_on": committed_units,
            "storage_kw": discharge - charge,
            "charge_kw": charge,
            "discharge_kw": discharge,
            "unserved_kw": unserved,
            "spill_kw": spill,
            "soc": soc_next,
            "grid_headroom_storage_preservation_kw": (
                grid_headroom_storage_preservation_kw
            ),
            "grid_headroom_storage_charge_kw": grid_headroom_storage_charge_kw,
        },
        soc_next,
    )


def _rule_command(
    load_kw: float,
    grid_capacity_kw: float,
    soc: float,
    plant: Plant,
    generator_available_kw: float | None = None,
    storage_available_kw: float | None = None,
) -> dict[str, float]:
    grid = min(load_kw, grid_capacity_kw)
    deficit = max(0.0, load_kw - grid)
    max_discharge, max_charge = _available_storage_power(soc, plant)
    if storage_available_kw is not None:
        max_discharge = min(max_discharge, float(storage_available_kw))
        max_charge = min(max_charge, float(storage_available_kw))
    generator_capacity = min(
        plant.generator_max_kw,
        plant.generator_max_kw if generator_available_kw is None else float(generator_available_kw),
    )
    generator_efficient_kw = min(0.75 * plant.generator_max_kw, generator_capacity)
    generator = min(deficit, generator_efficient_kw)
    discharge = min(max(0.0, deficit - generator), max_discharge)
    charge = 0.0
    if deficit <= 0.0 and soc < 0.75:
        charge = min(0.20 * plant.storage_power_kw, max_charge, grid_capacity_kw - grid)
        grid += charge
    return {
        "grid_kw": grid,
        "generator_kw": generator,
        "charge_kw": charge,
        "discharge_kw": discharge,
    }


def _supervisory_transition(
    protection_active: bool,
    safe_exit_counter: int,
    risk_level: int,
    soc: float,
    config: dict,
) -> tuple[bool, int]:
    """Causal hysteresis for the risk/SOC supervisory controller."""

    enter_level = int(config["risk_enter_level"])
    exit_level = int(config["risk_exit_level"])
    soc_enter = float(config["soc_enter"])
    soc_exit = float(config["soc_exit"])
    dwell = int(config["exit_dwell_steps"])
    if not 0 <= exit_level < enter_level <= 3:
        raise ValueError("监督控制风险退出等级必须低于进入等级")
    if not 0.0 <= soc_enter < soc_exit <= 1.0:
        raise ValueError("监督控制SOC退出阈值必须高于进入阈值")
    if dwell < 1:
        raise ValueError("监督控制退出驻留步数必须至少为1")

    enter = int(risk_level) >= enter_level or float(soc) <= soc_enter
    if not protection_active:
        return (True, 0) if enter else (False, 0)
    if enter:
        return True, 0
    safe_to_exit = int(risk_level) <= exit_level and float(soc) >= soc_exit
    counter = int(safe_exit_counter) + 1 if safe_to_exit else 0
    if counter >= dwell:
        return False, 0
    return True, counter


def _terminal_soc_drop_for_comparison(
    robust_cfg: dict,
    risk_level: int,
    method_specific_drop: float,
) -> float:
    """Apply V22's common terminal-SOC policy to every MILP comparator."""

    matched = robust_cfg.get("matched_terminal_soc", {})
    fallback = float(method_specific_drop)
    if not bool(matched.get("enabled", False)):
        return fallback
    by_level = matched.get("terminal_soc_drop_by_level", {})
    level = str(int(risk_level))
    if level not in by_level:
        raise ValueError(f"共同终端SOC策略缺少风险等级 {level}")
    selected = float(by_level[level])
    if not 0.0 <= selected <= 1.0:
        raise ValueError("共同终端SOC下降量必须在[0, 1]内")
    return selected


def _simulate_method(
    method: str,
    actual_kw: np.ndarray,
    forecast_kw: np.ndarray,
    persistence_kw: np.ndarray,
    grid_capacity_kw: float | np.ndarray,
    plant: Plant,
    robust_cfg: dict,
    risk_levels: np.ndarray | None = None,
    risk_reserve_kw: np.ndarray | None = None,
    generator_available_kw: np.ndarray | None = None,
    storage_available_kw: np.ndarray | None = None,
    initial_soc: float | None = None,
    initial_generator_kw: float = 0.0,
    initial_generator_units: int | None = None,
    initial_startup_history: tuple[int, ...] | list[int] = (),
    initial_shutdown_history: tuple[int, ...] | list[int] = (),
    initial_start_command_history: tuple[int, ...] | list[int] = (),
    scenario_forecasts_kw: np.ndarray | None = None,
    scenario_cvar_cfg: dict | None = None,
    supervisory_cfg: dict | None = None,
) -> tuple[pd.DataFrame, float]:
    point_scenario_ablation = method == "Full-Point-Scenario-Ablation-MILP"
    fixed_cvar_supervisor = method == "Full-Fixed-CVaR-Ablation-MILP"
    no_generator_first_ablation = (
        method == "Full-No-Generator-First-Ablation-MILP"
    )
    method = {
        "Persistence-MILP": "Persistence-MPC",
        "Point-Forecast-MILP": "Point-Forecast-MPC",
        "Fixed-Reserve-MILP": "Fixed-Reserve-MPC",
        "Residual-CVaR-MILP": "Scenario-CVaR-MPC",
        "Risk-Adaptive-Residual-CVaR-MILP": "Risk-Adaptive-CVaR-MPC",
        "Full-Risk-SOC-Supervisory-MILP": "Risk-SOC-Supervisory-MPC",
        "Full-Point-Scenario-Ablation-MILP": "Risk-SOC-Supervisory-MPC",
        "Full-Fixed-CVaR-Ablation-MILP": "Risk-SOC-Supervisory-MPC",
        "Full-No-Generator-First-Ablation-MILP": "Risk-SOC-Supervisory-MPC",
    }.get(method, method)
    records = []
    soc = plant.soc_initial if initial_soc is None else float(np.clip(initial_soc, plant.soc_min, plant.soc_max))
    previous_generator = float(
        np.clip(initial_generator_kw, 0.0, plant.generator_max_kw)
    )
    inferred_initial_units = int(
        np.ceil(max(previous_generator, 0.0) / plant.generator_unit_rated_kw)
    )
    previous_generator_units = int(
        np.clip(
            inferred_initial_units
            if initial_generator_units is None
            else initial_generator_units,
            0,
            plant.generator_unit_count,
        )
    )
    startup_keep = max(0, plant.generator_min_up_steps - 1)
    shutdown_keep = max(0, plant.generator_min_down_steps - 1)
    startup_history = list(initial_startup_history)[-startup_keep:] if startup_keep else []
    shutdown_history = (
        list(initial_shutdown_history)[-shutdown_keep:] if shutdown_keep else []
    )
    startup_delay_steps = int(robust_cfg.get("generator_startup_delay_steps", 0))
    if startup_delay_steps < 0:
        raise ValueError("generator_startup_delay_steps cannot be negative")
    pending_start_commands = (
        list(initial_start_command_history)[-startup_delay_steps:]
        if startup_delay_steps
        else []
    )
    if startup_delay_steps and len(pending_start_commands) < startup_delay_steps:
        pending_start_commands = [0] * (
            startup_delay_steps - len(pending_start_commands)
        ) + pending_start_commands
    _cancel_recent_transition_locks(
        startup_history,
        max(0, sum(startup_history) - previous_generator_units),
    )
    _cancel_recent_transition_locks(
        shutdown_history,
        max(
            0,
            sum(shutdown_history)
            - (plant.generator_unit_count - previous_generator_units),
        ),
    )
    grid_capacity_array = (
        np.full(len(actual_kw), float(grid_capacity_kw))
        if np.ndim(grid_capacity_kw) == 0
        else np.asarray(grid_capacity_kw, dtype=float)
    )
    if len(grid_capacity_array) != len(actual_kw):
        raise ValueError("动态网电容量与负荷长度不一致")
    supervisory_method = method == "Risk-SOC-Supervisory-MPC"
    if supervisory_method and (risk_levels is None or supervisory_cfg is None):
        raise ValueError("风险-SOC监督调度缺少风险信号或监督配置")
    if supervisory_method and (
        supervisory_cfg.get("economic_controller")
        != "Risk-Adaptive-CVaR-MPC"
        or supervisory_cfg.get("protection_controller") != "ML-Robust-MPC"
    ):
        raise ValueError("V9监督调度只允许预声明的经济/保护控制器组合")
    protection_active = False
    safe_exit_counter = 0
    started = time.perf_counter()
    for index, actual in enumerate(actual_kw):
        decision_started = time.perf_counter()
        newly_synchronized_units = (
            int(pending_start_commands.pop(0)) if startup_delay_steps else 0
        )
        grid_limit = float(grid_capacity_array[index])
        generator_limit = (
            None if generator_available_kw is None else float(generator_available_kw[index])
        )
        storage_limit = (
            None if storage_available_kw is None else float(storage_available_kw[index])
        )
        if plant.generator_unit_count > 1 and generator_limit is not None:
            available_units = min(
                plant.generator_unit_count,
                int(
                    np.floor(
                        (max(generator_limit, 0.0) + 1e-9)
                        / plant.generator_unit_min_kw
                    )
                ),
            )
            if previous_generator_units > available_units:
                previous_generator_units = available_units
                previous_generator = min(previous_generator, max(generator_limit, 0.0))
                _cancel_recent_transition_locks(
                    startup_history,
                    max(0, sum(startup_history) - previous_generator_units),
                )
        applied_risk_reserve = 0.0
        applied_cvar_alpha = np.nan
        applied_cvar_weight = np.nan
        applied_terminal_soc_drop = np.nan
        applied_minimum_generator_units = 0
        low_soc_commitment_guard_evaluated = False
        low_soc_commitment_guard_triggered = False
        low_soc_commitment_reachability_truncated = False
        low_soc_commitment_forecast_peak_kw = np.nan
        low_soc_single_unit_firm_supply_kw = np.nan
        low_soc_commitment_deficit_kw = np.nan
        current_risk_level = (
            int(risk_levels[index]) if risk_levels is not None else 0
        )
        if supervisory_method:
            protection_active, safe_exit_counter = _supervisory_transition(
                protection_active,
                safe_exit_counter,
                current_risk_level,
                soc,
                supervisory_cfg,
            )
        supervisory_protected = supervisory_method and protection_active
        supervisory_economic = supervisory_method and not protection_active
        selected_controller = (
            "ML-Robust-MPC"
            if supervisory_protected
            else (
                (
                    "Scenario-CVaR-MPC"
                    if fixed_cvar_supervisor
                    else "Risk-Adaptive-CVaR-MPC"
                )
                if supervisory_economic
                else method
            )
        )
        if (
            supervisory_protected
            and soc <= float(supervisory_cfg["soc_enter"]) + 1e-12
        ):
            applied_minimum_generator_units = int(
                supervisory_cfg.get("low_soc_minimum_committed_units", 0)
            )
            if not 0 <= applied_minimum_generator_units <= plant.generator_unit_count:
                raise ValueError("低SOC保护的最小开机台数超出物理范围")
            if bool(
                supervisory_cfg.get(
                    "low_soc_forecast_deficit_guard_enabled", False
                )
            ):
                if scenario_forecasts_kw is None:
                    raise ValueError("低SOC预测缺口门缺少因果预测场景")
                low_soc_commitment_guard_evaluated = True
                low_soc_commitment_forecast_peak_kw = float(
                    np.max(scenario_forecasts_kw[index])
                )
                single_unit_firm_power = min(
                    plant.generator_unit_rated_kw,
                    plant.generator_max_kw
                    if generator_limit is None
                    else max(generator_limit, 0.0),
                )
                low_soc_single_unit_firm_supply_kw = (
                    grid_limit + single_unit_firm_power
                )
                low_soc_commitment_deficit_kw = (
                    low_soc_commitment_forecast_peak_kw
                    - low_soc_single_unit_firm_supply_kw
                )
                guard_margin_kw = float(
                    supervisory_cfg.get(
                        "low_soc_forecast_deficit_margin_kw", 0.0
                    )
                )
                if guard_margin_kw < 0.0:
                    raise ValueError("低SOC预测缺口门裕量不得为负")
                low_soc_commitment_guard_triggered = bool(
                    low_soc_commitment_deficit_kw > guard_margin_kw
                )
                if not low_soc_commitment_guard_triggered:
                    applied_minimum_generator_units = 0
            if plant.generator_unit_count > 1 and applied_minimum_generator_units > 0:
                generator_capacity_for_commitment = min(
                    plant.generator_max_kw,
                    plant.generator_max_kw
                    if generator_limit is None
                    else max(generator_limit, 0.0),
                )
                capacity_units = min(
                    plant.generator_unit_count,
                    int(
                        np.floor(
                            (generator_capacity_for_commitment + 1e-9)
                            / plant.generator_unit_min_kw
                        )
                    ),
                )
                locked_off_units = _history_still_locked(
                    shutdown_history, plant.generator_min_down_steps, 0
                )
                reachable_units = min(
                    capacity_units,
                    max(0, plant.generator_unit_count - locked_off_units),
                )
                requested_minimum_units = applied_minimum_generator_units
                applied_minimum_generator_units = min(
                    applied_minimum_generator_units, reachable_units
                )
                low_soc_commitment_reachability_truncated = bool(
                    applied_minimum_generator_units < requested_minimum_units
                )
        risk_gate_level = int(
            (scenario_cvar_cfg or {}).get("risk_adaptive", {}).get(
                "activation_level", 2
            )
        )
        gated_cvar_active = (
            method == "Risk-Gated-CVaR-MPC"
            and risk_levels is not None
            and int(risk_levels[index]) >= risk_gate_level
        )
        if method == "Rule-Based":
            command = _rule_command(
                float(actual), grid_limit, soc, plant, generator_limit, storage_limit
            )
        elif (
            method in {"Scenario-CVaR-MPC", "Risk-Adaptive-CVaR-MPC"}
            or gated_cvar_active
            or supervisory_economic
        ):
            if scenario_forecasts_kw is None or scenario_cvar_cfg is None:
                raise ValueError("场景-CVaR调度缺少预测场景或配置")
            if method in {
                "Risk-Adaptive-CVaR-MPC",
                "Risk-Gated-CVaR-MPC",
                "Risk-SOC-Supervisory-MPC",
            } and not fixed_cvar_supervisor:
                if risk_levels is None:
                    raise ValueError("风险自适应CVaR调度缺少四级风险信号")
                level = str(int(risk_levels[index]))
                adaptive = scenario_cvar_cfg["risk_adaptive"]
                applied_cvar_alpha = float(adaptive["alpha_by_level"][level])
                applied_cvar_weight = float(adaptive["weight_by_level"][level])
                terminal_drop = float(adaptive["terminal_soc_drop_by_level"][level])
                applied_minimum_generator_units = int(
                    adaptive.get("minimum_committed_units_by_level", {}).get(
                        level, 0
                    )
                )
            else:
                applied_cvar_alpha = float(scenario_cvar_cfg["alpha"])
                applied_cvar_weight = float(scenario_cvar_cfg["weight"])
                terminal_drop = float(scenario_cvar_cfg["terminal_soc_drop"])
            terminal_drop = _terminal_soc_drop_for_comparison(
                robust_cfg, current_risk_level, terminal_drop
            )
            applied_terminal_soc_drop = terminal_drop
            selected_scenarios = scenario_forecasts_kw[index]
            if point_scenario_ablation:
                selected_scenarios = np.repeat(
                    forecast_kw[index][None, :],
                    scenario_forecasts_kw.shape[1],
                    axis=0,
                )
            command = _solve_scenario_cvar_mpc(
                selected_scenarios,
                grid_limit,
                soc,
                min(
                    previous_generator,
                    plant.generator_max_kw
                    if generator_limit is None
                    else generator_limit,
                ),
                plant,
                terminal_soc_drop=terminal_drop,
                cvar_alpha=applied_cvar_alpha,
                cvar_weight=applied_cvar_weight,
                generator_available_kw=generator_limit,
                storage_available_kw=storage_limit,
                previous_generator_units=previous_generator_units,
                startup_history=startup_history,
                shutdown_history=shutdown_history,
                minimum_generator_units=applied_minimum_generator_units,
            )
        else:
            if method == "Persistence-MPC":
                demand = persistence_kw[index]
                terminal_drop = float(robust_cfg["deterministic_terminal_soc_drop"])
            elif method == "Point-Forecast-MPC":
                demand = forecast_kw[index]
                terminal_drop = float(
                    robust_cfg["deterministic_terminal_soc_drop"]
                )
            elif method == "Fixed-Reserve-MPC":
                demand = forecast_kw[index] + float(
                    robust_cfg["fixed_reserve_kw"]
                )
                terminal_drop = float(robust_cfg["robust_terminal_soc_drop"])
            elif method == "Risk-Gated-CVaR-MPC":
                if risk_levels is None:
                    raise ValueError("风险门控CVaR调度缺少四级风险信号")
                # 正常/关注级使用项目点预测的低开销MPC；只在
                # 预警/严重级启用多场景CVaR，避免全时域过度保守。
                demand = forecast_kw[index]
                terminal_drop = float(
                    robust_cfg["deterministic_terminal_soc_drop"]
                )
            elif method == "ML-Robust-MPC" or supervisory_protected:
                disagreement = np.abs(forecast_kw[index] - persistence_kw[index])
                demand = (
                    forecast_kw[index]
                    + float(robust_cfg["fixed_reserve_kw"])
                    + float(robust_cfg["forecast_disagreement_weight"]) * disagreement
                )
                terminal_drop = float(robust_cfg["robust_terminal_soc_drop"])
            elif method == "Risk-Aware-Ensemble-MPC":
                if risk_levels is None or risk_reserve_kw is None:
                    raise ValueError("风险感知调度缺少 risk_levels 或 risk_reserve_kw")
                disagreement = np.abs(forecast_kw[index] - persistence_kw[index])
                base_demand = (
                    forecast_kw[index]
                    + float(robust_cfg["fixed_reserve_kw"])
                    + float(robust_cfg["forecast_disagreement_weight"]) * disagreement
                )
                # A virtual reserve must not exceed physical headroom.  Under
                # severe scarcity, inflating an already infeasible demand only
                # suppresses useful storage cycling; the controller therefore
                # switches from reserve procurement to emergency energy release.
                firm_power_limit = (
                    grid_limit
                    + (
                        plant.generator_max_kw
                        if generator_limit is None
                        else generator_limit
                    )
                    + (
                        plant.storage_power_kw
                        if storage_limit is None
                        else storage_limit
                    )
                )
                feasible_headroom = max(0.0, firm_power_limit - float(base_demand.max()))
                applied_risk_reserve = min(
                    float(risk_reserve_kw[index]), feasible_headroom
                )
                demand = base_demand + applied_risk_reserve
                terminal_map = robust_cfg["risk_terminal_soc_drop"]
                terminal_drop = float(terminal_map[str(int(risk_levels[index]))])
            else:
                raise ValueError(f"未知调度方法: {method}")
            terminal_drop = _terminal_soc_drop_for_comparison(
                robust_cfg, current_risk_level, terminal_drop
            )
            applied_terminal_soc_drop = terminal_drop
            command = _solve_linear_mpc(
                np.maximum(demand, 0.0),
                grid_limit,
                soc,
                min(
                    previous_generator,
                    plant.generator_max_kw
                    if generator_limit is None
                    else generator_limit,
                ),
                plant,
                terminal_soc_drop=terminal_drop,
                generator_available_kw=generator_limit,
                storage_available_kw=storage_limit,
                previous_generator_units=previous_generator_units,
                startup_history=startup_history,
                shutdown_history=shutdown_history,
                minimum_generator_units=applied_minimum_generator_units,
            )
        supervisory_generator_first = bool(
            supervisory_method
            and supervisory_cfg is not None
            and supervisory_cfg.get(
                "generator_first_storage_reserve_enabled",
                supervisory_protected,
            )
            and not no_generator_first_ablation
        )
        generator_before_storage = bool(
            supervisory_generator_first
            or (
                method in {"Risk-Adaptive-CVaR-MPC", "Risk-Gated-CVaR-MPC"}
                and risk_levels is not None
                and int(risk_levels[index]) >= 2
            )
        )
        requested_generator_units = int(
            command.get(
                "generator_units_on",
                np.ceil(
                    max(float(command["generator_kw"]), 0.0)
                    / plant.generator_unit_rated_kw
                ),
            )
        )
        available_discharge_kw, _ = _available_storage_power(soc, plant)
        if storage_limit is not None:
            available_discharge_kw = min(
                available_discharge_kw, max(0.0, storage_limit)
            )
        emergency_deficit_kw = max(
            0.0, float(actual) - grid_limit - available_discharge_kw
        )
        if emergency_deficit_kw > 1e-9:
            requested_generator_units = max(
                requested_generator_units,
                int(
                    np.ceil(
                        emergency_deficit_kw / plant.generator_unit_min_kw
                    )
                ),
            )
        synchronized_capacity_units: int | None = None
        startup_command_units = 0
        if startup_delay_steps:
            synchronized_capacity_units = min(
                plant.generator_unit_count,
                previous_generator_units + newly_synchronized_units,
            )
            already_online_or_pending = (
                synchronized_capacity_units + sum(pending_start_commands)
            )
            startup_command_units = int(
                np.clip(
                    requested_generator_units - already_online_or_pending,
                    0,
                    plant.generator_unit_count - already_online_or_pending,
                )
            )
            pending_start_commands.append(startup_command_units)
        executed, soc = _execute_with_safety_layer(
            float(actual),
            command,
            grid_limit,
            soc,
            previous_generator,
            plant,
            generator_limit,
            storage_limit,
            previous_generator_units,
            startup_history,
            shutdown_history,
            generator_before_storage,
            applied_minimum_generator_units,
            (
                float(supervisory_cfg["grid_headroom_reserve_target_soc"])
                if supervisory_method
                and bool(
                    supervisory_cfg.get(
                        "grid_headroom_reserve_enabled", False
                    )
                )
                else None
            ),
            (
                float(
                    supervisory_cfg.get(
                        "grid_headroom_reserve_max_storage_power_fraction", 0.0
                    )
                )
                * plant.storage_power_kw
                if supervisory_method
                and bool(
                    supervisory_cfg.get(
                        "grid_headroom_reserve_enabled", False
                    )
                )
                else 0.0
            ),
            synchronized_capacity_units,
            newly_synchronized_units,
        )
        executed["decision_seconds"] = time.perf_counter() - decision_started
        current_generator_units = int(executed["generator_units_on"])
        _cancel_recent_transition_locks(
            startup_history,
            max(0, sum(startup_history) - current_generator_units),
        )
        _cancel_recent_transition_locks(
            shutdown_history,
            max(
                0,
                sum(shutdown_history)
                - (plant.generator_unit_count - current_generator_units),
            ),
        )
        startup_units = max(0, current_generator_units - previous_generator_units)
        shutdown_units = max(0, previous_generator_units - current_generator_units)
        executed["generator_startup_units"] = startup_units
        executed["generator_start_command_units"] = (
            startup_command_units if startup_delay_steps else startup_units
        )
        executed["generator_synchronized_units"] = startup_units
        executed["generator_startup_delay_steps"] = startup_delay_steps
        executed["generator_pending_start_units"] = int(
            sum(pending_start_commands)
        )
        executed["generator_shutdown_units"] = shutdown_units
        executed["step"] = index
        executed["risk_level"] = (
            int(risk_levels[index]) if risk_levels is not None else 0
        )
        executed["risk_reserve_kw"] = (
            float(risk_reserve_kw[index]) if risk_reserve_kw is not None else 0.0
        )
        executed["applied_risk_reserve_kw"] = applied_risk_reserve
        executed["forecast_scenario_count"] = (
            int(scenario_forecasts_kw.shape[1])
            if (
                method in {"Scenario-CVaR-MPC", "Risk-Adaptive-CVaR-MPC"}
                or gated_cvar_active
                or supervisory_economic
            )
            and scenario_forecasts_kw is not None
            else 1
        )
        executed["cvar_alpha"] = applied_cvar_alpha
        executed["cvar_weight"] = applied_cvar_weight
        executed["predicted_expected_shortage_cost_yuan"] = float(
            command.get("predicted_expected_shortage_cost_yuan", np.nan)
        )
        executed["predicted_cvar_shortage_cost_yuan"] = float(
            command.get("predicted_cvar_shortage_cost_yuan", np.nan)
        )
        executed["predicted_weighted_cvar_term_yuan"] = float(
            command.get("predicted_weighted_cvar_term_yuan", np.nan)
        )
        executed["terminal_soc_drop"] = applied_terminal_soc_drop
        executed["minimum_committed_generator_units"] = (
            applied_minimum_generator_units
        )
        executed["low_soc_commitment_guard_evaluated"] = bool(
            low_soc_commitment_guard_evaluated
        )
        executed["low_soc_commitment_guard_triggered"] = bool(
            low_soc_commitment_guard_triggered
        )
        executed["low_soc_commitment_reachability_truncated"] = bool(
            low_soc_commitment_reachability_truncated
        )
        executed["low_soc_commitment_forecast_peak_kw"] = (
            low_soc_commitment_forecast_peak_kw
        )
        executed["low_soc_single_unit_firm_supply_kw"] = (
            low_soc_single_unit_firm_supply_kw
        )
        executed["low_soc_commitment_deficit_kw"] = (
            low_soc_commitment_deficit_kw
        )
        executed["generator_before_storage"] = generator_before_storage
        executed["supervisory_protection_active"] = bool(protection_active)
        executed["supervisory_safe_exit_counter"] = int(safe_exit_counter)
        executed["selected_controller"] = selected_controller
        executed["grid_available_capacity_kw"] = grid_limit
        executed["generator_available_capacity_kw"] = (
            plant.generator_max_kw if generator_limit is None else generator_limit
        )
        executed["storage_available_power_kw"] = (
            plant.storage_power_kw if storage_limit is None else storage_limit
        )
        records.append(executed)
        previous_generator = executed["generator_kw"]
        previous_generator_units = current_generator_units
        startup_history.append(startup_units)
        shutdown_history.append(shutdown_units)
        startup_history = startup_history[-startup_keep:] if startup_keep else []
        shutdown_history = (
            shutdown_history[-shutdown_keep:] if shutdown_keep else []
        )
    return pd.DataFrame(records), time.perf_counter() - started


def _metrics(
    frame: pd.DataFrame,
    plant: Plant,
    solve_seconds: float,
    reference_soc: float | None = None,
) -> dict[str, float]:
    dt = plant.step_hours
    fuel_l = float((_generator_fuel_lph(frame, plant) * dt).sum())
    grid_energy = float(frame["grid_kw"].sum() * dt)
    generator_energy = float(frame["generator_kw"].sum() * dt)
    storage_throughput = float((frame["charge_kw"] + frame["discharge_kw"]).sum() * dt)
    unserved_energy = float(frame["unserved_kw"].sum() * dt)
    on = frame["generator_kw"].to_numpy() > 1.0
    if "generator_synchronized_units" in frame:
        startups = int(frame["generator_synchronized_units"].sum())
    elif "generator_startup_units" in frame:
        startups = int(frame["generator_startup_units"].sum())
    else:
        startups = int(np.sum(on & ~np.r_[False, on[:-1]]))
    start_commands = (
        int(frame["generator_start_command_units"].sum())
        if "generator_start_command_units" in frame
        else startups
    )
    startup_cost = start_commands * plant.generator_startup_cost
    energy_cost = (
        grid_energy * plant.grid_price
        + fuel_l * plant.diesel_price
        + storage_throughput * plant.storage_degradation
        + startup_cost
    )
    final_soc = float(frame["soc"].iloc[-1])
    restoration_reference = (
        plant.soc_initial if reference_soc is None else float(reference_soc)
    )
    restoration_energy = (
        max(0.0, restoration_reference - final_soc)
        * plant.storage_energy_kwh
        / plant.charge_efficiency
    )
    restoration_cost = restoration_energy * plant.grid_price
    reliability_penalty = unserved_energy * plant.shortage_penalty
    total_social_cost = energy_cost + restoration_cost + reliability_penalty
    loss_of_load = frame["unserved_kw"].to_numpy(dtype=float) > 1e-9
    mean_load_pct = (
        float(frame.loc[on, "generator_kw"].mean() / plant.generator_max_kw * 100.0)
        if on.any()
        else 0.0
    )
    decision_seconds = (
        frame["decision_seconds"].to_numpy(dtype=float)
        if "decision_seconds" in frame
        else np.zeros(len(frame), dtype=float)
    )
    protection = (
        frame["supervisory_protection_active"].to_numpy(dtype=bool)
        if "supervisory_protection_active" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    supervisory_switches = int(
        np.sum(protection != np.r_[False, protection[:-1]])
    )
    predicted_cvar = (
        frame["predicted_cvar_shortage_cost_yuan"].to_numpy(dtype=float)
        if "predicted_cvar_shortage_cost_yuan" in frame
        else np.asarray([], dtype=float)
    )
    predicted_cvar = predicted_cvar[np.isfinite(predicted_cvar)]
    predicted_weighted_cvar = (
        frame["predicted_weighted_cvar_term_yuan"].to_numpy(dtype=float)
        if "predicted_weighted_cvar_term_yuan" in frame
        else np.asarray([], dtype=float)
    )
    predicted_weighted_cvar = predicted_weighted_cvar[
        np.isfinite(predicted_weighted_cvar)
    ]
    return {
        "realized_operating_cost_yuan": energy_cost,
        "terminal_energy_adjustment_yuan": restoration_cost,
        "reliability_penalty_yuan": reliability_penalty,
        "total_social_cost_yuan": total_social_cost,
        "energy_cost_yuan": energy_cost,
        "soc_restoration_cost_yuan": restoration_cost,
        "equivalent_cost_yuan": energy_cost + restoration_cost,
        # Historical alias retained for frozen V15/V19/V21 readers. New paper
        # tables must use total_social_cost_yuan and show its three components.
        "risk_adjusted_cost_yuan": total_social_cost,
        "grid_energy_kwh": grid_energy,
        "generator_energy_kwh": generator_energy,
        "diesel_fuel_l": fuel_l,
        "generator_startup_cost_yuan": startup_cost,
        "storage_throughput_kwh": storage_throughput,
        "unserved_energy_kwh": unserved_energy,
        "loss_of_load_duration_hours": float(loss_of_load.sum() * dt),
        "loss_of_load_step_fraction": float(loss_of_load.mean()),
        "VOLL_yuan_per_kwh": plant.shortage_penalty,
        "mean_predicted_cvar_shortage_cost_yuan": (
            float(predicted_cvar.mean()) if len(predicted_cvar) else float("nan")
        ),
        "maximum_predicted_cvar_shortage_cost_yuan": (
            float(predicted_cvar.max()) if len(predicted_cvar) else float("nan")
        ),
        "mean_predicted_weighted_cvar_term_yuan": (
            float(predicted_weighted_cvar.mean())
            if len(predicted_weighted_cvar)
            else float("nan")
        ),
        "max_unserved_power_kw": float(frame["unserved_kw"].max()),
        "spill_energy_kwh": float(frame["spill_kw"].sum() * dt),
        "minimum_soc_pct": float(frame["soc"].min() * 100.0),
        "final_soc_pct": final_soc * 100.0,
        "restoration_reference_soc_pct": restoration_reference * 100.0,
        "generator_startups": startups,
        "generator_start_commands": start_commands,
        "mean_generator_load_pct_when_on": mean_load_pct,
        "mean_decision_seconds": float(decision_seconds.mean()),
        "p95_decision_seconds": float(np.quantile(decision_seconds, 0.95)),
        "max_decision_seconds": float(decision_seconds.max()),
        "supervisory_protection_fraction": float(protection.mean()),
        "supervisory_mode_switches": supervisory_switches,
        "solve_seconds": solve_seconds,
    }


def _select_operation_aligned_windows(
    y_true: np.ndarray,
    transition: np.ndarray,
    operation_state_codes: np.ndarray,
    length: int,
    specs: dict,
    supply_regimes: np.ndarray | None = None,
) -> tuple[dict[str, slice], dict[str, dict]]:
    """Select drilling-operation scenarios under state and optional supply gates."""

    series = np.asarray(y_true, dtype=float)[:, 0]
    flags = np.asarray(transition, dtype=bool)
    state_codes = np.asarray(operation_state_codes, dtype=np.int16)
    if not (len(series) == len(flags) == len(state_codes)):
        raise ValueError("负荷、切换标志与作业工况序列长度不一致")
    if length <= 0 or len(series) < length:
        raise ValueError("作业场景窗口长度不合理")
    supply = (
        None
        if supply_regimes is None
        else np.asarray(supply_regimes, dtype=object)
    )
    if supply is not None and len(supply) != len(series):
        raise ValueError("供给工况与作业工况序列长度不一致")

    stride = max(1, length // 20)
    windows: dict[str, slice] = {}
    evidence: dict[str, dict] = {}
    for scenario, spec in specs.items():
        primary_codes = {
            STATE_TO_CODE[str(name)] for name in spec["primary_states"]
        }
        anchor_codes = {
            STATE_TO_CODE[str(name)] for name in spec.get("anchor_states", [])
        }
        minimum_primary = float(spec.get("minimum_primary_fraction", 0.0))
        minimum_anchor = float(spec.get("minimum_anchor_fraction", 0.0))
        acceptable_supply = {
            str(name) for name in spec.get("acceptable_supply_regimes", [])
        }
        minimum_supply = float(spec.get("minimum_supply_fraction", 0.0))
        maximum_emergency = float(spec.get("maximum_emergency_fraction", 1.0))
        if acceptable_supply and supply is None:
            raise ValueError(f"作业场景 {scenario} 声明了供给门槛但未提供供给序列")
        candidates = []
        for start in range(0, len(series) - length + 1, stride):
            stop = start + length
            states = state_codes[start:stop]
            values = series[start:stop]
            primary_fraction = float(np.isin(states, list(primary_codes)).mean())
            anchor_fraction = (
                float(np.isin(states, list(anchor_codes)).mean())
                if anchor_codes
                else 0.0
            )
            if primary_fraction + 1e-12 < minimum_primary:
                continue
            if anchor_codes and anchor_fraction + 1e-12 < minimum_anchor:
                continue
            window_supply = supply[start:stop] if supply is not None else None
            acceptable_supply_fraction = (
                float(np.isin(window_supply, list(acceptable_supply)).mean())
                if acceptable_supply
                else 0.0
            )
            emergency_supply_fraction = (
                float(np.mean(window_supply == "emergency"))
                if window_supply is not None
                else 0.0
            )
            if acceptable_supply and (
                acceptable_supply_fraction + 1e-12 < minimum_supply
                or emergency_supply_fraction > maximum_emergency + 1e-12
            ):
                continue
            transition_count = int(flags[start:stop].sum())
            load_range = float(values.max() - values.min())
            load_std = float(values.std())
            score_name = str(spec.get("score", "stable"))
            if score_name == "stable":
                score = 1000.0 * primary_fraction - load_std
            elif score_name == "transition":
                score = (
                    1000.0 * anchor_fraction
                    + 40.0 * transition_count
                    + load_range
                )
            elif score_name == "impact":
                score = 1000.0 * primary_fraction + load_range + 30.0 * transition_count
            else:
                raise ValueError(f"未知作业场景评分方式: {score_name}")
            candidates.append(
                {
                    "start": start,
                    "stop": stop,
                    "score": score,
                    "primary_fraction": primary_fraction,
                    "anchor_fraction": anchor_fraction,
                    "acceptable_supply_fraction": acceptable_supply_fraction,
                    "emergency_supply_fraction": emergency_supply_fraction,
                    "transition_count": transition_count,
                    "load_mean_kw": float(values.mean()),
                    "load_std_kw": load_std,
                    "load_range_kw": load_range,
                }
            )
        if not candidates:
            raise ValueError(
                f"作业场景 {scenario} 找不到满足工况覆盖门槛的连续窗口"
            )
        selected = max(candidates, key=lambda item: (item["score"], -item["start"]))
        start = int(selected["start"])
        stop = int(selected["stop"])
        selected_states = state_codes[start:stop]
        counts = {
            CODE_TO_STATE.get(int(code), "unknown"): float(
                np.mean(selected_states == code)
            )
            for code in np.unique(selected_states)
        }
        windows[scenario] = slice(start, stop)
        evidence[scenario] = {
            **selected,
            "state_fractions": counts,
            "primary_states": [str(name) for name in spec["primary_states"]],
            "anchor_states": [str(name) for name in spec.get("anchor_states", [])],
            "acceptable_supply_regimes": sorted(acceptable_supply),
            "selection_rule": str(spec.get("score", "stable")),
        }
    return windows, evidence


def _select_scenario_windows(y_true: np.ndarray, transition: np.ndarray, length: int) -> dict[str, slice]:
    series = y_true[:, 0]
    flags = transition.astype(float)
    candidates = range(0, len(series) - length, max(12, length // 12))
    stats = []
    for start in candidates:
        values = series[start : start + length]
        stats.append(
            {
                "start": start,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "range": float(values.max() - values.min()),
                "transitions": float(flags[start : start + length].sum()),
            }
        )
    stable = min((s for s in stats if s["mean"] > 600.0), key=lambda s: s["std"])
    impact = max(stats, key=lambda s: s["range"] + 30.0 * s["transitions"])
    weak = max(stats, key=lambda s: s["mean"])
    return {
        "A_supply_adequate": slice(stable["start"], stable["start"] + length),
        "B_constrained_impact": slice(impact["start"], impact["start"] + length),
        "C_emergency_supply": slice(weak["start"], weak["start"] + length),
    }


def _select_supply_aligned_windows(
    y_true: np.ndarray,
    transition: np.ndarray,
    supply_frame: pd.DataFrame,
    length: int,
) -> dict[str, slice]:
    """Select scenarios using the same supply context that produced risk levels."""

    series = y_true[:, 0]
    flags = transition.astype(float)
    regimes = supply_frame["supply_regime"].astype(str).to_numpy()
    firm_supply = (
        supply_frame["grid_available_capacity_kw"].to_numpy(float)
        + supply_frame["generator_available_capacity_kw"].to_numpy(float)
        + supply_frame["storage_available_discharge_power_kw"].to_numpy(float)
    )
    candidates = range(0, len(series) - length, max(12, length // 18))
    stats = []
    for start in candidates:
        stop = start + length
        values = series[start:stop]
        window_regimes = regimes[start:stop]
        fractions = {
            name: float(np.mean(window_regimes == name))
            for name in ("normal", "constrained", "weak", "emergency")
        }
        stats.append(
            {
                "start": start,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "range": float(values.max() - values.min()),
                "transitions": float(flags[start:stop].sum()),
                "firm_mean": float(firm_supply[start:stop].mean()),
                **fractions,
            }
        )
    stable = max(stats, key=lambda item: 2500.0 * item["normal"] - item["std"])
    impact = max(
        stats,
        key=lambda item: item["range"]
        + 30.0 * item["transitions"]
        + 800.0 * (item["constrained"] + item["weak"])
        - 500.0 * item["emergency"],
    )
    weak = max(
        stats,
        key=lambda item: item["mean"]
        - item["firm_mean"]
        + 1800.0 * item["emergency"]
        + 700.0 * item["weak"],
    )
    return {
        "A_supply_adequate": slice(stable["start"], stable["start"] + length),
        "B_constrained_impact": slice(impact["start"], impact["start"] + length),
        "C_emergency_supply": slice(weak["start"], weak["start"] + length),
    }


def _plot_scenario(trajectories: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(
        len(trajectories), 2, figsize=(15, 3.1 * len(trajectories)), sharex=True
    )
    for row, (method, frame) in enumerate(trajectories.items()):
        x = frame["step"] * 5.0 / 60.0
        ax = axes[row, 0]
        ax.plot(x, frame["load_kw"], color="#111827", linewidth=1.3, label="负荷")
        ax.plot(x, frame["grid_kw"], color="#2563EB", linewidth=1.0, label="网电")
        ax.plot(x, frame["generator_kw"], color="#F59E0B", linewidth=1.0, label="柴油机")
        ax.plot(x, frame["storage_kw"], color="#10B981", linewidth=1.0, label="储能(+放电)")
        ax.set_ylabel(method + "\n功率/kW")
        ax.grid(alpha=0.18)
        if row == 0:
            ax.legend(ncol=4, frameon=False)
        soc_ax = axes[row, 1]
        soc_ax.plot(x, frame["soc"] * 100.0, color="#7C3AED", linewidth=1.2)
        soc_ax.fill_between(x, 15.0, 90.0, color="#7C3AED", alpha=0.06)
        soc_ax.set_ylabel("SOC/%")
        soc_ax.set_ylim(10, 95)
        soc_ax.grid(alpha=0.18)
    axes[-1, 0].set_xlabel("场景时间/分钟")
    axes[-1, 1].set_xlabel("场景时间/分钟")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary(metrics: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    columns = [
        ("equivalent_cost_yuan", "等SOC运行成本/元"),
        ("diesel_fuel_l", "柴油消耗/L"),
        ("unserved_energy_kwh", "未供电量/kWh"),
        ("minimum_soc_pct", "最低SOC/%"),
    ]
    labels = metrics["scenario"] + "\n" + metrics["method"]
    for ax, (column, title) in zip(axes.flat, columns):
        ax.barh(labels, metrics[column], color="#3B82F6", alpha=0.82)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_dispatch_benchmark(
    prediction_path: Path,
    prediction_key_path: Path,
    config_path: Path,
    artifact_dir: Path,
    risk_signal_path: Path | None = None,
    scenario_residual_path: Path | None = None,
    scenario_seed: int | None = None,
    comparison_methods: list[str] | None = None,
) -> pd.DataFrame:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 每次重建只保留本次声明的方法轨迹，防止已撤下的探索候选
    # 残留在验收目录中被误当成正式结果。
    for stale_trajectory in artifact_dir.glob("trajectory_*.csv"):
        stale_trajectory.unlink()
    config, plant = _load_config(config_path)
    arrays = np.load(prediction_path)
    key_map = json.loads(Path(prediction_key_path).read_text(encoding="utf-8"))
    name_to_key = {name: key for key, name in key_map.items()}
    y_true = arrays["y_true"]
    transition = arrays["transition_flags"]
    operation_state_codes = (
        arrays["operation_state_codes"]
        if "operation_state_codes" in arrays.files
        else None
    )
    robust_cfg = config["robust_mpc"]
    forecast_model = str(robust_cfg.get("forecast_model", "StateAware-TCN-Attention"))
    if forecast_model not in name_to_key:
        raise ValueError(
            f"预测轨迹中不存在 {forecast_model}，可用模型: {sorted(name_to_key)}"
        )
    proposed = arrays[name_to_key[forecast_model]]
    persistence = arrays[name_to_key["Persistence"]]
    scenario_cvar_cfg = config["scenario_cvar_mpc"]
    scenario_member_names: list[str] = []
    residual_library: np.ndarray | None = None
    scenario_count: int | None = None
    if scenario_residual_path is not None:
        residual_arrays = np.load(scenario_residual_path)
        if residual_arrays.files != ["validation_residuals_kw"]:
            raise ValueError(
                "V22残差产物只能包含validation_residuals_kw"
            )
        residual_library = np.asarray(
            residual_arrays["validation_residuals_kw"], dtype=float
        )
        if residual_library.ndim != 2 or residual_library.shape[1] != proposed.shape[1]:
            raise ValueError("V22验证残差库与调度预测时域不一致")
        scenario_count = int(scenario_cvar_cfg["residual_scenario_count"])
        probabilities = np.full(scenario_count, 1.0 / scenario_count)
        minimum_tail = float(
            scenario_cvar_cfg["minimum_effective_tail_scenarios"]
        )
        configured_alphas = {float(scenario_cvar_cfg["alpha"])}
        configured_alphas.update(
            float(value)
            for value in scenario_cvar_cfg.get("risk_adaptive", {})
            .get("alpha_by_level", {})
            .values()
        )
        for configured_alpha in configured_alphas:
            audit_cvar_discretization(
                probabilities,
                alpha=configured_alpha,
                minimum_effective_tail_scenarios=minimum_tail,
            )
        scenario_forecasts: np.ndarray | None = None
    else:
        scenario_member_names = [
            str(name) for name in scenario_cvar_cfg["forecast_members"]
        ]
        missing_scenario_members = [
            name for name in scenario_member_names if name not in name_to_key
        ]
        if missing_scenario_members:
            raise ValueError(f"场景-CVaR预测成员不存在: {missing_scenario_members}")
        if len(scenario_member_names) < 3:
            raise ValueError("场景-CVaR对比至少需要3个预测成员")
        scenario_forecasts = np.stack(
            [arrays[name_to_key[name]] for name in scenario_member_names], axis=1
        )
    risk_levels: np.ndarray | None = None
    risk_reserve: np.ndarray | None = None
    risk_frame: pd.DataFrame | None = None
    source_indices = np.arange(len(y_true))
    if risk_signal_path is not None:
        risk_frame = pd.read_csv(risk_signal_path)
        required_risk_columns = {
            "sample_index",
            "predicted_risk_level",
            "reserve_adder_kw",
            "supply_regime",
            "grid_available_capacity_kw",
            "generator_available_capacity_kw",
            "storage_soc_pct",
            "storage_available_discharge_power_kw",
        }
        missing = sorted(required_risk_columns - set(risk_frame.columns))
        if missing:
            raise ValueError(f"风险信号文件缺少字段: {missing}")
        source_indices = risk_frame["sample_index"].to_numpy(dtype=int)
        if len(source_indices) == 0 or source_indices.min() < 0 or source_indices.max() >= len(y_true):
            raise ValueError("风险信号 sample_index 超出预测数组范围")
        if not np.all(np.diff(source_indices) == 1):
            raise ValueError("风险信号 sample_index 必须连续递增，才能选择连续调度场景")
        y_true = y_true[source_indices]
        transition = transition[source_indices]
        if operation_state_codes is not None:
            operation_state_codes = operation_state_codes[source_indices]
        proposed = proposed[source_indices]
        persistence = persistence[source_indices]
        if scenario_forecasts is not None:
            scenario_forecasts = scenario_forecasts[source_indices]
        dispatch_risk_column = (
            "dispatch_risk_level"
            if "dispatch_risk_level" in risk_frame.columns
            else "predicted_risk_level"
        )
        risk_levels = risk_frame[dispatch_risk_column].to_numpy(dtype=int)
        risk_reserve = risk_frame["reserve_adder_kw"].to_numpy(dtype=float)

    length = int(config["dispatch"]["scenario_steps"])
    requested_warmup = int(config["dispatch"].get("warmup_steps", 0))
    scenario_selection_cfg = config.get("scenario_selection", {})
    selection_mode = str(scenario_selection_cfg.get("mode", "supply_aligned"))
    operation_evidence: dict[str, dict] = {}
    if selection_mode == "operation_aligned":
        if operation_state_codes is None:
            raise ValueError(
                "operation_aligned场景选取需要预测产物中的operation_state_codes"
            )
        windows, operation_evidence = _select_operation_aligned_windows(
            y_true,
            transition,
            operation_state_codes,
            length,
            scenario_selection_cfg["scenarios"],
            supply_regimes=(
                risk_frame["supply_regime"].astype(str).to_numpy()
                if risk_frame is not None
                else None
            ),
        )
    elif selection_mode == "supply_aligned":
        windows = (
            _select_supply_aligned_windows(y_true, transition, risk_frame, length)
            if risk_frame is not None
            else _select_scenario_windows(y_true, transition, length)
        )
    else:
        raise ValueError(f"未知场景选取模式: {selection_mode}")
    scenario_cfg = config["scenarios"]
    default_methods = [
        "Rule-Based",
        "Persistence-MPC",
        "ML-Robust-MPC",
        "Scenario-CVaR-MPC",
    ]
    if risk_signal_path is not None:
        default_methods.append("Risk-Aware-Ensemble-MPC")
        default_methods.append("Risk-Adaptive-CVaR-MPC")
        if bool(config.get("supervisory_mpc", {}).get("enabled", False)):
            default_methods.append("Risk-SOC-Supervisory-MPC")
    methods = [
        str(value)
        for value in (
            comparison_methods
            if comparison_methods is not None
            else config.get("comparison_methods", default_methods)
        )
    ]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("比较方法列表必须非空且不得重复")
    all_metrics = []
    selection = {}
    scenario_generation_audits: dict[str, dict] = {}
    for scenario_index, (scenario, selected) in enumerate(windows.items()):
        execution_start = max(0, selected.start - requested_warmup)
        warmup = selected.start - execution_start
        execution_slice = slice(execution_start, selected.stop)
        full_actual = y_true[execution_slice, 0]
        full_ml_forecast = proposed[execution_slice]
        full_persistence_forecast = persistence[execution_slice]
        full_scenario_risk_levels = (
            risk_levels[execution_slice] if risk_levels is not None else None
        )
        full_scenario_risk_reserve = (
            risk_reserve[execution_slice] if risk_reserve is not None else None
        )
        if risk_frame is not None:
            scenario_supply = risk_frame.iloc[execution_slice]
            grid_capacity: float | np.ndarray = scenario_supply[
                "grid_available_capacity_kw"
            ].to_numpy(float)
            generator_available = scenario_supply[
                "generator_available_capacity_kw"
            ].to_numpy(float)
            storage_available = scenario_supply[
                "storage_available_discharge_power_kw"
            ].to_numpy(float)
            initial_soc = float(scenario_supply["storage_soc_pct"].iloc[0]) / 100.0
            scored_supply = risk_frame.iloc[selected]
            grid_metadata: float | dict[str, float] = {
                "mean": float(scored_supply["grid_available_capacity_kw"].mean()),
                "minimum": float(scored_supply["grid_available_capacity_kw"].min()),
                "maximum": float(scored_supply["grid_available_capacity_kw"].max()),
            }
            regime_counts = scored_supply["supply_regime"].value_counts(normalize=True).to_dict()
        else:
            grid_capacity = float(scenario_cfg[scenario]["grid_capacity_kw"])
            generator_available = None
            storage_available = None
            initial_soc = None
            grid_metadata = float(grid_capacity)
            regime_counts = {}
        if residual_library is not None:
            assert scenario_count is not None
            full_scenario_forecasts, probabilities, sampled_indices = (
                build_residual_scenarios(
                    full_ml_forecast,
                    residual_library,
                    scenario_count=scenario_count,
                    seed=(
                        int(scenario_seed)
                        if scenario_seed is not None
                        else int(config.get("metadata", {}).get("seed", 0))
                    )
                    + scenario_index,
                )
            )
            scenario_generation_audits[scenario] = {
                "source": "validation_residual_library",
                "scenario_count": scenario_count,
                "probability_sum": float(probabilities.sum()),
                "unique_residual_rows": int(len(np.unique(sampled_indices))),
                "test_targets_used": False,
            }
        else:
            assert scenario_forecasts is not None
            full_scenario_forecasts = scenario_forecasts[execution_slice]
        # All compared controllers start scoring from one identical causal
        # warm-up state.  Method-specific warm-ups would otherwise confound the
        # comparison by giving each controller a different SOC and unit status.
        comparison_initial_soc = (
            plant.soc_initial if initial_soc is None else float(initial_soc)
        )
        comparison_initial_generator = 0.0
        comparison_initial_generator_units = 0
        comparison_initial_startup_history: list[int] = []
        comparison_initial_shutdown_history: list[int] = []
        comparison_initial_start_command_history: list[int] = []
        if warmup > 0:
            warmup_trajectory, _ = _simulate_method(
                "Rule-Based",
                full_actual[:warmup],
                full_ml_forecast[:warmup],
                full_persistence_forecast[:warmup],
                (
                    grid_capacity[:warmup]
                    if np.ndim(grid_capacity) > 0
                    else grid_capacity
                ),
                plant,
                robust_cfg,
                generator_available_kw=(
                    generator_available[:warmup]
                    if generator_available is not None
                    else None
                ),
                storage_available_kw=(
                    storage_available[:warmup]
                    if storage_available is not None
                    else None
                ),
                initial_soc=comparison_initial_soc,
            )
            comparison_initial_soc = float(warmup_trajectory["soc"].iloc[-1])
            comparison_initial_generator = float(
                warmup_trajectory["generator_kw"].iloc[-1]
            )
            comparison_initial_generator_units = int(
                warmup_trajectory["generator_units_on"].iloc[-1]
            )
            startup_keep = max(0, plant.generator_min_up_steps - 1)
            shutdown_keep = max(0, plant.generator_min_down_steps - 1)
            if startup_keep:
                comparison_initial_startup_history = (
                    warmup_trajectory["generator_startup_units"]
                    .tail(startup_keep)
                    .astype(int)
                    .tolist()
                )
            if shutdown_keep:
                comparison_initial_shutdown_history = (
                    warmup_trajectory["generator_shutdown_units"]
                    .tail(shutdown_keep)
                    .astype(int)
                    .tolist()
                )
            startup_delay_steps = int(
                robust_cfg.get("generator_startup_delay_steps", 0)
            )
            if startup_delay_steps:
                comparison_initial_start_command_history = (
                    warmup_trajectory["generator_start_command_units"]
                    .tail(startup_delay_steps)
                    .astype(int)
                    .tolist()
                )
        score_slice = slice(warmup, None)
        actual = full_actual[score_slice]
        ml_forecast = full_ml_forecast[score_slice]
        persistence_forecast = full_persistence_forecast[score_slice]
        scenario_risk_levels = (
            full_scenario_risk_levels[score_slice]
            if full_scenario_risk_levels is not None
            else None
        )
        scenario_risk_reserve = (
            full_scenario_risk_reserve[score_slice]
            if full_scenario_risk_reserve is not None
            else None
        )
        if np.ndim(grid_capacity) > 0:
            grid_capacity = grid_capacity[score_slice]
        if generator_available is not None:
            generator_available = generator_available[score_slice]
        if storage_available is not None:
            storage_available = storage_available[score_slice]
        scored_scenario_forecasts = full_scenario_forecasts[score_slice]
        trajectories = {}
        selection[scenario] = {
            **operation_evidence.get(scenario, {}),
            "start_index": int(source_indices[selected.start]),
            "stop_index": int(source_indices[selected.stop - 1] + 1),
            "warmup_steps": warmup,
            "warmup_policy": "shared_rule_based_state_initialization",
            "comparison_initial_soc_pct": comparison_initial_soc * 100.0,
            "comparison_initial_generator_kw": comparison_initial_generator,
            "comparison_initial_generator_units": comparison_initial_generator_units,
            "comparison_initial_start_command_history": (
                comparison_initial_start_command_history
            ),
            "grid_capacity_kw": grid_metadata,
            "supply_regime_fractions": regime_counts,
            "mean_load_kw": float(actual.mean()),
            "max_load_kw": float(actual.max()),
            "scenario_generation": scenario_generation_audits.get(
                scenario,
                {
                    "source": "forecast_model_members",
                    "test_targets_used_for_generation": False,
                },
            ),
        }
        for method in methods:
            trajectory, solve_seconds = _simulate_method(
                method,
                actual,
                ml_forecast,
                persistence_forecast,
                grid_capacity,
                plant,
                robust_cfg,
                risk_levels=scenario_risk_levels,
                risk_reserve_kw=scenario_risk_reserve,
                generator_available_kw=generator_available,
                storage_available_kw=storage_available,
                initial_soc=comparison_initial_soc,
                initial_generator_kw=comparison_initial_generator,
                initial_generator_units=comparison_initial_generator_units,
                initial_startup_history=comparison_initial_startup_history,
                initial_shutdown_history=comparison_initial_shutdown_history,
                initial_start_command_history=(
                    comparison_initial_start_command_history
                ),
                scenario_forecasts_kw=scored_scenario_forecasts,
                scenario_cvar_cfg=scenario_cvar_cfg,
                supervisory_cfg=config.get("supervisory_mpc"),
            )
            scored_trajectory = trajectory.copy().reset_index(drop=True)
            scored_trajectory["step"] = np.arange(len(scored_trajectory))
            scored_trajectory.to_csv(
                artifact_dir / f"trajectory_{scenario}_{method.lower().replace('-', '_')}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            row = {"scenario": scenario, "method": method}
            row.update(
                _metrics(
                    scored_trajectory,
                    plant,
                    solve_seconds,
                    reference_soc=comparison_initial_soc,
                )
            )
            all_metrics.append(row)
            trajectories[method] = scored_trajectory
        _plot_scenario(
            trajectories,
            artifact_dir / f"dispatch_{scenario}.png",
            scenario_cfg[scenario]["title"],
        )

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(artifact_dir / "dispatch_metrics.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "scenario_selection.json").write_text(
        json.dumps(
            {
                "forecast_model": forecast_model,
                "scenario_cvar": {
                    "forecast_members": scenario_member_names,
                    "scenario_source": (
                        "validation_residual_library"
                        if residual_library is not None
                        else "forecast_model_members"
                    ),
                    "residual_scenario_count": scenario_count,
                    "alpha": float(scenario_cvar_cfg["alpha"]),
                    "weight": float(scenario_cvar_cfg["weight"]),
                    "power_nonanticipativity": "first_control_action_only",
                    "commitment_nonanticipativity": "shared_generator_on_off_schedule",
                },
                "risk_signal_path": str(risk_signal_path) if risk_signal_path else None,
                "selection_mode": selection_mode,
                "scenarios": selection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_summary(metrics, artifact_dir / "dispatch_metric_comparison.png")
    return metrics
