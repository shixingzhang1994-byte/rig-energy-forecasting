from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog


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


def _load_config(path: Path) -> tuple[dict, Plant]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    interval_seconds = float(config["dispatch"]["interval_seconds"])
    grid = config["plant"]["grid"]
    gen = config["plant"]["generator"]
    storage = config["plant"]["storage"]
    cost = config["cost"]
    plant = Plant(
        step_hours=interval_seconds / 3600.0,
        grid_price=float(cost["grid_price_yuan_per_kwh"]),
        diesel_price=float(cost["diesel_price_yuan_per_l"]),
        storage_degradation=float(cost["storage_degradation_yuan_per_kwh"]),
        shortage_penalty=float(cost["unserved_energy_penalty_yuan_per_kwh"]),
        generator_min_kw=float(gen["minimum_stable_power_kw"]),
        generator_max_kw=float(gen["rated_power_kw"]),
        generator_ramp_kw_per_step=float(gen["ramp_kw_per_step"]),
        storage_energy_kwh=float(storage["energy_capacity_kwh"]),
        storage_power_kw=float(storage["rated_power_kw"]),
        soc_min=float(storage["soc_min"]),
        soc_max=float(storage["soc_max"]),
        soc_initial=float(storage["soc_initial"]),
        charge_efficiency=float(storage["charge_efficiency"]),
        discharge_efficiency=float(storage["discharge_efficiency"]),
    )
    return config, plant


def _fuel_lph(power_kw: np.ndarray | float, rated_power_kw: float) -> np.ndarray:
    """CAT 300 ekW公开数据按容量比例缩放的分段线性燃油曲线。"""
    power = np.asarray(power_kw, dtype=float)
    scale = rated_power_kw / 300.0
    points_kw = np.array([0.0, 150.0, 225.0, 300.0]) * scale
    points_lph = np.array([0.0, 51.3, 66.7, 86.1]) * scale
    return np.interp(np.clip(power, 0.0, rated_power_kw), points_kw, points_lph)


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
    return (
        min(plant.storage_power_kw, discharge_energy_limited),
        min(plant.storage_power_kw, charge_energy_limited),
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
) -> dict[str, float]:
    demand = np.asarray(forecast_kw, dtype=float)
    horizon = len(demand)
    # Variable blocks: grid, generator, charge, discharge, unserved, spill, SOC[0:H+1].
    g0, d0, c0, b0, u0, w0, s0 = (i * horizon for i in range(7))
    size = 7 * horizon + 1
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
        previous_index = None if step == 0 else d0 + step - 1
        previous_value = previous_generator_kw if step == 0 else None
        up = np.zeros(size)
        up[d0 + step] = 1.0
        down = np.zeros(size)
        down[d0 + step] = -1.0
        if previous_index is None:
            a_ub.extend([up, down])
            b_ub.extend(
                [
                    plant.generator_ramp_kw_per_step + previous_value,
                    plant.generator_ramp_kw_per_step - previous_value,
                ]
            )
        else:
            up[previous_index] = -1.0
            down[previous_index] = 1.0
            a_ub.extend([up, down])
            b_ub.extend([plant.generator_ramp_kw_per_step] * 2)

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"滚动优化求解失败: {result.message}")
    x = result.x
    return {
        "grid_kw": float(x[g0]),
        "generator_kw": float(x[d0]),
        "charge_kw": float(x[c0]),
        "discharge_kw": float(x[b0]),
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
) -> dict[str, float]:
    """Two-stage scenario MPC with CVaR on unserved-energy cost.

    The first control action is non-anticipative across all forecast-member
    scenarios; later horizon actions are scenario recourse.  This is materially
    different from adding a fixed reserve to one point forecast.
    """

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
    # Per-scenario blocks: grid, generator, charge, discharge, unserved, spill,
    # SOC[0:H+1].  eta and z_s form the standard linear CVaR epigraph.
    block_size = 7 * horizon + 1
    eta_index = scenario_count * block_size
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
            up = np.zeros(size)
            down = np.zeros(size)
            up[d0 + step] = 1.0
            down[d0 + step] = -1.0
            if step == 0:
                b_ub.extend(
                    [
                        plant.generator_ramp_kw_per_step + previous_generator_kw,
                        plant.generator_ramp_kw_per_step - previous_generator_kw,
                    ]
                )
            else:
                up[d0 + step - 1] = -1.0
                down[d0 + step - 1] = 1.0
                b_ub.extend([plant.generator_ramp_kw_per_step] * 2)
            a_ub.extend([up, down])
        # shortage_cost_s - eta - z_s <= 0
        row = np.zeros(size)
        row[u0 : u0 + horizon] = plant.shortage_penalty * plant.step_hours
        row[eta_index] = -1.0
        row[z_start + scenario] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"场景-CVaR滚动优化求解失败: {result.message}")
    x = result.x
    return {
        "grid_kw": float(x[0]),
        "generator_kw": float(x[horizon]),
        "charge_kw": float(x[2 * horizon]),
        "discharge_kw": float(x[3 * horizon]),
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
    generator_upper = min(
        generator_capacity,
        previous_generator_kw + plant.generator_ramp_kw_per_step,
    )
    generator_lower = min(
        generator_upper,
        max(0.0, previous_generator_kw - plant.generator_ramp_kw_per_step),
    )
    generator = float(np.clip(command["generator_kw"], generator_lower, generator_upper))
    discharge = float(np.clip(command["discharge_kw"], 0.0, max_discharge))
    charge = float(np.clip(command["charge_kw"], 0.0, max_charge))

    net_supply = grid + generator + discharge - charge
    deficit = actual_load_kw - net_supply
    if deficit > 0.0:
        add = min(deficit, grid_capacity_kw - grid)
        grid += add
        deficit -= add
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
            "storage_kw": discharge - charge,
            "charge_kw": charge,
            "discharge_kw": discharge,
            "unserved_kw": unserved,
            "spill_kw": spill,
            "soc": soc_next,
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
    scenario_forecasts_kw: np.ndarray | None = None,
    scenario_cvar_cfg: dict | None = None,
) -> tuple[pd.DataFrame, float]:
    records = []
    soc = plant.soc_initial if initial_soc is None else float(np.clip(initial_soc, plant.soc_min, plant.soc_max))
    previous_generator = 0.0
    grid_capacity_array = (
        np.full(len(actual_kw), float(grid_capacity_kw))
        if np.ndim(grid_capacity_kw) == 0
        else np.asarray(grid_capacity_kw, dtype=float)
    )
    if len(grid_capacity_array) != len(actual_kw):
        raise ValueError("动态网电容量与负荷长度不一致")
    started = time.perf_counter()
    for index, actual in enumerate(actual_kw):
        grid_limit = float(grid_capacity_array[index])
        generator_limit = (
            None if generator_available_kw is None else float(generator_available_kw[index])
        )
        storage_limit = (
            None if storage_available_kw is None else float(storage_available_kw[index])
        )
        applied_risk_reserve = 0.0
        applied_cvar_alpha = np.nan
        applied_cvar_weight = np.nan
        if method == "Rule-Based":
            command = _rule_command(
                float(actual), grid_limit, soc, plant, generator_limit, storage_limit
            )
        elif method in {"Scenario-CVaR-MPC", "Risk-Adaptive-CVaR-MPC"}:
            if scenario_forecasts_kw is None or scenario_cvar_cfg is None:
                raise ValueError("场景-CVaR调度缺少预测场景或配置")
            if method == "Risk-Adaptive-CVaR-MPC":
                if risk_levels is None:
                    raise ValueError("风险自适应CVaR调度缺少四级风险信号")
                level = str(int(risk_levels[index]))
                adaptive = scenario_cvar_cfg["risk_adaptive"]
                applied_cvar_alpha = float(adaptive["alpha_by_level"][level])
                applied_cvar_weight = float(adaptive["weight_by_level"][level])
                terminal_drop = float(adaptive["terminal_soc_drop_by_level"][level])
            else:
                applied_cvar_alpha = float(scenario_cvar_cfg["alpha"])
                applied_cvar_weight = float(scenario_cvar_cfg["weight"])
                terminal_drop = float(scenario_cvar_cfg["terminal_soc_drop"])
            command = _solve_scenario_cvar_mpc(
                scenario_forecasts_kw[index],
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
            )
        else:
            if method == "Persistence-MPC":
                demand = persistence_kw[index]
                terminal_drop = float(robust_cfg["deterministic_terminal_soc_drop"])
            elif method == "ML-Robust-MPC":
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
            )
        executed, soc = _execute_with_safety_layer(
            float(actual),
            command,
            grid_limit,
            soc,
            previous_generator,
            plant,
            generator_limit,
            storage_limit,
        )
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
            if method in {"Scenario-CVaR-MPC", "Risk-Adaptive-CVaR-MPC"}
            and scenario_forecasts_kw is not None
            else 1
        )
        executed["cvar_alpha"] = applied_cvar_alpha
        executed["cvar_weight"] = applied_cvar_weight
        executed["grid_available_capacity_kw"] = grid_limit
        executed["generator_available_capacity_kw"] = (
            plant.generator_max_kw if generator_limit is None else generator_limit
        )
        executed["storage_available_power_kw"] = (
            plant.storage_power_kw if storage_limit is None else storage_limit
        )
        records.append(executed)
        previous_generator = executed["generator_kw"]
    return pd.DataFrame(records), time.perf_counter() - started


def _metrics(
    frame: pd.DataFrame,
    plant: Plant,
    solve_seconds: float,
    reference_soc: float | None = None,
) -> dict[str, float]:
    dt = plant.step_hours
    fuel_l = float((_fuel_lph(frame["generator_kw"].to_numpy(), plant.generator_max_kw) * dt).sum())
    grid_energy = float(frame["grid_kw"].sum() * dt)
    generator_energy = float(frame["generator_kw"].sum() * dt)
    storage_throughput = float((frame["charge_kw"] + frame["discharge_kw"]).sum() * dt)
    unserved_energy = float(frame["unserved_kw"].sum() * dt)
    energy_cost = (
        grid_energy * plant.grid_price
        + fuel_l * plant.diesel_price
        + storage_throughput * plant.storage_degradation
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
    on = frame["generator_kw"].to_numpy() > 1.0
    startups = int(np.sum(on & ~np.r_[False, on[:-1]]))
    mean_load_pct = (
        float(frame.loc[on, "generator_kw"].mean() / plant.generator_max_kw * 100.0)
        if on.any()
        else 0.0
    )
    return {
        "energy_cost_yuan": energy_cost,
        "soc_restoration_cost_yuan": restoration_cost,
        "equivalent_cost_yuan": energy_cost + restoration_cost,
        "risk_adjusted_cost_yuan": (
            energy_cost + restoration_cost + unserved_energy * plant.shortage_penalty
        ),
        "grid_energy_kwh": grid_energy,
        "generator_energy_kwh": generator_energy,
        "diesel_fuel_l": fuel_l,
        "storage_throughput_kwh": storage_throughput,
        "unserved_energy_kwh": unserved_energy,
        "max_unserved_power_kw": float(frame["unserved_kw"].max()),
        "spill_energy_kwh": float(frame["spill_kw"].sum() * dt),
        "minimum_soc_pct": float(frame["soc"].min() * 100.0),
        "final_soc_pct": final_soc * 100.0,
        "restoration_reference_soc_pct": restoration_reference * 100.0,
        "generator_startups": startups,
        "mean_generator_load_pct_when_on": mean_load_pct,
        "solve_seconds": solve_seconds,
    }


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
) -> pd.DataFrame:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config, plant = _load_config(config_path)
    arrays = np.load(prediction_path)
    key_map = json.loads(Path(prediction_key_path).read_text(encoding="utf-8"))
    name_to_key = {name: key for key, name in key_map.items()}
    y_true = arrays["y_true"]
    transition = arrays["transition_flags"]
    robust_cfg = config["robust_mpc"]
    forecast_model = str(robust_cfg.get("forecast_model", "StateAware-TCN-Attention"))
    if forecast_model not in name_to_key:
        raise ValueError(
            f"预测轨迹中不存在 {forecast_model}，可用模型: {sorted(name_to_key)}"
        )
    proposed = arrays[name_to_key[forecast_model]]
    persistence = arrays[name_to_key["Persistence"]]
    scenario_cvar_cfg = config["scenario_cvar_mpc"]
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
        proposed = proposed[source_indices]
        persistence = persistence[source_indices]
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
    windows = (
        _select_supply_aligned_windows(y_true, transition, risk_frame, length)
        if risk_frame is not None
        else _select_scenario_windows(y_true, transition, length)
    )
    scenario_cfg = config["scenarios"]
    methods = [
        "Rule-Based",
        "Persistence-MPC",
        "ML-Robust-MPC",
        "Scenario-CVaR-MPC",
    ]
    if risk_signal_path is not None:
        methods.append("Risk-Aware-Ensemble-MPC")
        methods.append("Risk-Adaptive-CVaR-MPC")
    all_metrics = []
    selection = {}
    for scenario, selected in windows.items():
        execution_start = max(0, selected.start - requested_warmup)
        warmup = selected.start - execution_start
        execution_slice = slice(execution_start, selected.stop)
        actual = y_true[execution_slice, 0]
        ml_forecast = proposed[execution_slice]
        persistence_forecast = persistence[execution_slice]
        scenario_risk_levels = (
            risk_levels[execution_slice] if risk_levels is not None else None
        )
        scenario_risk_reserve = (
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
        trajectories = {}
        selection[scenario] = {
            "start_index": int(source_indices[selected.start]),
            "stop_index": int(source_indices[selected.stop - 1] + 1),
            "warmup_steps": warmup,
            "grid_capacity_kw": grid_metadata,
            "supply_regime_fractions": regime_counts,
            "mean_load_kw": float(actual[warmup:].mean()),
            "max_load_kw": float(actual[warmup:].max()),
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
                initial_soc=initial_soc,
                scenario_forecasts_kw=scenario_forecasts[execution_slice],
                scenario_cvar_cfg=scenario_cvar_cfg,
            )
            scored_trajectory = trajectory.iloc[warmup:].copy().reset_index(drop=True)
            scored_trajectory["step"] = np.arange(len(scored_trajectory))
            scored_trajectory.to_csv(
                artifact_dir / f"trajectory_{scenario}_{method.lower().replace('-', '_')}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            row = {"scenario": scenario, "method": method}
            scoring_reference_soc = (
                float(trajectory["soc"].iloc[warmup - 1])
                if warmup > 0
                else (plant.soc_initial if initial_soc is None else initial_soc)
            )
            row.update(
                _metrics(
                    scored_trajectory,
                    plant,
                    solve_seconds,
                    reference_soc=scoring_reference_soc,
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
                    "alpha": float(scenario_cvar_cfg["alpha"]),
                    "weight": float(scenario_cvar_cfg["weight"]),
                    "nonanticipativity": "first_control_action_only",
                },
                "risk_signal_path": str(risk_signal_path) if risk_signal_path else None,
                "scenarios": selection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_summary(metrics, artifact_dir / "dispatch_metric_comparison.png")
    return metrics
