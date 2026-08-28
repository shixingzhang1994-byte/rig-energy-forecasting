from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.optimization.dispatch import (  # noqa: E402
    _load_config,
    _metrics,
    _simulate_method,
    _solve_linear_mpc,
    _solve_scenario_cvar_mpc,
    _fuel_lph,
)


def test_scaled_generator_fuel_curve_is_monotonic():
    power = np.array([0.0, 300.0, 600.0, 900.0, 1200.0])
    fuel = _fuel_lph(power, 1200.0)
    assert fuel[0] == 0.0
    assert np.all(np.diff(fuel) >= 0.0)


def test_linear_mpc_respects_first_step_bounds():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    command = _solve_linear_mpc(
        np.full(12, 1500.0),
        grid_capacity_kw=900.0,
        soc=0.65,
        previous_generator_kw=300.0,
        plant=plant,
        terminal_soc_drop=0.005,
    )
    assert 0.0 <= command["grid_kw"] <= 900.0
    assert 180.0 <= command["generator_kw"] <= 420.0
    assert 0.0 <= command["charge_kw"] <= plant.storage_power_kw
    assert 0.0 <= command["discharge_kw"] <= plant.storage_power_kw


def test_risk_aware_mpc_consumes_level_and_reserve_signals():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    actual = np.full(4, 1000.0)
    forecast = np.full((4, 12), 1000.0)
    risk_levels = np.array([0, 1, 2, 3])
    risk_reserve = np.array([0.0, 30.0, 80.0, 150.0])
    grid_available = np.array([900.0, 850.0, 800.0, 750.0])
    generator_available = np.array([700.0, 650.0, 600.0, 550.0])
    storage_available = np.array([300.0, 250.0, 200.0, 150.0])
    trajectory, _ = _simulate_method(
        "Risk-Aware-Ensemble-MPC",
        actual,
        forecast,
        forecast,
        grid_available,
        plant,
        config["robust_mpc"],
        risk_levels,
        risk_reserve,
        generator_available,
        storage_available,
        0.60,
    )
    assert trajectory["risk_level"].tolist() == risk_levels.tolist()
    assert trajectory["risk_reserve_kw"].tolist() == risk_reserve.tolist()
    assert trajectory["unserved_kw"].max() == 0.0
    assert np.all(trajectory["grid_kw"].to_numpy() <= grid_available + 1e-9)
    assert np.all(trajectory["generator_kw"].to_numpy() <= generator_available + 1e-9)
    assert trajectory["grid_available_capacity_kw"].tolist() == grid_available.tolist()


def test_scenario_cvar_mpc_hedges_a_rare_high_load_tail_within_bounds():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    low = np.full((9, 12), 500.0)
    high = np.full((1, 12), 1700.0)
    scenarios = np.vstack([low, high])
    point_command = _solve_linear_mpc(
        scenarios.mean(axis=0),
        grid_capacity_kw=900.0,
        soc=0.65,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.005,
    )
    cvar_command = _solve_scenario_cvar_mpc(
        scenarios,
        grid_capacity_kw=900.0,
        soc=0.65,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.005,
        cvar_alpha=0.90,
        cvar_weight=1.0,
    )
    point_firm = point_command["generator_kw"] + point_command["discharge_kw"]
    cvar_firm = cvar_command["generator_kw"] + cvar_command["discharge_kw"]
    assert cvar_firm >= point_firm - 1e-9
    assert 0.0 <= cvar_command["grid_kw"] <= 900.0
    assert 0.0 <= cvar_command["generator_kw"] <= 120.0
    assert 0.0 <= cvar_command["discharge_kw"] <= plant.storage_power_kw


def test_scenario_cvar_method_runs_in_the_same_rolling_safety_layer():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    actual = np.array([900.0, 1100.0])
    point = np.full((2, 12), 950.0)
    scenarios = np.stack(
        [
            np.full((2, 12), 850.0),
            np.full((2, 12), 1000.0),
            np.full((2, 12), 1250.0),
        ],
        axis=1,
    )
    trajectory, _ = _simulate_method(
        "Scenario-CVaR-MPC",
        actual,
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
    )
    assert len(trajectory) == len(actual)
    assert trajectory["unserved_kw"].max() == 0.0
    assert (trajectory["grid_kw"] <= 900.0 + 1e-9).all()


def test_risk_adaptive_cvar_consumes_four_level_signal_and_scenarios():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    actual = np.array([900.0, 1100.0])
    point = np.full((2, 12), 950.0)
    scenarios = np.stack(
        [
            np.full((2, 12), 850.0),
            np.full((2, 12), 1000.0),
            np.full((2, 12), 1250.0),
        ],
        axis=1,
    )
    risk_levels = np.array([0, 3])
    trajectory, _ = _simulate_method(
        "Risk-Adaptive-CVaR-MPC",
        actual,
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        risk_levels=risk_levels,
        risk_reserve_kw=np.zeros(2),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
    )
    assert trajectory["risk_level"].tolist() == [0, 3]
    assert trajectory["forecast_scenario_count"].tolist() == [3, 3]
    assert trajectory["cvar_alpha"].iloc[1] > trajectory["cvar_alpha"].iloc[0]
    assert trajectory["cvar_weight"].iloc[1] > trajectory["cvar_weight"].iloc[0]


def test_equivalent_cost_restores_the_actual_scoring_start_soc():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    frame = pd.DataFrame(
        {
            "grid_kw": [0.0],
            "generator_kw": [0.0],
            "charge_kw": [0.0],
            "discharge_kw": [0.0],
            "unserved_kw": [0.0],
            "spill_kw": [0.0],
            "soc": [0.60],
        }
    )
    metrics = _metrics(frame, plant, solve_seconds=0.0, reference_soc=0.80)
    expected_energy = (0.80 - 0.60) * plant.storage_energy_kwh / plant.charge_efficiency
    assert np.isclose(
        metrics["soc_restoration_cost_yuan"], expected_energy * plant.grid_price
    )
