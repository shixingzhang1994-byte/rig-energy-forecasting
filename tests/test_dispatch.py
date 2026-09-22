from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.optimization.dispatch import (  # noqa: E402
    _cancel_recent_transition_locks,
    _available_storage_power,
    _convex_generator_fuel_coefficients,
    _execute_with_safety_layer,
    _load_config,
    _metrics,
    _select_operation_aligned_windows,
    _simulate_method,
    _solve_linear_mpc,
    _solve_scenario_cvar_mpc,
    _terminal_soc_drop_for_comparison,
    _fuel_lph,
    _generator_fuel_lph,
    _supervisory_transition,
)
from rig_energy.data.schema import STATE_TO_CODE  # noqa: E402


def test_scaled_generator_fuel_curve_is_monotonic():
    power = np.array([0.0, 300.0, 600.0, 900.0, 1200.0])
    fuel = _fuel_lph(power, 1200.0)
    assert fuel[0] == 0.0
    assert np.all(np.diff(fuel) >= 0.0)


def test_multi_unit_fuel_uses_actual_committed_unit_loading():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
    )
    frame = pd.DataFrame(
        {"generator_kw": [0.0, 500.0], "generator_units_on": [0, 2]}
    )
    fuel = _generator_fuel_lph(frame, plant)
    assert fuel[0] == 0.0
    assert fuel[1] == 2.0 * _fuel_lph(250.0, 300.0)


def test_v22_cat_curve_has_exact_convex_milp_coefficients(monkeypatch):
    from rig_energy.optimization import dispatch as dispatch_module

    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    from run_v14_dispatch_benchmark import _v14_fuel_lph

    monkeypatch.setattr(dispatch_module, "_fuel_lph", _v14_fuel_lph)
    _, base = _load_config(PROJECT_DIR / "configs/v22_dispatch_matched.yaml")
    slope, intercept, breakpoint, high_increment = (
        _convex_generator_fuel_coefficients(base)
    )

    assert breakpoint == pytest.approx(180.0)
    for power in (120.0, 150.0, 180.0, 210.0, 240.0):
        modeled = slope * power + intercept + high_increment * max(
            0.0, power - breakpoint
        )
        assert modeled == pytest.approx(float(_v14_fuel_lph(power, 240.0)))


def test_v22_storage_power_derates_near_soc_limits() -> None:
    _, plant = _load_config(PROJECT_DIR / "configs/v22_dispatch_matched.yaml")

    low_discharge, _ = _available_storage_power(plant.soc_min + 0.025, plant)
    rated_discharge, _ = _available_storage_power(plant.soc_min + 0.05, plant)
    _, high_charge = _available_storage_power(plant.soc_max - 0.025, plant)

    assert low_discharge == pytest.approx(0.5 * plant.storage_power_kw)
    assert rated_discharge == pytest.approx(plant.storage_power_kw)
    assert high_charge == pytest.approx(0.5 * plant.storage_power_kw)


def test_forced_outage_cancels_newest_minimum_up_locks():
    history = [1, 2, 1]
    _cancel_recent_transition_locks(history, 2)
    assert history == [1, 1, 0]


def test_supervisor_enters_immediately_and_exits_only_after_safe_dwell():
    config = {
        "risk_enter_level": 2,
        "risk_exit_level": 1,
        "soc_enter": 0.30,
        "soc_exit": 0.35,
        "exit_dwell_steps": 12,
    }
    active, counter = _supervisory_transition(False, 0, 2, 0.60, config)
    assert active is True
    assert counter == 0

    for _ in range(11):
        active, counter = _supervisory_transition(
            active, counter, 1, 0.40, config
        )
        assert active is True
    active, counter = _supervisory_transition(active, counter, 1, 0.40, config)
    assert active is False
    assert counter == 0


def test_supervisor_low_soc_enters_and_unsafe_row_resets_exit_dwell():
    config = {
        "risk_enter_level": 2,
        "risk_exit_level": 1,
        "soc_enter": 0.30,
        "soc_exit": 0.35,
        "exit_dwell_steps": 3,
    }
    active, counter = _supervisory_transition(False, 0, 0, 0.29, config)
    assert active is True
    active, counter = _supervisory_transition(active, counter, 0, 0.40, config)
    assert counter == 1
    active, counter = _supervisory_transition(active, counter, 2, 0.40, config)
    assert active is True
    assert counter == 0


def test_supervisory_method_switches_controllers_with_shared_state():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    supervisor = dict(config["supervisory_mpc"])
    supervisor["exit_dwell_steps"] = 2
    actual = np.full(4, 900.0)
    point = np.full((4, 12), 900.0)
    scenarios = np.stack(
        [
            np.full((4, 12), 850.0),
            np.full((4, 12), 900.0),
            np.full((4, 12), 1000.0),
        ],
        axis=1,
    )
    trajectory, _ = _simulate_method(
        "Risk-SOC-Supervisory-MPC",
        actual,
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0, 2, 1, 1]),
        risk_reserve_kw=np.zeros(4),
        initial_soc=0.60,
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )
    assert trajectory["selected_controller"].tolist() == [
        "Risk-Adaptive-CVaR-MPC",
        "ML-Robust-MPC",
        "ML-Robust-MPC",
        "Risk-Adaptive-CVaR-MPC",
    ]
    assert trajectory["supervisory_protection_active"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert (trajectory["decision_seconds"] > 0.0).all()


def test_supervisory_low_soc_commitment_floor_is_wired_into_dispatch():
    config, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=12,
        generator_min_down_steps=6,
    )
    supervisor = dict(config["supervisory_mpc"])
    supervisor["low_soc_minimum_committed_units"] = 2
    actual = np.array([1550.0])
    point = np.full((1, 12), 1550.0)
    scenarios = np.stack(
        [
            np.full((1, 12), 1500.0),
            np.full((1, 12), 1550.0),
            np.full((1, 12), 1600.0),
        ],
        axis=1,
    )
    trajectory, _ = _simulate_method(
        "Risk-SOC-Supervisory-MPC",
        actual,
        point,
        point,
        1200.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        generator_available_kw=np.array([800.0]),
        storage_available_kw=np.array([500.0]),
        initial_soc=0.22,
        initial_generator_kw=300.0,
        initial_generator_units=1,
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )
    assert trajectory.loc[0, "supervisory_protection_active"]
    assert trajectory.loc[0, "minimum_committed_generator_units"] == 2
    assert trajectory.loc[0, "generator_units_on"] >= 2


def test_low_soc_commitment_guard_suppresses_floor_without_forecast_deficit():
    config, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=12,
        generator_min_down_steps=6,
    )
    supervisor = dict(config["supervisory_mpc"])
    supervisor.update(
        {
            "low_soc_minimum_committed_units": 2,
            "low_soc_forecast_deficit_guard_enabled": True,
            "low_soc_forecast_deficit_margin_kw": 0.0,
        }
    )
    point = np.full((1, 12), 1400.0)
    scenarios = np.stack(
        [
            np.full((1, 12), 1350.0),
            np.full((1, 12), 1400.0),
            np.full((1, 12), 1450.0),
        ],
        axis=1,
    )
    trajectory, _ = _simulate_method(
        "Risk-SOC-Supervisory-MPC",
        np.array([1400.0]),
        point,
        point,
        1200.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        generator_available_kw=np.array([800.0]),
        storage_available_kw=np.array([500.0]),
        initial_soc=0.22,
        initial_generator_kw=300.0,
        initial_generator_units=1,
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )
    assert trajectory.loc[0, "low_soc_commitment_guard_evaluated"]
    assert not trajectory.loc[0, "low_soc_commitment_guard_triggered"]
    assert trajectory.loc[0, "low_soc_commitment_deficit_kw"] == -50.0
    assert trajectory.loc[0, "minimum_committed_generator_units"] == 0


def test_low_soc_commitment_guard_applies_floor_for_forecast_deficit():
    config, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=12,
        generator_min_down_steps=6,
    )
    supervisor = dict(config["supervisory_mpc"])
    supervisor.update(
        {
            "low_soc_minimum_committed_units": 2,
            "low_soc_forecast_deficit_guard_enabled": True,
            "low_soc_forecast_deficit_margin_kw": 0.0,
        }
    )
    point = np.full((1, 12), 1550.0)
    scenarios = np.stack(
        [
            np.full((1, 12), 1500.0),
            np.full((1, 12), 1550.0),
            np.full((1, 12), 1600.0),
        ],
        axis=1,
    )
    trajectory, _ = _simulate_method(
        "Risk-SOC-Supervisory-MPC",
        np.array([1550.0]),
        point,
        point,
        1200.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        generator_available_kw=np.array([800.0]),
        storage_available_kw=np.array([500.0]),
        initial_soc=0.22,
        initial_generator_kw=300.0,
        initial_generator_units=1,
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )
    assert trajectory.loc[0, "low_soc_commitment_guard_triggered"]
    assert trajectory.loc[0, "low_soc_commitment_deficit_kw"] == 100.0
    assert trajectory.loc[0, "minimum_committed_generator_units"] == 2
    assert trajectory.loc[0, "generator_units_on"] >= 2


def test_low_soc_commitment_floor_is_truncated_by_minimum_down_reachability():
    config, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=12,
        generator_min_down_steps=6,
    )
    command = _solve_linear_mpc(
        np.full(12, 1550.0),
        grid_capacity_kw=1200.0,
        soc=0.22,
        previous_generator_kw=150.0,
        plant=plant,
        terminal_soc_drop=0.005,
        generator_available_kw=800.0,
        storage_available_kw=500.0,
        previous_generator_units=1,
        shutdown_history=[1, 1, 1, 0, 0],
        minimum_generator_units=2,
    )
    assert command["generator_units_on"] == 1


def test_grid_headroom_reserve_charges_without_starting_generator():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    executed, next_soc = _execute_with_safety_layer(
        actual_load_kw=800.0,
        command={
            "grid_kw": 800.0,
            "generator_kw": 0.0,
            "generator_units_on": 0,
            "charge_kw": 0.0,
            "discharge_kw": 0.0,
        },
        grid_capacity_kw=1200.0,
        soc=0.70,
        previous_generator_kw=0.0,
        plant=plant,
        grid_headroom_reserve_target_soc=0.75,
        grid_headroom_reserve_max_charge_kw=100.0,
    )
    assert executed["grid_kw"] == 900.0
    assert executed["generator_kw"] == 0.0
    assert executed["charge_kw"] == 100.0
    assert executed["grid_headroom_storage_charge_kw"] == 100.0
    assert next_soc > 0.70


def test_grid_headroom_reserve_replaces_avoidable_storage_discharge():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=900.0,
        command={
            "grid_kw": 800.0,
            "generator_kw": 0.0,
            "generator_units_on": 0,
            "charge_kw": 0.0,
            "discharge_kw": 100.0,
        },
        grid_capacity_kw=1200.0,
        soc=0.70,
        previous_generator_kw=0.0,
        plant=plant,
        grid_headroom_reserve_target_soc=0.75,
        grid_headroom_reserve_max_charge_kw=100.0,
    )
    assert executed["grid_kw"] == 900.0
    assert executed["discharge_kw"] == 0.0
    assert executed["charge_kw"] == 0.0
    assert executed["grid_headroom_storage_preservation_kw"] == 100.0


def test_grid_headroom_reserve_is_inactive_at_target_soc():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=800.0,
        command={
            "grid_kw": 800.0,
            "generator_kw": 0.0,
            "generator_units_on": 0,
            "charge_kw": 0.0,
            "discharge_kw": 0.0,
        },
        grid_capacity_kw=1200.0,
        soc=0.75,
        previous_generator_kw=0.0,
        plant=plant,
        grid_headroom_reserve_target_soc=0.75,
        grid_headroom_reserve_max_charge_kw=100.0,
    )
    assert executed["grid_kw"] == 800.0
    assert executed["charge_kw"] == 0.0
    assert executed["grid_headroom_storage_charge_kw"] == 0.0


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
    assert 0.0 <= command["generator_kw"] <= 420.0
    assert (
        command["generator_kw"] == 0.0
        or command["generator_kw"] >= plant.generator_min_kw
    )
    assert 0.0 <= command["charge_kw"] <= plant.storage_power_kw
    assert 0.0 <= command["discharge_kw"] <= plant.storage_power_kw


def test_executed_dispatch_never_runs_generator_below_stable_power():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    actual = np.array([950.0])
    forecast = np.full((1, 12), 950.0)
    trajectory, _ = _simulate_method(
        "Rule-Based",
        actual,
        forecast,
        forecast,
        np.array([900.0]),
        plant,
        config["robust_mpc"],
        generator_available_kw=np.array([1200.0]),
        storage_available_kw=np.array([0.0]),
    )
    generator = float(trajectory.loc[0, "generator_kw"])
    assert generator == 0.0 or generator >= plant.generator_min_kw
    supplied = (
        trajectory["grid_kw"]
        + trajectory["generator_kw"]
        + trajectory["storage_kw"]
        + trajectory["unserved_kw"]
        - trajectory["spill_kw"]
    )
    assert np.allclose(supplied, actual)


def test_safety_layer_starts_generator_when_real_time_load_exceeds_other_supply():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=1100.0,
        command={
            "grid_kw": 900.0,
            "generator_kw": 0.0,
            "charge_kw": 0.0,
            "discharge_kw": 0.0,
        },
        grid_capacity_kw=900.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        generator_available_kw=plant.generator_max_kw,
        storage_available_kw=0.0,
    )
    assert executed["generator_kw"] >= plant.generator_min_kw
    assert executed["unserved_kw"] == 0.0


def test_linear_mpc_command_is_off_or_above_generator_stable_power():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    command = _solve_linear_mpc(
        np.full(12, 1000.0),
        grid_capacity_kw=900.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.0,
        storage_available_kw=0.0,
    )
    assert (
        command["generator_kw"] == 0.0
        or command["generator_kw"] >= plant.generator_min_kw
    )


def test_clustered_unit_commitment_starts_only_physical_unit_counts():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = _solve_linear_mpc(
        np.full(12, 250.0),
        grid_capacity_kw=0.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.0,
        storage_available_kw=0.0,
        previous_generator_units=0,
    )
    assert command["generator_units_on"] == 4
    assert command["generator_kw"] == 240.0


def test_clustered_commitment_honours_recent_start_minimum_up_time():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=3,
    )
    command = _solve_linear_mpc(
        np.zeros(12),
        grid_capacity_kw=0.0,
        soc=plant.soc_min,
        previous_generator_kw=60.0,
        plant=plant,
        terminal_soc_drop=0.0,
        storage_available_kw=0.0,
        previous_generator_units=1,
        startup_history=(1,),
    )
    assert command["generator_units_on"] >= 1
    assert command["generator_kw"] >= 60.0


def test_safety_layer_applies_multi_unit_startup_ramp_and_reports_commitment():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=500.0,
        command={
            "grid_kw": 0.0,
            "generator_kw": 0.0,
            "generator_units_on": 0,
            "charge_kw": 0.0,
            "discharge_kw": 0.0,
        },
        grid_capacity_kw=0.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        generator_available_kw=plant.generator_max_kw,
        storage_available_kw=0.0,
        previous_generator_units=0,
    )
    assert executed["generator_units_on"] == 4
    assert executed["generator_kw"] == 240.0
    assert executed["unserved_kw"] == 260.0


def test_high_risk_corrective_dispatch_uses_online_ramp_before_storage():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = {
        "grid_kw": 0.0,
        "generator_kw": 240.0,
        "generator_units_on": 4,
        "charge_kw": 0.0,
        "discharge_kw": 260.0,
    }
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=500.0,
        command=command,
        grid_capacity_kw=0.0,
        soc=0.65,
        previous_generator_kw=240.0,
        plant=plant,
        storage_available_kw=500.0,
        previous_generator_units=4,
        generator_before_storage=True,
    )
    assert executed["generator_kw"] == 360.0
    assert executed["discharge_kw"] == 140.0
    assert executed["unserved_kw"] == 0.0


def test_low_soc_commitment_floor_starts_second_unit_before_storage():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = {
        "grid_kw": 1200.0,
        "generator_kw": 300.0,
        "generator_units_on": 1,
        "charge_kw": 0.0,
        "discharge_kw": 50.0,
    }
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=1550.0,
        command=command,
        grid_capacity_kw=1200.0,
        soc=0.22,
        previous_generator_kw=300.0,
        plant=plant,
        generator_available_kw=800.0,
        storage_available_kw=500.0,
        previous_generator_units=1,
        generator_before_storage=True,
        minimum_generator_units=2,
    )
    assert executed["generator_units_on"] == 2
    assert executed["generator_kw"] == 350.0
    assert executed["discharge_kw"] == 0.0
    assert executed["unserved_kw"] == 0.0


def test_low_soc_commitment_floor_is_truncated_by_available_capacity():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = {
        "grid_kw": 0.0,
        "generator_kw": 0.0,
        "generator_units_on": 0,
        "charge_kw": 0.0,
        "discharge_kw": 0.0,
    }
    executed, _ = _execute_with_safety_layer(
        actual_load_kw=100.0,
        command=command,
        grid_capacity_kw=0.0,
        soc=0.22,
        previous_generator_kw=0.0,
        plant=plant,
        generator_available_kw=100.0,
        storage_available_kw=0.0,
        previous_generator_units=0,
        generator_before_storage=True,
        minimum_generator_units=2,
    )
    assert executed["generator_units_on"] == 1
    assert executed["generator_kw"] == 60.0
    assert executed["unserved_kw"] == 40.0


def test_clustered_linear_mpc_respects_low_soc_commitment_floor():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = _solve_linear_mpc(
        np.full(12, 1500.0),
        grid_capacity_kw=1200.0,
        soc=0.22,
        previous_generator_kw=300.0,
        plant=plant,
        terminal_soc_drop=0.005,
        generator_available_kw=800.0,
        storage_available_kw=500.0,
        previous_generator_units=1,
        minimum_generator_units=2,
    )
    assert command["generator_units_on"] >= 2
    assert command["generator_kw"] >= 120.0


def test_clustered_scenario_cvar_uses_shared_integer_commitment():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
        generator_min_up_steps=3,
    )
    scenarios = np.vstack(
        [np.full(12, 200.0), np.full(12, 250.0), np.full(12, 500.0)]
    )
    command = _solve_scenario_cvar_mpc(
        scenarios,
        grid_capacity_kw=0.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.0,
        cvar_alpha=0.90,
        cvar_weight=1.0,
        storage_available_kw=0.0,
        previous_generator_units=0,
    )
    assert command["generator_units_on"] == 4
    assert command["generator_kw"] == 240.0
    assert command["predicted_expected_shortage_cost_yuan"] >= 0.0
    assert command["predicted_cvar_shortage_cost_yuan"] >= 0.0
    assert command["predicted_weighted_cvar_term_yuan"] == pytest.approx(
        command["predicted_cvar_shortage_cost_yuan"]
    )


def test_risk_reachability_floor_precommits_generator_units():
    _, base = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    plant = replace(
        base,
        generator_unit_count=4,
        generator_unit_rated_kw=300.0,
        generator_unit_min_kw=60.0,
        generator_unit_ramp_kw_per_step=30.0,
    )
    command = _solve_scenario_cvar_mpc(
        np.zeros((3, 12)),
        grid_capacity_kw=0.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.0,
        cvar_alpha=0.90,
        cvar_weight=1.0,
        storage_available_kw=0.0,
        previous_generator_units=0,
        minimum_generator_units=2,
    )
    assert command["generator_units_on"] >= 2
    assert command["generator_kw"] >= 120.0


def test_rolling_execution_never_charges_and_discharges_storage_together():
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    actual = np.array([500.0, 520.0])
    overforecast = np.full((2, 12), 1500.0)
    trajectory, _ = _simulate_method(
        "Persistence-MPC",
        actual,
        overforecast,
        overforecast,
        900.0,
        plant,
        config["robust_mpc"],
    )
    simultaneous = (
        (trajectory["charge_kw"] > 1e-8)
        & (trajectory["discharge_kw"] > 1e-8)
    )
    assert not simultaneous.any()


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
    assert 0.0 <= cvar_command["generator_kw"] <= plant.generator_max_kw
    assert (
        cvar_command["generator_kw"] == 0.0
        or cvar_command["generator_kw"] >= plant.generator_min_kw
    )
    assert 0.0 <= cvar_command["discharge_kw"] <= plant.storage_power_kw
    assert cvar_command["predicted_cvar_shortage_cost_yuan"] >= (
        cvar_command["predicted_expected_shortage_cost_yuan"] - 1e-9
    )


def test_scenario_cvar_command_is_off_or_above_generator_stable_power():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    scenarios = np.full((3, 12), 1000.0)
    command = _solve_scenario_cvar_mpc(
        scenarios,
        grid_capacity_kw=900.0,
        soc=plant.soc_min,
        previous_generator_kw=0.0,
        plant=plant,
        terminal_soc_drop=0.0,
        cvar_alpha=0.90,
        cvar_weight=1.0,
        storage_available_kw=0.0,
    )
    assert (
        command["generator_kw"] == 0.0
        or command["generator_kw"] >= plant.generator_min_kw
    )


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


def test_risk_gated_cvar_uses_point_mpc_normally_and_cvar_at_high_risk():
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
        "Risk-Gated-CVaR-MPC",
        actual,
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0, 3]),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
    )
    assert trajectory["forecast_scenario_count"].tolist() == [1, 3]
    assert np.isnan(trajectory["cvar_alpha"].iloc[0])
    assert trajectory["cvar_alpha"].iloc[1] > 0.90


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


def test_cost_metrics_separate_operation_terminal_adjustment_and_reliability():
    _, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    frame = pd.DataFrame(
        {
            "grid_kw": [100.0, 100.0],
            "generator_kw": [0.0, 0.0],
            "charge_kw": [0.0, 0.0],
            "discharge_kw": [0.0, 0.0],
            "unserved_kw": [25.0, 0.0],
            "spill_kw": [0.0, 0.0],
            "soc": [0.70, 0.60],
        }
    )

    metrics = _metrics(frame, plant, solve_seconds=0.0, reference_soc=0.80)

    assert metrics["realized_operating_cost_yuan"] == pytest.approx(
        metrics["energy_cost_yuan"]
    )
    assert metrics["terminal_energy_adjustment_yuan"] == pytest.approx(
        metrics["soc_restoration_cost_yuan"]
    )
    assert metrics["reliability_penalty_yuan"] == pytest.approx(
        metrics["unserved_energy_kwh"] * plant.shortage_penalty
    )
    assert metrics["total_social_cost_yuan"] == pytest.approx(
        metrics["realized_operating_cost_yuan"]
        + metrics["terminal_energy_adjustment_yuan"]
        + metrics["reliability_penalty_yuan"]
    )
    assert metrics["loss_of_load_duration_hours"] == pytest.approx(
        plant.step_hours
    )


def test_v22_matched_terminal_policy_overrides_method_specific_drop() -> None:
    robust = {
        "matched_terminal_soc": {
            "enabled": True,
            "terminal_soc_drop_by_level": {
                "0": 0.005,
                "1": 0.010,
                "2": 0.020,
                "3": 0.030,
            },
        }
    }

    assert _terminal_soc_drop_for_comparison(robust, 0, 0.025) == 0.005
    assert _terminal_soc_drop_for_comparison(robust, 3, 0.005) == 0.030
    assert _terminal_soc_drop_for_comparison({}, 3, 0.025) == 0.025


@pytest.mark.parametrize(
    "method",
    [
        "Persistence-MILP",
        "Point-Forecast-MILP",
        "Fixed-Reserve-MILP",
        "Residual-CVaR-MILP",
        "Risk-Adaptive-Residual-CVaR-MILP",
        "Full-Risk-SOC-Supervisory-MILP",
        "Full-Point-Scenario-Ablation-MILP",
        "Full-Fixed-CVaR-Ablation-MILP",
        "Full-No-Generator-First-Ablation-MILP",
    ],
)
def test_v22_matched_method_names_execute(method: str) -> None:
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    robust = dict(config["robust_mpc"])
    robust["matched_terminal_soc"] = {
        "enabled": True,
        "terminal_soc_drop_by_level": {
            "0": 0.005,
            "1": 0.010,
            "2": 0.020,
            "3": 0.030,
        },
    }
    supervisor = dict(config["supervisory_mpc"])
    supervisor["enabled"] = True
    point = np.full((1, 12), 900.0)
    scenarios = np.stack(
        [
            np.full((1, 12), 850.0),
            np.full((1, 12), 900.0),
            np.full((1, 12), 950.0),
        ],
        axis=1,
    )

    trajectory, _ = _simulate_method(
        method,
        np.array([900.0]),
        point,
        point,
        1200.0,
        plant,
        robust,
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )

    assert trajectory.loc[0, "terminal_soc_drop"] == pytest.approx(0.005)


def test_v22_full_controller_honors_generator_first_flag_at_low_risk() -> None:
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    supervisor = dict(config["supervisory_mpc"])
    supervisor.update(
        {
            "enabled": True,
            "generator_first_storage_reserve_enabled": True,
        }
    )
    point = np.full((1, 12), 1300.0)
    scenarios = np.repeat(point[:, None, :], 3, axis=1)

    trajectory, _ = _simulate_method(
        "Full-Risk-SOC-Supervisory-MILP",
        np.array([1300.0]),
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )

    assert bool(trajectory.loc[0, "generator_before_storage"])


def test_v22_no_generator_first_ablation_disables_action_ordering() -> None:
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    supervisor = dict(config["supervisory_mpc"])
    supervisor.update(
        {
            "enabled": True,
            "generator_first_storage_reserve_enabled": True,
        }
    )
    point = np.full((1, 12), 1300.0)
    scenarios = np.repeat(point[:, None, :], 3, axis=1)

    trajectory, _ = _simulate_method(
        "Full-No-Generator-First-Ablation-MILP",
        np.array([1300.0]),
        point,
        point,
        900.0,
        plant,
        config["robust_mpc"],
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )

    assert not bool(trajectory.loc[0, "generator_before_storage"])


def test_v22_fixed_cvar_supervisor_reports_fixed_alpha() -> None:
    config, plant = _load_config(PROJECT_DIR / "configs/dispatch_default.yaml")
    robust = dict(config["robust_mpc"])
    robust["matched_terminal_soc"] = {"enabled": False}
    supervisor = dict(config["supervisory_mpc"])
    supervisor["enabled"] = True
    point = np.full((1, 12), 900.0)
    scenarios = np.repeat(point[:, None, :], 3, axis=1)

    trajectory, _ = _simulate_method(
        "Full-Fixed-CVaR-Ablation-MILP",
        np.array([900.0]),
        point,
        point,
        1200.0,
        plant,
        robust,
        risk_levels=np.array([0]),
        risk_reserve_kw=np.zeros(1),
        scenario_forecasts_kw=scenarios,
        scenario_cvar_cfg=config["scenario_cvar_mpc"],
        supervisory_cfg=supervisor,
    )

    assert trajectory.loc[0, "cvar_alpha"] == pytest.approx(
        config["scenario_cvar_mpc"]["alpha"]
    )


def test_generator_start_commands_observe_fixed_synchronization_delay() -> None:
    config, plant = _load_config(PROJECT_DIR / "configs/v22_dispatch_matched.yaml")
    robust = dict(config["robust_mpc"])
    robust["generator_startup_delay_steps"] = 2
    actual = np.full(4, 500.0)
    point = np.full((4, 12), 500.0)

    trajectory, _ = _simulate_method(
        "Rule-Based",
        actual,
        point,
        point,
        0.0,
        plant,
        robust,
        storage_available_kw=np.zeros(4),
        initial_soc=plant.soc_min,
    )

    assert trajectory["generator_kw"].iloc[0] == 0.0
    assert trajectory["generator_kw"].iloc[1] == 0.0
    assert trajectory["generator_start_command_units"].iloc[0] > 0
    assert trajectory["generator_synchronized_units"].iloc[2] > 0
    assert trajectory["generator_kw"].iloc[2] > 0.0


def test_operation_scenarios_cover_drilling_connection_and_tripping():
    block = 40
    states = np.array(
        [STATE_TO_CODE["drilling"]] * block
        + [STATE_TO_CODE["drilling"]] * 15
        + [STATE_TO_CODE["connection"]] * 10
        + [STATE_TO_CODE["drilling"]] * 15
        + [STATE_TO_CODE["tripping"]] * block,
        dtype=np.int16,
    )
    load = np.concatenate(
        [
            np.full(block, 1400.0),
            np.r_[np.full(15, 1400.0), np.linspace(1400.0, 700.0, 10), np.full(15, 1400.0)],
            700.0 + 300.0 * np.abs(np.sin(np.linspace(0, 4 * np.pi, block))),
        ]
    )
    transitions = np.r_[False, states[1:] != states[:-1]]
    specs = {
        "A_stable_drilling": {
            "primary_states": ["drilling"],
            "minimum_primary_fraction": 0.90,
            "score": "stable",
        },
        "B_connection_transition": {
            "primary_states": ["drilling", "connection"],
            "anchor_states": ["connection"],
            "minimum_primary_fraction": 0.90,
            "minimum_anchor_fraction": 0.20,
            "score": "transition",
        },
        "C_tripping_impact": {
            "primary_states": ["tripping"],
            "minimum_primary_fraction": 0.90,
            "score": "impact",
        },
    }
    windows, evidence = _select_operation_aligned_windows(
        load.reshape(-1, 1), transitions, states, length=40, specs=specs
    )
    assert set(windows) == set(specs)
    assert evidence["A_stable_drilling"]["primary_fraction"] >= 0.90
    assert evidence["B_connection_transition"]["anchor_fraction"] >= 0.20
    assert evidence["C_tripping_impact"]["primary_fraction"] >= 0.90


def test_operation_scenarios_can_enforce_independent_supply_regime_gates():
    states = np.array(
        [STATE_TO_CODE["drilling"]] * 30
        + [STATE_TO_CODE["connection"]] * 10,
        dtype=np.int16,
    )
    supply = np.array(["emergency"] * 20 + ["constrained"] * 20, dtype=object)
    specs = {
        "connection": {
            "primary_states": ["drilling", "connection"],
            "anchor_states": ["connection"],
            "minimum_primary_fraction": 1.0,
            "minimum_anchor_fraction": 0.20,
            "acceptable_supply_regimes": ["constrained", "weak"],
            "minimum_supply_fraction": 0.40,
            "maximum_emergency_fraction": 0.60,
            "score": "transition",
        }
    }
    _, evidence = _select_operation_aligned_windows(
        np.linspace(800.0, 1400.0, 40).reshape(-1, 1),
        np.r_[False, states[1:] != states[:-1]],
        states,
        length=40,
        specs=specs,
        supply_regimes=supply,
    )
    assert evidence["connection"]["acceptable_supply_fraction"] == 0.5
    assert evidence["connection"]["emergency_supply_fraction"] == 0.5
