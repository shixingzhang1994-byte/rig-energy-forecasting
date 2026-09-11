from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from run_independent_causal_full_milp import solve_causal_window


def _config() -> dict[str, float]:
    return {
        "unit_count": 2,
        "unit_rated_power_kw": 240.0,
        "unit_minimum_stable_power_kw": 120.0,
        "ramp_kw_per_step": 60.0,
        "minimum_up_steps": 3,
        "minimum_down_steps": 2,
        "energy_capacity_kwh": 133.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "interval_seconds": 5.0,
        "soc_min": 0.15,
        "soc_max": 0.90,
        "storage_degradation": 0.15,
        "grid_price": 0.71,
        "diesel_price": 7.50,
        "fuel_lph_at_rated": 62.50,
        "rated_power_kw": 240.0,
        "unserved_penalty": 100.0,
        "spill_penalty": 1.0,
        "startup_cost": 5.0,
    }


def test_recent_start_history_keeps_unit_committed() -> None:
    plan = solve_causal_window(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [480.0, 480.0, 480.0],
        [0.0, 0.0, 0.0],
        _config(),
        soc_initial=0.65,
        previous_generator_kw=120.0,
        previous_generator_units=1,
        startup_history=[1],
        shutdown_history=[],
    )
    assert plan["generator_units_on"] == 1.0


def test_without_start_history_unit_can_shut_down() -> None:
    plan = solve_causal_window(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [480.0, 480.0, 480.0],
        [0.0, 0.0, 0.0],
        _config(),
        soc_initial=0.65,
        previous_generator_kw=120.0,
        previous_generator_units=1,
        startup_history=[],
        shutdown_history=[],
    )
    assert plan["generator_units_on"] == 0.0


def test_forced_capacity_derating_overrides_unavailable_minimum_up_locks() -> None:
    plan = solve_causal_window(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [120.0, 120.0, 120.0],
        [0.0, 0.0, 0.0],
        _config(),
        soc_initial=0.65,
        previous_generator_kw=240.0,
        previous_generator_units=2,
        startup_history=[2],
        shutdown_history=[],
    )
    assert plan["generator_units_on"] == 1.0


def test_historical_start_locks_are_netted_to_current_online_units() -> None:
    plan = solve_causal_window(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [480.0, 480.0, 480.0],
        [0.0, 0.0, 0.0],
        _config(),
        soc_initial=0.65,
        previous_generator_kw=120.0,
        previous_generator_units=1,
        startup_history=[2],
        shutdown_history=[1],
    )
    assert plan["generator_units_on"] == 1.0
