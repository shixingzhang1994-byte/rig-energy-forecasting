#!/usr/bin/env python3
"""Reconstruct the exact shared Rule-Based warm-up state for paper baselines.

The frozen benchmark stored SOC, generator power, and committed units at the
comparison boundary, but not the recent startup/shutdown vectors needed by
minimum up/down constraints.  This script deterministically replays only the
declared shared warm-up and exports those vectors.  It does not rerun or alter
the scored V15 controller trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("./rig-energy-forecasting")
sys.path.insert(0, str(PROJECT / "src"))

from rig_energy.optimization import dispatch as dispatch_module  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    dispatch_dir = root / "dispatch"
    forecast_dir = root / "source/forecast"
    config_path = root / "resolved_dispatch_config.yaml"
    risk_path = root / "risk/risk_predictions.csv"
    selection_path = dispatch_dir / "scenario_selection.json"

    config, plant = dispatch_module._load_config(config_path)
    arrays = np.load(forecast_dir / "forecast_predictions.npz")
    key_map = json.loads(
        (forecast_dir / "forecast_prediction_keys.json").read_text(encoding="utf-8")
    )
    name_to_key = {name: key for key, name in key_map.items()}
    risk = pd.read_csv(risk_path)
    source_indices = risk["sample_index"].to_numpy(dtype=int)
    y_true = np.asarray(arrays["y_true"], dtype=float)[source_indices, 0]
    forecast_model = str(config["robust_mpc"]["forecast_model"])
    ml_forecast = np.asarray(arrays[name_to_key[forecast_model]], dtype=float)[
        source_indices
    ]
    persistence = np.asarray(arrays[name_to_key["Persistence"]], dtype=float)[
        source_indices
    ]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    reconstructed: dict[str, object] = {}
    for scenario, metadata in selection["scenarios"].items():
        warmup = int(metadata["warmup_steps"])
        positions = np.flatnonzero(source_indices == int(metadata["start_index"]))
        if len(positions) != 1:
            raise ValueError(f"cannot locate unique source start for {scenario}")
        scored_start = int(positions[0])
        execution_start = scored_start - warmup
        if execution_start < 0:
            raise ValueError(f"warm-up begins before available data for {scenario}")

        warmup_slice = slice(execution_start, scored_start)
        initial_soc = float(risk["storage_soc_pct"].iloc[execution_start]) / 100.0
        trajectory, _ = dispatch_module._simulate_method(
            "Rule-Based",
            y_true[warmup_slice],
            ml_forecast[warmup_slice],
            persistence[warmup_slice],
            risk["grid_available_capacity_kw"].to_numpy(dtype=float)[warmup_slice],
            plant,
            config["robust_mpc"],
            generator_available_kw=risk[
                "generator_available_capacity_kw"
            ].to_numpy(dtype=float)[warmup_slice],
            storage_available_kw=risk[
                "storage_available_discharge_power_kw"
            ].to_numpy(dtype=float)[warmup_slice],
            initial_soc=initial_soc,
        )
        startup_keep = max(0, plant.generator_min_up_steps - 1)
        shutdown_keep = max(0, plant.generator_min_down_steps - 1)
        state = {
            "warmup_steps": warmup,
            "policy": "shared_rule_based_state_initialization",
            "comparison_initial_soc_pct": float(trajectory["soc"].iloc[-1]) * 100.0,
            "comparison_initial_generator_kw": float(
                trajectory["generator_kw"].iloc[-1]
            ),
            "comparison_initial_generator_units": int(
                trajectory["generator_units_on"].iloc[-1]
            ),
            "startup_history": (
                trajectory["generator_startup_units"]
                .tail(startup_keep)
                .astype(int)
                .tolist()
                if startup_keep
                else []
            ),
            "shutdown_history": (
                trajectory["generator_shutdown_units"]
                .tail(shutdown_keep)
                .astype(int)
                .tolist()
                if shutdown_keep
                else []
            ),
        }
        checks = {
            "soc_matches_frozen_metadata": bool(
                np.isclose(
                    state["comparison_initial_soc_pct"],
                    float(metadata["comparison_initial_soc_pct"]),
                    atol=1e-9,
                )
            ),
            "generator_kw_matches_frozen_metadata": bool(
                np.isclose(
                    state["comparison_initial_generator_kw"],
                    float(metadata["comparison_initial_generator_kw"]),
                    atol=1e-9,
                )
            ),
            "units_match_frozen_metadata": bool(
                state["comparison_initial_generator_units"]
                == int(metadata["comparison_initial_generator_units"])
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"warm-up reconstruction mismatch for {scenario}: {checks}")
        reconstructed[scenario] = {**state, "checks": checks}

    payload = {
        "seed": root.name,
        "purpose": "exact shared pre-window transition histories for independent paper baselines",
        "scenarios": reconstructed,
        "source_hashes": {
            "config": sha256(config_path),
            "risk_signals": sha256(risk_path),
            "forecast_predictions": sha256(
                forecast_dir / "forecast_predictions.npz"
            ),
            "scenario_selection": sha256(selection_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "scenarios": list(reconstructed)}))


if __name__ == "__main__":
    main()
