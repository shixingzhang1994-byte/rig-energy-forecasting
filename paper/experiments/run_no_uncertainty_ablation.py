"""Run a paired no-forecast-uncertainty ablation on one frozen V15 seed.

The original V15 artifacts are read-only inputs. This script creates a new
configuration and output directory under paper/experiments. The ablation keeps
the risk/SOC supervisor and dispatch protocol, but removes the forecast
envelope in both places where it enters the controller:

* the robust protection path has zero fixed reserve and zero disagreement term;
* all scenario-CVaR members are the same causal point forecast.

This isolates scenario/envelope diversity, not the complete emergency-load
allocation module, which is a separate V16 diagnostic path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT = Path("./rig-energy-forecasting")
PAPER = Path("./paper/experiments")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_method(metrics: pd.DataFrame, method: str) -> dict[str, float]:
    rows = metrics[metrics["method"] == method]
    if len(rows) != 3:
        raise RuntimeError(f"expected three scenarios for {method}, got {len(rows)}")
    return {
        "unserved_energy_kwh": float(rows["unserved_energy_kwh"].sum()),
        "risk_adjusted_cost_yuan": float(rows["risk_adjusted_cost_yuan"].sum()),
        "diesel_fuel_l": float(rows["diesel_fuel_l"].sum()),
        "generator_startups": float(rows["generator_startups"].sum()),
        "max_decision_seconds": float(rows["max_decision_seconds"].max()),
    }


def audit_trajectories(paths: list[Path]) -> dict[str, object]:
    maximums = {
        "max_power_balance_error_kw": 0.0,
        "max_capacity_violation_kw": 0.0,
        "max_soc_violation": 0.0,
        "max_integer_commitment_error": 0.0,
    }
    simultaneous = 0
    rows_seen = 0
    all_finite = True
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            rows_seen += 1
            numeric = []
            for value in row.values():
                if value in (None, "", "True", "False"):
                    continue
                try:
                    numeric.append(float(value))
                except ValueError:
                    continue
            all_finite = all_finite and bool(np.isfinite(numeric).all())
            balance = float(row["load_kw"]) - (
                float(row["grid_kw"])
                + float(row["generator_kw"])
                + float(row["storage_kw"])
                + float(row["unserved_kw"])
            )
            maximums["max_power_balance_error_kw"] = max(
                maximums["max_power_balance_error_kw"], abs(balance)
            )
            maximums["max_capacity_violation_kw"] = max(
                maximums["max_capacity_violation_kw"],
                max(
                    0.0,
                    float(row["grid_kw"]) - float(row["grid_available_capacity_kw"]),
                    float(row["generator_kw"]) - float(row["generator_available_capacity_kw"]),
                    abs(float(row["storage_kw"])) - float(row["storage_available_power_kw"]),
                ),
            )
            maximums["max_soc_violation"] = max(
                maximums["max_soc_violation"],
                max(0.0, 0.15 - float(row["soc"]), float(row["soc"]) - 0.90),
            )
            units = float(row["generator_units_on"])
            maximums["max_integer_commitment_error"] = max(
                maximums["max_integer_commitment_error"], abs(units - round(units))
            )
            simultaneous += int(
                float(row["charge_kw"]) > 1e-8 and float(row["discharge_kw"]) > 1e-8
            )
    passed = bool(
        all_finite
        and all(value <= 1e-8 for value in maximums.values())
        and simultaneous == 0
    )
    return {
        **maximums,
        "trajectory_count": len(paths),
        "row_count": rows_seen,
        "simultaneous_charge_discharge_count": simultaneous,
        "all_numeric_values_finite": all_finite,
        "physical_checks_pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifact_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(source.name.removeprefix("seed_"))
    source_config = source / "resolved_dispatch_config.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["robust_mpc"]["fixed_reserve_kw"] = 0.0
    config["robust_mpc"]["forecast_disagreement_weight"] = 0.0
    point_forecast = str(config["robust_mpc"]["forecast_model"])
    config["scenario_cvar_mpc"]["forecast_members"] = [point_forecast] * 7
    config.setdefault("paper_ablation", {})
    config["paper_ablation"].update(
        {
            "name": "no_forecast_uncertainty_envelope",
            "interpretation": "same causal point forecast in robust protection and all CVaR scenario members",
            "source_seed": seed,
            "source_config": str(source_config),
            "not_disabled": [
                "risk level signal",
                "SOC supervisory switching",
                "generator-first execution wrapper",
                "emergency priority-load diagnostic module",
            ],
        }
    )
    config_path = output / "no_uncertainty_envelope_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(PROJECT / "scripts/run_v15_dispatch_benchmark.py"),
        "--predictions",
        str(source / "source/forecast/forecast_predictions.npz"),
        "--prediction-keys",
        str(source / "source/forecast/forecast_prediction_keys.json"),
        "--config",
        str(config_path),
        "--artifact-dir",
        str(output / "dispatch"),
        "--risk-signals",
        str(source / "risk/risk_predictions.csv"),
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)

    original_selection = json.loads((source / "dispatch/scenario_selection.json").read_text(encoding="utf-8"))
    ablated_selection = json.loads((output / "dispatch/scenario_selection.json").read_text(encoding="utf-8"))
    selection_match = all(
        int(original_selection["scenarios"][name][key])
        == int(ablated_selection["scenarios"][name][key])
        for name in original_selection["scenarios"]
        for key in ("start_index", "stop_index", "warmup_steps")
    )
    if not selection_match:
        raise RuntimeError("uncertainty ablation changed selected scenario windows")

    new_metrics_path = output / "dispatch/dispatch_metrics.csv"
    new_metrics = pd.read_csv(new_metrics_path)
    original_summary = pd.read_csv(source / "method_summary.csv").set_index("method")
    envelope_off = aggregate_method(new_metrics, "Risk-SOC-Supervisory-MPC")
    envelope_on = {key: float(original_summary.loc["Risk-SOC-Supervisory-MPC", key]) for key in envelope_off}
    delta = {key: envelope_on[key] - envelope_off[key] for key in envelope_on}
    trajectories = sorted((output / "dispatch").glob("trajectory_*.csv"))
    audit = audit_trajectories(trajectories)
    payload = {
        "analysis_type": "orthogonal_forecast_uncertainty_envelope_ablation",
        "seed": seed,
        "status": "preregistered_paper_extension_not_original_v15_acceptance",
        "held_constant": [
            "causal point forecast",
            "online risk signal and SOC supervisory switching",
            "generator-first and grid-headroom reserve actions",
            "supply regime, scenario windows, plant, costs, and solver settings",
        ],
        "intervention": [
            "set robust fixed reserve and disagreement weight to zero",
            "replace all seven CVaR members by the same causal point forecast",
        ],
        "scenario_selection_match": selection_match,
        "envelope_on": envelope_on,
        "envelope_off": envelope_off,
        "envelope_on_minus_off": delta,
        "physical_audit": audit,
        "source_hashes": {
            "source_config": sha256(source_config),
            "ablation_config": sha256(config_path),
            "original_selection": sha256(source / "dispatch/scenario_selection.json"),
            "ablated_selection": sha256(output / "dispatch/scenario_selection.json"),
            "new_metrics": sha256(new_metrics_path),
        },
        "interpretation": "Paired mechanism intervention. A negative unserved-energy delta means the uncertainty envelope reduced shortage relative to the otherwise identical point-forecast controller.",
        "reproduction_command": " ".join(command),
    }
    result_path = output / "uncertainty_ablation_result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(result_path), "delta": delta, "audit": audit}))


if __name__ == "__main__":
    main()
