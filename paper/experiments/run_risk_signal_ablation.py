#!/usr/bin/env python3
"""Run a clean risk-on/risk-off ablation for one frozen V15 seed.

Only the online risk signal consumed by the proposed controller is neutralized:
the predicted/dispatch risk level and reserve adder are set to zero.  Supply
regimes, capacity traces, scenario windows, forecasts, SOC supervision,
generator-first reserve logic, plant parameters, and solver settings remain
unchanged.  Original project artifacts are read-only inputs.
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


PROJECT = Path("./rig-energy-forecasting")


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
                    float(row["grid_kw"])
                    - float(row["grid_available_capacity_kw"]),
                    float(row["generator_kw"])
                    - float(row["generator_available_capacity_kw"]),
                    abs(float(row["storage_kw"]))
                    - float(row["storage_available_power_kw"]),
                ),
            )
            maximums["max_soc_violation"] = max(
                maximums["max_soc_violation"],
                max(0.0, 0.15 - float(row["soc"]), float(row["soc"]) - 0.90),
            )
            units = float(row["generator_units_on"])
            maximums["max_integer_commitment_error"] = max(
                maximums["max_integer_commitment_error"],
                abs(units - round(units)),
            )
            simultaneous += int(
                float(row["charge_kw"]) > 1e-8
                and float(row["discharge_kw"]) > 1e-8
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
    original_risk_path = source / "risk/risk_predictions.csv"
    neutral_risk_path = output / "risk_predictions_neutral.csv"
    risk = pd.read_csv(original_risk_path)
    neutralized_columns = []
    for column in ("predicted_risk_level", "dispatch_risk_level"):
        if column in risk.columns:
            risk[column] = 0
            neutralized_columns.append(column)
    if "reserve_adder_kw" in risk.columns:
        risk["reserve_adder_kw"] = 0.0
        neutralized_columns.append("reserve_adder_kw")
    if not neutralized_columns:
        raise RuntimeError("no risk columns were available to neutralize")
    risk.to_csv(neutral_risk_path, index=False)

    command = [
        sys.executable,
        str(PROJECT / "scripts/run_v15_dispatch_benchmark.py"),
        "--predictions",
        str(source / "source/forecast/forecast_predictions.npz"),
        "--prediction-keys",
        str(source / "source/forecast/forecast_prediction_keys.json"),
        "--config",
        str(source / "resolved_dispatch_config.yaml"),
        "--artifact-dir",
        str(output / "dispatch"),
        "--risk-signals",
        str(neutral_risk_path),
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)

    original_selection_path = source / "dispatch/scenario_selection.json"
    ablated_selection_path = output / "dispatch/scenario_selection.json"
    original_selection = json.loads(original_selection_path.read_text(encoding="utf-8"))
    ablated_selection = json.loads(ablated_selection_path.read_text(encoding="utf-8"))
    selection_match = all(
        int(original_selection["scenarios"][name][key])
        == int(ablated_selection["scenarios"][name][key])
        for name in original_selection["scenarios"]
        for key in ("start_index", "stop_index", "warmup_steps")
    )
    if not selection_match:
        raise RuntimeError("risk ablation changed the selected scenario windows")

    new_metrics_path = output / "dispatch/dispatch_metrics.csv"
    new_metrics = pd.read_csv(new_metrics_path)
    original_summary = pd.read_csv(source / "method_summary.csv").set_index("method")
    risk_off = aggregate_method(new_metrics, "Risk-SOC-Supervisory-MPC")
    risk_on = {
        key: float(original_summary.loc["Risk-SOC-Supervisory-MPC", key])
        for key in risk_off
    }
    delta = {key: risk_on[key] - risk_off[key] for key in risk_on}
    trajectories = sorted(
        (output / "dispatch").glob(
            "trajectory_*_risk_soc_supervisory_mpc.csv"
        )
    )
    audit = audit_trajectories(trajectories)
    payload = {
        "analysis_type": "orthogonal_risk_signal_ablation",
        "seed": source.name,
        "status": "paper_extension_not_original_v15_acceptance",
        "neutralized_columns": neutralized_columns,
        "held_constant": [
            "supply regime and dynamic capacity traces",
            "selected scenario windows and shared warm-up",
            "forecast members and uncertainty envelope",
            "SOC thresholds and hysteresis",
            "generator-first and grid-headroom reserve actions",
            "plant, cost, and solver configuration",
        ],
        "scenario_selection_match": selection_match,
        "risk_on": risk_on,
        "risk_off": risk_off,
        "risk_on_minus_risk_off": delta,
        "physical_audit": audit,
        "source_hashes": {
            "original_risk": sha256(original_risk_path),
            "neutral_risk": sha256(neutral_risk_path),
            "config": sha256(source / "resolved_dispatch_config.yaml"),
            "original_selection": sha256(original_selection_path),
            "ablated_selection": sha256(ablated_selection_path),
            "new_metrics": sha256(new_metrics_path),
        },
        "interpretation": "Paired mechanism diagnostic. A positive unserved-energy delta means the risk-on controller reduced shortage relative to the otherwise identical risk-off controller.",
        "reproduction_command": " ".join(command),
    }
    result_path = output / "risk_ablation_result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(result_path), "delta": delta, "audit": audit}))


if __name__ == "__main__":
    main()
