"""Audit and compare the paper-only no-uncertainty-envelope run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path("./rig-energy-forecasting")
PAPER = Path("./paper/experiments")
OUTPUT = PAPER / "results/no_uncertainty_envelope_v15_seed_20261011"
ORIGINAL = PROJECT / "artifacts/v15_generator_first_reserve/seed_20261011"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    metrics_path = OUTPUT / "dispatch_metrics.csv"
    metrics = read_csv(metrics_path)
    trajectory_files = sorted(OUTPUT.glob("trajectory_*.csv"))
    max_balance_error = 0.0
    max_capacity_violation = 0.0
    max_soc_violation = 0.0
    simultaneous_count = 0
    max_integer_error = 0.0
    finite = True
    row_count = 0
    for path in trajectory_files:
        rows = read_csv(path)
        for row in rows:
            row_count += 1
            ignored = {
                "v15_generator_first_guard_active",
                "low_soc_commitment_guard_evaluated",
                "low_soc_commitment_guard_triggered",
                "low_soc_commitment_reachability_truncated",
                "generator_before_storage",
                "supervisory_protection_active",
                "selected_controller",
            }
            for key, value in row.items():
                if key in ignored or value in ("", None):
                    continue
                try:
                    finite = finite and bool(float(value) == float(value))
                except ValueError:
                    finite = False
            balance = float(row["load_kw"]) - (
                float(row["grid_kw"])
                + float(row["generator_kw"])
                + float(row["storage_kw"])
                + float(row["unserved_kw"])
            )
            max_balance_error = max(max_balance_error, abs(balance))
            max_capacity_violation = max(
                max_capacity_violation,
                max(0.0, float(row["grid_kw"]) - float(row["grid_available_capacity_kw"])),
                max(0.0, float(row["generator_kw"]) - float(row["generator_available_capacity_kw"])),
                max(0.0, abs(float(row["storage_kw"])) - float(row["storage_available_power_kw"])),
            )
            max_soc_violation = max(
                max_soc_violation,
                max(0.0, 0.15 - float(row["soc"])),
                max(0.0, float(row["soc"]) - 0.90),
            )
            simultaneous_count += int(
                float(row["charge_kw"]) > 1e-8
                and float(row["discharge_kw"]) > 1e-8
            )
            units = float(row["generator_units_on"])
            max_integer_error = max(max_integer_error, abs(units - round(units)))

    original_rows = {
        row["method"]: row
        for row in read_csv(ORIGINAL / "method_summary.csv")
    }
    new_rows = {
        row["method"]: row
        for row in metrics
        if row["scenario"] == "A_stable_drilling"
    }
    # Aggregate the three scenarios for the two controller variants.
    new_by_method: dict[str, dict[str, float]] = {}
    for method in ("Risk-SOC-Supervisory-MPC", "ML-Robust-MPC"):
        selected = [row for row in metrics if row["method"] == method]
        new_by_method[method] = {
            "unserved_energy_kwh": sum(float(row["unserved_energy_kwh"]) for row in selected),
            "risk_adjusted_cost_yuan": sum(float(row["risk_adjusted_cost_yuan"]) for row in selected),
            "diesel_fuel_l": sum(float(row["diesel_fuel_l"]) for row in selected),
            "max_decision_seconds": max(float(row["max_decision_seconds"]) for row in selected),
        }

    proposed = new_by_method["Risk-SOC-Supervisory-MPC"]
    original_proposed = {
        "unserved_energy_kwh": float(original_rows["Risk-SOC-Supervisory-MPC"]["unserved_energy_kwh"]),
        "risk_adjusted_cost_yuan": float(original_rows["Risk-SOC-Supervisory-MPC"]["risk_adjusted_cost_yuan"]),
        "diesel_fuel_l": float(original_rows["Risk-SOC-Supervisory-MPC"]["diesel_fuel_l"]),
        "max_decision_seconds": float(original_rows["Risk-SOC-Supervisory-MPC"]["max_decision_seconds"]),
    }
    delta = {key: proposed[key] - original_proposed[key] for key in proposed}
    audit = {
        "analysis_type": "single_seed_no_uncertainty_envelope_ablation",
        "seed": 20261011,
        "status": "completed_diagnostic_not_new_frozen_acceptance",
        "protocol_boundary": "paper_extension_using_existing_v15_seed_20261011_inputs",
        "trajectory_files": len(trajectory_files),
        "trajectory_rows": row_count,
        "physical_audit": {
            "all_numeric_values_finite": finite,
            "max_power_balance_error_kw": max_balance_error,
            "max_capacity_violation_kw": max_capacity_violation,
            "max_soc_violation": max_soc_violation,
            "simultaneous_charge_discharge_count": simultaneous_count,
            "max_integer_commitment_error": max_integer_error,
            "physical_checks_pass": bool(
                finite
                and max_balance_error <= 1e-8
                and max_capacity_violation <= 1e-8
                and max_soc_violation <= 1e-8
                and simultaneous_count == 0
                and max_integer_error <= 1e-8
            ),
        },
        "new_no_uncertainty_envelope_full_method": proposed,
        "original_v15_full_method_same_seed": original_proposed,
        "new_minus_original": delta,
        "source_hashes": {
            "new_config": sha256(PAPER / "configs/no_uncertainty_envelope_v15_seed_20261011.yaml"),
            "new_metrics": sha256(metrics_path),
            "original_method_summary": sha256(ORIGINAL / "method_summary.csv"),
        },
        "interpretation": "This single-seed diagnostic does not replace the two frozen V15 holdouts and does not prove that uncertainty envelopes improve every scenario.",
    }
    result = OUTPUT / "ablation_audit.json"
    result.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result)
    print(json.dumps(audit["physical_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
