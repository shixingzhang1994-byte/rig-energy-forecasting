"""Summarize the existing V16 emergency-load allocation comparison for the paper."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path("./rig-energy-forecasting")
PAPER = Path("./paper/experiments")
SOURCE = PROJECT / "artifacts/v16_expert_remediation_surrogate/emergency_load_metrics.csv"
OUTPUT = PAPER / "results/emergency_priority_ablation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in source_rows
        if row["method"] in {"proportional-curtailment", "priority-greedy"}
    ]
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 rows, found {len(rows)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "method",
        "total_unserved_energy_kwh",
        "critical_unserved_energy_kwh",
        "essential_unserved_energy_kwh",
        "interruptible_unserved_energy_kwh",
        "critical_served_fraction",
        "highs_equivalence_max_kw",
    ]
    selected = [{key: row[key] for key in fields} for row in rows]
    csv_path = OUTPUT / "emergency_priority_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    scenarios = [
        "base_v15_normalized",
        "grid_derate_25pct",
        "one_generator_unavailable",
        "storage_derate_350kw",
        "joint_extreme",
    ]
    labels = {
        "base_v15_normalized": "Base",
        "grid_derate_25pct": "Grid -25%",
        "one_generator_unavailable": "One genset unavailable",
        "storage_derate_350kw": "Storage 350 kW",
        "joint_extreme": "Joint stress",
    }
    by_key = {(row["scenario"], row["method"]): row for row in selected}
    critical_reductions = {
        scenario: float(by_key[(scenario, "proportional-curtailment")]["critical_unserved_energy_kwh"])
        - float(by_key[(scenario, "priority-greedy")]["critical_unserved_energy_kwh"])
        for scenario in scenarios
    }
    summary = {
        "analysis_type": "paper_side_existing_v16_emergency_priority_ablation",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "data_claim": "mechanistic_synthetic_v15_aligned",
        "status": "diagnostic_not_field_protection_validation",
        "methods": ["proportional-curtailment", "priority-greedy"],
        "scenarios": scenarios,
        "critical_unserved_reduction_kwh_proportional_minus_priority": critical_reductions,
        "highs_equivalence_max_kw": max(
            float(row["highs_equivalence_max_kw"]) for row in selected
        ),
        "approval_boundary": "load priority fractions remain NOT_OPERATIONALLY_APPROVED pending drilling/electrical/HSE review",
    }
    json_path = OUTPUT / "emergency_priority_ablation_manifest.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = list(range(len(scenarios)))
    width = 0.36
    for axis, metric, title, ylabel in (
        (axes[0], "critical_unserved_energy_kwh", "Critical-load unserved energy", "kWh"),
        (axes[1], "total_unserved_energy_kwh", "Total unserved energy", "kWh"),
    ):
        proportional = [
            float(by_key[(scenario, "proportional-curtailment")][metric])
            for scenario in scenarios
        ]
        priority = [
            float(by_key[(scenario, "priority-greedy")][metric])
            for scenario in scenarios
        ]
        axis.bar([item - width / 2 for item in x], proportional, width, label="Proportional curtailment", color="#DC2626")
        axis.bar([item + width / 2 for item in x], priority, width, label="Priority protection", color="#059669")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [labels[scenario] for scenario in scenarios], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Emergency load-priority ablation under stress scenarios")
    plot_path = OUTPUT / "emergency_priority_ablation.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(csv_path)
    print(json_path)
    print(plot_path)


if __name__ == "__main__":
    main()
