"""Create a paper-side ablation summary from the two frozen V15 holdouts.

This script does not rerun or modify the project artifacts. It only aggregates
the already executed, same-protocol method rows and labels the available
mechanism comparisons conservatively.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path("./rig-energy-forecasting")
PAPER = Path("./paper/experiments")
SOURCE_ROOT = PROJECT / "artifacts/v15_generator_first_reserve"
SEEDS = (20261011, 20261012)

METHODS = {
    "ML-Robust-MPC": "robust baseline without scenario CVaR or SOC supervisory switching",
    "Scenario-CVaR-MPC": "static scenario-CVaR without risk adaptation or SOC supervisory switching",
    "Risk-Adaptive-CVaR-MPC": "risk-adaptive scenario-CVaR without SOC supervisory switching",
    "Risk-SOC-Supervisory-MPC": "complete risk-adaptive CVaR plus SOC supervisory controller",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for seed in SEEDS:
        source = SOURCE_ROOT / f"seed_{seed}/method_summary.csv"
        sources.append({"path": str(source), "sha256": sha256(source)})
        with source.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["method"] not in METHODS:
                    continue
                row["seed"] = str(seed)
                rows.append(row)
    return rows, sources


def main() -> None:
    rows, sources = read_rows()
    summary: list[dict[str, object]] = []
    for method, interpretation in METHODS.items():
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(SEEDS):
            raise RuntimeError(f"expected one row per holdout for {method}")
        summary.append(
            {
                "method": method,
                "ablation_interpretation": interpretation,
                "holdout_seeds": ";".join(str(seed) for seed in SEEDS),
                "unserved_energy_kwh_sum": round(
                    sum(float(row["unserved_energy_kwh"]) for row in selected), 6
                ),
                "risk_adjusted_cost_yuan_sum": round(
                    sum(float(row["risk_adjusted_cost_yuan"]) for row in selected), 6
                ),
                "diesel_fuel_l_sum": round(
                    sum(float(row["diesel_fuel_l"]) for row in selected), 6
                ),
                "generator_startups_sum": sum(
                    int(row["generator_startups"]) for row in selected
                ),
                "max_decision_seconds": round(
                    max(float(row["max_decision_seconds"]) for row in selected), 6
                ),
                "source_status": "existing_frozen_v15_holdout_rows",
            }
        )

    PAPER.mkdir(parents=True, exist_ok=True)
    result_csv = PAPER / "results/ablation_summary.csv"
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0])
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    result_json = PAPER / "results/ablation_summary_manifest.json"
    result_json.write_text(
        json.dumps(
            {
                "analysis_type": "paper_side_frozen_holdout_ablation_summary",
                "protocol": "V15 generator-first reserve",
                "seeds": list(SEEDS),
                "sources": sources,
                "method_interpretations": METHODS,
                "not_isolated_by_this_summary": [
                    "emergency protection fully disabled",
                    "forecast uncertainty envelope fully disabled",
                    "independent full MILP reimplementation",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    labels = {
        "ML-Robust-MPC": "ML-Robust",
        "Scenario-CVaR-MPC": "Static CVaR",
        "Risk-Adaptive-CVaR-MPC": "Adaptive CVaR",
        "Risk-SOC-Supervisory-MPC": "Full method",
    }
    plot_rows = list(reversed(summary))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    y = list(range(len(plot_rows)))
    names = [labels[str(row["method"])] for row in plot_rows]
    axes[0].barh(y, [float(row["unserved_energy_kwh_sum"]) for row in plot_rows], color="#2563EB")
    axes[1].barh(y, [float(row["risk_adjusted_cost_yuan_sum"]) for row in plot_rows], color="#059669")
    axes[0].set_title("Unserved energy")
    axes[0].set_xlabel("kWh; sum of two frozen holdouts")
    axes[1].set_title("Risk-adjusted cost")
    axes[1].set_xlabel("yuan; sum of two frozen holdouts")
    for ax in axes:
        ax.set_yticks(y, names)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Mechanism comparison from frozen V15 holdouts")
    plot_path = PAPER / "results/ablation_summary.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(result_csv)
    print(result_json)
    print(plot_path)


if __name__ == "__main__":
    main()
