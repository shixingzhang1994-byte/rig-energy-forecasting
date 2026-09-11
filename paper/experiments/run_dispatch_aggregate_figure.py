"""Create a clean aggregate dispatch figure from the two frozen V15 holdouts."""

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
METHODS = ("Risk-SOC-Supervisory-MPC", "ML-Robust-MPC", "Rule-Based")
LABELS = {
    "Risk-SOC-Supervisory-MPC": "RSS",
    "ML-Robust-MPC": "MLR",
    "Rule-Based": "RB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = []
    rows: dict[str, list[dict[str, str]]] = {method: [] for method in METHODS}
    for seed in SEEDS:
        path = SOURCE_ROOT / f"seed_{seed}/method_summary.csv"
        sources.append({"path": str(path), "sha256": sha256(path)})
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["method"] in rows:
                    rows[row["method"]].append(row)
    if any(len(values) != len(SEEDS) for values in rows.values()):
        raise RuntimeError("each plotted method must have one row per frozen holdout")

    values = {
        "unserved_energy_kwh": [sum(float(r["unserved_energy_kwh"]) for r in rows[m]) for m in METHODS],
        "risk_adjusted_cost_yuan": [sum(float(r["risk_adjusted_cost_yuan"]) for r in rows[m]) for m in METHODS],
        "diesel_fuel_l": [sum(float(r["diesel_fuel_l"]) for r in rows[m]) for m in METHODS],
    }
    labels = [LABELS[m] for m in METHODS]
    colors = ["#7C3AED", "#2563EB", "#6B7280"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    panels = [
        ("unserved_energy_kwh", "Unserved energy", "kWh"),
        ("risk_adjusted_cost_yuan", "Risk-adjusted cost", "yuan"),
        ("diesel_fuel_l", "Diesel fuel", "L"),
    ]
    for ax, (key, title, unit) in zip(axes, panels):
        bars = ax.bar(labels, values[key], color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(unit + " (sum of two frozen holdouts)")
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values[key]):
            fmt = f"{value:,.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Aggregate dispatch comparison on frozen V15 holdouts")
    out_dir = PAPER / "results/dispatch_aggregate_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "dispatch_aggregate_comparison.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    manifest = {
        "analysis_type": "paper_side_frozen_v15_dispatch_aggregate_figure",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "aggregation": "sum across the two frozen holdout method_summary.csv rows",
        "sources": sources,
        "output": str(plot_path),
        "claim_boundary": "descriptive simulation result; no field validation or statistical significance claim",
    }
    manifest_path = out_dir / "dispatch_aggregate_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(plot_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
