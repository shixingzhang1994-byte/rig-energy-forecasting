#!/usr/bin/env python3
"""Create a compact reader-facing plot from the audited V19 seed table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper/experiments/results/v19_evidential_risk"


def main() -> None:
    with (RESULTS / "v19_seed_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    baseline = np.asarray([float(row["baseline_dispatch_macro_f1"]) for row in rows])
    evidential = np.asarray([float(row["evidential_dispatch_macro_f1"]) for row in rows])
    action_rate = np.asarray(
        [float(row["physical_action_changed_rows"]) / 1080.0 * 100.0 for row in rows]
    )
    seed_labels = [str(row["seed"])[-2:] for row in rows]

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    ax = axes[0]
    for before, after in zip(baseline, evidential, strict=True):
        ax.plot([0, 1], [before, after], color="#9aa0a6", linewidth=1.0, alpha=0.75)
        ax.scatter([0, 1], [before, after], color=["#6b7280", "#2474b5"], s=18, zorder=3)
    mean_before = float(baseline.mean())
    mean_after = float(evidential.mean())
    ax.plot([0, 1], [mean_before, mean_after], color="#b42318", linewidth=2.2, zorder=4)
    ax.scatter([0, 1], [mean_before, mean_after], color="#b42318", s=34, zorder=5, label="Mean")
    ax.set_xticks([0, 1], ["RSS", "E-RSS"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("Dispatch Macro-F1")
    ax.set_title("(a) Paired risk-recognition change")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.98, f"Mean: {mean_before:.2f} → {mean_after:.2f}",
            transform=ax.transAxes, va="top", ha="left")

    ax = axes[1]
    bars = ax.bar(seed_labels, action_rate, color="#2474b5", width=0.68)
    for bar, value in zip(bars, action_rate, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Holdout seed suffix")
    ax.set_ylabel("Changed physical actions (%)")
    ax.set_title("(b) Action change without shortage change")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.98, "Unserved-energy Δ = 0.00 kWh\non all 6 seeds",
            transform=ax.transAxes, va="top", ha="right",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1"})
    # Leave a dedicated annotation band above the tallest value label.
    ax.set_ylim(0.0, max(2.70, float(action_rate.max()) + 0.75))

    png = RESULTS / "v19_evidential_risk_audit.png"
    pdf = RESULTS / "v19_evidential_risk_audit.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
