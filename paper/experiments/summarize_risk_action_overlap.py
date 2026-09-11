#!/usr/bin/env python3
"""Quantify whether the risk channel changes supervisory actions or outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SCENARIOS = ("A_stable_drilling", "B_connection_transition", "C_tripping_impact")
SUFFIX = "risk_soc_supervisory_mpc"
ACTION_COLUMNS = (
    "grid_kw",
    "generator_kw",
    "storage_kw",
    "unserved_kw",
    "generator_units_on",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, action="append", required=True)
    parser.add_argument("--risk-ablation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    windows = []
    totals = {
        "steps": 0,
        "high_risk": 0,
        "soc_low": 0,
        "risk_only": 0,
        "action_changed": 0,
        "mode_changed": 0,
    }
    for root in args.seed_root:
        seed = root.name.removeprefix("seed_")
        for scenario in SCENARIOS:
            original = read(root / f"dispatch/trajectory_{scenario}_{SUFFIX}.csv")
            ablated = read(
                args.risk_ablation_root
                / f"seed_{seed}/dispatch/trajectory_{scenario}_{SUFFIX}.csv"
            )
            if len(original) != len(ablated):
                raise RuntimeError(f"trajectory length mismatch: {seed}/{scenario}")
            counts = {
                "high_risk": 0,
                "soc_low": 0,
                "risk_only": 0,
                "action_changed": 0,
                "mode_changed": 0,
            }
            for on, off in zip(original, ablated):
                risk_trigger = int(float(on["risk_level"])) >= 2
                soc_trigger = float(on["soc"]) <= 0.30 + 1e-12
                counts["high_risk"] += int(risk_trigger)
                counts["soc_low"] += int(soc_trigger)
                counts["risk_only"] += int(risk_trigger and not soc_trigger)
                counts["action_changed"] += int(
                    any(
                        abs(float(on[column]) - float(off[column])) > 1e-8
                        for column in ACTION_COLUMNS
                    )
                )
                counts["mode_changed"] += int(
                    on["supervisory_protection_active"]
                    != off["supervisory_protection_active"]
                )
            steps = len(original)
            windows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "steps": steps,
                    **{f"{key}_fraction": value / steps for key, value in counts.items()},
                }
            )
            totals["steps"] += steps
            for key, value in counts.items():
                totals[key] += value

    payload = {
        "analysis_type": "risk_trigger_action_overlap",
        "window_count": len(windows),
        "total_steps": totals["steps"],
        "per_window": windows,
        "aggregate": {
            f"{key}_fraction": value / totals["steps"]
            for key, value in totals.items()
            if key != "steps"
        },
        "interpretation": "A nonzero mode-change fraction with a zero action-change fraction means the risk channel changes the internal supervisory label but is behaviorally masked by the constrained optimizer and reserve layer on these windows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.figure is not None:
        labels = ["High-risk\nflag", "SOC-low\nflag", "Mode\nchanged", "Action\nchanged"]
        values = [
            100.0 * payload["aggregate"]["high_risk_fraction"],
            100.0 * payload["aggregate"]["soc_low_fraction"],
            100.0 * payload["aggregate"]["mode_changed_fraction"],
            100.0 * payload["aggregate"]["action_changed_fraction"],
        ]
        colors = ["#7f8c8d", "#4c78a8", "#f2b134", "#d95f02"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        bars = ax.bar(labels, values, color=colors, width=0.66)
        ax.set_ylabel("Share of control steps (%)")
        ax.set_ylim(0.0, 85.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title("Risk trigger and realized supervisory response")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.2,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"output": str(args.output), "aggregate": payload["aggregate"]}))


if __name__ == "__main__":
    main()
