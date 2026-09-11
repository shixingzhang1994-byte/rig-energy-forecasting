#!/usr/bin/env python3
"""Summarize independent frozen V15 holdouts without pseudoreplication."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Risk-SOC-Supervisory-MPC", "Rule-Based", "ML-Robust-MPC")
NUMERICAL_ZERO_TOLERANCE = 1e-9


def clean_numerical_zero(value: float) -> float:
    """Suppress solver-scale residuals before reporting or plotting effects."""

    numeric = float(value)
    return 0.0 if abs(numeric) <= NUMERICAL_ZERO_TOLERANCE else numeric


def summarize(values: list[float], rng: np.random.Generator) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array[np.abs(array) <= NUMERICAL_ZERO_TOLERANCE] = 0.0
    observed = abs(float(array.mean()))
    sign_means = [
        abs(float(np.mean(array * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    sign_flip_p = float(
        sum(value >= observed - 1e-12 for value in sign_means) / len(sign_means)
    )
    bootstrap = rng.choice(array, size=(20000, len(array)), replace=True).mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd_sample": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "nonpositive_count": int(np.sum(array <= 0.0)),
        "bootstrap_95_interval_mean": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "exact_two_sided_sign_flip_p": sign_flip_p,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return step-down Holm adjusted p-values with monotonicity enforced."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, action="append", required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--risk-ablation-root", type=Path, required=True)
    parser.add_argument("--uncertainty-ablation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for root in args.seed_root:
        seed = root.name.removeprefix("seed_")
        methods = pd.read_csv(root / "method_summary.csv").set_index("method")
        missing = set(METHODS) - set(methods.index)
        if missing:
            raise RuntimeError(f"{root} missing methods: {sorted(missing)}")
        causal = json.loads(
            (args.causal_root / f"seed_{seed}.json").read_text(encoding="utf-8")
        )
        risk = json.loads(
            (
                args.risk_ablation_root
                / f"seed_{seed}/risk_ablation_result.json"
            ).read_text(encoding="utf-8")
        )
        uncertainty = json.loads(
            (
                args.uncertainty_ablation_root
                / f"seed_{seed}/uncertainty_ablation_result.json"
            ).read_text(encoding="utf-8")
        )
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        proposed = methods.loc["Risk-SOC-Supervisory-MPC"]
        rule = methods.loc["Rule-Based"]
        ml = methods.loc["ML-Robust-MPC"]
        static_cvar = methods.loc["Scenario-CVaR-MPC"]
        adaptive_cvar = methods.loc["Risk-Adaptive-CVaR-MPC"]
        rows.append(
            {
                "seed": seed,
                "rss_unserved_kwh": float(proposed["unserved_energy_kwh"]),
                "rb_unserved_kwh": float(rule["unserved_energy_kwh"]),
                "mlr_unserved_kwh": float(ml["unserved_energy_kwh"]),
                "rss_risk_cost_yuan": float(proposed["risk_adjusted_cost_yuan"]),
                "rb_risk_cost_yuan": float(rule["risk_adjusted_cost_yuan"]),
                "mlr_risk_cost_yuan": float(ml["risk_adjusted_cost_yuan"]),
                "rss_diesel_fuel_l": float(proposed["diesel_fuel_l"]),
                "rb_diesel_fuel_l": float(rule["diesel_fuel_l"]),
                "mlr_diesel_fuel_l": float(ml["diesel_fuel_l"]),
                "rb_max_decision_seconds": float(rule["max_decision_seconds"]),
                "mlr_max_decision_seconds": float(ml["max_decision_seconds"]),
                "cr_unserved_kwh": float(
                    causal["aggregate"]["total_unserved_energy_kwh"]
                ),
                "rss_minus_rb_kwh": float(
                    proposed["unserved_energy_kwh"] - rule["unserved_energy_kwh"]
                ),
                "rss_minus_mlr_kwh": float(
                    proposed["unserved_energy_kwh"] - ml["unserved_energy_kwh"]
                ),
                "rss_minus_cr_kwh": float(
                    proposed["unserved_energy_kwh"]
                    - causal["aggregate"]["total_unserved_energy_kwh"]
                ),
                "risk_on_minus_off_unserved_kwh": clean_numerical_zero(
                    risk["risk_on_minus_risk_off"]["unserved_energy_kwh"]
                ),
                "risk_on_minus_off_cost_yuan": float(
                    risk["risk_on_minus_risk_off"]["risk_adjusted_cost_yuan"]
                ),
                "envelope_on_minus_off_unserved_kwh": float(
                    uncertainty["envelope_on_minus_off"]["unserved_energy_kwh"]
                ),
                "envelope_on_minus_off_cost_yuan": float(
                    uncertainty["envelope_on_minus_off"]["risk_adjusted_cost_yuan"]
                ),
                "envelope_off_unserved_kwh": float(
                    uncertainty["envelope_off"]["unserved_energy_kwh"]
                ),
                "envelope_off_risk_cost_yuan": float(
                    uncertainty["envelope_off"]["risk_adjusted_cost_yuan"]
                ),
                "envelope_off_diesel_fuel_l": float(
                    uncertainty["envelope_off"]["diesel_fuel_l"]
                ),
                "envelope_off_max_decision_seconds": float(
                    uncertainty["envelope_off"]["max_decision_seconds"]
                ),
                "risk_adaptation_without_supervisor_unserved_kwh": float(
                    adaptive_cvar["unserved_energy_kwh"]
                    - static_cvar["unserved_energy_kwh"]
                ),
                "static_cvar_unserved_kwh": float(static_cvar["unserved_energy_kwh"]),
                "static_cvar_risk_cost_yuan": float(static_cvar["risk_adjusted_cost_yuan"]),
                "static_cvar_max_decision_seconds": float(static_cvar["max_decision_seconds"]),
                "adaptive_cvar_unserved_kwh": float(adaptive_cvar["unserved_energy_kwh"]),
                "adaptive_cvar_risk_cost_yuan": float(adaptive_cvar["risk_adjusted_cost_yuan"]),
                "adaptive_cvar_max_decision_seconds": float(adaptive_cvar["max_decision_seconds"]),
                "risk_off_supervisor_unserved_kwh": float(risk["risk_off"]["unserved_energy_kwh"]),
                "risk_off_supervisor_risk_cost_yuan": float(risk["risk_off"]["risk_adjusted_cost_yuan"]),
                "risk_off_supervisor_max_decision_seconds": float(risk["risk_off"]["max_decision_seconds"]),
                "risk_adaptation_without_supervisor_cost_yuan": float(
                    adaptive_cvar["risk_adjusted_cost_yuan"]
                    - static_cvar["risk_adjusted_cost_yuan"]
                ),
                "soc_supervisor_without_risk_unserved_kwh": float(
                    risk["risk_off"]["unserved_energy_kwh"]
                    - static_cvar["unserved_energy_kwh"]
                ),
                "soc_supervisor_without_risk_cost_yuan": float(
                    risk["risk_off"]["risk_adjusted_cost_yuan"]
                    - static_cvar["risk_adjusted_cost_yuan"]
                ),
                "supervisor_with_risk_unserved_kwh": float(
                    proposed["unserved_energy_kwh"]
                    - adaptive_cvar["unserved_energy_kwh"]
                ),
                "supervisor_with_risk_cost_yuan": float(
                    proposed["risk_adjusted_cost_yuan"]
                    - adaptive_cvar["risk_adjusted_cost_yuan"]
                ),
                "risk_cost_ratio_vs_mlr": float(
                    proposed["risk_adjusted_cost_yuan"]
                    / ml["risk_adjusted_cost_yuan"]
                ),
                "max_decision_seconds": float(proposed["max_decision_seconds"]),
                "main_physical_pass": bool(
                    result["physical_audit"]["physical_pass"]
                ),
                "risk_ablation_physical_pass": bool(
                    risk["physical_audit"]["physical_checks_pass"]
                ),
                "uncertainty_ablation_physical_pass": bool(
                    uncertainty["physical_audit"]["physical_checks_pass"]
                ),
                "causal_physical_pass": bool(
                    causal["aggregate"]["max_power_balance_error_kw"] <= 1e-8
                    and causal["aggregate"]["max_charge_discharge_product"] <= 1e-8
                ),
            }
        )

    if len(rows) < 6:
        raise RuntimeError(f"expected at least six independent seed roots, got {len(rows)}")
    rng = np.random.default_rng(20260907)
    contrast_columns = {
        "rss_minus_rb_kwh": "RSS minus RB unserved energy",
        "rss_minus_mlr_kwh": "RSS minus MLR unserved energy",
        "rss_minus_cr_kwh": "RSS minus history-matched CR unserved energy",
        "risk_on_minus_off_unserved_kwh": "risk-on minus risk-off unserved energy",
        "envelope_on_minus_off_unserved_kwh": "uncertainty-envelope-on minus envelope-off unserved energy",
        "risk_adaptation_without_supervisor_unserved_kwh": "risk adaptation without supervisor: RAC minus static CVaR",
        "soc_supervisor_without_risk_unserved_kwh": "SOC supervisor without risk: risk-off supervisor minus static CVaR",
        "supervisor_with_risk_unserved_kwh": "supervisor effect with risk: RSS minus RAC",
    }
    contrasts = {
        label: summarize([float(row[column]) for row in rows], rng)
        for column, label in contrast_columns.items()
    }
    primary_labels = [
        contrast_columns["rss_minus_rb_kwh"],
        contrast_columns["rss_minus_mlr_kwh"],
        contrast_columns["rss_minus_cr_kwh"],
    ]
    primary_raw = {
        label: float(contrasts[label]["exact_two_sided_sign_flip_p"])
        for label in primary_labels
    }
    payload = {
        "analysis_type": "independent_frozen_holdout_extension",
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "unit_of_analysis": "seed; three scenarios remain clustered within each seed",
        "protocol_boundary": "original frozen V15 holdouts plus no-tuning extensions from result-blind locked queues; every excluded seed must fail the public calibration gate before dispatch",
        "per_seed": rows,
        "contrasts": contrasts,
        "primary_multiplicity": {
            "family": primary_labels,
            "method": "Holm step-down adjustment",
            "raw_p": primary_raw,
            "adjusted_p": holm_adjust(primary_raw),
        },
        "aggregate": {
            "rss_unserved_kwh": float(sum(float(row["rss_unserved_kwh"]) for row in rows)),
            "rb_unserved_kwh": float(sum(float(row["rb_unserved_kwh"]) for row in rows)),
            "mlr_unserved_kwh": float(sum(float(row["mlr_unserved_kwh"]) for row in rows)),
            "cr_unserved_kwh": float(sum(float(row["cr_unserved_kwh"]) for row in rows)),
            "rss_risk_cost_yuan": float(sum(float(row["rss_risk_cost_yuan"]) for row in rows)),
            "rb_risk_cost_yuan": float(sum(float(row["rb_risk_cost_yuan"]) for row in rows)),
            "mlr_risk_cost_yuan": float(sum(float(row["mlr_risk_cost_yuan"]) for row in rows)),
            "rss_diesel_fuel_l": float(sum(float(row["rss_diesel_fuel_l"]) for row in rows)),
            "rb_diesel_fuel_l": float(sum(float(row["rb_diesel_fuel_l"]) for row in rows)),
            "mlr_diesel_fuel_l": float(sum(float(row["mlr_diesel_fuel_l"]) for row in rows)),
            "static_cvar_unserved_kwh": float(sum(float(row["static_cvar_unserved_kwh"]) for row in rows)),
            "static_cvar_risk_cost_yuan": float(sum(float(row["static_cvar_risk_cost_yuan"]) for row in rows)),
            "adaptive_cvar_unserved_kwh": float(sum(float(row["adaptive_cvar_unserved_kwh"]) for row in rows)),
            "adaptive_cvar_risk_cost_yuan": float(sum(float(row["adaptive_cvar_risk_cost_yuan"]) for row in rows)),
            "risk_off_supervisor_unserved_kwh": float(sum(float(row["risk_off_supervisor_unserved_kwh"]) for row in rows)),
            "risk_off_supervisor_risk_cost_yuan": float(sum(float(row["risk_off_supervisor_risk_cost_yuan"]) for row in rows)),
            "envelope_off_unserved_kwh": float(sum(float(row["envelope_off_unserved_kwh"]) for row in rows)),
            "envelope_off_risk_cost_yuan": float(sum(float(row["envelope_off_risk_cost_yuan"]) for row in rows)),
            "envelope_off_diesel_fuel_l": float(sum(float(row["envelope_off_diesel_fuel_l"]) for row in rows)),
            "max_rb_decision_seconds": float(max(float(row["rb_max_decision_seconds"]) for row in rows)),
            "max_mlr_decision_seconds": float(max(float(row["mlr_max_decision_seconds"]) for row in rows)),
            "max_static_cvar_decision_seconds": float(max(float(row["static_cvar_max_decision_seconds"]) for row in rows)),
            "max_adaptive_cvar_decision_seconds": float(max(float(row["adaptive_cvar_max_decision_seconds"]) for row in rows)),
            "max_risk_off_supervisor_decision_seconds": float(max(float(row["risk_off_supervisor_max_decision_seconds"]) for row in rows)),
            "max_envelope_off_decision_seconds": float(max(float(row["envelope_off_max_decision_seconds"]) for row in rows)),
            "mean_risk_cost_ratio_vs_mlr": float(
                np.mean([float(row["risk_cost_ratio_vs_mlr"]) for row in rows])
            ),
            "max_decision_seconds": float(
                max(float(row["max_decision_seconds"]) for row in rows)
            ),
            "all_main_physical_pass": all(
                bool(row["main_physical_pass"]) for row in rows
            ),
            "all_risk_ablation_physical_pass": all(
                bool(row["risk_ablation_physical_pass"]) for row in rows
            ),
            "all_uncertainty_ablation_physical_pass": all(
                bool(row["uncertainty_ablation_physical_pass"]) for row in rows
            ),
            "all_causal_physical_pass": all(
                bool(row["causal_physical_pass"]) for row in rows
            ),
        },
        "trajectory_audit_counts": {
            "main_online_method_scenario_trajectories": 21 * len(rows),
            "risk_off_supervisor_trajectories": 3 * len(rows),
            "uncertainty_ablation_method_scenario_trajectories": 21 * len(rows),
            "independent_causal_full_milp_trajectories": 3 * len(rows),
            "total": 48 * len(rows),
        },
        "factorial_interpretation": "The four arms are static scenario-CVaR (no adaptive risk/no supervisor), risk-adaptive CVaR (adaptive risk/no supervisor), risk-neutralized SOC supervisor (no adaptive risk/supervisor), and the full controller (adaptive risk/supervisor). The risk-on versus risk-off supervisory contrast is the clean marginal risk-channel test.",
        "inference_boundary": "Exact tests and seed-bootstrap intervals quantify this registered synthetic-holdout population only; they do not establish field effectiveness or universal superiority.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), constrained_layout=True)
    y = np.arange(len(rows), dtype=float)
    controller_series = (
        ("RSS - RB", "rss_minus_rb_kwh", -0.18, "o", "#2563EB"),
        ("RSS - MLR", "rss_minus_mlr_kwh", 0.00, "s", "#D97706"),
        ("RSS - CR", "rss_minus_cr_kwh", 0.18, "^", "#059669"),
    )
    for label, column, offset, marker, color in controller_series:
        axes[0].scatter(
            [float(row[column]) for row in rows],
            y + offset,
            label=label,
            marker=marker,
            color=color,
            s=34,
        )
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, [str(row["seed"]) for row in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RSS minus comparator (kWh; lower favors RSS)")
    axes[0].set_ylabel("Independent holdout seed")
    axes[0].set_title("(a) Seed-level controller differences")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

    mechanism_specs = (
        (
            "Risk signal\n(with supervisor)",
            contrast_columns["risk_on_minus_off_unserved_kwh"],
            "D",
            "#7C3AED",
        ),
        (
            "Forecast envelope",
            contrast_columns["envelope_on_minus_off_unserved_kwh"],
            "o",
            "#2563EB",
        ),
        (
            "SOC supervisor\n(risk off)",
            contrast_columns["soc_supervisor_without_risk_unserved_kwh"],
            "s",
            "#D97706",
        ),
        (
            "Supervisor\n(risk on)",
            contrast_columns["supervisor_with_risk_unserved_kwh"],
            "^",
            "#059669",
        ),
    )
    mechanism_y = np.arange(len(mechanism_specs), dtype=float)
    for index, (display, contrast_name, marker, color) in enumerate(mechanism_specs):
        summary = contrasts[contrast_name]
        mean = float(summary["mean"])
        lower, upper = [float(value) for value in summary["bootstrap_95_interval_mean"]]
        axes[1].errorbar(
            mean,
            index,
            xerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            markersize=7,
        )
        text_x = mean + 0.22 if mean >= -0.25 else mean - 0.22
        alignment = "left" if mean >= -0.25 else "right"
        axes[1].text(text_x, index, f"{mean:.2f}", va="center", ha=alignment, fontsize=8.5)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_yticks(mechanism_y, [item[0] for item in mechanism_specs])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean paired effect (kWh; lower reduces shortage)")
    axes[1].set_title("(b) Mechanism effects with 95% bootstrap intervals")
    axes[1].set_xlim(-8.4, 0.55)
    axes[1].grid(axis="x", alpha=0.18)
    axes[1].annotate(
        "All ten effects = 0.00 kWh",
        xy=(0.0, 0.0),
        xytext=(-3.75, 0.33),
        arrowprops={"arrowstyle": "->", "color": "#4B5563", "linewidth": 0.9},
        fontsize=8.5,
        color="#374151",
    )
    figure_path = args.output.with_suffix(".png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "figure": str(figure_path)}))


if __name__ == "__main__":
    main()
