from __future__ import annotations

"""Aggregate the first 12 eligible V22 seeds at the seed-cluster level."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v22_submission_revision.yaml"
PROPOSED = "Full-Risk-SOC-Supervisory-MILP"
PRIMARY_REFERENCES = [
    "Rule-Based",
    "Point-Forecast-MILP",
    "Risk-Adaptive-Residual-CVaR-MILP",
]


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    paired = np.asarray(values, dtype=float)
    paired = paired[np.abs(paired) > 1e-12]
    if len(paired) == 0:
        return 1.0
    observed = abs(float(paired.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(paired)):
        statistics.append(abs(float(np.mean(paired * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def bootstrap_mean_interval(
    values: np.ndarray, *, draws: int = 20_000, seed: int = 20260914
) -> tuple[float, float]:
    paired = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(draws, len(paired)))
    means = paired[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _retained_seed_roots(protocol: dict, artifact_root: Path) -> list[Path]:
    retained = []
    required = int(protocol["required_eligible_holdouts"])
    for seed in map(int, protocol["prospective_holdout_seed_order"]):
        root = artifact_root / f"seed_{seed}"
        eligibility_path = root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"V22 queue is incomplete at seed {seed}")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if not eligibility.get("eligible"):
            continue
        if not (root / "result.json").exists():
            raise RuntimeError(f"eligible V22 seed {seed} is incomplete")
        retained.append(root)
        if len(retained) == required:
            return retained
    raise RuntimeError(f"only {len(retained)} eligible V22 seeds are complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    document = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    output = args.output_dir or artifact_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    roots = _retained_seed_roots(protocol, artifact_root)

    rows = []
    scenario_rows = []
    for root in roots:
        seed = int(root.name.split("_")[-1])
        metrics = pd.read_csv(root / "dispatch_matched/dispatch_metrics.csv")
        for _, item in metrics.iterrows():
            scenario_rows.append({"seed": seed, **item.to_dict()})
        for method, frame in metrics.groupby("method"):
            trajectory_paths = list(
                (root / "dispatch_matched").glob(
                    "trajectory_*_" + method.lower().replace("-", "_") + ".csv"
                )
            )
            demand_kwh = 0.0
            for path in trajectory_paths:
                trajectory = pd.read_csv(path)
                demand_kwh += float(trajectory["load_kw"].sum() * 5.0 / 3600.0)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "demand_energy_kwh": demand_kwh,
                    **{
                        column: float(frame[column].sum())
                        for column in [
                            "unserved_energy_kwh",
                            "loss_of_load_duration_hours",
                            "realized_operating_cost_yuan",
                            "terminal_energy_adjustment_yuan",
                            "reliability_penalty_yuan",
                            "total_social_cost_yuan",
                            "diesel_fuel_l",
                            "generator_startups",
                            "generator_start_commands",
                        ]
                    },
                    "minimum_soc_pct": float(frame["minimum_soc_pct"].min()),
                    "max_decision_seconds": float(
                        frame["max_decision_seconds"].max()
                    ),
                }
            )
    seed_summary = pd.DataFrame(rows)
    seed_summary["normalized_eens_kwh_per_mwh"] = (
        1000.0
        * seed_summary["unserved_energy_kwh"]
        / seed_summary["demand_energy_kwh"]
    )
    seed_summary.to_csv(output / "seed_method_summary.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(
        output / "seed_scenario_metrics.csv", index=False
    )

    aggregate_rows = []
    for method, frame in seed_summary.groupby("method"):
        aggregate_rows.append(
            {
                "method": method,
                "seed_count": int(len(frame)),
                "mean_eens_kwh": float(frame["unserved_energy_kwh"].mean()),
                "median_eens_kwh": float(frame["unserved_energy_kwh"].median()),
                "mean_normalized_eens_kwh_per_mwh": float(
                    frame["normalized_eens_kwh_per_mwh"].mean()
                ),
                "mean_operating_cost_yuan": float(
                    frame["realized_operating_cost_yuan"].mean()
                ),
                "mean_loss_of_load_duration_hours": float(
                    frame["loss_of_load_duration_hours"].mean()
                ),
                "mean_diesel_fuel_l": float(frame["diesel_fuel_l"].mean()),
                "mean_start_commands": float(
                    frame["generator_start_commands"].mean()
                ),
                "worst_max_decision_seconds": float(
                    frame["max_decision_seconds"].max()
                ),
            }
        )
    pd.DataFrame(aggregate_rows).sort_values("mean_eens_kwh").to_csv(
        output / "method_summary.csv", index=False
    )

    pivot_eens = seed_summary.pivot(
        index="seed", columns="method", values="unserved_energy_kwh"
    )
    pivot_cost = seed_summary.pivot(
        index="seed", columns="method", values="realized_operating_cost_yuan"
    )
    contrast_rows = []
    raw_pvalues = []
    for reference in PRIMARY_REFERENCES:
        differences = (pivot_eens[PROPOSED] - pivot_eens[reference]).to_numpy()
        cost_differences = (
            pivot_cost[PROPOSED] - pivot_cost[reference]
        ).to_numpy()
        low, high = bootstrap_mean_interval(differences)
        pvalue = exact_sign_flip_pvalue(differences)
        raw_pvalues.append(pvalue)
        leave_one_out = np.asarray(
            [np.delete(differences, index).mean() for index in range(len(differences))]
        )
        contrast_rows.append(
            {
                "contrast": f"{PROPOSED}_minus_{reference}",
                "seed_count": int(len(differences)),
                "mean_eens_difference_kwh": float(differences.mean()),
                "sample_sd_kwh": float(differences.std(ddof=1)),
                "median_difference_kwh": float(np.median(differences)),
                "q25_kwh": float(np.quantile(differences, 0.25)),
                "q75_kwh": float(np.quantile(differences, 0.75)),
                "bootstrap_95_low_kwh": low,
                "bootstrap_95_high_kwh": high,
                "exact_sign_flip_pvalue": pvalue,
                "improve_count": int((differences < -1e-12).sum()),
                "tie_count": int((np.abs(differences) <= 1e-12).sum()),
                "worsen_count": int((differences > 1e-12).sum()),
                "leave_one_seed_out_mean_min_kwh": float(leave_one_out.min()),
                "leave_one_seed_out_mean_max_kwh": float(leave_one_out.max()),
                "mean_operating_cost_difference_yuan": float(
                    cost_differences.mean()
                ),
            }
        )
    adjusted = holm_adjust(raw_pvalues)
    for row, value in zip(contrast_rows, adjusted):
        row["holm_adjusted_pvalue"] = value
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame.to_csv(output / "primary_contrasts.csv", index=False)

    manifest = {
        "protocol_id": protocol["id"],
        "retained_seeds": [int(root.name.split("_")[-1]) for root in roots],
        "analysis_unit": "seed_with_three_scenarios_clustered",
        "primary_endpoint": "EENS_kWh",
        "primary_contrasts": contrast_rows,
        "failed_or_adverse_results_retained": True,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
