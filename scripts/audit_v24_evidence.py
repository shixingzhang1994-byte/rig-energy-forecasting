from __future__ import annotations

"""Reconstruct and audit the frozen V24 evidence without modifying it.

This post-freeze audit deliberately writes to a separate directory.  It uses
exact trajectory filenames because the historical V24 aggregation glob for
``Residual-CVaR-MILP`` also matched ``Risk-Adaptive-Residual-CVaR-MILP`` and
therefore doubled only that method's demand denominator.  Raw EENS and the
three preregistered primary contrasts were not affected by that issue.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_v23_confirmatory import (  # noqa: E402
    bootstrap_mean_interval,
    exact_sign_flip_pvalue,
    holm_adjust,
)


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v24_submission_revision.yaml"
PROPOSED = "Full-Risk-SOC-Supervisory-MILP"
PRIMARY_REFERENCES = [
    "Rule-Based",
    "Point-Forecast-MILP",
    "Risk-Adaptive-Residual-CVaR-MILP",
]
COMPONENT_CONTRASTS = [
    ("residual_scenarios", "Residual-CVaR-MILP", "Point-Forecast-MILP"),
    (
        "risk_adaptation",
        "Risk-Adaptive-Residual-CVaR-MILP",
        "Residual-CVaR-MILP",
    ),
    (
        "reserve_soc_supervisor",
        PROPOSED,
        "Risk-Adaptive-Residual-CVaR-MILP",
    ),
]
FORECAST_REFERENCE = "Causal-ErrorFeedback-Ensemble"


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def trajectory_path(root: Path, scenario: str, method: str) -> Path:
    """Return the one exact trajectory path for a scenario-method pair."""

    return root / "dispatch_matched" / f"trajectory_{scenario}_{method_slug(method)}.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_summary(values: np.ndarray) -> dict[str, float | int]:
    paired = np.asarray(values, dtype=float)
    if paired.ndim != 1 or len(paired) < 2:
        raise ValueError("paired summary requires at least two seed-level values")
    low, high = bootstrap_mean_interval(paired)
    leave_one_out = np.asarray(
        [np.delete(paired, index).mean() for index in range(len(paired))]
    )
    return {
        "seed_count": int(len(paired)),
        "mean_difference": float(paired.mean()),
        "sample_sd": float(paired.std(ddof=1)),
        "median_difference": float(np.median(paired)),
        "q25": float(np.quantile(paired, 0.25)),
        "q75": float(np.quantile(paired, 0.75)),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "exact_sign_flip_pvalue": exact_sign_flip_pvalue(paired),
        "improve_count": int((paired < -1e-12).sum()),
        "tie_count": int((np.abs(paired) <= 1e-12).sum()),
        "worsen_count": int((paired > 1e-12).sum()),
        "leave_one_seed_out_mean_min": float(leave_one_out.min()),
        "leave_one_seed_out_mean_max": float(leave_one_out.max()),
    }


def retained_roots(protocol: dict, artifact_root: Path) -> list[Path]:
    retained: list[Path] = []
    required = int(protocol["required_eligible_holdouts"])
    for seed in map(int, protocol["prospective_holdout_seed_order"]):
        root = artifact_root / f"seed_{seed}"
        eligibility_path = root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"V24 queue is incomplete at seed {seed}")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if not eligibility.get("eligible"):
            continue
        if not (root / "result.json").exists():
            raise RuntimeError(f"eligible V24 seed {seed} is incomplete")
        retained.append(root)
        if len(retained) == required:
            return retained
    raise RuntimeError(f"only {len(retained)} eligible V24 seeds are complete")


def verify_manifest_entries(manifest_path: Path) -> list[dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    for item in manifest.get("frozen_files", manifest.get("files", [])):
        path = PROJECT_DIR / item["path"]
        if not path.exists():
            mismatches.append({"path": item["path"], "reason": "missing"})
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            mismatches.append(
                {
                    "path": item["path"],
                    "reason": "sha256_mismatch",
                    "expected": item["sha256"],
                    "actual": actual,
                }
            )
    return mismatches


def reconstruct_dispatch(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict] = []
    scenario_rows: list[dict] = []
    summed_columns = [
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
    for root in roots:
        seed = int(root.name.split("_")[-1])
        metrics = pd.read_csv(root / "dispatch_matched/dispatch_metrics.csv")
        for _, item in metrics.iterrows():
            scenario_rows.append({"seed": seed, **item.to_dict()})
        for method, frame in metrics.groupby("method", sort=False):
            demand_kwh = 0.0
            for scenario in frame["scenario"]:
                path = trajectory_path(root, str(scenario), str(method))
                if not path.exists():
                    raise RuntimeError(f"missing exact trajectory: {path}")
                trajectory = pd.read_csv(path)
                demand_kwh += float(trajectory["load_kw"].sum() * 5.0 / 3600.0)
            row = {
                "seed": seed,
                "method": method,
                "demand_energy_kwh": demand_kwh,
                **{column: float(frame[column].sum()) for column in summed_columns},
                "minimum_soc_pct": float(frame["minimum_soc_pct"].min()),
                "max_decision_seconds": float(frame["max_decision_seconds"].max()),
            }
            row["normalized_eens_kwh_per_mwh"] = (
                1000.0 * row["unserved_energy_kwh"] / demand_kwh
            )
            seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    demand_spread = seed_frame.groupby("seed")["demand_energy_kwh"].agg(
        lambda values: float(values.max() - values.min())
    )
    if float(demand_spread.max()) > 1e-9:
        raise RuntimeError("demand reconstruction differs across methods")
    return seed_frame, pd.DataFrame(scenario_rows)


def dispatch_summaries(
    seed_frame: pd.DataFrame, scenario_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_summary = (
        seed_frame.groupby("method")
        .agg(
            seed_count=("seed", "count"),
            mean_eens_kwh=("unserved_energy_kwh", "mean"),
            median_eens_kwh=("unserved_energy_kwh", "median"),
            mean_normalized_eens_kwh_per_mwh=(
                "normalized_eens_kwh_per_mwh",
                "mean",
            ),
            mean_operating_cost_yuan=("realized_operating_cost_yuan", "mean"),
            mean_loss_of_load_duration_hours=(
                "loss_of_load_duration_hours",
                "mean",
            ),
            mean_diesel_fuel_l=("diesel_fuel_l", "mean"),
            mean_start_commands=("generator_start_commands", "mean"),
            worst_max_decision_seconds=("max_decision_seconds", "max"),
        )
        .reset_index()
        .sort_values("mean_eens_kwh")
    )
    pivot_eens = seed_frame.pivot(
        index="seed", columns="method", values="unserved_energy_kwh"
    )
    pivot_cost = seed_frame.pivot(
        index="seed", columns="method", values="realized_operating_cost_yuan"
    )
    component_rows: list[dict] = []
    for label, treatment, reference in COMPONENT_CONTRASTS:
        differences = (pivot_eens[treatment] - pivot_eens[reference]).to_numpy()
        row = {
            "component": label,
            "treatment": treatment,
            "reference": reference,
            **paired_summary(differences),
            "mean_operating_cost_difference_yuan": float(
                (pivot_cost[treatment] - pivot_cost[reference]).mean()
            ),
            "claim_status": (
                "confirmatory_primary"
                if treatment == PROPOSED
                and reference == "Risk-Adaptive-Residual-CVaR-MILP"
                else "preregistered_exploratory"
            ),
        }
        component_rows.append(row)
    component_frame = pd.DataFrame(component_rows)

    scenario_contrast_rows: list[dict] = []
    scenario_pivot = scenario_frame.pivot_table(
        index=["seed", "scenario"],
        columns="method",
        values=["unserved_energy_kwh", "realized_operating_cost_yuan"],
        aggfunc="first",
    )
    for scenario in sorted(scenario_frame["scenario"].unique()):
        selected = scenario_pivot.xs(scenario, level="scenario")
        for reference in PRIMARY_REFERENCES:
            differences = (
                selected[("unserved_energy_kwh", PROPOSED)]
                - selected[("unserved_energy_kwh", reference)]
            ).to_numpy()
            scenario_contrast_rows.append(
                {
                    "scenario": scenario,
                    "contrast": f"{PROPOSED}_minus_{reference}",
                    **paired_summary(differences),
                    "mean_operating_cost_difference_yuan": float(
                        (
                            selected[("realized_operating_cost_yuan", PROPOSED)]
                            - selected[("realized_operating_cost_yuan", reference)]
                        ).mean()
                    ),
                    "claim_status": "descriptive_scenario_breakdown",
                }
            )
    return method_summary, component_frame, pd.DataFrame(scenario_contrast_rows)


def forecast_summaries(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for root in roots:
        frame = pd.read_csv(
            root / "source/forecast/benchmark_metrics.csv", encoding="utf-8-sig"
        )
        frame.insert(0, "seed", int(root.name.split("_")[-1]))
        frames.append(frame)
    seed_metrics = pd.concat(frames, ignore_index=True)
    summary = (
        seed_metrics.groupby("model")
        .agg(
            seed_count=("seed", "count"),
            mean_mae_kw=("mae_kw", "mean"),
            sd_mae_kw=("mae_kw", "std"),
            mean_rmse_kw=("rmse_kw", "mean"),
            mean_r2=("r2", "mean"),
            mean_peak_mae_kw=("peak_mae_kw", "mean"),
            mean_transition_mae_kw=("transition_mae_kw", "mean"),
        )
        .reset_index()
        .sort_values("mean_mae_kw")
    )
    pivot = seed_metrics.pivot(index="seed", columns="model", values="mae_kw")
    contrast_rows: list[dict] = []
    raw_pvalues: list[float] = []
    for reference in pivot.columns:
        if reference == FORECAST_REFERENCE:
            continue
        differences = (pivot[FORECAST_REFERENCE] - pivot[reference]).to_numpy()
        row = {
            "contrast": f"{FORECAST_REFERENCE}_minus_{reference}",
            **paired_summary(differences),
            "claim_status": "secondary_descriptive_with_Holm_adjustment",
        }
        raw_pvalues.append(float(row["exact_sign_flip_pvalue"]))
        contrast_rows.append(row)
    for row, adjusted in zip(contrast_rows, holm_adjust(raw_pvalues)):
        row["holm_adjusted_pvalue"] = adjusted
    return seed_metrics, summary, pd.DataFrame(contrast_rows)


def risk_summaries(roots: list[Path]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "high_risk_recall",
        "severe_recall",
        "severe_underclassification_rate",
        "severity_mae",
        "log_loss",
        "multiclass_brier",
    ]
    for root in roots:
        selection = json.loads(
            (root / "risk/operational_model_selection.json").read_text(
                encoding="utf-8"
            )
        )
        metrics = json.loads(
            (root / "risk/dispatch_guard_metrics.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "seed": int(root.name.split("_")[-1]),
                "validation_gate_fallback_active": bool(
                    selection["validation_gate_fallback_active"]
                ),
                "selected_reporting_model": selection["selected_model"],
                "dispatch_guard_model": metrics["model"],
                **{name: float(metrics[name]) for name in metric_names},
            }
        )
    frame = pd.DataFrame(rows)
    learned = frame.loc[~frame["validation_gate_fallback_active"]]
    summary = {
        "seed_count": int(len(frame)),
        "learned_guard_seed_count": int(len(learned)),
        "fallback_seed_count": int(frame["validation_gate_fallback_active"].sum()),
        "fallback_seeds": frame.loc[
            frame["validation_gate_fallback_active"], "seed"
        ].astype(int).tolist(),
        "learned_guard_macro_f1_mean": float(learned["macro_f1"].mean()),
        "learned_guard_macro_f1_min": float(learned["macro_f1"].min()),
        "learned_guard_high_risk_recall_mean": float(
            learned["high_risk_recall"].mean()
        ),
        "learned_guard_high_risk_recall_min": float(
            learned["high_risk_recall"].min()
        ),
        "learned_guard_severe_recall_mean": float(learned["severe_recall"].mean()),
        "learned_guard_severe_recall_min": float(learned["severe_recall"].min()),
        "fallback_is_not_certified_learned_risk": True,
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    document = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    output = args.output_dir or artifact_root / "postfreeze_audit"
    output.mkdir(parents=True, exist_ok=True)
    roots = retained_roots(protocol, artifact_root)

    freeze_manifest = artifact_root / "freeze_manifest.json"
    hash_mismatches = verify_manifest_entries(freeze_manifest)
    seed_manifest_mismatches: list[dict] = []
    technical_failures: list[int] = []
    for root in roots:
        seed = int(root.name.split("_")[-1])
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        if not result.get("technical_gate_pass"):
            technical_failures.append(seed)
        for mismatch in verify_manifest_entries(root / "evidence_manifest.json"):
            seed_manifest_mismatches.append({"seed": seed, **mismatch})
    if hash_mismatches or seed_manifest_mismatches or technical_failures:
        raise RuntimeError(
            "V24 integrity audit failed: "
            f"freeze={len(hash_mismatches)}, "
            f"seed={len(seed_manifest_mismatches)}, "
            f"technical={technical_failures}"
        )

    dispatch_seed, dispatch_scenario = reconstruct_dispatch(roots)
    dispatch_method, component, scenario = dispatch_summaries(
        dispatch_seed, dispatch_scenario
    )
    forecast_seed, forecast_model, forecast_contrasts = forecast_summaries(roots)
    risk_seed, risk_summary = risk_summaries(roots)

    outputs = {
        "dispatch_seed_method_summary.csv": dispatch_seed,
        "dispatch_method_summary.csv": dispatch_method,
        "dispatch_component_contrasts.csv": component,
        "dispatch_scenario_contrasts.csv": scenario,
        "forecast_seed_metrics.csv": forecast_seed,
        "forecast_model_summary.csv": forecast_model,
        "forecast_cef_contrasts.csv": forecast_contrasts,
        "risk_seed_summary.csv": risk_seed,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    (output / "risk_summary.json").write_text(
        json.dumps(risk_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    legacy = artifact_root / "analysis/seed_method_summary.csv"
    legacy_issue_detected = False
    if legacy.exists():
        old = pd.read_csv(legacy)
        residual = old.loc[old["method"] == "Residual-CVaR-MILP"]
        other = old.loc[old["method"] == "Point-Forecast-MILP"]
        merged = residual.merge(other, on="seed", suffixes=("_residual", "_point"))
        legacy_issue_detected = bool(
            np.allclose(
                merged["demand_energy_kwh_residual"],
                2.0 * merged["demand_energy_kwh_point"],
                rtol=0.0,
                atol=1e-9,
            )
        )
    manifest = {
        "protocol_id": protocol["id"],
        "status": "pass" if legacy_issue_detected else "pass_no_legacy_issue_found",
        "retained_seeds": [int(root.name.split("_")[-1]) for root in roots],
        "freeze_hash_mismatch_count": 0,
        "seed_manifest_hash_mismatch_count": 0,
        "technical_failure_count": 0,
        "exact_trajectory_paths_used": True,
        "common_demand_denominator_verified_across_methods": True,
        "legacy_normalized_eens_issue": {
            "detected": legacy_issue_detected,
            "affected_method": "Residual-CVaR-MILP",
            "cause": (
                "historical wildcard also matched Risk-Adaptive-Residual-CVaR-MILP"
            ),
            "affected_fields": [
                "Residual-CVaR-MILP demand_energy_kwh",
                "Residual-CVaR-MILP normalized_eens_kwh_per_mwh",
            ],
            "unaffected_fields": [
                "raw EENS",
                "three preregistered primary EENS contrasts",
                "exact sign-flip p-values",
            ],
            "correction_location": str(output.relative_to(PROJECT_DIR)),
        },
        "risk_validation_fallback": risk_summary,
        "claim_boundaries": [
            "V24 is publicly constrained synthetic active-power simulation, not field validation.",
            "The fallback seed remains in every confirmatory dispatch estimate.",
            "Component contrasts other than the preregistered supervisor contrast are exploratory.",
            "Operating-cost increases must be reported separately from EENS reductions.",
        ],
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
