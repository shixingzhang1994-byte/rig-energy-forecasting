from __future__ import annotations

"""Analyze V26 at the preassigned parameter-set-by-seed unit level."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_v23_confirmatory import holm_adjust  # noqa: E402
from audit_v24_evidence import (  # noqa: E402
    PRIMARY_REFERENCES,
    PROPOSED,
    paired_summary,
    reconstruct_dispatch,
)


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v26_heterogeneous_validation.yaml"


def assigned_units(document: dict) -> list[dict]:
    family_path = PROJECT_DIR / document["heterogeneous_family"]["definition"]
    family = yaml.safe_load(family_path.read_text(encoding="utf-8"))
    return family["prospective_units"]


def factor_breakdown(seed_summary: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    pivot_eens = seed_summary.pivot(
        index="seed", columns="method", values="unserved_energy_kwh"
    )
    joined = units.set_index("seed")
    rows: list[dict] = []
    for reference in PRIMARY_REFERENCES:
        differences = (pivot_eens[PROPOSED] - pivot_eens[reference]).rename("difference")
        frame = joined.join(differences)
        for factor in ("load", "transition", "plant"):
            for level, group in frame.groupby(factor):
                values = group["difference"].to_numpy(float)
                rows.append(
                    {
                        "contrast": f"{PROPOSED}_minus_{reference}",
                        "factor": factor,
                        "level": level,
                        "unit_count": len(values),
                        "mean_eens_difference_kwh": float(values.mean()),
                        "median_eens_difference_kwh": float(np.median(values)),
                        "improve_count": int((values < -1e-12).sum()),
                        "tie_count": int((np.abs(values) <= 1e-12).sum()),
                        "worsen_count": int((values > 1e-12).sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    document = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    output = artifact_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    units = assigned_units(document)
    expected_seeds = [int(unit["seed"]) for unit in units]
    roots = [artifact_root / f"seed_{seed}" for seed in expected_seeds]
    failures: list[dict] = []
    for root in roots:
        seed = int(root.name.split("_")[-1])
        eligibility_path = root / "eligibility.json"
        result_path = root / "result.json"
        if not eligibility_path.exists() or not result_path.exists():
            failures.append({"seed": seed, "reason": "missing_unit_artifact"})
            continue
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not eligibility.get("eligible") or not result.get("technical_gate_pass"):
            failures.append(
                {
                    "seed": seed,
                    "reason": "upstream_ineligible_or_technical_failure",
                }
            )
    if failures:
        manifest = {
            "protocol_id": protocol["id"],
            "status": "incomplete_no_efficacy_estimates",
            "all_preassigned_units_retained": True,
            "failures": failures,
        }
        (output / "analysis_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    seed_summary, scenario_metrics = reconstruct_dispatch(roots)
    unit_frame = pd.DataFrame(units)
    seed_summary = seed_summary.merge(unit_frame, on="seed", validate="many_to_one")
    scenario_metrics = scenario_metrics.merge(unit_frame, on="seed", validate="many_to_one")
    seed_summary.to_csv(output / "unit_method_summary.csv", index=False)
    scenario_metrics.to_csv(output / "unit_scenario_metrics.csv", index=False)

    method_rows = []
    for method, frame in seed_summary.groupby("method"):
        method_rows.append(
            {
                "method": method,
                "unit_count": len(frame),
                "mean_eens_kwh": float(frame["unserved_energy_kwh"].mean()),
                "median_eens_kwh": float(frame["unserved_energy_kwh"].median()),
                "mean_normalized_eens_kwh_per_mwh": float(
                    frame["normalized_eens_kwh_per_mwh"].mean()
                ),
                "mean_operating_cost_yuan": float(
                    frame["realized_operating_cost_yuan"].mean()
                ),
                "worst_max_decision_seconds": float(frame["max_decision_seconds"].max()),
            }
        )
    pd.DataFrame(method_rows).sort_values("mean_eens_kwh").to_csv(
        output / "method_summary.csv", index=False
    )

    pivot_eens = seed_summary.pivot(index="seed", columns="method", values="unserved_energy_kwh")
    pivot_cost = seed_summary.pivot(
        index="seed", columns="method", values="realized_operating_cost_yuan"
    )
    contrasts = []
    pvalues = []
    for reference in PRIMARY_REFERENCES:
        differences = (pivot_eens[PROPOSED] - pivot_eens[reference]).to_numpy(float)
        row = {
            "contrast": f"{PROPOSED}_minus_{reference}",
            **paired_summary(differences),
            "mean_operating_cost_difference_yuan": float(
                (pivot_cost[PROPOSED] - pivot_cost[reference]).mean()
            ),
        }
        contrasts.append(row)
        pvalues.append(float(row["exact_sign_flip_pvalue"]))
    for row, adjusted in zip(contrasts, holm_adjust(pvalues)):
        row["holm_adjusted_pvalue"] = adjusted
    pd.DataFrame(contrasts).to_csv(output / "primary_contrasts.csv", index=False)
    factor_breakdown(seed_summary, unit_frame).to_csv(
        output / "factor_breakdown.csv", index=False
    )

    fallback_seeds = []
    for root in roots:
        metadata = json.loads((root / "risk/run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("validation_gate_fallback_active"):
            fallback_seeds.append(int(root.name.split("_")[-1]))
    manifest = {
        "protocol_id": protocol["id"],
        "status": "complete",
        "analysis_unit": "preassigned_parameter_set_by_seed",
        "unit_count": len(roots),
        "seeds": expected_seeds,
        "primary_contrasts": contrasts,
        "fallback_seeds": fallback_seeds,
        "all_preassigned_null_adverse_and_failed_results_retained": True,
        "evidence_boundary": "heterogeneous simulation, not independent rigs or field validation",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
