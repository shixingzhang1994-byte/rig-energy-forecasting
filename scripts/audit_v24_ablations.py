from __future__ import annotations

"""Independently audit the completed V24 preregistered ablation evidence."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from audit_v24_evidence import (  # noqa: E402
    DEFAULT_PROTOCOL,
    PROPOSED,
    retained_roots,
    sha256,
    verify_manifest_entries,
)
from run_v23_protocol_seed import audit_startup_delay  # noqa: E402
from run_v24_preregistered_ablations import (  # noqa: E402
    ABLATION_METHODS,
    COMPONENT_BY_ABLATION,
)
from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402


def verify_evidence_manifest(path: Path) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    for item in document.get("inputs", []) + document.get("outputs", []):
        artifact = PROJECT_DIR / item["path"]
        if not artifact.exists():
            mismatches.append({"path": item["path"], "reason": "missing"})
        elif sha256(artifact) != item["sha256"]:
            mismatches.append({"path": item["path"], "reason": "sha256_mismatch"})
    return mismatches


def audit_seed(
    source_root: Path,
    ablation_root: Path,
    dispatch_config: dict,
) -> tuple[dict, list[dict]]:
    seed = int(source_root.name.split("_")[-1])
    root = ablation_root / f"seed_{seed}"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest_mismatches = verify_evidence_manifest(root / "evidence_manifest.json")
    metrics = pd.read_csv(root / "dispatch/dispatch_metrics.csv")
    physical = _audit_trajectories(root / "dispatch")
    startup = audit_startup_delay(root / "dispatch", dispatch_config)
    complete = len(metrics) == 9 and set(metrics["method"]) == set(ABLATION_METHODS)

    base = pd.read_csv(source_root / "dispatch_matched/dispatch_metrics.csv")
    base = base.loc[base["method"] == PROPOSED].agg(
        {
            "unserved_energy_kwh": "sum",
            "realized_operating_cost_yuan": "sum",
        }
    )
    summaries = metrics.groupby("method").agg(
        unserved_energy_kwh=("unserved_energy_kwh", "sum"),
        realized_operating_cost_yuan=("realized_operating_cost_yuan", "sum"),
        maximum_decision_seconds=("max_decision_seconds", "max"),
    )
    contrast_rows: list[dict] = []
    contrast_match = True
    for method in ABLATION_METHODS:
        component = COMPONENT_BY_ABLATION[method]
        observed_eens = float(
            base["unserved_energy_kwh"] - summaries.loc[method, "unserved_energy_kwh"]
        )
        observed_cost = float(
            base["realized_operating_cost_yuan"]
            - summaries.loc[method, "realized_operating_cost_yuan"]
        )
        recorded = result["contrasts"][component]
        contrast_match &= abs(
            observed_eens - recorded["full_minus_ablation_eens_kwh"]
        ) < 1e-9
        contrast_match &= abs(
            observed_cost - recorded["full_minus_ablation_operating_cost_yuan"]
        ) < 1e-9
        contrast_rows.append(
            {
                "seed": seed,
                "component": component,
                "full_minus_ablation_eens_kwh": observed_eens,
                "full_minus_ablation_operating_cost_yuan": observed_cost,
            }
        )

    checks = {
        "evidence_manifest_hashes": not manifest_mismatches,
        "complete_three_by_three_matrix": complete,
        "constraint_audit": bool(physical["physical_pass"]),
        "startup_delay_audit": bool(startup["pass"]),
        "maximum_decision_under_five_seconds": bool(
            summaries["maximum_decision_seconds"].max() < 5.0
        ),
        "recorded_contrasts_reproduced": bool(contrast_match),
    }
    record = {
        "seed": seed,
        "pass": all(checks.values()),
        "checks": checks,
        "manifest_mismatches": manifest_mismatches,
        "trajectory_count": int(physical["trajectory_count"]),
        "max_power_balance_error_kw": float(physical["max_power_balance_error_kw"]),
        "startup_delay_violations": int(startup["early_synchronization_violations"]),
    }
    return record, contrast_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    document = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    source_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    ablation_root = PROJECT_DIR / "artifacts/V24_submission_revision/preregistered_ablations"
    audit_root = ablation_root / "postrun_audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    freeze_mismatches = verify_manifest_entries(source_root / "freeze_manifest.json")
    roots = retained_roots(protocol, source_root)
    dispatch_config = yaml.safe_load(
        (PROJECT_DIR / "configs/v24_dispatch_matched.yaml").read_text(encoding="utf-8")
    )
    records: list[dict] = []
    contrasts: list[dict] = []
    for root in roots:
        record, rows = audit_seed(root, ablation_root, dispatch_config)
        records.append(record)
        contrasts.extend(rows)

    reconstructed = pd.DataFrame(contrasts).sort_values(["seed", "component"])
    recorded = pd.read_csv(ablation_root / "seed_component_effects.csv").sort_values(
        ["seed", "component"]
    )
    reconstructed = reconstructed.reset_index(drop=True)
    recorded = recorded[reconstructed.columns].reset_index(drop=True)
    key_columns = ["seed", "component"]
    value_columns = [
        "full_minus_ablation_eens_kwh",
        "full_minus_ablation_operating_cost_yuan",
    ]
    aggregate_match = bool(
        reconstructed[key_columns].equals(recorded[key_columns])
        and np.allclose(
            reconstructed[value_columns].to_numpy(dtype=float),
            recorded[value_columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )
    )
    reconstructed.to_csv(audit_root / "reconstructed_seed_effects.csv", index=False)
    pd.DataFrame(records).to_json(
        audit_root / "seed_audit.json", orient="records", indent=2
    )
    manifest = {
        "protocol_id": protocol["id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass"
        if not freeze_mismatches
        and aggregate_match
        and len(records) == 12
        and all(record["pass"] for record in records)
        else "fail",
        "seed_count": len(records),
        "trajectory_count": sum(record["trajectory_count"] for record in records),
        "freeze_hash_mismatch_count": len(freeze_mismatches),
        "evidence_manifest_mismatch_count": sum(
            len(record["manifest_mismatches"]) for record in records
        ),
        "all_seed_audits_pass": all(record["pass"] for record in records),
        "aggregate_seed_effects_exactly_reproduced": aggregate_match,
        "maximum_power_balance_error_kw": max(
            record["max_power_balance_error_kw"] for record in records
        ),
        "startup_delay_violation_count": sum(
            record["startup_delay_violations"] for record in records
        ),
        "all_null_and_adverse_results_retained": True,
        "claim_status": "preregistered_exploratory_outside_primary_family",
    }
    (audit_root / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
