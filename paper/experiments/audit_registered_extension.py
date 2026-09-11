#!/usr/bin/env python3
"""Audit the registered ten-holdout paper extension and hash its evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(".")
PROJECT = ROOT / "rig-energy-forecasting"
PAPER = ROOT / "paper/experiments"
RESULTS = PAPER / "results"
PROTOCOL = PROJECT / "configs/paper_v15_extension_holdouts_v3.yaml"
PREREGISTRATION = PAPER / "preregistered_extension_v3.yaml"
FREEZE = PROJECT / "artifacts/paper_v15_extension_holdouts_v3/freeze_manifest.json"
OUTPUT = RESULTS / "registered_extension_v3_audit.json"
V3_SEEDS = [20261016, 20261017, 20261018, 20261019, 20261020, 20261021]
ALL_SEEDS = [20261011, 20261012, 20261014, 20261015, *V3_SEEDS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    summary_path = RESULTS / "extended_holdout_analysis.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_v3_dirs = sorted(
        int(path.name.removeprefix("seed_"))
        for path in (PROJECT / "artifacts/paper_v15_extension_holdouts_v3").glob("seed_*")
        if path.is_dir()
    )
    checks: dict[str, bool] = {
        "protocol_starts_at_20261016": protocol["protocol"]["holdout_seed_order"][0] == 20261016,
        "protocol_requires_six_eligible": protocol["protocol"]["required_eligible_seeds_per_phase"] == 6,
        "only_registered_v3_seeds_ran": actual_v3_dirs == V3_SEEDS,
        "freeze_protocol_hash_matches": freeze["extension_protocol_sha256"] == sha256(PROTOCOL),
        "summary_has_exact_ten_seeds": [int(seed) for seed in summary["seeds"]] == ALL_SEEDS,
        "summary_seed_is_independent_unit": summary["unit_of_analysis"].startswith("seed;"),
        "all_main_physical_pass": bool(summary["aggregate"]["all_main_physical_pass"]),
        "all_risk_ablation_physical_pass": bool(summary["aggregate"]["all_risk_ablation_physical_pass"]),
        "all_uncertainty_ablation_physical_pass": bool(summary["aggregate"]["all_uncertainty_ablation_physical_pass"]),
        "all_causal_physical_pass": bool(summary["aggregate"]["all_causal_physical_pass"]),
        "trajectory_total_is_480": summary["trajectory_audit_counts"]["total"] == 480,
    }
    files = [
        PROTOCOL,
        PREREGISTRATION,
        FREEZE,
        PAPER / "run_registered_extension_pipeline.py",
        PAPER / "summarize_extended_holdouts.py",
        PAPER / "run_independent_causal_full_milp.py",
        PAPER / "export_shared_initial_histories.py",
        PAPER / "run_risk_signal_ablation.py",
        PAPER / "run_no_uncertainty_ablation.py",
        PAPER / "summarize_risk_action_overlap.py",
        PAPER / "tests/test_independent_causal_full_milp.py",
        PAPER / "tests/test_extended_holdout_statistics.py",
        summary_path,
        RESULTS / "extended_holdout_analysis.csv",
        RESULTS / "risk_action_overlap.json",
    ]
    for seed in V3_SEEDS:
        root = PROJECT / f"artifacts/paper_v15_extension_holdouts_v3/seed_{seed}"
        eligibility = json.loads((root / "eligibility.json").read_text(encoding="utf-8"))
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        checks[f"seed_{seed}_eligible_before_dispatch"] = bool(
            eligibility["eligible"]
            and eligibility["timing"] == "before_dispatch_and_before_any_controller_outcome"
        )
        checks[f"seed_{seed}_all_frozen_gates_pass"] = bool(result["overall_pass"])
        files.extend([root / "eligibility.json", root / "result.json", root / "method_summary.csv"])
    for seed in ALL_SEEDS:
        files.extend(
            [
                RESULTS / f"shared_initial_histories/seed_{seed}.json",
                RESULTS / f"independent_causal_full_milp_history_matched/seed_{seed}.json",
                RESULTS / f"risk_signal_ablation/seed_{seed}/risk_ablation_result.json",
                RESULTS / f"uncertainty_envelope_ablation/seed_{seed}/uncertainty_ablation_result.json",
            ]
        )
    missing = [rel(path) for path in files if not path.exists()]
    checks["all_registered_evidence_files_exist"] = not missing
    payload = {
        "audit": "registered_v15_extension_v3_and_ten_holdout_evidence",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "missing": missing,
        "v3_seed_directories": actual_v3_dirs,
        "all_reported_seeds": ALL_SEEDS,
        "trajectory_audit_counts": summary["trajectory_audit_counts"],
        "primary_multiplicity": summary["primary_multiplicity"],
        "file_count": len(files),
        "files": [
            {"path": rel(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in files
            if path.exists()
        ],
        "reproduction_command": "conda run -n qz-rig-energy-ml python paper/experiments/run_registered_extension_pipeline.py --workers 4",
        "claim_boundary": "Internal evidence for the registered public-evidence-constrained synthetic holdout population; not field validation.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": payload["status"], "checks": checks, "file_count": len(files)}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
