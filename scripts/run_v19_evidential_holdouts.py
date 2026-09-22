from __future__ import annotations

"""Run the frozen V19 evidential-risk candidate on prospective holdouts.

The runner is restartable and fail-closed. It freezes the execution layer,
verifies the pre-existing V19 candidate and V15 parent hashes, then processes
the declared seed queue in order. The first six upstream-eligible seeds form
the complete analysis set regardless of their outcomes.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs/v19_evidential_risk_development.yaml"
HOLDOUT_CONFIG_PATH = PROJECT_DIR / "configs/v19_evidential_risk_holdouts.yaml"
CANDIDATE_FREEZE_PATH = PROJECT_DIR / "artifacts/v19_evidential_risk/risk_candidate_freeze.json"
SELECTED_CONFIG_PATH = PROJECT_DIR / "artifacts/v19_evidential_risk/development_search/selected_evidential_config.yaml"
PARENT_FREEZE_PATH = PROJECT_DIR / "artifacts/v15_generator_first_reserve/freeze_manifest.json"
RUN_FREEZE_PATH = PROJECT_DIR / "artifacts/v19_evidential_risk/prospective_run_freeze.json"
SOURCE_ROOT = PROJECT_DIR / "artifacts/v19_evidential_risk/prospective_source"
OUTPUT_ROOT = PROJECT_DIR / "artifacts/v19_evidential_risk/prospective"
SUMMARY_PATH = OUTPUT_ROOT / "prospective_summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT / "prospective_seed_results.csv"
MANIFEST_PATH = OUTPUT_ROOT / "prospective_evidence_manifest.json"
ACTION_COLUMNS = [
    "grid_kw",
    "generator_kw",
    "generator_units_on",
    "storage_kw",
    "unserved_kw",
    "generator_startup_units",
    "generator_shutdown_units",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_DIR))


def run_command(command: list[str], allowed_returncodes: set[int] | None = None) -> int:
    allowed = allowed_returncodes or {0}
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)
    return int(result.returncode)


def verify_candidate_freeze() -> dict[str, object]:
    freeze = json.loads(CANDIDATE_FREEZE_PATH.read_text(encoding="utf-8"))
    changed: list[str] = []
    for item in freeze["files"]:
        path = PROJECT_DIR / item["path"]
        if not path.exists() or sha256(path) != item["sha256"]:
            changed.append(item["path"])
    if sha256(PARENT_FREEZE_PATH) != freeze["parent_v15_freeze_sha256"]:
        changed.append(relative(PARENT_FREEZE_PATH))
    if changed:
        raise RuntimeError("frozen V19/V15 inputs changed: " + ", ".join(changed))

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    holdout = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_queue = [int(v) for v in protocol["protocol"]["prospective_holdout_seed_order"]]
    actual_queue = [int(v) for v in holdout["protocol"]["holdout_seed_order"]]
    if actual_queue != expected_queue:
        raise RuntimeError("holdout queue differs from the frozen V19 protocol")
    expected_count = int(protocol["protocol"]["required_eligible_holdouts"])
    actual_count = int(holdout["protocol"]["required_eligible_seeds_per_phase"])
    if actual_count != expected_count:
        raise RuntimeError("required holdout count differs from the frozen V19 protocol")
    return freeze


def ensure_run_freeze() -> dict[str, object]:
    candidate = verify_candidate_freeze()
    paths = [
        HOLDOUT_CONFIG_PATH,
        Path(__file__).resolve(),
        PROJECT_DIR / "scripts/run_evidential_risk_fusion.py",
        PROJECT_DIR / "tests/test_evidence_theory.py",
        PROJECT_DIR / "tests/test_v19_evidential_holdouts.py",
        SELECTED_CONFIG_PATH,
        CANDIDATE_FREEZE_PATH,
        PARENT_FREEZE_PATH,
    ]
    files = [
        {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]
    holdout = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    stable = {
        "protocol_id": candidate["protocol_id"],
        "status": "frozen_before_any_prospective_seed_result",
        "seed_queue": holdout["protocol"]["holdout_seed_order"],
        "required_eligible_seeds": holdout["protocol"]["required_eligible_seeds_per_phase"],
        "selected_candidate_index": candidate["selected_candidate_index"],
        "files": files,
        "claim_boundary": holdout["claim_boundary"],
    }
    if RUN_FREEZE_PATH.exists():
        existing = json.loads(RUN_FREEZE_PATH.read_text(encoding="utf-8"))
        if {key: existing[key] for key in stable} != stable:
            raise RuntimeError("prospective run files changed after the run freeze")
        return existing
    RUN_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {**stable, "created_utc": datetime.now(timezone.utc).isoformat()}
    RUN_FREEZE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def aggregate_method(metrics: pd.DataFrame, method: str) -> dict[str, float]:
    rows = metrics[metrics["method"] == method]
    if len(rows) != 3:
        raise RuntimeError(f"expected three scenarios for {method}, found {len(rows)}")
    return {
        "unserved_energy_kwh": float(rows["unserved_energy_kwh"].sum()),
        "risk_adjusted_cost_yuan": float(rows["risk_adjusted_cost_yuan"].sum()),
        "diesel_fuel_l": float(rows["diesel_fuel_l"].sum()),
        "generator_startups": float(rows["generator_startups"].sum()),
        "max_decision_seconds": float(rows["max_decision_seconds"].max()),
    }


def exact_sign_flip_p(values: np.ndarray, tolerance: float = 1e-12) -> float:
    sample = np.asarray(values, dtype=float)
    nonzero = sample[np.abs(sample) > tolerance]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    extreme = 0
    total = 2 ** len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(float(np.mean(nonzero * np.asarray(signs))))
        extreme += int(statistic >= observed - tolerance)
    return float(extreme / total)


def bootstrap_mean_interval(
    values: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    if len(sample) == 0:
        raise ValueError("bootstrap sample must not be empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compare_actions(base_dir: Path, evidential_dir: Path) -> dict[str, object]:
    changed_rows = 0
    row_count = 0
    maximum_absolute_change = {column: 0.0 for column in ACTION_COLUMNS}
    files = sorted(base_dir.glob("trajectory_*_risk_soc_supervisory_mpc.csv"))
    if len(files) != 3:
        raise RuntimeError(f"expected three base RSS trajectories, found {len(files)}")
    for base_path in files:
        evidential_path = evidential_dir / base_path.name
        if not evidential_path.exists():
            raise FileNotFoundError(evidential_path)
        base = pd.read_csv(base_path)
        evidential = pd.read_csv(evidential_path)
        if len(base) != len(evidential):
            raise RuntimeError(f"trajectory length mismatch: {base_path.name}")
        missing = [
            column
            for column in ACTION_COLUMNS
            if column not in base.columns or column not in evidential.columns
        ]
        if missing:
            raise RuntimeError(f"action columns missing from {base_path.name}: {missing}")
        differences = (base[ACTION_COLUMNS] - evidential[ACTION_COLUMNS]).abs()
        changed_rows += int((differences.max(axis=1) > 1e-8).sum())
        row_count += int(len(differences))
        for column in ACTION_COLUMNS:
            maximum_absolute_change[column] = max(
                maximum_absolute_change[column], float(differences[column].max())
            )
    return {
        "trajectory_count": len(files),
        "row_count": row_count,
        "physical_action_changed_rows": changed_rows,
        "physical_action_change_rate": float(changed_rows / row_count),
        "maximum_absolute_change": maximum_absolute_change,
    }


def scenario_selection_matches(base_path: Path, evidential_path: Path) -> bool:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    evidential = json.loads(evidential_path.read_text(encoding="utf-8"))
    return all(
        int(base["scenarios"][name][key]) == int(evidential["scenarios"][name][key])
        for name in base["scenarios"]
        for key in ("start_index", "stop_index", "warmup_steps")
    )


def process_evidential_seed(seed: int, source: Path, force: bool = False) -> Path:
    output = OUTPUT_ROOT / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists() and not force:
        print("REUSE", result_path, flush=True)
        return result_path

    risk_dir = output / "risk"
    risk_metrics_path = risk_dir / "evidential_risk_metrics.json"
    risk_predictions_path = risk_dir / "evidential_risk_predictions.csv"
    if force or not (risk_metrics_path.exists() and risk_predictions_path.exists()):
        run_command(
            [
                sys.executable,
                "scripts/run_evidential_risk_fusion.py",
                "--input",
                str(source / "risk/risk_predictions.csv"),
                "--config",
                str(SELECTED_CONFIG_PATH),
                "--output-dir",
                str(risk_dir),
            ]
        )

    dispatch_dir = output / "dispatch"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    if force or not metrics_path.exists():
        run_command(
            [
                sys.executable,
                "scripts/run_v15_dispatch_benchmark.py",
                "--predictions",
                str(source / "source/forecast/forecast_predictions.npz"),
                "--prediction-keys",
                str(source / "source/forecast/forecast_prediction_keys.json"),
                "--config",
                str(source / "resolved_dispatch_config.yaml"),
                "--risk-signals",
                str(risk_predictions_path),
                "--artifact-dir",
                str(dispatch_dir),
            ]
        )

    if not scenario_selection_matches(
        source / "dispatch/scenario_selection.json",
        dispatch_dir / "scenario_selection.json",
    ):
        raise RuntimeError(f"evidential risk changed scenario selection for seed {seed}")

    base_metrics = pd.read_csv(source / "dispatch/dispatch_metrics.csv")
    evidential_metrics = pd.read_csv(metrics_path)
    baseline = aggregate_method(base_metrics, "Risk-SOC-Supervisory-MPC")
    evidential = aggregate_method(evidential_metrics, "Risk-SOC-Supervisory-MPC")
    delta = {key: evidential[key] - baseline[key] for key in baseline}
    action = compare_actions(source / "dispatch", dispatch_dir)
    physical = _audit_trajectories(dispatch_dir)
    risk_report = json.loads(risk_metrics_path.read_text(encoding="utf-8"))
    acceptance = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "evidential_holdout_acceptance"
    ]
    checks = {
        "scenario_selection_match": True,
        "unserved_not_worse_on_seed": delta["unserved_energy_kwh"]
        <= float(acceptance["aggregate_unserved_tolerance_kwh"]),
        "cost_ratio_within_seed_cap": evidential["risk_adjusted_cost_yuan"]
        / baseline["risk_adjusted_cost_yuan"]
        <= float(acceptance["maximum_aggregate_risk_adjusted_cost_ratio"]),
        "physical_feasibility": bool(physical["physical_pass"]),
    }
    source_risk = pd.read_csv(source / "risk/risk_predictions.csv")
    evidential_risk = pd.read_csv(risk_predictions_path)
    risk_changed = int(
        (
            source_risk["dispatch_risk_level"].to_numpy(int)
            != evidential_risk["dispatch_risk_level"].to_numpy(int)
        ).sum()
    )
    payload = {
        "analysis": "frozen_evidential_risk_prospective_holdout",
        "seed": seed,
        "eligible_before_controller_execution": True,
        "source_root": relative(source),
        "selected_candidate_index": json.loads(
            CANDIDATE_FREEZE_PATH.read_text(encoding="utf-8")
        )["selected_candidate_index"],
        "baseline_rss": baseline,
        "evidential_rss": evidential,
        "evidential_minus_baseline": delta,
        "risk_recognition": {
            "baseline_prediction": risk_report["baseline_prediction"],
            "baseline_dispatch_guard": risk_report["baseline_dispatch_guard"],
            "evidential_prediction": risk_report["evidential_prediction"],
            "evidential_dispatch_guard": risk_report["evidential_dispatch_guard"],
            "diagnostics": risk_report["diagnostics"],
            "dispatch_risk_changed_rows": risk_changed,
            "dispatch_risk_change_rate": float(risk_changed / len(source_risk)),
        },
        "action_audit": action,
        "physical_audit": physical,
        "checks": checks,
        "seed_gate_pass": bool(all(checks.values())),
        "claim_boundary": (
            "Prospective synthetic holdout; classification, action, and outcome "
            "effects are reported separately."
        ),
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [
        source / "eligibility.json",
        source / "result.json",
        source / "resolved_dispatch_config.yaml",
        source / "risk/risk_predictions.csv",
        risk_metrics_path,
        risk_predictions_path,
        metrics_path,
        dispatch_dir / "scenario_selection.json",
        result_path,
    ]
    (output / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "files": [
                    {
                        "path": relative(path),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"seed": seed, "output": str(result_path), "checks": checks}))
    return result_path


def summarize(result_paths: list[Path]) -> dict[str, object]:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    seeds = [int(result["seed"]) for result in results]
    unserved_delta = np.asarray(
        [result["evidential_minus_baseline"]["unserved_energy_kwh"] for result in results],
        dtype=float,
    )
    cost_delta = np.asarray(
        [result["evidential_minus_baseline"]["risk_adjusted_cost_yuan"] for result in results],
        dtype=float,
    )
    baseline_unserved = float(
        sum(result["baseline_rss"]["unserved_energy_kwh"] for result in results)
    )
    evidential_unserved = float(
        sum(result["evidential_rss"]["unserved_energy_kwh"] for result in results)
    )
    baseline_cost = float(
        sum(result["baseline_rss"]["risk_adjusted_cost_yuan"] for result in results)
    )
    evidential_cost = float(
        sum(result["evidential_rss"]["risk_adjusted_cost_yuan"] for result in results)
    )
    acceptance = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "evidential_holdout_acceptance"
    ]
    interval = bootstrap_mean_interval(
        unserved_delta,
        resamples=int(acceptance["bootstrap_resamples"]),
        seed=int(acceptance["bootstrap_seed"]),
    )
    total_action_rows = int(sum(r["action_audit"]["row_count"] for r in results))
    changed_action_rows = int(
        sum(r["action_audit"]["physical_action_changed_rows"] for r in results)
    )
    checks = {
        "exact_first_six_eligible_seeds": len(results) == 6,
        "aggregate_unserved_not_worse": evidential_unserved - baseline_unserved
        <= float(acceptance["aggregate_unserved_tolerance_kwh"]),
        "aggregate_cost_ratio_within_cap": evidential_cost / baseline_cost
        <= float(acceptance["maximum_aggregate_risk_adjusted_cost_ratio"]),
        "all_physical_audits_pass": all(
            bool(result["physical_audit"]["physical_pass"]) for result in results
        ),
        "nonzero_physical_action_change": changed_action_rows > 0,
    }
    rows: list[dict[str, object]] = []
    for result in results:
        recognition = result["risk_recognition"]
        rows.append(
            {
                "seed": result["seed"],
                "baseline_macro_f1": recognition["baseline_prediction"]["macro_f1"],
                "evidential_macro_f1": recognition["evidential_prediction"]["macro_f1"],
                "baseline_high_risk_recall": recognition["baseline_dispatch_guard"]["high_risk_recall"],
                "evidential_high_risk_recall": recognition["evidential_dispatch_guard"]["high_risk_recall"],
                "baseline_unserved_kwh": result["baseline_rss"]["unserved_energy_kwh"],
                "evidential_unserved_kwh": result["evidential_rss"]["unserved_energy_kwh"],
                "unserved_delta_kwh": result["evidential_minus_baseline"]["unserved_energy_kwh"],
                "cost_delta_yuan": result["evidential_minus_baseline"]["risk_adjusted_cost_yuan"],
                "action_change_rate": result["action_audit"]["physical_action_change_rate"],
                "mean_ignorance": recognition["diagnostics"]["mean_ignorance"],
                "mean_conflict": recognition["diagnostics"]["mean_conflict"],
                "seed_gate_pass": result["seed_gate_pass"],
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SUMMARY_CSV_PATH, index=False)
    if float(unserved_delta.mean()) < -1e-9:
        effect = "improved"
    elif float(unserved_delta.mean()) > 1e-9:
        effect = "worse"
    else:
        effect = "null_within_tolerance"
    payload = {
        "analysis": "V19 frozen evidential-risk prospective holdouts",
        "unit_of_analysis": "seed; three scenarios remain clustered within seed",
        "seeds": seeds,
        "n": len(seeds),
        "aggregate": {
            "baseline_unserved_energy_kwh": baseline_unserved,
            "evidential_unserved_energy_kwh": evidential_unserved,
            "evidential_minus_baseline_unserved_energy_kwh": evidential_unserved
            - baseline_unserved,
            "baseline_risk_adjusted_cost_yuan": baseline_cost,
            "evidential_risk_adjusted_cost_yuan": evidential_cost,
            "risk_adjusted_cost_ratio": evidential_cost / baseline_cost,
            "physical_action_changed_rows": changed_action_rows,
            "physical_action_total_rows": total_action_rows,
            "physical_action_change_rate": changed_action_rows / total_action_rows,
        },
        "paired_unserved_effect": {
            "mean_kwh_per_seed": float(unserved_delta.mean()),
            "sample_sd_kwh": float(unserved_delta.std(ddof=1)),
            "median_kwh": float(np.median(unserved_delta)),
            "seed_bootstrap_95_interval_kwh": list(interval),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(unserved_delta),
            "improved_tied_worse": [
                int((unserved_delta < -1e-9).sum()),
                int((np.abs(unserved_delta) <= 1e-9).sum()),
                int((unserved_delta > 1e-9).sum()),
            ],
            "interpretation": effect,
        },
        "paired_cost_effect": {
            "mean_yuan_per_seed": float(cost_delta.mean()),
            "sample_sd_yuan": float(cost_delta.std(ddof=1)),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(cost_delta),
        },
        "mean_risk_metrics": {
            key: float(frame[key].mean())
            for key in (
                "baseline_macro_f1",
                "evidential_macro_f1",
                "baseline_high_risk_recall",
                "evidential_high_risk_recall",
                "mean_ignorance",
                "mean_conflict",
            )
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "claim_boundary": [
            "Risk recognition, physical action, and outcome effects are distinct estimands.",
            "A null shortage effect cannot be described as control improvement.",
            "The evidence is prospective within the declared synthetic generator, not field validation.",
        ],
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_manifest(result_paths: list[Path]) -> None:
    paths = [
        PROTOCOL_PATH,
        HOLDOUT_CONFIG_PATH,
        CANDIDATE_FREEZE_PATH,
        RUN_FREEZE_PATH,
        SELECTED_CONFIG_PATH,
        Path(__file__).resolve(),
        SUMMARY_PATH,
        SUMMARY_CSV_PATH,
        *result_paths,
        *(path.parent / "evidence_manifest.json" for path in result_paths),
    ]
    payload = {
        "analysis": "V19 prospective evidential-risk evidence package",
        "file_count": len(paths),
        "files": [
            {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in paths
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    ensure_run_freeze()
    holdout = yaml.safe_load(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    queue = [int(v) for v in holdout["protocol"]["holdout_seed_order"]]
    required = int(holdout["protocol"]["required_eligible_seeds_per_phase"])
    eligible: list[int] = []
    result_paths: list[Path] = []
    for seed in queue:
        if len(eligible) >= required:
            break
        source = SOURCE_ROOT / f"seed_{seed}"
        eligibility_path = source / "eligibility.json"
        if not eligibility_path.exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_paper_v15_extension.py",
                    "--protocol",
                    str(HOLDOUT_CONFIG_PATH),
                    "--seed",
                    str(seed),
                ],
                allowed_returncodes={0, 2, 3},
            )
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if not bool(eligibility["eligible"]):
            print("INELIGIBLE", seed, flush=True)
            continue
        if not (source / "result.json").exists():
            raise RuntimeError(f"eligible source seed lacks result.json: {seed}")
        eligible.append(seed)
        if not args.source_only:
            result_paths.append(process_evidential_seed(seed, source, args.force))
    if len(eligible) != required:
        raise RuntimeError(f"only {len(eligible)}/{required} eligible seeds were completed")
    if args.source_only:
        print(json.dumps({"eligible_seeds": eligible, "status": "source_complete"}))
        return
    summary = summarize(result_paths)
    write_manifest(result_paths)
    print(
        json.dumps(
            {"eligible_seeds": eligible, "summary": str(SUMMARY_PATH), "status": summary["status"]}
        )
    )


if __name__ == "__main__":
    main()
