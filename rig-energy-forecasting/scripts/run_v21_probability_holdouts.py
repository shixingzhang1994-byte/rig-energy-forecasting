from __future__ import annotations

"""Run the frozen V21 probability-calibration candidate on prospective seeds."""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v19_evidential_holdouts import (  # noqa: E402
    _audit_trajectories,
    aggregate_method,
    compare_actions,
    scenario_selection_matches,
)


PROTOCOL_PATH = PROJECT_DIR / "configs/v21_probability_calibration_protocol.yaml"
SOURCE_CONFIG_PATH = PROJECT_DIR / "configs/v21_probability_calibration_holdouts.yaml"
CANDIDATE_CONFIG_PATH = PROJECT_DIR / "configs/v21_selected_probability_candidate.yaml"
V19_CONFIG_PATH = PROJECT_DIR / "artifacts/v19_evidential_risk/development_search/selected_evidential_config.yaml"
RUN_FREEZE_PATH = PROJECT_DIR / "artifacts/v21_probability_calibration/run_freeze.json"
SOURCE_ROOT = PROJECT_DIR / "artifacts/v21_probability_calibration/prospective_source"
OUTPUT_ROOT = PROJECT_DIR / "artifacts/v21_probability_calibration/prospective"
SUMMARY_PATH = OUTPUT_ROOT / "prospective_summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT / "prospective_seed_results.csv"
MANIFEST_PATH = OUTPUT_ROOT / "evidence_manifest.json"


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


def bootstrap_mean_interval(values: np.ndarray, *, resamples: int, seed: int) -> list[float]:
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))
    low, high = np.quantile(sample[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def ensure_run_freeze() -> dict[str, object]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = yaml.safe_load(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    if source["protocol"]["holdout_seed_order"] != protocol["protocol"][
        "prospective_holdout_seed_order"
    ]:
        raise RuntimeError("V21 source queue differs from scientific protocol")
    if source["protocol"]["required_eligible_seeds_per_phase"] != protocol[
        "protocol"
    ]["required_eligible_holdouts"]:
        raise RuntimeError("V21 required holdout count differs")
    paths = [
        PROTOCOL_PATH,
        SOURCE_CONFIG_PATH,
        CANDIDATE_CONFIG_PATH,
        V19_CONFIG_PATH,
        Path(__file__).resolve(),
        PROJECT_DIR / "src/rig_energy/risk/evidence_theory_v20.py",
        PROJECT_DIR / "scripts/run_v20_evidential_calibration.py",
        PROJECT_DIR / "scripts/run_evidential_risk_fusion.py",
        PROJECT_DIR / "tests/test_v21_probability_holdouts.py",
        PROJECT_DIR / "artifacts/v20_evidential_calibration/development_freeze.json",
        PROJECT_DIR / "artifacts/v20_evidential_calibration/development_audit.json",
        PROJECT_DIR / "artifacts/v20_evidential_calibration/development_search/candidate_results.json",
        PROJECT_DIR / "artifacts/v19_evidential_risk/risk_candidate_freeze.json",
        PROJECT_DIR / "artifacts/v15_generator_first_reserve/freeze_manifest.json",
    ]
    files = [
        {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]
    stable = {
        "protocol_id": protocol["protocol"]["id"],
        "status": "frozen_before_any_V21_holdout_result",
        "seed_queue": protocol["protocol"]["prospective_holdout_seed_order"],
        "required_eligible_seeds": protocol["protocol"]["required_eligible_holdouts"],
        "candidate_id": "V21-Dempster-T050",
        "files": files,
        "claim_boundary": protocol["claim_boundary"],
    }
    if RUN_FREEZE_PATH.exists():
        existing = json.loads(RUN_FREEZE_PATH.read_text(encoding="utf-8"))
        if existing != stable:
            raise RuntimeError("V21 run freeze differs from current files")
        return existing
    if SOURCE_ROOT.exists() and any(SOURCE_ROOT.iterdir()):
        raise RuntimeError("V21 source results exist before run freeze")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("V21 candidate results exist before run freeze")
    RUN_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_FREEZE_PATH.write_text(
        json.dumps(stable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stable


def probability_audit(path: Path, prefix: str) -> dict[str, float | bool]:
    frame = pd.read_csv(path)
    columns = [f"{prefix}_{name}" for name in ("normal", "watch", "warning", "severe")]
    probability = frame[columns].to_numpy(float)
    error = float(np.abs(probability.sum(axis=1) - 1.0).max())
    return {
        "maximum_absolute_row_sum_error": error,
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "pass": bool(error <= 1e-12 and probability.min() >= 0.0 and probability.max() <= 1.0),
    }


def process_seed(seed: int, source: Path, force: bool = False) -> Path:
    output = OUTPUT_ROOT / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists() and not force:
        print("REUSE", result_path, flush=True)
        return result_path

    yager_risk = output / "v19_yager_risk"
    candidate_risk = output / "candidate_risk"
    if force or not (yager_risk / "evidential_risk_metrics.json").exists():
        run_command(
            [
                sys.executable,
                "scripts/run_evidential_risk_fusion.py",
                "--input",
                str(source / "risk/risk_predictions.csv"),
                "--config",
                str(V19_CONFIG_PATH),
                "--output-dir",
                str(yager_risk),
            ]
        )
    if force or not (candidate_risk / "v20_risk_metrics.json").exists():
        run_command(
            [
                sys.executable,
                "scripts/run_v20_evidential_calibration.py",
                "--input",
                str(source / "risk/risk_predictions.csv"),
                "--config",
                str(CANDIDATE_CONFIG_PATH),
                "--output-dir",
                str(candidate_risk),
            ]
        )

    dispatches = {
        "v19_yager": (
            yager_risk / "evidential_risk_predictions.csv",
            output / "v19_yager_dispatch",
        ),
        "candidate": (
            candidate_risk / "v20_risk_predictions.csv",
            output / "candidate_dispatch",
        ),
    }
    for _, (signals, dispatch_dir) in dispatches.items():
        if force or not (dispatch_dir / "dispatch_metrics.csv").exists():
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
                    str(signals),
                    "--artifact-dir",
                    str(dispatch_dir),
                ]
            )
        if not scenario_selection_matches(
            source / "dispatch/scenario_selection.json",
            dispatch_dir / "scenario_selection.json",
        ):
            raise RuntimeError(f"scenario selection changed for seed {seed}")

    base_dispatch = source / "dispatch"
    yager_dispatch = dispatches["v19_yager"][1]
    candidate_dispatch = dispatches["candidate"][1]
    base = aggregate_method(pd.read_csv(base_dispatch / "dispatch_metrics.csv"), "Risk-SOC-Supervisory-MPC")
    yager = aggregate_method(pd.read_csv(yager_dispatch / "dispatch_metrics.csv"), "Risk-SOC-Supervisory-MPC")
    candidate = aggregate_method(pd.read_csv(candidate_dispatch / "dispatch_metrics.csv"), "Risk-SOC-Supervisory-MPC")
    yager_report = json.loads((yager_risk / "evidential_risk_metrics.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_risk / "v20_risk_metrics.json").read_text(encoding="utf-8"))
    candidate_prob_audit = probability_audit(
        candidate_risk / "v20_risk_predictions.csv", "v20_calibrated_prob"
    )
    raw = candidate_report["raw_evidential_prediction"]
    calibrated = candidate_report["calibrated_evidential_prediction"]
    v19_probability = yager_report["evidential_prediction"]
    baseline_probability = candidate_report["baseline_prediction"]
    payload = {
        "seed": seed,
        "eligible_before_controller_execution": True,
        "probability": {
            "frozen_RSS": baseline_probability,
            "raw_V19_Yager": v19_probability,
            "raw_Dempster": raw,
            "calibrated_Dempster": calibrated,
            "calibration_brier_delta": calibrated["multiclass_brier"] - raw["multiclass_brier"],
            "rule_brier_delta": raw["multiclass_brier"] - v19_probability["multiclass_brier"],
            "total_brier_delta_vs_Yager": calibrated["multiclass_brier"] - v19_probability["multiclass_brier"],
            "candidate_brier_delta_vs_RSS": calibrated["multiclass_brier"] - baseline_probability["multiclass_brier"],
            "candidate_macro_f1_delta_vs_RSS": calibrated["macro_f1"] - baseline_probability["macro_f1"],
        },
        "diagnostics": candidate_report["diagnostics"],
        "decision_invariance": candidate_report["decision_invariance"],
        "probability_numerical_audit": candidate_prob_audit,
        "dispatch": {
            "frozen_RSS": base,
            "V19_Yager_RSS": yager,
            "V21_candidate_RSS": candidate,
            "candidate_minus_RSS": {key: candidate[key] - base[key] for key in base},
            "candidate_minus_Yager": {key: candidate[key] - yager[key] for key in yager},
        },
        "action_audit": {
            "Yager_vs_RSS": compare_actions(base_dispatch, yager_dispatch),
            "candidate_vs_RSS": compare_actions(base_dispatch, candidate_dispatch),
            "candidate_vs_Yager": compare_actions(yager_dispatch, candidate_dispatch),
        },
        "physical_audit": {
            "Yager": _audit_trajectories(yager_dispatch),
            "candidate": _audit_trajectories(candidate_dispatch),
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [
        source / "eligibility.json",
        source / "result.json",
        yager_risk / "evidential_risk_metrics.json",
        yager_risk / "evidential_risk_predictions.csv",
        candidate_risk / "v20_risk_metrics.json",
        candidate_risk / "v20_risk_predictions.csv",
        yager_dispatch / "dispatch_metrics.csv",
        candidate_dispatch / "dispatch_metrics.csv",
        result_path,
    ]
    (output / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "files": [
                    {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                    for path in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_path


def effect_summary(values: np.ndarray, protocol: dict) -> dict[str, object]:
    sample = np.asarray(values, dtype=float)
    stats = protocol["acceptance"]["statistics"]
    return {
        "mean": float(sample.mean()),
        "sample_sd": float(sample.std(ddof=1)),
        "median": float(np.median(sample)),
        "improved_tied_worse": [
            int((sample < -1e-12).sum()),
            int((np.abs(sample) <= 1e-12).sum()),
            int((sample > 1e-12).sum()),
        ],
        "seed_bootstrap_95_interval": bootstrap_mean_interval(
            sample,
            resamples=int(stats["seed_bootstrap_resamples"]),
            seed=int(stats["bootstrap_seed"]),
        ),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(sample),
    }


def summarize(paths: list[Path]) -> dict[str, object]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = []
    for result in results:
        probability = result["probability"]
        dispatch = result["dispatch"]
        rows.append(
            {
                "seed": result["seed"],
                "RSS_brier": probability["frozen_RSS"]["multiclass_brier"],
                "Yager_raw_brier": probability["raw_V19_Yager"]["multiclass_brier"],
                "Dempster_raw_brier": probability["raw_Dempster"]["multiclass_brier"],
                "Dempster_calibrated_brier": probability["calibrated_Dempster"]["multiclass_brier"],
                "calibration_brier_delta": probability["calibration_brier_delta"],
                "rule_brier_delta": probability["rule_brier_delta"],
                "total_brier_delta_vs_Yager": probability["total_brier_delta_vs_Yager"],
                "candidate_brier_delta_vs_RSS": probability["candidate_brier_delta_vs_RSS"],
                "candidate_macro_f1_delta_vs_RSS": probability["candidate_macro_f1_delta_vs_RSS"],
                "candidate_high_risk_recall": probability["calibrated_Dempster"]["high_risk_recall"],
                "candidate_severe_recall": probability["calibrated_Dempster"]["severe_recall"],
                "candidate_unserved_delta_kwh": dispatch["candidate_minus_RSS"]["unserved_energy_kwh"],
                "candidate_cost_delta_yuan": dispatch["candidate_minus_RSS"]["risk_adjusted_cost_yuan"],
                "candidate_action_change_rows": result["action_audit"]["candidate_vs_RSS"]["physical_action_changed_rows"],
                "candidate_vs_Yager_action_change_rows": result["action_audit"]["candidate_vs_Yager"]["physical_action_changed_rows"],
                "mean_conflict": result["diagnostics"]["mean_conflict"],
                "mean_ignorance": result["diagnostics"]["mean_ignorance"],
                "fallback_rows": result["diagnostics"]["total_conflict_fallback_rows"],
            }
        )
    frame = pd.DataFrame(rows)
    acceptance = protocol["acceptance"]
    baseline_cost = float(sum(result["dispatch"]["frozen_RSS"]["risk_adjusted_cost_yuan"] for result in results))
    candidate_cost = float(sum(result["dispatch"]["V21_candidate_RSS"]["risk_adjusted_cost_yuan"] for result in results))
    baseline_unserved = float(sum(result["dispatch"]["frozen_RSS"]["unserved_energy_kwh"] for result in results))
    candidate_unserved = float(sum(result["dispatch"]["V21_candidate_RSS"]["unserved_energy_kwh"] for result in results))
    checks = {
        "exact_first_six_eligible_seeds": len(results) == int(protocol["protocol"]["required_eligible_holdouts"]),
        "calibration_improves_required_seed_count": int((frame["calibration_brier_delta"] < 0.0).sum())
        >= int(acceptance["probability"]["require_calibration_brier_improvement_on_at_least_seeds"]),
        "mean_calibration_brier_improves": float(frame["calibration_brier_delta"].mean())
        < float(acceptance["probability"]["require_mean_calibration_brier_delta_below"]),
        "candidate_mean_brier_vs_RSS_within_margin": float(frame["candidate_brier_delta_vs_RSS"].mean())
        <= float(acceptance["probability"]["require_mean_candidate_brier_not_worse_than_RSS_by_more_than"]),
        "macro_f1_noninferior": float(frame["candidate_macro_f1_delta_vs_RSS"].mean())
        >= float(acceptance["recognition_noninferiority"]["require_mean_macro_f1_delta_vs_RSS_at_least"]),
        "high_risk_recall_gate": bool(
            (frame["candidate_high_risk_recall"] >= float(acceptance["recognition_noninferiority"]["require_every_seed_high_risk_recall_at_least"])).all()
        ),
        "severe_recall_gate": bool(
            (frame["candidate_severe_recall"] >= float(acceptance["recognition_noninferiority"]["require_every_seed_severe_recall_at_least"])).all()
        ),
        "aggregate_unserved_not_worse": candidate_unserved - baseline_unserved
        <= float(acceptance["control_safety"]["require_aggregate_unserved_not_worse_than_RSS_kwh"]),
        "aggregate_cost_ratio_within_cap": candidate_cost / baseline_cost
        <= float(acceptance["control_safety"]["maximum_risk_adjusted_cost_ratio_vs_RSS"]),
        "all_physical_audits_pass": all(
            result["physical_audit"]["Yager"]["physical_pass"]
            and result["physical_audit"]["candidate"]["physical_pass"]
            for result in results
        ),
        "probability_numerical_audits_pass": all(
            result["probability_numerical_audit"]["pass"] for result in results
        ),
        "zero_total_conflict_fallback_rows": int(frame["fallback_rows"].sum()) == 0,
    }
    payload = {
        "analysis": "V21 frozen probability-calibration prospective holdouts",
        "unit_of_analysis": "seed; three dispatch scenarios clustered within seed",
        "seeds": [int(value) for value in frame["seed"]],
        "n": len(frame),
        "probability_effects": {
            "calibration_effect_calibrated_minus_raw_Dempster": effect_summary(frame["calibration_brier_delta"].to_numpy(), protocol),
            "rule_effect_raw_Dempster_minus_raw_Yager": effect_summary(frame["rule_brier_delta"].to_numpy(), protocol),
            "total_effect_calibrated_Dempster_minus_raw_Yager": effect_summary(frame["total_brier_delta_vs_Yager"].to_numpy(), protocol),
            "candidate_minus_RSS": effect_summary(frame["candidate_brier_delta_vs_RSS"].to_numpy(), protocol),
        },
        "recognition": {
            "macro_f1_delta_vs_RSS": effect_summary(frame["candidate_macro_f1_delta_vs_RSS"].to_numpy(), protocol),
            "mean_candidate_high_risk_recall": float(frame["candidate_high_risk_recall"].mean()),
            "mean_candidate_severe_recall": float(frame["candidate_severe_recall"].mean()),
        },
        "control": {
            "baseline_unserved_energy_kwh": baseline_unserved,
            "candidate_unserved_energy_kwh": candidate_unserved,
            "candidate_minus_baseline_unserved_energy_kwh": candidate_unserved - baseline_unserved,
            "unserved_effect": effect_summary(frame["candidate_unserved_delta_kwh"].to_numpy(), protocol),
            "baseline_risk_adjusted_cost_yuan": baseline_cost,
            "candidate_risk_adjusted_cost_yuan": candidate_cost,
            "risk_adjusted_cost_ratio": candidate_cost / baseline_cost,
            "cost_effect": effect_summary(frame["candidate_cost_delta_yuan"].to_numpy(), protocol),
            "candidate_action_changed_rows": int(frame["candidate_action_change_rows"].sum()),
            "candidate_vs_Yager_action_changed_rows": int(frame["candidate_vs_Yager_action_change_rows"].sum()),
        },
        "diagnostics": {
            "mean_conflict": float(frame["mean_conflict"].mean()),
            "mean_ignorance": float(frame["mean_ignorance"].mean()),
            "total_conflict_fallback_rows": int(frame["fallback_rows"].sum()),
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "multiplicity_note": "All endpoint-wise p-values are descriptive; no multiplicity-adjusted family-wise claim.",
        "claim_boundary": protocol["claim_boundary"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SUMMARY_CSV_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_manifest(paths: list[Path]) -> None:
    files = [
        PROTOCOL_PATH,
        SOURCE_CONFIG_PATH,
        CANDIDATE_CONFIG_PATH,
        RUN_FREEZE_PATH,
        SUMMARY_PATH,
        SUMMARY_CSV_PATH,
        *paths,
        *(path.parent / "evidence_manifest.json" for path in paths),
    ]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "analysis": "V21 prospective probability-calibration evidence package",
                "file_count": len(files),
                "files": [
                    {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                    for path in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_run_freeze()
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    required = int(protocol["protocol"]["required_eligible_holdouts"])
    eligible = []
    result_paths = []
    for seed in protocol["protocol"]["prospective_holdout_seed_order"]:
        if len(eligible) >= required:
            break
        seed = int(seed)
        source = SOURCE_ROOT / f"seed_{seed}"
        eligibility_path = source / "eligibility.json"
        if not eligibility_path.exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_paper_v15_extension.py",
                    "--protocol",
                    str(SOURCE_CONFIG_PATH),
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
            raise RuntimeError(f"eligible seed lacks source result: {seed}")
        eligible.append(seed)
        if not args.source_only:
            result_paths.append(process_seed(seed, source, args.force))
    if len(eligible) != required:
        raise RuntimeError(f"only {len(eligible)}/{required} eligible V21 seeds")
    if args.source_only:
        print(json.dumps({"eligible_seeds": eligible, "status": "source_complete"}))
        return
    summary = summarize(result_paths)
    write_manifest(result_paths)
    print(json.dumps({"eligible_seeds": eligible, "status": summary["status"], "summary": str(SUMMARY_PATH)}))


if __name__ == "__main__":
    main()
