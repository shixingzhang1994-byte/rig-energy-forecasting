from __future__ import annotations

"""Independent, read-only reconstruction of the frozen V21 holdout evidence."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR / "artifacts/v21_probability_calibration"
SOURCE_ROOT = ROOT / "prospective_source"
RESULT_ROOT = ROOT / "prospective"
PROTOCOL_PATH = PROJECT_DIR / "configs/v21_probability_calibration_protocol.yaml"
FREEZE_PATH = ROOT / "run_freeze.json"
CLARIFICATION_PATH = ROOT / "execution_clarification_1.json"
RUNNER_MANIFEST_PATH = RESULT_ROOT / "evidence_manifest.json"
OUTPUT_PATH = ROOT / "independent_audit.json"
AUDIT_MANIFEST_PATH = ROOT / "independent_audit_manifest.json"
CLASS_NAMES = ("normal", "watch", "warning", "severe")
ACTION_COLUMNS = (
    "grid_kw",
    "generator_kw",
    "generator_units_on",
    "storage_kw",
    "unserved_kw",
    "generator_startup_units",
    "generator_shutdown_units",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_DIR))


def verify_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    changed: list[str] = []
    for item in manifest["files"]:
        target = PROJECT_DIR / item["path"]
        if (
            not target.exists()
            or sha256(target) != item["sha256"]
            or target.stat().st_size != int(item["size_bytes"])
        ):
            changed.append(item["path"])
    return changed


def macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for level in range(4):
        tp = int(((truth == level) & (prediction == level)).sum())
        fp = int(((truth != level) & (prediction == level)).sum())
        fn = int(((truth == level) & (prediction != level)).sum())
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def probability_metrics(
    truth: np.ndarray, probability: np.ndarray
) -> dict[str, float | int | bool]:
    prediction = probability.argmax(axis=1)
    one_hot = np.eye(4, dtype=float)[truth]
    high_denominator = int((truth >= 2).sum())
    severe_denominator = int((truth == 3).sum())
    row_error = float(np.abs(probability.sum(axis=1) - 1.0).max())
    return {
        "row_count": int(len(truth)),
        "multiclass_brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "macro_f1": macro_f1(truth, prediction),
        "high_risk_recall": float(
            ((truth >= 2) & (prediction >= 2)).sum() / high_denominator
        ),
        "severe_recall": float(
            ((truth == 3) & (prediction == 3)).sum() / severe_denominator
        ),
        "maximum_absolute_row_sum_error": row_error,
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "probability_valid": bool(
            row_error <= 1e-12
            and probability.min() >= 0.0
            and probability.max() <= 1.0
        ),
    }


def exact_sign_flip_p(values: np.ndarray, tolerance: float = 1e-12) -> float:
    nonzero = np.asarray(values, dtype=float)
    nonzero = nonzero[np.abs(nonzero) > tolerance]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(float(np.mean(nonzero * np.asarray(signs, dtype=float))))
        extreme += int(statistic >= observed - tolerance)
    return float(extreme / (2 ** len(nonzero)))


def bootstrap_interval(
    values: np.ndarray, *, resamples: int, seed: int
) -> list[float]:
    sample = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(sample), size=(resamples, len(sample)))
    return [
        float(value)
        for value in np.quantile(sample[indices].mean(axis=1), [0.025, 0.975])
    ]


def effect_summary(
    values: np.ndarray,
    *,
    lower_is_better: bool,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    sample = np.asarray(values, dtype=float)
    favorable = sample < -1e-12 if lower_is_better else sample > 1e-12
    adverse = sample > 1e-12 if lower_is_better else sample < -1e-12
    return {
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "mean": float(sample.mean()),
        "sample_sd": float(sample.std(ddof=1)),
        "median": float(np.median(sample)),
        "favorable_tied_adverse": [
            int(favorable.sum()),
            int((np.abs(sample) <= 1e-12).sum()),
            int(adverse.sum()),
        ],
        "seed_bootstrap_95_interval": bootstrap_interval(
            sample, resamples=resamples, seed=bootstrap_seed
        ),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(sample),
    }


def aggregate_rss_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    rows = frame[frame["method"] == "Risk-SOC-Supervisory-MPC"]
    if len(rows) != 3:
        raise RuntimeError(f"expected three RSS scenario rows in {path}")
    return {
        "unserved_energy_kwh": float(rows["unserved_energy_kwh"].sum()),
        "risk_adjusted_cost_yuan": float(rows["risk_adjusted_cost_yuan"].sum()),
    }


def compare_actions(left_root: Path, right_root: Path) -> dict[str, object]:
    files = sorted(left_root.glob("trajectory_*_risk_soc_supervisory_mpc.csv"))
    if len(files) != 3:
        raise RuntimeError(f"expected three RSS trajectories in {left_root}")
    rows = 0
    changed = 0
    maximum = {column: 0.0 for column in ACTION_COLUMNS}
    for left_path in files:
        right_path = right_root / left_path.name
        left = pd.read_csv(left_path)
        right = pd.read_csv(right_path)
        if len(left) != len(right):
            raise RuntimeError(f"trajectory length mismatch: {left_path.name}")
        difference = (left[list(ACTION_COLUMNS)] - right[list(ACTION_COLUMNS)]).abs()
        changed += int((difference.max(axis=1) > 1e-8).sum())
        rows += len(difference)
        for column in ACTION_COLUMNS:
            maximum[column] = max(maximum[column], float(difference[column].max()))
    return {
        "trajectory_count": len(files),
        "row_count": int(rows),
        "physical_action_changed_rows": int(changed),
        "physical_action_change_rate": float(changed / rows),
        "maximum_absolute_change": maximum,
    }


def physical_audit(root: Path) -> dict[str, object]:
    files = sorted(root.glob("trajectory_*.csv"))
    if len(files) != 21:
        raise RuntimeError(f"expected 21 controller-scenario trajectories in {root}")
    result: dict[str, float | int | bool] = {
        "trajectory_count": len(files),
        "max_power_balance_error_kw": 0.0,
        "max_grid_capacity_violation_kw": 0.0,
        "max_generator_capacity_violation_kw": 0.0,
        "max_storage_capacity_violation_kw": 0.0,
        "soc_violation_count": 0,
        "simultaneous_charge_discharge_count": 0,
        "noninteger_commitment_count": 0,
    }
    for path in files:
        frame = pd.read_csv(path)
        balance = (
            frame["grid_kw"]
            + frame["generator_kw"]
            + frame["storage_kw"]
            + frame["unserved_kw"]
            - frame["spill_kw"]
            - frame["load_kw"]
        )
        result["max_power_balance_error_kw"] = max(
            float(result["max_power_balance_error_kw"]), float(balance.abs().max())
        )
        for key, value in (
            (
                "max_grid_capacity_violation_kw",
                frame["grid_kw"] - frame["grid_available_capacity_kw"],
            ),
            (
                "max_generator_capacity_violation_kw",
                frame["generator_kw"] - frame["generator_available_capacity_kw"],
            ),
            (
                "max_storage_capacity_violation_kw",
                frame["storage_kw"].abs() - frame["storage_available_power_kw"],
            ),
        ):
            result[key] = max(float(result[key]), float(np.maximum(value, 0.0).max()))
        result["soc_violation_count"] = int(result["soc_violation_count"]) + int(
            ((frame["soc"] < 0.15 - 1e-9) | (frame["soc"] > 0.90 + 1e-9)).sum()
        )
        result["simultaneous_charge_discharge_count"] = int(
            result["simultaneous_charge_discharge_count"]
        ) + int(((frame["charge_kw"] > 1e-8) & (frame["discharge_kw"] > 1e-8)).sum())
        result["noninteger_commitment_count"] = int(
            result["noninteger_commitment_count"]
        ) + int(
            (
                np.abs(frame["generator_units_on"] - np.rint(frame["generator_units_on"]))
                > 1e-9
            ).sum()
        )
    result["physical_pass"] = bool(
        float(result["max_power_balance_error_kw"]) <= 1e-8
        and float(result["max_grid_capacity_violation_kw"]) <= 1e-8
        and float(result["max_generator_capacity_violation_kw"]) <= 1e-8
        and float(result["max_storage_capacity_violation_kw"]) <= 1e-8
        and int(result["soc_violation_count"]) == 0
        and int(result["simultaneous_charge_discharge_count"]) == 0
        and int(result["noninteger_commitment_count"]) == 0
    )
    return result


def assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise RuntimeError(f"{label}: reconstructed {actual}, reported {expected}")


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_changed = []
    for item in freeze["files"]:
        target = PROJECT_DIR / item["path"]
        if not target.exists() or sha256(target) != item["sha256"]:
            freeze_changed.append(item["path"])
    if freeze_changed:
        raise RuntimeError("V21 frozen inputs changed: " + ", ".join(freeze_changed))
    if verify_manifest(RUNNER_MANIFEST_PATH):
        raise RuntimeError("runner evidence manifest hash failure")

    clarification = json.loads(CLARIFICATION_PATH.read_text(encoding="utf-8"))
    if clarification["original_run_freeze"]["sha256"] != sha256(FREEZE_PATH):
        raise RuntimeError("execution clarification does not reference the original freeze")
    for item in clarification["supporting_files"]:
        target = PROJECT_DIR / item["path"]
        if sha256(target) != item["sha256"]:
            raise RuntimeError(f"clarification support changed: {item['path']}")
    eligibility_target = PROJECT_DIR / clarification["eligibility_record"]["path"]
    if sha256(eligibility_target) != clarification["eligibility_record"]["sha256"]:
        raise RuntimeError("clarified eligibility record changed")

    queue = [int(seed) for seed in protocol["protocol"]["prospective_holdout_seed_order"]]
    required = int(protocol["protocol"]["required_eligible_holdouts"])
    eligible: list[int] = []
    screened: list[dict[str, object]] = []
    eligibility_paths: list[Path] = []
    for seed in queue:
        if len(eligible) >= required:
            break
        path = SOURCE_ROOT / f"seed_{seed}/eligibility.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        eligibility_paths.append(path)
        is_eligible = bool(record["eligible"])
        screened.append({"seed": seed, "eligible": is_eligible})
        if is_eligible:
            eligible.append(seed)
    expected = [20261040, 20261041, 20261042, 20261045, 20261046, 20261047]
    if eligible != expected:
        raise RuntimeError(f"unexpected V21 eligible seed set: {eligible}")

    seed_rows: list[dict[str, object]] = []
    action_rows_vs_rss = 0
    action_rows_vs_yager = 0
    physical_reports: list[dict[str, object]] = []
    baseline_unserved = candidate_unserved = 0.0
    baseline_cost = candidate_cost = 0.0
    for seed in eligible:
        source = SOURCE_ROOT / f"seed_{seed}"
        root = RESULT_ROOT / f"seed_{seed}"
        manifest_path = root / "evidence_manifest.json"
        changed = verify_manifest(manifest_path)
        if changed:
            raise RuntimeError(f"seed {seed} manifest hash failure: {changed}")
        reported = json.loads((root / "result.json").read_text(encoding="utf-8"))
        baseline_frame = pd.read_csv(source / "risk/risk_predictions.csv")
        yager_frame = pd.read_csv(root / "v19_yager_risk/evidential_risk_predictions.csv")
        candidate_frame = pd.read_csv(root / "candidate_risk/v20_risk_predictions.csv")
        truth = baseline_frame["true_risk_level"].to_numpy(int)
        for frame, label in ((yager_frame, "Yager"), (candidate_frame, "candidate")):
            if not np.array_equal(frame["true_risk_level"].to_numpy(int), truth):
                raise RuntimeError(f"seed {seed} {label} truth rows changed")
            if not np.array_equal(
                frame["sample_index"].to_numpy(int),
                baseline_frame["sample_index"].to_numpy(int),
            ):
                raise RuntimeError(f"seed {seed} {label} sample order changed")
        arrays = {
            "RSS": baseline_frame[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(float),
            "Yager_raw": yager_frame[
                [f"evidential_prob_{name}" for name in CLASS_NAMES]
            ].to_numpy(float),
            "Dempster_raw": candidate_frame[
                [f"v20_raw_prob_{name}" for name in CLASS_NAMES]
            ].to_numpy(float),
            "Dempster_calibrated": candidate_frame[
                [f"v20_calibrated_prob_{name}" for name in CLASS_NAMES]
            ].to_numpy(float),
        }
        metrics = {name: probability_metrics(truth, values) for name, values in arrays.items()}
        report_probability = reported["probability"]
        report_keys = {
            "RSS": "frozen_RSS",
            "Yager_raw": "raw_V19_Yager",
            "Dempster_raw": "raw_Dempster",
            "Dempster_calibrated": "calibrated_Dempster",
        }
        for name, report_key in report_keys.items():
            for metric in ("multiclass_brier", "macro_f1", "high_risk_recall", "severe_recall"):
                assert_close(
                    float(metrics[name][metric]),
                    float(report_probability[report_key][metric]),
                    f"seed {seed} {name} {metric}",
                )
        raw_argmax = arrays["Dempster_raw"].argmax(axis=1)
        calibrated_argmax = arrays["Dempster_calibrated"].argmax(axis=1)
        argmax_changes = int((raw_argmax != calibrated_argmax).sum())
        if argmax_changes != int(reported["decision_invariance"]["calibration_changed_argmax_rows"]):
            raise RuntimeError(f"seed {seed} calibration argmax mismatch")
        base_dispatch = source / "dispatch"
        yager_dispatch = root / "v19_yager_dispatch"
        candidate_dispatch = root / "candidate_dispatch"
        action_vs_rss = compare_actions(base_dispatch, candidate_dispatch)
        action_vs_yager = compare_actions(yager_dispatch, candidate_dispatch)
        action_rows_vs_rss += int(action_vs_rss["physical_action_changed_rows"])
        action_rows_vs_yager += int(action_vs_yager["physical_action_changed_rows"])
        yager_physical = physical_audit(yager_dispatch)
        candidate_physical = physical_audit(candidate_dispatch)
        physical_reports.extend([yager_physical, candidate_physical])
        base_dispatch_metrics = aggregate_rss_metrics(base_dispatch / "dispatch_metrics.csv")
        candidate_dispatch_metrics = aggregate_rss_metrics(
            candidate_dispatch / "dispatch_metrics.csv"
        )
        baseline_unserved += base_dispatch_metrics["unserved_energy_kwh"]
        candidate_unserved += candidate_dispatch_metrics["unserved_energy_kwh"]
        baseline_cost += base_dispatch_metrics["risk_adjusted_cost_yuan"]
        candidate_cost += candidate_dispatch_metrics["risk_adjusted_cost_yuan"]
        seed_rows.append(
            {
                "seed": seed,
                "metrics": metrics,
                "calibration_brier_delta": float(
                    metrics["Dempster_calibrated"]["multiclass_brier"]
                    - metrics["Dempster_raw"]["multiclass_brier"]
                ),
                "rule_brier_delta": float(
                    metrics["Dempster_raw"]["multiclass_brier"]
                    - metrics["Yager_raw"]["multiclass_brier"]
                ),
                "total_brier_delta_vs_yager": float(
                    metrics["Dempster_calibrated"]["multiclass_brier"]
                    - metrics["Yager_raw"]["multiclass_brier"]
                ),
                "candidate_brier_delta_vs_rss": float(
                    metrics["Dempster_calibrated"]["multiclass_brier"]
                    - metrics["RSS"]["multiclass_brier"]
                ),
                "macro_f1_delta_vs_rss": float(
                    metrics["Dempster_calibrated"]["macro_f1"]
                    - metrics["RSS"]["macro_f1"]
                ),
                "calibration_argmax_changed_rows": argmax_changes,
                "action_vs_rss": action_vs_rss,
                "action_vs_yager": action_vs_yager,
                "candidate_dispatch": candidate_dispatch_metrics,
                "baseline_dispatch": base_dispatch_metrics,
                "candidate_physical": candidate_physical,
                "yager_physical": yager_physical,
            }
        )

    statistics = protocol["acceptance"]["statistics"]
    resamples = int(statistics["seed_bootstrap_resamples"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    calibration_delta = np.asarray([row["calibration_brier_delta"] for row in seed_rows])
    rule_delta = np.asarray([row["rule_brier_delta"] for row in seed_rows])
    total_delta = np.asarray([row["total_brier_delta_vs_yager"] for row in seed_rows])
    rss_delta = np.asarray([row["candidate_brier_delta_vs_rss"] for row in seed_rows])
    macro_delta = np.asarray([row["macro_f1_delta_vs_rss"] for row in seed_rows])
    acceptance = protocol["acceptance"]
    high_recall = np.asarray(
        [row["metrics"]["Dempster_calibrated"]["high_risk_recall"] for row in seed_rows]
    )
    severe_recall = np.asarray(
        [row["metrics"]["Dempster_calibrated"]["severe_recall"] for row in seed_rows]
    )
    all_probability_valid = all(
        bool(metric["probability_valid"])
        for row in seed_rows
        for metric in row["metrics"].values()
    )
    all_candidate_probability_valid = all(
        bool(row["metrics"][name]["probability_valid"])
        for row in seed_rows
        for name in ("Yager_raw", "Dempster_raw", "Dempster_calibrated")
    )
    maximum_rss_export_row_sum_error = max(
        float(row["metrics"]["RSS"]["maximum_absolute_row_sum_error"])
        for row in seed_rows
    )
    fallback_rows = int(
        sum(
            pd.read_csv(
                RESULT_ROOT / f"seed_{seed}/candidate_risk/v20_risk_predictions.csv"
            )["v20_total_conflict_fallback"].astype(bool).sum()
            for seed in eligible
        )
    )
    checks = {
        "frozen_input_hashes_pass": True,
        "runner_evidence_hashes_pass": True,
        "execution_clarification_hashes_pass": True,
        "exact_first_six_eligible_seeds": eligible == expected,
        "calibration_improves_at_least_five_seeds": int((calibration_delta < 0).sum()) >= 5,
        "mean_calibration_brier_improves": float(calibration_delta.mean()) < 0.0,
        "mean_candidate_brier_vs_rss_within_margin": float(rss_delta.mean()) <= 0.005,
        "macro_f1_noninferior": float(macro_delta.mean()) >= -0.01,
        "every_seed_high_risk_recall_at_least_0_97": bool((high_recall >= 0.97).all()),
        "every_seed_severe_recall_at_least_0_95": bool((severe_recall >= 0.95).all()),
        "aggregate_unserved_not_worse": candidate_unserved - baseline_unserved <= 1e-9,
        "aggregate_cost_ratio_within_cap": candidate_cost / baseline_cost <= 1.005,
        "all_physical_audits_pass": all(bool(item["physical_pass"]) for item in physical_reports),
        "candidate_probability_rows_valid": all_candidate_probability_valid,
        "zero_total_conflict_fallback_rows": fallback_rows == 0,
        "zero_calibration_argmax_changes": sum(
            int(row["calibration_argmax_changed_rows"]) for row in seed_rows
        )
        == 0,
    }
    runner_summary = json.loads(
        (RESULT_ROOT / "prospective_summary.json").read_text(encoding="utf-8")
    )
    payload = {
        "analysis": "independent read-only reconstruction of V21 prospective holdouts",
        "protocol_id": protocol["protocol"]["id"],
        "run_freeze_sha256": sha256(FREEZE_PATH),
        "screened_seed_queue_prefix": screened,
        "eligible_seeds": eligible,
        "n": len(eligible),
        "per_seed": seed_rows,
        "effects": {
            "calibration_calibrated_minus_raw_dempster": effect_summary(
                calibration_delta,
                lower_is_better=True,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "conflict_rule_raw_dempster_minus_raw_yager": effect_summary(
                rule_delta,
                lower_is_better=True,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "total_calibrated_dempster_minus_raw_yager": effect_summary(
                total_delta,
                lower_is_better=True,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "candidate_minus_rss_brier": effect_summary(
                rss_delta,
                lower_is_better=True,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "candidate_minus_rss_macro_f1": effect_summary(
                macro_delta,
                lower_is_better=False,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            ),
        },
        "recognition": {
            "mean_high_risk_recall": float(high_recall.mean()),
            "minimum_high_risk_recall": float(high_recall.min()),
            "mean_severe_recall": float(severe_recall.mean()),
            "minimum_severe_recall": float(severe_recall.min()),
        },
        "control": {
            "baseline_unserved_energy_kwh": baseline_unserved,
            "candidate_unserved_energy_kwh": candidate_unserved,
            "candidate_minus_baseline_unserved_energy_kwh": candidate_unserved - baseline_unserved,
            "baseline_risk_adjusted_cost_yuan": baseline_cost,
            "candidate_risk_adjusted_cost_yuan": candidate_cost,
            "candidate_minus_baseline_cost_yuan": candidate_cost - baseline_cost,
            "candidate_to_baseline_cost_ratio": candidate_cost / baseline_cost,
            "candidate_vs_rss_action_changed_rows": action_rows_vs_rss,
            "candidate_vs_yager_action_changed_rows": action_rows_vs_yager,
        },
        "numerical_and_physical": {
            "total_conflict_fallback_rows": fallback_rows,
        "candidate_probability_rows_valid": all_candidate_probability_valid,
        "all_compared_exported_probability_rows_valid_at_1e_12": all_probability_valid,
        "maximum_rss_export_row_sum_error": maximum_rss_export_row_sum_error,
        "rss_export_rows_valid_at_2e_7_serialization_tolerance": bool(
            maximum_rss_export_row_sum_error <= 2e-7
        ),
            "all_physical_audits_pass": all(
                bool(item["physical_pass"]) for item in physical_reports
            ),
            "maximum_power_balance_error_kw": max(
                float(item["max_power_balance_error_kw"]) for item in physical_reports
            ),
        },
        "checks": checks,
        "status": (
            "pass_with_nonblocking_reporting_findings"
            if all(checks.values())
            else "fail"
        ),
        "runner_reporting_issue": {
            "field": "recognition.macro_f1_delta_vs_RSS.improved_tied_worse",
            "runner_value": runner_summary["recognition"]["macro_f1_delta_vs_RSS"][
                "improved_tied_worse"
            ],
            "corrected_value": effect_summary(
                macro_delta,
                lower_is_better=False,
                resamples=resamples,
                bootstrap_seed=bootstrap_seed,
            )["favorable_tied_adverse"],
            "cause": "generic lower-is-better label applied to a higher-is-better endpoint",
            "effect_on_gate_or_status": "none; the gate used the signed mean directly",
        },
        "extended_export_precision_finding": {
            "scope": "frozen RSS probabilities exported by the upstream source pipeline",
            "maximum_absolute_row_sum_error": maximum_rss_export_row_sum_error,
            "strict_1e_12_pass": bool(maximum_rss_export_row_sum_error <= 1e-12),
            "serialization_tolerance_2e_7_pass": bool(
                maximum_rss_export_row_sum_error <= 2e-7
            ),
            "candidate_outputs_affected": False,
            "effect_on_protocol_gate_or_reported_metrics": (
                "none; V21's frozen numerical gate applies to candidate output, and "
                "independently reconstructed RSS metrics match the frozen reports"
            ),
        },
        "claim_boundary": [
            "V20 failed its original composite development gates and remains a retained negative result.",
            "V21 supports probability-calibration improvement relative to raw Dempster evidence fusion.",
            "The calibrated candidate is not established as better than frozen RSS probability output: only three of six seeds improved and the bootstrap interval crosses zero.",
            "Temperature calibration changed no argmax or dispatch input; it cannot be credited with action or control changes.",
            "The 57 candidate-versus-RSS and 9 candidate-versus-Yager changed action rows are attributable to raw Dempster conflict handling, not scalar calibration.",
            "Endpoint-wise p-values are descriptive and unadjusted; n=6 has a minimum attainable two-sided p-value of 0.03125.",
            "Evidence is synthetic simulation, not field, HIL, or causal-effect validation.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_files = [
        FREEZE_PATH,
        CLARIFICATION_PATH,
        RUNNER_MANIFEST_PATH,
        RESULT_ROOT / "prospective_summary.json",
        RESULT_ROOT / "prospective_seed_results.csv",
        *eligibility_paths,
        *(RESULT_ROOT / f"seed_{seed}/evidence_manifest.json" for seed in eligible),
        OUTPUT_PATH,
    ]
    AUDIT_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "analysis": "V21 independent audit evidence package",
                "file_count": len(audit_files),
                "files": [
                    {
                        "path": relative(path),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in audit_files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT_PATH)}))


if __name__ == "__main__":
    main()
