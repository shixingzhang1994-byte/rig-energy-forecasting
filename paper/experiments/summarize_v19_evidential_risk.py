#!/usr/bin/env python3
"""Independent paper-side audit of the frozen V19 prospective results.

This script does not tune, transform, or rerun the controller.  It verifies the
frozen evidence manifest, recomputes paired seed-level summaries, and checks
that both baseline and evidential probability vectors are normalized.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "rig-energy-forecasting"
ARTIFACT = PROJECT / "artifacts/v19_evidential_risk/prospective"
EXPECTED_SEEDS = [20261022, 20261024, 20261025, 20261027, 20261028, 20261029]
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0, atol=1e-12):
        return 1.0
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def bootstrap_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def verify_manifest() -> dict[str, object]:
    manifest_path = ARTIFACT / "prospective_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["files"]:
        path = PROJECT / record["path"]
        if not path.exists() or sha256(path) != record["sha256"]:
            failures.append(record["path"])
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "declared_file_count": manifest["file_count"],
        "hash_failures": failures,
        "pass": not failures,
    }


def probability_audit(seed: int) -> dict[str, float | int | bool]:
    path = ARTIFACT / f"seed_{seed}" / "risk/evidential_risk_predictions.csv"
    compressed_path = path.with_suffix(path.suffix + ".gz")
    baseline_columns = ["prob_normal", "prob_watch", "prob_warning", "prob_severe"]
    evidence_columns = [
        "evidential_prob_normal",
        "evidential_prob_watch",
        "evidential_prob_warning",
        "evidential_prob_severe",
    ]
    maximum_baseline_error = 0.0
    maximum_evidence_error = 0.0
    raw_brier_sum = 0.0
    normalized_brier_sum = 0.0
    row_count = 0
    if path.exists():
        handle_context = path.open("r", encoding="utf-8-sig", newline="")
    else:
        handle_context = gzip.open(
            compressed_path, "rt", encoding="utf-8-sig", newline=""
        )
    with handle_context as handle:
        for row in csv.DictReader(handle):
            baseline_values = np.asarray([float(row[column]) for column in baseline_columns])
            baseline_sum = float(baseline_values.sum())
            evidence_sum = sum(float(row[column]) for column in evidence_columns)
            maximum_baseline_error = max(maximum_baseline_error, abs(baseline_sum - 1.0))
            maximum_evidence_error = max(maximum_evidence_error, abs(evidence_sum - 1.0))
            truth = int(row["true_risk_level"])
            target = np.eye(4)[truth]
            raw_brier_sum += float(np.sum((baseline_values - target) ** 2))
            normalized = baseline_values / baseline_sum
            normalized_brier_sum += float(np.sum((normalized - target) ** 2))
            row_count += 1
    # The archived baseline CSV contains one float-serialization residual of
    # 1.15e-7.  A separate sensitivity check changes its multiclass Brier
    # score by 6.84e-11 after row normalization, so 1e-6 is used here as a
    # numerical-integrity tolerance rather than a statistical acceptance gate.
    tolerance = 1e-6
    return {
        "row_count": row_count,
        "sum_tolerance": tolerance,
        "baseline_max_abs_sum_error": maximum_baseline_error,
        "evidential_max_abs_sum_error": maximum_evidence_error,
        "baseline_brier_change_after_normalization": (
            normalized_brier_sum - raw_brier_sum
        ) / row_count,
        "pass": maximum_baseline_error <= tolerance and maximum_evidence_error <= tolerance,
    }


def paired_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "bootstrap_95_interval": bootstrap_interval(values),
        "exact_two_sided_sign_flip_p": exact_sign_flip(values),
        "improved_tied_worse": [
            int(np.sum(values > 1e-12)),
            int(np.sum(np.abs(values) <= 1e-12)),
            int(np.sum(values < -1e-12)),
        ],
    }


def main() -> None:
    records: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    metric_paths = {
        "prediction_macro_f1": ("baseline_prediction", "evidential_prediction", "macro_f1"),
        "prediction_balanced_accuracy": (
            "baseline_prediction",
            "evidential_prediction",
            "balanced_accuracy",
        ),
        "prediction_brier": (
            "baseline_prediction",
            "evidential_prediction",
            "multiclass_brier",
        ),
        "dispatch_macro_f1": (
            "baseline_dispatch_guard",
            "evidential_dispatch_guard",
            "macro_f1",
        ),
        "dispatch_high_risk_recall": (
            "baseline_dispatch_guard",
            "evidential_dispatch_guard",
            "high_risk_recall",
        ),
    }
    for seed in EXPECTED_SEEDS:
        result_path = ARTIFACT / f"seed_{seed}" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["seed"] != seed or not result["seed_gate_pass"]:
            raise RuntimeError(f"unexpected or failed V19 result for seed {seed}")
        risk = result["risk_recognition"]
        record: dict[str, object] = {"seed": seed}
        for label, (baseline_group, evidence_group, metric) in metric_paths.items():
            baseline = float(risk[baseline_group][metric])
            evidence = float(risk[evidence_group][metric])
            record[f"baseline_{label}"] = baseline
            record[f"evidential_{label}"] = evidence
            record[f"delta_{label}"] = evidence - baseline
        record["unserved_delta_kwh"] = result["evidential_minus_baseline"][
            "unserved_energy_kwh"
        ]
        record["cost_delta_yuan"] = result["evidential_minus_baseline"][
            "risk_adjusted_cost_yuan"
        ]
        record["physical_action_changed_rows"] = result["action_audit"][
            "physical_action_changed_rows"
        ]
        records.append(record)
        audits[str(seed)] = probability_audit(seed)

    summaries: dict[str, object] = {}
    for label in metric_paths:
        deltas = np.asarray([float(row[f"delta_{label}"]) for row in records])
        summaries[label] = paired_summary(deltas)
    for label in ("unserved_delta_kwh", "cost_delta_yuan"):
        deltas = np.asarray([float(row[label]) for row in records])
        summaries[label] = paired_summary(deltas)

    manifest_audit = verify_manifest()
    probability_pass = all(bool(item["pass"]) for item in audits.values())
    output = {
        "analysis": "independent paper-side V19 evidential-risk audit",
        "unit_of_analysis": "seed; scenarios remain clustered within seed",
        "seeds": EXPECTED_SEEDS,
        "n": len(EXPECTED_SEEDS),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95%",
        },
        "paired_deltas": summaries,
        "probability_audit": audits,
        "manifest_audit": manifest_audit,
        "checks": {
            "exact_seed_set": [int(row["seed"]) for row in records] == EXPECTED_SEEDS,
            "all_probability_rows_normalized": probability_pass,
            "evidence_manifest_hashes_match": bool(manifest_audit["pass"]),
            "all_seed_gates_pass": True,
        },
        "claim_boundary": [
            "Positive classification deltas do not imply a control-outcome improvement.",
            "Pignistic scores failed to improve Brier calibration and are not claimed as calibrated probabilities.",
            "The evidence is synthetic prospective holdout evidence, not field validation.",
        ],
    }
    output["status"] = "pass" if all(output["checks"].values()) else "fail"

    output_dir = ROOT / "paper/experiments/results/v19_evidential_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "v19_paper_audit.json"
    csv_path = output_dir / "v19_seed_metrics.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"status": output["status"], "json": str(json_path), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
