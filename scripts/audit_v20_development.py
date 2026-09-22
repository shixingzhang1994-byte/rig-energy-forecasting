from __future__ import annotations

"""Independent, read-only reconstruction of the V20 development decision."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory_v20 import fuse_risk_frame_v20  # noqa: E402


PROTOCOL = PROJECT_DIR / "configs/v20_evidential_calibration_protocol.yaml"
FREEZE = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_freeze.json"
PARENT = PROJECT_DIR / "artifacts/v19_evidential_risk/development_search/selected_evidential_config.yaml"
RESULTS = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_search/candidate_results.json"
SOURCE = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_source"
OUTPUT = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_audit.json"
NAMES = ("normal", "watch", "warning", "severe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for level in range(4):
        true_positive = int(((truth == level) & (prediction == level)).sum())
        false_positive = int(((truth != level) & (prediction == level)).sum())
        false_negative = int(((truth == level) & (prediction != level)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    changed = []
    for item in freeze["files"]:
        path = PROJECT_DIR / item["path"]
        if not path.exists() or sha256(path) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("V20 frozen file changed: " + ", ".join(changed))
    candidates = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(candidates) != int(protocol["candidate_space"]["candidate_count"]):
        raise RuntimeError("candidate result count differs from protocol")
    if any(bool(candidate["eligible"]) for candidate in candidates):
        raise RuntimeError("candidate file does not preserve the V20 fail-closed result")
    best = min(
        candidates,
        key=lambda candidate: (
            candidate["mean_calibrated_brier"],
            candidate["worst_brier_delta_vs_baseline"],
            candidate["mean_calibrated_log_loss"],
            candidate["rule_index"],
            candidate["temperature_index"],
        ),
    )
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    config = deepcopy(parent)
    config.pop("selection_provenance", None)
    config["combination_rule"] = best["combination_rule"]
    config["calibration"] = {
        "method": "scalar_temperature_power",
        "temperature": best["temperature"],
        "fit_scope": "V20_development_seeds_only",
        "dispatch_uses": "raw_pignistic_probability",
    }
    reconstructed = []
    max_row_sum_error = 0.0
    minimum_probability = 1.0
    maximum_probability = 0.0
    for expected in best["per_seed"]:
        seed = int(expected["seed"])
        frame = pd.read_csv(SOURCE / f"seed_{seed}/risk/risk_predictions.csv")
        fused = fuse_risk_frame_v20(frame, config)
        probability = fused[
            [f"v20_calibrated_prob_{name}" for name in NAMES]
        ].to_numpy(float)
        truth = frame["true_risk_level"].to_numpy(int)
        prediction = probability.argmax(axis=1)
        one_hot = np.eye(4)[truth]
        brier = float(np.square(probability - one_hot).sum(axis=1).mean())
        f1 = macro_f1(truth, prediction)
        if abs(brier - float(expected["calibrated"]["multiclass_brier"])) > 1e-12:
            raise RuntimeError(f"Brier reconstruction failed for seed {seed}")
        if abs(f1 - float(expected["calibrated"]["macro_f1"])) > 1e-12:
            raise RuntimeError(f"Macro-F1 reconstruction failed for seed {seed}")
        max_row_sum_error = max(
            max_row_sum_error, float(np.abs(probability.sum(axis=1) - 1.0).max())
        )
        minimum_probability = min(minimum_probability, float(probability.min()))
        maximum_probability = max(maximum_probability, float(probability.max()))
        reconstructed.append(
            {
                "seed": seed,
                "baseline_macro_f1": expected["baseline"]["macro_f1"],
                "candidate_macro_f1": f1,
                "macro_f1_delta": expected["macro_f1_gain_vs_baseline"],
                "baseline_brier": expected["baseline"]["multiclass_brier"],
                "candidate_brier": brier,
                "brier_delta": expected["brier_delta_vs_baseline"],
                "high_risk_recall": expected["dispatch"]["high_risk_recall"],
                "severe_recall": expected["dispatch"]["severe_recall"],
            }
        )
    payload = {
        "protocol_id": protocol["protocol"]["id"],
        "development_freeze_sha256": sha256(FREEZE),
        "candidate_results_sha256": sha256(RESULTS),
        "frozen_file_hashes_pass": True,
        "candidate_count": len(candidates),
        "eligible_candidate_count": 0,
        "v20_decision": "stop_before_holdout_no_candidate_passed_all_hard_gates",
        "best_probability_candidate_descriptive_only": {
            "candidate_index": best["candidate_index"],
            "combination_rule": best["combination_rule"],
            "temperature": best["temperature"],
            "mean_calibrated_brier": best["mean_calibrated_brier"],
            "mean_baseline_brier": float(
                np.mean([row["baseline_brier"] for row in reconstructed])
            ),
            "mean_brier_delta": float(
                np.mean([row["brier_delta"] for row in reconstructed])
            ),
            "mean_macro_f1_delta": float(
                np.mean([row["macro_f1_delta"] for row in reconstructed])
            ),
            "per_seed": reconstructed,
        },
        "probability_numerical_audit": {
            "maximum_absolute_row_sum_error": max_row_sum_error,
            "minimum_probability": minimum_probability,
            "maximum_probability": maximum_probability,
            "pass": max_row_sum_error <= 1e-12
            and minimum_probability >= 0.0
            and maximum_probability <= 1.0,
        },
        "interpretation": (
            "Probability calibration improved on average, but the frozen V20 "
            "classification-superiority and recall gates were not all met."
        ),
        "holdout_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
