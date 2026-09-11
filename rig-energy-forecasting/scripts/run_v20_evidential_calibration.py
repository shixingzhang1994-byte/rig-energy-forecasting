from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory_v20 import fuse_risk_frame_v20  # noqa: E402


CLASS_NAMES = ("normal", "watch", "warning", "severe")


def top_label_ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(labels, dtype=int)
    values = np.asarray(probability, dtype=float)
    prediction = values.argmax(axis=1)
    confidence = values.max(axis=1)
    correct = prediction == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        member = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if member.any():
            score += float(member.mean()) * abs(
                float(correct[member].mean()) - float(confidence[member].mean())
            )
    return score


def metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    values = np.asarray(probability, dtype=float)
    one_hot = np.eye(4)[truth]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "high_risk_recall": float(recall_score(truth >= 2, predicted >= 2, zero_division=0)),
        "severe_recall": float(recall_score(truth == 3, predicted == 3, zero_division=0)),
        "severity_mae": float(np.abs(truth - predicted).mean()),
        "log_loss": float(log_loss(truth, values, labels=[0, 1, 2, 3])),
        "multiclass_brier": float(np.square(values - one_hot).sum(axis=1).mean()),
        "top_label_ece_10bin": top_label_ece(truth, values, bins=10),
    }


def run(input_path: Path, config_path: Path, output_dir: Path) -> dict[str, object]:
    source = pd.read_csv(input_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fused = fuse_risk_frame_v20(source, config)
    output = source.copy()
    output["baseline_predicted_risk_level"] = output["predicted_risk_level"]
    output["baseline_dispatch_risk_level"] = output["dispatch_risk_level"]
    output = pd.concat([output, fused], axis=1)
    output["dispatch_risk_level"] = fused["v20_dispatch_risk_level"].astype(int)
    reserve_map = {int(level): float(value) for level, value in config["reserve_adder_kw"].items()}
    output["reserve_adder_kw"] = output["dispatch_risk_level"].map(reserve_map)

    labels = output["true_risk_level"].to_numpy(int)
    baseline_probability = output[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(float)
    raw_probability = fused[[f"v20_raw_prob_{name}" for name in CLASS_NAMES]].to_numpy(float)
    calibrated_probability = fused[
        [f"v20_calibrated_prob_{name}" for name in CLASS_NAMES]
    ].to_numpy(float)
    baseline = metrics(labels, output["baseline_predicted_risk_level"].to_numpy(int), baseline_probability)
    raw = metrics(labels, fused["v20_risk_level"].to_numpy(int), raw_probability)
    calibrated = metrics(
        labels, fused["v20_calibrated_risk_level"].to_numpy(int), calibrated_probability
    )
    dispatch = metrics(
        labels, fused["v20_dispatch_risk_level"].to_numpy(int), calibrated_probability
    )
    report: dict[str, object] = {
        "input": str(input_path),
        "config": str(config_path),
        "sample_count": int(len(output)),
        "transform_uses_current_row_truth": False,
        "selection_labels_are_development_only": True,
        "combination_rule": config["combination_rule"],
        "temperature": float(config["calibration"]["temperature"]),
        "dispatch_probability_source": "raw_pignistic_probability",
        "baseline_prediction": baseline,
        "raw_evidential_prediction": raw,
        "calibrated_evidential_prediction": calibrated,
        "evidential_dispatch_guard": dispatch,
        "calibration_delta": {
            "brier_calibrated_minus_raw": calibrated["multiclass_brier"] - raw["multiclass_brier"],
            "brier_calibrated_minus_baseline": calibrated["multiclass_brier"] - baseline["multiclass_brier"],
            "log_loss_calibrated_minus_raw": calibrated["log_loss"] - raw["log_loss"],
            "ece_calibrated_minus_raw": calibrated["top_label_ece_10bin"] - raw["top_label_ece_10bin"],
        },
        "decision_invariance": {
            "calibration_changed_argmax_rows": int(fused["v20_calibration_changed_argmax"].sum()),
            "dispatch_reads_calibrated_probability": False,
        },
        "diagnostics": {
            "mean_ignorance": float(fused["v20_ignorance"].mean()),
            "p95_ignorance": float(fused["v20_ignorance"].quantile(0.95)),
            "mean_conflict": float(fused["v20_conflict"].mean()),
            "p95_conflict": float(fused["v20_conflict"].quantile(0.95)),
            "total_conflict_fallback_rows": int(fused["v20_total_conflict_fallback"].sum()),
            "dispatch_escalation_rate_vs_baseline": float(
                (fused["v20_dispatch_risk_level"].to_numpy(int) > output["baseline_dispatch_risk_level"].to_numpy(int)).mean()
            ),
            "dispatch_deescalation_rate_vs_baseline": float(
                (fused["v20_dispatch_risk_level"].to_numpy(int) < output["baseline_dispatch_risk_level"].to_numpy(int)).mean()
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "v20_risk_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "v20_risk_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_v20_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
