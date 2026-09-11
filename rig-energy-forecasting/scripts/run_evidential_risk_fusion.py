from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory import fuse_risk_frame  # noqa: E402


def _metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
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
    }


def run(input_path: Path, config_path: Path, output_dir: Path) -> dict[str, object]:
    source = pd.read_csv(input_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fused = fuse_risk_frame(source, config)
    output = source.copy()
    output["baseline_predicted_risk_level"] = output["predicted_risk_level"]
    output["baseline_dispatch_risk_level"] = output["dispatch_risk_level"]
    output = pd.concat([output, fused], axis=1)
    output["dispatch_risk_level"] = fused["evidential_dispatch_risk_level"].astype(int)
    reserve_map = {int(level): float(value) for level, value in config["reserve_adder_kw"].items()}
    output["reserve_adder_kw"] = output["dispatch_risk_level"].map(reserve_map)

    probability = fused[
        [
            "evidential_prob_normal",
            "evidential_prob_watch",
            "evidential_prob_warning",
            "evidential_prob_severe",
        ]
    ].to_numpy(float)
    labels = output["true_risk_level"].to_numpy(int)
    baseline_probability = output[
        ["prob_normal", "prob_watch", "prob_warning", "prob_severe"]
    ].to_numpy(float)
    report = {
        "input": str(input_path),
        "config": str(config_path),
        "sample_count": int(len(output)),
        "transform_uses_future_truth": False,
        "baseline_prediction": _metrics(
            labels,
            output["baseline_predicted_risk_level"].to_numpy(int),
            baseline_probability,
        ),
        "baseline_dispatch_guard": _metrics(
            labels,
            output["baseline_dispatch_risk_level"].to_numpy(int),
            np.eye(4)[output["baseline_dispatch_risk_level"].to_numpy(int)] * 0.97 + 0.03 / 4.0,
        ),
        "evidential_prediction": _metrics(
            labels,
            fused["evidential_risk_level"].to_numpy(int),
            probability,
        ),
        "evidential_dispatch_guard": _metrics(
            labels,
            fused["evidential_dispatch_risk_level"].to_numpy(int),
            probability,
        ),
        "diagnostics": {
            "mean_ignorance": float(fused["evidential_ignorance"].mean()),
            "p95_ignorance": float(fused["evidential_ignorance"].quantile(0.95)),
            "mean_conflict": float(fused["evidential_conflict"].mean()),
            "p95_conflict": float(fused["evidential_conflict"].quantile(0.95)),
            "dispatch_escalation_rate_vs_baseline": float(
                (
                    fused["evidential_dispatch_risk_level"].to_numpy(int)
                    > output["baseline_dispatch_risk_level"].to_numpy(int)
                ).mean()
            ),
            "dispatch_deescalation_rate_vs_baseline": float(
                (
                    fused["evidential_dispatch_risk_level"].to_numpy(int)
                    < output["baseline_dispatch_risk_level"].to_numpy(int)
                ).mean()
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "evidential_risk_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "evidential_risk_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_evidential_config.yaml").write_text(
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
