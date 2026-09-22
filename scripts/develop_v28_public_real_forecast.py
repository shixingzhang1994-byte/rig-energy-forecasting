from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import sha256_file
from rig_energy.models.ensemble import (
    apply_weighted_ensemble,
    fit_validation_weighted_ensemble,
)
from rig_energy.validation.public_electrical_forecast import (
    apply_timestamp_causal_error_feedback,
    fit_candidate_models,
    make_causal_forecast_table,
    mase_scale_from_training,
    prediction_map,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop the V28 real-data forecast using training/validation only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v28_public_real_forecast_joint_draft.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/forecast_development",
    )
    return parser.parse_args()


def _with_mase(metrics: dict[str, float], scale: float) -> dict[str, float]:
    return {**metrics, "mase": float(metrics["mae_kw"] / scale)}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol"]["tests_may_be_opened"]:
        raise ValueError("Development runner requires locked tests")
    shared = config["shared_forecast"]
    horizons = [int(config["protocol"]["primary_horizon_minutes"])] + [
        int(value) for value in config["protocol"]["secondary_horizons_minutes"]
    ]
    seed = int(config["protocol"]["random_seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol_id": config["protocol"]["id"],
        "phase": "training_and_validation_only",
        "test_targets_read": False,
        "datasets": {},
    }

    for dataset_name, dataset_config in config["datasets"].items():
        if dataset_config.get("eligibility") != "included":
            continue
        path = PROJECT_DIR / dataset_config["prepared_path"]
        observed_hash = sha256_file(path)
        if observed_hash != dataset_config["prepared_sha256"]:
            raise ValueError(f"Prepared hash mismatch for {dataset_name}")
        validation_end = pd.Timestamp(dataset_config["validation_target_end"])
        # Predicate pushdown prevents the locked test portion from being read.
        frame = pd.read_parquet(
            path,
            filters=[("timestamp", "<=", validation_end.to_pydatetime())],
        )
        if pd.to_datetime(frame["timestamp"], utc=True).max() > validation_end:
            raise RuntimeError("Test-lock predicate was not enforced")
        train_end = pd.Timestamp(dataset_config["train_target_end"])
        training_raw = frame.loc[pd.to_datetime(frame["timestamp"], utc=True).le(train_end)]
        mase_scale = mase_scale_from_training(training_raw)
        dataset_report: dict[str, object] = {
            "prepared_sha256_verified": observed_hash,
            "maximum_timestamp_read": pd.to_datetime(frame["timestamp"], utc=True).max(),
            "validation_end": validation_end,
            "mase_training_scale_kw": mase_scale,
            "horizons": {},
        }

        for horizon in horizons:
            table, features = make_causal_forecast_table(
                frame,
                horizon_minutes=horizon,
                load_lags_minutes=[int(value) for value in shared["load_lags_minutes"]],
                rolling_windows_minutes=[
                    int(value) for value in shared["rolling_windows_minutes"]
                ],
            )
            target_time = pd.to_datetime(table["target_timestamp"], utc=True)
            validation_start = pd.Timestamp(dataset_config["validation_target_start"])
            train = table.loc[target_time.le(train_end)].copy()
            validation = table.loc[
                target_time.ge(validation_start) & target_time.le(validation_end)
            ].copy()
            if min(len(train), len(validation)) == 0:
                raise ValueError(f"Empty development split for {dataset_name}, h={horizon}")
            cut_timestamp = validation_start + (validation_end - validation_start) / 2
            calibration_mask = pd.to_datetime(
                validation["target_timestamp"], utc=True
            ).le(cut_timestamp)
            evaluation_mask = ~calibration_mask
            if min(int(calibration_mask.sum()), int(evaluation_mask.sum())) == 0:
                raise ValueError("Validation calibration/evaluation half is empty")

            models = fit_candidate_models(train, features, seed=seed + horizon)
            predictions = prediction_map(models, validation, features)
            weights, weight_audit = fit_validation_weighted_ensemble(
                validation.loc[calibration_mask, "target_kw"].to_numpy(float),
                {
                    name: predictions[name][calibration_mask.to_numpy()]
                    for name in shared["base_models"]
                },
                l2_penalty=float(shared["ensemble"]["l2_penalty"]),
            )
            ensemble = apply_weighted_ensemble(predictions, weights).astype(float)
            clip = (
                float(shared["causal_error_feedback"]["correction_clip_training_mase_multipler"])
                * mase_scale
            )
            corrected, bias_trace = apply_timestamp_causal_error_feedback(
                ensemble,
                validation["target_kw"].to_numpy(float),
                validation["origin_timestamp"],
                validation["target_timestamp"],
                smoothing=float(shared["causal_error_feedback"]["smoothing"]),
                correction_clip_kw=clip,
            )
            y_eval = validation.loc[evaluation_mask, "target_kw"].to_numpy(float)
            eval_predictions = {
                "persistence": predictions["persistence"][evaluation_mask.to_numpy()],
                "rolling_mean_60": predictions["rolling_mean_60"][evaluation_mask.to_numpy()],
                "ridge": predictions["ridge"][evaluation_mask.to_numpy()],
                "hist_gradient_boosting": predictions["hist_gradient_boosting"][
                    evaluation_mask.to_numpy()
                ],
                "validation_weighted_ensemble": ensemble[evaluation_mask.to_numpy()],
                "causal_error_feedback_ensemble": corrected[evaluation_mask.to_numpy()],
            }
            metrics = {
                name: _with_mase(regression_metrics(y_eval, prediction), mase_scale)
                for name, prediction in eval_predictions.items()
            }
            dataset_report["horizons"][str(horizon)] = {
                "train_rows": len(train),
                "validation_calibration_rows": int(calibration_mask.sum()),
                "validation_evaluation_rows": int(evaluation_mask.sum()),
                "ensemble_weights": weights,
                "ensemble_weight_audit": weight_audit,
                "feedback_smoothing": float(
                    shared["causal_error_feedback"]["smoothing"]
                ),
                "feedback_clip_kw": clip,
                "feedback_final_bias_kw": float(bias_trace[-1]),
                "validation_evaluation_metrics": metrics,
                "causal_ensemble_minus_persistence_mase": (
                    metrics["causal_error_feedback_ensemble"]["mase"]
                    - metrics["persistence"]["mase"]
                ),
            }
        report["datasets"][dataset_name] = dataset_report

    output = args.output_dir / "development_report.json"
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.isoformat()
            if isinstance(value, pd.Timestamp)
            else float(value),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
