from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import sha256_file
from rig_energy.validation.public_electrical_forecast import (
    fit_candidate_models,
    make_candidate_models,
    make_causal_forecast_table,
    mase_scale_from_training,
    paired_daily_bootstrap,
    prediction_map,
    regression_metrics,
    split_forecast_table,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _metrics(y_true: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    result = regression_metrics(y_true, prediction)
    result["mase"] = float(result["mae_kw"] / scale)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v28_public_real_forecast_frozen.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/forecast_confirmatory",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    protocol = config["protocol"]
    if protocol["status"] != "frozen_before_any_listed_test_target_was_read":
        raise ValueError("Confirmation requires a frozen protocol")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared = config["shared_forecast"]
    horizons = [int(protocol["primary_horizon_minutes"])] + [
        int(value) for value in protocol["secondary_horizons_minutes"]
    ]
    seed = int(protocol["random_seed"])
    report: dict[str, object] = {
        "protocol_id": protocol["id"],
        "config_path": str(args.config.resolve()),
        "config_sha256_before_test_open": config_sha256,
        "phase": "confirmatory_open_once",
        "all_listed_datasets_retained": True,
        "future_storage_grid_or_pv_channels_used": False,
        "datasets": {},
    }
    metric_rows: list[dict[str, object]] = []

    for dataset_index, (dataset_name, data_config) in enumerate(config["datasets"].items()):
        path = PROJECT_DIR / data_config["prepared_path"]
        observed_hash = sha256_file(path)
        if observed_hash != data_config["prepared_sha256"]:
            raise ValueError(f"Prepared hash mismatch for {dataset_name}")
        frame = pd.read_parquet(path)
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        train_end = pd.Timestamp(data_config["train_target_end"])
        training_raw = frame.loc[timestamps.le(train_end)]
        scale = mase_scale_from_training(training_raw)
        dataset_report: dict[str, object] = {
            "prepared_sha256_verified": observed_hash,
            "target_provenance": data_config["target_provenance"],
            "mase_training_scale_kw": scale,
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
            split = split_forecast_table(table, data_config)
            train_models = fit_candidate_models(
                split.train, features, seed=seed + dataset_index * 100 + horizon
            )
            validation_predictions = prediction_map(
                train_models, split.validation, features
            )
            y_validation = split.validation["target_kw"].to_numpy(float)
            validation_metrics = {
                name: _metrics(y_validation, prediction, scale)
                for name, prediction in validation_predictions.items()
            }
            selected = min(
                shared["candidate_models"],
                key=lambda name: validation_metrics[name]["mae_kw"],
            )
            refit = pd.concat([split.train, split.validation], ignore_index=True)
            model = make_candidate_models(
                seed=seed + dataset_index * 100 + horizon
            )[selected]
            model.fit(refit.loc[:, features], refit["target_kw"])
            predictions = {
                selected: np.asarray(
                    model.predict(split.test.loc[:, features]), dtype=float
                ),
                "persistence": split.test["load_lag_0"].to_numpy(float),
                "rolling_mean_60": split.test["load_roll_mean_60"].to_numpy(float),
            }
            y_test = split.test["target_kw"].to_numpy(float)
            test_metrics = {
                name: _metrics(y_test, prediction, scale)
                for name, prediction in predictions.items()
            }
            bootstrap = paired_daily_bootstrap(
                split.test["target_timestamp"],
                y_test,
                predictions[selected],
                predictions["persistence"],
                replicates=int(shared["paired_interval"]["replicates"]),
                confidence_level=float(shared["paired_interval"]["confidence_level"]),
                seed=seed + 10_000 + dataset_index * 100 + horizon,
            )
            output = split.test.loc[
                :, ["origin_timestamp", "target_timestamp", "target_kw"]
            ].copy()
            for name, prediction in predictions.items():
                output[f"prediction_{name}_kw"] = prediction
                output[f"absolute_error_{name}_kw"] = np.abs(prediction - y_test)
            output.to_parquet(
                args.output_dir / f"{dataset_name}_test_predictions_h{horizon}.parquet",
                index=False,
            )
            dataset_report["horizons"][str(horizon)] = {
                "split_rows": {
                    "train": len(split.train),
                    "validation": len(split.validation),
                    "test": len(split.test),
                },
                "selected_model_before_test": selected,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "paired_daily_bootstrap_vs_persistence": bootstrap,
                "selected_beats_persistence_mae": bool(
                    test_metrics[selected]["mae_kw"]
                    < test_metrics["persistence"]["mae_kw"]
                ),
            }
            for phase, values in (("validation", validation_metrics), ("test", test_metrics)):
                for model_name, metrics in values.items():
                    metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "target_provenance": data_config["target_provenance"],
                            "horizon_minutes": horizon,
                            "phase": phase,
                            "model": model_name,
                            "selected_model": model_name == selected,
                            **metrics,
                        }
                    )
        report["datasets"][dataset_name] = dataset_report

    report_path = args.output_dir / "forecast_confirmatory_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "forecast_confirmatory_metrics.csv", index=False
    )
    print(
        json.dumps(
            {
                "protocol_id": protocol["id"],
                "config_sha256_before_test_open": config_sha256,
                "datasets_completed": len(report["datasets"]),
                "horizons": horizons,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
