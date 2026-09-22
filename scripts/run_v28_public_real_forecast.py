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
    paired_daily_bootstrap,
    prediction_map,
    regression_metrics,
    split_forecast_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen V28 OpenCEM forecast test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v28_public_real_validation.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/forecast",
    )
    return parser.parse_args()


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = config["opencem"]
    prepared_path = PROJECT_DIR / data_config["prepared_path"]
    observed_hash = sha256_file(prepared_path)
    if observed_hash != data_config["prepared_sha256"]:
        raise ValueError(
            f"Frozen prepared-data hash mismatch: {observed_hash} != "
            f"{data_config['prepared_sha256']}"
        )
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    seed = int(config["protocol"]["random_seed"])
    frame = pd.read_parquet(prepared_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "protocol_id": config["protocol"]["id"],
        "config_path": str(args.config.resolve()),
        "config_sha256": config_hash,
        "prepared_path": str(prepared_path.resolve()),
        "prepared_sha256_verified": observed_hash,
        "test_opened_after_protocol_freeze": True,
        "future_storage_or_grid_channels_used": False,
        "horizons": {},
    }
    metric_rows: list[dict[str, object]] = []

    for horizon in data_config["horizons_minutes"]:
        horizon = int(horizon)
        table, features = make_causal_forecast_table(
            frame,
            horizon_minutes=horizon,
            load_lags_minutes=[int(value) for value in data_config["load_lags_minutes"]],
            rolling_windows_minutes=[
                int(value) for value in data_config["rolling_windows_minutes"]
            ],
        )
        split = split_forecast_table(table, data_config)
        if tuple(features) != split.feature_columns:
            raise RuntimeError("Feature-column audit mismatch")

        train_models = fit_candidate_models(split.train, features, seed=seed + horizon)
        validation_predictions = prediction_map(train_models, split.validation, features)
        validation_metrics = {
            name: regression_metrics(
                split.validation["target_kw"].to_numpy(float), prediction
            )
            for name, prediction in validation_predictions.items()
        }
        selected = min(
            data_config["candidate_models"],
            key=lambda name: validation_metrics[name][data_config["selection_metric"]],
        )

        refit_rows = pd.concat([split.train, split.validation], ignore_index=True)
        selected_model = make_candidate_models(seed=seed + horizon)[selected]
        selected_model.fit(refit_rows.loc[:, features], refit_rows["target_kw"])
        test_predictions = {
            selected: np.asarray(
                selected_model.predict(split.test.loc[:, features]), dtype=float
            ),
            "persistence": split.test["load_lag_0"].to_numpy(float),
            "rolling_mean_60": split.test["load_roll_mean_60"].to_numpy(float),
        }
        y_test = split.test["target_kw"].to_numpy(float)
        test_metrics = {
            name: regression_metrics(y_test, prediction)
            for name, prediction in test_predictions.items()
        }
        bootstrap = paired_daily_bootstrap(
            split.test["target_timestamp"],
            y_test,
            test_predictions[selected],
            test_predictions["persistence"],
            replicates=int(data_config["bootstrap"]["replicates"]),
            confidence_level=float(data_config["bootstrap"]["confidence_level"]),
            seed=seed + 1000 + horizon,
        )
        test_output = split.test.loc[
            :, ["origin_timestamp", "target_timestamp", "target_kw"]
        ].copy()
        for name, prediction in test_predictions.items():
            test_output[f"prediction_{name}_kw"] = prediction
            test_output[f"absolute_error_{name}_kw"] = np.abs(prediction - y_test)
        test_output.to_parquet(
            args.output_dir / f"opencem_test_predictions_h{horizon}.parquet", index=False
        )

        horizon_report = {
            "horizon_minutes": horizon,
            "feature_columns": list(features),
            "split_rows": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "split_target_ranges": {
                "train": [
                    split.train["target_timestamp"].min(),
                    split.train["target_timestamp"].max(),
                ],
                "validation": [
                    split.validation["target_timestamp"].min(),
                    split.validation["target_timestamp"].max(),
                ],
                "test": [
                    split.test["target_timestamp"].min(),
                    split.test["target_timestamp"].max(),
                ],
            },
            "validation_metrics": validation_metrics,
            "selected_model_before_test": selected,
            "test_metrics": test_metrics,
            "paired_daily_bootstrap_vs_persistence": bootstrap,
            "selected_beats_persistence_mae": (
                test_metrics[selected]["mae_kw"] < test_metrics["persistence"]["mae_kw"]
            ),
        }
        report["horizons"][str(horizon)] = horizon_report
        for phase, metrics_by_model in (
            ("validation", validation_metrics),
            ("test", test_metrics),
        ):
            for model, metrics in metrics_by_model.items():
                metric_rows.append(
                    {
                        "horizon_minutes": horizon,
                        "phase": phase,
                        "model": model,
                        "selected_model": model == selected,
                        **metrics,
                    }
                )

    report_path = args.output_dir / "forecast_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "forecast_metrics.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
