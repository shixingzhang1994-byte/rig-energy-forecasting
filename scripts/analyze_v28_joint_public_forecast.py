from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * float(p_values[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _bootstrap_tail_probability(
    daily_difference: np.ndarray, *, replicates: int, seed: int
) -> float:
    rng = np.random.default_rng(int(seed))
    values = np.asarray(daily_difference, dtype=float)
    samples = rng.choice(values, size=(int(replicates), len(values)), replace=True).mean(axis=1)
    lower = (np.count_nonzero(samples <= 0.0) + 1.0) / (len(samples) + 1.0)
    upper = (np.count_nonzero(samples >= 0.0) + 1.0) / (len(samples) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v28_public_real_forecast_frozen.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/forecast_confirmatory",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report_path = args.input_dir / "forecast_confirmatory_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    datasets = list(config["datasets"])
    horizons = [int(config["protocol"]["primary_horizon_minutes"])] + [
        int(value) for value in config["protocol"]["secondary_horizons_minutes"]
    ]
    replicates = int(config["shared_forecast"]["paired_interval"]["replicates"])
    seed = int(config["protocol"]["random_seed"])
    rows: list[dict[str, object]] = []
    primary_p: dict[str, float] = {}

    for dataset_index, dataset in enumerate(datasets):
        dataset_report = report["datasets"][dataset]
        for horizon in horizons:
            horizon_report = dataset_report["horizons"][str(horizon)]
            selected = horizon_report["selected_model_before_test"]
            predictions = pd.read_parquet(
                args.input_dir / f"{dataset}_test_predictions_h{horizon}.parquet"
            )
            difference = (
                predictions[f"absolute_error_{selected}_kw"]
                - predictions["absolute_error_persistence_kw"]
            )
            day = pd.to_datetime(predictions["target_timestamp"], utc=True).dt.floor("D")
            daily = pd.DataFrame({"day": day, "difference": difference}).groupby("day")[
                "difference"
            ].mean()
            p_value = _bootstrap_tail_probability(
                daily.to_numpy(float),
                replicates=replicates,
                seed=seed + 10_000 + dataset_index * 100 + horizon,
            )
            metrics = horizon_report["test_metrics"]
            row = {
                "dataset": dataset,
                "target_provenance": dataset_report["target_provenance"],
                "horizon_minutes": horizon,
                "selected_model": selected,
                "test_rows": int(len(predictions)),
                "test_days": int(len(daily)),
                "selected_mase": float(metrics[selected]["mase"]),
                "persistence_mase": float(metrics["persistence"]["mase"]),
                "selected_minus_persistence_mase": float(
                    metrics[selected]["mase"] - metrics["persistence"]["mase"]
                ),
                "selected_minus_persistence_mae_kw": float(difference.mean()),
                "ci_low_kw": float(
                    horizon_report["paired_daily_bootstrap_vs_persistence"]["ci_low_kw"]
                ),
                "ci_high_kw": float(
                    horizon_report["paired_daily_bootstrap_vs_persistence"]["ci_high_kw"]
                ),
                "bootstrap_tail_probability": p_value,
                "beats_persistence": bool(difference.mean() < 0.0),
            }
            rows.append(row)
            if horizon == int(config["protocol"]["primary_horizon_minutes"]):
                primary_p[dataset] = p_value

    adjusted = _holm(primary_p)
    for row in rows:
        if row["horizon_minutes"] == int(config["protocol"]["primary_horizon_minutes"]):
            row["holm_adjusted_tail_probability"] = adjusted[str(row["dataset"])]
            row["primary_significant_improvement"] = bool(
                row["beats_persistence"]
                and float(row["holm_adjusted_tail_probability"]) < 0.05
            )
        else:
            row["holm_adjusted_tail_probability"] = np.nan
            row["primary_significant_improvement"] = False

    summary = pd.DataFrame(rows)
    horizon_summary = {}
    for horizon, group in summary.groupby("horizon_minutes"):
        horizon_summary[str(int(horizon))] = {
            "datasets": int(len(group)),
            "datasets_beating_persistence": int(group["beats_persistence"].sum()),
            "median_selected_minus_persistence_mase": float(
                group["selected_minus_persistence_mase"].median()
            ),
            "mean_selected_minus_persistence_mase_equal_dataset_weight": float(
                group["selected_minus_persistence_mase"].mean()
            ),
        }
    primary = summary.loc[
        summary["horizon_minutes"].eq(
            int(config["protocol"]["primary_horizon_minutes"])
        )
    ]
    analysis = {
        "protocol_id": report["protocol_id"],
        "config_sha256_before_test_open": report["config_sha256_before_test_open"],
        "all_8_datasets_and_3_horizons_present": bool(len(summary) == 24),
        "horizon_summary": horizon_summary,
        "primary_family": {
            "horizon_minutes": int(config["protocol"]["primary_horizon_minutes"]),
            "datasets_with_holm_significant_improvement": int(
                primary["primary_significant_improvement"].sum()
            ),
            "datasets_with_directional_improvement": int(primary["beats_persistence"].sum()),
            "conclusion_rule": (
                "Report dataset-specific effects; no universal-improvement claim unless "
                "all eight effects are negative and each Holm-adjusted tail probability < 0.05."
            ),
            "universal_improvement_supported": bool(
                primary["beats_persistence"].all()
                and primary["primary_significant_improvement"].all()
            ),
        },
        "claim_boundary": (
            "Cross-domain public electrical forecasting only; not target-rig field "
            "forecasting or closed-loop controller validation."
        ),
    }
    summary.to_csv(args.input_dir / "forecast_confirmatory_summary.csv", index=False)
    (args.input_dir / "forecast_confirmatory_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
