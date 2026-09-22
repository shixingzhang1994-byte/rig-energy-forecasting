from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ForecastSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]


def make_causal_forecast_table(
    frame: pd.DataFrame,
    *,
    horizon_minutes: int,
    load_lags_minutes: list[int],
    rolling_windows_minutes: list[int],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build one-minute forecast rows using information available at origin.

    Reindexing occurs before lag construction. Consequently, a missing or
    quality-rejected minute breaks rolling features instead of being bridged by
    row position. Calendar features describe the known target timestamp.
    """

    required = {"timestamp", "measured_load_kw", "quality_ok"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared electrical frame missing columns: {missing}")
    data = frame.loc[:, list(required)].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    data = data.set_index("timestamp").asfreq("1min")
    load = data["measured_load_kw"].where(data["quality_ok"].fillna(False))

    result = pd.DataFrame(index=data.index)
    feature_columns: list[str] = []
    for lag in load_lags_minutes:
        column = f"load_lag_{int(lag)}"
        result[column] = load.shift(int(lag))
        feature_columns.append(column)
    for window in rolling_windows_minutes:
        window = int(window)
        mean_column = f"load_roll_mean_{window}"
        std_column = f"load_roll_std_{window}"
        result[mean_column] = load.rolling(window, min_periods=window).mean()
        result[std_column] = load.rolling(window, min_periods=window).std(ddof=0)
        feature_columns.extend([mean_column, std_column])

    target_time = result.index + pd.to_timedelta(int(horizon_minutes), unit="min")
    result["target_timestamp"] = target_time
    result["target_kw"] = load.shift(-int(horizon_minutes))
    minute_of_day = target_time.hour * 60 + target_time.minute
    result["target_sin_day"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    result["target_cos_day"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    result["target_sin_week"] = np.sin(
        2.0 * np.pi * (target_time.dayofweek * 1440 + minute_of_day) / 10080.0
    )
    result["target_cos_week"] = np.cos(
        2.0 * np.pi * (target_time.dayofweek * 1440 + minute_of_day) / 10080.0
    )
    feature_columns.extend(
        ["target_sin_day", "target_cos_day", "target_sin_week", "target_cos_week"]
    )
    result["origin_timestamp"] = result.index
    result = result.dropna(subset=feature_columns + ["target_kw"]).reset_index(drop=True)
    return result, tuple(feature_columns)


def split_forecast_table(table: pd.DataFrame, config: dict[str, Any]) -> ForecastSplit:
    target = pd.to_datetime(table["target_timestamp"], utc=True)
    train_end = pd.Timestamp(config["train_target_end"])
    validation_start = pd.Timestamp(config["validation_target_start"])
    validation_end = pd.Timestamp(config["validation_target_end"])
    test_start = pd.Timestamp(config["test_target_start"])
    test_end = pd.Timestamp(config["test_target_end"])
    train = table.loc[target.le(train_end)].copy()
    validation = table.loc[target.ge(validation_start) & target.le(validation_end)].copy()
    test = table.loc[target.ge(test_start) & target.le(test_end)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("At least one chronological forecast split is empty")
    feature_columns = tuple(
        column
        for column in table.columns
        if column.startswith("load_") or column.startswith("target_sin_") or column.startswith("target_cos_")
    )
    return ForecastSplit(train, validation, test, feature_columns)


def make_candidate_models(*, seed: int) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(seed),
        ),
    }


def fit_candidate_models(
    train: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    x = train.loc[:, feature_columns]
    y = train["target_kw"]
    models = make_candidate_models(seed=seed)
    for model in models.values():
        model.fit(x, y)
    return models


def prediction_map(
    models: dict[str, Any],
    split: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, np.ndarray]:
    predictions = {
        name: np.asarray(model.predict(split.loc[:, feature_columns]), dtype=float)
        for name, model in models.items()
    }
    predictions["persistence"] = split["load_lag_0"].to_numpy(float)
    predictions["rolling_mean_60"] = split["load_roll_mean_60"].to_numpy(float)
    return predictions


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    absolute = np.abs(error)
    denominator = max(float(np.mean(np.abs(y_true))), 1e-12)
    return {
        "mae_kw": float(np.mean(absolute)),
        "rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
        "nmae_mean_load": float(np.mean(absolute) / denominator),
        "bias_kw": float(np.mean(error)),
    }


def paired_daily_bootstrap(
    target_timestamp: pd.Series,
    y_true: np.ndarray,
    selected_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    day = pd.to_datetime(target_timestamp, utc=True).dt.floor("D")
    difference = np.abs(selected_prediction - y_true) - np.abs(baseline_prediction - y_true)
    daily = pd.DataFrame({"day": day, "difference": difference}).groupby("day")[
        "difference"
    ].mean()
    rng = np.random.default_rng(int(seed))
    values = daily.to_numpy(float)
    samples = rng.choice(values, size=(int(replicates), len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "selected_minus_persistence_mae_kw": float(np.mean(difference)),
        "ci_low_kw": float(np.quantile(samples, alpha)),
        "ci_high_kw": float(np.quantile(samples, 1.0 - alpha)),
        "bootstrap_days": int(len(values)),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
    }


def apply_timestamp_causal_error_feedback(
    base_prediction: np.ndarray,
    actual: np.ndarray,
    origin_timestamp: pd.Series,
    target_timestamp: pd.Series,
    *,
    smoothing: float,
    correction_clip_kw: float,
    initial_bias_kw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply EWMA residual feedback when prior targets have become observable.

    Unlike a fixed row-delay implementation, this uses timestamps and remains
    causal when quality filtering creates irregular gaps between forecast rows.
    The returned bias trace is the value used for each prediction.
    """

    base = np.asarray(base_prediction, dtype=float)
    target = np.asarray(actual, dtype=float)
    origins = pd.to_datetime(origin_timestamp, utc=True).to_numpy()
    targets = pd.to_datetime(target_timestamp, utc=True).to_numpy()
    if not (base.ndim == target.ndim == 1 and len(base) == len(target) == len(origins)):
        raise ValueError("Feedback inputs must be aligned one-dimensional arrays")
    if np.any(origins[1:] < origins[:-1]) or np.any(targets[1:] < targets[:-1]):
        raise ValueError("Forecast rows must be sorted chronologically")
    if not 0.0 < float(smoothing) <= 1.0:
        raise ValueError("smoothing must be in (0, 1]")
    if float(correction_clip_kw) < 0.0:
        raise ValueError("correction_clip_kw must be nonnegative")

    corrected = np.empty_like(base)
    bias_trace = np.empty_like(base)
    bias = float(initial_bias_kw)
    release_index = 0
    for index, origin in enumerate(origins):
        while release_index < index and targets[release_index] <= origin:
            residual = target[release_index] - base[release_index]
            bias = (1.0 - float(smoothing)) * bias + float(smoothing) * residual
            release_index += 1
        clipped = float(np.clip(bias, -correction_clip_kw, correction_clip_kw))
        corrected[index] = max(base[index] + clipped, 0.0)
        bias_trace[index] = clipped
    return corrected, bias_trace


def mase_scale_from_training(frame: pd.DataFrame) -> float:
    """Return the one-step MASE scale using adjacent valid training minutes."""

    ordered = frame.sort_values("timestamp")
    timestamp = pd.to_datetime(ordered["timestamp"], utc=True)
    load = ordered["measured_load_kw"].where(ordered["quality_ok"].fillna(False))
    adjacent = timestamp.diff().eq(pd.Timedelta(minutes=1)) & load.notna() & load.shift(1).notna()
    differences = load.diff().abs().loc[adjacent]
    if differences.empty:
        raise ValueError("No adjacent valid training minutes for MASE scale")
    scale = float(differences.mean())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Training MASE scale is nonpositive or nonfinite")
    return scale
