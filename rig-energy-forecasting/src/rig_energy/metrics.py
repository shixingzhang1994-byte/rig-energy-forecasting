from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    peak_threshold: float,
    transition_flags: np.ndarray | None = None,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"预测与真实形状不一致: {pred.shape} vs {true.shape}")
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mean_load = max(float(np.mean(np.abs(true))), 1e-9)
    smape = float(100 * np.mean(2 * np.abs(error) / (np.abs(true) + np.abs(pred) + 1e-6)))
    flat_true = true.reshape(-1)
    flat_pred = pred.reshape(-1)
    peak_mask = flat_true >= peak_threshold
    peak_mae = float(np.mean(np.abs(flat_pred[peak_mask] - flat_true[peak_mask]))) if np.any(peak_mask) else float("nan")
    transition_mae = float("nan")
    if transition_flags is not None and np.any(transition_flags):
        transition_mae = float(np.mean(np.abs(error[np.asarray(transition_flags, dtype=bool)])))
    return {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "nmae_pct_of_mean": 100 * mae / mean_load,
        "smape_pct": smape,
        "r2": float(r2_score(flat_true, flat_pred)),
        "peak_mae_kw": peak_mae,
        "transition_mae_kw": transition_mae,
    }

