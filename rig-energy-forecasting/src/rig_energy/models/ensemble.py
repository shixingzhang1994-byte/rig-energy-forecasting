from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize


def fit_validation_weighted_ensemble(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    l2_penalty: float = 1e-4,
) -> tuple[dict[str, float], dict[str, float | bool | str]]:
    """Fit non-negative convex weights using validation data only."""

    if len(predictions) < 2:
        raise ValueError("集成学习至少需要两个成员模型")
    names = list(predictions)
    target = np.asarray(y_true, dtype=np.float64)
    members = []
    for name in names:
        prediction = np.asarray(predictions[name], dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError(
                f"{name} 验证预测形状 {prediction.shape} 与目标 {target.shape} 不一致"
            )
        members.append(prediction)
    stacked = np.stack(members, axis=-1)

    def objective(weights: np.ndarray) -> float:
        blended = np.tensordot(stacked, weights, axes=([-1], [0]))
        mse = np.mean(np.square(blended - target))
        return float(mse + l2_penalty * np.sum(np.square(weights)))

    initial = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if result.success and np.all(np.isfinite(result.x)):
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= weights.sum()
        method = "validation_slsqp"
    else:
        mse = np.array([np.mean(np.square(member - target)) for member in members])
        inverse = 1.0 / np.maximum(mse, 1e-12)
        weights = inverse / inverse.sum()
        method = "inverse_validation_mse_fallback"

    metadata: dict[str, float | bool | str] = {
        "fit_method": method,
        "validation_objective": objective(weights),
        "l2_penalty": float(l2_penalty),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    return {name: float(weight) for name, weight in zip(names, weights)}, metadata


def apply_weighted_ensemble(
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    missing = [name for name in weights if name not in predictions]
    if missing:
        raise ValueError(f"集成预测缺少成员: {missing}")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"集成权重之和必须为1，当前为 {total}")
    blended = None
    for name, weight in weights.items():
        member = np.asarray(predictions[name], dtype=np.float64)
        contribution = float(weight) * member
        blended = contribution if blended is None else blended + contribution
    return np.asarray(blended, dtype=np.float32)


def apply_causal_error_feedback(
    prediction: np.ndarray,
    actual: np.ndarray,
    *,
    feedback_delay_steps: int,
    smoothing: float,
    correction_clip_kw: float,
    initial_bias_kw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct forecasts with delayed EWMA residuals, without future leakage.

    At sample ``i`` only the residual of sample ``i-feedback_delay_steps`` is
    released.  For stride-one multi-step forecasts, setting the delay equal to
    the horizon means every target in that historical window has been observed
    before it can update the bias vector.
    """

    base = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("误差反馈要求预测与真值为同形状二维数组")
    if feedback_delay_steps < base.shape[1]:
        raise ValueError("反馈延迟不得小于预测时域，否则存在未来信息泄漏")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("误差平滑系数必须在 (0, 1] 内")
    bias = (
        np.zeros(base.shape[1], dtype=np.float64)
        if initial_bias_kw is None
        else np.asarray(initial_bias_kw, dtype=np.float64).copy()
    )
    if bias.shape != (base.shape[1],):
        raise ValueError("初始误差偏置长度与预测时域不一致")
    corrected = np.empty_like(base)
    for index in range(len(base)):
        released = index - feedback_delay_steps
        if released >= 0:
            residual = target[released] - base[released]
            bias = (1.0 - smoothing) * bias + smoothing * residual
        clipped_bias = np.clip(bias, -correction_clip_kw, correction_clip_kw)
        corrected[index] = np.maximum(base[index] + clipped_bias, 0.0)
    return corrected.astype(np.float32), bias.astype(np.float32)
