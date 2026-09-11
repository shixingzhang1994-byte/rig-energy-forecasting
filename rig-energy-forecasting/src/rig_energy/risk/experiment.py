from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from ..data.schema import validate_canonical_frame
from ..data.windowing import split_frame_by_time
from .labels import RISK_NAMES, RISK_NAMES_ZH, assign_risk_levels, build_risk_evidence
from .models import (
    apply_probability_ensemble,
    fit_probability_ensemble,
    predict_transformer_probabilities,
    train_transformer_classifier,
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _causal_transition_feature(
    arrays: object,
    expected_length: int,
) -> tuple[np.ndarray, str]:
    """Load an online-available transition feature without future truth.

    ``transition_flags`` and ``future_transition_flags`` are deliberately not
    accepted: both describe whether a transition occurs inside the future
    scoring horizon and are valid only for offline forecast metrics.
    """

    keys = set(arrays.keys())
    if "history_transition_flags" in keys:
        feature = np.asarray(arrays["history_transition_flags"], dtype=np.float32)
        source = "history_transition_flags"
    elif "operation_state_codes" in keys:
        states = np.asarray(arrays["operation_state_codes"])
        feature = np.zeros(len(states), dtype=np.float32)
        feature[1:] = (states[1:] != states[:-1]).astype(np.float32)
        source = "observed_operation_state_change"
    else:
        raise ValueError(
            "风险模型缺少因果工况切换特征；禁止回退到未来transition_flags"
        )
    if feature.ndim != 1 or len(feature) != int(expected_length):
        raise ValueError("因果工况切换特征与预测样本长度不一致")
    return feature, source


def _smooth_noise(rng: np.random.Generator, count: int, sigma: float) -> np.ndarray:
    raw = rng.normal(0.0, sigma, count)
    smooth = np.empty(count, dtype=float)
    smooth[0] = raw[0]
    for index in range(1, count):
        smooth[index] = 0.97 * smooth[index - 1] + 0.12 * raw[index]
    return smooth


def _simulate_supply_context(
    timestamps: pd.Series,
    forecast_required_kw: np.ndarray,
    config: dict,
    seed: int,
) -> pd.DataFrame:
    """Generate an auditable synthetic supply process.

    ``independent_process`` is the acceptance mode: supply capacity and SOC are
    generated only from the declared regime process and random seed.  The
    legacy ``forecast_margin_stratified`` mode remains available solely for
    reproducing frozen V3 artifacts.
    """

    rng = np.random.default_rng(seed)
    count = len(timestamps)
    supply_cfg = config["synthetic_supply"]
    regimes = supply_cfg["regimes"]
    cycle = list(supply_cfg["regime_cycle"])
    minimum, maximum = map(int, supply_cfg["block_length_steps"])
    # Reset the stress-test cycle at each chronological classifier split. This
    # preserves time separation while ensuring train/validation/test all cover
    # the four supply regimes instead of inheriting an accidental regime shift.
    split_cfg = config["split"]
    boundaries = [
        0,
        int(count * float(split_cfg["train_fraction"])),
        int(
            count
            * (
                float(split_cfg["train_fraction"])
                + float(split_cfg["val_fraction"])
            )
        ),
        count,
    ]
    names = [""] * count
    for part_start, part_stop in zip(boundaries[:-1], boundaries[1:]):
        cursor = part_start
        block = 0
        while cursor < part_stop:
            length = int(rng.integers(minimum, maximum + 1))
            regime = cycle[block % len(cycle)]
            stop = min(cursor + length, part_stop)
            names[cursor:stop] = [regime] * (stop - cursor)
            cursor = stop
            block += 1
    regime_array = np.asarray(names, dtype=object)
    required = np.asarray(forecast_required_kw, dtype=float)
    if len(required) != count:
        raise ValueError("forecast_required_kw 与时间戳行数不一致")
    generation_mode = str(
        supply_cfg.get("generation_mode", "forecast_margin_stratified")
    )
    if generation_mode == "independent_process":
        grid_base = np.asarray(
            [float(regimes[name]["grid_capacity_kw"]) for name in regime_array]
        )
        generator_base = np.asarray(
            [
                float(regimes[name]["generator_capacity_kw"])
                for name in regime_array
            ]
        )
        soc_base = np.asarray(
            [float(regimes[name]["storage_soc_pct"]) for name in regime_array]
        )
        grid = grid_base + _smooth_noise(
            rng, count, float(supply_cfg.get("grid_noise_sigma_kw", 18.0))
        )
        generator = generator_base + _smooth_noise(
            rng,
            count,
            float(supply_cfg.get("generator_noise_sigma_kw", 12.0)),
        )
        soc = soc_base + _smooth_noise(
            rng, count, float(supply_cfg.get("soc_noise_sigma_pct", 2.5))
        )
        grid = np.clip(grid, 0.0, float(supply_cfg["grid_rated_capacity_kw"]))
        generator = np.clip(
            generator, 0.0, float(supply_cfg["generator_rated_power_kw"])
        )
        soc = np.clip(
            soc,
            float(supply_cfg["soc_min_pct"]),
            float(supply_cfg["soc_max_pct"]),
        )
        soc_availability = np.clip(
            (soc - float(supply_cfg["soc_min_pct"]))
            / float(supply_cfg["soc_full_power_above_pct"]),
            0.0,
            1.0,
        )
        storage = np.clip(
            float(supply_cfg["storage_rated_power_kw"]) * soc_availability
            + _smooth_noise(
                rng,
                count,
                float(supply_cfg.get("storage_noise_sigma_kw", 8.0)),
            ),
            0.0,
            float(supply_cfg["storage_rated_power_kw"]),
        )
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps).to_numpy(),
                "supply_regime": regime_array,
                "grid_available_capacity_kw": grid,
                "generator_available_capacity_kw": generator,
                "storage_soc_pct": soc,
                "storage_available_discharge_power_kw": storage,
                "supply_source_type": "synthetic_independent_supply",
            }
        )
    if generation_mode != "forecast_margin_stratified":
        raise ValueError(f"未知合成供给生成模式: {generation_mode}")

    grid = np.zeros(count, dtype=float)
    generator = np.zeros(count, dtype=float)
    storage = np.zeros(count, dtype=float)
    soc = np.zeros(count)
    for name, values in regimes.items():
        mask = regime_array == name
        size = int(mask.sum())
        target_margin = float(values["target_forecast_margin_ratio"])
        total = required[mask] * (
            1.0
            + target_margin
            + rng.normal(0.0, float(values["margin_sigma_ratio"]), size)
        )
        total += rng.normal(0.0, float(values["firm_supply_sigma_kw"]), size)
        grid[mask] = total * float(values["grid_share"]) + rng.normal(0.0, 12.0, size)
        generator[mask] = total * float(values["generator_share"]) + rng.normal(
            0.0, 10.0, size
        )
        soc[mask] = float(values["storage_soc_pct"]) + rng.normal(
            0.0, float(values["soc_sigma_pct"]), size
        )
    grid += _smooth_noise(rng, count, 18.0)
    generator += _smooth_noise(rng, count, 12.0)
    storage += _smooth_noise(rng, count, 8.0)
    grid = np.clip(grid, 0.0, float(supply_cfg["grid_rated_capacity_kw"]))
    generator = np.clip(generator, 0.0, float(supply_cfg["generator_rated_power_kw"]))
    soc = np.clip(soc, float(supply_cfg["soc_min_pct"]), float(supply_cfg["soc_max_pct"]))
    soc_availability = np.clip(
        (soc - float(supply_cfg["soc_min_pct"]))
        / float(supply_cfg["soc_full_power_above_pct"]),
        0.0,
        1.0,
    )
    desired_total = required * np.asarray(
        [
            1.0 + float(regimes[name]["target_forecast_margin_ratio"])
            for name in regime_array
        ]
    )
    storage = np.minimum(
        np.maximum(desired_total - grid - generator, 0.0),
        float(supply_cfg["storage_rated_power_kw"]) * soc_availability,
    )
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).to_numpy(),
            "supply_regime": regime_array,
            "grid_available_capacity_kw": grid,
            "generator_available_capacity_kw": generator,
            "storage_soc_pct": soc,
            "storage_available_discharge_power_kw": storage,
            "supply_source_type": "synthetic_forecast_stratified_supply",
        }
    )


def _supply_from_canonical(
    origin_frame: pd.DataFrame,
    timestamps: pd.Series,
) -> pd.DataFrame | None:
    required = [
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_soc_pct",
        "storage_available_discharge_power_kw",
    ]
    if not all(column in origin_frame.columns for column in required):
        return None
    supply = origin_frame[required].apply(pd.to_numeric, errors="coerce")
    if supply.isna().any().any():
        return None
    supply.insert(0, "timestamp", pd.to_datetime(timestamps).to_numpy())
    supply["supply_regime"] = "observed"
    supply["supply_source_type"] = "scada"
    return supply


def _load_external_supply(path: Path, timestamps: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = [
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_soc_pct",
        "storage_available_discharge_power_kw",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"外部供能数据缺少字段: {missing}")
    if len(frame) != len(timestamps):
        raise ValueError(f"外部供能数据行数 {len(frame)} 与预测样本数 {len(timestamps)} 不一致")
    result = frame.copy().reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(timestamps).to_numpy()
    if "supply_regime" not in result:
        result["supply_regime"] = "external"
    result["supply_source_type"] = "external"
    return result


def _classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    true_high = labels >= 2
    pred_high = prediction >= 2
    high_risk_recall = recall_score(true_high, pred_high, zero_division=0)
    true_severe = labels == 3
    pred_severe = prediction == 3
    severe_recall = recall_score(true_severe, pred_severe, zero_division=0)
    severe_count = int(true_severe.sum())
    severe_underclassification_rate = (
        float(np.logical_and(true_severe, prediction < 3).sum() / severe_count)
        if severe_count
        else 0.0
    )
    one_hot = np.eye(4)[labels]
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted")),
        "high_risk_recall": float(high_risk_recall),
        "severe_recall": float(severe_recall),
        "severe_underclassification_rate": severe_underclassification_rate,
        "severity_mae": float(np.abs(labels - prediction).mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1, 2, 3])),
        "multiclass_brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
    }


def _fit_minimum_recall_threshold(
    severe_scores: np.ndarray,
    labels: np.ndarray,
    minimum_recall: float,
) -> float:
    """Fit the most selective severe-score threshold meeting validation recall.

    Only positive validation examples determine the threshold.  Choosing the
    highest feasible threshold minimizes unnecessary escalations while meeting
    the predeclared safety target; test labels are never consulted.
    """

    target = float(minimum_recall)
    if not 0.0 < target <= 1.0:
        raise ValueError("minimum_recall 必须在 (0, 1] 内")
    scores = np.asarray(severe_scores, dtype=float)
    truth = np.asarray(labels) == 3
    positive_scores = np.sort(scores[truth])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("验证集中没有严重风险样本，无法校准哨兵阈值")
    required = int(np.ceil(target * len(positive_scores)))
    return float(positive_scores[required - 1])


def _apply_severe_sentinel(
    base_probability: np.ndarray,
    severe_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Escalate only sentinel-qualified rows to severe, preserving confidence."""

    output = np.asarray(base_probability, dtype=float).copy()
    scores = np.asarray(severe_scores, dtype=float)
    if len(output) != len(scores):
        raise ValueError("严重风险得分与概率行数不一致")
    current = output.argmax(axis=1)
    trigger_rows = np.flatnonzero(np.logical_and(scores >= threshold, current != 3))
    for row in trigger_rows:
        left = current[row]
        output[row, left], output[row, 3] = output[row, 3], output[row, left]
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def _causal_probability_filter(
    probability: np.ndarray,
    smoothing: float,
    initial_probability: np.ndarray | None = None,
) -> np.ndarray:
    """Exponentially smooth class probabilities without access to future rows."""

    values = np.asarray(probability, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("风险概率必须是非空二维数组")
    alpha = float(smoothing)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("smoothing 必须在 [0, 1) 内")
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    output = np.empty_like(normalized)
    if initial_probability is None:
        state = normalized[0].copy()
        output[0] = state
        start = 1
    else:
        state = np.asarray(initial_probability, dtype=float).copy()
        state /= max(float(state.sum()), 1e-12)
        start = 0
    for index in range(start, len(normalized)):
        state = alpha * state + (1.0 - alpha) * normalized[index]
        state /= max(float(state.sum()), 1e-12)
        # Escalation is immediate; only de-escalation is allowed to benefit
        # from persistence.  Swap retains normalization and confidence scale.
        raw_class = int(normalized[index].argmax())
        filtered_class = int(state.argmax())
        if raw_class > filtered_class:
            state[filtered_class], state[raw_class] = (
                state[raw_class],
                state[filtered_class],
            )
        output[index] = state
    return output


def _one_sided_conformal_correction(
    actual_peak: np.ndarray,
    forecast_peak: np.ndarray,
    coverage: float,
) -> float:
    """Finite-sample one-sided correction for underestimated future peaks."""

    target = float(coverage)
    if not 0.0 < target < 1.0:
        raise ValueError("coverage 必须在 (0, 1) 内")
    residual = np.asarray(actual_peak, dtype=float) - np.asarray(
        forecast_peak, dtype=float
    )
    if residual.ndim != 1 or len(residual) == 0:
        raise ValueError("峰值误差必须是非空一维序列")
    quantile_level = min(1.0, np.ceil((len(residual) + 1) * target) / len(residual))
    correction = float(np.quantile(residual, quantile_level, method="higher"))
    return max(correction, 0.0)


def _margin_classes(margin_ratio: np.ndarray, label_cfg: dict) -> np.ndarray:
    levels = np.full(len(margin_ratio), 3, dtype=np.int64)
    levels[margin_ratio >= float(label_cfg["warning_margin_ratio"])] = 2
    levels[margin_ratio >= float(label_cfg["watch_margin_ratio"])] = 1
    levels[margin_ratio >= float(label_cfg["normal_margin_ratio"])] = 0
    return levels


def _calibrated_evidence_probabilities(
    calibration_margin: np.ndarray,
    calibration_labels: np.ndarray,
    target_margin: np.ndarray,
    label_cfg: dict,
    smoothing: float = 2.0,
) -> np.ndarray:
    """Calibrate the transparent margin rule without changing its evidence basis."""

    calibration_class = _margin_classes(calibration_margin, label_cfg)
    table = np.full((4, 4), smoothing, dtype=float)
    for predicted, actual in zip(calibration_class, calibration_labels):
        table[predicted, actual] += 1.0
    table /= table.sum(axis=1, keepdims=True)
    return table[_margin_classes(target_margin, label_cfg)]


def _ordinal_probabilities(models: list[xgb.XGBClassifier], features: np.ndarray) -> np.ndarray:
    cumulative = np.column_stack(
        [model.predict_proba(features)[:, 1] for model in models]
    )
    # Project P(Y>=1), P(Y>=2), P(Y>=3) onto a monotone sequence.
    cumulative[:, 1] = np.minimum(cumulative[:, 1], cumulative[:, 0])
    cumulative[:, 2] = np.minimum(cumulative[:, 2], cumulative[:, 1])
    probability = np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ]
    )
    probability = np.maximum(probability, 0.0)
    return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)


def _apply_ordinal_cumulative_threshold(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply one shared threshold to the three monotone ordinal boundaries."""

    output = np.asarray(probability, dtype=float).copy()
    value = float(threshold)
    if not 0.0 < value < 1.0:
        raise ValueError("ordinal threshold 必须在 (0, 1) 内")
    cumulative = np.column_stack(
        [
            1.0 - output[:, 0],
            output[:, 2] + output[:, 3],
            output[:, 3],
        ]
    )
    target_class = (cumulative >= value).sum(axis=1).astype(np.int64)
    current_class = output.argmax(axis=1)
    for row in np.flatnonzero(current_class != target_class):
        left = current_class[row]
        right = target_class[row]
        output[row, left], output[row, right] = output[row, right], output[row, left]
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def _evidence_guided_probabilities(
    learned_probability: np.ndarray,
    evidence_probability: np.ndarray,
) -> np.ndarray:
    """Keep learned confidence while enforcing the auditable adequacy grade.

    When the learned ensemble crosses an adjacent threshold, swap the two class
    probabilities.  The reported grade therefore always remains traceable to
    the configured supply-margin rule, while confidence still reflects the ML
    ensemble and its forecast-error experience.
    """

    output = np.asarray(learned_probability, dtype=float).copy()
    learned_class = output.argmax(axis=1)
    evidence_class = evidence_probability.argmax(axis=1)
    for row in np.flatnonzero(learned_class != evidence_class):
        left = learned_class[row]
        right = evidence_class[row]
        output[row, left], output[row, right] = output[row, right], output[row, left]
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def _safety_union_probabilities(
    *probabilities: np.ndarray,
) -> np.ndarray:
    """Conservative ensemble for a safety-oriented operational signal.

    Confidence is the mean of three complementary estimators, while the hard
    grade is never lower than any constituent grade.  This implements an
    auditable OR rule for warning/severe evidence and avoids a high-risk vote
    being diluted by probability averaging.
    """

    if len(probabilities) < 2:
        raise ValueError("Safety-Union 至少需要两个互补概率源")
    sources = [np.asarray(value, dtype=float) for value in probabilities]
    output = np.mean(sources, axis=0)
    target_class = np.maximum.reduce([source.argmax(axis=1) for source in sources])
    current_class = output.argmax(axis=1)
    for row in np.flatnonzero(current_class != target_class):
        left = current_class[row]
        right = target_class[row]
        output[row, left], output[row, right] = output[row, right], output[row, left]
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def _plot_metric_comparison(metrics: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    columns = [
        ("macro_f1", "宏平均F1（越高越好）"),
        ("high_risk_recall", "高风险召回率（越高越好）"),
        ("severe_recall", "严重风险召回率（越高越好）"),
        ("log_loss", "对数损失（越低越好）"),
    ]
    for ax, (column, title) in zip(axes.flat, columns):
        values = metrics.set_index("model")[column]
        ax.barh(values.index, values.values, color="#2563EB", alpha=0.84)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusions(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    count = len(probabilities)
    columns = 3 if count > 4 else 2
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.5 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    names_zh = [RISK_NAMES_ZH[index] for index in range(4)]
    for ax, (name, probability) in zip(axes_array, probabilities.items()):
        matrix = confusion_matrix(labels, probability.argmax(axis=1), labels=[0, 1, 2, 3])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(4):
            for column in range(4):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[row, column] > 0.55 else "#111827",
                )
        ax.set_title(name)
        ax.set_xticks(range(4), names_zh)
        ax.set_yticks(range(4), names_zh)
        ax.set_xlabel("预测等级")
        ax.set_ylabel("真实等级")
    for ax in axes_array[count:]:
        ax.axis("off")
    fig.colorbar(image, ax=axes_array[:count].tolist(), shrink=0.75)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline(predictions: pd.DataFrame, path: Path, limit: int = 1800) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    view = predictions.iloc[: min(limit, len(predictions))]
    x = np.arange(len(view)) * 5.0 / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(x, view["actual_peak_kw"], label="未来实际峰值", color="#111827")
    axes[0].plot(x, view["forecast_peak_kw"], label="预测峰值", color="#2563EB", alpha=0.85)
    axes[0].plot(x, view["firm_supply_kw"], label="可用供电能力", color="#10B981")
    axes[0].set_ylabel("功率/kW")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(alpha=0.18)
    axes[1].step(x, view["true_risk_level"], where="post", label="真实风险", color="#111827")
    axes[1].step(
        x,
        view["predicted_risk_level"],
        where="post",
        label="集成预测风险",
        color="#DC2626",
        alpha=0.8,
    )
    axes[1].set_yticks(range(4), [RISK_NAMES_ZH[index] for index in range(4)])
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    axes[2].plot(x, view["risk_confidence"], color="#7C3AED", label="分类置信度")
    axes[2].plot(x, view["prob_warning"] + view["prob_severe"], color="#F97316", label="高风险概率")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_ylabel("概率")
    axes[2].set_xlabel("测试时段/分钟")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_risk_benchmark(
    prediction_path: Path,
    prediction_key_path: Path,
    data_path: Path,
    forecast_config_path: Path,
    risk_config_path: Path,
    artifact_dir: Path,
    *,
    supply_path: Path | None = None,
    force_cpu: bool = False,
) -> pd.DataFrame:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(exist_ok=True)
    risk_cfg = _load_yaml(risk_config_path)
    forecast_cfg = _load_yaml(forecast_config_path)
    seed = int(risk_cfg["project"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    arrays = np.load(prediction_path)
    key_map = json.loads(Path(prediction_key_path).read_text(encoding="utf-8"))
    name_to_key = {name: key for key, name in key_map.items()}
    model_name = str(risk_cfg["input"]["forecast_model"])
    if model_name not in name_to_key:
        raise ValueError(f"风险输入预测模型不存在: {model_name}; 可用: {sorted(name_to_key)}")
    forecast_sequence = arrays[name_to_key[model_name]].astype(np.float32)
    actual_future = arrays["y_true"].astype(np.float32)
    transition_flags, transition_feature_source = _causal_transition_feature(
        arrays, len(forecast_sequence)
    )
    member_names = [
        name
        for name in risk_cfg["input"]["uncertainty_members"]
        if name in name_to_key
    ]
    member_predictions = np.stack(
        [arrays[name_to_key[name]].astype(np.float32) for name in member_names], axis=0
    )
    uncertainty_sequence = member_predictions.std(axis=0)

    frame = pd.read_parquet(data_path)
    frame, schema_report = validate_canonical_frame(frame)
    _, _, test_frame = split_frame_by_time(
        frame,
        float(forecast_cfg["forecast"]["train_fraction"]),
        float(forecast_cfg["forecast"]["val_fraction"]),
    )
    history = int(forecast_cfg["forecast"]["history_steps"])
    horizon = int(forecast_cfg["forecast"]["horizon_steps"])
    eval_stride = int(forecast_cfg["forecast"]["eval_stride"])
    origins = np.arange(
        history,
        len(test_frame) - horizon + 1,
        max(1, eval_stride),
    )
    if len(origins) != len(actual_future):
        raise ValueError(
            f"预测样本数 {len(actual_future)} 与按配置对齐的数据窗口数 {len(origins)} 不一致"
        )
    origin_frame = test_frame.iloc[origins - 1].reset_index(drop=True)
    target_timestamps = test_frame.iloc[origins]["timestamp"].reset_index(drop=True)

    if supply_path is not None:
        supply = _load_external_supply(supply_path, target_timestamps)
    else:
        supply = _supply_from_canonical(origin_frame, target_timestamps)
        if supply is None:
            preliminary_peak = forecast_sequence.max(axis=1).astype(float)
            preliminary_required = preliminary_peak + np.maximum(
                float(risk_cfg["label"]["reserve_floor_kw"]),
                float(risk_cfg["label"]["reserve_ratio"]) * preliminary_peak,
            )
            supply = _simulate_supply_context(
                target_timestamps, preliminary_required, risk_cfg, seed
            )
    supply = supply.reset_index(drop=True)
    firm_supply = (
        supply["grid_available_capacity_kw"].to_numpy(float)
        + supply["generator_available_capacity_kw"].to_numpy(float)
        + supply["storage_available_discharge_power_kw"].to_numpy(float)
    )

    label_cfg = risk_cfg["label"]
    labels, actual_margin_kw, actual_margin_ratio = assign_risk_levels(
        actual_future,
        firm_supply,
        reserve_ratio=float(label_cfg["reserve_ratio"]),
        reserve_floor_kw=float(label_cfg["reserve_floor_kw"]),
        normal_margin_ratio=float(label_cfg["normal_margin_ratio"]),
        watch_margin_ratio=float(label_cfg["watch_margin_ratio"]),
        warning_margin_ratio=float(label_cfg["warning_margin_ratio"]),
    )
    current_load = arrays[name_to_key["Persistence"]][:, 0].astype(float)
    forecast_peak = forecast_sequence.max(axis=1).astype(float)
    forecast_required = forecast_peak + np.maximum(
        float(label_cfg["reserve_floor_kw"]),
        float(label_cfg["reserve_ratio"]) * forecast_peak,
    )
    forecast_margin_ratio = (firm_supply - forecast_required) / np.maximum(
        forecast_required, 1.0
    )
    interval_seconds = float(forecast_cfg["project"]["frequency_seconds"])
    forecast_ramp = np.maximum(
        np.diff(np.column_stack([current_load, forecast_sequence]), axis=1), 0.0
    ).max(axis=1) / interval_seconds
    timestamp = pd.to_datetime(target_timestamps)
    seconds = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    phase = 2.0 * np.pi * seconds.to_numpy() / 86400.0
    uncertainty_mean = uncertainty_sequence.mean(axis=1)
    uncertainty_max = uncertainty_sequence.max(axis=1)
    static_names = [
        "current_load_kw",
        "forecast_mean_kw",
        "forecast_max_kw",
        "forecast_min_kw",
        "forecast_std_kw",
        "forecast_ramp_kw_per_s",
        "uncertainty_mean_kw",
        "uncertainty_max_kw",
        "grid_capacity_kw",
        "generator_available_kw",
        "storage_soc_pct",
        "storage_discharge_available_kw",
        "firm_supply_kw",
        "forecast_margin_ratio",
        "history_transition_flag",
        "operation_state_code",
        "time_sin",
        "time_cos",
    ]
    static = np.column_stack(
        [
            current_load,
            forecast_sequence.mean(axis=1),
            forecast_peak,
            forecast_sequence.min(axis=1),
            forecast_sequence.std(axis=1),
            forecast_ramp,
            uncertainty_mean,
            uncertainty_max,
            supply["grid_available_capacity_kw"],
            supply["generator_available_capacity_kw"],
            supply["storage_soc_pct"],
            supply["storage_available_discharge_power_kw"],
            firm_supply,
            forecast_margin_ratio,
            transition_flags,
            origin_frame["operation_state_code"].to_numpy(),
            np.sin(phase),
            np.cos(phase),
        ]
    ).astype(np.float32)
    tree_features = np.column_stack([forecast_sequence, static]).astype(np.float32)
    tree_feature_names = [f"forecast_t+{index + 1}_kw" for index in range(horizon)] + static_names

    split_cfg = risk_cfg["split"]
    train_stop = int(len(labels) * float(split_cfg["train_fraction"]))
    val_stop = int(len(labels) * (float(split_cfg["train_fraction"]) + float(split_cfg["val_fraction"])))
    train_idx = np.arange(0, train_stop)
    val_idx = np.arange(train_stop, val_stop)
    test_idx = np.arange(val_stop, len(labels))
    for split_name, indices in {"train": train_idx, "validation": val_idx, "test": test_idx}.items():
        missing = set(range(4)) - set(labels[indices].tolist())
        if missing:
            raise ValueError(f"{split_name} 时间切分缺少风险等级: {sorted(missing)}，请调整仿真供能场景")

    tree_scaler = StandardScaler().fit(tree_features[train_idx])
    static_scaler = StandardScaler().fit(static[train_idx])
    sequence_mean = float(forecast_sequence[train_idx].mean())
    sequence_std = max(float(forecast_sequence[train_idx].std()), 1e-6)
    sequence_scaled = (forecast_sequence - sequence_mean) / sequence_std
    static_scaled = static_scaler.transform(static).astype(np.float32)

    probabilities_val: dict[str, np.ndarray] = {}
    probabilities_test: dict[str, np.ndarray] = {}
    training_rows = []
    probabilities_val["Evidence-Margin"] = _calibrated_evidence_probabilities(
        forecast_margin_ratio[train_idx],
        labels[train_idx],
        forecast_margin_ratio[val_idx],
        label_cfg,
    )
    probabilities_test["Evidence-Margin"] = _calibrated_evidence_probabilities(
        forecast_margin_ratio[train_idx],
        labels[train_idx],
        forecast_margin_ratio[test_idx],
        label_cfg,
    )
    training_rows.append({"model": "Evidence-Margin", "training_seconds": 0.0})
    conformal_coverage = float(risk_cfg.get("conformal", {}).get("coverage", 0.90))
    conformal_correction_kw = _one_sided_conformal_correction(
        actual_future[train_idx].max(axis=1),
        forecast_peak[train_idx],
        conformal_coverage,
    )
    conformal_peak = forecast_peak + conformal_correction_kw
    conformal_required = conformal_peak + np.maximum(
        float(label_cfg["reserve_floor_kw"]),
        float(label_cfg["reserve_ratio"]) * conformal_peak,
    )
    conformal_margin_ratio = (firm_supply - conformal_required) / np.maximum(
        conformal_required, 1.0
    )
    probabilities_val["Conformal-Margin"] = _calibrated_evidence_probabilities(
        conformal_margin_ratio[train_idx],
        labels[train_idx],
        conformal_margin_ratio[val_idx],
        label_cfg,
    )
    probabilities_test["Conformal-Margin"] = _calibrated_evidence_probabilities(
        conformal_margin_ratio[train_idx],
        labels[train_idx],
        conformal_margin_ratio[test_idx],
        label_cfg,
    )
    training_rows.append({"model": "Conformal-Margin", "training_seconds": 0.0})
    logistic_cfg = risk_cfg["models"]["logistic_regression"]
    started = time.perf_counter()
    logistic = LogisticRegression(
        C=float(logistic_cfg["C"]),
        max_iter=int(logistic_cfg["max_iter"]),
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    logistic.fit(tree_scaler.transform(tree_features[train_idx]), labels[train_idx])
    elapsed = time.perf_counter() - started
    probabilities_val["Logistic-Regression"] = logistic.predict_proba(
        tree_scaler.transform(tree_features[val_idx])
    )
    probabilities_test["Logistic-Regression"] = logistic.predict_proba(
        tree_scaler.transform(tree_features[test_idx])
    )
    training_rows.append({"model": "Logistic-Regression", "training_seconds": elapsed})
    joblib.dump(
        {"model": logistic, "scaler": tree_scaler, "feature_names": tree_feature_names},
        model_dir / "logistic_regression.joblib",
    )

    xgb_cfg = risk_cfg["models"]["xgboost"]
    started = time.perf_counter()
    ordinal_models: list[xgb.XGBClassifier] = []
    for boundary in (1, 2, 3):
        binary_label = (labels[train_idx] >= boundary).astype(np.int64)
        positive = max(int(binary_label.sum()), 1)
        negative = max(len(binary_label) - positive, 1)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=int(xgb_cfg["n_estimators"]),
            max_depth=int(xgb_cfg["max_depth"]),
            learning_rate=float(xgb_cfg["learning_rate"]),
            subsample=float(xgb_cfg["subsample"]),
            colsample_bytree=float(xgb_cfg["colsample_bytree"]),
            min_child_weight=float(xgb_cfg["min_child_weight"]),
            reg_lambda=float(xgb_cfg["reg_lambda"]),
            scale_pos_weight=negative / positive,
            tree_method="hist",
            device="cuda" if device.type == "cuda" else "cpu",
            random_state=seed + boundary,
            n_jobs=1,
        )
        model.fit(tree_features[train_idx], binary_label)
        ordinal_models.append(model)
    elapsed = time.perf_counter() - started
    probabilities_val["Ordinal-XGBoost"] = _ordinal_probabilities(
        ordinal_models, tree_features[val_idx]
    )
    probabilities_test["Ordinal-XGBoost"] = _ordinal_probabilities(
        ordinal_models, tree_features[test_idx]
    )
    training_rows.append({"model": "Ordinal-XGBoost", "training_seconds": elapsed})
    joblib.dump(
        {"models": ordinal_models, "boundaries": [1, 2, 3], "feature_names": tree_feature_names},
        model_dir / "ordinal_xgboost_risk.joblib",
    )

    started = time.perf_counter()
    class_counts = np.bincount(labels[train_idx], minlength=4).astype(float)
    class_weights = len(train_idx) / np.maximum(4.0 * class_counts, 1.0)
    multiclass_xgboost = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=float(xgb_cfg["min_child_weight"]),
        reg_lambda=float(xgb_cfg["reg_lambda"]),
        tree_method="hist",
        device="cuda" if device.type == "cuda" else "cpu",
        random_state=seed + 10,
        n_jobs=1,
    )
    multiclass_xgboost.fit(
        tree_features[train_idx],
        labels[train_idx],
        sample_weight=class_weights[labels[train_idx]],
    )
    elapsed = time.perf_counter() - started
    probabilities_val["Balanced-Multiclass-XGBoost"] = (
        multiclass_xgboost.predict_proba(tree_features[val_idx])
    )
    probabilities_test["Balanced-Multiclass-XGBoost"] = (
        multiclass_xgboost.predict_proba(tree_features[test_idx])
    )
    training_rows.append(
        {"model": "Balanced-Multiclass-XGBoost", "training_seconds": elapsed}
    )
    joblib.dump(
        {
            "model": multiclass_xgboost,
            "class_weights": class_weights.tolist(),
            "feature_names": tree_feature_names,
        },
        model_dir / "balanced_multiclass_xgboost_risk.joblib",
    )

    transformer_cfg = risk_cfg["models"]["temporal_transformer"]
    started = time.perf_counter()
    transformer_result = train_transformer_classifier(
        sequence_scaled[train_idx].astype(np.float32),
        static_scaled[train_idx],
        labels[train_idx],
        sequence_scaled[val_idx].astype(np.float32),
        static_scaled[val_idx],
        labels[val_idx],
        device=device,
        hidden_size=int(transformer_cfg["hidden_size"]),
        attention_heads=int(transformer_cfg["attention_heads"]),
        layers=int(transformer_cfg["layers"]),
        dropout=float(transformer_cfg["dropout"]),
        learning_rate=float(transformer_cfg["learning_rate"]),
        weight_decay=float(transformer_cfg["weight_decay"]),
        batch_size=int(transformer_cfg["batch_size"]),
        max_epochs=int(transformer_cfg["max_epochs"]),
        patience=int(transformer_cfg["patience"]),
    )
    elapsed = time.perf_counter() - started
    probabilities_val["Temporal-Transformer"] = predict_transformer_probabilities(
        transformer_result.model,
        sequence_scaled[val_idx].astype(np.float32),
        static_scaled[val_idx],
        device,
        int(transformer_cfg["batch_size"]),
    )
    probabilities_test["Temporal-Transformer"] = predict_transformer_probabilities(
        transformer_result.model,
        sequence_scaled[test_idx].astype(np.float32),
        static_scaled[test_idx],
        device,
        int(transformer_cfg["batch_size"]),
    )
    training_rows.append({"model": "Temporal-Transformer", "training_seconds": elapsed})
    torch.save(
        {
            "state_dict": transformer_result.model.state_dict(),
            "sequence_mean": sequence_mean,
            "sequence_std": sequence_std,
            "static_scaler": static_scaler,
            "static_names": static_names,
            "config": transformer_cfg,
        },
        model_dir / "temporal_transformer.pt",
    )

    ensemble_weights = fit_probability_ensemble(
        probabilities_val,
        labels[val_idx],
        float(risk_cfg["ensemble"]["l2_penalty"]),
    )
    probabilities_val["Validation-Weighted-Ensemble"] = apply_probability_ensemble(
        probabilities_val, ensemble_weights
    )
    probabilities_test["Validation-Weighted-Ensemble"] = apply_probability_ensemble(
        probabilities_test, ensemble_weights
    )
    probabilities_val["Evidence-Guided-Ensemble"] = _evidence_guided_probabilities(
        probabilities_val["Validation-Weighted-Ensemble"],
        probabilities_val["Evidence-Margin"],
    )
    probabilities_test["Evidence-Guided-Ensemble"] = _evidence_guided_probabilities(
        probabilities_test["Validation-Weighted-Ensemble"],
        probabilities_test["Evidence-Margin"],
    )
    probabilities_val["Safety-Union-Ensemble"] = _safety_union_probabilities(
        probabilities_val["Validation-Weighted-Ensemble"],
        probabilities_val["Evidence-Margin"],
        probabilities_val["Logistic-Regression"],
    )
    probabilities_test["Safety-Union-Ensemble"] = _safety_union_probabilities(
        probabilities_test["Validation-Weighted-Ensemble"],
        probabilities_test["Evidence-Margin"],
        probabilities_test["Logistic-Regression"],
    )
    selection_cfg = risk_cfg.get("operational_selection", {})
    minimum_high_recall = float(selection_cfg.get("minimum_validation_high_risk_recall", 0.80))
    minimum_severe_recall = float(
        selection_cfg.get("minimum_validation_severe_recall", 0.80)
    )
    minimum_macro_f1 = float(
        selection_cfg.get("minimum_validation_macro_f1", 0.70)
    )
    ordinal_threshold_trials = []
    ordinal_threshold_outputs = {}
    for threshold in risk_cfg.get("ordinal_calibration", {}).get(
        "threshold_candidates", [0.3, 0.4, 0.5, 0.6, 0.7]
    ):
        threshold = float(threshold)
        calibrated = _apply_ordinal_cumulative_threshold(
            probabilities_val["Ordinal-XGBoost"], threshold
        )
        item = {"threshold": threshold}
        item.update(_classification_metrics(labels[val_idx], calibrated))
        item["selection_score"] = (
            0.40 * item["macro_f1"]
            + 0.20 * item["balanced_accuracy"]
            + 0.20 * item["high_risk_recall"]
            + 0.20 * item["severe_recall"]
            - 0.10 * item["severity_mae"]
        )
        item["eligible"] = (
            item["macro_f1"] >= minimum_macro_f1
            and item["high_risk_recall"] >= minimum_high_recall
            and item["severe_recall"] >= minimum_severe_recall
        )
        ordinal_threshold_trials.append(item)
        ordinal_threshold_outputs[threshold] = calibrated
    ordinal_threshold_pool = [
        item for item in ordinal_threshold_trials if item["eligible"]
    ] or ordinal_threshold_trials
    selected_ordinal_threshold = max(
        ordinal_threshold_pool,
        key=lambda item: (item["selection_score"], -item["log_loss"]),
    )
    selected_ordinal_threshold_value = float(selected_ordinal_threshold["threshold"])
    probabilities_val["Validation-Calibrated-Ordinal"] = ordinal_threshold_outputs[
        selected_ordinal_threshold_value
    ]
    probabilities_test["Validation-Calibrated-Ordinal"] = (
        _apply_ordinal_cumulative_threshold(
            probabilities_test["Ordinal-XGBoost"],
            selected_ordinal_threshold_value,
        )
    )
    sentinel_target_recall = float(
        selection_cfg.get("severe_sentinel_validation_recall_target", 0.85)
    )
    severe_sentinel_threshold = _fit_minimum_recall_threshold(
        probabilities_val["Ordinal-XGBoost"][:, 3],
        labels[val_idx],
        sentinel_target_recall,
    )
    probabilities_val["Safety-Union-Ensemble"] = _apply_severe_sentinel(
        probabilities_val["Safety-Union-Ensemble"],
        probabilities_val["Ordinal-XGBoost"][:, 3],
        severe_sentinel_threshold,
    )
    probabilities_test["Safety-Union-Ensemble"] = _apply_severe_sentinel(
        probabilities_test["Safety-Union-Ensemble"],
        probabilities_test["Ordinal-XGBoost"][:, 3],
        severe_sentinel_threshold,
    )
    probabilities_val["Severe-Guarded-Logistic"] = _apply_severe_sentinel(
        probabilities_val["Logistic-Regression"],
        probabilities_val["Ordinal-XGBoost"][:, 3],
        severe_sentinel_threshold,
    )
    probabilities_test["Severe-Guarded-Logistic"] = _apply_severe_sentinel(
        probabilities_test["Logistic-Regression"],
        probabilities_test["Ordinal-XGBoost"][:, 3],
        severe_sentinel_threshold,
    )
    probabilities_val["Severe-Guarded-Multiclass-XGBoost"] = (
        _apply_severe_sentinel(
            probabilities_val["Balanced-Multiclass-XGBoost"],
            probabilities_val["Ordinal-XGBoost"][:, 3],
            severe_sentinel_threshold,
        )
    )
    probabilities_test["Severe-Guarded-Multiclass-XGBoost"] = (
        _apply_severe_sentinel(
            probabilities_test["Balanced-Multiclass-XGBoost"],
            probabilities_test["Ordinal-XGBoost"][:, 3],
            severe_sentinel_threshold,
        )
    )
    hybrid_trials = []
    hybrid_outputs = {}
    for balanced_weight in risk_cfg.get("balanced_ordinal_hybrid", {}).get(
        "balanced_weight_candidates", [0.0, 0.25, 0.5, 0.75, 1.0]
    ):
        balanced_weight = float(balanced_weight)
        ordinal_weight = 1.0 - balanced_weight
        hybrid_val = (
            balanced_weight * probabilities_val["Balanced-Multiclass-XGBoost"]
            + ordinal_weight * probabilities_val["Ordinal-XGBoost"]
        )
        item = {
            "balanced_weight": balanced_weight,
            "ordinal_weight": ordinal_weight,
        }
        item.update(_classification_metrics(labels[val_idx], hybrid_val))
        item["selection_score"] = (
            0.40 * item["macro_f1"]
            + 0.20 * item["balanced_accuracy"]
            + 0.20 * item["high_risk_recall"]
            + 0.20 * item["severe_recall"]
            - 0.10 * item["severity_mae"]
        )
        item["eligible"] = (
            item["macro_f1"] >= minimum_macro_f1
            and item["high_risk_recall"] >= minimum_high_recall
            and item["severe_recall"] >= minimum_severe_recall
        )
        hybrid_trials.append(item)
        hybrid_outputs[balanced_weight] = hybrid_val
    hybrid_pool = [item for item in hybrid_trials if item["eligible"]] or hybrid_trials
    selected_hybrid = max(
        hybrid_pool,
        key=lambda item: (item["selection_score"], -item["log_loss"]),
    )
    selected_balanced_weight = float(selected_hybrid["balanced_weight"])
    selected_ordinal_weight = 1.0 - selected_balanced_weight
    probabilities_val["Balanced-Ordinal-Hybrid"] = hybrid_outputs[
        selected_balanced_weight
    ]
    probabilities_test["Balanced-Ordinal-Hybrid"] = (
        selected_balanced_weight
        * probabilities_test["Balanced-Multiclass-XGBoost"]
        + selected_ordinal_weight * probabilities_test["Ordinal-XGBoost"]
    )
    probabilities_val["Balanced-Ordinal-Safety-Max"] = _safety_union_probabilities(
        probabilities_val["Balanced-Multiclass-XGBoost"],
        probabilities_val["Ordinal-XGBoost"],
    )
    probabilities_test["Balanced-Ordinal-Safety-Max"] = _safety_union_probabilities(
        probabilities_test["Balanced-Multiclass-XGBoost"],
        probabilities_test["Ordinal-XGBoost"],
    )
    filter_cfg = risk_cfg.get("causal_filter", {})
    filter_trials = []
    filter_outputs: dict[tuple[str, float], np.ndarray] = {}
    for source_name in filter_cfg.get(
        "source_models", ["Ordinal-XGBoost", "Safety-Union-Ensemble"]
    ):
        if source_name not in probabilities_val:
            raise ValueError(f"因果风险滤波输入不存在: {source_name}")
        for smoothing in filter_cfg.get(
            "smoothing_candidates", [0.0, 0.2, 0.5, 0.8]
        ):
            smoothing = float(smoothing)
            filtered = _causal_probability_filter(
                probabilities_val[source_name], smoothing
            )
            item = {
                "source_model": source_name,
                "smoothing": smoothing,
            }
            item.update(_classification_metrics(labels[val_idx], filtered))
            item["selection_score"] = (
                0.40 * item["macro_f1"]
                + 0.20 * item["balanced_accuracy"]
                + 0.20 * item["high_risk_recall"]
                + 0.20 * item["severe_recall"]
                - 0.10 * item["severity_mae"]
            )
            item["eligible"] = (
                item["macro_f1"] >= minimum_macro_f1
                and item["high_risk_recall"] >= minimum_high_recall
                and item["severe_recall"] >= minimum_severe_recall
            )
            filter_trials.append(item)
            filter_outputs[(source_name, smoothing)] = filtered
    eligible_filter_trials = [item for item in filter_trials if item["eligible"]]
    filter_pool = eligible_filter_trials or filter_trials
    selected_filter = max(
        filter_pool,
        key=lambda item: (item["selection_score"], -item["log_loss"]),
    )
    filter_key = (
        str(selected_filter["source_model"]),
        float(selected_filter["smoothing"]),
    )
    filtered_validation = filter_outputs[filter_key]
    probabilities_val["Causal-Risk-Filter"] = filtered_validation
    probabilities_test["Causal-Risk-Filter"] = _causal_probability_filter(
        probabilities_test[filter_key[0]],
        filter_key[1],
        initial_probability=filtered_validation[-1],
    )
    validation_selection = []
    for name, probability in probabilities_val.items():
        item = {"model": name}
        item.update(_classification_metrics(labels[val_idx], probability))
        item["selection_score"] = (
            0.40 * item["macro_f1"]
            + 0.20 * item["balanced_accuracy"]
            + 0.20 * item["high_risk_recall"]
            + 0.20 * item["severe_recall"]
            - 0.10 * item["severity_mae"]
        )
        item["eligible"] = (
            item["macro_f1"] >= minimum_macro_f1
            and item["high_risk_recall"] >= minimum_high_recall
            and item["severe_recall"] >= minimum_severe_recall
        )
        validation_selection.append(item)
    allowed_models = set(
        selection_cfg.get(
            "candidate_models",
            ["Validation-Weighted-Ensemble", "Evidence-Guided-Ensemble"],
        )
    )
    allowed_items = [
        item for item in validation_selection if item["model"] in allowed_models
    ]
    eligible = [item for item in allowed_items if item["eligible"]]
    if not eligible:
        raise ValueError(
            "无风险模型同时通过验证集 Macro-F1、高风险召回和严重风险召回门槛"
        )
    selection_pool = eligible
    best_score = max(item["selection_score"] for item in selection_pool)
    tie_tolerance = float(selection_cfg.get("score_tie_tolerance", 0.01))
    statistically_tied = [
        item for item in selection_pool if item["selection_score"] >= best_score - tie_tolerance
    ]
    selected_by_validation = min(
        statistically_tied, key=lambda item: item["log_loss"]
    )["model"]
    fixed_operational_model = selection_cfg.get("fixed_operational_model")
    if fixed_operational_model is not None:
        operational_model = str(fixed_operational_model)
        fixed_item = next(
            (item for item in validation_selection if item["model"] == operational_model),
            None,
        )
        if fixed_item is None:
            raise ValueError(f"预声明运行模型不存在: {operational_model}")
        if not fixed_item["eligible"]:
            raise ValueError(
                f"预声明运行模型 {operational_model} 未通过验证集高风险/"
                "严重风险召回门槛"
            )
        selection_mode = "predeclared_safety_policy_with_validation_gate"
    else:
        operational_model = selected_by_validation
        selection_mode = "validation_only"
    selected_validation_family = operational_model
    dispatch_guard_model = str(
        selection_cfg.get(
            "dispatch_guard_model", "Balanced-Ordinal-Safety-Max"
        )
    )
    guard_validation_item = next(
        (
            item
            for item in validation_selection
            if item["model"] == dispatch_guard_model
        ),
        None,
    )
    if guard_validation_item is None:
        raise ValueError(f"预声明调度保护模型不存在: {dispatch_guard_model}")
    if (
        guard_validation_item["high_risk_recall"] < minimum_high_recall
        or guard_validation_item["severe_recall"] < minimum_severe_recall
    ):
        raise ValueError("调度保护模型未通过验证集高风险/严重风险召回门槛")
    metrics_rows = []
    for name, probability in probabilities_test.items():
        row = {
            "model": name,
            "operational_selected": name == operational_model,
            "dispatch_guard_selected": name == dispatch_guard_model,
        }
        row.update(_classification_metrics(labels[test_idx], probability))
        training_match = next(
            (item["training_seconds"] for item in training_rows if item["model"] == name), 0.0
        )
        row["training_seconds"] = training_match
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(artifact_dir / "risk_metrics.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "ensemble_weights.json").write_text(
        json.dumps(ensemble_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "operational_model_selection.json").write_text(
        json.dumps(
            {
                "selected_model": operational_model,
                "selected_validation_family": selected_validation_family,
                "dispatch_guard_model": dispatch_guard_model,
                "dispatch_guard_selection": (
                    "predeclared_model_with_validation_high_and_severe_recall_gates"
                ),
                "validation_score_selected_model": selected_by_validation,
                "selection_mode": selection_mode,
                "minimum_validation_macro_f1": minimum_macro_f1,
                "minimum_validation_high_risk_recall": minimum_high_recall,
                "minimum_validation_severe_recall": minimum_severe_recall,
                "severe_sentinel": {
                    "source_model": "Ordinal-XGBoost",
                    "score": "P(Y>=severe)",
                    "validation_recall_target": sentinel_target_recall,
                    "validation_fitted_threshold": severe_sentinel_threshold,
                    "fit_data": "validation_only; test labels excluded",
                },
                "ordinal_threshold_calibration": {
                    "selected_threshold": selected_ordinal_threshold_value,
                    "selection_data": "validation_only; test labels excluded",
                    "trials": ordinal_threshold_trials,
                },
                "balanced_ordinal_hybrid": {
                    "selected_balanced_weight": selected_balanced_weight,
                    "selected_ordinal_weight": selected_ordinal_weight,
                    "selection_data": "validation_only; test labels excluded",
                    "trials": hybrid_trials,
                },
                "causal_filter": {
                    "selected_source_model": filter_key[0],
                    "selected_smoothing": filter_key[1],
                    "selection_data": "validation_only; test labels excluded",
                    "escalation_rule": "immediate; smoothing applies to de-escalation",
                    "trials": filter_trials,
                },
                "candidate_models": sorted(allowed_models),
                "score_tie_tolerance": tie_tolerance,
                "selection_data": "validation_only; test labels excluded",
                "candidates": validation_selection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "transformer_training_log.json").write_text(
        json.dumps(transformer_result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    class_rows = []
    for split_name, indices in {"train": train_idx, "validation": val_idx, "test": test_idx}.items():
        for level in range(4):
            count = int((labels[indices] == level).sum())
            class_rows.append(
                {
                    "split": split_name,
                    "risk_level": level,
                    "risk_name": RISK_NAMES[level],
                    "risk_name_zh": RISK_NAMES_ZH[level],
                    "sample_count": count,
                    "sample_fraction": count / len(indices),
                }
            )
    pd.DataFrame(class_rows).to_csv(
        artifact_dir / "class_distribution.csv", index=False, encoding="utf-8-sig"
    )

    ensemble_probability = probabilities_test[operational_model]
    predicted = ensemble_probability.argmax(axis=1)
    dispatch_guard_probability = probabilities_test[dispatch_guard_model]
    dispatch_guard_predicted = dispatch_guard_probability.argmax(axis=1)
    dispatch_guard_metrics = _classification_metrics(
        labels[test_idx], dispatch_guard_probability
    )
    (artifact_dir / "dispatch_guard_metrics.json").write_text(
        json.dumps(
            {
                "model": dispatch_guard_model,
                "selection_data": "validation_only; test labels excluded",
                **dispatch_guard_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    test_supply = supply.iloc[test_idx].reset_index(drop=True)
    evidence_cfg = risk_cfg["evidence"]
    evidence = build_risk_evidence(
        forecast_margin_ratio[test_idx],
        test_supply["grid_available_capacity_kw"].to_numpy(),
        test_supply["storage_soc_pct"].to_numpy(),
        uncertainty_max[test_idx],
        forecast_ramp[test_idx],
        grid_derating_threshold_kw=float(evidence_cfg["grid_derating_threshold_kw"]),
        low_soc_threshold_pct=float(evidence_cfg["low_soc_threshold_pct"]),
        uncertainty_threshold_kw=float(evidence_cfg["uncertainty_threshold_kw"]),
        ramp_threshold_kw_per_s=float(evidence_cfg["ramp_threshold_kw_per_s"]),
    )
    reserve_adders = risk_cfg["dispatch_signal"]["reserve_adder_kw"]
    output = pd.DataFrame(
        {
            "sample_index": test_idx,
            "timestamp": pd.to_datetime(target_timestamps.iloc[test_idx]).to_numpy(),
            "supply_regime": test_supply["supply_regime"].to_numpy(),
            "supply_source_type": test_supply["supply_source_type"].to_numpy(),
            "current_load_kw": current_load[test_idx],
            "forecast_peak_kw": forecast_peak[test_idx],
            "actual_peak_kw": actual_future[test_idx].max(axis=1),
            "firm_supply_kw": firm_supply[test_idx],
            "actual_margin_kw": actual_margin_kw[test_idx],
            "actual_margin_ratio": actual_margin_ratio[test_idx],
            "forecast_margin_ratio": forecast_margin_ratio[test_idx],
            "grid_available_capacity_kw": test_supply["grid_available_capacity_kw"],
            "generator_available_capacity_kw": test_supply[
                "generator_available_capacity_kw"
            ],
            "storage_soc_pct": test_supply["storage_soc_pct"],
            "storage_available_discharge_power_kw": test_supply[
                "storage_available_discharge_power_kw"
            ],
            "forecast_uncertainty_kw": uncertainty_max[test_idx],
            "forecast_ramp_kw_per_s": forecast_ramp[test_idx],
            "true_risk_level": labels[test_idx],
            "true_risk_name": [RISK_NAMES[value] for value in labels[test_idx]],
            "predicted_risk_level": predicted,
            "predicted_risk_name": [RISK_NAMES[value] for value in predicted],
            "dispatch_risk_level": dispatch_guard_predicted,
            "dispatch_risk_name": [
                RISK_NAMES[value] for value in dispatch_guard_predicted
            ],
            "risk_confidence": ensemble_probability.max(axis=1),
            "prob_normal": ensemble_probability[:, 0],
            "prob_watch": ensemble_probability[:, 1],
            "prob_warning": ensemble_probability[:, 2],
            "prob_severe": ensemble_probability[:, 3],
            "risk_evidence": evidence,
            "reserve_adder_kw": [
                float(reserve_adders[str(value)])
                for value in dispatch_guard_predicted
            ],
        }
    )
    output.to_csv(artifact_dir / "risk_predictions.csv", index=False, encoding="utf-8-sig")
    supply.to_csv(artifact_dir / "supply_context.csv", index=False, encoding="utf-8-sig")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels[test_idx], predicted, labels=[0, 1, 2, 3], zero_division=0
    )
    pd.DataFrame(
        {
            "risk_level": range(4),
            "risk_name": [RISK_NAMES[index] for index in range(4)],
            "risk_name_zh": [RISK_NAMES_ZH[index] for index in range(4)],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(artifact_dir / "ensemble_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    guard_precision, guard_recall, guard_f1, guard_support = (
        precision_recall_fscore_support(
            labels[test_idx],
            dispatch_guard_predicted,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
    )
    pd.DataFrame(
        {
            "risk_level": range(4),
            "risk_name": [RISK_NAMES[index] for index in range(4)],
            "risk_name_zh": [RISK_NAMES_ZH[index] for index in range(4)],
            "precision": guard_precision,
            "recall": guard_recall,
            "f1": guard_f1,
            "support": guard_support,
        }
    ).to_csv(
        artifact_dir / "dispatch_guard_per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _plot_metric_comparison(metrics, artifact_dir / "risk_metric_comparison.png")
    _plot_confusions(labels[test_idx], probabilities_test, artifact_dir / "confusion_matrices.png")
    _plot_timeline(output, artifact_dir / "risk_timeline.png")
    metadata = {
        "forecast_model": model_name,
        "uncertainty_members": member_names,
        "schema_report": schema_report.as_dict(),
        "sample_count": len(labels),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "supply_source_type": str(supply["supply_source_type"].iloc[0]),
        "synthetic_supply_generation_mode": str(
            risk_cfg.get("synthetic_supply", {}).get(
                "generation_mode", "forecast_margin_stratified"
            )
        ),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "xgboost": xgb.__version__,
        "transformer_best_epoch": transformer_result.best_epoch,
        "transformer_validation_macro_f1": transformer_result.validation_macro_f1,
        "operational_model": operational_model,
        "dispatch_guard_model": dispatch_guard_model,
        "operational_model_selection": selection_mode,
        "transition_feature": {
            "source": transition_feature_source,
            "causal": True,
            "future_transition_flags_excluded": True,
        },
        "conformal_peak_correction": {
            "coverage": conformal_coverage,
            "correction_kw": conformal_correction_kw,
            "calibration_data": "risk_train_only; validation/test labels excluded",
        },
        "risk_definition": {
            "required_power": "未来一分钟实际峰值 + max(备用比例×峰值, 备用下限)",
            "margin_ratio": "(可用供电能力-所需功率)/所需功率",
            "normal": f">={label_cfg['normal_margin_ratio']}",
            "watch": f"[{label_cfg['watch_margin_ratio']}, {label_cfg['normal_margin_ratio']})",
            "warning": f"[{label_cfg['warning_margin_ratio']}, {label_cfg['watch_margin_ratio']})",
            "severe": f"<{label_cfg['warning_margin_ratio']}",
        },
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
