from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..data.schema import validate_canonical_frame


TOTAL_POWER_COLUMN = "total_active_power_kw"


def _safe_cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values, ddof=0) / abs(mean)) if abs(mean) > 1e-9 else float("nan")


def _count_events(mask: np.ndarray) -> int:
    flags = np.asarray(mask, dtype=bool)
    if not flags.size:
        return 0
    return int(np.sum(flags & ~np.r_[False, flags[:-1]]))


def _contiguous_durations_seconds(mask: np.ndarray, step_seconds: float) -> np.ndarray:
    flags = np.asarray(mask, dtype=bool)
    if not flags.size or not np.any(flags):
        return np.empty(0, dtype=float)
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    stops = np.flatnonzero(flags & ~np.r_[flags[1:], False]) + 1
    return (stops - starts).astype(float) * step_seconds


def _durations_with_breaks(
    mask: np.ndarray,
    break_before: np.ndarray,
    step_seconds: float,
) -> np.ndarray:
    durations = []
    current = 0.0
    for index, flag in enumerate(np.asarray(mask, dtype=bool)):
        if break_before[index] and current > 0.0:
            durations.append(current)
            current = 0.0
        if flag:
            current += step_seconds
        elif current > 0.0:
            durations.append(current)
            current = 0.0
    if current > 0.0:
        durations.append(current)
    return np.asarray(durations, dtype=float)


def _peak_event_durations(
    power: np.ndarray,
    break_before: np.ndarray,
    step_seconds: float,
    start_threshold_kw: float,
    release_threshold_kw: float,
    minimum_duration_seconds: float,
) -> np.ndarray:
    durations = []
    active = False
    duration = 0.0
    for index, value in enumerate(power):
        if break_before[index] and active:
            if duration >= minimum_duration_seconds:
                durations.append(duration)
            active = False
            duration = 0.0
        if not active and value >= start_threshold_kw:
            active = True
        if active:
            duration += step_seconds
            if value <= release_threshold_kw:
                if duration >= minimum_duration_seconds:
                    durations.append(duration)
                active = False
                duration = 0.0
    if active and duration >= minimum_duration_seconds:
        durations.append(duration)
    return np.asarray(durations, dtype=float)


def _segment_table(frame: pd.DataFrame, step_seconds: float) -> pd.DataFrame:
    state = frame["operation_state"].astype(str)
    group_id = state.ne(state.shift()).cumsum()
    segments = (
        frame.assign(_segment_id=group_id)
        .groupby("_segment_id", sort=False)
        .agg(
            operation_state=("operation_state", "first"),
            start_time=("timestamp", "first"),
            end_time=("timestamp", "last"),
            sample_count=("timestamp", "size"),
            mean_power_kw=(TOTAL_POWER_COLUMN, "mean"),
            max_power_kw=(TOTAL_POWER_COLUMN, "max"),
        )
        .reset_index(drop=True)
    )
    segments["duration_seconds"] = segments["sample_count"] * step_seconds
    return segments


def _metric_rows(
    values: np.ndarray,
    *,
    timestamps: np.ndarray,
    state: str,
    equipment: str,
    step_seconds: float,
    peak_threshold_kw: float,
    peak_release_ratio: float,
    minimum_peak_duration_seconds: float,
    impact_ramp_threshold_kw_per_s: float,
) -> list[dict]:
    power = np.asarray(values, dtype=float)
    time = pd.to_datetime(np.asarray(timestamps))
    finite = np.isfinite(power) & ~pd.isna(time)
    power = power[finite]
    time = time[finite]
    if power.size == 0:
        return []
    break_before = np.zeros(power.size, dtype=bool)
    if power.size > 1:
        time_delta = np.diff(time).astype("timedelta64[ns]").astype(np.int64) / 1e9
        break_before[1:] = time_delta > 1.5 * step_seconds
        full_ramp = np.diff(power) / step_seconds
        valid_ramp = ~break_before[1:]
        ramp = full_ramp[valid_ramp]
        full_impact_mask = (np.abs(full_ramp) >= impact_ramp_threshold_kw_per_s) & valid_ramp
    else:
        ramp = np.empty(0)
        full_ramp = np.empty(0)
        full_impact_mask = np.empty(0, dtype=bool)
    peak_durations = _peak_event_durations(
        power,
        break_before,
        step_seconds,
        peak_threshold_kw,
        peak_threshold_kw * peak_release_ratio,
        minimum_peak_duration_seconds,
    )
    energy_kwh = float(np.sum(power) * step_seconds / 3600.0)
    maximum = float(np.max(power))
    metrics = [
        ("sample_count", float(power.size), "count"),
        ("mean_power", float(np.mean(power)), "kW"),
        ("minimum_power", float(np.min(power)), "kW"),
        ("maximum_power", maximum, "kW"),
        ("p95_power", float(np.quantile(power, 0.95)), "kW"),
        ("peak_to_valley", float(np.ptp(power)), "kW"),
        ("standard_deviation", float(np.std(power, ddof=0)), "kW"),
        ("coefficient_of_variation", _safe_cv(power), "ratio"),
        ("energy", energy_kwh, "kWh"),
        ("load_factor", float(np.mean(power) / maximum) if maximum > 1e-9 else float("nan"), "ratio"),
        ("maximum_ramp_up", float(np.max(ramp)) if ramp.size else 0.0, "kW/s"),
        ("maximum_ramp_down", float(np.min(ramp)) if ramp.size else 0.0, "kW/s"),
        ("maximum_absolute_ramp", float(np.max(np.abs(ramp))) if ramp.size else 0.0, "kW/s"),
        ("peak_event_count", float(len(peak_durations)), "count"),
        ("peak_total_duration", float(peak_durations.sum()), "s"),
        ("peak_maximum_duration", float(peak_durations.max()) if peak_durations.size else 0.0, "s"),
        ("impact_event_count", float(_count_events(full_impact_mask)), "count"),
        (
            "mean_impact_amplitude",
            float(np.mean(np.abs(full_ramp[full_impact_mask]) * step_seconds))
            if np.any(full_impact_mask)
            else 0.0,
            "kW",
        ),
    ]
    return [
        {
            "operation_state": state,
            "equipment": equipment,
            "metric": metric,
            "value": value,
            "unit": unit,
        }
        for metric, value, unit in metrics
    ]


def _transition_metrics(
    frame: pd.DataFrame,
    step_seconds: float,
    window_seconds: float,
) -> pd.DataFrame:
    state = frame["operation_state"].astype(str).to_numpy()
    power = frame[TOTAL_POWER_COLUMN].to_numpy(dtype=float)
    transition_indices = np.flatnonzero(state[1:] != state[:-1]) + 1
    half_window = max(1, int(round(window_seconds / step_seconds)))
    rows = []
    for index in transition_indices:
        before = power[max(0, index - half_window) : index]
        after = power[index : min(len(power), index + half_window)]
        if not len(before) or not len(after):
            continue
        rows.append(
            {
                "timestamp": frame["timestamp"].iloc[index],
                "from_state": state[index - 1],
                "to_state": state[index],
                "before_mean_power_kw": float(np.mean(before)),
                "after_mean_power_kw": float(np.mean(after)),
                "power_change_kw": float(np.mean(after) - np.mean(before)),
                "absolute_power_change_kw": float(abs(np.mean(after) - np.mean(before))),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "from_state",
                "to_state",
                "transition_count",
                "mean_power_change_kw",
                "mean_absolute_power_change_kw",
                "maximum_absolute_power_change_kw",
            ]
        )
    return (
        events.groupby(["from_state", "to_state"], as_index=False)
        .agg(
            transition_count=("power_change_kw", "size"),
            mean_power_change_kw=("power_change_kw", "mean"),
            mean_absolute_power_change_kw=("absolute_power_change_kw", "mean"),
            maximum_absolute_power_change_kw=("absolute_power_change_kw", "max"),
        )
        .sort_values("mean_absolute_power_change_kw", ascending=False)
        .reset_index(drop=True)
    )


def _plot_feature_summary(
    frame: pd.DataFrame,
    state_metrics: pd.DataFrame,
    peak_threshold_kw: float,
    path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    states = state_metrics["operation_state"]
    x = np.arange(len(states))
    axes[0, 0].bar(x - 0.25, state_metrics["mean_power_kw"], width=0.25, label="均值")
    axes[0, 0].bar(x, state_metrics["p95_power_kw"], width=0.25, label="P95")
    axes[0, 0].bar(x + 0.25, state_metrics["maximum_power_kw"], width=0.25, label="最大值")
    axes[0, 0].set_xticks(x, states, rotation=25, ha="right")
    axes[0, 0].set_ylabel("功率/kW")
    axes[0, 0].set_title("分工况负荷水平")
    axes[0, 0].legend(frameon=False)

    duration = np.sort(frame[TOTAL_POWER_COLUMN].to_numpy(dtype=float))[::-1]
    exceedance = np.arange(1, len(duration) + 1) / len(duration) * 100.0
    axes[0, 1].plot(exceedance, duration, color="#2563EB")
    axes[0, 1].axhline(peak_threshold_kw, color="#DC2626", linestyle="--", label="P95峰值阈值")
    axes[0, 1].set_xlabel("超越时间比例/%")
    axes[0, 1].set_ylabel("功率/kW")
    axes[0, 1].set_title("负荷持续曲线")
    axes[0, 1].legend(frameon=False)

    step_seconds = float(frame["timestamp"].diff().dt.total_seconds().dropna().median())
    ramp = frame[TOTAL_POWER_COLUMN].diff() / step_seconds
    clipped = ramp.clip(ramp.quantile(0.005), ramp.quantile(0.995)).dropna()
    axes[1, 0].hist(clipped, bins=80, color="#10B981", alpha=0.85)
    axes[1, 0].set_xlabel("功率变化率/(kW/s)")
    axes[1, 0].set_ylabel("频数")
    axes[1, 0].set_title("功率变化率分布（0.5%–99.5%）")

    count = min(len(frame), int(round(12 * 3600 / step_seconds)))
    time_hours = np.arange(count) * step_seconds / 3600.0
    axes[1, 1].plot(
        time_hours,
        frame[TOTAL_POWER_COLUMN].iloc[:count],
        color="#111827",
        linewidth=0.8,
    )
    axes[1, 1].axhline(peak_threshold_kw, color="#DC2626", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("时间/h")
    axes[1, 1].set_ylabel("功率/kW")
    axes[1, 1].set_title("前12小时典型负荷")

    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_typical_states(frame: pd.DataFrame, path: Path, minutes: float) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    states = list(pd.unique(frame["operation_state"]))
    columns = 2
    rows = int(np.ceil(len(states) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.1 * rows), squeeze=False)
    step_seconds = float(frame["timestamp"].diff().dt.total_seconds().dropna().median())
    desired = max(2, int(round(minutes * 60.0 / step_seconds)))
    for ax, state in zip(axes.flat, states):
        indices = np.flatnonzero(frame["operation_state"].astype(str).to_numpy() == str(state))
        start = int(indices[0])
        stop = start + 1
        while (
            stop < len(frame)
            and stop < start + desired
            and str(frame["operation_state"].iloc[stop]) == str(state)
        ):
            stop += 1
        values = frame[TOTAL_POWER_COLUMN].iloc[start:stop].to_numpy(dtype=float)
        time_minutes = np.arange(len(values)) * step_seconds / 60.0
        ax.plot(time_minutes, values, color="#2563EB", linewidth=1.0)
        ax.set_title(str(state))
        ax.set_xlabel("片段时间/min")
        ax.set_ylabel("功率/kW")
        ax.grid(alpha=0.18)
    for ax in axes.flat[len(states) :]:
        ax.axis("off")
    fig.suptitle("分工况典型负荷片段")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_load_features(
    data_path: Path,
    config_path: Path,
    artifact_dir: Path,
) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg = config["feature_analysis"]
    frame = pd.read_parquet(data_path)
    frame, schema_report = validate_canonical_frame(frame)
    interval = frame["timestamp"].diff().dt.total_seconds().dropna()
    step_seconds = float(interval.median())
    if step_seconds <= 0.0:
        raise ValueError("采样间隔必须大于0")

    total_power = frame[TOTAL_POWER_COLUMN].to_numpy(dtype=float)
    peak_threshold_kw = float(
        np.quantile(total_power, float(cfg.get("peak_quantile", 0.95)))
    )
    absolute_ramp = np.abs(np.diff(total_power) / step_seconds)
    impact_ramp_threshold = max(
        float(cfg.get("minimum_impact_ramp_kw_per_s", 30.0)),
        float(np.quantile(absolute_ramp, float(cfg.get("impact_ramp_quantile", 0.99)))),
    )

    equipment_columns = [TOTAL_POWER_COLUMN]
    if bool(cfg.get("include_component_power", True)):
        equipment_columns.extend(
            column
            for column in frame.columns
            if column.endswith("_power_kw") and column != TOTAL_POWER_COLUMN
        )
    equipment_columns = list(dict.fromkeys(equipment_columns))

    rows = []
    state_values = ["all"] + [str(value) for value in pd.unique(frame["operation_state"])]
    for state in state_values:
        state_frame = frame if state == "all" else frame.loc[frame["operation_state"] == state]
        for column in equipment_columns:
            rows.extend(
                _metric_rows(
                    state_frame[column].to_numpy(dtype=float),
                    timestamps=state_frame["timestamp"].to_numpy(),
                    state=state,
                    equipment=column.removesuffix("_power_kw"),
                    step_seconds=step_seconds,
                    peak_threshold_kw=peak_threshold_kw,
                    peak_release_ratio=float(cfg.get("peak_release_ratio", 0.97)),
                    minimum_peak_duration_seconds=float(
                        cfg.get("minimum_peak_duration_seconds", 10.0)
                    ),
                    impact_ramp_threshold_kw_per_s=impact_ramp_threshold,
                )
            )
    acceptance_table = pd.DataFrame(rows)
    acceptance_table.to_csv(
        artifact_dir / "acceptance_feature_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segments = _segment_table(frame, step_seconds)
    segments.to_csv(artifact_dir / "state_segments.csv", index=False, encoding="utf-8-sig")
    state_total = acceptance_table.loc[acceptance_table["equipment"] == "total_active"]
    pivot = state_total.pivot(index="operation_state", columns="metric", values="value")
    pivot = pivot.drop(index="all", errors="ignore").reset_index()
    state_duration = (
        segments.groupby("operation_state", as_index=False)
        .agg(
            segment_count=("duration_seconds", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
            median_segment_duration_seconds=("duration_seconds", "median"),
            maximum_segment_duration_seconds=("duration_seconds", "max"),
        )
    )
    state_metrics = pivot.merge(state_duration, on="operation_state", how="left")
    state_metrics = state_metrics.rename(
        columns={
            "mean_power": "mean_power_kw",
            "maximum_power": "maximum_power_kw",
            "minimum_power": "minimum_power_kw",
            "p95_power": "p95_power_kw",
            "energy": "energy_kwh",
            "maximum_ramp_up": "maximum_ramp_up_kw_per_s",
            "maximum_ramp_down": "maximum_ramp_down_kw_per_s",
        }
    )
    state_metrics.to_csv(artifact_dir / "state_metrics.csv", index=False, encoding="utf-8-sig")

    transitions = _transition_metrics(
        frame,
        step_seconds,
        float(cfg.get("transition_window_seconds", 30.0)),
    )
    transitions.to_csv(
        artifact_dir / "transition_metrics.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "schema": schema_report.as_dict(),
        "sampling_interval_seconds": step_seconds,
        "duration_hours": len(frame) * step_seconds / 3600.0,
        "equipment_columns": equipment_columns,
        "operation_states": [state for state in state_values if state != "all"],
        "peak_quantile": float(cfg.get("peak_quantile", 0.95)),
        "peak_threshold_kw": peak_threshold_kw,
        "impact_ramp_threshold_kw_per_s": impact_ramp_threshold,
        "quality_flag_counts": (
            frame["quality_flag"].value_counts(dropna=False).astype(int).to_dict()
        ),
        "source_type_counts": (
            frame["source_type"].value_counts(dropna=False).astype(int).to_dict()
        ),
        "output_files": [
            "acceptance_feature_table.csv",
            "state_metrics.csv",
            "state_segments.csv",
            "transition_metrics.csv",
            "feature_summary.png",
            "typical_state_curves.png",
        ],
    }
    (artifact_dir / "feature_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_feature_summary(
        frame,
        state_metrics,
        peak_threshold_kw,
        artifact_dir / "feature_summary.png",
    )
    _plot_typical_states(
        frame,
        artifact_dir / "typical_state_curves.png",
        float(cfg.get("typical_window_minutes", 30.0)),
    )
    return summary
