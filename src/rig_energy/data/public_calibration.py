from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


COMPONENT_COLUMNS = (
    "auxiliary_power_kw",
    "mud_pump_power_kw",
    "topdrive_power_kw",
    "drawworks_power_kw",
    "other_power_kw",
)


@dataclass(frozen=True)
class CalibrationAudit:
    passed: bool
    issues: tuple[str, ...]


def load_public_calibration(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("公开证据标定文件必须是YAML对象")
    return data


def validate_public_calibration(config: dict) -> CalibrationAudit:
    issues: list[str] = []
    sources = config.get("sources", [])
    source_ids = {str(item.get("id")) for item in sources}
    if len(sources) < 12:
        issues.append("证据源少于12条，不足以支撑跨类型标定")
    if len(source_ids) != len(sources):
        issues.append("证据源ID重复")
    for source in sources:
        if not source.get("url"):
            issues.append(f"证据源 {source.get('id')} 缺少URL")
        if source.get("source_type") not in {
            "peer_reviewed_paper",
            "patent",
            "manufacturer",
            "industry_news",
            "textbook_or_handbook",
            "standard",
            "government_price",
        }:
            issues.append(f"证据源 {source.get('id')} 类型不受支持")

    targets = config.get("load_calibration", {}).get("state_targets", {})
    required_states = {
        "idle",
        "circulation",
        "drilling",
        "connection",
        "tripping",
        "maintenance",
    }
    missing_states = sorted(required_states - set(targets))
    if missing_states:
        issues.append("缺少工况标定: " + ", ".join(missing_states))
    for state, target in targets.items():
        mean_kw = float(target["target_mean_kw"])
        low, high = map(float, target["acceptance_mean_kw"])
        cap = float(target["hard_cap_kw"])
        if not low <= mean_kw <= high:
            issues.append(f"{state}目标均值不在验收区间内")
        if cap < high:
            issues.append(f"{state}硬上限低于均值验收上限")
        unknown = sorted(set(target.get("source_ids", [])) - source_ids)
        if unknown:
            issues.append(f"{state}引用未定义证据: {', '.join(unknown)}")

    measurement = config.get("measurement_calibration", {})
    if float(measurement.get("relative_error_clip", 0.0)) <= 0.0:
        issues.append("测量相对误差上限必须为正")
    return CalibrationAudit(not issues, tuple(issues))


def _colored_relative_noise(
    rng: np.random.Generator, n: int, sigma: float, phi: float
) -> np.ndarray:
    innovations = rng.normal(0.0, sigma, n)
    values = np.zeros(n, dtype=np.float64)
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovations[index]
    observed_sigma = float(np.std(values))
    if observed_sigma > 0.0:
        values *= sigma / observed_sigma
    return values


def _scale_state_components(
    frame: pd.DataFrame, mask: np.ndarray, target_mean_kw: float, hard_cap_kw: float
) -> float:
    current = frame.loc[mask, list(COMPONENT_COLUMNS)].sum(axis=1).to_numpy(float)
    if len(current) == 0 or float(current.mean()) <= 0.0:
        raise ValueError("工况样本为空或功率非正，无法标定")

    # Bisection accounts for the state cap, so the mean remains reproducible
    # even when a few impact-load rows saturate.
    lower, upper = 0.0, max(1.0, target_mean_kw / float(current.mean()) * 3.0)
    for _ in range(60):
        middle = (lower + upper) / 2.0
        achieved = float(np.minimum(current * middle, hard_cap_kw).mean())
        if achieved < target_mean_kw:
            lower = middle
        else:
            upper = middle
    scale = (lower + upper) / 2.0

    selected = frame.loc[mask, list(COMPONENT_COLUMNS)].to_numpy(float) * scale
    totals = selected.sum(axis=1)
    row_scale = np.minimum(1.0, hard_cap_kw / np.maximum(totals, 1e-9))
    selected *= row_scale[:, None]
    frame.loc[mask, list(COMPONENT_COLUMNS)] = selected
    return float(scale)


def calibrate_synthetic_frame(
    base_frame: pd.DataFrame, calibration: dict, *, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Create a public-evidence-calibrated synthetic frame.

    Calibration uses only predeclared public targets and the synthetic source
    frame. It never reads model predictions, dispatch outcomes, or holdout
    scores. This prevents outcome-driven tuning from entering data generation.
    """

    audit = validate_public_calibration(calibration)
    if not audit.passed:
        raise ValueError("; ".join(audit.issues))
    frame = base_frame.copy()
    rng = np.random.default_rng(int(seed) + 14_000_003)
    for column in COMPONENT_COLUMNS:
        frame[column] = frame[column].astype("float64")

    # In the legacy generator, switching events were added only to the meter
    # channel. They represent real contactor/VFD load changes, so V14 assigns
    # them to the physical "other" component. Calibration is fitted to the
    # underlying operating state first; the sparse event is then restored so
    # that it remains a physical impact rather than a fake sensor artefact.
    transient = frame["transient_event_kw"].to_numpy(float)

    scales: dict[str, float] = {}
    targets = calibration["load_calibration"]["state_targets"]
    state_values = frame["operation_state"].astype(str).to_numpy()
    for state, target in targets.items():
        mask = state_values == str(state)
        scales[state] = _scale_state_components(
            frame,
            mask,
            float(target["target_mean_kw"]),
            float(target["hard_cap_kw"]),
        )

    other = frame["other_power_kw"].to_numpy(float)
    physical_transient = np.maximum(other + transient, 0.0) - other
    frame["other_power_kw"] = other + physical_transient
    frame["transient_event_kw"] = physical_transient

    # State caps apply to the complete physical load, including switching
    # events. Scale the full component row so the component balance remains
    # exact; record the correspondingly clipped event contribution.
    for state, target in targets.items():
        mask = state_values == str(state)
        components = frame.loc[mask, list(COMPONENT_COLUMNS)].to_numpy(float)
        totals = components.sum(axis=1)
        cap = float(target["hard_cap_kw"])
        row_scale = np.minimum(1.0, cap / np.maximum(totals, 1e-9))
        frame.loc[mask, list(COMPONENT_COLUMNS)] = components * row_scale[:, None]
        frame.loc[mask, "transient_event_kw"] = (
            frame.loc[mask, "transient_event_kw"].to_numpy(float) * row_scale
        )

    physical = frame.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1).to_numpy(float)
    total_low, total_high = map(
        float, calibration["load_calibration"]["total_power_hard_bounds_kw"]
    )
    physical = np.clip(physical, total_low, total_high)
    component_sum = frame.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1).to_numpy(float)
    correction = physical / np.maximum(component_sum, 1e-9)
    frame.loc[:, list(COMPONENT_COLUMNS)] = (
        frame.loc[:, list(COMPONENT_COLUMNS)].to_numpy(float) * correction[:, None]
    )
    frame["physical_total_power_kw"] = physical

    measurement = calibration["measurement_calibration"]
    relative_white = rng.normal(
        0.0, float(measurement["relative_white_noise_sigma"]), len(frame)
    )
    relative_drift = _colored_relative_noise(
        rng,
        len(frame),
        float(measurement["relative_drift_sigma"]),
        float(measurement["relative_drift_phi"]),
    )
    relative_error = np.clip(
        relative_white + relative_drift,
        -float(measurement["relative_error_clip"]),
        float(measurement["relative_error_clip"]),
    )
    observed = physical * (1.0 + relative_error)
    quantization = float(measurement["quantization_kw"])
    if quantization > 0.0:
        observed = np.round(observed / quantization) * quantization
    observed = np.clip(observed, total_low, total_high)

    missing = frame["quality_flag"].astype(str).eq("imputed").to_numpy()
    raw = observed.copy()
    raw[missing] = np.nan
    imputed = pd.Series(raw).interpolate(limit_direction="both").to_numpy(float)
    frame["total_active_power_kw_raw"] = raw.astype("float32")
    frame["total_active_power_kw"] = imputed.astype("float32")
    frame["source_type"] = "public_evidence_calibrated_synthetic_v14"
    frame["calibration_version"] = "V14"

    state_profile = (
        frame.groupby("operation_state")["physical_total_power_kw"]
        .agg(["count", "mean", "std", "min", "max"])
        .to_dict(orient="index")
    )
    report = {
        "calibration_version": "V14",
        "seed": int(seed),
        "state_component_scale": scales,
        "overall_mean_kw": float(frame["physical_total_power_kw"].mean()),
        "overall_max_kw": float(frame["physical_total_power_kw"].max()),
        "measurement_mae_kw": float(
            np.mean(np.abs(observed - frame["physical_total_power_kw"].to_numpy(float)))
        ),
        "measurement_max_relative_error": float(np.max(np.abs(relative_error))),
        "imputed_rows": int(missing.sum()),
        "state_profile": state_profile,
        "outcome_data_used_for_calibration": False,
        "field_scada_claimed": False,
    }
    return frame, report
