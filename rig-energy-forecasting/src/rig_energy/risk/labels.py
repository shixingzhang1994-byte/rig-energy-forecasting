from __future__ import annotations

import numpy as np


RISK_NAMES = {
    0: "normal",
    1: "watch",
    2: "warning",
    3: "severe",
}
RISK_NAMES_ZH = {
    0: "正常",
    1: "关注",
    2: "预警",
    3: "严重",
}


def assign_risk_levels(
    actual_future_kw: np.ndarray,
    firm_supply_kw: np.ndarray,
    *,
    reserve_ratio: float = 0.05,
    reserve_floor_kw: float = 50.0,
    normal_margin_ratio: float = 0.15,
    watch_margin_ratio: float = 0.05,
    warning_margin_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create auditable four-level adequacy labels from realized future demand.

    The label is allowed to use future load because it is the supervised target.
    Model features must only use information available at the forecast origin.
    """

    future = np.asarray(actual_future_kw, dtype=float)
    supply = np.asarray(firm_supply_kw, dtype=float).reshape(-1)
    if future.ndim != 2 or len(future) != len(supply):
        raise ValueError("actual_future_kw 必须为二维数组且与 firm_supply_kw 行数一致")
    required_peak = future.max(axis=1)
    required_with_reserve = required_peak + np.maximum(
        reserve_floor_kw, reserve_ratio * required_peak
    )
    margin_kw = supply - required_with_reserve
    margin_ratio = margin_kw / np.maximum(required_with_reserve, 1.0)
    levels = np.full(len(future), 3, dtype=np.int64)
    levels[margin_ratio >= warning_margin_ratio] = 2
    levels[margin_ratio >= watch_margin_ratio] = 1
    levels[margin_ratio >= normal_margin_ratio] = 0
    return levels, margin_kw.astype(np.float32), margin_ratio.astype(np.float32)


def build_risk_evidence(
    forecast_margin_ratio: np.ndarray,
    grid_capacity_kw: np.ndarray,
    storage_soc_pct: np.ndarray,
    uncertainty_kw: np.ndarray,
    forecast_ramp_kw_per_s: np.ndarray,
    *,
    grid_derating_threshold_kw: float,
    low_soc_threshold_pct: float,
    uncertainty_threshold_kw: float,
    ramp_threshold_kw_per_s: float,
) -> list[str]:
    """Return concise, rule-based evidence alongside the learned classifier."""

    evidence: list[str] = []
    for margin, grid, soc, uncertainty, ramp in zip(
        forecast_margin_ratio,
        grid_capacity_kw,
        storage_soc_pct,
        uncertainty_kw,
        forecast_ramp_kw_per_s,
    ):
        items = []
        if margin < 0.0:
            items.append("预测供电缺口")
        elif margin < 0.05:
            items.append("预测裕度不足5%")
        elif margin < 0.15:
            items.append("预测裕度不足15%")
        if grid < grid_derating_threshold_kw:
            items.append("网电受限")
        if soc < low_soc_threshold_pct:
            items.append("储能SOC偏低")
        if uncertainty > uncertainty_threshold_kw:
            items.append("模型分歧较大")
        if ramp > ramp_threshold_kw_per_s:
            items.append("负荷快速上升")
        evidence.append("；".join(items) if items else "供电裕度充足")
    return evidence
