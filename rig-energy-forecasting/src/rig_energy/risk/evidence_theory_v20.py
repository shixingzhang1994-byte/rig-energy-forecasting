from __future__ import annotations

"""V20 conflict-rule and probability-calibration extensions.

The V19 implementation is frozen evidence and is therefore imported rather
than modified.  V20 keeps the source construction and dispatch thresholds
identical to V19, varies only the declared combination rule, and applies a
scalar temperature to the reported pignistic probabilities.  The calibrated
probabilities never enter the dispatch decision.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from rig_energy.risk.evidence_theory import (
    HIGH_RISK,
    SEVERE_RISK,
    UNIVERSE,
    _row_sources,
    _validate_mass,
    belief,
    pignistic_probability,
    plausibility,
    yager_combine,
)


YAGER_RULE = "yager_conflict_to_ignorance"
DEMPSTER_RULE = "dempster_normalized"
SUPPORTED_RULES = (YAGER_RULE, DEMPSTER_RULE)


def dempster_combine(
    masses: Iterable[Mapping[int, float]],
    *,
    total_conflict_tolerance: float = 1e-12,
) -> tuple[dict[int, float], float, bool]:
    """Combine masses with Dempster normalization and expose raw conflict.

    A deterministic full-ignorance fallback is used when normalization is
    undefined under (near-)total conflict.  The fallback flag prevents this
    edge case from being silently presented as ordinary Dempster fusion.
    """

    sources = [_validate_mass(mass) for mass in masses]
    if not sources:
        raise ValueError("at least one evidence source is required")
    combined: dict[int, float] = {UNIVERSE: 1.0}
    for source in sources:
        next_mass: dict[int, float] = {}
        for left, left_value in combined.items():
            for right, right_value in source.items():
                focal = int(left) & int(right)
                next_mass[focal] = next_mass.get(focal, 0.0) + left_value * right_value
        combined = next_mass
    conflict = float(combined.pop(0, 0.0))
    denominator = 1.0 - conflict
    if denominator <= float(total_conflict_tolerance):
        return {UNIVERSE: 1.0}, conflict, True
    normalized = {focal: value / denominator for focal, value in combined.items()}
    return _validate_mass(normalized), conflict, False


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar power/temperature calibration along the last axis."""

    values = np.asarray(probability, dtype=float)
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if values.ndim not in (1, 2) or values.shape[-1] != 4:
        raise ValueError("probability must end in four risk classes")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probability values must be finite and nonnegative")
    row_sum = values.sum(axis=-1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("probability rows must have positive mass")
    normalized = values / row_sum
    powered = np.power(np.clip(normalized, 1e-15, 1.0), 1.0 / value)
    return powered / powered.sum(axis=-1, keepdims=True)


def _combine(
    sources: list[dict[int, float]], rule: str
) -> tuple[dict[int, float], float, bool]:
    if rule == YAGER_RULE:
        mass, conflict = yager_combine(sources)
        return mass, conflict, False
    if rule == DEMPSTER_RULE:
        return dempster_combine(sources)
    raise ValueError(f"unsupported combination rule: {rule}")


def fuse_risk_frame_v20(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    """Fuse causal evidence and return raw, calibrated, and dispatch outputs."""

    required = {
        "forecast_peak_kw",
        "forecast_margin_ratio",
        "grid_available_capacity_kw",
        "storage_soc_pct",
        "forecast_uncertainty_kw",
        "forecast_ramp_kw_per_s",
        "prob_normal",
        "prob_watch",
        "prob_warning",
        "prob_severe",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evidential risk input is missing columns: {missing}")
    rule = str(config["combination_rule"])
    temperature = float(config["calibration"]["temperature"])
    if config["calibration"].get("dispatch_uses", "raw_pignistic_probability") != (
        "raw_pignistic_probability"
    ):
        raise ValueError("V20 dispatch must use raw pignistic probabilities")
    decision = config["decision"]
    labels = ("normal", "watch", "warning", "severe")
    records: list[dict[str, float | int | bool | str]] = []
    for _, row in frame.iterrows():
        mass, conflict, fallback = _combine(_row_sources(row, config), rule)
        raw_probability = pignistic_probability(mass)
        calibrated_probability = temperature_scale(raw_probability, temperature)
        predicted = int(np.argmax(raw_probability))
        calibrated_predicted = int(np.argmax(calibrated_probability))
        high_score = float(raw_probability[2:].sum()) + float(
            decision["conflict_weight"]
        ) * conflict
        severe_score = float(raw_probability[3]) + float(
            decision["conflict_weight"]
        ) * conflict
        dispatch_level = predicted
        if high_score >= float(decision["high_score_threshold"]):
            dispatch_level = max(dispatch_level, 2)
        if severe_score >= float(decision["severe_score_threshold"]):
            dispatch_level = 3
        record: dict[str, float | int | bool | str] = {
            "v20_combination_rule": rule,
            "v20_temperature": temperature,
            "v20_risk_level": predicted,
            "v20_calibrated_risk_level": calibrated_predicted,
            "v20_dispatch_risk_level": dispatch_level,
            "v20_high_belief": belief(mass, HIGH_RISK),
            "v20_high_plausibility": plausibility(mass, HIGH_RISK),
            "v20_severe_belief": belief(mass, SEVERE_RISK),
            "v20_severe_plausibility": plausibility(mass, SEVERE_RISK),
            "v20_ignorance": float(mass.get(UNIVERSE, 0.0)),
            "v20_conflict": conflict,
            "v20_total_conflict_fallback": fallback,
            "v20_calibration_changed_argmax": calibrated_predicted != predicted,
        }
        for index, label in enumerate(labels):
            record[f"v20_raw_prob_{label}"] = float(raw_probability[index])
            record[f"v20_calibrated_prob_{label}"] = float(
                calibrated_probability[index]
            )
        records.append(record)
    return pd.DataFrame.from_records(records, index=frame.index)
