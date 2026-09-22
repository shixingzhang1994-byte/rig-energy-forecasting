from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


RISK_LEVEL_COUNT = 4
UNIVERSE = (1 << RISK_LEVEL_COUNT) - 1
HIGH_RISK = (1 << 2) | (1 << 3)
SEVERE_RISK = 1 << 3


def _validate_mass(mass: Mapping[int, float]) -> dict[int, float]:
    cleaned: dict[int, float] = {}
    for focal, value in mass.items():
        focal_set = int(focal)
        numeric = float(value)
        if focal_set <= 0 or focal_set > UNIVERSE:
            raise ValueError(f"invalid nonempty focal set: {focal_set}")
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError("basic probability assignments must be finite and nonnegative")
        if numeric > 0.0:
            cleaned[focal_set] = cleaned.get(focal_set, 0.0) + numeric
    total = float(sum(cleaned.values()))
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"basic probability assignments must sum to one, got {total}")
    return cleaned


def discount_mass(mass: Mapping[int, float], reliability: float) -> dict[int, float]:
    """Discount one evidence source and transfer unsupported mass to ignorance."""

    source = _validate_mass(mass)
    value = float(reliability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("source reliability must be in [0, 1]")
    discounted = {
        focal: value * amount
        for focal, amount in source.items()
        if focal != UNIVERSE
    }
    discounted[UNIVERSE] = value * source.get(UNIVERSE, 0.0) + (1.0 - value)
    return _validate_mass(discounted)


def yager_combine(masses: Iterable[Mapping[int, float]]) -> tuple[dict[int, float], float]:
    """Combine sources conjunctively and assign total conflict to ignorance.

    The empty-set mass is retained until every source has been combined, then
    moved to the full frame. This avoids the high-conflict normalization used
    by Dempster's rule and makes conflict available as a separate diagnostic.
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
    combined[UNIVERSE] = combined.get(UNIVERSE, 0.0) + conflict
    return _validate_mass(combined), conflict


def belief(mass: Mapping[int, float], hypothesis: int) -> float:
    source = _validate_mass(mass)
    target = int(hypothesis)
    return float(sum(value for focal, value in source.items() if focal & ~target == 0))


def plausibility(mass: Mapping[int, float], hypothesis: int) -> float:
    source = _validate_mass(mass)
    target = int(hypothesis)
    return float(sum(value for focal, value in source.items() if focal & target))


def pignistic_probability(mass: Mapping[int, float]) -> np.ndarray:
    source = _validate_mass(mass)
    probability = np.zeros(RISK_LEVEL_COUNT, dtype=float)
    for focal, value in source.items():
        members = [level for level in range(RISK_LEVEL_COUNT) if focal & (1 << level)]
        share = float(value) / len(members)
        probability[members] += share
    probability /= probability.sum()
    return probability


def margin_level(margin_ratio: float, thresholds: Mapping[str, float]) -> int:
    margin = float(margin_ratio)
    if margin >= float(thresholds["normal_margin_ratio"]):
        return 0
    if margin >= float(thresholds["watch_margin_ratio"]):
        return 1
    if margin >= float(thresholds["warning_margin_ratio"]):
        return 2
    return 3


def margin_interval_focal(
    margin_ratio: float,
    uncertainty_ratio: float,
    thresholds: Mapping[str, float],
) -> int:
    """Map a forecast-margin interval to the ordinal classes it can support."""

    radius = max(float(uncertainty_ratio), 0.0)
    lower = margin_level(float(margin_ratio) - radius, thresholds)
    upper = margin_level(float(margin_ratio) + radius, thresholds)
    low_level, high_level = sorted((lower, upper))
    focal = 0
    for level in range(low_level, high_level + 1):
        focal |= 1 << level
    return focal


def _categorical_mass(probability: np.ndarray) -> dict[int, float]:
    values = np.asarray(probability, dtype=float)
    if values.shape != (RISK_LEVEL_COUNT,) or np.any(values < 0.0):
        raise ValueError("risk probabilities must contain four nonnegative values")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("risk probabilities must have a positive finite sum")
    values = values / total
    return {1 << level: float(values[level]) for level in range(RISK_LEVEL_COUNT)}


def _simple_support(focal: int) -> dict[int, float]:
    return {int(focal): 1.0}


def _row_sources(row: pd.Series, config: Mapping[str, object]) -> list[dict[int, float]]:
    discounts = config["discounts"]
    thresholds = config["margin_thresholds"]
    probability = row[["prob_normal", "prob_watch", "prob_warning", "prob_severe"]].to_numpy(float)
    sources = [
        discount_mass(
            _categorical_mass(probability),
            float(discounts["learned_probability"]),
        )
    ]

    required_power = max(float(row["forecast_peak_kw"]), 1.0)
    uncertainty_ratio = (
        float(config["uncertainty_interval_scale"])
        * max(float(row["forecast_uncertainty_kw"]), 0.0)
        / required_power
    )
    focal = margin_interval_focal(
        float(row["forecast_margin_ratio"]), uncertainty_ratio, thresholds
    )
    sources.append(discount_mass(_simple_support(focal), float(discounts["margin_interval"])))

    soc = float(row["storage_soc_pct"])
    if soc <= float(config["soc_warning_pct"]):
        sources.append(
            discount_mass(_simple_support(HIGH_RISK), float(discounts["low_soc"]))
        )
    elif soc >= float(config["soc_healthy_pct"]):
        sources.append(
            discount_mass(
                _simple_support((1 << 0) | (1 << 1)),
                float(discounts["healthy_soc"]),
            )
        )

    grid = float(row["grid_available_capacity_kw"])
    if grid < float(config["grid_derating_threshold_kw"]):
        sources.append(
            discount_mass(
                _simple_support((1 << 1) | HIGH_RISK),
                float(discounts["grid_derating"]),
            )
        )

    ramp = float(row["forecast_ramp_kw_per_s"])
    if ramp > float(config["ramp_threshold_kw_per_s"]):
        sources.append(
            discount_mass(_simple_support(HIGH_RISK), float(discounts["rapid_ramp"]))
        )
    return sources


def fuse_risk_frame(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Create causal evidential risk outputs without reading future truth columns."""

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
    decision = config["decision"]
    records: list[dict[str, float | int]] = []
    for _, row in frame.iterrows():
        mass, conflict = yager_combine(_row_sources(row, config))
        probability = pignistic_probability(mass)
        predicted = int(np.argmax(probability))
        high_belief = belief(mass, HIGH_RISK)
        high_plausibility = plausibility(mass, HIGH_RISK)
        severe_belief = belief(mass, SEVERE_RISK)
        severe_plausibility = plausibility(mass, SEVERE_RISK)
        ignorance = float(mass.get(UNIVERSE, 0.0))
        high_score = float(probability[2:].sum()) + float(
            decision["conflict_weight"]
        ) * conflict
        severe_score = float(probability[3]) + float(
            decision["conflict_weight"]
        ) * conflict
        dispatch_level = predicted
        if high_score >= float(decision["high_score_threshold"]):
            dispatch_level = max(dispatch_level, 2)
        if severe_score >= float(decision["severe_score_threshold"]):
            dispatch_level = 3
        records.append(
            {
                "evidential_risk_level": predicted,
                "evidential_dispatch_risk_level": dispatch_level,
                "evidential_prob_normal": float(probability[0]),
                "evidential_prob_watch": float(probability[1]),
                "evidential_prob_warning": float(probability[2]),
                "evidential_prob_severe": float(probability[3]),
                "evidential_high_belief": high_belief,
                "evidential_high_plausibility": high_plausibility,
                "evidential_severe_belief": severe_belief,
                "evidential_severe_plausibility": severe_plausibility,
                "evidential_ignorance": ignorance,
                "evidential_conflict": conflict,
            }
        )
    return pd.DataFrame.from_records(records, index=frame.index)
