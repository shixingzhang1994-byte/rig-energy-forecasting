"""Leakage-safe empirical residual scenarios for the V22 dispatch study.

The calibration library is deliberately restricted to a chronological
validation split.  Each sampled item is a complete multi-step residual vector,
so the temporal dependence across the dispatch horizon is not destroyed by
independent per-lead-time sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def audit_cvar_discretization(
    probabilities: np.ndarray,
    *,
    alpha: float,
    minimum_effective_tail_scenarios: float,
) -> dict[str, Any]:
    """Audit whether a discrete scenario set can resolve the requested tail.

    ``effective_tail_scenarios`` is a conservative probability-resolution
    measure: tail probability divided by the largest atomic scenario weight.
    For equally weighted scenarios it reduces to ``(1-alpha) * S``.  A value
    no larger than one means empirical CVaR is the single worst scenario (with
    at most a fractional contribution of that same atom), making high alpha
    settings observationally indistinguishable.
    """

    weights = np.asarray(probabilities, dtype=float)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("probabilities must contain at least two scenarios")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("probabilities must be finite and strictly positive")
    total = float(weights.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"scenario probabilities must sum to one, got {total:.16g}")
    confidence = float(alpha)
    if not 0.0 < confidence < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    minimum = float(minimum_effective_tail_scenarios)
    if minimum <= 1.0:
        raise ValueError("minimum_effective_tail_scenarios must exceed one")

    tail_probability = 1.0 - confidence
    largest_atom = float(weights.max())
    effective_tail = tail_probability / largest_atom
    collapse = effective_tail <= 1.0 + 1e-12
    if effective_tail + 1e-12 < minimum:
        raise ValueError(
            "CVaR effective tail has only "
            f"{effective_tail:.6g} scenarios at alpha={confidence:.6g}; "
            f"at least {minimum:.6g} are required"
        )
    return {
        "scenario_count": int(len(weights)),
        "alpha": confidence,
        "tail_probability": tail_probability,
        "largest_scenario_probability": largest_atom,
        "effective_tail_scenarios": effective_tail,
        "collapses_to_single_worst_scenario": collapse,
        "minimum_effective_tail_scenarios": minimum,
    }


def build_validation_residual_library(
    actual_kw: np.ndarray,
    point_forecast_kw: np.ndarray,
    *,
    split_name: str,
    warmup_samples: int,
    source_artifact: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a residual library while enforcing validation-only provenance."""

    if str(split_name).strip().lower() != "validation":
        raise ValueError(
            "residual scenarios may be calibrated from the validation split only"
        )
    actual = _finite_matrix(actual_kw, "actual_kw")
    prediction = _finite_matrix(point_forecast_kw, "point_forecast_kw")
    if actual.shape != prediction.shape:
        raise ValueError("actual_kw and point_forecast_kw must have identical shapes")
    warmup = int(warmup_samples)
    if warmup < 0 or warmup >= len(actual):
        raise ValueError("warmup_samples must retain at least one validation sample")
    residuals = (actual[warmup:] - prediction[warmup:]).astype(np.float32)
    metadata: dict[str, Any] = {
        "split_name": "validation",
        "test_targets_used": False,
        "residual_definition": "actual_kw_minus_point_forecast_kw",
        "sampling_unit": "complete_horizon_residual_vector",
        "validation_samples_before_warmup": int(len(actual)),
        "excluded_warmup_samples": warmup,
        "residual_sample_count": int(len(residuals)),
        "horizon_steps": int(residuals.shape[1]),
    }
    if source_artifact is not None:
        metadata["source_artifact"] = str(source_artifact)
    return residuals, metadata


def build_residual_scenarios(
    point_forecast_kw: np.ndarray,
    residual_library_kw: np.ndarray,
    *,
    scenario_count: int,
    seed: int,
    minimum_load_kw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add stratified validation residual blocks to each point forecast.

    Residual vectors are ordered by their maximum positive forecast error and
    sampled once from each empirical quantile stratum.  This retains complete
    horizon vectors while preventing a small random sample from accidentally
    omitting the upper residual tail.
    """

    point = _finite_matrix(point_forecast_kw, "point_forecast_kw")
    residuals = _finite_matrix(residual_library_kw, "residual_library_kw")
    if point.shape[1] != residuals.shape[1]:
        raise ValueError(
            "point forecast and residual library horizon dimensions must match"
        )
    count = int(scenario_count)
    if count < 2:
        raise ValueError("scenario_count must be at least two")
    floor = float(minimum_load_kw)
    if not np.isfinite(floor):
        raise ValueError("minimum_load_kw must be finite")

    severity_order = np.argsort(np.max(residuals, axis=1), kind="stable")
    rng = np.random.default_rng(int(seed))
    sampled_indices = np.empty((len(point), count), dtype=np.int64)
    quantile_width = 1.0 / count
    for origin in range(len(point)):
        positions = (
            np.arange(count, dtype=float) * quantile_width
            + rng.random(count) * quantile_width
        )
        ranks = np.minimum(
            (positions * len(residuals)).astype(np.int64),
            len(residuals) - 1,
        )
        chosen = severity_order[ranks]
        sampled_indices[origin] = chosen[rng.permutation(count)]

    scenarios = point[:, None, :] + residuals[sampled_indices]
    scenarios = np.maximum(scenarios, floor).astype(np.float32)
    probabilities = np.full(count, 1.0 / count, dtype=np.float64)
    return scenarios, probabilities, sampled_indices


def write_validation_residual_artifact(
    artifact_dir: Path,
    actual_kw: np.ndarray,
    point_forecast_kw: np.ndarray,
    *,
    warmup_samples: int,
    source_model: str,
    source_config: str,
) -> tuple[Path, Path]:
    """Persist the validation-only residual library and its leakage audit."""

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    residuals, metadata = build_validation_residual_library(
        actual_kw,
        point_forecast_kw,
        split_name="validation",
        warmup_samples=warmup_samples,
        source_artifact="validation_residual_library.npz",
    )
    metadata.update(
        {
            "source_model": str(source_model),
            "source_config": str(source_config),
        }
    )
    npz_path = output / "validation_residual_library.npz"
    metadata_path = output / "validation_residual_library.json"
    np.savez_compressed(npz_path, validation_residuals_kw=residuals)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return npz_path, metadata_path


__all__ = [
    "audit_cvar_discretization",
    "build_residual_scenarios",
    "build_validation_residual_library",
    "write_validation_residual_artifact",
]
