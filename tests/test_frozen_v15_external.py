from __future__ import annotations

import numpy as np
import pandas as pd

from rig_energy.validation.frozen_v15_external import (
    _causal_supply_fill,
    _forecast_metrics,
    _supply_regime,
)


def test_causal_supply_fill_uses_only_previous_values() -> None:
    frame = pd.DataFrame({"capacity": [100.0, np.nan, np.nan, 80.0]})
    filled, counts = _causal_supply_fill(frame, ["capacity"], limit=2)
    assert filled["capacity"].tolist() == [100.0, 100.0, 100.0, 80.0]
    assert counts == {"capacity": 2}


def test_supply_regime_boundaries_are_preregistered_margin_bands() -> None:
    values = np.asarray([0.15, 0.149, 0.05, 0.049, 0.0, -0.001])
    assert _supply_regime(values).tolist() == [
        "normal",
        "constrained",
        "constrained",
        "weak",
        "weak",
        "emergency",
    ]


def test_forecast_metrics_exclude_cold_start_warmup_for_every_model() -> None:
    truth = np.arange(24, dtype=float).reshape(4, 6)
    transition = np.asarray([False, False, True, False])
    predictions = {
        "perfect": truth.copy(),
        "bad_only_in_warmup": truth.copy(),
    }
    predictions["bad_only_in_warmup"][0] += 1000.0
    metrics = _forecast_metrics(
        truth,
        predictions,
        transition,
        warmup=1,
        peak_threshold=10.0,
    ).set_index("model")
    assert metrics.loc["perfect", "mae_kw"] == 0.0
    assert metrics.loc["bad_only_in_warmup", "mae_kw"] == 0.0
    assert metrics.loc["perfect", "scored_samples"] == 3
