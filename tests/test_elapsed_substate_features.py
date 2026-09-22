from __future__ import annotations

import numpy as np
import pandas as pd

from rig_energy.validation.multivariate_causal import add_causal_derived_features


def test_elapsed_features_reset_only_on_observed_state_change() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="5s"),
            "total_active_power_kw": [1, 2, 3, 4, 5],
            "operation_state": ["drilling", "drilling", "connection", "connection", "drilling"],
            "operation_substate": ["rotary", "rotary", "pipe", "pipe", "rotary"],
        }
    )
    output, vocabulary = add_causal_derived_features(frame)
    np.testing.assert_allclose(
        output["operation_state_elapsed_minutes"],
        [0.0, 0.083333, 0.0, 0.083333, 0.0],
        atol=1e-6,
    )
    assert set(vocabulary) == {"pipe", "rotary"}
