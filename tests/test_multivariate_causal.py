from __future__ import annotations

import numpy as np
import pandas as pd

from rig_energy.validation.multivariate_causal import (
    FeatureScaler,
    MultivariateCausalDataset,
)


def test_multivariate_dataset_never_places_future_rows_in_input() -> None:
    frame = pd.DataFrame(
        {
            "total_active_power_kw": np.arange(20, dtype=np.float32),
            "sensor": np.arange(100, 120, dtype=np.float32),
            "operation_state_code": np.ones(20, dtype=np.int16),
        }
    )
    scaler = FeatureScaler.fit(frame.iloc[:10], ["total_active_power_kw", "sensor"])
    dataset = MultivariateCausalDataset(frame, scaler, history=5, horizon=3, stride=1)
    item = dataset[0]
    recovered_input = item["x"][:, 0].numpy() * scaler.std[0] + scaler.mean[0]
    recovered_target = item["y"].numpy() * scaler.std[0] + scaler.mean[0]
    np.testing.assert_allclose(
        recovered_input, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-6
    )
    np.testing.assert_allclose(recovered_target, [5.0, 6.0, 7.0], atol=1e-6)


def test_substate_embedding_does_not_change_main_state_transition_label() -> None:
    frame = pd.DataFrame(
        {
            "total_active_power_kw": np.arange(10, dtype=np.float32),
            "sensor": np.arange(20, 30, dtype=np.float32),
            "operation_state_code": np.ones(10, dtype=np.int16),
            "operation_substate_code": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    scaler = FeatureScaler.fit(frame, ["total_active_power_kw", "sensor"])
    dataset = MultivariateCausalDataset(
        frame,
        scaler,
        history=3,
        horizon=3,
        stride=1,
        state_column="operation_substate_code",
    )
    assert not bool(dataset[0]["future_transition"])
