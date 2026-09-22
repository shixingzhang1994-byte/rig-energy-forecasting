from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.windowing import PowerScaler, WindowedRigDataset  # noqa: E402


def test_future_transition_flag_excludes_changes_only_seen_in_history():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="5s"),
            "total_active_power_kw": np.linspace(400.0, 460.0, 7),
            # 历史4点内发生0->1，预测目标两点保持1不变。
            "operation_state_code": [0, 1, 1, 1, 1, 1, 1],
        }
    )
    scaler = PowerScaler.fit(frame["total_active_power_kw"].to_numpy())
    dataset = WindowedRigDataset(
        frame, scaler, history_steps=4, horizon_steps=2, stride=1
    )
    sample = dataset[0]
    assert bool(sample["history_transition"])
    assert not bool(sample["future_transition"])
    assert not bool(sample["transition"])
