from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.validation.public_electrical_forecast import (
    apply_timestamp_causal_error_feedback,
    make_causal_forecast_table,
    mase_scale_from_training,
)


def test_forecast_table_uses_time_lags_and_does_not_cross_gap() -> None:
    timestamp = pd.date_range("2026-01-01", periods=200, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "measured_load_kw": np.arange(200, dtype=float),
            "quality_ok": True,
        }
    )
    frame.loc[100, "quality_ok"] = False

    table, features = make_causal_forecast_table(
        frame,
        horizon_minutes=15,
        load_lags_minutes=[0, 1, 5],
        rolling_windows_minutes=[5],
    )

    assert "load_lag_0" in features
    row = table.loc[table["origin_timestamp"].eq(timestamp[50])].iloc[0]
    assert row["load_lag_0"] == 50.0
    assert row["load_lag_5"] == 45.0
    assert row["target_kw"] == 65.0
    assert not table["origin_timestamp"].between(timestamp[100], timestamp[104]).any()


def test_forecast_target_calendar_is_known_future_time() -> None:
    timestamp = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {"timestamp": timestamp, "measured_load_kw": 1.0, "quality_ok": True}
    )
    table, _ = make_causal_forecast_table(
        frame,
        horizon_minutes=5,
        load_lags_minutes=[0],
        rolling_windows_minutes=[2],
    )
    assert (table["target_timestamp"] - table["origin_timestamp"]).eq(
        pd.Timedelta(minutes=5)
    ).all()


def test_timestamp_feedback_releases_only_observed_targets_across_gap() -> None:
    origins = pd.Series(pd.to_datetime(
        ["2026-01-01T00:00Z", "2026-01-01T00:01Z", "2026-01-01T00:10Z"],
        utc=True,
    ))
    targets = origins + pd.Timedelta(minutes=5)
    base = np.array([1.0, 1.0, 1.0])
    actual = np.array([2.0, 3.0, 4.0])

    corrected, bias = apply_timestamp_causal_error_feedback(
        base,
        actual,
        origins,
        targets,
        smoothing=1.0,
        correction_clip_kw=10.0,
    )

    assert corrected[:2].tolist() == [1.0, 1.0]
    # At 00:10, targets from 00:05 and 00:06 are both observable; the latest
    # released residual is 2 kW.
    assert corrected[2] == 3.0
    assert bias[2] == 2.0


def test_mase_scale_uses_only_adjacent_valid_minutes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC"),
            "measured_load_kw": [1.0, 2.0, 100.0, 4.0, 7.0],
            "quality_ok": [True, True, False, True, True],
        }
    )
    assert mase_scale_from_training(frame) == 2.0
