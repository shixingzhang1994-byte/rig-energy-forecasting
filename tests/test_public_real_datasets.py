from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import (
    aggregate_utah_forge_one_minute,
    audit_three_w,
)


def _utah_row(timestamp: str, *, rop: float, rpm: float, flow: float) -> dict[str, object]:
    stamp = pd.Timestamp(timestamp)
    return {
        "Date": stamp.strftime("%-m/%-d/%y"),
        "Time": stamp.strftime("%H:%M:%S"),
        "Depth (m)": 100.0,
        "Block Ht (m)": 10.0,
        "Bit Depth (m)": 99.0,
        "ROP(1 m)": rop,
        "Hookload (kg)": 10_000.0,
        "Pump Pressure (psi)": 1000.0,
        "Pump Press (KPa)": 6000.0,
        "Surface Torque (psi)": 100.0,
        "Surface Torque (KPa)": 700.0,
        "Rotary Speed (rpm)": rpm,
        "Pump 1 (spm)": 0.0,
        "Pump 2 (spm)": 0.0,
        "Flow In (liters/min)": flow,
    }


def test_utah_adapter_builds_process_proxy_without_power_claim(tmp_path: Path) -> None:
    path = tmp_path / "forge.csv"
    pd.DataFrame(
        [
            _utah_row("2017-01-01T00:00:00", rop=1.0, rpm=60.0, flow=600.0),
            _utah_row("2017-01-01T00:00:01", rop=1.0, rpm=60.0, flow=600.0),
            _utah_row("2017-01-01T00:01:00", rop=0.0, rpm=0.0, flow=600.0),
        ]
    ).to_csv(path, index=False)

    canonical, metadata = aggregate_utah_forge_one_minute(path, chunksize=2)

    assert canonical["external_coarse_state"].tolist() == ["drilling", "circulation"]
    assert np.isclose(canonical.loc[0, "pump_proxy_kw_mean"], 60.0)
    assert metadata["raw_rows"] == 3
    assert metadata["supports_real_operation_validation"] is True
    assert metadata["supports_measured_total_power_validation"] is False
    assert metadata["electrical_power_channels"] == []


def test_three_w_audit_keeps_real_events_out_of_power_claim(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    dataset = root / "dataset"
    (dataset / "0").mkdir(parents=True)
    (dataset / "3").mkdir(parents=True)
    pd.DataFrame(
        {"P-PDG": [1.0, 2.0], "class": pd.Series([0, 0], dtype="Int64")}
    ).to_parquet(dataset / "0" / "WELL-00001_20170101000000.parquet")
    pd.DataFrame(
        {"P-PDG": [3.0], "class": pd.Series([3], dtype="Int64")}
    ).to_parquet(dataset / "3" / "SIMULATED_00001.parquet")

    inventory, metadata = audit_three_w(dataset)

    assert len(inventory) == 2
    assert metadata["file_count_by_source"] == {"real": 1, "simulated": 1}
    assert metadata["real_well_count"] == 1
    assert metadata["supports_real_well_event_validation"] is True
    assert metadata["supports_drilling_process_validation"] is False
    assert metadata["supports_measured_total_power_validation"] is False
    assert metadata["electrical_power_channels"] == []
