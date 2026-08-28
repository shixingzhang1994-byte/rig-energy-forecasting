from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.adapters import CanonicalFileDataSource, SyntheticDataSource  # noqa: E402
from rig_energy.data.schema import REQUIRED_COLUMNS, validate_canonical_frame  # noqa: E402


def test_synthetic_conforms_to_canonical_schema():
    config = yaml.safe_load((PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8"))
    config["project"]["days"] = 1
    data = SyntheticDataSource(config).load()
    assert all(column in data.columns for column in REQUIRED_COLUMNS)
    assert data["timestamp"].is_monotonic_increasing
    assert data["total_active_power_kw"].notna().all()
    assert set(data["operation_state"].unique()) <= {
        "idle", "circulation", "drilling", "connection", "tripping", "maintenance"
    }


def test_synthetic_process_signals_are_physical_and_interface_optional():
    config = yaml.safe_load((PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8"))
    config["project"]["days"] = 1
    data = SyntheticDataSource(config).load()
    process_columns = {
        "well_depth_m",
        "formation_hardness_index",
        "pump_strokes_per_min",
        "standpipe_pressure_mpa",
        "topdrive_rpm",
        "hookload_pct",
        "hook_speed_m_per_s",
        "rate_of_penetration_m_per_h",
    }
    assert process_columns <= set(data.columns)
    assert data["well_depth_m"].is_monotonic_increasing
    assert data["hookload_pct"].between(0.0, 100.0).all()
    assert data["pump_strokes_per_min"].between(0.0, 120.0).all()
    assert (data.loc[data["operation_state"] != "drilling", "rate_of_penetration_m_per_h"] == 0.0).all()


def test_csv_mapping_and_power_unit_conversion(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "Time": pd.date_range("2026-01-01", periods=5, freq="5s"),
            "ActivePower": [1_000_000, 1_050_000, 1_100_000, 1_080_000, 1_020_000],
            "OperationState": ["idle"] * 5,
        }
    )
    csv_path = tmp_path / "scada.csv"
    mapping_path = tmp_path / "mapping.yaml"
    raw.to_csv(csv_path, index=False)
    mapping_path.write_text(
        """
column_mapping:
  timestamp: Time
  total_active_power_kw: ActivePower
  operation_state: OperationState
adapter:
  frequency_seconds: 5
  power_unit: W
""".strip(),
        encoding="utf-8",
    )
    data = CanonicalFileDataSource(csv_path, mapping_path).load()
    data, report = validate_canonical_frame(data)
    assert report.row_count == 5
    assert abs(data["total_active_power_kw"].iloc[0] - 1000.0) < 1e-6


def test_state_value_and_optional_power_unit_mapping(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "Time": pd.date_range("2026-01-01", periods=3, freq="5s"),
            "ActivePower": [0.5, 0.6, 0.7],
            "State": ["钻进", "钻进", "起下钻"],
            "GridPowerW": [100_000, 120_000, 130_000],
        }
    )
    csv_path = tmp_path / "scada.csv"
    mapping_path = tmp_path / "mapping.yaml"
    raw.to_csv(csv_path, index=False)
    mapping_path.write_text(
        """
column_mapping:
  timestamp: Time
  total_active_power_kw: ActivePower
  operation_state: State
  grid_active_power_kw: GridPowerW
state_value_mapping:
  钻进: drilling
  起下钻: tripping
unit_mapping:
  total_active_power_kw: MW
  grid_active_power_kw: W
adapter:
  frequency_seconds: 5
  power_unit: kW
""".strip(),
        encoding="utf-8",
    )

    data = CanonicalFileDataSource(csv_path, mapping_path).load()
    assert list(data["operation_state"]) == ["drilling", "drilling", "tripping"]
    assert list(data["total_active_power_kw"].round(3)) == [500.0, 600.0, 700.0]
    assert list(data["grid_active_power_kw"].round(3)) == [100.0, 120.0, 130.0]
