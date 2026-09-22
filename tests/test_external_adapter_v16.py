from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.external_adapter_v16 import load_external_data_v16  # noqa: E402


def test_v16_adapter_supports_gzip_null_mapping_and_provenance_guard(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5s"),
            "total_active_power_kw": [100.0, None, 120.0, 130.0],
            "operation_state": ["drilling"] * 4,
            "quality_flag": ["GOOD", "BAD_SENSOR", "GOOD", "GOOD"],
            "record_origin": ["MECHANISTIC_SYNTHETIC_PUBLIC_CALIBRATED"] * 4,
            "storage_soc_pct": [70.0, None, 69.8, 69.7],
        }
    )
    path = tmp_path / "surrogate.csv.gz"
    raw.to_csv(path, index=False)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        """
column_mapping: {}
state_value_mapping:
unit_mapping:
adapter:
  frequency_seconds: 5
  timezone: Asia/Shanghai
  power_unit: kW
  max_interpolation_gap_steps: 3
""".strip(),
        encoding="utf-8",
    )

    data, audit = load_external_data_v16(path, mapping_path=mapping)
    assert audit.provenance_class == "synthetic_surrogate"
    assert audit.total_power_imputed_rows == 1
    assert audit.raw_quality_counts["BAD_SENSOR"] == 1
    assert data.loc[1, "source_quality_flag"] == "BAD_SENSOR"
    assert data.loc[1, "total_active_power_quality_flag"] == "imputed"
    assert data.loc[1, "quality_flag"] == "BAD_SENSOR"
    assert data["source_type"].eq("synthetic_surrogate").all()
    assert data["storage_soc_pct"].isna().sum() == 1
    assert str(data["timestamp"].dt.tz) == "Asia/Shanghai"

    with pytest.raises(ValueError, match="目标钻机实测数据"):
        load_external_data_v16(
            path,
            mapping_path=mapping,
            require_target_rig_field_data=True,
        )


def test_v16_adapter_accepts_explicit_field_origin(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=3, freq="5s", tz="Asia/Shanghai"
            ),
            "total_active_power_kw": [500.0, 510.0, 505.0],
            "operation_state": ["drilling"] * 3,
            "quality_flag": ["GOOD"] * 3,
            "record_origin": ["TARGET_RIG_FIELD_SCADA"] * 3,
        }
    )
    path = tmp_path / "field.csv"
    raw.to_csv(path, index=False)
    data, audit = load_external_data_v16(
        path, require_target_rig_field_data=True
    )
    assert audit.provenance_class == "target_rig_field_scada"
    assert data["source_type"].eq("scada").all()
    assert data["quality_flag"].eq("observed").all()

