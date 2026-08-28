from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

from .schema import OPTIONAL_LOAD_COLUMNS, OPTIONAL_SUPPLY_COLUMNS, validate_canonical_frame
from .synthetic import generate_synthetic_rig_load


class RigLoadDataSource(Protocol):
    def load(self) -> pd.DataFrame: ...


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"不支持的数据格式: {suffix}; 请使用 CSV、XLSX 或 Parquet")


def _convert_power_to_kw(series: pd.Series, unit: str) -> pd.Series:
    factors = {"w": 0.001, "kw": 1.0, "mw": 1000.0}
    key = unit.strip().lower()
    if key not in factors:
        raise ValueError(f"不支持的功率单位: {unit}")
    return pd.to_numeric(series, errors="coerce") * factors[key]


def _is_kw_quantity(column: str) -> bool:
    return column.endswith("_power_kw") or column.endswith("_capacity_kw")


@dataclass
class CanonicalFileDataSource:
    path: Path
    mapping_path: Path | None = None

    def load(self) -> pd.DataFrame:
        raw = _read_table(Path(self.path))
        mapping_cfg: dict = {}
        if self.mapping_path is not None:
            mapping_cfg = yaml.safe_load(Path(self.mapping_path).read_text(encoding="utf-8"))
        column_mapping = mapping_cfg.get("column_mapping", {})
        state_value_mapping = mapping_cfg.get("state_value_mapping", {})
        unit_mapping = mapping_cfg.get("unit_mapping", {})
        adapter_cfg = mapping_cfg.get("adapter", {})

        rename = {source: canonical for canonical, source in column_mapping.items() if source in raw}
        data = raw.rename(columns=rename).copy()
        if "timestamp" not in data or "total_active_power_kw" not in data:
            raise ValueError(
                "映射后仍缺少 timestamp 或 total_active_power_kw，请检查 column_mapping"
            )

        timestamp_format = adapter_cfg.get("timestamp_format")
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], format=timestamp_format, errors="coerce"
        )
        default_power_unit = adapter_cfg.get("power_unit", "kW")
        canonical_optional = set(OPTIONAL_LOAD_COLUMNS) | set(OPTIONAL_SUPPLY_COLUMNS)
        quantity_columns = {
            "total_active_power_kw",
            *(name for name in canonical_optional if _is_kw_quantity(name)),
        }
        for column in quantity_columns.intersection(data.columns):
            data[column] = _convert_power_to_kw(
                data[column], unit_mapping.get(column, default_power_unit)
            )

        if "operation_state" in data and state_value_mapping:
            normalized_mapping = {
                str(source).strip().lower(): str(target).strip().lower()
                for source, target in state_value_mapping.items()
            }
            raw_state = data["operation_state"].astype("string").str.strip().str.lower()
            data["operation_state"] = raw_state.map(normalized_mapping).fillna(raw_state)

        data, _ = validate_canonical_frame(data, allow_missing_power=True)
        frequency_seconds = int(adapter_cfg.get("frequency_seconds", 5))
        data = data.set_index("timestamp")

        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        text_columns = [c for c in data.columns if c not in numeric_columns]
        numeric = data[numeric_columns].resample(f"{frequency_seconds}s").mean()
        text = data[text_columns].resample(f"{frequency_seconds}s").ffill()
        data = pd.concat([numeric, text], axis=1).sort_index().reset_index()

        max_gap = int(adapter_cfg.get("max_interpolation_gap_steps", 3))
        missing_before = data["total_active_power_kw"].isna()
        data["total_active_power_kw"] = data["total_active_power_kw"].interpolate(
            method="linear", limit=max_gap, limit_area="inside"
        )
        data["quality_flag"] = "observed"
        data.loc[missing_before & data["total_active_power_kw"].notna(), "quality_flag"] = "imputed"
        data["source_type"] = "scada"
        data = data.dropna(subset=["total_active_power_kw"])
        data, _ = validate_canonical_frame(data)
        return data


@dataclass
class SyntheticDataSource:
    config: dict

    def load(self) -> pd.DataFrame:
        data = generate_synthetic_rig_load(self.config)
        data, _ = validate_canonical_frame(data)
        return data
