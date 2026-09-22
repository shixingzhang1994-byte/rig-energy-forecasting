from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .schema import OPTIONAL_LOAD_COLUMNS, OPTIONAL_SUPPLY_COLUMNS, validate_canonical_frame


FIELD_ORIGIN_TOKENS = ("TARGET_RIG_FIELD", "FIELD_SCADA", "FIELD_MEASURED")
SYNTHETIC_ORIGIN_TOKENS = ("SYNTHETIC", "SIMULATION", "SURROGATE")


@dataclass(frozen=True)
class ExternalDataAudit:
    path: str
    provenance_class: str
    record_origins: list[str]
    row_count_raw: int
    row_count_canonical: int
    start_time: str
    end_time: str
    median_interval_seconds: float
    duplicate_timestamps_raw: int
    raw_quality_counts: dict[str, int]
    final_quality_counts: dict[str, int]
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    total_power_imputed_rows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".csv" or name.endswith((".csv.gz", ".csv.gzip")):
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(
        f"不支持的数据格式: {path.name}; 请使用 CSV、CSV.GZ、XLSX 或 Parquet"
    )


def _mapping_section(mapping: dict[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"映射配置 {name} 必须是字典")
    return value


def _infer_provenance(frame: pd.DataFrame) -> tuple[str, list[str]]:
    if "record_origin" not in frame:
        return "unknown", []
    origins = sorted(
        {
            str(value).strip()
            for value in frame["record_origin"].dropna().unique()
            if str(value).strip()
        }
    )
    upper = "|".join(origins).upper()
    if any(token in upper for token in SYNTHETIC_ORIGIN_TOKENS):
        return "synthetic_surrogate", origins
    if any(token in upper for token in FIELD_ORIGIN_TOKENS):
        return "target_rig_field_scada", origins
    return "unknown", origins


def _is_kw_quantity(column: str) -> bool:
    return column.endswith("_power_kw") or column.endswith("_capacity_kw")


def _convert_power_to_kw(series: pd.Series, unit: str) -> pd.Series:
    factors = {"w": 0.001, "kw": 1.0, "mw": 1000.0}
    key = str(unit).strip().lower()
    if key not in factors:
        raise ValueError(f"不支持的功率单位: {unit}")
    return pd.to_numeric(series, errors="coerce") * factors[key]


def _parse_timestamp(
    values: pd.Series, *, timestamp_format: str | None, timezone: str | None
) -> pd.Series:
    parsed = pd.to_datetime(values, format=timestamp_format, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"timestamp 中有 {int(parsed.isna().sum())} 行无法解析")
    if timezone:
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(
                timezone, ambiguous="NaT", nonexistent="NaT"
            )
        else:
            parsed = parsed.dt.tz_convert(timezone)
        if parsed.isna().any():
            raise ValueError("timestamp 在时区本地化后出现歧义或不存在的时间")
    return parsed


def load_external_data_v16(
    path: Path,
    *,
    mapping_path: Path | None = None,
    require_target_rig_field_data: bool = False,
) -> tuple[pd.DataFrame, ExternalDataAudit]:
    """Load external data without allowing provenance or quality flags to drift.

    This V16-only adapter intentionally lives outside the frozen V15 adapter. It
    may be used for engineering preflight with surrogate data, while
    ``require_target_rig_field_data=True`` is the hard gate for an external-field
    claim.
    """

    path = Path(path)
    raw = _read_table(path)
    mapping: dict[str, Any] = {}
    if mapping_path is not None:
        mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8")) or {}
        if not isinstance(mapping, dict):
            raise ValueError("映射配置根节点必须是字典")

    column_mapping = _mapping_section(mapping, "column_mapping")
    state_mapping = _mapping_section(mapping, "state_value_mapping")
    unit_mapping = _mapping_section(mapping, "unit_mapping")
    adapter_cfg = _mapping_section(mapping, "adapter")

    rename = {
        source: canonical
        for canonical, source in column_mapping.items()
        if source in raw.columns
    }
    data = raw.rename(columns=rename).copy()
    if "timestamp" not in data or "total_active_power_kw" not in data:
        raise ValueError("映射后仍缺少 timestamp 或 total_active_power_kw")

    provenance_class, origins = _infer_provenance(data)
    if require_target_rig_field_data and provenance_class != "target_rig_field_scada":
        raise ValueError(
            "外部现场测试要求目标钻机实测数据；"
            f"当前 provenance_class={provenance_class}, record_origins={origins}"
        )

    duplicate_count = int(data["timestamp"].duplicated().sum())
    missing_before = {
        name: int(value)
        for name, value in data.isna().sum().items()
        if int(value) > 0
    }
    raw_quality = (
        data["quality_flag"].astype("string").fillna("missing").value_counts().to_dict()
        if "quality_flag" in data
        else {}
    )

    data["timestamp"] = _parse_timestamp(
        data["timestamp"],
        timestamp_format=adapter_cfg.get("timestamp_format"),
        timezone=adapter_cfg.get("timezone"),
    )
    default_power_unit = adapter_cfg.get("power_unit", "kW")
    optional = set(OPTIONAL_LOAD_COLUMNS) | set(OPTIONAL_SUPPLY_COLUMNS)
    quantity_columns = {
        "total_active_power_kw",
        *(name for name in optional if _is_kw_quantity(name)),
    }
    for column in quantity_columns.intersection(data.columns):
        data[column] = _convert_power_to_kw(
            data[column], unit_mapping.get(column, default_power_unit)
        )

    if "operation_state" in data and state_mapping:
        normalized = {
            str(source).strip().lower(): str(target).strip().lower()
            for source, target in state_mapping.items()
        }
        state = data["operation_state"].astype("string").str.strip().str.lower()
        data["operation_state"] = state.map(normalized).fillna(state)

    original_quality = (
        data["quality_flag"].astype("string").fillna("missing")
        if "quality_flag" in data
        else pd.Series("unknown", index=data.index, dtype="string")
    )
    data["source_quality_flag"] = original_quality

    frequency_seconds = int(adapter_cfg.get("frequency_seconds", 5))
    data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    data = data.set_index("timestamp")
    discrete_columns = {
        name
        for name in data.columns
        if name.endswith(("_status", "_code"))
        or name
        in {
            "operation_state",
            "operation_substate",
            "quality_flag",
            "source_quality_flag",
            "source_type",
            "source_tag",
            "record_origin",
            "generator_units_on",
        }
    }
    numeric_columns = [
        name
        for name in data.select_dtypes(include="number").columns
        if name not in discrete_columns
    ]
    other_columns = [name for name in data.columns if name not in numeric_columns]
    numeric = data[numeric_columns].resample(f"{frequency_seconds}s").mean()
    other = data[other_columns].resample(f"{frequency_seconds}s").last().ffill()
    data = pd.concat([numeric, other], axis=1).sort_index().reset_index()

    missing_power_before = data["total_active_power_kw"].isna()
    max_gap = int(adapter_cfg.get("max_interpolation_gap_steps", 3))
    if max_gap > 0:
        data["total_active_power_kw"] = data["total_active_power_kw"].interpolate(
            method="linear", limit=max_gap, limit_area="inside"
        )
    imputed_power = missing_power_before & data["total_active_power_kw"].notna()
    data["total_active_power_quality_flag"] = np.where(
        imputed_power,
        "imputed",
        np.where(data["total_active_power_kw"].isna(), "missing", "observed"),
    )
    source_quality = data["source_quality_flag"].astype("string").fillna("missing")
    source_good = source_quality.str.upper().isin({"GOOD", "OK", "VALID", "OBSERVED"})
    data["quality_flag"] = source_quality
    data.loc[source_good, "quality_flag"] = data.loc[
        source_good, "total_active_power_quality_flag"
    ]

    configured_source = adapter_cfg.get("source_type")
    if configured_source:
        data["source_type"] = str(configured_source)
    elif provenance_class == "synthetic_surrogate":
        data["source_type"] = "synthetic_surrogate"
    elif provenance_class == "target_rig_field_scada":
        data["source_type"] = "scada"
    else:
        data["source_type"] = "unknown"

    data = data.dropna(subset=["total_active_power_kw"])
    data, schema_report = validate_canonical_frame(data)
    missing_after = {
        name: int(value)
        for name, value in data.isna().sum().items()
        if int(value) > 0
    }
    audit = ExternalDataAudit(
        path=str(path),
        provenance_class=provenance_class,
        record_origins=origins,
        row_count_raw=len(raw),
        row_count_canonical=len(data),
        start_time=schema_report.start_time,
        end_time=schema_report.end_time,
        median_interval_seconds=schema_report.median_interval_seconds,
        duplicate_timestamps_raw=duplicate_count,
        raw_quality_counts={str(k): int(v) for k, v in raw_quality.items()},
        final_quality_counts={
            str(k): int(v)
            for k, v in data["quality_flag"].value_counts(dropna=False).items()
        },
        missing_before=missing_before,
        missing_after=missing_after,
        total_power_imputed_rows=int(imputed_power.sum()),
    )
    return data, audit
