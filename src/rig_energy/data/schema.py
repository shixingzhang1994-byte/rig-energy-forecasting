from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "total_active_power_kw")

OPTIONAL_LOAD_COLUMNS = (
    "operation_state",
    "operation_state_code",
    "auxiliary_power_kw",
    "mud_pump_power_kw",
    "topdrive_power_kw",
    "drawworks_power_kw",
    "other_power_kw",
    "quality_flag",
    "source_type",
    "well_depth_m",
    "formation_hardness_index",
    "pump_strokes_per_min",
    "standpipe_pressure_mpa",
    "topdrive_rpm",
    "hookload_pct",
    "hook_speed_m_per_s",
    "rate_of_penetration_m_per_h",
)

# 这些字段当前训练阶段不强制使用，但接口预留给后续供需匹配和协同优化。
OPTIONAL_SUPPLY_COLUMNS = (
    "grid_active_power_kw",
    "grid_available_capacity_kw",
    "generator_active_power_kw",
    "generator_available_capacity_kw",
    "storage_active_power_kw",
    "storage_soc_pct",
    "storage_available_charge_power_kw",
    "storage_available_discharge_power_kw",
    "electricity_price_yuan_per_kwh",
    "diesel_price_yuan_per_l",
)

STATE_TO_CODE = {
    "unknown": 0,
    "idle": 1,
    "circulation": 2,
    "drilling": 3,
    "connection": 4,
    "tripping": 5,
    "maintenance": 6,
}
CODE_TO_STATE = {value: key for key, value in STATE_TO_CODE.items()}


@dataclass(frozen=True)
class SchemaReport:
    row_count: int
    start_time: str
    end_time: str
    median_interval_seconds: float
    missing_power_rows: int
    unknown_state_rows: int
    min_power_kw: float
    max_power_kw: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少项目统一字段: {missing}")


def validate_canonical_frame(
    frame: pd.DataFrame,
    *,
    allow_missing_power: bool = False,
    sort: bool = True,
) -> tuple[pd.DataFrame, SchemaReport]:
    """Validate and normalize data to the canonical project schema.

    The minimum contract is timestamp + total_active_power_kw.  All model code
    reads this schema, so a real SCADA file only needs an adapter mapping.
    """

    _require_columns(frame, REQUIRED_COLUMNS)
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        bad = int(data["timestamp"].isna().sum())
        raise ValueError(f"timestamp 中有 {bad} 行无法解析")

    data["total_active_power_kw"] = pd.to_numeric(
        data["total_active_power_kw"], errors="coerce"
    )
    missing_power = int(data["total_active_power_kw"].isna().sum())
    if missing_power and not allow_missing_power:
        raise ValueError(f"total_active_power_kw 中有 {missing_power} 个缺失值")

    if sort:
        data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    data = data.reset_index(drop=True)

    if "operation_state" not in data:
        data["operation_state"] = "unknown"
    data["operation_state"] = (
        data["operation_state"].astype("string").str.strip().str.lower().fillna("unknown")
    )
    known_states = set(STATE_TO_CODE)
    data.loc[~data["operation_state"].isin(known_states), "operation_state"] = "unknown"
    data["operation_state_code"] = data["operation_state"].map(STATE_TO_CODE).astype("int16")

    if "quality_flag" not in data:
        data["quality_flag"] = np.where(
            data["total_active_power_kw"].isna(), "missing", "observed"
        )
    if "source_type" not in data:
        data["source_type"] = "unknown"

    interval = data["timestamp"].diff().dt.total_seconds().dropna()
    median_interval = float(interval.median()) if len(interval) else 0.0
    finite_power = data["total_active_power_kw"].dropna()
    report = SchemaReport(
        row_count=len(data),
        start_time=str(data["timestamp"].iloc[0]),
        end_time=str(data["timestamp"].iloc[-1]),
        median_interval_seconds=median_interval,
        missing_power_rows=missing_power,
        unknown_state_rows=int((data["operation_state"] == "unknown").sum()),
        min_power_kw=float(finite_power.min()) if len(finite_power) else float("nan"),
        max_power_kw=float(finite_power.max()) if len(finite_power) else float("nan"),
    )
    return data, report
