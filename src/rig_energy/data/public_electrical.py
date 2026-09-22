from __future__ import annotations

"""Adapters for public measured electrical-load datasets.

The canonical output deliberately separates measured load from endogenous
storage/grid channels.  The latter are useful for auditing the physical trace,
but they must not be used as future-known forecast inputs or claimed as rig
measurements.
"""

from collections.abc import Iterable
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from rig_energy.data.public_real_datasets import sha256_file


OPENCEM_COLUMNS = (
    "read_ts",
    "inverter",
    "outsumw",
    "pv1power",
    "battvolt",
    "battcurr",
    "battsoc",
    "battchgpower",
    "gridpowerw_a",
    "linepowerw_a",
)

OPENCEM_PRIMARY_QUALITY_COLUMNS = (
    "outsumw",
    "pv1power",
    "linepowerw_a",
    "gridpowerw_a",
    "battvolt",
    "battsoc",
)

OPENCEM_PHYSICAL_BOUNDS = {
    # Two 8 kW inverters are present. Bounds below are per inverter and retain
    # headroom above rating for telemetry transients while rejecting register
    # wrap/column-shift artefacts.
    "outsumw": (0.0, 10_000.0),
    "pv1power": (0.0, 15_000.0),
    "battvolt": (30.0, 65.0),
    "battcurr": (-250.0, 250.0),
    "battsoc": (0.0, 100.0),
    "battchgpower": (0.0, 10_000.0),
    "gridpowerw_a": (-20_000.0, 20_000.0),
    "linepowerw_a": (-20_000.0, 20_000.0),
}

TSUKUBA_COLUMNS = (
    "timestamp_local",
    "battery_active_power_kw",
    "battery_dc_voltage_v",
    "battery_dc_current_a",
    "grid_voltage_v",
    "grid_active_power_kw",
    "pv_active_power_kw",
    "battery_command_kw",
    "battery_soc_pct",
    "pv_array_1_kw",
    "pv_array_2_kw",
    "pv_array_3_kw",
    "pv_array_4_kw",
)

TSUKUBA_PHYSICAL_BOUNDS = {
    "battery_active_power_kw": (-500.0, 500.0),
    "battery_dc_voltage_v": (100.0, 600.0),
    "battery_dc_current_a": (-2_000.0, 2_000.0),
    "grid_voltage_v": (1_000.0, 15_000.0),
    "grid_active_power_kw": (0.0, 2_000.0),
    # The source includes small negative night-time offsets around zero.
    "pv_active_power_kw": (-5.0, 500.0),
    "battery_command_kw": (-500.0, 500.0),
    "battery_soc_pct": (0.0, 100.0),
}


def _select_best_opencem_record(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve duplicate inverter/timestamp records by observed field quality.

    The public export contains a known duplicate representation in some
    partitions: one record has the full 82-column layout and a second record
    has shifted/missing tail fields.  Selection is based only on completeness
    and physical plausibility, never on downstream model error.
    """

    frame = frame.copy()
    for column in OPENCEM_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["read_ts", "inverter"])
    frame["read_ts"] = frame["read_ts"].astype("int64")
    frame["inverter"] = frame["inverter"].astype("int16")
    frame = frame.loc[frame["inverter"].isin([1, 2])].copy()

    duplicate_mask = frame.duplicated(["read_ts", "inverter"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(
        frame.loc[duplicate_mask, ["read_ts", "inverter"]].drop_duplicates().shape[0]
    )
    if duplicate_keys:
        conflict = (
            frame.loc[duplicate_mask]
            .groupby(["read_ts", "inverter"], sort=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        conflicting_keys = int(conflict.sum())
    else:
        conflicting_keys = 0

    completeness = frame.loc[:, OPENCEM_PRIMARY_QUALITY_COLUMNS].notna().sum(axis=1)
    plausibility = pd.Series(0, index=frame.index, dtype="int16")
    for column, (lower, upper) in OPENCEM_PHYSICAL_BOUNDS.items():
        plausibility += frame[column].between(lower, upper, inclusive="both").astype("int16")
    frame["_quality_score"] = completeness * 10 + plausibility
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["read_ts", "inverter", "_quality_score", "_source_order"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["read_ts", "inverter"], keep="first")
    frame = frame.drop(columns=["_quality_score", "_source_order"])
    return frame, {
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "conflicting_duplicate_keys": conflicting_keys,
    }


def _apply_opencem_bounds(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    invalid_counts: dict[str, int] = {}
    for column, (lower, upper) in OPENCEM_PHYSICAL_BOUNDS.items():
        invalid = frame[column].notna() & ~frame[column].between(
            lower, upper, inclusive="both"
        )
        invalid_counts[column] = int(invalid.sum())
        frame.loc[invalid, column] = np.nan
    return frame, invalid_counts


def prepare_opencem_one_minute(
    paths: Iterable[str | Path],
    *,
    minimum_samples_per_minute: int = 3,
    minimum_complete_pair_fraction: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean OpenCEM inverter readings and aggregate them to one minute.

    ``measured_load_kw`` is the sum of the two inverter ``outsumw`` channels.
    PV, battery, and grid values are retained as audit-only measured context.
    The function does not interpolate the forecast target and explicitly marks
    incomplete minutes, preventing windows from silently crossing data gaps.
    """

    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one OpenCEM measurement CSV is required")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing OpenCEM files: " + ", ".join(missing))
    if not 0.0 <= minimum_complete_pair_fraction <= 1.0:
        raise ValueError("minimum_complete_pair_fraction must be in [0, 1]")

    parts: list[pd.DataFrame] = []
    raw_rows = 0
    duplicate_rows = 0
    duplicate_keys = 0
    conflicting_keys = 0
    invalid_counts = {column: 0 for column in OPENCEM_PHYSICAL_BOUNDS}
    file_records: list[dict[str, Any]] = []

    for path in sorted(resolved):
        raw = pd.read_csv(path, usecols=list(OPENCEM_COLUMNS), low_memory=False)
        raw_rows += len(raw)
        selected, duplicate_audit = _select_best_opencem_record(raw)
        selected, invalid = _apply_opencem_bounds(selected)
        parts.append(selected)
        duplicate_rows += duplicate_audit["duplicate_rows"]
        duplicate_keys += duplicate_audit["duplicate_keys"]
        conflicting_keys += duplicate_audit["conflicting_duplicate_keys"]
        for column, count in invalid.items():
            invalid_counts[column] += count
        file_records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "raw_rows": int(len(raw)),
            }
        )

    readings = pd.concat(parts, ignore_index=True)
    # Resolve any exact partition-boundary overlap using the same fixed rule.
    readings, cross_file_audit = _select_best_opencem_record(readings)
    duplicate_rows += cross_file_audit["duplicate_rows"]
    duplicate_keys += cross_file_audit["duplicate_keys"]
    conflicting_keys += cross_file_audit["conflicting_duplicate_keys"]

    readings = readings.dropna(subset=["outsumw"])
    readings["timestamp"] = pd.to_datetime(readings["read_ts"], unit="s", utc=True)
    timestamp_level = readings.groupby("timestamp", sort=True).agg(
        inverter_count=("inverter", "nunique"),
        measured_load_w=("outsumw", lambda values: values.sum(min_count=1)),
        measured_pv_w=("pv1power", lambda values: values.sum(min_count=1)),
        measured_battery_charge_w=(
            "battchgpower",
            lambda values: values.sum(min_count=1),
        ),
        measured_grid_power_w=(
            "gridpowerw_a",
            lambda values: values.sum(min_count=1),
        ),
        measured_line_power_w=(
            "linepowerw_a",
            lambda values: values.sum(min_count=1),
        ),
        measured_battery_soc_pct=("battsoc", "mean"),
        measured_battery_voltage_v=("battvolt", "mean"),
    )
    timestamp_level["complete_inverter_pair"] = timestamp_level["inverter_count"].eq(2)

    minute = timestamp_level.resample("1min").agg(
        sample_count=("measured_load_w", "count"),
        complete_pair_fraction=("complete_inverter_pair", "mean"),
        measured_load_kw=("measured_load_w", "mean"),
        measured_load_kw_p95=("measured_load_w", lambda x: x.quantile(0.95)),
        measured_pv_kw=("measured_pv_w", "mean"),
        measured_battery_charge_kw=("measured_battery_charge_w", "mean"),
        measured_grid_power_kw=("measured_grid_power_w", "mean"),
        measured_line_power_kw=("measured_line_power_w", "mean"),
        measured_battery_soc_pct=("measured_battery_soc_pct", "mean"),
        measured_battery_voltage_v=("measured_battery_voltage_v", "mean"),
    )
    watt_columns = [
        "measured_load_kw",
        "measured_load_kw_p95",
        "measured_pv_kw",
        "measured_battery_charge_kw",
        "measured_grid_power_kw",
        "measured_line_power_kw",
    ]
    minute.loc[:, watt_columns] = minute.loc[:, watt_columns] / 1000.0
    minute["quality_ok"] = (
        minute["sample_count"].ge(int(minimum_samples_per_minute))
        & minute["complete_pair_fraction"].ge(float(minimum_complete_pair_fraction))
        & minute["measured_load_kw"].notna()
    )
    minute["quality_issue"] = "ok"
    minute.loc[minute["sample_count"].lt(int(minimum_samples_per_minute)), "quality_issue"] = (
        "insufficient_samples"
    )
    minute.loc[
        minute["sample_count"].ge(int(minimum_samples_per_minute))
        & minute["complete_pair_fraction"].lt(float(minimum_complete_pair_fraction)),
        "quality_issue",
    ] = "incomplete_inverter_pair"
    minute.index.name = "timestamp"
    minute = minute.reset_index()

    metadata: dict[str, Any] = {
        "source_class": "official_public_real_microgrid_measurements",
        "source_name": "OpenCEM Dataset",
        "source_repository": "https://github.com/OpenCEM-platform/opencem-dataset",
        "repository_commit": "5884d253a5267fb240b7a8df6fa9e4d49a905167",
        "license": "CC BY 4.0 (dataset metadata); repository code MIT",
        "raw_files": file_records,
        "raw_rows": int(raw_rows),
        "deduplicated_inverter_rows": int(len(readings)),
        "duplicate_rows_encountered": int(duplicate_rows),
        "duplicate_keys_encountered": int(duplicate_keys),
        "conflicting_duplicate_keys": int(conflicting_keys),
        "invalid_values_replaced_with_null": invalid_counts,
        "timestamp_rows": int(len(timestamp_level)),
        "one_minute_rows": int(len(minute)),
        "quality_ok_rows": int(minute["quality_ok"].sum()),
        "quality_ok_fraction": float(minute["quality_ok"].mean()),
        "start": minute["timestamp"].min().isoformat(),
        "end": minute["timestamp"].max().isoformat(),
        "measured_total_active_power": True,
        "measured_load_definition": "sum(outsumw across inverter IDs) / 1000",
        "target_imputed": False,
        "supports_real_electrical_load_forecasting": True,
        "supports_target_rig_field_validation": False,
        "endogenous_audit_only_channels": [
            "measured_battery_charge_kw",
            "measured_battery_soc_pct",
            "measured_grid_power_kw",
        ],
        "evidence_boundary": (
            "Real campus microgrid electrical measurements, not drilling-rig power. "
            "Storage/grid channels are endogenous and are not future-known forecast inputs."
        ),
    }
    return minute, metadata


def prepare_refit_house_one_minute(
    path: str | Path,
    *,
    house_id: str,
    chunksize: int = 500_000,
    maximum_aggregate_w: float = 30_000.0,
    minimum_valid_samples_per_minute: int = 3,
    minimum_issue_free_fraction: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare one REFIT house using aggregate active-power measurements.

    Rows flagged by the public ``Issues`` column or outside the fixed
    residential-meter range are excluded from the minute mean. No target
    interpolation is performed.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {"Unix", "Aggregate", "Issues"}
    partials: list[pd.DataFrame] = []
    raw_rows = 0
    issue_rows = 0
    invalid_power_rows = 0
    for raw in pd.read_csv(path, usecols=list(required), chunksize=int(chunksize)):
        raw_rows += len(raw)
        timestamp = pd.to_datetime(
            pd.to_numeric(raw["Unix"], errors="coerce"), unit="s", utc=True
        )
        power = pd.to_numeric(raw["Aggregate"], errors="coerce")
        issue_free = pd.to_numeric(raw["Issues"], errors="coerce").eq(0)
        physically_valid = power.between(0.0, float(maximum_aggregate_w), inclusive="both")
        issue_rows += int((~issue_free).sum())
        invalid_power_rows += int((power.notna() & ~physically_valid).sum())
        valid = timestamp.notna() & issue_free & physically_valid
        chunk = pd.DataFrame(
            {
                "minute": timestamp.dt.floor("min"),
                "raw_count": 1,
                "valid_count": valid.astype("int16"),
                "valid_power_w": power.where(valid, 0.0),
            }
        ).dropna(subset=["minute"])
        partials.append(
            chunk.groupby("minute", as_index=False).agg(
                raw_count=("raw_count", "sum"),
                valid_count=("valid_count", "sum"),
                valid_power_sum_w=("valid_power_w", "sum"),
            )
        )
    if not partials:
        raise ValueError(f"REFIT file contains no readable timestamps: {path}")
    minute = (
        pd.concat(partials, ignore_index=True)
        .groupby("minute", as_index=False)
        .agg(
            raw_count=("raw_count", "sum"),
            valid_count=("valid_count", "sum"),
            valid_power_sum_w=("valid_power_sum_w", "sum"),
        )
        .set_index("minute")
        .asfreq("1min")
    )
    minute[["raw_count", "valid_count"]] = minute[["raw_count", "valid_count"]].fillna(0)
    minute["issue_free_fraction"] = minute["valid_count"].div(
        minute["raw_count"].replace(0, np.nan)
    )
    minute["measured_load_kw"] = minute["valid_power_sum_w"].div(
        minute["valid_count"].replace(0, np.nan)
    ) / 1000.0
    minute["quality_ok"] = (
        minute["valid_count"].ge(int(minimum_valid_samples_per_minute))
        & minute["issue_free_fraction"].ge(float(minimum_issue_free_fraction))
        & minute["measured_load_kw"].notna()
    )
    minute["quality_issue"] = "ok"
    minute.loc[
        minute["valid_count"].lt(int(minimum_valid_samples_per_minute)),
        "quality_issue",
    ] = "insufficient_valid_samples"
    minute.loc[
        minute["valid_count"].ge(int(minimum_valid_samples_per_minute))
        & minute["issue_free_fraction"].lt(float(minimum_issue_free_fraction)),
        "quality_issue",
    ] = "issue_flag_fraction"
    minute["site_id"] = str(house_id)
    minute.index.name = "timestamp"
    minute = minute.reset_index()
    minute = minute.drop(columns=["valid_power_sum_w"])
    metadata: dict[str, Any] = {
        "source_class": "official_public_real_residential_electrical_measurements",
        "source_name": "REFIT Electrical Load Measurements",
        "source_doi": "10.5281/zenodo.5063428",
        "license": "CC BY 4.0",
        "house_id": str(house_id),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "raw_rows": int(raw_rows),
        "issue_flagged_rows": int(issue_rows),
        "invalid_power_rows": int(invalid_power_rows),
        "one_minute_rows": int(len(minute)),
        "quality_ok_rows": int(minute["quality_ok"].sum()),
        "quality_ok_fraction": float(minute["quality_ok"].mean()),
        "start": minute["timestamp"].min().isoformat(),
        "end": minute["timestamp"].max().isoformat(),
        "measured_total_active_power": True,
        "measured_load_definition": "issue-free Aggregate watts averaged per minute / 1000",
        "target_imputed": False,
        "supports_real_electrical_load_forecasting": True,
        "supports_target_rig_field_validation": False,
        "evidence_boundary": (
            "Real residential aggregate electrical measurements, not microgrid or "
            "drilling-rig power; used as an independent load-shape stress test."
        ),
    }
    return minute, metadata


def prepare_uci_household_one_minute(
    path: str | Path,
    *,
    maximum_active_power_kw: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare the UCI one-minute household active-power series.

    The source is already on a complete one-minute calendar grid. Missing
    measurement fields are represented by ``?`` and remain missing.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = [
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
    ]
    raw = pd.read_csv(path, sep=";", usecols=columns, na_values=["?"], low_memory=False)
    timestamp = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        utc=True,
    )
    load = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    invalid_power = load.notna() & ~load.between(
        0.0, float(maximum_active_power_kw), inclusive="both"
    )
    load = load.mask(invalid_power)
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "measured_load_kw": load,
            "measured_reactive_power_kvar": pd.to_numeric(
                raw["Global_reactive_power"], errors="coerce"
            ),
            "measured_voltage_v": pd.to_numeric(raw["Voltage"], errors="coerce"),
            "measured_current_a": pd.to_numeric(
                raw["Global_intensity"], errors="coerce"
            ),
        }
    ).dropna(subset=["timestamp"])
    duplicate_rows = int(frame["timestamp"].duplicated(keep=False).sum())
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    frame = frame.set_index("timestamp").asfreq("1min")
    frame["quality_ok"] = frame["measured_load_kw"].notna()
    frame["quality_issue"] = np.where(frame["quality_ok"], "ok", "missing_or_invalid_power")
    frame["site_id"] = "UCI_household_235"
    frame = frame.reset_index()
    metadata: dict[str, Any] = {
        "source_class": "official_public_real_residential_electrical_measurements",
        "source_name": "UCI Individual Household Electric Power Consumption",
        "source_doi": "10.24432/C58K54",
        "license": "CC BY 4.0",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "raw_rows": int(len(raw)),
        "duplicate_timestamp_rows": duplicate_rows,
        "invalid_power_rows": int(invalid_power.sum()),
        "one_minute_rows": int(len(frame)),
        "quality_ok_rows": int(frame["quality_ok"].sum()),
        "quality_ok_fraction": float(frame["quality_ok"].mean()),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "measured_total_active_power": True,
        "measured_load_definition": "Global_active_power in kW on the original one-minute grid",
        "target_imputed": False,
        "supports_real_electrical_load_forecasting": True,
        "supports_target_rig_field_validation": False,
        "evidence_boundary": (
            "Real single-household electrical measurements, not microgrid or drilling-rig "
            "power; used as an independent long-duration load-shape stress test."
        ),
    }
    return frame, metadata


def prepare_tsukuba_microgrid_one_minute(
    cleaned_seconds_zip: str | Path,
    *,
    parent_archive: str | Path | None = None,
    chunksize: int = 500_000,
    minimum_valid_samples_per_minute: int = 45,
    minimum_valid_fraction: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream and prepare Tsukuba's cleaned one-second microgrid data.

    The public archive uses Deflate64, which Python's standard ``zipfile``
    reader cannot decode.  The system ``unzip`` implementation is therefore
    used as a streaming decoder; the 9.47 GB of CSV members never need to be
    materialised on disk.

    The dataset does not expose a direct native-load meter.  A physically
    balanced native-load estimate is derived from measured channels as
    ``grid import + PV generation + battery active power``.  Battery power is
    positive while discharging: this sign convention is independently
    checked from the negative relationship between power and SOC change.
    """

    path = Path(cleaned_seconds_zip)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in [0, 1]")

    # Some late source rows contain extra comma-delimited fields despite the
    # publisher's archive-level integrity check.  Count and isolate such rows
    # before pandas parsing rather than silently accepting a shifted schema.
    stream_filter = r'''
unzip -Z1 "$1" | while IFS= read -r member; do
  unzip -p "$1" "$member"
  printf '\n'
done | awk -F, '
BEGIN { OFS=","; headers=0; data=0; extended=0; bad=0; blank=0 }
{ sub(/\r$/, "") }
/^#/ { headers++; next }
/^$/ { blank++; next }
NF == 13 { print; data++; next }
NF == 15 && $14 == "" && $15 == "" {
  line=$1
  for (i=2; i<=13; i++) line=line OFS $i
  print line
  data++
  extended++
  next
}
{ bad++ }
END {
  printf "TSUKUBA_STREAM_AUDIT headers=%d data=%d extended=%d bad=%d blank=%d\n", headers, data, extended, bad, blank > "/dev/stderr"
}
' '''
    command = ["bash", "-o", "pipefail", "-c", stream_filter, "_", str(path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        raise RuntimeError("Unable to open unzip stream")

    partials: list[pd.DataFrame] = []
    raw_rows = 0
    invalid_timestamp_rows = 0
    invalid_counts = {column: 0 for column in TSUKUBA_PHYSICAL_BOUNDS}
    try:
        chunks = pd.read_csv(
            process.stdout,
            names=list(TSUKUBA_COLUMNS),
            encoding="cp932",
            chunksize=int(chunksize),
            low_memory=False,
        )
        for raw in chunks:
            raw_rows += len(raw)
            timestamp = pd.to_datetime(
                raw["timestamp_local"].astype(str).str.lstrip("'"),
                format="%Y/%m/%d %H:%M:%S",
                errors="coerce",
            )
            timestamp = timestamp.dt.tz_localize(
                "Asia/Tokyo", ambiguous="NaT", nonexistent="NaT"
            ).dt.tz_convert("UTC")
            invalid_timestamp_rows += int(timestamp.isna().sum())
            values = raw.loc[:, TSUKUBA_COLUMNS[1:]].apply(
                pd.to_numeric, errors="coerce"
            )
            for column, (lower, upper) in TSUKUBA_PHYSICAL_BOUNDS.items():
                invalid = values[column].notna() & ~values[column].between(
                    lower, upper, inclusive="both"
                )
                invalid_counts[column] += int(invalid.sum())
                values.loc[invalid, column] = np.nan

            balance_valid = (
                timestamp.notna()
                & values["grid_active_power_kw"].notna()
                & values["pv_active_power_kw"].notna()
                & values["battery_active_power_kw"].notna()
            )
            native_load = (
                values["grid_active_power_kw"]
                + values["pv_active_power_kw"]
                + values["battery_active_power_kw"]
            ).where(balance_valid)
            second = pd.DataFrame(
                {
                    "minute": timestamp.dt.floor("min"),
                    "raw_count": timestamp.notna().astype("int16"),
                    "valid_count": balance_valid.astype("int16"),
                    "native_load_sum_kw": native_load.fillna(0.0),
                    "grid_sum_kw": values["grid_active_power_kw"].fillna(0.0),
                    "grid_count": values["grid_active_power_kw"].notna().astype("int16"),
                    "pv_sum_kw": values["pv_active_power_kw"].fillna(0.0),
                    "pv_count": values["pv_active_power_kw"].notna().astype("int16"),
                    "battery_sum_kw": values["battery_active_power_kw"].fillna(0.0),
                    "battery_count": values["battery_active_power_kw"].notna().astype("int16"),
                    "soc_sum_pct": values["battery_soc_pct"].fillna(0.0),
                    "soc_count": values["battery_soc_pct"].notna().astype("int16"),
                    "battery_voltage_sum_v": values["battery_dc_voltage_v"].fillna(0.0),
                    "battery_voltage_count": values["battery_dc_voltage_v"].notna().astype("int16"),
                }
            ).dropna(subset=["minute"])
            partials.append(second.groupby("minute", as_index=False).sum())
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"unzip failed with exit code {return_code}: {stderr[-2000:]}")
    if not partials:
        raise ValueError(f"No readable Tsukuba timestamps found in {path}")
    stream_audit = {
        "header_rows": 0,
        "data_rows": int(raw_rows),
        "extended_trailing_empty_field_rows": 0,
        "bad_field_count_rows": 0,
        "blank_separator_rows": 0,
    }
    for line in stderr.splitlines():
        if not line.startswith("TSUKUBA_STREAM_AUDIT "):
            continue
        for token in line.split()[1:]:
            key, value = token.split("=", maxsplit=1)
            if key == "headers":
                stream_audit["header_rows"] = int(value)
            elif key == "data":
                stream_audit["data_rows"] = int(value)
            elif key == "extended":
                stream_audit["extended_trailing_empty_field_rows"] = int(value)
            elif key == "bad":
                stream_audit["bad_field_count_rows"] = int(value)
            elif key == "blank":
                stream_audit["blank_separator_rows"] = int(value)

    minute = pd.concat(partials, ignore_index=True).groupby(
        "minute", as_index=False
    ).sum()
    minute = minute.set_index("minute").asfreq("1min")
    count_columns = [column for column in minute.columns if column.endswith("count")]
    minute[count_columns] = minute[count_columns].fillna(0)
    minute["valid_fraction"] = minute["valid_count"].div(
        minute["raw_count"].replace(0, np.nan)
    )
    minute["measured_load_kw"] = minute["native_load_sum_kw"].div(
        minute["valid_count"].replace(0, np.nan)
    )
    minute["measured_grid_power_kw"] = minute["grid_sum_kw"].div(
        minute["grid_count"].replace(0, np.nan)
    )
    minute["measured_pv_kw"] = minute["pv_sum_kw"].div(
        minute["pv_count"].replace(0, np.nan)
    )
    minute["measured_battery_power_kw"] = minute["battery_sum_kw"].div(
        minute["battery_count"].replace(0, np.nan)
    )
    minute["measured_battery_soc_pct"] = minute["soc_sum_pct"].div(
        minute["soc_count"].replace(0, np.nan)
    )
    minute["measured_battery_voltage_v"] = minute["battery_voltage_sum_v"].div(
        minute["battery_voltage_count"].replace(0, np.nan)
    )
    minute["quality_ok"] = (
        minute["valid_count"].ge(int(minimum_valid_samples_per_minute))
        & minute["valid_fraction"].ge(float(minimum_valid_fraction))
        & minute["measured_load_kw"].notna()
        & minute["measured_load_kw"].ge(0.0)
    )
    minute["quality_issue"] = "ok"
    minute.loc[
        minute["valid_count"].lt(int(minimum_valid_samples_per_minute)),
        "quality_issue",
    ] = "insufficient_valid_samples"
    minute.loc[
        minute["valid_count"].ge(int(minimum_valid_samples_per_minute))
        & minute["valid_fraction"].lt(float(minimum_valid_fraction)),
        "quality_issue",
    ] = "low_valid_fraction"
    minute.loc[
        minute["measured_load_kw"].lt(0.0), "quality_issue"
    ] = "negative_power_balance"
    minute["site_id"] = "Tsukuba_building_microgrid"
    minute.index.name = "timestamp"
    minute = minute.reset_index()
    minute = minute.drop(
        columns=[
            "native_load_sum_kw",
            "grid_sum_kw",
            "grid_count",
            "pv_sum_kw",
            "pv_count",
            "battery_sum_kw",
            "battery_count",
            "soc_sum_pct",
            "soc_count",
            "battery_voltage_sum_v",
            "battery_voltage_count",
        ]
    )

    soc_change = minute["measured_battery_soc_pct"].diff()
    sign_mask = (
        minute["measured_battery_power_kw"].notna()
        & soc_change.notna()
        & soc_change.abs().lt(20.0)
    )
    sign_correlation = (
        float(
            minute.loc[sign_mask, "measured_battery_power_kw"].corr(
                soc_change[sign_mask]
            )
        )
        if int(sign_mask.sum()) >= 2
        else float("nan")
    )
    parent = Path(parent_archive) if parent_archive is not None else None
    metadata: dict[str, Any] = {
        "source_class": "official_public_real_building_microgrid_measurements",
        "source_name": "Tsukuba building microgrid one-second dataset",
        "source_doi": "10.6084/m9.figshare.7403954.v1",
        "license": "CC0 1.0",
        "cleaned_seconds_zip_path": str(path.resolve()),
        "cleaned_seconds_zip_size_bytes": path.stat().st_size,
        "cleaned_seconds_zip_sha256": sha256_file(path),
        "parent_archive_path": str(parent.resolve()) if parent is not None else None,
        "parent_archive_size_bytes": parent.stat().st_size if parent is not None else None,
        "parent_archive_sha256": sha256_file(parent) if parent is not None else None,
        "raw_rows": int(raw_rows),
        "invalid_timestamp_rows": int(invalid_timestamp_rows),
        "stream_schema_audit": stream_audit,
        "invalid_values_replaced_with_null": invalid_counts,
        "one_minute_rows": int(len(minute)),
        "quality_ok_rows": int(minute["quality_ok"].sum()),
        "quality_ok_fraction": float(minute["quality_ok"].mean()),
        "start": minute["timestamp"].min().isoformat(),
        "end": minute["timestamp"].max().isoformat(),
        "timezone_conversion": "Asia/Tokyo source time converted to UTC",
        "measured_total_active_power": False,
        "measured_load_definition": (
            "derived native load = measured grid import + measured PV generation + "
            "measured battery active power; battery positive means discharge"
        ),
        "battery_power_soc_change_correlation": sign_correlation,
        "battery_sign_check_passed": bool(sign_correlation < -0.20),
        "target_imputed": False,
        "supports_real_electrical_load_forecasting": True,
        "supports_real_bess_validation": True,
        "supports_target_rig_field_validation": False,
        "endogenous_audit_only_channels": [
            "measured_grid_power_kw",
            "measured_pv_kw",
            "measured_battery_power_kw",
            "measured_battery_soc_pct",
        ],
        "evidence_boundary": (
            "Real building microgrid and BESS measurements, not drilling-rig power. "
            "The forecasting target is a measured-channel power-balance derivation, "
            "not a direct load meter; storage/grid/PV channels are audit-only inputs."
        ),
    }
    return minute, metadata


def _bounded_numeric(
    frame: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
) -> tuple[pd.Series, int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.notna() & ~values.between(lower, upper, inclusive="both")
    return values.mask(invalid), int(invalid.sum())


def prepare_m5bat_one_minute(
    extracted_dir: str | Path,
    *,
    source_archive: str | Path | None = None,
    minimum_samples_per_minute: int = 30,
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare the complete 2017--2025 M5BAT field record at one minute.

    Annual BMS and power-conversion-controller Parquet files are streamed in
    bounded Arrow batches.  No measurement channel is interpolated.  The
    resulting table is intended for real BESS validation, not as a surrogate
    drilling-rig load dataset.
    """

    import pyarrow.parquet as pq

    directory = Path(extracted_dir)
    bms_paths = sorted(directory.glob("Exide1_BMS_*.parquet"))
    bsc_paths = sorted(directory.glob("Exide1_BSC_*.parquet"))
    if len(bms_paths) != 9 or len(bsc_paths) != 9:
        raise ValueError(
            f"Expected nine BMS and nine BSC annual files; found "
            f"{len(bms_paths)} and {len(bsc_paths)}"
        )

    bms_parts: list[pd.DataFrame] = []
    bsc_parts: list[pd.DataFrame] = []
    raw_bms_rows = 0
    raw_bsc_rows = 0
    invalid_counts = {
        "power_dc_W_bms": 0,
        "current_A_bms": 0,
        "voltage_bat_V_bms": 0,
        "soc_pct_bms": 0,
        "power_ac_kW_bsc": 0,
        "frequency_Hz_bsc": 0,
        "temperature_degC_bsc": 0,
        "power_ac_setpoint_kW_bsc": 0,
    }

    bms_columns = [
        "timestamp_utc",
        "power_dc_W_bms",
        "current_A_bms",
        "voltage_bat_V_bms",
        "soc_pct_bms",
        "flag_imbalance_voltage_bms",
        "flag_alarm_voltage_imbalance_bms",
    ]
    for path in bms_paths:
        parquet = pq.ParquetFile(path)
        raw_bms_rows += parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=int(batch_size), columns=bms_columns):
            raw = batch.to_pandas()
            if "timestamp_utc" not in raw.columns:
                raw = raw.reset_index()
            timestamp = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
            # Despite the published column name/codebook saying W, the stored
            # values are numerically in kW: e.g. -299 accompanies about -406 A
            # and 737 V (roughly -299 kW). Retain the numeric values as kW.
            dc_power, invalid = _bounded_numeric(raw, "power_dc_W_bms", -1_000.0, 1_000.0)
            invalid_counts["power_dc_W_bms"] += invalid
            dc_current, invalid = _bounded_numeric(raw, "current_A_bms", -2_000, 2_000)
            invalid_counts["current_A_bms"] += invalid
            dc_voltage, invalid = _bounded_numeric(raw, "voltage_bat_V_bms", 400, 800)
            invalid_counts["voltage_bat_V_bms"] += invalid
            soc, invalid = _bounded_numeric(raw, "soc_pct_bms", 0, 100)
            invalid_counts["soc_pct_bms"] += invalid
            core_valid = (
                timestamp.notna()
                & dc_power.notna()
                & dc_current.notna()
                & dc_voltage.notna()
                & soc.notna()
            )
            rows = pd.DataFrame(
                {
                    "timestamp": timestamp.dt.floor("min"),
                    "bms_raw_count": timestamp.notna().astype("int16"),
                    "bms_valid_count": core_valid.astype("int16"),
                    "dc_power_sum_kw": dc_power.fillna(0.0),
                    "dc_power_count": dc_power.notna().astype("int16"),
                    "dc_current_sum_a": dc_current.fillna(0.0),
                    "dc_current_count": dc_current.notna().astype("int16"),
                    "dc_voltage_sum_v": dc_voltage.fillna(0.0),
                    "dc_voltage_count": dc_voltage.notna().astype("int16"),
                    "soc_sum_pct": soc.fillna(0.0),
                    "soc_count": soc.notna().astype("int16"),
                    "imbalance_warning": pd.to_numeric(
                        raw["flag_imbalance_voltage_bms"], errors="coerce"
                    ).fillna(0),
                    "imbalance_alarm": pd.to_numeric(
                        raw["flag_alarm_voltage_imbalance_bms"], errors="coerce"
                    ).fillna(0),
                }
            ).dropna(subset=["timestamp"])
            bms_parts.append(
                rows.groupby("timestamp", as_index=False).agg(
                    bms_raw_count=("bms_raw_count", "sum"),
                    bms_valid_count=("bms_valid_count", "sum"),
                    dc_power_sum_kw=("dc_power_sum_kw", "sum"),
                    dc_power_count=("dc_power_count", "sum"),
                    dc_current_sum_a=("dc_current_sum_a", "sum"),
                    dc_current_count=("dc_current_count", "sum"),
                    dc_voltage_sum_v=("dc_voltage_sum_v", "sum"),
                    dc_voltage_count=("dc_voltage_count", "sum"),
                    soc_sum_pct=("soc_sum_pct", "sum"),
                    soc_count=("soc_count", "sum"),
                    imbalance_warning=("imbalance_warning", "max"),
                    imbalance_alarm=("imbalance_alarm", "max"),
                )
            )

    bsc_columns = [
        "timestamp_utc",
        "power_ac_kW_bsc",
        "frequency_Hz_bsc",
        "temperature_degC_bsc",
        "power_ac_setpoint_kW_bsc",
        "flag_fcr_active_bsc",
        "flag_standby_bsc",
        "flag_stop_bsc",
        "flag_failure_bsc",
        "flag_const_bsc",
    ]
    for path in bsc_paths:
        parquet = pq.ParquetFile(path)
        raw_bsc_rows += parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=int(batch_size), columns=bsc_columns):
            raw = batch.to_pandas()
            if "timestamp_utc" not in raw.columns:
                raw = raw.reset_index()
            timestamp = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
            ac_power, invalid = _bounded_numeric(raw, "power_ac_kW_bsc", -1_000, 1_000)
            invalid_counts["power_ac_kW_bsc"] += invalid
            frequency, invalid = _bounded_numeric(raw, "frequency_Hz_bsc", 45, 55)
            invalid_counts["frequency_Hz_bsc"] += invalid
            temperature, invalid = _bounded_numeric(raw, "temperature_degC_bsc", -30, 80)
            invalid_counts["temperature_degC_bsc"] += invalid
            setpoint, invalid = _bounded_numeric(raw, "power_ac_setpoint_kW_bsc", -1_000, 1_000)
            invalid_counts["power_ac_setpoint_kW_bsc"] += invalid
            core_valid = timestamp.notna() & ac_power.notna()
            rows = pd.DataFrame(
                {
                    "timestamp": timestamp.dt.floor("min"),
                    "bsc_raw_count": timestamp.notna().astype("int16"),
                    "bsc_valid_count": core_valid.astype("int16"),
                    "ac_power_sum_kw": ac_power.fillna(0.0),
                    "ac_power_count": ac_power.notna().astype("int16"),
                    "frequency_sum_hz": frequency.fillna(0.0),
                    "frequency_count": frequency.notna().astype("int16"),
                    "temperature_sum_degC": temperature.fillna(0.0),
                    "temperature_count": temperature.notna().astype("int16"),
                    "setpoint_sum_kw": setpoint.fillna(0.0),
                    "setpoint_count": setpoint.notna().astype("int16"),
                    "fcr_active": pd.to_numeric(raw["flag_fcr_active_bsc"], errors="coerce").fillna(0),
                    "standby": pd.to_numeric(raw["flag_standby_bsc"], errors="coerce").fillna(0),
                    "stopped": pd.to_numeric(raw["flag_stop_bsc"], errors="coerce").fillna(0),
                    "failure": pd.to_numeric(raw["flag_failure_bsc"], errors="coerce").fillna(0),
                    "constant_power": pd.to_numeric(raw["flag_const_bsc"], errors="coerce").fillna(0),
                }
            ).dropna(subset=["timestamp"])
            bsc_parts.append(
                rows.groupby("timestamp", as_index=False).agg(
                    bsc_raw_count=("bsc_raw_count", "sum"),
                    bsc_valid_count=("bsc_valid_count", "sum"),
                    ac_power_sum_kw=("ac_power_sum_kw", "sum"),
                    ac_power_count=("ac_power_count", "sum"),
                    frequency_sum_hz=("frequency_sum_hz", "sum"),
                    frequency_count=("frequency_count", "sum"),
                    temperature_sum_degC=("temperature_sum_degC", "sum"),
                    temperature_count=("temperature_count", "sum"),
                    setpoint_sum_kw=("setpoint_sum_kw", "sum"),
                    setpoint_count=("setpoint_count", "sum"),
                    fcr_active=("fcr_active", "max"),
                    standby=("standby", "max"),
                    stopped=("stopped", "max"),
                    failure=("failure", "max"),
                    constant_power=("constant_power", "max"),
                )
            )

    bms = pd.concat(bms_parts, ignore_index=True).groupby("timestamp", as_index=False).agg(
        bms_raw_count=("bms_raw_count", "sum"),
        bms_valid_count=("bms_valid_count", "sum"),
        dc_power_sum_kw=("dc_power_sum_kw", "sum"),
        dc_power_count=("dc_power_count", "sum"),
        dc_current_sum_a=("dc_current_sum_a", "sum"),
        dc_current_count=("dc_current_count", "sum"),
        dc_voltage_sum_v=("dc_voltage_sum_v", "sum"),
        dc_voltage_count=("dc_voltage_count", "sum"),
        soc_sum_pct=("soc_sum_pct", "sum"),
        soc_count=("soc_count", "sum"),
        imbalance_warning=("imbalance_warning", "max"),
        imbalance_alarm=("imbalance_alarm", "max"),
    )
    bsc = pd.concat(bsc_parts, ignore_index=True).groupby("timestamp", as_index=False).agg(
        bsc_raw_count=("bsc_raw_count", "sum"),
        bsc_valid_count=("bsc_valid_count", "sum"),
        ac_power_sum_kw=("ac_power_sum_kw", "sum"),
        ac_power_count=("ac_power_count", "sum"),
        frequency_sum_hz=("frequency_sum_hz", "sum"),
        frequency_count=("frequency_count", "sum"),
        temperature_sum_degC=("temperature_sum_degC", "sum"),
        temperature_count=("temperature_count", "sum"),
        setpoint_sum_kw=("setpoint_sum_kw", "sum"),
        setpoint_count=("setpoint_count", "sum"),
        fcr_active=("fcr_active", "max"),
        standby=("standby", "max"),
        stopped=("stopped", "max"),
        failure=("failure", "max"),
        constant_power=("constant_power", "max"),
    )
    minute = bms.merge(bsc, on="timestamp", how="outer").sort_values("timestamp")
    minute = minute.set_index("timestamp").asfreq("1min")
    count_columns = [column for column in minute.columns if column.endswith("count")]
    minute[count_columns] = minute[count_columns].fillna(0)
    minute["measured_battery_dc_power_kw"] = minute["dc_power_sum_kw"].div(
        minute["dc_power_count"].replace(0, np.nan)
    )
    minute["measured_battery_dc_current_a"] = minute["dc_current_sum_a"].div(
        minute["dc_current_count"].replace(0, np.nan)
    )
    minute["measured_battery_dc_voltage_v"] = minute["dc_voltage_sum_v"].div(
        minute["dc_voltage_count"].replace(0, np.nan)
    )
    minute["measured_battery_soc_pct"] = minute["soc_sum_pct"].div(
        minute["soc_count"].replace(0, np.nan)
    )
    minute["measured_ac_power_kw"] = minute["ac_power_sum_kw"].div(
        minute["ac_power_count"].replace(0, np.nan)
    )
    minute["measured_grid_frequency_hz"] = minute["frequency_sum_hz"].div(
        minute["frequency_count"].replace(0, np.nan)
    )
    minute["measured_temperature_degC"] = minute["temperature_sum_degC"].div(
        minute["temperature_count"].replace(0, np.nan)
    )
    minute["measured_ac_setpoint_kw"] = minute["setpoint_sum_kw"].div(
        minute["setpoint_count"].replace(0, np.nan)
    )
    minute["quality_ok"] = (
        minute["bms_valid_count"].ge(int(minimum_samples_per_minute))
        & minute["bsc_valid_count"].ge(int(minimum_samples_per_minute))
    )
    minute["quality_issue"] = "ok"
    minute.loc[
        minute["bms_valid_count"].lt(int(minimum_samples_per_minute)),
        "quality_issue",
    ] = "insufficient_bms_samples"
    minute.loc[
        minute["bms_valid_count"].ge(int(minimum_samples_per_minute))
        & minute["bsc_valid_count"].lt(int(minimum_samples_per_minute)),
        "quality_issue",
    ] = "insufficient_bsc_samples"
    minute["site_id"] = "M5BAT_Exide1"
    minute = minute.reset_index()
    minute = minute.drop(
        columns=[
            "dc_power_sum_kw", "dc_power_count", "dc_current_sum_a",
            "dc_current_count", "dc_voltage_sum_v", "dc_voltage_count",
            "soc_sum_pct", "soc_count", "ac_power_sum_kw", "ac_power_count",
            "frequency_sum_hz", "frequency_count", "temperature_sum_degC",
            "temperature_count", "setpoint_sum_kw", "setpoint_count",
        ]
    )

    overlap = minute["quality_ok"]
    ac_dc_correlation = (
        float(
            minute.loc[overlap, "measured_ac_power_kw"].corr(
                minute.loc[overlap, "measured_battery_dc_power_kw"]
            )
        )
        if int(overlap.sum()) >= 2
        else float("nan")
    )
    archive = Path(source_archive) if source_archive is not None else None
    codebook = directory / "codebook.csv"
    metadata: dict[str, Any] = {
        "source_class": "official_public_real_grid_scale_bess_measurements",
        "source_name": "M5BAT Large-Scale Battery Storage System Pb1 2017-2025",
        "source_doi": "10.18154/RWTH-2026-06637",
        "license": "CC BY 4.0",
        "source_archive_path": str(archive.resolve()) if archive is not None else None,
        "source_archive_size_bytes": archive.stat().st_size if archive is not None else None,
        "source_archive_sha256": sha256_file(archive) if archive is not None else None,
        "codebook_path": str(codebook.resolve()),
        "codebook_sha256": sha256_file(codebook),
        "annual_bms_files": [path.name for path in bms_paths],
        "annual_bsc_files": [path.name for path in bsc_paths],
        "raw_bms_rows": int(raw_bms_rows),
        "raw_bsc_rows": int(raw_bsc_rows),
        "invalid_values_replaced_with_null": invalid_counts,
        "dc_power_source_unit_resolution": (
            "Published field power_dc_W_bms is numerically stored in kW despite "
            "the W suffix/codebook unit; retained as kW after annual cross-checks "
            "against DC current times terminal voltage and AC-side power."
        ),
        "one_minute_rows": int(len(minute)),
        "quality_ok_rows": int(minute["quality_ok"].sum()),
        "quality_ok_fraction": float(minute["quality_ok"].mean()),
        "start": minute["timestamp"].min().isoformat(),
        "end": minute["timestamp"].max().isoformat(),
        "ac_dc_power_correlation_on_quality_minutes": ac_dc_correlation,
        "target_imputed": False,
        "supports_real_bess_validation": True,
        "supports_real_electrical_load_forecasting": False,
        "supports_target_rig_field_validation": False,
        "evidence_boundary": (
            "Real grid-scale lead-acid BESS field measurements. It validates storage "
            "power/SOC/control behavior but is not a drilling-rig load trace."
        ),
    }
    return minute, metadata
