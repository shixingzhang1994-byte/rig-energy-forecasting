from __future__ import annotations

"""Auditable adapters for public real-world drilling and well datasets.

The adapters intentionally keep process evidence, anomaly evidence, and
electrical-power evidence separate.  In particular, a mechanically derived
proxy is never renamed to measured active power.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import hashlib
import re

import numpy as np
import pandas as pd


UTAH_REQUIRED_COLUMNS = {
    "Date",
    "Time",
    "Depth (m)",
    "Block Ht (m)",
    "Bit Depth (m)",
    "ROP(1 m)",
    "Hookload (kg)",
    "Pump Press (KPa)",
    "Surface Torque (KPa)",
    "Rotary Speed (rpm)",
    "Flow In (liters/min)",
}

THREE_W_ELECTRICAL_NAMES = {
    "kw",
    "mw",
    "active_power",
    "electrical_power",
    "generator_power",
    "generator_kw",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_sentinels(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.mask(values <= -9000.0)


def _utah_chunk(frame: pd.DataFrame, previous: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(UTAH_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Utah FORGE CSV missing required columns: {missing}")

    numeric = [column for column in frame.columns if column not in {"Date", "Time"}]
    _replace_sentinels(frame, numeric)
    timestamp = pd.to_datetime(
        frame["Date"].astype(str).str.strip()
        + " "
        + frame["Time"].astype(str).str.strip(),
        format="%m/%d/%y %H:%M:%S",
        errors="coerce",
        utc=True,
    )
    frame = frame.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    if frame.empty:
        return frame

    seconds = frame["timestamp"].astype("int64").diff().div(1e9)
    block_delta = frame["Block Ht (m)"].diff()
    if previous:
        first = frame.index[0]
        seconds.loc[first] = (
            frame.loc[first, "timestamp"] - previous["timestamp"]
        ).total_seconds()
        block_delta.loc[first] = frame.loc[first, "Block Ht (m)"] - previous["block_m"]
    velocity = block_delta.div(seconds.where(seconds > 0.0)).replace([np.inf, -np.inf], np.nan)
    frame["block_velocity_m_per_s"] = velocity
    frame["pump_hydraulic_power_proxy_kw"] = (
        frame["Pump Press (KPa)"].clip(lower=0.0)
        * frame["Flow In (liters/min)"].clip(lower=0.0)
        / 60000.0
    )
    frame["hoist_mechanical_power_proxy_kw"] = (
        frame["Hookload (kg)"].clip(lower=0.0)
        * 9.80665
        * frame["block_velocity_m_per_s"].abs()
        / 1000.0
    )
    frame["rotary_uncalibrated_intensity"] = (
        frame["Surface Torque (KPa)"].clip(lower=0.0)
        * frame["Rotary Speed (rpm)"].clip(lower=0.0)
    )

    rpm = frame["Rotary Speed (rpm)"].fillna(0.0)
    rop = frame["ROP(1 m)"].fillna(0.0)
    flow = frame["Flow In (liters/min)"].fillna(0.0)
    pump_spm = frame.get("Pump 1 (spm)", pd.Series(0.0, index=frame.index)).fillna(0.0)
    pump_spm += frame.get("Pump 2 (spm)", pd.Series(0.0, index=frame.index)).fillna(0.0)
    drilling = (rop > 0.1) & (rpm > 5.0)
    circulation = ~drilling & ((flow > 100.0) | (pump_spm > 5.0))
    hoisting = ~drilling & ~circulation & (velocity.abs().fillna(0.0) > 0.01)
    state = pd.Series("idle_or_other", index=frame.index, dtype="object")
    state.loc[hoisting] = "hoisting"
    state.loc[circulation] = "circulation"
    state.loc[drilling] = "drilling"
    frame["external_coarse_state"] = state
    return frame


def aggregate_utah_forge_one_minute(
    path: Path,
    *,
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream the 1 Hz FORGE file and return auditable one-minute process data."""

    path = Path(path)
    minute_parts: list[pd.DataFrame] = []
    state_parts: list[pd.DataFrame] = []
    total_rows = 0
    valid_time_rows = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    sample_seconds: list[float] = []
    previous: dict[str, Any] = {}
    columns: list[str] | None = None

    for raw in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(raw)
        columns = list(raw.columns)
        frame = _utah_chunk(raw, previous)
        if frame.empty:
            continue
        valid_time_rows += len(frame)
        chunk_start = frame["timestamp"].min()
        chunk_end = frame["timestamp"].max()
        start = chunk_start if start is None else min(start, chunk_start)
        end = chunk_end if end is None else max(end, chunk_end)
        delta = frame["timestamp"].astype("int64").diff().div(1e9)
        sample_seconds.extend(delta[(delta > 0.0) & (delta <= 60.0)].head(5000).tolist())
        previous = {
            "timestamp": frame["timestamp"].iloc[-1],
            "block_m": frame["Block Ht (m)"].iloc[-1],
        }
        frame["minute"] = frame["timestamp"].dt.floor("min")
        numeric_map = {
            "row_count": ("timestamp", "size"),
            "depth_m_mean": ("Depth (m)", "mean"),
            "bit_depth_m_mean": ("Bit Depth (m)", "mean"),
            "rop_m_per_h_mean": ("ROP(1 m)", "mean"),
            "rpm_mean": ("Rotary Speed (rpm)", "mean"),
            "flow_l_per_min_mean": ("Flow In (liters/min)", "mean"),
            "pump_proxy_kw_mean": ("pump_hydraulic_power_proxy_kw", "mean"),
            "pump_proxy_kw_max": ("pump_hydraulic_power_proxy_kw", "max"),
            "hoist_proxy_kw_mean": ("hoist_mechanical_power_proxy_kw", "mean"),
            "hoist_proxy_kw_max": ("hoist_mechanical_power_proxy_kw", "max"),
            "rotary_intensity_mean": ("rotary_uncalibrated_intensity", "mean"),
        }
        minute_parts.append(frame.groupby("minute", sort=False).agg(**numeric_map).reset_index())
        counts = (
            frame.groupby(["minute", "external_coarse_state"], sort=False)
            .size()
            .rename("count")
            .reset_index()
        )
        state_parts.append(counts)

    if not minute_parts:
        raise ValueError(f"Utah FORGE CSV contains no valid timestamp rows: {path}")
    numeric = pd.concat(minute_parts, ignore_index=True)
    weighted_columns = [
        column for column in numeric.columns if column.endswith("_mean")
    ]
    for column in weighted_columns:
        numeric[f"_{column}_weighted"] = numeric[column] * numeric["row_count"]
    aggregation: dict[str, str] = {"row_count": "sum"}
    aggregation.update({f"_{column}_weighted": "sum" for column in weighted_columns})
    aggregation.update(
        {column: "max" for column in numeric.columns if column.endswith("_max")}
    )
    numeric = numeric.groupby("minute", as_index=False).agg(aggregation)
    for column in weighted_columns:
        numeric[column] = numeric.pop(f"_{column}_weighted") / numeric["row_count"]

    state_counts = (
        pd.concat(state_parts, ignore_index=True)
        .groupby(["minute", "external_coarse_state"], as_index=False)["count"]
        .sum()
    )
    state = state_counts.loc[
        state_counts.groupby("minute")["count"].idxmax(),
        ["minute", "external_coarse_state"],
    ]
    canonical = numeric.merge(state, on="minute", validate="one_to_one")
    canonical = canonical.rename(columns={"minute": "timestamp"}).sort_values("timestamp")

    metadata = {
        "source_class": "official_public_real_world_drilling_process",
        "source_name": "Utah FORGE Well 58-32 raw Pason log",
        "source_doi": "10.15121/1495411",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "raw_rows": int(total_rows),
        "valid_timestamp_rows": int(valid_time_rows),
        "one_minute_rows": int(len(canonical)),
        "columns": columns or [],
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
        "median_sample_seconds": float(np.median(sample_seconds)) if sample_seconds else None,
        "electrical_power_channels": [],
        "derived_proxies": {
            "pump_hydraulic_power_proxy_kw": "pressure[kPa]*flow[L/min]/60000; hydraulic output only",
            "hoist_mechanical_power_proxy_kw": "hookload[kg]*g*abs(block_velocity[m/s])/1000",
            "rotary_uncalibrated_intensity": "surface torque instrument pressure[kPa]*rpm; not kW",
        },
        "supports_real_operation_validation": True,
        "supports_measured_total_power_validation": False,
        "evidence_boundary": (
            "Real 1 Hz drilling-process and mechanical/hydraulic proxy evidence; "
            "no measured rig-bus electrical active-power channel."
        ),
    }
    return canonical.reset_index(drop=True), metadata


def _three_w_source(filename: str) -> str:
    upper = filename.upper()
    if upper.startswith("WELL-"):
        return "real"
    if upper.startswith("SIMULATED"):
        return "simulated"
    if upper.startswith("DRAWN") or upper.startswith("HAND"):
        return "hand_drawn"
    return "unknown"


def audit_three_w(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit the Petrobras 3W corpus without treating it as drilling power data."""

    import pyarrow.parquet as pq

    root = Path(root)
    files = sorted(root.glob("[0-9]/*.parquet"))
    if not files:
        raise ValueError(f"No 3W parquet files found under {root}")
    rows: list[dict[str, Any]] = []
    schema_names: set[str] = set()
    wells: set[str] = set()
    source_counts: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    event_rows: Counter[str] = Counter()
    total_bytes = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        schema_names.update(names)
        source = _three_w_source(path.name)
        event_type = path.parent.name
        num_rows = int(parquet.metadata.num_rows)
        match = re.match(r"(WELL-\d+)", path.name.upper())
        if match:
            wells.add(match.group(1))
        source_counts[source] += 1
        source_rows[source] += num_rows
        event_rows[event_type] += num_rows
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "event_type": int(event_type),
                "source": source,
                "path": str(path.relative_to(root)),
                "rows": num_rows,
                "size_bytes": size,
                "has_class": "class" in names,
                "has_state": "state" in names,
            }
        )
    normalized = {re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") for name in schema_names}
    electrical = sorted(normalized & THREE_W_ELECTRICAL_NAMES)
    inventory = pd.DataFrame(rows).sort_values(["event_type", "source", "path"])
    metadata = {
        "source_class": "official_public_real_and_synthetic_production_well_events",
        "source_name": "Petrobras 3W Dataset",
        "dataset_version": "2.0.0",
        "repository_commit": _git_head(root.parent),
        "license": "CC BY 4.0 for dataset files",
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_rows": int(sum(source_rows.values())),
        "total_bytes": int(total_bytes),
        "file_count_by_source": dict(sorted(source_counts.items())),
        "row_count_by_source": dict(sorted(source_rows.items())),
        "row_count_by_event_type": dict(sorted(event_rows.items())),
        "real_well_count": len(wells),
        "schema_columns": sorted(schema_names),
        "electrical_power_channels": electrical,
        "supports_real_well_event_validation": source_counts["real"] > 0,
        "supports_drilling_process_validation": False,
        "supports_measured_total_power_validation": bool(electrical),
        "eligible_role": "external anomaly-dataset provenance and scope audit only",
        "evidence_boundary": (
            "Production-well pressure/temperature/flow event data; not rig drilling-process "
            "data, not power-system risk labels, and not rig-bus active power."
        ),
    }
    return inventory.reset_index(drop=True), metadata


def _git_head(path: Path) -> str | None:
    head = path / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = path / ".git" / value.removeprefix("ref: ")
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = path / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(value.removeprefix("ref: ")):
                    return line.split()[0]
        return None
    return value
