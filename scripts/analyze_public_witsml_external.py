from __future__ import annotations

"""Turn an official public WITSML archive into auditable external evidence.

This tool deliberately produces a process/operating-condition validation set,
not a fabricated total-power label.  Rotary mechanical power is reported only
as a proxy derived from surface torque and rotary speed.
"""

import argparse
import csv
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SELECTED_CURVES = (
    "TIME",
    "TQA",  # average surface torque, kN.m
    "RPM",  # average rotary speed, rpm
    "TFLO",  # mud flow in, L/min
    "SWOB",  # weight on bit, kkgf
    "ROP",  # rate of penetration, m/h
    "BPOS",  # block position, m
    "BONB",
    "MBOT",
    "STIS",
    "RIG_STATE_HSPM",
)
NULL_VALUES = {"", "-999", "-999.0", "-999.25", "-999.2500"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_child_text(node: ET.Element, name: str) -> str:
    for child in node:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> list[str]:
    members = []
    for info in archive.infolist():
        if info.filename.lower().endswith(".xml") and "/log/" in f"/{info.filename}":
            members.append(info.filename)
    return sorted(members)


def _to_float(value: str) -> float:
    value = value.strip()
    if value in NULL_VALUES:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _extract_fragment(archive: zipfile.ZipFile, member: str) -> pd.DataFrame | None:
    with archive.open(member) as stream:
        root = ET.parse(stream).getroot()
    log = next((node for node in root.iter() if _local_name(node.tag) == "log"), None)
    if log is None or _first_child_text(log, "indexType").lower() != "date time":
        return None
    log_uid = log.attrib.get("uid", "")
    log_name = _first_child_text(log, "name")
    curves = [
        _first_child_text(node, "mnemonic")
        for node in log
        if _local_name(node.tag) == "logCurveInfo"
    ]
    index_curve = _first_child_text(log, "indexCurve")
    # WITSML stores the index value as the first CSV field.  The index curve
    # (TIME) is therefore not at its logCurveInfo ordinal in the data row.
    indices = {index_curve: 0}
    non_index_curves = [name for name in curves if name != index_curve]
    indices.update(
        {
            name: index
            for index, name in enumerate(non_index_curves, start=1)
            if name in SELECTED_CURVES
        }
    )
    if index_curve != "TIME":
        return None
    rows: list[dict[str, object]] = []
    log_data = next((node for node in log if _local_name(node.tag) == "logData"), None)
    if log_data is None:
        return None
    for data_node in log_data.iter():
        if _local_name(data_node.tag) != "data":
            continue
        values = (data_node.text or "").split(",")
        row: dict[str, object] = {
            "log_uid": log_uid,
            "log_name": log_name,
            "archive_member": member,
        }
        for name, index in indices.items():
            raw = values[index] if index < len(values) else ""
            row[name] = raw.strip() if name == "TIME" else _to_float(raw)
        rows.append(row)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        frame["TIME"], format="mixed", utc=True, errors="coerce"
    )
    frame = frame.drop(columns=["TIME"]).dropna(subset=["timestamp"])
    frame = frame.loc[frame["timestamp"].dt.year.between(2000, 2100)].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _coarse_state(
    frame: pd.DataFrame,
    *,
    rop_threshold: float = 0.1,
    rpm_threshold: float = 5.0,
    flow_threshold: float = 100.0,
    hoist_velocity_threshold: float = 0.01,
) -> pd.Series:
    """Conservative process-state labels from publicly logged channels."""

    observed = frame[[column for column in ("RPM", "TFLO", "ROP", "BONB", "MBOT", "BPOS") if column in frame]].notna().sum(axis=1)
    rpm = frame.get("RPM", pd.Series(np.nan, index=frame.index)).fillna(0.0)
    flow = frame.get("TFLO", pd.Series(np.nan, index=frame.index)).fillna(0.0)
    rop = frame.get("ROP", pd.Series(np.nan, index=frame.index)).fillna(0.0)
    bonb = frame.get("BONB", pd.Series(np.nan, index=frame.index)).fillna(0.0)
    mbot = frame.get("MBOT", pd.Series(np.nan, index=frame.index)).fillna(0.0)
    drilling = (bonb >= 0.5) | (mbot >= 0.5) | ((rop > rop_threshold) & (rpm > rpm_threshold))
    circulation = (flow > flow_threshold) & ~drilling
    hoisting = pd.Series(False, index=frame.index)
    if "BPOS" in frame:
        time_seconds = frame["timestamp"].astype("int64").diff().div(1e9).replace(0.0, np.nan)
        velocity = frame["BPOS"].diff().abs().div(time_seconds).fillna(0.0)
        hoisting = (velocity > hoist_velocity_threshold) & ~drilling & ~circulation
    state = pd.Series("unknown", index=frame.index, dtype="object")
    state.loc[observed > 0] = "idle_or_other"
    state.loc[hoisting] = "hoisting"
    state.loc[circulation] = "circulation"
    state.loc[drilling] = "drilling"
    return state


def _run_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (log_uid, log_name), group in frame.groupby(["log_uid", "log_name"], sort=False):
        group = group.sort_values("timestamp")
        seconds = group["timestamp"].astype("int64").diff().div(1e9)
        minute_states = (
            group.assign(minute=group["timestamp"].dt.floor("min"))
            .groupby("minute", sort=True)["external_coarse_state_1min"]
            .first()
        )
        durations = []
        previous = None
        start = None
        for index, state in enumerate(minute_states.tolist()):
            if state != previous:
                if previous is not None and start is not None:
                    durations.append(index - start)
                previous, start = state, index
        if previous is not None and start is not None:
            durations.append(len(minute_states) - start)
        proxy = group["rotary_mechanical_power_proxy_kw"].dropna()
        rows.append(
            {
                "log_uid": log_uid,
                "log_name": log_name,
                "rows": len(group),
                "start": group["timestamp"].min().isoformat(),
                "end": group["timestamp"].max().isoformat(),
                "duration_hours": (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600.0,
                "median_sample_seconds": float(np.nanmedian(seconds.iloc[1:])),
                "proxy_observed_rows": int(proxy.size),
                "proxy_p50_kw": float(proxy.quantile(0.50)) if len(proxy) else None,
                "proxy_p95_kw": float(proxy.quantile(0.95)) if len(proxy) else None,
                "proxy_max_kw": float(proxy.max()) if len(proxy) else None,
                "state_fractions": json.dumps(minute_states.value_counts(normalize=True).to_dict(), ensure_ascii=False, sort_keys=True),
                "state_duration_p50_min": float(np.nanmedian(durations)) if durations else None,
                "state_duration_p95_min": float(np.nanpercentile(durations, 95)) if durations else None,
            }
        )
    return pd.DataFrame(rows)


def _state_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (log_uid, log_name), group in frame.groupby(["log_uid", "log_name"], sort=False):
        minute_states = (
            group.assign(minute=group["timestamp"].dt.floor("min"))
            .groupby("minute", sort=True)["external_coarse_state_1min"]
            .first()
        )
        run_ids = minute_states.ne(minute_states.shift()).cumsum()
        for _, run in minute_states.groupby(run_ids):
            state = str(run.iloc[0])
            if state == "unknown":
                continue
            rows.append(
                {
                    "log_uid": log_uid,
                    "log_name": log_name,
                    "state": state,
                    "start_minute": run.index[0].isoformat(),
                    "end_minute": run.index[-1].isoformat(),
                    "duration_minutes": int(len(run)),
                }
            )
    return pd.DataFrame(rows)


def _state_label_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rop_threshold in (0.1, 1.0):
        for rpm_threshold in (5.0, 10.0):
            for flow_threshold in (100.0, 500.0):
                for hoist_velocity_threshold in (0.01, 0.05):
                    labels = _coarse_state(
                        frame,
                        rop_threshold=rop_threshold,
                        rpm_threshold=rpm_threshold,
                        flow_threshold=flow_threshold,
                        hoist_velocity_threshold=hoist_velocity_threshold,
                    )
                    known = labels[labels != "unknown"]
                    fractions = labels.value_counts(normalize=True)
                    known_fractions = known.value_counts(normalize=True)
                    rows.append(
                        {
                            "rop_threshold_m_per_h": rop_threshold,
                            "rpm_threshold": rpm_threshold,
                            "flow_threshold_l_per_min": flow_threshold,
                            "hoist_velocity_threshold_m_per_s": hoist_velocity_threshold,
                            "known_row_coverage": float(len(known) / max(len(labels), 1)),
                            "drilling_fraction_known": float(known_fractions.get("drilling", 0.0)),
                            "circulation_fraction_known": float(known_fractions.get("circulation", 0.0)),
                            "hoisting_fraction_known": float(known_fractions.get("hoisting", 0.0)),
                            "idle_or_other_fraction_known": float(known_fractions.get("idle_or_other", 0.0)),
                            "unknown_fraction_all_rows": float(fractions.get("unknown", 0.0)),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audited real WITSML process evidence")
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(args.archive) as archive:
        for member in _safe_members(archive):
            frame = _extract_fragment(archive, member)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise RuntimeError("archive contains no date-time WITSML log with TIME curve")
    frame = pd.concat(frames, ignore_index=True)
    if {"TQA", "RPM"}.issubset(frame.columns):
        torque = frame["TQA"].clip(lower=0.0)
        rpm = frame["RPM"].clip(lower=0.0)
        frame["rotary_mechanical_power_proxy_kw"] = 2.0 * math.pi * torque * rpm / 60.0
    else:
        frame["rotary_mechanical_power_proxy_kw"] = np.nan
    frame["external_coarse_state"] = _coarse_state(frame)
    # Collapse noisy 2--10 s samples to one-minute modal states before any
    # dwell-time calculation.  This prevents sensor chatter from appearing as
    # implausibly short operating states.
    minute_state = (
        frame.assign(minute=frame["timestamp"].dt.floor("min"))
        .groupby(["log_uid", "minute"], sort=False)["external_coarse_state"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else "unknown")
        .rename("external_coarse_state_1min")
        .reset_index()
    )
    frame["minute"] = frame["timestamp"].dt.floor("min")
    frame = frame.merge(minute_state, on=["log_uid", "minute"], how="left", sort=False)
    frame = frame.drop(columns=["minute"])
    frame.to_csv(args.output_dir / "real_process_timeseries.csv", index=False, encoding="utf-8-sig")
    _run_statistics(frame).to_csv(args.output_dir / "real_process_log_statistics.csv", index=False, encoding="utf-8-sig")
    _state_runs(frame).to_csv(
        args.output_dir / "real_process_state_runs.csv", index=False, encoding="utf-8-sig"
    )
    _state_label_sensitivity(frame).to_csv(
        args.output_dir / "state_label_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "status": "exploratory_external_anchor_before_manuscript_revision",
        "reviewer_concern": "whether simulator operating states and transitions are detached from real drilling operations",
        "experiment_question": "Do publicly released real WITSML process channels support the simulator's process-state and transition plausibility, without pretending to provide total bus power labels?",
        "source_class": "official_public_real_world_witsml",
        "archive_path": str(args.archive.resolve()),
        "archive_sha256": _sha256(args.archive),
        "date_time_rows_extracted": int(len(frame)),
        "logical_logs_extracted": int(frame[["log_uid", "log_name"]].drop_duplicates().shape[0]),
        "selected_curves": [column for column in SELECTED_CURVES if column in frame.columns],
        "selected_curve_observed_rows": {
            column: int(frame[column].notna().sum())
            for column in SELECTED_CURVES
            if column in frame.columns
        },
        "electrical_power_channels_extracted": [],
        "derived_fields": {
            "rotary_mechanical_power_proxy_kw": "2*pi*TQA[kN.m]*RPM[rpm]/60; mechanical rotary proxy only, not total electrical active power",
        "external_coarse_state": "BONB/MBOT/ROP/RPM/TFLO/BPOS rule labels; external process evidence only",
        "unknown_state_policy": "rows with no observed process channel are unknown and excluded from state-fraction comparisons",
        },
        "state_thresholds": {
            "BONB_or_MBOT": ">=0.5",
            "ROP_and_RPM": "ROP>0.1 m/h and RPM>5",
            "circulation": "TFLO>100 L/min when not drilling",
            "hoisting": "absolute block velocity>0.01 m/s when not drilling or circulating",
        },
        "state_threshold_sensitivity_grid": {
            "rop_threshold_m_per_h": [0.1, 1.0],
            "rpm_threshold": [5.0, 10.0],
            "flow_threshold_l_per_min": [100.0, 500.0],
            "hoist_velocity_threshold_m_per_s": [0.01, 0.05],
        },
        "claim_boundary": "supports real process/transition and mechanical-proxy validation; does not validate total_active_power_kw or target-rig SCADA performance",
    }
    (args.output_dir / "external_validation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
