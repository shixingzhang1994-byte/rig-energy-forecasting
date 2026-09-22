from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from datetime import datetime
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile


PROCESS_CHANNEL_RULES = {
    "rotation": ("rpm", "torque", "rotary"),
    "hoisting": ("hookload", "block position", "slips"),
    "circulation": ("pump", "stand pipe pressure", "standpipe pressure", "flow"),
    "drilling": ("weight on bit", "wob", "rate of penetration", "rop", "on bottom"),
}
ELECTRICAL_POWER_UNITS = {"w", "kw", "mw", "kva", "kvar"}
ELECTRICAL_POWER_PHRASES = (
    "active power",
    "electric power",
    "electrical power",
    "generator power",
    "generator load",
    "power demand",
)
ELECTRICAL_POWER_MNEMONICS = {
    "KW",
    "MW",
    "PWR",
    "POWER",
    "ACTPWR",
    "ACTIVE_POWER",
    "GEN_KW",
    "GENERATOR_KW",
}
STATE_REFERENCE_MNEMONICS = {"RIG_STATE_HSPM", "BONB", "MBOT", "STIS"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children_by_name(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_name(child.tag) == name)


def _first_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text.strip() if child.text else None
    return None


def _normalized_channel_text(curve: "WitsmlCurve") -> str:
    return " ".join(
        part for part in (curve.mnemonic, curve.description, curve.data_source) if part
    ).lower()


def _process_roles(curve: "WitsmlCurve") -> list[str]:
    text = _normalized_channel_text(curve)
    return sorted(
        role
        for role, terms in PROCESS_CHANNEL_RULES.items()
        if any(term in text for term in terms)
    )


def _is_electrical_power(curve: "WitsmlCurve") -> bool:
    mnemonic = re.sub(r"[^A-Z0-9_]+", "_", curve.mnemonic.upper()).strip("_")
    unit = curve.unit.strip().lower()
    description = (curve.description or "").lower()
    return (
        unit in ELECTRICAL_POWER_UNITS
        or mnemonic in ELECTRICAL_POWER_MNEMONICS
        or any(phrase in description for phrase in ELECTRICAL_POWER_PHRASES)
    )


@dataclass(frozen=True)
class WitsmlCurve:
    archive_member: str
    log_uid: str
    log_name: str
    index_type: str
    mnemonic: str
    unit: str
    description: str
    data_source: str
    process_roles: list[str]
    is_electrical_power: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitsmlLog:
    archive_member: str
    log_uid: str
    log_name: str
    index_type: str
    index_curve: str
    start_index: str
    end_index: str
    actual_start_index: str
    actual_end_index: str
    direction: str
    row_count: int
    curve_count: int
    process_roles: list[str]
    electrical_power_channels: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitsmlArchiveAudit:
    archive_path: str
    archive_sha256: str
    source_class: str
    source_url: str
    source_standard: str
    xml_log_fragment_count: int
    logical_log_count: int
    total_rows: int
    time_index_rows: int
    measured_depth_rows: int
    observed_time_start: str
    observed_time_end: str
    observed_time_span_hours: float | None
    unique_curve_count: int
    index_types: list[str]
    process_roles: list[str]
    state_reference_channels: list[str]
    electrical_power_channels: list[str]
    supports_real_operation_validation: bool
    supports_measured_total_power_validation: bool
    evidence_boundary: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_xml_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"ZIP 中存在不安全路径: {info.filename}")
        if "/log/" in f"/{info.filename}" and info.filename.lower().endswith(".xml"):
            members.append(info.filename)
    return sorted(members)


def _parse_log_fragment(
    archive: zipfile.ZipFile, member: str
) -> tuple[WitsmlLog, list[WitsmlCurve]]:
    with archive.open(member) as stream:
        root = ET.parse(stream).getroot()
    log = next((node for node in root.iter() if _local_name(node.tag) == "log"), None)
    if log is None:
        raise ValueError(f"WITSML XML 不含 log 对象: {member}")

    uid = log.attrib.get("uid", "")
    name = _first_text(log, "name") or ""
    index_type = _first_text(log, "indexType") or ""
    index_curve = _first_text(log, "indexCurve") or ""
    start_index = _first_text(log, "startIndex") or ""
    end_index = _first_text(log, "endIndex") or ""
    direction = _first_text(log, "direction") or ""

    curves: list[WitsmlCurve] = []
    for curve_node in _children_by_name(log, "logCurveInfo"):
        curve = WitsmlCurve(
            archive_member=member,
            log_uid=uid,
            log_name=name,
            index_type=index_type,
            mnemonic=_first_text(curve_node, "mnemonic") or "",
            unit=_first_text(curve_node, "unit") or "",
            description=_first_text(curve_node, "curveDescription") or "",
            data_source=_first_text(curve_node, "dataSource") or "",
            process_roles=[],
            is_electrical_power=False,
        )
        curve = WitsmlCurve(
            **{
                **curve.as_dict(),
                "process_roles": _process_roles(curve),
                "is_electrical_power": _is_electrical_power(curve),
            }
        )
        curves.append(curve)

    row_count = 0
    actual_start_index = ""
    actual_end_index = ""
    for node in log.iter():
        if _local_name(node.tag) != "data":
            continue
        row_count += 1
        value = (node.text or "").split(",", 1)[0].strip()
        if not actual_start_index:
            actual_start_index = value
        actual_end_index = value
    roles = sorted({role for curve in curves for role in curve.process_roles})
    electrical = sorted(
        {curve.mnemonic for curve in curves if curve.is_electrical_power}
    )
    return (
        WitsmlLog(
            archive_member=member,
            log_uid=uid,
            log_name=name,
            index_type=index_type,
            index_curve=index_curve,
            start_index=start_index,
            end_index=end_index,
            actual_start_index=actual_start_index,
            actual_end_index=actual_end_index,
            direction=direction,
            row_count=row_count,
            curve_count=len(curves),
            process_roles=roles,
            electrical_power_channels=electrical,
        ),
        curves,
    )


def audit_witsml_archive(
    path: Path,
    *,
    source_url: str,
    source_standard: str = "WITSML 1.4.1.1",
) -> tuple[WitsmlArchiveAudit, list[WitsmlLog], list[WitsmlCurve]]:
    """Audit a WITSML ZIP without extracting it or upgrading its evidence class."""

    path = Path(path)
    logs: list[WitsmlLog] = []
    curves: list[WitsmlCurve] = []
    with zipfile.ZipFile(path) as archive:
        for member in _safe_xml_members(archive):
            parsed_log, parsed_curves = _parse_log_fragment(archive, member)
            logs.append(parsed_log)
            curves.extend(parsed_curves)

    unique_curves = {
        (curve.log_uid, curve.mnemonic, curve.unit, curve.description) for curve in curves
    }
    roles = sorted({role for curve in curves for role in curve.process_roles})
    electrical = sorted(
        {curve.mnemonic for curve in curves if curve.is_electrical_power}
    )
    state_reference = sorted(
        {
            curve.mnemonic
            for curve in curves
            if curve.mnemonic.upper() in STATE_REFERENCE_MNEMONICS
        }
    )
    logical_logs = {(log.log_uid, log.log_name, log.index_type) for log in logs}
    supports_operation = bool(roles)
    supports_power = bool(electrical)
    time_logs = [log for log in logs if log.index_type.lower() == "date time"]
    depth_logs = [log for log in logs if log.index_type.lower() == "measured depth"]
    observed_times: list[datetime] = []
    for log in time_logs:
        for value in (log.actual_start_index, log.actual_end_index):
            try:
                observed_times.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
            except ValueError:
                continue
    observed_start = min(observed_times) if observed_times else None
    observed_end = max(observed_times) if observed_times else None
    boundary = (
        "Official public real-world WITSML drilling-process evidence; usable for "
        "channel, operating-condition, and transition-method validation. "
        "No measured electrical-power channel was found, so it cannot validate "
        "total active-power prediction or target-rig SCADA performance."
        if not supports_power
        else "Official public real-world WITSML evidence containing conservatively "
        "detected electrical-power channels; channel semantics still require manual review."
    )
    audit = WitsmlArchiveAudit(
        archive_path=str(path.resolve()),
        archive_sha256=_sha256(path),
        source_class="official_public_real_world_witsml",
        source_url=source_url,
        source_standard=source_standard,
        xml_log_fragment_count=len(logs),
        logical_log_count=len(logical_logs),
        total_rows=sum(log.row_count for log in logs),
        time_index_rows=sum(log.row_count for log in time_logs),
        measured_depth_rows=sum(log.row_count for log in depth_logs),
        observed_time_start=observed_start.isoformat() if observed_start else "",
        observed_time_end=observed_end.isoformat() if observed_end else "",
        observed_time_span_hours=(
            (observed_end - observed_start).total_seconds() / 3600.0
            if observed_start and observed_end
            else None
        ),
        unique_curve_count=len(unique_curves),
        index_types=sorted({log.index_type for log in logs if log.index_type}),
        process_roles=roles,
        state_reference_channels=state_reference,
        electrical_power_channels=electrical,
        supports_real_operation_validation=supports_operation,
        supports_measured_total_power_validation=supports_power,
        evidence_boundary=boundary,
    )
    return audit, logs, curves
