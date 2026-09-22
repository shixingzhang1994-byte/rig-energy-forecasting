from __future__ import annotations

import sys
from pathlib import Path
import zipfile


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.witsml_external import audit_witsml_archive  # noqa: E402


WITSML = """<?xml version="1.0" encoding="UTF-8"?>
<logs xmlns="http://www.witsml.org/schemas/1series" version="1.4.1.1">
  <log uid="L-1">
    <name>Time log</name><indexType>date time</indexType>
    <startIndex>2026-01-01T00:00:00Z</startIndex>
    <endIndex>2026-01-01T00:00:05Z</endIndex>
    <direction>increasing</direction><indexCurve>TIME</indexCurve>
    <logCurveInfo uid="TIME"><mnemonic>TIME</mnemonic><unit>unitless</unit></logCurveInfo>
    <logCurveInfo uid="RPM"><mnemonic>RPM</mnemonic><unit>rpm</unit>
      <curveDescription>Rotary speed</curveDescription><dataSource>Drilling</dataSource>
    </logCurveInfo>
    <logCurveInfo uid="WOB"><mnemonic>WOB</mnemonic><unit>kN</unit>
      <curveDescription>Weight on bit</curveDescription><dataSource>Drilling</dataSource>
    </logCurveInfo>
    <logCurveInfo uid="BONB"><mnemonic>BONB</mnemonic><unit>unitless</unit>
      <curveDescription>Bit on Bottom</curveDescription><dataSource>Drilling</dataSource>
    </logCurveInfo>
    <logCurveInfo uid="MBOT"><mnemonic>MBOT</mnemonic><unit>unitless</unit>
      <curveDescription>On Bottom Status</curveDescription><dataSource>Drilling</dataSource>
    </logCurveInfo>
    <logData><mnemonicList>TIME,RPM,WOB</mnemonicList><unitList>,rpm,kN</unitList>
      <data>2026-01-01T00:00:00Z,60,100</data>
      <data>2026-01-01T00:00:05Z,61,101</data>
    </logData>
  </log>
</logs>
"""


def test_witsml_audit_keeps_process_and_power_evidence_separate(tmp_path: Path):
    archive_path = tmp_path / "well.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("well/1/log/1/00001.xml", WITSML)

    audit, logs, curves = audit_witsml_archive(
        archive_path, source_url="https://example.invalid/well.zip"
    )

    assert audit.xml_log_fragment_count == 1
    assert audit.logical_log_count == 1
    assert audit.total_rows == 2
    assert audit.time_index_rows == 2
    assert audit.measured_depth_rows == 0
    assert audit.observed_time_span_hours == 5 / 3600
    assert audit.supports_real_operation_validation is True
    assert audit.supports_measured_total_power_validation is False
    assert audit.electrical_power_channels == []
    assert audit.state_reference_channels == ["BONB", "MBOT"]
    assert logs[0].row_count == 2
    assert logs[0].actual_start_index == "2026-01-01T00:00:00Z"
    assert logs[0].actual_end_index == "2026-01-01T00:00:05Z"
    assert {role for curve in curves for role in curve.process_roles} >= {
        "rotation",
        "drilling",
    }


def test_witsml_audit_detects_explicit_kw_channel(tmp_path: Path):
    content = WITSML.replace(
        "</log>",
        "<logCurveInfo uid=\"GEN_KW\"><mnemonic>GEN_KW</mnemonic>"
        "<unit>kW</unit><curveDescription>Generator active power</curveDescription>"
        "</logCurveInfo></log>",
    )
    archive_path = tmp_path / "well.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("well/1/log/1/00001.xml", content)

    audit, _, _ = audit_witsml_archive(
        archive_path, source_url="https://example.invalid/well.zip"
    )
    assert audit.supports_measured_total_power_validation is True
    assert audit.electrical_power_channels == ["GEN_KW"]
