from __future__ import annotations

"""Build the authoritative V28 data-readiness evidence package.

This script performs read-only validation of prepared Parquet files and writes
only audit artifacts.  It does not train models or inspect reserved test data.
"""

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import sha256_file


PREPARED_DIR = PROJECT_DIR / "artifacts/v28_public_real_validation/prepared"
OUTPUT_DIR = PROJECT_DIR / "artifacts/v28_public_real_validation/data_readiness"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _audit_prepared(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prepared_path = Path(metadata["prepared_path"])
    errors: list[str] = []
    if not prepared_path.is_file():
        return {
            "dataset": metadata_path.name.removesuffix("_preparation_audit.json"),
            "status": "missing",
            "errors": [f"missing prepared file: {prepared_path}"],
        }

    parquet = pq.ParquetFile(prepared_path)
    row_count = int(parquet.metadata.num_rows)
    columns = set(parquet.schema_arrow.names)
    required = {"timestamp", "quality_ok"}
    if not required.issubset(columns):
        errors.append(f"missing required columns: {sorted(required - columns)}")
        frame = pd.DataFrame(columns=["timestamp", "quality_ok"])
    else:
        frame = pd.read_parquet(prepared_path, columns=["timestamp", "quality_ok"])
    timestamps = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    quality = frame.get("quality_ok", pd.Series(dtype=bool)).fillna(False).astype(bool)
    duplicate_timestamps = int(timestamps.duplicated(keep=False).sum())
    missing_timestamps = int(timestamps.isna().sum())
    sorted_timestamps = bool(timestamps.is_monotonic_increasing)
    quality_ok_rows = int(quality.sum())
    if row_count != int(metadata.get("one_minute_rows", row_count)):
        errors.append("row count differs from preparation audit")
    if quality_ok_rows != int(metadata.get("quality_ok_rows", quality_ok_rows)):
        errors.append("quality count differs from preparation audit")
    if duplicate_timestamps:
        errors.append(f"duplicate timestamps: {duplicate_timestamps}")
    if missing_timestamps:
        errors.append(f"missing timestamps: {missing_timestamps}")
    if not sorted_timestamps:
        errors.append("timestamps are not sorted")

    return {
        "dataset": metadata_path.name.removesuffix("_preparation_audit.json"),
        "source_name": metadata.get("source_name"),
        "source_doi": metadata.get("source_doi"),
        "license": metadata.get("license"),
        "status": "complete" if not errors else "failed_audit",
        "prepared_path": _relative(prepared_path),
        "prepared_size_bytes": prepared_path.stat().st_size,
        "prepared_sha256": sha256_file(prepared_path),
        "rows": row_count,
        "quality_ok_rows": quality_ok_rows,
        "quality_ok_fraction": quality_ok_rows / row_count if row_count else 0.0,
        "start": timestamps.min().isoformat() if row_count else None,
        "end": timestamps.max().isoformat() if row_count else None,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_timestamps": missing_timestamps,
        "timestamps_sorted": sorted_timestamps,
        "target_imputed": metadata.get("target_imputed"),
        "supports_real_electrical_load_forecasting": metadata.get(
            "supports_real_electrical_load_forecasting"
        ),
        "supports_real_bess_validation": metadata.get("supports_real_bess_validation"),
        "supports_target_rig_field_validation": metadata.get(
            "supports_target_rig_field_validation"
        ),
        "evidence_boundary": metadata.get("evidence_boundary"),
        "preparation_audit_path": _relative(metadata_path),
        "preparation_audit_sha256": sha256_file(metadata_path),
        "errors": errors,
    }


def _verify_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    if path.is_file():
        record["sha256"] = sha256_file(path)
        record["hash_matches_registry"] = (
            record["sha256"] == expected_sha256 if expected_sha256 else None
        )
    return record


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_records = [
        _audit_prepared(path)
        for path in sorted(PREPARED_DIR.glob("*_preparation_audit.json"))
    ]

    process_records = [
        {
            "dataset": "energistics_well_b",
            "role": "real drilling process/transition anchor",
            "status": "complete",
            "raw": _verify_file(
                PROJECT_DIR
                / "data/public_external/energistics_well_b/NA-NA-EnergisticsWell2016-B.zip",
                "e68526fcfb26e23506e0be5fd1f69b585b5c6341cb3db0a359bbe160ca4a8904",
            ),
            "supports_target_rig_field_validation": False,
        },
        {
            "dataset": "utah_forge_58_32",
            "role": "independent real drilling process/mechanical-hydraulic anchor",
            "status": "complete",
            "raw": _verify_file(
                PROJECT_DIR
                / "data/public_external/utah_forge_58_32/Well_58-32_raw_pason_log.csv",
                "5af0241d325b9cf5b4d6a9d524218cb2cbc3694fa324816cfb879f1f556161c2",
            ),
            "prepared": _verify_file(
                PROJECT_DIR
                / "artifacts/public_real_multi_dataset_validation/utah_forge_58_32/process_1min.parquet"
            ),
            "supports_target_rig_field_validation": False,
        },
        {
            "dataset": "petrobras_3w",
            "role": "real production-well anomaly scope audit only",
            "status": "complete_scope_audit",
            "inventory": _verify_file(
                PROJECT_DIR
                / "artifacts/public_real_multi_dataset_validation/petrobras_3w/file_inventory.csv"
            ),
            "supports_target_rig_field_validation": False,
        },
    ]

    required_prepared = {
        "opencem",
        "refit_house1",
        "refit_house2",
        "refit_house3",
        "refit_house4",
        "refit_house5",
        "refit_house11",
        "uci_household",
        "tsukuba_microgrid",
        "m5bat_exide1",
    }
    by_name = {record["dataset"]: record for record in prepared_records}
    missing_required = sorted(required_prepared - by_name.keys())
    failed_required = sorted(
        name
        for name in required_prepared & by_name.keys()
        if by_name[name]["status"] != "complete"
    )
    failed_process = sorted(
        record["dataset"]
        for record in process_records
        if record["status"] not in {"complete", "complete_scope_audit"}
        or any(
            value.get("exists") is False
            for value in record.values()
            if isinstance(value, dict) and "exists" in value
        )
    )
    complete = not missing_required and not failed_required and not failed_process

    manifest = {
        "protocol_id": "V28-PUBLIC-REAL-DATA-READINESS-20260921",
        "dataset_stage_complete": complete,
        "completion_definition": (
            "All preselected anonymous public sources are downloaded from official "
            "repositories, integrity checked, adapted, quality audited, and registered."
        ),
        "prepared_datasets": prepared_records,
        "drilling_process_datasets": process_records,
        "missing_required": missing_required,
        "failed_required": failed_required,
        "failed_process": failed_process,
        "excluded_or_blocked_sources": [
            {
                "dataset": "equinor_volve",
                "reason": "requires an authenticated institutional Databricks account",
            },
            {
                "dataset": "mesa_del_sol",
                "reason": "Dryad metadata is public but current download endpoint rejects this network",
            },
            {
                "dataset": "spark",
                "reason": (
                    "curated/raw archives are 76-246 GB; excluded from the frozen core "
                    "because OpenCEM, Tsukuba, REFIT, UCI, and M5BAT already cover the "
                    "predeclared load/microgrid/BESS evidence roles"
                ),
            },
            {
                "dataset": "tsukuba_nims_mirror",
                "reason": (
                    "publisher mirror is truncated despite matching its advertised MD5; "
                    "replaced by the intact official Figshare versioned archive"
                ),
            },
        ],
        "claim_boundary": (
            "No selected public source is direct target-rig active-power telemetry; "
            "target-rig field validation therefore remains unavailable."
        ),
    }
    manifest_path = OUTPUT_DIR / "dataset_readiness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = OUTPUT_DIR / "prepared_dataset_quality.csv"
    fields = [
        "dataset",
        "status",
        "rows",
        "quality_ok_rows",
        "quality_ok_fraction",
        "start",
        "end",
        "prepared_size_bytes",
        "prepared_sha256",
        "target_imputed",
        "supports_real_electrical_load_forecasting",
        "supports_real_bess_validation",
        "supports_target_rig_field_validation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prepared_records)

    readme_path = OUTPUT_DIR / "DATASET_READINESS.md"
    table_rows = "\n".join(
        "| {dataset} | {status} | {rows:,} | {quality_ok_fraction:.2%} | {target_imputed} |".format(
            **record
        )
        for record in prepared_records
    )
    readme_path.write_text(
        "# V28 public real-data readiness\n\n"
        f"Dataset stage complete: **{str(complete).lower()}**\n\n"
        "| Prepared series | Status | Minute rows | Quality OK | Target imputed |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{table_rows}\n\n"
        "The selected suite also retains Energistics Well B and Utah FORGE as real "
        "drilling-process anchors, and Petrobras 3W only as a scope audit. None is "
        "direct target-rig active-power telemetry.\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset_stage_complete": complete, "manifest": str(manifest_path), "prepared_count": len(prepared_records), "missing_required": missing_required, "failed_required": failed_required}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
