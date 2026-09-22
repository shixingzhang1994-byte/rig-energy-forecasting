from __future__ import annotations

"""Build a traceable evidence package for all identified public datasets."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import (  # noqa: E402
    aggregate_utah_forge_one_minute,
    audit_three_w,
    sha256_file,
)


DEFAULT_REGISTRY = PROJECT_DIR / "configs/public_real_dataset_registry.yaml"
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts/public_real_multi_dataset_validation"


def _state_summary(frame: pd.DataFrame, dataset: str) -> dict[str, object]:
    states = frame["external_coarse_state"].astype(str)
    runs = states.ne(states.shift()).cumsum()
    durations = states.groupby(runs).size()
    fractions = states.value_counts(normalize=True).to_dict()
    return {
        "dataset": dataset,
        "one_minute_rows": int(len(frame)),
        "state_fractions": {key: float(value) for key, value in fractions.items()},
        "state_run_count": int(len(durations)),
        "dwell_p50_minutes": float(durations.median()),
        "dwell_p95_minutes": float(durations.quantile(0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = registry["datasets"]
    utah_path = PROJECT_DIR / datasets["utah_forge_58_32"]["local_raw_path"]
    utah_processed_path = (
        PROJECT_DIR / datasets["utah_forge_58_32"]["local_processed_path"]
    )
    three_w_root = PROJECT_DIR / datasets["petrobras_3w"]["local_path"]
    energistics_path = PROJECT_DIR / datasets["energistics_well_b"]["local_path"]

    actual_energistics_hash = sha256_file(energistics_path)
    if actual_energistics_hash != datasets["energistics_well_b"]["sha256"]:
        raise RuntimeError("Energistics Well B source hash mismatch")
    if sha256_file(utah_processed_path) != datasets["utah_forge_58_32"][
        "processed_sha256"
    ]:
        raise RuntimeError("Utah FORGE processed source hash mismatch")

    utah_dir = args.output_dir / "utah_forge_58_32"
    utah_dir.mkdir(parents=True, exist_ok=True)
    utah, utah_metadata = aggregate_utah_forge_one_minute(utah_path)
    if utah_metadata["sha256"] != datasets["utah_forge_58_32"]["raw_sha256"]:
        raise RuntimeError("Utah FORGE raw source hash mismatch")
    utah.to_parquet(utah_dir / "process_1min.parquet", index=False)
    utah_metadata["reviewer_concern"] = datasets["utah_forge_58_32"]["reviewer_question"]
    (utah_dir / "metadata.json").write_text(
        json.dumps(utah_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    three_w_dir = args.output_dir / "petrobras_3w"
    three_w_dir.mkdir(parents=True, exist_ok=True)
    inventory, three_w_metadata = audit_three_w(three_w_root)
    inventory.to_csv(three_w_dir / "file_inventory.csv", index=False)
    three_w_metadata["reviewer_concern"] = datasets["petrobras_3w"]["reviewer_question"]
    (three_w_dir / "metadata.json").write_text(
        json.dumps(three_w_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    energistics_audit_path = (
        PROJECT_DIR / "artifacts/v18_public_real_drilling_anchor/archive_audit.json"
    )
    energistics_audit = json.loads(energistics_audit_path.read_text(encoding="utf-8"))
    energistics_process_path = (
        PROJECT_DIR
        / "artifacts/v18_public_real_drilling_anchor/external_validation/real_process_timeseries.csv"
    )
    energistics = pd.read_csv(
        energistics_process_path,
        usecols=["timestamp", "external_coarse_state_1min"],
    )
    energistics["timestamp"] = pd.to_datetime(
        energistics["timestamp"], format="mixed", utc=True
    )
    energistics = (
        energistics.assign(minute=energistics["timestamp"].dt.floor("min"))
        .groupby("minute", as_index=False)["external_coarse_state_1min"]
        .first()
        .rename(
            columns={
                "minute": "timestamp",
                "external_coarse_state_1min": "external_coarse_state",
            }
        )
    )
    state_summaries = [
        _state_summary(energistics, "Energistics-Well-B"),
        _state_summary(utah, "Utah-FORGE-58-32"),
    ]
    (args.output_dir / "cross_dataset_process_summary.json").write_text(
        json.dumps(state_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prior_comparison_path = (
        PROJECT_DIR
        / "artifacts/v18_public_real_drilling_anchor/external_validation/comparison/comparison_summary.json"
    )
    prior_comparison = json.loads(
        prior_comparison_path.read_text(encoding="utf-8")
    )
    real_p50 = [item["dwell_p50_minutes"] for item in state_summaries]
    real_p95 = [item["dwell_p95_minutes"] for item in state_summaries]
    simulator_p50 = float(prior_comparison["simulation_dwell_p50_minutes"])
    simulator_p95 = float(prior_comparison["simulation_dwell_p95_minutes"])
    temporal_assessment = {
        "question": (
            "Do independently sourced real drilling records support the simulated "
            "state-persistence distribution used by the current experimental pipeline?"
        ),
        "real_datasets": [item["dataset"] for item in state_summaries],
        "real_dwell_p50_minutes": real_p50,
        "real_dwell_p95_minutes": real_p95,
        "simulation_dwell_p50_minutes": simulator_p50,
        "simulation_dwell_p95_minutes": simulator_p95,
        "simulation_to_real_median_ratio_p50": simulator_p50
        / float(pd.Series(real_p50).median()),
        "simulation_to_real_median_ratio_p95": simulator_p95
        / float(pd.Series(real_p95).median()),
        "simulator_temporal_state_persistence_supported": False,
        "interpretation": (
            "Both public real datasets show much shorter state runs than the current "
            "simulation. This is an external-validation failure that must be addressed "
            "by real-data-informed temporal calibration or robustness analysis; it is "
            "not evidence of real-site power or controller performance."
        ),
    }
    (args.output_dir / "temporal_external_validity_assessment.json").write_text(
        json.dumps(temporal_assessment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    matrix = pd.DataFrame(
        [
            {
                "dataset": "Energistics-Well-B",
                "downloaded": True,
                "actual_data_analysis": True,
                "real_process": True,
                "real_anomaly_labels": False,
                "measured_total_active_power": False,
                "paper_role": "process/state/transition anchor",
            },
            {
                "dataset": "Utah-FORGE-58-32",
                "downloaded": True,
                "actual_data_analysis": True,
                "real_process": True,
                "real_anomaly_labels": False,
                "measured_total_active_power": False,
                "paper_role": "independent process and proxy anchor",
            },
            {
                "dataset": "Petrobras-3W-2.0.0",
                "downloaded": True,
                "actual_data_analysis": True,
                "real_process": False,
                "real_anomaly_labels": True,
                "measured_total_active_power": False,
                "paper_role": "scope audit; excluded from core efficacy",
            },
            {
                "dataset": "Equinor-Volve",
                "downloaded": False,
                "actual_data_analysis": False,
                "real_process": True,
                "real_anomaly_labels": False,
                "measured_total_active_power": None,
                "paper_role": "blocked pending authenticated access",
            },
        ]
    )
    matrix.to_csv(args.output_dir / "dataset_evidence_matrix.csv", index=False)

    manifest = {
        "protocol_id": registry["protocol"]["id"],
        "status": "partial_auth_blocked",
        "completed_datasets": [
            "Energistics-Well-B",
            "Utah-FORGE-58-32",
            "Petrobras-3W-2.0.0",
        ],
        "blocked_datasets": {
            "Equinor-Volve": datasets["equinor_volve"]["blocking_condition"]
        },
        "source_hashes": {
            "energistics_well_b": actual_energistics_hash,
            "utah_forge_58_32": utah_metadata["sha256"],
            "petrobras_3w_commit": three_w_metadata["repository_commit"],
        },
        "row_evidence": {
            "energistics_total_rows": energistics_audit["total_rows"],
            "utah_raw_rows": utah_metadata["raw_rows"],
            "petrobras_3w_total_rows": three_w_metadata["total_rows"],
            "petrobras_3w_real_rows": three_w_metadata["row_count_by_source"].get(
                "real", 0
            ),
        },
        "core_power_validation_closed": False,
        "simulator_temporal_state_persistence_supported": False,
        "reason": (
            "None of the completed public datasets contains measured total rig-bus "
            "active power synchronized with the required supply-side state."
        ),
        "claim_policy": registry["claim_policy"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
