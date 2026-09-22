from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    relative_paths = [
        "configs/v17_user_designated_external_acceptance.yaml",
        "configs/v17_external_dispatch_data_aligned.yaml",
        "configs/v17_target_domain_adaptation.yaml",
        "configs/v17_multivariate_causal_development.yaml",
        "configs/v17_elapsed_substate_development.yaml",
        "src/rig_energy/validation/frozen_v15_external.py",
        "src/rig_energy/validation/multivariate_causal.py",
        "scripts/run_v17_external_acceptance.py",
        "scripts/prepare_v17_target_domain_data.py",
        "scripts/run_v17_multivariate_development.py",
        "scripts/audit_v17_dispatch_critical_load.py",
        "scripts/audit_v17_highs_solver_lock.py",
        "tests/test_frozen_v15_external.py",
        "tests/test_multivariate_causal.py",
        "tests/test_elapsed_substate_features.py",
        "artifacts/v17_user_designated_external_acceptance/locked_input_manifest.json",
        "artifacts/v17_user_designated_external_acceptance/external_data_audit.json",
        "artifacts/v17_user_designated_external_acceptance/acceptance_result.json",
        "artifacts/v17_user_designated_external_acceptance/forecast_metrics.csv",
        "artifacts/v17_user_designated_external_acceptance/risk_metrics.json",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen/original_scenario_selection_failure.json",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/dispatch_metrics.csv",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/scenario_selection.json",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/critical_load_audit/critical_load_audit.json",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/critical_load_audit/critical_load_metrics.csv",
        "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/multi_period_highs_solver_lock.json",
        "artifacts/v17_target_domain_adaptation/data/preparation_audit.json",
        "artifacts/v17_target_domain_adaptation/forecast/benchmark_metrics.csv",
        "artifacts/v17_target_domain_adaptation/forecast/run_metadata.json",
        "artifacts/v17_target_domain_adaptation/risk_development/failure.json",
        "artifacts/v17_multivariate_causal_development/benchmark_metrics.csv",
        "artifacts/v17_multivariate_causal_development/development_result.json",
        "artifacts/v17_elapsed_substate_development_v2/benchmark_metrics.csv",
        "artifacts/v17_elapsed_substate_development_v2/development_result.json",
    ]
    records = []
    for relative in relative_paths:
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "name": "v17-user-designated-data-convergence-evidence",
        "created_after_execution": True,
        "file_count": len(records),
        "records": records,
        "manifest_self_hash_excluded": True,
    }
    output = PROJECT_DIR / "artifacts/v17_evidence_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "file_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
