from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
OUTPUT = PROJECT_DIR / "artifacts/v18_convergence_manifest.json"

FILES = [
    PROJECT_DIR / "README.md",
    PROJECT_DIR / "configs/v18_evidence_triage_convergence.yaml",
    PROJECT_DIR / "src/rig_energy/data/witsml_external.py",
    PROJECT_DIR / "scripts/audit_energistics_witsml.py",
    PROJECT_DIR / "scripts/build_v18_convergence_manifest.py",
    PROJECT_DIR / "tests/test_witsml_external.py",
    PROJECT_DIR / "data/public_external/energistics_well_b/NA-NA-EnergisticsWell2016-B.zip",
    PROJECT_DIR / "artifacts/v18_public_real_drilling_anchor/archive_audit.json",
    PROJECT_DIR / "artifacts/v18_public_real_drilling_anchor/log_inventory.csv",
    PROJECT_DIR / "artifacts/v18_public_real_drilling_anchor/channel_inventory.csv",
    PROJECT_DIR / "artifacts/v17_target_domain_adaptation/forecast/models/itransformer.pt",
    PROJECT_DIR / "artifacts/v17_target_domain_adaptation/forecast/benchmark_metrics.csv",
    PROJECT_DIR / "artifacts/v17_target_domain_adaptation/forecast/dispatch_uncertainty_envelope.json",
    PROJECT_DIR / "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/multi_period_highs_solver_lock.json",
    PROJECT_DIR / "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/critical_load_audit/critical_load_audit.json",
    PROJECT_DIR / "artifacts/v17_evidence_manifest.json",
    WORKSPACE_DIR / "现场采集困难下的项目收敛方案.md",
    WORKSPACE_DIR / "青年专项_最终研发状态.md",
]

EXPECTED_LOCKS = {
    "rig-energy-forecasting/data/public_external/energistics_well_b/NA-NA-EnergisticsWell2016-B.zip": "e68526fcfb26e23506e0be5fd1f69b585b5c6341cb3db0a359bbe160ca4a8904",
    "rig-energy-forecasting/artifacts/v17_target_domain_adaptation/forecast/models/itransformer.pt": "4c83b9d37ac4bb8b534876a26f2999833b4f595654827a4d7fcb7a78a6c53392",
    "rig-energy-forecasting/artifacts/v17_target_domain_adaptation/forecast/dispatch_uncertainty_envelope.json": "b374c83988d2a2b6ec0195136ec6036e861641744c5f264a8486da465cdc6993",
    "rig-energy-forecasting/artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/multi_period_highs_solver_lock.json": "bf64ca36927d51d93e76556bf5af0be2e8ee9cdb768030ab49cf66002c13aec6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("V18证据文件缺失: " + ", ".join(missing))

    items = []
    for path in FILES:
        relative = str(path.relative_to(WORKSPACE_DIR))
        digest = _sha256(path)
        items.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "expected_lock": EXPECTED_LOCKS.get(relative),
                "lock_ok": (
                    digest == EXPECTED_LOCKS[relative]
                    if relative in EXPECTED_LOCKS
                    else None
                ),
            }
        )

    manifest = {
        "version": "V18",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_file_count": len(items),
        "all_files_exist": True,
        "all_expected_locks_ok": all(
            item["lock_ok"] is not False for item in items
        ),
        "claim_boundary": (
            "Technical convergence for method-and-simulation evidence; public real "
            "WITSML anchors drilling operations but does not prove measured rig-bus power."
        ),
        "files": items,
    }
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
