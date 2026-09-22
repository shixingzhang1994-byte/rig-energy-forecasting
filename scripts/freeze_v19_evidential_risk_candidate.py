from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_DIR / "configs/v19_evidential_risk_development.yaml"
SEARCH_ROOT = PROJECT_DIR / "artifacts/v19_evidential_risk/development_search"
OUTPUT = PROJECT_DIR / "artifacts/v19_evidential_risk/risk_candidate_freeze.json"
FILES = [
    PROTOCOL,
    SEARCH_ROOT / "selected_evidential_config.yaml",
    SEARCH_ROOT / "selection_summary.json",
    PROJECT_DIR / "src/rig_energy/risk/evidence_theory.py",
    PROJECT_DIR / "scripts/run_evidential_risk_fusion.py",
    PROJECT_DIR / "tests/test_evidence_theory.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"freeze already exists: {OUTPUT}")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    selected = yaml.safe_load(
        (SEARCH_ROOT / "selected_evidential_config.yaml").read_text(encoding="utf-8")
    )
    payload = {
        "protocol_id": protocol["protocol"]["id"],
        "status": "risk_candidate_frozen_before_external_diagnostic_and_prospective_holdouts",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate_index": selected["candidate_index"],
        "target_v17_labels_used_for_selection": False,
        "prospective_holdout_labels_used_for_selection": False,
        "files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in FILES
        ],
        "parent_v15_freeze": "artifacts/v15_generator_first_reserve/freeze_manifest.json",
        "parent_v15_freeze_sha256": _sha256(
            PROJECT_DIR / "artifacts/v15_generator_first_reserve/freeze_manifest.json"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": len(FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
