from __future__ import annotations

"""Freeze V20 code, candidate space, and source queue before development outcomes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_freeze.json"
FILES = [
    "configs/v20_evidential_calibration_protocol.yaml",
    "configs/v20_evidential_calibration_development_source.yaml",
    "src/rig_energy/risk/evidence_theory_v20.py",
    "scripts/run_v20_evidential_calibration.py",
    "scripts/search_v20_conflict_calibration.py",
    "scripts/freeze_v20_development.py",
    "tests/test_evidence_theory_v20.py",
    "tests/test_v20_protocol.py",
    "artifacts/v19_evidential_risk/risk_candidate_freeze.json",
    "artifacts/v15_generator_first_reserve/freeze_manifest.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for relative in FILES:
        path = PROJECT_DIR / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(
            {"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        )
    stable = {
        "protocol_id": "V20-CONFLICT-CALIBRATION-20260909",
        "status": "frozen_before_any_v20_development_outcome",
        "files": files,
        "forbidden_result_namespaces_at_freeze": [
            "artifacts/v20_evidential_calibration/development_source/seed_*",
            "artifacts/v20_evidential_calibration/development_search",
            "artifacts/v20_evidential_calibration/prospective",
        ],
    }
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        comparable = {key: existing[key] for key in stable}
        if comparable != stable:
            raise RuntimeError("existing V20 development freeze differs from current files")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return
    result_roots = [
        PROJECT_DIR / "artifacts/v20_evidential_calibration/development_source",
        PROJECT_DIR / "artifacts/v20_evidential_calibration/development_search",
        PROJECT_DIR / "artifacts/v20_evidential_calibration/prospective",
    ]
    if any(path.exists() and any(path.iterdir()) for path in result_roots):
        raise RuntimeError("V20 result artifacts already exist; cannot assert pre-outcome freeze")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {**stable, "created_utc": datetime.now(timezone.utc).isoformat()}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
