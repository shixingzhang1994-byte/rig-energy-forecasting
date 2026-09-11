from __future__ import annotations

"""Record a pre-controller upstream risk-validation failure without changing V21."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED = 20261043
ROOT = PROJECT_DIR / f"artifacts/v21_probability_calibration/prospective_source/seed_{SEED}"
ELIGIBILITY = ROOT / "eligibility.json"
AMENDMENT = PROJECT_DIR / "artifacts/v21_probability_calibration/execution_clarification_1.json"
RUN_FREEZE = PROJECT_DIR / "artifacts/v21_probability_calibration/run_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if ELIGIBILITY.exists():
        raise RuntimeError("eligibility record already exists")
    forbidden = [
        ROOT / "result.json",
        ROOT / "dispatch",
        PROJECT_DIR / f"artifacts/v21_probability_calibration/prospective/seed_{SEED}",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("controller or V21 candidate output exists; cannot record upstream failure")
    required = [
        ROOT / "source/data/audit/calibration_audit.json",
        ROOT / "source/forecast/benchmark_metrics.csv",
        ROOT / "source/resolved_risk_config.yaml",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    audit = json.loads(required[0].read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise RuntimeError("this clarification is only for post-calibration risk-validation failure")
    payload = {
        "seed": SEED,
        "phase": "holdout",
        "timing": "before_dispatch_and_before_any_V21_candidate_outcome",
        "eligible": False,
        "reason": "upstream_risk_model_validation_gate_failed",
        "public_calibration_status": "pass_22_of_22",
        "failure_message": "No risk model passed validation Macro-F1, high-risk recall, and severe-recall gates simultaneously.",
        "controller_executed": False,
        "V21_candidate_executed": False,
        "queue_action": "retain_as_ineligible_and_advance_to_next_predeclared_seed",
    }
    ELIGIBILITY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clarification = {
        "protocol_id": "V21-PROBABILITY-CALIBRATION-HOLDOUT-20260909",
        "type": "execution_clarification_not_threshold_or_candidate_change",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_run_freeze": {
            "path": str(RUN_FREEZE.relative_to(PROJECT_DIR)),
            "sha256": sha256(RUN_FREEZE),
        },
        "seed": SEED,
        "trigger": "source pipeline stopped at the frozen risk-model validation gate before dispatch",
        "resolution": "record the seed as upstream-ineligible and continue the existing queue",
        "candidate_changed": False,
        "threshold_changed": False,
        "seed_order_changed": False,
        "V21_outcomes_seen": False,
        "eligibility_record": {
            "path": str(ELIGIBILITY.relative_to(PROJECT_DIR)),
            "sha256": sha256(ELIGIBILITY),
        },
        "supporting_files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in required
        ],
    }
    AMENDMENT.write_text(
        json.dumps(clarification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(clarification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
