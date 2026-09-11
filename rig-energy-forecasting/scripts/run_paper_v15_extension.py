from __future__ import annotations

"""Run the two predeclared V15 reserve seeds as a paper-only extension.

The original V15 namespace and acceptance summary remain untouched.  This
runner verifies the original frozen file hashes, creates a separate extension
freeze manifest, and reuses the frozen V15 data, risk, and dispatch pipeline.
"""

import hashlib
import json
import subprocess as real_subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v10_protocol_seed as protocol_runner  # noqa: E402
import run_v14_protocol_seed as v14_runner  # noqa: E402
import run_v15_protocol_seed as v15_runner  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/paper_v15_extension_holdouts.yaml"
ORIGINAL_FREEZE = (
    PROJECT_DIR / "artifacts/v15_generator_first_reserve/freeze_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_original_freeze() -> dict:
    manifest = json.loads(ORIGINAL_FREEZE.read_text(encoding="utf-8"))
    changed = []
    for item in manifest["frozen_files"]:
        target = PROJECT_DIR / item["path"]
        if not target.exists() or _sha256(target) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("V15 frozen files changed: " + ", ".join(changed))
    return manifest


def _ensure_extension_freeze(protocol_path: Path) -> Path:
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    path = root / "freeze_manifest.json"
    original = _verify_original_freeze()
    payload = {
        "protocol": protocol["name"],
        "status": "locked_before_extension_results",
        "artifact_namespace": protocol["artifact_namespace"],
        "holdout_seed_order": protocol["holdout_seed_order"],
        "required_eligible_seeds_per_phase": protocol[
            "required_eligible_seeds_per_phase"
        ],
        "source_v15_freeze_manifest": str(ORIGINAL_FREEZE.relative_to(PROJECT_DIR)),
        "source_v15_freeze_sha256": _sha256(ORIGINAL_FREEZE),
        "extension_protocol": str(protocol_path.relative_to(PROJECT_DIR)),
        "extension_protocol_sha256": _sha256(protocol_path),
        "frozen_files": original["frozen_files"],
        "claim_boundary": document["claim_boundary"],
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing extension freeze manifest does not match")
    else:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def _dispatch_aware_run(command, *args, **kwargs):
    command = list(command)
    try:
        index = command.index("scripts/run_dispatch_benchmark.py")
    except ValueError:
        pass
    else:
        command[index] = "scripts/run_v15_dispatch_benchmark.py"
    return real_subprocess.run(command, *args, **kwargs)


def _refresh_result_manifest(protocol_path: Path, seed: int) -> None:
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    root = (
        PROJECT_DIR
        / "artifacts"
        / document["protocol"]["artifact_namespace"]
        / f"seed_{seed}"
    )
    result_path = root / "result.json"
    manifest_path = root / "evidence_manifest.json"
    if not result_path.exists() or not manifest_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["reproduction_command"] = result["reproduction_command"].replace(
        "scripts/run_dispatch_benchmark.py", "scripts/run_v15_dispatch_benchmark.py"
    )
    result.update(
        {
            "controller_delta": "committed_generator_before_storage",
            "frozen_controller": "V15",
            "extension_status": "prospective_paper_extension_no_tuning",
            "field_scada_claimed": False,
        }
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(result_path.relative_to(PROJECT_DIR))
    for item in manifest.get("files", []):
        if item.get("path") == relative:
            item["sha256"] = _sha256(result_path)
            item["size_bytes"] = result_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _record_upstream_ineligibility(protocol_path: Path, seed: int) -> bool:
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    root = (
        PROJECT_DIR
        / "artifacts"
        / document["protocol"]["artifact_namespace"]
        / f"seed_{seed}"
    )
    audit_path = root / "source/data/audit/calibration_audit.json"
    if not audit_path.exists():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "fail":
        return False
    eligibility = {
        "seed": seed,
        "phase": "holdout",
        "timing": "before_dispatch_and_before_any_controller_outcome",
        "eligible": False,
        "reason": "public_evidence_calibration_gate_failed",
        "failed_checks": audit.get("failed_checks", []),
        "calibration_audit": str(audit_path.relative_to(PROJECT_DIR)),
        "calibration_audit_sha256": _sha256(audit_path),
        "controller_executed": False,
    }
    path = root / "eligibility.json"
    path.write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(eligibility, ensure_ascii=False), flush=True)
    return True


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        protocol_runner.main()
        return
    protocol_path = DEFAULT_PROTOCOL
    if "--protocol" in sys.argv:
        protocol_path = Path(sys.argv[sys.argv.index("--protocol") + 1]).resolve()
    if "--seed" not in sys.argv:
        raise SystemExit("--seed is required")
    seed = int(sys.argv[sys.argv.index("--seed") + 1])
    freeze_path = _ensure_extension_freeze(protocol_path)
    print(f"EXTENSION_FREEZE_OK {freeze_path}", flush=True)

    protocol_runner.DEFAULT_PROTOCOL_PATH = protocol_path
    protocol_runner._prepare_inputs = v14_runner._prepare_inputs
    protocol_runner._build_dispatch = v14_runner._build_dispatch
    protocol_runner.subprocess = SimpleNamespace(run=_dispatch_aware_run)
    try:
        protocol_runner.main()
    except real_subprocess.CalledProcessError:
        if _record_upstream_ineligibility(protocol_path, seed):
            raise SystemExit(3)
        raise
    finally:
        _refresh_result_manifest(protocol_path, seed)


if __name__ == "__main__":
    main()
