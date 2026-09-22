from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs/v14_public_calibration_acceptance.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_DIR)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    required = int(protocol["required_eligible_seeds_per_phase"])
    accepted = []
    for seed in map(int, protocol["development_seed_order"]):
        seed_root = root / f"seed_{seed}"
        eligibility_path = seed_root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"开发种子尚未按顺序评估: {seed}")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if not eligibility.get("eligible"):
            continue
        result_path = seed_root / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"可评估开发种子缺少结果: {seed}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("overall_pass"):
            raise RuntimeError(f"开发硬门未通过，拒绝冻结: {seed}")
        accepted.append(
            {
                "seed": seed,
                "eligibility_path": str(eligibility_path.relative_to(PROJECT_DIR)),
                "eligibility_sha256": _sha256(eligibility_path),
                "result_path": str(result_path.relative_to(PROJECT_DIR)),
                "result_sha256": _sha256(result_path),
            }
        )
        if len(accepted) == required:
            break
    if len(accepted) != required:
        raise RuntimeError(f"仅{len(accepted)}个可评估且通过的开发种子，需{required}个")

    frozen_paths = [
        PROTOCOL_PATH,
        PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml",
        PROJECT_DIR / "configs/v5_acceptance.yaml",
        PROJECT_DIR / "configs/v6_dispatch_development.yaml",
        PROJECT_DIR / "configs/synthetic_default.yaml",
        PROJECT_DIR / "configs/risk_default.yaml",
        PROJECT_DIR / "configs/dispatch_default.yaml",
        PROJECT_DIR / "environment.yml",
        PROJECT_DIR / "scripts/generate_v14_calibrated.py",
        PROJECT_DIR / "scripts/audit_v14_calibration.py",
        PROJECT_DIR / "scripts/run_v14_dispatch_benchmark.py",
        PROJECT_DIR / "scripts/run_v14_protocol_seed.py",
        PROJECT_DIR / "scripts/run_benchmark.py",
        PROJECT_DIR / "scripts/run_risk_benchmark.py",
        PROJECT_DIR / "scripts/run_v10_protocol_seed.py",
        Path(__file__).resolve(),
        *sorted((PROJECT_DIR / "src/rig_energy").rglob("*.py")),
    ]
    missing = [str(path.relative_to(PROJECT_DIR)) for path in frozen_paths if not path.is_file()]
    if missing:
        raise RuntimeError("冻结文件缺失: " + ", ".join(missing))
    manifest = {
        "protocol": protocol["name"],
        "status": "frozen_before_holdout",
        "artifact_namespace": protocol["artifact_namespace"],
        "development_seed_order": list(map(int, protocol["development_seed_order"])),
        "holdout_seed_order": list(map(int, protocol["holdout_seed_order"])),
        "required_eligible_seeds_per_phase": required,
        "accepted_development_records": accepted,
        "calibration": "configs/v14_public_evidence_calibration.yaml",
        "fixed_unit_commitment": document["fixed_unit_commitment"],
        "plant_and_cost_overrides": document["plant_and_cost_overrides"],
        "supervisor": document["supervisor"],
        "performance_gate": document["performance_gate"],
        "claim_boundary": document["claim_boundary"],
        "frozen_files": [_record(path) for path in frozen_paths],
        "post_freeze_rule": "任一冻结文件哈希变化均使留出无效；必须连续报告最先两个可评估留出种子。",
    }
    output = root / "freeze_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

