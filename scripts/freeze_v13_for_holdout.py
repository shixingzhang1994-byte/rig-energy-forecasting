from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs/v13_grid_headroom_reserve.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.relative_to(PROJECT_DIR)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    protocol_document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = protocol_document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    required = int(protocol["required_eligible_seeds_per_phase"])

    accepted_development_records: list[dict[str, str | int]] = []
    for seed in [int(value) for value in protocol["development_seed_order"]]:
        seed_root = artifact_root / f"seed_{seed}"
        eligibility_path = seed_root / "eligibility.json"
        if not eligibility_path.exists():
            raise RuntimeError(f"开发种子尚未按预注册顺序评估: {seed}")
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if not bool(eligibility.get("eligible")):
            continue

        result_path = seed_root / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"可评估开发种子缺少结果: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not bool(result.get("overall_pass")):
            raise RuntimeError(f"开发硬门未通过，拒绝冻结: {result_path}")

        accepted_development_records.append(
            {
                "seed": seed,
                "eligibility_path": str(eligibility_path.relative_to(PROJECT_DIR)),
                "eligibility_sha256": _sha256(eligibility_path),
                "result_path": str(result_path.relative_to(PROJECT_DIR)),
                "result_sha256": _sha256(result_path),
            }
        )
        if len(accepted_development_records) == required:
            break

    if len(accepted_development_records) != required:
        raise RuntimeError(
            f"仅有{len(accepted_development_records)}个可评估且通过的开发种子，"
            f"要求{required}个，拒绝冻结"
        )

    fixed_paths = [
        PROTOCOL_PATH,
        PROJECT_DIR / "configs/v5_acceptance.yaml",
        PROJECT_DIR / "configs/v6_dispatch_development.yaml",
        PROJECT_DIR / "configs/synthetic_default.yaml",
        PROJECT_DIR / "configs/risk_default.yaml",
        PROJECT_DIR / "configs/dispatch_default.yaml",
        PROJECT_DIR / "environment.yml",
        PROJECT_DIR / "scripts/generate_synthetic.py",
        PROJECT_DIR / "scripts/run_benchmark.py",
        PROJECT_DIR / "scripts/run_risk_benchmark.py",
        PROJECT_DIR / "scripts/run_dispatch_benchmark.py",
        PROJECT_DIR / "scripts/run_v9_supervisor_first_seed.py",
        PROJECT_DIR / "scripts/run_v10_protocol_seed.py",
        Path(__file__).resolve(),
        *sorted((PROJECT_DIR / "src/rig_energy").rglob("*.py")),
    ]
    missing = [str(path.relative_to(PROJECT_DIR)) for path in fixed_paths if not path.is_file()]
    if missing:
        raise RuntimeError("冻结文件缺失: " + ", ".join(missing))

    manifest = {
        "protocol": protocol["name"],
        "status": "frozen_before_holdout",
        "artifact_namespace": protocol["artifact_namespace"],
        "diagnostic_seeds_excluded_from_evidence": [
            int(value)
            for value in protocol["diagnostic_seeds_excluded_from_evidence"]
        ],
        "development_seed_order": [
            int(value) for value in protocol["development_seed_order"]
        ],
        "holdout_seed_order": [int(value) for value in protocol["holdout_seed_order"]],
        "required_eligible_seeds_per_phase": required,
        "accepted_development_records": accepted_development_records,
        "supervisor": protocol_document["supervisor"],
        "fixed_unit_commitment": protocol_document["fixed_unit_commitment"],
        "performance_gate": protocol_document["performance_gate"],
        "claim_boundary": protocol_document["claim_boundary"],
        "frozen_files": [_frozen_record(path) for path in fixed_paths],
        "post_freeze_rule": (
            "Any frozen-file hash change invalidates holdout execution. Holdout seeds "
            "must be processed in the preregistered order, and the first two eligible "
            "holdouts must be reported without parameter changes."
        ),
    }
    output = artifact_root / "freeze_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
