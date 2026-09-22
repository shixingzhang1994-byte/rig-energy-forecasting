from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol_path = PROJECT_DIR / "configs/v9_causal_supervisor_development.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    development_seeds = [int(v) for v in protocol["protocol"]["development_seeds"]]
    gate_names = ["first_seed_gate.json", "second_seed_gate.json"]
    gate_paths = [
        PROJECT_DIR
        / f"artifacts/v9_causal_supervisor/seed_{seed}/{gate_name}"
        for seed, gate_name in zip(development_seeds, gate_names, strict=True)
    ]
    for path in gate_paths:
        if not path.exists() or not bool(
            json.loads(path.read_text(encoding="utf-8")).get("overall_pass")
        ):
            raise RuntimeError(f"开发硬门未通过，拒绝冻结: {path}")

    fixed_paths = [
        protocol_path,
        PROJECT_DIR / "configs/v5_acceptance.yaml",
        PROJECT_DIR / "configs/v6_dispatch_development.yaml",
        PROJECT_DIR / "configs/synthetic_default.yaml",
        PROJECT_DIR / "configs/risk_default.yaml",
        PROJECT_DIR / "configs/dispatch_default.yaml",
        PROJECT_DIR / "scripts/generate_synthetic.py",
        PROJECT_DIR / "scripts/run_benchmark.py",
        PROJECT_DIR / "scripts/run_risk_benchmark.py",
        PROJECT_DIR / "scripts/run_dispatch_benchmark.py",
        PROJECT_DIR / "scripts/run_v9_supervisor_first_seed.py",
        *sorted((PROJECT_DIR / "src/rig_energy").rglob("*.py")),
    ]
    frozen_files = [
        {
            "path": str(path.relative_to(PROJECT_DIR)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in fixed_paths
    ]
    manifest = {
        "protocol": protocol["protocol"]["name"],
        "status": "frozen_before_holdout",
        "development_seeds": development_seeds,
        "holdout_seeds": [int(v) for v in protocol["protocol"]["holdout_seeds"]],
        "supervisor": protocol["supervisor"],
        "fixed_unit_commitment": protocol["fixed_unit_commitment"],
        "development_gate_files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": _sha256(path),
            }
            for path in gate_paths
        ],
        "frozen_files": frozen_files,
        "post_freeze_rule": (
            "Any frozen-file hash change invalidates holdout execution; both holdout "
            "seeds must be reported without parameter changes."
        ),
    }
    output = PROJECT_DIR / "artifacts/v9_causal_supervisor/freeze_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
