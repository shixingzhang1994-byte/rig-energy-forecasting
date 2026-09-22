from __future__ import annotations

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


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v15_generator_first_reserve_acceptance.yaml"


def _dispatch_aware_run(command, *args, **kwargs):
    command = list(command)
    try:
        index = command.index("scripts/run_dispatch_benchmark.py")
    except ValueError:
        pass
    else:
        command[index] = "scripts/run_v15_dispatch_benchmark.py"
    return real_subprocess.run(command, *args, **kwargs)


def _refresh_result_manifest(root: Path) -> None:
    result_path = root / "result.json"
    manifest_path = root / "evidence_manifest.json"
    if not result_path.exists() or not manifest_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["reproduction_command"] = result["reproduction_command"].replace(
        "scripts/run_dispatch_benchmark.py",
        "scripts/run_v15_dispatch_benchmark.py",
    )
    result["controller_delta"] = "committed_generator_before_storage"
    result["diagnostic_seed_excluded"] = 20261004
    result["field_scada_claimed"] = False
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(result_path.relative_to(PROJECT_DIR))
    for item in manifest.get("files", []):
        if item.get("path") == relative:
            item["sha256"] = digest
            item["size_bytes"] = result_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    protocol_runner.DEFAULT_PROTOCOL_PATH = DEFAULT_PROTOCOL
    protocol_runner._prepare_inputs = v14_runner._prepare_inputs
    protocol_runner._build_dispatch = v14_runner._build_dispatch
    protocol_runner.subprocess = SimpleNamespace(run=_dispatch_aware_run)
    try:
        protocol_runner.main()
    finally:
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
            document = yaml.safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
            root = (
                PROJECT_DIR
                / "artifacts"
                / document["protocol"]["artifact_namespace"]
                / f"seed_{seed}"
            )
            _refresh_result_manifest(root)


if __name__ == "__main__":
    main()
