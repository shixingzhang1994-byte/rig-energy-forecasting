from __future__ import annotations

"""Run one V24 seed while retaining every unexpected preparation failure."""

import json
import subprocess
import sys
import traceback
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v23_protocol_seed as implementation  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs/v24_submission_revision.yaml"
DISPATCH_PATH = PROJECT_DIR / "configs/v24_dispatch_matched.yaml"
implementation.DEFAULT_PROTOCOL = PROTOCOL_PATH
implementation.DISPATCH_CONFIG = DISPATCH_PATH


def _argument_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _write_preparation_failure(exc: BaseException) -> None:
    seed_text = _argument_value("--seed")
    if seed_text is None:
        return
    protocol_arg = _argument_value("--protocol")
    protocol_path = Path(protocol_arg).resolve() if protocol_arg else PROTOCOL_PATH
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    root = (
        PROJECT_DIR
        / "artifacts"
        / protocol["artifact_namespace"]
        / f"seed_{int(seed_text)}"
    )
    root.mkdir(parents=True, exist_ok=True)
    command = getattr(exc, "cmd", None)
    command_text = (
        " ".join(map(str, command))
        if isinstance(command, (list, tuple))
        else str(command or "")
    )
    failure = {
        "seed": int(seed_text),
        "phase": "confirmatory",
        "retained_regardless_of_outcome_direction": True,
        "technical_gate_pass": False,
        "performance_gate_applied": False,
        "failure_stage": "input_preparation",
        "failure_type": type(exc).__name__,
        "failed_command": command_text,
        "controller_outcomes_generated": False,
        "traceback": "".join(traceback.format_exception(exc)),
    }
    (root / "result.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    try:
        implementation.main()
    except SystemExit:
        raise
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        _write_preparation_failure(exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
