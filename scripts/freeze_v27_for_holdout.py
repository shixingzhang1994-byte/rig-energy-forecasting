from __future__ import annotations

"""Freeze the V27 heterogeneous family before any prospective unit outcome."""

import sys
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import freeze_v23_for_holdout as implementation  # noqa: E402


PROTOCOL = PROJECT_DIR / "configs/v27_heterogeneous_validation.yaml"
FAMILY = PROJECT_DIR / "configs/v27_heterogeneous_family.yaml"
PARAMETER_ROOT = PROJECT_DIR / "artifacts/V27_heterogeneous_validation/parameter_sets"
DEVELOPMENT = PROJECT_DIR / "artifacts/V27_heterogeneous_validation/development/seed_20261100"
D00_DISPATCH = PARAMETER_ROOT / "D00_seed_20261100/dispatch.yaml"


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))["protocol"]
    family = yaml.safe_load(FAMILY.read_text(encoding="utf-8"))
    prospective = family["prospective_units"]
    factors = {
        (unit["load"], unit["transition"], unit["plant"])
        for unit in prospective
    }
    if len(prospective) != 12 or len(factors) != 12:
        raise RuntimeError("V27 factor design is not the declared balanced 3x2x2 family")
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    revealed = [
        seed
        for seed in protocol["prospective_holdout_seed_order"]
        if (artifact_root / f"seed_{seed}").exists()
    ]
    if revealed:
        raise RuntimeError(f"prospective V27 unit directories already exist: {revealed}")

    parameter_files = sorted(
        path
        for path in PARAMETER_ROOT.rglob("*")
        if path.is_file()
    )
    implementation.PROTOCOL_PATH = PROTOCOL
    implementation.DISPATCH_PATH = D00_DISPATCH
    implementation.DEVELOPMENT_ROOT = DEVELOPMENT
    implementation.RUNNER_PATHS = [
        PROJECT_DIR / "scripts/prepare_v27_parameter_sets.py",
        PROJECT_DIR / "scripts/run_v27_protocol_seed.py",
        PROJECT_DIR / "scripts/run_v27_development.py",
        PROJECT_DIR / "scripts/run_v26_dispatch_benchmark.py",
        FAMILY,
        *parameter_files,
    ]
    implementation.ANALYSIS_PATHS = [
        PROJECT_DIR / "scripts/analyze_v27_heterogeneous.py",
    ]
    implementation.FREEZE_ENTRY_PATH = Path(__file__).resolve()
    implementation.main()


if __name__ == "__main__":
    main()
