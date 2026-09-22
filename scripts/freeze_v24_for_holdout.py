from __future__ import annotations

"""Freeze V24 after the revealed V23 failure seed passes development replay."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import freeze_v23_for_holdout as implementation  # noqa: E402


implementation.PROTOCOL_PATH = PROJECT_DIR / "configs/v24_submission_revision.yaml"
implementation.DISPATCH_PATH = PROJECT_DIR / "configs/v24_dispatch_matched.yaml"
implementation.DEVELOPMENT_ROOT = (
    PROJECT_DIR / "artifacts/V24_submission_revision/development/seed_20261100"
)
implementation.RUNNER_PATHS = [
    PROJECT_DIR / "scripts/run_v23_protocol_seed.py",
    PROJECT_DIR / "scripts/run_v24_protocol_seed.py",
]
implementation.ANALYSIS_PATHS = [
    PROJECT_DIR / "scripts/analyze_v23_confirmatory.py",
    PROJECT_DIR / "scripts/analyze_v24_confirmatory.py",
]
implementation.FREEZE_ENTRY_PATH = Path(__file__).resolve()


if __name__ == "__main__":
    implementation.main()
