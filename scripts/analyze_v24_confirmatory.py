from __future__ import annotations

"""Aggregate the first 12 eligible V24 seeds at the seed-cluster level."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import analyze_v23_confirmatory as implementation  # noqa: E402


implementation.DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v24_submission_revision.yaml"


if __name__ == "__main__":
    implementation.main()
