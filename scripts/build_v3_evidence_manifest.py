from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "artifacts/v3_competitive_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_patterns = [
        "configs/*.yaml",
        "src/**/*.py",
        "scripts/*.py",
        "tests/*.py",
    ]
    evidence_paths = [
        "artifacts/v3_data/data_profile.json",
        "artifacts/v3_forecast_acceptance/benchmark_metrics.csv",
        "artifacts/v3_competitive_forecast/benchmark_metrics.csv",
        "artifacts/v3_competitive_forecast/run_metadata.json",
        "artifacts/v3_competitive_forecast/ensemble_weights.json",
        "artifacts/v3_robustness/summary/pairwise_vs_modern_baselines.csv",
        "artifacts/v3_robustness/summary/robustness_summary.json",
        "artifacts/v3_competitive_risk/risk_metrics.csv",
        "artifacts/v3_competitive_risk/operational_model_selection.json",
        "artifacts/v3_competitive_risk/dispatch_guard_metrics.json",
        "artifacts/v3_competitive_dispatch/dispatch_metrics.csv",
        "artifacts/v3_competitive_dispatch/scenario_selection.json",
        "artifacts/v3_competitive_audit/competitive_audit.csv",
        "artifacts/v3_competitive_audit/competitive_audit.json",
        "artifacts/acceptance_audit/acceptance_audit.json",
    ]
    paths: set[Path] = set()
    for pattern in source_patterns:
        paths.update(path for path in PROJECT_DIR.glob(pattern) if path.is_file())
    for relative in evidence_paths:
        path = PROJECT_DIR / relative
        if path.exists():
            paths.add(path)

    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "xgboost",
        "PyYAML",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "evidence_manifest.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_DIR),
        "git_repository": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "file_count": len(rows),
        "files": rows,
        "scope": "V3 competitive simulation evidence; hashes support integrity, not field validity",
    }
    (OUTPUT_DIR / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("file_count", "packages", "scope")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
