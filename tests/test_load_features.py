from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.analysis import analyze_load_features  # noqa: E402
from rig_energy.data.adapters import SyntheticDataSource  # noqa: E402


def test_feature_analysis_outputs_acceptance_tables(tmp_path: Path):
    config = yaml.safe_load(
        (PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8")
    )
    config["project"]["days"] = 1
    frame = SyntheticDataSource(config).load()
    data_path = tmp_path / "synthetic.parquet"
    artifact_dir = tmp_path / "features"
    frame.to_parquet(data_path, index=False)

    summary = analyze_load_features(
        data_path,
        PROJECT_DIR / "configs/feature_analysis.yaml",
        artifact_dir,
    )
    acceptance = pd.read_csv(artifact_dir / "acceptance_feature_table.csv")
    states = pd.read_csv(artifact_dir / "state_metrics.csv")
    assert summary["sampling_interval_seconds"] == 5.0
    assert {"mean_power", "p95_power", "maximum_absolute_ramp"} <= set(
        acceptance["metric"]
    )
    assert {"operation_state", "mean_power_kw", "energy_kwh"} <= set(states.columns)
    assert (artifact_dir / "feature_summary.png").stat().st_size > 0
