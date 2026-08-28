from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.adapters import SyntheticDataSource  # noqa: E402
from rig_energy.data.schema import validate_canonical_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/synthetic_default.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "data/processed/rig_load_synthetic_v2.parquet")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/v2_data")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=PROJECT_DIR / "configs/simulation_parameter_evidence.yaml",
    )
    parser.add_argument("--seed", type=int, default=None, help="可选的仿真随机种子覆盖")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["project"]["seed"] = int(args.seed)
    data = SyntheticDataSource(config).load()
    data, report = validate_canonical_frame(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False)

    state_profile = (
        data.groupby("operation_state")["total_active_power_kw"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .reset_index()
    )
    state_profile.to_csv(args.artifact_dir / "state_power_profile.csv", index=False, encoding="utf-8-sig")
    if args.evidence.exists():
        evidence = yaml.safe_load(args.evidence.read_text(encoding="utf-8"))
        pd.DataFrame(evidence["parameters"]).to_csv(
            args.artifact_dir / "simulation_parameter_evidence.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (args.artifact_dir / "simulation_parameter_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    profile = report.as_dict()
    profile.update(
        {
            "simulation_seed": int(config["project"]["seed"]),
            "imputed_rows": int((data["quality_flag"] == "imputed").sum()),
            "state_counts": data["operation_state"].value_counts().to_dict(),
            "component_balance_mae_kw": float(
                (
                    data["physical_total_power_kw"]
                    - data[
                        [
                            "auxiliary_power_kw",
                            "mud_pump_power_kw",
                            "topdrive_power_kw",
                            "drawworks_power_kw",
                            "other_power_kw",
                        ]
                    ].sum(axis=1)
                )
                .abs()
                .mean()
            ),
            "well_depth_start_m": float(data["well_depth_m"].iloc[0]),
            "well_depth_end_m": float(data["well_depth_m"].iloc[-1]),
            "process_signal_columns": [
                "well_depth_m",
                "formation_hardness_index",
                "pump_strokes_per_min",
                "standpipe_pressure_mpa",
                "topdrive_rpm",
                "hookload_pct",
                "hook_speed_m_per_s",
                "rate_of_penetration_m_per_h",
            ],
        }
    )
    (args.artifact_dir / "data_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    view = data.iloc[: int(12 * 3600 / config["project"]["frequency_seconds"])]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(view["timestamp"], view["total_active_power_kw"], linewidth=0.8, color="#1D4ED8")
    ax.set_title("拟真钻机总功率：前12小时")
    ax.set_ylabel("有功功率 / kW")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.artifact_dir / "synthetic_load_first_12h.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"output": str(args.output), "profile": profile}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
