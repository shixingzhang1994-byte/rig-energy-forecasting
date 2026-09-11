from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_calibration import (  # noqa: E402
    calibrate_synthetic_frame,
    load_public_calibration,
    validate_public_calibration,
)
from rig_energy.data.schema import validate_canonical_frame  # noqa: E402
from rig_energy.data.synthetic import generate_synthetic_rig_load  # noqa: E402


def _deep_merge(base: dict, overrides: dict) -> dict:
    output = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成不使用模型结果调参的V14公开证据标定合成数据"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_DIR / "configs/synthetic_default.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data/processed/rig_load_v14_calibrated.parquet",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v14_public_calibration/data",
    )
    args = parser.parse_args()

    calibration = load_public_calibration(args.calibration)
    evidence_audit = validate_public_calibration(calibration)
    if not evidence_audit.passed:
        raise ValueError("; ".join(evidence_audit.issues))
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    resolved = _deep_merge(base, calibration["synthetic_overrides"])
    if args.seed is not None:
        resolved["project"]["seed"] = int(args.seed)
    if args.days is not None:
        resolved["project"]["days"] = int(args.days)

    seed = int(resolved["project"]["seed"])
    source = generate_synthetic_rig_load(resolved)
    data, report = calibrate_synthetic_frame(source, calibration, seed=seed)
    data, schema = validate_canonical_frame(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False)
    (args.artifact_dir / "resolved_synthetic_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (args.artifact_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.artifact_dir / "evidence_audit.json").write_text(
        json.dumps(
            {"passed": evidence_audit.passed, "issues": list(evidence_audit.issues)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    profile = (
        data.groupby("operation_state")["physical_total_power_kw"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    profile.to_csv(
        args.artifact_dir / "state_power_profile.csv", index=False, encoding="utf-8-sig"
    )

    frequency = int(resolved["project"]["frequency_seconds"])
    view = data.iloc[: int(12 * 3600 / frequency)]
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(view["timestamp"], view["physical_total_power_kw"], linewidth=0.8)
    ax.set_title("V14公开证据标定负荷：前12小时")
    ax.set_ylabel("有功功率 / kW")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.artifact_dir / "calibrated_load_first_12h.png", dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": schema.as_dict(),
                "calibration": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

