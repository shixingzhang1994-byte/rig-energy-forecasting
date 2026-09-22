from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_calibration import (  # noqa: E402
    COMPONENT_COLUMNS,
    load_public_calibration,
    validate_public_calibration,
)


def _check(name: str, passed: bool, observed: object, expected: object) -> dict:
    return {
        "check": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计V14公开证据标定数据")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data/processed/rig_load_v14_calibrated.parquet",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v14_public_calibration/audit",
    )
    args = parser.parse_args()

    config = load_public_calibration(args.calibration)
    evidence = validate_public_calibration(config)
    data = pd.read_parquet(args.data)
    physical = data["physical_total_power_kw"].to_numpy(float)
    measured = data["total_active_power_kw"].to_numpy(float)
    raw = data["total_active_power_kw_raw"]
    frequency = float(
        data["timestamp"].sort_values().diff().dt.total_seconds().dropna().median()
    )
    checks: list[dict] = [
        _check("evidence_schema", evidence.passed, list(evidence.issues), "no issues"),
        _check(
            "source_type_coverage",
            len({item["source_type"] for item in config["sources"]}) >= 7,
            sorted({item["source_type"] for item in config["sources"]}),
            "paper/patent/manufacturer/news/handbook/standard/government",
        ),
        _check("sampling_interval_seconds", frequency == 5.0, frequency, 5.0),
        _check(
            "component_balance_mae_kw",
            float(
                np.mean(
                    np.abs(
                        physical
                        - data.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1).to_numpy(float)
                    )
                )
            )
            < 1e-3,
            float(
                np.mean(
                    np.abs(
                        physical
                        - data.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1).to_numpy(float)
                    )
                )
            ),
            "< 0.001 kW",
        ),
    ]

    load = config["load_calibration"]
    overall_mean_range = list(map(float, load["overall_acceptance_mean_kw"]))
    overall_max_range = list(map(float, load["overall_acceptance_max_kw"]))
    checks.extend(
        [
            _check(
                "overall_mean_kw",
                overall_mean_range[0] <= float(physical.mean()) <= overall_mean_range[1],
                float(physical.mean()),
                overall_mean_range,
            ),
            _check(
                "overall_max_kw",
                overall_max_range[0] <= float(physical.max()) <= overall_max_range[1],
                float(physical.max()),
                overall_max_range,
            ),
        ]
    )
    for state, target in load["state_targets"].items():
        values = data.loc[data["operation_state"].astype(str) == state, "physical_total_power_kw"]
        mean_range = list(map(float, target["acceptance_mean_kw"]))
        checks.append(
            _check(
                f"state_mean_kw::{state}",
                len(values) > 0 and mean_range[0] <= float(values.mean()) <= mean_range[1],
                None if len(values) == 0 else float(values.mean()),
                mean_range,
            )
        )
        checks.append(
            _check(
                f"state_hard_cap_kw::{state}",
                len(values) > 0 and float(values.max()) <= float(target["hard_cap_kw"]) + 1e-6,
                None if len(values) == 0 else float(values.max()),
                float(target["hard_cap_kw"]),
            )
        )

    observed_mask = raw.notna().to_numpy()
    relative_error = np.abs(measured[observed_mask] - physical[observed_mask]) / np.maximum(
        physical[observed_mask], 1.0
    )
    relative_limit = float(config["measurement_calibration"]["relative_error_clip"])
    quantization = float(config["measurement_calibration"]["quantization_kw"])
    relative_tolerance = relative_limit + quantization / max(float(physical.min()), 1.0)
    checks.extend(
        [
            _check(
                "measurement_relative_error_bound",
                float(relative_error.max()) <= relative_tolerance + 1e-9,
                float(relative_error.max()),
                relative_tolerance,
            ),
            _check(
                "physical_impact_rate_kw_per_s",
                float(np.max(np.abs(np.diff(physical)) / frequency)) >= 41.0,
                float(np.max(np.abs(np.diff(physical)) / frequency)),
                ">= 41 kW/s (field-observed impact exists)",
            ),
            _check(
                "source_label",
                set(data["source_type"].astype(str).unique())
                == {"public_evidence_calibrated_synthetic_v14"},
                sorted(data["source_type"].astype(str).unique()),
                ["public_evidence_calibrated_synthetic_v14"],
            ),
            _check(
                "missingness_is_not_hidden",
                int(raw.isna().sum()) == int(data["quality_flag"].astype(str).eq("imputed").sum()),
                {"raw_missing": int(raw.isna().sum()), "imputed_flags": int(data["quality_flag"].astype(str).eq("imputed").sum())},
                "equal",
            ),
        ]
    )

    frame = pd.DataFrame(checks)
    overall = bool(frame["passed"].all())
    result = {
        "version": "V14",
        "status": "pass" if overall else "fail",
        "passed_checks": int(frame["passed"].sum()),
        "total_checks": int(len(frame)),
        "field_scada_claimed": False,
        "claim": "public-evidence-calibrated synthetic data only",
        "failed_checks": frame.loc[~frame["passed"], "check"].tolist(),
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.artifact_dir / "calibration_checks.csv", index=False, encoding="utf-8-sig")
    (args.artifact_dir / "calibration_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

