from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_real_datasets import sha256_file


def _adjacent_soc_response(
    frame: pd.DataFrame,
    power_column: str,
    *,
    power_threshold: float,
    soc_change_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    ordered = frame.sort_values("timestamp").copy()
    timestamp = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered["soc_change_pct"] = ordered["measured_battery_soc_pct"].diff()
    adjacent = timestamp.diff().eq(pd.Timedelta(minutes=1))
    eligible = (
        ordered["quality_ok"].fillna(False)
        & adjacent
        & ordered[power_column].abs().ge(float(power_threshold))
        & ordered["soc_change_pct"].abs().ge(float(soc_change_threshold))
    )
    response = ordered.loc[
        eligible, ["timestamp", power_column, "soc_change_pct"]
    ].copy()
    expected = -np.sign(response[power_column].to_numpy(float))
    observed = np.sign(response["soc_change_pct"].to_numpy(float))
    match = observed == expected
    correlation = float(
        response[power_column].corr(response["soc_change_pct"])
    ) if len(response) >= 2 else float("nan")
    return response, {
        "eligible_adjacent_minutes": int(len(response)),
        "direction_consistent_minutes": int(match.sum()),
        "soc_direction_consistency": float(match.mean()) if len(match) else float("nan"),
        "power_soc_change_correlation": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v28_bess_observational_frozen.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/bess_observational",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["protocol"]["status"] != "frozen_before_metric_execution":
        raise ValueError("BESS observational protocol is not frozen")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol_id": config["protocol"]["id"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "causal_controller_efficacy_claimed": False,
        "target_rig_field_validation_claimed": False,
        "datasets": {},
    }
    annual_rows: list[dict[str, object]] = []

    m5_config = config["m5bat"]
    m5_path = PROJECT_DIR / m5_config["prepared_path"]
    if sha256_file(m5_path) != m5_config["prepared_sha256"]:
        raise ValueError("M5BAT prepared hash mismatch")
    m5 = pd.read_parquet(m5_path)
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)
    m5_quality = m5.loc[m5["quality_ok"].fillna(False)].copy()
    threshold = float(m5_config["active_power_threshold_kw"])
    active = m5_quality.loc[
        m5_quality["measured_battery_dc_power_kw"].abs().ge(threshold)
        & m5_quality["measured_ac_power_kw"].abs().ge(threshold)
    ].copy()
    same_sign = np.sign(active["measured_battery_dc_power_kw"]) == np.sign(
        active["measured_ac_power_kw"]
    )
    active = active.loc[same_sign]
    discharge = active.loc[
        active["measured_battery_dc_power_kw"].gt(0)
        & active["measured_ac_power_kw"].gt(0)
    ]
    charge = active.loc[
        active["measured_battery_dc_power_kw"].lt(0)
        & active["measured_ac_power_kw"].lt(0)
    ]
    efficiency_low, efficiency_high = map(
        float, m5_config["plausible_one_way_efficiency_range"]
    )
    discharge_ratio = (
        discharge["measured_ac_power_kw"]
        / discharge["measured_battery_dc_power_kw"]
    )
    charge_ratio = (
        charge["measured_battery_dc_power_kw"].abs()
        / charge["measured_ac_power_kw"].abs()
    )
    discharge_eff = discharge_ratio.between(
        efficiency_low, efficiency_high, inclusive="both"
    )
    charge_eff = charge_ratio.between(
        efficiency_low, efficiency_high, inclusive="both"
    )
    m5_response, m5_soc = _adjacent_soc_response(
        m5,
        "measured_battery_dc_power_kw",
        power_threshold=threshold,
        soc_change_threshold=float(m5_config["soc_change_threshold_pct"]),
    )
    setpoint = m5_quality.loc[
        m5_quality["measured_ac_setpoint_kw"].abs().ge(threshold)
        & m5_quality["measured_ac_power_kw"].notna()
    ]
    m5_metrics = {
        "rows": int(len(m5)),
        "quality_ok_rows": int(len(m5_quality)),
        "quality_coverage": float(len(m5_quality) / len(m5)),
        "active_same_sign_minutes": int(len(active)),
        "ac_dc_power_correlation": float(
            active["measured_ac_power_kw"].corr(
                active["measured_battery_dc_power_kw"]
            )
        ),
        "discharge_minutes": int(discharge_eff.sum()),
        "discharge_energy_weighted_efficiency": float(
            discharge.loc[discharge_eff, "measured_ac_power_kw"].sum()
            / discharge.loc[
                discharge_eff, "measured_battery_dc_power_kw"
            ].sum()
        ),
        "charge_minutes": int(charge_eff.sum()),
        "charge_energy_weighted_efficiency": float(
            charge.loc[charge_eff, "measured_battery_dc_power_kw"].abs().sum()
            / charge.loc[charge_eff, "measured_ac_power_kw"].abs().sum()
        ),
        "setpoint_tracking_minutes": int(len(setpoint)),
        "setpoint_tracking_mae_kw": float(
            (
                setpoint["measured_ac_power_kw"]
                - setpoint["measured_ac_setpoint_kw"]
            ).abs().mean()
        ),
        "imbalance_warning_minutes": int(m5_quality["imbalance_warning"].gt(0).sum()),
        "imbalance_alarm_minutes": int(m5_quality["imbalance_alarm"].gt(0).sum()),
        **m5_soc,
    }
    report["datasets"]["m5bat"] = {
        "prepared_sha256_verified": m5_config["prepared_sha256"],
        "metrics": m5_metrics,
        "interpretation_boundary": (
            "Real grid-scale lead-acid BESS behavior; observational, not a test of "
            "the paper controller."
        ),
    }

    for year, group in m5_quality.groupby(m5_quality["timestamp"].dt.year):
        annual_rows.append(
            {
                "dataset": "m5bat",
                "year": int(year),
                "quality_rows": int(len(group)),
                "mean_abs_battery_power_kw": float(
                    group["measured_battery_dc_power_kw"].abs().mean()
                ),
                "mean_soc_pct": float(group["measured_battery_soc_pct"].mean()),
            }
        )

    ts_config = config["tsukuba"]
    ts_path = PROJECT_DIR / ts_config["prepared_path"]
    if sha256_file(ts_path) != ts_config["prepared_sha256"]:
        raise ValueError("Tsukuba prepared hash mismatch")
    ts = pd.read_parquet(ts_path)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts_quality = ts.loc[ts["quality_ok"].fillna(False)].copy()
    ts_threshold = float(ts_config["active_power_threshold_kw"])
    ts_response, ts_soc = _adjacent_soc_response(
        ts,
        "measured_battery_power_kw",
        power_threshold=ts_threshold,
        soc_change_threshold=float(ts_config["soc_change_threshold_pct"]),
    )
    ts_metrics = {
        "rows": int(len(ts)),
        "quality_ok_rows": int(len(ts_quality)),
        "quality_coverage": float(len(ts_quality) / len(ts)),
        "active_battery_minutes": int(
            ts_quality["measured_battery_power_kw"].abs().ge(ts_threshold).sum()
        ),
        "active_battery_fraction": float(
            ts_quality["measured_battery_power_kw"].abs().ge(ts_threshold).mean()
        ),
        **ts_soc,
    }
    report["datasets"]["tsukuba"] = {
        "prepared_sha256_verified": ts_config["prepared_sha256"],
        "metrics": ts_metrics,
        "interpretation_boundary": (
            "Real building-microgrid BESS behavior; observational and not target-rig "
            "controller validation."
        ),
    }
    for year, group in ts_quality.groupby(ts_quality["timestamp"].dt.year):
        annual_rows.append(
            {
                "dataset": "tsukuba",
                "year": int(year),
                "quality_rows": int(len(group)),
                "mean_abs_battery_power_kw": float(
                    group["measured_battery_power_kw"].abs().mean()
                ),
                "mean_soc_pct": float(group["measured_battery_soc_pct"].mean()),
            }
        )

    report_path = args.output_dir / "bess_observational_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(annual_rows).to_csv(
        args.output_dir / "bess_annual_sensitivity.csv", index=False
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
