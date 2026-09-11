from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _check(name: str, value: float, unit: str, rule: str, passed: bool, basis: str) -> dict:
    return {
        "check": name,
        "value": float(value),
        "unit": unit,
        "acceptance_rule": rule,
        "passed": bool(passed),
        "basis": basis,
    }


def _tripping_period_seconds(data: pd.DataFrame, interval_seconds: float) -> float:
    mask = data["operation_state"].eq("tripping").to_numpy()
    segment_id = np.cumsum(np.r_[True, mask[1:] != mask[:-1]])
    periods = []
    for identifier in np.unique(segment_id[mask]):
        values = data.loc[(segment_id == identifier) & mask, "drawworks_power_kw"].to_numpy()
        if len(values) < 100:
            continue
        peaks, _ = find_peaks(
            values,
            distance=max(1, int(180 / interval_seconds)),
            prominence=250.0,
        )
        if len(peaks) >= 2:
            periods.extend(np.diff(peaks) * interval_seconds)
    return float(np.median(periods)) if periods else float("nan")


def _plot_process_windows(data: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row, state in enumerate(["drilling", "tripping"]):
        indices = np.flatnonzero(data["operation_state"].eq(state).to_numpy())
        start = int(indices[0])
        stop = min(len(data), start + 720)
        view = data.iloc[start:stop]
        x = np.arange(len(view)) * 5.0 / 60.0
        axes[row, 0].plot(x, view["total_active_power_kw"], label="总功率", color="#111827")
        axes[row, 0].plot(x, view["mud_pump_power_kw"], label="泥浆泵", color="#2563EB")
        axes[row, 0].plot(x, view["topdrive_power_kw"], label="顶驱", color="#10B981")
        axes[row, 0].plot(x, view["drawworks_power_kw"], label="绞车", color="#F59E0B")
        axes[row, 0].set_title(f"{state}功率组成")
        axes[row, 0].set_ylabel("功率/kW")
        axes[row, 0].grid(alpha=0.18)
        axes[row, 0].legend(ncol=4, frameon=False, fontsize=8)
        if state == "drilling":
            axes[row, 1].plot(x, view["pump_strokes_per_min"], label="泵冲/min")
            axes[row, 1].plot(x, view["standpipe_pressure_mpa"], label="立管压力/MPa")
            axes[row, 1].plot(x, view["topdrive_rpm"], label="顶驱转速/rpm")
        else:
            axes[row, 1].plot(x, view["hookload_pct"], label="钩载/%")
            axes[row, 1].plot(x, view["hook_speed_m_per_s"] * 50.0, label="钩速×50")
        axes[row, 1].set_title(f"{state}工艺变量")
        axes[row, 1].grid(alpha=0.18)
        axes[row, 1].legend(frameon=False)
    axes[-1, 0].set_xlabel("分钟")
    axes[-1, 1].set_xlabel("分钟")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="按公开参数和机理关系验证拟真数据")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data/processed/rig_load_synthetic_v2.parquet",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "configs/synthetic_default.yaml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/v2_data"
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = pd.read_parquet(args.data)
    interval = float(data["timestamp"].diff().dt.total_seconds().median())
    equipment = config["simulation"]["equipment"]
    component_sum = data[
        [
            "auxiliary_power_kw",
            "mud_pump_power_kw",
            "topdrive_power_kw",
            "drawworks_power_kw",
            "other_power_kw",
        ]
    ].sum(axis=1)
    drilling = data[data["operation_state"] == "drilling"]
    state_mean = data.groupby("operation_state")["total_active_power_kw"].mean()
    trip_period = _tripping_period_seconds(data, interval)
    checks = [
        _check("sampling_interval", interval, "s", "=5", abs(interval - 5.0) < 1e-9, "project contract"),
        _check("component_balance_mae", (data["physical_total_power_kw"] - component_sum).abs().mean(), "kW", "<0.1", (data["physical_total_power_kw"] - component_sum).abs().mean() < 0.1, "energy balance"),
        _check("total_peak", data["total_active_power_kw"].max(), "kW", "1500—2100", 1500 <= data["total_active_power_kw"].max() <= 2100, "US8446037B2"),
        _check("idle_mean", state_mean["idle"], "kW", "300—800", 300 <= state_mean["idle"] <= 800, "published base/average ranges"),
        _check("drawworks_peak_ratio", data["drawworks_power_kw"].max() / float(equipment["drawworks_short_time_peak_kw"]), "p.u.", "<=1.0", data["drawworks_power_kw"].max() <= float(equipment["drawworks_short_time_peak_kw"]) + 1e-6, "NOV rating"),
        _check("tripping_cycle", trip_period, "s", "300—420", 300 <= trip_period <= 420, "5—7 min operating cycle"),
        _check("pump_pressure_correlation", drilling["mud_pump_power_kw"].corr(drilling["standpipe_pressure_mpa"]), "corr", ">0.70", drilling["mud_pump_power_kw"].corr(drilling["standpipe_pressure_mpa"]) > 0.70, "hydraulic mechanism"),
        _check("hardness_topdrive_correlation", drilling["formation_hardness_index"].corr(drilling["topdrive_power_kw"]), "corr", ">0.15", drilling["formation_hardness_index"].corr(drilling["topdrive_power_kw"]) > 0.15, "rock-breaking mechanism"),
        _check("hardness_rop_correlation", drilling["formation_hardness_index"].corr(drilling["rate_of_penetration_m_per_h"]), "corr", "<-0.25", drilling["formation_hardness_index"].corr(drilling["rate_of_penetration_m_per_h"]) < -0.25, "rock-breaking mechanism"),
        _check("depth_monotonicity", float((data["well_depth_m"].diff().fillna(0) >= 0).mean()), "fraction", "=1.0", data["well_depth_m"].is_monotonic_increasing, "well construction sequence"),
        _check("state_coverage", data["operation_state"].nunique(), "states", "=6", data["operation_state"].nunique() == 6, "acceptance scenarios"),
        _check("missing_fraction", data["quality_flag"].eq("imputed").mean(), "fraction", "<0.001", data["quality_flag"].eq("imputed").mean() < 0.001, "short SCADA dropout assumption"),
        _check("state_power_order", state_mean["drilling"] - state_mean["circulation"], "kW", ">0", state_mean["drilling"] > state_mean["circulation"] > state_mean["connection"], "operating mechanism"),
    ]
    result = pd.DataFrame(checks)
    result.to_csv(args.artifact_dir / "realism_validation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "checks": len(result),
        "passed": int(result["passed"].sum()),
        "failed": int((~result["passed"]).sum()),
        "all_passed": bool(result["passed"].all()),
        "data": str(args.data),
    }
    (args.artifact_dir / "realism_validation.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_process_windows(data, args.artifact_dir / "process_signal_validation.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
