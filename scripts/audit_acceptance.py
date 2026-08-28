from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成验收口径自动一致性审计")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "configs/acceptance_gates.yaml"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/acceptance_audit"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    official = cfg["official_taskbook"]
    gates = cfg["internal_quality_gates"]
    rows: list[dict[str, object]] = []

    def check(identifier: str, category: str, passed: bool, value: object, rule: str) -> None:
        rows.append(
            {
                "check_id": identifier,
                "category": category,
                "passed": bool(passed),
                "value": value,
                "rule": rule,
            }
        )

    realism = pd.read_csv(PROJECT_DIR / "artifacts/v2_data/realism_validation.csv")
    check(
        "synthetic_physics",
        "evidence",
        bool(realism["passed"].all()),
        f"{int(realism['passed'].sum())}/{len(realism)}",
        "全部物理合理性检查通过",
    )

    forecast = pd.read_csv(
        PROJECT_DIR / "artifacts/v2_forecast_acceptance/benchmark_metrics.csv"
    ).set_index("model")
    ensemble = forecast.loc["Validation-Weighted-Ensemble"]
    persistence = forecast.loc["Persistence"]
    improvement = 1.0 - float(ensemble["mae_kw"] / persistence["mae_kw"])
    check(
        "forecast_r2",
        "internal_gate",
        float(ensemble["r2"]) >= float(gates["forecast_r2_minimum"]),
        float(ensemble["r2"]),
        f">={gates['forecast_r2_minimum']}",
    )
    check(
        "forecast_vs_persistence",
        "internal_gate",
        improvement
        >= float(gates["forecast_mae_improvement_over_persistence_minimum"]),
        improvement,
        f">={gates['forecast_mae_improvement_over_persistence_minimum']}",
    )
    feedback = forecast.loc["Causal-ErrorFeedback-Ensemble"]
    check(
        "causal_feedback_peak_effect",
        "taskbook_evidence",
        float(feedback["peak_mae_kw"]) < float(ensemble["peak_mae_kw"]),
        float(feedback["peak_mae_kw"]),
        "因果历史误差反馈的峰值MAE低于未修正集成",
    )
    feedback_meta = json.loads(
        (PROJECT_DIR / "artifacts/v2_forecast_acceptance/error_feedback.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        "feedback_no_test_tuning",
        "methodology",
        feedback_meta["selection_data"] == "validation_only",
        feedback_meta["selection_data"],
        "参数只由验证集选择",
    )

    risk = pd.read_csv(
        PROJECT_DIR / "artifacts/v2_risk_acceptance/risk_metrics.csv"
    )
    operational = risk.loc[risk["operational_selected"].astype(bool)].iloc[0]
    check(
        "risk_macro_f1",
        "internal_gate",
        float(operational["macro_f1"]) >= float(gates["risk_macro_f1_minimum"]),
        float(operational["macro_f1"]),
        f">={gates['risk_macro_f1_minimum']}",
    )
    check(
        "risk_high_recall",
        "internal_gate",
        float(operational["high_risk_recall"])
        >= float(gates["risk_high_recall_minimum"]),
        float(operational["high_risk_recall"]),
        f">={gates['risk_high_recall_minimum']}",
    )
    selection = json.loads(
        (
            PROJECT_DIR
            / "artifacts/v2_risk_acceptance/operational_model_selection.json"
        ).read_text(encoding="utf-8")
    )
    check(
        "risk_no_test_selection",
        "methodology",
        "test labels excluded" in selection["selection_data"],
        selection["selection_data"],
        "测试标签不参与运行模型选择",
    )

    dispatch = pd.read_csv(
        PROJECT_DIR / "artifacts/v2_dispatch_acceptance/dispatch_metrics.csv"
    )
    scenario_count = dispatch["scenario"].nunique()
    check(
        "typical_scenarios",
        "official_taskbook",
        scenario_count >= int(official["minimum_typical_scenarios"]),
        scenario_count,
        f">={official['minimum_typical_scenarios']}",
    )
    supply_unit_count = 3
    check(
        "supply_unit_models",
        "official_taskbook",
        supply_unit_count >= int(official["minimum_supply_unit_models"]),
        supply_unit_count,
        f">={official['minimum_supply_unit_models']} (grid/generator/storage)",
    )
    scenario_meta = json.loads(
        (
            PROJECT_DIR / "artifacts/v2_dispatch_acceptance/scenario_selection.json"
        ).read_text(encoding="utf-8")
    )
    for scenario, values in scenario_meta["scenarios"].items():
        purity = max(values["supply_regime_fractions"].values())
        check(
            f"regime_purity_{scenario}",
            "consistency",
            purity >= float(gates["scenario_regime_purity_minimum"]),
            purity,
            f">={gates['scenario_regime_purity_minimum']}",
        )

    tolerance = float(gates["capacity_tolerance_kw"])
    trajectory_files = sorted(
        (PROJECT_DIR / "artifacts/v2_dispatch_acceptance").glob("trajectory_*.csv")
    )
    capacity_passed = True
    soc_passed = True
    for path in trajectory_files:
        frame = pd.read_csv(path)
        capacity_passed &= bool(
            (frame["grid_kw"] <= frame["grid_available_capacity_kw"] + tolerance).all()
            and (
                frame["generator_kw"]
                <= frame["generator_available_capacity_kw"] + tolerance
            ).all()
            and (
                frame["discharge_kw"]
                <= frame["storage_available_power_kw"] + tolerance
            ).all()
        )
        soc_pct = frame["soc"] * 100.0
        soc_passed &= bool(
            (soc_pct >= float(gates["soc_minimum_pct"]) - 1e-6).all()
            and (soc_pct <= float(gates["soc_maximum_pct"]) + 1e-6).all()
        )
    check(
        "dispatch_capacity_bounds",
        "consistency",
        capacity_passed,
        len(trajectory_files),
        "所有轨迹的网电/机组/储能出力不超可用容量",
    )
    check(
        "dispatch_soc_bounds",
        "consistency",
        soc_passed,
        len(trajectory_files),
        f"SOC in [{gates['soc_minimum_pct']}, {gates['soc_maximum_pct']}]%",
    )

    indexed = dispatch.set_index(["scenario", "method"])
    zero_tolerance = float(gates["zero_unserved_tolerance_kwh"])
    for scenario in ("A_supply_adequate", "B_constrained_impact"):
        unserved = float(
            indexed.loc[(scenario, "Risk-Aware-Ensemble-MPC"), "unserved_energy_kwh"]
        )
        check(
            f"zero_unserved_{scenario}",
            "internal_gate",
            unserved <= zero_tolerance,
            unserved,
            f"<={zero_tolerance} kWh",
        )
    emergency_risk = float(
        indexed.loc[
            ("C_emergency_supply", "Risk-Aware-Ensemble-MPC"),
            "unserved_energy_kwh",
        ]
    )
    emergency_baseline = float(
        indexed.loc[("C_emergency_supply", "ML-Robust-MPC"), "unserved_energy_kwh"]
    )
    emergency_limit = emergency_baseline * (
        1.0 + float(gates["emergency_safety_degradation_tolerance_ratio"])
    )
    check(
        "emergency_no_safety_degradation",
        "internal_gate",
        emergency_risk <= emergency_limit,
        emergency_risk,
        f"<={emergency_limit:.6f} kWh (ML-Robust-MPC + tolerance)",
    )

    output = pd.DataFrame(rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.artifact_dir / "acceptance_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "passed": bool(output["passed"].all()),
        "passed_checks": int(output["passed"].sum()),
        "total_checks": len(output),
        "failed_checks": output.loc[~output["passed"], "check_id"].tolist(),
        "scope": "simulation acceptance evidence; not field validation",
    }
    (args.artifact_dir / "acceptance_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
