from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="审计V3先进基线PK与风险调度证据")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/competitive_gates.yaml",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v3_competitive_audit",
    )
    args = parser.parse_args()
    gates = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    def check(identifier: str, passed: bool, value: object, rule: str) -> None:
        rows.append(
            {
                "check_id": identifier,
                "passed": bool(passed),
                "value": value,
                "rule": rule,
            }
        )

    robustness = json.loads(
        (
            PROJECT_DIR
            / "artifacts/v3_robustness/summary/robustness_summary.json"
        ).read_text(encoding="utf-8")
    )
    forecast_gates = gates["forecast"]
    check(
        "forecast_seed_count",
        int(robustness["seed_count"]) >= int(forecast_gates["minimum_seed_count"]),
        robustness["seed_count"],
        f">={forecast_gates['minimum_seed_count']}",
    )
    check(
        "forecast_mae_win_rate",
        float(robustness["project_mae_win_rate"])
        >= float(forecast_gates["project_mae_win_rate_minimum"]),
        robustness["project_mae_win_rate"],
        f">={forecast_gates['project_mae_win_rate_minimum']}",
    )
    check(
        "forecast_transition_win_rate",
        float(robustness["project_transition_win_rate"])
        >= float(forecast_gates["project_transition_win_rate_minimum"]),
        robustness["project_transition_win_rate"],
        f">={forecast_gates['project_transition_win_rate_minimum']}",
    )
    check(
        "forecast_worst_seed_improvement",
        float(robustness["project_worst_mae_improvement_ratio"])
        >= float(forecast_gates["worst_mae_improvement_ratio_minimum"]),
        robustness["project_worst_mae_improvement_ratio"],
        f">={forecast_gates['worst_mae_improvement_ratio_minimum']}",
    )
    forecast_meta = json.loads(
        (
            PROJECT_DIR / "artifacts/v3_competitive_forecast/run_metadata.json"
        ).read_text(encoding="utf-8")
    )
    check(
        "modern_baselines_not_in_project_ensemble",
        not bool(
            forecast_meta["comparison_protocol"][
                "modern_comparators_in_project_ensemble"
            ]
        ),
        forecast_meta["comparison_protocol"][
            "modern_comparators_in_project_ensemble"
        ],
        "must be false",
    )

    risk = pd.read_csv(
        PROJECT_DIR / "artifacts/v3_competitive_risk/risk_metrics.csv"
    )
    operational = risk.loc[risk["operational_selected"].astype(bool)].iloc[0]
    risk_gates = gates["risk"]
    check(
        "risk_macro_f1",
        float(operational["macro_f1"]) >= float(risk_gates["macro_f1_minimum"]),
        float(operational["macro_f1"]),
        f">={risk_gates['macro_f1_minimum']}",
    )
    dispatch_guard = json.loads(
        (
            PROJECT_DIR
            / "artifacts/v3_competitive_risk/dispatch_guard_metrics.json"
        ).read_text(encoding="utf-8")
    )
    for column, gate_name in (
        ("high_risk_recall", "high_risk_recall_minimum"),
        ("severe_recall", "severe_recall_minimum"),
    ):
        check(
            f"dispatch_guard_{column}",
            float(dispatch_guard[column]) >= float(risk_gates[gate_name]),
            float(dispatch_guard[column]),
            f">={risk_gates[gate_name]}",
        )
    selection = json.loads(
        (
            PROJECT_DIR
            / "artifacts/v3_competitive_risk/operational_model_selection.json"
        ).read_text(encoding="utf-8")
    )
    check(
        "risk_test_labels_excluded",
        "test labels excluded" in selection["selection_data"],
        selection["selection_data"],
        "selection/calibration must exclude test labels",
    )

    dispatch = pd.read_csv(
        PROJECT_DIR / "artifacts/v3_competitive_dispatch/dispatch_metrics.csv"
    ).set_index(["scenario", "method"])
    dispatch_gates = gates["dispatch"]
    project_method = "Risk-Adaptive-CVaR-MPC"
    comparator = "Scenario-CVaR-MPC"
    adequate = "A_supply_adequate"
    constrained = "B_constrained_impact"
    emergency = "C_emergency_supply"
    adequate_project = float(
        dispatch.loc[(adequate, project_method), "equivalent_cost_yuan"]
    )
    adequate_comparator = float(
        dispatch.loc[(adequate, comparator), "equivalent_cost_yuan"]
    )
    adequate_limit = adequate_comparator * (
        1.0
        + float(dispatch_gates["adequate_cost_degradation_tolerance_ratio"])
    )
    check(
        "dispatch_adequate_non_degradation",
        adequate_project <= adequate_limit,
        adequate_project,
        f"<={adequate_limit}",
    )
    constrained_project = float(
        dispatch.loc[(constrained, project_method), "equivalent_cost_yuan"]
    )
    constrained_comparator = float(
        dispatch.loc[(constrained, comparator), "equivalent_cost_yuan"]
    )
    constrained_improvement = 1.0 - constrained_project / constrained_comparator
    check(
        "dispatch_constrained_cost_improvement",
        constrained_improvement
        >= float(dispatch_gates["constrained_cost_improvement_minimum"]),
        constrained_improvement,
        f">={dispatch_gates['constrained_cost_improvement_minimum']}",
    )
    emergency_project = float(
        dispatch.loc[(emergency, project_method), "unserved_energy_kwh"]
    )
    emergency_comparator = float(
        dispatch.loc[(emergency, comparator), "unserved_energy_kwh"]
    )
    emergency_limit = emergency_comparator * (
        1.0
        + float(
            dispatch_gates["emergency_unserved_degradation_tolerance_ratio"]
        )
    )
    check(
        "dispatch_emergency_safety_non_degradation",
        emergency_project <= emergency_limit,
        emergency_project,
        f"<={emergency_limit}",
    )
    project_total = float(
        dispatch.xs(project_method, level="method")["risk_adjusted_cost_yuan"].sum()
    )
    comparator_total = float(
        dispatch.xs(comparator, level="method")["risk_adjusted_cost_yuan"].sum()
    )
    total_limit = comparator_total * (
        1.0
        + float(
            dispatch_gates[
                "aggregate_risk_adjusted_cost_degradation_tolerance_ratio"
            ]
        )
    )
    check(
        "dispatch_aggregate_risk_adjusted_cost",
        project_total <= total_limit,
        project_total,
        f"<={total_limit}",
    )

    trajectory_paths = sorted(
        (PROJECT_DIR / "artifacts/v3_competitive_dispatch").glob(
            "trajectory_*_risk_adaptive_cvar_mpc.csv"
        )
    )
    bounds_ok = True
    soc_ok = True
    tolerance = float(dispatch_gates["capacity_tolerance_kw"])
    for path in trajectory_paths:
        frame = pd.read_csv(path)
        bounds_ok &= bool(
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
        soc_ok &= bool(
            (soc_pct >= float(dispatch_gates["soc_minimum_pct"]) - 1e-9).all()
            and (soc_pct <= float(dispatch_gates["soc_maximum_pct"]) + 1e-9).all()
        )
    check(
        "dispatch_physical_power_bounds",
        bounds_ok and len(trajectory_paths) == 3,
        len(trajectory_paths),
        "3 scenario trajectories and all source bounds satisfied",
    )
    check(
        "dispatch_soc_bounds",
        soc_ok and len(trajectory_paths) == 3,
        len(trajectory_paths),
        f"SOC in [{dispatch_gates['soc_minimum_pct']}, {dispatch_gates['soc_maximum_pct']}]%",
    )

    output = pd.DataFrame(rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        args.artifact_dir / "competitive_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "passed": bool(output["passed"].all()),
        "passed_checks": int(output["passed"].sum()),
        "total_checks": int(len(output)),
        "failed_checks": output.loc[~output["passed"], "check_id"].tolist(),
        "scope": "competitive simulation evidence; not field validation",
    }
    (args.artifact_dir / "competitive_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
