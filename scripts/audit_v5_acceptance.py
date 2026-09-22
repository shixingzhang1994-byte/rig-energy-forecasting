from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR / "artifacts/v5_acceptance_rework"


def main() -> None:
    checks: list[dict] = []

    def add(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    arrays = np.load(ROOT / "forecast/forecast_predictions.npz")
    required_arrays = {
        "y_true",
        "future_transition_flags",
        "history_transition_flags",
        "operation_state_codes",
    }
    add(
        "预测产物含作业工况与拆分后的切换元数据",
        required_arrays.issubset(arrays.files),
        {"required": sorted(required_arrays), "actual": sorted(arrays.files)},
    )
    forecast_metadata = json.loads(
        (ROOT / "forecast/run_metadata.json").read_text(encoding="utf-8")
    )
    transition_definition = forecast_metadata["comparison_protocol"].get(
        "transition_metric_definition", ""
    )
    add(
        "切换评价只使用未来预测时域",
        "future forecast horizon only" in transition_definition,
        transition_definition,
    )

    forecast_metrics = pd.read_csv(ROOT / "forecast/benchmark_metrics.csv")
    project = forecast_metrics.loc[
        forecast_metrics["model"] == "Validation-Weighted-Ensemble"
    ].iloc[0]
    modern = forecast_metrics.loc[
        forecast_metrics["model"].isin(["PatchTST", "iTransformer"])
    ].sort_values("mae_kw").iloc[0]
    add(
        "项目集成总体MAE不劣于最强现代对手",
        float(project["mae_kw"]) <= float(modern["mae_kw"]),
        {
            "project_mae_kw": float(project["mae_kw"]),
            "comparator": str(modern["model"]),
            "comparator_mae_kw": float(modern["mae_kw"]),
        },
    )
    add(
        "项目集成未来切换MAE不劣于最强现代对手",
        float(project["transition_mae_kw"])
        <= float(modern["transition_mae_kw"]),
        {
            "project_transition_mae_kw": float(project["transition_mae_kw"]),
            "comparator": str(modern["model"]),
            "comparator_transition_mae_kw": float(modern["transition_mae_kw"]),
        },
    )

    risk_predictions = pd.read_csv(ROOT / "risk/risk_predictions.csv")
    supply_types = sorted(risk_predictions["supply_source_type"].unique().tolist())
    add(
        "风险与调度使用独立供给过程",
        supply_types == ["synthetic_independent_supply"],
        supply_types,
    )
    class_distribution = pd.read_csv(ROOT / "risk/class_distribution.csv")
    missing_classes = {}
    for split, frame in class_distribution.groupby("split"):
        missing = frame.loc[frame["sample_count"] <= 0, "risk_level"].astype(int).tolist()
        if missing:
            missing_classes[str(split)] = missing
    add("风险时间切分均覆盖四级风险", not missing_classes, missing_classes)

    dispatch_config = yaml.safe_load(
        (ROOT / "resolved_dispatch_config.yaml").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (ROOT / "dispatch/scenario_selection.json").read_text(encoding="utf-8")
    )
    specs = dispatch_config["scenario_selection"]["scenarios"]
    add(
        "调度场景按钻井作业而非供给等级选取",
        selection.get("selection_mode") == "operation_aligned",
        selection.get("selection_mode"),
    )
    selected_scenarios = selection["scenarios"]
    purity_failures = {}
    for name, spec in specs.items():
        evidence = selected_scenarios.get(name, {})
        failures = []
        if float(evidence.get("primary_fraction", -1.0)) + 1e-12 < float(
            spec.get("minimum_primary_fraction", 0.0)
        ):
            failures.append("primary_fraction")
        if spec.get("anchor_states") and float(
            evidence.get("anchor_fraction", -1.0)
        ) + 1e-12 < float(spec.get("minimum_anchor_fraction", 0.0)):
            failures.append("anchor_fraction")
        if spec.get("acceptable_supply_regimes") and float(
            evidence.get("acceptable_supply_fraction", -1.0)
        ) + 1e-12 < float(spec.get("minimum_supply_fraction", 0.0)):
            failures.append("acceptable_supply_fraction")
        if float(evidence.get("emergency_supply_fraction", 0.0)) > float(
            spec.get("maximum_emergency_fraction", 1.0)
        ) + 1e-12:
            failures.append("emergency_supply_fraction")
        if failures:
            purity_failures[name] = failures
    add(
        "三个作业场景均通过预声明工况与供给纯度门槛",
        set(selected_scenarios) == set(specs) and not purity_failures,
        {
            "selected": sorted(selected_scenarios),
            "required": sorted(specs),
            "failures": purity_failures,
        },
    )

    metrics = pd.read_csv(ROOT / "dispatch/dispatch_metrics.csv")
    expected_methods = 6
    add(
        "每个作业场景均完成六种调度方法对比",
        metrics.groupby("scenario")["method"].nunique().eq(expected_methods).all()
        and metrics["scenario"].nunique() == 3,
        metrics.groupby("scenario")["method"].nunique().to_dict(),
    )

    minimum_stable = float(
        dispatch_config["plant"]["generator"]["minimum_stable_power_kw"]
    )
    soc_min = float(dispatch_config["plant"]["storage"]["soc_min"])
    soc_max = float(dispatch_config["plant"]["storage"]["soc_max"])
    trajectory_files = sorted((ROOT / "dispatch").glob("trajectory_*.csv"))
    physical = {
        "trajectory_count": len(trajectory_files),
        "minimum_positive_generator_kw": None,
        "simultaneous_storage_steps": 0,
        "maximum_balance_error_kw": 0.0,
        "generator_capacity_violations": 0,
        "storage_capacity_violations": 0,
        "soc_violations": 0,
    }
    positive_generator = []
    for path in trajectory_files:
        frame = pd.read_csv(path)
        positive_generator.extend(
            frame.loc[frame["generator_kw"] > 1e-6, "generator_kw"].tolist()
        )
        physical["simultaneous_storage_steps"] += int(
            ((frame["charge_kw"] > 1e-6) & (frame["discharge_kw"] > 1e-6)).sum()
        )
        balance = (
            frame["grid_kw"]
            + frame["generator_kw"]
            + frame["storage_kw"]
            + frame["unserved_kw"]
            - frame["spill_kw"]
            - frame["load_kw"]
        )
        physical["maximum_balance_error_kw"] = max(
            physical["maximum_balance_error_kw"], float(balance.abs().max())
        )
        physical["generator_capacity_violations"] += int(
            (
                frame["generator_kw"]
                > frame["generator_available_capacity_kw"] + 1e-6
            ).sum()
        )
        physical["storage_capacity_violations"] += int(
            (
                frame[["charge_kw", "discharge_kw"]].max(axis=1)
                > frame["storage_available_power_kw"] + 1e-6
            ).sum()
        )
        physical["soc_violations"] += int(
            ((frame["soc"] < soc_min - 1e-8) | (frame["soc"] > soc_max + 1e-8)).sum()
        )
    if positive_generator:
        physical["minimum_positive_generator_kw"] = float(min(positive_generator))
    generator_valid = not positive_generator or min(positive_generator) >= minimum_stable - 1e-6
    physical_valid = (
        len(trajectory_files) == 18
        and generator_valid
        and physical["simultaneous_storage_steps"] == 0
        and physical["maximum_balance_error_kw"] <= 1e-5
        and physical["generator_capacity_violations"] == 0
        and physical["storage_capacity_violations"] == 0
        and physical["soc_violations"] == 0
    )
    add("全部调度轨迹通过物理可行性审计", physical_valid, physical)

    aggregate = metrics.groupby("method")[[
        "equivalent_cost_yuan",
        "risk_adjusted_cost_yuan",
        "unserved_energy_kwh",
        "solve_seconds",
    ]].sum()
    proposed_name = "Risk-Adaptive-CVaR-MPC"
    fixed_name = "Scenario-CVaR-MPC"
    proposed = aggregate.loc[proposed_name]
    fixed = aggregate.loc[fixed_name]
    add(
        "风险自适应CVaR总体不劣于固定CVaR消融基线",
        float(proposed["equivalent_cost_yuan"])
        <= float(fixed["equivalent_cost_yuan"]) + 1e-9
        and float(proposed["risk_adjusted_cost_yuan"])
        <= float(fixed["risk_adjusted_cost_yuan"]) + 1e-9,
        {
            "proposed": proposed.to_dict(),
            "fixed_cvar": fixed.to_dict(),
        },
    )
    conventional_names = ["Rule-Based", "Persistence-MPC", "ML-Robust-MPC"]
    strongest_conventional_name = str(
        aggregate.loc[conventional_names, "risk_adjusted_cost_yuan"].idxmin()
    )
    strongest_conventional = aggregate.loc[strongest_conventional_name]
    competitive_passed = (
        float(proposed["risk_adjusted_cost_yuan"])
        <= float(strongest_conventional["risk_adjusted_cost_yuan"]) + 1e-9
    )
    add(
        "项目风险自适应调度总体不劣于最强常规基线",
        competitive_passed,
        {
            "criterion": "aggregate risk_adjusted_cost_yuan",
            "proposed": proposed.to_dict(),
            "strongest_conventional_name": strongest_conventional_name,
            "strongest_conventional": strongest_conventional.to_dict(),
        },
    )

    protocol_and_physical_passed = all(
        item["passed"] for item in checks[:-2]
    )
    ablation_passed = bool(checks[-2]["passed"])
    ready_for_final_acceptance = all(item["passed"] for item in checks)
    report = {
        "version": "V5 acceptance rework",
        "scope": "synthetic evidence only; V2/V3 frozen artifacts untouched",
        "passed": ready_for_final_acceptance,
        "protocol_and_physical_passed": protocol_and_physical_passed,
        "risk_adaptive_ablation_passed": ablation_passed,
        "dispatch_competitiveness_passed": competitive_passed,
        "ready_for_final_acceptance": ready_for_final_acceptance,
        "checks": checks,
    }
    (ROOT / "acceptance_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V5 验收审计",
        "",
        f"- 最终验收结论：{'PASS' if report['passed'] else 'FAIL'}",
        f"- 协议与物理审计：{'PASS' if protocol_and_physical_passed else 'FAIL'}",
        f"- 调度竞争性：{'PASS' if competitive_passed else 'FAIL'}",
        "- 边界：仅支持合成数据上的方法和物理可行性，不替代现场SCADA验证。",
        "",
    ]
    for item in checks:
        lines.extend(
            [
                f"- [{'x' if item['passed'] else ' '}] {item['name']}",
                f"  - 证据：`{json.dumps(item['evidence'], ensure_ascii=False)}`",
            ]
        )
    (ROOT / "acceptance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["protocol_and_physical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
