from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402
from run_v10_protocol_seed import _verify_protocol_freeze  # noqa: E402


SEED = 20261008
ACCEPTANCE_ROOT = PROJECT_DIR / "artifacts/v15_generator_first_reserve"
SOURCE_ROOT = ACCEPTANCE_ROOT / f"seed_{SEED}"
OUTPUT_ROOT = PROJECT_DIR / "artifacts/v15_development_sensitivity"
PROPOSED = "Risk-SOC-Supervisory-MPC"
REFERENCES = ("Rule-Based", "ML-Robust-MPC")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variants(base: dict) -> list[tuple[str, str, dict]]:
    variants: list[tuple[str, str, dict]] = []

    def add(name: str, change: str, update) -> None:
        document = deepcopy(base)
        update(document)
        variants.append((name, change, document))

    add(
        "generator_ramp_low",
        "单机5 s爬坡由30降至15 kW",
        lambda d: (
            d["plant"]["generator"].update(
                {"unit_ramp_kw_per_step": 15.0, "ramp_kw_per_step": 60.0}
            )
        ),
    )
    add(
        "generator_dwell_long",
        "最小开停时间均取300 s",
        lambda d: d["plant"]["generator"].update(
            {"minimum_up_steps": 60, "minimum_down_steps": 60}
        ),
    )
    add(
        "storage_power_low",
        "储能功率由500降至350 kW",
        lambda d: d["plant"]["storage"].update({"rated_power_kw": 350.0}),
    )
    add(
        "storage_energy_low",
        "储能能量由133降至117.5 kWh",
        lambda d: d["plant"]["storage"].update({"energy_capacity_kwh": 117.5}),
    )

    def grid_low(document: dict) -> None:
        document["plant"]["grid"]["rated_capacity_kw"] = 975.0
        for scenario in document["scenarios"].values():
            scenario["grid_capacity_kw"] = 0.75 * float(
                scenario["grid_capacity_kw"]
            )

    add(
        "grid_dynamic_unannounced_derating",
        "风险判级后动态网电可用容量整体突降25%，风险等级保持冻结",
        grid_low,
    )
    add(
        "storage_degradation_high",
        "储能退化价由0.15升至0.50元/kWh",
        lambda d: d["cost"].update(
            {"storage_degradation_yuan_per_kwh": 0.50}
        ),
    )
    add(
        "unserved_penalty_low",
        "失供惩罚由100降至50元/kWh",
        lambda d: d["cost"].update(
            {"unserved_energy_penalty_yuan_per_kwh": 50.0}
        ),
    )
    return variants


def _evaluate(metrics: pd.DataFrame, dispatch_dir: Path) -> dict:
    sum_columns = [
        "equivalent_cost_yuan",
        "risk_adjusted_cost_yuan",
        "unserved_energy_kwh",
        "generator_startups",
        "diesel_fuel_l",
    ]
    summary = metrics.groupby("method")[sum_columns].sum()
    summary["max_decision_seconds"] = metrics.groupby("method")[
        "max_decision_seconds"
    ].max()
    proposed = summary.loc[PROPOSED]
    rule = summary.loc["Rule-Based"]
    robust = summary.loc["ML-Robust-MPC"]
    scenario = metrics.pivot(
        index="scenario", columns="method", values="unserved_energy_kwh"
    )
    best_reference = scenario[list(REFERENCES)].min(axis=1)
    regret = scenario[PROPOSED] - best_reference
    physical = _audit_trajectories(dispatch_dir)
    checks = {
        "aggregate_safety_noninferiority": float(proposed["unserved_energy_kwh"])
        <= min(
            float(rule["unserved_energy_kwh"]),
            float(robust["unserved_energy_kwh"]),
        )
        + 1e-9,
        "single_scenario_regret_cap": float(regret.max()) <= 0.50,
        "risk_adjusted_cost_cap": float(proposed["risk_adjusted_cost_yuan"])
        / float(robust["risk_adjusted_cost_yuan"])
        <= 1.005,
        "real_time_latency": float(proposed["max_decision_seconds"]) < 5.0,
        "physical_feasibility": bool(physical["physical_pass"]),
    }
    return {
        "proposed_unserved_energy_kwh": float(proposed["unserved_energy_kwh"]),
        "rule_unserved_energy_kwh": float(rule["unserved_energy_kwh"]),
        "ml_robust_unserved_energy_kwh": float(robust["unserved_energy_kwh"]),
        "risk_adjusted_cost_ratio_vs_ml_robust": float(
            proposed["risk_adjusted_cost_yuan"]
            / robust["risk_adjusted_cost_yuan"]
        ),
        "scenario_regret_kwh_vs_best_reference": {
            key: float(value) for key, value in regret.items()
        },
        "maximum_scenario_regret_kwh": float(regret.max()),
        "max_decision_seconds": float(proposed["max_decision_seconds"]),
        "physical_audit": physical,
        "checks": checks,
        "all_selected_stress_gates_pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只在V15开发种子上运行预登记参数的单因素压力测试"
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    _verify_protocol_freeze(20261012, ACCEPTANCE_ROOT)
    base_path = SOURCE_ROOT / "resolved_dispatch_config.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    predictions = SOURCE_ROOT / "source/forecast/forecast_predictions.npz"
    prediction_keys = SOURCE_ROOT / "source/forecast/forecast_prediction_keys.json"
    risk_signals = SOURCE_ROOT / "risk/risk_predictions.csv"
    records = []
    for name, change, document in _variants(base):
        root = OUTPUT_ROOT / name
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "resolved_dispatch_config.yaml"
        config_path.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        dispatch_dir = root / "dispatch"
        metrics_path = dispatch_dir / "dispatch_metrics.csv"
        active_risk_signals = risk_signals
        if name == "grid_dynamic_unannounced_derating":
            risk_frame = pd.read_csv(risk_signals)
            actual_required = (
                risk_frame["firm_supply_kw"] - risk_frame["actual_margin_kw"]
            )
            forecast_required = risk_frame["firm_supply_kw"] / (
                1.0 + risk_frame["forecast_margin_ratio"]
            )
            risk_frame["grid_available_capacity_kw"] *= 0.75
            risk_frame["firm_supply_kw"] = (
                risk_frame["grid_available_capacity_kw"]
                + risk_frame["generator_available_capacity_kw"]
                + risk_frame["storage_available_discharge_power_kw"]
            )
            risk_frame["actual_margin_kw"] = (
                risk_frame["firm_supply_kw"] - actual_required
            )
            risk_frame["actual_margin_ratio"] = (
                risk_frame["actual_margin_kw"] / actual_required.clip(lower=1e-9)
            )
            risk_frame["forecast_margin_ratio"] = (
                risk_frame["firm_supply_kw"] - forecast_required
            ) / forecast_required.clip(lower=1e-9)
            active_risk_signals = root / "risk_predictions_derated.csv"
            risk_frame.to_csv(active_risk_signals, index=False, encoding="utf-8-sig")
        command = [
            sys.executable,
            "scripts/run_v15_dispatch_benchmark.py",
            "--predictions",
            str(predictions),
            "--prediction-keys",
            str(prediction_keys),
            "--config",
            str(config_path),
            "--risk-signals",
            str(active_risk_signals),
            "--artifact-dir",
            str(dispatch_dir),
        ]
        if not (args.reuse_existing and metrics_path.exists()):
            print("RUN", name, change, flush=True)
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
        else:
            print("REUSE", name, flush=True)
        metrics = pd.read_csv(metrics_path)
        record = {
            "variant": name,
            "single_change": change,
            "source_seed": SEED,
            "phase": "post_acceptance_development_sensitivity",
            "config_path": str(config_path.relative_to(PROJECT_DIR)),
            "config_sha256": _sha256(config_path),
            "dispatch_metrics_path": str(metrics_path.relative_to(PROJECT_DIR)),
            "dispatch_metrics_sha256": _sha256(metrics_path),
            "risk_signals_path": str(active_risk_signals.relative_to(PROJECT_DIR)),
            "risk_signals_sha256": _sha256(active_risk_signals),
            **_evaluate(metrics, dispatch_dir),
        }
        (root / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "variant": name,
                    "pass": record["all_selected_stress_gates_pass"],
                    "proposed_unserved_energy_kwh": record[
                        "proposed_unserved_energy_kwh"
                    ],
                    "maximum_scenario_regret_kwh": record[
                        "maximum_scenario_regret_kwh"
                    ],
                    "cost_ratio": record[
                        "risk_adjusted_cost_ratio_vs_ml_robust"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary = {
        "status": "post_acceptance_development_sensitivity_complete",
        "source_seed": SEED,
        "holdout_data_used": False,
        "one_factor_at_a_time": True,
        "selected_from_predeclared_matrix": True,
        "variant_count": len(records),
        "passing_variant_count": sum(
            record["all_selected_stress_gates_pass"] for record in records
        ),
        "failing_variants": [
            record["variant"]
            for record in records
            if not record["all_selected_stress_gates_pass"]
        ],
        "records": records,
        "limitations": [
            "这是单因素开发集压力测试，不是新的留出验收。",
            "没有运行组合极端参数，不能证明联合扰动稳健。",
            "测量误差和缺失块会影响训练数据，需单独重训协议，未混入本调度敏感性。",
            "动态网电降额变体冻结原风险等级，用于模拟判级后突发降额，不等同于重训后的稳态网电缩放。",
            "稳健性区间不是特定现场置信区间。",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / "sensitivity_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "status", "variant_count", "passing_variant_count", "failing_variants"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
