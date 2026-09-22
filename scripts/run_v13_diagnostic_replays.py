from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs/v13_grid_headroom_reserve.yaml"
DIAGNOSTIC_SOURCES = {
    20260905: PROJECT_DIR / "artifacts/v10_evaluable_holdout/seed_20260905",
    20260910: PROJECT_DIR / "artifacts/v11_low_soc_commitment/seed_20260910",
    20260912: PROJECT_DIR / "artifacts/v12_causal_deficit_commitment/seed_20260912",
}


def _run_seed(seed: int, source: Path, supervisor: dict) -> dict:
    output = PROJECT_DIR / "artifacts/v13_grid_headroom_reserve" / f"diagnostic_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(
        (source / "resolved_dispatch_config.yaml").read_text(encoding="utf-8")
    )
    config["supervisory_mpc"] = supervisor
    resolved = output / "resolved_dispatch_config.yaml"
    resolved.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    dispatch_dir = output / "dispatch"
    command = [
        sys.executable,
        "scripts/run_dispatch_benchmark.py",
        "--predictions",
        str(source / "source/forecast/forecast_predictions.npz"),
        "--prediction-keys",
        str(source / "source/forecast/forecast_prediction_keys.json"),
        "--config",
        str(resolved),
        "--risk-signals",
        str(source / "risk/risk_predictions.csv"),
        "--artifact-dir",
        str(dispatch_dir),
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    metrics = pd.read_csv(dispatch_dir / "dispatch_metrics.csv")
    proposed = metrics[metrics["method"] == "Risk-SOC-Supervisory-MPC"].set_index(
        "scenario"
    )
    rule = metrics[metrics["method"] == "Rule-Based"].set_index("scenario")
    ml = metrics[metrics["method"] == "ML-Robust-MPC"].set_index("scenario")
    scenario_regret = proposed["unserved_energy_kwh"] - pd.concat(
        [rule["unserved_energy_kwh"], ml["unserved_energy_kwh"]], axis=1
    ).min(axis=1)
    reserve_actions = {}
    for scenario in proposed.index:
        trajectory = pd.read_csv(
            dispatch_dir / f"trajectory_{scenario}_risk_soc_supervisory_mpc.csv",
            encoding="utf-8-sig",
        )
        reserve_actions[scenario] = {
            "preserved_discharge_energy_kwh": float(
                trajectory["grid_headroom_storage_preservation_kw"].sum()
                * 5.0
                / 3600.0
            ),
            "charged_energy_kwh": float(
                trajectory["grid_headroom_storage_charge_kw"].sum()
                * 5.0
                / 3600.0
            ),
        }
    proposed_unserved = float(proposed["unserved_energy_kwh"].sum())
    rule_unserved = float(rule["unserved_energy_kwh"].sum())
    ml_unserved = float(ml["unserved_energy_kwh"].sum())
    cost_ratio = float(
        proposed["risk_adjusted_cost_yuan"].sum()
        / ml["risk_adjusted_cost_yuan"].sum()
    )
    return {
        "seed": seed,
        "role": "diagnostic_replay_only_not_acceptance_evidence",
        "proposed_unserved_energy_kwh": proposed_unserved,
        "rule_unserved_energy_kwh": rule_unserved,
        "ml_robust_unserved_energy_kwh": ml_unserved,
        "risk_adjusted_cost_ratio_vs_ml_robust": cost_ratio,
        "scenario_unserved_regret_kwh_vs_best_reference": {
            key: float(value) for key, value in scenario_regret.items()
        },
        "reserve_actions": reserve_actions,
        "diagnostic_gate_pass": bool(
            proposed_unserved <= min(rule_unserved, ml_unserved) + 1e-9
            and float(scenario_regret.max()) <= 0.50
            and cost_ratio <= 1.005
        ),
        "reproduction_command": " ".join(command),
    }


def main() -> None:
    protocol_document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    results = [
        _run_seed(seed, source, protocol_document["supervisor"])
        for seed, source in DIAGNOSTIC_SOURCES.items()
    ]
    summary = {
        "protocol": protocol_document["protocol"]["name"],
        "role": "failure_mechanism_diagnostic_only_not_acceptance_evidence",
        "single_change_from_v12": "causal_grid_headroom_storage_reserve_barrier",
        "results": results,
        "all_diagnostic_gates_pass": all(
            item["diagnostic_gate_pass"] for item in results
        ),
    }
    output = PROJECT_DIR / "artifacts/v13_grid_headroom_reserve/diagnostic_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
