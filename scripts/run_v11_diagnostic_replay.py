from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    seed = 20260905
    source = PROJECT_DIR / f"artifacts/v10_evaluable_holdout/seed_{seed}"
    output = PROJECT_DIR / f"artifacts/v11_low_soc_commitment/diagnostic_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(
        (source / "resolved_dispatch_config.yaml").read_text(encoding="utf-8")
    )
    config["supervisory_mpc"]["low_soc_minimum_committed_units"] = 2
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
    totals = metrics.groupby("method")["unserved_energy_kwh"].sum()
    old = json.loads((source / "result.json").read_text(encoding="utf-8"))
    scenario_a = pd.read_csv(
        dispatch_dir / "trajectory_A_stable_drilling_risk_soc_supervisory_mpc.csv"
    )
    floor_rows = scenario_a.index[scenario_a["soc"] <= 0.150000001]
    result = {
        "seed": seed,
        "role": "diagnostic_replay_only_not_acceptance_evidence",
        "single_change": "low_soc_minimum_committed_units=2",
        "old_proposed_unserved_energy_kwh": old["proposed_unserved_energy_kwh"],
        "new_proposed_unserved_energy_kwh": float(
            totals["Risk-SOC-Supervisory-MPC"]
        ),
        "rule_unserved_energy_kwh": float(totals["Rule-Based"]),
        "ml_robust_unserved_energy_kwh": float(totals["ML-Robust-MPC"]),
        "scenario_a_first_soc_floor_step": (
            None if len(floor_rows) == 0 else int(floor_rows[0])
        ),
        "scenario_a_low_soc_minimum_units_observed": int(
            scenario_a.loc[scenario_a["soc"] <= 0.30, "generator_units_on"].min()
        ),
        "reproduction_command": " ".join(command),
    }
    (output / "diagnostic_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
