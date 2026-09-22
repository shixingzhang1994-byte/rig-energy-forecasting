from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.schema import CODE_TO_STATE  # noqa: E402
from rig_energy.safety.emergency_load import (  # noqa: E402
    TIER_NAMES,
    allocate_priority_greedy,
    allocate_priority_highs,
    allocate_proportional,
)


def _metrics(allocation, dt_hours: float) -> dict:
    return {
        "method": allocation.method,
        "total_unserved_energy_kwh": float(allocation.shed_kw.sum() * dt_hours),
        **{
            f"{name}_unserved_energy_kwh": float(
                allocation.shed_kw[:, index].sum() * dt_hours
            )
            for index, name in enumerate(TIER_NAMES)
        },
        "maximum_critical_unserved_kw": float(allocation.shed_kw[:, 0].max()),
        "solver_metadata": allocation.solver_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对 V17 外部调度轨迹执行关键负荷保护和 HiGHS 独立复核"
    )
    parser.add_argument(
        "--dispatch-dir",
        type=Path,
        default=PROJECT_DIR
        / "artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR
        / "artifacts/v17_user_designated_external_acceptance/forecast_predictions.npz",
    )
    parser.add_argument(
        "--priority-config",
        type=Path,
        default=PROJECT_DIR / "configs/v16_expert_remediation_surrogate.yaml",
    )
    args = parser.parse_args()
    selection = json.loads(
        (args.dispatch_dir / "scenario_selection.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(args.priority_config.read_text(encoding="utf-8"))
    fractions = config["load_priority_placeholder"]["fractions_by_state"]
    arrays = np.load(args.predictions)
    state_codes = arrays["operation_state_codes"].astype(int)
    dt_hours = 5.0 / 3600.0
    rows = []
    detail = {}
    for scenario, info in selection["scenarios"].items():
        trajectory_path = args.dispatch_dir / (
            f"trajectory_{scenario}_risk_soc_supervisory_mpc.csv"
        )
        trajectory = pd.read_csv(trajectory_path)
        start = int(info["start_index"])
        stop = int(info["stop_index"])
        scenario_states = state_codes[start:stop]
        if len(scenario_states) != len(trajectory):
            raise ValueError(f"{scenario}: 作业状态与调度轨迹未对齐")
        state_names = [CODE_TO_STATE.get(int(value), "unknown") for value in scenario_states]
        tier_fraction = np.stack(
            [np.asarray(fractions.get(name, fractions["unknown"]), dtype=float) for name in state_names]
        )
        demand = trajectory["load_kw"].to_numpy(float)[:, None] * tier_fraction
        available = np.maximum(
            0.0,
            trajectory["load_kw"].to_numpy(float)
            - trajectory["unserved_kw"].to_numpy(float),
        )
        proportional = allocate_proportional(demand, available)
        priority = allocate_priority_greedy(demand, available)
        highs = allocate_priority_highs(demand, available)
        equivalence = float(np.max(np.abs(priority.served_kw - highs.served_kw)))
        if equivalence > 1e-5:
            raise RuntimeError(f"{scenario}: 优先级逻辑与 HiGHS 不等价 {equivalence}")
        scenario_metrics = []
        for allocation in (proportional, priority, highs):
            item = {
                "scenario": scenario,
                **{
                    key: value
                    for key, value in _metrics(allocation, dt_hours).items()
                    if key != "solver_metadata"
                },
                "highs_equivalence_max_kw": equivalence,
            }
            rows.append(item)
            scenario_metrics.append(
                {**item, "solver_metadata": allocation.solver_metadata}
            )
        detail[scenario] = {
            "metrics": scenario_metrics,
            "state_fractions": pd.Series(state_names).value_counts(normalize=True).to_dict(),
            "priority_definition_status": config["load_priority_placeholder"][
                "approval_status"
            ],
        }
    output = args.dispatch_dir / "critical_load_audit"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "critical_load_metrics.csv", index=False, encoding="utf-8-sig")
    priority_rows = table[table["method"] == "priority-greedy"]
    result = {
        "controller": "Risk-SOC-Supervisory-MPC",
        "evaluation_type": "post_failure_critical_load_allocation_audit",
        "total_unserved_energy_kwh": float(
            priority_rows["total_unserved_energy_kwh"].sum()
        ),
        "critical_unserved_energy_kwh": float(
            priority_rows["critical_unserved_energy_kwh"].sum()
        ),
        "critical_unserved_maximum_kw": float(
            priority_rows["maximum_critical_unserved_kw"].max()
        ),
        "critical_load_gate_passed": bool(
            np.all(priority_rows["critical_unserved_energy_kwh"] <= 1e-12)
        ),
        "zero_total_unserved_gate_passed": bool(
            np.all(priority_rows["total_unserved_energy_kwh"] <= 1e-12)
        ),
        "independent_solver": "HiGHS-priority-LP",
        "highs_matches_priority_logic": bool(
            np.all(priority_rows["highs_equivalence_max_kw"] <= 1e-5)
        ),
        "load_priority_definition_status": config["load_priority_placeholder"][
            "approval_status"
        ],
        "operational_claim_allowed": False,
        "scenarios": detail,
    }
    (output / "critical_load_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
