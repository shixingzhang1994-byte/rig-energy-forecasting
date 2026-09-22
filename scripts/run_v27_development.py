from __future__ import annotations

"""Technical development replay for V27 before the prospective freeze."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v23_protocol_seed as common  # noqa: E402
from run_v27_protocol_seed import (  # noqa: E402
    PROTOCOL_PATH,
    parameter_record,
    prepare_inputs,
)
from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402


SEED = 20261100
ROOT = PROJECT_DIR / "artifacts/V27_heterogeneous_validation/development/seed_20261100"


def main() -> None:
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    record, _ = parameter_record(SEED)
    dispatch_path = PROJECT_DIR / record["dispatch_path"]
    dispatch = yaml.safe_load(dispatch_path.read_text(encoding="utf-8"))
    ROOT.mkdir(parents=True, exist_ok=True)
    forecast_dir, risk_signals, risk_metadata_path, residuals, calibration_path = prepare_inputs(
        SEED, int(protocol["simulation_days"]), ROOT, True, PROTOCOL_PATH
    )
    risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
    transition = risk_metadata.get("transition_feature", {})
    if transition.get("source") != "history_transition_flags" or not bool(
        transition.get("future_transition_flags_excluded")
    ):
        raise RuntimeError("V27 development risk inputs failed the causal gate")

    eligibility = common.legacy_runner._eligibility(
        forecast_dir / "forecast_predictions.npz",
        risk_signals,
        dispatch,
        SEED,
        "development",
    )
    eligibility["outcomes_inspected_before_eligibility"] = False
    (ROOT / "eligibility.json").write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not eligibility["eligible"]:
        raise RuntimeError("V27 development parameter unit is upstream-ineligible")

    dispatch_dir = ROOT / "dispatch_matched"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    deadline_audit_path = dispatch_dir / "deadline_audit.json"
    command = [
        sys.executable,
        "scripts/run_v26_dispatch_benchmark.py",
        "--predictions", str(forecast_dir / "forecast_predictions.npz"),
        "--prediction-keys", str(forecast_dir / "forecast_prediction_keys.json"),
        "--config", str(dispatch_path),
        "--risk-signals", str(risk_signals),
        "--scenario-residuals", str(residuals),
        "--scenario-seed", str(SEED),
        "--artifact-dir", str(dispatch_dir),
        "--deadline-audit", str(deadline_audit_path),
    ]
    if not metrics_path.exists():
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    metrics = pd.read_csv(metrics_path)
    expected = set(document["matched_comparison"]["methods"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    deadline_audit = json.loads(deadline_audit_path.read_text(encoding="utf-8"))
    physical = _audit_trajectories(dispatch_dir)
    startup = common.audit_startup_delay(dispatch_dir, dispatch)
    proposed = metrics.loc[
        metrics["method"] == "Full-Risk-SOC-Supervisory-MILP"
    ]
    checks = {
        "eligible_before_outcomes": bool(eligibility["eligible"]),
        "complete_method_scenario_matrix": len(metrics) == len(expected) * 3
        and set(metrics["method"]) == expected,
        "validation_only_residuals": bool(calibration["leakage_gate_pass"]),
        "cvar_discretization": not bool(
            calibration["cvar_discretization_audit"]["collapses_to_single_worst_scenario"]
        ),
        "constraint_audit": bool(physical["physical_pass"]),
        "startup_delay_audit": bool(startup["pass"]),
        "real_time_latency": float(proposed["max_decision_seconds"].max())
        < float(document["hard_gates"]["maximum_decision_seconds"]),
        "deadline_audit_completed": bool(
            deadline_audit["completed_without_exception"]
        ),
        "no_rejected_deadline_incumbent": int(
            deadline_audit["rejected_deadline_count"]
        )
        == 0,
        "accepted_incumbents_independently_feasible": float(
            deadline_audit["maximum_accepted_incumbent_violation"]
        )
        <= float(
            document["hard_gates"]["maximum_incumbent_constraint_violation"]
        ),
    }
    result = {
        "seed": SEED,
        "parameter_unit": record,
        "role": "development_only_previously_revealed_seed",
        "confirmatory_evidence": False,
        "checks": checks,
        "technical_gate_pass": all(checks.values()),
        "physical_audit": physical,
        "startup_delay_audit": startup,
        "deadline_policy_audit": {
            key: deadline_audit[key]
            for key in (
                "call_count",
                "deadline_count",
                "accepted_feasible_incumbent_count",
                "rejected_deadline_count",
                "maximum_solver_elapsed_seconds",
                "maximum_accepted_incumbent_violation",
            )
        },
        "risk_validation_fallback_active": bool(
            risk_metadata.get("validation_gate_fallback_active")
        ),
        "reproduction_command": " ".join(command),
    }
    (ROOT / "development_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["technical_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
