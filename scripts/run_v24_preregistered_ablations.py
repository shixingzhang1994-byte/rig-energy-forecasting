from __future__ import annotations

"""Run and audit the three preregistered V24 single-factor ablations.

The ablation method names and their implementation were frozen before V24
holdout outcomes.  These results remain exploratory because they are outside
the three-test confirmatory family.  Every retained V24 seed is processed in
queue order, including the validation-fallback seed.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from audit_v24_evidence import (  # noqa: E402
    DEFAULT_PROTOCOL,
    PROPOSED,
    paired_summary,
    retained_roots,
    sha256,
    verify_manifest_entries,
)
from run_v23_protocol_seed import audit_startup_delay  # noqa: E402
from run_v9_supervisor_first_seed import _audit_trajectories  # noqa: E402


ABLATION_METHODS = [
    "Full-Point-Scenario-Ablation-MILP",
    "Full-Fixed-CVaR-Ablation-MILP",
    "Full-No-Generator-First-Ablation-MILP",
]
COMPONENT_BY_ABLATION = {
    "Full-Point-Scenario-Ablation-MILP": "residual_scenarios",
    "Full-Fixed-CVaR-Ablation-MILP": "risk_adaptive_cvar",
    "Full-No-Generator-First-Ablation-MILP": "generator_first_execution",
}


def _command(seed: int, root: Path, output: Path, dispatch_path: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_v22_dispatch_benchmark.py",
        "--predictions",
        str(root / "source/forecast/forecast_predictions.npz"),
        "--prediction-keys",
        str(root / "source/forecast/forecast_prediction_keys.json"),
        "--config",
        str(dispatch_path),
        "--risk-signals",
        str(root / "risk/risk_predictions.csv"),
        "--scenario-residuals",
        str(root / "source/forecast/validation_residual_library.npz"),
        "--scenario-seed",
        str(seed),
        "--artifact-dir",
        str(output / "dispatch"),
    ]
    for method in ABLATION_METHODS:
        command.extend(["--method", method])
    return command


def run_seed(
    root: Path,
    output_root: Path,
    dispatch_path: Path,
    maximum_decision_seconds: float,
) -> dict:
    seed = int(root.name.split("_")[-1])
    output = output_root / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = _command(seed, root, output, dispatch_path)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    dispatch_dir = output / "dispatch"
    metrics_path = dispatch_dir / "dispatch_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    observed_methods = set(metrics["method"])
    if observed_methods != set(ABLATION_METHODS) or len(metrics) != 9:
        raise RuntimeError(
            f"seed {seed} ablation matrix incomplete: {sorted(observed_methods)}"
        )
    physical = _audit_trajectories(dispatch_dir)
    dispatch = yaml.safe_load(dispatch_path.read_text(encoding="utf-8"))
    startup = audit_startup_delay(dispatch_dir, dispatch)
    summary = metrics.groupby("method").agg(
        unserved_energy_kwh=("unserved_energy_kwh", "sum"),
        realized_operating_cost_yuan=("realized_operating_cost_yuan", "sum"),
        maximum_decision_seconds=("max_decision_seconds", "max"),
    )
    base = (
        pd.read_csv(root / "dispatch_matched/dispatch_metrics.csv")
        .loc[lambda frame: frame["method"] == PROPOSED]
        .agg(
            {
                "unserved_energy_kwh": "sum",
                "realized_operating_cost_yuan": "sum",
                "max_decision_seconds": "max",
            }
        )
    )
    contrasts = {}
    for ablation in ABLATION_METHODS:
        contrasts[COMPONENT_BY_ABLATION[ablation]] = {
            "full_minus_ablation_eens_kwh": float(
                base["unserved_energy_kwh"]
                - summary.loc[ablation, "unserved_energy_kwh"]
            ),
            "full_minus_ablation_operating_cost_yuan": float(
                base["realized_operating_cost_yuan"]
                - summary.loc[ablation, "realized_operating_cost_yuan"]
            ),
        }
    technical_checks = {
        "complete_three_method_by_three_scenario_matrix": True,
        "constraint_audit": bool(physical["physical_pass"]),
        "startup_delay_audit": bool(startup["pass"]),
        "real_time_latency": float(summary["maximum_decision_seconds"].max())
        < maximum_decision_seconds,
    }
    result = {
        "seed": seed,
        "claim_status": "preregistered_exploratory",
        "retained_regardless_of_outcome_direction": True,
        "technical_checks": technical_checks,
        "technical_gate_pass": all(technical_checks.values()),
        "contrasts": contrasts,
        "physical_audit": physical,
        "startup_delay_audit": startup,
        "reproduction_command": " ".join(command),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [metrics_path, dispatch_dir / "scenario_selection.json", result_path]
    files.extend(sorted(dispatch_dir.glob("trajectory_*.csv")))
    manifest = {
        "seed": seed,
        "claim_status": "preregistered_exploratory",
        "inputs": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": sha256(path),
            }
            for path in [
                dispatch_path,
                root / "source/forecast/forecast_predictions.npz",
                root / "source/forecast/forecast_prediction_keys.json",
                root / "source/forecast/validation_residual_library.npz",
                root / "risk/risk_predictions.csv",
            ]
        ],
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def aggregate(results: list[dict], output_root: Path, protocol_id: str) -> dict:
    rows = []
    for result in results:
        for component, values in result["contrasts"].items():
            rows.append({"seed": result["seed"], "component": component, **values})
    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(output_root / "seed_component_effects.csv", index=False)
    summaries = []
    for component, frame in seed_frame.groupby("component"):
        summary = paired_summary(frame["full_minus_ablation_eens_kwh"].to_numpy())
        summary.update(
            {
                "component": component,
                "effect_definition": "full_controller_minus_single_component_removed",
                "negative_eens_difference_means_component_improves_reliability": True,
                "mean_operating_cost_difference_yuan": float(
                    frame["full_minus_ablation_operating_cost_yuan"].mean()
                ),
                "claim_status": "preregistered_exploratory",
            }
        )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_root / "component_effect_summary.csv", index=False)
    manifest = {
        "protocol_id": protocol_id,
        "status": "pass" if all(r["technical_gate_pass"] for r in results) else "fail",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_count": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "methods": ABLATION_METHODS,
        "all_null_and_adverse_results_retained": True,
        "claim_status": "preregistered_exploratory_outside_primary_family",
        "summary_path": str(
            (output_root / "component_effect_summary.csv").relative_to(PROJECT_DIR)
        ),
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    document = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    artifact_root = PROJECT_DIR / "artifacts" / protocol["artifact_namespace"]
    mismatches = verify_manifest_entries(artifact_root / "freeze_manifest.json")
    if mismatches:
        raise RuntimeError(f"frozen V24 files changed: {mismatches}")
    roots = retained_roots(protocol, artifact_root)
    if args.seed is not None:
        roots = [root for root in roots if int(root.name.split("_")[-1]) == args.seed]
        if not roots:
            raise ValueError("seed is not one of the twelve retained V24 units")
    output_root = PROJECT_DIR / "artifacts/V24_submission_revision/preregistered_ablations"
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_seed(
            root,
            output_root,
            PROJECT_DIR / "configs/v24_dispatch_matched.yaml",
            float(document["hard_gates"]["maximum_decision_seconds"]),
        )
        for root in roots
    ]
    if args.seed is None:
        print(json.dumps(aggregate(results, output_root, protocol["id"]), indent=2))
    else:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
