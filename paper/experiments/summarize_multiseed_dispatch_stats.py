#!/usr/bin/env python3
"""Descriptive paired statistics for the frozen V15 runs.

The two development seeds and two frozen holdout seeds are reported separately
and together.  The combined four-seed result is exploratory: development seeds
are not treated as independent confirmatory test replicates.  A second view
reports the same contrasts on the 12 paired seed-by-scenario observations,
while retaining the clustering by seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


METHODS = ("Risk-SOC-Supervisory-MPC", "Rule-Based", "ML-Robust-MPC")
SEED_PHASE = {
    "20261008": "development",
    "20261009": "development",
    "20261011": "holdout",
    "20261012": "holdout",
}
SCENARIOS = ("A_stable_drilling", "B_connection_transition", "C_tripping_impact")
TRAJECTORY_SUFFIX = {
    "Risk-SOC-Supervisory-MPC": "risk_soc_supervisory_mpc",
    "Rule-Based": "rule_based",
    "ML-Robust-MPC": "ml_robust_mpc",
}


def read_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["method"] in METHODS:
            result[row["method"]] = {key: float(value) for key, value in row.items() if key != "method"}
    missing = set(METHODS) - set(result)
    if missing:
        raise ValueError(f"{path} missing methods: {sorted(missing)}")
    return result


def read_trajectory_unserved(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "unserved_kw" not in rows[0]:
        raise ValueError(f"{path} missing unserved_kw")
    return float(sum(float(row["unserved_kw"]) for row in rows) * 5.0 / 3600.0)


def read_causal_results(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["scenario"]): float(row["unserved_energy_kwh"])
        for row in payload["trajectories"]
    }


def percentile_bootstrap(values: Sequence[float], rng: np.random.Generator, draws: int = 20000) -> List[float]:
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize(values: Sequence[float], rng: np.random.Generator) -> Dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd_sample": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "bootstrap_95ci_mean": percentile_bootstrap(array, rng),
        "nonpositive_count": int(np.sum(array <= 0.0)),
    }


def build_rows(root: Path, seeds: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    causal_root = Path(__file__).resolve().parent / "results" / "independent_causal_full_milp"
    for seed in seeds:
        methods = read_summary(root / f"seed_{seed}" / "method_summary.csv")
        proposed = methods["Risk-SOC-Supervisory-MPC"]
        rule = methods["Rule-Based"]
        ml = methods["ML-Robust-MPC"]
        causal = json.loads((causal_root / f"seed_{seed}.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "phase": SEED_PHASE[seed],
                "proposed_unserved_kwh": proposed["unserved_energy_kwh"],
                "rule_unserved_kwh": rule["unserved_energy_kwh"],
                "ml_robust_unserved_kwh": ml["unserved_energy_kwh"],
                "causal_full_milp_unserved_kwh": causal["aggregate"]["total_unserved_energy_kwh"],
                "delta_unserved_vs_rule_kwh": proposed["unserved_energy_kwh"] - rule["unserved_energy_kwh"],
                "delta_unserved_vs_ml_kwh": proposed["unserved_energy_kwh"] - ml["unserved_energy_kwh"],
                "delta_unserved_vs_causal_full_milp_kwh": proposed["unserved_energy_kwh"] - causal["aggregate"]["total_unserved_energy_kwh"],
                "risk_adjusted_cost_ratio_vs_ml": proposed["risk_adjusted_cost_yuan"] / ml["risk_adjusted_cost_yuan"],
                "max_decision_seconds": proposed["max_decision_seconds"],
            }
        )
    return rows


def build_scenario_rows(root: Path, seeds: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    causal_root = Path(__file__).resolve().parent / "results" / "independent_causal_full_milp"
    for seed in seeds:
        seed_root = root / f"seed_{seed}" / "dispatch"
        causal = read_causal_results(causal_root / f"seed_{seed}.json")
        values: Dict[str, Dict[str, float]] = {}
        for method, suffix in TRAJECTORY_SUFFIX.items():
            values[method] = {
                scenario: read_trajectory_unserved(
                    seed_root / f"trajectory_{scenario}_{suffix}.csv"
                )
                for scenario in SCENARIOS
            }
        for scenario in SCENARIOS:
            proposed = values["Risk-SOC-Supervisory-MPC"][scenario]
            rule = values["Rule-Based"][scenario]
            ml = values["ML-Robust-MPC"][scenario]
            rows.append(
                {
                    "seed": seed,
                    "phase": SEED_PHASE[seed],
                    "scenario": scenario,
                    "proposed_unserved_kwh": proposed,
                    "rule_unserved_kwh": rule,
                    "ml_robust_unserved_kwh": ml,
                    "causal_full_milp_unserved_kwh": causal[scenario],
                    "delta_unserved_vs_rule_kwh": proposed - rule,
                    "delta_unserved_vs_ml_kwh": proposed - ml,
                    "delta_unserved_vs_causal_full_milp_kwh": proposed - causal[scenario],
                }
            )
    return rows


def group_summary(rows: Sequence[Dict[str, object]], rng: np.random.Generator) -> Dict[str, object]:
    deltas_rule = [float(row["delta_unserved_vs_rule_kwh"]) for row in rows]
    deltas_ml = [float(row["delta_unserved_vs_ml_kwh"]) for row in rows]
    deltas_causal = [float(row["delta_unserved_vs_causal_full_milp_kwh"]) for row in rows]
    ratios = [float(row["risk_adjusted_cost_ratio_vs_ml"]) for row in rows]
    return {
        "seed_count": len(rows),
        "seeds": [str(row["seed"]) for row in rows],
        "delta_unserved_vs_rule_kwh": summarize(deltas_rule, rng),
        "delta_unserved_vs_ml_kwh": summarize(deltas_ml, rng),
        "delta_unserved_vs_causal_full_milp_kwh": summarize(deltas_causal, rng),
        "risk_adjusted_cost_ratio_vs_ml": summarize([v - 1.0 for v in ratios], rng),
        "risk_adjusted_cost_ratio_vs_ml_raw": {
            "mean": float(np.mean(ratios)),
            "sd_sample": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
        },
        "max_decision_seconds": {
            "max": float(np.max([float(row["max_decision_seconds"]) for row in rows]))
        },
    }


def scenario_group_summary(rows: Sequence[Dict[str, object]], rng: np.random.Generator) -> Dict[str, object]:
    return {
        "paired_observation_count": len(rows),
        "seed_count": len({str(row["seed"]) for row in rows}),
        "scenario_count": len({str(row["scenario"]) for row in rows}),
        "interpretation": "Clustered descriptive seed-by-scenario observations; n is not an independent inferential sample size.",
        "delta_unserved_vs_rule_kwh": summarize(
            [float(row["delta_unserved_vs_rule_kwh"]) for row in rows], rng
        ),
        "delta_unserved_vs_ml_kwh": summarize(
            [float(row["delta_unserved_vs_ml_kwh"]) for row in rows], rng
        ),
        "delta_unserved_vs_causal_full_milp_kwh": summarize(
            [float(row["delta_unserved_vs_causal_full_milp_kwh"]) for row in rows], rng
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(SEED_PHASE)
    rows = build_rows(args.artifact_root, seeds)
    scenario_rows = build_scenario_rows(args.artifact_root, seeds)
    rng = np.random.default_rng(20260903)
    development = [row for row in rows if row["phase"] == "development"]
    holdout = [row for row in rows if row["phase"] == "holdout"]
    payload = {
        "protocol": {
            "seed_list": seeds,
            "bootstrap_draws": 20000,
            "bootstrap_rng_seed": 20260903,
            "interpretation": "Descriptive paired stability analysis; the four-seed pooled result is exploratory because two seeds were development runs.",
            "confirmatory_holdout_n": len(holdout),
            "scenario_level_interpretation": "The 12 seed-by-scenario rows are paired descriptive observations clustered within four seeds, not 12 independent replicates.",
        },
        "per_seed": rows,
        "development_summary": group_summary(development, rng),
        "holdout_summary": group_summary(holdout, rng),
        "all_four_seed_summary": group_summary(rows, rng),
        "development_scenario_summary": scenario_group_summary(
            [row for row in scenario_rows if row["phase"] == "development"], rng
        ),
        "holdout_scenario_summary": scenario_group_summary(
            [row for row in scenario_rows if row["phase"] == "holdout"], rng
        ),
        "all_seed_scenario_summary": scenario_group_summary(scenario_rows, rng),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_csv_path = args.output.with_name(args.output.stem + "_scenario.csv")
    with scenario_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "scenario_csv": str(scenario_csv_path), "all_four_seed_summary": payload["all_four_seed_summary"], "all_seed_scenario_summary": payload["all_seed_scenario_summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
