from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_v10_protocol_seed import _verify_protocol_freeze  # noqa: E402


ROOT = PROJECT_DIR / "artifacts/v15_generator_first_reserve"
DEVELOPMENT_SEEDS = (20261008, 20261009)
HOLDOUT_SEEDS = (20261011, 20261012)
METHODS = ("Risk-SOC-Supervisory-MPC", "Rule-Based", "ML-Robust-MPC")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verified_result(seed: int) -> tuple[dict, dict]:
    seed_root = ROOT / f"seed_{seed}"
    result_path = seed_root / "result.json"
    manifest_path = seed_root / "evidence_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(result_path.relative_to(PROJECT_DIR))
    record = next(item for item in manifest["files"] if item["path"] == relative)
    digest = _sha256(result_path)
    if digest != record["sha256"]:
        raise RuntimeError(f"结果与证据清单哈希不一致: {seed}")
    if not result.get("eligible") or not result.get("overall_pass"):
        raise RuntimeError(f"种子未通过预注册硬门: {seed}")
    return result, {
        "seed": seed,
        "result_path": relative,
        "result_sha256": digest,
        "evidence_manifest_path": str(manifest_path.relative_to(PROJECT_DIR)),
        "evidence_manifest_sha256": _sha256(manifest_path),
    }


def _method_summary(seed: int) -> dict[str, dict[str, float]]:
    path = ROOT / f"seed_{seed}/method_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        method = row["method"]
        if method in METHODS:
            selected[method] = {
                "equivalent_cost_yuan": float(row["equivalent_cost_yuan"]),
                "risk_adjusted_cost_yuan": float(row["risk_adjusted_cost_yuan"]),
                "unserved_energy_kwh": float(row["unserved_energy_kwh"]),
                "generator_startups": int(row["generator_startups"]),
                "diesel_fuel_l": float(row["diesel_fuel_l"]),
                "max_decision_seconds": float(row["max_decision_seconds"]),
            }
    if set(selected) != set(METHODS):
        raise RuntimeError(f"方法汇总不完整: {seed}")
    return selected


def _supply_demand_summary(seed: int) -> dict[str, float | int]:
    path = ROOT / f"seed_{seed}/risk/risk_predictions.csv"
    margins: list[float] = []
    ratios: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            margins.append(float(row["actual_margin_kw"]))
            ratios.append(float(row["actual_margin_ratio"]))
    deficits = [max(-value, 0.0) for value in margins]
    redundancies = [max(value, 0.0) for value in margins]
    count = len(margins)
    return {
        "seed": seed,
        "sample_count": count,
        "mean_supply_margin_kw": sum(margins) / count,
        "minimum_supply_margin_kw": min(margins),
        "mean_supply_margin_ratio": sum(ratios) / count,
        "load_deficit_sample_count": sum(value > 0.0 for value in deficits),
        "load_deficit_sample_fraction": sum(value > 0.0 for value in deficits) / count,
        "mean_load_deficit_kw": sum(deficits) / count,
        "maximum_load_deficit_kw": max(deficits),
        "mean_power_redundancy_kw": sum(redundancies) / count,
        "maximum_power_redundancy_kw": max(redundancies),
    }


def main() -> None:
    _verify_protocol_freeze(HOLDOUT_SEEDS[-1], ROOT)
    results: dict[int, dict] = {}
    evidence = []
    for seed in DEVELOPMENT_SEEDS + HOLDOUT_SEEDS:
        results[seed], record = _load_verified_result(seed)
        evidence.append(record)

    method_totals = {
        method: {
            "equivalent_cost_yuan": 0.0,
            "risk_adjusted_cost_yuan": 0.0,
            "unserved_energy_kwh": 0.0,
            "generator_startups": 0,
            "diesel_fuel_l": 0.0,
            "max_decision_seconds": 0.0,
        }
        for method in METHODS
    }
    for seed in HOLDOUT_SEEDS:
        for method, metrics in _method_summary(seed).items():
            for key in (
                "equivalent_cost_yuan",
                "risk_adjusted_cost_yuan",
                "unserved_energy_kwh",
                "generator_startups",
                "diesel_fuel_l",
            ):
                method_totals[method][key] += metrics[key]
            method_totals[method]["max_decision_seconds"] = max(
                method_totals[method]["max_decision_seconds"],
                metrics["max_decision_seconds"],
            )

    proposed = method_totals["Risk-SOC-Supervisory-MPC"]
    rule = method_totals["Rule-Based"]
    robust = method_totals["ML-Robust-MPC"]
    supply_rows = [_supply_demand_summary(seed) for seed in HOLDOUT_SEEDS]
    supply_path = ROOT / "supply_demand_indicator_summary.csv"
    with supply_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(supply_rows[0]))
        writer.writeheader()
        writer.writerows(supply_rows)

    freeze_path = ROOT / "freeze_manifest.json"
    summary = {
        "protocol": "v15-generator-first-storage-reserve",
        "status": "internal_public_evidence_simulation_acceptance_pass",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "all_declared_seeds_eligible_and_passed": True,
        "freeze_manifest_path": str(freeze_path.relative_to(PROJECT_DIR)),
        "freeze_manifest_sha256": _sha256(freeze_path),
        "evidence_records": evidence,
        "holdout_results": [results[seed] for seed in HOLDOUT_SEEDS],
        "holdout_method_totals": method_totals,
        "holdout_comparison": {
            "unserved_reduction_vs_rule_kwh": (
                rule["unserved_energy_kwh"] - proposed["unserved_energy_kwh"]
            ),
            "unserved_reduction_vs_rule_pct": 100.0
            * (rule["unserved_energy_kwh"] - proposed["unserved_energy_kwh"])
            / rule["unserved_energy_kwh"],
            "unserved_reduction_vs_ml_robust_kwh": (
                robust["unserved_energy_kwh"] - proposed["unserved_energy_kwh"]
            ),
            "unserved_reduction_vs_ml_robust_pct": 100.0
            * (robust["unserved_energy_kwh"] - proposed["unserved_energy_kwh"])
            / robust["unserved_energy_kwh"],
            "risk_adjusted_cost_ratio_vs_ml_robust": (
                proposed["risk_adjusted_cost_yuan"]
                / robust["risk_adjusted_cost_yuan"]
            ),
            "equivalent_cost_reduction_vs_ml_robust_pct": 100.0
            * (robust["equivalent_cost_yuan"] - proposed["equivalent_cost_yuan"])
            / robust["equivalent_cost_yuan"],
        },
        "supply_demand_indicator_formulas": {
            "supply_margin_kw": "available_supply_kw - required_power_kw",
            "load_deficit_kw": "max(-supply_margin_kw, 0)",
            "power_redundancy_kw": "max(supply_margin_kw, 0)",
        },
        "supply_demand_indicator_summary_path": str(
            supply_path.relative_to(PROJECT_DIR)
        ),
        "supply_demand_indicator_summary_sha256": _sha256(supply_path),
        "supply_demand_indicators": supply_rows,
        "claim_boundary": {
            "field_scada_validated": False,
            "hardware_in_loop_validated": False,
            "statistical_superiority_claimed": False,
            "allowed_claim": (
                "V15在公开证据约束合成仿真中完成两开发、冻结和两按序留出硬门，"
                "对两种强参考达到安全非劣并取得描述性小幅改善。"
            ),
        },
    }
    output = ROOT / "final_acceptance_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
