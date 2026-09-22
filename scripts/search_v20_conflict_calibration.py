from __future__ import annotations

"""Select one predeclared V20 rule/temperature candidate on development seeds."""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from rig_energy.risk.evidence_theory_v20 import fuse_risk_frame_v20  # noqa: E402
from run_v20_evidential_calibration import CLASS_NAMES, metrics  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs/v20_evidential_calibration_protocol.yaml"
DEVELOPMENT_FREEZE_PATH = (
    PROJECT_DIR / "artifacts/v20_evidential_calibration/development_freeze.json"
)
PARENT_CONFIG_PATH = (
    PROJECT_DIR
    / "artifacts/v19_evidential_risk/development_search/selected_evidential_config.yaml"
)
SOURCE_ROOT = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_source"
OUTPUT_ROOT = PROJECT_DIR / "artifacts/v20_evidential_calibration/development_search"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_development_freeze() -> dict[str, object]:
    freeze = json.loads(DEVELOPMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    changed = []
    for item in freeze["files"]:
        path = PROJECT_DIR / item["path"]
        if not path.exists() or sha256(path) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise RuntimeError("V20 development freeze changed: " + ", ".join(changed))
    return freeze


def eligible_development_seeds(protocol: dict) -> list[int]:
    seeds: list[int] = []
    for seed in protocol["protocol"]["development_seed_order"]:
        path = SOURCE_ROOT / f"seed_{int(seed)}" / "eligibility.json"
        if not path.exists():
            raise FileNotFoundError(f"development eligibility is missing: {path}")
        eligibility = json.loads(path.read_text(encoding="utf-8"))
        if bool(eligibility["eligible"]):
            seeds.append(int(seed))
        if len(seeds) == int(protocol["protocol"]["required_eligible_development_seeds"]):
            break
    required = int(protocol["protocol"]["required_eligible_development_seeds"])
    if len(seeds) != required:
        raise RuntimeError(f"only {len(seeds)}/{required} eligible development seeds")
    return seeds


def candidate_config(parent: dict, rule: str, temperature: float) -> dict:
    config = deepcopy(parent)
    config.pop("selection_provenance", None)
    config["combination_rule"] = rule
    config["calibration"] = {
        "method": "scalar_temperature_power",
        "temperature": float(temperature),
        "fit_scope": "V20_development_seeds_only",
        "dispatch_uses": "raw_pignistic_probability",
    }
    return config


def evaluate_seed(seed: int, config: dict) -> dict[str, object]:
    path = SOURCE_ROOT / f"seed_{seed}" / "risk/risk_predictions.csv"
    source = pd.read_csv(path)
    fused = fuse_risk_frame_v20(source, config)
    truth = source["true_risk_level"].to_numpy(int)
    baseline_probability = source[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(float)
    calibrated_probability = fused[
        [f"v20_calibrated_prob_{name}" for name in CLASS_NAMES]
    ].to_numpy(float)
    baseline = metrics(
        truth, source["predicted_risk_level"].to_numpy(int), baseline_probability
    )
    calibrated = metrics(
        truth,
        fused["v20_calibrated_risk_level"].to_numpy(int),
        calibrated_probability,
    )
    dispatch = metrics(
        truth, fused["v20_dispatch_risk_level"].to_numpy(int), calibrated_probability
    )
    return {
        "seed": seed,
        "baseline": baseline,
        "calibrated": calibrated,
        "dispatch": dispatch,
        "macro_f1_gain_vs_baseline": calibrated["macro_f1"] - baseline["macro_f1"],
        "brier_delta_vs_baseline": calibrated["multiclass_brier"]
        - baseline["multiclass_brier"],
        "mean_conflict": float(fused["v20_conflict"].mean()),
        "mean_ignorance": float(fused["v20_ignorance"].mean()),
        "calibration_changed_argmax_rows": int(
            fused["v20_calibration_changed_argmax"].sum()
        ),
        "total_conflict_fallback_rows": int(
            fused["v20_total_conflict_fallback"].sum()
        ),
    }


def candidate_passes(rows: list[dict[str, object]], gates: dict) -> bool:
    return all(
        float(row["macro_f1_gain_vs_baseline"])
        >= float(gates["macro_f1_gain_vs_frozen_RSS_minimum"])
        and float(row["dispatch"]["high_risk_recall"])
        >= float(gates["high_risk_recall_minimum"])
        and float(row["dispatch"]["severe_recall"])
        >= float(gates["severe_recall_minimum"])
        and float(row["brier_delta_vs_baseline"])
        <= float(gates["calibrated_brier_minus_frozen_RSS_maximum"])
        and int(row["calibration_changed_argmax_rows"])
        <= int(gates["calibration_changed_argmax_rows_maximum"])
        and int(row["total_conflict_fallback_rows"])
        <= int(gates["total_conflict_fallback_rows_maximum"])
        for row in rows
    )


def run() -> dict[str, object]:
    freeze = verify_development_freeze()
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = yaml.safe_load(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = eligible_development_seeds(protocol)
    rules = [str(value) for value in protocol["candidate_space"]["combination_rules"]]
    temperatures = [
        float(value) for value in protocol["candidate_space"]["scalar_temperature_grid"]
    ]
    gates = protocol["development_selection"]["hard_gates_each_development_seed"]
    candidates: list[dict[str, object]] = []
    index = 0
    for rule_index, rule in enumerate(rules):
        for temperature_index, temperature in enumerate(temperatures):
            config = candidate_config(parent, rule, temperature)
            rows = [evaluate_seed(seed, config) for seed in seeds]
            candidate = {
                "candidate_index": index,
                "rule_index": rule_index,
                "temperature_index": temperature_index,
                "combination_rule": rule,
                "temperature": temperature,
                "eligible": candidate_passes(rows, gates),
                "mean_calibrated_brier": float(
                    np.mean([row["calibrated"]["multiclass_brier"] for row in rows])
                ),
                "worst_brier_delta_vs_baseline": float(
                    max(row["brier_delta_vs_baseline"] for row in rows)
                ),
                "mean_calibrated_log_loss": float(
                    np.mean([row["calibrated"]["log_loss"] for row in rows])
                ),
                "per_seed": rows,
            }
            candidates.append(candidate)
            index += 1
    expected = int(protocol["candidate_space"]["candidate_count"])
    if len(candidates) != expected:
        raise RuntimeError(f"candidate count differs from protocol: {len(candidates)} != {expected}")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "candidate_results.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError("no V20 candidate passed every predeclared development gate")
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["mean_calibrated_brier"],
            candidate["worst_brier_delta_vs_baseline"],
            candidate["mean_calibrated_log_loss"],
            candidate["rule_index"],
            candidate["temperature_index"],
        ),
    )
    selected_config = candidate_config(
        parent, str(selected["combination_rule"]), float(selected["temperature"])
    )
    selected_config["v20_selection_provenance"] = {
        "protocol_id": protocol["protocol"]["id"],
        "development_freeze_sha256": sha256(DEVELOPMENT_FREEZE_PATH),
        "development_seeds": seeds,
        "candidate_count": len(candidates),
        "selected_candidate_index": selected["candidate_index"],
        "v19_or_v20_holdout_labels_used": False,
    }
    summary = {
        "protocol_id": protocol["protocol"]["id"],
        "development_freeze_sha256": sha256(DEVELOPMENT_FREEZE_PATH),
        "development_seeds": seeds,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected": selected,
        "selection_used_holdout_labels": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "candidate_results.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "selected_v20_config.yaml").write_text(
        yaml.safe_dump(selected_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
