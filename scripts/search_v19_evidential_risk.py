from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.risk.evidence_theory import fuse_risk_frame  # noqa: E402


def _candidate(index: int, base: dict, rng: np.random.Generator) -> dict:
    ranges = base["fusion"]["search_ranges"]
    draw = lambda key: float(rng.uniform(*map(float, ranges[key])))
    fixed = base["fusion"]["fixed_operational_thresholds"]
    return {
        "candidate_index": index,
        "combination_rule": base["fusion"]["combination_rule"],
        "discounts": {
            "learned_probability": draw("learned_probability_discount"),
            "margin_interval": draw("margin_interval_discount"),
            "low_soc": draw("low_soc_discount"),
            "healthy_soc": draw("healthy_soc_discount"),
            "grid_derating": draw("grid_derating_discount"),
            "rapid_ramp": draw("rapid_ramp_discount"),
        },
        "margin_thresholds": dict(base["fusion"]["fixed_margin_thresholds"]),
        "uncertainty_interval_scale": draw("uncertainty_interval_scale"),
        "soc_warning_pct": float(fixed["soc_warning_pct"]),
        "soc_healthy_pct": float(fixed["soc_healthy_pct"]),
        "grid_derating_threshold_kw": float(fixed["grid_derating_threshold_kw"]),
        "ramp_threshold_kw_per_s": float(fixed["ramp_threshold_kw_per_s"]),
        "decision": {
            "high_score_threshold": draw("high_score_threshold"),
            "severe_score_threshold": draw("severe_score_threshold"),
            "conflict_weight": draw("conflict_weight"),
        },
        "reserve_adder_kw": {0: 0.0, 1: 30.0, 2: 80.0, 3: 150.0},
    }


def _evaluate_one(payload: tuple[dict, list[tuple[int, pd.DataFrame]], dict]) -> dict:
    candidate, frames, gates = payload
    seed_rows = []
    for seed, frame in frames:
        fused = fuse_risk_frame(frame, candidate)
        truth = frame["true_risk_level"].to_numpy(int)
        prediction = fused["evidential_risk_level"].to_numpy(int)
        dispatch = fused["evidential_dispatch_risk_level"].to_numpy(int)
        seed_rows.append(
            {
                "seed": seed,
                "macro_f1": float(f1_score(truth, prediction, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                "high_risk_recall": float(recall_score(truth >= 2, dispatch >= 2, zero_division=0)),
                "severe_recall": float(recall_score(truth == 3, dispatch == 3, zero_division=0)),
                "mean_ignorance": float(fused["evidential_ignorance"].mean()),
                "mean_conflict": float(fused["evidential_conflict"].mean()),
                "dispatch_escalation_rate": float(
                    (dispatch > frame["dispatch_risk_level"].to_numpy(int)).mean()
                ),
            }
        )
    worst = {
        name: min(float(row[name]) for row in seed_rows)
        for name in ("macro_f1", "balanced_accuracy", "high_risk_recall", "severe_recall")
    }
    means = {
        name: float(np.mean([float(row[name]) for row in seed_rows]))
        for name in ("mean_ignorance", "mean_conflict", "dispatch_escalation_rate")
    }
    eligible = (
        worst["macro_f1"] >= float(gates["evidential_macro_f1_minimum"])
        and worst["high_risk_recall"] >= float(gates["dispatch_high_risk_recall_minimum"])
        and worst["severe_recall"] >= float(gates["dispatch_severe_recall_minimum"])
    )
    score = (
        0.40 * worst["macro_f1"]
        + 0.20 * worst["balanced_accuracy"]
        + 0.20 * worst["high_risk_recall"]
        + 0.20 * worst["severe_recall"]
        - 0.05 * means["dispatch_escalation_rate"]
    )
    return {
        "candidate_index": candidate["candidate_index"],
        "eligible": eligible,
        "selection_score": score,
        "worst_seed": worst,
        "means": means,
        "per_seed": seed_rows,
        "config": candidate,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/v19_evidential_risk_development.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts/v19_evidential_risk/development_search")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    frames = []
    for seed in protocol["protocol"]["development_seed_order"]:
        path = PROJECT_DIR / f"artifacts/v15_generator_first_reserve/seed_{seed}/risk/risk_predictions.csv"
        frames.append((int(seed), pd.read_csv(path)))
    rng = np.random.default_rng(int(protocol["protocol"]["candidate_generator_seed"]))
    candidates = [
        _candidate(index, protocol, rng)
        for index in range(int(protocol["protocol"]["candidate_count"]))
    ]
    gates = protocol["selection"]["require_each_development_seed"]
    payloads = [(candidate, frames, gates) for candidate in candidates]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_evaluate_one, payloads))
    eligible = [result for result in results if result["eligible"]]
    selection_pool = eligible or results
    selected = max(
        selection_pool,
        key=lambda result: (
            result["selection_score"],
            -result["means"]["mean_ignorance"],
            -result["means"]["mean_conflict"],
            -result["candidate_index"],
        ),
    )
    selected_config = dict(selected["config"])
    selected_config["selection_provenance"] = {
        "protocol": str(args.protocol.relative_to(PROJECT_DIR)),
        "protocol_sha256": _sha256(args.protocol),
        "development_seeds": protocol["protocol"]["development_seed_order"],
        "candidate_count": len(candidates),
        "selected_candidate_index": selected["candidate_index"],
        "test_or_holdout_labels_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_results.json").write_text(
        json.dumps(
            {
                "protocol": protocol["protocol"],
                "eligible_candidate_count": len(eligible),
                "selected_candidate_index": selected["candidate_index"],
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected_path = args.output_dir / "selected_evidential_config.yaml"
    selected_path.write_text(
        yaml.safe_dump(selected_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"eligible": len(eligible), "selected": selected, "config": str(selected_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
