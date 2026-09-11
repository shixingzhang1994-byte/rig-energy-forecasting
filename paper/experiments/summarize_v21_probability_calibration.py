from __future__ import annotations

"""Build the manuscript-side V21 table evidence from the independent audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PAPER_DIR.parent
SOURCE = (
    WORKSPACE_DIR
    / "rig-energy-forecasting/artifacts/v21_probability_calibration/independent_audit.json"
)
OUTPUT_DIR = PAPER_DIR / "experiments/results/v21_probability_calibration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["status"] != "pass_with_nonblocking_reporting_findings":
        raise RuntimeError("V21 independent audit has not passed")
    if not all(bool(value) for value in source["checks"].values()):
        raise RuntimeError("at least one frozen V21 protocol check failed")
    rows = []
    for record in source["per_seed"]:
        metrics = record["metrics"]
        rows.append(
            {
                "seed": int(record["seed"]),
                "RSS_brier": metrics["RSS"]["multiclass_brier"],
                "E_RSS_raw_yager_brier": metrics["Yager_raw"]["multiclass_brier"],
                "D_RSS_raw_dempster_brier": metrics["Dempster_raw"]["multiclass_brier"],
                "DC_RSS_calibrated_dempster_brier": metrics["Dempster_calibrated"][
                    "multiclass_brier"
                ],
                "RSS_macro_f1": metrics["RSS"]["macro_f1"],
                "DC_RSS_macro_f1": metrics["Dempster_calibrated"]["macro_f1"],
                "DC_RSS_high_risk_recall": metrics["Dempster_calibrated"][
                    "high_risk_recall"
                ],
                "DC_RSS_severe_recall": metrics["Dempster_calibrated"][
                    "severe_recall"
                ],
                "calibration_brier_delta": record["calibration_brier_delta"],
                "rule_brier_delta": record["rule_brier_delta"],
                "total_brier_delta_vs_yager": record["total_brier_delta_vs_yager"],
                "candidate_brier_delta_vs_rss": record[
                    "candidate_brier_delta_vs_rss"
                ],
                "candidate_macro_f1_delta_vs_rss": record[
                    "macro_f1_delta_vs_rss"
                ],
                "candidate_action_changed_rows_vs_rss": record["action_vs_rss"][
                    "physical_action_changed_rows"
                ],
                "candidate_action_changed_rows_vs_yager": record[
                    "action_vs_yager"
                ]["physical_action_changed_rows"],
                "RSS_unserved_kwh": record["baseline_dispatch"][
                    "unserved_energy_kwh"
                ],
                "DC_RSS_unserved_kwh": record["candidate_dispatch"][
                    "unserved_energy_kwh"
                ],
                "RSS_risk_cost_yuan": record["baseline_dispatch"][
                    "risk_adjusted_cost_yuan"
                ],
                "DC_RSS_risk_cost_yuan": record["candidate_dispatch"][
                    "risk_adjusted_cost_yuan"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    if frame["seed"].tolist() != source["eligible_seeds"]:
        raise RuntimeError("paper-side V21 seed order differs from independent audit")
    reconstructed = {
        "mean_calibration_brier_delta": float(frame["calibration_brier_delta"].mean()),
        "mean_rule_brier_delta": float(frame["rule_brier_delta"].mean()),
        "mean_total_brier_delta_vs_yager": float(
            frame["total_brier_delta_vs_yager"].mean()
        ),
        "mean_candidate_brier_delta_vs_rss": float(
            frame["candidate_brier_delta_vs_rss"].mean()
        ),
        "mean_macro_f1_delta_vs_rss": float(
            frame["candidate_macro_f1_delta_vs_rss"].mean()
        ),
    }
    expected = source["effects"]
    comparisons = {
        "mean_calibration_brier_delta": expected[
            "calibration_calibrated_minus_raw_dempster"
        ]["mean"],
        "mean_rule_brier_delta": expected[
            "conflict_rule_raw_dempster_minus_raw_yager"
        ]["mean"],
        "mean_total_brier_delta_vs_yager": expected[
            "total_calibrated_dempster_minus_raw_yager"
        ]["mean"],
        "mean_candidate_brier_delta_vs_rss": expected[
            "candidate_minus_rss_brier"
        ]["mean"],
        "mean_macro_f1_delta_vs_rss": expected[
            "candidate_minus_rss_macro_f1"
        ]["mean"],
    }
    if any(
        not np.isclose(reconstructed[key], value, rtol=0.0, atol=1e-12)
        for key, value in comparisons.items()
    ):
        raise RuntimeError("paper-side V21 effect reconstruction mismatch")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "v21_seed_metrics.csv"
    audit_path = OUTPUT_DIR / "v21_paper_audit.json"
    frame.to_csv(csv_path, index=False)
    paper_audit = {
        "analysis": "manuscript-side reconstruction of independently audited V21 evidence",
        "source": str(SOURCE.relative_to(WORKSPACE_DIR)),
        "source_sha256": sha256(SOURCE),
        "eligible_seeds": source["eligible_seeds"],
        "reconstructed_effects": reconstructed,
        "reported_effects": expected,
        "recognition": source["recognition"],
        "control": source["control"],
        "numerical_and_physical": source["numerical_and_physical"],
        "runner_reporting_issue": source["runner_reporting_issue"],
        "extended_export_precision_finding": source[
            "extended_export_precision_finding"
        ],
        "claim_boundary": source["claim_boundary"],
        "pass": True,
    }
    audit_path.write_text(
        json.dumps(paper_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = OUTPUT_DIR / "manifest.json"
    files = [SOURCE, csv_path, audit_path]
    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "V21 manuscript evidence package",
                "files": [
                    {
                        "path": str(path.relative_to(WORKSPACE_DIR)),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pass": True, "rows": len(frame), "output": str(audit_path)}))


if __name__ == "__main__":
    main()
