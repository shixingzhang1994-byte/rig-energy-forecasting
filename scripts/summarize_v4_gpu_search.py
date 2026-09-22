from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_DIR / "artifacts/v4_gpu_search"
FINAL_SEEDS = (20260832, 20260833)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    initial = _read_json(ARTIFACT_ROOT / "model/holdout_summary.json")
    final = []
    expert_weights = []
    for seed in FINAL_SEEDS:
        root = ARTIFACT_ROOT / f"ensemble_seed_{seed}"
        final.append(_read_json(root / "v4_summary.json"))
        weights = _read_json(root / "ensemble_weights.json")["weights"]
        expert_weights.append(
            float(weights["StateAware-DualBranch-Patch-Transformer"])
        )

    mae_improvements = [item["mae_improvement_pct"] for item in final]
    transition_improvements = [
        item["transition_improvement_pct"] for item in final
    ]
    summary = {
        "version": "V4 GPU exploration",
        "scope": "synthetic confirmation only; V2/V3 frozen artifacts are not overwritten",
        "selection_protocol": {
            "development_seeds": [20260828, 20260829],
            "first_holdout_seeds": [20260830, 20260831],
            "final_ensemble_confirmation_seeds": list(FINAL_SEEDS),
            "test_labels_used_for_candidate_selection": False,
            "modern_comparators_in_project_ensemble": False,
        },
        "standalone_first_holdout": initial,
        "standalone_gate_passed": bool(initial["accepted"]),
        "final_ensemble_confirmation": {
            "per_seed": final,
            "mean_mae_improvement_pct": float(sum(mae_improvements) / len(final)),
            "worst_mae_improvement_pct": float(min(mae_improvements)),
            "mae_win_rate": float(
                sum(value > 0 for value in mae_improvements) / len(final)
            ),
            "mean_transition_improvement_pct": float(
                sum(transition_improvements) / len(final)
            ),
            "worst_transition_improvement_pct": float(
                min(transition_improvements)
            ),
            "transition_win_rate": float(
                sum(value > 0 for value in transition_improvements) / len(final)
            ),
            "expert_weight_mean": float(sum(expert_weights) / len(final)),
            "expert_weight_min": float(min(expert_weights)),
            "accepted": bool(
                min(mae_improvements) > 0 and min(transition_improvements) > 0
            ),
        },
        "claim_boundary": [
            "The first standalone holdout gate failed because mean MAE degraded by 1.45%; this result is retained.",
            "The later two-seed result supports the validation-weighted project ensemble on synthetic data only.",
            "No field-SCADA generalization, universal SOTA, or dispatch improvement is established by V4.",
        ],
    }
    summary_dir = ARTIFACT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "aggregate_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evidence_files = [
        PROJECT_DIR / "configs/v4_gpu_search.yaml",
        PROJECT_DIR / "configs/synthetic_default.yaml",
        PROJECT_DIR / "scripts/search_state_patch_gpu.py",
        PROJECT_DIR / "scripts/run_v4_expert_ensemble.py",
        PROJECT_DIR / "scripts/summarize_v4_gpu_search.py",
        PROJECT_DIR / "src/rig_energy/models/neural.py",
        PROJECT_DIR / "src/rig_energy/training.py",
        ARTIFACT_ROOT / "model/protocol.json",
        ARTIFACT_ROOT / "model/development_trials.csv",
        ARTIFACT_ROOT / "model/development_ranking.csv",
        ARTIFACT_ROOT / "model/selection.json",
        ARTIFACT_ROOT / "model/holdout_metrics.csv",
        ARTIFACT_ROOT / "model/holdout_summary.json",
        summary_path,
    ]
    for seed in FINAL_SEEDS:
        root = ARTIFACT_ROOT / f"ensemble_seed_{seed}"
        evidence_files.extend(
            [
                root / "resolved_config.yaml",
                root / "benchmark_metrics.csv",
                root / "ensemble_weights.json",
                root / "run_metadata.json",
                root / "training_log.json",
                root / "v4_summary.json",
            ]
        )
    missing = [str(path) for path in evidence_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4证据文件缺失: {missing}")
    manifest = {
        "file_count": len(evidence_files),
        "files": [
            {
                "path": str(path.relative_to(PROJECT_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in evidence_files
        ],
    }
    (summary_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"evidence_files={len(evidence_files)}")
    if not summary["final_ensemble_confirmation"]["accepted"]:
        raise SystemExit("V4最终集成确认未通过双指标逐种子非退化门槛")


if __name__ == "__main__":
    main()
