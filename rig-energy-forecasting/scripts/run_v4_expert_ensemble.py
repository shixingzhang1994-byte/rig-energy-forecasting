from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.experiment import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行包含V4切换工况专家分支的完整预测集成"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_DIR / "configs/synthetic_default.yaml",
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["project"]["seed"] = int(args.seed)
    config["models"]["state_aware_dual_branch_patch_transformer"]["enabled"] = True
    resolved = args.artifact_dir / "resolved_config.yaml"
    resolved.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    metrics = run_benchmark(
        args.data,
        resolved,
        args.artifact_dir,
        seed_override=int(args.seed),
    )
    project = metrics[metrics["model"] == "Validation-Weighted-Ensemble"].iloc[0]
    modern = metrics[metrics["model"].isin(["PatchTST", "iTransformer"])].sort_values(
        "mae_kw"
    ).iloc[0]
    summary = {
        "seed": int(args.seed),
        "project_model": "Validation-Weighted-Ensemble",
        "strongest_modern_comparator": str(modern["model"]),
        "project_mae_kw": float(project["mae_kw"]),
        "comparator_mae_kw": float(modern["mae_kw"]),
        "mae_improvement_pct": float(
            100.0 * (modern["mae_kw"] - project["mae_kw"]) / modern["mae_kw"]
        ),
        "project_transition_mae_kw": float(project["transition_mae_kw"]),
        "comparator_transition_mae_kw": float(modern["transition_mae_kw"]),
        "transition_improvement_pct": float(
            100.0
            * (modern["transition_mae_kw"] - project["transition_mae_kw"])
            / modern["transition_mae_kw"]
        ),
        "modern_comparators_in_project_ensemble": False,
    }
    (args.artifact_dir / "v4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
