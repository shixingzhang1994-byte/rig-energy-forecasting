from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    20260825: PROJECT_DIR / "artifacts/v3_competitive_forecast/benchmark_metrics.csv",
    20260826: PROJECT_DIR
    / "artifacts/v3_robustness/seed_20260826/forecast/benchmark_metrics.csv",
    20260827: PROJECT_DIR
    / "artifacts/v3_robustness/seed_20260827/forecast/benchmark_metrics.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总V3多井况种子的现代预测基线稳健性"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v3_robustness/summary",
    )
    args = parser.parse_args()

    detail_rows: list[dict[str, float | int | str]] = []
    pairwise_rows: list[dict[str, float | int | str | bool]] = []
    for seed, path in DEFAULT_RUNS.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少稳健性结果: {path}")
        metrics = pd.read_csv(path)
        metrics.insert(0, "seed", seed)
        detail_rows.extend(metrics.to_dict(orient="records"))
        indexed = metrics.set_index("model")
        project = indexed.loc["Validation-Weighted-Ensemble"]
        modern = indexed.loc[["PatchTST", "iTransformer"]].sort_values("mae_kw")
        strongest_name = str(modern.index[0])
        strongest = modern.iloc[0]
        state_patch = indexed.loc["StateAware-Patch-Transformer"]
        patchtst = indexed.loc["PatchTST"]
        pairwise_rows.append(
            {
                "seed": seed,
                "strongest_modern_comparator": strongest_name,
                "project_mae_kw": float(project["mae_kw"]),
                "comparator_mae_kw": float(strongest["mae_kw"]),
                "project_mae_improvement_ratio": float(
                    1.0 - project["mae_kw"] / strongest["mae_kw"]
                ),
                "project_transition_mae_kw": float(project["transition_mae_kw"]),
                "comparator_transition_mae_kw": float(
                    strongest["transition_mae_kw"]
                ),
                "project_wins_mae": bool(project["mae_kw"] < strongest["mae_kw"]),
                "project_wins_transition": bool(
                    project["transition_mae_kw"]
                    < strongest["transition_mae_kw"]
                ),
                "state_patch_mae_kw": float(state_patch["mae_kw"]),
                "patchtst_mae_kw": float(patchtst["mae_kw"]),
                "state_patch_wins_patchtst": bool(
                    state_patch["mae_kw"] < patchtst["mae_kw"]
                ),
            }
        )

    detail = pd.DataFrame(detail_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.artifact_dir / "all_seed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pairwise.to_csv(
        args.artifact_dir / "pairwise_vs_modern_baselines.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "seed_count": int(len(pairwise)),
        "seeds": pairwise["seed"].astype(int).tolist(),
        "project_mae_win_rate": float(pairwise["project_wins_mae"].mean()),
        "project_transition_win_rate": float(
            pairwise["project_wins_transition"].mean()
        ),
        "project_mean_mae_improvement_ratio": float(
            pairwise["project_mae_improvement_ratio"].mean()
        ),
        "project_worst_mae_improvement_ratio": float(
            pairwise["project_mae_improvement_ratio"].min()
        ),
        "state_patch_vs_patchtst_win_rate": float(
            pairwise["state_patch_wins_patchtst"].mean()
        ),
        "scope": "three independent synthetic drilling/load seeds; not field validation",
    }
    (args.artifact_dir / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pairwise.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
