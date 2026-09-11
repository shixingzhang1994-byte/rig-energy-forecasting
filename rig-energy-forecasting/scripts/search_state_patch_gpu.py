from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.schema import STATE_TO_CODE, validate_canonical_frame  # noqa: E402
from rig_energy.data.windowing import (  # noqa: E402
    PowerScaler,
    WindowedRigDataset,
    split_frame_by_time,
)
from rig_energy.metrics import regression_metrics  # noqa: E402
from rig_energy.models import (  # noqa: E402
    PatchTSTForecaster,
    StateAwareDualBranchPatchTransformer,
    StateAwarePatchTransformer,
)
from rig_energy.training import (  # noqa: E402
    predict_neural_model,
    save_torch_checkpoint,
    seed_everything,
    train_neural_model,
)


def _loader(dataset, cfg: dict, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        drop_last=False,
    )


def _data_bundle(path: Path, cfg: dict, device: torch.device) -> dict:
    frame, _ = validate_canonical_frame(pd.read_parquet(path))
    train, val, test = split_frame_by_time(
        frame, float(cfg["train_fraction"]), float(cfg["val_fraction"])
    )
    scaler = PowerScaler.fit(train["total_active_power_kw"].to_numpy(dtype=np.float32))
    history = int(cfg["history_steps"])
    horizon = int(cfg["horizon_steps"])
    train_ds = WindowedRigDataset(
        train, scaler, history, horizon, int(cfg["train_stride"])
    )
    val_ds = WindowedRigDataset(val, scaler, history, horizon, int(cfg["eval_stride"])
    )
    test_ds = WindowedRigDataset(
        test, scaler, history, horizon, int(cfg["eval_stride"])
    )
    return {
        "scaler": scaler,
        "train": _loader(train_ds, cfg, True, device),
        "val": _loader(val_ds, cfg, False, device),
        "test": _loader(test_ds, cfg, False, device),
        "peak_threshold": float(
            train["total_active_power_kw"].quantile(float(cfg["peak_quantile"]))
        ),
        "counts": {
            "train": len(train_ds), "val": len(val_ds), "test": len(test_ds)
        },
    }


def _build_patchtst(cfg: dict, protocol: dict) -> PatchTSTForecaster:
    return PatchTSTForecaster(
        input_size=4,
        history=int(protocol["history_steps"]),
        horizon=int(protocol["horizon_steps"]),
        patch_length=int(cfg["patch_length"]),
        patch_stride=int(cfg["patch_stride"]),
        hidden_size=int(cfg["hidden_size"]),
        attention_heads=int(cfg["attention_heads"]),
        layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
    )


def _build_candidate(cfg: dict, protocol: dict) -> torch.nn.Module:
    common = dict(
        numeric_input_size=4,
        num_states=len(STATE_TO_CODE),
        state_embedding_dim=int(cfg["state_embedding_dim"]),
        history=int(protocol["history_steps"]),
        horizon=int(protocol["horizon_steps"]),
        patch_length=int(cfg["patch_length"]),
        patch_stride=int(cfg["patch_stride"]),
        hidden_size=int(cfg["hidden_size"]),
        attention_heads=int(cfg["attention_heads"]),
        layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
    )
    if cfg["architecture"] == "fused":
        return StateAwarePatchTransformer(**common)
    if cfg["architecture"] == "dual_branch":
        return StateAwareDualBranchPatchTransformer(
            transition_embedding_dim=int(cfg["transition_embedding_dim"]), **common
        )
    raise ValueError(f"未知候选架构: {cfg['architecture']}")


def _fit_and_score(
    model: torch.nn.Module,
    bundle: dict,
    protocol: dict,
    model_cfg: dict,
    device: torch.device,
    evaluation_split: str,
) -> tuple[dict, torch.nn.Module, dict]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, training = train_neural_model(
        model,
        bundle["train"],
        bundle["val"],
        device=device,
        max_epochs=int(protocol["max_epochs"]),
        patience=int(protocol["patience"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        peak_weight=float(model_cfg.get("peak_weight", 0.0)),
        transition_weight=float(model_cfg.get("transition_weight", 0.0)),
    )
    target, prediction, transition, inference_seconds = predict_neural_model(
        model,
        bundle[evaluation_split],
        bundle["scaler"],
        device=device,
    )
    metrics = regression_metrics(
        target,
        prediction,
        peak_threshold=float(bundle["peak_threshold"]),
        transition_flags=transition,
    )
    metrics.update(
        {
            "best_val_loss": float(training["best_val_loss"]),
            "epochs_ran": int(training["epochs_ran"]),
            "training_seconds": float(training["training_seconds"]),
            "inference_seconds": float(inference_seconds),
            "parameter_count": int(training["parameter_count"]),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    return metrics, model, training


def _relative_selection(trials: pd.DataFrame, weights: dict) -> pd.DataFrame:
    anchors = (
        trials[trials["candidate"] == "PatchTST-anchor"]
        .set_index("seed")[["mae_kw", "transition_mae_kw"]]
        .rename(columns=lambda column: f"anchor_{column}")
    )
    candidate_rows = trials[trials["candidate"] != "PatchTST-anchor"].join(
        anchors, on="seed"
    )
    candidate_rows["mae_ratio"] = (
        candidate_rows["mae_kw"] / candidate_rows["anchor_mae_kw"]
    )
    candidate_rows["transition_ratio"] = (
        candidate_rows["transition_mae_kw"]
        / candidate_rows["anchor_transition_mae_kw"]
    )
    candidate_rows["selection_score"] = (
        float(weights["mae"]) * candidate_rows["mae_ratio"]
        + float(weights["transition_mae"]) * candidate_rows["transition_ratio"]
    )
    return (
        candidate_rows.groupby("candidate", as_index=False)
        .agg(
            selection_score=("selection_score", "mean"),
            mean_mae_ratio=("mae_ratio", "mean"),
            mean_transition_ratio=("transition_ratio", "mean"),
            dev_mae_kw=("mae_kw", "mean"),
            dev_transition_mae_kw=("transition_mae_kw", "mean"),
        )
        .sort_values(["selection_score", "mean_mae_ratio", "candidate"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="状态感知 Patch 模型的有界 GPU 搜索")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "configs/v4_gpu_search.yaml"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / "data/processed/v4_gpu_search"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_DIR / "artifacts/v4_gpu_search/model"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol_record = {
        **protocol,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "selection_rule": (
            "minimum mean validation-only score; score = 0.60 * MAE ratio "
            "+ 0.40 * transition-MAE ratio versus PatchTST on each development seed"
        ),
        "holdout_rule": "holdout test labels are evaluated only after candidate selection",
        "candidate_count": len(config["candidates"]),
    }
    (args.artifact_dir / "protocol.json").write_text(
        json.dumps(protocol_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trial_rows: list[dict] = []
    for seed in protocol["development_seeds"]:
        data_path = args.data_dir / f"rig_load_seed_{seed}.parquet"
        bundle = _data_bundle(data_path, protocol, device)
        seed_everything(int(seed))
        anchor_metrics, anchor_model, _ = _fit_and_score(
            _build_patchtst(config["patchtst_anchor"], protocol),
            bundle,
            protocol,
            config["patchtst_anchor"],
            device,
            "val",
        )
        trial_rows.append(
            {"seed": seed, "candidate": "PatchTST-anchor", **anchor_metrics}
        )
        del anchor_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for candidate in config["candidates"]:
            seed_everything(int(seed))
            metrics, model, _ = _fit_and_score(
                _build_candidate(candidate, protocol),
                bundle,
                protocol,
                candidate,
                device,
                "val",
            )
            trial_rows.append(
                {"seed": seed, "candidate": candidate["name"], **metrics}
            )
            print(
                f"DEV seed={seed} candidate={candidate['name']} "
                f"MAE={metrics['mae_kw']:.4f} transition={metrics['transition_mae_kw']:.4f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.artifact_dir / "development_trials.csv", index=False)
    ranking = _relative_selection(trials, protocol["selection_weights"])
    ranking.to_csv(args.artifact_dir / "development_ranking.csv", index=False)
    selected_name = str(ranking.iloc[0]["candidate"])
    selected_cfg = next(
        candidate for candidate in config["candidates"] if candidate["name"] == selected_name
    )
    selection = {
        "selected_candidate": selected_name,
        "selected_config": selected_cfg,
        "ranking": ranking.to_dict(orient="records"),
        "test_labels_used_for_selection": False,
    }
    (args.artifact_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SELECTED {selected_name}")

    holdout_rows: list[dict] = []
    training_logs: dict[str, dict] = {}
    checkpoint_dir = args.artifact_dir / "holdout_checkpoints"
    for seed in protocol["holdout_seeds"]:
        data_path = args.data_dir / f"rig_load_seed_{seed}.parquet"
        bundle = _data_bundle(data_path, protocol, device)
        for model_name, model_cfg, builder in (
            ("PatchTST-anchor", config["patchtst_anchor"], _build_patchtst),
            (selected_name, selected_cfg, _build_candidate),
        ):
            seed_everything(int(seed))
            metrics, model, training = _fit_and_score(
                builder(model_cfg, protocol),
                bundle,
                protocol,
                model_cfg,
                device,
                "test",
            )
            holdout_rows.append({"seed": seed, "model": model_name, **metrics})
            training_logs[f"{seed}:{model_name}"] = training
            save_torch_checkpoint(
                model,
                checkpoint_dir / f"seed_{seed}_{model_name}.pt",
                {"seed": seed, "model_config": model_cfg, "training": training},
            )
            print(
                f"HOLDOUT seed={seed} model={model_name} "
                f"MAE={metrics['mae_kw']:.4f} transition={metrics['transition_mae_kw']:.4f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(args.artifact_dir / "holdout_metrics.csv", index=False)
    (args.artifact_dir / "holdout_training_log.json").write_text(
        json.dumps(training_logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    anchor = holdout[holdout["model"] == "PatchTST-anchor"].set_index("seed")
    project = holdout[holdout["model"] == selected_name].set_index("seed")
    per_seed = pd.DataFrame(
        {
            "anchor_mae_kw": anchor["mae_kw"],
            "project_mae_kw": project["mae_kw"],
            "anchor_transition_mae_kw": anchor["transition_mae_kw"],
            "project_transition_mae_kw": project["transition_mae_kw"],
        }
    )
    per_seed["mae_improvement_pct"] = 100.0 * (
        per_seed["anchor_mae_kw"] - per_seed["project_mae_kw"]
    ) / per_seed["anchor_mae_kw"]
    per_seed["transition_improvement_pct"] = 100.0 * (
        per_seed["anchor_transition_mae_kw"] - per_seed["project_transition_mae_kw"]
    ) / per_seed["anchor_transition_mae_kw"]
    per_seed.to_csv(args.artifact_dir / "holdout_pairwise.csv")
    summary = {
        "selected_candidate": selected_name,
        "holdout_seed_count": len(per_seed),
        "mean_mae_improvement_pct": float(per_seed["mae_improvement_pct"].mean()),
        "worst_mae_improvement_pct": float(per_seed["mae_improvement_pct"].min()),
        "mae_win_rate": float((per_seed["mae_improvement_pct"] > 0).mean()),
        "mean_transition_improvement_pct": float(
            per_seed["transition_improvement_pct"].mean()
        ),
        "worst_transition_improvement_pct": float(
            per_seed["transition_improvement_pct"].min()
        ),
        "transition_win_rate": float(
            (per_seed["transition_improvement_pct"] > 0).mean()
        ),
        "acceptance": {
            "mean_mae_non_degradation": bool(per_seed["mae_improvement_pct"].mean() >= 0),
            "mean_transition_non_degradation": bool(
                per_seed["transition_improvement_pct"].mean() >= 0
            ),
        },
    }
    summary["accepted"] = bool(all(summary["acceptance"].values()))
    (args.artifact_dir / "holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
