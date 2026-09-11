from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ..data.schema import STATE_TO_CODE, validate_canonical_frame
from ..data.windowing import _transition_flags
from ..experiment import _plot_forecasts, _plot_metric_bars
from ..metrics import regression_metrics
from ..models import (
    ITransformerForecaster,
    PatchTSTForecaster,
    StateAwareDualBranchPatchTransformer,
    StateAwareTCNAttention,
    apply_causal_error_feedback,
    apply_weighted_ensemble,
    fit_validation_weighted_ensemble,
)
from ..training import seed_everything, train_neural_model


@dataclass(frozen=True)
class FeatureScaler:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, names: list[str]) -> "FeatureScaler":
        values = frame[names].to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("因果多变量特征中存在缺失或无穷值")
        std = values.std(axis=0)
        return cls(
            names=tuple(names),
            mean=values.mean(axis=0).astype(np.float32),
            std=np.maximum(std, 1e-6).astype(np.float32),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.names)].to_numpy(np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * float(self.std[0]) + float(self.mean[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "fit_data": "chronological_training_partition_only",
        }


class MultivariateCausalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        scaler: FeatureScaler,
        history: int,
        horizon: int,
        stride: int,
        state_column: str = "operation_state_code",
    ) -> None:
        self.features = scaler.transform(frame)
        self.states = frame[state_column].to_numpy(np.int64)
        self.transition_states = frame["operation_state_code"].to_numpy(np.int64)
        self.target = self.features[:, 0].astype(np.float32)
        self.history = int(history)
        self.horizon = int(horizon)
        self.indices = np.arange(
            self.history,
            len(frame) - self.horizon + 1,
            max(1, int(stride)),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        stop = int(self.indices[item])
        start = stop - self.history
        target_stop = stop + self.horizon
        history_transition, future_transition = _transition_flags(
            self.transition_states, start, stop, target_stop
        )
        return {
            "x": torch.from_numpy(self.features[start:stop]),
            "state": torch.from_numpy(self.states[start:stop]),
            "y": torch.from_numpy(self.target[stop:target_stop]),
            "transition": torch.tensor(future_transition),
            "history_transition": torch.tensor(history_transition),
            "future_transition": torch.tensor(future_transition),
        }


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: FeatureScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    transitions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(x, state)
            predictions.append(prediction.float().cpu().numpy())
            targets.append(batch["y"].numpy())
            transitions.append(batch["transition"].numpy())
    return (
        scaler.inverse_target(np.concatenate(targets)),
        scaler.inverse_target(np.concatenate(predictions)),
        np.concatenate(transitions).astype(bool),
        time.perf_counter() - started,
    )


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _models(
    config: dict[str, Any],
    feature_count: int,
    num_states: int,
    history: int,
    horizon: int,
) -> list[tuple[str, torch.nn.Module]]:
    cfg = config["models"]
    output: list[tuple[str, torch.nn.Module]] = []
    if "multivariate_itransformer" in cfg:
        item = cfg["multivariate_itransformer"]
        output.append((
            str(item.get("name", "Multivariate-iTransformer")),
            ITransformerForecaster(
                input_size=feature_count,
                history=history,
                horizon=horizon,
                hidden_size=int(item["hidden_size"]),
                attention_heads=int(item["attention_heads"]),
                layers=int(item["layers"]),
                dropout=float(item["dropout"]),
            ),
        ))
    if "multivariate_patchtst" in cfg:
        item = cfg["multivariate_patchtst"]
        output.append(
        (
            "Multivariate-PatchTST",
            PatchTSTForecaster(
                input_size=feature_count,
                history=history,
                horizon=horizon,
                patch_length=int(item["patch_length"]),
                patch_stride=int(item["patch_stride"]),
                hidden_size=int(item["hidden_size"]),
                attention_heads=int(item["attention_heads"]),
                layers=int(item["layers"]),
                dropout=float(item["dropout"]),
            ),
        )
    )
    if "multivariate_stateaware_dualbranch" in cfg:
        item = cfg["multivariate_stateaware_dualbranch"]
        output.append(
        (
            str(item.get("name", "Multivariate-StateAware-DualBranch")),
            StateAwareDualBranchPatchTransformer(
                numeric_input_size=feature_count,
                num_states=num_states,
                state_embedding_dim=int(item["state_embedding_dim"]),
                transition_embedding_dim=int(item["transition_embedding_dim"]),
                history=history,
                horizon=horizon,
                patch_length=int(item["patch_length"]),
                patch_stride=int(item["patch_stride"]),
                hidden_size=int(item["hidden_size"]),
                attention_heads=int(item["attention_heads"]),
                layers=int(item["layers"]),
                dropout=float(item["dropout"]),
            ),
        )
    )
    if "multivariate_stateaware_tcn_attention" in cfg:
        item = cfg["multivariate_stateaware_tcn_attention"]
        output.append(
        (
            "Multivariate-StateAware-TCN-Attention",
            StateAwareTCNAttention(
                numeric_input_size=feature_count,
                num_states=num_states,
                state_embedding_dim=int(item["state_embedding_dim"]),
                hidden_size=int(item["hidden_size"]),
                levels=int(item["levels"]),
                kernel_size=int(item["kernel_size"]),
                attention_heads=int(item["attention_heads"]),
                horizon=horizon,
                dropout=float(item["dropout"]),
            ),
        )
    )
    return output


def add_causal_derived_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add only origin-available timing and run-length features."""

    output = frame.copy()
    power = output["total_active_power_kw"].to_numpy(np.float32)
    output["total_power_delta_kw"] = np.r_[0.0, np.diff(power)].astype(np.float32)
    timestamp = pd.to_datetime(output["timestamp"])
    seconds = (
        timestamp.dt.hour.to_numpy() * 3600
        + timestamp.dt.minute.to_numpy() * 60
        + timestamp.dt.second.to_numpy()
    )
    phase = 2.0 * np.pi * seconds / 86400.0
    output["time_sin"] = np.sin(phase).astype(np.float32)
    output["time_cos"] = np.cos(phase).astype(np.float32)
    for source, target in [
        ("operation_state", "operation_state_elapsed_minutes"),
        ("operation_substate", "operation_substate_elapsed_minutes"),
    ]:
        values = output[source].astype(str)
        block = values.ne(values.shift()).cumsum()
        elapsed_steps = output.groupby(block, sort=False).cumcount().to_numpy()
        output[target] = (elapsed_steps * 5.0 / 60.0).astype(np.float32)
    substates = sorted(output["operation_substate"].astype(str).unique())
    substate_to_code = {name: index for index, name in enumerate(substates)}
    output["operation_substate_code"] = (
        output["operation_substate"].astype(str).map(substate_to_code).astype("int16")
    )
    return output, substate_to_code


def run_multivariate_causal_development(
    config_path: Path, artifact_dir: Path, *, force_cpu: bool = False
) -> pd.DataFrame:
    config_path = Path(config_path).resolve()
    project_dir = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "development_protocol.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    seed_everything(int(config["project"]["seed"]))
    data_path = (project_dir / config["data"]["path"]).resolve()
    frame = pd.read_parquet(data_path)
    frame, schema_report = validate_canonical_frame(frame)
    if bool(config["data"].get("add_causal_derived_features", False)):
        frame, substate_to_code = add_causal_derived_features(frame)
    else:
        substate_to_code = {}
    features = [str(name) for name in config["data"]["causal_numeric_features"]]
    if features[0] != "total_active_power_kw":
        raise ValueError("第一个因果特征必须是预测目标总有功功率")
    forbidden_tokens = ("future", "reference", "risk_level", "load_shed")
    forbidden = [name for name in features if any(token in name for token in forbidden_tokens)]
    if forbidden:
        raise ValueError(f"多变量输入包含禁止的未来/标签字段: {forbidden}")
    count = len(frame)
    train_stop = int(count * float(config["data"]["train_fraction"]))
    val_stop = int(
        count
        * (
            float(config["data"]["train_fraction"])
            + float(config["data"]["validation_fraction"])
        )
    )
    train_frame = frame.iloc[:train_stop].reset_index(drop=True)
    val_frame = frame.iloc[train_stop:val_stop].reset_index(drop=True)
    test_frame = frame.iloc[val_stop:].reset_index(drop=True)
    scaler = FeatureScaler.fit(train_frame, features)
    state_column = str(config["data"].get("state_embedding_column", "operation_state_code"))
    if state_column not in frame:
        raise ValueError(f"状态嵌入字段不存在: {state_column}")
    num_states = int(frame[state_column].max()) + 1
    history = int(config["data"]["history_steps"])
    horizon = int(config["data"]["horizon_steps"])
    train_set = MultivariateCausalDataset(
        train_frame,
        scaler,
        history,
        horizon,
        int(config["data"]["train_stride"]),
        state_column,
    )
    val_set = MultivariateCausalDataset(
        val_frame,
        scaler,
        history,
        horizon,
        int(config["data"]["evaluation_stride"]),
        state_column,
    )
    test_set = MultivariateCausalDataset(
        test_frame,
        scaler,
        history,
        horizon,
        int(config["data"]["evaluation_stride"]),
        state_column,
    )
    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    training = config["training"]
    loaders = {
        "train": _loader(
            train_set,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            workers=int(training["num_workers"]),
            device=device,
        ),
        "validation": _loader(
            val_set,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            workers=int(training["num_workers"]),
            device=device,
        ),
        "test": _loader(
            test_set,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            workers=int(training["num_workers"]),
            device=device,
        ),
    }
    target = test_frame["total_active_power_kw"].to_numpy(np.float32)
    stops = test_set.indices
    y_test = np.stack([target[stop : stop + horizon] for stop in stops])
    persistence = np.repeat(target[stops - 1, None], horizon, axis=1)
    val_target = val_frame["total_active_power_kw"].to_numpy(np.float32)
    val_stops = val_set.indices
    y_val = np.stack([val_target[stop : stop + horizon] for stop in val_stops])
    val_persistence = np.repeat(val_target[val_stops - 1, None], horizon, axis=1)
    states = test_frame["operation_state_code"].to_numpy(np.int16)
    transitions = np.asarray(
        [
            _transition_flags(states, stop - history, stop, stop + horizon)[1]
            for stop in stops
        ],
        dtype=bool,
    )
    predictions: dict[str, np.ndarray] = {"Persistence": persistence}
    validation_predictions: dict[str, np.ndarray] = {
        "Persistence": val_persistence
    }
    logs: dict[str, Any] = {}
    model_dir = artifact_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, model in _models(
        config, len(features), num_states, history, horizon
    ):
        model, info = train_neural_model(
            model,
            loaders["train"],
            loaders["validation"],
            device=device,
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            peak_weight=float(training["peak_weight"]),
            transition_weight=float(training["transition_weight"]),
        )
        neural_test, test_prediction, neural_transition, test_seconds = _predict(
            model, loaders["test"], scaler, device
        )
        neural_val, val_prediction, _, val_seconds = _predict(
            model, loaders["validation"], scaler, device
        )
        if not np.allclose(neural_test, y_test, rtol=0.0, atol=1e-3):
            raise ValueError(f"{name} 多变量测试窗口未对齐")
        if not np.allclose(neural_val, y_val, rtol=0.0, atol=1e-3):
            raise ValueError(f"{name} 多变量验证窗口未对齐")
        if not np.array_equal(neural_transition, transitions):
            raise ValueError(f"{name} 切换标记未对齐")
        predictions[name] = test_prediction.astype(np.float32)
        validation_predictions[name] = val_prediction.astype(np.float32)
        info["test_inference_seconds"] = test_seconds
        info["validation_inference_seconds"] = val_seconds
        logs[name] = info
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_scaler": scaler.as_dict(),
                "config": config,
            },
            model_dir / f"{name.lower().replace('-', '_')}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    member_val = {
        name: value
        for name, value in validation_predictions.items()
        if name != "Persistence"
    }
    weights, fit = fit_validation_weighted_ensemble(
        y_val, member_val, l2_penalty=float(config["ensemble"]["l2_penalty"])
    )
    ensemble = apply_weighted_ensemble(predictions, weights)
    val_ensemble = apply_weighted_ensemble(validation_predictions, weights)
    predictions["Multivariate-Validation-Weighted-Ensemble"] = ensemble
    feedback = config["error_feedback"]
    delay = max(horizon, int(feedback["feedback_delay_steps"]))
    candidates = []
    for smoothing in feedback["smoothing_candidates"]:
        for clip in feedback["correction_clip_candidates_kw"]:
            corrected, final_bias = apply_causal_error_feedback(
                val_ensemble,
                y_val,
                feedback_delay_steps=delay,
                smoothing=float(smoothing),
                correction_clip_kw=float(clip),
            )
            candidates.append(
                {
                    "smoothing": float(smoothing),
                    "clip_kw": float(clip),
                    "validation_mae_kw": float(
                        np.mean(np.abs(corrected[delay:] - y_val[delay:]))
                    ),
                    "final_bias_kw": final_bias,
                }
            )
    selected = min(candidates, key=lambda item: item["validation_mae_kw"])
    corrected, _ = apply_causal_error_feedback(
        ensemble,
        y_test,
        feedback_delay_steps=delay,
        smoothing=selected["smoothing"],
        correction_clip_kw=selected["clip_kw"],
        initial_bias_kw=selected["final_bias_kw"],
    )
    predictions["Multivariate-Causal-ErrorFeedback-Ensemble"] = corrected
    peak_threshold = float(
        train_frame["total_active_power_kw"].quantile(0.90)
    )
    rows = []
    for name, prediction in predictions.items():
        row = {"model": name}
        row.update(
            regression_metrics(
                y_test,
                prediction,
                peak_threshold=peak_threshold,
                transition_flags=transitions,
            )
        )
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("mae_kw").reset_index(drop=True)
    metrics.to_csv(artifact_dir / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "ensemble_weights.json").write_text(
        json.dumps({"weights": weights, "fit": fit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "error_feedback.json").write_text(
        json.dumps(
            {
                "selected": {
                    key: value
                    for key, value in selected.items()
                    if key != "final_bias_kw"
                },
                "selection_data": "validation_only",
                "feedback_delay_steps": delay,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "feature_audit.json").write_text(
        json.dumps(
            {
                "features": features,
                "state_embedding_column": state_column,
                "state_vocabulary": substate_to_code,
                "all_features_observed_at_or_before_forecast_origin": True,
                "future_or_label_fields": [],
                "scaler": scaler.as_dict(),
                "schema_report": schema_report.as_dict(),
                "split_rows": {
                    "train": len(train_frame),
                    "validation": len(val_frame),
                    "test": len(test_frame),
                },
                "evaluation_status": config["protocol"]["status"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    arrays: dict[str, np.ndarray] = {
        "y_true": y_test.astype(np.float32),
        "transition_flags": transitions.astype(np.int8),
        "operation_state_codes": states[stops],
    }
    keys = {}
    for index, (name, prediction) in enumerate(predictions.items()):
        key = f"prediction_{index}"
        arrays[key] = np.asarray(prediction, dtype=np.float32)
        keys[key] = name
    np.savez_compressed(artifact_dir / "forecast_predictions.npz", **arrays)
    (artifact_dir / "forecast_prediction_keys.json").write_text(
        json.dumps(keys, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_forecasts(y_test, predictions, artifact_dir / "forecast_comparison.png")
    _plot_metric_bars(metrics, artifact_dir / "metric_comparison.png")
    primary = config["development_gates"]["primary_model"]
    lookup = metrics.set_index("model")
    p = lookup.loc[primary]
    b = lookup.loc["Persistence"]
    result = {
        "primary_model": primary,
        "r2": float(p["r2"]),
        "mae_kw": float(p["mae_kw"]),
        "mae_improvement_over_persistence": float(
            (b["mae_kw"] - p["mae_kw"]) / b["mae_kw"]
        ),
        "transition_mae_improvement_over_persistence": float(
            (b["transition_mae_kw"] - p["transition_mae_kw"])
            / b["transition_mae_kw"]
        ),
        "development_only": True,
        "new_post_development_data_required": True,
        "device": str(device),
    }
    gates = config["development_gates"]
    result["gate_results"] = {
        "r2": result["r2"] >= float(gates["r2_min"]),
        "mae_improvement": result["mae_improvement_over_persistence"]
        >= float(gates["mae_improvement_over_persistence_min"]),
        "transition_mae_improvement": result[
            "transition_mae_improvement_over_persistence"
        ]
        >= float(gates["transition_mae_improvement_over_persistence_min"]),
    }
    (artifact_dir / "development_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
