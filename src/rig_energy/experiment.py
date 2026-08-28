from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader

from .data.schema import STATE_TO_CODE, validate_canonical_frame
from .data.windowing import (
    PowerScaler,
    WindowedRigDataset,
    make_tree_windows,
    split_frame_by_time,
)
from .metrics import regression_metrics
from .models import (
    ITransformerForecaster,
    LSTMForecaster,
    PatchTSTForecaster,
    StateAwarePatchTransformer,
    StateAwareTCNAttention,
    TCNForecaster,
    apply_causal_error_feedback,
    apply_weighted_ensemble,
    fit_validation_weighted_ensemble,
)
from .training import (
    predict_neural_model,
    save_torch_checkpoint,
    seed_everything,
    train_neural_model,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _device_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "xgboost": xgb.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_runtime": torch.version.cuda,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _plot_forecasts(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
    limit: int = 720,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    count = min(limit, len(y_true))
    x_axis = np.arange(count)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x_axis, y_true[:count, 0], color="#111827", linewidth=1.6, label="真实/仿真目标")
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(predictions)))
    for color, (name, pred) in zip(colors, predictions.items()):
        ax.plot(x_axis, pred[:count, 0], linewidth=1.0, alpha=0.88, label=name, color=color)
    ax.set_title("测试集下一时刻功率预测对比")
    ax.set_xlabel("测试样本序号（按时间连续）")
    ax.set_ylabel("有功功率 / kW")
    ax.grid(alpha=0.18)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    columns = ["mae_kw", "rmse_kw", "peak_mae_kw", "transition_mae_kw"]
    labels = ["MAE", "RMSE", "峰值MAE", "工况切换MAE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, column, label in zip(axes.flat, columns, labels):
        values = metrics.set_index("model")[column].sort_values()
        ax.barh(values.index, values.values, color="#3B82F6", alpha=0.82)
        ax.set_title(label + " / kW")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(
    data_path: Path,
    config_path: Path,
    artifact_dir: Path,
    *,
    quick: bool = False,
    force_cpu: bool = False,
    seed_override: int | None = None,
) -> pd.DataFrame:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    seed = int(
        config["project"]["seed"] if seed_override is None else seed_override
    )
    seed_everything(seed)

    frame = pd.read_parquet(data_path)
    frame, schema_report = validate_canonical_frame(frame)
    forecast_cfg = config["forecast"]
    train_frame, val_frame, test_frame = split_frame_by_time(
        frame,
        float(forecast_cfg["train_fraction"]),
        float(forecast_cfg["val_fraction"]),
    )
    history = int(forecast_cfg["history_steps"])
    horizon = int(forecast_cfg["horizon_steps"])
    train_stride = int(forecast_cfg["train_stride"])
    eval_stride = int(forecast_cfg["eval_stride"])
    if quick:
        train_stride = max(train_stride, 6)
        max_epochs = min(3, int(forecast_cfg["max_epochs"]))
    else:
        max_epochs = int(forecast_cfg["max_epochs"])

    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    device_info = _device_info(device)
    (artifact_dir / "device_info.json").write_text(
        json.dumps(device_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    peak_threshold = float(
        train_frame["total_active_power_kw"].quantile(float(forecast_cfg["peak_quantile"]))
    )
    results: list[dict] = []
    prediction_store: dict[str, np.ndarray] = {}
    validation_prediction_store: dict[str, np.ndarray] = {}

    x_train, y_train, _ = make_tree_windows(train_frame, history, horizon, train_stride)
    x_val, y_val, _ = make_tree_windows(val_frame, history, horizon, eval_stride)
    x_test, y_test, transition_test = make_tree_windows(test_frame, history, horizon, eval_stride)

    persistence = np.repeat(x_test[:, [0]], horizon, axis=1)
    result = {"model": "Persistence"}
    result.update(
        regression_metrics(
            y_test,
            persistence,
            peak_threshold=peak_threshold,
            transition_flags=transition_test,
        )
    )
    result.update({"training_seconds": 0.0, "inference_seconds": 0.0, "parameter_count": 0})
    results.append(result)
    prediction_store["Persistence"] = persistence
    validation_prediction_store["Persistence"] = np.repeat(x_val[:, [0]], horizon, axis=1)

    xgb_cfg = config["models"]["xgboost"]
    xgb_device = "cuda" if device.type == "cuda" else "cpu"
    estimator = xgb.XGBRegressor(
        n_estimators=int(xgb_cfg["n_estimators"] if not quick else min(60, xgb_cfg["n_estimators"])),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        objective="reg:squarederror",
        tree_method="hist",
        device=xgb_device,
        random_state=seed,
        n_jobs=1,
    )
    tree_model = MultiOutputRegressor(estimator, n_jobs=1)
    started = time.perf_counter()
    tree_model.fit(x_train, y_train)
    tree_train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    xgb_pred = tree_model.predict(x_test)
    tree_inference_seconds = time.perf_counter() - started
    xgb_val_pred = tree_model.predict(x_val)
    result = {"model": "XGBoost"}
    result.update(
        regression_metrics(
            y_test,
            xgb_pred,
            peak_threshold=peak_threshold,
            transition_flags=transition_test,
        )
    )
    result.update(
        {
            "training_seconds": tree_train_seconds,
            "inference_seconds": tree_inference_seconds,
            "parameter_count": int(sum(e.get_booster().num_boosted_rounds() for e in tree_model.estimators_)),
        }
    )
    results.append(result)
    prediction_store["XGBoost"] = xgb_pred
    validation_prediction_store["XGBoost"] = xgb_val_pred
    joblib.dump(tree_model, model_dir / "xgboost_multi_horizon.joblib")

    scaler = PowerScaler.fit(train_frame["total_active_power_kw"].to_numpy(dtype=np.float32))
    train_dataset = WindowedRigDataset(train_frame, scaler, history, horizon, train_stride)
    val_dataset = WindowedRigDataset(val_frame, scaler, history, horizon, eval_stride)
    test_dataset = WindowedRigDataset(test_frame, scaler, history, horizon, eval_stride)
    batch_size = int(forecast_cfg["batch_size"])
    num_workers = int(forecast_cfg["num_workers"])
    train_loader = _make_loader(train_dataset, batch_size, True, num_workers, device)
    val_loader = _make_loader(val_dataset, batch_size, False, num_workers, device)
    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, device)

    model_specs = []
    lstm_cfg = config["models"]["lstm"]
    model_specs.append(
        (
            "LSTM",
            LSTMForecaster(
                input_size=4,
                hidden_size=int(lstm_cfg["hidden_size"]),
                num_layers=int(lstm_cfg["num_layers"]),
                horizon=horizon,
                dropout=float(lstm_cfg["dropout"]),
            ),
            0.0,
        )
    )
    tcn_cfg = config["models"]["tcn"]
    model_specs.append(
        (
            "TCN",
            TCNForecaster(
                input_size=4,
                hidden_size=int(tcn_cfg["hidden_size"]),
                levels=int(tcn_cfg["levels"]),
                kernel_size=int(tcn_cfg["kernel_size"]),
                horizon=horizon,
                dropout=float(tcn_cfg["dropout"]),
            ),
            0.0,
        )
    )
    patchtst_cfg = config["models"]["patchtst"]
    model_specs.append(
        (
            "PatchTST",
            PatchTSTForecaster(
                input_size=4,
                history=history,
                horizon=horizon,
                patch_length=int(patchtst_cfg["patch_length"]),
                patch_stride=int(patchtst_cfg["patch_stride"]),
                hidden_size=int(patchtst_cfg["hidden_size"]),
                attention_heads=int(patchtst_cfg["attention_heads"]),
                layers=int(patchtst_cfg["layers"]),
                dropout=float(patchtst_cfg["dropout"]),
            ),
            0.0,
        )
    )
    itransformer_cfg = config["models"]["itransformer"]
    model_specs.append(
        (
            "iTransformer",
            ITransformerForecaster(
                input_size=4,
                history=history,
                horizon=horizon,
                hidden_size=int(itransformer_cfg["hidden_size"]),
                attention_heads=int(itransformer_cfg["attention_heads"]),
                layers=int(itransformer_cfg["layers"]),
                dropout=float(itransformer_cfg["dropout"]),
            ),
            0.0,
        )
    )
    state_patch_cfg = config["models"]["state_aware_patch_transformer"]
    model_specs.append(
        (
            "StateAware-Patch-Transformer",
            StateAwarePatchTransformer(
                numeric_input_size=4,
                num_states=len(STATE_TO_CODE),
                state_embedding_dim=int(state_patch_cfg["state_embedding_dim"]),
                history=history,
                horizon=horizon,
                patch_length=int(state_patch_cfg["patch_length"]),
                patch_stride=int(state_patch_cfg["patch_stride"]),
                hidden_size=int(state_patch_cfg["hidden_size"]),
                attention_heads=int(state_patch_cfg["attention_heads"]),
                layers=int(state_patch_cfg["layers"]),
                dropout=float(state_patch_cfg["dropout"]),
            ),
            float(state_patch_cfg["peak_weight"]),
        )
    )
    proposed_cfg = config["models"]["state_aware_tcn_attention"]
    model_specs.append(
        (
            "StateAware-TCN-Attention",
            StateAwareTCNAttention(
                numeric_input_size=4,
                num_states=len(STATE_TO_CODE),
                state_embedding_dim=int(proposed_cfg["state_embedding_dim"]),
                hidden_size=int(proposed_cfg["hidden_size"]),
                levels=int(proposed_cfg["levels"]),
                kernel_size=int(proposed_cfg["kernel_size"]),
                attention_heads=int(proposed_cfg["attention_heads"]),
                horizon=horizon,
                dropout=float(proposed_cfg["dropout"]),
            ),
            float(proposed_cfg["peak_weight"]),
        )
    )

    training_log: dict[str, dict] = {}
    for name, model, peak_weight in model_specs:
        model, info = train_neural_model(
            model,
            train_loader,
            val_loader,
            device=device,
            max_epochs=max_epochs,
            patience=int(forecast_cfg["patience"]),
            learning_rate=float(forecast_cfg["learning_rate"]),
            weight_decay=float(forecast_cfg["weight_decay"]),
            peak_weight=peak_weight,
        )
        y_true_neural, pred, flags, inference_seconds = predict_neural_model(
            model, test_loader, scaler, device=device
        )
        y_val_neural, val_pred, _, _ = predict_neural_model(
            model, val_loader, scaler, device=device
        )
        if not np.allclose(y_val_neural, y_val, rtol=0.0, atol=1e-3):
            raise RuntimeError("树模型与神经网络的验证窗口未对齐")
        result = {"model": name}
        result.update(
            regression_metrics(
                y_true_neural,
                pred,
                peak_threshold=peak_threshold,
                transition_flags=flags,
            )
        )
        result.update(
            {
                "training_seconds": info["training_seconds"],
                "inference_seconds": inference_seconds,
                "parameter_count": info["parameter_count"],
            }
        )
        results.append(result)
        prediction_store[name] = pred
        validation_prediction_store[name] = val_pred
        training_log[name] = info
        save_torch_checkpoint(
            model,
            model_dir / f"{name.lower().replace('-', '_')}.pt",
            {"scaler": scaler.__dict__, "config": config, "training": info},
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 收敛版可选增强：只根据验证集证据学习透明的凸组合权重。
    # 测试目标从不参与调权，避免测试泄漏。
    ensemble_members = [
        "XGBoost",
        "LSTM",
        "TCN",
        "StateAware-TCN-Attention",
        "StateAware-Patch-Transformer",
    ]
    ensemble_val_predictions = {
        name: validation_prediction_store[name] for name in ensemble_members
    }
    ensemble_weights, ensemble_metadata = fit_validation_weighted_ensemble(
        y_val,
        ensemble_val_predictions,
        l2_penalty=float(config.get("ensemble", {}).get("l2_penalty", 1e-4)),
    )
    ensemble_started = time.perf_counter()
    ensemble_prediction = apply_weighted_ensemble(prediction_store, ensemble_weights)
    ensemble_inference_seconds = time.perf_counter() - ensemble_started
    result = {"model": "Validation-Weighted-Ensemble"}
    result.update(
        regression_metrics(
            y_test,
            ensemble_prediction,
            peak_threshold=peak_threshold,
            transition_flags=transition_test,
        )
    )
    result.update(
        {
            "training_seconds": 0.0,
            "inference_seconds": ensemble_inference_seconds,
            "parameter_count": len(ensemble_weights),
        }
    )
    results.append(result)
    prediction_store["Validation-Weighted-Ensemble"] = ensemble_prediction
    (artifact_dir / "ensemble_weights.json").write_text(
        json.dumps(
            {"weights": ensemble_weights, "fit": ensemble_metadata},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 任务书要求的“历史误差反馈修正”：仅释放已完整到达的
    # 预测窗口残差。平滑与限幅参数只在验证集上选择。
    feedback_cfg = config.get("error_feedback", {})
    delay = max(horizon, int(feedback_cfg.get("feedback_delay_steps", horizon)))
    candidates = []
    ensemble_val_prediction = apply_weighted_ensemble(
        validation_prediction_store, ensemble_weights
    )
    for smoothing in feedback_cfg.get("smoothing_candidates", [0.02, 0.05, 0.10]):
        for clip_kw in feedback_cfg.get("correction_clip_candidates_kw", [50.0, 100.0]):
            corrected_val, final_bias = apply_causal_error_feedback(
                ensemble_val_prediction,
                y_val,
                feedback_delay_steps=delay,
                smoothing=float(smoothing),
                correction_clip_kw=float(clip_kw),
            )
            validation_mae = float(
                np.mean(np.abs(corrected_val[delay:] - y_val[delay:]))
            )
            candidates.append(
                {
                    "smoothing": float(smoothing),
                    "correction_clip_kw": float(clip_kw),
                    "validation_mae_kw": validation_mae,
                    "final_bias_kw": final_bias,
                }
            )
    selected_feedback = min(candidates, key=lambda item: item["validation_mae_kw"])
    feedback_started = time.perf_counter()
    feedback_prediction, _ = apply_causal_error_feedback(
        ensemble_prediction,
        y_test,
        feedback_delay_steps=delay,
        smoothing=selected_feedback["smoothing"],
        correction_clip_kw=selected_feedback["correction_clip_kw"],
        initial_bias_kw=selected_feedback["final_bias_kw"],
    )
    feedback_inference_seconds = time.perf_counter() - feedback_started
    result = {"model": "Causal-ErrorFeedback-Ensemble"}
    result.update(
        regression_metrics(
            y_test,
            feedback_prediction,
            peak_threshold=peak_threshold,
            transition_flags=transition_test,
        )
    )
    result.update(
        {
            "training_seconds": 0.0,
            "inference_seconds": feedback_inference_seconds,
            "parameter_count": len(ensemble_weights) + horizon,
        }
    )
    results.append(result)
    prediction_store["Causal-ErrorFeedback-Ensemble"] = feedback_prediction
    (artifact_dir / "error_feedback.json").write_text(
        json.dumps(
            {
                "causality_rule": "sample i only uses residuals from samples <= i-feedback_delay_steps",
                "feedback_delay_steps": delay,
                "selection_data": "validation_only",
                "selected_smoothing": selected_feedback["smoothing"],
                "selected_correction_clip_kw": selected_feedback[
                    "correction_clip_kw"
                ],
                "selected_validation_mae_kw": selected_feedback[
                    "validation_mae_kw"
                ],
                "candidates": [
                    {key: value for key, value in item.items() if key != "final_bias_kw"}
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = pd.DataFrame(results).sort_values("mae_kw").reset_index(drop=True)
    metrics.to_csv(artifact_dir / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "training_log.json").write_text(
        json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_metadata = {
        "schema_report": schema_report.as_dict(),
        "data_path": str(data_path),
        "config_path": str(config_path),
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "history_steps": history,
        "horizon_steps": horizon,
        "peak_threshold_kw": peak_threshold,
        "quick": quick,
        "seed": seed,
        "device": device_info,
        "comparison_protocol": {
            "same_chronological_split": True,
            "same_history_and_horizon": True,
            "same_numeric_inputs": True,
            "modern_comparators": ["PatchTST", "iTransformer"],
            "modern_comparators_in_project_ensemble": False,
            "project_models": [
                "StateAware-TCN-Attention",
                "StateAware-Patch-Transformer",
            ],
            "project_specific_extra_input": "operation_state_code",
        },
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 保留各模型的多步预测轨迹，供后续多源动力滚动优化直接复用。
    prediction_arrays = {
        "y_true": y_test.astype(np.float32),
        "transition_flags": transition_test.astype(np.int8),
    }
    prediction_key_map = {}
    for index, (name, prediction) in enumerate(prediction_store.items()):
        key = f"prediction_{index}"
        prediction_arrays[key] = prediction.astype(np.float32)
        prediction_key_map[key] = name
    np.savez_compressed(artifact_dir / "forecast_predictions.npz", **prediction_arrays)
    (artifact_dir / "forecast_prediction_keys.json").write_text(
        json.dumps(prediction_key_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_forecasts(y_test, prediction_store, artifact_dir / "forecast_comparison.png")
    _plot_metric_bars(metrics, artifact_dir / "metric_comparison.png")
    return metrics
