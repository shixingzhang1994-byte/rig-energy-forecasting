from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost
import yaml
from torch.utils.data import DataLoader

from ..data.external_adapter_v16 import load_external_data_v16
from ..data.schema import STATE_TO_CODE
from ..data.windowing import (
    PowerScaler,
    WindowedRigDataset,
    _transition_flags,
    make_tree_windows,
)
from ..experiment import _plot_forecasts, _plot_metric_bars
from ..metrics import regression_metrics
from ..models import (
    ITransformerForecaster,
    LSTMForecaster,
    PatchTSTForecaster,
    StateAwareDualBranchPatchTransformer,
    StateAwarePatchTransformer,
    StateAwareTCNAttention,
    TCNForecaster,
    apply_causal_error_feedback,
    apply_weighted_ensemble,
)
from ..risk.experiment import (
    _classification_metrics,
    _ordinal_probabilities,
    _plot_timeline,
    _safety_union_probabilities,
)
from ..risk.labels import RISK_NAMES, assign_risk_levels, build_risk_evidence
from ..training import predict_neural_model


FORECAST_MODEL_FILES = {
    "LSTM": "lstm.pt",
    "TCN": "tcn.pt",
    "PatchTST": "patchtst.pt",
    "iTransformer": "itransformer.pt",
    "StateAware-TCN-Attention": "stateaware_tcn_attention.pt",
    "StateAware-Patch-Transformer": "stateaware_patch_transformer.pt",
    "StateAware-DualBranch-Patch-Transformer": (
        "stateaware_dualbranch_patch_transformer.pt"
    ),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点必须为字典: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_path(release_root: Path, key: str) -> Path:
    if key == "data":
        raise ValueError("data 路径由验收协议单独解析")
    if key == "resolved_forecast_config.yaml":
        return release_root / "source" / key
    if key == "resolved_risk_config.yaml":
        return release_root / "source" / key
    if key.startswith("forecast/"):
        return release_root / "source" / key
    return release_root / key


def verify_locked_inputs(
    protocol: dict[str, Any], project_dir: Path
) -> list[dict[str, Any]]:
    paths = protocol["paths"]
    release_root = (project_dir / paths["release_root"]).resolve()
    data_path = Path(paths["data"]).resolve()
    records: list[dict[str, Any]] = []
    for key, expected in protocol["expected_sha256"].items():
        path = data_path if key == "data" else _release_path(release_root, key)
        actual = _sha256(path)
        records.append(
            {
                "key": key,
                "path": str(path),
                "expected_sha256": str(expected),
                "actual_sha256": actual,
                "match": actual == str(expected),
            }
        )
    failures = [item for item in records if not item["match"]]
    if failures:
        raise ValueError(f"冻结输入哈希不匹配: {[item['key'] for item in failures]}")
    return records


def _build_forecaster(
    name: str, config: dict[str, Any], history: int, horizon: int
) -> torch.nn.Module:
    models = config["models"]
    if name == "LSTM":
        cfg = models["lstm"]
        return LSTMForecaster(
            input_size=4,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            horizon=horizon,
            dropout=float(cfg["dropout"]),
        )
    if name == "TCN":
        cfg = models["tcn"]
        return TCNForecaster(
            input_size=4,
            hidden_size=int(cfg["hidden_size"]),
            levels=int(cfg["levels"]),
            kernel_size=int(cfg["kernel_size"]),
            horizon=horizon,
            dropout=float(cfg["dropout"]),
        )
    if name == "PatchTST":
        cfg = models["patchtst"]
        return PatchTSTForecaster(
            input_size=4,
            history=history,
            horizon=horizon,
            patch_length=int(cfg["patch_length"]),
            patch_stride=int(cfg["patch_stride"]),
            hidden_size=int(cfg["hidden_size"]),
            attention_heads=int(cfg["attention_heads"]),
            layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]),
        )
    if name == "iTransformer":
        cfg = models["itransformer"]
        return ITransformerForecaster(
            input_size=4,
            history=history,
            horizon=horizon,
            hidden_size=int(cfg["hidden_size"]),
            attention_heads=int(cfg["attention_heads"]),
            layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]),
        )
    if name == "StateAware-TCN-Attention":
        cfg = models["state_aware_tcn_attention"]
        return StateAwareTCNAttention(
            numeric_input_size=4,
            num_states=len(STATE_TO_CODE),
            state_embedding_dim=int(cfg["state_embedding_dim"]),
            hidden_size=int(cfg["hidden_size"]),
            levels=int(cfg["levels"]),
            kernel_size=int(cfg["kernel_size"]),
            attention_heads=int(cfg["attention_heads"]),
            horizon=horizon,
            dropout=float(cfg["dropout"]),
        )
    if name == "StateAware-Patch-Transformer":
        cfg = models["state_aware_patch_transformer"]
        return StateAwarePatchTransformer(
            numeric_input_size=4,
            num_states=len(STATE_TO_CODE),
            state_embedding_dim=int(cfg["state_embedding_dim"]),
            history=history,
            horizon=horizon,
            patch_length=int(cfg["patch_length"]),
            patch_stride=int(cfg["patch_stride"]),
            hidden_size=int(cfg["hidden_size"]),
            attention_heads=int(cfg["attention_heads"]),
            layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]),
        )
    if name == "StateAware-DualBranch-Patch-Transformer":
        cfg = models["state_aware_dual_branch_patch_transformer"]
        return StateAwareDualBranchPatchTransformer(
            numeric_input_size=4,
            num_states=len(STATE_TO_CODE),
            state_embedding_dim=int(cfg["state_embedding_dim"]),
            transition_embedding_dim=int(cfg["transition_embedding_dim"]),
            history=history,
            horizon=horizon,
            patch_length=int(cfg["patch_length"]),
            patch_stride=int(cfg["patch_stride"]),
            hidden_size=int(cfg["hidden_size"]),
            attention_heads=int(cfg["attention_heads"]),
            layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]),
        )
    raise ValueError(f"未支持的冻结预测模型: {name}")


def _load_scaler(checkpoint_path: Path) -> PowerScaler:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    value = checkpoint["metadata"]["scaler"]
    return PowerScaler(mean=float(value["mean"]), std=float(value["std"]))


def _load_neural_model(
    name: str,
    path: Path,
    config: dict[str, Any],
    history: int,
    horizon: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = _build_forecaster(name, config, history, horizon)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)


def _causal_supply_fill(
    frame: pd.DataFrame, columns: list[str], limit: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    counts: dict[str, int] = {}
    for column in columns:
        before = output[column].isna()
        output[column] = pd.to_numeric(output[column], errors="coerce").ffill(limit=limit)
        counts[column] = int((before & output[column].notna()).sum())
    remaining = {name: int(output[name].isna().sum()) for name in columns}
    if any(remaining.values()):
        raise ValueError(f"因果前向填充后供能字段仍缺失: {remaining}")
    return output, counts


def _supply_regime(margin_ratio: np.ndarray) -> np.ndarray:
    value = np.asarray(margin_ratio, dtype=float)
    return np.select(
        [value >= 0.15, value >= 0.05, value >= 0.0],
        ["normal", "constrained", "weak"],
        default="emergency",
    )


def _forecast_metrics(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    transition_flags: np.ndarray,
    warmup: int,
    peak_threshold: float,
) -> pd.DataFrame:
    rows = []
    for name, prediction in predictions.items():
        row = {"model": name, "scored_samples": len(y_true) - warmup}
        row.update(
            regression_metrics(
                y_true[warmup:],
                prediction[warmup:],
                peak_threshold=peak_threshold,
                transition_flags=transition_flags[warmup:],
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_kw").reset_index(drop=True)


def _build_risk_outputs(
    *,
    frame: pd.DataFrame,
    stops: np.ndarray,
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    history_transition_flags: np.ndarray,
    forecast_config: dict[str, Any],
    risk_config: dict[str, Any],
    release_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    risk_cfg = risk_config
    horizon = int(forecast_config["forecast"]["horizon_steps"])
    forecast_sequence = predictions[str(risk_cfg["input"]["forecast_model"])]
    member_names = [str(name) for name in risk_cfg["input"]["uncertainty_members"]]
    member_predictions = np.stack([predictions[name] for name in member_names], axis=0)
    uncertainty_sequence = member_predictions.std(axis=0)
    origin = frame.iloc[stops - 1].reset_index(drop=True)
    target_timestamps = frame.iloc[stops]["timestamp"].reset_index(drop=True)
    supply_columns = [
        "grid_available_capacity_kw",
        "generator_available_capacity_kw",
        "storage_soc_pct",
        "storage_available_discharge_power_kw",
    ]
    supply = origin[supply_columns].astype(float).reset_index(drop=True)
    firm_supply = supply.sum(axis=1).to_numpy(float) - supply[
        "storage_soc_pct"
    ].to_numpy(float)

    label_cfg = risk_cfg["label"]
    labels, actual_margin_kw, actual_margin_ratio = assign_risk_levels(
        y_true,
        firm_supply,
        reserve_ratio=float(label_cfg["reserve_ratio"]),
        reserve_floor_kw=float(label_cfg["reserve_floor_kw"]),
        normal_margin_ratio=float(label_cfg["normal_margin_ratio"]),
        watch_margin_ratio=float(label_cfg["watch_margin_ratio"]),
        warning_margin_ratio=float(label_cfg["warning_margin_ratio"]),
    )
    current_load = predictions["Persistence"][:, 0].astype(float)
    forecast_peak = forecast_sequence.max(axis=1).astype(float)
    forecast_required = forecast_peak + np.maximum(
        float(label_cfg["reserve_floor_kw"]),
        float(label_cfg["reserve_ratio"]) * forecast_peak,
    )
    forecast_margin_ratio = (firm_supply - forecast_required) / np.maximum(
        forecast_required, 1.0
    )
    interval_seconds = float(forecast_config["project"]["frequency_seconds"])
    forecast_ramp = np.maximum(
        np.diff(np.column_stack([current_load, forecast_sequence]), axis=1), 0.0
    ).max(axis=1) / interval_seconds
    timestamp = pd.to_datetime(target_timestamps)
    seconds = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    phase = 2.0 * np.pi * seconds.to_numpy() / 86400.0
    uncertainty_mean = uncertainty_sequence.mean(axis=1)
    uncertainty_max = uncertainty_sequence.max(axis=1)
    static_names = [
        "current_load_kw",
        "forecast_mean_kw",
        "forecast_max_kw",
        "forecast_min_kw",
        "forecast_std_kw",
        "forecast_ramp_kw_per_s",
        "uncertainty_mean_kw",
        "uncertainty_max_kw",
        "grid_capacity_kw",
        "generator_available_kw",
        "storage_soc_pct",
        "storage_discharge_available_kw",
        "firm_supply_kw",
        "forecast_margin_ratio",
        "history_transition_flag",
        "operation_state_code",
        "time_sin",
        "time_cos",
    ]
    static = np.column_stack(
        [
            current_load,
            forecast_sequence.mean(axis=1),
            forecast_peak,
            forecast_sequence.min(axis=1),
            forecast_sequence.std(axis=1),
            forecast_ramp,
            uncertainty_mean,
            uncertainty_max,
            supply["grid_available_capacity_kw"],
            supply["generator_available_capacity_kw"],
            supply["storage_soc_pct"],
            supply["storage_available_discharge_power_kw"],
            firm_supply,
            forecast_margin_ratio,
            history_transition_flags,
            origin["operation_state_code"].to_numpy(),
            np.sin(phase),
            np.cos(phase),
        ]
    ).astype(np.float32)
    tree_features = np.column_stack([forecast_sequence, static]).astype(np.float32)
    expected_feature_names = [
        f"forecast_t+{index + 1}_kw" for index in range(horizon)
    ] + static_names
    model_dir = release_root / "risk" / "models"
    balanced_bundle = joblib.load(model_dir / "balanced_multiclass_xgboost_risk.joblib")
    ordinal_bundle = joblib.load(model_dir / "ordinal_xgboost_risk.joblib")
    if list(balanced_bundle["feature_names"]) != expected_feature_names:
        raise ValueError("冻结多分类风险模型特征顺序不一致")
    if list(ordinal_bundle["feature_names"]) != expected_feature_names:
        raise ValueError("冻结序数风险模型特征顺序不一致")
    balanced_probability = balanced_bundle["model"].predict_proba(tree_features)
    ordinal_probability = _ordinal_probabilities(
        ordinal_bundle["models"], tree_features
    )
    guard_probability = _safety_union_probabilities(
        balanced_probability, ordinal_probability
    )
    predicted = guard_probability.argmax(axis=1)
    risk_metrics = _classification_metrics(labels, guard_probability)

    evidence_cfg = risk_cfg["evidence"]
    evidence = build_risk_evidence(
        forecast_margin_ratio,
        supply["grid_available_capacity_kw"].to_numpy(),
        supply["storage_soc_pct"].to_numpy(),
        uncertainty_max,
        forecast_ramp,
        grid_derating_threshold_kw=float(evidence_cfg["grid_derating_threshold_kw"]),
        low_soc_threshold_pct=float(evidence_cfg["low_soc_threshold_pct"]),
        uncertainty_threshold_kw=float(evidence_cfg["uncertainty_threshold_kw"]),
        ramp_threshold_kw_per_s=float(evidence_cfg["ramp_threshold_kw_per_s"]),
    )
    reserve_adders = risk_cfg["dispatch_signal"]["reserve_adder_kw"]
    output = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true), dtype=int),
            "timestamp": timestamp,
            "supply_regime": _supply_regime(forecast_margin_ratio),
            "supply_source_type": origin["source_type"].astype(str).to_numpy(),
            "current_load_kw": current_load,
            "forecast_peak_kw": forecast_peak,
            "actual_peak_kw": y_true.max(axis=1),
            "firm_supply_kw": firm_supply,
            "actual_margin_kw": actual_margin_kw,
            "actual_margin_ratio": actual_margin_ratio,
            "forecast_margin_ratio": forecast_margin_ratio,
            "grid_available_capacity_kw": supply["grid_available_capacity_kw"],
            "generator_available_capacity_kw": supply[
                "generator_available_capacity_kw"
            ],
            "storage_soc_pct": supply["storage_soc_pct"],
            "storage_available_discharge_power_kw": supply[
                "storage_available_discharge_power_kw"
            ],
            "forecast_uncertainty_kw": uncertainty_max,
            "forecast_ramp_kw_per_s": forecast_ramp,
            "true_risk_level": labels,
            "true_risk_name": [RISK_NAMES[value] for value in labels],
            "predicted_risk_level": predicted,
            "predicted_risk_name": [RISK_NAMES[value] for value in predicted],
            "dispatch_risk_level": predicted,
            "dispatch_risk_name": [RISK_NAMES[value] for value in predicted],
            "risk_confidence": guard_probability.max(axis=1),
            "prob_normal": guard_probability[:, 0],
            "prob_watch": guard_probability[:, 1],
            "prob_warning": guard_probability[:, 2],
            "prob_severe": guard_probability[:, 3],
            "risk_evidence": evidence,
            "reserve_adder_kw": [
                float(reserve_adders[str(value)]) for value in predicted
            ],
        }
    )
    metrics = {
        "model": "Balanced-Ordinal-Safety-Max",
        "sample_count": len(output),
        "class_counts": {
            str(level): int((labels == level).sum()) for level in range(4)
        },
        **risk_metrics,
    }
    return output, metrics


def run_frozen_v15_external_acceptance(
    protocol_path: Path,
    *,
    force_cpu: bool = False,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    project_dir = protocol_path.parents[1]
    protocol = _read_yaml(protocol_path)
    paths = protocol["paths"]
    artifact_dir = (project_dir / paths["artifact_dir"]).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_records = verify_locked_inputs(protocol, project_dir)
    (artifact_dir / "locked_input_manifest.json").write_text(
        json.dumps(lock_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "preregistered_protocol.yaml").write_text(
        protocol_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    release_root = (project_dir / paths["release_root"]).resolve()
    data_path = Path(paths["data"]).resolve()
    frame, data_audit = load_external_data_v16(
        data_path, require_target_rig_field_data=False
    )
    required_supply = [str(value) for value in protocol["data_handling"]["required_supply_columns"]]
    frame, supply_fill_counts = _causal_supply_fill(
        frame,
        required_supply,
        int(protocol["data_handling"]["supply_forward_fill_limit_steps"]),
    )
    (artifact_dir / "external_data_audit.json").write_text(
        json.dumps(
            {
                **data_audit.as_dict(),
                "acceptance_usage_class": protocol["protocol"]["data_usage_class"],
                "original_provenance_preserved": True,
                "field_origin_claim_allowed": bool(
                    protocol["protocol"]["field_origin_claim_allowed"]
                ),
                "supply_causal_forward_fill_rows": supply_fill_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    forecast_config = _read_yaml(release_root / "source" / "resolved_forecast_config.yaml")
    risk_config = _read_yaml(release_root / "source" / "resolved_risk_config.yaml")
    release_cfg = protocol["forecast_release"]
    history = int(release_cfg["history_steps"])
    horizon = int(release_cfg["horizon_steps"])
    stride = int(release_cfg["stride"])
    if history != int(forecast_config["forecast"]["history_steps"]):
        raise ValueError("外部协议与冻结历史窗长不一致")
    if horizon != int(forecast_config["forecast"]["horizon_steps"]):
        raise ValueError("外部协议与冻结预测时域不一致")

    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    x_tree, y_true, transition_flags = make_tree_windows(
        frame, history, horizon, stride
    )
    stops = np.arange(history, len(frame) - horizon + 1, stride, dtype=np.int64)
    persistence = np.repeat(
        frame["total_active_power_kw"].to_numpy(np.float32)[stops - 1, None],
        horizon,
        axis=1,
    )
    predictions: dict[str, np.ndarray] = {"Persistence": persistence}
    started = time.perf_counter()
    tree_model = joblib.load(
        release_root / "source" / "forecast" / "models" / "xgboost_multi_horizon.joblib"
    )
    predictions["XGBoost"] = tree_model.predict(x_tree).astype(np.float32)
    inference_seconds: dict[str, float] = {"XGBoost": time.perf_counter() - started}

    checkpoint_dir = release_root / "source" / "forecast" / "models"
    scaler = _load_scaler(checkpoint_dir / "lstm.pt")
    dataset = WindowedRigDataset(frame, scaler, history, horizon, stride)
    loader = DataLoader(
        dataset,
        batch_size=int(release_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    for name, filename in FORECAST_MODEL_FILES.items():
        model = _load_neural_model(
            name,
            checkpoint_dir / filename,
            forecast_config,
            history,
            horizon,
            device,
        )
        neural_true, prediction, neural_transition, elapsed = predict_neural_model(
            model, loader, scaler, device=device
        )
        if not np.allclose(neural_true, y_true, rtol=0.0, atol=1e-3):
            raise ValueError(f"{name} 与树模型外部窗口未对齐")
        if not np.array_equal(neural_transition, transition_flags):
            raise ValueError(f"{name} 工况切换标记未对齐")
        predictions[name] = prediction.astype(np.float32)
        inference_seconds[name] = elapsed
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_record = json.loads(
        (release_root / "source" / "forecast" / "ensemble_weights.json").read_text(
            encoding="utf-8"
        )
    )
    ensemble_weights = {
        str(name): float(weight) for name, weight in ensemble_record["weights"].items()
    }
    predictions["Validation-Weighted-Ensemble"] = apply_weighted_ensemble(
        predictions, ensemble_weights
    ).astype(np.float32)
    feedback_record = json.loads(
        (release_root / "source" / "forecast" / "error_feedback.json").read_text(
            encoding="utf-8"
        )
    )
    feedback_prediction, final_bias = apply_causal_error_feedback(
        predictions["Validation-Weighted-Ensemble"],
        y_true,
        feedback_delay_steps=int(feedback_record["feedback_delay_steps"]),
        smoothing=float(feedback_record["selected_smoothing"]),
        correction_clip_kw=float(feedback_record["selected_correction_clip_kw"]),
        initial_bias_kw=np.zeros(horizon, dtype=np.float32),
    )
    predictions["Causal-ErrorFeedback-Ensemble"] = feedback_prediction.astype(
        np.float32
    )
    envelope_record = json.loads(
        (
            release_root
            / "source"
            / "forecast"
            / "dispatch_uncertainty_envelope.json"
        ).read_text(encoding="utf-8")
    )
    for item in envelope_record["candidates"]:
        correction = np.asarray(item["horizon_correction_kw"], dtype=np.float32)
        predictions[str(item["name"])] = np.maximum(
            feedback_prediction + correction[None, :], 0.0
        ).astype(np.float32)

    finite_status = {
        name: bool(np.isfinite(value).all()) for name, value in predictions.items()
    }
    if not all(finite_status.values()):
        raise ValueError(
            f"外部预测含非有限值: {[k for k, v in finite_status.items() if not v]}"
        )
    warmup = int(release_cfg["warmup_excluded_samples"])
    peak_threshold = float(np.quantile(y_true, 0.90))
    metrics = _forecast_metrics(
        y_true, predictions, transition_flags, warmup, peak_threshold
    )
    metrics.to_csv(
        artifact_dir / "forecast_metrics.csv", index=False, encoding="utf-8-sig"
    )

    states = frame["operation_state_code"].to_numpy(np.int16)
    history_transition = np.asarray(
        [
            _transition_flags(states, stop - history, stop, stop + horizon)[0]
            for stop in stops
        ],
        dtype=np.int8,
    )
    prediction_arrays: dict[str, np.ndarray] = {
        "y_true": y_true.astype(np.float32),
        "transition_flags": transition_flags.astype(np.int8),
        "future_transition_flags": transition_flags.astype(np.int8),
        "history_transition_flags": history_transition,
        "operation_state_codes": states[stops],
        "window_stops": stops,
    }
    key_map: dict[str, str] = {}
    for index, (name, prediction) in enumerate(predictions.items()):
        key = f"prediction_{index}"
        prediction_arrays[key] = prediction.astype(np.float32)
        key_map[key] = name
    np.savez_compressed(
        artifact_dir / "forecast_predictions.npz", **prediction_arrays
    )
    (artifact_dir / "forecast_prediction_keys.json").write_text(
        json.dumps(key_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_forecasts(
        y_true[warmup:],
        {
            "Persistence": predictions["Persistence"][warmup:],
            "Validation-Weighted-Ensemble": predictions[
                "Validation-Weighted-Ensemble"
            ][warmup:],
            "Causal-ErrorFeedback-Ensemble": predictions[
                "Causal-ErrorFeedback-Ensemble"
            ][warmup:],
        },
        artifact_dir / "external_forecast_overview.png",
    )
    _plot_metric_bars(metrics, artifact_dir / "external_forecast_metrics.png")

    risk_output, risk_metrics = _build_risk_outputs(
        frame=frame,
        stops=stops,
        predictions=predictions,
        y_true=y_true,
        history_transition_flags=history_transition,
        forecast_config=forecast_config,
        risk_config=risk_config,
        release_root=release_root,
    )
    risk_output.to_csv(
        artifact_dir / "risk_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (artifact_dir / "risk_metrics.json").write_text(
        json.dumps(risk_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_timeline(risk_output, artifact_dir / "external_risk_timeline.png")

    primary_name = str(release_cfg["primary_model"])
    baseline_name = str(release_cfg["baseline_model"])
    metric_lookup = metrics.set_index("model")
    primary_mae = float(metric_lookup.loc[primary_name, "mae_kw"])
    baseline_mae = float(metric_lookup.loc[baseline_name, "mae_kw"])
    improvement = (baseline_mae - primary_mae) / baseline_mae
    gates = protocol["acceptance_gates"]
    gate_results = {
        "forecast_primary_r2": {
            "value": float(metric_lookup.loc[primary_name, "r2"]),
            "threshold": float(gates["forecast_primary_r2_min"]),
            "passed": float(metric_lookup.loc[primary_name, "r2"])
            >= float(gates["forecast_primary_r2_min"]),
        },
        "forecast_primary_mae_improvement_over_persistence": {
            "value": improvement,
            "threshold": float(
                gates["forecast_primary_mae_improvement_over_persistence_min"]
            ),
            "passed": improvement
            >= float(gates["forecast_primary_mae_improvement_over_persistence_min"]),
        },
        "forecast_predictions_finite": {
            "value": all(finite_status.values()),
            "threshold": True,
            "passed": all(finite_status.values()),
        },
        "risk_macro_f1": {
            "value": float(risk_metrics["macro_f1"]),
            "threshold": float(gates["risk_macro_f1_min"]),
            "passed": float(risk_metrics["macro_f1"])
            >= float(gates["risk_macro_f1_min"]),
        },
        "risk_high_recall": {
            "value": float(risk_metrics["high_risk_recall"]),
            "threshold": float(gates["risk_high_recall_min"]),
            "passed": float(risk_metrics["high_risk_recall"])
            >= float(gates["risk_high_recall_min"]),
        },
        "risk_severe_recall": {
            "value": float(risk_metrics["severe_recall"]),
            "threshold": float(gates["risk_severe_recall_min"]),
            "passed": float(risk_metrics["severe_recall"])
            >= float(gates["risk_severe_recall_min"]),
        },
    }
    result = {
        "protocol_id": protocol["protocol"]["id"],
        "protocol_status_at_execution": protocol["protocol"]["status"],
        "acceptance_usage_class": protocol["protocol"]["data_usage_class"],
        "original_provenance_class": data_audit.provenance_class,
        "original_record_origins": data_audit.record_origins,
        "field_origin_claim_allowed": protocol["protocol"]["field_origin_claim_allowed"],
        "no_retraining_performed": True,
        "forecast_sample_count": len(y_true),
        "forecast_scored_sample_count": len(y_true) - warmup,
        "forecast_time_start": str(frame.iloc[stops[0]]["timestamp"]),
        "forecast_time_end": str(frame.iloc[stops[-1] + horizon - 1]["timestamp"]),
        "feedback_initialization": "zero_bias_cold_start",
        "feedback_final_bias_kw": np.asarray(final_bias, dtype=float).tolist(),
        "inference_seconds": inference_seconds,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "risk_release_limitation": protocol["risk_release"][
            "unavailable_release_component"
        ],
        "gate_results": gate_results,
        "forecast_and_risk_all_gates_passed": all(
            item["passed"] for item in gate_results.values()
        ),
    }
    (artifact_dir / "acceptance_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
