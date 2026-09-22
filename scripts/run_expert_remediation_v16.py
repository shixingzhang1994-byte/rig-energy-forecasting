from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.external_adapter_v16 import load_external_data_v16  # noqa: E402
from rig_energy.safety.emergency_load import (  # noqa: E402
    TIER_NAMES,
    allocate_priority_greedy,
    allocate_priority_highs,
    allocate_proportional,
)
from rig_energy.validation.statistical_claims import (  # noqa: E402
    exact_sign_flip_pvalue,
    paired_block_bootstrap_ci,
)


ANALYSIS_REQUIRED_COLUMNS = (
    "total_active_power_kw",
    "pre_shed_demand_kw",
    "grid_available_capacity_kw",
    "generator_available_capacity_kw",
    "storage_available_discharge_power_kw",
    "storage_soc_pct",
    *(f"genset_{unit}_active_power_kw" for unit in range(1, 5)),
    *(f"genset_{unit}_run_status" for unit in range(1, 5)),
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_DIR / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _verify_source_manifest(manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    records = []
    failures = []
    for item in manifest.get("files", []):
        path = manifest_path.parent / item["name"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = _sha256(path) if exists else None
        valid = bool(
            exists
            and actual_bytes == int(item["bytes"])
            and actual_sha256 == str(item["sha256"])
        )
        records.append(
            {
                "name": item["name"],
                "exists": exists,
                "bytes_match": exists and actual_bytes == int(item["bytes"]),
                "sha256_match": exists and actual_sha256 == str(item["sha256"]),
                "valid": valid,
            }
        )
        if not valid:
            failures.append(item["name"])
    expected_count = int(manifest.get("file_count_excluding_manifest", len(records)))
    result = {
        "dataset_id": manifest.get("dataset_id"),
        "record_origin": manifest.get("record_origin"),
        "target_rig_field_data": manifest.get("target_rig_field_data"),
        "expected_file_count": expected_count,
        "checked_file_count": len(records),
        "all_declared_files_valid": not failures and len(records) == expected_count,
        "failures": failures,
        "records": records,
    }
    if not result["all_declared_files_valid"]:
        raise RuntimeError(f"来源数据包清单校验失败: {failures}")
    return result


def _select_analysis_eligible(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_columns = [name for name in ANALYSIS_REQUIRED_COLUMNS if name not in frame]
    if missing_columns:
        raise ValueError(f"整改分析缺少必需字段: {missing_columns}")
    finite = np.ones(len(frame), dtype=bool)
    missing_by_column = {}
    for name in ANALYSIS_REQUIRED_COLUMNS:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(float))
        missing_by_column[name] = int(invalid.sum())
        finite &= ~invalid
    selected = frame.loc[finite].reset_index(drop=True)
    if selected.empty:
        raise ValueError("整改分析没有满足必需字段完整性的记录")
    return selected, {
        "input_rows_after_resampling": int(len(frame)),
        "analysis_eligible_rows": int(len(selected)),
        "excluded_rows": int((~finite).sum()),
        "selection_rule": "all emergency-allocation and parameter-identification fields finite",
        "missing_by_required_column": missing_by_column,
        "truth_used_for_repair": False,
    }


def _tier_demands(frame: pd.DataFrame, config: dict) -> np.ndarray:
    mapping = config["load_priority_placeholder"]["fractions_by_state"]
    unknown = np.asarray(mapping["unknown"], dtype=float)
    fractions = np.stack(
        [np.asarray(mapping.get(str(state), unknown), dtype=float) for state in frame["operation_state"]]
    )
    if not np.allclose(fractions.sum(axis=1), 1.0):
        raise ValueError("各工况三级负荷比例之和必须为1")
    demand_column = str(config["input"].get("demand_column", "total_active_power_kw"))
    return frame[demand_column].to_numpy(float)[:, None] * fractions


def _normalized_capacities(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    norm = config["v15_capacity_normalization"]
    grid = (
        frame["grid_available_capacity_kw"].to_numpy(float)
        / float(norm["source_grid_rated_kw"])
        * float(norm["target_grid_rated_kw"])
    )
    generator = (
        frame["generator_available_capacity_kw"].to_numpy(float)
        / float(norm["source_generator_unit_rated_kw"])
        * float(norm["target_generator_unit_rated_kw"])
    )
    storage = np.minimum(
        frame["storage_available_discharge_power_kw"].to_numpy(float),
        float(norm["target_storage_power_kw"]),
    )
    storage = np.where(
        frame["storage_soc_pct"].to_numpy(float)
        <= float(norm["target_storage_soc_min_pct"]),
        0.0,
        storage,
    )
    return {"grid": grid, "generator": generator, "storage": storage}


def _stress_case(
    frame: pd.DataFrame,
    tier_demand: np.ndarray,
    base_capacity: dict[str, np.ndarray],
    case: dict,
    config: dict,
) -> dict[str, np.ndarray]:
    norm = config["v15_capacity_normalization"]
    load_factor = float(case["load_measurement_factor"])
    demand = tier_demand * load_factor
    grid = base_capacity["grid"] * float(case["grid_capacity_factor"])
    generator = np.maximum(
        0.0,
        base_capacity["generator"]
        - int(case["unavailable_generator_units"])
        * float(norm["target_generator_unit_rated_kw"]),
    )
    biased_soc = (
        frame["storage_soc_pct"].to_numpy(float)
        + float(case["storage_soc_bias_pct"])
    )
    storage = np.minimum(
        base_capacity["storage"], float(case["storage_power_limit_kw"])
    )
    storage = np.where(
        biased_soc <= float(norm["target_storage_soc_min_pct"]), 0.0, storage
    )
    actual_available = grid + generator + storage
    delay = int(case["capacity_telemetry_delay_steps"])
    reported_available = actual_available.copy()
    if delay > 0:
        reported_available[delay:] = actual_available[:-delay]
        reported_available[:delay] = actual_available[0]
    return {
        "demand_by_tier": demand,
        "actual_available": actual_available,
        "reported_available": reported_available,
        "telemetry_overstatement": np.maximum(
            0.0, reported_available - actual_available
        ),
    }


def _allocation_metrics(allocation, *, dt_hours: float) -> dict:
    shed = allocation.shed_kw
    served = allocation.served_kw
    demand = shed + served
    result = {
        "method": allocation.method,
        "total_unserved_energy_kwh": float(shed.sum() * dt_hours),
        "maximum_unserved_power_kw": float(shed.sum(axis=1).max()),
        "unserved_points": int((shed.sum(axis=1) > 1e-6).sum()),
        "solver_metadata": allocation.solver_metadata,
    }
    for index, name in enumerate(TIER_NAMES):
        demand_energy = float(demand[:, index].sum() * dt_hours)
        unserved = float(shed[:, index].sum() * dt_hours)
        result[f"{name}_demand_energy_kwh"] = demand_energy
        result[f"{name}_unserved_energy_kwh"] = unserved
        result[f"{name}_maximum_unserved_power_kw"] = float(shed[:, index].max())
        result[f"{name}_served_fraction"] = (
            1.0 if demand_energy <= 1e-12 else 1.0 - unserved / demand_energy
        )
    return result


def _run_emergency_audit(frame: pd.DataFrame, config: dict, output: Path) -> dict:
    dt_hours = float(config["protocol"]["interval_seconds"]) / 3600.0
    tier_demand = _tier_demands(frame, config)
    base_capacity = _normalized_capacities(frame, config)
    rows = []
    state_rows = []
    case_details = {}
    joint_plot = None
    for name, case in config["stress_scenarios"].items():
        stressed = _stress_case(frame, tier_demand, base_capacity, case, config)
        proportional = allocate_proportional(
            stressed["demand_by_tier"], stressed["actual_available"]
        )
        priority = allocate_priority_greedy(
            stressed["demand_by_tier"], stressed["actual_available"]
        )
        highs = allocate_priority_highs(
            stressed["demand_by_tier"], stressed["actual_available"]
        )
        max_difference = float(np.max(np.abs(priority.served_kw - highs.served_kw)))
        if max_difference > 1e-5:
            raise RuntimeError(f"{name}: 优先策略与HiGHS参考不一致 {max_difference}")
        metrics = []
        case_state_metrics = []
        for allocation in (proportional, priority, highs):
            item = _allocation_metrics(allocation, dt_hours=dt_hours)
            item.update({"scenario": name, "highs_equivalence_max_kw": max_difference})
            metrics.append(item)
            flat = {key: value for key, value in item.items() if key != "solver_metadata"}
            rows.append(flat)
            for state in sorted(frame["operation_state"].astype(str).unique()):
                state_mask = frame["operation_state"].astype(str).to_numpy() == state
                state_metrics = _allocation_metrics(
                    type(allocation)(
                        served_kw=allocation.served_kw[state_mask],
                        shed_kw=allocation.shed_kw[state_mask],
                        method=allocation.method,
                        solver_metadata=allocation.solver_metadata,
                    ),
                    dt_hours=dt_hours,
                )
                state_item = {
                    "scenario": name,
                    "operation_state": state,
                    **{
                        key: value
                        for key, value in state_metrics.items()
                        if key != "solver_metadata"
                    },
                }
                state_rows.append(state_item)
                case_state_metrics.append(state_item)
        case_details[name] = {
            "metrics": metrics,
            "minimum_actual_available_kw": float(stressed["actual_available"].min()),
            "maximum_demand_kw": float(stressed["demand_by_tier"].sum(axis=1).max()),
            "telemetry_overstatement_energy_kwh": float(
                stressed["telemetry_overstatement"].sum() * dt_hours
            ),
            "critical_unserved_reduction_vs_proportional_kwh": float(
                (proportional.shed_kw[:, 0] - priority.shed_kw[:, 0]).sum()
                * dt_hours
            ),
            "priority_definition_status": config["load_priority_placeholder"][
                "approval_status"
            ],
            "state_metrics": case_state_metrics,
        }
        if name == "base_v15_normalized" and "load_shed_kw" in frame:
            source_unserved = float(frame["load_shed_kw"].sum() * dt_hours)
            replay_unserved = next(
                item["total_unserved_energy_kwh"]
                for item in metrics
                if item["method"] == "priority-greedy"
            )
            case_details[name]["source_load_shed_energy_kwh"] = source_unserved
            case_details[name]["replay_minus_source_kwh"] = float(
                replay_unserved - source_unserved
            )
        if name == "joint_extreme":
            joint_plot = (stressed, proportional, priority)

    pd.DataFrame(rows).to_csv(
        output / "emergency_load_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(state_rows).to_csv(
        output / "emergency_load_state_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if joint_plot is not None:
        stressed, proportional, priority = joint_plot
        total_shed = proportional.shed_kw.sum(axis=1)
        worst = int(np.argmax(total_shed))
        start = max(0, worst - 180)
        stop = min(len(frame), worst + 181)
        x = frame["timestamp"].iloc[start:stop]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(
            x,
            stressed["demand_by_tier"][start:stop].sum(axis=1),
            label="Pre-shed demand",
            color="#111827",
        )
        axes[0].plot(
            x,
            stressed["actual_available"][start:stop],
            label="Actual available supply under joint stress",
            color="#DC2626",
        )
        axes[0].legend()
        axes[0].set_ylabel("Power (kW)")
        axes[0].grid(alpha=0.2)
        axes[1].plot(
            x,
            proportional.shed_kw[start:stop, 0],
            label="Critical unserved: proportional",
            color="#DC2626",
        )
        axes[1].plot(
            x,
            priority.shed_kw[start:stop, 0],
            label="Critical unserved: priority protection",
            color="#059669",
        )
        axes[1].set_ylabel("Critical unserved (kW)")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        fig.suptitle("Critical-load protection under joint stress")
        fig.tight_layout()
        fig.savefig(output / "joint_extreme_critical_load_protection.png", dpi=180)
        plt.close(fig)
    return case_details


def _segments(binary: np.ndarray, interval_seconds: float) -> tuple[list[float], list[float]]:
    values = np.asarray(binary, dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    stops = np.r_[starts[1:], len(values)]
    on = []
    off = []
    for start, stop in zip(starts, stops):
        target = on if values[start] == 1 else off
        target.append(float((stop - start) * interval_seconds))
    return on, off


def _parameter_identification(frame: pd.DataFrame, config: dict, output: Path) -> list[dict]:
    dt = float(config["protocol"]["interval_seconds"])
    records = []
    for unit in range(1, 5):
        power = frame[f"genset_{unit}_active_power_kw"].to_numpy(float)
        run = frame[f"genset_{unit}_run_status"].to_numpy(int)
        both_on = (run[1:] == 1) & (run[:-1] == 1)
        ramp = np.diff(power)[both_on] if both_on.any() else np.array([0.0])
        on, off = _segments(run, dt)
        positive = power[power > 1e-6]
        values = {
            "observed_max_power_kw": float(power.max()),
            "observed_p05_positive_power_kw": float(np.quantile(positive, 0.05)) if len(positive) else 0.0,
            "observed_max_ramp_up_kw_per_step": float(max(0.0, ramp.max())),
            "observed_max_ramp_down_kw_per_step": float(max(0.0, -ramp.min())),
            "observed_min_on_seconds": float(min(on)) if on else 0.0,
            "observed_min_off_seconds": float(min(off)) if off else 0.0,
            "observed_median_on_seconds": float(np.median(on)) if on else 0.0,
            "observed_median_off_seconds": float(np.median(off)) if off else 0.0,
        }
        for parameter, value in values.items():
            records.append(
                {
                    "asset_id": f"G{unit}",
                    "parameter": parameter,
                    "identified_value": value,
                    "provenance_class": "SURROGATE_IDENTIFIED",
                    "can_replace_target_rig_parameter": False,
                }
            )
    storage_values = {
        "observed_max_discharge_power_kw": float(frame["storage_active_power_kw"].max()),
        "observed_max_charge_power_kw": float(-frame["storage_active_power_kw"].min()),
        "observed_min_soc_pct": float(frame["storage_soc_pct"].min()),
        "observed_max_soc_pct": float(frame["storage_soc_pct"].max()),
    }
    for parameter, value in storage_values.items():
        records.append(
            {
                "asset_id": "BESS01",
                "parameter": parameter,
                "identified_value": value,
                "provenance_class": "SURROGATE_IDENTIFIED",
                "can_replace_target_rig_parameter": False,
            }
        )
    pd.DataFrame(records).to_csv(
        output / "surrogate_parameter_identification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return records


def _parameter_gap_register(config: dict, output: Path) -> list[dict]:
    records = [dict(item) for item in config.get("parameter_evidence_gaps", [])]
    pd.DataFrame(records).to_csv(
        output / "target_parameter_evidence_gap_register.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return records


def _inject_missing_blocks(
    values: np.ndarray,
    *,
    target_fraction: float,
    block_seconds: list[int],
    interval_seconds: int,
    rng: np.random.Generator,
    protected_prefix: int,
) -> tuple[np.ndarray, np.ndarray]:
    corrupted = np.asarray(values, dtype=float).copy()
    mask = np.zeros(len(corrupted), dtype=bool)
    target = int(round(target_fraction * len(mask)))
    if target == 0:
        return corrupted, mask
    low = max(1, int(np.ceil(block_seconds[0] / interval_seconds)))
    high = max(low, int(np.ceil(block_seconds[1] / interval_seconds)))
    while int(mask.sum()) < target:
        length = int(rng.integers(low, high + 1))
        start = int(rng.integers(protected_prefix, max(protected_prefix + 1, len(mask) - length)))
        mask[start : start + length] = True
    corrupted[mask] = np.nan
    series = pd.Series(corrupted).ffill()
    if series.isna().any():
        series = series.bfill()
    return series.to_numpy(float), mask


def _window_matrix(
    input_power: np.ndarray,
    truth_power: np.ndarray,
    missing_mask: np.ndarray,
    *,
    history: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stops = np.arange(history, len(input_power) - horizon + 1, stride)
    lag_offsets = np.unique(np.linspace(0, history - 1, 24, dtype=int))
    x_power = np.stack([input_power[stop - history + lag_offsets] for stop in stops])
    x_missing = np.stack([missing_mask[stop - history + lag_offsets] for stop in stops])
    x = np.concatenate(
        [x_power, x_missing.astype(float), missing_mask[stops - 1][:, None]],
        axis=1,
    )
    y = np.stack([truth_power[stop : stop + horizon] for stop in stops])
    return x, y, stops


def _missingness_retraining(frame: pd.DataFrame, config: dict, output: Path) -> list[dict]:
    cfg = config["missingness_retraining_diagnostic"]
    truth = frame["total_active_power_kw"].to_numpy(float)
    history = int(cfg["history_steps"])
    horizon = int(cfg["horizon_steps"])
    stride = int(cfg["stride"])
    interval = int(config["protocol"]["interval_seconds"])
    rows = []
    for offset, (name, condition) in enumerate(cfg["conditions"].items()):
        rng = np.random.default_rng(int(config["protocol"]["random_seed"]) + offset)
        corrupted, mask = _inject_missing_blocks(
            truth,
            target_fraction=float(condition["target_missing_fraction"]),
            block_seconds=list(condition["block_seconds"]),
            interval_seconds=interval,
            rng=rng,
            protected_prefix=history,
        )
        x, y, _ = _window_matrix(
            corrupted,
            truth,
            mask,
            history=history,
            horizon=horizon,
            stride=stride,
        )
        n = len(x)
        train_stop = int(float(cfg["train_fraction"]) * n)
        test_start = int((1.0 - float(cfg["test_fraction"])) * n)
        estimator = xgb.XGBRegressor(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            learning_rate=float(cfg["learning_rate"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            objective="reg:squarederror",
            tree_method="hist",
            device=str(cfg["device"]),
            random_state=int(config["protocol"]["random_seed"]) + offset,
            n_jobs=1,
        )
        model = MultiOutputRegressor(estimator, n_jobs=1)
        started = time.perf_counter()
        model.fit(x[:train_stop], y[:train_stop])
        training_seconds = time.perf_counter() - started
        prediction = model.predict(x[test_start:])
        target = y[test_start:]
        error = np.abs(prediction - target)
        peak_threshold = float(np.quantile(target, 0.90))
        peak_mask = target >= peak_threshold
        rows.append(
            {
                "condition": name,
                "model": cfg["model"],
                "target_missing_fraction": float(condition["target_missing_fraction"]),
                "realized_missing_fraction": float(mask.mean()),
                "mae_kw": float(mean_absolute_error(target, prediction)),
                "rmse_kw": float(np.sqrt(mean_squared_error(target, prediction))),
                "peak_mae_kw": float(error[peak_mask].mean()),
                "training_seconds": training_seconds,
                "training_device": str(cfg["device"]),
                "xgboost_version": xgb.__version__,
                "train_windows": train_stop,
                "test_windows": n - test_start,
                "claim_boundary": cfg["claim_boundary"],
            }
        )
        joblib.dump(model, output / f"missingness_model_{name}.joblib")
    reference_mae = next(row["mae_kw"] for row in rows if row["condition"] == "reference")
    for row in rows:
        degradation = 100.0 * (row["mae_kw"] / reference_mae - 1.0)
        row["mae_degradation_vs_reference_pct"] = float(degradation)
        if row["condition"] == "moderate_blocks":
            row["diagnostic_gate_pass"] = bool(
                degradation <= float(cfg["moderate_mae_degradation_limit_pct"])
            )
        elif row["condition"] == "severe_blocks":
            row["diagnostic_gate_pass"] = bool(
                degradation <= float(cfg["severe_mae_degradation_limit_pct"])
            )
        else:
            row["diagnostic_gate_pass"] = True
    pd.DataFrame(rows).to_csv(
        output / "missingness_retraining_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return rows


def _statistical_claim_audit(config: dict, output: Path) -> dict:
    cfg = config["statistical_claim_audit"]
    root = PROJECT_DIR / "artifacts/v15_generator_first_reserve"
    records = []
    for seed in cfg["holdout_seeds"]:
        metrics = pd.read_csv(root / f"seed_{seed}/dispatch/dispatch_metrics.csv")
        for scenario, group in metrics.groupby("scenario"):
            project = float(
                group.loc[
                    group["method"] == cfg["project_method"], "unserved_energy_kwh"
                ].iloc[0]
            )
            references = {
                method: float(
                    group.loc[group["method"] == method, "unserved_energy_kwh"].iloc[0]
                )
                for method in cfg["reference_methods"]
            }
            best_name = min(references, key=references.get)
            best = references[best_name]
            records.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "project_unserved_kwh": project,
                    "best_reference": best_name,
                    "best_reference_unserved_kwh": best,
                    "improvement_kwh": best - project,
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(
        output / "v15_paired_scenario_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences = table["improvement_kwh"].to_numpy(float)
    low, high = paired_block_bootstrap_ci(
        differences,
        confidence=float(cfg["confidence"]),
        samples=int(cfg["bootstrap_samples"]),
        seed=int(config["protocol"]["random_seed"]),
    )
    pvalue = exact_sign_flip_pvalue(differences)
    significant = bool(pvalue < 0.05 and low > 0.0)
    result = {
        "paired_units": int(len(differences)),
        "positive_units": int((differences > 1e-12).sum()),
        "ties": int((np.abs(differences) <= 1e-12).sum()),
        "mean_improvement_kwh": float(differences.mean()),
        "median_improvement_kwh": float(np.median(differences)),
        "bootstrap_mean_ci_kwh": [low, high],
        "exact_two_sided_sign_flip_pvalue": pvalue,
        "statistically_significant_at_0p05": significant,
        "sota_claim_allowed": False,
        "allowed_claim": "冻结V15在两个合成留出上安全非劣并取得描述性小幅改善",
        "forbidden_claim": "统计显著SOTA或普遍优于强基线",
    }
    _json_dump(output / "v15_statistical_claim_audit.json", result)
    return result


def _write_report(
    output: Path,
    *,
    audit: dict,
    emergency: dict,
    parameters: list[dict],
    missingness: list[dict],
    statistics: dict,
    selection: dict,
    parameter_gaps: list[dict],
    source_manifest: dict,
    config: dict,
) -> None:
    joint = emergency["joint_extreme"]
    priority = next(
        item for item in joint["metrics"] if item["method"] == "priority-greedy"
    )
    proportional = next(
        item
        for item in joint["metrics"]
        if item["method"] == "proportional-curtailment"
    )
    base = emergency["base_v15_normalized"]
    base_tripping_priority = next(
        item
        for item in base["state_metrics"]
        if item["operation_state"] == "tripping"
        and item["method"] == "priority-greedy"
    )
    base_tripping_proportional = next(
        item
        for item in base["state_metrics"]
        if item["operation_state"] == "tripping"
        and item["method"] == "proportional-curtailment"
    )
    missing_rows = "\n".join(
        f"| {row['condition']} | {row['realized_missing_fraction']:.3%} | "
        f"{row['mae_kw']:.3f} | {row['peak_mae_kw']:.3f} | "
        f"{row['mae_degradation_vs_reference_pct']:.2f}% | {row['diagnostic_gate_pass']} |"
        for row in missingness
    )
    closure = config["closure_boundary"]
    report = f"""# 六项专家问题整改复审报告（V16前置，机理数据）

## 结论

本轮使用的数据类别为 `{audit['provenance_class']}`，不是目标钻机SCADA。完成的是来源防误标、统计声明审计、HiGHS独立安全基线、关键负荷优先、参数识别、缺失重训诊断和联合极端软件回放；未关闭现场数据和半实物闭环外部依赖。

输入重采样后 {selection['input_rows_after_resampling']} 行，其中 {selection['analysis_eligible_rows']} 行满足本轮供能与参数分析字段完整性；排除 {selection['excluded_rows']} 行，未使用生成器真值修补。

来源数据包清单声明的 {source_manifest['checked_file_count']} 个文件已逐一通过字节数和SHA-256校验。

## 六项问题闭环矩阵

| 专家问题 | 本轮处理 | 状态 |
|---|---|---|
| 无目标钻机真实SCADA | 增加硬来源门；本数据被拒绝为现场外部证据 | {closure['target_rig_scada']} |
| 0.7%—1.3%不能称显著SOTA | 对6个留出场景配对单元做精确随机化和区块自助审计，锁定描述性口径 | {closure['statistical_sota']} |
| 缺独立先进算法/工业优化器 | 锁定SciPy {scipy.__version__}内嵌HiGHS；作为应急负荷保护LP最优参考 | {closure['independent_solver_baseline']} |
| 极端起下钻仍约100 kWh失供 | 实现三级负荷优先保护与等比例削减对照；分级仍待现场HAZOP批准 | {closure['emergency_load_protection']} |
| 部分设备参数为工程假设 | 输出机理数据识别表并强制 `can_replace_target_rig_parameter=false` | {closure['target_equipment_parameters']} |
| 缺失重训、联合极端、半实物 | 完成诊断模型缺失重训和联合极端SIL；半实物仍未完成 | {closure['hardware_in_loop']} |

## 统计声明审计

- 配对场景单元：{statistics['paired_units']}，真正改善：{statistics['positive_units']}，并列：{statistics['ties']}。
- 平均改善：{statistics['mean_improvement_kwh']:.6f} kWh；中位数：{statistics['median_improvement_kwh']:.6f} kWh。
- 95%区块自助区间：[{statistics['bootstrap_mean_ci_kwh'][0]:.6f}, {statistics['bootstrap_mean_ci_kwh'][1]:.6f}] kWh。
- 双侧精确符号翻转 p 值：{statistics['exact_two_sided_sign_flip_pvalue']:.6f}。
- 允许结论：{statistics['allowed_claim']}。
- 禁止结论：{statistics['forbidden_claim']}。

## 联合极端和关键负荷保护

- 数据包原始供能边界软件回放失供 {base['source_load_shed_energy_kwh']:.3f} kWh；按 `pre_shed_demand_kw` 重算差值 {base['replay_minus_source_kwh']:.6f} kWh。
- 其中起下钻失供 {base_tripping_priority['total_unserved_energy_kwh']:.3f} kWh。等比例削减的关键负荷失供 {base_tripping_proportional['critical_unserved_energy_kwh']:.3f} kWh，优先保护后降至 {base_tripping_priority['critical_unserved_energy_kwh']:.3f} kWh。
- 若要求该起下钻回放中的关键负荷零失供，仍至少需要覆盖 {base_tripping_priority['critical_maximum_unserved_power_kw']:.3f} kW 峰值、{base_tripping_priority['critical_unserved_energy_kwh']:.3f} kWh 能量的快速备用；该值只是本机理轨迹下限，不是设备选型值。

- 等比例削减总失供：{proportional['total_unserved_energy_kwh']:.3f} kWh，其中关键负荷失供 {proportional['critical_unserved_energy_kwh']:.3f} kWh。
- 优先保护总失供：{priority['total_unserved_energy_kwh']:.3f} kWh，其中关键负荷失供 {priority['critical_unserved_energy_kwh']:.3f} kWh。
- 关键负荷失供减少：{joint['critical_unserved_reduction_vs_proportional_kwh']:.3f} kWh。
- 优先策略与HiGHS参考最大差：{priority['highs_equivalence_max_kw']:.3e} kW。
- 当前负荷分级状态：`{joint['priority_definition_status']}`；不得直接下装现场。

## 测量缺失重训诊断

| 条件 | 实际缺失比例 | MAE/kW | 峰值MAE/kW | 相对退化 | 诊断门 |
|---|---:|---:|---:|---:|---|
{missing_rows}

该诊断使用GPU多步XGBoost、缺失掩码和因果前向填补，并在每个缺失条件下独立重训；它仍不代表冻结V15主模型或目标现场缺失分布已经验收。

## 参数识别边界

共输出 {len(parameters)} 条机组/储能观测参数。全部标为 `SURROGATE_IDENTIFIED`，不得替代目标机组爬坡、最小开停、燃油、储能效率和退化参数。

目标参数证据缺口登记表共 {len(parameter_gaps)} 项，当前仍开放 {sum(item['status'] == 'OPEN' for item in parameter_gaps)} 项；必须用目标设备OEM资料、控制器设定、BMS/燃油表或调试试验关闭。

## 最终评审判断

本轮提升了工程可信度和失败处置完整性，但项目仍不能宣称现场验证、统计显著SOTA、现场关键负荷保护已批准或半实物闭环完成。下一硬门仍是目标钻机SCADA、设备参数签字表和HIL/现场闭环。
"""
    (output / "expert_remediation_report.md").write_text(report, encoding="utf-8")


def _evidence_manifest(output: Path, inputs: list[Path]) -> None:
    files = sorted(
        {path.resolve() for path in inputs}
        | {
            path.resolve()
            for path in output.iterdir()
            if path.is_file() and path.name != "evidence_manifest.json"
        }
    )
    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "status": "surrogate_engineering_preflight_not_acceptance",
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    _json_dump(output / "evidence_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行六项专家问题的V16前置整改审计")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/v16_expert_remediation_surrogate.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "artifacts/v16_expert_remediation_surrogate",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data_path = _resolve(config["input"]["data_path"])
    manifest_path = _resolve(config["input"]["manifest_path"])
    source_manifest = _verify_source_manifest(manifest_path)
    _json_dump(output / "source_dataset_manifest_audit.json", source_manifest)
    frame, data_audit = load_external_data_v16(data_path)
    _json_dump(output / "data_provenance_and_quality_audit.json", data_audit.as_dict())
    frame, selection = _select_analysis_eligible(frame)
    _json_dump(output / "analysis_eligibility_audit.json", selection)

    emergency = _run_emergency_audit(frame, config, output)
    _json_dump(output / "emergency_load_audit.json", emergency)
    parameters = _parameter_identification(frame, config, output)
    parameter_gaps = _parameter_gap_register(config, output)
    missingness = _missingness_retraining(frame, config, output)
    statistics = _statistical_claim_audit(config, output)
    summary = {
        "protocol": config["protocol"],
        "data_audit": data_audit.as_dict(),
        "source_dataset_manifest_audit": {
            key: value for key, value in source_manifest.items() if key != "records"
        },
        "analysis_eligibility": selection,
        "closure_boundary": config["closure_boundary"],
        "solver_lock": next(
            item["solver_metadata"]
            for item in emergency["joint_extreme"]["metrics"]
            if item["method"] == "HiGHS-priority-LP"
        ),
        "runtime_lock": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
        "statistical_claim_audit": statistics,
        "missingness_retraining": missingness,
        "parameter_evidence_gaps": parameter_gaps,
    }
    _json_dump(output / "final_summary.json", summary)
    _write_report(
        output,
        audit=data_audit.as_dict(),
        emergency=emergency,
        parameters=parameters,
        missingness=missingness,
        statistics=statistics,
        selection=selection,
        parameter_gaps=parameter_gaps,
        source_manifest=source_manifest,
        config=config,
    )
    _evidence_manifest(
        output,
        [
            args.config,
            Path(__file__),
            data_path,
            manifest_path,
            _resolve(config["input"]["equipment_path"]),
            _resolve(config["input"]["event_path"]),
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
