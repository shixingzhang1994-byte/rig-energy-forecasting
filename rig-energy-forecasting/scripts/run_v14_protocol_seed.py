from __future__ import annotations

import json
import subprocess as real_subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_v10_protocol_seed as protocol_runner  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_DIR / "configs/v14_public_calibration_acceptance.yaml"
CALIBRATION = PROJECT_DIR / "configs/v14_public_evidence_calibration.yaml"
_ORIGINAL_BUILD_DISPATCH = protocol_runner._build_dispatch


def _prepare_inputs(
    seed: int, days: int, root: Path, reuse: bool
) -> tuple[Path, Path, Path]:
    v5 = yaml.safe_load(
        (PROJECT_DIR / "configs/v5_acceptance.yaml").read_text(encoding="utf-8")
    )
    v6 = yaml.safe_load(
        (PROJECT_DIR / "configs/v6_dispatch_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    calibration = yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))
    base = yaml.safe_load(
        (PROJECT_DIR / "configs/synthetic_default.yaml").read_text(encoding="utf-8")
    )
    forecast = protocol_runner._deep_merge(base, calibration["synthetic_overrides"])
    forecast = protocol_runner._deep_merge(forecast, v5["forecast_overrides"])
    forecast["project"]["seed"] = int(seed)
    forecast["project"]["days"] = int(days)
    forecast["dispatch_uncertainty"] = {
        "enabled": True,
        "coverage_candidates": v6["protocol"]["coverage_candidates"],
    }
    forecast_config = root / "source/resolved_forecast_config.yaml"
    protocol_runner._write_yaml(forecast, forecast_config)

    risk = protocol_runner._deep_merge(
        yaml.safe_load(
            (PROJECT_DIR / "configs/risk_default.yaml").read_text(encoding="utf-8")
        ),
        v5["risk_overrides"],
    )
    risk["project"]["seed"] = int(seed)
    risk["split"]["train_fraction"] = float(
        v6["protocol"]["risk_split"]["train_fraction"]
    )
    risk["split"]["val_fraction"] = float(
        v6["protocol"]["risk_split"]["val_fraction"]
    )
    risk_config = root / "source/resolved_risk_config.yaml"
    protocol_runner._write_yaml(risk, risk_config)

    data_path = PROJECT_DIR / f"data/processed/rig_load_v14_{seed}.parquet"
    data_artifacts = root / "source/data"
    forecast_dir = root / "source/forecast"
    predictions = forecast_dir / "forecast_predictions.npz"
    keys = forecast_dir / "forecast_prediction_keys.json"
    if not (reuse and data_path.exists()):
        protocol_runner._run(
            "scripts/generate_v14_calibrated.py",
            "--calibration",
            str(CALIBRATION),
            "--seed",
            str(seed),
            "--days",
            str(days),
            "--output",
            str(data_path),
            "--artifact-dir",
            str(data_artifacts),
        )
        protocol_runner._run(
            "scripts/audit_v14_calibration.py",
            "--data",
            str(data_path),
            "--calibration",
            str(CALIBRATION),
            "--artifact-dir",
            str(data_artifacts / "audit"),
        )
    if not (reuse and predictions.exists() and keys.exists()):
        protocol_runner._run(
            "scripts/run_benchmark.py",
            "--data",
            str(data_path),
            "--config",
            str(forecast_config),
            "--artifact-dir",
            str(forecast_dir),
        )

    risk_dir = root / "risk"
    risk_signals = risk_dir / "risk_predictions.csv"
    risk_metadata = risk_dir / "run_metadata.json"
    if not (reuse and risk_signals.exists() and risk_metadata.exists()):
        protocol_runner._run(
            "scripts/run_risk_benchmark.py",
            "--predictions",
            str(predictions),
            "--prediction-keys",
            str(keys),
            "--data",
            str(data_path),
            "--forecast-config",
            str(forecast_config),
            "--risk-config",
            str(risk_config),
            "--artifact-dir",
            str(risk_dir),
        )
    return forecast_dir, risk_signals, risk_metadata


def _build_dispatch(full_protocol: dict) -> dict:
    dispatch = _ORIGINAL_BUILD_DISPATCH(full_protocol)
    dispatch = protocol_runner._deep_merge(
        dispatch, full_protocol["plant_and_cost_overrides"]
    )
    # The legacy builder applies fixed unit fields before plant overrides.
    # Reassert them so the aggregate and per-unit ratings cannot diverge.
    fixed = full_protocol["fixed_unit_commitment"]
    generator = dispatch["plant"]["generator"]
    generator.update(
        {
            "unit_count": int(fixed["unit_count"]),
            "unit_rated_power_kw": float(fixed["unit_rated_power_kw"]),
            "unit_minimum_stable_power_kw": float(
                fixed["unit_minimum_stable_power_kw"]
            ),
            "unit_ramp_kw_per_step": float(fixed["unit_ramp_kw_per_step"]),
            "minimum_up_steps": int(fixed["minimum_up_steps"]),
            "minimum_down_steps": int(fixed["minimum_down_steps"]),
        }
    )
    dispatch["cost"]["generator_startup_cost_yuan_per_unit"] = float(
        fixed["startup_cost_yuan_per_unit"]
    )
    dispatch["fuel_model"] = {
        "name": "CAT-XQP300-50Hz-prime",
        "load_fraction": [0.0, 0.5, 0.75, 1.0],
        "liters_per_hour": [0.0, 33.8, 47.3, 62.5],
        "source": "M_CAT_XQP300",
    }
    return dispatch


def _dispatch_aware_run(command, *args, **kwargs):
    command = list(command)
    try:
        index = command.index("scripts/run_dispatch_benchmark.py")
    except ValueError:
        pass
    else:
        command[index] = "scripts/run_v14_dispatch_benchmark.py"
    return real_subprocess.run(command, *args, **kwargs)


def main() -> None:
    protocol_runner.DEFAULT_PROTOCOL_PATH = DEFAULT_PROTOCOL
    protocol_runner._prepare_inputs = _prepare_inputs
    protocol_runner._build_dispatch = _build_dispatch
    protocol_runner.subprocess = SimpleNamespace(run=_dispatch_aware_run)
    protocol_runner.main()

    # Make the calibrated fuel runner explicit in the evidence result instead
    # of leaving the inherited V10 command string ambiguous.
    protocol = yaml.safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))["protocol"]
    seed = int(sys.argv[sys.argv.index("--seed") + 1])
    result_path = (
        PROJECT_DIR
        / "artifacts"
        / protocol["artifact_namespace"]
        / f"seed_{seed}"
        / "result.json"
    )
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["reproduction_command"] = result["reproduction_command"].replace(
            "scripts/run_dispatch_benchmark.py",
            "scripts/run_v14_dispatch_benchmark.py",
        )
        result["calibration"] = "configs/v14_public_evidence_calibration.yaml"
        result["field_scada_claimed"] = False
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_path = result_path.parent / "evidence_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            relative_result = str(result_path.relative_to(PROJECT_DIR))
            for item in manifest.get("files", []):
                if item.get("path") == relative_result:
                    item["sha256"] = protocol_runner._sha256(result_path)
                    item["size_bytes"] = result_path.stat().st_size
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
