from __future__ import annotations

"""Materialize the preregistered V27 heterogeneous parameter sets."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
FAMILY_PATH = PROJECT_DIR / "configs/v27_heterogeneous_family.yaml"
OUTPUT_ROOT = PROJECT_DIR / "artifacts/V27_heterogeneous_validation/parameter_sets"


def deep_merge(base: dict, override: dict) -> dict:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(unit: dict, family: dict, calibration_base: dict, dispatch_base: dict) -> dict:
    unit_root = OUTPUT_ROOT / f"{unit['unit_id']}_seed_{unit['seed']}"
    unit_root.mkdir(parents=True, exist_ok=True)

    calibration = deepcopy(calibration_base)
    calibration["metadata"]["version"] = "V27-heterogeneous-family"
    calibration["metadata"]["claim_boundary"] = family["family"]["evidence_boundary"]
    calibration["synthetic_overrides"]["simulation"]["states"] = deepcopy(
        family["transition_levels"][unit["transition"]]
    )
    load_level = family["load_levels"][unit["load"]]
    for state in ("idle", "circulation", "drilling", "connection", "tripping", "maintenance"):
        mean_kw = load_level[state]
        calibration["load_calibration"]["state_targets"][state][
            "target_mean_kw"
        ] = float(mean_kw)
    calibration["load_calibration"]["overall_acceptance_mean_kw"] = list(
        map(float, load_level["overall_acceptance_mean_kw"])
    )
    calibration["load_calibration"]["overall_acceptance_max_kw"] = list(
        map(float, load_level["overall_acceptance_max_kw"])
    )
    plant = family["plant_levels"][unit["plant"]]
    calibration["v27_risk_overrides"] = deepcopy(plant["risk_overrides"])
    calibration["v27_unit"] = deepcopy(unit)

    dispatch = deep_merge(dispatch_base, plant["dispatch_overrides"])
    dispatch["metadata"]["version"] = "V27-heterogeneous-family"
    dispatch["metadata"]["parameter_unit"] = unit["unit_id"]
    dispatch["metadata"]["factor_levels"] = {
        key: unit[key] for key in ("load", "transition", "plant")
    }

    calibration_path = unit_root / "calibration.yaml"
    dispatch_path = unit_root / "dispatch.yaml"
    calibration_path.write_text(
        yaml.safe_dump(calibration, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dispatch_path.write_text(
        yaml.safe_dump(dispatch, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    record = {
        **unit,
        "calibration_path": str(calibration_path.relative_to(PROJECT_DIR)),
        "calibration_sha256": sha256(calibration_path),
        "dispatch_path": str(dispatch_path.relative_to(PROJECT_DIR)),
        "dispatch_sha256": sha256(dispatch_path),
        "outcomes_used": False,
    }
    (unit_root / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    family = yaml.safe_load(FAMILY_PATH.read_text(encoding="utf-8"))
    base_calibration_path = PROJECT_DIR / family["family"]["base_calibration"]
    base_dispatch_path = PROJECT_DIR / family["family"]["base_dispatch"]
    calibration = yaml.safe_load(base_calibration_path.read_text(encoding="utf-8"))
    dispatch = yaml.safe_load(base_dispatch_path.read_text(encoding="utf-8"))
    units = [family["development_unit"], *family["prospective_units"]]
    records = [materialize(unit, family, calibration, dispatch) for unit in units]
    manifest = {
        "family_id": family["family"]["id"],
        "family_sha256": sha256(FAMILY_PATH),
        "unit_count": len(records),
        "development_unit_count": 1,
        "prospective_unit_count": 12,
        "outcomes_used": False,
        "records": records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
