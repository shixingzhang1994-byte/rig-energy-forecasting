from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.public_electrical import (
    prepare_m5bat_one_minute,
    prepare_opencem_one_minute,
    prepare_refit_house_one_minute,
    prepare_tsukuba_microgrid_one_minute,
    prepare_uci_household_one_minute,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe one-minute public electrical measurements for V28."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_DIR / "data/public_external/opencem/data/measurements",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/v28_public_real_validation/prepared",
    )
    parser.add_argument(
        "--refit-house4",
        type=Path,
        default=PROJECT_DIR / "data/public_external/refit/CLEAN_House4.csv",
    )
    parser.add_argument("--skip-opencem", action="store_true")
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--skip-uci", action="store_true")
    parser.add_argument(
        "--all-refit",
        action="store_true",
        help="Prepare every completed CLEAN_House*.csv in the REFIT directory.",
    )
    parser.add_argument(
        "--uci-household",
        type=Path,
        default=PROJECT_DIR
        / "data/public_external/uci_household_power/household_power_consumption.txt",
    )
    parser.add_argument(
        "--tsukuba-cleaned-zip",
        type=Path,
        default=PROJECT_DIR
        / "data/public_external/tsukuba_microgrid/figshare_v1/extracted/2 Cleaned_data_per_second.zip",
    )
    parser.add_argument(
        "--tsukuba-parent-archive",
        type=Path,
        default=PROJECT_DIR
        / "data/public_external/tsukuba_microgrid/figshare_v1/294ecef46cba01f6392560efe6c78a28.zip",
    )
    parser.add_argument("--prepare-tsukuba", action="store_true")
    parser.add_argument(
        "--m5bat-extracted-dir",
        type=Path,
        default=PROJECT_DIR / "data/public_external/m5bat/full/extracted",
    )
    parser.add_argument(
        "--m5bat-archive",
        type=Path,
        default=PROJECT_DIR
        / "data/public_external/m5bat/full/Full_Dataset_M5BAT_Battery_Unit_Pb1.zip",
    )
    parser.add_argument("--prepare-m5bat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_opencem:
        paths = sorted(args.input_dir.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No measurement CSV files found in {args.input_dir}")
        frame, metadata = prepare_opencem_one_minute(paths)
        parquet_path = args.output_dir / "opencem_1min.parquet"
        metadata_path = args.output_dir / "opencem_preparation_audit.json"
        frame.to_parquet(parquet_path, index=False)
        metadata["prepared_path"] = str(parquet_path.resolve())
        metadata["prepared_size_bytes"] = parquet_path.stat().st_size
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    refit_paths = (
        sorted(args.refit_house4.parent.glob("CLEAN_House*.csv"))
        if args.all_refit
        else [args.refit_house4]
    )
    if not args.skip_refit:
        for refit_source in refit_paths:
            if not refit_source.is_file() or refit_source.stat().st_size == 0:
                continue
            house_token = refit_source.stem.removeprefix("CLEAN_")
            output_token = house_token.lower()
            refit, refit_metadata = prepare_refit_house_one_minute(
                refit_source, house_id=f"REFIT_{house_token}"
            )
            refit_path = args.output_dir / f"refit_{output_token}_1min.parquet"
            refit_metadata_path = args.output_dir / f"refit_{output_token}_preparation_audit.json"
            refit.to_parquet(refit_path, index=False)
            refit_metadata["prepared_path"] = str(refit_path.resolve())
            refit_metadata["prepared_size_bytes"] = refit_path.stat().st_size
            refit_metadata_path.write_text(
                json.dumps(refit_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(refit_metadata, ensure_ascii=False, indent=2))
    if (
        not args.skip_uci
        and args.uci_household.is_file()
        and args.uci_household.stat().st_size > 0
    ):
        uci, uci_metadata = prepare_uci_household_one_minute(args.uci_household)
        uci_path = args.output_dir / "uci_household_1min.parquet"
        uci_metadata_path = args.output_dir / "uci_household_preparation_audit.json"
        uci.to_parquet(uci_path, index=False)
        uci_metadata["prepared_path"] = str(uci_path.resolve())
        uci_metadata["prepared_size_bytes"] = uci_path.stat().st_size
        uci_metadata_path.write_text(
            json.dumps(uci_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(uci_metadata, ensure_ascii=False, indent=2))
    if args.prepare_tsukuba:
        tsukuba, tsukuba_metadata = prepare_tsukuba_microgrid_one_minute(
            args.tsukuba_cleaned_zip,
            parent_archive=args.tsukuba_parent_archive,
        )
        tsukuba_path = args.output_dir / "tsukuba_microgrid_1min.parquet"
        tsukuba_metadata_path = args.output_dir / "tsukuba_microgrid_preparation_audit.json"
        tsukuba.to_parquet(tsukuba_path, index=False)
        tsukuba_metadata["prepared_path"] = str(tsukuba_path.resolve())
        tsukuba_metadata["prepared_size_bytes"] = tsukuba_path.stat().st_size
        tsukuba_metadata_path.write_text(
            json.dumps(tsukuba_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(tsukuba_metadata, ensure_ascii=False, indent=2))
    if args.prepare_m5bat:
        m5bat, m5bat_metadata = prepare_m5bat_one_minute(
            args.m5bat_extracted_dir,
            source_archive=args.m5bat_archive,
        )
        m5bat_path = args.output_dir / "m5bat_exide1_1min.parquet"
        m5bat_metadata_path = args.output_dir / "m5bat_exide1_preparation_audit.json"
        m5bat.to_parquet(m5bat_path, index=False)
        m5bat_metadata["prepared_path"] = str(m5bat_path.resolve())
        m5bat_metadata["prepared_size_bytes"] = m5bat_path.stat().st_size
        m5bat_metadata_path.write_text(
            json.dumps(m5bat_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(m5bat_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
