from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.adapters import CanonicalFileDataSource  # noqa: E402
from rig_energy.data.schema import validate_canonical_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="将现场CSV/XLSX映射为项目统一数据接口")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = CanonicalFileDataSource(args.input, args.mapping).load()
    data, report = validate_canonical_frame(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "schema": report.as_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

