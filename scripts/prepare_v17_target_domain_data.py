from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.external_adapter_v16 import load_external_data_v16  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 V17 目标域适配生成项目统一 Parquet，不改写来源证据"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/home/zsx/桌面/QZ/data/V15_rig_industrial_dataset_v2_download/"
            "rig_industrial_dataset_v2/10_canonical_observed_5s.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "artifacts/v17_target_domain_adaptation/data/canonical_72h.parquet",
    )
    args = parser.parse_args()
    expected = "2d58aef196e19c7ba59ec60a13cd2857c49870dce7bb26895736f92f85b605b0"
    actual = _sha256(args.input)
    if actual != expected:
        raise ValueError(f"来源数据哈希不匹配: {actual}")
    frame, audit = load_external_data_v16(
        args.input, require_target_rig_field_data=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    record = {
        **audit.as_dict(),
        "acceptance_usage_class": "USER_DESIGNATED_TARGET_RIG_ACCEPTANCE_DATA",
        "original_provenance_preserved": True,
        "field_origin_claim_allowed": False,
        "source_sha256": actual,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "target_interpolation_performed_by_this_step": False,
    }
    (args.output.parent / "preparation_audit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
