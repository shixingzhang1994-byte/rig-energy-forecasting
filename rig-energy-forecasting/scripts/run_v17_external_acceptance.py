from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.validation.frozen_v15_external import (  # noqa: E402
    run_frozen_v15_external_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在用户指定目标钻机数据上执行冻结 V15 外部验收"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs" / "v17_user_designated_external_acceptance.yaml",
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    result = run_frozen_v15_external_acceptance(
        args.protocol, force_cpu=args.force_cpu
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
