from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rig_energy.data.witsml_external import audit_witsml_archive  # noqa: E402


DEFAULT_SOURCE_URL = (
    "https://energistics.org/sites/default/files/2023-03/"
    "NA-NA-EnergisticsWell2016-B.zip"
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "|".join(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official Energistics WITSML ZIP")
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    args = parser.parse_args()

    audit, logs, curves = audit_witsml_archive(
        args.archive,
        source_url=args.source_url,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "archive_audit.json").write_text(
        json.dumps(audit.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "log_inventory.csv", [row.as_dict() for row in logs])
    _write_csv(
        args.output_dir / "channel_inventory.csv",
        [row.as_dict() for row in curves],
    )
    print(json.dumps(audit.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
