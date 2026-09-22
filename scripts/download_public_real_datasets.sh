#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
external_root="$project_root/data/public_external"

forge_root="$external_root/utah_forge_58_32"
mkdir -p "$forge_root"
curl -L --fail --retry 8 --retry-delay 3 --continue-at - \
  --output "$forge_root/Well_58-32_raw_pason_log.csv" \
  'https://gdr.openei.org/files/1113/Well_58-32_raw_pason_log.csv'
curl -L --fail --retry 8 --retry-delay 3 --continue-at - \
  --output "$forge_root/Well_58-32_processed_pason_log.csv" \
  'https://gdr.openei.org/files/1113/Well_58-32_processed_pason_log.csv'

three_w_root="$external_root/petrobras_3w"
if [[ ! -d "$three_w_root/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/petrobras/3W.git "$three_w_root"
fi

cd "$three_w_root"
patterns=(
  /README.md
  /3W_DATASET_STRUCTURE.md
  /CITATION.md
  /dataset/dataset.ini
  /dataset/README.md
)
for category in {0..9}; do
  patterns+=("/dataset/$category/")
  completed=0
  for attempt in {1..5}; do
    if git sparse-checkout set --no-cone "${patterns[@]}"; then
      completed=1
      break
    fi
    sleep 3
  done
  if [[ "$completed" -ne 1 ]]; then
    echo "Petrobras 3W category $category failed after five attempts" >&2
    exit 20
  fi
done

echo "Utah FORGE raw SHA-256: $(sha256sum "$forge_root/Well_58-32_raw_pason_log.csv" | cut -d' ' -f1)"
echo "Petrobras 3W commit: $(git rev-parse HEAD)"
echo "Petrobras 3W parquet files: $(find dataset -type f -name '*.parquet' | wc -l)"
echo "Equinor Volve is not downloaded by this script because the official route requires authenticated Databricks Marketplace access."
