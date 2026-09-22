#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zsx/miniconda3/envs/qz-rig-energy-ml/bin/python"
LOG_ROOT="${PROJECT_DIR}/artifacts/V27_heterogeneous_validation/confirmatory/queue_logs"
mkdir -p "${LOG_ROOT}"

for seed in 20261302 20261303 20261304 20261305 20261306 20261307 20261308 20261309 20261310 20261311; do
  log="${LOG_ROOT}/seed_${seed}.log"
  printf 'START seed=%s\n' "${seed}" | tee "${LOG_ROOT}/seed_${seed}.status"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_v27_protocol_seed.py" --seed "${seed}" >"${log}" 2>&1
  rc=$?
  printf 'DONE seed=%s exit_code=%s\n' "${seed}" "${rc}" | tee -a "${LOG_ROOT}/seed_${seed}.status"
done
