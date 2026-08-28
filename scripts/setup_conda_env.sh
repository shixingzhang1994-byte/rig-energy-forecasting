#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-/home/zsx/miniconda3/bin/conda}"
ENV_NAME="qz-rig-energy-ml"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" env create -f "$PROJECT_DIR/environment.yml"
else
  "$CONDA_BIN" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
fi

# RTX 50 系列使用 CUDA 12.8 构建；版本来自 PyTorch 官方发布矩阵。
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

"$CONDA_BIN" run -n "$ENV_NAME" python "$PROJECT_DIR/scripts/inspect_gpu.py"

