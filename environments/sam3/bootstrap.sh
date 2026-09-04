#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${SAM3_ENV_DIR:-/workspace/venvs/sam3}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
SAM3_REV=660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
command -v git >/dev/null
command -v ffmpeg >/dev/null
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
"$PYTHON_BIN" -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade 'pip==25.1.1' 'setuptools==80.9.0' wheel
"$PY" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$ROOT/environments/sam3/requirements.txt"
"$PY" -m pip install --no-build-isolation "sam3 @ git+https://github.com/facebookresearch/sam3.git@$SAM3_REV"
"$PY" -m pip check
"$PY" -c 'from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor; from sam3.model.sam3_image_processor import Sam3Processor; print("SAM3 imports OK")'
"$PY" -m pip freeze > "$ENV_DIR/installed.freeze.txt"
printf 'Environment ready. Activate with: source %s/bin/activate\nRun the GPU smoke test before annotating.\n' "$ENV_DIR"
