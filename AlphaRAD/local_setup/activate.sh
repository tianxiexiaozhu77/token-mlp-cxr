#!/usr/bin/env bash
# Source this file to activate AlphaRAD's independent virtual environment.
export ALPHARAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ALPHARAD_ASSETS="$ALPHARAD_ROOT/local_assets"
export ALPHARAD_DATA_ROOT="$(dirname -- "$ALPHARAD_ROOT")/data/alpharad"
export HF_HOME="$ALPHARAD_ASSETS/.cache/huggingface"
export TORCH_HOME="$ALPHARAD_ASSETS/.cache/torch"
export XDG_CACHE_HOME="$ALPHARAD_ASSETS/.cache"
export MPLCONFIGDIR="$ALPHARAD_ASSETS/.cache/matplotlib"
export UV_CACHE_DIR="$ALPHARAD_ASSETS/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ALPHARAD_ASSETS/tools/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ALPHARAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$ALPHARAD_ROOT/.venv/bin/activate"
