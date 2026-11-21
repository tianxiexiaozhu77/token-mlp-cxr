#!/usr/bin/env bash
set -euo pipefail
RADZERO_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RADZERO_WORKSPACE="$(dirname -- "$RADZERO_REPO")"
RADZERO_PYTHON="$RADZERO_WORKSPACE/.envs/radzero/bin/python"
if [[ ! -x "$RADZERO_PYTHON" ]]; then
  echo "RadZero environment is missing. See $RADZERO_WORKSPACE/_downloads/radzero_env/ENVIRONMENT_NOTES.txt" >&2
  exit 1
fi
export PYTHONPATH="$RADZERO_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RADZERO_REPO/hf_cache"
export HF_MODULES_CACHE="$RADZERO_REPO/hf_cache/modules"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$RADZERO_REPO/hf_cache/matplotlib"
export XDG_CACHE_HOME="$RADZERO_REPO/hf_cache/xdg"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$RADZERO_PYTHON" "$@"
