#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$BASELINE_ROOT/../.cache/huggingface" XDG_CACHE_HOME="$BASELINE_ROOT/../.cache" TMPDIR="$BASELINE_ROOT/../.cache/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$BASELINE_ROOT/local_setup:$BASELINE_ROOT/official${PYTHONPATH:+:$PYTHONPATH}"
exec "$BASELINE_ROOT/.venv/bin/python" "$@"
