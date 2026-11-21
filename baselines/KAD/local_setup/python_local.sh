#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$BASELINE_ROOT/../.cache/huggingface"
export TORCH_HOME="$BASELINE_ROOT/../.cache/torch"
export XDG_CACHE_HOME="$BASELINE_ROOT/../.cache"
export TMPDIR="$BASELINE_ROOT/../.cache/tmp"
export PIP_CACHE_DIR="$BASELINE_ROOT/../.cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Common evaluator only: reduce NFS progress writes without changing numerics.
if [[ $# -gt 0 && "${1##*/}" == "evaluate_baseline.py" ]]; then
 if [[ "$(readlink -f -- "$1")" == "$(readlink -f -- "$BASELINE_ROOT/../local_setup/evaluate_baseline.py")" ]]; then
  shift
  set -- "$BASELINE_ROOT/../local_setup/evaluate_baseline_buffered.py" "$@"
 fi
fi
exec "$BASELINE_ROOT/.venv/bin/python" "$@"
