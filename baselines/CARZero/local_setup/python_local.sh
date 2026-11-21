#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$root/../.cache/huggingface"
export TORCH_HOME="$root/../.cache/torch"
export TMPDIR="$root/../.cache/tmp"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
if [[ "${1:-}" == "$(cd "$root/.." && pwd)/local_setup/evaluate_baseline.py" ]]; then
 shift
 set -- "$root/../local_setup/evaluate_baseline_buffered.py" "$@"
fi
exec "$root/.venv/bin/python" "$@"
