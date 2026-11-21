#!/usr/bin/env bash
set -euo pipefail
alpharad_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$alpharad_script_dir/activate.sh"
exec "$ALPHARAD_ROOT/.venv/bin/python" "$@"
