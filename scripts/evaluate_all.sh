#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export MINEDOJO_HEADLESS=1
# Use the requested environment even when launched from an unactivated terminal.
PYTHON_BIN="${LS_IMAGINE_PYTHON:-/home/user1/dl/conda_envs/ls_imagine/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
exec "$PYTHON_BIN" -u evaluate_checkpoints.py "$@"
