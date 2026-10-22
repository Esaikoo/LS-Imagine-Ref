#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/evaluate_all.sh" --heatmap-mode zero --runs 1 --episodes 10 \
  --output evaluation_results/no_unet_5tasks_10episodes "$@"
