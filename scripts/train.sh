#!/bin/bash
# POSIX-compatible so the usual "sh ./scripts/train.sh ..." also works.
set -eu

if [ "$#" -eq 0 ]; then
  echo "Usage: sh ./scripts/train.sh <task_name> [extra_args...]"
  exit 1
fi

TASK_NAME=$1
shift

export MINEDOJO_HEADLESS=1
export PYTHONUNBUFFERED=1

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_NAME=$(basename "$PROJECT_ROOT")
LOG_ROOT="${LOG_ROOT:-/root/rivermind-data/mine/projects/tb_logs}"
MODE="${A_MODE:-full}"

case "$TASK_NAME" in
  harvest_log_in_plains) TARGET_ITEM=log ;;
  harvest_sand) TARGET_ITEM=sand ;;
  harvest_water_with_bucket) TARGET_ITEM=water_bucket ;;
  mine_iron_ore) TARGET_ITEM=iron_ore ;;
  shear_sheep) TARGET_ITEM=wool ;;
  *) echo "Unknown task: $TASK_NAME" >&2; exit 2 ;;
esac

# full selects repaired A v3 (B disabled); explicit extra arguments remain last.
case "$MODE" in
  full) set -- --configs minedojo apm_a "$@" ;;
  off) set -- --configs minedojo apm_a_off "$@" ;;
  loss_only) set -- --configs minedojo apm_a apm_a_loss_only "$@" ;;
  ab) set -- --configs minedojo apm_ab "$@" ;;
  b) set -- --configs minedojo apm_b "$@" ;;
  *) echo "A_MODE must be full, off, loss_only, ab, or b" >&2; exit 2 ;;
esac

RUN_LOG_ROOT="$LOG_ROOT/$PROJECT_NAME/$MODE"
mkdir -p "$RUN_LOG_ROOT"
cd "$PROJECT_ROOT"

echo "[train] task=$TASK_NAME A_MODE=$MODE logdir=$RUN_LOG_ROOT"
exec python expr.py \
    --task "minedojo_$TASK_NAME" \
    --target_item "$TARGET_ITEM" \
    --steps 1000000 \
    --log_every 1000 \
    --logdir "$RUN_LOG_ROOT" \
    "$@"
