#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./scripts/train.sh <task_name> [extra_args...]"
  exit 1
fi

TASK_NAME=$1
shift

export MINEDOJO_HEADLESS=1

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_NAME=$(basename "$PROJECT_ROOT")

LOG_ROOT=/root/rivermind-data/mine/projects/tb_logs

mkdir -p "$LOG_ROOT/$PROJECT_NAME"

cd "$PROJECT_ROOT"

python expr.py \
    --configs minedojo \
    --task minedojo_${TASK_NAME} \
    --logdir "$LOG_ROOT/$PROJECT_NAME" \
    "$@"