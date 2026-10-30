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
MODE="${A_MODE:-vnext}"

case "$TASK_NAME" in
  harvest_log_in_plains) TARGET_ITEM=log ;;
  harvest_sand) TARGET_ITEM=sand ;;
  harvest_water_with_bucket) TARGET_ITEM=water_bucket ;;
  mine_iron_ore) TARGET_ITEM=iron_ore ;;
  shear_sheep) TARGET_ITEM=wool ;;
  *) echo "Unknown task: $TASK_NAME" >&2; exit 2 ;;
esac

# Default vnext is the task-preserving revision; abc/full preserve historical runs.
case "$MODE" in
  vnext) set -- --configs minedojo apm_vnext "$@" ;;
  vnext_ac) set -- --configs minedojo apm_vnext --vnext_memory_enabled False "$@" ;;
  vnext_no_search) set -- --configs minedojo apm_vnext --vnext_control False "$@" ;;
  vnext_no_interaction) set -- --configs minedojo apm_vnext --vnext_interaction False "$@" ;;
  full) set -- --configs minedojo apm_a "$@" ;;
  off) set -- --configs minedojo apm_a_off "$@" ;;
  loss_only) set -- --configs minedojo apm_a apm_a_loss_only "$@" ;;
  ab) set -- --configs minedojo apm_a apm_ab_v4 "$@" ;;
  abc) set -- --configs minedojo apm_a apm_full "$@" ;;
  abc_v1) set -- --configs minedojo apm_a apm_abc "$@" ;;
  a5) set -- --configs minedojo apm_a apm_full --search_b_enabled False --search_c_enabled False --search_branch_enabled False "$@" ;;
  ab5) set -- --configs minedojo apm_a apm_full --search_c_enabled False --search_branch_enabled False "$@" ;;
  ac5) set -- --configs minedojo apm_a apm_full --search_b_enabled False "$@" ;;
  bc2) set -- --configs minedojo apm_a apm_full --search_a_enabled False --search_branch_enabled False "$@" ;;
  b3) set -- --configs minedojo apm_a apm_full --search_a_enabled False --search_c_enabled False --search_branch_enabled False "$@" ;;
  ac) set -- --configs minedojo apm_a apm_ac "$@" ;;
  bc) set -- --configs minedojo apm_a apm_bc "$@" ;;
  b) set -- --configs minedojo apm_b "$@" ;;
  *) echo "A_MODE must be vnext, vnext_ac, vnext_no_search, vnext_no_interaction, abc, a5, ab5, ac5, bc2, b3, full, off, loss_only, ab, abc_v1, ac, bc, or b" >&2; exit 2 ;;
esac

RUN_LOG_ROOT="$LOG_ROOT/$PROJECT_NAME/$MODE"

# Opt in with NTFY_TOPIC. Keep notification addresses out of tracked files.
NTFY_TOPIC="${NTFY_TOPIC:-}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
NTFY_SERVER="${NTFY_SERVER%/}"

notify_training_exit() {
  TRAIN_EXIT_STATUS="$1"
  if [ -n "${TRAIN_SIGNAL_EXIT:-}" ]; then
    TRAIN_EXIT_STATUS=$TRAIN_SIGNAL_EXIT
  fi
  trap - 0 HUP INT TERM
  set +e
  if [ "$TRAIN_EXIT_STATUS" -eq 0 ]; then
    TRAIN_NOTIFY_TITLE="Training completed: $TASK_NAME"
    TRAIN_NOTIFY_RESULT="训练正常结束"
  else
    TRAIN_NOTIFY_TITLE="Training stopped: $TASK_NAME"
    TRAIN_NOTIFY_RESULT="训练异常退出或被中断"
  fi
  TRAIN_END_TIME=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')
  TRAIN_NOTIFY_CONTENT=$(printf '%s\n' \
    "$TRAIN_NOTIFY_RESULT" \
    "项目：$PROJECT_NAME" \
    "任务：$TASK_NAME；模式：$MODE" \
    "退出码：$TRAIN_EXIT_STATUS" \
    "结束时间：$TRAIN_END_TIME" \
    "训练数据：$RUN_LOG_ROOT" \
    "输出日志：${TRAIN_LOG_FILE:-见本次 nohup 启动日志}" \
    "请到算力平台控制台手动关闭实例，确认实例已经关机。")
  if curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 15 --retry 2 --retry-delay 3 \
      --output /dev/null \
      -H "Title: $TRAIN_NOTIFY_TITLE" -H "Priority: high" \
      --data-binary "$TRAIN_NOTIFY_CONTENT" "$NTFY_SERVER/$NTFY_TOPIC"; then
    echo "[ntfy] End notification accepted; training exit=$TRAIN_EXIT_STATUS."
  else
    echo "[ntfy] WARNING: notification request failed; training exit=$TRAIN_EXIT_STATUS." >&2
  fi
  exit "$TRAIN_EXIT_STATUS"
}

forward_training_signal() {
  TRAIN_SIGNAL=$1
  TRAIN_SIGNAL_EXIT=$2
  trap '' HUP INT TERM
  if [ -n "${TRAIN_CHILD_PID:-}" ]; then
    TRAIN_FORWARD_SIGNAL=$TRAIN_SIGNAL
    # Async commands can inherit ignored INT; TERM reliably forwards the stop request.
    if [ "$TRAIN_SIGNAL" = INT ]; then
      TRAIN_FORWARD_SIGNAL=TERM
    fi
    kill -"$TRAIN_FORWARD_SIGNAL" "$TRAIN_CHILD_PID" 2>/dev/null || :
    wait "$TRAIN_CHILD_PID" 2>/dev/null || :
  fi
  notify_training_exit "$TRAIN_SIGNAL_EXIT"
}

if [ -n "$NTFY_TOPIC" ]; then
  case "$NTFY_TOPIC" in
    *[!A-Za-z0-9_-]*) echo "NTFY_TOPIC must contain only letters, digits, _ or -." >&2; exit 2 ;;
  esac
  case "$NTFY_SERVER" in
    http://*|https://*) ;;
    *) echo "NTFY_SERVER must be an HTTP(S) address." >&2; exit 2 ;;
  esac
  command -v curl >/dev/null 2>&1 || {
    echo "ntfy enabled but curl is unavailable; install curl before starting training." >&2
    exit 2
  }
  trap 'notify_training_exit "$?"' 0
  trap 'forward_training_signal HUP 129' HUP
  trap 'forward_training_signal INT 130' INT
  trap 'forward_training_signal TERM 143' TERM
  echo "[ntfy] End notifications enabled (normal exit and errors)."
fi

mkdir -p "$RUN_LOG_ROOT"
cd "$PROJECT_ROOT"

echo "[train] task=$TASK_NAME A_MODE=$MODE logdir=$RUN_LOG_ROOT"
set -- python expr.py \
    --task "minedojo_$TASK_NAME" \
    --target_item "$TARGET_ITEM" \
    --steps 1000000 \
    --log_every 1000 \
    --logdir "$RUN_LOG_ROOT" \
    "$@"


if [ -z "$NTFY_TOPIC" ]; then
  exec "$@"
fi

# Keep this shell alive so its exit trap can notify after Python exits.
"$@" &
TRAIN_CHILD_PID=$!
echo "[train] python_pid=$TRAIN_CHILD_PID"
wait "$TRAIN_CHILD_PID"
