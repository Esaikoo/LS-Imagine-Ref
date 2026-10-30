#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--minedojo-recovery" ]]; then
  shift
  export MINEDOJO_HEADLESS=1 PYTHONUNBUFFERED=1
  exec python tests/test_env_recovery.py --minedojo "$@"
fi
if [[ "${1:-}" == "--startup-replay" ]]; then
  shift
  exec python tests/test_search_a.py --startup-replay "$@"
fi
if [[ "${1:-}" == "--branch-checkpoint" ]]; then
  shift
  export MINEDOJO_HEADLESS=1 PYTHONUNBUFFERED=1
  exec python tests/test_search_a.py --branch-checkpoint "$@"
fi
if [[ "${1:-}" == "--audit-replay" ]]; then
  shift
  exec python tests/test_search_a.py --audit-replay "$@"
fi
python tests/test_search_a.py
python tests/test_env_recovery.py
python tests/test_training_state.py
python tests/test_search_b.py
python tests/test_search_vnext_evidence.py
python tests/test_search_vnext_models.py
python tests/test_search_vnext_control.py
python tests/test_search_vnext_integration.py
python -m py_compile search_actuator.py search_vnext.py search_vnext_control.py training_state.py envs/recovery.py networks.py search_a.py search_c.py search_feedback.py search_branches.py search_memory.py search_evidence.py expr.py models.py tools.py test.py evaluate_checkpoints.py \
  envs/tasks/minedojo/__init__.py envs/tasks/minedojo/wrappers.py envs/tasks/base/ls_imagine_wrapper.py
bash -n scripts/test_a.sh
sh -n scripts/train.sh
echo '[APM] CPU tests and syntax: PASS'
if [[ "${1:-}" == "--minedojo" || "${1:-}" == "--minedojo-off" || "${1:-}" == "--minedojo-abc" || "${1:-}" == "--minedojo-full" || "${1:-}" == "--minedojo-vnext" ]]; then
  MODE=full
  EXTRA=()
  PREFILL=32
  CONFIGS=(minedojo apm_a apm_a_smoke)
  if [[ "$1" == "--minedojo-abc" ]]; then
    MODE=abc
    CONFIGS=(minedojo apm_a apm_abc apm_a_smoke)
    EXTRA=(--search_b_slots 4 --search_b_dim 16 --search_b_hidden 32 --search_b_lookback 12)
  fi
  if [[ "$1" == "--minedojo-full" ]]; then
    MODE=complete
    CONFIGS=(minedojo apm_a apm_full apm_a_smoke)
    EXTRA=(--search_b_slots 4 --search_b_dim 16 --search_b_hidden 32 --search_b_lookback 12
      --search_branch_every 64 --search_branch_count 2 --search_branch_prefix 2
      --eval_every 32)
  fi
  if [[ "$1" == "--minedojo-vnext" ]]; then
    MODE=vnext
    PREFILL=40
    CONFIGS=(minedojo apm_vnext apm_a_smoke)
    EXTRA=(--vnext_slots 4 --vnext_memory_dim 16 --vnext_hidden 64
      --vnext_future_dim 16 --vnext_start_step 32 --vnext_warmup 64
      --vnext_full_step 96 --vnext_base_episode_fraction 0
      --vnext_probe_probability 1 --eval_every 32)
  fi
  if [[ "$1" == "--minedojo-off" ]]; then
    MODE=off
    EXTRA=(--search_a_enabled False)
  fi
  # Fresh run root prevents stale checkpoints from satisfying verification.
  RUN_ROOT="./logs/test_apm_a_v4/${MODE}_$(date +%Y%m%dT%H%M%S)_$$"
  export MINEDOJO_HEADLESS=1 PYTHONUNBUFFERED=1
  echo "[APM] MineDojo $MODE: 96 train steps including $PREFILL prefill; short eval."
  python expr.py \
    --configs "${CONFIGS[@]}" \
    --task minedojo_harvest_sand --target_item sand \
    --steps 96 --prefill "$PREFILL" --eval_every 96 --eval_episode_num 1 \
    --episode_max_steps_override 32 --log_every 16 --train_ratio 4 \
    --search_a_start_step 32 --search_a_override_prob 1.0 --search_a_epsilon 1.0 \
    --search_a_min_labels 1 --search_a_min_positives 0 \
    --search_a_low_evidence 1.1 --compile False --video_pred_log False \
    --log_all_metrics True --env_seed 0 --logdir "$RUN_ROOT" "${EXTRA[@]}"
  python - "$RUN_ROOT" "$MODE" <<'CHECK'
import json
import pathlib
import sys
import numpy as np
import torch
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
paths = list(root.glob('**/latest.pt'))
assert len(paths) == 1, paths
latest = paths[0]
checkpoint = torch.load(latest, map_location='cpu')
assert checkpoint['step'] == 96
assert all(torch.isfinite(v).all() for v in checkpoint['agent_state_dict'].values())
rows = [json.loads(line) for line in (latest.parent/'metrics.jsonl').read_text().splitlines()]
assert all(np.isfinite(value) for row in rows for value in row.values())
episodes = list(latest.parent.glob('*_eps/*.npz'))
assert episodes
override = 0
known = 0
for episode in episodes:
    with np.load(episode) as data:
        assert len({len(data[k]) for k in data.files}) == 1, episode
        if (episode.parent.name == 'eval_eps' and mode not in ('complete','vnext')) or mode == 'off':
            assert all(not data[k].any() for k in data.files if k.startswith('search_') and k.endswith('_valid'))
        else:
            known += data['search_visible_valid'].sum()
        if episode.parent.name == 'train_eps' and 'log_search_override' in data.files:
            override += data['log_search_override'].sum()
if mode == 'vnext':
    from search_vnext import check_vnext_checkpoint
    check_vnext_checkpoint(checkpoint, True)
    assert checkpoint['search_vnext_version'] == 1
    assert checkpoint['search_a_label_version'] == 3
    assert not any('_wm.search_a.' in k or '_wm.search_b.' in k for k in checkpoint['agent_state_dict'])
    for key in ('A/prediction_loss','B/total_loss','C/transitions'):
        assert any(key in row for row in rows), key
    assert any(row.get('C/updates',0)>0 for row in rows)
    for path in episodes:
        with np.load(path) as d:
            assert len({len(d[k]) for k in d.files}) == 1
            if 'log_vnext_applied' in d:
                active=d['log_vnext_version']==1
                np.testing.assert_allclose(d['log_vnext_applied'][active],d['log_applied_control'][active])
            if 'log_vnext_memory_refs' in d:
                for t,refs in enumerate(d['log_vnext_memory_refs']):
                    if not d['is_first'][t]:
                        assert np.all(refs[refs>=0]<t), (path,t,refs)
elif mode in ('full', 'abc', 'complete'):
    assert checkpoint['search_a_version'] == (5 if mode == 'complete' else 4)
    for key in ('A/future_loss','A/prediction_loss','A/new_loss','A/discovery_loss','A/revisit_loss','A/current_visible_loss','A/future_visible_loss','A/shared_grad_scale'):
        assert any(key in row for row in rows), key
    assert any('_a_opt' in key for key in checkpoint['optims_state_dict'])
    assert checkpoint['search_a_control_version'] == 4
    assert checkpoint['search_a_label_version'] == 2
    assert known > 0, known
    if mode == 'full':
        assert override > 0, override
    if mode in ('abc', 'complete'):
        assert checkpoint['search_c_version'] == (2 if mode == 'complete' else 1)
        assert any(row.get('C/transitions', 0) > 0 for row in rows)
        assert any(row.get('C/updates', 0) > 0 for row in rows)
        assert any('B/total_loss' in row for row in rows)
    if mode == 'complete':
        assert checkpoint['search_b_version'] == 3
        assert any(row.get('C/imagined_transitions', 0) > 0 for row in rows)
        assert any('C/task_actor_search_bonus' in row for row in rows)
        assert any(row.get('C/branch_real_steps', 0) > 0 for row in rows)
        assert any(row.get('C/branch_verified_options', 0) >= 2 for row in rows)
        assert any(row.get('eval_observed_surface_cells', 0) > 0 for row in rows)
    assert any(row.get('A/new_valid',0) > 0 for row in rows)
    assert any(row.get('A/discovery_valid',0) > 0 for row in rows)
    assert any(row.get('eval_search_decision_rate',0) > 0 for row in rows)
else:
    assert not any('.search_a.' in key for key in checkpoint['agent_state_dict'])
    assert not any(row.get('eval_search_rate',0) for row in rows)
    assert not any('A/prediction_loss' in row for row in rows)
print('[APM] Verified model, real labels, eval isolation, replay and mode:', mode, latest)
CHECK
  echo "[APM] MineDojo $MODE: PASS"
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: bash scripts/test_a.sh [--minedojo|--minedojo-off|--minedojo-abc|--minedojo-full|--minedojo-vnext|--audit-replay ...|--branch-checkpoint ...|--startup-replay ...|--minedojo-recovery --output ...]' >&2
  exit 2
fi
