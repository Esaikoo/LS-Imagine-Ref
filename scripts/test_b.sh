#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/test_a.sh
python tests/test_search_b.py
python -m py_compile search_a.py search_memory.py expr.py models.py tools.py test.py
bash -n scripts/test_b.sh
sh -n scripts/train.sh
echo '[B] Unit tests and syntax: PASS'
if [[ "${1:-}" == "--minedojo" ]]; then
  export MINEDOJO_HEADLESS=1 PYTHONUNBUFFERED=1
  ROOT="./logs/test_apm_b/run_$(date +%Y%m%dT%H%M%S)_$$"
  python expr.py --configs minedojo apm_ab apm_a_smoke \
    --task minedojo_harvest_log_in_plains --target_item log \
    --steps 96 --prefill 32 --eval_every 96 --eval_episode_num 1 \
    --episode_max_steps_override 32 --log_every 16 --train_ratio 4 \
    --search_a_start_step 32 --search_a_override_prob 1.0 --search_a_epsilon 1.0 \
    --search_a_min_labels 1 --search_a_min_positives 0 \
    --search_a_low_evidence 1.1 --compile False --video_pred_log False \
    --search_b_dim 32 --search_b_hidden 64 --search_b_slots 8 --search_b_lookback 16 \
    --log_all_metrics True --env_seed 0 --logdir "$ROOT"
  python - "$ROOT" <<'PY'
import sys, pathlib, torch, json, numpy as np
paths=list(pathlib.Path(sys.argv[1]).rglob('latest.pt'))
assert len(paths)==1,paths
p=paths[0]
c=torch.load(p,map_location='cpu')
assert c['search_a_version']==2 and c['search_b_version']==2
assert c['step']==96
assert any('.search_b.' in k for k in c['agent_state_dict'])
assert all(torch.isfinite(x).all() for x in c['agent_state_dict'].values())
rows=[json.loads(line) for line in (p.parent/'metrics.jsonl').read_text().splitlines()]
for key in ('B/revisit_loss','B/consistency_loss','B/place_loss','model_loss','actor_loss','value_loss','image_loss'):
 assert any(key in r for r in rows), key
 assert all(np.isfinite(r[key]) for r in rows if key in r), key
assert any('A/future_loss' in r for r in rows)
assert any('eval_b_revisit_rate' in r for r in rows)
assert any('eval_success' in r for r in rows)
eps=list(p.parent.glob('*_eps/*.npz'))
assert eps
has_memory = has_place = False
for e in eps:
 with np.load(e) as d:
  assert len({len(d[k]) for k in d.files})==1,e
  if 'log_b_memory_used' in d.files and e.parent.name == 'train_eps':
   has_memory = has_memory or bool(np.asarray(d['log_b_memory_used']).sum()>0)
   has_place = has_place or bool(np.asarray(d['search_place_valid']).sum()>0)
   assert d['log_b_keyframe_indices'].shape[-1]==8
   refs=d['log_b_keyframe_indices']
   tick=d['log_b_observation_index']
   assert np.all(refs <= tick[:,None])
   assert np.all(tick < np.arange(len(tick)))
  if e.parent.name=='eval_eps':
   assert not d['search_place_valid'].any()
assert has_memory, 'No post-prefill B memory transition was saved'
assert has_place, 'No training-only place labels were collected'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tags=EventAccumulator(str(p.parent)).Reload().Tags()['scalars']
for tag in ('B/revisit_loss','B/consistency_loss','scalars/model_loss','scalars/actor_loss','scalars/value_loss'):
 assert tag in tags,tag
print('[B] MineDojo+RSSM+A+B integration checkpoint/labels/logging PASS')
# Restore the exact saved architecture and exercise fixed-checkpoint memory ablation.
from types import SimpleNamespace
import test
cfg=SimpleNamespace(**c['config'])
cfg.agent_checkpoint_dir=str(p.parent)
cfg.logdir=str(pathlib.Path(sys.argv[1])/'memory_off_eval')
cfg.evaldir=''
cfg.search_b_control=False
cfg.eval_episode_num=1
test.main(cfg)
print('[B] Standalone checkpoint restore + memory-off evaluation PASS')
PY
elif [[ -n "${1:-}" ]]; then
 echo 'Usage: bash scripts/test_b.sh [--minedojo]' >&2; exit 2
fi
