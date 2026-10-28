#!/usr/bin/env python3
"""Evaluate original or A/B LS-Imagine checkpoints; real RGB MP4s, no training."""
import argparse
import copy
import csv
import hashlib
import json
import os
import socket
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
TASKS = {
    'harvest_log_in_plains': 'log',
    'harvest_water_with_bucket': 'water',
    'harvest_sand': 'sand',
    'mine_iron_ore': 'iron',
    'shear_sheep': 'wool',
}


class Mp4Writer:
    """Stream native RGB directly to the bundled FFmpeg (imageio versions vary)."""
    def __init__(self, path, fps, executable):
        self.path, self.fps, self.executable = path, fps, executable
        self.process = None
        self.shape = None

    def append_data(self, frame):
        if self.process is None:
            height, width, channels = frame.shape
            if channels != 3 or height % 2 or width % 2:
                raise ValueError('H.264 requires RGB frames with even dimensions')
            self.shape = frame.shape
            self.process = subprocess.Popen([
                self.executable, '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', '%dx%d' % (width, height), '-r', str(self.fps), '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        if frame.shape != self.shape or str(frame.dtype) != 'uint8':
            raise ValueError('RGB frame shape or dtype changed')
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            try:
                process.stdin.close()
                code = process.wait(timeout=60)
            except BaseException:
                process.kill()
                process.wait()
                raise
            if code:
                raise RuntimeError('FFmpeg exited with code %d' % code)


def heatmap_frame(rgb, native_heatmap, policy_heatmap, task, step, action, success, heatmap_mode='unet'):
    """Fixed 0..1 colors; show native U-Net output and the exact policy input."""
    import cv2
    import numpy as np
    height, width = rgb.shape[:2]
    native = np.clip(np.asarray(native_heatmap).squeeze() * 255, 0, 255).astype(np.uint8)
    native = cv2.resize(native, (width, height), interpolation=cv2.INTER_LINEAR)
    policy = cv2.resize(np.asarray(policy_heatmap).squeeze(), (width, height), interpolation=cv2.INTER_NEAREST)
    color = lambda mask: cv2.cvtColor(cv2.applyColorMap(mask, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.55, color(policy), 0.45, 0) if heatmap_mode == 'unet' else rgb
    panels = np.concatenate([rgb, color(native), overlay], axis=1)
    canvas = np.full((height + 56, width * 3, 3), 22, dtype=np.uint8)
    canvas[28:28+height] = panels
    labels = (('Original RGB', 'U-Net heatmap (native)', 'Policy heatmap overlay (64x64)')
              if heatmap_mode == 'unet' else ('Original RGB', 'U-Net OFF: zero heatmap', 'Policy heatmap: all zeros'))
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (index * width + 5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240,240,240), 1, cv2.LINE_AA)
    footer = '%s | step %d | %s | success=%s | blue=0 red=1' % (task, step, action, success)
    cv2.putText(canvas, footer, (5, height + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240,240,240), 1, cv2.LINE_AA)
    return np.ascontiguousarray(canvas)



def configure_heatmap_specs(task_specs, mode):
    """Disable the generator itself, preserving task, reward and model structure."""
    result = copy.deepcopy(task_specs)
    if mode == 'zero':
        result.pop('concentration_specs', None)
    elif mode != 'unet':
        raise ValueError('Unknown heatmap mode: ' + mode)
    return result


def validate_policy_heatmap(obs, mode):
    if mode == 'zero':
        import numpy as np
        heatmap = np.asarray(obs['heatmap'])
        if heatmap.shape != (64, 64) or heatmap.dtype != np.uint8 or np.any(heatmap):
            raise RuntimeError('No-U-Net evaluation requires an all-zero uint8 64x64 policy heatmap')


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def organize(checkpoint_root, tasks=None):
    """Rename on the same filesystem; never overwrite an existing checkpoint."""
    plan = []
    base = checkpoint_root.resolve()
    for task in (tasks or TASKS):
        alias = TASKS[task]
        source = base / ('latest_' + alias + '.pt')
        dest = base / task / 'latest.pt'
        if dest.exists():
            if source.exists():
                raise FileExistsError('Both source and destination exist: ' + str(dest))
        elif source.is_file():
            if not dest.resolve().is_relative_to(base):
                raise ValueError('Checkpoint destination escapes checkpoint root')
            plan.append((source, dest))
        else:
            raise FileNotFoundError(str(source) + ' or ' + str(dest))
    for source, dest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
        print('Organized:', dest, flush=True)


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def summarize(output, manifest):
    records = []
    for path in sorted(output.glob('*/run_*/episodes/episode_*.json')):
        records.append(json.loads(path.read_text()))
    fields = ['task', 'run', 'episode', 'seed', 'status', 'success', 'steps',
              'return', 'first_success_step', 'termination', 'wall_seconds', 'video', 'trace', 'heatmaps', 'heatmap_mode',
              'search_actions', 'search_decisions', 'search_rate', 'reference_start_match']
    with (output / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    tasks = {}
    for task in manifest['tasks']:
        runs = []
        for run in range(manifest['runs']):
            selected = [r for r in records if r['task'] == task and r['run'] == run]
            completed = [r for r in selected if r['status'] == 'completed']
            successes = sum(r['success'] for r in completed)
            runs.append(dict(run=run, completed=len(completed), errors=len(selected)-len(completed),
                             successes=successes, expected=manifest['episodes'],
                             success_rate=successes / len(completed) if completed else None))
        completed = [r for r in records if r['task'] == task and r['status'] == 'completed']
        rates = [r['success_rate'] for r in runs if r['completed'] == r['expected']]
        tasks[task] = dict(expected=manifest['runs'] * manifest['episodes'], completed=len(completed),
                           successes=sum(r['success'] for r in completed),
                           success_rate=sum(r['success'] for r in completed)/len(completed) if completed else None,
                           complete=len(completed) == manifest['runs'] * manifest['episodes'],
                           full_run_mean=statistics.mean(rates) if rates else None,
                           full_run_std=statistics.stdev(rates) if len(rates) > 1 else None, runs=runs)
    write_json(output / 'summary.json', tasks)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['task', 'expected', 'completed', 'successes', 'success_rate',
                                         'complete', 'full_run_mean', 'full_run_std'])
        writer.writeheader()
        for task, result in tasks.items():
            writer.writerow(dict(task=task, **{k:v for k,v in result.items() if k != 'runs'}))
    return tasks


def configure_search_checkpoint(config, checkpoint):
    """Restore the trained search module without changing the base evaluation."""
    state = checkpoint['agent_state_dict']
    saved = checkpoint.get('config', {})
    has_a = any('.search_a.' in k for k in state)
    has_b = any('.search_b.' in k for k in state)
    has_c = any(k.startswith('_search_c.') for k in state)
    if (has_a or has_b or has_c) and not saved:
        raise ValueError('Search checkpoint requires its saved configuration')
    for key, value in saved.items():
        if key.startswith(('search_a_', 'search_b_', 'search_c_')):
            config[key] = copy.deepcopy(value)
    config['search_a_enabled'] = has_a
    config['search_b_enabled'] = has_b
    config['search_b_model_version'] = checkpoint.get('search_b_version') or 2
    if (has_a or has_c) and 'step' not in checkpoint:
        raise ValueError('A/C checkpoint requires training step to restore warmup')
    from search_a import check_checkpoint, restore_a_config
    from search_memory import check_b_checkpoint
    restore_a_config(config, checkpoint)
    check_checkpoint(checkpoint, has_a)
    check_b_checkpoint(checkpoint, has_b)
    from search_c import restore_c_config
    restore_c_config(config, checkpoint)
    return int(checkpoint.get('step', 0))


def has_control_override(args):
    return (getattr(args, "a_control", "checkpoint") != "checkpoint"
            or getattr(args, "c_control", "checkpoint") != "checkpoint")


def apply_a_control_mode(config, mode):
    """A runtime-only ablation: keep every trained module and its saved weights."""
    if mode == 'checkpoint':
        return
    if config.get('search_c_enabled') and config.get('search_c_control'):
        raise ValueError('Use explicit C ablation settings for a C checkpoint; legacy A-only control override is ambiguous')
    if mode == 'repaired':
        from search_a import apply_repaired_control
        apply_repaired_control(config)
        return
    if mode != 'off':
        raise ValueError('Unknown A control mode')
    if not config.get('search_a_enabled'):
        raise ValueError('Disabling A control requires an A checkpoint')
    config['search_a_control'] = False


def validate_reference(args, task, run, config, task_specs, sim_specs):
    """Fail before evaluation if common settings differ from the existing run."""
    if args.reference_output is None:
        return None
    root = args.reference_output
    reference = json.loads((root / 'manifest.json').read_text())
    for key in ('seed', 'runs', 'episodes', 'fps'):
        if reference[key] != getattr(args, key):
            raise ValueError('Reference protocol mismatch: ' + key)
    if reference.get('heatmap_mode', 'unet') != args.heatmap_mode:
        raise ValueError('Reference heatmap mode differs')
    settings_dir = root / task / ('run_%02d' % run)
    settings_file = (settings_dir / ('settings_episode_%03d.json' % args.worker_episode)
                     if args.worker_episode is not None else settings_dir / 'settings.json')
    saved = json.loads(settings_file.read_text())
    runtime = {'logdir', 'results_dir', 'device', 'name', 'traindir', 'evaldir'}
    different = [key for key, value in saved['config'].items()
                 if key not in runtime and not key.startswith(('search_a_', 'search_b_', 'search_c_'))
                 and config.get(key) != value]
    if has_control_override(args):
        if getattr(args, 'a_control', 'checkpoint') != 'checkpoint':
            if not saved['config'].get('search_a_control') or not saved['config'].get('search_a_enabled'):
                raise ValueError('A override requires a reference with A control enabled')
        if getattr(args, 'c_control', 'checkpoint') != 'checkpoint':
            if not saved['config'].get('search_c_enabled'):
                raise ValueError('C override requires a C reference checkpoint')
        expected_path = (args.checkpoint_root / task / 'latest.pt').resolve()
        if Path(saved['checkpoint']).resolve() != expected_path:
            raise ValueError('A-off ablation must load exactly the reference checkpoint')
        from search_a import REPAIRED_CONTROL_DEFAULTS
        allowed = set()
        if getattr(args, 'a_control', 'checkpoint') != 'checkpoint':
            allowed.add('search_a_control')
        if getattr(args, 'c_control', 'checkpoint') == 'off':
            allowed.add('search_c_control')
        if getattr(args, 'c_control', 'checkpoint') == 'no_predictions':
            allowed.add('search_c_use_predictions')
        if args.a_control == 'repaired':
            allowed.update(REPAIRED_CONTROL_DEFAULTS)
        for key, value in saved['config'].items():
            if key.startswith(('search_a_', 'search_b_', 'search_c_')) and key not in allowed:
                if config.get(key) != value:
                    different.append(key)
    clean = lambda spec: {k: v for k, v in spec.items() if k != 'log_dir'}
    if clean(saved['task_specs']) != clean(task_specs):
        different.append('task_specs')
    if saved['sim_specs'] != sim_specs:
        different.append('sim_specs')
    if different:
        raise ValueError('Reference settings differ: ' + ', '.join(different))
    return dict(output=str(root), common_config_matches=True,
                a_control_mode=getattr(args, 'a_control', 'checkpoint'),
                c_control_mode=getattr(args, 'c_control', 'checkpoint'),
                runtime_changes={k: dict(before=saved['config'].get(k), after=v)
                    for k, v in config.items() if k.startswith(("search_a_", "search_c_"))
                    and saved['config'].get(k) != v},
                task_specs_match=True, sim_specs_match=True,
                note='Same configured seeds and protocol; world identity checked separately from traces.')



def locate_sim(environment):
    from minedojo.sim import MineDojoSim
    current = environment
    while not isinstance(current, MineDojoSim):
        if not hasattr(current, 'env'):
            raise RuntimeError('Cannot locate MineDojo simulator')
        current = current.env
    return current


def prepare_fixed_mission(mission_xml):
    """Set simulator-only options missing from HarvestMeta's public arguments."""
    from lxml import etree
    generators = mission_xml.xpath('//*[local-name()="DefaultWorldGenerator"]')
    if len(generators) != 1 or not generators[0].get('seed'):
        raise RuntimeError('Fixed evaluation needs one explicitly seeded default world')
    generators[0].set('forceReset', 'true')
    times = mission_xml.xpath('//*[local-name()="ServerInitialConditions"]/*[local-name()="Time"]')
    if len(times) != 1:
        raise RuntimeError('Missing mission time handler')
    time_element = times[0]
    start = time_element.find('{*}StartTime')
    if start is None:
        namespace = etree.QName(time_element).namespace
        start = etree.Element(('{' + namespace + '}' if namespace else '') + 'StartTime')
        time_element.insert(0, start)
    start.text = '6000'
    passage = time_element.find('{*}AllowPassageOfTime')
    if passage is None or passage.text != 'false':
        raise RuntimeError('Fixed evaluation requires a frozen mission clock')


def seed_clean_bridge(environment, episode_seed, initial_dir, stem):
    """Seed both the Java launch queue and mission token; never reuse a client."""
    from minedojo.sim.bridge.bridge_env import BridgeEnv
    from minedojo.sim.bridge.mc_instance import InstanceManager
    from lxml import etree
    sim = locate_sim(environment)
    sim._bridge_env.close()

    class FixedSeedBridge(BridgeEnv):
        def _get_new_instance(self, port=None, instance_id=None):
            if port is not None:
                raise RuntimeError('Paired evaluation requires a new managed client')
            instance = InstanceManager.get_instance(os.getpid(), instance_id=instance_id)
            if instance.running:
                raise RuntimeError('Refusing to reuse a Minecraft instance')
            # MineDojo 0.1 seeds only the first mission through this launch queue.
            instance._seed = int(episode_seed)
            instance.launch(replaceable=self._is_fault_tolerant)
            instance.had_to_clean = False
            return instance

        def _send_mission(self, instance, mission_xml_etree, token_in, agent_count=1, seed=None):
            prepare_fixed_mission(mission_xml_etree)
            (initial_dir / (stem + '_mission.xml')).write_bytes(
                etree.tostring(mission_xml_etree, pretty_print=True))
            return super()._send_mission(instance, mission_xml_etree, token_in,
                                         agent_count=agent_count, seed=int(episode_seed))

    sim._bridge_env = FixedSeedBridge(is_fault_tolerant=True, seed=int(episode_seed))


def terrain_fingerprint(names, metadata):
    """Keep block variants; ignore only the leaves CHECK_DECAY bit (Minecraft 1.11.2)."""
    import numpy as np
    names = np.asarray(names).astype(str)
    raw = np.asarray(metadata)
    stable = raw.copy()
    leaves = np.isin(names, ['leaves', 'leaves2', 'minecraft:leaves', 'minecraft:leaves2'])
    stable[leaves] = np.bitwise_and(stable[leaves], ~8)
    def digest(values):
        return hashlib.sha256(json.dumps(dict(names=names.tolist(), meta=values.tolist()),
                                         sort_keys=True).encode()).hexdigest()
    return digest(stable), digest(raw)


def compare_initial_scenes(original, candidate):
    """Coordinates/terrain are audit data only, never model observations."""
    import numpy as np
    exact = ('world_seed', 'mission_seed', 'java_launch_seed', 'terrain_sha256',
             'inventory', 'life_stats', 'policy_observation_keys')
    checks = {key: original[key] == candidate[key] for key in exact}
    for key in ('pos', 'pitch', 'yaw'):
        checks[key] = bool(np.allclose(original['location_stats'][key],
                                      candidate['location_stats'][key], rtol=0, atol=1e-5))
    for key in ('biome_id', 'is_raining', 'sea_level'):
        checks[key] = original['location_stats'][key] == candidate['location_stats'][key]
    return dict(match=all(checks.values()), checks=checks)


def paired_summary(output, task, episodes):
    records = {}
    for label in ('original', 'a_model'):
        folder = output / label / task / 'run_00'
        records[label] = {int(p.stem.split('_')[-1]): json.loads(p.read_text())
                         for p in (folder / 'episodes').glob('episode_*.json')}
    pairs = []
    for episode in range(episodes):
        left, right = (records[label].get(episode) for label in ('original', 'a_model'))
        if not (left and right and left['status'] == right['status'] == 'completed'):
            continue
        initial = []
        for label in ('original', 'a_model'):
            initial.append(json.loads((output/label/task/'run_00/initials'/
                                       ('episode_%03d.json' % episode)).read_text()))
        check = compare_initial_scenes(*initial)
        if not check['match']:
            raise RuntimeError('Completed pair has mismatching initial conditions')
        pairs.append(dict(episode=episode, world_seed=initial[0]['world_seed'],
                          original_success=left['success'], a_success=right['success'],
                          original_steps=left['steps'], a_steps=right['steps'], initial_match=True))
    result = dict(expected_pairs=episodes, completed_pairs=len(pairs),
                  completed_rollouts=sum(r['status'] == 'completed'
                      for values in records.values() for r in values.values()),
                  both_success=sum(p['original_success'] and p['a_success'] for p in pairs),
                  original_only=sum(p['original_success'] and not p['a_success'] for p in pairs),
                  a_only=sum(p['a_success'] and not p['original_success'] for p in pairs),
                  both_fail=sum(not p['original_success'] and not p['a_success'] for p in pairs),
                  models={label: dict(completed=sum(r['status']=='completed' for r in values.values()),
                      successes=sum(r['success'] for r in values.values() if r['status']=='completed'))
                          for label, values in records.items()}, pairs=pairs)
    for values in result['models'].values():
        values['success_rate'] = values['successes']/values['completed'] if values['completed'] else None
    write_json(output/'paired_summary.json', result)
    return result


def paired_main(args):
    """Two validation maps first, then original/A on each independent new map."""
    import fcntl
    task = args.tasks[0]
    roots = dict(original=args.checkpoint_root, a_model=args.paired_checkpoint_root)
    for checkpoint_root in roots.values():
        organize(checkpoint_root, args.tasks)
    manifest = dict(protocol_version=4, task=task, pairs=args.episodes,
        policy_seed_start=args.seed, world_seed_start=args.world_seed_start,
        probe_policy_seed_start=args.seed+1000, probe_world_seed_start=args.world_seed_start+1000,
        order=['original','a_model'], fresh_process_per_rollout=True, fast_reset=None,
        fixed_time=6000, allow_time_passage=False, weather='clear',
        heatmap_mode=args.heatmap_mode, fps=args.fps, device=args.device, threads=args.threads,
        checkpoint_roots={k:str(v) for k,v in roots.items()},
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_config=(ROOT/'configs.yaml').read_text(),
        task_specs=(ROOT/'envs/tasks/task_specs.yaml').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'paired_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Paired evaluation settings/code differ; use another output directory')
    if not path.exists():
        write_json(path, manifest)
    with (args.output/'.paired_lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def run_one(label, episode, probe):
            base = args.output/'preflight' if probe else args.output
            output = base/label
            folder = output/task/'run_00'
            record_path = folder/'episodes'/('episode_%03d.json' % episode)
            expected_status = 'probe' if probe else 'completed'
            if record_path.exists() and json.loads(record_path.read_text())['status'] == expected_status:
                return
            folder.mkdir(parents=True, exist_ok=True)
            seed = args.seed + (1000 if probe else 0)
            world_seed = args.world_seed_start + (1000 if probe else 0)
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                '--checkpoint-root',str(roots[label]), '--output',str(output),
                '--tasks',task, '--runs','1', '--episodes',str(2 if probe else args.episodes),
                '--seed',str(seed), '--world-seed-start',str(world_seed),
                '--fps',str(args.fps), '--device',args.device, '--threads',str(args.threads),
                '--heatmap-mode',args.heatmap_mode, '--worker-task',task, '--worker-run','0',
                '--worker-episode',str(episode)]
            if label == 'a_model':
                command += ['--reference-output',str(base/'original')]
            if probe:
                command += ['--reset-only']
            progress = dict(status='running', phase='preflight' if probe else 'evaluation',
                            episode=episode, model=label, world_seed=world_seed+episode,
                            pid=os.getpid())
            write_json(args.output/'status.json', progress)
            with (args.output/'order.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='start', **progress))+'\n')
            print(json.dumps(progress), flush=True)
            os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = str(folder)
            with (folder/'console.log').open('a') as stream:
                result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(args.output/'status.json',dict(progress,status='failed',
                    returncode=result.returncode, console=str(folder/'console.log')))
                raise RuntimeError('Paired worker failed: '+str(folder/'console.log'))
            record = json.loads(record_path.read_text())
            if record['status'] != expected_status:
                raise RuntimeError('Unexpected worker result: '+str(record_path))
            with (args.output/'order.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='finish', **progress,
                                             result=record['status']))+'\n')

        validation = args.output/'preflight.json'
        if not validation.exists():
            checks = []
            for episode in range(2):
                for label in ('original','a_model'):
                    run_one(label,episode,True)
                scenes = [json.loads((args.output/'preflight'/label/task/'run_00/initials'/
                                      ('episode_%03d.json'%episode)).read_text())
                          for label in ('original','a_model')]
                check = compare_initial_scenes(*scenes)
                checks.append(dict(episode=episode,world_seed=scenes[0]['world_seed'],**check))
                if not check['match']:
                    write_json(validation,dict(status='failed',checks=checks))
                    write_json(args.output/'status.json',dict(status='failed',phase='preflight',
                                                              reason='scene mismatch'))
                    raise RuntimeError('Scene validation failed; full evaluation was not started')
            write_json(validation,dict(status='passed',checks=checks,
                note='Two unseen seeds, each verified in fresh original/A clients before policy actions.'))
        if json.loads(validation.read_text())['status'] != 'passed':
            raise RuntimeError('Preflight has not passed')
        if args.probe_only:
            write_json(args.output/'status.json',dict(status='validated',phase='preflight'))
            print('Preflight passed. No scored episodes run.', flush=True)
            return
        try:
            for episode in range(args.episodes):
                for label in ('original','a_model'):
                    run_one(label,episode,False)
                    paired_summary(args.output,task,args.episodes)
            result = paired_summary(args.output,task,args.episodes)
            assert result['completed_pairs'] == args.episodes
            write_json(args.output/'status.json',dict(status='completed',
                completed_pairs=args.episodes,completed_rollouts=2*args.episodes))
        except BaseException:
            status=json.loads((args.output/'status.json').read_text())
            write_json(args.output/'status.json',dict(status,status='failed',error=traceback.format_exc()))
            raise
        print(json.dumps(result,indent=2), flush=True)


def a_control_ablation_summary(args):
    if args.a_control == "repaired":
        task = args.tasks[0]
        pairs = []
        for episode in range(args.episodes):
            stem = "episode_%03d.json" % episode
            old_path = args.reference_output/task/"run_00"/"episodes"/stem
            new_path = args.output/task/"run_00"/"episodes"/stem
            if not old_path.exists() or not new_path.exists():
                continue
            old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
            if old["status"] != "completed" or new["status"] != "completed":
                continue
            if new.get("a_control_mode") != "repaired":
                raise ValueError("Mixed controller versions in repaired results")
            pairs.append(dict(episode=episode, world_seed=args.world_seed_start+episode,
                              old_success=bool(old["success"]), repaired_success=bool(new["success"]),
                              old_steps=old["steps"], repaired_steps=new["steps"],
                              repaired_search_rate=new.get("search_rate", 0)))
        summary = dict(completed=len(pairs), requested=args.episodes,
                       old_successes=sum(p["old_success"] for p in pairs),
                       repaired_successes=sum(p["repaired_success"] for p in pairs), pairs=pairs)
        write_json(args.output/"a_repaired_comparison.json", summary)
        return summary
    """Compare only maps completed in both the existing on run and the new off run."""
    task = args.tasks[0]
    pairs = []
    for episode in range(args.episodes):
        stem = 'episode_%03d' % episode
        folder = args.output / task / 'run_00'
        off_path = folder / 'episodes' / (stem + '.json')
        if not off_path.exists():
            continue
        off = json.loads(off_path.read_text())
        if off['status'] != 'completed':
            continue
        reference = args.reference_output / task / 'run_00'
        on = json.loads((reference / 'episodes' / (stem + '.json')).read_text())
        if on['status'] != 'completed':
            raise RuntimeError('Missing completed A-on reference')
        scenes = [json.loads((f / 'initials' / (stem + '.json')).read_text())
                  for f in (reference, folder)]
        if not compare_initial_scenes(*scenes)['match']:
            raise RuntimeError('A-off completed with different starting conditions')
        if args.a_control == 'off' and (off['search_actions'] or off['search_decisions']):
            raise RuntimeError('A controller was not fully disabled')
        pairs.append(dict(episode=episode, world_seed=off['world_seed'],
            on_success=on['success'], off_success=off['success'],
            on_steps=on['steps'], off_steps=off['steps'],
            on_control_rate=on['search_rate'], off_control_rate=off['search_rate'],
            off_action_counts=off.get('action_counts', {}),
            off_cells_2m=off.get('visited_cells_2m'), initial_match=True))
    result = dict(expected=args.episodes, completed=len(pairs), same_checkpoint=True,
        trained_weights_retained=True, runtime_override_only=True,
        on_successes=sum(p['on_success'] for p in pairs),
        off_successes=sum(p['off_success'] for p in pairs),
        off_only=sum(p['off_success'] and not p['on_success'] for p in pairs),
        on_only=sum(p['on_success'] and not p['off_success'] for p in pairs),
        both_success=sum(p['off_success'] and p['on_success'] for p in pairs),
        both_fail=sum(not p['off_success'] and not p['on_success'] for p in pairs), pairs=pairs)
    for label in ('on', 'off'):
        result[label+'_success_rate'] = result[label+'_successes']/len(pairs) if pairs else None
    c_override = getattr(args, 'c_control', 'checkpoint') != 'checkpoint'
    result['a_control'] = args.a_control
    result['c_control'] = getattr(args, 'c_control', 'checkpoint')
    write_json(args.output/('c_control_comparison.json' if c_override else 'a_control_comparison.json'), result)
    lines = ['# 同一 A 检查点关闭接管：定位测试',
        '', '本报告自动更新；仅统计已完成且起始条件匹配的地图，部分结果不能当成完整 20 回合结果。',
        '', '完成：%d/%d。两组使用同一检查点，A 网络保留，仅关闭在线接管。' % (len(pairs),args.episodes)]
    if pairs:
        lines += ['', 'A 开启：%d/%d；A 关闭：%d/%d。仅关闭成功 %d 张，仅开启成功 %d 张。' %
            (result['on_successes'],len(pairs),result['off_successes'],len(pairs),
             result['off_only'],result['on_only'])]
    lines += ['', '| 地图种子 | 开启结果/步数 | 关闭结果/步数 |',
              '| --- | ---: | ---: |']
    for p in pairs:
        lines.append('| %s | %s / %d | %s / %d |' % (p['world_seed'],
            '成功' if p['on_success'] else '失败', p['on_steps'],
            '成功' if p['off_success'] else '失败', p['off_steps']))
    if c_override:
        lines = [line.replace('同一 A 检查点关闭接管', '同一检查点 C 消融')
                 .replace('A 网络保留，仅关闭在线接管', '全部权重保留，C 模式为 '+args.c_control)
                 .replace('A 开启', '原控制').replace('A 关闭', '消融控制')
                 for line in lines]
    (args.output/'diagnosis_progress_CN.md').write_text('\n'.join(lines)+'\n')
    return result


def retryable_initial_scene(folder, episode):
    """Never resample a rollout outcome: only a terrain mismatch before action zero."""
    stem = 'episode_%03d' % episode
    record_path = folder/'episodes'/(stem+'.json')
    check_path = folder/'initials'/(stem+'_comparison.json')
    if not record_path.exists() or not check_path.exists():
        return False
    record = json.loads(record_path.read_text())
    check = json.loads(check_path.read_text())
    return (record.get('status') == 'error' and record.get('steps') == 0
            and record.get('termination') == 'error' and not record.get('success')
            and 'Initial scene mismatch' in record.get('error','')
            and check.get('match') is False
            and {k for k,v in check.get('checks',{}).items() if not v} == {'terrain_sha256'})


def fixed_reference_main(args, manifest):
    """Reuse completed A-on references; run one clean A-off process per map."""
    import fcntl
    task = args.tasks[0]
    order = args.episode_order or list(range(args.episodes))
    checkpoint = args.checkpoint_root / task / 'latest.pt'
    stat = checkpoint.stat()
    identity = dict(checkpoint=str(checkpoint),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        episode_order=order, a_control=args.a_control,
        c_control=getattr(args, "c_control", "checkpoint"), reference=str(args.reference_output))
    identity_path = args.output/'fixed_run_manifest.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('A-off evaluator/order/checkpoint changed; use a new output directory')
    if not identity_path.exists():
        write_json(identity_path,identity)
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder = args.output/task/'run_00'
        folder.mkdir(parents=True,exist_ok=True)
        try:
            for episode in order:
                record_path = folder/'episodes'/('episode_%03d.json'%episode)
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if (record['status']=='completed' and record.get('a_control_mode')==args.a_control
                            and record.get('c_control_mode', 'checkpoint')==args.c_control):
                        continue
                command=[sys.executable,'-u',str(Path(__file__).resolve()),
                    '--checkpoint-root',str(args.checkpoint_root),'--output',str(args.output),
                    '--tasks',task,'--runs','1','--episodes',str(args.episodes),
                    '--seed',str(args.seed),'--world-seed-start',str(args.world_seed_start),
                    '--fps',str(args.fps),'--device',args.device,'--threads',str(args.threads),
                    '--heatmap-mode',args.heatmap_mode,'--reference-output',str(args.reference_output),
                    '--a-control',args.a_control,'--c-control',args.c_control,'--worker-task',task,'--worker-run','0',
                    '--worker-episode',str(episode)]
                progress=dict(status='running',phase='a_control_' + args.a_control,episode=episode,
                    world_seed=args.world_seed_start+episode,pid=os.getpid())
                write_json(args.output/'status.json',progress)
                print(json.dumps(progress),flush=True)
                env=os.environ.copy()
                env['MALMO_MINECRAFT_OUTPUT_LOGDIR']=str(folder)
                for scene_attempt in range(3):
                    with (folder/'console.log').open('a') as stream:
                        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
                    if result.returncode == 0:
                        break
                    if scene_attempt == 2 or not retryable_initial_scene(folder,episode):
                        raise RuntimeError('A-off worker failed (%d): %s' % (result.returncode,folder/'console.log'))
                    import shutil
                    stamp=time.time_ns()
                    archive=folder/'errors'/('initial_scene_%03d_%d'%(episode,stamp))
                    archive.mkdir(parents=True)
                    shutil.copy2(record_path,archive/'episode.json')
                    for initial in (folder/'initials').glob('episode_%03d*'%episode):
                        shutil.copy2(initial,archive/initial.name)
                    with (args.output/'scene_validation_retries.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(episode=episode,attempt=scene_attempt+1,
                            world_seed=args.world_seed_start+episode,executed_actions=0,
                            archive=str(archive),reason='terrain mismatch before any policy action'))+'\n')
                    print('Fresh same-seed retry after zero-action terrain mismatch:',episode,flush=True)
                result=a_control_ablation_summary(args)
                print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)
            result=a_control_ablation_summary(args)
            assert result['completed']==args.episodes
            write_json(args.output/'status.json',dict(status='completed',phase='a_control_' + args.a_control,
                completed_rollouts=result['completed'],
                comparison={k:v for k,v in result.items() if k != 'pairs'}))
        except BaseException:
            write_json(args.output/'status.json',dict(status='failed',phase='a_control_' + args.a_control,
                episode=episode,error=traceback.format_exc(),console=str(folder/'console.log')))
            raise


def worker(args, manifest):
    # Imports stay out of the coordinator, leaving its memory available to Minecraft.
    os.environ.setdefault('MINEDOJO_HEADLESS', '1')
    import gym
    import imageio_ffmpeg
    import numpy as np
    import torch
    import ruamel.yaml as yaml
    import tools
    from expr import LS_Imagine
    from envs.tasks import get_specs
    from envs.tasks.minedojo import make_minedojo
    from envs.tasks.base.ls_imagine_wrapper import LSImagineWrapper
    from envs.wrappers import OneHotAction, RewardObs, SelectAction

    torch.set_num_threads(args.threads)
    task = args.worker_task
    run = args.worker_run
    run_seed = args.seed + run * 100000
    tools.set_seed_everywhere(run_seed)
    folder = args.output / task / ('run_%02d' % run)
    for name in ('episodes', 'videos', 'traces', 'heatmaps', 'initials'):
        (folder / name).mkdir(parents=True, exist_ok=True)

    class Capture(gym.Wrapper):
        """Tap native RGB and game state before the 64x64 policy conversion."""
        def remember(self, obs):
            self.frame = np.ascontiguousarray(obs['rgb'].transpose(1, 2, 0))
            self.heatmap = np.asarray(obs.get('heatmap',
                np.zeros((*self.frame.shape[:2], 1), dtype=np.float32))).copy()
            inventory = obs.get('inventory', {})
            counts = {}
            for name, count in zip(inventory.get('name', []), inventory.get('quantity', [])):
                if int(count):
                    counts[str(name)] = counts.get(str(name), 0) + int(count)
            self.terrain = obs.get('voxels')
            self.details = dict(inventory=counts,
                                life_stats=convert(obs.get('life_stats', {})),
                                location_stats=convert(obs.get('location_stats', {})))
        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            self.remember(obs)
            return obs
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.remember(obs)
            return obs, reward, done, info

    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    config_dict = yaml.safe_load((ROOT / 'configs.yaml').read_text())
    config = copy.deepcopy(config_dict['defaults'])
    merge(config, config_dict['minedojo'])
    checkpoint_path = args.checkpoint_root / task / 'latest.pt'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', mmap=True)
    checkpoint_step = configure_search_checkpoint(config, checkpoint)
    from search_c import apply_c_control_mode
    apply_c_control_mode(config, args.c_control)
    apply_a_control_mode(config, args.a_control)
    config.update(compile=False, video_pred_log=False, use_wandb=False, device=args.device,
                  seed=run_seed, task='minedojo_test_' + task, target_item=TASKS[task],
                  logdir=str(folder), results_dir=str(folder), name=task)
    config = SimpleNamespace(**config)
    task_id, task_specs, sim_specs = copy.deepcopy(get_specs('test_' + task))
    max_steps = task_specs['terminal_specs']['max_steps']
    config.episode_max_steps = max_steps
    task_specs['concentration_specs'].update(max_steps=max_steps,
        gaussian_reward_weight=config.gaussian_reward_weight, gaussian_sigma_weight=config.gaussian_sigma_weight)
    condition = task_specs['success_specs'].get('all', task_specs['success_specs'].get('any'))
    task_specs['clip_specs']['target_object'] = condition['item']['type']
    task_specs = configure_heatmap_specs(task_specs, args.heatmap_mode)
    task_specs['log_dir'] = str(folder)
    sim_specs['seed'] = run_seed
    fixed = args.world_seed_start is not None
    if fixed:
        if args.worker_episode is None:
            raise ValueError('Fixed worlds require one fresh worker per episode')
        episode_seed = run_seed + args.worker_episode
        task_specs['fast_reset'] = None
        sim_specs.update(seed=episode_seed, world_seed=str(args.world_seed_start + args.worker_episode),
                         allow_time_passage=False, initial_weather='clear',
                         use_voxel=True, voxel_size=dict(xmin=-5,xmax=5,ymin=-2,ymax=2,zmin=-5,zmax=5))
    reference_check = validate_reference(args, task, run, vars(config), task_specs, sim_specs)
    write_json(folder / 'settings.json', dict(config=vars(config), task_specs=task_specs, sim_specs=sim_specs,
               checkpoint=str(checkpoint_path), checkpoint_step=checkpoint_step,
               a_control_mode=args.a_control, c_control_mode=args.c_control,
               search_a_version=checkpoint.get('search_a_version'),
               search_a_control_version=config.search_a_control_version,
               search_b_version=checkpoint.get('search_b_version'),
               search_c_version=checkpoint.get('search_c_version'),
               reference_check=reference_check, max_steps=max_steps, heatmap_mode=args.heatmap_mode,
               seed_note=('Fresh world/client, explicit world/Java/mission seed; XML forces new world at time 6000.'
                          if fixed else 'Group seed and per-episode simulator/policy RNG seeds; world generation follows task YAML.')))
    if args.worker_episode is not None:
        settings = json.loads((folder/'settings.json').read_text())
        write_json(folder/('settings_episode_%03d.json'%args.worker_episode), settings)

    def restart_bridge(environment):
        # A dead client can hang in PAUSING_OLD_SERVER. Replace only its bridge;
        # retain all policy/reward wrappers and the bridge's Python RNG stream.
        from minedojo.sim import MineDojoSim
        from minedojo.sim.bridge.bridge_env import BridgeEnv
        sim = environment
        while not isinstance(sim, MineDojoSim):
            if not hasattr(sim, 'env'):
                raise RuntimeError('Cannot locate MineDojo simulator for recovery')
            sim = sim.env
        old = sim._bridge_env
        rng_state = copy.deepcopy(old._rng.bit_generator.state)
        old.close()
        new = BridgeEnv(is_fault_tolerant=True, seed=0)
        new._rng.bit_generator.state = rng_state
        sim._bridge_env = new

    def record_recovery(episode, reason):
        with (folder / 'recovery.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(episode=episode, reason=reason,
                time=time.strftime('%Y-%m-%dT%H:%M:%S'))) + '\n')

    env = None
    try:
        with torch.no_grad():
            env = make_minedojo(task_id, task_specs, sim_specs)
            if fixed:
                wrapped = env
                while hasattr(wrapped, 'env'):
                    if 'FastReset' in wrapped.__class__.__name__:
                        raise RuntimeError('Fast-reset wrapper found in fixed-world evaluation')
                    wrapped = wrapped.env
                seed_clean_bridge(env, episode_seed, folder/'initials',
                                  'episode_%03d'%args.worker_episode)
            if args.heatmap_mode == 'zero':
                wrapped = env
                while hasattr(wrapped, 'env'):
                    if wrapped.__class__.__name__ == 'ConcentrationWrapper':
                        raise RuntimeError('U-Net concentration wrapper unexpectedly enabled')
                    wrapped = wrapped.env
                print('U-Net disabled: no generator loaded; policy heatmap fixed to zeros.', flush=True)
            current = env
            while not isinstance(current, LSImagineWrapper):
                current = current.env
            capture = Capture(current.env)
            current.env = capture
            action_names = current._action_names
            env = RewardObs(SelectAction(OneHotAction(env), 'action'))
            config.num_actions = env.action_space.shape[0]
            agent = LS_Imagine(env.observation_space, env.action_space, config,
                               SimpleNamespace(step=0), None).to(args.device)
            # torch.compile checkpoints add this component to module names.
            state_dict = {k.replace('._orig_mod.', '.'): v
                          for k, v in checkpoint['agent_state_dict'].items()}
            if len(state_dict) != len(checkpoint['agent_state_dict']):
                raise ValueError('Checkpoint key collision')
            agent.load_state_dict(state_dict, strict=True)
            if args.a_control == 'off':
                if agent._wm.search_a is None or agent._search_a is not None:
                    raise RuntimeError('A-off must retain A weights and disable only the controller')
                print('A control OFF: A network retained, identical weights loaded strictly.', flush=True)
            if agent._search_a is not None:
                write_json(folder / "search_control.json", dict(
                    model_version=config.search_a_model_version,
                    control_version=agent._search_a.control_version,
                    macro_names=list(agent._search_a.macro_names),
                    interrupt_codes={"1":"evidence", "2":"interaction", "3":"budget"}))
            agent._step = checkpoint_step
            agent._should_pretrain._once = False
            agent._logger.step = checkpoint_step * config.action_repeat
            del checkpoint, state_dict
            agent.eval().requires_grad_(False)
            print('Loaded checkpoint strictly:', checkpoint_path, flush=True)
            episode_indices = [args.worker_episode] if args.worker_episode is not None else range(args.episodes)
            for episode in episode_indices:
                stem = 'episode_%03d' % episode
                record_path = folder / 'episodes' / (stem + '.json')
                previous_record = json.loads(record_path.read_text()) if record_path.exists() else None
                if previous_record and previous_record['status'] == 'completed':
                    if fixed:
                        print('Resume: retaining completed independent episode', record_path, flush=True)
                        continue
                    # Consume the initial mission when resuming; otherwise the next
                    # episode silently reuses the seeded first world. No policy rollout.
                    tools.set_seed_everywhere(run_seed + episode)
                    env.seed(run_seed + episode)
                    env.reset()
                    record_recovery(episode, 'resume_reset_only; completed rollout retained')
                    print('Resume: retained completed rollout; reset only:', record_path, flush=True)
                    continue
                if previous_record:
                    attempts = folder / 'errors'
                    attempts.mkdir(exist_ok=True)
                    write_json(attempts / (stem + '_%d.json' % time.time_ns()), previous_record)
                seed = run_seed + episode
                tools.set_seed_everywhere(seed)
                env.seed(seed)
                video_path = folder / 'videos' / (stem + '.mp4')
                trace_path = folder / 'traces' / (stem + '.jsonl')
                heatmap_path = folder / 'heatmaps' / (stem + '.npz')
                started = time.monotonic()
                record = dict(task=task, run=run, episode=episode, seed=seed, status='error',
                              success=False, steps=0, return_=0.0, first_success_step=None,
                              video=str(video_path.relative_to(args.output)), trace=str(trace_path.relative_to(args.output)),
                              heatmaps=str(heatmap_path.relative_to(args.output)), heatmap_mode=args.heatmap_mode)
                writer = None
                search_actions = search_decisions = 0
                action_counts, visited_cells, pitch_up_steps = {}, set(), 0
                record['a_control_mode'] = args.a_control
                record['c_control_mode'] = args.c_control
                try:
                    for reset_attempt in range(3):
                        try:
                            obs = env.reset()
                            break
                        except (socket.timeout, ConnectionError) as error:
                            record_recovery(episode, 'reset failed: ' + repr(error))
                            if reset_attempt == 2:
                                raise
                            if fixed:
                                seed_clean_bridge(env, seed, folder/'initials', stem)
                            else:
                                restart_bridge(env)
                            tools.set_seed_everywhere(seed)
                            env.seed(seed)
                    validate_policy_heatmap(obs, args.heatmap_mode)
                    if fixed:
                        if 'voxels' in obs or 'location_stats' in obs:
                            raise RuntimeError('Audit information leaked into policy observations')
                        if capture.terrain is None:
                            raise RuntimeError('Missing terrain audit')
                        names = np.asarray(capture.terrain['block_name']).astype(str)
                        meta = np.asarray(capture.terrain['block_meta'])
                        terrain_hash, terrain_raw_hash = terrain_fingerprint(names, meta)
                        initial = dict(world_seed=int(sim_specs['world_seed']),mission_seed=seed,
                            java_launch_seed=int(locate_sim(env)._bridge_env._instances[0]._seed),
                            terrain_sha256=terrain_hash, terrain_raw_sha256=terrain_raw_hash,
                            terrain_check='Exact names/metadata except leaves CHECK_DECAY bit 8; raw arrays retained',
                            policy_observation_keys=sorted(obs),
                            **capture.details)
                        write_json(folder/'initials'/(stem+'.json'),initial)
                        np.savez_compressed(folder/'initials'/(stem+'.npz'),rgb=capture.frame,
                            image=obs['image'],heatmap=obs['heatmap'],block_name=names,block_meta=meta)
                        import cv2
                        cv2.imwrite(str(folder/'initials'/(stem+'.png')),cv2.cvtColor(capture.frame,cv2.COLOR_RGB2BGR))
                        record.update(world_seed=initial['world_seed'],mission_seed=seed,
                                      initial_terrain_sha256=terrain_hash)
                        if args.reference_output:
                            reference_initial = json.loads((args.reference_output/task/('run_%02d'%run)/
                                'initials'/(stem+'.json')).read_text())
                            verification = compare_initial_scenes(reference_initial,initial)
                            write_json(folder/'initials'/(stem+'_comparison.json'),verification)
                            if not verification['match']:
                                raise RuntimeError('Initial scene mismatch: '+json.dumps(verification))
                        if args.reset_only:
                            record.update(status='probe',termination='probe')
                            return
                    if args.reference_output:
                        ref_trace = (args.reference_output / task / ('run_%02d' % run)
                                     / 'traces' / (stem + '.jsonl'))
                        with ref_trace.open() as reference_stream:
                            reference_start = json.loads(reference_stream.readline())['location_stats']
                        actual_start = capture.details['location_stats']
                        fields = ('pos', 'biome_id', 'pitch', 'yaw')
                        record['reference_start_match'] = all(actual_start.get(k) == reference_start.get(k)
                                                              for k in fields)
                        record['initial_location'] = actual_start
                        record['reference_initial_location'] = reference_start
                    state = None  # Never carry recurrent state across episodes.
                    writer = Mp4Writer(video_path, args.fps, imageio_ffmpeg.get_ffmpeg_exe())
                    policy_heatmaps = [obs['heatmap'].copy()]
                    native_heatmaps = [np.clip(capture.heatmap.squeeze()*255, 0, 255).astype(np.uint8)]
                    writer.append_data(heatmap_frame(capture.frame, capture.heatmap, obs['heatmap'], task, 0, 'reset', False, args.heatmap_mode))
                    total_reward = 0.0
                    with trace_path.open('w') as trace:
                        trace.write(json.dumps(dict(step=0, **capture.details)) + '\n')
                        for step in range(1, max_steps + 1):
                            batch = {k: np.stack([v]) for k, v in obs.items()}
                            result, state = agent(batch, np.array([step == 1]), state, training=False)
                            action = result['action'][0].detach().cpu().numpy()
                            search_log = {k: convert(v[0]) for k, v in result.items()
                                          if k.startswith(('log_search_', 'log_b_'))}
                            search_actions += int(search_log.get('log_search_override', 0))
                            search_decisions += int(search_log.get('log_search_decision', 0))
                            obs, reward, done, info = env.step({'action': action})
                            action_name = action_names[int(action.argmax())]
                            action_counts[action_name] = action_counts.get(action_name,0)+1
                            location = capture.details['location_stats']
                            pos = location['pos']
                            visited_cells.add((int(np.floor(pos[0]/2)),int(np.floor(pos[2]/2))))
                            pitch_up_steps += int(float(location['pitch'][0]) < -45)
                            if args.a_control == 'off' and (search_actions or search_decisions):
                                raise RuntimeError('Unexpected A intervention during control-off evaluation')
                            validate_policy_heatmap(obs, args.heatmap_mode)
                            policy_heatmaps.append(obs['heatmap'].copy())
                            native_heatmaps.append(np.clip(capture.heatmap.squeeze()*255, 0, 255).astype(np.uint8))
                            writer.append_data(heatmap_frame(capture.frame, capture.heatmap, obs['heatmap'],
                                task, step, action_names[int(action.argmax())], bool(info.get('success', False)), args.heatmap_mode))
                            total_reward += float(reward)
                            record['steps'] = step
                            record['return_'] = total_reward
                            success = bool(info.get('success', False))
                            if success and record['first_success_step'] is None:
                                record['first_success_step'] = step  # 1-based executed action count.
                            record['success'] = record['success'] or success
                            trace.write(json.dumps(dict(step=step, action_index=int(action.argmax()),
                                action=action_names[int(action.argmax())], reward=float(reward),
                                success=success, done=bool(done), info=convert(info), **search_log, **capture.details)) + '\n')
                            if 'error' in info:
                                raise RuntimeError('Simulator error: ' + str(info['error']))
                            if step % 100 == 0:
                                trace.flush()
                                print('%s run=%d episode=%d step=%d' % (task, run, episode, step), flush=True)
                            if done:
                                break
                    writer.close()
                    writer = None
                    np.savez_compressed(heatmap_path, policy_heatmap=np.stack(policy_heatmaps),
                                        unet_heatmap_uint8=np.stack(native_heatmaps))
                    record['status'] = 'completed'
                    record['termination'] = ('success' if record['success'] else
                        'death' if info.get('real_done', False) else
                        'time_limit' if step >= max_steps else 'environment_done')
                except BaseException:
                    record['error'] = traceback.format_exc()
                    record['termination'] = 'error'
                    raise
                finally:
                    if writer is not None:
                        writer.close()
                    record['return'] = record.pop('return_')
                    record['wall_seconds'] = time.monotonic() - started
                    record['search_actions'] = search_actions
                    record['search_decisions'] = search_decisions
                    record['search_rate'] = search_actions / max(1, record['steps'])
                    record['action_counts'] = action_counts
                    record['visited_cells_2m'] = len(visited_cells)
                    record['pitch_up_over_45_rate'] = pitch_up_steps / max(1,record['steps'])
                    write_json(record_path, record)
                    summarize(args.output, manifest)
                print(json.dumps(record), flush=True)
                if not fixed and record['termination'] == 'death' and episode + 1 < args.episodes:
                    record_recovery(episode, 'restart Minecraft after death; completed failure retained')
                    restart_bridge(env)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT / 'ckpt/latest')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation_results/author_5tasks_20episodes')
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    parser.add_argument('--heatmap-mode', choices=['unet', 'zero'], default='unet',
                        help='zero skips U-Net generation and supplies zero heatmaps to the existing checkpoint')
    parser.add_argument('--reference-output', type=Path,
                        help='Existing evaluation root whose common config, seeds and task specs must match')
    parser.add_argument('--a-control',choices=['checkpoint','off','repaired'],default='checkpoint',
                        help='off disables A control; repaired applies v3 control to the same strictly loaded weights')
    parser.add_argument('--c-control', choices=['checkpoint', 'off', 'no_predictions'],
                        default='checkpoint',
                        help='off keeps A fixed selector; no_predictions removes C prediction inputs only')
    parser.add_argument('--episode-order',type=int,nargs='+',
                        help='Permutation of all episode indices for a fixed-world A-off diagnosis')
    parser.add_argument('--paired-checkpoint-root', type=Path,
                        help='Alternate original checkpoint-root and this checkpoint on each fixed new world')
    parser.add_argument('--world-seed-start', type=int,
                        help='Positive nonzero Minecraft world seed for episode zero')
    parser.add_argument('--probe-only', action='store_true',
                        help='Validate two original/A scene pairs without scored evaluation')
    parser.add_argument('--worker-episode', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--reset-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=20, help='Episodes PER run, per task')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=20, help='Minecraft runs at 20 ticks/s, repeat=1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--organize-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--worker-task', choices=list(TASKS), help=argparse.SUPPRESS)
    parser.add_argument('--worker-run', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    if args.paired_checkpoint_root:
        args.paired_checkpoint_root = args.paired_checkpoint_root.resolve()
        if args.world_seed_start is None or args.world_seed_start <= 0:
            parser.error('Paired evaluation requires a positive --world-seed-start')
        if len(args.tasks) != 1 or args.runs != 1 or args.reference_output:
            parser.error('Paired mode requires one task, runs=1, and no legacy reference-output')
    if args.world_seed_start is not None:
        if not 0 < args.world_seed_start < 2**63 - 100000:
            parser.error('world-seed-start must be a positive signed 64-bit seed')
        if not 0 < args.seed < 2**31 - args.episodes - 1001:
            parser.error('Fixed evaluation requires a positive policy seed below signed 32-bit maximum')
        if not args.paired_checkpoint_root and args.worker_episode is None:
            if not (has_control_override(args) and args.reference_output):
                parser.error('Fixed-world evaluation requires paired mode or an A-off reference run')
    if args.worker_episode is not None and not 0 <= args.worker_episode < args.episodes:
        parser.error('worker-episode out of range')
    if has_control_override(args) and args.paired_checkpoint_root:
        parser.error('A-off diagnosis reuses reference-output; do not pass paired-checkpoint-root')
    if has_control_override(args) and (args.world_seed_start is None or args.reference_output is None):
        parser.error('A-off diagnosis requires fixed-world-seed and reference-output')
    if has_control_override(args) and (len(args.tasks) != 1 or args.runs != 1):
        parser.error('A-off diagnosis requires one task and runs=1')
    if args.episode_order is not None:
        if not has_control_override(args) or sorted(args.episode_order) != list(range(args.episodes)):
            parser.error('episode-order must be a permutation of all A-off episodes')
    if args.probe_only and not args.paired_checkpoint_root:
        parser.error('probe-only requires paired mode')
    if args.reset_only and args.worker_episode is None:
        parser.error('reset-only requires a single episode worker')
    if args.reference_output:
        args.reference_output = args.reference_output.resolve()
        if args.output == args.reference_output:
            parser.error('A reference evaluation must not be overwritten')
    if min(args.runs, args.episodes, args.fps, args.threads) <= 0:
        parser.error('runs, episodes, fps and threads must be positive')
    if args.summarize_only:
        manifest = json.loads((args.output / 'manifest.json').read_text())
        print(json.dumps(summarize(args.output, manifest), indent=2))
        return
    if args.paired_checkpoint_root:
        paired_main(args)
        return
    organize(args.checkpoint_root, args.tasks)
    if args.organize_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(tasks=args.tasks, runs=args.runs, episodes=args.episodes, seed=args.seed,
                    fps=args.fps, device=args.device, checkpoint_root=str(args.checkpoint_root),
                    task_specs=(ROOT / 'envs/tasks/task_specs.yaml').read_text(),
                    model_config=(ROOT / 'configs.yaml').read_text(), protocol_version=2, video_layout='rgb_native-unet_policy-overlay',
                    heatmap_storage='uint8 npz: policy_heatmap and unet_heatmap_uint8; frame 0 is reset')
    if args.heatmap_mode == 'zero':
        manifest.update(protocol_version=3, heatmap_mode='zero', unet_loaded=False,
                        video_layout='rgb_zero-heatmap_rgb',
                        ablation='pretrained checkpoint evaluated with zero heatmap; no retraining')
    if args.reference_output:
        manifest.update(reference_output=str(args.reference_output),
                        checkpoint_config='auto search_a/search_b from checkpoint; restore training step',
                        search_diagnostics='per-action trace log_search_* and episode search_rate')
    if args.world_seed_start is not None:
        manifest.update(protocol_version=4,world_seed_start=args.world_seed_start,
            fresh_process_per_rollout=True,fast_reset=None,mission_seed='per-episode',
            initial_audit='location, orientation, life/inventory, local terrain hash; not policy inputs',
            fixed_time=6000,allow_time_passage=False,weather='clear')
    if args.a_control != 'checkpoint':
        manifest.update(a_control=args.a_control,ablation='Same A weights, explicit online control override: ' + args.a_control)
    if args.c_control != 'checkpoint':
        manifest.update(c_control=args.c_control,
                        ablation='Same checkpoint, explicit C runtime override: ' + args.c_control)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Output has different evaluation settings; use another --output directory')
    else:
        write_json(manifest_path, manifest)
    if args.worker_task:
        worker(args, manifest)
        return
    if args.world_seed_start is not None:
        fixed_reference_main(args,manifest)
        return
    import fcntl
    with (args.output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = []
        for task in args.tasks:
            for run in range(args.runs):
                folder = args.output / task / ('run_%02d' % run)
                folder.mkdir(parents=True, exist_ok=True)
                completed = list((folder / 'episodes').glob('episode_*.json'))
                if len(completed) == args.episodes and all(json.loads(p.read_text())['status'] == 'completed' for p in completed):
                    continue
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                           '--checkpoint-root', str(args.checkpoint_root), '--output', str(args.output),
                           '--tasks', *args.tasks, '--runs', str(args.runs), '--episodes', str(args.episodes),
                           '--seed', str(args.seed), '--fps', str(args.fps), '--device', args.device,
                           '--threads', str(args.threads), '--heatmap-mode', args.heatmap_mode, '--worker-task', task, '--worker-run', str(run)]
                if args.reference_output:
                    command += ['--reference-output', str(args.reference_output)]
                print('Evaluating %s group %d/%d' % (task, run + 1, args.runs), flush=True)
                write_json(args.output / 'status.json', dict(status='running', pid=os.getpid(), task=task, run=run))
                with (folder / 'console.log').open('a') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                summarize(args.output, manifest)
                if result.returncode:
                    failures.append(dict(task=task, run=run, returncode=result.returncode))
                    # Stop on infrastructure failures; resuming retries unfinished episodes.
                    write_json(args.output / 'status.json', dict(status='failed', failures=failures))
                    raise RuntimeError('Worker failed; inspect ' + str(folder / 'console.log'))
        summary = summarize(args.output, manifest)
        complete = all(r['complete'] for r in summary.values())
        write_json(args.output / 'status.json', dict(status='completed' if complete else 'incomplete', failures=failures))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
