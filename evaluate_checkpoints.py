#!/usr/bin/env python3
"""Evaluate the five released LS-Imagine agents; real RGB MP4s, no training."""
import argparse
import copy
import csv
import json
import os
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


def organize(checkpoint_root):
    """Rename on the same filesystem; never overwrite an existing checkpoint."""
    plan = []
    base = checkpoint_root.resolve()
    for task, alias in TASKS.items():
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
              'return', 'first_success_step', 'termination', 'wall_seconds', 'video', 'trace', 'heatmaps', 'heatmap_mode']
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
    for name in ('episodes', 'videos', 'traces', 'heatmaps'):
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
    checkpoint_path = args.checkpoint_root / task / 'latest.pt'
    write_json(folder / 'settings.json', dict(config=vars(config), task_specs=task_specs, sim_specs=sim_specs,
               checkpoint=str(checkpoint_path), max_steps=max_steps, heatmap_mode=args.heatmap_mode,
               seed_note='Group seed and per-episode simulator/policy RNG seeds; world generation follows task YAML.'))
    env = None
    try:
        with torch.no_grad():
            env = make_minedojo(task_id, task_specs, sim_specs)
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
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # torch.compile checkpoints add this component to module names.
            state_dict = {k.replace('._orig_mod.', '.'): v
                          for k, v in checkpoint['agent_state_dict'].items()}
            if len(state_dict) != len(checkpoint['agent_state_dict']):
                raise ValueError('Checkpoint key collision')
            agent.load_state_dict(state_dict, strict=True)
            del checkpoint, state_dict
            agent.eval().requires_grad_(False)
            print('Loaded checkpoint strictly:', checkpoint_path, flush=True)
            for episode in range(args.episodes):
                stem = 'episode_%03d' % episode
                record_path = folder / 'episodes' / (stem + '.json')
                if record_path.exists() and json.loads(record_path.read_text())['status'] == 'completed':
                    print('Resume: skipping', record_path, flush=True)
                    continue
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
                try:
                    obs = env.reset()
                    validate_policy_heatmap(obs, args.heatmap_mode)
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
                            obs, reward, done, info = env.step({'action': action})
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
                                success=success, done=bool(done), info=convert(info), **capture.details)) + '\n')
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
                    write_json(record_path, record)
                    summarize(args.output, manifest)
                print(json.dumps(record), flush=True)
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
    if min(args.runs, args.episodes, args.fps, args.threads) <= 0:
        parser.error('runs, episodes, fps and threads must be positive')
    if args.summarize_only:
        manifest = json.loads((args.output / 'manifest.json').read_text())
        print(json.dumps(summarize(args.output, manifest), indent=2))
        return
    organize(args.checkpoint_root)
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
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Output has different evaluation settings; use another --output directory')
    else:
        write_json(manifest_path, manifest)
    if args.worker_task:
        worker(args, manifest)
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
