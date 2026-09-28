#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global-mean heatmap ablation evaluation for a trained LS-Imagine checkpoint.

This changes ONLY the heatmap seen by the trained agent.
Natural Zoom, NACLIP-P85, score, score_on_zoomed and ScoreStorage are
computed inside the original environment BEFORE this outer wrapper.

Modes:
  normal      : original heatmap
  frame_mean  : each frame -> constant map equal to that frame mean
  global_mean : every frame -> ONE constant map equal to the global mean
                computed from all train_eps heatmaps

Interpretation:
  normal >> frame_mean and fixed_shuffle
      => spatial heatmap information is genuinely useful.
  normal ~= frame_mean ~= fixed_shuffle
      => the policy/world model barely uses spatial heatmap structure.
  zero worse but frame_mean/shuffle similar to normal
      => do NOT claim spatial heatmap usefulness; zero may simply be OOD.

Put this file in:
  <LS-Imagine>/relevance_map/heatmap_ablation_eval.py

Edit CHECKPOINT_PATH and REPLAY_DIR, then run directly in PyCharm.
"""

from pathlib import Path
from types import SimpleNamespace
import collections
import copy
import functools
import math
import os
import sys
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml


# =========================
# CONFIG -- EDIT THIS
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

# Can be a latest.pt file OR its containing directory.
CHECKPOINT_PATH = Path(
    r"/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/"
    r"minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt"
)

REPLAY_DIR = Path(
    r"/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/"
    r"minedojo_harvest_log_in_plains/seed_0/20260922T091826/train_eps"
)

TASK = "minedojo_harvest_log_in_plains"

# Quick diagnosis: 20. Paper-quality: preferably 50~100 per mode.
EVAL_EPISODES = 50

MODES = [
    "normal",
    "frame_mean",
    "global_mean",
]

SEED = 0
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "global_mean_ablation_outputs"
)

# None = use the compile setting from your local configs.yaml.
# If checkpoint loading reports _orig_mod key mismatch, set this to the
# same compile setting that was used during training.
FORCE_COMPILE = None


def recursive_update(base, update):
    for k, v in update.items():
        if (
            isinstance(v, dict)
            and k in base
            and isinstance(base[k], dict)
        ):
            recursive_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def load_config():
    configs = yaml.safe_load(
        (PROJECT_ROOT / "configs.yaml").read_text(
            encoding="utf-8"
        )
    )

    merged = {}
    recursive_update(merged, configs["defaults"])
    recursive_update(merged, configs["minedojo"])

    merged["task"] = TASK
    merged["parallel"] = False
    merged["envs"] = 1
    merged["eval_episode_num"] = EVAL_EPISODES
    merged["video_pred_log"] = False
    merged["use_wandb"] = False

    if FORCE_COMPILE is not None:
        merged["compile"] = bool(FORCE_COMPILE)

    return SimpleNamespace(**merged)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}\n"
            "Edit CHECKPOINT_PATH."
        )
    return path


def wilson(success, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = success / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    ) / d
    return c - h, c + h


def compute_global_heatmap_mean(replay_dir):
    """Compute one mean over ALL pixels of ALL ep['heatmap'] arrays."""
    replay_dir = Path(replay_dir).expanduser().resolve()
    if not replay_dir.exists():
        raise FileNotFoundError(f"Replay directory not found:\n{replay_dir}")

    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found in:\n{replay_dir}")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    total_frames = 0
    valid_files = 0
    gmin = np.inf
    gmax = -np.inf
    first_dtype = None

    print("=" * 100)
    print("COMPUTING GLOBAL MEAN FROM TRAIN REPLAY")
    print("=" * 100)
    print("replay dir:", replay_dir)
    print("npz files :", len(files))

    for i, path in enumerate(files):
        with np.load(path, allow_pickle=False) as ep:
            if "heatmap" not in ep.files:
                continue
            h = np.asarray(ep["heatmap"])
            if first_dtype is None:
                first_dtype = h.dtype
            x = h.astype(np.float64, copy=False)
            total_sum += float(x.sum())
            total_sq_sum += float(np.square(x).sum())
            total_pixels += int(x.size)
            total_frames += int(x.shape[0]) if x.ndim >= 3 else 1
            gmin = min(gmin, float(x.min()))
            gmax = max(gmax, float(x.max()))
            valid_files += 1

        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"[{i+1}/{len(files)}] frames={total_frames:,}, pixels={total_pixels:,}")

    if total_pixels == 0:
        raise RuntimeError("No valid heatmap pixels found")

    mean_raw = total_sum / total_pixels
    var_raw = max(0.0, total_sq_sum / total_pixels - mean_raw ** 2)
    std_raw = math.sqrt(var_raw)
    scale = 255.0 if gmax > 1.5 else 1.0

    stats = {
        "valid_files": valid_files,
        "frames": total_frames,
        "pixels": total_pixels,
        "dtype": str(first_dtype),
        "min_raw": gmin,
        "max_raw": gmax,
        "global_mean_raw": mean_raw,
        "global_std_raw": std_raw,
        "global_mean_normalized": mean_raw / scale,
        "global_std_normalized": std_raw / scale,
    }

    print("GLOBAL_MEAN raw       =", mean_raw)
    print("GLOBAL_MEAN normalized=", mean_raw / scale)
    print("GLOBAL_STD raw        =", std_raw)
    print("=" * 100)
    return mean_raw, stats


class NullLogger:
    def __init__(self):
        self.step = 0

    def scalar(self, *args, **kwargs):
        pass

    def image(self, *args, **kwargs):
        pass

    def video(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def offline_scalar(self, *args, **kwargs):
        pass

    def offline_video(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class HeatmapAblationWrapper(gym.Wrapper):
    VALID = {"normal", "frame_mean", "global_mean"}

    def __init__(self, env, mode, global_mean_raw):
        super().__init__(env)
        if mode not in self.VALID:
            raise ValueError(mode)
        self.mode = mode
        self.global_mean_raw = float(global_mean_raw)
        self.transform_calls = 0
        self.changed_calls = 0

    @staticmethod
    def _safe_scalar(x, value):
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            return int(np.clip(np.rint(value), info.min, info.max))
        return float(value)

    def transform_heatmap(self, value):
        if value is None:
            return None
        x = np.asarray(value)
        if self.mode == "normal":
            return x.copy()
        if x.shape[0] != 64 or x.shape[1] != 64:
            raise ValueError(f"Expected 64x64 heatmap, got {x.shape}")

        self.transform_calls += 1
        if self.mode == "frame_mean":
            m = self._safe_scalar(x, float(np.mean(x, dtype=np.float64)))
        elif self.mode == "global_mean":
            m = self._safe_scalar(x, self.global_mean_raw)
        else:
            raise RuntimeError(self.mode)

        y = np.full_like(x, m)
        if not np.array_equal(x, y):
            self.changed_calls += 1
        return y

    def transform_obs(self, obs):
        obs = dict(obs)
        if "heatmap" in obs:
            obs["heatmap"] = self.transform_heatmap(obs["heatmap"])
        if "heatmap_on_zoomed" in obs:
            obs["heatmap_on_zoomed"] = self.transform_heatmap(obs["heatmap_on_zoomed"])
        return obs

    def reset(self, **kwargs):
        return self.transform_obs(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.transform_obs(obs), reward, done, info

def prepare_task(config):
    """
    Mirror the important task setup done by the repo's test.py.
    """
    from envs.tasks import get_specs

    suite, task = config.task.split("_", 1)
    assert suite == "minedojo"

    task_id, task_specs, sim_specs = get_specs(
        task,
        target_item=config.target_item,
    )

    config.episode_max_steps = (
        task_specs["terminal_specs"]["max_steps"]
    )

    if "concentration_specs" in task_specs:
        c = task_specs["concentration_specs"]
        c["max_steps"] = config.episode_max_steps
        c["gaussian_reward_weight"] = (
            config.gaussian_reward_weight
        )
        c["gaussian_sigma_weight"] = (
            config.gaussian_sigma_weight
        )

    if (
        "clip_specs" in task_specs
        and "success_specs" in task_specs
    ):
        s = task_specs["success_specs"]
        if "all" in s:
            item = s["all"]["item"]["type"]
        else:
            item = s["any"]["item"]["type"]
        task_specs["clip_specs"]["target_object"] = item


def make_env_for_mode(config, mode, global_mean_raw):
    from expr import make_env

    # Full original environment first.
    env = make_env(config, "eval", 0)

    # Then alter only the final observation shown to the agent.
    return HeatmapAblationWrapper(env, mode, global_mean_raw)


def summarize_eps(directory, max_steps):
    rows = []

    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as ep:
            reward = np.asarray(ep["reward"]).reshape(-1)
            length = len(reward) - 1

            if "success" in ep:
                suc_arr = (
                    np.asarray(ep["success"])
                    .reshape(-1)
                    .astype(bool)
                )
                success = bool(
                    np.any(suc_arr[: max_steps + 1])
                )
            else:
                success = bool(
                    np.sum(reward[: max_steps + 1]) > 0
                )

            if "first_success_step" in ep:
                fss = (
                    np.asarray(ep["first_success_step"])
                    .reshape(-1)
                )
                first_success = float(
                    min(fss[-1], max_steps)
                )
            elif success and "success" in ep:
                idx = np.flatnonzero(suc_arr)
                first_success = (
                    float(idx[0])
                    if len(idx)
                    else float(max_steps)
                )
            else:
                first_success = float(max_steps)

            rows.append({
                "file": path.name,
                "success": int(success),
                "first_success_step": first_success,
                "length": int(length),
                "return": float(
                    np.sum(reward[: max_steps + 1])
                ),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No eval episodes saved in {directory}"
        )

    n = len(df)
    s = int(df["success"].sum())
    lo, hi = wilson(s, n)

    success_df = df[df["success"] == 1]

    summary = {
        "episodes": n,
        "successes": s,
        "success_rate": s / n,
        "success_ci95_low": lo,
        "success_ci95_high": hi,
        "mean_first_success_all":
            float(df["first_success_step"].mean()),
        "mean_first_success_success_only":
            float(success_df["first_success_step"].mean())
            if len(success_df) else np.nan,
        "median_first_success_success_only":
            float(success_df["first_success_step"].median())
            if len(success_df) else np.nan,
        "mean_length": float(df["length"].mean()),
        "mean_return": float(df["return"].mean()),
    }

    return df, summary


def main():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    import tools
    from expr import LS_Imagine
    from parallel import Damy

    checkpoint_path = resolve_checkpoint(
        CHECKPOINT_PATH
    )

    global_mean_raw, global_stats = compute_global_heatmap_mean(REPLAY_DIR)

    run_dir = (
        OUTPUT_ROOT
        / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([global_stats]).to_csv(
        run_dir / "global_heatmap_stats.csv", index=False, encoding="utf-8-sig"
    )

    config = load_config()
    config.results_dir = str(run_dir / "_env_logs")
    config.logdir = run_dir
    config.evaldir = run_dir / "_eval"
    config.name = "global_mean_ablation"

    prepare_task(config)

    print("=" * 100)
    print("GLOBAL-MEAN HEATMAP ABLATION")
    print("=" * 100)
    print("checkpoint:", checkpoint_path)
    print("task:", config.task)
    print("replay dir:", Path(REPLAY_DIR).resolve())
    print("global mean raw:", global_mean_raw)
    print("global mean normalized:", global_stats["global_mean_normalized"])
    print("episodes/mode:", EVAL_EPISODES)
    print("modes:", MODES)
    print("compile:", config.compile)
    print("output:", run_dir)

    # Build reference env to define spaces.
    tools.set_seed_everywhere(SEED)
    ref_raw = make_env_for_mode(config, "normal", global_mean_raw)
    ref_env = Damy(ref_raw)

    config.num_actions = (
        ref_env.action_space.n
        if hasattr(ref_env.action_space, "n")
        else ref_env.action_space.shape[0]
    )

    agent = LS_Imagine(
        ref_env.observation_space,
        ref_env.action_space,
        config,
        NullLogger(),
        dataset=None,
    ).to(config.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
    )
    agent.load_state_dict(
        checkpoint["agent_state_dict"],
        strict=True,
    )
    agent.requires_grad_(False)
    agent.eval()

    try:
        ref_env.close()
    except Exception:
        pass

    print("[PASS] checkpoint loaded")

    summaries = []

    for mode in MODES:
        print()
        print("-" * 100)
        print("MODE:", mode)

        # Try to make each mode start from the same RNG state.
        tools.set_seed_everywhere(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        config.name = f"global_mean_ablation_{mode}"

        raw_env = make_env_for_mode(config, mode, global_mean_raw)
        ablation_wrapper = raw_env
        env = Damy(raw_env)

        eps_dir = run_dir / mode / "episodes"
        eps_dir.mkdir(parents=True, exist_ok=True)

        cache = collections.OrderedDict()
        score_storage = tools.ScoreStorage(
            max_steps=config.episode_max_steps
        )

        policy = functools.partial(
            agent,
            training=False,
        )

        with torch.no_grad():
            tools.simulate(
                policy,
                [env],
                cache,
                eps_dir,
                NullLogger(),
                score_storage,
                config.episode_max_steps,
                config.discount,
                is_eval=True,
                episodes=EVAL_EPISODES,
                is_training=False,
            )

        ep_df, summary = summarize_eps(
            eps_dir,
            config.episode_max_steps,
        )

        summary["mode"] = mode
        summary["global_mean_raw"] = global_mean_raw
        summary["global_mean_normalized"] = global_stats["global_mean_normalized"]
        summary["heatmap_transform_calls"] = int(
            ablation_wrapper.transform_calls
        )
        summary["heatmap_changed_calls"] = int(
            ablation_wrapper.changed_calls
        )

        ep_df.to_csv(
            run_dir / mode / "episodes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summaries.append(summary)

        print(
            f"success: {summary['successes']}/"
            f"{summary['episodes']} = "
            f"{summary['success_rate']:.3f}"
        )
        print(
            "95% CI: "
            f"[{summary['success_ci95_low']:.3f}, "
            f"{summary['success_ci95_high']:.3f}]"
        )
        print(
            "first_success(success-only): "
            f"mean={summary['mean_first_success_success_only']:.1f}, "
            f"median={summary['median_first_success_success_only']:.1f}"
        )
        print(
            "changed heatmap calls:",
            summary["heatmap_changed_calls"],
        )

        try:
            env.close()
        except Exception:
            pass

    summary_df = pd.DataFrame(summaries)
    cols = ["mode"] + [
        c for c in summary_df.columns
        if c != "mode"
    ]
    summary_df = summary_df[cols]

    summary_df.to_csv(
        run_dir / "ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    text = [
        "Global-Mean Heatmap Ablation Summary",
        "=" * 72,
        f"checkpoint: {checkpoint_path}",
        f"episodes/mode: {EVAL_EPISODES}",
        "",
        summary_df[
            [
                "mode",
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "mean_first_success_success_only",
                "median_first_success_success_only",
            ]
        ].to_string(index=False),
        "",
        "Interpretation:",
        f"global_mean_raw = {global_mean_raw:.8f}",
        f"global_mean_normalized = {global_stats['global_mean_normalized']:.8f}",
        "",
        "1) normal ~= frame_mean ~= global_mean:",
        "   meaningful heatmap content is probably not needed at inference;",
        "   keeping the heatmap channel in-distribution may be enough.",
        "2) normal ~= frame_mean, but frame_mean >> global_mean:",
        "   per-frame average relevance is useful, spatial structure is not.",
        "3) normal >> frame_mean:",
        "   spatial heatmap structure matters.",
        "",
        "Reminder: this is inference-time ablation only.",
    ]

    report = "\n".join(text)
    (run_dir / "00_READ_ME_FIRST.txt").write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(report)
    print()
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
