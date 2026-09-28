#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heatmap ablation evaluation for a trained LS-Imagine checkpoint.

This changes ONLY the heatmap seen by the trained agent.
Natural Zoom, NACLIP-P85, score, score_on_zoomed and ScoreStorage are
computed inside the original environment BEFORE this outer wrapper.

Modes:
  normal       : original heatmap
  frame_mean   : each 64x64 heatmap -> constant equal to its own mean
  fixed_shuffle: fixed spatial permutation; exact histogram preserved
  zero         : all zeros (strong/OOD control)

Interpretation:
  normal >> frame_mean and fixed_shuffle
      => spatial heatmap information is genuinely useful.
  normal ~= frame_mean ~= fixed_shuffle
      => the policy/world model barely uses spatial heatmap structure.
  zero worse but frame_mean/shuffle similar to normal
      => do NOT claim spatial heatmap usefulness; zero may simply be OOD.

Put this file in:
  <LS-Imagine>/relevance_map/heatmap_ablation_eval.py

Edit CHECKPOINT_PATH, then run directly in PyCharm.
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
    r"/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt"
)

TASK = "minedojo_harvest_log_in_plains"

# Quick diagnosis: 20. Paper-quality: preferably 50~100 per mode.
EVAL_EPISODES = 100

MODES = [
    "normal",
    "frame_mean",
    "fixed_shuffle",
    "zero",
]

SEED = 0
SHUFFLE_SEED = 20260927

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "heatmap_ablation_outputs"
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
    VALID = {
        "normal",
        "frame_mean",
        "fixed_shuffle",
        "zero",
    }

    def __init__(self, env, mode):
        super().__init__(env)
        if mode not in self.VALID:
            raise ValueError(mode)

        self.mode = mode
        rng = np.random.RandomState(SHUFFLE_SEED)
        self.perm = rng.permutation(64 * 64)

        self.transform_calls = 0
        self.changed_calls = 0

    def transform_heatmap(self, value):
        if value is None:
            return None

        x = np.asarray(value)

        if self.mode == "normal":
            return x.copy()

        if x.shape[0] != 64 or x.shape[1] != 64:
            raise ValueError(
                f"Expected 64x64 heatmap, got {x.shape}"
            )

        self.transform_calls += 1

        if self.mode == "zero":
            y = np.zeros_like(x)

        elif self.mode == "frame_mean":
            m = float(np.mean(x, dtype=np.float64))
            if np.issubdtype(x.dtype, np.integer):
                m = int(np.rint(m))
            y = np.full_like(x, m)

        elif self.mode == "fixed_shuffle":
            if x.ndim == 2:
                y = x.reshape(-1)[self.perm].reshape(x.shape)
            elif x.ndim == 3:
                h, w, c = x.shape
                y = (
                    x.reshape(h * w, c)[self.perm]
                    .reshape(x.shape)
                )
            else:
                raise ValueError(x.shape)

        else:
            raise RuntimeError(self.mode)

        if not np.array_equal(x, y):
            self.changed_calls += 1

        return y

    def transform_obs(self, obs):
        obs = dict(obs)

        if "heatmap" in obs:
            obs["heatmap"] = self.transform_heatmap(
                obs["heatmap"]
            )

        # Not needed by normal online policy inference, but keep saved
        # evaluation replay internally consistent.
        if "heatmap_on_zoomed" in obs:
            obs["heatmap_on_zoomed"] = self.transform_heatmap(
                obs["heatmap_on_zoomed"]
            )

        return obs

    def reset(self, **kwargs):
        return self.transform_obs(
            self.env.reset(**kwargs)
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (
            self.transform_obs(obs),
            reward,
            done,
            info,
        )


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


def make_env_for_mode(config, mode):
    from expr import make_env

    # Full original environment first.
    env = make_env(config, "eval", 0)

    # Then alter only the final observation shown to the agent.
    return HeatmapAblationWrapper(env, mode)


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

    run_dir = (
        OUTPUT_ROOT
        / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config.results_dir = str(run_dir / "_env_logs")
    config.logdir = run_dir
    config.evaldir = run_dir / "_eval"
    config.name = "heatmap_ablation"

    prepare_task(config)

    print("=" * 100)
    print("HEATMAP ABLATION")
    print("=" * 100)
    print("checkpoint:", checkpoint_path)
    print("task:", config.task)
    print("episodes/mode:", EVAL_EPISODES)
    print("modes:", MODES)
    print("compile:", config.compile)
    print("output:", run_dir)

    # Build reference env to define spaces.
    tools.set_seed_everywhere(SEED)
    ref_raw = make_env_for_mode(config, "normal")
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

        config.name = f"heatmap_ablation_{mode}"

        raw_env = make_env_for_mode(config, mode)
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
        "Heatmap Ablation Summary",
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
        "1) normal >> frame_mean and fixed_shuffle:",
        "   spatial heatmap information matters.",
        "2) normal ~= frame_mean ~= fixed_shuffle:",
        "   spatial heatmap contributes little.",
        "3) zero alone is worse:",
        "   zero may be OOD; do not treat that alone as proof.",
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
