#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone coarse_8x8 visualization / smoke test for LS-Imagine.

What it verifies
----------------
1. Natural Zoom still owns the FULL spatial Value heatmap.
2. The World-Model heatmap is:
       full spatial map
       -> 8x8 regional average
       -> nearest/block expand back to original HxW
3. Global mean is preserved.
4. Every coarse cell is spatially constant.
5. Real MineDojo frames are saved as:
       RGB
       Full spatial heatmap
       Native 8x8 values
       Expanded coarse_8x8 heatmap
       RGB + coarse overlay

No checkpoint is needed. This uses the current environment + MineCLIP pipeline.

Usage
-----
Put this file under:
    <LS-Imagine>/relevance_map/test_coarse_8x8_heatmap.py

Then:
    cd <LS-Imagine>
    MINEDOJO_HEADLESS=1 python relevance_map/test_coarse_8x8_heatmap.py

No argparse. Edit globals below if needed.
"""

from pathlib import Path
from types import SimpleNamespace
import copy
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml


# ============================================================
# CONFIG
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

TASK = "minedojo_harvest_log_in_plains"

# Number of random environment steps.
NUM_STEPS = 60

# Save one visualization every N steps, plus reset frame.
SAVE_EVERY = 5

SEED = 0

OUTPUT_DIR = (
    PROJECT_ROOT
    / "relevance_map"
    / "coarse_8x8_visual_test"
)


# ============================================================
# Config helpers
# ============================================================

def recursive_update(base, update):
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


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
    merged["video_pred_log"] = False
    merged["use_wandb"] = False

    return SimpleNamespace(**merged)


def prepare_task_config(config):
    """
    Mirrors the task preparation already used by the successful
    heatmap-ablation scripts.
    """
    from envs.tasks import get_specs

    suite, task = config.task.split("_", 1)
    if suite != "minedojo":
        raise ValueError(f"Expected minedojo task, got {suite}")

    _, task_specs, _ = get_specs(
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
            item_type = s["all"]["item"]["type"]
        else:
            item_type = s["any"]["item"]["type"]

        task_specs["clip_specs"]["target_object"] = item_type


# ============================================================
# Wrapper lookup
# ============================================================

def find_wrapper(env, wanted_name):
    """
    Strict wrapper lookup.

    IMPORTANT:
    gym.Wrapper.__getattr__ delegates public attributes to inner wrappers.
    Therefore using getattr(cur, "wrapper_name", ...) can falsely make an
    OUTER wrapper look like an INNER wrapper.

    We only inspect:
      1) the current object's real class name
      2) wrapper_name stored in the current object's own __dict__
    """
    cur = env

    for _ in range(100):
        class_name = cur.__class__.__name__
        own_wrapper_name = cur.__dict__.get(
            "wrapper_name",
            None,
        )

        if (
            class_name == wanted_name
            or own_wrapper_name == wanted_name
        ):
            return cur

        if "env" not in cur.__dict__:
            break

        cur = cur.__dict__["env"]

    raise RuntimeError(
        f"Could not find wrapper: {wanted_name}"
    )


def print_wrapper_chain(env):
    """Print the actual wrapper chain without delegated attributes."""
    print("[INFO] Actual wrapper chain:")

    cur = env

    for depth in range(100):
        print(
            f"  {depth:02d}: "
            f"{cur.__class__.__module__}."
            f"{cur.__class__.__name__}"
        )

        if "env" not in cur.__dict__:
            break

        cur = cur.__dict__["env"]


# ============================================================
# Independent 8x8 reference implementation
# ============================================================

def independent_coarse_8x8(spatial_heatmap):
    """
    Independent reference implementation, intentionally not calling
    ConcentrationWrapper._to_world_model_heatmap().
    """
    x = np.asarray(
        spatial_heatmap,
        dtype=np.float32
    )

    if x.ndim == 3:
        assert x.shape[-1] == 1
        x2 = x[..., 0]
    else:
        x2 = x

    H, W = x2.shape

    assert H % 8 == 0, (H, W)
    assert W % 8 == 0, (H, W)

    bh = H // 8
    bw = W // 8

    coarse = (
        x2
        .reshape(8, bh, 8, bw)
        .mean(
            axis=(1, 3),
            dtype=np.float64,
        )
        .astype(np.float32)
    )

    expanded = np.repeat(
        np.repeat(
            coarse,
            bh,
            axis=0,
        ),
        bw,
        axis=1,
    ).astype(np.float32)

    return coarse, expanded[..., None]


def max_within_cell_std(expanded):
    x = np.asarray(expanded)[..., 0]

    H, W = x.shape
    bh = H // 8
    bw = W // 8

    max_std = 0.0

    for gy in range(8):
        for gx in range(8):
            cell = x[
                gy * bh:(gy + 1) * bh,
                gx * bw:(gx + 1) * bw,
            ]

            max_std = max(
                max_std,
                float(cell.std()),
            )

    return max_std


# ============================================================
# Safe random action for the ACTUAL LS-Imagine wrapper stack
# ============================================================

def sample_safe_random_action(
        outer_env,
        basic_action_names,
        rng,
):
    """
    Actual action path in expr.make_env():

        MinedojoLSImagineWrapper
            Discrete(len(BASIC_ACTIONS))
                ↓
        OneHotAction
            Box(shape=(N,))
                ↓
        SelectAction(key="action")
            expects {"action": one_hot}
                ↓
        UUID
                ↓
        RewardObs   <- outer_env

    Therefore this function MUST NOT use MineDojo's raw 8-D
    MultiDiscrete.no_op() interface.

    It samples only safe LS-Imagine discrete actions:
        noop
        forward
        turn_left
        turn_right

    Then converts the selected discrete index to the one-hot vector
    required by OneHotAction, and finally wraps it in {"action": ...}
    for SelectAction.
    """
    safe_names = (
        "noop",
        "forward",
        "turn_left",
        "turn_right",
    )

    # Mostly move forward, occasionally rotate or stop.
    probs = np.asarray(
        [0.15, 0.75, 0.05, 0.05],
        dtype=np.float64,
    )

    for name in safe_names:
        if name not in basic_action_names:
            raise RuntimeError(
                f"Missing BASIC_ACTION '{name}'. "
                f"Available: {basic_action_names}"
            )

    chosen_name = str(
        rng.choice(
            safe_names,
            p=probs,
        )
    )

    discrete_index = basic_action_names.index(
        chosen_name
    )

    # The outer advertised action space is inherited from OneHotAction:
    # Box(low=0, high=1, shape=(N,), dtype=float32)
    if not hasattr(
        outer_env.action_space,
        "shape",
    ):
        raise RuntimeError(
            "Expected outer action_space to be the Box produced by "
            f"OneHotAction, got: {outer_env.action_space}"
        )

    num_actions = int(
        outer_env.action_space.shape[0]
    )

    if num_actions != len(basic_action_names):
        raise RuntimeError(
            "OneHotAction dimension mismatch: "
            f"outer Box has {num_actions}, "
            f"BASIC_ACTIONS has {len(basic_action_names)}"
        )

    one_hot = np.zeros(
        (num_actions,),
        dtype=np.float32,
    )
    one_hot[discrete_index] = 1.0

    # SelectAction.step() will extract action["action"].
    action = {
        "action": one_hot
    }

    return (
        action,
        chosen_name,
        discrete_index,
    )


# ============================================================
# Visualization
# ============================================================

def save_visualization(
        step,
        rgb,
        spatial,
        coarse_grid,
        coarse_expanded,
        score,
        is_zoomed,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb = np.asarray(rgb)
    spatial2 = np.asarray(spatial)[..., 0]
    coarse2 = np.asarray(coarse_expanded)[..., 0]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(19, 4.2),
    )

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(
        spatial2,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title(
        "Full spatial\n"
        f"mean={spatial2.mean():.4f}, "
        f"std={spatial2.std():.4f}"
    )

    im = axes[2].imshow(
        coarse_grid,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[2].set_title(
        "Native 8x8\n"
        f"std={coarse_grid.std():.4f}"
    )

    # Draw native 8x8 grid.
    axes[2].set_xticks(
        np.arange(-0.5, 8, 1),
        minor=True,
    )
    axes[2].set_yticks(
        np.arange(-0.5, 8, 1),
        minor=True,
    )
    axes[2].grid(
        which="minor",
        linewidth=0.5,
    )

    axes[3].imshow(
        coarse2,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[3].set_title(
        "8x8 -> original size\n"
        f"mean={coarse2.mean():.4f}, "
        f"std={coarse2.std():.4f}"
    )

    axes[4].imshow(rgb)
    axes[4].imshow(
        coarse2,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        alpha=0.35,
        interpolation="nearest",
    )
    axes[4].set_title(
        "RGB + coarse_8x8\n"
        f"P85={score:.4f}, zoom={int(is_zoomed)}"
    )

    for ax in axes:
        ax.axis("off")

    # Re-enable native-grid axes so grid lines remain visible.
    axes[2].axis("on")
    axes[2].tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    fig.tight_layout()

    out = OUTPUT_DIR / f"step_{step:04d}.png"
    fig.savefig(
        out,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    np.savetxt(
        OUTPUT_DIR / f"step_{step:04d}_grid8x8.csv",
        coarse_grid,
        delimiter=",",
        fmt="%.7f",
    )

    return out


# ============================================================
# Main
# ============================================================

def main():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    os.chdir(PROJECT_ROOT)

    import tools
    from expr import make_env
    from envs.tasks.base.ls_imagine_wrapper import BASIC_ACTIONS

    tools.set_seed_everywhere(SEED)
    np.random.seed(SEED)

    # Separate RNG used only by the safe random action policy.
    rng = np.random.RandomState(SEED)

    config = load_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.results_dir = str(
        OUTPUT_DIR / "_env_logs"
    )
    config.logdir = OUTPUT_DIR
    config.evaldir = OUTPUT_DIR / "_eval"
    config.name = "coarse_8x8_visual_test"

    prepare_task_config(config)

    print("=" * 100)
    print("COARSE 8x8 HEATMAP VISUAL TEST")
    print("=" * 100)
    print("task   :", TASK)
    print("steps  :", NUM_STEPS)
    print("output :", OUTPUT_DIR)
    print("=" * 100)

    env = make_env(
        config,
        "eval",
        0,
    )

    print_wrapper_chain(env)

    cw = find_wrapper(
        env,
        "ConcentrationWrapper",
    )

    # Use the source-of-truth action names directly.
    # This avoids relying on private attributes of a wrapped object.
    basic_action_names = tuple(
        BASIC_ACTIONS.keys()
    )

    # Force this test mode regardless of task_specs.yaml,
    # but only AFTER wrapper construction and BEFORE reset().
    cw.wm_heatmap_mode = "coarse_8x8"

    if not hasattr(
        cw,
        "_coarse_average_pool_and_expand",
    ):
        raise RuntimeError(
            "Your concentration_wrapper.py does not contain "
            "_coarse_average_pool_and_expand(). "
            "Apply the coarse_8x8 modification first."
        )

    print(
        "[PASS] Found actual ConcentrationWrapper:",
        cw.__class__.__name__,
    )
    print(
        "[INFO] BASIC_ACTIONS:",
        basic_action_names,
    )
    print(
        "[INFO] outer action_space:",
        env.action_space,
    )
    print("[PASS] Forced wm_heatmap_mode=coarse_8x8")

    summary_rows = []

    def inspect(step, obs):
        spatial = cw.concentration.get_heatmap(
            is_zoomed=False
        )

        transformed = cw._to_world_model_heatmap(
            spatial
        )

        ref_grid, ref_expanded = (
            independent_coarse_8x8(
                spatial
            )
        )

        err = float(
            np.max(
                np.abs(
                    transformed
                    - ref_expanded
                )
            )
        )

        mean_err = float(
            abs(
                float(np.mean(spatial))
                - float(np.mean(transformed))
            )
        )

        cell_std = max_within_cell_std(
            transformed
        )

        assert transformed.shape == spatial.shape
        assert transformed.dtype == np.float32
        assert err < 1e-6, err
        assert mean_err < 1e-6, mean_err
        assert cell_std < 1e-6, cell_std

        rgb = np.asarray(
            cw.concentration.curr_frame
        )

        score = float(
            obs.get(
                "score",
                cw.concentration.get_progress_score(
                    is_zoomed=False
                ),
            )
        )

        is_zoomed = bool(
            obs.get(
                "is_zoomed",
                False,
            )
        )

        print(
            f"step={step:03d} | "
            f"spatial=[{spatial.min():.4f},"
            f"{spatial.max():.4f}] "
            f"mean={spatial.mean():.6f} "
            f"std={spatial.std():.6f} | "
            f"grid8 std={ref_grid.std():.6f} | "
            f"mean_err={mean_err:.2e} | "
            f"formula_err={err:.2e} | "
            f"P85={score:.4f} | "
            f"zoom={int(is_zoomed)}"
        )

        summary_rows.append(
            {
                "step": step,
                "spatial_min": float(spatial.min()),
                "spatial_max": float(spatial.max()),
                "spatial_mean": float(spatial.mean()),
                "spatial_std": float(spatial.std()),
                "coarse_min": float(ref_grid.min()),
                "coarse_max": float(ref_grid.max()),
                "coarse_mean": float(ref_grid.mean()),
                "coarse_std": float(ref_grid.std()),
                "mean_abs_error": mean_err,
                "formula_max_abs_error": err,
                "max_within_cell_std": cell_std,
                "score_p85": score,
                "is_zoomed": int(is_zoomed),
            }
        )

        if (
            step == 0
            or step % SAVE_EVERY == 0
            or is_zoomed
        ):
            out = save_visualization(
                step,
                rgb,
                spatial,
                ref_grid,
                transformed,
                score,
                is_zoomed,
            )

            print("  saved:", out)

    obs = env.reset()
    inspect(0, obs)

    for step in range(
        1,
        NUM_STEPS + 1,
    ):
        # The actual LS-Imagine stack exposes one-hot BASIC_ACTIONS,
        # not MineDojo's raw 8-D MultiDiscrete action.
        action, action_name, action_index = (
            sample_safe_random_action(
                env,
                basic_action_names,
                rng,
            )
        )

        if (
            step <= 5
            or step % 10 == 0
        ):
            print(
                f"  action step={step:03d}: "
                f"{action_name} "
                f"(discrete={action_index})"
            )

        obs, reward, done, info = env.step(
            action
        )

        inspect(
            step,
            obs,
        )

        if done:
            print(
                f"[INFO] episode ended at step={step}; resetting."
            )
            obs = env.reset()

    # Save summary.
    fields = list(
        summary_rows[0].keys()
    )

    with (
        OUTPUT_DIR / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 100)
    print("[PASS] coarse_8x8 formula matches independent reference")
    print("[PASS] global frame mean is preserved")
    print("[PASS] every expanded 8x8 cell is constant")
    print("Saved visualizations and CSV to:")
    print(OUTPUT_DIR)
    print("=" * 100)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
