from typing import Dict
from omegaconf import OmegaConf
from minedojo.tasks import MetaTaskBase, _meta_task_make, _parse_inventory_dict, ALL_TASKS_SPECS
from minedojo.sim import MineDojoSim

from envs.tasks.minedojo.wrappers import *


def _get_minedojo_specs(task_id, task_specs, sim_specs):
    if task_id in ALL_TASKS_SPECS:
        minedojo_specs = ALL_TASKS_SPECS[task_id]
        if OmegaConf.is_config(minedojo_specs):
            minedojo_specs = OmegaConf.to_container(minedojo_specs)
        minedojo_specs.pop("prompt", None)
        meta_task_cls = minedojo_specs.pop("__cls__")
    else:
        minedojo_specs = dict()
        meta_task_cls = task_id

    minedojo_specs.update(dict(
        image_size=(160, 256),
        fast_reset=False,
        event_level_control=False,
        use_voxel=False,
        use_lidar=False
    ))

    # If using blocks condition, activate voxels
    if ("terminal_specs" in task_specs and \
        ("all" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["all"]) or \
         "any" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["any"]))) or \
            ("success_specs" in task_specs and \
             ("all" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["all"]) or \
              "any" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["any"]))):
        minedojo_specs["use_voxel"] = True
        minedojo_specs["voxel_size"] = dict(xmin=-3, ymin=-1, zmin=-3, xmax=3, ymax=1, zmax=3)

    minedojo_specs.update(**sim_specs)

    if "initial_inventory" in minedojo_specs:
        minedojo_specs["initial_inventory"] = _parse_inventory_dict(minedojo_specs["initial_inventory"])

    return meta_task_cls, minedojo_specs


def _add_wrappers(
        env: MetaTaskBase,
        task_id: str,
        LS_Imagine_specs: Dict = None,
        screenshot_specs: Dict = None,
        reward_specs: Dict = None,
        success_specs: Dict = None,
        terminal_specs: Dict = None,
        clip_specs: Dict = None,
        concentration_specs: Dict = None,
        fast_reset: int = None,
        log_dir: str = None,
        freeze_equipped: bool = True,
        **kwargs
):
    if terminal_specs is None:
        terminal_specs = dict(max_steps=500, on_death=True)
    if success_specs and terminal_specs:
        success_specs["max_steps"] = terminal_specs["max_steps"]
    if screenshot_specs:
        env = MinedojoScreenshotWrapper(env, log_dir=log_dir, **screenshot_specs)
    if reward_specs:
        env = MinedojoRewardWrapper(env, **reward_specs)
    if success_specs:
        env = MinedojoSuccessWrapper(env, **success_specs)

    env = MinedojoTerminalWrapper(env, **terminal_specs)

    # 一个共享 MineCLIP
    shared_clip_reward = None
    if (
            clip_specs is not None
            or concentration_specs is not None
    ):
        shared_clip_reward = (
            MinedojoClipReward()
        )

    # Add reward shaping wrapper
    # MineCLIP intrinsic reward
    if clip_specs is not None:
        env = ClipWrapper(
            env,
            shared_clip_reward,
            **clip_specs,
        )
    # if clip_specs is not None:
    #     clip_reward = MinedojoClipReward()
    #     env = ClipWrapper(env, clip_reward, **clip_specs)

    # MineCLIP relevance map
    # ============================================================

    if concentration_specs is not None:
        # ----------------------------
        # Relevance
        # ----------------------------
        gaussian_sigma_weight = (
            concentration_specs["gaussian_sigma_weight"]
            if "gaussian_sigma_weight" in concentration_specs
            else 0.5
        )

        # ScoreStorage progress readout.
        # Global setting, shared across tasks; no task-specific calibration.
        progress_percentile = (
            concentration_specs.get(
                "progress_percentile",
                85.0,
            )
        )

        # ----------------------------
        # Gate 1: cooldown
        # ----------------------------
        zoom_cooldown_steps = (
            concentration_specs.get(
                "zoom_cooldown_steps",
                5,
            )
        )

        # ----------------------------
        # Gate 2: semantic gate
        # ----------------------------
        semantic_gate_enabled = (
            concentration_specs.get(
                "semantic_gate_enabled",
                False,
            )
        )
        semantic_min_p95_p50 = (
            concentration_specs.get(
                "semantic_min_p95_p50",
                0.025,
            )
        )
        semantic_min_cc_fraction = (
            concentration_specs.get(
                "semantic_min_cc_fraction",
                0.02,
            )
        )

        # ----------------------------
        # Gate 3: post-zoom gain
        # ----------------------------
        post_zoom_semantic_gate_enabled = (
            concentration_specs.get(
                "post_zoom_semantic_gate_enabled",
                False,
            )
        )
        post_zoom_min_p95_gain = (
            concentration_specs.get(
                "post_zoom_min_p95_gain",
                0.0,
            )
        )
        post_zoom_min_contrast_gain = (
            concentration_specs.get(
                "post_zoom_min_contrast_gain",
                -0.005,
            )
        )
        post_zoom_min_cc_retention = (
            concentration_specs.get(
                "post_zoom_min_cc_retention",
                0.50,
            )
        )

        # ----------------------------
        # Zoom scale
        # ----------------------------
        max_zoom_factor = (
            concentration_specs.get(
                "max_zoom_factor",
                3.5,
            )
        )

        # ====================================================
        # 创建 ConcentrationReward
        # ====================================================
        concentration_reward = (
            MinedojoConcentrationReward(
                clip_reward=shared_clip_reward,
                output_dir=log_dir,
                gaussian_sigma_weight=gaussian_sigma_weight,
                progress_percentile=progress_percentile,
                # Gate 1
                zoom_cooldown_steps=zoom_cooldown_steps,
                # Gate 2
                semantic_gate_enabled=semantic_gate_enabled,
                semantic_min_p95_p50=semantic_min_p95_p50,
                semantic_min_cc_fraction=semantic_min_cc_fraction,
                # Gate 3
                post_zoom_semantic_gate_enabled=post_zoom_semantic_gate_enabled,
                post_zoom_min_p95_gain=post_zoom_min_p95_gain,
                post_zoom_min_contrast_gain=post_zoom_min_contrast_gain,
                post_zoom_min_cc_retention=post_zoom_min_cc_retention,
                # Zoom limit
                max_zoom_factor=max_zoom_factor,
            )
        )

        env = ConcentrationWrapper(
            env,
            concentration_reward,
            **concentration_specs,
        )
    # if concentration_specs is not None:
    #     unet_checkpoint_dir = concentration_specs["unet_checkpoint_dir"] if "unet_checkpoint_dir" in concentration_specs else "envs/tasks/base/unet_checkpoint"
    #     gaussian_sigma_weight = concentration_specs["gaussian_sigma_weight"] if "gaussian_sigma_weight" in concentration_specs else 0.5
    #     concentration_reward = MinedojoConcentrationReward(unet_checkpoint_dir=unet_checkpoint_dir, output_dir=log_dir, gaussian_sigma_weight=gaussian_sigma_weight)
    #     env = ConcentrationWrapper(env, concentration_reward, **concentration_specs)

    env = MinedojoLSImagineWrapper(env, **LS_Imagine_specs)

    # If we don't care about start position, use fast reset to speed training and prevent memory leaks
    if fast_reset is not None:
        wrapped = env
        while hasattr(wrapped, "env"):
            if isinstance(wrapped.env, MineDojoSim):
                wrapped.env = MinedojoSemifastResetWrapper(
                    wrapped.env,
                    reset_freq=fast_reset,
                    random_teleport_range=200
                )
                break
            wrapped = wrapped.env

    return env


def make_minedojo(task_id: str, task_specs, sim_specs):
    # Get minedojo specs
    meta_task_cls, minedojo_specs = _get_minedojo_specs(task_id, task_specs, sim_specs)

    # Make minedojo env
    env = _meta_task_make(meta_task_cls, **minedojo_specs)
    env = _add_wrappers(env, task_id, **task_specs)

    return env
