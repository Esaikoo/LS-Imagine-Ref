import argparse
import functools
import os
import pathlib
import sys
import torch
import ruamel.yaml as yaml
from datetime import datetime

import tools
from parallel import Parallel, Damy

from expr import LS_Imagine, make_env, make_dataset, check_b_checkpoint
from search_a import check_checkpoint, restore_a_config, apply_repaired_control
from search_c import restore_c_config, apply_c_control_mode

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()

def configure_evaluation(config, checkpoint):
    """Restore trained shapes, then honor explicit runtime control-off flags."""
    from search_vnext import restore_vnext_config
    runtime_vnext_control = getattr(config, "vnext_control", True)
    runtime_vnext_interaction = getattr(config, "vnext_interaction", True)
    runtime_vnext_memory = getattr(config, "vnext_memory_enabled", True)
    restore_vnext_config(vars(config), checkpoint)
    if config.search_vnext_enabled:
        if not runtime_vnext_memory:
            config.vnext_memory_enabled = False
        if config.search_a_eval_mode != "checkpoint" or config.search_c_eval_mode != "checkpoint":
            raise ValueError("Use vnext_control/vnext_memory_enabled/vnext_interaction for vNext ablations")
        if not runtime_vnext_control:
            config.vnext_control = False
        if not runtime_vnext_interaction:
            config.vnext_interaction = False
        if "step" not in checkpoint:
            raise ValueError("vNext checkpoint requires its training step")
    requested_control = config.search_a_control
    requested_b_control = config.search_b_control
    requested_c_control = config.search_c_control
    requested_c_mode = config.search_c_eval_mode
    requested_a_mode = config.search_a_eval_mode
    restore_a_config(vars(config), checkpoint)
    for key, value in checkpoint.get("config", {}).items():
        if key.startswith("search_b_"):
            setattr(config, key, value)
    config.search_b_enabled = any(".search_b." in k for k in checkpoint["agent_state_dict"])
    config.search_b_model_version = checkpoint.get("search_b_version") or 2
    if not requested_b_control:
        config.search_b_control = False
    restore_c_config(vars(config), checkpoint)
    apply_c_control_mode(vars(config), requested_c_mode)
    if not requested_c_control:
        config.search_c_control = False
    config.search_a_eval_mode = requested_a_mode
    if (config.search_c_enabled and config.search_c_control
            and (not requested_control or requested_a_mode != "checkpoint")):
        raise ValueError("Disable C explicitly before overriding A control")
    if not requested_control or config.search_a_eval_mode == "off":
        config.search_a_control = False
    elif config.search_a_eval_mode == "repaired":
        apply_repaired_control(vars(config))
    elif config.search_a_eval_mode != "checkpoint":
        raise ValueError("search_a_eval_mode must be checkpoint, off, or repaired")
    check_checkpoint(checkpoint, config.search_a_enabled, config.search_a_model_version)
    check_b_checkpoint(checkpoint, config.search_b_enabled)
    if (config.search_a_enabled or config.search_c_enabled) and "step" not in checkpoint:
        raise ValueError("A/C checkpoint requires step to restore control warmup")


def main(config):

    checkpoint_path = pathlib.Path(config.agent_checkpoint_dir) / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    configure_evaluation(config, checkpoint)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser() 
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
    config.logdir = logdir

    config.evaldir = config.evaldir or logdir / "eval_eps"
    logdir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = 0
    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_evaldir: # False
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    eval_eps = tools.load_episodes(directory, limit=1)

    make = lambda mode, id: make_env(config, mode, id)
    suite, task = config.task.split("_", 1)

    from envs.tasks import get_specs

    kwargs=dict(
            # log_dir=log_dir,
            target_item=config.target_item
        )
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict

    config.episode_max_steps = (config.episode_max_steps_override
                                or task_specs['terminal_specs']['max_steps'])
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        eval_envs = [Damy(env) for env in eval_envs]
    acts = eval_envs[0].action_space

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step_calculator = tools.ScoreStorage(max_steps=config.episode_max_steps)

    state = None

    print("Start evaluation.")
    eval_dataset = make_dataset(eval_eps, config)
    agent = LS_Imagine(
        eval_envs[0].observation_space,
        eval_envs[0].action_space,
        config,
        logger,
        eval_dataset,
    ).to(config.device)

    checkpoint_path = pathlib.Path(config.agent_checkpoint_dir) / "latest.pt"
    print(checkpoint_path)
    
    if checkpoint_path.exists():
        print("Loading checkpoint from", config.agent_checkpoint_dir)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.requires_grad_(False)  # Optimizer state is unnecessary for evaluation.
        agent._should_pretrain._once = False
        if getattr(config, "search_a_enabled", False) and "step" not in checkpoint:
            raise ValueError("A checkpoint has no training step; cannot restore search warmup")
        agent._step = int(checkpoint.get("step", agent._step))
        if agent._vnext_control is not None:
            agent._vnext_control.load_rng_state(checkpoint.get("vnext_rng_state"))
        logger.step = config.action_repeat * agent._step

    else:
        raise ValueError("no checkpoint found")
    
    if config.eval_episode_num > 0:
        eval_policy = functools.partial(agent, training=False) 
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            is_eval=True,
            episodes=config.eval_episode_num,
            is_training=False,
        )
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_openl", to_np(video_pred))

    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name]) 

    parser = argparse.ArgumentParser()

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))

