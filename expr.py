import argparse
import functools
import os
import pathlib
import sys
import torch
import numpy as np
import ruamel.yaml as yaml
from torch import nn
from torch import distributions as torchd
from datetime import datetime

import exploration as expl
import models
import tools
from search_a import SearchAController, actor_action_sequence, check_checkpoint, MODEL_VERSION
from search_memory import describe_torch, sample_episodes_with_history, B_VERSION, check_b_checkpoint
import envs.wrappers as wrappers
from parallel import Parallel, Damy

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()


class LS_Imagine(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(LS_Imagine, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        # APM-IMAGINE A STAGE: low-frequency counterfactual exploration control.
        self._search_a = (SearchAController(
            horizon=config.search_a_horizon,
            override_prob=config.search_a_override_prob,
            low_evidence=config.search_a_low_evidence,
            new_weight=config.search_a_new_weight,
            discovery_weight=config.search_a_discovery_weight,
            revisit_weight=config.search_a_revisit_weight,
            cost_weight=config.search_a_cost_weight,
            min_labels=config.search_a_min_labels,
            min_positives=config.search_a_min_positives,
            margin=config.search_a_margin,
            start_step=config.search_a_start_step,
            control_version=config.search_a_control_version,
            commit_steps=config.search_a_commit_steps,
            budget_window=config.search_a_budget_window,
            budget_fraction=config.search_a_budget_fraction,
            cooldown=config.search_a_cooldown,
            revisit_gain_cap=config.search_a_revisit_gain_cap,
            min_information_gain=config.search_a_min_information_gain,
            repeat_cost=config.search_a_repeat_cost,
            visible_score_weight=config.search_a_visible_score_weight,
            visible_high=config.search_a_visible_high,
            visible_low=config.search_a_visible_low,
            conflict_steps=config.search_a_conflict_steps,
            protect_interaction=config.search_a_protect_interaction,
            seed=config.seed,
            # Random macros gather event supervision only while A is learning.
            epsilon=(config.search_a_epsilon if config.search_a_train
                     and config.search_a_loss_weight > 0 and (config.search_a_event_weight > 0
                         or (config.search_a_model_version >= 3 and config.search_a_visible_weight > 0)) else 0.0),
        ) if config.search_a_enabled and config.search_a_control else None)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    if np.size(values):
                        self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        # APM-IMAGINE A STAGE: keep a macro progress tensor in recurrent policy state.
        # APM-IMAGINE B STAGE: persistent bank is part of the recurrent policy state.
        if state is None:
            latent = action = search_state = memory_state = None
        elif len(state) == 2:  # backward-compatible initial states
            latent, action = state
            search_state = memory_state = None
        elif len(state) == 3:
            latent, action, search_state = state
            memory_state = None
        else:
            latent, action, search_state, memory_state = state
        # Strip training-only evidence even before tensor conversion.
        obs = self._wm.preprocess({k: v for k, v in obs.items() if not k.startswith("search_")})
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        memory_context = None
        memory_info = {}
        if self._wm.search_b is not None:
            scene = describe_torch(obs["image"], obs["heatmap"])
            memory_context, memory_state, memory_info = self._wm.search_b.online_step(
                scene, memory_state, obs["is_first"], previous_action=action)
        policy_memory = memory_context if self._config.search_b_control else (
            torch.zeros_like(memory_context) if memory_context is not None else None)
        policy_feat = torch.cat((feat, policy_memory), -1) if policy_memory is not None else feat
        if not training:
            actor = self._task_behavior.actor(policy_feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(policy_feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(policy_feat)
            action = actor.sample()
        search_info = {}
        if self._search_a is not None:
            proposed_action = action
            def continuation(indices):
                return actor_action_sequence(
                    self._wm.dynamics, self._task_behavior.actor,
                    {k: v[indices] for k, v in latent.items()},
                    proposed_action[indices], self._config.search_a_horizon,
                    memory=policy_memory[indices] if policy_memory is not None else None,
                    memory_transition=self._wm.search_b.imagine_step
                    if self._wm.search_b is not None and self._config.search_b_control else None)
            def candidate_continuations(indices, templates):
                # Imagine the committed exploration prefix, then Actor in each resulting state.
                candidate_ids = torch.tensor([0, 1, 2, 3, 4, 6, 7], device=feat.device)
                count, num = len(indices), len(candidate_ids)
                prefix = templates[candidate_ids, :self._search_a.commit_steps]
                prefix = prefix[None].expand(count, -1, -1, -1).reshape(
                    count * num, self._search_a.commit_steps, self._config.num_actions)
                expanded = {k: v[indices, None].expand(-1, num, *v.shape[1:]).reshape(
                    count * num, *v.shape[1:]) for k, v in latent.items()}
                expanded_memory = (policy_memory[indices, None].expand(-1, num, -1).reshape(
                    count * num, -1) if policy_memory is not None else None)
                imagined = actor_action_sequence(
                    self._wm.dynamics, self._task_behavior.actor, expanded, prefix[:, 0],
                    self._config.search_a_horizon, memory=expanded_memory,
                    memory_transition=self._wm.search_b.imagine_step
                    if self._wm.search_b is not None and self._config.search_b_control else None,
                    prefix_actions=prefix)
                result = templates[None].expand(count, -1, -1, -1).clone()
                result[:, candidate_ids] = imagined.reshape(
                    count, num, self._config.search_a_horizon, self._config.num_actions)
                return result
            action, search_state, search_info = self._search_a.choose(
                self._wm.search_a, feat, obs, proposed_action, search_state,
                obs["is_first"], self._step, training, continuation,
                memory=(memory_context if self._config.search_b_control else None),
                candidate_sequence=candidate_continuations)
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        policy_output.update(search_info)
        policy_output.update(memory_info)
        if self._wm.search_b is not None:
            state = (latent, action, search_state, memory_state)
        else:
            state = (latent, action, search_state) if self._search_a is not None else (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, post_zoomed, context, mets = self._wm._train(data, step=self._step)
        metrics.update(mets)
        # start = (post, post_zoomed)

        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        intrinsic = lambda f, s, a: self._wm.heads["intrinsic"](
            self._wm.dynamics.get_feat(s)
        ).mode() 

        jumping_steps = lambda f, s, a: self._wm.heads["jumping_steps"](
            f
        ).mean().clamp_min(1).int()

        accumulated_reward = lambda f, s, a: self._wm.heads["accumulated_reward"](
            f
        ).mode()

        jump_indicator = lambda s: self._wm.heads["jump"](
            self._wm.dynamics.get_feat(s)
        ).mean

        is_end = lambda s: self._wm.heads["end"](
            self._wm.dynamics.get_feat(s)
        ).mean

        metrics.update(self._task_behavior._train(post, post_zoomed, reward, intrinsic, jumping_steps,
                        accumulated_reward, jump_indicator, is_end,
                        search_context=context.get("search_memory"),
                        search_context_zoomed=context.get("search_memory_zoomed"))[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = (sample_episodes_with_history(
        episodes, config.batch_length, slots=config.search_b_slots,
        lookback=config.search_b_lookback, save_every=config.search_b_save_every) if config.search_b_enabled
        else tools.sample_episodes(episodes, config.batch_length))
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "minedojo":
        import envs.minedojo as minedojo
        log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        collect_labels = (mode == "train" and (
            (config.search_a_enabled and config.search_a_train
             and config.search_a_loss_weight > 0 and (config.search_a_event_weight > 0
                  or (config.search_a_model_version >= 3 and config.search_a_visible_weight > 0)))
            or (config.search_b_enabled and config.search_b_loss_weight > 0)))
        kwargs = dict(
            log_dir=log_dir, target_item=config.target_item,
            search_evidence_specs=dict(
                enabled=collect_labels, target=config.target_item,
                distance=config.search_a_ray_distance, cell_size=config.search_a_cell_size,
                revisit_gap=config.search_a_horizon),
            simulator_seed=(config.env_seed + id + (100000 if mode == "eval" else 0)
                            if config.env_seed >= 0 else None))

        from envs.tasks import get_specs
        _, specs, _ = get_specs(task, target_item=config.target_item)
        max_steps = config.episode_max_steps_override or specs["terminal_specs"]["max_steps"]
        kwargs["terminal_specs"] = dict(specs["terminal_specs"], max_steps=max_steps)
        if config.episode_max_steps_override:
            kwargs["terminal_specs"]["stagger_max_steps"] = False
        kwargs["concentration_specs"] = dict(
            specs["concentration_specs"], max_steps=max_steps,
            gaussian_reward_weight=config.gaussian_reward_weight,
            gaussian_sigma_weight=config.gaussian_sigma_weight)
        success = specs["success_specs"]
        target = success["all"]["item"]["type"] if "all" in success else success["any"]["item"]["type"]
        kwargs["clip_specs"] = dict(specs["clip_specs"], target_object=target)
        env = minedojo.make_env(task, **kwargs)
        env = wrappers.OneHotAction(env)

    else:
        raise NotImplementedError(suite)
    
    # env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)

    return env


def main(config): # config is namespace

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser() 
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
    config.logdir = logdir
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat 
    config.log_every //= config.action_repeat 
    config.time_limit //= config.action_repeat
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    
    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_traindir: # False
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir

    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

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
    
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = ([make("eval", i) for i in range(config.envs)]
                 if config.eval_episode_num > 0 else [])

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step_calculator = tools.ScoreStorage(max_steps=config.episode_max_steps)

    state = None

    if not config.offline_traindir: 
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=prefill,
            is_training=False,
        )

        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = LS_Imagine(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)
    
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        check_checkpoint(checkpoint, config.search_a_enabled, config.search_a_model_version)
        check_b_checkpoint(checkpoint, config.search_b_enabled)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        agent._step = int(checkpoint.get("step", agent._step))
        if agent._search_a is not None:
            agent._search_a.load_rng_state(checkpoint.get("search_a_rng_state"))
        if "model_amp_scaler" in checkpoint:
            agent._wm._model_opt._scaler.load_state_dict(checkpoint["model_amp_scaler"])
        logger.step = config.action_repeat * agent._step

    
    # make sure eval will be executed once after config.steps
    while True:
        logger.write()
        
        if config.eval_episode_num > 0:
            print("Start evaluation.")
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

        if agent._step >= config.steps:
            break
        print("Start training.")

        state = tools.simulate(
            agent, # LS_Imagine
            train_envs, 
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=min(config.eval_every, config.steps - agent._step),
            state=state,
            is_training=True,
        )

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": agent._step,
            "config": vars(config),
            "search_a_version": config.search_a_model_version if config.search_a_enabled else None,
            "search_a_control_version": config.search_a_control_version if config.search_a_enabled else None,
            "search_a_label_version": 2 if config.search_a_enabled else None,
            "search_a_rng_state": agent._search_a.rng_state() if agent._search_a is not None else None,
            "search_a_macro_names": list(agent._search_a.macro_names) if agent._search_a is not None else [],
            "model_amp_scaler": agent._wm._model_opt._scaler.state_dict(),
            "search_b_version": B_VERSION if config.search_b_enabled else None,
        }
        
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
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
