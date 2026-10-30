import argparse
import functools
import os
import pathlib
import sys
import time
import torch
import numpy as np
import ruamel.yaml as yaml
from torch import nn
from torch import distributions as torchd
from datetime import datetime

import exploration as expl
import models
import tools
from search_a import (SearchAController, PrefixSearchController, actor_action_sequence,
                      prefix_macro_actions, prefix_candidate_predictions, check_checkpoint, MODEL_VERSION)
from search_c import SearchBehavior, build_transitions, check_c_checkpoint, C_VERSION
from search_memory import describe_torch, sample_episodes_with_history, B_VERSION, check_b_checkpoint
from search_vnext import SearchVNext, sample_episodes_vnext
from search_vnext_control import VNextController
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
        if getattr(config, "search_vnext_enabled", False):
            if any((config.search_a_enabled, config.search_b_enabled, config.search_c_enabled, config.search_branch_enabled)):
                raise ValueError("vNext cannot mix with legacy ABC modules or branch collection")
            if config.compile:
                raise ValueError("vNext requires compile=False for explicit isolated optimizers")
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        self._search_c = None
        if config.search_c_enabled:
            if config.search_a_control_version != 4:
                raise ValueError("C requires the v4 executable prefix interface")
            size = (config.dyn_stoch * config.dyn_discrete if config.dyn_discrete else config.dyn_stoch) + config.dyn_deter
            size += config.search_b_dim if config.search_b_enabled else 0
            with torch.random.fork_rng(devices=[]):
                self._search_c = SearchBehavior(size, config)
        controller = PrefixSearchController if config.search_a_control_version == 4 else SearchAController
        # APM-IMAGINE A STAGE: low-frequency counterfactual exploration control.
        self._search_a = (controller(
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
        ) if (config.search_a_enabled and config.search_a_control)
             or (config.search_c_enabled and config.search_c_control) else None)
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
        self._vnext = self._vnext_control = None
        if getattr(config, "search_vnext_enabled", False):
            size = (config.dyn_stoch * config.dyn_discrete if config.dyn_discrete
                    else config.dyn_stoch) + config.dyn_deter
            # Additional modules cannot perturb original Actor/RSSM initialization.
            with torch.random.fork_rng(devices=[]):
                self._vnext = SearchVNext(size, config)
                self._vnext_control = VNextController(size, config, memory_dim=self._vnext.memory_dim)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                batch_start = time.perf_counter()
                data = next(self._dataset)
                self._last_batch_prepare_ms = (time.perf_counter() - batch_start) * 1000.
                self._train(data)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                self.flush_metrics()
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def flush_metrics(self):
        """Persist final partial logging intervals without performing an update."""
        for name, values in self._metrics.items():
            if np.size(values):
                self._logger.scalar(name, float(np.mean(values)))
            self._metrics[name] = []

    def _policy(self, obs, state, training):
        if self._vnext is not None:
            return self._vnext_policy(obs, state, training)
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
                scene, memory_state, obs["is_first"], previous_action=action, feature=feat)
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
            extra = {}
            if self._search_a.control_version == 4:
                extra = dict(selector=self._search_c if self._config.search_c_control else None,
                             policy_memory=policy_memory)
            action, search_state, search_info = self._search_a.choose(
                self._wm.search_a, feat, obs, proposed_action, search_state,
                obs["is_first"], self._step, training, continuation,
                memory=(memory_context if self._config.search_b_control else None),
                candidate_sequence=candidate_continuations, **extra)
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
        policy_output["search_data_source"] = search_info.get(
            "log_search_override", torch.zeros_like(logprob)).float()
        if self._wm.search_b is not None:
            state = (latent, action, search_state, memory_state)
        else:
            state = (latent, action, search_state) if self._search_a is not None else (latent, action)
        return policy_output, state

    @torch.no_grad()
    def _vnext_policy(self, obs, state, training):
        latent, previous_action, memory_state, control_state = (
            (None, None, None, None) if state is None else state)
        # Only original, legal policy inputs reach the shared encoder or search.
        allowed = {"image", "heatmap", "is_first", "is_last", "is_terminal",
                   "is_zoomed", "jump", "is_calculated", "obs_reward",
                   "inventory", "inventory_max", "equipped", "health", "hunger", "breath"}
        clean = {key: value for key, value in obs.items() if key in allowed}
        clean = self._wm.preprocess(clean)
        embed = self._wm.encoder(clean)
        latent, _ = self._wm.dynamics.obs_step(latent, previous_action, embed, clean["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feature = self._wm.dynamics.get_feat(latent).detach()
        actor = (self._expl_behavior.actor(feature)
                 if training and self._should_expl(self._step) else self._task_behavior.actor(feature))
        proposed = actor.sample() if training else actor.mode()
        if control_state is None and "vnext_bootstrap_actuator" in obs:
            control_state = self._vnext_control.initial(len(feature), feature.device)
            control_state["actuator"] = torch.as_tensor(obs["vnext_bootstrap_actuator"],device=feature.device).float()
            control_state["previous_control"] = torch.as_tensor(obs["vnext_bootstrap_control"],device=feature.device).float()
            control_state["age"] = torch.as_tensor(obs["vnext_episode_step"],device=feature.device).long()
        previous_control = self._vnext_control.previous_control(
            control_state, clean["is_first"], feature.device)
        current, memory, memory_state, memory_info = self._vnext.online(
            feature, clean["image"], clean["heatmap"], memory_state,
            clean["is_first"], previous_control=previous_control,
            reward=clean.get("obs_reward"))

        def continuation(indices, k=3):
            return actor_action_sequence(
                self._wm.dynamics, self._task_behavior.actor,
                {key: value[indices] for key, value in latent.items()},
                proposed[indices], k)

        action, control_state, control_info = self._vnext_control.choose(
            self._vnext, feature, memory, current, clean, proposed, control_state,
            clean["is_first"], self._step, training, actor_sequence=continuation)
        action = action.detach()
        output = {"action": action, "logprob": actor.log_prob(action)}
        output.update(memory_info)
        output.update(control_info)
        # Preserve the intervention/decision metric interface; protocol 7 also
        # records separate option and task-stage rates in vNext diagnostics.
        output["log_search_override"] = control_info["log_vnext_override"]
        output["log_search_decision"] = control_info["log_vnext_decision"]
        output["search_data_source"] = control_info.get(
            "log_vnext_override", feature.new_zeros(len(feature))).float()
        latent = {key: value.detach() for key, value in latent.items()}
        return output, (latent, action, memory_state, control_state)

    def _train(self, data):
        metrics = {}
        profile = (self._vnext is not None and self._update_count %
                   max(1, int(getattr(self._config, "vnext_profile_every", 32))) == 0)
        def clock():
            if profile and torch.device(self._config.device).type == "cuda":
                torch.cuda.synchronize(self._config.device)
            return time.perf_counter()
        stage = clock() if profile else None
        post, post_zoomed, context, mets = self._wm._train(data, step=self._step)
        metrics.update(mets)
        if profile:
            now = clock()
            metrics["diagnostics/time_world_model_ms"] = (now-stage)*1000.
            stage = now
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
        if profile:
            now = clock()
            metrics["diagnostics/time_task_behavior_ms"] = (now-stage)*1000.
            stage = now
        if self._search_c is not None:
            metrics.update(self._train_search_c(post, context.get("search_memory"), data))
        if self._vnext is not None:
            features = self._wm.dynamics.get_feat(post).detach()
            metrics.update(self._vnext.train_step(features, data, self._step))
            with torch.no_grad():
                memory = self._vnext.memory_batch(features, data)
            if profile:
                now = clock()
                metrics["diagnostics/time_search_ab_ms"] = (now-stage)*1000.
                stage = now
            def actor_continuation(batch_indices, time_indices, k=3):
                latent = {key: value[batch_indices, time_indices] for key, value in post.items()}
                first = self._task_behavior.actor(self._wm.dynamics.get_feat(latent)).mode()
                return actor_action_sequence(self._wm.dynamics, self._task_behavior.actor,
                                             latent, first, k)
            metrics.update(self._vnext_control.train_batch(
                features, memory, self._vnext, data, actor_sequence=actor_continuation))
            if profile:
                now = clock()
                metrics["diagnostics/time_search_c_ms"] = (now-stage)*1000.
                metrics["diagnostics/time_batch_prepare_ms"] = getattr(self, "_last_batch_prepare_ms", 0.)
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


    def _train_search_c(self, post, memory, data):
        records = build_transitions(data, self._config, self._config.search_a_commit_steps)
        if not records:
            return {"C/transitions": 0.}
        device = next(self.parameters()).device
        snapshots = {}
        def inputs(endpoint):
            b = torch.tensor([r["batch"] for r in records], device=device)
            t = torch.tensor([r[endpoint] for r in records], device=device)
            latent = {k: value[b, t] for k, value in post.items()}
            feat = self._wm.dynamics.get_feat(latent).detach()
            mem = memory[b, t].detach() if memory is not None else None
            policy_memory = mem if self._config.search_b_control else (
                torch.zeros_like(mem) if mem is not None else None)
            state = torch.cat((feat, policy_memory), -1) if policy_memory is not None else feat
            context_key = "context" if endpoint == "start" else "next_context"
            history = torch.as_tensor(np.stack([r[context_key] for r in records]), device=device)
            k = self._config.search_a_commit_steps
            with torch.no_grad():
                first_action = self._task_behavior.actor(state).mode()
                sequence = prefix_macro_actions(k, device)[None].expand(len(records), -1, -1, -1).clone()
                sequence[:, 5] = actor_action_sequence(
                    self._wm.dynamics, self._task_behavior.actor, latent, first_action, k,
                    memory=policy_memory,
                    memory_transition=self._wm.search_b.imagine_step
                    if self._wm.search_b is not None and self._config.search_b_control else None)
                result = prefix_candidate_predictions(
                    self._wm.search_a, feat, sequence, history[:, 8:].argmax(-1),
                    mem if self._config.search_b_control else None,
                    self._config.search_a_min_labels, self._config.search_a_min_positives)
                # Use current A outputs, not stale collection-time predictions.
                for i, record in enumerate(records):
                    current = record[context_key].copy()
                    current[1] = float(result[2][i])
                    current[2] = float(result[3][3])
                    record[context_key] = current
            snapshots[endpoint] = (latent, policy_memory, sequence)
            return state, result[4]
        state, prediction = inputs("start")
        next_state, next_prediction = inputs("end")
        from search_feedback import imagined_search_transitions
        imagined = imagined_search_transitions(self, snapshots["start"], state, prediction, records)
        return self._search_c.train_transitions(
            state, prediction, next_state, next_prediction, records, imagined=imagined)

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    if getattr(config, "search_vnext_enabled", False):
        generator = sample_episodes_vnext(
            episodes, config.batch_length, slots=config.vnext_slots, seed=config.seed)
        return tools.from_generator(generator, config.batch_size)
    generator = (sample_episodes_with_history(
        episodes, config.batch_length, slots=config.search_b_slots,
        lookback=config.search_b_lookback, save_every=config.search_b_save_every,
        observation_keys=(("image", "heatmap", "inventory", "inventory_max", "equipped", "health",
                           "hunger", "breath", "obs_reward") if config.search_b_model_version >= 3 else ())) if config.search_b_enabled
        else tools.sample_episodes(episodes, config.batch_length))
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "minedojo":
        import envs.minedojo as minedojo
        log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        collect_labels = ((mode == "train" or config.search_eval_evidence) and (
            (config.search_a_enabled and config.search_a_train
             and config.search_a_loss_weight > 0 and (config.search_a_event_weight > 0
                  or (config.search_a_model_version >= 3 and config.search_a_visible_weight > 0)))
            or (config.search_b_enabled and config.search_b_loss_weight > 0)
            or config.search_c_enabled or getattr(config, "search_vnext_enabled", False)))
        kwargs = dict(
            log_dir=log_dir, target_item=config.target_item,
            search_evidence_specs=dict(
                enabled=collect_labels, target=config.target_item,
                distance=config.search_a_ray_distance, cell_size=config.search_a_cell_size,
                revisit_gap=config.search_a_horizon,
                version=3 if getattr(config, "search_vnext_enabled", False) else 2),
            simulator_seed=(config.env_seed + id + (100000 if mode == "eval" else 0)
                            if config.env_seed >= 0 else None))

        from envs.tasks import get_specs
        _, specs, _ = get_specs(task, target_item=config.target_item)
        if getattr(config, "search_vnext_enabled", False):
            kwargs["LS_Imagine_specs"] = dict(specs["LS_Imagine_specs"], vnext=True,
                                               vnext_world_seed=kwargs["simulator_seed"])
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

    from envs.recovery import ResetRecovery
    return ResetRecovery(env, retries=config.minedojo_reset_retries,
                         log_path=pathlib.Path(config.logdir) / "env_recovery.jsonl",
                         label=f"{mode}-{id}")


def save_active_replay(envs, cache, directory):
    """Persist unfinished real prefixes without inventing terminal outcomes."""
    active = {getattr(env, "id", None): cache[getattr(env, "id", None)]
              for env in envs if getattr(env, "id", None) in cache}
    if active:
        tools.save_episodes(directory, active)


def main(config): # config is namespace

    from training_state import (prepare_resume, restore_training_state,
                                write_resume_manifest, save_training_checkpoint)
    resume_plan = prepare_resume(config)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S') + ("_resume" if resume_plan else "")
    logdir = logdir / timestamp
    if logdir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {logdir}")
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
    step = resume_plan["environment_step"] if resume_plan else count_steps(config.traindir)

    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_traindir: # False
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir

    train_eps = (resume_plan["replay"] if resume_plan else
                 tools.load_episodes(directory, limit=config.dataset_size))

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

    if not config.offline_traindir and resume_plan is None:
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

    if resume_plan:
        restore_training_state(agent, resume_plan, logger)
        write_resume_manifest(resume_plan, config)
        print(f"Resume model at {resume_plan['model_step']}; "
              f"real interaction budget already used: {agent._step}; "
              f"total target: {int(config.steps)}. Skipping random prefill/pretrain.", flush=True)
    else:
        replay_source = (pathlib.Path(config.offline_traindir.format(**vars(config)))
                         if config.offline_traindir else config.traindir)
        agent._replay_sources = [str(replay_source.resolve())]
        agent._replay_budget_offset = 0

    # Branch interactions share the same total real-interaction budget.
    next_branch_step = (resume_plan["next_branch_step"] if resume_plan else
                       (agent._step // config.search_branch_every + 1) * config.search_branch_every)
    try:
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
                agent.flush_metrics()
                break
            if config.search_branch_enabled and agent._step >= next_branch_step:
                from search_branches import collect_real_branches
                collect_real_branches(agent, config, train_eps, logger, next_branch_step // config.search_branch_every)
                next_branch_step += config.search_branch_every
                save_training_checkpoint(agent, config, next_branch_step)
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

            save_active_replay(train_envs, train_eps, config.traindir)
            save_training_checkpoint(agent, config, next_branch_step)
    except BaseException as error:
        try:
            save_active_replay(train_envs, train_eps, config.traindir)
        except Exception as replay_error:
            print(f"[replay] Emergency prefix save failed: {replay_error}", flush=True)
        try:
            path = save_training_checkpoint(agent, config, next_branch_step, name="interrupted.pt")
            print(f"[checkpoint] Saved interrupted training state: {path}", flush=True)
        except Exception as save_error:
            print(f"[checkpoint] Emergency save failed: {save_error}", flush=True)
        raise
    finally:
        try:
            agent.flush_metrics()
            logger.finish()
        except Exception as log_error:
            print(f"[logger] Final flush failed: {log_error}", flush=True)
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


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
