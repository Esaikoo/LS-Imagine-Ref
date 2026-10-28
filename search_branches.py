"""Verified equivalent-scene real interventions, sharing the normal Replay/budget.

A fresh Minecraft client/world per branch; replay one common action prefix.
Audit-only simulator state never reaches the policy. Nonmatching scenes remain
ordinary real data and are explicitly NOT labeled as counterfactual pairs.
"""
import copy
import gc
import json
import random
from pathlib import Path
from datetime import datetime
import uuid
import numpy as np
import torch
import gym

from search_evidence import complete_episode_labels


def native_value(x):
    if isinstance(x, dict):
        return {str(k): native_value(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


class SceneAudit(gym.Wrapper):
    def remember(self, obs):
        from evaluate_checkpoints import terrain_fingerprint
        terrain = obs.get("voxels")
        if terrain is None:
            raise RuntimeError("Real branch comparison requires a terrain audit")
        fingerprint, raw = terrain_fingerprint(terrain["block_name"], terrain["block_meta"])
        inventory = obs.get("inventory", {})
        counts = {}
        for name, count in zip(inventory.get("name", []), inventory.get("quantity", [])):
            if int(count):
                counts[str(name)] = counts.get(str(name), 0) + int(count)
        import hashlib
        rays = obs.get("rays", {})
        ray_keys = ("block_name", "block_distance", "entity_name", "entity_distance",
                    "traced_block_x", "traced_block_y", "traced_block_z")
        if not all(key in rays for key in ray_keys):
            raise RuntimeError("Missing ray geometry for real branch audit")
        ray_data = {}
        for key in ray_keys:
            values = np.asarray(rays[key])
            if np.issubdtype(values.dtype, np.number):
                values = values.round(4)
            ray_data[key] = native_value(values)
        ray_hash = hashlib.sha256(json.dumps(ray_data, sort_keys=True).encode()).hexdigest()
        self.scene = dict(terrain_sha256=fingerprint, terrain_raw_sha256=raw,
                          rays_sha256=ray_hash,
                          inventory=counts, life_stats=native_value(obs.get("life_stats", {})),
                          location_stats=native_value(obs.get("location_stats", {})))
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.remember(obs)
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.remember(obs)
        return obs, reward, done, info


def compare_branch_scenes(left, right, left_prefix=None, right_prefix=None):
    """Exact world/pose/ray checks plus bounded renderer noise, never outcome filtering."""
    from evaluate_checkpoints import compare_initial_scenes
    result = compare_initial_scenes(left, right)
    checks = result["checks"]
    checks["ray_geometry"] = (left.get("rays_sha256") is not None
                              and left["rays_sha256"] == right.get("rays_sha256"))
    if left_prefix is None or right_prefix is None:
        checks["policy_rgb"] = left["image_sha256"] == right["image_sha256"]
        checks["policy_heatmap"] = left["heatmap_sha256"] == right["heatmap_sha256"]
        checks["common_history"] = left.get("history_sha256") == right.get("history_sha256")
    else:
        visual = {}
        for key, mean_limit, percentile_limit in (("image", 2., 8.), ("heatmap", 1., 2.)):
            a, b = np.asarray(left_prefix[key]), np.asarray(right_prefix[key])
            if a.shape != b.shape:
                checks["history_" + key] = False
                continue
            delta = np.abs(a.astype(float)-b.astype(float)).reshape(len(a), -1)
            means, p95 = delta.mean(-1), np.quantile(delta,.95,axis=-1)
            checks["history_" + key] = bool((means <= mean_limit).all() and (p95 <= percentile_limit).all())
            visual[key] = dict(max_frame_mae=float(means.max()), max_frame_p95=float(p95.max()),
                               mae_limit=mean_limit, p95_limit=percentile_limit, range=255)
        for key in ("action", "obs_reward", "inventory", "inventory_max", "equipped", "health", "hunger", "breath"):
            if key in left_prefix or key in right_prefix:
                checks["history_" + key] = bool(key in left_prefix and key in right_prefix
                    and np.array_equal(left_prefix[key], right_prefix[key]))
        result["visual_comparison"] = visual
    result["match"] = all(checks.values())
    result["protocol_version"] = 2
    return result


def make_branch_env(config, seed, folder, name, steps):
    from envs.tasks import get_specs
    from envs.tasks.minedojo import make_minedojo
    from envs.tasks.base.ls_imagine_wrapper import LSImagineWrapper
    from envs import wrappers
    from evaluate_checkpoints import seed_clean_bridge
    task = config.task.split("_", 1)[1]
    task_id, task_specs, sim_specs = copy.deepcopy(get_specs(task, target_item=config.target_item))
    task_specs.update(fast_reset=None, log_dir=str(folder),
        search_evidence_specs=dict(enabled=True, target=config.target_item,
            distance=config.search_a_ray_distance, cell_size=config.search_a_cell_size,
            revisit_gap=config.search_a_horizon))
    # Collection horizon must not change task reward/time-limit semantics.
    # The collector marks its own shorter end as a nonterminal truncation.
    task_specs["terminal_specs"].update(max_steps=config.episode_max_steps, stagger_max_steps=False)
    task_specs["concentration_specs"].update(max_steps=config.episode_max_steps,
        gaussian_reward_weight=config.gaussian_reward_weight,
        gaussian_sigma_weight=config.gaussian_sigma_weight)
    success = task_specs["success_specs"].get("all", task_specs["success_specs"].get("any"))
    task_specs["clip_specs"]["target_object"] = success["item"]["type"]
    sim_specs.update(seed=seed, world_seed=str(seed), allow_time_passage=False,
        initial_weather="clear", use_voxel=True,
        voxel_size=dict(xmin=-5, xmax=5, ymin=-2, ymax=2, zmin=-5, zmax=5))
    env = make_minedojo(task_id, task_specs, sim_specs)
    seed_clean_bridge(env, seed, folder, name)
    current = env
    while not isinstance(current, LSImagineWrapper):
        current = current.env
    audit = SceneAudit(current.env)
    current.env = audit
    env = wrappers.RewardObs(wrappers.SelectAction(wrappers.OneHotAction(env), "action"))
    return env, audit


def scene_record(audit, obs, env, seed):
    from evaluate_checkpoints import locate_sim
    import hashlib
    return dict(audit.scene, world_seed=seed, mission_seed=seed,
                java_launch_seed=int(locate_sim(env)._bridge_env._instances[0]._seed),
                policy_observation_keys=sorted(obs),
                image_sha256=hashlib.sha256(np.asarray(obs["image"]).tobytes()).hexdigest(),
                heatmap_sha256=hashlib.sha256(np.asarray(obs["heatmap"]).tobytes()).hexdigest())


@torch.no_grad()
def collect_real_branches(agent, config, replay, logger, group):
    import tools
    count, prefix, k = config.search_branch_count, config.search_branch_prefix, config.search_a_commit_steps
    if not 2 <= count <= 8 or prefix < 0:
        raise ValueError("Real branch collection needs 2..8 options and a nonnegative common prefix")
    required = count * (prefix + k)
    if config.steps - agent._step < required:
        logger.scalar("C/branch_skipped_budget", 1)
        return 0
    seed = int(config.search_branch_seed + group)
    folder = Path(config.logdir) / "branches" / ("%06d" % group)
    folder.mkdir(parents=True, exist_ok=True)
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    controller = agent._search_a
    agent._search_a = None  # fixed experimental actions, not controller-selected outcomes
    start_step = agent._step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    options = [5] + [int(i) for i in np.random.RandomState(seed).permutation([0, 1, 2, 3, 4, 6, 7])[:count-1]]
    common = [6, 6, 4, 6, 6, 5, 6, 6]
    episodes, report, scenes = {}, [], []
    try:
        for option in options:
            print("Real branch", group, "requested option", option, "seed", seed, flush=True)
            tools.set_seed_everywhere(seed)
            env = None
            name = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
            local = {}
            branch_scene = None
            saved_prediction = None
            frames = []
            try:
                env, audit = make_branch_env(config, seed, folder, name, prefix+k)
                obs = env.reset()
                first = dict(obs, reward=np.float32(0), discount=np.float32(1))
                tools.add_to_cache(local, name, first)
                state = None
                control_state = None
                actual_choice = option
                class Intervention:
                    model_version = agent._search_c.model_version if agent._search_c is not None else 0
                    option = 5
                    @staticmethod
                    def choose(state, predictions, context, mask, training):
                        choice = torch.full((len(mask),), selector.option, device=mask.device, dtype=torch.long)
                        choice = torch.where(mask[:, selector.option].bool(), choice, torch.full_like(choice, 5))
                        return choice, torch.ones(len(mask), device=mask.device)
                selector = Intervention()
                for t in range(prefix+k):
                    batched = {key: np.asarray(value)[None] for key, value in obs.items() if not key.startswith("log_")}
                    output, state = agent._policy(batched, state, training=False)
                    latent = state[0]
                    full_memory = state[3][-1] if config.search_b_enabled else None
                    memory = (full_memory if config.search_b_control else
                              torch.zeros_like(full_memory) if full_memory is not None else None)
                    feat = agent._wm.dynamics.get_feat(latent)
                    # Reuse the actual executor, including interruption/protection rules.
                    # Only the first macro choice is externally intervened on.
                    from search_a import actor_action_sequence
                    def continuation(indices):
                        return actor_action_sequence(agent._wm.dynamics, agent._task_behavior.actor,
                            {key: value[indices] for key, value in latent.items()},
                            output["action"][indices], config.search_a_horizon,
                            memory=memory[indices] if memory is not None else None,
                            memory_transition=agent._wm.search_b.imagine_step
                                if memory is not None and config.search_b_control else None)
                    policy_obs = agent._wm.preprocess({key: value for key, value in batched.items()
                                                       if not key.startswith("search_")})
                    if t == prefix:
                        if control_state is not None:
                            control_state = list(control_state)
                            control_state[0] = torch.full_like(control_state[0], -1)
                            control_state[1] = torch.zeros_like(control_state[1])
                            control_state = tuple(control_state)
                        selector.option = option
                        branch_scene = scene_record(audit, obs, env, seed)
                        import hashlib
                        history = hashlib.sha256()
                        for row in range(len(local[name]["image"])):
                            for key in ("image", "heatmap", "action", "obs_reward",
                                        "inventory", "inventory_max", "equipped", "health", "hunger", "breath"):
                                if key in local[name]:
                                    history.update(np.asarray(local[name][key][row]).tobytes())
                        branch_scene["history_sha256"] = history.hexdigest()
                        frames.append(np.asarray(obs["image"]).copy())
                    action_batch, control_state, info_control = controller.choose(
                        agent._wm.search_a, feat, policy_obs, output["action"], control_state,
                        policy_obs["is_first"], agent._step, False, continuation,
                        memory=memory if config.search_b_control else None,
                        selector=selector, policy_memory=memory)
                    if t > prefix and info_control["search_c_decision"][0] > 0:
                        # The real controller ended/interrupted this option before
                        # executing its next action. Close the preceding real transition.
                        break
                    if t == prefix:
                        actual_choice = int(info_control["search_c_choice"][0])
                        if "log_search_return_predictions" in info_control:
                            saved_prediction = dict(
                                returns=info_control["log_search_return_predictions"][0,actual_choice].cpu().tolist(),
                                reliability=info_control["log_search_reliability"][0,actual_choice].cpu().tolist(),
                                planned_actions=info_control["log_search_plan_actions"][0,:k].cpu().tolist())
                        selector.option = 5  # next boundary belongs to ordinary Actor
                    action = (torch.nn.functional.one_hot(torch.tensor(common[t % len(common)],
                              device=config.device), 12).float() if t < prefix else action_batch[0])
                    # Prefix actions also feed the controller's real action history.
                    control_state = list(control_state)
                    control_state[10] = action.argmax(-1)[None]
                    control_state = tuple(control_state)
                    state = tuple(action[None] if i == 1 else value for i, value in enumerate(state))
                    obs, reward, done, info = env.step({"action": action.cpu().numpy()})
                    agent._step += 1  # Count every real action, including the repeated prefix.
                    logger.step = agent._step * config.action_repeat
                    transition = dict(obs, action=action.cpu().numpy(), reward=np.float32(reward),
                        discount=np.float32(0 if bool(obs["is_terminal"]) else 1),
                        success=np.float32(info.get("success", False)), logprob=np.float32(0),
                        first_success_step=np.float32(info.get("first_success_step", config.episode_max_steps)),
                        search_data_source=np.float32(2), search_branch_group=np.float32(group),
                        search_branch_choice=np.float32(option), search_branch_start=np.float32(t == prefix))
                    if t >= prefix:
                        transition.update({key: value[0].cpu().numpy() for key, value in info_control.items()})
                        if agent._search_c is None:
                            transition["search_c_version"] = np.float32(0)
                    transition.update({key:value[0].cpu().numpy() for key,value in output.items()
                                       if key.startswith("log_b_")})
                    tools.add_to_cache(local, name, transition)
                    if done:
                        break
                # Explicit truncation, never an invented terminal or reward.
                local[name]["is_last"][-1] = np.array(True)
                frames.append(np.asarray(obs["image"]).copy())
                episode = complete_episode_labels({key: np.asarray(value) for key, value in
                    tools.replace_none_with_zeros(local[name]).items()})
                episodes[name] = episode
                scenes.append(branch_scene)
                episode["search_branch_choice"][:] = actual_choice
                factual = None
                duration = len(episode["reward"]) - 1 - prefix
                if branch_scene is not None and duration > 0:
                    from search_feedback import window_targets
                    tensor = {key: torch.as_tensor(value)[None] for key, value in episode.items()
                              if value.dtype != object}
                    targets, known, _, _ = window_targets(tensor, duration, config)
                    factual = dict(returns=targets[0, prefix].tolist(), known=known[0, prefix].tolist(),
                                   duration=duration, planned_duration=k,
                                   executed_actions=episode["action"][prefix+1:,:12].argmax(-1).tolist())
                    # Interrupted options have a shorter target; never compare
                    # their factual return to a different-horizon forecast.
                    factual["matches_prediction_duration"] = duration == k
                print("Real branch completed:", name, "choice", actual_choice,
                      "actions", len(episode["reward"])-1, flush=True)
                report.append(dict(option=actual_choice, requested_option=option, episode=name, steps=len(episode["reward"])-1,
                                   start=branch_scene, prediction=saved_prediction, actual=factual))
                from PIL import Image
                Image.fromarray(np.concatenate(frames, axis=1)).save(folder / ("option_%d.png" % option))
            finally:
                if env is not None:
                    env.close()
                env = None
                audit = None
                # The audit wrapper also owns references to MineCLIP/U-Net.
                # Release them BEFORE constructing the next branch environment.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        ref_index = next((i for i, scene in enumerate(scenes) if scene is not None), None)
        reference = scenes[ref_index] if ref_index is not None else None
        reference_episode = list(episodes.values())[ref_index] if ref_index is not None else None
        def common_data(episode):
            return {key: value[:prefix+1] for key,value in episode.items()}
        for (name, episode), record, scene in zip(episodes.items(), report, scenes):
            comparison = (compare_branch_scenes(reference, scene, common_data(reference_episode), common_data(episode))
                          if reference is not None and scene is not None else dict(match=False))
            record["comparison"] = comparison
            # At least two verified equal endpoints are required to claim paired data.
        matched = sum(record["comparison"]["match"] for record in report)
        diverse = len({record["option"] for record in report if record["comparison"]["match"]}) >= 2
        for (name, episode), record in zip(episodes.items(), report):
            verified = matched >= 2 and diverse and record["comparison"]["match"]
            episode["search_branch_verified"][:] = float(verified)
            replay[name] = episode
        tools.save_episodes(config.traindir, episodes)
        # Preserve any currently open rollout; normal simulation performs FIFO eviction.
        logger.scalar("C/branch_real_steps", agent._step-start_step)
        logger.scalar("C/branch_verified_options", matched if matched >= 2 and diverse else 0)
        logger.scalar("C/branch_mismatched_options", len(report)-matched)
        (folder / "feedback.json").write_text(json.dumps(dict(
            group=group, world_seed=seed, checkpoint_step=start_step,
            real_steps=agent._step-start_step, options=report,
            budget_note="All agent actions, including repeated prefixes, count toward config.steps; engine reset ticks follow normal MineDojo accounting."
        ), indent=2))
        logger.write()
    finally:
        agent._search_a = controller
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        if rng_state[3] is not None:
            torch.cuda.set_rng_state_all(rng_state[3])
    return agent._step-start_step
