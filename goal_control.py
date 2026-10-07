"""T05 diagnostic real goal execution, isolated from training/ScoreStorage.

Goals are real endpoints of short reference executions. Class IDs only
describe them; they never determine success or terminate execution.
"""

import copy
import json
from pathlib import Path
import random

import numpy as np
import torch

import goal_bc as bc
import goal_library as gl
import goal_segments as gs
import long_horizon as lh


FORMAT = "ls_imagine_goal_control_benchmark_v1"
WORLD_SEED_POLICY = "case_seed_plus_one_v1"
ENVIRONMENT_PROTOCOL = "fresh_task_fixed_pose_nonzero_world_seed_clear_weather_frozen_time_no_fast_reset_task_visual_preprocessing_v3"
VISUAL_PREPROCESSING_POLICY = "preserve_task_screenshot_wrapper_disable_file_output_v1"
MODES = ("goal", "zero_goal", "shuffled_goal", "original")
OBS_KEYS = ("image", "heatmap", "obs_reward", "is_first", "is_last", "is_terminal")
PAIR_LIMITS = {"rgb_mae": 1.0, "rgb_p99": 8.0, "heatmap_mae": 0.25,
               "state_relative_l2": 1e-4, "position": 0.05, "angle": 0.25,
               "reward": 1e-6}


def make_scenario(seed, start_position=None):
    """Keep experiment RNG seed separate from Malmo's nonzero world seed.

    BiomeGenerator/DefaultWorldGenerator treat numeric world seed zero as a
    request to retain a randomly generated seed. Map every case bijectively
    to seed+1, and persist that mapping for discovery, references and trials.
    """
    gl.require(type(seed) is int and 0 <= seed < 2**31 - 10000, "T05 实验种子必须为合法非负整数")
    scenario = {"seed": seed, "world_seed": str(seed + 1), "world_seed_policy": WORLD_SEED_POLICY}
    if start_position is not None:
        scenario["start_position"] = copy.deepcopy(start_position)
    return scenario


def validate_scenario(scenario):
    expected = make_scenario(scenario["seed"])
    gl.require(scenario.get("world_seed_policy") == WORLD_SEED_POLICY and
               scenario.get("world_seed") == expected["world_seed"],
               "T05 世界种子必须明确保存为 seed+1；world_seed=0 会随机生成世界，旧 scenario 需要重新 prepare")
    return scenario["world_seed"]


def visual_preprocessing_specs(task_specs):
    """Disable screenshot files without removing the task's RGB processing.

    ScreenshotWrapper also removes HUD/hand pixels before MineCLIP and WM
    see them. Dropping this wrapper changes inputs relative to T04 replay.
    Keep its presence, HUD setting and all non-output options unchanged.
    """
    specs = copy.deepcopy(task_specs.get("screenshot_specs"))
    if specs:
        specs.update(reset_flag=False, step_flag=False)
    return specs


def load_model(payload, cache, device):
    """Strict inference load without constructing/restoring any optimizer."""
    gl.require(payload.get("checkpoint_format") == bc.CHECKPOINT_FORMAT and
               payload.get("stage") == "t04_offline_bc" and payload.get("experiment_mode") == "goal_worker" and
               payload.get("worker_architecture") == "expanded_actor_v1" and
               payload.get("candidate_semantics") == "current_state_future_class_forecast" and
               not payload.get("verification_artifact"), "T05 需要真实 T04 checkpoint，不能用原 agent 或合成验收文件")
    gl.require(payload["cache_id"] == cache.cache_id, "T05 checkpoint 与状态缓存不同")
    bc.validate_bundle(payload["frozen_bundle"])
    gl.require(payload["frozen_bundle"]["bundle_id"] == cache.bundle["bundle_id"], "T05 冻结模型/目标库不同")
    options, counters = payload["options"], payload["counters"]
    bc.validate_checkpoint_sampling(payload)
    step = counters["step"]
    gl.require(type(step) is int and step > 0 and counters == dict(step=step, worker_updates=step,
        candidate_updates=step, worker_version=step, worker_labels_seen=step * options["batch_size"],
        candidate_labels_seen=step * options["batch_size"], new_env_steps=0), "T05 checkpoint 训练计数异常")
    gl.require(options["conditioning"] in ("goal", "no_goal"), "未知 BC 条件模式")
    # Constructors initialize temporary modules. Do not change caller RNG.
    rng = bc.capture_rng(device)
    try:
        model = bc.GoalBCModel(cache.bundle, options, device)
    finally:
        bc.restore_rng(rng, device)
    for name in ("worker", "candidate"):
        module = getattr(model, name)
        lh.require_compatible(module.state_dict(), payload[name])
        gl.require(all(bool(torch.isfinite(v).all()) for v in payload[name].values()), "T05 权重含非有限值")
        module.load_state_dict(payload[name], strict=True)
    return model.requires_grad_(False).eval()


def model_identity(model):
    return gl.tensor_digest(model.state_dict(), {"bundle_id": model.bundle["bundle_id"], "options": model.options})


def angle_error(a, b):
    return float(abs((float(a) - float(b) + 180) % 360 - 180))


def telemetry(observation):
    """Read simulator measurements before LSImagineWrapper drops them."""
    loc, life = observation.get("location_stats", {}), observation.get("life_stats", {})
    pos = loc.get("pos")
    pose = None
    if pos is not None and np.asarray(pos).shape == (3,) and all(key in loc for key in ("yaw", "pitch")):
        values = [*np.asarray(pos).tolist(), float(np.asarray(loc["yaw"]).item()), float(np.asarray(loc["pitch"]).item())]
        if np.isfinite(values).all():
            pose = dict(zip(("x", "y", "z", "yaw", "pitch"), values))
    inventory = None
    inv = observation.get("inventory", {})
    if "name" in inv and "quantity" in inv:
        names, quantities = np.asarray(inv["name"]).reshape(-1), np.asarray(inv["quantity"]).reshape(-1)
        gl.require(len(names) == len(quantities) and np.isfinite(quantities).all(), "物品遥测长度或数值异常")
        inventory = {}
        for name, quantity in zip(names, quantities):
            name = str(name).replace(" ", "_")
            if float(quantity) > 0 and name not in ("air", "none"):
                inventory[name] = inventory.get(name, 0.0) + float(quantity)
    health = float(np.asarray(life["life"]).item()) if "life" in life else None
    gl.require(health is None or np.isfinite(health), "生命遥测非有限")
    return {"pose": pose, "inventory": inventory, "health": health}


def pose_difference(left, right):
    if left is None or right is None:
        return None
    position = float(np.linalg.norm([left[k] - right[k] for k in ("x", "y", "z")]))
    return {"position": position, "yaw": angle_error(left["yaw"], right["yaw"]),
            "pitch": abs(left["pitch"] - right["pitch"])}


def onehot(action, dimension):
    gl.require(type(action) is int and 0 <= action < dimension, "真实动作编号越界")
    result = np.zeros(dimension, dtype=np.float32)
    result[action] = 1
    return result


def select_action(probabilities, policy, uniform):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gl.require(probabilities.ndim == 1 and np.isfinite(probabilities).all() and
               np.all(probabilities >= 0) and np.isclose(probabilities.sum(), 1, atol=1e-5), "动作概率异常")
    gl.require(policy in ("mode", "sample") and 0 <= uniform < 1, "执行策略或共同随机数错误")
    if policy == "mode":
        return int(probabilities.argmax())
    return min(int(np.searchsorted(np.cumsum(probabilities / probabilities.sum()), uniform, side="right")), len(probabilities) - 1)


@torch.no_grad()
def encode_goal(model, observation):
    device = next(model.parameters()).device
    obs = {key: torch.as_tensor(np.asarray(observation[key])[None], device=device) for key in ("image", "heatmap")}
    result = model.library(obs)[0].cpu().numpy()
    gl.require(np.isfinite(result).all() and np.isclose(np.linalg.norm(result), 1, atol=1e-5), "真实目标编码异常")
    return result


@torch.no_grad()
def probabilities(model, features, goal, remaining, mode):
    gl.require(mode in (*MODES, "no_goal_bc"), "未知 T05 对照")
    goals = torch.as_tensor(np.asarray(goal)[None], device=features.device)
    steps = torch.tensor([remaining], device=features.device, dtype=torch.int64)
    branch = "original" if mode == "original" else "zero_goal" if mode == "zero_goal" else "trained"
    return model.action_dist(features, goals, steps, branch).probs[0].cpu().numpy()


def prefix_comparison(reference, actual, limits=PAIR_LIMITS):
    """Check the whole real history, including WM state and raw pose/items."""
    result = {"passed": True, "limits": dict(limits), "missing_telemetry": False}
    gl.require(len(reference["image"]) == len(actual["image"]), "配对历史长度不同")
    rgb = np.abs(reference["image"].astype(float) - actual["image"].astype(float)).reshape(len(reference["image"]), -1)
    heat = np.abs(reference["heatmap"].astype(float) - actual["heatmap"].astype(float)).reshape(len(rgb), -1)
    result.update(rgb_mae=float(rgb.mean(1).max()), rgb_p99=float(np.percentile(rgb, 99, axis=1).max()),
                  heatmap_mae=float(heat.mean(1).max()),
                  reward=float(np.abs(reference["obs_reward"] - actual["obs_reward"]).max()))
    ref_feat, feat = reference["features"], actual["features"]
    result["state_relative_l2"] = float((np.linalg.norm(ref_feat - feat, axis=1) / np.maximum(np.linalg.norm(ref_feat, axis=1), 1e-8)).max())
    result.update(position=0.0, angle=0.0, inventory_equal=True, health_equal=True, flags_equal=True, actions_equal=True)
    for left, right in zip(reference["telemetry"], actual["telemetry"]):
        delta = pose_difference(left["pose"], right["pose"])
        if delta is None or left["inventory"] is None or right["inventory"] is None or left["health"] is None or right["health"] is None:
            result["missing_telemetry"] = True
        else:
            result["position"] = max(result["position"], delta["position"])
            result["angle"] = max(result["angle"], delta["yaw"], delta["pitch"])
        result["inventory_equal"] &= left["inventory"] == right["inventory"]
        result["health_equal"] &= left["health"] == right["health"]
    for key in ("is_first", "is_last", "is_terminal"):
        result["flags_equal"] &= bool(np.array_equal(reference[key], actual[key]))
    result["actions_equal"] = bool(np.array_equal(reference["action"], actual["action"]))
    boolean_checks = ("inventory_equal", "health_equal", "flags_equal", "actions_equal")
    result["failed_checks"] = [key for key, limit in limits.items() if result[key] > limit]
    result["failed_checks"] += [key for key in boolean_checks if not result[key]]
    if result["missing_telemetry"]:
        result["failed_checks"].append("missing_telemetry")
    result["passed"] = bool(not result["missing_telemetry"] and all(result[key] <= limit for key, limit in limits.items()) and
                            all(result[key] for key in boolean_checks))
    # Preserve the original aggregate checks and thresholds, and expose the
    # first diverging observation rather than only the maximum over 33 frames.
    state_errors = np.linalg.norm(ref_feat - feat, axis=1) / np.maximum(np.linalg.norm(ref_feat, axis=1), 1e-8)
    frames = []
    for index, (left, right) in enumerate(zip(reference["telemetry"], actual["telemetry"])):
        delta = pose_difference(left["pose"], right["pose"])
        frame = {"frame": index, "is_reset": index == 0, "rgb_mae": float(rgb[index].mean()),
                 "rgb_p99": float(np.percentile(rgb[index], 99)), "heatmap_mae": float(heat[index].mean()),
                 "state_relative_l2": float(state_errors[index]),
                 "reward": float(np.abs(reference["obs_reward"][index] - actual["obs_reward"][index]).max()),
                 "position": 0.0 if delta is None else delta["position"],
                 "angle": 0.0 if delta is None else max(delta["yaw"], delta["pitch"]),
                 "inventory_equal": left["inventory"] == right["inventory"], "health_equal": left["health"] == right["health"],
                 "flags_equal": all(np.array_equal(reference[key][index], actual[key][index]) for key in ("is_first", "is_last", "is_terminal")),
                 "actions_equal": bool(np.array_equal(reference["action"][index], actual["action"][index])),
                 "missing_telemetry": delta is None or any(item[key] is None for item in (left, right) for key in ("inventory", "health"))}
        frame["failed_checks"] = [key for key, limit in limits.items() if frame[key] > limit]
        frame["failed_checks"] += [key for key in boolean_checks if not frame[key]]
        if frame["missing_telemetry"]:
            frame["failed_checks"].append("missing_telemetry")
        frame["passed"] = bool(not frame["missing_telemetry"] and all(frame[key] <= limit for key, limit in limits.items()) and
                               all(frame[key] for key in boolean_checks))
        frames.append(frame)
    result["per_frame"] = frames
    result["first_mismatch_frame"] = next((frame["frame"] for frame in frames if not frame["passed"]), None)
    return result


def environment_fingerprint(root):
    """Bind real heatmap execution to the repository code and actual weights."""
    from importlib import metadata
    paths = sorted((Path(root) / "envs").rglob("*.py")) + [Path(root) / "envs/tasks/task_specs.yaml", Path(root) / "envs/tasks/base/HUD_mask.png", Path(root) / "weights/mineclip_attn.pth",
                                                          Path(root) / "goal_control.py", Path(root) / "scripts/t05_goal_control.py"]
    result = {str(path.relative_to(root)): bc.file_hash(path) for path in paths}
    for package in ("minedojo", "mineclip", "torch", "opencv-python"):
        try:
            result["package:" + package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result["package:" + package] = None
    return result


def make_environment(bundle, scenario, log_dir):
    """Fresh task/simulator for every arm; no semifast reset/teleport reuse.

    These kwargs are supported by HarvestMeta as well as MineDojoSim.
    Do not pass sim-only kwargs (e.g. start_time) through HarvestMeta.
    """
    world_seed = validate_scenario(scenario)
    import gym
    from envs.tasks import get_specs
    from envs.tasks import minedojo as factory
    from envs.tasks.base.ls_imagine_wrapper import LSImagineWrapper
    from envs.tasks.base.screenshot_wrapper import ScreenshotWrapper

    config = bundle["config"]
    suite, _, task = config["task"].partition("_")
    gl.require(suite == "minedojo", "T05 当前仅支持 MineDojo")
    task_id, task_specs, sim_specs = copy.deepcopy(get_specs(task, target_item=config["target_item"]))
    task_specs["fast_reset"] = None
    task_specs["log_dir"] = str(log_dir)
    task_specs["screenshot_specs"] = visual_preprocessing_specs(task_specs)
    task_specs["concentration_specs"].update(max_steps=config["episode_max_steps"],
        gaussian_reward_weight=config["gaussian_reward_weight"], gaussian_sigma_weight=config["gaussian_sigma_weight"])
    success = task_specs["success_specs"]
    task_specs["clip_specs"]["target_object"] = success["all"]["item"]["type"] if "all" in success else success["any"]["item"]["type"]
    sim_specs.update(world_seed=world_seed, seed=scenario["seed"], fast_reset=False,
                     initial_weather="clear", allow_time_passage=False)
    if scenario.get("start_position") is not None:
        sim_specs["start_position"] = copy.deepcopy(scenario["start_position"])
    # The existing factory pops fields in the global task catalogue. Isolate
    # that mutation as well as get_specs' nested dictionaries in this tool.
    saved_spec = copy.deepcopy(factory.ALL_TASKS_SPECS[task_id]) if task_id in factory.ALL_TASKS_SPECS else None
    try:
        env = factory.make_minedojo(task_id, task_specs, sim_specs)
    finally:
        if saved_spec is not None:
            factory.ALL_TASKS_SPECS[task_id] = saved_spec

    screenshot_specs = task_specs["screenshot_specs"]
    remove_hud = bool(screenshot_specs) and not bool(screenshot_specs.get("HUD", False))
    try:
        if screenshot_specs:
            screenshot = env
            while not isinstance(screenshot, ScreenshotWrapper):
                gl.require(hasattr(screenshot, "env"), "T05 缺少任务原有 ScreenshotWrapper；不能改变训练时的 RGB 预处理")
                screenshot = screenshot.env
            gl.require(bool(screenshot.HUD) == bool(screenshot_specs.get("HUD", False)) and
                       not screenshot.reset_flag and not screenshot.step_flag,
                       "T05 必须保留任务 HUD 设置，仅关闭截图文件输出")
            if remove_hud:
                gl.require(screenshot.mask is not None and screenshot.mask.shape == screenshot.get_resolution(),
                           "T05 HUD mask 缺失或尺寸不匹配")
    except BaseException:
        env.close()
        raise
    visual = {"policy": VISUAL_PREPROCESSING_POLICY, "screenshot_wrapper": bool(screenshot_specs),
              "remove_hud": remove_hud, "screenshot_file_output": False,
              "screenshot_specs": copy.deepcopy(screenshot_specs)}
    print(f"[ENV] visual_preprocessing screenshot_wrapper={int(bool(screenshot_specs))} "
          f"remove_hud={int(remove_hud)} screenshot_file_output=0", flush=True)

    class Tap(gym.Wrapper):
        def capture(self, obs):
            self.current = telemetry(obs)
            self.rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0).astype(np.uint8, copy=True)

        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            self.capture(obs)
            self.executed = []
            return obs

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.capture(obs)
            self.executed.append(copy.deepcopy(action))
            return obs, reward, done, info

    wrapped = env
    while not isinstance(wrapped, LSImagineWrapper):
        gl.require(hasattr(wrapped, "env"), "找不到真实 RGB/遥测采集位置")
        wrapped = wrapped.env
    tap = Tap(wrapped.env)
    wrapped.env = tap
    gl.require(env.action_space.n == config["num_actions"], "T05 动作空间不兼容")
    return env, tap, {"task_id": task_id, "task_specs": task_specs, "sim_specs": sim_specs,
                      "visual_preprocessing": visual}


class Session:
    """Online RSSM carries only actual observation/action history."""
    def __init__(self, model, scenario, log_dir, on_step):
        self.model, self.scenario, self.on_step = model, scenario, on_step
        # Common Python/NumPy seeds help task hooks; empirical prefix checks
        # remain mandatory and are the evidence for matching starts.
        random.seed(scenario["seed"])
        np.random.seed(scenario["seed"])
        torch.manual_seed(scenario["seed"])
        self.env, self.tap, self.specs = make_environment(model.bundle, scenario, log_dir)
        self.rows, self.events, self.video = [], [], []
        self.state = None
        self.done = False
        try:
            obs = self.env.reset()
            self.record(obs, np.zeros(model.bundle["config"]["num_actions"], np.float32), 0.0, {})
        except BaseException:
            self.close()
            raise

    def record(self, obs, incoming, reward, info):
        obs = dict(obs, obs_reward=np.array([reward], dtype=np.float32))
        self.state, self.features = self.model.state_encoder.step(obs, incoming, self.state)
        self.observation = obs
        self.rows.append({key: np.array(obs[key], copy=True) for key in OBS_KEYS})
        self.rows[-1].update(action=incoming.copy(), features=self.features[0].cpu().numpy().copy())
        self.events.append({"frame": len(self.rows) - 1, "reward": float(reward), "success": bool(info.get("success", False)),
                            "done": bool(self.done), "error": str(info["error"]) if "error" in info else None,
                            "telemetry": copy.deepcopy(self.tap.current),
                            "native_actions": copy.deepcopy(self.tap.executed)})
        self.tap.executed.clear()
        self.video.append(self.tap.rgb.copy())
        gl.require("error" not in info, f"环境执行错误: {info.get('error')}")

    def step(self, action):
        gl.require(not self.done, "T05 禁止在真实结束后继续执行")
        obs, reward, done, info = self.env.step(int(action))
        self.on_step()
        self.done = bool(done or obs["is_last"] or obs["is_terminal"])
        self.record(obs, onehot(int(action), self.model.bundle["config"]["num_actions"]), reward, info)

    def arrays(self, stop=None):
        rows = self.rows if stop is None else self.rows[:stop + 1]
        result = {key: np.stack([row[key] for row in rows]) for key in (*OBS_KEYS, "action", "features")}
        result["telemetry"] = [event["telemetry"] for event in self.events[:len(rows)]]
        return result

    def close(self):
        self.env.close()


def save_session(session, directory, write_video):
    directory = Path(directory)
    # The environment may already have created its log subdirectory.
    # Diagnostic files themselves must never overwrite an earlier trial.
    gl.require(not any((directory / name).exists() for name in ("trajectory.npz", "events.json", "video.mp4")),
               "拒绝覆盖已有真实诊断轨迹")
    directory.mkdir(parents=True, exist_ok=True)
    arrays = session.arrays()
    del arrays["telemetry"]
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    # Only diagnostic real data, never inserted into the training replay.
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    (directory / "events.json").write_text(json.dumps(plain(session.events), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return write_video(directory / "video.mp4", np.stack(session.video))


def read_trajectory(directory):
    directory = Path(directory)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as file:
        result = {key: file[key].copy() for key in file.files}
    events = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    gl.require(len(events) == len(result["image"]), "真实轨迹与遥测长度不匹配")
    if len(events) == 1:
        # A failed pairing with zero warmup still has a useful reset frame.
        # It is diagnostic data only, never a BC segment/trajectory label.
        gl.require(result["image"].shape == (1, 64, 64, 3) and result["image"].dtype == np.uint8 and
                   result["heatmap"].shape == (1, 64, 64) and result["heatmap"].dtype == np.uint8 and
                   bool(result["is_first"][0]) and not bool(result["is_last"][0]) and not bool(result["is_terminal"][0]) and
                   np.all(result["action"] == 0) and np.all(result["obs_reward"] == 0), "单帧诊断必须为真实 reset")
    else:
        gs.validate_episode(result, result["action"].shape[1])
    gl.require(result["features"].ndim == 2 and len(result["features"]) == len(events) and
               np.isfinite(result["features"]).all(), "真实轨迹状态数组异常")
    result["telemetry"] = [event["telemetry"] for event in events]
    return result


def distances(features, goals):
    return np.clip(1 - np.asarray(features) @ np.asarray(goals).T, 0, 2)


def trial_metrics(trajectory, goals, target_index, prefix_steps, target_telemetry):
    encoded = trajectory["goal_features"][prefix_steps:]
    d = distances(encoded, goals)
    target = d[:, target_index]
    other = 1 - target_index
    start, end = trajectory["telemetry"][prefix_steps], trajectory["telemetry"][-1]
    inventory_delta = None
    if start["inventory"] is not None and end["inventory"] is not None:
        inventory_delta = {k: end["inventory"].get(k, 0) - start["inventory"].get(k, 0)
                           for k in sorted(set(start["inventory"]) | set(end["inventory"]))}
    pose = pose_difference(end["pose"], target_telemetry["pose"])
    return {"start_distance": float(target[0]), "end_distance": float(target[-1]),
            "best_distance": float(target.min()), "distance_improvement": float(target[0] - target[-1]),
            "relative_distance_improvement": float((target[0] - target[-1]) / max(float(target[0]), 1e-8)),
            "target_preference_margin": float(d[-1, other] - d[-1, target_index]),
            "reference_pose_error": pose,
            "reference_pose_match": None if pose is None else bool(pose["position"] <= 1 and pose["yaw"] <= 15 and pose["pitch"] <= 15),
            "movement_from_start": pose_difference(end["pose"], start["pose"]),
            "inventory_delta": inventory_delta,
            "metric_scope": "visual endpoint/pose agreement; not semantic task completion"}
