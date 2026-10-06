"""T03 real-episode hindsight segments and causal frozen WM features.

The future observation supplies supervision only. State inputs are strictly
the real prefix from reset through the current observation, with incoming
action[t]; the BC label at that observation is outgoing action[t + 1].
"""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import goal_library as gl
import networks


FORMAT = "ls_imagine_goal_segments_v1"
STATE_POLICY = "posterior_mode_full_reset_prefix_fp32_v1"
HISTORY_KEYS = ("image", "heatmap", "obs_reward", "is_first", "action")


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_episode(path, max_steps, action_dim):
    episode = gl.read_episode(path, max_steps)
    with np.load(path, allow_pickle=False) as data:
        gl.require({"action", "obs_reward"} <= set(data.files), "片段回放缺少 action/obs_reward")
        length = episode["original_length"]
        action = data["action"]
        gl.require(action.shape == (length, action_dim) and np.isfinite(action).all(), "真实动作形状或数值错误")
        reward = data["obs_reward"]
        gl.require(reward.shape in ((length,), (length, 1)) and np.isfinite(reward).all(), "obs_reward 形状或数值错误")
        usable = episode["usable_length"]
        episode["action"] = action[:usable].astype(np.float32, copy=True)
        episode["obs_reward"] = reward[:usable].reshape(usable, 1).astype(np.float32, copy=True)
    validate_episode(episode, action_dim)
    return episode


def validate_episode(episode, action_dim):
    length = len(episode["image"])
    gl.require(length >= 2 and episode["heatmap"].shape == episode["image"].shape[:3], "片段观测尺寸错误")
    gl.require(episode["image"].dtype == np.uint8 and episode["heatmap"].dtype == np.uint8, "片段需要真实 uint8 观测")
    for key in ("is_first", "is_last", "is_terminal"):
        gl.require(episode[key].shape == (length,), f"{key} 长度错误")
    gl.require(bool(episode["is_first"][0]) and not episode["is_first"][1:].any(), "片段源含跨 reset")
    endings = np.flatnonzero(episode["is_last"] | episode["is_terminal"])
    gl.require(not len(endings) or int(endings[0]) == length - 1, "片段源含终止后后缀")
    for key in ("is_virtual", "is_synthetic"):
        gl.require(key not in episode or not np.any(episode[key]), "片段源含虚拟观测")
    action = episode["action"]
    gl.require(action.shape == (length, action_dim) and np.isfinite(action).all(), "真实动作形状或数值错误")
    gl.require(np.allclose(action[0], 0, atol=1e-6), "reset 的 action[0] 应为零占位，不是训练标签")
    binary = np.isclose(action[1:], 0, atol=1e-6) | np.isclose(action[1:], 1, atol=1e-6)
    gl.require(binary.all() and np.allclose(action[1:].sum(1), 1, atol=1e-6), "真实动作必须是 onehot，不能含虚拟动作")
    gl.require(episode["obs_reward"].shape == (length, 1) and np.isfinite(episode["obs_reward"]).all(), "obs_reward 形状或数值错误")


def sample_pairs(length, horizon, count, seed):
    """Stratify duration within each episode, with no duplicate (start,end)."""
    gl.require(length >= 2 and horizon >= 1 and count >= 1, "片段采样参数错误")
    rng = np.random.RandomState(seed)
    durations = range(1, min(horizon, length - 1) + 1)
    pools = {h: rng.permutation(length - h).tolist() for h in durations}
    pairs = []
    while len(pairs) < count:
        available = [h for h in pools if pools[h]]
        if not available:
            break
        for h in rng.permutation(available):
            start = int(pools[int(h)].pop())
            pairs.append((start, start + int(h)))
            if len(pairs) == count:
                break
    return sorted(pairs)


def segment_view(episode, start, end, goal, goal_id, horizon):
    gl.require(type(start) is int and type(end) is int and 0 <= start < end < len(episode["image"]), "片段索引越界或终点不在未来")
    gl.require(end - start <= horizon, "片段超过最大执行步数")
    gl.require(episode["is_first"][0] and not episode["is_first"][1:end + 1].any(), "片段跨 reset")
    gl.require(not (episode["is_last"][:end] | episode["is_terminal"][:end]).any(), "片段跨终止或含后缀")
    for key in ("is_virtual", "is_synthetic"):
        gl.require(key not in episode or not np.any(episode[key][:end + 1]), "片段含虚拟观测")
    goal = np.asarray(goal, dtype=np.float32)
    gl.require(goal.ndim == 1 and np.isfinite(goal).all() and np.isclose(np.linalg.norm(goal), 1, atol=1e-5), "事后目标向量异常")
    # History stops at end-1. No endpoint image, future obs_reward or label
    # action[end] is exposed as an RSSM state input.
    return {"history": {key: np.array(episode[key][:end], copy=True) for key in HISTORY_KEYS},
            "current_indices": np.arange(start, end, dtype=np.int64),
            "label_indices": np.arange(start + 1, end + 1, dtype=np.int64),
            "next_actions": np.array(episode["action"][start + 1:end + 1], copy=True),
            "goals": np.repeat(goal[None], end - start, axis=0),
            "goal_id": int(goal_id), "endpoint_index": end,
            "remaining": np.arange(end - start, 0, -1, dtype=np.int64),
            "loss_mask": np.arange(end) >= start, "prefix_steps": start}


def content_id(arrays, metadata):
    tensors = {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()}
    return gl.tensor_digest(tensors, metadata)


def validate_dataset(arrays, metadata):
    gl.require(metadata.get("format") == FORMAT and metadata.get("state_policy") == STATE_POLICY, "未知片段格式或状态恢复约定")
    gl.require(metadata.get("action_alignment") == "obs[t] -> action[t+1]; incoming action[t]", "动作对齐约定不兼容")
    required = {"episode_index", "start", "end", "goal_id", "goals"}
    gl.require(set(arrays) == required, "片段数组键错误")
    count = len(arrays["start"])
    gl.require(count > 0 and len(metadata["episodes"]) >= 2, "片段或 episode 为空")
    for key in required - {"goals"}:
        gl.require(arrays[key].shape == (count,) and arrays[key].dtype == np.int64, f"{key} 形状或类型错误")
    goals = arrays["goals"]
    gl.require(goals.shape == (count, metadata["goal_dim"]) and goals.dtype == np.float32 and np.isfinite(goals).all(), "目标数组形状或数值错误")
    gl.require(np.allclose(np.linalg.norm(goals, axis=1), 1, atol=1e-5), "目标数组不是单位向量")
    ep = arrays["episode_index"]
    gl.require(np.all((ep >= 0) & (ep < len(metadata["episodes"]))), "episode 编号越界")
    gl.require(np.array_equal(np.unique(ep), np.arange(len(metadata["episodes"]))), "片段没有覆盖清单中的所有 episode")
    gl.require(np.all((arrays["goal_id"] >= 0) & (arrays["goal_id"] < metadata["goal_count"])), "目标类别越界")
    maximum = np.array([entry["usable_length"] - 1 for entry in metadata["episodes"]])[ep]
    gl.require(np.all((arrays["start"] >= 0) & (arrays["end"] > arrays["start"]) & (arrays["end"] <= maximum)), "片段终点越界")
    gl.require(np.all(arrays["end"] - arrays["start"] <= metadata["horizon"]), "片段超过最大执行步数")
    gl.require(len(set(zip(ep.tolist(), arrays["start"].tolist(), arrays["end"].tolist()))) == count, "片段重复")
    paths = [entry["path"] for entry in metadata["episodes"]]
    gl.require(len(paths) == len(set(paths)), "episode 重复或分割泄漏")
    splits = {entry["split"] for entry in metadata["episodes"]}
    gl.require(splits == {"train", "validation"}, "缺少按 episode 分开的训练/检查集")
    gl.require(metadata["seed_holdout"] == "unavailable_in_legacy_replay", "旧回放不能虚构环境种子划分")


class SegmentDataset:
    """Small persisted indices/targets; real pixels remain in source replay."""
    def __init__(self, directory, cache_size=2):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "segments_manifest.json").read_text(encoding="utf-8"))
        self.metadata = {key: value for key, value in manifest.items() if key != "content_id"}
        with np.load(self.directory / "segments.npz", allow_pickle=False) as data:
            self.arrays = {key: data[key].copy() for key in data.files}
        validate_dataset(self.arrays, self.metadata)
        gl.require(content_id(self.arrays, self.metadata) == manifest["content_id"], "片段内容哈希不匹配")
        self.content_id = manifest["content_id"]
        self.cache_size = max(1, cache_size)
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.arrays["start"])

    def episode(self, index):
        entry = self.metadata["episodes"][index]
        path = Path(entry["path"])
        signature = file_signature(path)
        # Use the exact T00 signature convention, without importing scripts.
        expected = entry["signature"]
        gl.require(signature == expected, "源 replay 已变更")
        if index not in self.cache:
            self.cache[index] = read_episode(path, self.metadata["max_steps"], self.metadata["action_dim"])
            gl.require(self.cache[index]["usable_length"] == entry["usable_length"], "源 episode 边界改变")
            gl.require(file_signature(path) == expected, "读取过程中源 replay 已变更")
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[index]

    def rows(self, split):
        gl.require(split in ("train", "validation"), "未知片段 split")
        return np.flatnonzero([self.metadata["episodes"][int(ep)]["split"] == split for ep in self.arrays["episode_index"]])

    def __getitem__(self, row):
        gl.require(isinstance(row, (int, np.integer)) and 0 <= row < len(self), "片段行号越界")
        a = self.arrays
        return segment_view(self.episode(int(a["episode_index"][row])), int(a["start"][row]), int(a["end"][row]),
                            a["goals"][row], int(a["goal_id"][row]), self.metadata["horizon"])


class FrozenStateEncoder(nn.Module):
    """Only the original WM encoder/RSSM, no heads, optimizers or actor.

    Deterministic posterior mode is a prototype convention, not a claim of
    matching the original policy's sampled latent rollout. Later execution
    must use the same convention as BC training.
    """
    def __init__(self, config, shapes, device):
        super().__init__()
        device = torch.device(device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            self.encoder = networks.MultiEncoder({key: tuple(value) for key, value in shapes.items()}, **config["encoder"])
            gl.require(set(self.encoder.cnn_shapes) == {"image", "heatmap"} and set(self.encoder.mlp_shapes) == {"obs_reward"},
                       "T03 当前状态接口需要 image/heatmap/obs_reward，其他输入尚未支持")
            self.dynamics = networks.RSSM(config["dyn_stoch"], config["dyn_deter"], config["dyn_hidden"],
                config["dyn_rec_depth"], config["dyn_discrete"], config["act"], config["norm"],
                config["dyn_mean_act"], config["dyn_std_act"], config["dyn_min_std"], config["unimix_ratio"],
                config["initial"], config["num_actions"], self.encoder.outdim, str(device))
        self.action_dim = config["num_actions"]
        self.to(device).requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def step(self, observation, incoming, state=None):
        """One *real* observation; incoming is the action that produced it.

        The returned state belongs to this episode only. Reset requires a
        zero incoming action and discards any previous episode's state.
        Uses exactly the preprocessing/posterior convention of rollout().
        """
        image = np.asarray(observation["image"])
        heatmap = np.asarray(observation["heatmap"])
        incoming = np.asarray(incoming, dtype=np.float32)
        reward = np.asarray(observation["obs_reward"], dtype=np.float32).reshape(1, 1)
        first = bool(observation["is_first"])
        gl.require(image.dtype == np.uint8 and image.shape == (64, 64, 3) and
                   heatmap.dtype == np.uint8 and heatmap.shape == (64, 64), "在线状态需要真实 uint8 image/heatmap")
        gl.require(incoming.shape == (self.action_dim,) and np.isfinite(incoming).all() and
                   np.isfinite(reward).all(), "在线 incoming action/obs_reward 异常")
        if first:
            gl.require(np.all(incoming == 0), "reset 的 incoming action 必须为零")
            state = None
        else:
            gl.require(state is not None and np.all((incoming == 0) | (incoming == 1)) and
                       incoming.sum() == 1, "非 reset 状态需要本局历史及真实 onehot incoming action")
        device = next(self.parameters()).device
        self.dynamics._device = str(device)
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), gl.encoder_precision(device):
            obs = {"image": torch.as_tensor(image[None], device=device).float() / 255,
                   "heatmap": torch.as_tensor(heatmap[None, ..., None], device=device).float() / 255,
                   "obs_reward": torch.as_tensor(reward, device=device)}
            action = torch.as_tensor(incoming[None], device=device)
            action = torch.cat((action, action.new_zeros((1, 1))), -1)
            state, _ = self.dynamics.obs_step(state, action, self.encoder(obs),
                                            torch.tensor([float(first)], device=device), sample=False)
            features = self.dynamics.get_feat(state)
        gl.require(bool(torch.isfinite(features).all()), "在线 RSSM 状态非有限")
        return state, features

    @torch.no_grad()
    def rollout(self, episode, stop):
        gl.require(type(stop) is int and 0 <= stop < len(episode["image"]), "因果历史终点越界")
        gl.require(episode["is_first"][0] and not episode["is_first"][1:stop + 1].any(), "因果历史必须从真实 reset 开始")
        if "is_last" in episode and "is_terminal" in episode:
            gl.require(not (episode["is_last"][:stop] | episode["is_terminal"][:stop]).any(), "因果历史跨终止")
        device = next(self.parameters()).device
        self.dynamics._device = str(device)
        devices = [device.index] if device.type == "cuda" else []
        state, features = None, []
        # obs_step's discarded prior still samples; keep it from advancing
        # the caller's RNG. Posterior mode and initial learned state are fixed.
        with torch.random.fork_rng(devices=devices), gl.encoder_precision(device):
            for t in range(stop + 1):
                obs = {"image": torch.as_tensor(episode["image"][t:t + 1], device=device).float() / 255,
                       "heatmap": torch.as_tensor(episode["heatmap"][t:t + 1], device=device).float().unsqueeze(-1) / 255,
                       "obs_reward": torch.as_tensor(episode["obs_reward"][t:t + 1], device=device).float()}
                incoming = torch.as_tensor(episode["action"][t:t + 1], device=device).float()
                incoming = torch.cat((incoming, incoming.new_zeros((1, 1))), -1)
                first = torch.as_tensor(episode["is_first"][t:t + 1], device=device).float()
                state, _ = self.dynamics.obs_step(state, incoming, self.encoder(obs), first, sample=False)
                features.append(self.dynamics.get_feat(state)[0])
        result = torch.stack(features)
        gl.require(bool(torch.isfinite(result).all()), "因果 RSSM 特征非有限")
        return result


def state_encoder_from_checkpoint(checkpoint, config, shapes, device):
    import long_horizon as lh

    frozen = FrozenStateEncoder(config, shapes, device)
    source = lh.normalize(checkpoint["agent_state_dict"])
    selected = {name[len("_wm."):]: value for name, value in source.items()
                if name.startswith(("_wm.encoder.", "_wm.dynamics."))}
    lh.require_compatible(frozen.state_dict(), selected)
    frozen.load_state_dict(selected, strict=True)
    expected = source["_task_behavior.actor.layers.Actor_linear0.weight"].shape[1]
    actual = config["dyn_deter"] + config["dyn_stoch"] * (config["dyn_discrete"] or 1)
    gl.require(actual == expected, "因果当前状态维度与原 actor 不一致")
    return frozen
