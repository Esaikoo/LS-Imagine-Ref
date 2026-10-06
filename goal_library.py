"""T02 frozen, observation-only goal representation and immutable goal IDs.

No environment, MineCLIP, RSSM, reward, action or future-state inputs. The
standalone artifact embeds the visual weights, projection and goal anchors.
"""

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import networks


FORMAT = "ls_imagine_goal_library_v1"
NUMERIC_POLICY = "float32_no_amp_no_tf32_v1"


@contextmanager
def encoder_precision(device):
    """Keep goal identities independent of an enclosing training AMP context.

    TF32/AMP are useful for training, but their reduced precision must not
    change the frozen features used as persistent goal vectors. Restore the
    caller's settings after inference, including when inference raises.
    """
    device = torch.device(device)
    with torch.autocast(device_type=device.type, enabled=False):
        if device.type != "cuda":
            yield
            return
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                yield
        finally:
            # Restoring only allow_tf32=True would turn a caller's "medium"
            # matmul policy into "high". Preserve the complete policy.
            torch.set_float32_matmul_precision(previous_precision)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_digest(state, metadata=None):
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, value in sorted(state.items()):
        require(torch.is_tensor(value), f"非 Tensor 权重项: {name}")
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cnn_weights(state):
    # Canonicalize compile names without importing a full agent or environment.
    normalized = {}
    for name, value in state.items():
        key = ".".join(part for part in name.split(".") if part != "_orig_mod")
        require(key not in normalized, f"checkpoint 权重规范化名称冲突: {key}")
        normalized[key] = value
    prefix = "_wm.encoder._cnn."
    selected = {name[len(prefix):]: value for name, value in normalized.items() if name.startswith(prefix)}
    require(bool(selected), "checkpoint 没有 WM 视觉 CNN 权重")
    for alias in ("_task_behavior._world_model.encoder._cnn.", "_expl_behavior._world_model.encoder._cnn."):
        other = {name[len(alias):]: value for name, value in normalized.items() if name.startswith(alias)}
        if other:
            require(set(other) == set(selected), "checkpoint 的共享视觉 CNN 名称不一致")
            require(all(torch.equal(other[key], selected[key]) for key in selected), "checkpoint 的共享视觉 CNN 数值冲突")
    return selected


def encoder_spec(config, observation_shapes, goal_dim, pool_size):
    visual = {key: list(shape) for key, shape in observation_shapes.items()
              if len(shape) == 3 and re.match(config["encoder"]["cnn_keys"], key)}
    require(set(visual) == {"image", "heatmap"}, "T02 原型需要原 WM 的 image+heatmap 视觉接口")
    require(visual["image"][-1] == 3 and visual["heatmap"][-1] == 1, "RGB/heatmap 通道数不符合约定")
    require(visual["image"][:2] == visual["heatmap"][:2], "RGB 与 heatmap 尺寸不一致")
    cnn_config = config["encoder"]
    height, width = visual["image"][:2]
    stages = int(np.log2(height) - np.log2(cnn_config["minres"]))
    require(stages > 0 and 1 <= pool_size <= min(height // 2**stages, width // 2**stages), "pool_size 超过视觉特征图尺寸")
    channels = cnn_config["cnn_depth"] * 2 ** (stages - 1)
    return {"input_keys": list(visual), "shapes": visual, "goal_dim": goal_dim, "pool_size": pool_size,
            "raw_dim": channels * pool_size**2, "projection": "center_randomized_pca_l2_v1",
            "numeric_policy": NUMERIC_POLICY,
            "cnn": {"depth": cnn_config["cnn_depth"], "act": cnn_config["act"], "norm": cnn_config["norm"],
                    "kernel_size": cnn_config["kernel_size"], "minres": cnn_config["minres"]}}


class FrozenGoalEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        height, width = spec["shapes"]["image"][:2]
        channels = sum(spec["shapes"][key][-1] for key in spec["input_keys"])
        # Construction must not change the control agent's random stream.
        with torch.random.fork_rng(devices=[]):
            self.cnn = networks.ConvEncoder((height, width, channels), **spec["cnn"])
        self.register_buffer("mean", torch.zeros(spec["raw_dim"]))
        self.register_buffer("components", torch.zeros(spec["raw_dim"], spec["goal_dim"]))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def raw_features(self, observation):
        require({"image", "heatmap"} <= set(observation), "目标编码器需要真实 image 和 heatmap，不能只提供 zoomed 图")
        device = self.mean.device
        values = {}
        for key in self.spec["input_keys"]:
            value = torch.as_tensor(observation[key], device=device)
            require(value.dtype == torch.uint8, f"{key} 必须是原始 uint8 像素，不能重复归一化")
            if key == "heatmap" and value.ndim >= 2 and tuple(value.shape[-2:]) == tuple(self.spec["shapes"][key][:2]):
                value = value.unsqueeze(-1)
            shape = tuple(self.spec["shapes"][key])
            require(tuple(value.shape[-3:]) == shape, f"{key} 尺寸错误: {tuple(value.shape)}，期望尾部 {shape}")
            values[key] = value.to(torch.float32) / 255.0
        require(values["image"].shape[:-3] == values["heatmap"].shape[:-3], "image/heatmap 批次不一致")
        inputs = torch.cat([values[key] for key in self.spec["input_keys"]], -1) - 0.5
        leading = inputs.shape[:-3]
        inputs = inputs.reshape(-1, *inputs.shape[-3:]).permute(0, 3, 1, 2)
        with encoder_precision(device):
            maps = self.cnn.layers(inputs)
            pooled = F.adaptive_avg_pool2d(maps, self.spec["pool_size"]).flatten(1)
        require(pooled.shape[-1] == self.spec["raw_dim"], "目标 CNN 输出维度与已保存结构不一致")
        return pooled.reshape(*leading, self.spec["raw_dim"])

    @torch.no_grad()
    def project(self, raw):
        raw = torch.as_tensor(raw, dtype=torch.float32, device=self.mean.device)
        require(raw.shape[-1] == self.spec["raw_dim"] and bool(torch.isfinite(raw).all()), "目标原始特征维度或数值异常")
        with encoder_precision(self.mean.device):
            projected = (raw - self.mean) @ self.components
            norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
            require(bool((norms > 1e-8).all()), "目标特征坍缩为零；请检查回放多样性和降维维度")
            return projected / norms

    def forward(self, observation):
        return self.project(self.raw_features(observation))


def encoder_from_checkpoint(checkpoint, spec):
    encoder = FrozenGoalEncoder(spec)
    saved = cnn_weights(checkpoint["agent_state_dict"])
    expected = encoder.cnn.state_dict()
    require(set(saved) == set(expected), "目标 CNN 的保存键与当前结构不匹配")
    require(all(saved[key].shape == expected[key].shape and saved[key].dtype == expected[key].dtype for key in expected),
            "目标 CNN 权重形状或类型不匹配")
    encoder.cnn.load_state_dict(saved, strict=True)
    return encoder


def fit_pca(raw, train_indices, dimension, seed, iterations=2):
    # Fit on train episodes only. CPU randomized PCA avoids a new sklearn
    # dependency and does not allocate a raw_dim x raw_dim covariance matrix.
    x = torch.from_numpy(np.array(raw[train_indices], dtype=np.float32, copy=True))
    require(x.ndim == 2 and min(x.shape) > dimension and bool(torch.isfinite(x).all()), "PCA 训练样本不足或特征非有限")
    mean = x.mean(0)
    x -= mean
    total = float(x.square().sum())
    require(total > 1e-8, "真实观测特征没有方差，不能构建目标库")
    rank = min(dimension + 16, min(x.shape))
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        _, singular, vectors = torch.pca_lowrank(x, q=rank, center=False, niter=iterations)
    components = vectors[:, :dimension].contiguous()
    # Fix the arbitrary sign of each PCA axis before persistent numbering.
    pivots = components.abs().argmax(0)
    signs = torch.sign(components[pivots, torch.arange(dimension)])
    components *= signs
    stats = {"fit_frames": len(train_indices), "raw_dim": x.shape[1], "goal_dim": dimension,
             "retained_variance": float(singular[:dimension].square().sum()) / total,
             "effective_rank": int((singular > singular[0] * 1e-6).sum()), "whiten": False}
    require(float(singular[dimension - 1]) > float(singular[0]) * 1e-6, "PCA 有效维度不足，请增加真实数据或减小 goal_dim")
    return mean, components, stats


def project_array(raw, mean, components, batch_size=1024):
    result = np.empty((len(raw), components.shape[1]), dtype=np.float32)
    for start in range(0, len(raw), batch_size):
        values = torch.from_numpy(np.array(raw[start:start + batch_size], dtype=np.float32, copy=True))
        values = (values - mean) @ components
        norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        require(bool((norms > 1e-8).all()), "真实样本的 PCA 特征为零，不能用任意方向替代")
        result[start:start + len(values)] = (values / norms).numpy()
    return result


def fit_spherical_kmeans(features, goal_count, seed, restarts=3, iterations=100):
    x = np.asarray(features, dtype=np.float32)
    require(x.ndim == 2 and len(x) >= goal_count * 4 and np.isfinite(x).all(), "聚类样本不足或特征非有限")
    best = None
    for attempt in range(restarts):
        rng = np.random.RandomState(seed + attempt)
        chosen = [int(rng.randint(len(x)))]
        distance = np.maximum(0, 1 - x @ x[chosen[0]])
        for _ in range(1, goal_count):
            weights = distance.astype(np.float64)
            weights[chosen] = 0
            require(weights.sum() > 1e-8, "观测特征缺少足够的不同方向；增加数据或减小 goal_count")
            index = int(rng.choice(len(x), p=weights / weights.sum()))
            chosen.append(index)
            distance = np.minimum(distance, np.maximum(0, 1 - x @ x[index]))
        centers = x[chosen].copy()
        previous = None
        for _ in range(iterations):
            similarities = x @ centers.T
            labels = similarities.argmax(1)
            if previous is not None and np.array_equal(previous, labels):
                break
            previous = labels.copy()
            used = set()
            for index in range(goal_count):
                members = x[labels == index]
                if len(members):
                    value = members.mean(0)
                    norm = np.linalg.norm(value)
                    require(norm > 1e-8, "目标聚类中心坍缩")
                    centers[index] = value / norm
                else:
                    order = np.argsort(similarities.max(1), kind="stable")
                    replacement = next(int(row) for row in order if int(row) not in used)
                    used.add(replacement)
                    centers[index] = x[replacement]
        labels = (x @ centers.T).argmax(1)
        if np.any(np.bincount(labels, minlength=goal_count) == 0):
            continue
        representatives = np.array([np.flatnonzero(labels == index)[
            np.argmax(x[labels == index] @ centers[index])] for index in range(goal_count)])
        # Stable numbering is saved, not recomputed on load. Canonicalize
        # fitting runs by representative row order, rather than random init ID.
        order = np.argsort(representatives, kind="stable")
        centers = centers[order].copy()
        representatives = representatives[order]
        labels = (x @ centers.T).argmax(1)
        loss = float(np.mean(1 - np.sum(x * centers[labels], axis=1)))
        candidate = {"centers": centers, "labels": labels, "representatives": representatives, "loss": loss}
        if best is None or loss < best["loss"]:
            best = candidate
    require(best is not None, "聚类存在空类别；增加真实回放或减小 goal_count")
    return best


class GoalLibrary(nn.Module):
    def __init__(self, encoder, centers, goals, images, heatmaps, metadata):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32).clone())
        self.register_buffer("goals", torch.as_tensor(goals, dtype=torch.float32).clone())
        self.register_buffer("images", torch.as_tensor(images, dtype=torch.uint8).clone())
        self.register_buffer("heatmaps", torch.as_tensor(heatmaps, dtype=torch.uint8).clone())
        self.metadata = copy.deepcopy(metadata)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, observation):
        return self.encoder(observation)

    @torch.no_grad()
    def assign(self, features):
        value = torch.as_tensor(features, dtype=torch.float32, device=self.centers.device)
        require(value.shape[-1] == self.goals.shape[1] and bool(torch.isfinite(value).all()), "目标特征维度或数值错误")
        require(bool((torch.linalg.vector_norm(value, dim=-1) > 1e-8).all()), "不能分配零目标特征")
        with encoder_precision(self.centers.device):
            return (F.normalize(value, dim=-1) @ self.centers.T).argmax(-1)

    def goal(self, ids):
        ids = torch.as_tensor(ids, device=self.goals.device)
        require(ids.dtype in (torch.int32, torch.int64) and bool(((ids >= 0) & (ids < len(self.goals))).all()), "目标编号越界或类型错误")
        return self.goals[ids.long()]

    def payload(self):
        state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
        spec = self.encoder.spec
        enc_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
        encoder_id = tensor_digest(enc_state, spec)
        library_id = tensor_digest({key: value for key, value in state.items() if not key.startswith("encoder.")},
                                   {"encoder_id": encoder_id, "task": self.metadata["task"]})
        return {"format": FORMAT, "encoder_spec": copy.deepcopy(spec), "state_dict": state,
                "encoder_id": encoder_id, "library_id": library_id,
                "source_encoder_hash": tensor_digest(self.encoder.cnn.state_dict()), "metadata": copy.deepcopy(self.metadata)}

    @classmethod
    def from_payload(cls, payload, device="cpu"):
        validate_payload(payload)
        state = payload["state_dict"]
        encoder = FrozenGoalEncoder(payload["encoder_spec"])
        library = cls(encoder, state["centers"], state["goals"], state["images"], state["heatmaps"], payload["metadata"])
        require(set(library.state_dict()) == set(state), "目标库保存键与结构不匹配")
        library.load_state_dict(state, strict=True)
        return library.to(device)


def validate_payload(payload, goal_dim=None, goal_count=None):
    require(isinstance(payload, dict) and payload.get("format") == FORMAT, "未知目标库格式")
    state, spec = payload["state_dict"], payload["encoder_spec"]
    require(spec.get("numeric_policy") == NUMERIC_POLICY,
            "目标库缺少当前 FP32 数值约定；请用修复后的 T02 build 重新构建，不复用旧缓存")
    require(spec["projection"] == "center_randomized_pca_l2_v1" and set(spec["input_keys"]) == {"image", "heatmap"}, "未知目标编码接口")
    dimension = spec["goal_dim"]
    require(type(dimension) is int and dimension > 0 and type(spec["raw_dim"]) is int and spec["raw_dim"] >= dimension,
            "目标库编码维度不合法")
    require(len(spec["input_keys"]) == 2 and set(spec["shapes"]) == {"image", "heatmap"}, "目标输入键不合法")
    count = state["centers"].shape[0]
    require(count > 0 and len(payload["metadata"]["representatives"]) == count, "目标库代表帧数量不匹配")
    require(goal_dim is None or goal_dim == dimension, "目标库维度与 worker 配置不同")
    require(goal_count is None or goal_count == count, "目标库类别数与高层配置不同")
    shapes = {"centers": (count, dimension), "goals": (count, dimension),
              "encoder.mean": (spec["raw_dim"],), "encoder.components": (spec["raw_dim"], dimension),
              "images": (count, *spec["shapes"]["image"]), "heatmaps": (count, *spec["shapes"]["image"][:2])}
    for name, shape in shapes.items():
        require(state[name].shape == shape, f"目标库 {name} 形状错误")
    for name, value in state.items():
        expected_dtype = torch.uint8 if name in ("images", "heatmaps") else torch.float32
        require(torch.is_tensor(value) and value.dtype == expected_dtype and bool(torch.isfinite(value).all()), f"目标库 {name} 数值/类型异常")
    for name in ("centers", "goals"):
        require(torch.allclose(torch.linalg.vector_norm(state[name], dim=-1), torch.ones(count), atol=1e-5), f"{name} 不是单位向量")
    require(torch.allclose(state["encoder.components"].T @ state["encoder.components"], torch.eye(dimension), atol=1e-4, rtol=1e-4), "PCA 投影不是正交基")
    enc_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
    require(tensor_digest(enc_state, spec) == payload["encoder_id"], "目标编码器内容哈希不匹配")
    other = {key: value for key, value in state.items() if not key.startswith("encoder.")}
    require(tensor_digest(other, {"encoder_id": payload["encoder_id"], "task": payload["metadata"]["task"]}) == payload["library_id"], "目标库内容哈希不匹配")
    weights = {key[len("encoder.cnn."):]: value for key, value in state.items() if key.startswith("encoder.cnn.")}
    require(tensor_digest(weights) == payload["source_encoder_hash"], "目标库的原 CNN 指纹不匹配")


def save_library(library, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), "拒绝覆盖已有目标库或临时文件")
    payload = library.payload()
    validate_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_library(path, device="cpu"):
    import inspect

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return GoalLibrary.from_payload(torch.load(Path(path), **kwargs), device)


def attach(agent, library):
    require(agent.experiment_mode != "flat_ls", "flat_ls 不使用目标库，请选择目标策略模式")
    payload = library.payload()
    validate_payload(payload, agent._config.goal_dim, agent._config.goal_count)
    require(payload["metadata"]["task"] == agent._config.task, "目标库任务与当前任务不同")
    if agent._goal_encoder_id is not None:
        require(agent._goal_encoder_id == payload["encoder_id"], "不能替换已有目标编码器")
    if getattr(agent, "_goal_library_id", None) is not None:
        require(agent._goal_library_id == payload["library_id"], "不能替换已有目标库编号语义")
    # Plain CPU payload, not a newly registered module/optimizer. Future
    # workers obtain the immutable runtime view through from_agent().
    agent._goal_library = payload
    agent._goal_encoder_id = payload["encoder_id"]
    agent._goal_library_id = payload["library_id"]


def from_agent(agent, device=None):
    require(agent._goal_library is not None, "agent 没有正式目标库")
    validate_payload(agent._goal_library, agent._config.goal_dim, agent._config.goal_count)
    require(agent._goal_encoder_id == agent._goal_library["encoder_id"] and
            getattr(agent, "_goal_library_id", None) == agent._goal_library["library_id"], "agent 目标库内容标识不同")
    return GoalLibrary.from_payload(agent._goal_library, device or agent._config.device)


def read_episode(path, max_steps):
    """Return only the first real episode, before any collector tail.

    A zoom trigger does NOT make image virtual: its original image/heatmap
    are real observations. Auxiliary zoomed_image is never read here.
    """
    with np.load(path, allow_pickle=False) as data:
        require({"image", "heatmap", "is_first", "is_last", "is_terminal"} <= set(data.files), "replay 缺少真实观测/边界字段")
        images, heatmaps = data["image"], data["heatmap"]
        length = len(images)
        require(length >= 2 and images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3, "replay RGB 形状/类型错误")
        if heatmaps.ndim == 4 and heatmaps.shape[-1] == 1:
            heatmaps = heatmaps[..., 0]
        require(heatmaps.dtype == np.uint8 and heatmaps.shape == images.shape[:3], "replay heatmap 形状/类型错误")
        flags = {}
        for key in ("is_first", "is_last", "is_terminal", "is_zoomed"):
            value = data[key] if key in data.files else np.zeros(length, dtype=bool)
            require(value.shape in ((length,), (length, 1)) and np.isin(value, (0, 1)).all(), f"replay {key} 长度或标志错误")
            flags[key] = value.reshape(length).astype(bool)
        require(flags["is_first"][0] and not flags["is_first"][1:].any(), "单个 replay 含跨 reset 或缺少起点")
        endings = np.flatnonzero(flags["is_last"] | flags["is_terminal"])
        end = min(length - 1, max_steps, int(endings[0]) if len(endings) else length - 1)
        require(end >= 1, "episode 起点已终止")
        # Reject explicitly flagged synthetic observations; legacy LS replay
        # normally has none. Never infer that is_zoomed means synthetic RGB.
        for key in ("is_virtual", "is_synthetic"):
            if key in data.files:
                value = data[key]
                require(value.shape in ((length,), (length, 1)) and not value[:end + 1].any(), "replay 含显式虚拟观测")
        return {"image": images[:end + 1], "heatmap": heatmaps[:end + 1],
                **{key: value[:end + 1] for key, value in flags.items()},
                "original_length": length, "usable_length": end + 1, "discarded_tail": length - end - 1}
