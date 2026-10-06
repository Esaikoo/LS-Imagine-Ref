"""T04 frozen-state BC worker and current-state future-class forecaster.

This offline prototype has its own compact checkpoint contract. It reuses
the T01 expanded actor and T03 causal encoder, not the flat training loop.
"""

import copy
import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_library as gl
import goal_segments as gs
import long_horizon as lh
import networks


BUNDLE_FORMAT = "ls_imagine_frozen_bc_bundle_v1"
CACHE_FORMAT = "ls_imagine_bc_cache_v1"
CHECKPOINT_FORMAT = "ls_imagine_goal_bc_checkpoint_v1"
CHECKPOINT_STORAGE = "external_frozen_bundle_v1"
DISK_RESERVE_BYTES = 64 * 1024 * 1024


def _torch_load(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(Path(path), **kwargs)


def torch_load(path):
    """Read legacy embedded snapshots or resolve a checked shared dependency."""
    path = Path(path)
    payload = _torch_load(path)
    if not isinstance(payload, dict) or payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        return payload
    storage = payload.get("checkpoint_storage")
    if storage is None:
        gl.require("frozen_bundle_ref" not in payload, "checkpoint 冻结依赖声明不完整")
        return payload
    gl.require(storage == CHECKPOINT_STORAGE and "frozen_bundle" not in payload,
               "checkpoint 存储格式不兼容或冻结模型重复声明")
    reference = payload["frozen_bundle_ref"]
    gl.require(isinstance(reference, dict) and isinstance(reference.get("path"), str) and
               reference["path"] and not Path(reference["path"]).is_absolute(), "冻结依赖需要相对路径")
    dependency = (path.parent / reference["path"]).resolve()
    if not dependency.is_file():
        raise FileNotFoundError(f"checkpoint 依赖缺失：{dependency}；请保留 T04 缓存中的 frozen_bundle.pt，迁移时保持相对目录关系")
    gl.require(file_hash(dependency) == reference["sha256"], "checkpoint 冻结依赖文件 SHA256 不匹配")
    bundle = _torch_load(dependency)
    validate_bundle(bundle)
    gl.require(bundle["bundle_id"] == reference["bundle_id"], "checkpoint 冻结依赖内容 ID 不匹配")
    return dict(payload, frozen_bundle=bundle)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def tensor_storage_bytes(value):
    """Count tensor storages once, including views' complete saved storage."""
    seen = set()

    def count(item):
        if torch.is_tensor(item):
            storage = item.untyped_storage() if hasattr(item, "untyped_storage") else item.storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, np.ndarray):
            return item.nbytes
        if isinstance(item, dict):
            return sum(count(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(count(child) for child in item)
        return 0

    return count(value)


def require_disk_space(directory, data_bytes, reserve_bytes=DISK_RESERVE_BYTES):
    free = shutil.disk_usage(directory).free
    required = int(data_bytes) + reserve_bytes
    if free < required:
        raise OSError(errno.ENOSPC,
            f"磁盘空间不足：{directory}，可用 {free / 2**20:.1f} MiB，预计需要至少 {required / 2**20:.1f} MiB "
            "（含临时写入及日志余量）；请清理空间或指定其他文件系统上的 --output-dir")
    return {"free_bytes": free, "required_bytes": required, "reserve_bytes": reserve_bytes}


def save_atomic(payload, path, overwrite=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    gl.require(not temporary.exists() and (overwrite or not path.exists()), "拒绝覆盖已有产物或未完成临时文件")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_space(path.parent, tensor_storage_bytes(payload))
    # Exclusive creation makes ownership explicit: never remove a pre-existing
    # .tmp file. A failed write leaves the previous destination untouched.
    owned = False
    try:
        with temporary.open("xb", buffering=0) as stream:
            owned = True
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        if owned:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(f"[WARN] temporary_cleanup: {temporary}: {cleanup_error}", file=sys.stderr, flush=True)
        raise


def frozen_bundle_reference(cache):
    path = (cache.directory / "frozen_bundle.pt").resolve()
    return {"source_path": path, "sha256": file_hash(path), "bundle_id": cache.bundle["bundle_id"]}


def compact_checkpoint(payload, path, reference):
    """Keep all mutable training state, sharing only the immutable bundle."""
    gl.require(payload.get("checkpoint_format") == CHECKPOINT_FORMAT and
               payload["frozen_bundle"]["bundle_id"] == reference["bundle_id"], "checkpoint 与冻结依赖不同")
    result = {key: value for key, value in payload.items() if key not in ("frozen_bundle", "frozen_bundle_ref")}
    result["checkpoint_storage"] = CHECKPOINT_STORAGE
    result["frozen_bundle_ref"] = {"path": os.path.relpath(reference["source_path"], Path(path).resolve().parent),
                                  "sha256": reference["sha256"], "bundle_id": reference["bundle_id"]}
    return result


def save_checkpoint(payload, path, reference, overwrite=False):
    save_atomic(compact_checkpoint(payload, path, reference), path, overwrite=overwrite)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_id(bundle):
    tensors = {f"state.{key}": value for key, value in bundle["state_encoder"].items()}
    tensors.update({f"original_actor.{key}": value for key, value in bundle["original_actor"].items()})
    return gl.tensor_digest(tensors, {"config": bundle["config"], "shapes": bundle["shapes"],
        "state_policy": bundle["state_policy"], "library_id": bundle["library"]["library_id"],
        "horizon": bundle["horizon"], "feature_dim": bundle["feature_dim"]})


def validate_bundle(bundle):
    gl.require(bundle.get("format") == BUNDLE_FORMAT and bundle.get("state_policy") == gs.STATE_POLICY, "冻结 bundle 格式或状态约定不兼容")
    gl.validate_payload(bundle["library"])
    gl.require(bundle_id(bundle) == bundle["bundle_id"], "冻结 bundle 内容哈希不匹配")
    gl.require(bundle["config"]["actor"]["dist"] == "onehot", "T04 仅支持原 onehot 动作接口")
    gl.require(bundle["config"]["task"] == bundle["library"]["metadata"]["task"], "冻结 bundle 任务不兼容")


def original_actor(config, feature_dim, state, device):
    a = config["actor"]
    actor = networks.MLP(feature_dim, (config["num_actions"],), a["layers"], config["units"],
        config["act"], config["norm"], a["dist"], a["std"], a["min_std"], a["max_std"],
        absmax=1.0, temp=a["temp"], unimix_ratio=a["unimix_ratio"], outscale=a["outscale"],
        device=device, name="Actor").to(device)
    lh.require_compatible(actor.state_dict(), state)
    actor.load_state_dict(state, strict=True)
    return actor.requires_grad_(False).eval()


class Candidate(nn.Module):
    """P(future class | current state) over sampled 1..horizon durations.

    There is no goal, remaining-duration or success input, and no claim that
    the forecast probabilities measure controllability.
    """
    def __init__(self, feature_dim, hidden, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden),
                                    nn.SiLU(), nn.Linear(hidden, classes))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return self.layers(features)


class GoalBCModel(nn.Module):
    def __init__(self, bundle, options, device):
        super().__init__()
        validate_bundle(bundle)
        self.bundle = bundle  # immutable CPU source; not an optimizer member
        self.options = copy.deepcopy(options)
        self.state_encoder = gs.FrozenStateEncoder(bundle["config"], bundle["shapes"], device)
        lh.require_compatible(self.state_encoder.state_dict(), bundle["state_encoder"])
        self.state_encoder.load_state_dict(bundle["state_encoder"], strict=True)
        self.library = gl.GoalLibrary.from_payload(bundle["library"], device)
        self.original_actor = original_actor(bundle["config"], bundle["feature_dim"], bundle["original_actor"], device)
        self.worker = lh.GoalWorker(self.original_actor, self.library.goals.shape[1], bundle["horizon"]).to(device)
        # The original actor is frozen; deep-copying it also copied that flag.
        # Only the new worker's full actor is trainable, not just its first layer.
        self.worker.requires_grad_(True)
        self.candidate = Candidate(bundle["feature_dim"], options["candidate_hidden"], len(self.library.goals)).to(device)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.original_actor.eval()
        self.state_encoder.eval()
        self.library.eval()
        return self

    def action_dist(self, features, goals, remaining, mode="trained"):
        with gl.encoder_precision(features.device):
            if mode == "original":
                return self.original_actor(features)
            gl.require(mode in ("trained", "zero_goal"), "未知低层行为模式")
            if mode == "zero_goal" or self.options["conditioning"] == "no_goal":
                goals = torch.zeros_like(goals)
            return self.worker(features, goals, remaining)

    @torch.no_grad()
    def propose(self, features, count=4):
        gl.require(1 <= count <= len(self.library.goals), "候选数量越界")
        with gl.encoder_precision(features.device):
            probabilities = self.candidate(features).softmax(-1)
            scores, ids = probabilities.topk(count, dim=-1)
        return {"goal_ids": ids, "forecast_probabilities": scores, "goals": self.library.goal(ids)}

    def frozen_identity(self):
        current = dict(self.bundle, state_encoder=self.state_encoder.state_dict(),
                       original_actor=self.original_actor.state_dict(), library=self.library.payload())
        return bundle_id(current)


class TrainingCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "cache_manifest.json").read_text(encoding="utf-8"))
        self.metadata = {key: value for key, value in manifest.items() if key != "cache_id"}
        gl.require(self.metadata.get("format") == CACHE_FORMAT and self.metadata.get("state_policy") == gs.STATE_POLICY, "未知训练缓存格式")
        with np.load(self.directory / "tables.npz", allow_pickle=False) as data:
            self.tables = {key: data[key].copy() for key in data.files}
        self.states = np.load(self.directory / "states.npy", mmap_mode="r", allow_pickle=False)
        gl.require(self.states.dtype == np.float32 and self.states.ndim == 2 and len(self.states) > 0 and
                   self.states.shape == tuple(self.metadata["state_shape"]), "当前状态缓存形状/类型错误")
        gl.require(file_hash(self.directory / "states.npy") == self.metadata["states_sha256"], "当前状态缓存内容哈希不匹配")
        self.cache_id = gs.content_id(self.tables, self.metadata)
        gl.require(self.cache_id == manifest["cache_id"], "训练表/缓存元数据哈希不匹配")
        report = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        gl.require(report.get("command") == "prepare" and report.get("status") in ("passed", "passed_with_warnings") and
                   report.get("cache_id") == self.cache_id, "需要通过 T04 prepare 的训练缓存")
        self.bundle = torch_load(self.directory / "frozen_bundle.pt")
        validate_bundle(self.bundle)
        gl.require(self.bundle["bundle_id"] == self.metadata["bundle_id"], "训练缓存与冻结 bundle 不匹配")
        gl.require(self.states.shape[1] == self.bundle["feature_dim"] and
                   self.metadata["action_dim"] == self.bundle["config"]["num_actions"] and
                   self.metadata["goal_dim"] == self.bundle["library"]["state_dict"]["goals"].shape[1] and
                   self.metadata["goal_count"] == len(self.bundle["library"]["state_dict"]["goals"]), "缓存与模型接口维度不匹配")
        t = self.tables
        required = {"worker_state", "worker_segment", "remaining", "action_id", "worker_train",
                    "candidate_state", "candidate_goal", "candidate_train", "goals"}
        gl.require(set(t) == required, "训练表字段错误")
        n, m = len(t["worker_state"]), len(t["candidate_state"])
        for key in ("worker_state", "worker_segment", "remaining", "action_id"):
            gl.require(t[key].shape == (n,) and t[key].dtype == np.int64, f"{key} 表结构错误")
        for key in ("candidate_state", "candidate_goal"):
            gl.require(t[key].shape == (m,) and t[key].dtype == np.int64, f"{key} 表结构错误")
        gl.require(t["worker_train"].shape == (n,) and t["candidate_train"].shape == (m,) and
                   t["worker_train"].dtype == bool and t["candidate_train"].dtype == bool, "训练/验证标志错误")
        gl.require(t["goals"].shape == (m, self.metadata["goal_dim"]) and t["goals"].dtype == np.float32 and
                   np.isfinite(t["goals"]).all() and np.allclose(np.linalg.norm(t["goals"], axis=1), 1, atol=1e-5), "训练目标特征错误")
        gl.require(np.all((t["worker_state"] >= 0) & (t["worker_state"] < len(self.states))) and
                   np.all((t["candidate_state"] >= 0) & (t["candidate_state"] < len(self.states))), "当前状态行号越界")
        gl.require(np.all((t["worker_segment"] >= 0) & (t["worker_segment"] < m)), "目标片段行号越界")
        gl.require(np.array_equal(t["worker_train"], t["candidate_train"][t["worker_segment"]]), "监督表跨 episode 划分")
        gl.require(np.all((t["remaining"] >= 1) & (t["remaining"] <= self.bundle["horizon"])) and
                   np.all((t["action_id"] >= 0) & (t["action_id"] < self.metadata["action_dim"])) and
                   np.all((t["candidate_goal"] >= 0) & (t["candidate_goal"] < self.metadata["goal_count"])), "监督标签越界")
        self.worker_rows = {split: np.flatnonzero(t["worker_train"] == (split == "train")) for split in ("train", "validation")}
        self.candidate_rows = {split: np.flatnonzero(t["candidate_train"] == (split == "train")) for split in ("train", "validation")}
        gl.require(all(len(rows) for rows in list(self.worker_rows.values()) + list(self.candidate_rows.values())), "训练或验证表为空")

    def worker_batch(self, rows, device):
        t = self.tables
        segments = t["worker_segment"][rows]
        return (torch.as_tensor(np.array(self.states[t["worker_state"][rows]], copy=True), device=device),
                torch.as_tensor(t["goals"][segments], device=device),
                torch.as_tensor(t["remaining"][rows], device=device),
                torch.as_tensor(t["action_id"][rows], device=device))

    def candidate_batch(self, rows, device):
        return (torch.as_tensor(np.array(self.states[self.tables["candidate_state"][rows]], copy=True), device=device),
                torch.as_tensor(self.tables["candidate_goal"][rows], device=device))


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(saved, device):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if torch.device(device).type == "cuda":
        gl.require(saved["cuda"] is not None, "CUDA RNG 缺失，不能精确 resume")
        torch.cuda.set_rng_state(saved["cuda"], device)


def backend_info(device):
    device = torch.device(device)
    cuda = device.type == "cuda"
    return {"device_type": device.type, "torch": str(torch.__version__),
            "cuda": torch.version.cuda if cuda else None,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "device_name": torch.cuda.get_device_name(device) if cuda else None,
            "capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
            "cpu_threads": torch.get_num_threads() if not cuda else None}


class Trainer:
    def __init__(self, cache, options, device):
        self.cache, self.options, self.device = cache, copy.deepcopy(options), device
        gl.require(options["conditioning"] in ("goal", "no_goal"), "未知 BC 条件模式")
        gl.require(all(np.isfinite(options[key]) and options[key] > 0 for key in
                       ("batch_size", "candidate_hidden", "learning_rate", "candidate_learning_rate", "grad_clip")), "训练超参数必须有限且为正")
        gl.require(type(options["batch_size"]) is int and type(options["candidate_hidden"]) is int and
                   type(options["seed"]) is int and 0 <= options["seed"] < 2**32, "训练批次/维度/种子类型错误")
        self.model = GoalBCModel(cache.bundle, options, device)
        self.optimizers = {"worker": torch.optim.Adam(self.model.worker.parameters(), lr=options["learning_rate"]),
                           "candidate": torch.optim.Adam(self.model.candidate.parameters(), lr=options["candidate_learning_rate"])}
        self.sampler = np.random.RandomState(options["seed"])
        self.step = 0
        self.check_ownership()

    def check_ownership(self):
        groups = [{id(p) for group in optimizer.param_groups for p in group["params"]} for optimizer in self.optimizers.values()]
        frozen = {id(p) for module in (self.model.state_encoder, self.model.library, self.model.original_actor) for p in module.parameters()}
        gl.require(not groups[0] & groups[1] and not (groups[0] | groups[1]) & frozen, "优化器包含冻结模块或共享参数")
        gl.require(groups[0] == {id(p) for p in self.model.worker.parameters()} and
                   groups[1] == {id(p) for p in self.model.candidate.parameters()}, "优化器未覆盖对应新模块的全部参数")
        gl.require(all(p.requires_grad for module in (self.model.worker, self.model.candidate) for p in module.parameters()), "新模块存在意外冻结参数")
        gl.require(not any(p.requires_grad for module in (self.model.state_encoder, self.model.library, self.model.original_actor) for p in module.parameters()), "原模型或目标编码器被解冻")

    def update(self):
        self.model.train(True)
        size = self.options["batch_size"]
        wr = self.sampler.choice(self.cache.worker_rows["train"], size=size, replace=True)
        cr = self.sampler.choice(self.cache.candidate_rows["train"], size=size, replace=True)
        features, goals, remaining, actions = self.cache.worker_batch(wr, self.device)
        current, classes = self.cache.candidate_batch(cr, self.device)
        with gl.encoder_precision(self.device):
            onehot = F.one_hot(actions, self.cache.metadata["action_dim"]).float()
            worker_loss = -self.model.action_dist(features, goals, remaining).log_prob(onehot).mean()
            candidate_loss = F.cross_entropy(self.model.candidate(current), classes)
            gl.require(bool(torch.isfinite(worker_loss)) and bool(torch.isfinite(candidate_loss)), "训练损失非有限")
            norms = {}
            for name, loss, module in (("worker", worker_loss, self.model.worker), ("candidate", candidate_loss, self.model.candidate)):
                optimizer = self.optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(module.parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), f"{name} 梯度非有限")
                optimizer.step()
                norms[name] = float(norm)
        self.step += 1
        return {"step": self.step, "worker_nll": float(worker_loss.detach()),
                "candidate_ce": float(candidate_loss.detach()), "worker_grad_norm": norms["worker"],
                "candidate_grad_norm": norms["candidate"]}

    def counters(self):
        return {"step": self.step, "worker_updates": self.step, "candidate_updates": self.step,
                "worker_version": self.step, "worker_labels_seen": self.step * self.options["batch_size"],
                "candidate_labels_seen": self.step * self.options["batch_size"], "new_env_steps": 0}

    def payload(self, verification_artifact=False):
        return {"checkpoint_format": CHECKPOINT_FORMAT, "experiment_mode": "goal_worker", "stage": "t04_offline_bc",
                "worker_architecture": "expanded_actor_v1", "candidate_semantics": "current_state_future_class_forecast",
                "frozen_bundle": self.cache.bundle, "cache_id": self.cache.cache_id, "options": self.options,
                "worker": cpu_tree(self.model.worker.state_dict()), "candidate": cpu_tree(self.model.candidate.state_dict()),
                "optimizers": {name: cpu_tree(opt.state_dict()) for name, opt in self.optimizers.items()},
                "counters": self.counters(), "sampler_state": self.sampler.get_state(),
                "rng_state": capture_rng(self.device), "backend": backend_info(self.device),
                "verification_artifact": bool(verification_artifact)}

    def restore(self, payload, allow_verification=False):
        gl.require(payload.get("checkpoint_format") == CHECKPOINT_FORMAT and payload.get("experiment_mode") == "goal_worker" and
                   payload.get("stage") == "t04_offline_bc" and payload.get("worker_architecture") == "expanded_actor_v1" and
                   payload.get("candidate_semantics") == "current_state_future_class_forecast", "T04 checkpoint 格式或模式不兼容")
        gl.require(allow_verification or not payload.get("verification_artifact"), "合成验收 checkpoint 不能恢复真实训练")
        gl.require(payload["cache_id"] == self.cache.cache_id, "训练缓存 ID 不同，不能精确 resume")
        validate_bundle(payload["frozen_bundle"])
        gl.require(payload["frozen_bundle"]["bundle_id"] == self.cache.bundle["bundle_id"], "冻结模型/目标库 ID 不同")
        gl.require(payload["options"] == self.options, "训练超参数不同，不能精确 resume")
        gl.require(payload["backend"] == backend_info(self.device), "精确 resume 需要相同设备类型、PyTorch 及 CUDA/cuDNN/设备信息")
        counters = payload["counters"]
        step = counters["step"]
        gl.require(type(step) is int and step >= 0 and counters == dict(step=step, worker_updates=step, candidate_updates=step,
            worker_version=step, worker_labels_seen=step * self.options["batch_size"], candidate_labels_seen=step * self.options["batch_size"],
            new_env_steps=0), "checkpoint 训练计数异常")
        for name, module in (("worker", self.model.worker), ("candidate", self.model.candidate)):
            lh.require_compatible(module.state_dict(), payload[name])
            gl.require(all(bool(torch.isfinite(value).all()) for value in payload[name].values()), "新模块权重含非有限值")
            lh.validate_optimizer(self.optimizers[name], payload["optimizers"][name], name)
            expected_lr = self.options["learning_rate" if name == "worker" else "candidate_learning_rate"]
            gl.require(all(group["lr"] == expected_lr for group in payload["optimizers"][name]["param_groups"]), "优化器学习率与配置不同")
            for current, saved in zip(self.optimizers[name].param_groups, payload["optimizers"][name]["param_groups"]):
                gl.require({k: v for k, v in current.items() if k != "params"} ==
                           {k: v for k, v in saved.items() if k != "params"}, "优化器超参数与配置不同")
            states = payload["optimizers"][name]["state"]
            gl.require((step == 0 and not states) or (step > 0 and bool(states)), "优化器状态与训练计数不匹配")
            indices = {i for group in payload["optimizers"][name]["param_groups"] for i in group["params"]}
            gl.require(step == 0 or set(states) == indices, "优化器动量状态不完整")
            gl.require(all({"step", "exp_avg", "exp_avg_sq"} <= set(state) and
                           all(bool(torch.isfinite(state[key]).all()) for key in ("step", "exp_avg", "exp_avg_sq"))
                           for state in states.values()), "优化器动量含非有限值或缺失")
            gl.require(all(float(state["step"]) == step for state in states.values()), "优化器步数与训练计数不匹配")
        # Validate streams before mutating the runtime model.
        np.random.RandomState().set_state(payload["sampler_state"])
        random.Random().setstate(payload["rng_state"]["python"])
        np.random.RandomState().set_state(payload["rng_state"]["numpy"])
        torch.Generator(device="cpu").set_state(payload["rng_state"]["torch"])
        if torch.device(self.device).type == "cuda":
            gl.require(payload["rng_state"]["cuda"] is not None, "CUDA RNG 缺失，不能精确 resume")
            torch.Generator(device=self.device).set_state(payload["rng_state"]["cuda"])
        self.model.worker.load_state_dict(payload["worker"], strict=True)
        self.model.candidate.load_state_dict(payload["candidate"], strict=True)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        self.step = step
        self.sampler.set_state(payload["sampler_state"])
        restore_rng(payload["rng_state"], self.device)
        self.check_ownership()


@torch.no_grad()
def evaluate(model, cache, split="validation", batch_size=256, max_worker_rows=None):
    model.eval()
    device = next(model.worker.parameters()).device
    rows = cache.worker_rows[split]
    if max_worker_rows and len(rows) > max_worker_rows:
        rows = np.sort(np.random.RandomState(19).choice(rows, max_worker_rows, replace=False))
    sums = {key: 0.0 for key in ("worker_nll", "original_nll", "zero_goal_nll", "shuffled_goal_nll", "prototype_goal_nll",
                                "worker_accuracy", "original_accuracy", "worker_entropy", "goal_sensitivity_l1")}
    action_hist = np.zeros(cache.metadata["action_dim"], dtype=np.int64)
    prediction_hist = np.zeros_like(action_hist)
    rng = np.random.RandomState(23)
    for start in range(0, len(rows), batch_size):
        chosen = rows[start:start + batch_size]
        features, goals, remaining, actions = cache.worker_batch(chosen, device)
        alternate = rng.choice(cache.candidate_rows[split], len(chosen), replace=True)
        shuffled = torch.as_tensor(cache.tables["goals"][alternate], device=device)
        classes = cache.tables["candidate_goal"][cache.tables["worker_segment"][chosen]]
        prototype = model.library.goal(torch.as_tensor(classes, device=device))
        with gl.encoder_precision(device):
            distributions = {"worker": model.action_dist(features, goals, remaining),
                "original": model.action_dist(features, goals, remaining, "original"),
                "zero_goal": model.action_dist(features, goals, remaining, "zero_goal"),
                "shuffled_goal": model.action_dist(features, shuffled, remaining),
                "prototype_goal": model.action_dist(features, prototype, remaining)}
            onehot = F.one_hot(actions, cache.metadata["action_dim"]).float()
            for name, distribution in distributions.items():
                sums[name + "_nll"] += float(-distribution.log_prob(onehot).sum())
            for name in ("worker", "original"):
                sums[name + "_accuracy"] += int((distributions[name].probs.argmax(-1) == actions).sum())
            sums["worker_entropy"] += float(distributions["worker"].entropy().sum())
            sums["goal_sensitivity_l1"] += float((distributions["worker"].probs - distributions["shuffled_goal"].probs).abs().sum())
        action_hist += np.bincount(actions.cpu().numpy(), minlength=len(action_hist))
        prediction_hist += np.bincount(distributions["worker"].probs.argmax(-1).cpu().numpy(), minlength=len(action_hist))
    result = {key: value / len(rows) for key, value in sums.items()}
    result.update(worker_labels=len(rows), action_label_histogram=action_hist.tolist(), action_prediction_histogram=prediction_hist.tolist())
    cr = cache.candidate_rows[split]
    labels = cache.tables["candidate_goal"][cr]
    train_counts = np.bincount(cache.tables["candidate_goal"][cache.candidate_rows["train"]], minlength=cache.metadata["goal_count"])
    majority = int(train_counts.argmax())
    frequency_top4 = np.argsort(-train_counts, kind="stable")[:min(4, cache.metadata["goal_count"])]
    frequency_probabilities = (train_counts + 1) / (train_counts.sum() + len(train_counts))
    predicted, correct_top4, cross_entropy = [], 0, 0.0
    for start in range(0, len(cr), batch_size):
        features, target = cache.candidate_batch(cr[start:start + batch_size], device)
        with gl.encoder_precision(device):
            logits = model.candidate(features)
            cross_entropy += float(F.cross_entropy(logits, target, reduction="sum"))
            predicted.extend(logits.argmax(-1).cpu().tolist())
            top = logits.topk(min(4, cache.metadata["goal_count"]), -1).indices
            correct_top4 += int((top == target[:, None]).any(-1).sum())
    predicted = np.array(predicted)
    recalls = [float((predicted[labels == c] == c).mean()) if np.any(labels == c) else None for c in range(cache.metadata["goal_count"])]
    result.update(candidate_ce=cross_entropy / len(cr), candidate_accuracy=float((predicted == labels).mean()),
        candidate_top4_recall=correct_top4 / len(cr), candidate_macro_recall=float(np.mean([v for v in recalls if v is not None])),
        candidate_per_class_recall=recalls, candidate_majority_baseline=float((labels == majority).mean()), candidate_majority_class=majority,
        candidate_frequency_top4_baseline=float(np.isin(labels, frequency_top4).mean()), candidate_frequency_top4_classes=frequency_top4.tolist(),
        candidate_frequency_ce=float(-np.log(frequency_probabilities[labels]).mean()),
        candidate_label_histogram=np.bincount(labels, minlength=cache.metadata["goal_count"]).tolist(),
        candidate_prediction_histogram=np.bincount(predicted, minlength=cache.metadata["goal_count"]).tolist(), candidate_segments=len(cr))
    gl.require(all(np.isfinite(value) for value in result.values() if isinstance(value, (int, float))), "验证指标非有限")
    return result
