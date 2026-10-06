"""T01 experiment scaffolding and explicit checkpoint migration.

Goal-conditioned execution/training is connected in later TODO steps. T01
provides initialized modules and checkpoint contracts, never a flat fallback
masquerading as a trained hierarchical policy.
"""

import copy
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

import tools


MODES = ("flat_ls", "goal_worker", "hierarchical")
FORMAT = "ls_imagine_experiment_v1"


def mode(config):
    value = getattr(config, "experiment_mode", "flat_ls")
    if value not in MODES:
        raise ValueError(f"未知 experiment_mode={value}；可选 {MODES}")
    return value


def require_flat_execution(config):
    if mode(config) != "flat_ls":
        raise NotImplementedError(
            "T01 新模式已支持模块初始化及保存/恢复；真实目标控制需要 T02/T04，"
            "分层训练还需要宏模型与高层策略。请先运行 scripts/t01_checkpoint_check.py。"
        )
    if getattr(config, "freeze_wm", False):
        raise ValueError("原 flat_ls 训练路径不支持冻结 WM；冻结原型请使用新模式的独立入口")


class GoalWorker(nn.Module):
    def __init__(self, actor, goal_dim, horizon):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.goal_dim = goal_dim
        self.horizon = horizon
        first = self.actor.layers.Actor_linear0
        self.feature_dim = first.in_features
        self.actor.layers.Actor_linear0 = nn.Linear(
            self.feature_dim + goal_dim + 1, first.out_features,
            bias=first.bias is not None, device=first.weight.device, dtype=first.weight.dtype,
        )
        self.initialize_from_actor(actor)

    def initialize_from_actor(self, actor):
        source = actor.state_dict()
        target = self.actor.state_dict()
        if set(source) != set(target):
            raise ValueError("低层与原 actor 的权重名称不一致")
        first_name = "layers.Actor_linear0.weight"
        expanded = torch.zeros_like(target[first_name])
        old = source[first_name]
        if old.shape != (expanded.shape[0], self.feature_dim):
            raise ValueError("原 actor 第一层形状不符合低层初始化约定")
        expanded[:, :self.feature_dim] = old.to(expanded.device)
        replacement = dict(source)
        replacement[first_name] = expanded
        self.actor.load_state_dict(replacement, strict=True)

    def forward(self, features, goal, remaining):
        if features.shape[-1] != self.feature_dim or goal.shape != features.shape[:-1] + (self.goal_dim,):
            raise ValueError("低层需要匹配的当前状态特征和 goal 特征")
        if remaining.shape == features.shape[:-1]:
            remaining = remaining.unsqueeze(-1)
        if remaining.shape != features.shape[:-1] + (1,):
            raise ValueError("remaining 需要与状态批次匹配的一个剩余步数")
        if not torch.isfinite(goal).all() or not torch.isfinite(remaining).all():
            raise ValueError("goal/remaining 含非有限值")
        if torch.any(remaining < 1) or torch.any(remaining > self.horizon):
            raise ValueError(f"remaining 必须位于 [1, {self.horizon}]")
        inputs = torch.cat((features, goal, remaining / self.horizon), dim=-1)
        return self.actor(inputs)


def configure_experiment(agent, config):
    current_mode = mode(config)
    agent.experiment_mode = current_mode
    agent._experiment_modules = nn.ModuleDict()
    agent._experiment_optimizers = {}
    agent._goal_library = None
    agent._goal_encoder_id = None
    agent._worker_version = 0
    agent._initialization_source = None
    if current_mode == "flat_ls":
        return  # No new parameters, initializers or RNG draws on the baseline.
    if config.compile:
        raise ValueError("T01 新模式需要 compile=False，以便明确检查模块及优化器")
    if config.actor["dist"] != "onehot":
        raise ValueError("T01 目标策略原型目前使用 MineDojo 的 onehot 动作接口")
    if min(config.goal_dim, config.goal_count, config.macro_max_steps) < 1:
        raise ValueError("goal_dim/goal_count/macro_max_steps 必须为正数")
    worker = GoalWorker(agent._task_behavior.actor, config.goal_dim, config.macro_max_steps).to(config.device)
    agent._experiment_modules["worker"] = worker
    if current_mode == "hierarchical":
        # Initial goal logits only; candidate generation and macro dynamics
        # are implemented later, and are not considered trained by this head.
        manager = nn.Linear(worker.feature_dim, config.goal_count).to(config.device)
        nn.init.zeros_(manager.weight)
        nn.init.zeros_(manager.bias)
        agent._experiment_modules["manager"] = manager
    for name, module in agent._experiment_modules.items():
        agent._experiment_optimizers[name] = torch.optim.Adam(module.parameters(), lr=config.goal_learning_rate)
    if config.freeze_wm:
        agent._wm.requires_grad_(False)


def canonical(name):
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def normalize(mapping):
    result = {}
    for name, value in mapping.items():
        key = canonical(name)
        if key in result:
            raise ValueError(f"编译前缀规范化后名称冲突: {key}")
        result[key] = value
    return result


def read_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    result = torch.load(path, **kwargs)
    if not isinstance(result, dict) or not isinstance(result.get("agent_state_dict"), dict):
        raise ValueError("checkpoint 必须包含 agent_state_dict")
    return result


def json_config(config):
    return json.loads(json.dumps(vars(config), default=str))


def identity(config):
    values = json_config(config)
    # Resume checks every semantic/learning setting, including optimizer and
    # wrapper settings. Only destinations, budget and execution backend vary.
    runtime = {
        "logdir", "traindir", "evaldir", "offline_traindir", "offline_evaldir",
        "results_dir", "name", "steps", "eval_every", "eval_episode_num",
        "device", "compile", "init_checkpoint", "checkpoint_load",
        "use_wandb", "wandb_key", "video_pred_log",
    }
    return {key: value for key, value in values.items() if key not in runtime}


def compatibility(target, source):
    return {
        "missing_keys": sorted(set(target) - set(source)),
        "unexpected_keys": sorted(set(source) - set(target)),
        "mismatches": {
            key: {"saved_shape": list(source[key].shape), "expected_shape": list(target[key].shape),
                  "saved_dtype": str(source[key].dtype), "expected_dtype": str(target[key].dtype)}
            for key in set(source) & set(target)
            if source[key].shape != target[key].shape or source[key].dtype != target[key].dtype
        },
    }


def require_compatible(target, source):
    if not all(torch.is_tensor(value) for value in source.values()):
        raise ValueError("agent_state_dict 中存在非 Tensor 项")
    result = compatibility(target, source)
    if any(result.values()):
        raise ValueError("权重不兼容: " + json.dumps(result, ensure_ascii=False))
    # Different saved values for aliases would silently overwrite one another
    # even with strict=True. Check aliases before loading any weights.
    groups = {}
    for key, tensor in target.items():
        signature = (tensor.device, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if signature in groups:
            previous = source[groups[signature]]
            if previous.data_ptr() != source[key].data_ptr() and not torch.equal(previous, source[key]):
                raise ValueError(f"共享参数的保存值冲突: {groups[signature]} / {key}")
        else:
            groups[signature] = key
    return result


def base_optimizers(agent):
    saved = tools.recursively_collect_optim_state_dict(agent)
    result = {}
    for path in saved:
        obj = agent
        for part in path.split("."):
            obj = getattr(obj, part)
        key = canonical(path)
        if key in result:
            raise ValueError(f"优化器路径冲突: {key}")
        result[key] = (path, obj)
    return result


def amp_scalers(agent):
    result = {}
    for name, (path, _) in base_optimizers(agent).items():
        owner = agent
        for part in path.split(".")[:-1]:
            owner = getattr(owner, part)
        if hasattr(owner, "_scaler"):
            result[name] = owner._scaler
    return result


def initialize_from_checkpoint(agent, checkpoint, source_path):
    if checkpoint.get("verification_artifact"):
        raise ValueError("合成验收 checkpoint 不能用于真实模型初始化")
    if checkpoint.get("checkpoint_format") not in (None, FORMAT):
        raise ValueError("未知 checkpoint 格式")
    if checkpoint.get("checkpoint_format") == FORMAT and checkpoint["experiment"]["mode"] != "flat_ls":
        raise ValueError("已有目标策略 checkpoint 请使用 resume，不能当作旧 flat 模型重新初始化")
    current = agent.state_dict()
    target = normalize(current)
    base = {key: value for key, value in target.items() if not key.startswith("_experiment_modules.")}
    source = normalize(checkpoint["agent_state_dict"])
    result = require_compatible(base, source)
    combined = dict(target)
    combined.update(source)
    agent.load_state_dict({name: combined[canonical(name)] for name in current}, strict=True)
    if agent.experiment_mode != "flat_ls":
        agent._experiment_modules["worker"].initialize_from_actor(agent._task_behavior.actor)
    # Preserve counters for already collected data in this new run. Legacy
    # checkpoint counters/optimizers are never inferred or transplanted.
    for _, optimizer in base_optimizers(agent).values():
        optimizer.state.clear()
    for optimizer in agent._experiment_optimizers.values():
        optimizer.state.clear()
    agent._should_pretrain._once = False
    agent._initialization_source = {
        "path": str(Path(source_path).resolve()), "legacy_training_step": None,
        "optimizer_states_imported": False,
    }
    agent._goal_library = agent._goal_encoder_id = None
    agent._worker_version = 0
    return result


def counters(agent):
    return {
        "step": agent._step, "update_count": agent._update_count, "logger_step": agent._logger.step,
        "schedulers": {name: getattr(agent, name)._last for name in ("_should_train", "_should_log", "_should_reset")},
        "pretrain_once": agent._should_pretrain._once,
        "value_updates": getattr(agent._task_behavior, "_updates", None),
        "jump_prob": agent._task_behavior.jump_prob,
    }


def capture_rng(agent):
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    device = torch.device(agent._config.device)
    result["cuda"] = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return result


def save_checkpoint(agent, path, *, verification_artifact=False, metadata=None, overwrite=False):
    path = Path(path).expanduser()
    temporary = path.with_name(path.name + ".tmp")
    if (path.exists() and not overwrite) or temporary.exists():
        raise FileExistsError(f"拒绝覆盖已有 checkpoint/临时文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format": FORMAT, "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "new_optims_state_dict": {name: opt.state_dict() for name, opt in agent._experiment_optimizers.items()},
        "amp_scalers_state_dict": {name: scaler.state_dict() for name, scaler in amp_scalers(agent).items()},
        "config": json_config(agent._config), "identity": identity(agent._config),
        "experiment": {"mode": agent.experiment_mode, "stage": "t01_scaffold", "freeze_wm": getattr(agent._config, "freeze_wm", False),
                       "goal_encoder_id": agent._goal_encoder_id, "goal_library": agent._goal_library,
                       "worker_version": agent._worker_version, "worker_architecture": "expanded_actor_v1",
                       "initialization_source": agent._initialization_source},
        "training_state": counters(agent), "rng_state": capture_rng(agent),
        "verification_artifact": bool(verification_artifact), "metadata": metadata or {},
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_optimizer(optimizer, saved, name):
    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError(f"优化器状态格式异常: {name}")
    groups = saved.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError(f"优化器参数组数量不匹配: {name}")
    indices = []
    for current, old in zip(optimizer.param_groups, groups):
        if len(current["params"]) != len(old["params"]):
            raise ValueError(f"优化器参数数量不匹配: {name}")
        for parameter, index in zip(current["params"], old["params"]):
            indices.append(index)
            for slot, value in saved.get("state", {}).get(index, {}).items():
                if torch.is_tensor(value) and slot != "step" and value.shape != parameter.shape:
                    raise ValueError(f"优化器状态形状不匹配: {name}/{slot}")
    if len(indices) != len(set(indices)) or set(saved["state"]) - set(indices):
        raise ValueError(f"优化器参数编号异常: {name}")


def validate_training_state(training, rng, experiment, config):
    for key in ("step", "update_count", "logger_step"):
        if type(training.get(key)) is not int or training[key] < 0:
            raise ValueError("checkpoint 训练计数器异常")
    if training["logger_step"] != training["step"] * config.action_repeat:
        raise ValueError("logger_step 与真实交互计数不一致")
    schedulers = training.get("schedulers", {})
    if set(schedulers) != {"_should_train", "_should_log", "_should_reset"}:
        raise ValueError("checkpoint 调度器集合不匹配")
    if any(value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0)
           for value in schedulers.values()):
        raise ValueError("checkpoint 调度器计数异常")
    if type(training.get("pretrain_once")) is not bool:
        raise ValueError("checkpoint 预训练标志异常")
    updates = training.get("value_updates")
    if config.critic["slow_target"] and (type(updates) is not int or updates < 0):
        raise ValueError("checkpoint value 更新计数异常")
    if not isinstance(training.get("jump_prob"), (int, float)) or not 0 <= training["jump_prob"] <= 1:
        raise ValueError("checkpoint 跳跃概率异常")
    if type(experiment.get("worker_version")) is not int or experiment["worker_version"] < 0:
        raise ValueError("checkpoint 低层版本异常")
    if experiment.get("freeze_wm") != config.freeze_wm:
        raise ValueError("checkpoint 冻结设置不匹配")
    if experiment.get("goal_library") is not None and not experiment.get("goal_encoder_id"):
        raise ValueError("目标库必须带有目标编码器标识")
    # Validate RNG metadata with local generators before changing the agent.
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator(device="cpu").set_state(rng["torch"])
    if rng["cuda"] is not None and torch.device(config.device).type == "cuda":
        torch.Generator(device=config.device).set_state(rng["cuda"])


def restore_checkpoint(agent, checkpoint, *, allow_verification=False):
    if checkpoint.get("checkpoint_format") != FORMAT:
        raise ValueError("旧 latest.pt 缺少训练计数等元数据，只能 initialize，不能精确 resume")
    if checkpoint.get("verification_artifact") and not allow_verification:
        raise ValueError("这是合成验收 checkpoint，不能恢复到真实训练；请使用原模型初始化")
    if checkpoint.get("identity") != identity(agent._config):
        raise ValueError("checkpoint 的模式、任务或模型/优化器配置不匹配")
    saved_experiment = checkpoint["experiment"]
    if saved_experiment["mode"] != agent.experiment_mode or saved_experiment["worker_architecture"] != "expanded_actor_v1":
        raise ValueError("模式或低层结构版本不兼容")
    if agent._goal_encoder_id is not None and agent._goal_encoder_id != saved_experiment["goal_encoder_id"]:
        raise ValueError("目标编码器标识不同，不能复用目标编号")
    current = agent.state_dict()
    saved = normalize(checkpoint["agent_state_dict"])
    require_compatible(normalize(current), saved)
    base = base_optimizers(agent)
    saved_base = normalize(checkpoint["optims_state_dict"])
    saved_new = checkpoint["new_optims_state_dict"]
    if set(base) != set(saved_base) or set(agent._experiment_optimizers) != set(saved_new):
        raise ValueError("checkpoint 的优化器集合不匹配")
    for name, (_, optimizer) in base.items():
        validate_optimizer(optimizer, saved_base[name], name)
    for name, optimizer in agent._experiment_optimizers.items():
        validate_optimizer(optimizer, saved_new[name], name)
    scalers = amp_scalers(agent)
    saved_scalers = checkpoint["amp_scalers_state_dict"]
    if set(scalers) != set(saved_scalers):
        raise ValueError("checkpoint AMP scaler 集合不匹配")
    for name, scaler in scalers.items():
        # A scaler with AMP enabled may still be lazily initialized. Loading
        # a copied scaler validates metadata without changing the live one.
        copy.deepcopy(scaler).load_state_dict(saved_scalers[name])
    training = checkpoint["training_state"]
    validate_training_state(training, checkpoint["rng_state"], saved_experiment, agent._config)
    agent.load_state_dict({name: saved[canonical(name)] for name in current}, strict=True)
    for name, (_, optimizer) in base.items():
        optimizer.load_state_dict(saved_base[name])
    for name, optimizer in agent._experiment_optimizers.items():
        optimizer.load_state_dict(saved_new[name])
    for name, scaler in scalers.items():
        scaler.load_state_dict(saved_scalers[name])
    agent._step, agent._update_count = training["step"], training["update_count"]
    agent._logger.step = training["logger_step"]
    for name, last in training["schedulers"].items():
        getattr(agent, name)._last = last
    agent._should_pretrain._once = training["pretrain_once"]
    if training["value_updates"] is not None:
        agent._task_behavior._updates = training["value_updates"]
    agent._task_behavior.jump_prob = training["jump_prob"]
    agent._goal_library = saved_experiment["goal_library"]
    agent._goal_encoder_id = saved_experiment["goal_encoder_id"]
    agent._worker_version = saved_experiment["worker_version"]
    agent._initialization_source = saved_experiment["initialization_source"]
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.device(agent._config.device).type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], torch.device(agent._config.device))

