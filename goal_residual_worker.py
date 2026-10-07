"""T05 paired executable residual policies over a frozen no-goal BC actor.

Training consumes accepted real state/action tables. WM and goal encoding
are instantiated only by the optional online runtime, never by the trainer.
Raw actor preferences are corrected before the original onehot unimix.
"""

import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_information_probe as probe
import goal_library as gl
import long_horizon as lh
import tools
from goal_sampling import WorkerRowSampler


FORMAT = "ls_imagine_goal_residual_worker_v1"
ARCHITECTURE = "frozen_bc_raw_preference_zero_initialized_residual_v1"
SEMANTICS = "paired_real_goal_and_no_goal_residual_action_policies"
SAMPLING = "expanded_row_uniform_v1"
NAMES = ("goal", "no_goal")
DEFAULTS = copy.deepcopy(probe.DEFAULTS)
DEPENDENCIES = ("base_checkpoint", "base_verify_report", "frozen_bundle")


def base_identity(payload):
    return gl.tensor_digest(payload["worker"], {
        "cache_id": payload["cache_id"], "bundle_id": payload["frozen_bundle"]["bundle_id"],
        "options": payload["options"], "counters": payload["counters"],
        "architecture": payload["worker_architecture"]})


def validate_base(payload, cache):
    gl.require(payload.get("checkpoint_format") == bc.CHECKPOINT_FORMAT and
               payload.get("stage") == "t04_offline_bc" and
               payload.get("experiment_mode") == "goal_worker" and
               payload.get("worker_architecture") == "expanded_actor_v1" and
               payload.get("candidate_semantics") == "current_state_future_class_forecast" and
               not payload.get("verification_artifact"), "需要真实已训练T04底座，不能用探针或合成验收文件")
    bc.validate_checkpoint_sampling(payload)
    options = bc.normalize_training_options(payload["options"])
    gl.require(options["conditioning"] == "no_goal" and options["worker_sampling"] == "uniform_rows",
               "残差实验底座必须为独立no_goal且使用uniform_rows")
    bc.validate_bundle(payload["frozen_bundle"])
    gl.require(payload["cache_id"] == cache.cache_id and
               payload["frozen_bundle"]["bundle_id"] == cache.bundle["bundle_id"], "底座与缓存/冻结依赖身份不同")
    step = payload["counters"]["step"]
    gl.require(type(step) is int and step > 0 and payload["counters"] == dict(
        step=step, worker_updates=step, candidate_updates=step, worker_version=step,
        worker_labels_seen=step * options["batch_size"],
        candidate_labels_seen=step * options["batch_size"], new_env_steps=0), "底座训练计数异常")


def load_base(path, verify_report, cache):
    """Require the passed T04 verify record for this exact base file."""
    path, verify_report = Path(path).resolve(), Path(verify_report).resolve()
    payload = bc.torch_load(path)
    validate_base(payload, cache)
    report = json.loads(verify_report.read_text(encoding="utf-8"))
    stat = path.stat()
    signature = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    required = {"strict_load", "real_supervision", "causal_cache", "next_update_equivalence", "frozen_ownership"}
    gl.require(report.get("command") == "verify" and report.get("status") in ("passed", "passed_with_warnings") and
               report.get("cache_id") == cache.cache_id and report.get("bundle_id") == cache.bundle["bundle_id"] and
               report.get("counters") == payload["counters"] and
               report.get("source_inputs_before", {}).get(str(path)) == signature and
               required <= {row["name"] for row in report["checks"] if row["level"] == "PASS"},
               "底座验收报告与指定文件/缓存不同；请对该底座独立运行T04 verify")
    reference = {
        "base_checkpoint": {"source_path": path, "sha256": bc.file_hash(path)},
        "base_verify_report": {"source_path": verify_report, "sha256": bc.file_hash(verify_report)},
        "frozen_bundle": {"source_path": (cache.directory / "frozen_bundle.pt").resolve(),
                          "sha256": bc.file_hash(cache.directory / "frozen_bundle.pt")}}
    return payload, reference


class ResidualData(probe.ProbeData):
    def __init__(self, cache, options, base_id):
        super().__init__(cache, options)
        self.base_id = base_id

    def identity(self):
        return dict(super().identity(), base_id=self.base_id)


class ResidualHead(probe.ActionHead):
    def __init__(self, identity, options):
        super().__init__(identity, options)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)


class ResidualPolicy(nn.Module):
    def __init__(self, cache, base_payload, identity, options, device):
        super().__init__()
        validate_base(base_payload, cache)
        gl.require(base_identity(base_payload) == identity["base_id"], "底座内容身份不匹配")
        self.identity, self.options = copy.deepcopy(identity), copy.deepcopy(options)
        self.bundle = cache.bundle
        original = bc.original_actor(cache.bundle["config"], cache.bundle["feature_dim"],
                                     cache.bundle["original_actor"], device)
        self.base = lh.GoalWorker(original, identity["goal_dim"], identity["horizon"]).to(device)
        lh.require_compatible(self.base.state_dict(), base_payload["worker"])
        gl.require(all(bool(torch.isfinite(value).all()) for value in base_payload["worker"].values()), "底座权重非有限")
        self.base.load_state_dict(base_payload["worker"], strict=True)
        self.base.requires_grad_(False).eval()
        first = ResidualHead(identity, options).to(device)
        self.heads = nn.ModuleDict({"goal": first, "no_goal": copy.deepcopy(first)})
        self.base_hash = gl.tensor_digest(self.base.state_dict(), {})
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def validate_inputs(self, features, goals, remaining):
        i = self.identity
        gl.require(features.ndim == 2 and features.shape[1] == i["feature_dim"] and
                   goals.shape == (len(features), i["goal_dim"]) and remaining.shape == (len(features),),
                   "残差worker输入形状错误")
        gl.require(features.dtype == goals.dtype == torch.float32 and remaining.dtype == torch.int64 and
                   features.device == goals.device == remaining.device == next(self.parameters()).device and
                   bool(torch.isfinite(features).all()) and bool(torch.isfinite(goals).all()),
                   "残差worker输入类型、设备或数值错误")
        gl.require(bool(((remaining >= 1) & (remaining <= i["horizon"])).all()),
                   f"remaining 必须位于 [1, {i['horizon']}]")

    @torch.no_grad()
    def base_preferences(self, features, goals, remaining):
        self.validate_inputs(features, goals, remaining)
        # Match GoalWorker + MLP.forward exactly. Do not take log(probs):
        # those probabilities already include onehot's uniform mixture.
        inputs = torch.cat((features, torch.zeros_like(goals),
                            remaining.unsqueeze(-1) / self.identity["horizon"]), -1)
        actor = self.base.actor
        gl.require(actor._dist == "onehot" and not isinstance(actor._shape, dict), "底座需要单一onehot动作头")
        if actor._symlog_inputs:
            inputs = tools.symlog(inputs)
        return actor.mean_layer(actor.layers(inputs))

    def correction(self, name, features, goals, remaining):
        gl.require(name in NAMES, "未知残差条件")
        self.validate_inputs(features, goals, remaining)
        return self.heads[name](features, torch.zeros_like(goals) if name == "no_goal" else goals, remaining)

    def distribution(self, raw_preferences):
        gl.require(bool(torch.isfinite(raw_preferences).all()), "残差动作偏好非有限")
        return tools.OneHotDist(raw_preferences, unimix_ratio=self.base.actor._unimix_ratio)

    def action_dist(self, features, goals, remaining, mode="goal"):
        gl.require(mode in (*NAMES, "base", "zero_goal"), "未知残差worker行为模式")
        with gl.encoder_precision(features.device):
            base = self.base_preferences(features, goals, remaining)
            if mode == "base":
                return self.distribution(base)
            name = "goal" if mode == "zero_goal" else mode
            goal = torch.zeros_like(goals) if mode == "zero_goal" else goals
            return self.distribution(base + self.correction(name, features, goal, remaining))

    def check_frozen(self, content=False):
        gl.require(not self.base.training and all(not p.requires_grad and p.grad is None for p in self.base.parameters()),
                   "底座冻结或梯度所有权异常")
        if content:
            gl.require(gl.tensor_digest(self.base.state_dict(), {}) == self.base_hash, "冻结底座内容改变")


class Trainer:
    def __init__(self, data, base_payload, reference, options, device):
        probe.validate_options(options)
        self.data, self.options, self.device = data, copy.deepcopy(options), device
        self.identity, self.reference = data.identity(), reference
        self.model = ResidualPolicy(data.cache, base_payload, self.identity, options, device)
        self.optimizers = {name: torch.optim.Adam(self.model.heads[name].parameters(), lr=options["learning_rate"])
                           for name in NAMES}
        self.sampler = np.random.RandomState(options["seed"])
        self.worker_sampler = WorkerRowSampler(data.cache.tables["remaining"], data.cache.tables["worker_train"],
                                               self.identity["horizon"], "uniform_rows")
        self.step = 0
        self.check_ownership()

    def check_ownership(self):
        groups = [{id(p) for group in self.optimizers[name].param_groups for p in group["params"]} for name in NAMES]
        frozen = {id(p) for p in self.model.base.parameters()}
        gl.require(not groups[0] & groups[1] and all(not group & frozen and
                   group == {id(p) for p in self.model.heads[name].parameters()} and
                   all(p.requires_grad for p in self.model.heads[name].parameters())
                   for group, name in zip(groups, NAMES)), "残差优化器包含底座/共享参数或分支不可训练")
        self.model.check_frozen()

    def sampling_plan(self):
        return dict(self.worker_sampler.plan(), paired_batches_identical=True,
                    candidate_sampling="not applicable; candidate not loaded or trained",
                    validation_weighting="full expanded rows and fixed deduplicated queries")

    def update(self):
        rows = self.worker_sampler.draw(self.sampler, self.options["batch_size"])
        features, goals, remaining, labels = self.data.cache.worker_batch(rows, self.device)
        result = dict(step=self.step + 1, shared_batch_id=probe.array_digest({"rows": rows}),
                      remaining_histogram=self.worker_sampler.histogram(rows).tolist())
        self.model.train(True)
        with gl.encoder_precision(self.device):
            base = self.model.base_preferences(features, goals, remaining)
            for name in NAMES:
                optimizer = self.optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                dist = self.model.distribution(base + self.model.correction(name, features, goals, remaining))
                # Train the same mixed action distribution that is deployed.
                loss = F.nll_loss(dist.logits, labels)
                gl.require(bool(torch.isfinite(loss)), "残差BC损失非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.heads[name].parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "残差BC梯度非有限")
                optimizer.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.model.heads[name].parameters()), "残差权重非有限")
                result[name + "_nll"], result[name + "_grad_norm"] = float(loss.detach()), float(norm)
        self.step += 1
        self.check_ownership()
        return result

    def counters(self):
        return dict(step=self.step, goal_updates=self.step, no_goal_updates=self.step,
                    labels_seen_per_branch=self.step * self.options["batch_size"], optimizer_updates=2 * self.step,
                    base_updates=0, wm_updates=0, new_env_steps=0)

    def payload(self, verification_artifact=False):
        return dict(checkpoint_format=FORMAT, architecture=ARCHITECTURE, semantics=SEMANTICS,
                    sampling_protocol=SAMPLING, input_identity=self.identity, options=self.options,
                    heads={name: bc.cpu_tree(head.state_dict()) for name, head in self.model.heads.items()},
                    optimizers={name: bc.cpu_tree(opt.state_dict()) for name, opt in self.optimizers.items()},
                    counters=self.counters(), sampler_state=self.sampler.get_state(),
                    rng_state=bc.capture_rng(self.device), backend=bc.backend_info(self.device),
                    verification_artifact=bool(verification_artifact))

    def restore(self, payload, allow_verification=False):
        validate_payload(payload, self.identity, self.options, allow_verification)
        gl.require(payload["backend"] == bc.backend_info(self.device), "残差精确恢复需要相同运行后端")
        step = payload["counters"]["step"]
        for name in NAMES:
            lh.require_compatible(self.model.heads[name].state_dict(), payload["heads"][name])
            opt, saved = self.optimizers[name], payload["optimizers"][name]
            lh.validate_optimizer(opt, saved, name)
            for current, group in zip(opt.param_groups, saved["param_groups"]):
                gl.require({k: v for k, v in current.items() if k != "params"} ==
                           {k: v for k, v in group.items() if k != "params"}, "残差优化器配置不同")
            indices = {index for group in saved["param_groups"] for index in group["params"]}
            states = saved["state"]
            gl.require((step == 0 and not states) or (step > 0 and set(states) == indices), "残差优化器状态缺失")
            for state in states.values():
                gl.require({"step", "exp_avg", "exp_avg_sq"} <= set(state) and float(state["step"]) == step and
                           all(bool(torch.isfinite(state[key]).all()) for key in ("step", "exp_avg", "exp_avg_sq")),
                           "残差优化器步数或数值异常")
        np.random.RandomState(0).set_state(payload["sampler_state"])
        random.Random(0).setstate(payload["rng_state"]["python"])
        np.random.RandomState(0).set_state(payload["rng_state"]["numpy"])
        torch.Generator(device="cpu").set_state(payload["rng_state"]["torch"])
        if torch.device(self.device).type == "cuda":
            gl.require(payload["rng_state"]["cuda"] is not None, "残差CUDA RNG缺失")
            torch.Generator(device=self.device).set_state(payload["rng_state"]["cuda"])
        # Every stream is validated before any runtime mutation.
        for name in NAMES:
            self.model.heads[name].load_state_dict(payload["heads"][name], strict=True)
            self.optimizers[name].load_state_dict(payload["optimizers"][name])
        self.step = step
        self.sampler.set_state(payload["sampler_state"])
        bc.restore_rng(payload["rng_state"], self.device)
        self.check_ownership()


def validate_payload(payload, identity, options, allow_verification=False):
    gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("architecture") == ARCHITECTURE and
               payload.get("semantics") == SEMANTICS, "残差worker格式不兼容；不能加载小探针或原T04文件")
    gl.require(allow_verification or not payload.get("verification_artifact"), "合成验收残差文件不能用于训练或控制")
    gl.require(payload["input_identity"] == identity, "残差缓存/目标/底座/查询身份不同")
    gl.require(payload["options"] == options and payload["sampling_protocol"] == SAMPLING, "残差配置或采样协议不同")
    step = payload["counters"]["step"]
    gl.require(type(step) is int and step >= 0 and payload["counters"] == dict(
        step=step, goal_updates=step, no_goal_updates=step, labels_seen_per_branch=step * options["batch_size"],
        optimizer_updates=2 * step, base_updates=0, wm_updates=0, new_env_steps=0), "残差训练计数异常")
    gl.require(set(payload["heads"]) == set(payload["optimizers"]) == set(NAMES), "残差独立对照缺失")
    gl.require(all(bool(torch.isfinite(v).all()) for state in payload["heads"].values() for v in state.values()),
               "残差权重非有限")


def save_checkpoint(trainer, path, check_sha256, verification_artifact=False, selection=None, overwrite=False):
    path = Path(path)
    payload = trainer.payload(verification_artifact)
    payload["accepted_check_sha256"] = check_sha256
    payload["dependencies"] = {name: dict(path=os.path.relpath(item["source_path"], path.resolve().parent),
                                        sha256=item["sha256"]) for name, item in trainer.reference.items()}
    if selection is not None:
        payload["selection"] = copy.deepcopy(selection)
    bc.save_atomic(payload, path, overwrite)
    return payload


def checkpoint_dependencies(payload, path):
    refs = payload.get("dependencies", {})
    gl.require(set(refs) == set(DEPENDENCIES), "残差共享依赖声明缺失")
    result = {}
    for name, reference in refs.items():
        gl.require(isinstance(reference.get("path"), str) and reference["path"] and
                   not Path(reference["path"]).is_absolute(), "残差共享依赖需要相对路径")
        dependency = (Path(path).resolve().parent / reference["path"]).resolve()
        gl.require(dependency.is_file() and bc.file_hash(dependency) == reference["sha256"],
                   f"残差共享依赖缺失或SHA256不匹配：{name}；请保留底座、验收报告和T04缓存")
        result[name] = dependency
    return result


def read_checkpoint(path, trainer=None, check_sha256=None):
    payload = bc._torch_load(path)
    gl.require(payload.get("checkpoint_format") == FORMAT, "需要残差worker文件，不能加载小探针或原T04文件")
    paths = checkpoint_dependencies(payload, path)
    if trainer is not None:
        gl.require(all(paths[name] == trainer.reference[name]["source_path"] and
                       payload["dependencies"][name]["sha256"] == trainer.reference[name]["sha256"]
                       for name in DEPENDENCIES), "残差恢复时共享依赖不同")
    if check_sha256 is not None:
        gl.require(payload.get("accepted_check_sha256") == check_sha256, "残差checkpoint与独立check记录不同")
    return payload, paths


def load_policy(path, cache, device):
    """Dedicated inference loader; no optimizers or WM constructors."""
    payload, refs = read_checkpoint(path)
    gl.require(not payload.get("verification_artifact"), "合成验收残差文件不能用于训练或控制")
    options = payload["options"]
    probe.validate_options(options)
    base, _ = load_base(refs["base_checkpoint"], refs["base_verify_report"], cache)
    base = {key: value for key, value in base.items() if key not in ("optimizers", "candidate")}
    base["frozen_bundle"] = cache.bundle
    data = ResidualData(cache, options, base_identity(base))
    validate_payload(payload, data.identity(), options)
    gl.require(payload["counters"]["step"] > 0, "真实残差控制需要已训练分支")
    gl.require(refs["frozen_bundle"] == (cache.directory / "frozen_bundle.pt").resolve(), "残差推理冻结依赖路径不同")
    rng = bc.capture_rng(device)
    try:
        model = ResidualPolicy(cache, base, data.identity(), options, device)
        for name in NAMES:
            lh.require_compatible(model.heads[name].state_dict(), payload["heads"][name])
            model.heads[name].load_state_dict(payload["heads"][name], strict=True)
    finally:
        bc.restore_rng(rng, device)
    model.worker_version = payload["counters"]["step"]
    model.model_id = gl.tensor_digest(model.heads.state_dict(), {
        "architecture": ARCHITECTURE, "input_identity": model.identity,
        "options": options, "worker_version": model.worker_version})
    return model.requires_grad_(False).eval()


class OnlineRuntime(nn.Module):
    """Real observation/history -> causal state -> onehot action distribution.

    Caller supplies the fixed real goal and decreasing remaining budget.
    Environment/reward handling and experimental condition order stay with
    the T05 evaluator; this object never declares a visual goal reached.
    """
    def __init__(self, policy, device):
        super().__init__()
        import goal_segments as gs

        self.policy, self.bundle = policy, policy.bundle
        self.state_encoder = gs.FrozenStateEncoder(self.bundle["config"], self.bundle["shapes"], device)
        lh.require_compatible(self.state_encoder.state_dict(), self.bundle["state_encoder"])
        self.state_encoder.load_state_dict(self.bundle["state_encoder"], strict=True)
        self.library = gl.GoalLibrary.from_payload(self.bundle["library"], device)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def action_dist(self, features, goals, remaining, mode="goal"):
        return self.policy.action_dist(features, goals, remaining, mode)

    @torch.no_grad()
    def step(self, observation, incoming, goal, remaining, state=None, mode="goal"):
        gl.require(not observation.get("is_zoomed", False), "残差控制只接受真实观测")
        gl.require(not observation["is_last"] and not observation["is_terminal"], "真实结束后不能继续发出动作")
        state, features = self.state_encoder.step(observation, incoming, state)
        device = features.device
        goals = torch.as_tensor(np.asarray(goal)[None], dtype=torch.float32, device=device)
        gl.require(type(remaining) is int, "remaining必须为整数")
        steps = torch.tensor([remaining], dtype=torch.int64, device=device)
        return state, self.action_dist(features, goals, steps, mode)


@torch.no_grad()
def predict(trainer, rows, donors=None):
    cache, model = trainer.data.cache, trainer.model
    model.eval()
    names = (*NAMES, "base", "zero_goal", "swapped_goal") if donors is not None else (*NAMES, "base", "zero_goal")
    result = dict(rows=np.asarray(rows, np.int64), donors=np.asarray(donors, np.int64) if donors is not None else None,
                  nll={name: [] for name in names}, probabilities={name: [] for name in names},
                  correction_rms={name: [] for name in NAMES})
    for start in range(0, len(rows), trainer.options["batch_size"]):
        chosen = rows[start:start + trainer.options["batch_size"]]
        features, goals, remaining, labels = cache.worker_batch(chosen, trainer.device)
        with gl.encoder_precision(trainer.device):
            base = model.base_preferences(features, goals, remaining)
            corrections = {name: model.correction(name, features, goals, remaining) for name in NAMES}
            raw = {name: base + corrections[name] for name in NAMES}
            raw.update(base=base, zero_goal=base + model.correction("goal", features, torch.zeros_like(goals), remaining))
            if donors is not None:
                alternate = donors[start:start + len(chosen)]
                safe = np.where(alternate >= 0, alternate, chosen)
                swapped = torch.as_tensor(cache.tables["goals"][cache.tables["worker_segment"][safe]], device=trainer.device)
                raw["swapped_goal"] = base + model.correction("goal", features, swapped, remaining)
            for name, value in raw.items():
                dist = model.distribution(value)
                result["nll"][name].append(F.nll_loss(dist.logits, labels, reduction="none").cpu().numpy())
                result["probabilities"][name].append(dist.probs.cpu().numpy())
            for name, value in corrections.items():
                # Preferences have an arbitrary shared offset; report centered RMS.
                centered = value - value.mean(-1, keepdim=True)
                result["correction_rms"][name].append(centered.square().mean(-1).sqrt().cpu().numpy())
    for key in ("nll", "probabilities", "correction_rms"):
        result[key] = {name: np.concatenate(parts) for name, parts in result[key].items()}
    return result


def summarize(data, prediction, positions=None):
    positions = np.arange(len(prediction["rows"])) if positions is None else np.asarray(positions, np.int64)
    result = probe.summarize(data, prediction, positions)
    rows, t = prediction["rows"][positions], data.cache.tables
    episodes, budgets = data.episodes[t["worker_state"][rows]], t["remaining"][rows]

    def extra(selected):
        ix = positions[selected]
        losses, probs = prediction["nll"], prediction["probabilities"]
        return dict(base_minus_goal_nll=float((losses["base"][ix].astype(np.float64) - losses["goal"][ix]).mean()),
                    base_minus_no_goal_nll=float((losses["base"][ix].astype(np.float64) - losses["no_goal"][ix]).mean()),
                    goal_minus_base_accuracy=float(((probs["goal"][ix].argmax(-1) == t["action_id"][rows[selected]]).astype(float) -
                                                   (probs["base"][ix].argmax(-1) == t["action_id"][rows[selected]])).mean()),
                    goal_vs_base_l1=float(np.abs(probs["goal"][ix] - probs["base"][ix]).sum(-1).mean()),
                    goal_vs_base_mode_change_fraction=float((probs["goal"][ix].argmax(-1) != probs["base"][ix].argmax(-1)).mean()),
                    **{name + "_centered_correction_rms": float(prediction["correction_rms"][name][ix].mean()) for name in NAMES})

    result["row_weighted"].update(extra(np.arange(len(rows))))
    for row in result["per_episode"]:
        row.update(extra(np.flatnonzero(episodes == row["episode"])))
    for row in result["per_remaining"]:
        row.update(extra(np.flatnonzero(budgets == row["remaining"])))
    if "remaining_13_plus" in result:
        result["remaining_13_plus"].update(extra(np.flatnonzero(budgets >= 13)))
    for label, values in (("episode_macro", result["per_episode"]), ("remaining_macro", result["per_remaining"])):
        for key in extra(np.arange(len(rows))):
            result[label][key] = float(np.mean([row[key] for row in values]))
    return result


def evaluate(trainer):
    from scripts.t01_checkpoint_check import same

    rng = bc.capture_rng(trainer.device)
    queries = predict(trainer, trainer.data.rows, trainer.data.donors)
    flags = trainer.data.cache.tables["worker_train"][trainer.data.rows]
    full = predict(trainer, trainer.data.cache.worker_rows["validation"])
    result = dict(step=trainer.step, input_identity=trainer.identity,
                  train_queries=summarize(trainer.data, queries, np.flatnonzero(flags)),
                  validation_queries=summarize(trainer.data, queries, np.flatnonzero(~flags)),
                  validation_full=summarize(trainer.data, full),
                  scope="held-out action prediction of runnable policies; real control and success not yet evaluated")
    gl.require(same(rng, bc.capture_rng(trainer.device)), "残差评价推进训练RNG")
    return result, queries, full
