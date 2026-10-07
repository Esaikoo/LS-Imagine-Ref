"""Small paired action predictors, isolated from all runnable agent policies."""

import copy
import hashlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_library as gl
import goal_learning_diagnostics as diag
import long_horizon as lh
from goal_sampling import WorkerRowSampler


FORMAT = "ls_imagine_goal_information_probe_v1"
ARCHITECTURE = "normalized_state_goal_action_head_v1"
SEMANTICS = "diagnostic_next_action_predictor_not_executable_worker"
SAMPLING = "expanded_row_uniform_v1"
NAMES = ("goal", "no_goal")
DEFAULTS = dict(seed=0, batch_size=256, state_dim=128, hidden=256,
                learning_rate=3e-4, grad_clip=10.0,
                train_per_episode=8, validation_per_episode=32, donor_candidates=8)


def validate_options(options):
    gl.require(set(options) == set(DEFAULTS), "探针训练选项字段不兼容")
    for key in set(DEFAULTS) - {"learning_rate", "grad_clip"}:
        gl.require(type(options[key]) is int and (0 <= options[key] < 2**32 if key == "seed" else options[key] > 0),
                   f"探针 {key} 必须为合法整数")
    gl.require(all(np.isfinite(options[key]) and options[key] > 0 for key in ("learning_rate", "grad_clip")),
               "探针学习率和梯度阈值必须有限且为正")


def array_digest(arrays):
    result = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        array = np.ascontiguousarray(array)
        result.update(str((name, array.dtype.str, array.shape)).encode("utf-8"))
        result.update(array.tobytes())
    return result.hexdigest()


class ProbeData:
    def __init__(self, cache, options):
        validate_options(options)
        self.cache = cache
        self.episodes, self.frames = diag.state_index(cache)
        self.rows = diag.probe_rows(cache, self.episodes, options["train_per_episode"],
                                    options["validation_per_episode"], options["seed"])
        self.donors = diag.alternate_rows(cache, self.rows, self.episodes, options["seed"], options["donor_candidates"])
        t = cache.tables
        supported = self.donors >= 0
        donors, rows = self.donors[supported], self.rows[supported]
        gl.require(np.array_equal(t["remaining"][rows], t["remaining"][donors]) and
                   np.array_equal(t["worker_train"][rows], t["worker_train"][donors]) and
                   np.all(self.episodes[t["worker_state"][rows]] != self.episodes[t["worker_state"][donors]]),
                   "目标替换跨划分、持续时间或来自同局")
        self.query_id = array_digest({"rows": self.rows, "donors": self.donors})
        gl.require(all(np.any(t["worker_train"][self.rows] == train) for train in (True, False)), "探针查询缺少训练或留出局")

    def identity(self):
        c = self.cache
        return dict(cache_id=c.cache_id, bundle_id=c.bundle["bundle_id"], dataset_id=c.metadata["dataset_id"],
                    states_sha256=c.metadata["states_sha256"], feature_dim=c.states.shape[1],
                    goal_dim=c.metadata["goal_dim"], action_dim=c.metadata["action_dim"],
                    horizon=c.bundle["horizon"], query_id=self.query_id,
                    action_alignment="obs[t] -> action[t+1]; incoming action[t]")


class ActionHead(nn.Module):
    def __init__(self, identity, options):
        super().__init__()
        self.identity = copy.deepcopy(identity)
        self.state_projection = nn.Linear(identity["feature_dim"], options["state_dim"])
        self.state_norm = nn.LayerNorm(options["state_dim"], elementwise_affine=False)
        self.goal_norm = nn.LayerNorm(identity["goal_dim"], elementwise_affine=False)
        self.layers = nn.Sequential(nn.Linear(options["state_dim"] + identity["goal_dim"] + 1, options["hidden"]),
                                    nn.SiLU(), nn.Linear(options["hidden"], identity["action_dim"]))

    def normalized_inputs(self, features, goals, remaining):
        i = self.identity
        gl.require(features.ndim == 2 and features.shape[1] == i["feature_dim"] and
                   goals.shape == (len(features), i["goal_dim"]) and remaining.shape == (len(features),), "探针输入形状错误")
        gl.require(features.dtype == goals.dtype == torch.float32 and remaining.dtype == torch.int64 and
                   bool(torch.isfinite(features).all()) and bool(torch.isfinite(goals).all()), "探针输入类型或数值错误")
        gl.require(bool(((remaining >= 1) & (remaining <= i["horizon"])).all()), "探针remaining越界")
        return (self.state_norm(self.state_projection(features)), self.goal_norm(goals),
                remaining.float().unsqueeze(-1) / i["horizon"])

    def forward(self, features, goals, remaining):
        return self.layers(torch.cat(self.normalized_inputs(features, goals, remaining), -1))


class Trainer:
    def __init__(self, data, options, device):
        validate_options(options)
        self.data, self.options, self.device = data, copy.deepcopy(options), device
        self.identity = data.identity()
        first = ActionHead(self.identity, options).to(device)
        self.heads = {"goal": first, "no_goal": copy.deepcopy(first)}
        self.optimizers = {name: torch.optim.Adam(head.parameters(), lr=options["learning_rate"])
                           for name, head in self.heads.items()}
        self.sampler = np.random.RandomState(options["seed"])
        self.worker_sampler = WorkerRowSampler(data.cache.tables["remaining"], data.cache.tables["worker_train"],
                                               self.identity["horizon"], "uniform_rows")
        self.step = 0
        self.check_ownership()

    def check_ownership(self):
        groups = [{id(p) for group in self.optimizers[name].param_groups for p in group["params"]} for name in NAMES]
        gl.require(not groups[0] & groups[1] and all(group == {id(p) for p in self.heads[name].parameters()}
                   for group, name in zip(groups, NAMES)), "探针优化器共享参数或包含外部模块")

    def sampling_plan(self):
        return dict(self.worker_sampler.plan(), candidate_sampling="not applicable; no candidate model",
                    validation_weighting="full expanded rows and fixed deduplicated episode queries; no resampling")

    def logits(self, name, features, goals, remaining):
        gl.require(name in NAMES, "未知探针条件")
        return self.heads[name](features, torch.zeros_like(goals) if name == "no_goal" else goals, remaining)

    def update(self):
        rows = self.worker_sampler.draw(self.sampler, self.options["batch_size"])
        features, goals, remaining, labels = self.data.cache.worker_batch(rows, self.device)
        result = {"step": self.step + 1, "shared_batch_id": array_digest({"rows": rows}),
                  "remaining_histogram": self.worker_sampler.histogram(rows).tolist()}
        with gl.encoder_precision(self.device):
            for name in NAMES:
                self.heads[name].train()
                optimizer = self.optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.logits(name, features, goals, remaining), labels)
                gl.require(bool(torch.isfinite(loss)), "探针损失非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.heads[name].parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "探针梯度非有限")
                optimizer.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.heads[name].parameters()), "探针更新后权重非有限")
                result[name + "_nll"], result[name + "_grad_norm"] = float(loss.detach()), float(norm)
        self.step += 1
        return result

    def counters(self):
        return dict(step=self.step, goal_updates=self.step, no_goal_updates=self.step,
                    labels_seen_per_head=self.step * self.options["batch_size"], optimizer_updates=2 * self.step,
                    new_env_steps=0)

    def payload(self, verification_artifact=False):
        return dict(checkpoint_format=FORMAT, architecture=ARCHITECTURE, semantics=SEMANTICS,
                    sampling_protocol=SAMPLING, input_identity=self.identity, options=self.options,
                    heads={name: bc.cpu_tree(head.state_dict()) for name, head in self.heads.items()},
                    optimizers={name: bc.cpu_tree(opt.state_dict()) for name, opt in self.optimizers.items()},
                    counters=self.counters(), sampler_state=self.sampler.get_state(),
                    rng_state=bc.capture_rng(self.device), backend=bc.backend_info(self.device),
                    verification_artifact=bool(verification_artifact))

    def restore(self, payload, allow_verification=False):
        gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("architecture") == ARCHITECTURE and
                   payload.get("semantics") == SEMANTICS, "探针格式或用途不兼容，不能作为真实worker")
        gl.require(allow_verification or not payload.get("verification_artifact"), "合成验收探针不能resume真实训练")
        gl.require(payload["input_identity"] == self.identity, "探针缓存或查询身份不同")
        gl.require(payload["options"] == self.options and payload["sampling_protocol"] == SAMPLING, "探针配置或采样协议不同")
        gl.require(payload["backend"] == bc.backend_info(self.device), "探针精确恢复需要相同运行后端")
        step = payload["counters"]["step"]
        gl.require(type(step) is int and step >= 0 and payload["counters"] == dict(
            step=step, goal_updates=step, no_goal_updates=step, labels_seen_per_head=step * self.options["batch_size"],
            optimizer_updates=2 * step, new_env_steps=0), "探针训练计数异常")
        gl.require(set(payload["heads"]) == set(payload["optimizers"]) == set(NAMES), "探针对照不完整")
        for name in NAMES:
            lh.require_compatible(self.heads[name].state_dict(), payload["heads"][name])
            gl.require(all(bool(torch.isfinite(value).all()) for value in payload["heads"][name].values()), "探针权重非有限")
            optimizer, saved = self.optimizers[name], payload["optimizers"][name]
            lh.validate_optimizer(optimizer, saved, name)
            for current, group in zip(optimizer.param_groups, saved["param_groups"]):
                gl.require({k: v for k, v in current.items() if k != "params"} ==
                           {k: v for k, v in group.items() if k != "params"}, "探针优化器配置不同")
            states = saved["state"]
            indices = {index for group in saved["param_groups"] for index in group["params"]}
            gl.require((step == 0 and not states) or (step > 0 and set(states) == indices), "探针优化器状态缺失")
            for state in states.values():
                gl.require({"step", "exp_avg", "exp_avg_sq"} <= set(state) and float(state["step"]) == step and
                           all(bool(torch.isfinite(state[key]).all()) for key in ("step", "exp_avg", "exp_avg_sq")),
                           "探针优化器步数或数值异常")
        # Validate both streams before mutating weights or optimizers.
        np.random.RandomState(0).set_state(payload["sampler_state"])
        import random
        random.Random(0).setstate(payload["rng_state"]["python"])
        np.random.RandomState(0).set_state(payload["rng_state"]["numpy"])
        torch.Generator(device="cpu").set_state(payload["rng_state"]["torch"])
        if torch.device(self.device).type == "cuda":
            gl.require(payload["rng_state"]["cuda"] is not None, "探针CUDA RNG缺失")
            torch.Generator(device=self.device).set_state(payload["rng_state"]["cuda"])
        for name in NAMES:
            self.heads[name].load_state_dict(payload["heads"][name], strict=True)
            self.optimizers[name].load_state_dict(payload["optimizers"][name])
        self.step = step
        self.sampler.set_state(payload["sampler_state"])
        bc.restore_rng(payload["rng_state"], self.device)
        self.check_ownership()


@torch.no_grad()
def predict(trainer, rows, donors=None):
    cache = trainer.data.cache
    for head in trainer.heads.values():
        head.eval()
    names = (*NAMES, "zero_goal", "swapped_goal") if donors is not None else (*NAMES, "zero_goal")
    result = {"rows": np.asarray(rows, np.int64), "donors": np.asarray(donors, np.int64) if donors is not None else None,
              "nll": {name: [] for name in names}, "probabilities": {name: [] for name in names}}
    for start in range(0, len(rows), trainer.options["batch_size"]):
        chosen = rows[start:start + trainer.options["batch_size"]]
        features, goals, remaining, labels = cache.worker_batch(chosen, trainer.device)
        with gl.encoder_precision(trainer.device):
            logits = {name: trainer.logits(name, features, goals, remaining) for name in NAMES}
            logits["zero_goal"] = trainer.logits("goal", features, torch.zeros_like(goals), remaining)
            if donors is not None:
                alternate = donors[start:start + len(chosen)]
                safe = np.where(alternate >= 0, alternate, chosen)
                swapped = torch.as_tensor(cache.tables["goals"][cache.tables["worker_segment"][safe]], device=trainer.device)
                logits["swapped_goal"] = trainer.logits("goal", features, swapped, remaining)
            for name, value in logits.items():
                gl.require(bool(torch.isfinite(value).all()), "探针前向非有限")
                result["nll"][name].append(F.cross_entropy(value, labels, reduction="none").cpu().numpy())
                result["probabilities"][name].append(value.softmax(-1).cpu().numpy())
    for key in ("nll", "probabilities"):
        result[key] = {name: np.concatenate(parts) for name, parts in result[key].items()}
    return result


def summarize(data, prediction, positions=None):
    positions = np.arange(len(prediction["rows"])) if positions is None else np.asarray(positions, np.int64)
    gl.require(len(positions) > 0, "探针汇总集合为空")
    rows = prediction["rows"][positions]
    t = data.cache.tables
    episodes = data.episodes[t["worker_state"][rows]]
    budgets, labels = t["remaining"][rows], t["action_id"][rows]
    losses = {name: values[positions].astype(np.float64) for name, values in prediction["nll"].items()}
    probabilities = {name: values[positions] for name, values in prediction["probabilities"].items()}
    supported = prediction["donors"][positions] >= 0 if prediction["donors"] is not None else None

    def aggregate(selected):
        result = dict(rows=int(len(selected)))
        if supported is not None:
            result["supported_goal_swaps"] = int(supported[selected].sum())
        for name in losses:
            used = selected if name != "swapped_goal" else selected[supported[selected]]
            result[name + "_nll"] = float(losses[name][used].mean()) if len(used) else None
            result[name + "_accuracy"] = float((probabilities[name][used].argmax(-1) == labels[used]).mean()) if len(used) else None
        for name in ("no_goal", "zero_goal", "swapped_goal"):
            if name not in losses:
                continue
            used = selected if name != "swapped_goal" else selected[supported[selected]]
            result[name + "_minus_goal_nll"] = float((losses[name][used] - losses["goal"][used]).mean()) if len(used) else None
            result["goal_vs_" + name + "_l1"] = float(np.abs(probabilities["goal"][used] - probabilities[name][used]).sum(-1).mean()) if len(used) else None
            result["goal_vs_" + name + "_mode_change_fraction"] = float((probabilities["goal"][used].argmax(-1) != probabilities[name][used].argmax(-1)).mean()) if len(used) else None
        return result

    result = {"row_weighted": aggregate(np.arange(len(rows))),
              "per_episode": [{"episode": int(index), **aggregate(np.flatnonzero(episodes == index))} for index in np.unique(episodes)],
              "per_remaining": [{"remaining": int(h), **aggregate(np.flatnonzero(budgets == h))} for h in np.unique(budgets)]}
    for label, values in (("episode_macro", result["per_episode"]), ("remaining_macro", result["per_remaining"])):
        result[label] = {key: float(np.mean([row[key] for row in values if row[key] is not None]))
                         for key in result["row_weighted"] if key not in ("rows", "supported_goal_swaps") and any(row[key] is not None for row in values)}
    deltas = [row["no_goal_minus_goal_nll"] for row in result["per_episode"]]
    result["episode_comparison"] = dict(episodes=len(deltas), goal_better=int(np.sum(np.asarray(deltas) > 0)),
                                        no_goal_better=int(np.sum(np.asarray(deltas) < 0)),
                                        difference_descriptive=diag.describe(deltas))
    if np.any(budgets >= 13):
        result["remaining_13_plus"] = aggregate(np.flatnonzero(budgets >= 13))
    return result


def evaluate(trainer):
    rng_before = bc.capture_rng(trainer.device)
    queries = predict(trainer, trainer.data.rows, trainer.data.donors)
    train = trainer.data.cache.tables["worker_train"][trainer.data.rows]
    full = predict(trainer, trainer.data.cache.worker_rows["validation"])
    result = dict(step=trainer.step, input_identity=trainer.identity,
                  train_queries=summarize(trainer.data, queries, np.flatnonzero(train)),
                  validation_queries=summarize(trainer.data, queries, np.flatnonzero(~train)),
                  validation_full=summarize(trainer.data, full),
                  scope="action prediction on held-out episodes; no reachability, task success or seed-generalization claim")
    # Evaluation has no random layers or action sampling and may not advance RNG.
    import scripts.t01_checkpoint_check as t01
    gl.require(t01.same(rng_before, bc.capture_rng(trainer.device)), "探针评价推进训练RNG")
    return result, queries


def query_records(data, prediction):
    records = []
    t = data.cache.tables
    for position, row in enumerate(prediction["rows"]):
        state = t["worker_state"][row]
        donor = int(prediction["donors"][position])
        record = dict(row=int(row), episode=int(data.episodes[state]), frame=int(data.frames[state]),
                      split="train" if t["worker_train"][row] else "validation",
                      remaining=int(t["remaining"][row]), label_action=int(t["action_id"][row]), donor_row=donor,
                      donor_goal_cosine_distance=float(1 - np.clip(t["goals"][t["worker_segment"][row]] @
                          t["goals"][t["worker_segment"][donor]], -1, 1)) if donor >= 0 else None)
        for name, losses in prediction["nll"].items():
            supported = name != "swapped_goal" or donor >= 0
            record[name + "_nll"] = float(losses[position]) if supported else None
            record[name + "_action"] = int(prediction["probabilities"][name][position].argmax()) if supported else None
        records.append(record)
    return records
