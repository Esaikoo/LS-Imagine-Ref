"""Paired residual BC repair from whole, SHA-bound real reference histories.

The reference pool is development data from one already inspected world.
Only each history's actual next action and its own achieved endpoint train
the heads. Alternate goals are evaluation inputs, never action labels.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_library as gl
import goal_information_probe as probe
import goal_residual_worker as residual
import long_horizon as lh


FORMAT = "ls_imagine_reference_repair_worker_v1"
PROTOCOL = "real_reference_repeat0_train_repeat1_validation_mixture_v1"
SEMANTICS = "paired_real_endpoint_residual_repair_not_control_acceptance"
DEPENDENCIES = (*residual.DEPENDENCIES, "source_checkpoint", "source_verify_report",
                "calibration_manifest", "diagnosis_report", "repair_check_report")


def sampling_plan(batch_size):
    gl.require(type(batch_size) is int and batch_size >= 8 and batch_size % 8 == 0,
               "修复batch_size必须为8的正整数倍")
    return dict(protocol=PROTOCOL, original_rows_per_batch=3 * batch_size // 4,
                reference_rows_per_batch=batch_size // 4,
                reference_rows_per_target_per_batch=batch_size // 8,
                original_sampling="uniform real training expanded rows with replacement",
                reference_sampling="equal targets; uniform real repeat-0 rows with replacement",
                split="whole reference episode: repeat 0 train; repeat 1 validation",
                paired_batches_identical=True, action_reweighting=False,
                swapped_goal_training=False, loss="unmodified deployed mixed onehot action NLL")


def validate_reference(arrays, runs, horizon, feature_dim, goal_dim, dimension):
    required = {"features", "goals", "remaining", "labels", "train", "run", "target", "frame"}
    gl.require(set(arrays) == required and len(runs) == 4, "真实修复监督字段或整局数量不同")
    n = 4 * horizon
    gl.require(arrays["features"].shape == (n, feature_dim) and arrays["goals"].shape == (n, goal_dim) and
               arrays["features"].dtype == arrays["goals"].dtype == np.float32 and
               np.isfinite(arrays["features"]).all() and np.isfinite(arrays["goals"]).all() and
               arrays["train"].shape == (n,) and arrays["train"].dtype == np.bool_ and all(arrays[key].shape == (n,) and
               arrays[key].dtype == np.int64 for key in required - {"features", "goals", "train"}),
               "真实修复监督形状或类型不同")
    gl.require(np.all((arrays["labels"] >= 0) & (arrays["labels"] < dimension)), "真实动作标签越界")
    gl.require(sorted((run["repeat"], run["target"]) for run in runs) == [(0, 0), (0, 1), (1, 0), (1, 1)] and
               len({run["trajectory_path"] for run in runs}) == 4,
               "必须保留四条不同真实历史，不能拆帧或复制历史跨划分")
    for index, run in enumerate(runs):
        ix = np.flatnonzero(arrays["run"] == index)
        gl.require(len(ix) == horizon and np.array_equal(arrays["frame"][ix],
                   np.arange(run["start_frame"], run["endpoint_frame"])) and
                   np.array_equal(arrays["remaining"][ix], np.arange(horizon, 0, -1)) and
                   np.array_equal(arrays["labels"][ix], run["action_ids"]) and
                   np.all(arrays["train"][ix] == (run["repeat"] == 0)) and
                   np.all(arrays["target"][ix] == run["target"]) and
                   np.array_equal(arrays["goals"][ix], np.repeat(arrays["goals"][ix[:1]], horizon, axis=0)) and
                   run["endpoint_frame"] == run["start_frame"] + horizon,
                   "整局划分/真实下一动作/本局终点/remaining不一致")


class RepairData(residual.ResidualData):
    def __init__(self, cache, options, base_id, arrays, runs, lineage, fixed_goals):
        super().__init__(cache, options, base_id)
        validate_reference(arrays, runs, cache.bundle["horizon"], cache.states.shape[1],
                           cache.metadata["goal_dim"], cache.metadata["action_dim"])
        self.reference_arrays, self.runs, self.lineage = arrays, runs, copy.deepcopy(lineage)
        self.fixed_goals = np.asarray(fixed_goals, np.float32)
        gl.require(self.fixed_goals.shape == (2, cache.metadata["goal_dim"]) and
                   np.isfinite(self.fixed_goals).all(), "原固定目标形状或数值异常")
        self.reference_id = probe.array_digest(arrays)
        self.reference_rows = {name: np.flatnonzero(arrays["train"] == flag)
                               for name, flag in (("train", True), ("validation", False))}
        self.target_rows = {target: np.flatnonzero(arrays["train"] & (arrays["target"] == target))
                            for target in (0, 1)}
        gl.require(all(len(rows) == cache.bundle["horizon"] for rows in self.target_rows.values()),
                   "参考训练池缺少任一完整目标脚本")

    def identity(self):
        return dict(super().identity(), repair_protocol=PROTOCOL, reference_id=self.reference_id,
                    reference_runs=self.runs, lineage=self.lineage,
                    fixed_goals_id=probe.array_digest({"goals": self.fixed_goals}))

    def reference_batch(self, rows, device):
        a = self.reference_arrays
        return tuple(torch.as_tensor(a[key][rows].copy(), device=device)
                     for key in ("features", "goals", "remaining", "labels"))


class Trainer(residual.Trainer):
    """Warm start weights; fresh paired optimizers; exact repair-only resume."""
    def __init__(self, data, base, references, options, source, device):
        sampling_plan(options["batch_size"])
        super().__init__(data, base, references, options, device)
        for name in residual.NAMES:
            lh.require_compatible(self.model.heads[name].state_dict(), source["heads"][name])
            self.model.heads[name].load_state_dict(source["heads"][name], strict=True)
        self.counts = dict(source_remaining=np.zeros((2, self.identity["horizon"]), np.int64),
                           source_action=np.zeros((2, self.identity["action_dim"]), np.int64),
                           reference_target=np.zeros(2, np.int64))
        self.check_ownership()

    def sampling_plan(self):
        return sampling_plan(self.options["batch_size"])

    def draw(self):
        plan, a = self.sampling_plan(), self.data.reference_arrays
        original = self.worker_sampler.draw(self.sampler, plan["original_rows_per_batch"])
        reference = np.concatenate([self.sampler.choice(self.data.target_rows[target],
            plan["reference_rows_per_target_per_batch"], replace=True) for target in (0, 1)])
        gl.require(np.all(self.data.cache.tables["worker_train"][original]) and np.all(a["train"][reference]),
                   "修复采样混入留出整局")
        order = self.sampler.permutation(self.options["batch_size"])
        return original, reference, order

    def batch(self, selection):
        original, reference, order = selection
        left = self.data.cache.worker_batch(original, self.device)
        right = self.data.reference_batch(reference, self.device)
        permutation = torch.as_tensor(order, device=self.device)
        return tuple(torch.cat((a, b), dim=0)[permutation] for a, b in zip(left, right))

    def update(self):
        original, reference, order = selection = self.draw()
        features, goals, remaining, labels = self.batch(selection)
        result = dict(step=self.step + 1, shared_batch_id=probe.array_digest(
                      dict(original=original, reference=reference, order=order)))
        self.model.train(True)
        with gl.encoder_precision(self.device):
            base = self.model.base_preferences(features, goals, remaining)
            for name in residual.NAMES:
                opt = self.optimizers[name]
                opt.zero_grad(set_to_none=True)
                dist = self.model.distribution(base + self.model.correction(name, features, goals, remaining))
                loss = F.nll_loss(dist.logits, labels)
                gl.require(bool(torch.isfinite(loss)), "修复BC损失非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.heads[name].parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "修复梯度非有限")
                opt.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.model.heads[name].parameters()),
                           "修复权重非有限")
                result[name + "_nll"], result[name + "_grad_norm"] = float(loss.detach()), float(norm)
        a, t = self.data.reference_arrays, self.data.cache.tables
        for source, pool, rows, label in ((0, t, original, "action_id"), (1, a, reference, "labels")):
            self.counts["source_remaining"][source] += np.bincount(pool["remaining"][rows] - 1,
                                                                  minlength=self.identity["horizon"])
            self.counts["source_action"][source] += np.bincount(pool[label][rows], minlength=self.identity["action_dim"])
        self.counts["reference_target"] += np.bincount(a["target"][reference], minlength=2)
        self.step += 1
        self.check_ownership()
        return result

    def payload(self, verification_artifact=False):
        result = super().payload(verification_artifact)
        result.update(checkpoint_format=FORMAT, semantics=SEMANTICS, sampling_protocol=PROTOCOL,
                      repair_sampling=self.sampling_plan(), sampling_counts=copy.deepcopy(self.counts))
        return result

    def restore(self, payload, allow_verification=False):
        validate_payload(payload, self.identity, self.options, allow_verification)
        # Reuse the established tensor/optimizer/RNG checks, without saving an
        # artifact under the old protocol or restoring the old optimizer.
        translated = dict(payload, checkpoint_format=residual.FORMAT, semantics=residual.SEMANTICS,
                          sampling_protocol=residual.SAMPLING)
        super().restore(translated, allow_verification)
        self.counts = {name: np.asarray(value).copy() for name, value in payload["sampling_counts"].items()}


def validate_payload(payload, identity, options, allow_verification=False):
    probe.validate_options(options)
    gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("semantics") == SEMANTICS and
               payload.get("sampling_protocol") == PROTOCOL and
               payload.get("repair_sampling") == sampling_plan(options["batch_size"]), "修复格式或采样协议不同")
    translated = dict(payload, checkpoint_format=residual.FORMAT, semantics=residual.SEMANTICS,
                      sampling_protocol=residual.SAMPLING)
    residual.validate_payload(translated, identity, options, allow_verification)
    counts, step, plan = payload["sampling_counts"], payload["counters"]["step"], payload["repair_sampling"]
    shapes = dict(source_remaining=(2, identity["horizon"]), source_action=(2, identity["action_dim"]), reference_target=(2,))
    gl.require(set(counts) == set(shapes) and all(isinstance(counts[name], np.ndarray) and np.asarray(counts[name]).shape == shape and
               np.asarray(counts[name]).dtype == np.int64 and np.all(np.asarray(counts[name]) >= 0)
               for name, shape in shapes.items()), "修复采样计数形状或数值异常")
    expected = np.array([plan["original_rows_per_batch"], plan["reference_rows_per_batch"]]) * step
    gl.require(all(np.array_equal(np.asarray(counts[name]).sum(1), expected) for name in ("source_remaining", "source_action")) and
               np.array_equal(counts["reference_target"], np.full(2, plan["reference_rows_per_target_per_batch"] * step)),
               "修复采样计数与更新预算不同")


def save_checkpoint(trainer, path, check_hash, verification_artifact=False):
    return residual.save_checkpoint(trainer, path, check_hash, verification_artifact=verification_artifact,
                                    overwrite=Path(path).exists())


def read_checkpoint(path, trainer=None, check_hash=None):
    payload = bc._torch_load(path)
    gl.require(payload.get("checkpoint_format") == FORMAT, "需要新修复worker，不能恢复旧残差/探针文件")
    gl.require(set(payload.get("dependencies", {})) == set(DEPENDENCIES), "修复共享依赖声明缺失")
    refs = {}
    for name, record in payload["dependencies"].items():
        gl.require(isinstance(record.get("path"), str) and record["path"] and not Path(record["path"]).is_absolute(),
                   "修复依赖必须用相对路径")
        dependency = (Path(path).resolve().parent / record["path"]).resolve()
        gl.require(dependency.is_file() and bc.file_hash(dependency) == record["sha256"],
                   f"修复共享依赖SHA256不同：{name}")
        refs[name] = dependency
    if trainer is not None:
        gl.require(all(refs[name] == trainer.reference[name]["source_path"] and
                   payload["dependencies"][name]["sha256"] == trainer.reference[name]["sha256"]
                   for name in DEPENDENCIES),
                   "修复恢复共享依赖路径不同")
    if check_hash is not None:
        gl.require(payload.get("accepted_check_sha256") == check_hash, "修复checkpoint与check不同")
    checked = json.loads(refs["repair_check_report"].read_text(encoding="utf-8"))
    gl.require(checked.get("command") == "check" and checked.get("status") in ("passed", "passed_with_warnings") and
               checked.get("repair_format") == FORMAT and checked.get("repair_identity") == payload["input_identity"] and
               checked.get("options") == payload["options"] and
               payload.get("accepted_check_sha256") == payload["dependencies"]["repair_check_report"]["sha256"],
               "修复文件未绑定已通过的独立check")
    root = Path(__file__).resolve().parent
    for name, digest in checked["repair_code"].items():
        code_path = (root / name).resolve()
        gl.require(root in code_path.parents and code_path.is_file() and bc.file_hash(code_path) == digest,
                   f"修复推理代码与已验收check不同：{name}")
    lineage = payload["input_identity"]["lineage"]
    for name, key in (("source_checkpoint", "source_checkpoint_sha256"), ("source_verify_report", "source_verify_sha256"),
                      ("calibration_manifest", "calibration_manifest_sha256"), ("diagnosis_report", "diagnosis_report_sha256")):
        gl.require(payload["dependencies"][name]["sha256"] == lineage[key], "修复依赖与声明的真实来源不同")
    return payload, refs


def load_policy(path, cache, device):
    """Inference only; original residual loader intentionally rejects this format."""
    payload, refs = read_checkpoint(path)
    validate_payload(payload, payload["input_identity"], payload["options"])
    checked = json.loads(refs["repair_check_report"].read_text(encoding="utf-8"))
    gl.require(payload["counters"]["step"] == checked["planned_repair_updates"] and
               cache.cache_id == payload["input_identity"]["cache_id"] and
               cache.metadata["bundle_id"] == payload["input_identity"]["bundle_id"],
               "推理需要已训练的修复worker及同一缓存")
    gl.require(refs["frozen_bundle"] == (cache.directory / "frozen_bundle.pt").resolve(), "冻结bundle路径不同")
    base, _ = residual.load_base(refs["base_checkpoint"], refs["base_verify_report"], cache)
    base = {key: value for key, value in base.items() if key not in ("optimizers", "candidate")}
    base["frozen_bundle"] = cache.bundle
    rng = bc.capture_rng(device)
    try:
        model = residual.ResidualPolicy(cache, base, payload["input_identity"], payload["options"], device)
        for name in residual.NAMES:
            lh.require_compatible(model.heads[name].state_dict(), payload["heads"][name])
            model.heads[name].load_state_dict(payload["heads"][name], strict=True)
    finally:
        bc.restore_rng(rng, device)
    model.repair_updates = payload["counters"]["step"]
    model.worker_version = payload["input_identity"]["lineage"]["source_branch_updates"] + model.repair_updates
    model.model_id = gl.tensor_digest(model.heads.state_dict(), dict(format=FORMAT, input_identity=model.identity,
        options=payload["options"], worker_version=model.worker_version, repair_updates=model.repair_updates))
    return model.requires_grad_(False).eval()


@torch.no_grad()
def reference_predictions(trainer, goal_source):
    a = trainer.data.reference_arrays
    features, goals, remaining, labels = trainer.data.reference_batch(np.arange(len(a["labels"])), trainer.device)
    alternate = np.empty_like(a["goals"])
    for run_index, run in enumerate(trainer.data.runs):
        own = a["run"] == run_index
        other = next(i for i, item in enumerate(trainer.data.runs)
                     if item["repeat"] == run["repeat"] and item["target"] != run["target"])
        alternate[own] = a["goals"][np.flatnonzero(a["run"] == other)[0]]
    if goal_source == "fixed_old_targets":
        goals = torch.as_tensor(trainer.data.fixed_goals[a["target"]], device=trainer.device)
        alternate = trainer.data.fixed_goals[1 - a["target"]]
    swapped = torch.as_tensor(alternate, device=trainer.device)
    model = trainer.model.eval()
    with gl.encoder_precision(trainer.device):
        dists = {name: model.action_dist(features, goals, remaining, name)
                 for name in ("goal", "no_goal", "zero_goal", "base")}
        dists["swapped_goal"] = model.action_dist(features, swapped, remaining, "goal")
        gl.require(torch.equal(dists["no_goal"].probs, model.action_dist(features, swapped, remaining, "no_goal").probs),
                   "独立无目标分支依赖目标")
        losses = {name: F.nll_loss(dist.logits, labels, reduction="none").cpu().numpy() for name, dist in dists.items()}
        probs = {name: dist.probs.cpu().numpy() for name, dist in dists.items()}
    frames = []
    for index in range(len(labels)):
        label = int(a["labels"][index])
        row = dict(goal_source=goal_source, run=int(a["run"][index]), target=int(a["target"][index]),
                   split="train" if a["train"][index] else "validation", frame=int(a["frame"][index]),
                   next_action_frame=int(a["frame"][index]) + 1, remaining=int(a["remaining"][index]), label=label)
        for name in dists:
            p = probs[name][index]
            row.update({name + "_nll": float(losses[name][index]), name + "_probability": float(p[label]),
                        name + "_mode": int(p.argmax()), name + "_match": bool(p.argmax() == label)})
        row.update(no_goal_minus_goal_nll=row["no_goal_nll"] - row["goal_nll"],
                   zero_minus_goal_nll=row["zero_goal_nll"] - row["goal_nll"],
                   swapped_minus_goal_nll=row["swapped_goal_nll"] - row["goal_nll"],
                   goal_tv=float(np.abs(probs["goal"][index] - probs["swapped_goal"][index]).sum() / 2),
                   goal_mode_changed=bool(row["goal_mode"] != row["swapped_goal_mode"]))
        frames.append(row)
    return frames


def reference_summary(rows):
    def mean(group):
        return dict(rows=len(group), **{key: float(np.mean([row[key] for row in group])) for key in group[0]
            if key.endswith(("_nll", "_probability", "_match")) or key in ("goal_tv", "goal_mode_changed")})
    result = dict(all_rows=mean(rows), by_split={}, per_run={}, per_action={})
    for split in ("train", "validation"):
        group = [row for row in rows if row["split"] == split]
        result["by_split"][split] = dict(all_rows=mean(group), by_target={
            str(target): mean([row for row in group if row["target"] == target]) for target in (0, 1)},
            after_common_prefix=mean([row for row in group if row["frame"] >= 68]))
    for run in sorted({row["run"] for row in rows}):
        result["per_run"][str(run)] = mean([row for row in rows if row["run"] == run])
    for label in sorted({row["label"] for row in rows}):
        result["per_action"][str(label)] = mean([row for row in rows if row["label"] == label])
    return result


def evaluate(trainer):
    from scripts.t01_checkpoint_check import same

    rng, sampling = bc.capture_rng(trainer.device), trainer.sampler.get_state()
    original, _, _ = residual.evaluate(trainer)
    frames = [row for goal_source in ("own_real_endpoint", "fixed_old_targets")
              for row in reference_predictions(trainer, goal_source)]
    result = dict(step=trainer.step, original_cache=original, reference={
        source: reference_summary([row for row in frames if row["goal_source"] == source])
        for source in ("own_real_endpoint", "fixed_old_targets")}, behavior_accepted=False, t06_approved=False)
    gl.require(same(rng, bc.capture_rng(trainer.device)) and same(sampling, trainer.sampler.get_state()),
               "修复评价推进训练随机数")
    return result, frames


@torch.no_grad()
def coverage(data, device):
    """Descriptive train-only neighbours selected by current state and budget."""
    t, a, dimension, horizon = data.cache.tables, data.reference_arrays, data.identity()["action_dim"], data.identity()["horizon"]
    rows = data.cache.worker_rows["train"]
    unique_rows = rows[np.unique(t["worker_state"][rows], return_index=True)[1]]
    counts = np.zeros((horizon, dimension), np.int64)
    np.add.at(counts, (t["remaining"][rows] - 1, t["action_id"][rows]), 1)
    result = dict(original_expanded_rows=len(rows), original_unique_action_states=len(unique_rows),
        expanded_action_counts=np.bincount(t["action_id"][rows], minlength=dimension).tolist(),
        unique_action_state_counts=np.bincount(t["action_id"][unique_rows], minlength=dimension).tolist(),
        remaining_action_counts=counts.tolist(), reference={}, nearest_context=[],
        neighbours_are_same_physical_state=False, used_for_training_selection=False,
        neighbour_ranking="current frozen state cosine; same remaining; training episodes only; no future goal ranking")
    for split, selected in data.reference_rows.items():
        result["reference"][split] = dict(rows=len(selected), action_counts=np.bincount(a["labels"][selected],
            minlength=dimension).tolist(), target_counts=np.bincount(a["target"][selected], minlength=2).tolist())
    for budget in range(1, horizon + 1):
        pool = rows[t["remaining"][rows] == budget]
        pool = pool[np.unique(t["worker_state"][pool], return_index=True)[1]]
        queries = np.flatnonzero(a["remaining"] == budget)
        query = F.normalize(torch.as_tensor(a["features"][queries], device=device), dim=-1)
        best_rows, best_distances = np.empty(0, np.int64), np.empty((len(queries), 0), np.float32)
        for start in range(0, len(pool), 1024):
            chosen = pool[start:start + 1024]
            state = torch.as_tensor(data.cache.states[t["worker_state"][chosen]].copy(), device=device)
            distances = (1 - query @ F.normalize(state, dim=-1).T).clamp(0, 2).cpu().numpy()
            combined = np.concatenate((best_distances, distances), axis=1)
            # Each query retains its own row indices; no future frame is consulted.
            if best_rows.ndim == 2:
                combined_rows = np.concatenate((best_rows, np.repeat(chosen[None], len(queries), axis=0)), axis=1)
            else:
                combined_rows = np.repeat(chosen[None], len(queries), axis=0)
            orders = np.stack([np.lexsort((rr, dd))[:32] for rr, dd in zip(combined_rows, combined)])
            best_rows = np.take_along_axis(combined_rows, orders, axis=1)
            best_distances = np.take_along_axis(combined, orders, axis=1)
        gl.require(len(pool) > 0, "原训练池缺少remaining预算")
        for local, index in enumerate(queries):
            chosen, label = best_rows[local], int(a["labels"][index])
            result["nearest_context"].append(dict(reference_row=int(index), run=int(a["run"][index]),
                frame=int(a["frame"][index]), remaining=budget, label=label, neighbours=len(chosen),
                actual_action_support_count=int((t["action_id"][chosen] == label).sum()),
                action_counts=np.bincount(t["action_id"][chosen], minlength=dimension).tolist(),
                state_cosine_distance_min=float(best_distances[local].min()),
                state_cosine_distance_max=float(best_distances[local].max()),
                original_row_ids=chosen.tolist()))
    return result
