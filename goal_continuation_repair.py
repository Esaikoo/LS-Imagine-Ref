"""Paired, fixed-budget repair from SHA-bound actual continuation endpoints.

Only manual train episodes enter the added pool. Worker continuations and
whole development holdouts remain evaluation data, including failed runs.
No counterfactual labels, remaining remapping, or corrective generalization
claim is made from the singleton observed correction.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_information_probe as probe
import goal_library as gl
import goal_reference_repair as reference
import goal_residual_worker as residual
import long_horizon as lh


FORMAT = "ls_imagine_continuation_repair_worker_v1"
PROTOCOL = "original_reference_manual_actual_endpoint_whole_episode_mixture_v1"
SEMANTICS = "paired_actual_continuation_repair_not_autonomous_acceptance"
PLANNED_UPDATES = 50
LEARNING_RATE = 5e-5
RETENTION_NLL_LIMIT = .02
DEPENDENCIES = (*residual.DEPENDENCIES, "source_checkpoint", "source_verify_report",
                "continuation_manifest", "continuation_index", "continuation_repair_check_report")


def sampling_plan(batch_size):
    gl.require(type(batch_size) is int and batch_size >= 32 and batch_size % 32 == 0,
               "续段修复batch_size必须为32的正整数倍")
    return dict(protocol=PROTOCOL, original_rows_per_batch=3 * batch_size // 4,
        reference_rows_per_batch=3 * batch_size // 16,
        reference_rows_per_target_per_batch=3 * batch_size // 32,
        continuation_rows_per_batch=batch_size // 16,
        original_sampling="uniform original real training expanded rows with replacement",
        reference_sampling="equal original targets; uniform real repeat-0 script rows with replacement",
        continuation_sampling="uniform all manual real repeat-0/1/2 continuation rows with replacement",
        split="whole episodes; manual repeat0-2 train, repeat3-4 development_holdout",
        worker_continuations_used_for_training=False, action_reweighting=False,
        correction_oversampling=False, swapped_goal_training=False, paired_batches_identical=True,
        goal="each history's actual achieved endpoint", loss="deployed mixed onehot action NLL")


def validate_continuations(a, runs, feature_dim, goal_dim, dimension):
    keys = {"features", "goals", "remaining", "labels", "train", "run", "target", "frame", "manual", "correction"}
    gl.require(set(a) == keys and len(runs) == 10 and
        a["features"].shape == (40, feature_dim) and a["goals"].shape == (40, goal_dim) and
        a["features"].dtype == a["goals"].dtype == np.float32 and
        np.isfinite(a["features"]).all() and np.isfinite(a["goals"]).all() and
        np.allclose(np.linalg.norm(a["goals"], axis=1), 1, atol=1e-5) and
        all(a[k].shape == (40,) and a[k].dtype == np.bool_ for k in ("train", "manual", "correction")) and
        all(a[k].shape == (40,) and a[k].dtype == np.int64 for k in keys - {"features", "goals", "train", "manual", "correction"}),
        "续段真实监督字段、形状或类型异常")
    gl.require(np.all((a["labels"] >= 0) & (a["labels"] < dimension)) and np.all(a["target"] == 1),
               "续段实际动作或请求目标异常")
    gl.require(sorted((r["repeat"], r["condition"]) for r in runs) ==
        [(repeat, condition) for repeat in range(5) for condition in ("visual_correct_hold", "worker_continuation")] and
        len({r["trajectory_path"] for r in runs}) == 10 and
        np.array_equal(a["run"], np.repeat(np.arange(10), 4)), "续段完整十局或顺序不同，不能复制/拆局")
    for index, run in enumerate(runs):
        ix = np.flatnonzero(a["run"] == index)
        manual = run["condition"] == "visual_correct_hold"
        corrections = np.asarray([source == "visual_turn_down_once" for source in run["decision_sources"]])
        gl.require(run["trial_index"] == index and run["start_frame"] == 80 and run["endpoint_frame"] == 84 and
            run["episode_split"] == ("train" if run["repeat"] < 3 else "development_holdout") and
            len(ix) == 4 and np.array_equal(a["frame"][ix], np.arange(80, 84)) and
            np.array_equal(a["remaining"][ix], np.arange(4, 0, -1)) and
            np.array_equal(a["labels"][ix], run["action_ids"]) and
            np.all(a["train"][ix] == (run["repeat"] < 3)) and np.all(a["manual"][ix] == manual) and
            np.array_equal(a["correction"][ix], corrections) and int(corrections.sum()) <= 1 and
            np.array_equal(a["goals"][ix], np.repeat(a["goals"][ix[:1]], 4, axis=0)),
            "续段整局划分/真实下一动作/remaining4-1/纠偏来源不一致")
        gl.require(probe.array_digest({"features": a["features"][ix], "goals": a["goals"][ix]}) ==
                   run["supervision_content_id"], "续段状态或本局真实终点被替换")
        if manual:
            gl.require(all(source in ("visual_noop_hold", "visual_turn_down_once") for source in run["decision_sources"]) and
                np.array_equal(a["labels"][ix], np.where(corrections, run["turn_down_action"], run["noop_action"])),
                "手工监督必须是实际下转/保持动作，不能移植worker建议")
        else:
            gl.require(run["decision_sources"] == ["worker_mode"] * 4 and not corrections.any(),
                       "worker事实续段不能伪装成手工纠偏")
    gl.require(np.count_nonzero(a["manual"] & a["train"]) == 12 and
               np.count_nonzero(a["manual"] & ~a["train"]) == 8, "手工监督必须保留12训练/8开发留出标签")


class RepairData(reference.RepairData):
    def __init__(self, cache, options, base_id, reference_arrays, reference_runs, source_lineage,
                 fixed_goals, continuation_arrays, continuation_runs, lineage):
        super().__init__(cache, options, base_id, reference_arrays, reference_runs, source_lineage, fixed_goals)
        validate_continuations(continuation_arrays, continuation_runs, cache.states.shape[1],
                               cache.metadata["goal_dim"], cache.metadata["action_dim"])
        self.continuation_arrays = continuation_arrays
        self.continuation_runs = copy.deepcopy(continuation_runs)
        self.continuation_lineage = copy.deepcopy(lineage)
        self.continuation_id = probe.array_digest(continuation_arrays)
        self.manual_train_rows = np.flatnonzero(continuation_arrays["manual"] & continuation_arrays["train"])

    def identity(self):
        return dict(super().identity(), continuation_protocol=PROTOCOL, continuation_id=self.continuation_id,
            continuation_runs=self.continuation_runs, continuation_lineage=self.continuation_lineage)

    def continuation_batch(self, rows, device):
        return tuple(torch.as_tensor(self.continuation_arrays[key][rows].copy(), device=device)
                     for key in ("features", "goals", "remaining", "labels"))


class Trainer(residual.Trainer):
    def __init__(self, data, base, references, options, source, device):
        sampling_plan(options["batch_size"])
        super().__init__(data, base, references, options, device)
        for name in residual.NAMES:
            lh.require_compatible(self.model.heads[name].state_dict(), source["heads"][name])
            self.model.heads[name].load_state_dict(source["heads"][name], strict=True)
        self.counts = dict(source_remaining=np.zeros((3, self.identity["horizon"]), np.int64),
            source_action=np.zeros((3, self.identity["action_dim"]), np.int64),
            reference_target=np.zeros(2, np.int64), continuation_row=np.zeros(40, np.int64))
        self.check_ownership()

    def sampling_plan(self):
        return sampling_plan(self.options["batch_size"])

    def draw(self):
        plan = self.sampling_plan()
        original = self.worker_sampler.draw(self.sampler, plan["original_rows_per_batch"])
        refs = np.concatenate([self.sampler.choice(self.data.target_rows[target],
            plan["reference_rows_per_target_per_batch"], replace=True) for target in (0, 1)])
        added = self.sampler.choice(self.data.manual_train_rows, plan["continuation_rows_per_batch"], replace=True)
        a = self.data.continuation_arrays
        gl.require(np.all(self.data.cache.tables["worker_train"][original]) and
            np.all(self.data.reference_arrays["train"][refs]) and np.all(a["train"][added] & a["manual"][added]),
            "混合采样混入开发留出或worker事实续段")
        return original, refs, added, self.sampler.permutation(self.options["batch_size"])

    def batch(self, selection):
        original, refs, added, order = selection
        pools = (self.data.cache.worker_batch(original, self.device), self.data.reference_batch(refs, self.device),
                 self.data.continuation_batch(added, self.device))
        permutation = torch.as_tensor(order, device=self.device)
        return tuple(torch.cat(items, dim=0)[permutation] for items in zip(*pools))

    def update(self):
        original, refs, added, order = selection = self.draw()
        features, goals, remaining, labels = self.batch(selection)
        result = dict(step=self.step + 1, shared_batch_id=probe.array_digest(
            dict(original=original, reference=refs, continuation=added, order=order)))
        self.model.train(True)
        with gl.encoder_precision(self.device):
            base = self.model.base_preferences(features, goals, remaining)
            for name in residual.NAMES:
                optimizer = self.optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                dist = self.model.distribution(base + self.model.correction(name, features, goals, remaining))
                loss = F.nll_loss(dist.logits, labels)
                gl.require(bool(torch.isfinite(loss)), "续段修复NLL非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.heads[name].parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "续段修复梯度非有限")
                optimizer.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.model.heads[name].parameters()), "续段修复权重非有限")
                result[name + "_nll"], result[name + "_grad_norm"] = float(loss.detach()), float(norm)
        for source, pool, rows, label in ((0, self.data.cache.tables, original, "action_id"),
            (1, self.data.reference_arrays, refs, "labels"), (2, self.data.continuation_arrays, added, "labels")):
            self.counts["source_remaining"][source] += np.bincount(pool["remaining"][rows] - 1, minlength=self.identity["horizon"])
            self.counts["source_action"][source] += np.bincount(pool[label][rows], minlength=self.identity["action_dim"])
        self.counts["reference_target"] += np.bincount(self.data.reference_arrays["target"][refs], minlength=2)
        self.counts["continuation_row"] += np.bincount(added, minlength=40)
        self.step += 1
        self.check_ownership()
        return result

    def payload(self, verification_artifact=False):
        result = super().payload(verification_artifact)
        result.update(checkpoint_format=FORMAT, semantics=SEMANTICS, sampling_protocol=PROTOCOL,
            repair_sampling=self.sampling_plan(), sampling_counts=copy.deepcopy(self.counts),
            planned_additional_updates=PLANNED_UPDATES, source_worker_version=300)
        return result

    def restore(self, payload, allow_verification=False):
        validate_payload(payload, self.identity, self.options, allow_verification)
        super().restore(dict(payload, checkpoint_format=residual.FORMAT,
            semantics=residual.SEMANTICS, sampling_protocol=residual.SAMPLING), allow_verification)
        self.counts = {name: value.copy() for name, value in payload["sampling_counts"].items()}


def validate_payload(payload, identity, options, allow_verification=False):
    probe.validate_options(options)
    gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("semantics") == SEMANTICS and
        payload.get("sampling_protocol") == PROTOCOL and payload.get("repair_sampling") == sampling_plan(options["batch_size"]) and
        payload.get("planned_additional_updates") == PLANNED_UPDATES and payload.get("source_worker_version") == 300 and
        options["batch_size"] == 256 and options["learning_rate"] == LEARNING_RATE,
        "续段修复格式/固定预算/采样协议不同")
    residual.validate_payload(dict(payload, checkpoint_format=residual.FORMAT, semantics=residual.SEMANTICS,
                              sampling_protocol=residual.SAMPLING), identity, options, allow_verification)
    step, counts, plan = payload["counters"]["step"], payload["sampling_counts"], payload["repair_sampling"]
    gl.require(step <= PLANNED_UPDATES, "固定追加预算不能延长")
    shapes = dict(source_remaining=(3, identity["horizon"]), source_action=(3, identity["action_dim"]),
                  reference_target=(2,), continuation_row=(40,))
    gl.require(set(counts) == set(shapes) and all(isinstance(counts[k], np.ndarray) and counts[k].shape == shape and
        counts[k].dtype == np.int64 and np.all(counts[k] >= 0) for k, shape in shapes.items()), "混合采样计数形状或数值异常")
    expected = np.array([plan[k] for k in ("original_rows_per_batch", "reference_rows_per_batch", "continuation_rows_per_batch")]) * step
    gl.require(all(np.array_equal(counts[k].sum(1), expected) for k in ("source_remaining", "source_action")) and
        np.array_equal(counts["reference_target"], np.full(2, plan["reference_rows_per_target_per_batch"] * step)) and
        counts["continuation_row"].sum() == expected[2], "混合采样计数与更新预算不同")
    allowed, labels = np.zeros(40, bool), np.empty(40, np.int64)
    for index, run in enumerate(identity["continuation_runs"]):
        allowed[index * 4:(index + 1) * 4] = run["repeat"] < 3 and run["condition"] == "visual_correct_hold"
        labels[index * 4:(index + 1) * 4] = run["action_ids"]
    gl.require(np.all(counts["continuation_row"][~allowed] == 0) and
        np.array_equal(counts["source_action"][2], np.bincount(labels, weights=counts["continuation_row"], minlength=identity["action_dim"])) and
        np.array_equal(counts["source_remaining"][2, :4], counts["continuation_row"].reshape(10, 4).sum(0)[::-1]) and
        np.all(counts["source_remaining"][2, 4:] == 0), "续段采样混入留出/worker或移植预算与动作")


def save_checkpoint(trainer, path, check_hash, verification_artifact=False):
    return residual.save_checkpoint(trainer, path, check_hash, verification_artifact=verification_artifact,
                                    overwrite=Path(path).exists())


def read_checkpoint(path, trainer=None, check_hash=None):
    payload = bc._torch_load(path)
    gl.require(payload.get("checkpoint_format") == FORMAT, "需要独立续段修复worker，不能加载旧修复/残差/探针")
    gl.require(set(payload.get("dependencies", {})) == set(DEPENDENCIES), "续段修复共享依赖缺失")
    refs = {}
    for name, record in payload["dependencies"].items():
        gl.require(isinstance(record.get("path"), str) and record["path"] and not Path(record["path"]).is_absolute(),
                   "续段修复依赖必须使用相对路径")
        dependency = (Path(path).resolve().parent / record["path"]).resolve()
        gl.require(dependency.is_file() and bc.file_hash(dependency) == record["sha256"], f"共享依赖SHA256不同：{name}")
        refs[name] = dependency
    if trainer is not None:
        gl.require(all(refs[k] == trainer.reference[k]["source_path"] and payload["dependencies"][k]["sha256"] ==
            trainer.reference[k]["sha256"] for k in DEPENDENCIES), "恢复共享依赖路径不同")
    if check_hash is not None:
        gl.require(payload.get("accepted_check_sha256") == check_hash, "续段修复checkpoint与独立check不同")
    checked = json.loads(refs["continuation_repair_check_report"].read_text(encoding="utf-8"))
    gl.require(checked.get("command") == "check" and checked.get("status") in ("passed", "passed_with_warnings") and
        checked.get("repair_format") == FORMAT and checked.get("repair_identity") == payload["input_identity"] and
        checked.get("options") == payload["options"] and checked.get("planned_repair_updates") == PLANNED_UPDATES and
        checked.get("sampling_plan") == sampling_plan(payload["options"]["batch_size"]) and
        {"real_continuation_supervision", "warm_start", "sampler_contract", "training_split_guard", "sample_holdout_guard",
         "sample_worker_guard", "distribution_contract", "diagnostic_contracts", "initial_metrics", "no_updates",
         "source_inputs_unchanged"} <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"} and
        payload.get("accepted_check_sha256") == payload["dependencies"]["continuation_repair_check_report"]["sha256"],
        "续段修复文件未绑定已通过的独立check")
    root = Path(__file__).resolve().parent
    for name, digest in checked["repair_code"].items():
        code_path = (root / name).resolve()
        gl.require(root in code_path.parents and code_path.is_file() and bc.file_hash(code_path) == digest,
                   f"续段修复代码与check不同：{name}")
    lineage = payload["input_identity"]["continuation_lineage"]
    for name in ("source_checkpoint", "source_verify_report", "continuation_manifest", "continuation_index"):
        gl.require(payload["dependencies"][name]["sha256"] == lineage[name + "_sha256"], "续段修复来源身份不同")
    return payload, refs


def load_policy(path, cache, device):
    """Strict inference only; no optimizer or RSSM is constructed."""
    payload, refs = read_checkpoint(path)
    validate_payload(payload, payload["input_identity"], payload["options"])
    gl.require(payload["counters"]["step"] == PLANNED_UPDATES and cache.cache_id == payload["input_identity"]["cache_id"] and
        cache.metadata["bundle_id"] == payload["input_identity"]["bundle_id"] and
        refs["frozen_bundle"] == (cache.directory / "frozen_bundle.pt").resolve(), "推理需要固定50步latest及同一冻结缓存")
    base, _ = residual.load_base(refs["base_checkpoint"], refs["base_verify_report"], cache)
    base = {k: v for k, v in base.items() if k not in ("optimizers", "candidate")}
    base["frozen_bundle"] = cache.bundle
    rng = bc.capture_rng(device)
    try:
        model = residual.ResidualPolicy(cache, base, payload["input_identity"], payload["options"], device)
        for name in residual.NAMES:
            lh.require_compatible(model.heads[name].state_dict(), payload["heads"][name])
            model.heads[name].load_state_dict(payload["heads"][name], strict=True)
    finally:
        bc.restore_rng(rng, device)
    model.continuation_updates = payload["counters"]["step"]
    model.worker_version = 300 + model.continuation_updates
    model.model_id = gl.tensor_digest(model.heads.state_dict(), dict(format=FORMAT, input_identity=model.identity,
        options=payload["options"], worker_version=model.worker_version, continuation_updates=model.continuation_updates))
    return model.requires_grad_(False).eval()


@torch.no_grad()
def continuation_predictions(trainer, goal_source):
    a, model = trainer.data.continuation_arrays, trainer.model.eval()
    features, goals, remaining, labels = trainer.data.continuation_batch(np.arange(40), trainer.device)
    if goal_source == "fixed_old_targets":
        goals = torch.as_tensor(trainer.data.fixed_goals[a["target"]], device=trainer.device)
    swapped = torch.as_tensor(trainer.data.fixed_goals[1 - a["target"]], device=trainer.device)
    with gl.encoder_precision(trainer.device):
        dists = {name: model.action_dist(features, goals, remaining, name) for name in ("goal", "no_goal", "zero_goal", "base")}
        dists["swapped_goal"] = model.action_dist(features, swapped, remaining, "goal")
        gl.require(torch.equal(dists["no_goal"].probs, model.action_dist(features, swapped, remaining, "no_goal").probs),
                   "续段独立无目标分支依赖目标")
        losses = {name: F.nll_loss(dist.logits, labels, reduction="none").cpu().numpy() for name, dist in dists.items()}
        probs = {name: dist.probs.cpu().numpy() for name, dist in dists.items()}
    rows = []
    for index in range(40):
        label, run = int(a["labels"][index]), trainer.data.continuation_runs[int(a["run"][index])]
        row = dict(goal_source=goal_source, trial_index=int(a["run"][index]), repeat=run["repeat"], condition=run["condition"],
            split=run["episode_split"], frame=int(a["frame"][index]), next_action_frame=int(a["frame"][index]) + 1,
            remaining=int(a["remaining"][index]), label=label, correction=bool(a["correction"][index]),
            decision_source=run["decision_sources"][index % 4], boundary_kind=run["boundary_kind"],
            alternate_goal_source="old_fixed_target0_evaluation_only", observed_action_is_expert_label=False)
        for name in dists:
            p = probs[name][index]
            row.update({name + "_nll": float(losses[name][index]), name + "_probability": float(p[label]),
                name + "_rank": int(np.count_nonzero(p > p[label]) + 1), name + "_mode": int(p.argmax()),
                name + "_match": bool(p.argmax() == label)})
        row.update(no_goal_minus_goal_nll=row["no_goal_nll"] - row["goal_nll"],
            zero_minus_goal_nll=row["zero_goal_nll"] - row["goal_nll"],
            swapped_minus_goal_nll=row["swapped_goal_nll"] - row["goal_nll"],
            goal_tv=float(np.abs(probs["goal"][index] - probs["swapped_goal"][index]).sum() / 2),
            goal_mode_changed=bool(row["goal_mode"] != row["swapped_goal_mode"]))
        rows.append(row)
    return rows


def mean_rows(rows):
    if not rows:
        return dict(available=False, rows=0, episodes=0)
    keys = [k for k in rows[0] if k.endswith(("_nll", "_probability", "_rank", "_match")) or k in ("goal_tv", "goal_mode_changed")]
    return dict(available=True, rows=len(rows), episodes=len({r["trial_index"] for r in rows}),
                **{k: float(np.mean([r[k] for r in rows])) for k in keys})


def continuation_summary(rows):
    result = {}
    for condition in ("visual_correct_hold", "worker_continuation"):
        condition_rows = [r for r in rows if r["condition"] == condition]
        groups = {}
        for split in ("train", "development_holdout"):
            selected = [r for r in condition_rows if r["split"] == split]
            groups[split] = dict(all_rows=mean_rows(selected),
                correction=mean_rows([r for r in selected if r["correction"]]),
                noop_hold=mean_rows([r for r in selected if r["decision_source"] == "visual_noop_hold"]),
                by_remaining={str(b): mean_rows([r for r in selected if r["remaining"] == b]) for b in range(1, 5)},
                by_action={str(label): mean_rows([r for r in selected if r["label"] == label]) for label in sorted({r["label"] for r in selected})})
        result[condition] = dict(by_split=groups, per_episode={str(index): mean_rows([r for r in condition_rows if r["trial_index"] == index])
            for index in sorted({r["trial_index"] for r in condition_rows})},
            train_split_used_for_added_supervision=condition == "visual_correct_hold")
    result["corrective_generalization_verified"] = False
    return result


def evaluate(trainer):
    from scripts.t01_checkpoint_check import same

    rng, sampling = bc.capture_rng(trainer.device), trainer.sampler.get_state()
    result, frames = reference.evaluate(trainer)
    added = [row for source in ("own_real_endpoint", "fixed_old_targets") for row in continuation_predictions(trainer, source)]
    result["continuation"] = {source: continuation_summary([r for r in added if r["goal_source"] == source])
                              for source in ("own_real_endpoint", "fixed_old_targets")}
    result.update(source_worker_version=300, additional_updates=trainer.step, worker_version=300 + trainer.step,
                  corrective_generalization_verified=False)
    gl.require(same(rng, bc.capture_rng(trainer.device)) and same(sampling, trainer.sampler.get_state()), "续段评价推进训练随机数")
    return result, [dict(pool="reference", **r) for r in frames] + [dict(pool="continuation", **r) for r in added]


def retention_review(initial, current):
    comparisons = {}
    def compare(key, before, after):
        delta = float(after - before)
        gl.require(np.isfinite([before, after, delta]).all(), "留出NLL必须有限")
        comparisons[key] = dict(initial_nll=before, current_nll=after, increase=delta,
                                within_limit=delta <= RETENTION_NLL_LIMIT)
    for mode in residual.NAMES:
        compare("original_full_26_episodes_" + mode,
            initial["original_cache"]["validation_full"]["row_weighted"][mode + "_nll"],
            current["original_cache"]["validation_full"]["row_weighted"][mode + "_nll"])
        for source in ("own_real_endpoint", "fixed_old_targets"):
            compare("reference_goal0_" + source + "_" + mode,
                initial["reference"][source]["by_split"]["validation"]["by_target"]["0"][mode + "_nll"],
                current["reference"][source]["by_split"]["validation"]["by_target"]["0"][mode + "_nll"])
    support = {}
    for source in ("own_real_endpoint", "fixed_old_targets"):
        before = initial["continuation"][source]["visual_correct_hold"]["by_split"]
        after = current["continuation"][source]["visual_correct_hold"]["by_split"]
        held = after["development_holdout"]["all_rows"]
        correction_before, correction_after = before["train"]["correction"], after["train"]["correction"]
        support[source] = dict(manual_holdout_goal_nll_change=held["goal_nll"] - before["development_holdout"]["all_rows"]["goal_nll"],
            manual_holdout_no_goal_gap=held["no_goal_minus_goal_nll"], manual_holdout_zero_gap=held["zero_minus_goal_nll"],
            manual_holdout_swap_gap=held["swapped_minus_goal_nll"], train_correction_rows=correction_after["rows"],
            train_correction_goal_nll_change=correction_after["goal_nll"] - correction_before["goal_nll"] if correction_after["available"] else None,
            holdout_correction_rows=after["development_holdout"]["correction"]["rows"])
    return dict(predeclared_max_absolute_nll_increase=RETENTION_NLL_LIMIT, comparisons=comparisons,
        within_predeclared_retention_limit=all(v["within_limit"] for v in comparisons.values()), support=support,
        corrective_generalization_verified=False, autonomous_confirmation_required=True, behavior_accepted=False, t06_approved=False)


def coverage(data, device):
    a, result = data.continuation_arrays, dict(original_reference=reference.coverage(data, device), continuation={})
    for condition, manual in (("visual_correct_hold", True), ("worker_continuation", False)):
        result["continuation"][condition] = {}
        for split, train in (("train", True), ("development_holdout", False)):
            ix = np.flatnonzero((a["manual"] == manual) & (a["train"] == train))
            correction = ix[a["correction"][ix]]
            result["continuation"][condition][split] = dict(rows=len(ix), episodes=len(np.unique(a["run"][ix])),
                action_counts=np.bincount(a["labels"][ix], minlength=data.identity()["action_dim"]).tolist(),
                remaining_counts=np.bincount(a["remaining"][ix], minlength=5)[1:].tolist(),
                correction_rows=len(correction), correction_episodes=len(np.unique(a["run"][correction])),
                added_training_pool=manual and train, unique_factual_row_definition="trial_index,state_frame")
    result.update(corrective_generalization_verified=False, action_reweighting=False,
                  resampled_labels_are_not_independent_evidence=True, coordinates_used_for_training=False)
    return result
