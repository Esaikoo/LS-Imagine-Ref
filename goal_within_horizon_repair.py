"""Fixed-budget paired factual repair on real frames76..79 and own frame80.

Whole manual training episodes only; retained continuations are evaluation
data. Neither hindsight endpoints nor factual actions certify optimality.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_continuation_repair as continuation
import goal_information_probe as probe
import goal_library as gl
import goal_reference_repair as reference
import goal_residual_worker as residual
import goal_within_horizon_supervision as supervision
import long_horizon as lh

FORMAT = "ls_imagine_within_horizon_repair_worker_v1"
PROTOCOL = "source350_original_reference_manual_frames76_79_endpoint80_mixture_v1"
SEMANTICS = "paired_within_horizon_factual_development_repair_not_behavior_acceptance"
PLANNED_UPDATES = 50
SOURCE_VERSION = 350
LEARNING_RATE = 5e-5
RETENTION_NLL_LIMIT = .02
DEPENDENCIES = (*residual.DEPENDENCIES, "source350_checkpoint", "source350_verify_report",
                "supervision_manifest", "feedback_manifest", "feedback_index", "within_horizon_repair_check_report")
CHECKS = ("real_within_horizon_supervision", "warm_start", "sampler_contract", "sample_holdout_guard",
          "sample_worker_guard", "sample_subset_guard", "training_split_guard", "own_endpoint_guard",
          "source350_roundtrip", "distribution_contract", "diagnostic_contracts", "initial_metrics", "no_updates",
          "source_inputs_unchanged")


def sampling_plan(batch_size):
    gl.require(type(batch_size) is int and batch_size == 256, "预算内修复batch_size必须固定256")
    return dict(protocol=PROTOCOL, original_rows_per_batch=192, reference_rows_per_batch=48,
        reference_rows_per_target_per_batch=24, feedback_rows_per_batch=16,
        original_sampling="uniform all accepted original real train expanded rows with replacement",
        reference_sampling="equal targets; uniform real repeat0 script rows with replacement",
        feedback_sampling="uniform all24 manual repeat0-2 factual rows with replacement",
        split="whole episodes; repeat0-2 train, repeat3-4 development_holdout",
        worker_facts_used_for_training=False, retained_continuations_used_for_training=False,
        action_reweighting=False, correction_oversampling=False, swapped_goal_training=False,
        paired_batches_identical=True, goal="each episode actual frame80 RGB/heatmap endpoint",
        loss="deployed mixed onehot action NLL", future_endpoint_is_not_deployment_input=True)


def validate_feedback(a, runs, identity):
    keys = {"features", "goals", "remaining", "labels", "train", "run", "target", "frame", "manual", "correction"}
    gl.require(set(a) == keys and len(runs) == 20 and
        a["features"].shape == (80, identity["feature_dim"]) and a["goals"].shape == (80, identity["goal_dim"]) and
        a["features"].dtype == a["goals"].dtype == np.float32 and
        np.isfinite(a["features"]).all() and np.isfinite(a["goals"]).all() and
        np.allclose(np.linalg.norm(a["goals"], axis=1), 1, atol=1e-5) and
        all(a[k].shape == (80,) and a[k].dtype == np.bool_ for k in ("train", "manual", "correction")) and
        all(a[k].shape == (80,) and a[k].dtype == np.int64 for k in keys - {"features", "goals", "train", "manual", "correction"}),
        "预算内事实监督字段/形状/类型异常")
    gl.require(sorted((r["repeat"], r["target"], r["condition"]) for r in runs) ==
        [(repeat, target, c) for repeat in range(5) for target in (0, 1) for c in ("visual_correct_hold", "worker_mode")] and
        len({r["trajectory_path"] for r in runs}) == 20 and
        np.array_equal(a["run"], np.repeat(np.arange(20), 4)), "必须保留完整20局和原顺序，不能拆局或筛选")
    for i, run in enumerate(runs):
        ix = np.arange(i * 4, i * 4 + 4)
        manual, train = run["condition"] == "visual_correct_hold", run["repeat"] < 3
        gl.require(run["trial_index"] == i and run["start_frame"] == 76 and run["endpoint_frame"] == 80 and
            run["episode_split"] == ("train" if train else "development_holdout") and
            run["goal_source"] == "this_episode_actual_frame80_rgb_heatmap" and run["expert_labels"] is False and
            np.array_equal(a["frame"][ix], np.arange(76, 80)) and
            np.array_equal(a["remaining"][ix], np.arange(4, 0, -1)) and
            np.array_equal(a["labels"][ix], run["action_ids"]) and
            np.all(a["train"][ix] == train) and np.all(a["manual"][ix] == manual) and
            np.all(a["target"][ix] == run["target"]) and
            np.array_equal(a["correction"][ix], run["correction_applied"]) and a["correction"][ix].sum() <= 1,
            "预算内整局划分/真实下一动作/remaining/纠偏来源不同")
        gl.require(probe.array_digest({"features": a["features"][ix], "goals": a["goals"][ix]}) ==
            run["supervision_content_id"] and
            np.array_equal(a["goals"][ix], np.repeat(a["goals"][ix[:1]], 4, axis=0)) and
            all(hashlib.sha256(a["features"][j].tobytes()).hexdigest() == h for j, h in zip(ix, run["state_sha256"])) and
            hashlib.sha256(a["goals"][ix[0]].tobytes()).hexdigest() == run["endpoint_goal_sha256"],
            "预算内状态或本局真实80帧终点被替换")
        if manual:
            gl.require(np.array_equal(a["labels"][ix], np.where(a["correction"][ix],
                2 if run["target"] == 0 else 3, 0)) and run["decision_sources"] ==
                ["visual_correction_once" if c else "visual_noop" for c in a["correction"][ix]],
                "只能使用实际视觉保持/纠偏，不能移植worker建议")
        else:
            gl.require(run["decision_sources"] == ["worker_mode"] * 4 and not a["correction"][ix].any(),
                       "worker事实不能伪装成手工训练")
    allowed = a["manual"] & a["train"]
    gl.require(allowed.sum() == 24 and (a["manual"] & ~a["train"]).sum() == 16 and (~a["manual"]).sum() == 40 and
        np.array_equal(np.bincount(a["labels"][allowed], minlength=12), [23, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) and
        a["correction"][allowed].sum() == 1 and not np.any(allowed & a["correction"] & (a["target"] == 1)),
        "当前固定协议只接受24/16/40及训练23noop/1上转/0下转，不能换采集或补样")


class RepairData(reference.RepairData):
    def __init__(self, old, options, base_id, arrays, runs, queries, plan, lineage):
        super().__init__(old.cache, options, base_id, old.reference_arrays, old.runs, old.lineage, old.fixed_goals)
        validate_feedback(arrays, runs, super().identity())
        self.feedback_arrays, self.feedback_runs = arrays, copy.deepcopy(runs)
        self.feedback_queries, self.feedback_plan = copy.deepcopy(queries), copy.deepcopy(plan)
        self.feedback_lineage = copy.deepcopy(lineage)
        self.feedback_id = probe.array_digest(arrays)
        self.manual_train_rows = np.flatnonzero(arrays["manual"] & arrays["train"])
        self.continuation_arrays = old.continuation_arrays
        self.continuation_runs = copy.deepcopy(old.continuation_runs)
        self.continuation_lineage = copy.deepcopy(old.continuation_lineage)

    def identity(self):
        return dict(super().identity(), within_horizon_protocol=PROTOCOL, within_horizon_id=self.feedback_id,
            within_horizon_runs=self.feedback_runs, within_horizon_lineage=self.feedback_lineage,
            retained_continuation_id=probe.array_digest(self.continuation_arrays),
            retained_continuation_runs=self.continuation_runs, retained_continuation_lineage=self.continuation_lineage,
            retained_continuations_role="evaluation_only_original80_83_endpoint84_no_label_transplant")

    def feedback_batch(self, rows, device):
        return tuple(torch.as_tensor(self.feedback_arrays[k][rows].copy(), device=device)
                     for k in ("features", "goals", "remaining", "labels"))

    def continuation_batch(self, rows, device):
        """Expose retained84 facts only to the existing evaluation routine."""
        return tuple(torch.as_tensor(self.continuation_arrays[k][rows].copy(), device=device)
                     for k in ("features", "goals", "remaining", "labels"))


class Trainer(residual.Trainer):
    def __init__(self, data, base, refs, options, source, device):
        sampling_plan(options["batch_size"])
        super().__init__(data, base, refs, options, device)
        for name in residual.NAMES:
            lh.require_compatible(self.model.heads[name].state_dict(), source["heads"][name])
            self.model.heads[name].load_state_dict(source["heads"][name], strict=True)
        self.counts = dict(source_remaining=np.zeros((3, 16), np.int64), source_action=np.zeros((3, 12), np.int64),
                           reference_target=np.zeros(2, np.int64), feedback_row=np.zeros(80, np.int64))
        self.check_ownership()

    def sampling_plan(self):
        return sampling_plan(self.options["batch_size"])

    def draw(self):
        a, plan = self.data.feedback_arrays, self.sampling_plan()
        gl.require(np.array_equal(self.data.manual_train_rows, np.flatnonzero(a["manual"] & a["train"])),
                   "手工采样池必须包含全部24训练事实，禁止留出/worker/子集或动作加权")
        original = self.worker_sampler.draw(self.sampler, plan["original_rows_per_batch"])
        refs = np.concatenate([self.sampler.choice(self.data.target_rows[t], 24, replace=True) for t in (0, 1)])
        added = self.sampler.choice(self.data.manual_train_rows, 16, replace=True)
        gl.require(np.all(self.data.cache.tables["worker_train"][original]) and
            np.all(self.data.reference_arrays["train"][refs]) and np.all(a["train"][added] & a["manual"][added]),
            "混合采样混入留出或worker事实")
        return original, refs, added, self.sampler.permutation(256)

    def batch(self, selection):
        original, refs, added, order = selection
        pools = (self.data.cache.worker_batch(original, self.device), self.data.reference_batch(refs, self.device),
                 self.data.feedback_batch(added, self.device))
        order = torch.as_tensor(order, device=self.device)
        return tuple(torch.cat(items, dim=0)[order] for items in zip(*pools))

    def update(self, verification_only=False):
        gl.require((self.step == 50 if verification_only else self.step < 50), "固定追加50步不能延长或重复验收更新")
        original, refs, added, order = selection = self.draw()
        x, g, r, labels = self.batch(selection)
        result = dict(step=self.step + 1, shared_batch_id=probe.array_digest(
            dict(original=original, reference=refs, feedback=added, order=order)), verification_only=verification_only)
        self.model.train(True)
        with gl.encoder_precision(self.device):
            base = self.model.base_preferences(x, g, r)
            for name in residual.NAMES:
                optimizer = self.optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss = F.nll_loss(self.model.distribution(base + self.model.correction(name, x, g, r)).logits, labels)
                gl.require(bool(torch.isfinite(loss)), "预算内事实NLL非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.heads[name].parameters(), self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "预算内修复梯度非有限")
                optimizer.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.model.heads[name].parameters()), "预算内权重非有限")
                result[name + "_nll"], result[name + "_grad_norm"] = float(loss.detach()), float(norm)
        for source, pool, rows, key in ((0, self.data.cache.tables, original, "action_id"),
            (1, self.data.reference_arrays, refs, "labels"), (2, self.data.feedback_arrays, added, "labels")):
            self.counts["source_remaining"][source] += np.bincount(pool["remaining"][rows] - 1, minlength=16)
            self.counts["source_action"][source] += np.bincount(pool[key][rows], minlength=12)
        self.counts["reference_target"] += np.bincount(self.data.reference_arrays["target"][refs], minlength=2)
        self.counts["feedback_row"] += np.bincount(added, minlength=80)
        self.step += 1
        self.check_ownership()
        return result

    def payload(self, verification_artifact=False):
        result = super().payload(verification_artifact)
        result.update(checkpoint_format=FORMAT, semantics=SEMANTICS, sampling_protocol=PROTOCOL,
            repair_sampling=self.sampling_plan(), sampling_counts=copy.deepcopy(self.counts),
            source_worker_version=350, planned_additional_updates=50)
        validate_payload(result, self.identity, self.options, verification_artifact)
        return result

    def restore(self, payload, allow_verification=False):
        validate_payload(payload, self.identity, self.options, allow_verification)
        super().restore(dict(payload, checkpoint_format=residual.FORMAT,
            semantics=residual.SEMANTICS, sampling_protocol=residual.SAMPLING), allow_verification)
        self.counts = {k: v.copy() for k, v in payload["sampling_counts"].items()}


def validate_payload(payload, identity, options, allow_verification=False):
    probe.validate_options(options)
    gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("semantics") == SEMANTICS and
        payload.get("sampling_protocol") == PROTOCOL and payload.get("source_worker_version") == 350 and
        payload.get("planned_additional_updates") == 50 and options["learning_rate"] == LEARNING_RATE and
        payload.get("repair_sampling") == sampling_plan(options["batch_size"]), "预算内修复格式/源350/预算或采样协议不同")
    residual.validate_payload(dict(payload, checkpoint_format=residual.FORMAT, semantics=residual.SEMANTICS,
        sampling_protocol=residual.SAMPLING), identity, options, allow_verification)
    step, counts = payload["counters"]["step"], payload["sampling_counts"]
    gl.require(step <= 50, "固定追加50步不能延长")
    shapes = dict(source_remaining=(3, 16), source_action=(3, 12), reference_target=(2,), feedback_row=(80,))
    gl.require(set(counts) == set(shapes) and all(isinstance(counts[k], np.ndarray) and counts[k].shape == shape and
        counts[k].dtype == np.int64 and np.all(counts[k] >= 0) for k, shape in shapes.items()), "混合采样计数形状异常")
    totals = np.array([192, 48, 16]) * step
    gl.require(all(np.array_equal(counts[k].sum(1), totals) for k in ("source_remaining", "source_action")) and
        np.array_equal(counts["reference_target"], np.full(2, 24 * step)) and counts["feedback_row"].sum() == totals[2],
        "混合采样计数与更新预算不同")
    allowed, labels = np.zeros(80, bool), np.empty(80, np.int64)
    for i, run in enumerate(identity["within_horizon_runs"]):
        allowed[i * 4:(i + 1) * 4] = run["repeat"] < 3 and run["condition"] == "visual_correct_hold"
        labels[i * 4:(i + 1) * 4] = run["action_ids"]
    gl.require(np.all(counts["feedback_row"][~allowed] == 0) and
        np.array_equal(counts["source_action"][2], np.bincount(labels, weights=counts["feedback_row"], minlength=12)) and
        np.array_equal(counts["source_remaining"][2, :4], counts["feedback_row"].reshape(20, 4).sum(0)[::-1]) and
        np.all(counts["source_remaining"][2, 4:] == 0), "预算内采样混入留出/worker或移植动作/预算")


def save_checkpoint(trainer, path, check_hash, verification_artifact=False):
    return residual.save_checkpoint(trainer, path, check_hash, verification_artifact=verification_artifact,
                                    overwrite=Path(path).exists())


def read_checkpoint(path, trainer=None, check_hash=None):
    path = Path(path).resolve()
    payload = bc._torch_load(path)
    gl.require(payload.get("checkpoint_format") == FORMAT and set(payload.get("dependencies", {})) == set(DEPENDENCIES),
               "需要独立预算内修复格式及全部共享依赖，不能加载旧版本")
    refs = {}
    for name, item in payload["dependencies"].items():
        gl.require(isinstance(item.get("path"), str) and item["path"] and not Path(item["path"]).is_absolute(),
                   "预算内共享依赖必须为相对路径")
        dependency = (path.parent / item["path"]).resolve()
        gl.require(dependency.is_file() and bc.file_hash(dependency) == item["sha256"], f"预算内共享依赖SHA不同：{name}")
        refs[name] = dependency
    if trainer is not None:
        gl.require(all(refs[k] == trainer.reference[k]["source_path"] and payload["dependencies"][k]["sha256"] ==
            trainer.reference[k]["sha256"] for k in DEPENDENCIES), "预算内恢复共享依赖身份不同")
    checked = json.loads(refs["within_horizon_repair_check_report"].read_text(encoding="utf-8"))
    gl.require(checked.get("command") == "check" and checked.get("status") in ("passed", "passed_with_warnings") and
        checked.get("repair_format") == FORMAT and checked.get("repair_identity") == payload["input_identity"] and
        checked.get("options") == payload["options"] and checked.get("planned_repair_updates") == 50 and
        checked.get("sampling_plan") == sampling_plan(256) and
        set(CHECKS) <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"} and
        payload.get("accepted_check_sha256") == payload["dependencies"]["within_horizon_repair_check_report"]["sha256"] and
        (check_hash is None or payload["accepted_check_sha256"] == check_hash), "预算内修复未绑定已通过独立check")
    root = Path(__file__).resolve().parent
    for name, digest in checked["repair_code"].items():
        code_path = (root / name).resolve()
        gl.require(root in code_path.parents and code_path.is_file() and bc.file_hash(code_path) == digest,
                   f"预算内修复代码与check不同：{name}")
    for name, digest in checked["check_artifacts"].items():
        artifact = (refs["within_horizon_repair_check_report"].parent / name).resolve()
        gl.require(artifact.parent == refs["within_horizon_repair_check_report"].parent and bc.file_hash(artifact) == digest,
                   f"预算内check产物改变：{name}")
    lineage = payload["input_identity"]["within_horizon_lineage"]
    for name in DEPENDENCIES[3:-1]:
        gl.require(payload["dependencies"][name]["sha256"] == lineage[name + "_sha256"], "预算内来源SHA身份不同")
    source, _ = continuation.read_checkpoint(refs["source350_checkpoint"])
    verified = json.loads(refs["source350_verify_report"].read_text(encoding="utf-8"))
    gl.require(refs["source350_checkpoint"].name == "latest.pt" and not source.get("verification_artifact") and
        source["counters"]["step"] == 50 and source["input_identity"] == lineage["source350_identity"] and
        verified.get("command") == "verify" and verified.get("status") in ("passed", "passed_with_warnings") and
        verified.get("worker_version") == 350 and verified.get("input_identity") == source["input_identity"] and
        verified.get("backend") == source["backend"] == payload["backend"] and
        verified.get("source_inputs_before") == verified.get("source_inputs_after") and
        Path(verified["arguments"]["checkpoint"]).resolve() == refs["source350_checkpoint"], "需要已独立验收的生产350来源")
    return payload, refs


def load_policy(path, cache, device):
    """Dedicated production inference; no optimizer, candidate or RSSM."""
    payload, refs = read_checkpoint(path)
    validate_payload(payload, payload["input_identity"], payload["options"])
    gl.require(Path(path).name == "latest.pt" and payload["counters"]["step"] == 50 and
        cache.cache_id == payload["input_identity"]["cache_id"] and cache.metadata["bundle_id"] == payload["input_identity"]["bundle_id"] and
        refs["frozen_bundle"] == (cache.directory / "frozen_bundle.pt").resolve() and payload["backend"] == bc.backend_info(device),
        "独立推理只接受同后端固定50步生产latest，不接受best/中途/验收副本")
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
    model.source_worker_version, model.within_horizon_updates, model.worker_version = 350, 50, 400
    model.model_id = gl.tensor_digest(model.heads.state_dict(), dict(format=FORMAT, input_identity=model.identity,
        options=payload["options"], worker_version=400, within_horizon_updates=50))
    return model.requires_grad_(False).eval()


@torch.no_grad()
def feedback_predictions(trainer):
    from scripts.t05_within_horizon_supervision import distribution
    rows, a = [], trainer.data.feedback_arrays
    for i, original in enumerate(trainer.data.feedback_queries):
        x, g, r, _ = trainer.data.feedback_batch(np.array([i]), trainer.device)
        fixed, swapped = trainer.data.fixed_goals[a["target"][i]], trainer.data.fixed_goals[1 - a["target"][i]]
        inputs = dict(actual_endpoint=(a["goals"][i], "goal"), fixed_goal=(fixed, "goal"), swapped_goal=(swapped, "goal"),
                      no_goal=(fixed, "no_goal"), zero_goal=(fixed, "zero_goal"), base=(fixed, "base"))
        probs, raw = {}, {}
        for c, (goal, mode) in inputs.items():
            probs[c], raw[c] = distribution(trainer.model.eval(), x, goal, int(r[0]), mode)
        for mode in ("no_goal", "zero_goal", "base"):
            for alternate in (a["goals"][i], swapped):
                probability, preferences = distribution(trainer.model, x, alternate, int(r[0]), mode)
                gl.require(np.array_equal(probability, probs[mode]) and np.array_equal(preferences, raw[mode]),
                           "评价中无目标/置零/底座受实际或交换目标影响")
        row = {k: copy.deepcopy(v) for k, v in original.items() if k not in ("metrics", "probabilities", "raw_preferences")}
        row.update(evaluated_worker_version=350 + trainer.step, repair_step=trainer.step,
            metrics=supervision.factual_metrics(probs, raw, int(a["labels"][i])),
            probabilities={k: v.tolist() for k, v in probs.items()}, raw_preferences={k: v.tolist() for k, v in raw.items()})
        rows.append(row)
    return rows


def evaluate(trainer):
    from scripts.t01_checkpoint_check import same
    rng, sampler = bc.capture_rng(trainer.device), trainer.sampler.get_state()
    result, retained = continuation.evaluate(trainer)
    rows = feedback_predictions(trainer)
    for source in ("own_real_endpoint", "fixed_old_targets"):
        for condition in ("visual_correct_hold", "worker_continuation"):
            result["continuation"][source][condition]["train_split_used_for_added_supervision"] = False
            result["continuation"][source][condition]["role"] = "retained84_evaluation_only"
    retained = [dict(row, pool="retained84_evaluation" if row["pool"] == "continuation" else row["pool"]) for row in retained]
    summary = supervision.summarize(rows, trainer.data.feedback_runs, trainer.data.feedback_plan)
    summary.update(format=FORMAT, comparison_protocol=PROTOCOL, source_diagnostic_format=supervision.FORMAT,
        source_factual_worker_version=350, evaluated_worker_version=350 + trainer.step,
        repair_step=trainer.step, optimizer_updates=trainer.counters()["optimizer_updates"],
        scope="fixed repair evaluation on unchanged source350 facts; no new rollout or expert labels")
    result.update(source_worker_version=350, additional_updates=trainer.step, worker_version=350 + trainer.step,
        within_horizon=summary, retained84_labels_used_for_training=False,
        dual_goal_correction_train_coverage=False, future_endpoint_is_not_deployment_input=True,
        behavior_accepted=False, t06_approved=False)
    gl.require(same(rng, bc.capture_rng(trainer.device)) and same(sampler, trainer.sampler.get_state()),
               "预算内评价推进训练RNG")
    return result, retained, rows


def retention_review(initial, current):
    comparisons = {}
    def compare(key, before, after):
        delta = float(after - before)
        gl.require(np.isfinite([before, after, delta]).all(), "保留NLL必须有限")
        comparisons[key] = dict(initial_nll=before, current_nll=after, increase=delta, within_limit=delta <= .02)
    for mode in residual.NAMES:
        compare("original_full26_" + mode, initial["original_cache"]["validation_full"]["row_weighted"][mode + "_nll"],
                current["original_cache"]["validation_full"]["row_weighted"][mode + "_nll"])
        for target in (0, 1):
            for source in ("own_real_endpoint", "fixed_old_targets"):
                compare(f"reference_goal{target}_{source}_{mode}",
                    initial["reference"][source]["by_split"]["validation"]["by_target"][str(target)][mode + "_nll"],
                    current["reference"][source]["by_split"]["validation"]["by_target"][str(target)][mode + "_nll"])
    return dict(compared_to="step0 production350", predeclared_max_absolute_nll_increase=.02,
        comparisons=comparisons, within_predeclared_retention_limit=all(v["within_limit"] for v in comparisons.values()),
        original84frame_continuation_role="separate descriptive retention evaluation; not new80frame labels",
        dual_goal_correction_train_coverage=False, autonomous_confirmation_required=True,
        automatic_budget_extension=False, checkpoint_selected_by_validation=False, behavior_accepted=False, t06_approved=False)
