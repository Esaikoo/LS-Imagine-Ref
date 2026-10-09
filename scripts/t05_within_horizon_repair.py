"""Fixed50 offline within-horizon factual repair: check/train/verify only."""
import argparse
import copy
from datetime import datetime
import gc
import hashlib
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_continuation_repair as old_repair
import t05_within_horizon_supervision as source_check

CODE = ("goal_within_horizon_repair.py", "scripts/t05_within_horizon_repair.py")


class Report(baseline.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        saved = self.save()
        failed = not saved or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_WITHIN_HORIZON_REPAIR_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return {**old_repair.code_identity(), **source_check.collection.code_identity(),
            **{n: bc.file_hash(ROOT / n) for n in (*source_check.CODE, *CODE)}}


def retention_plan():
    return dict(compared_to="step0 production350", max_absolute_nll_increase=.02,
        original_full26_modes=["goal", "no_goal"], reference_targets=[0, 1],
        reference_sources=["own_real_endpoint", "fixed_old_targets"], reference_modes=["goal", "no_goal"],
        planned_numerical_comparisons=10, original84frame_continuations="separate descriptive retention evaluation only",
        automatic_budget_extension=False, checkpoint_selected_by_validation=False, t06_approved=False)


def confirmation_plan(source):
    import goal_random_control as pilot
    import goal_repaired_control as acceptance
    plan = source["feedback_manifest"]["design"]
    return dict(worker_version=400, source_worker_version=350, fixed_additional_updates=50,
        execution_policy="mode", randomization_seed=6, planned_trials=30,
        schedule=pilot.schedule(0, 5, 6), maximum_new_env_steps=2400, gate=copy.deepcopy(acceptance.GATE),
        target_content_id=plan["target_content_id"], scenario=copy.deepcopy(plan["scenario"]),
        visual_preprocessing=copy.deepcopy(plan["visual_preprocessing"]),
        environment_fingerprint=copy.deepcopy(plan["environment_fingerprint"]), action_names=plan["action_names"],
        warmup_steps=32, prefix_actions=plan["prefix_actions"], start_actions=plan["start_actions"],
        control_start_frame=64, horizon=16, actual_endpoint_frame=80, independent_fresh_resets=True,
        matched_start_comparison=False, no_retries_or_replacements=True, implementation_pending=True,
        checkpoint_binding="new production350+50 fixed latest and independent verify; no best",
        requires_predeclared_retention_pass=True, dual_goal_correction_train_coverage=False, t06_approved=False)


def historical_inputs(args):
    import goal_bc as bc
    import goal_continuation_control as acceptance
    import goal_library as gl
    import goal_within_horizon_supervision as supervision
    directory = baseline.project_path(args.supervision_dir)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "check")
    manifest = legacy.read_json(directory / "supervision_manifest.json")
    source, protected, files = source_check.historical_inputs(SimpleNamespace(
        eval_dir=record["arguments"]["eval_dir"], device=args.device))
    expected = dict(checked_episodes=20, query_rows=80, diagnostic_distribution_rows=480,
        probability_roundtrip_queries=320, measurement_roundtrip_frames=340, maximum_probability_error=0,
        optimizer_updates=0, new_env_steps=0, worker_control_actions=0, intervention_actions=0,
        labels_generated=False, expert_labels_generated=False, approved_for_training=False,
        behavior_accepted=False, t06_approved=False, worker_version=350)
    gl.require(all(record.get(k) == v for k, v in expected.items()) and
        record.get("source_inputs_before") == record.get("source_inputs_after") and record["arguments"]["device"] == args.device and
        manifest.get("format") == supervision.FORMAT and manifest.get("comparison_protocol") == supervision.PROTOCOL and
        manifest["supervision_id"] == record["supervision_id"] == gl.tensor_digest({}, {
            k: v for k, v in manifest.items() if k != "supervision_id"}) and
        manifest["source_manifest_sha256"] == bc.file_hash(source["feedback_dir"] / "evaluation_manifest.json") and
        manifest["input_identity"] == source["feedback_manifest"]["input_identity"] and
        manifest["source_evaluation_id"] == source["feedback_manifest"]["evaluation_id"] and
        manifest["source_design_id"] == source["feedback_manifest"]["design_id"] and
        manifest["read_only_code"] == record["read_only_code"] == {n: bc.file_hash(ROOT / n) for n in source_check.CODE} and
        acceptance.passed(record, ("historical_identity", "strict_frozen_worker350", "fixed_visual_targets",
            "supervision_contracts", "real_query_alignment", "recorded_forward_roundtrip", "recorded_measurement_roundtrip",
            "same_state_goal_comparison", "no_training", "source_inputs_unchanged")),
        "需要完整已通过80事实/480分布监督检查、冻结350及47个原代码SHA；不能改写旧FAIL")
    required = {"queries.json", "frame_metrics.csv", "trials.json", "diagnostics.json", "training_review.json", "output_probe.json"}
    gl.require(required <= set(manifest["artifacts"]), "监督检查产物缺失")
    for name, digest in manifest["artifacts"].items():
        path = (directory / name).resolve()
        gl.require(not Path(name).is_absolute() and path.parent == directory and bc.file_hash(path) == digest,
                   f"原监督检查产物SHA不同：{name}")
        files.append(path)
    queries, trials = legacy.read_json(directory / "queries.json"), legacy.read_json(directory / "trials.json")
    summary = supervision.summarize(queries, trials, source["feedback_manifest"]["design"])
    saved = legacy.read_json(directory / "diagnostics.json")
    gl.require(all(saved[k] == v for k, v in summary.items()) and
        saved["source_manual_branch_coverage"] == source["feedback_coverage"] and
        saved["original_numerical_gate"] == source["feedback_manifest"]["design"]["original_numerical_gate"] and
        legacy.read_json(directory / "training_review.json") == supervision.review(summary, trials,
            source["feedback_manifest"]["design"]["action_names"]), "监督事实汇总/覆盖或旧门槛不同")
    source.update(supervision_dir=directory, supervision_record=record, supervision_manifest=manifest,
                  supervision_queries=queries, supervision_trials=trials)
    files += [directory / "report.json", directory / "supervision_manifest.json", *[ROOT / n for n in CODE]]
    protected.append(directory)
    return source, list(dict.fromkeys(protected)), list(dict.fromkeys(p.resolve() for p in files))


def feedback_data(source, library, device):
    import numpy as np
    import torch
    import goal_control as control
    import goal_information_probe as probe
    import goal_library as gl
    import goal_within_horizon_supervision as supervision
    plan = source["feedback_manifest"]["design"]
    arrays = {k: [] for k in ("features", "goals", "remaining", "labels", "train", "run", "target", "frame", "manual", "correction")}
    runs = []
    for i, (ep, row) in enumerate(zip(source["feedback_index"]["episodes"], source["feedback_manifest"]["rows"])):
        directory = baseline.project_path(ep["artifact_dir"])
        a = control.read_trajectory(directory)
        trace, measured = (legacy.read_json(directory / n) for n in ("control_trace.json", "frame_metrics.json"))
        with np.load(directory / "actual_endpoint.npz", allow_pickle=False) as stored:
            endpoint = {k: stored[k].copy() for k in stored.files}
        supervision.validate_saved_trial(a, legacy.read_json(directory / "events.json"), trace, measured, endpoint, ep,
            legacy.read_json(directory / "metrics.json"), legacy.read_json(directory / "history_check.json"), row, plan, i)
        with torch.no_grad(), gl.encoder_precision(device):
            obs = {k: torch.as_tensor(a[k][80:81].copy(), device=device) for k in ("image", "heatmap")}
            encoded = library(obs)[0].cpu().numpy()
        own = np.asarray(endpoint["goal_feature"], np.float32)
        gl.require(np.allclose(encoded, own, atol=3e-5, rtol=3e-5) and
            np.array_equal(endpoint["image"], a["image"][80]) and np.array_equal(endpoint["heatmap"], a["heatmap"][80]),
            "必须使用本局真实80帧，不能替换意向终点或旧84帧标签")
        selected = source["supervision_queries"][i * 4:i * 4 + 4]
        for j, query in enumerate(selected):
            identity = supervision.query_identity(ep, row, trace, measured, j)
            supervision.validate_query(query, identity)
            frame = 76 + j
            gl.require(query["state_sha256"] == hashlib.sha256(a["features"][frame].tobytes()).hexdigest() and
                query["endpoint_goal_sha256"] == hashlib.sha256(own.tobytes()).hexdigest() and
                query["endpoint_artifact_sha256"] == ep["artifacts_sha256"]["actual_endpoint.npz"] and
                np.array_equal(a["action"][frame + 1], np.eye(12, dtype=np.float32)[query["actual_action"]]),
                "事实状态/真实incoming/本局80帧SHA或下一动作不同")
            arrays["features"].append(a["features"][frame].copy())
            arrays["goals"].append(own.copy())
            for k, v in dict(remaining=4-j, labels=query["actual_action"], train=ep["episode_split"] == "train",
                run=i, target=ep["requested_target"], frame=frame, manual=ep["condition"] == "visual_correct_hold",
                correction=query["correction_applied"]).items():
                arrays[k].append(v)
        runs.append(dict(trial_index=i, repeat=ep["repeat"], target=ep["requested_target"], condition=ep["condition"],
            episode_split=ep["episode_split"], start_frame=76, endpoint_frame=80,
            action_ids=[q["actual_action"] for q in selected], correction_applied=[q["correction_applied"] for q in selected],
            decision_sources=[q["decision_source"] for q in selected], state_sha256=[q["state_sha256"] for q in selected],
            endpoint_goal_sha256=selected[0]["endpoint_goal_sha256"], trajectory_path=str(directory / "trajectory.npz"),
            trajectory_sha256=ep["artifacts_sha256"]["trajectory.npz"], endpoint_sha256=ep["artifacts_sha256"]["actual_endpoint.npz"],
            supervision_content_id=probe.array_digest(dict(features=a["features"][76:80].copy(), goals=np.repeat(own[None], 4, axis=0))),
            goal_source="this_episode_actual_frame80_rgb_heatmap", expert_labels=False))
    arrays = {k: np.asarray(v, dtype=np.float32 if k in ("features", "goals") else
        np.bool_ if k in ("train", "manual", "correction") else np.int64) for k, v in arrays.items()}
    return arrays, runs


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_continuation_repair as continuation
    import goal_library as gl
    import goal_reference_repair as reference
    import goal_residual_worker as residual
    import goal_within_horizon_repair as repair
    context = source["context"]
    old_source = dict(context["source"])
    original, dependencies = reference.read_checkpoint(context["refs"]["source_checkpoint"])
    old_source.update(source_checkpoint_payload=original, source_checkpoint_dependencies=dependencies)
    class Audit:
        def __init__(self):
            self.data = {}
        def check(self, name, level, detail):
            gl.require(level != "FAIL", f"原已验收监督加载失败：{name}: {detail}")
    # Reconstruct the accepted original/reference/84-frame evaluation pools.
    # This loader creates no optimizer, RSSM, candidate or environment.
    old, _, base, _, _ = old_repair.load_inputs(SimpleNamespace(command="verify", device=args.device,
        check_dir=str(context["check_dir"])), Audit(), old_source)
    saved = context["payload"]
    gl.require(old.identity() == saved["input_identity"] and context["checkpoint"].name == "latest.pt" and
        saved["counters"]["step"] == 50 and not saved.get("verification_artifact") and
        context["verified"]["backend"] == bc.backend_info(args.device), "原完整监督/同后端生产350身份不同")
    options = dict(saved["options"], learning_rate=repair.LEARNING_RATE)
    policy = continuation.load_policy(context["checkpoint"], old.cache, args.device)
    gl.require(policy.worker_version == 350 and policy.model_id == source["feedback_manifest"]["input_identity"]["model_id"],
               "原350生产模型内容不同")
    library = gl.GoalLibrary.from_payload(old.cache.bundle["library"], args.device).requires_grad_(False).eval()
    encoder_hash = gl.tensor_digest(library.state_dict(), {})
    arrays, runs = feedback_data(source, library, args.device)
    with np.load(source["feedback_dir"] / "targets.npz", allow_pickle=False) as bank:
        gl.require(np.array_equal(bank["goals"], old.fixed_goals), "原两固定视觉目标不能改变")
    gl.require(gl.tensor_digest(library.state_dict(), {}) == encoder_hash and
        all(not p.requires_grad and p.grad is None for p in library.parameters()), "实际终点复核改变冻结目标编码器")
    policy.check_frozen(content=True)
    del policy, library
    paths = dict(source350_checkpoint=context["checkpoint"], source350_verify_report=context["verify_dir"] / "report.json",
        supervision_manifest=source["supervision_dir"] / "supervision_manifest.json",
        feedback_manifest=source["feedback_dir"] / "evaluation_manifest.json", feedback_index=source["feedback_dir"] / "endpoint_data.json")
    lineage = dict(**{k + "_sha256": bc.file_hash(p) for k, p in paths.items()}, source350_identity=saved["input_identity"],
        source_supervision_id=source["supervision_manifest"]["supervision_id"], source350_model_id=source["supervision_record"]["model_id"],
        optimizer_initialization="fresh Adam step0; source350 head weights only", dual_goal_correction_train_coverage=False)
    data = repair.RepairData(old, options, residual.base_identity(base), arrays, runs,
        source["supervision_queries"], source["feedback_manifest"]["design"], lineage)
    _, refs = residual.load_base(context["refs"]["base_checkpoint"], context["refs"]["base_verify_report"], old.cache)
    refs.update({k: dict(source_path=p.resolve(), sha256=bc.file_hash(p)) for k, p in paths.items()})
    check_path = report.directory / "report.json" if args.command == "check" else baseline.project_path(args.check_dir) / "report.json"
    refs["within_horizon_repair_check_report"] = dict(source_path=check_path, sha256=None)
    future, retention = confirmation_plan(source), retention_plan()
    if args.command != "check":
        checked = legacy.read_json(check_path)
        legacy.accepted(checked, "check")
        gl.require(checked["repair_format"] == repair.FORMAT and checked["repair_code"] == code_identity() and
            checked["repair_identity"] == data.identity() and checked["options"] == options and
            checked["backend"] == bc.backend_info(args.device) and checked["planned_repair_updates"] == 50 and
            checked["sampling_plan"] == repair.sampling_plan(256) and checked["retention_plan"] == retention and
            checked["confirmation_plan"] == future and checked["source_inputs_before"] == checked["source_inputs_after"] and
            set(repair.CHECKS) <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"},
            "新修复check身份/代码/后端/固定预算/保留或确认计划不同")
        for name, digest in checked["check_artifacts"].items():
            path = (check_path.parent / name).resolve()
            gl.require(path.parent == check_path.parent and bc.file_hash(path) == digest, f"新check真实监督或指标改变：{name}")
        expected = {**{"reference__" + k: v for k, v in data.reference_arrays.items()},
                    **{"feedback__" + k: v for k, v in arrays.items()},
                    **{"retained84__" + k: v for k, v in data.continuation_arrays.items()}}
        with np.load(check_path.parent / "supervision.npz", allow_pickle=False) as stored:
            gl.require(set(stored.files) == set(expected) and all(np.array_equal(stored[k], v) for k, v in expected.items()),
                       "新check监督不能替换或移植旧84帧动作")
        refs["within_horizon_repair_check_report"]["sha256"] = bc.file_hash(check_path)
    base = {k: v for k, v in base.items() if k not in ("optimizers", "candidate")}
    base["frozen_bundle"] = old.cache.bundle
    report.data.update(repair_format=repair.FORMAT, architecture=residual.ARCHITECTURE, semantics=repair.SEMANTICS,
        repair_code=code_identity(), repair_identity=data.identity(),
        input_identity=data.identity(), options=options, backend=bc.backend_info(args.device), planned_repair_updates=50,
        source_worker_version=350, source_supervision_id=source["supervision_manifest"]["supervision_id"],
        sampling_plan=repair.sampling_plan(256), retention_plan=retention, confirmation_plan=future,
        original_numerical_gate=source["feedback_manifest"]["design"]["original_numerical_gate"],
        factual_pools=dict(manual_train=24, manual_development_holdout=16, worker_facts_only=40),
        dual_goal_correction_train_coverage=False, retained84_labels_used_for_training=False)
    report.check("real_within_horizon_supervision", "PASS", "完整20局/80事实状态，本局80帧目标/真实incoming/remaining4–1；只训24手工，开发16/worker40及旧84帧40只评价；原47代码不改")
    return data, options, base, refs, saved


def make_trainer(inputs, device):
    import numpy as np
    import torch
    import goal_within_horizon_repair as repair
    data, options, base, refs, saved = inputs
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return repair.Trainer(data, base, refs, options, saved, device)


def evaluation_outputs(trainer, report, initial=None):
    import goal_within_horizon_repair as repair
    metrics, retained, rows = repair.evaluate(trainer)
    if initial is not None:
        metrics["retention_review"] = repair.retention_review(initial, metrics)
        baseline.write_json(report.directory / "retention.json", metrics["retention_review"])
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    baseline.write_json(report.directory / "queries.json", rows)
    baseline.write_json(report.directory / "retained_frame_metrics.json", retained)
    flat = [dict({k: v for k, v in row.items() if k not in ("metrics", "probabilities", "raw_preferences")}, **row["metrics"]) for row in rows]
    baseline.write_csv(report.directory / "frame_metrics.csv", flat)
    hold = metrics["within_horizon"]["by_pool"]["manual_development_holdout"]
    print(f"[WITHIN REPAIR] step={trainer.step}/50 dev_actual_nll={hold['metrics']['actual_endpoint_factual_nll']:.6f} "
          f"dev_fixed_nll={hold['metrics']['fixed_goal_factual_nll']:.6f} dev_actual_mode={hold['mode_matches']['actual_endpoint']}/16 "
          f"source350+{trainer.step}", flush=True)
    return metrics, rows


def check(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_within_horizon_repair as repair
    import goal_within_horizon_supervision as supervision
    trainer = make_trainer(inputs, args.device)
    data, _, _, _, source = inputs
    gl.require(all(t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        not trainer.optimizers[k].state for k in ("goal", "no_goal")), "只能继承350权重，不能恢复旧优化器")
    report.check("warm_start", "PASS", "生产350双头逐值继承，新Adam步0；固定50步/5e-5，无RSSM/候选模型")
    rng = trainer.sampler.get_state()
    first = trainer.draw()
    trainer.sampler.set_state(rng)
    gl.require(t01.same(first, trainer.draw()), "混合采样RNG恢复后不同")
    trainer.sampler.set_state(rng)
    report.check("sampler_contract", "PASS", "每批192原训练+48原参考（每目标24）+16全部手工训练；双头同批，均匀有放回，无加权/纠偏过采样")
    t02.rejection(report, "sampling_budget_guard", lambda: repair.sampling_plan(255), "固定256")
    a, actual_pool = data.feedback_arrays, data.manual_train_rows.copy()
    try:
        for name, bad in (("sample_holdout_guard", np.flatnonzero(a["manual"] & ~a["train"])),
            ("sample_worker_guard", np.flatnonzero(~a["manual"] & a["train"])), ("sample_subset_guard", actual_pool[:-1])):
            data.manual_train_rows = bad
            t02.rejection(report, name, trainer.draw, "全部24")
    finally:
        data.manual_train_rows = actual_pool
        trainer.sampler.set_state(rng)
    def validate(changed):
        return repair.validate_feedback(changed, data.feedback_runs, data.identity())
    for name, key, value, message in (("training_split_guard", "train", not bool(a["train"][0]), "整局划分"),
        ("remaining_guard", "remaining", 1, "remaining"), ("action_guard", "labels", (int(a["labels"][0])+1)%12, "真实下一动作"),
        ("manual_source_guard", "manual", not bool(a["manual"][0]), "纠偏来源")):
        bad = {k: v.copy() for k, v in a.items()}
        bad[key][0] = value
        t02.rejection(report, name, lambda bad=bad: validate(bad), message)
    for name, key in (("own_endpoint_guard", "goals"), ("real_state_guard", "features")):
        bad = {k: v.copy() for k, v in a.items()}
        bad[key][:4] = -bad[key][:4]
        t02.rejection(report, name, lambda bad=bad: validate(bad), "真实80帧终点")
    payload = trainer.payload()
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "protocol_guard", lambda: trainer.restore(dict(payload, sampling_protocol="wrong")), "采样协议")
    x, g, r, _ = data.feedback_batch(np.arange(80), args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        raw = trainer.model.base_preferences(x, g, r) + trainer.model.correction("goal", x, g, r)
        gl.require(torch.equal(trainer.model.distribution(raw).probs, trainer.model.action_dist(x, g, r).probs), "部署unimix不同")
        for mode in ("no_goal", "zero_goal", "base"):
            gl.require(torch.equal(trainer.model.action_dist(x, g, r, mode).probs,
                trainer.model.action_dist(x, torch.flip(g, [0]), r, mode).probs), "无目标/置零/底座依赖目标")
    report.check("distribution_contract", "PASS", "raw偏好相加及原一次unimix不变；no_goal/zero/base严格目标不变")
    initial, rows = evaluation_outputs(trainer, report)
    maximum_error = 0.
    for actual, original in zip(rows, data.feedback_queries):
        for c in supervision.CONDITIONS:
            error = float(np.max(np.abs(np.asarray(actual["probabilities"][c]) - original["probabilities"][c])))
            maximum_error = max(maximum_error, error)
            gl.require(np.allclose(actual["probabilities"][c], original["probabilities"][c], atol=3e-5, rtol=3e-5) and
                np.allclose(actual["raw_preferences"][c], original["raw_preferences"][c], atol=3e-5, rtol=3e-5) and
                actual["metrics"][c + "_mode"] == original["metrics"][c + "_mode"], "step0不能重现原350六条件概率/模式")
    report.data.update(source350_roundtrip_queries=80, source350_distribution_roundtrip_rows=480, maximum_probability_error=maximum_error)
    report.check("source350_roundtrip", "PASS", f"未更新双头重现原80状态/480分布与mode，最大误差={maximum_error:.6g}；实际/固定目标分开")
    summary = initial["within_horizon"]
    gl.require(summary["missing_manual_branches"] == ["target1_train_correction"] and
        summary["by_target_split_branch"]["target1_train_correction"]["metrics"] is None,
        "目标1训练纠偏缺失必须保留null，不重新划分或填零")
    baseline.write_json(report.directory / "coverage.json", dict(by_pool={p: dict(rows=v["rows"], independent_episodes=v["independent_episodes"],
        action_counts=v["action_counts"]) for p, v in summary["by_pool"].items()},
        source_branch_coverage=summary["by_target_split_branch"], missing_manual_branches=summary["missing_manual_branches"],
        repeated_rows_are_not_independent_evidence=True, expert_labels_generated=False, t06_approved=False))
    report.check("diagnostic_contracts", "PASS", "两目标/整局划分/保持与纠偏分别报告；旧84帧只评价；空组null，偏航两局不筛除")
    report.check("correction_coverage", "WARN", "训练23noop/1上转/0下转；目标1预算内训练纠偏缺失，不能声称双目标纠偏泛化")
    np.savez_compressed(report.directory / "supervision.npz", **{"reference__"+k:v for k,v in data.reference_arrays.items()},
        **{"feedback__"+k:v for k,v in a.items()}, **{"retained84__"+k:v for k,v in data.continuation_arrays.items()})
    for name, content in (("initial_metrics.json", initial), ("retention_plan.json", report.data["retention_plan"]),
                           ("confirmation_plan.json", report.data["confirmation_plan"])):
        baseline.write_json(report.directory / name, content)
    names = ("supervision.npz", "coverage.json", "initial_metrics.json", "retention_plan.json", "confirmation_plan.json", "queries.json")
    report.data["check_artifacts"] = {n: bc.file_hash(report.directory / n) for n in names}
    report.check("initial_metrics", "PASS", "step0原26局/两目标两来源双分支10项保留限度0.02已声明；80帧事实与旧84帧保持/纠偏另报，不选best")
    trainer.model.check_frozen(content=True)
    gl.require(trainer.step == 0 and all(not trainer.optimizers[k].state and
        t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        all(p.grad is None for p in trainer.model.heads[k].parameters()) for k in ("goal", "no_goal")), "check不能更新权重/优化器/梯度")
    report.data.update(counters=trainer.counters(), optimizer_updates=0, worker_version=350)
    report.check("no_updates", "PASS", "check仅初始化/采样/前向，双头与新优化器步0；不启动训练或环境")


def train(args, report, inputs):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_within_horizon_repair as repair
    trainer = make_trainer(inputs, args.device)
    digest = trainer.reference["within_horizon_repair_check_report"]["sha256"]
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    baseline.write_json(report.directory / "initial_metrics.json", initial)
    current, _ = evaluation_outputs(trainer, report, initial)
    gl.require(t01.same({k:v for k,v in current.items() if k != "retention_review"}, initial), "训练step0与独立check不同")
    estimate = 3 * bc.tensor_storage_bytes(trainer.model.heads.state_dict()) + 1024**2
    bc.require_disk_space(report.directory, 3 * estimate)
    report.check("storage_preflight", "PASS", "双头/优化器快照空间已预检；原模型/缓存/轨迹只共享引用")
    report.check("paired_training", "PASS", "350权重warm start，双头同批192/48/16真实NLL；固定追加50步，开发/worker/旧84帧不训练")
    while trainer.step < 50:
        report.require_writable()
        values = trainer.update()
        report.data.update(counters=trainer.counters(), optimizer_updates=trainer.counters()["optimizer_updates"],
                           worker_version=350+trainer.step, additional_updates=trainer.step)
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % 10 == 0:
            print(f"[TRAIN WITHIN REPAIR] step={trainer.step}/50 goal_nll={values['goal_nll']:.4f} no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
            current, _ = evaluation_outputs(trainer, report, initial)
            repair.save_checkpoint(trainer, report.directory / "latest.pt", digest)
            counts, correction = trainer.counts["feedback_row"], trainer.data.feedback_arrays["correction"]
            baseline.write_json(report.directory / "sampling.json", dict(plan=trainer.sampling_plan(), end_step=trainer.step,
                cumulative_counts={k:v.tolist() for k,v in trainer.counts.items()}, sampled_manual_labels=int(counts.sum()),
                sampled_correction_labels=int(counts[correction].sum()), unique_manual_rows_seen=int(np.count_nonzero(counts)),
                unique_correction_rows_seen=int(np.count_nonzero(counts[correction])), labels_per_branch=trainer.counters()["labels_seen_per_branch"],
                resampled_labels_are_not_independent_evidence=True, retained84_labels_used_for_training=False))
            report.save()
    trainer.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle改变")
    report.data.update(final_metrics=current, worker_version=400, additional_updates=50,
                       checkpoint_sha256=bc.file_hash(report.directory / "latest.pt"))
    baseline.write_json(report.directory / "diagnostics.json", dict(initial=initial, final=current, worker_version=400,
        source_worker_version=350, additional_updates=50, fixed_budget_latest_primary=True, checkpoint_selected_by_validation=False,
        dual_goal_correction_train_coverage=False, behavior_accepted=False, t06_approved=False,
        retention_plan=report.data["retention_plan"], confirmation_plan=report.data["confirmation_plan"]))
    within = current["retention_review"]["within_predeclared_retention_limit"]
    report.check("retention_review", "PASS" if within else "WARN", "相对350 step0原26局及两个目标参考双分支10项NLL增加≤0.02；超限保留，不延预算/选best")
    report.check("training", "PASS", "双头各固定50次更新，源350+50=400；冻结底座/WM/目标更新0、新环境步0")
    report.check("frozen_dependencies", "PASS", "独立格式latest/新优化器/三池计数/RNG及来源SHA共享；无best选择或旧文件回写")


def verify(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_repair as continuation
    import goal_within_horizon_repair as repair
    trainer = make_trainer(inputs, args.device)
    path = baseline.project_path(args.checkpoint)
    digest = trainer.reference["within_horizon_repair_check_report"]["sha256"]
    payload, _ = repair.read_checkpoint(path, trainer, digest)
    trainer.restore(payload)
    trained = legacy.read_json(path.parent / "report.json")
    legacy.accepted(trained, "train")
    gl.require(path.name == "latest.pt" and trainer.step == 50 and trained.get("repair_identity") == trainer.identity and
        trained.get("counters") == payload["counters"] and trained.get("worker_version") == 400 and
        trained.get("source_worker_version") == 350 and trained.get("additional_updates") == 50 and
        trained.get("repair_code") == report.data["repair_code"] and trained.get("options") == trainer.options and
        trained.get("backend") == bc.backend_info(args.device) and
        trained.get("retention_plan") == report.data["retention_plan"] and
        trained.get("confirmation_plan") == report.data["confirmation_plan"] and
        trained.get("source_inputs_before") == trained.get("source_inputs_after") and
        trained.get("new_env_steps") == 0 and trained.get("expert_labels_generated") is False and
        {"paired_training", "training", "frozen_dependencies", "source_inputs_unchanged"} <=
            {r["name"] for r in trained["checks"] if r["level"] == "PASS"} and
        trained.get("checkpoint_sha256") == bc.file_hash(path), "只验收本轮固定50步生产latest，不接受best/中途或验收副本")
    report.check("strict_load", "PASS", "生产350+50=400双头/优化器/三池计数/采样RNG严格恢复")
    t02.rejection(report, "legacy_format_guard", lambda: continuation.read_checkpoint(path), "独立续段修复")
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "source_version_guard", lambda: trainer.restore(dict(payload, source_worker_version=300)), "源350")
    t02.rejection(report, "counter_guard", lambda: trainer.restore(dict(payload, counters=dict(payload["counters"], base_updates=1))), "计数")
    t02.rejection(report, "identity_guard", lambda: trainer.restore(dict(payload,
        input_identity=dict(payload["input_identity"], within_horizon_id="wrong"))), "身份")
    counts = copy.deepcopy(payload["sampling_counts"])
    counts["feedback_row"][0] += 1
    t02.rejection(report, "sample_count_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "计数")
    counts = copy.deepcopy(payload["sampling_counts"])
    donor = next(int(i) for i in trainer.data.manual_train_rows if counts["feedback_row"][i] > 0)
    counts["feedback_row"][donor] -= 1
    counts["feedback_row"][0] += 1  # trial0 is a worker episode; counts remain total-preserving.
    t02.rejection(report, "sample_split_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "混入留出/worker")
    heads = copy.deepcopy(payload["heads"])
    heads["goal"]["state_projection.weight"] = heads["goal"]["state_projection.weight"][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, heads=heads)), "权重不兼容")
    artifact_path = report.directory / "roundtrip_verification.pt"
    artifact = repair.save_checkpoint(trainer, artifact_path, digest, verification_artifact=True)
    other = make_trainer(inputs, args.device)
    again, _ = repair.read_checkpoint(artifact_path, other, digest)
    other.restore(again, allow_verification=True)
    restored = other.payload(True)
    for key in ("heads", "optimizers", "counters", "sampler_state", "rng_state", "sampling_counts", "input_identity", "options"):
        gl.require(t01.same(artifact[key], restored[key]), f"预算内恢复{key}不同")
    inference = repair.load_policy(path, trainer.data.cache, args.device)
    x, g, r, _ = trainer.data.feedback_batch(np.arange(80), args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            gl.require(torch.equal(inference.action_dist(x,g,r,mode).probs, trainer.model.action_dist(x,g,r,mode).probs) and
                torch.equal(trainer.model.action_dist(x,g,r,mode).probs, other.model.action_dist(x,g,r,mode).probs), "恢复/独立推理分布不同")
        swapped = torch.as_tensor(trainer.data.fixed_goals[1-trainer.data.feedback_arrays["target"]], device=args.device)
        gl.require(torch.equal(inference.action_dist(x,swapped,r).probs, trainer.model.action_dist(x,swapped,r).probs), "交换目标分布不同")
    t02.rejection(report, "inference_artifact_guard", lambda: repair.load_policy(artifact_path, trainer.data.cache, args.device), "合成验收")
    report.check("roundtrip", "PASS", "双头/优化器/计数/三池RNG往返相同；独立推理无优化器，五分布同值")
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    metrics, _ = evaluation_outputs(trainer, report, initial)
    gl.require(t01.same(metrics, trained["final_metrics"]), "独立verify与生产latest最终指标不同")
    rng = bc.capture_rng(args.device)
    left = trainer.update(verification_only=True)
    bc.restore_rng(rng, args.device)
    right = other.update(verification_only=True)
    gl.require(left["shared_batch_id"] == right["shared_batch_id"] and trainer.counters() == other.counters() and
        t01.same(trainer.counts, other.counts) and t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and
        all(t01.same(trainer.model.heads[k].state_dict(), other.model.heads[k].state_dict()) and
            t01.same(trainer.optimizers[k].state_dict(), other.optimizers[k].state_dict()) for k in ("goal", "no_goal")),
        "恢复后下一真实混合batch/优化器更新不同")
    trainer.model.check_frozen(content=True)
    other.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "验收改变冻结bundle")
    report.data.update(counters=payload["counters"], optimizer_updates=4, verification_only=True,
        verification_updates_per_copy=1, verification_updates_not_saved_to_training=True, worker_version=400,
        additional_updates=50, model_id=inference.model_id, metrics=metrics, checkpoint_sha256=bc.file_hash(path))
    report.check("next_update_equivalence", "PASS", "两个内存副本各一次真实混合更新完全相同；源latest不回写，不计入生产50步")
    report.check("frozen_dependencies", "PASS", "底座/WM/目标未变，验收快照禁止训练/控制，生产计数仍350+50=400")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--supervision-dir", required=True)
    training = commands.add_parser("train")
    training.add_argument("--check-dir", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--check-dir", required=True)
    verification.add_argument("--checkpoint", required=True)
    for command in (checking, training, verification):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
        command.add_argument("--feedback-dir")
    args = parser.parse_args()
    try:
        if args.command != "check":
            args.check_dir = str(baseline.project_path(args.check_dir))
            checked = legacy.read_json(Path(args.check_dir) / "report.json")
            legacy.accepted(checked, "check")
            args.supervision_dir = checked["arguments"]["supervision_dir"]
        args.supervision_dir = str(baseline.project_path(args.supervision_dir))
        source, protected, files = historical_inputs(args)
        if args.command != "check":
            protected.append(Path(args.check_dir))
            files += [Path(args.check_dir) / "report.json", *[Path(args.check_dir) / n for n in checked["check_artifacts"]]]
        if args.command == "verify":
            checkpoint = baseline.project_path(args.checkpoint)
            args.checkpoint = str(checkpoint)
            protected.append(checkpoint.parent)
            files += [checkpoint, checkpoint.parent / "report.json"]
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            f"within_horizon_repair_{args.command}_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        gl_paths = [p.resolve() for p in protected]
        if any(directory == p or p in directory.parents or directory in p.parents for p in gl_paths):
            parser.error("输出须独立于原模型/缓存/轨迹/目标/check/验收及快照，禁止覆盖")
        directory.mkdir(parents=True, exist_ok=False)
        args.output_dir = str(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, worker_control_actions=0, intervention_actions=0,
        labels_generated=False, expert_labels_generated=False, behavior_accepted=False, t06_approved=False,
        dual_goal_correction_train_coverage=False,
        training_scope="independent fixed-budget factual development repair; no expert or behavior approval")
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        files = list(dict.fromkeys(p.resolve() for p in files))
        before = {str(p): baseline.file_signature(p) for p in files}
        report.data["source_inputs_before"] = before
        bc.require_disk_space(directory, 128 * 1024**2)
        baseline.write_json(directory / "output_probe.json", dict(atomic_output_probe_only=True))
        gl.require(legacy.read_json(directory / "output_probe.json")["atomic_output_probe_only"] is True, "原子JSON预检失败")
        report.check("output_preflight", "PASS", "独立新目录/原子JSON/空间预检通过，旧目录拒绝覆盖")
        report.check("historical_identity", "PASS", "完整监督check/20反馈局/80事实/原350及verify/旧30局失败和47代码SHA绑定，不改旧记录")
        rng = bc.capture_rng(args.device)
        inputs = load_inputs(args, report, source)
        {"check":check, "train":train, "verify":verify}[args.command](args, report, inputs)
        gl.require(not any(n in ("minedojo", "mineclip") or n.startswith(("minedojo.", "mineclip.")) for n in sys.modules),
                   "离线修复不能导入MineDojo/MineCLIP")
        report.check("scope", "WARN", "原预算保持事实的固定短修复；目标1训练纠偏仍缺失，离线拟合不等于自主控制；新400确认入口待后续实现，T06未批准")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {name:baseline.file_signature(Path(name)) for name in before}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                    "原模型/缓存/轨迹/目标/监督/验收及47个旧代码大小和修改时间不变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    result = report.finish()
    try:
        import t05_result_bundle as feedback
        sources = [("run",directory), ("supervision",source["supervision_dir"]),
            ("feedback_evaluate",source["feedback_dir"]), ("feedback_check",source["feedback_checked_dir"]),
            ("diagnosis",source["diagnosis_directory"]), ("source_evaluate",source["eval_dir"]),
            ("source350_verify",source["context"]["verify_dir"])]
        if args.command != "check":
            sources.append(("repair_check",Path(args.check_dir)))
        if args.command == "verify":
            sources.append(("repair_train",baseline.project_path(args.checkpoint).parent))
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT / "relevance_map/t05_feedback" / (
            directory.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in [*protected,directory]):
            raise ValueError("反馈包输出须独立于全部源目录与新结果目录")
        index = feedback.build_bundle(sources, export, directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export / 'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"[FAIL] feedback_export: {type(error).__name__}: {error}；原报告已保留", file=sys.stderr, flush=True)
        result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())
