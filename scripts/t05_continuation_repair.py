"""Offline continuation mixture repair: check/train/verify; no environment."""

import argparse
import copy
import gc
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
import t05_goal_information_probe as information
import t05_repaired_control as confirmed
import t05_reference_repair as previous
import t05_closed_loop_continuation as collection


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_CONTINUATION_REPAIR_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(collection.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (
        *previous.CODE_FILES, "goal_continuation_repair.py", "scripts/t05_continuation_repair.py",
        "scripts/t00_baseline.py", "scripts/t01_checkpoint_check.py", "scripts/t02_goal_library.py",
        "scripts/t04_goal_bc.py", "scripts/t05_goal_control.py")})


def historical_inputs(directory):
    import goal_bc as bc
    import goal_library as gl
    import goal_closed_loop_continuation as protocol

    directory = baseline.project_path(directory)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "evaluate")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    plan = manifest["design"]
    protocol.validate_plan(plan)
    gl.require(manifest.get("format") == protocol.EVALUATION_FORMAT and
        manifest.get("comparison_protocol") == protocol.PROTOCOL and
        manifest["evaluation_id"] == record["evaluation_id"] == gl.tensor_digest({}, {
            k: v for k, v in manifest.items() if k != "evaluation_id"}) and
        manifest["design_id"] == record["design_id"] == plan["design_id"] == gl.tensor_digest({}, {
            k: v for k, v in plan.items() if k != "design_id"}) and
        legacy.read_json(directory / "design.json") == plan and
        record["input_identity"] == manifest["input_identity"] == plan["input_identity"] and
        record["evaluation_code"] == plan["evaluation_code"] == collection.code_identity(),
        "续段完整评价/设计/原代码或worker300身份不同")
    expected = dict(execution_valid_trials=10, completed_trial_records=10, continuation_episodes=10,
        continuation_action_records=40, worker_action_queries=200, worker_control_actions=180,
        intervention_actions=20, new_env_steps=840, probability_roundtrip_queries=200, measurement_roundtrip_frames=210)
    gl.require(all(record.get(k) == v for k, v in expected.items()) and record.get("optimizer_updates") == 0 and
        record.get("full_design_completed") is True and manifest.get("full_design_completed") is True and
        record.get("behavior_accepted") is False and record.get("t06_approved") is False and
        confirmed.passed_checks(record, ("historical_identity", "design_identity", "real_control_interface", "execution_counters",
            "fixed_horizon_endpoints", "actual_endpoint_data", "no_training", "source_inputs_unchanged")),
        "需要完整10局/40个真实续段动作的已通过采集，不能筛选或补样")
    source, protected, files = collection.historical_inputs(SimpleNamespace(command="check",
        intervention_dir=record["arguments"]["intervention_dir"]))
    gl.require(plan["source_intervention_id"] == source["intervention_manifest"]["evaluation_id"] and
        plan["source_evaluation_id"] == source["manifest"]["evaluation_id"] and
        plan["original_numerical_gate"] == source["record"]["numerical_gate"] and
        plan["input_identity"] == source["manifest"]["input_identity"], "原干预/30局失败/修复verify来源不同")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(bc.file_hash(checked_dir / "report.json") == manifest["check_report_sha256"] and
        legacy.read_json(checked_dir / "design.json") == plan and checked.get("evaluation_code") == collection.code_identity() and
        bc.file_hash(checked_dir / "targets.npz") == checked["targets_sha256"] and
        confirmed.passed_checks(checked, ("historical_identity", "continuation_contracts", "predeclared_continuation",
            "no_training", "source_inputs_unchanged")), "原续段check身份不同")
    rows = legacy.read_json(directory / "trials.json")
    summary = protocol.summarize(rows, plan)
    gl.require(rows == manifest["rows"] and summary["valid_histories"] == 10 and
        all(r["execution_valid"] and r["fixed_horizon_completed"] and r["actual_steps"] == 20 for r in rows),
        "必须保留完整十局续段，失败局不能删除")
    required = {"design.json", "targets.npz", "trials.json", "diagnostics.json", "continuation_data.json", "coverage.json"}
    gl.require(required <= set(manifest["summary_artifacts"]), "续段汇总缺少SHA256绑定")
    for index, row in enumerate(rows):
        trial = baseline.project_path(row["artifact_dir"])
        expected_dir = directory / f"trial_{index:03d}_repeat_{row['repeat']}_target_1_{row['condition']}"
        gl.require(trial == expected_dir and row["trial_index"] == index, "真实trial路径或顺序不同")
        for name in ("trajectory.npz", "events.json", "control_trace.json", "frame_metrics.json", "history_check.json",
                     "start.json", "metrics.json", "video.mp4", "continuation_endpoint.npz"):
            gl.require(str((trial / name).relative_to(directory)) in manifest["artifacts"], "真实轨迹/终点/完整视频缺少SHA256")
    files += [directory / "report.json", directory / "evaluation_manifest.json"]
    for table in (manifest["artifacts"], manifest["summary_artifacts"]):
        for name, digest in table.items():
            path = (directory / name).resolve()
            gl.require(directory in path.parents and not Path(name).is_absolute() and
                path.is_file() and bc.file_hash(path) == digest, f"真实续段产物SHA256不同：{name}")
            files.append(path)
    index = legacy.read_json(directory / "continuation_data.json")
    gl.require(index == collection.continuation_data(rows, plan), "真实续段索引/实际终点/整局划分不同")
    files += [checked_dir / name for name in ("report.json", "design.json", "targets.npz")]
    files += [ROOT / name for name in code_identity()]
    protected += [directory, checked_dir]
    source.update(continuation_record=record, continuation_manifest=manifest, continuation_directory=directory,
                  continuation_index=index, continuation_check=checked_dir)
    return source, protected, files


def confirmation_plan(source):
    import numpy as np
    import goal_repaired_control as acceptance

    rng = np.random.RandomState(4)
    cells = [(target, mode) for target in (0, 1) for mode in ("goal", "no_goal", "swapped_goal")]
    schedule = [dict(seed=0, repeat=repeat, target=cells[int(i)][0], mode=cells[int(i)][1])
                for repeat in range(5) for i in rng.permutation(6)]
    old = source["manifest"]["design"]
    return dict(worker_version=350, source_design_id=old["design_id"], randomization_seed=4, schedule=schedule,
        execution_policy="mode", planned_trials=30, maximum_new_env_steps=2400, gate=copy.deepcopy(acceptance.GATE),
        target_content_id=old["target_content_id"], scenario=copy.deepcopy(old["scenario"]),
        visual_preprocessing=copy.deepcopy(old["visual_preprocessing"]), environment_fingerprint=copy.deepcopy(old["environment_fingerprint"]),
        action_names=copy.deepcopy(old["action_names"]), warmup_steps=32,
        prefix_actions=copy.deepcopy(old["prefix_actions"]), start_actions=copy.deepcopy(old["start_actions"]),
        control_start_frame=64, horizon=16, actual_endpoint_frame=80, independent_fresh_resets=True,
        matched_start_comparison=False, no_retries_or_replacements=True, implementation_pending=True,
        checkpoint_binding="fixed step50 latest plus new independent verify; no best selection", t06_approved=False)


def continuations_from_histories(source, cache, policy, device):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_information_probe as probe
    import goal_library as gl
    import goal_closed_loop_continuation as protocol

    plan, index = source["continuation_manifest"]["design"], source["continuation_index"]
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], device)
    before = gl.tensor_digest(library.state_dict(), {})
    arrays = {k: [] for k in ("features", "goals", "remaining", "labels", "train", "run", "target", "frame", "manual", "correction")}
    runs, query_count, maximum_error = [], 0, 0.
    gl.require(policy.worker_version == 300 and policy.model_id == plan["input_identity"]["model_id"], "原worker300内容身份不同")
    with np.load(source["continuation_directory"] / "targets.npz", allow_pickle=False) as file:
        fixed = {k: file[k].copy() for k in file.files}
    gl.require(gl.tensor_digest({}, {k: v.tolist() for k, v in fixed.items()}) == plan["target_content_id"], "原两固定目标内容不同")
    for run_index, (item, row) in enumerate(zip(index["episodes"], source["continuation_manifest"]["rows"])):
        directory = baseline.project_path(item["artifact_dir"])
        trajectory = ctl.read_trajectory(directory)
        trace = legacy.read_json(directory / "control_trace.json")
        events = legacy.read_json(directory / "events.json")
        history = legacy.read_json(directory / "history_check.json")
        measured = legacy.read_json(directory / "frame_metrics.json")
        gl.require(protocol.validate_trace(trajectory, events, trace, plan, run_index) == 20 and
            history.get("saved_history_roundtrip") is True and history.get("own_causal_history_passed") is True and
            history.get("reset_is_real") is True and history.get("comparison_to_other_histories") is False and
            history.get("total_frames") == 85 and history.get("start_frame") == 64 and
            row["saved_history_roundtrip"] is True and row["saved_execution_verified"] is True and
            row["video_verified"] is True and row["video"]["frames"] == 85 and
            len(measured) == 21 and [m["frame"] for m in measured] == list(range(64, 85)),
            "续段必须读取本局完整已验收真实因果状态/原生事件/视频")
        x = torch.as_tensor(trajectory["features"][64:84].copy(), device=device)
        g = torch.as_tensor(np.repeat(fixed["goals"][1:2], 20, axis=0), device=device)
        r = torch.as_tensor(protocol.BUDGETS, dtype=torch.int64, device=device)
        with torch.no_grad(), gl.encoder_precision(device):
            probabilities = policy.action_dist(x, g, r, "goal").probs.cpu().numpy()
        error = float(np.max(np.abs(probabilities - np.asarray(trace["probabilities"]))))
        gl.require(np.allclose(probabilities, trace["probabilities"], atol=3e-5, rtol=3e-5) and
            np.array_equal(probabilities.argmax(-1), trace["worker_action_ids"]), "原worker300建议概率或mode不能重现")
        maximum_error = max(maximum_error, error)
        query_count += 20
        with np.load(directory / "continuation_endpoint.npz", allow_pickle=False) as endpoint:
            endpoint = {k: endpoint[k].copy() for k in endpoint.files}
        obs = {k: torch.as_tensor(trajectory[k][84:85], device=device) for k in ("image", "heatmap")}
        own = library(obs)[0].cpu().numpy().astype(np.float32)
        gl.require(np.array_equal(endpoint["image"], trajectory["image"][84]) and
            np.array_equal(endpoint["heatmap"], trajectory["heatmap"][84]) and
            np.allclose(own, endpoint["goal_feature"], atol=3e-5, rtol=3e-5) and
            np.allclose(own, measured[-1]["goal_feature"], atol=3e-5, rtol=3e-5), "本局真实第84帧目标重编码不同")
        # Keep the exact saved endpoint encoding after verifying its real content.
        own = np.asarray(endpoint["goal_feature"], dtype=np.float32)
        for query in item["queries"]:
            frame, label = query["state_frame"], query["actual_action"]
            observed = measured[frame - 64]
            gl.require(query["incoming_action_frame"] == frame + 1 and query["remaining"] == 84 - frame and
                observed["query_remaining"] == query["remaining"] and observed["actual_next_action"] == label and
                observed["decision_source"] == query["decision_source"] and
                np.array_equal(trajectory["action"][frame + 1], np.eye(12, dtype=np.float32)[label]),
                "真实续段obs[t]→incoming[t+1]/remaining或事实测量不同")
            arrays["features"].append(trajectory["features"][frame].copy())
            arrays["goals"].append(own.copy())
            for k, v in dict(remaining=query["remaining"], labels=label, train=item["episode_split"] == "train",
                run=run_index, target=1, frame=frame, manual=item["condition"] == "visual_correct_hold",
                correction=query["decision_source"] == "visual_turn_down_once").items():
                arrays[k].append(v)
        runs.append(dict(trial_index=run_index, repeat=item["repeat"], condition=item["condition"], episode_split=item["episode_split"],
            start_frame=80, endpoint_frame=84, action_ids=[q["actual_action"] for q in item["queries"]],
            decision_sources=[q["decision_source"] for q in item["queries"]], boundary_kind=row["boundary_kind"],
            noop_action=plan["noop_action"], turn_down_action=plan["turn_down_action"],
            trajectory_path=str(directory / "trajectory.npz"), trajectory_sha256=item["trajectory_sha256"],
            endpoint_sha256=item["endpoint_sha256"], events_sha256=bc.file_hash(directory / "events.json"),
            supervision_content_id=probe.array_digest(dict(features=trajectory["features"][80:84].copy(),
                                                           goals=np.repeat(own[None], 4, axis=0))),
            goal_source="this_episode_actual_frame84_rgb_heatmap", expert_labels=False))
    arrays = {k: np.asarray(v, dtype=np.float32 if k in ("features", "goals") else
        np.bool_ if k in ("train", "manual", "correction") else np.int64) for k, v in arrays.items()}
    gl.require(query_count == 200 and gl.tensor_digest(library.state_dict(), {}) == before and
        all(not p.requires_grad and p.grad is None for p in library.parameters()), "冻结目标编码器改变或原查询不完整")
    return arrays, runs, fixed["goals"], maximum_error


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_worker as residual
    import goal_reference_repair as reference
    import goal_continuation_repair as repair

    old_args = source["record"]["arguments"]
    source_path = baseline.project_path(old_args["checkpoint"])
    saved, paths = source["source_checkpoint_payload"], source["source_checkpoint_dependencies"]
    reference.validate_payload(saved, saved["input_identity"], saved["options"])
    gl.require(saved["counters"]["step"] == 100 and args.device == old_args["device"], "需要同一设备的源worker300，不能换模型")
    options = dict(saved["options"], learning_rate=repair.LEARNING_RATE)
    gl.require(options["batch_size"] == 256, "固定修复预算需要batch_size256")
    cache_args = SimpleNamespace(command="check", cache_dir=old_args["cache_dir"],
        t04_verify_dir=str(paths["base_verify_report"].parent), **options)
    original, options = information.load_data(cache_args, report)
    base, references = residual.load_base(paths["base_checkpoint"], paths["base_verify_report"], original.cache)
    calibration = source["calibration"]
    calibration_record = legacy.read_json(paths["calibration_manifest"].parent / "report.json")
    legacy.accepted(calibration_record, "evaluate")
    calibration_check = baseline.project_path(calibration_record["arguments"]["check_dir"])
    ref_arrays, ref_runs, fixed = previous.references_from_histories(calibration, calibration_check, original.cache, args.device)
    original_data = reference.RepairData(original.cache, saved["options"], residual.base_identity(base),
        ref_arrays, ref_runs, saved["input_identity"]["lineage"], fixed)
    gl.require(original_data.identity() == saved["input_identity"], "原参考/缓存/去重查询与worker300训练身份不同")
    policy = reference.load_policy(source_path, original.cache, args.device)
    arrays, runs, fixed_continuation, error = continuations_from_histories(source, original.cache, policy, args.device)
    gl.require(np.array_equal(fixed, fixed_continuation), "续段与参考的原固定目标不同")
    policy.check_frozen(content=True)
    del policy
    directory = source["continuation_directory"]
    lineage = dict(source_checkpoint_sha256=bc.file_hash(source_path), source_worker_version=300,
        source_verify_report_sha256=bc.file_hash(baseline.project_path(old_args["repair_verify_dir"]) / "report.json"),
        continuation_manifest_sha256=bc.file_hash(directory / "evaluation_manifest.json"),
        continuation_index_sha256=bc.file_hash(directory / "continuation_data.json"),
        continuation_evaluation_id=source["continuation_manifest"]["evaluation_id"],
        continuation_design_id=source["continuation_manifest"]["design_id"], optimizer_initialization="fresh Adam; weights only",
        corrective_generalization_verified=False, worker_continuations_role="evaluation only; all actual endpoints retained")
    data = repair.RepairData(original.cache, options, residual.base_identity(base), ref_arrays, ref_runs,
        saved["input_identity"]["lineage"], fixed, arrays, runs, lineage)
    check_path = report.directory / "report.json" if args.command == "check" else baseline.project_path(args.check_dir) / "report.json"
    extras = dict(source_checkpoint=source_path, source_verify_report=baseline.project_path(old_args["repair_verify_dir"]) / "report.json",
        continuation_manifest=directory / "evaluation_manifest.json", continuation_index=directory / "continuation_data.json")
    references.update({k: dict(source_path=p.resolve(), sha256=bc.file_hash(p)) for k, p in extras.items()})
    references["continuation_repair_check_report"] = dict(source_path=check_path.resolve(), sha256=None)
    future = confirmation_plan(source)
    retention = dict(max_absolute_nll_increase=repair.RETENTION_NLL_LIMIT,
        compared_to="step0 worker300", original_full_modes=["goal", "no_goal"],
        reference_goal0_sources=["own_real_endpoint", "fixed_old_targets"], reference_goal0_modes=["goal", "no_goal"],
        automatic_budget_extension=False, corrective_generalization_verified=False, t06_approved=False)
    if args.command != "check":
        checked = legacy.read_json(check_path)
        legacy.accepted(checked, "check")
        gl.require(checked.get("repair_format") == repair.FORMAT and checked.get("repair_code") == code_identity() and
            checked.get("repair_identity") == data.identity() and checked.get("options") == options and
            checked.get("planned_repair_updates") == repair.PLANNED_UPDATES and
            checked.get("backend") == bc.backend_info(args.device) and checked.get("confirmation_plan") == future and
            checked.get("retention_plan") == retention and
            confirmed.passed_checks(checked, ("real_continuation_supervision", "warm_start", "sampler_contract", "training_split_guard", "sample_holdout_guard", "sample_worker_guard",
                "distribution_contract", "diagnostic_contracts", "initial_metrics", "no_updates", "source_inputs_unchanged")),
            "续段修复check数据/代码/预算/后端/门槛不同")
        for name, digest in checked["check_artifacts"].items():
            gl.require(bc.file_hash(check_path.parent / name) == digest, f"check监督/初始指标改变：{name}")
        with np.load(check_path.parent / "supervision.npz", allow_pickle=False) as cached:
            expected = {**{"reference__" + k: v for k, v in ref_arrays.items()}, **{"continuation__" + k: v for k, v in arrays.items()}}
            gl.require(set(cached.files) == set(expected) and all(np.array_equal(cached[k], v) for k, v in expected.items()),
                       "check监督与真实来源不同")
        references["continuation_repair_check_report"]["sha256"] = bc.file_hash(check_path)
    base = {k: v for k, v in base.items() if k not in ("optimizers", "candidate")}
    base["frozen_bundle"] = original.cache.bundle
    report.data.update(repair_format=repair.FORMAT, architecture=residual.ARCHITECTURE, semantics=repair.SEMANTICS,
        repair_identity=data.identity(), input_identity=data.identity(), repair_code=code_identity(), options=options,
        backend=bc.backend_info(args.device), planned_repair_updates=repair.PLANNED_UPDATES, source_worker_version=300,
        sampling_plan=repair.sampling_plan(256), lineage=lineage, confirmation_plan=future, retention_plan=retention,
        source_evaluation_id=source["continuation_manifest"]["evaluation_id"], original_numerical_gate=source["record"]["numerical_gate"],
        probability_roundtrip_queries=200, maximum_probability_error=error)
    report.check("real_continuation_supervision", "PASS", "完整10局/40事实动作与worker300原200查询重现；只训手工12，开发留出8；本局第84帧目标、80–83→81–84/remaining4–1；原参考与原整局留出保持")
    return data, options, base, references, saved


def make_trainer(inputs, device):
    import numpy as np
    import torch
    import goal_continuation_repair as repair

    data, options, base, references, source = inputs
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return repair.Trainer(data, base, references, options, source, device)


def evaluation_outputs(trainer, report, initial=None):
    import goal_continuation_repair as repair

    metrics, frames = repair.evaluate(trainer)
    if initial is not None:
        review = repair.retention_review(initial, metrics)
        metrics["retention_review"] = review
        baseline.write_json(report.directory / "retention.json", review)
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    # The two pools have different metadata (run/target vs trial/condition).
    # Give the established CSV writer one complete, stable set of columns.
    columns = list(dict.fromkeys(k for row in frames for k in row))
    baseline.write_csv(report.directory / "frame_metrics.csv",
        [dict(step=trainer.step, **{k: row.get(k) for k in columns}) for row in frames])
    held = metrics["continuation"]["own_real_endpoint"]["visual_correct_hold"]["by_split"]["development_holdout"]["all_rows"]
    corr = metrics["continuation"]["own_real_endpoint"]["visual_correct_hold"]["by_split"]["train"]["correction"]
    original = metrics["original_cache"]["validation_full"]["row_weighted"]
    print(f"[CONT REPAIR] step={trainer.step} holdout_goal_nll={held['goal_nll']:.6f} "
          f"no_goal_gap={held['no_goal_minus_goal_nll']:.6f} zero_gap={held['zero_minus_goal_nll']:.6f} "
          f"swap_gap={held['swapped_minus_goal_nll']:.6f} correction_train_rows={corr['rows']} "
          f"original_goal_nll={original['goal_nll']:.6f}", flush=True)
    return metrics


def check(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_repair as repair

    trainer = make_trainer(inputs, args.device)
    data, _, _, _, source = inputs
    gl.require(all(t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        not trainer.optimizers[k].state for k in ("goal", "no_goal")), "warm start恢复了旧优化器或源权重不同")
    report.check("warm_start", "PASS", "分别继承worker300双分支权重，新Adam步0；固定追加50步、学习率5e-5，不要求两分支初始权重相同")
    rng = trainer.sampler.get_state()
    first = trainer.draw()
    trainer.sampler.set_state(rng)
    second = trainer.draw()
    gl.require(t01.same(first, second), "保存恢复采样RNG后混合batch不同")
    trainer.sampler.set_state(rng)
    report.check("sampler_contract", "PASS", "每批192原训练+48原参考（每目标24）+16手工续段；均匀有放回，同批双分支；无动作加权或纠偏过采样")
    t02.rejection(report, "sampling_budget_guard", lambda: repair.sampling_plan(255), "32的")
    a = data.continuation_arrays
    actual_pool = data.manual_train_rows.copy()
    try:
        for name, bad_rows in (("sample_holdout_guard", np.flatnonzero(a["manual"] & ~a["train"])),
                              ("sample_worker_guard", np.flatnonzero(~a["manual"] & a["train"]))):
            data.manual_train_rows = bad_rows
            t02.rejection(report, name, trainer.draw, "混入开发留出")
    finally:
        data.manual_train_rows = actual_pool
        trainer.sampler.set_state(rng)
    def validate(changed=a, runs=data.continuation_runs):
        return repair.validate_continuations(changed, runs, data.identity()["feature_dim"], data.identity()["goal_dim"], data.identity()["action_dim"])
    bad = {k: v.copy() for k, v in a.items()}
    bad["train"][0] = not bad["train"][0]
    t02.rejection(report, "training_split_guard", lambda: validate(bad), "整局划分")
    for name, key in (("remaining_guard", "remaining"), ("action_guard", "labels")):
        bad = {k: v.copy() for k, v in a.items()}
        bad[key][0] = 1 if key == "remaining" else (int(a[key][0]) + 1) % 12
        t02.rejection(report, name, lambda bad=bad: validate(bad), "真实下一动作")
    bad = {k: v.copy() for k, v in a.items()}
    bad["goals"][0:4] = -a["goals"][0:4]
    t02.rejection(report, "own_endpoint_guard", lambda: validate(bad), "真实终点")
    bad = {k: v.copy() for k, v in a.items()}
    bad["manual"][0:4] = ~bad["manual"][0:4]
    t02.rejection(report, "manual_source_guard", lambda: validate(bad), "纠偏来源")
    payload = trainer.payload()
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "protocol_guard", lambda: trainer.restore(dict(payload, sampling_protocol="wrong")), "采样协议")
    rows = np.flatnonzero(a["manual"] & ~a["train"])
    x, g, r, _ = data.continuation_batch(rows, args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        raw = trainer.model.base_preferences(x, g, r) + trainer.model.correction("goal", x, g, r)
        gl.require(torch.equal(trainer.model.distribution(raw).probs, trainer.model.action_dist(x, g, r).probs) and
            torch.equal(trainer.model.action_dist(x, g, r, "no_goal").probs,
                trainer.model.action_dist(x, torch.flip(g, [0]), r, "no_goal").probs), "部署unimix或无目标分布不同")
    report.check("distribution_contract", "PASS", "原宏动作/raw偏好相加/归一化及一次unimix不变；no_goal严格目标不变")
    empty = repair.mean_rows([])
    gl.require(empty == dict(available=False, rows=0, episodes=0), "缺纠偏覆盖不能报告零NLL或成功")
    np.savez_compressed(report.directory / "supervision.npz", **{"reference__" + k: v for k, v in data.reference_arrays.items()},
                        **{"continuation__" + k: v for k, v in a.items()})
    cover = repair.coverage(data, args.device)
    baseline.write_json(report.directory / "coverage.json", cover)
    train_cover = cover["continuation"]["visual_correct_hold"]["train"]
    held_cover = cover["continuation"]["visual_correct_hold"]["development_holdout"]
    gl.require(train_cover["rows"] == 12 and held_cover["rows"] == 8 and
        train_cover["correction_rows"] == 1 and held_cover["correction_rows"] == 0,
        "当前已分析数据应为训练1纠偏、开发留出0纠偏；改变采集不能沿用本预算")
    report.check("diagnostic_contracts", "PASS", "保持/实际纠偏/worker事实分别统计；唯一纠偏在train、开发留出不可用；缺覆盖不填零；合成守卫不写轨迹")
    report.check("correction_coverage", "WARN", "只有一条实际下转训练标签；开发留出仅noop，不能检验纠偏泛化，重采样次数不算独立证据")
    initial = evaluation_outputs(trainer, report)
    baseline.write_json(report.directory / "initial_metrics.json", initial)
    baseline.write_json(report.directory / "confirmation_plan.json", report.data["confirmation_plan"])
    baseline.write_json(report.directory / "retention_plan.json", report.data["retention_plan"])
    report.data["check_artifacts"] = {name: bc.file_hash(report.directory / name) for name in
        ("supervision.npz", "coverage.json", "initial_metrics.json", "confirmation_plan.json", "retention_plan.json")}
    report.check("initial_metrics", "PASS", "未更新双分支上报告原26局/参考两目标/手工保持与纠偏/全部worker事实；本局终点与旧固定目标分开，留出NLL增加限度0.02提前声明")
    trainer.model.check_frozen(content=True)
    gl.require(trainer.step == 0 and all(not trainer.optimizers[k].state and
        t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        all(p.grad is None for p in trainer.model.heads[k].parameters()) for k in ("goal", "no_goal")),
        "check不能更新双分支/优化器或产生梯度")
    report.data.update(counters=trainer.counters(), optimizer_updates=0)
    report.check("no_updates", "PASS", "check仅初始化/采样/前向；新优化器步0；无RSSM、候选模型、MineDojo或MineCLIP")


def train(args, report, inputs):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_repair as repair

    trainer = make_trainer(inputs, args.device)
    digest = trainer.reference["continuation_repair_check_report"]["sha256"]
    if args.resume:
        saved, _ = repair.read_checkpoint(baseline.project_path(args.resume), trainer, digest)
        trainer.restore(saved)
        report.check("resume", "PASS", f"仅恢复本实验精确快照：追加步{trainer.step}；总预算仍50步，输出独立目录")
    gl.require(trainer.step < repair.PLANNED_UPDATES, "固定50步预算已用完，不能继续追加或覆盖")
    estimate = 3 * bc.tensor_storage_bytes(trainer.model.heads.state_dict()) + 1024 ** 2
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, 3 * estimate)
    report.check("storage_preflight", "PASS", f"双头完整快照约{estimate / 2**20:.1f}MiB；原模型/缓存/轨迹共享引用，不复制")
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    baseline.write_json(report.directory / "initial_metrics.json", initial)
    current = evaluation_outputs(trainer, report, initial)
    if trainer.step == 0:
        gl.require(t01.same({k: v for k, v in current.items() if k != "retention_review"}, initial), "训练步0指标与check不同")
    else:
        baseline.write_json(report.directory / "resume_start_metrics.json", current)
    report.check("paired_training", "PASS", "固定追加50步，双分支同批真实NLL；实际续段目标/remaining不改，worker事实与开发留出不入训练")
    start_step = trainer.step
    report.data.update(start_step=start_step, updates_this_process_per_branch=0)
    while trainer.step < repair.PLANNED_UPDATES:
        report.require_writable()
        values = trainer.update()
        report.data.update(counters=trainer.counters(), optimizer_updates=trainer.counters()["optimizer_updates"],
                           updates_this_process_per_branch=trainer.step - start_step)
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % 10 == 0 or trainer.step == repair.PLANNED_UPDATES:
            print(f"[TRAIN CONT REPAIR] step={trainer.step}/50 goal_nll={values['goal_nll']:.4f} no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
            current = evaluation_outputs(trainer, report, initial)
            repair.save_checkpoint(trainer, report.directory / "latest.pt", digest)
            sampled = trainer.counts["continuation_row"]
            correction = trainer.data.continuation_arrays["correction"]
            baseline.write_json(report.directory / "sampling.json", dict(plan=trainer.sampling_plan(), start_step=start_step,
                end_step=trainer.step, cumulative_counts={k: v.tolist() for k, v in trainer.counts.items()},
                cumulative_including_resume=True, sampled_manual_labels=int(sampled.sum()),
                sampled_correction_labels=int(sampled[correction].sum()), unique_manual_rows_seen=int(np.count_nonzero(sampled)),
                unique_correction_rows_seen=int(np.count_nonzero(sampled[correction])),
                resampled_labels_are_not_independent_evidence=True, labels_per_branch=trainer.counters()["labels_seen_per_branch"]))
            report.save()
    trainer.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle改变")
    report.data.update(counters=trainer.counters(), optimizer_updates=trainer.counters()["optimizer_updates"],
        worker_version=350, source_worker_version=300, additional_updates=50, final_metrics=current)
    baseline.write_json(report.directory / "diagnostics.json", dict(initial=initial, final=current, fixed_budget_latest_primary=True,
        checkpoint_selected_by_validation=False, source_worker_version=300, additional_updates=50, worker_version=350,
        corrective_generalization_verified=False, behavior_accepted=False, t06_approved=False,
        confirmation_plan=report.data["confirmation_plan"], retention_plan=report.data["retention_plan"]))
    within = current["retention_review"]["within_predeclared_retention_limit"]
    report.check("retention_review", "PASS" if within else "WARN", "按step0检查原26局与目标0参考双分支NLL增加≤0.02；超限原样保留，不加预算或选best")
    report.check("training", "PASS", "两分支各追加50次更新；有限真实NLL/梯度；源300与追加50分开，底座/WM更新0、新环境步0")
    report.check("frozen_dependencies", "PASS", "独立新格式保存双头/新优化器/三池采样/计数/RNG；只存固定latest，无best筛选")


def verify(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_repair as reference
    import goal_residual_worker as residual
    import goal_continuation_repair as repair

    trainer = make_trainer(inputs, args.device)
    path = baseline.project_path(args.checkpoint)
    digest = trainer.reference["continuation_repair_check_report"]["sha256"]
    payload, _ = repair.read_checkpoint(path, trainer, digest)
    trainer.restore(payload)
    gl.require(trainer.step == 50, "只验收固定50步latest，不用中途/best或延长预算")
    report.check("strict_load", "PASS", "新格式双头/新优化器/三池计数/采样RNG严格恢复；源300+追加50分开")
    t02.rejection(report, "legacy_residual_guard", lambda: residual.read_checkpoint(path), "需要残差worker")
    t02.rejection(report, "legacy_reference_guard", lambda: reference.read_checkpoint(path), "需要新修复worker")
    t02.rejection(report, "counter_guard", lambda: trainer.restore(dict(payload, counters=dict(payload["counters"], base_updates=1))), "计数")
    t02.rejection(report, "identity_guard", lambda: trainer.restore(dict(payload,
        input_identity=dict(payload["input_identity"], continuation_id="wrong"))), "身份")
    counts = copy.deepcopy(payload["sampling_counts"])
    counts["continuation_row"][0] += 1
    t02.rejection(report, "sample_count_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "计数")
    counts = copy.deepcopy(payload["sampling_counts"])
    allowed = trainer.data.manual_train_rows
    forbidden = np.flatnonzero(~(trainer.data.continuation_arrays["manual"] & trainer.data.continuation_arrays["train"]))[0]
    donor = next(int(i) for i in allowed if counts["continuation_row"][i] > 0)
    counts["continuation_row"][donor] -= 1
    counts["continuation_row"][forbidden] += 1
    t02.rejection(report, "sample_split_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "混入留出")
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
        gl.require(t01.same(artifact[key], restored[key]), f"续段恢复{key}不同")
    inference = repair.load_policy(path, trainer.data.cache, args.device)
    x, g, r, _ = trainer.data.continuation_batch(np.arange(40), args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            gl.require(torch.equal(inference.action_dist(x, g, r, mode).probs, trainer.model.action_dist(x, g, r, mode).probs) and
                torch.equal(trainer.model.action_dist(x, g, r, mode).probs, other.model.action_dist(x, g, r, mode).probs),
                "续段训练/恢复/独立推理分布不同")
        gl.require(torch.equal(inference.action_dist(x, g, r, "no_goal").probs,
            inference.action_dist(x, torch.flip(g, [0]), r, "no_goal").probs), "独立推理无目标分支依赖目标")
    t02.rejection(report, "inference_artifact_guard", lambda: repair.load_policy(artifact_path, trainer.data.cache, args.device), "合成验收")
    report.data.update(model_id=inference.model_id, worker_version=inference.worker_version)
    report.check("roundtrip", "PASS", "双头/优化器/计数/三池采样/RNG逐值往返；独立推理无优化器，四种分布相同")
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    metrics = evaluation_outputs(trainer, report, initial)
    rng = bc.capture_rng(args.device)
    left = trainer.update()
    bc.restore_rng(rng, args.device)
    right = other.update()
    gl.require(left["shared_batch_id"] == right["shared_batch_id"] and trainer.counters() == other.counters() and
        t01.same(trainer.counts, other.counts) and t01.same(trainer.sampler.get_state(), other.sampler.get_state()), "恢复后下一真实混合batch/计数不同")
    for name in ("goal", "no_goal"):
        gl.require(t01.same(trainer.model.heads[name].state_dict(), other.model.heads[name].state_dict()) and
            t01.same(trainer.optimizers[name].state_dict(), other.optimizers[name].state_dict()), "下一真实更新/优化器不同")
    trainer.model.check_frozen(content=True)
    other.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle改变")
    report.data.update(counters=payload["counters"], metrics=metrics, verification_only=True,
        verification_updates_per_copy=1, optimizer_updates=4, verification_updates_not_saved_to_training=True)
    report.check("next_update_equivalence", "PASS", "两内存副本额外各一次真实混合更新完全一致，源latest不回写；不是追加训练预算")
    report.check("frozen_dependencies", "PASS", "原底座/WM/目标不变；验收快照禁止resume/控制；无RSSM/环境构造")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--continuation-dir", required=True)
    training = commands.add_parser("train")
    training.add_argument("--check-dir", required=True)
    training.add_argument("--resume", help="本实验未完成快照精确恢复；新目录、总追加预算仍50")
    verification = commands.add_parser("verify")
    verification.add_argument("--check-dir", required=True)
    verification.add_argument("--checkpoint", required=True)
    for command in (checking, training, verification):
        command.add_argument("--output-dir", required=True)
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    directory = baseline.project_path(args.output_dir)
    try:
        source_dir = args.continuation_dir if args.command == "check" else legacy.read_json(
            baseline.project_path(args.check_dir) / "report.json")["arguments"]["continuation_dir"]
        source, protected, files = historical_inputs(source_dir)
        old_args = source["record"]["arguments"]
        cache = baseline.project_path(old_args["cache_dir"])
        cache_manifest = legacy.read_json(cache / "cache_manifest.json")
        metadata = cache_manifest["source_metadata"]
        raw_sources = [Path(metadata["checkpoint"]["path"]), Path(metadata["library_path"]),
                       *[Path(entry["path"]) for entry in metadata["episodes"]]]
        files += raw_sources
        files += [Path(cache_manifest["dataset_dir"]) / name for name in ("segments_manifest.json", "segments.npz", "report.json")]
        files += [Path(cache_manifest["t03_verify_dir"]) / "report.json"]
        import goal_reference_repair as reference
        payload, dependencies = reference.read_checkpoint(baseline.project_path(old_args["checkpoint"]))
        source.update(source_checkpoint_payload=payload, source_checkpoint_dependencies=dependencies)
        calibration_dir = dependencies["calibration_manifest"].parent
        calibration_record = legacy.read_json(calibration_dir / "report.json")
        calibration_check = baseline.project_path(calibration_record["arguments"]["check_dir"])
        files += [calibration_dir / "report.json", *[calibration_check / name for name in ("report.json", "targets.npz", "design.json")]]
        protected += [calibration_dir, calibration_check]
        if args.command != "check":
            check_dir = baseline.project_path(args.check_dir)
            checked = legacy.read_json(check_dir / "report.json")
            legacy.accepted(checked, "check")
            protected.append(check_dir)
            files += [check_dir / "report.json", *[check_dir / name for name in checked["check_artifacts"]]]
            checkpoint = args.resume if args.command == "train" else args.checkpoint
            if checkpoint:
                path = baseline.project_path(checkpoint)
                protected.append(path.parent)
                files.append(path)
        protected += [cache, Path(cache_manifest["dataset_dir"]), Path(cache_manifest["t03_verify_dir"])]
        protected += [p.resolve().parent for p in raw_sources if p.resolve().parent != ROOT]
        protected += [path.parent for path in files if ROOT not in path.resolve().parents]
        if any(directory == path.resolve() or path.resolve() in directory.parents or directory in path.resolve().parents for path in protected):
            parser.error("输出必须独立于旧采集/失败/模型/缓存/check及输入快照，禁止覆盖")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    report.data.update(new_env_steps=0, worker_control_actions=0, corrective_generalization_verified=False,
                       expert_labels_generated=False, behavior_accepted=False, t06_approved=False)
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        before = {str(p): baseline.file_signature(p) for p in dict.fromkeys(path.resolve() for path in files)}
        report.data["source_inputs_before"] = before
        report.require_writable()
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        rng = bc.capture_rng(args.device)
        report.check("historical_identity", "PASS", "完整续段10局/干预10局/原30局/标定及修复verify和全部真实SHA256绑定；原代码/旧失败不改写")
        inputs = load_inputs(args, report, source)
        {"check": check, "train": train, "verify": verify}[args.command](args, report, inputs)
        gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or name == "mineclip" or
            name.startswith("mineclip.") for name in sys.modules), "离线修复不能导入MineDojo/MineCLIP")
        report.check("scope", "WARN", "同世界开发数据短修复；仅一条训练纠偏、留出只有保持；离线工程PASS不证明纠偏泛化，不批准T06；新自主确认入口待后续实现")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {name: baseline.file_signature(Path(name)) for name in before}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                    "原模型/缓存/轨迹/目标/验收/check与旧代码大小、修改时间未变；真实产物另按SHA256验真，只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
