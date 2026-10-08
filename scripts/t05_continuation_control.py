"""Worker350 confirmation: offline check, then the preregistered 30 real trials.

Only the fixed production latest and its independent continuation-repair verify
are accepted. No training, checkpoint selection, retries or partial resume.
"""
import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_random_control as pilot
import t05_continuation_repair as mixing


class Report(pilot.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_CONTINUATION_CONTROL_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(mixing.code_identity(), **{name: bc.file_hash(ROOT / name) for name in
        ("goal_continuation_control.py", "scripts/t05_continuation_control.py")})


def source_paths(args):
    import goal_bc as bc
    import goal_continuation_repair as repair
    import goal_continuation_control as acceptance
    import goal_library as gl

    verify_dir = baseline.project_path(args.repair_verify_dir)
    verified = legacy.read_json(verify_dir / "report.json")
    legacy.accepted(verified, "verify")
    checkpoint = baseline.project_path(verified["arguments"]["checkpoint"])
    payload, refs = repair.read_checkpoint(checkpoint)
    repair.validate_payload(payload, payload["input_identity"], payload["options"])
    check_dir = refs["continuation_repair_check_report"].parent
    checked = legacy.read_json(check_dir / "report.json")
    trained = legacy.read_json(checkpoint.parent / "report.json")
    acceptance.validate_source_records(checkpoint, check_dir, checked, trained, verified, payload)
    gl.require(baseline.project_path(verified["arguments"]["output_dir"]) == verify_dir and
        verified["arguments"]["device"] == args.device and
        verified["source_inputs_before"].get(str(checkpoint)) == baseline.file_signature(checkpoint),
        "需要同设备、同生产latest及原路径的独立verify")
    for name, digest in checked["check_artifacts"].items():
        path = (check_dir / name).resolve()
        gl.require(check_dir in path.parents and path.is_file() and bc.file_hash(path) == digest,
                   f"原修复check产物SHA256不同：{name}")
    locked = legacy.read_json(check_dir / "confirmation_plan.json")
    retention_plan = legacy.read_json(check_dir / "retention_plan.json")
    gl.require(locked == checked["confirmation_plan"] and retention_plan == checked["retention_plan"],
               "原修复check预声明计划不同")
    initial = legacy.read_json(check_dir / "initial_metrics.json")
    metrics = legacy.read_json(checkpoint.parent / "metrics.json")
    retention = repair.retention_review(initial, metrics)
    gl.require(retention["within_predeclared_retention_limit"] is True and
        retention == legacy.read_json(checkpoint.parent / "retention.json") ==
        legacy.read_json(verify_dir / "retention.json") == metrics["retention_review"] and
        metrics == trained["final_metrics"] == verified["metrics"] == legacy.read_json(verify_dir / "metrics.json") and
        initial == legacy.read_json(checkpoint.parent / "initial_metrics.json") and
        metrics["step"] == metrics["additional_updates"] == 50 and metrics["worker_version"] == 350,
        "固定50步最终指标或六项预声明退步检查不同")
    source, protected, files = mixing.historical_inputs(refs["continuation_manifest"].parent)
    gl.require(locked == mixing.confirmation_plan(source) and
        refs["continuation_manifest"] == source["continuation_directory"] / "evaluation_manifest.json" and
        refs["continuation_index"] == source["continuation_directory"] / "continuation_data.json",
        "预声明计划或真实续段来源不同")
    cache_dir = refs["frozen_bundle"].parent
    gl.require(cache_dir == baseline.project_path(source["record"]["arguments"]["cache_dir"]),
               "原300与新350必须使用同一真实缓存")
    cache_protected, cache_files, raw_source = legacy.source_paths(cache_dir,
        SimpleNamespace(checkpoint=str(checkpoint), check_dir=getattr(args, "check_dir", None)))
    protected += cache_protected + [verify_dir, check_dir, checkpoint.parent]
    files += cache_files + list(refs.values()) + [checkpoint, verify_dir / "report.json",
        verify_dir / "metrics.json", verify_dir / "retention.json",
        *[check_dir / name for name in checked["check_artifacts"]],
        *[checkpoint.parent / name for name in ("report.json", "initial_metrics.json", "metrics.json",
                                               "retention.json", "sampling.json", "diagnostics.json", "evaluation.jsonl")]]
    protected += [p.parent for p in refs.values()]
    files += [ROOT / name for name in code_identity()]
    if args.command == "evaluate":
        directory = baseline.project_path(args.check_dir)
        old = legacy.read_json(directory / "report.json")
        legacy.accepted(old, "check")
        protected.append(directory)
        files += [directory / name for name in ("report.json", "design.json", "targets.npz")]
    files = list(dict.fromkeys(p.resolve() for p in files))
    gl.require(all(p.is_file() for p in files), "输入依赖缺失，请保留服务器生产latest、verify、check和真实来源")
    context = dict(checkpoint=checkpoint, payload=payload, refs=refs, cache_dir=cache_dir,
        check_dir=check_dir, checked=checked, trained=trained, verified=verified, source=source,
        locked=locked, retention=retention, verify_dir=verify_dir, raw_source=raw_source)
    return context, list(dict.fromkeys(p.resolve() for p in protected)), files


def load_inputs(args, report, context):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_information_probe as probe
    import goal_library as gl
    import goal_random_control as randomized
    import goal_residual_worker as residual
    import goal_continuation_repair as repair
    import goal_continuation_control as acceptance

    cache = bc.TrainingCache(context["cache_dir"])
    policy = repair.load_policy(context["checkpoint"], cache, args.device)
    payload, verified, checked = context["payload"], context["verified"], context["checked"]
    gl.require(policy.worker_version == 350 and policy.continuation_updates == 50 and
        verified.get("model_id") == policy.model_id and verified.get("repair_identity") == policy.identity and
        verified.get("counters") == payload["counters"] and
        verified.get("backend") == checked.get("backend") == bc.backend_info(args.device),
        "worker350独立推理内容/版本/计数/后端与verify不同")
    identity = dict(checkpoint=baseline.file_signature(context["checkpoint"]),
        checkpoint_sha256=bc.file_hash(context["checkpoint"]), checkpoint_format=repair.FORMAT,
        input_identity=policy.identity, counters=payload["counters"], model_id=policy.model_id,
        worker_version=350, source_worker_version=300, additional_updates=50,
        inference_backend=bc.backend_info(args.device), dependencies=payload["dependencies"],
        repair_verify_sha256=bc.file_hash(context["verify_dir"] / "report.json"),
        repair_train_report_sha256=bc.file_hash(context["checkpoint"].parent / "report.json"),
        repair_check_report_sha256=bc.file_hash(context["check_dir"] / "report.json"),
        confirmation_plan_sha256=bc.file_hash(context["check_dir"] / "confirmation_plan.json"),
        retention_sha256=bc.file_hash(context["checkpoint"].parent / "retention.json"))
    runtime = residual.OnlineRuntime(policy, args.device)
    locked, source = context["locked"], context["source"]
    fingerprint = ctl.environment_fingerprint(ROOT)
    gl.require(fingerprint == locked["environment_fingerprint"], "真实环境或视觉预处理与预声明场景不同")
    with np.load(source["continuation_directory"] / "targets.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    dimension = len(locked["action_names"])
    gl.require(bank["goals"].shape == (2, cache.metadata["goal_dim"]) and bank["goals"].dtype == np.float32 and
        bank["images"].shape == (2, 64, 64, 3) and bank["heatmaps"].shape == (2, 64, 64) and
        bank["images"].dtype == bank["heatmaps"].dtype == np.uint8 and np.isfinite(bank["goals"]).all() and
        np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5) and
        probe.array_digest({"goals": bank["goals"]}) == policy.identity["fixed_goals_id"] and
        gl.tensor_digest({}, {k: v.tolist() for k, v in bank.items()}) == locked["target_content_id"] and
        dimension == runtime.bundle["config"]["num_actions"] and pilot.action_names() == locked["action_names"],
        "原两固定视觉目标、内容ID或动作接口不同")
    calibrated = source["calibration"]["design"]
    old = source["manifest"]["design"]
    gl.require(calibrated["target_content_id"] == old["target_content_id"] == locked["target_content_id"] and
        old["target_telemetry"] == calibrated["target_telemetry"] and
        all(locked[k] == old[k] for k in ("scenario", "visual_preprocessing", "start_actions", "prefix_actions")),
        "原已标定终点/初始化或诊断遥测不同")
    for target in (0, 1):
        encoded = ctl.encode_goal(runtime, dict(image=bank["images"][target], heatmap=bank["heatmaps"][target]))
        gl.require(np.allclose(encoded, bank["goals"][target], atol=3e-5, rtol=3e-5),
                   "原固定RGB/heatmap目标重编码不同")
    design = dict(seed=0, repetitions=5, randomization_seed=4, execution_policy="mode",
        modes=list(randomized.MODES), schedule=copy.deepcopy(locked["schedule"]))
    plan = dict(format=acceptance.FORMAT, comparison_protocol=acceptance.PROTOCOL,
        preregistered_confirmation=copy.deepcopy(locked), design=design,
        input_identity=identity, evaluation_code=code_identity(),
        **{name: copy.deepcopy(locked[name]) for name in acceptance.LOCKED_FIELDS})
    plan.update(source_worker_version=300, additional_updates=50, target_telemetry=copy.deepcopy(old["target_telemetry"]),
        same_hidden_state=False, matched_start_comparison=False, old_failures_unchanged=True,
        goal_source="unchanged calibrated real RGB/heatmap endpoints; coordinates diagnostic only",
        source_evaluation_id=source["continuation_manifest"]["evaluation_id"],
        source_confirmation_design_id=old["design_id"])
    acceptance.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(input_identity=identity, model_id=policy.model_id, worker_version=350,
        source_worker_version=300, additional_updates=50, counters=payload["counters"],
        evaluation_code=code_identity(), comparison_protocol=acceptance.PROTOCOL,
        environment_fingerprint=fingerprint, environment_protocol=ctl.ENVIRONMENT_PROTOCOL,
        visual_preprocessing_policy=ctl.VISUAL_PREPROCESSING_POLICY, design_id=plan["design_id"],
        planned_trials=30, maximum_new_env_steps=2400, execution_policy="mode", randomization_seed=4,
        original_numerical_gate=source["record"]["numerical_gate"], offline_retention=context["retention"],
        matched_start_comparison=False, old_failures_unchanged=True, candidate_used_for_selection=False,
        same_hidden_state=False, controlled_reachability_verified=False, corrective_generalization_verified=False,
        task_success_evaluated=False, optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0,
        worker_control_actions=0, worker_action_queries=0, probability_roundtrip_queries=0,
        measurement_roundtrip_frames=0, maximum_probability_error=0., intervention_actions=0,
        behavior_accepted=False, t06_approved=False)
    report.check("historical_identity", "PASS", "绑定本轮check/固定50步latest/独立verify与完整真实续段及旧SHA256；原300行为失败保持")
    report.check("strict_continuation_inference_load", "PASS", "源300+追加50=worker350；独立推理与verify同内容/后端；只构造冻结底座/WM/目标库与双分支，无优化器或候选模型")
    report.check("preregistered_confirmation", "PASS", "复算本轮六项退步检查并绑定原check计划；种子4/mode/30局/原16步预算不改")
    report.check("fixed_visual_targets", "PASS", "沿用原两已标定真实RGB/heatmap终点与完整初始化脚本；重编码通过，坐标只诊断")
    if args.command == "evaluate":
        checked_dir = baseline.project_path(args.check_dir)
        run_check = legacy.read_json(checked_dir / "report.json")
        legacy.accepted(run_check, "check")
        gl.require(run_check.get("comparison_protocol") == acceptance.PROTOCOL and
            run_check.get("input_identity") == identity and run_check.get("design_id") == plan["design_id"] and
            run_check.get("evaluation_code") == code_identity() and
            legacy.read_json(checked_dir / "design.json") == plan and
            run_check.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz") and
            acceptance.passed(run_check, ("historical_identity", "strict_continuation_inference_load",
                "preregistered_confirmation", "fixed_visual_targets", "trial_output_preflight",
                "causal_execution_interface", "independent_start_contract", "confirmation_gate_contract",
                "saved_execution_contract", "worker350_source_contract", "predeclared_design",
                "no_training", "source_inputs_unchanged")),
            "需要本次worker350确认check；旧check或修改后的计划不能执行")
        with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
            gl.require(all(np.array_equal(saved[k], bank[k]) for k in bank), "确认check的固定目标改变")
        report.check("design_identity", "PASS", "worker350/verify/种子4/mode/30次原16步计划及代码与check一致")
    gc.collect()
    return cache, runtime, plan, bank


def trace_identity(plan, cell, index):
    return dict(worker_version=350, source_worker_version=300, additional_updates=50,
        model_id=plan["input_identity"]["model_id"], comparison_protocol=plan["comparison_protocol"],
        design_id=plan["design_id"], trial_index=index, control_start_frame=64,
        condition=cell["mode"], evaluated_target=cell["target"],
        issued_target=None if cell["mode"] == "no_goal" else 1 - cell["target"] if cell["mode"] == "swapped_goal" else cell["target"])


def run_trial(report, runtime, plan, bank, cell, directory):
    # The accepted runner executes all actions and owns fresh reset/history/video.
    # This wrapper adds only the actual new worker identity, never an action override.
    row = pilot.run_trial(report, runtime, plan, bank, cell, directory)
    index = plan["design"]["schedule"].index(cell)
    trace = legacy.read_json(directory / "control_trace.json")
    trace.update(trace_identity(plan, cell, index))
    baseline.write_json(directory / "control_trace.json", trace)
    row.update(trial_index=index, worker_version=350, source_worker_version=300, additional_updates=50,
        model_id=runtime.policy.model_id, worker_action_queries=len(trace["probabilities"]),
        intervention_actions=0)
    report.data["worker_action_queries"] += row["worker_action_queries"]
    return row


def check(args, report, cache, runtime, plan, bank, context):
    import numpy as np
    import goal_library as gl
    import goal_continuation_control as confirmation
    import goal_continuation_repair as repair

    # Reuse accepted causal-state/distribution/output guards without changing
    # their fingerprinted implementations or the deployment policy.
    pilot.check(SimpleNamespace(device=args.device, repetitions=5), report, cache, runtime, plan, bank)
    first, dimension = plan["control_start_frame"], len(plan["action_names"])
    frames = first + plan["horizon"] + 1
    markers = dict(image=np.zeros((frames, 64, 64, 3), np.uint8), heatmap=np.zeros((frames, 64, 64), np.uint8),
        features=np.zeros((frames, runtime.bundle["feature_dim"]), np.float32), obs_reward=np.zeros((frames, 1), np.float32),
        action=np.concatenate([np.zeros((1, dimension), np.float32),
            np.eye(dimension, dtype=np.float32)[plan["start_actions"] + [0] * 16]]),
        is_first=np.arange(frames) == 0, is_last=np.zeros(frames, bool), is_terminal=np.zeros(frames, bool))
    telemetry = dict(pose=dict(x=0., y=0., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    events = [dict(frame=frame, reward=0., done=False, error=None,
                   native_actions=[] if frame == 0 else [[0]], telemetry=telemetry) for frame in range(frames)]
    cell = plan["design"]["schedule"][0]
    seed = (plan["design"]["randomization_seed"] + 701) % (2**31 - 1)
    trace = dict(trace_identity(plan, cell, 0), action_seed=seed, action_ids=[0] * 16, remaining=list(range(16, 0, -1)), execution_policy="mode",
        probabilities=np.eye(dimension)[[0] * 16].tolist(), uniforms=np.random.RandomState(seed).uniform(size=16).tolist())
    def validate_saved(arrays, native, saved):
        return confirmation.validate_trace(arrays, native, saved, plan, cell, 0, dimension)
    gl.require(validate_saved(markers, events, trace) == 16, "保存接口标记不一致")
    early = {name: value[:first + 7].copy() for name, value in markers.items()}
    early["is_last"][-1] = True
    early_events = copy.deepcopy(events[:first + 7])
    early_events[-1]["done"] = True
    early_trace = copy.deepcopy(trace)
    for name in ("action_ids", "remaining", "probabilities", "uniforms"):
        early_trace[name] = early_trace[name][:6]
    gl.require(validate_saved(early, early_events, early_trace) == 6, "真实提前结束没有保留实际终点")
    wrong = copy.deepcopy(markers)
    wrong["is_last"][-2] = True
    t02.rejection(report, "saved_terminal_guard", lambda: validate_saved(wrong, events, trace), "真实结束")
    wrong = copy.deepcopy(trace)
    wrong["remaining"][0] = 15
    t02.rejection(report, "saved_remaining_guard", lambda: validate_saved(markers, events, wrong), "剩余预算")
    wrong = copy.deepcopy(trace)
    wrong["action_ids"][0] = 1
    t02.rejection(report, "saved_next_action_guard", lambda: validate_saved(markers, events, wrong), "incoming")
    missing = copy.deepcopy(events)
    del missing[-1]["telemetry"]
    t02.rejection(report, "saved_telemetry_guard", lambda: validate_saved(markers, missing, trace), "遥测")
    wrong = copy.deepcopy(trace)
    wrong["probabilities"][0] = np.eye(dimension)[1].tolist()
    t02.rejection(report, "recorded_mode_guard", lambda: validate_saved(markers, events, wrong), "mode")
    last_index = len(plan["design"]["schedule"]) - 1
    last_cell = plan["design"]["schedule"][last_index]
    last_seed = (plan["design"]["randomization_seed"] + last_index * 1009 + 701) % (2**31 - 1)
    last_trace = dict(trace, **trace_identity(plan, last_cell, last_index))
    last_trace.update(action_seed=last_seed, uniforms=np.random.RandomState(last_seed).uniform(size=16).tolist())
    gl.require(confirmation.validate_trace(markers, events, last_trace, plan, last_cell, last_index, dimension) == 16,
               "后续trial计划种子校验不同")
    wrong = dict(trace, action_seed=seed + 1)
    t02.rejection(report, "action_seed_guard", lambda: validate_saved(markers, events, wrong), "种子")
    wrong = copy.deepcopy(trace)
    wrong["uniforms"][0] = (wrong["uniforms"][0] + .5) % 1
    t02.rejection(report, "action_uniform_guard", lambda: validate_saved(markers, events, wrong), "随机数")
    wrong = dict(trace, worker_version=300)
    t02.rejection(report, "saved_worker_version_guard", lambda: validate_saved(markers, events, wrong), "worker350")
    changed = copy.deepcopy(plan["preregistered_confirmation"])
    del changed["randomization_seed"]
    t02.rejection(report, "missing_design_seed_guard", lambda: confirmation.validate_predeclared(changed), "种子")
    changed = copy.deepcopy(plan["preregistered_confirmation"])
    changed["randomization_seed"] = 1
    t02.rejection(report, "design_seed_guard", lambda: confirmation.validate_predeclared(changed), "种子")
    checkpoint, checked_dir = context["checkpoint"], context["check_dir"]
    records = (context["checked"], context["trained"], context["verified"], context["payload"])
    t02.rejection(report, "best_checkpoint_guard",
        lambda: confirmation.validate_source_records(checkpoint.with_name("best.pt"), checked_dir, *records), "latest")
    wrong = dict(context["verified"], worker_version=300)
    t02.rejection(report, "source_worker_version_guard",
        lambda: confirmation.validate_source_records(checkpoint, checked_dir, context["checked"],
                                                     context["trained"], wrong, context["payload"]), "计数")
    wrong_payload = dict(context["payload"], verification_artifact=True)
    t02.rejection(report, "inference_artifact_guard",
        lambda: repair.validate_payload(wrong_payload, wrong_payload["input_identity"], wrong_payload["options"]), "合成验收")
    report.check("worker350_source_contract", "PASS", "只接受本次50步生产latest和独立verify；旧worker/best/验收副本拒绝；首尾trial种子来自design，内存守卫不写轨迹")
    report.check("saved_execution_contract", "PASS", "真实保存接口检查全历史遥测/原生事件/下一动作/mode/remaining；内存标记不写成轨迹")

    rows = []
    for index, cell in enumerate(plan["design"]["schedule"]):
        target, mode = cell["target"], cell["mode"]
        start, end = [1., 1.], [1., 1.]
        progress = .5 if mode == "goal" else .1
        end[target] -= progress
        rows.append(dict(cell, **trace_identity(plan, cell, index), execution_valid=True, saved_execution_verified=True, video_verified=True,
            actual_steps=16, end_reason="fixed_budget", task_success=False, start_telemetry=telemetry,
            prefix_movement=dict(position=0.), start_distance=1., end_distance=1. - progress,
            distance_improvement=progress, target_preference_margin=progress, preference_improvement=progress,
            start_distances_to_both_goals=start, end_distances_to_both_goals=end))
    summary = confirmation.summarize(rows, plan)
    gl.require(summary["numerical_gate"]["passed"] and not summary["t06_approved"], "数值与人工门槛未分开")
    boundary = copy.deepcopy(rows)
    failing = [row for row in boundary if row["target"] == 1 and row["mode"] == "goal"]
    for row in failing[:1]:
        row.update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                   preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(confirmation.summarize(boundary, plan)["numerical_gate"]["passed"], "4/5边界被错误拒绝")
    failing[1].update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                      preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(not confirmation.summarize(boundary, plan)["numerical_gate"]["passed"], "3/5被错误通过")
    weak = copy.deepcopy(rows)
    for row in weak:
        if row["target"] == 1 and row["mode"] == "goal":
            row.update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                       preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(not confirmation.summarize(weak, plan)["numerical_gate"]["passed"], "pooled结果掩盖一个目标失败")
    missing = copy.deepcopy(rows)
    missing[0].update(execution_valid=False, error="in-memory missing-measurement marker")
    gl.require(not confirmation.summarize(missing, plan)["numerical_gate"]["passed"] and
               not confirmation.summarize(rows[:-1], plan)["numerical_gate"]["passed"], "缺测/未完成被通过")
    equal = copy.deepcopy(rows)
    for row in equal:
        if row["mode"] != "goal":
            target = row["target"]
            end = [1., 1.]
            end[target] = .5
            row.update(end_distance=.5, distance_improvement=.5, target_preference_margin=.5,
                       preference_improvement=.5, end_distances_to_both_goals=end)
    gl.require(not confirmation.summarize(equal, plan)["numerical_gate"]["passed"], "与对照持平被当作优势")
    t02.rejection(report, "confirmation_duplicate_guard", lambda: confirmation.summarize(rows + rows[:1], plan), "重复")
    changed = copy.deepcopy(plan)
    changed["gate"]["minimum_directional_trials_per_target"] = 3
    t02.rejection(report, "gate_change_guard", lambda: confirmation.validate_plan(changed), "门槛")
    changed = copy.deepcopy(plan)
    changed["design"]["execution_policy"] = "sample"
    t02.rejection(report, "execution_policy_guard", lambda: confirmation.validate_plan(changed), "执行方式")
    wrong = copy.deepcopy(rows)
    wrong[0]["distance_improvement"] += .1
    t02.rejection(report, "own_progress_guard", lambda: confirmation.summarize(wrong, plan), "自身进展")
    report.check("confirmation_gate_contract", "PASS", "两个目标分别4/5正进展且正确偏好、五次均值胜两对照；缺测/持平/未完成不通过；数值达标仍需人工核对")
    gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or name == "mineclip" or
                       name.startswith("mineclip.") for name in sys.modules), "离线check不应导入环境/MineCLIP")


def verify_saved_trial(report, runtime, plan, bank, row):
    import numpy as np
    import torch
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_calibration as calibration
    import goal_residual_control as control
    import goal_continuation_control as confirmation

    directory = Path(row["artifact_dir"])
    arrays = ctl.read_trajectory(directory)
    events = legacy.read_json(directory / "events.json")
    trace = legacy.read_json(directory / "control_trace.json")
    history = legacy.read_json(directory / "history_check.json")
    dimension = runtime.bundle["config"]["num_actions"]
    first, target, mode = plan["control_start_frame"], row["target"], row["mode"]
    issued = None if mode == "no_goal" else 1 - target if mode == "swapped_goal" else target
    index = next(index for index, cell in enumerate(plan["design"]["schedule"])
                 if all(cell[name] == row[name] for name in ("seed", "repeat", "target", "mode")))
    cell = plan["design"]["schedule"][index]
    count = confirmation.validate_trace(arrays, events, trace, plan, cell, index, dimension)
    action_seed = (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)
    gl.require(trace["model_id"] == runtime.policy.model_id and trace["condition"] == mode and
               trace["evaluated_target"] == target and trace["issued_target"] == issued and
               trace["action_seed"] == action_seed and
               np.array_equal(trace["uniforms"], np.random.RandomState(action_seed).uniform(size=plan["horizon"])[:count]) and
               history.get("own_causal_history_passed") is True and history.get("reset_is_real") is True and
               history.get("comparison_to_other_histories") is False and history.get("start_frame") == first and
               history.get("total_frames") == len(arrays["image"]) and row["actual_steps"] == count and
               row["video"]["frames"] == len(arrays["image"]), "保存的模型/条件/因果历史或完整解码视频不一致")
    device = next(runtime.parameters()).device
    measurements, maximum_error = [], 0.
    for offset in range(count + 1):
        frame = first + offset
        feature = ctl.encode_goal(runtime, {name: arrays[name][frame] for name in ("image", "heatmap")})
        values = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                     events[frame]["telemetry"], bank, plan["target_telemetry"])
        next_action = trace["action_ids"][offset] if offset < count else None
        measurements.append(dict(frame=frame, control_step=offset, goal_feature=feature.tolist(),
            incoming_action_id=int(np.argmax(arrays["action"][frame])),
            next_action_id=next_action,
            next_action_name=plan["action_names"][next_action] if next_action is not None else None,
            remaining=plan["horizon"] - offset if offset < count else None,
            actual_next_action_probability=trace["probabilities"][offset][next_action] if offset < count else None,
            **values))
        if offset < count:
            state = torch.as_tensor(arrays["features"][frame:frame + 1].copy(), device=device)
            p = control.probabilities(runtime, state, bank["goals"], plan["horizon"] - offset, target, mode)
            maximum_error = max(maximum_error, float(np.max(np.abs(p - trace["probabilities"][offset]))))
            gl.require(np.allclose(p, trace["probabilities"][offset], atol=3e-5, rtol=3e-5) and
                       ctl.select_action(p, "mode", trace["uniforms"][offset]) == trace["action_ids"][offset],
                       "保存真实状态的控制概率或mode动作重现不一致")
    distances = [[item[f"distance_{target}"] for target in (0, 1)] for item in measurements]
    gl.require(np.allclose(distances, trace["distances_to_both_goals"], atol=3e-5, rtol=3e-5),
               "保存真实画面的目标距离重现不一致")
    start, end = np.asarray(distances[0]), np.asarray(distances[-1])
    margin = float(end[1 - target] - end[target])
    # Persist measurements computed from saved, real frames and actual endpoints.
    row.update(start_distances_to_both_goals=start.tolist(), end_distances_to_both_goals=end.tolist(),
        start_distance=float(start[target]), end_distance=float(end[target]),
        distance_improvement=float(start[target] - end[target]), target_preference_margin=margin,
        preference_improvement=margin - float(start[1 - target] - start[target]),
        saved_execution_verified=True, video_verified=True,
        fixed_horizon_completed=count == plan["horizon"], actual_endpoint_frame=first + count,
        worker_version=350, source_worker_version=300, additional_updates=50,
        maximum_probability_error=maximum_error, probability_roundtrip_queries=count,
        measurement_roundtrip_frames=count + 1,
        terminal_action_ids=trace["action_ids"][-2:],
        terminal_action_names=[plan["action_names"][a] for a in trace["action_ids"][-2:]],
        last_step_progress=measurements[-2][f"distance_{target}"] - measurements[-1][f"distance_{target}"],
        last_step_pose_change=ctl.pose_difference(events[-1]["telemetry"]["pose"], events[-2]["telemetry"]["pose"]),
        target_pose_errors_at_start={str(target): {key: value for key, value in measurements[0].items()
            if key.endswith(f"_error_{target}")} for target in (0, 1)},
        target_pose_errors_at_end={str(target): {key: value for key, value in measurements[-1].items()
            if key.endswith(f"_error_{target}")} for target in (0, 1)})
    trace.update(probability_roundtrip_queries=count, measurement_roundtrip_frames=count + 1,
                 maximum_probability_error=maximum_error, actual_endpoint_frame=first + count)
    baseline.write_json(directory / "frame_metrics.json", measurements)
    baseline.write_json(directory / "control_trace.json", trace)
    report.data["probability_roundtrip_queries"] += count
    report.data["measurement_roundtrip_frames"] += count + 1
    report.data["maximum_probability_error"] = max(report.data["maximum_probability_error"], maximum_error)
    return measurements


def outcome_gallery(report, plan, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw
    import goal_random_control as random_control

    for repeat in range(plan["design"]["repetitions"]):
        panel = Image.new("RGB", (660, 600), "white")
        draw = ImageDraw.Draw(panel)
        for target in (0, 1):
            x = 220 + target * 220
            draw.text((x, 3), f"fixed target {target}", fill="black")
            panel.paste(Image.fromarray(bank["images"][target]).resize((128, 128)), (x, 20))
        for index, mode in enumerate(random_control.MODES, 1):
            y = index * 150
            draw.text((3, y + 20), f"{mode} | repeat {repeat}", fill="black")
            for row in rows:
                if row["repeat"] != repeat or row["mode"] != mode:
                    continue
                x = 220 + row["target"] * 220
                label = f"d={row['end_distance']:.4f} progress={row['distance_improvement']:.4f}" if row["execution_valid"] else "EXECUTION ERROR (retained)"
                draw.text((x, y + 3), label, fill="black")
                path = Path(row["artifact_dir"]) / "trajectory.npz"
                if path.is_file():
                    # An invalid attempt's raw last image is a diagnostic only;
                    # previewing it must not revalidate or discard its failure.
                    try:
                        with np.load(path, allow_pickle=False) as saved:
                            image = saved["image"][-1].copy()
                        if image.shape == (64, 64, 3) and image.dtype == np.uint8:
                            panel.paste(Image.fromarray(image).resize((128, 128)), (x, y + 20))
                    except (ValueError, KeyError, IndexError):
                        draw.text((x, y + 40), "raw image unavailable", fill="black")
        panel.save(report.directory / f"outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    from PIL import Image
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_continuation_control as confirmation

    baseline.video_preflight(report)
    cells = plan["design"]["schedule"]
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, len(cells) * 20 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.data["attempted_env_steps"] = 0
    report.check("storage_preflight", "PASS", "预留30次完整真实历史/视频空间；不复制模型或旧轨迹")
    report.check("independent_starts", "PASS", "各局fresh reset并保留自身32noop+32前缀；直接按固定mode条件执行，不比较或筛选跨局起点")
    rows, frames, balance = [], [], []
    first_start = None
    for index, cell in enumerate(cells):
        report.require_writable()
        directory = report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['mode']}"
        row = run_trial(report, runtime, plan, bank, cell, directory)
        report.data["worker_control_actions"] += row["actual_steps"]
        row.update(saved_execution_verified=False, video_verified=False)
        if row["execution_valid"]:
            try:
                measurements = verify_saved_trial(report, runtime, plan, bank, row)
                frames.extend(dict(trial=index, target=cell["target"], mode=cell["mode"], repeat=cell["repeat"],
                    **{key: value for key, value in item.items() if key != "goal_feature"}) for item in measurements)
            except OSError:
                raise
            except Exception as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}")
                baseline.write_json(directory / "saved_execution_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
                report.check(f"trial_{index}_saved_execution", "FAIL", row["error"] + "；保留尝试，不补样")
        if row["execution_valid"] and (directory / "start.json").is_file() and (directory / "trajectory.npz").is_file():
            trajectory = ctl.read_trajectory(directory)
            first = plan["control_start_frame"]
            if len(trajectory["image"]) > first:
                if first_start is None:
                    first_start = trajectory
                rgb = np.abs(trajectory["image"][first].astype(np.float32) - first_start["image"][first].astype(np.float32))
                balance.append(dict(trial=index, target=cell["target"], mode=cell["mode"], repeat=cell["repeat"],
                    rgb_mae_to_first=float(rgb.mean()), rgb_p99_to_first=float(np.percentile(rgb, 99)),
                    heatmap_mae_to_first=float(np.abs(trajectory["heatmap"][first].astype(np.float32) -
                        first_start["heatmap"][first].astype(np.float32)).mean()),
                    pose_difference_to_first=ctl.pose_difference(trajectory["telemetry"][first]["pose"], first_start["telemetry"][first]["pose"]),
                    rssm_relative_l2_to_first=float(np.linalg.norm(trajectory["features"][first] - first_start["features"][first]) /
                        max(float(np.linalg.norm(first_start["features"][first])), 1e-8)), diagnostic_only=True))
                pair = Image.new("RGB", (256, 128), "white")
                pair.paste(Image.fromarray(trajectory["image"][first]).resize((128, 128)), (0, 0))
                pair.paste(Image.fromarray(trajectory["image"][-1]).resize((128, 128)), (128, 0))
                pair.save(directory / "start_and_outcome.png")
        rows.append(row)
        baseline.write_json(directory / "metrics.json", row)
        t04.append_json(report.directory / "trials.jsonl", row)
        summary = confirmation.summarize(rows, plan)
        summary["start_differences_diagnostic_only"] = balance
        initial_balance = dict(diagnostic_only=True, used_for_selection=False,
            reference_trial=balance[0]["trial"] if balance else None, attempted_trials=len(rows),
            rows=balance, missing_trials=[r["trial_index"] for r in rows
                if r["trial_index"] not in {b["trial"] for b in balance}],
            interpretation="起点差异只诊断；缺失明确列出，所有尝试保留，不改变分组或补样")
        baseline.write_json(report.directory / "initial_balance.json", initial_balance)
        baseline.write_json(report.directory / "diagnostics.json", summary)
        baseline.write_json(report.directory / "gate.json", summary["numerical_gate"])
        baseline.write_json(report.directory / "trials.json", rows)
        report.data.update(attempted_trials=len(rows), numerical_gate=summary["numerical_gate"])
        report.save()
        report.require_writable()
        detail = f"steps={row['actual_steps']} d={row['end_distance']:.5f} progress={row['distance_improvement']:.5f}" if row["execution_valid"] else row.get("error", "execution error")
        print(f"[CONFIRM 350 {index + 1}/30] repeat={cell['repeat']} target={cell['target']} mode={cell['mode']} {detail}", flush=True)
    if frames:
        baseline.write_csv(report.directory / "frame_metrics.csv", frames)
    else:
        # No measurement is invented when every attempt failed.
        baseline.write_text_atomic(report.directory / "frame_metrics.csv", "trial,target,mode,repeat,frame,control_step\n")
    baseline.write_csv(report.directory / "trials.csv", [{name: row.get(name) for name in (
        "trial_index", "worker_version", "source_worker_version", "additional_updates",
        "seed", "repeat", "target", "mode", "execution_valid", "saved_execution_verified", "video_verified",
        "actual_steps", "fixed_horizon_completed", "actual_endpoint_frame", "returned_env_steps", "end_reason", "task_success",
        "worker_action_queries", "probability_roundtrip_queries", "measurement_roundtrip_frames",
        "maximum_probability_error", "terminal_action_names", "last_step_progress", "intervention_actions",
        "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "preference_improvement",
        "error", "artifact_dir")} for row in rows])
    outcome_gallery(report, plan, bank, rows)
    # Five modest six-trial strips avoid an unreadable 7680-pixel thumbnail.
    for repeat in range(5):
        panel = Image.new("RGB", (1536, 150), "white")
        for index, row in enumerate(item for item in rows if item["repeat"] == repeat):
            path = Path(row["artifact_dir"]) / "start_and_outcome.png"
            if path.is_file():
                with Image.open(path) as pair:
                    panel.paste(pair, (index * 256, 22))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(panel)
        for index, row in enumerate(item for item in rows if item["repeat"] == repeat):
            draw.text((index * 256 + 2, 3), f"target{row['target']} {row['mode']} | own start / end", fill="black")
        panel.save(report.directory / f"starts_and_outcomes_repeat_{repeat}.png")
    saved = dict(format=confirmation.EVALUATION_FORMAT, comparison_protocol=confirmation.PROTOCOL,
        design=plan, design_id=plan["design_id"], rows=rows, input_identity=report.data["input_identity"],
        new_env_steps=report.data["new_env_steps"], attempted_env_steps=report.data["attempted_env_steps"],
        worker_version=350, source_worker_version=300, additional_updates=50,
        worker_control_actions=report.data["worker_control_actions"],
        worker_action_queries=report.data["worker_action_queries"],
        probability_roundtrip_queries=report.data["probability_roundtrip_queries"],
        measurement_roundtrip_frames=report.data["measurement_roundtrip_frames"],
        maximum_probability_error=report.data["maximum_probability_error"],
        intervention_actions=0, optimizer_updates=0, full_design_completed=len(rows) == 30,
        check_report_sha256=bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
        artifacts={str(path.relative_to(report.directory)): bc.file_hash(path)
                   for row in rows for path in Path(row["artifact_dir"]).iterdir() if path.is_file()},
        summary_artifacts={name: bc.file_hash(report.directory / name) for name in
            ("design.json", "targets.npz", "trials.json", "trials.jsonl", "trials.csv", "frame_metrics.csv",
             "diagnostics.json", "gate.json", "initial_balance.json",
             *[f"outcomes_repeat_{i}.png" for i in range(5)],
             *[f"starts_and_outcomes_repeat_{i}.png" for i in range(5)])},
        behavior_accepted=False, t06_approved=False)
    saved["evaluation_id"] = gl.tensor_digest({}, saved)
    baseline.write_json(report.directory / "evaluation_manifest.json", saved)
    valid = sum(row["execution_valid"] for row in rows)
    gl.require(report.data["new_env_steps"] <= plan["maximum_new_env_steps"], "真实执行超过预声明预算")
    report.data.update(evaluation_id=saved["evaluation_id"], execution_valid_trials=valid,
                       task_success_evaluated=True, full_design_completed=True)
    report.check("planned_attempts", "PASS", "按固定新随机顺序尝试30/30，全部保留，无重试、补样、best终点或失败续跑")
    report.check("real_control_interface", "PASS" if valid == 30 else "FAIL", f"完整保存、真实动作/遥测/概率及视频有效{valid}/30")
    actions = sum(row["actual_steps"] for row in rows)
    gl.require(report.data["worker_control_actions"] == actions and
        report.data["worker_action_queries"] == sum(row["worker_action_queries"] for row in rows) and
        report.data["new_env_steps"] == sum(row["returned_env_steps"] for row in rows) and
        report.data["intervention_actions"] == report.data["optimizer_updates"] == 0,
        "真实查询/动作/环境步或干预计数不同")
    if valid == 30:
        gl.require(report.data["new_env_steps"] == 30 * 64 + actions and
            report.data["worker_action_queries"] == report.data["probability_roundtrip_queries"] == actions and
            report.data["measurement_roundtrip_frames"] == actions + 30,
            "完整真实历史/概率/测量计数不同")
    report.check("execution_counters", "PASS" if valid == 30 else "FAIL",
        f"env.step={report.data['new_env_steps']}，查询={report.data['worker_action_queries']}，"
        f"执行={actions}，概率重现={report.data['probability_roundtrip_queries']}，"
        f"逐帧测量={report.data['measurement_roundtrip_frames']}；干预/更新0，缺测不补样")
    gate = summary["numerical_gate"]
    print(f"[NUMERICAL GATE] passed={int(gate['passed'])}; human_review_pending=1; t06_approved=0", flush=True)
    for target, values in gate["by_target"].items():
        print(f"[TARGET {target}] direction={values['positive_progress_and_correct_preference']}/5 "
              f"mean_progress={values['mean_progress']} advantages={values['progress_advantages']}", flush=True)
    report.check("numerical_behavior_gate", "WARN", "数值门槛" + ("达到" if gate["passed"] else "未达到") +
                 "；逐目标结果见gate.json；该结论与工程PASS分别报告")
    report.check("behavior_acceptance", "WARN", "须结合全部视频、位置/朝向和起点平衡人工核对；仅当前世界局部目标，T06未自动批准")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        command.add_argument("--repair-verify-dir", required=True,
            help="本次续段修复独立verify；自动绑定其固定50步生产latest、check和真实来源")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        context, protected, files = source_paths(args)
        directory = baseline.project_path(args.output_dir)
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于模型、依赖、缓存、回放、旧参考和check")
        directory.mkdir(parents=True, exist_ok=False)
    except Exception as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    source = context["raw_source"]
    report.data.update(new_env_steps=0, evaluation_env_steps=0, worker_control_actions=0,
                       behavior_accepted=False, t06_approved=False)
    os.chdir(ROOT)
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        else:
            gl.require(args.command != "evaluate", "真实MineCLIP路径需要CUDA")
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化模型变化")
        for episode in source["episodes"]:
            gl.require(baseline.file_signature(Path(episode["path"])) == episode["signature"], "原真实回放变化")
        rng = bc.capture_rng(args.device)
        cache, runtime, plan, bank = load_inputs(args, report, context)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            if args.command == "check":
                check(args, report, cache, runtime, plan, bank, context)
            else:
                evaluate(args, report, cache, runtime, plan, bank)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()),
                   "确认改变冻结模型或产生梯度")
        report.check("no_training", "PASS", f"optimizer_updates=0；冻结worker/底座/WM/目标内容未变；"
                     f"worker查询={report.data['worker_action_queries']}，执行={report.data['worker_control_actions']}，"
                     f"干预=0，新env.step={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                             "源模型/依赖/回放/目标/验收/check与旧代码大小及修改时间未变；旧FAIL不改写")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
