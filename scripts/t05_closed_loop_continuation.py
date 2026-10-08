"""T05 real visual correction/holding continuations after autonomous16.

Ten fresh development histories, original16 plus separately budgeted4.
Current visual feedback selects at most one turn_down, otherwise noop.
No training, expert labels, retries, start filtering or T06 approval.
"""

import argparse
import copy
from datetime import datetime
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
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_random_control as pilot
import t05_repaired_control as confirmed
import t05_terminal_intervention as prior


class Report(pilot.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_CLOSED_LOOP_CONTINUATION_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(prior.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (
        "goal_closed_loop_continuation.py", "scripts/t05_closed_loop_continuation.py")})


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_terminal_intervention as old

    directory = baseline.project_path(args.intervention_dir)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "evaluate")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    plan = manifest["design"]
    old.validate_plan(plan)
    unsigned = dict(manifest)
    unsigned.pop("evaluation_id")
    gl.require(manifest["evaluation_id"] == gl.tensor_digest({}, unsigned) and
        manifest["evaluation_id"] == record["evaluation_id"] and
        manifest["format"] == old.EVALUATION_FORMAT and
        manifest["design_id"] == record["design_id"] == plan["design_id"] and
        record["evaluation_code"] == plan["evaluation_code"] == prior.code_identity() and
        record["input_identity"] == manifest["input_identity"] == plan["input_identity"] and
        record.get("execution_valid_trials") == 10 and record.get("completed_trial_records") == 10 and
        record.get("full_design_completed") is True and manifest["full_design_completed"] is True and
        record.get("optimizer_updates") == 0 and record.get("new_env_steps") == 800 and
        record.get("worker_action_queries") == 160 and record.get("worker_control_actions") == 155 and
        record.get("intervention_actions") == 5 and record.get("probability_roundtrip_queries") == 160 and
        record.get("measurement_roundtrip_frames") == 170 and
        record.get("behavior_accepted") is False and record.get("t06_approved") is False and
        confirmed.passed_checks(record, ("historical_identity", "design_identity", "fixed_visual_targets",
            "real_control_interface", "execution_counters", "fixed_horizon_endpoints", "no_training", "source_inputs_unchanged")),
        "需要完整通过的10局真实末步干预及原代码，失败/子集不能用于续段采集")
    source, protected, files = prior.historical_inputs(SimpleNamespace(command="check",
        diagnosis_dir=record["arguments"]["diagnosis_dir"]))
    gl.require(plan["source_evaluation_id"] == source["manifest"]["evaluation_id"] and
        plan["source_diagnosis_sha256"] == source["diagnosis_hashes"] and
        plan["original_numerical_gate"] == source["record"]["numerical_gate"], "干预与原失败/诊断身份不同")
    rows = legacy.read_json(directory / "trials.json")
    summary = old.summarize(rows, plan)
    gl.require(rows == manifest["rows"] and summary["valid_histories"] == 10 and
               all(r["fixed_horizon_completed"] for r in rows), "完整十局真实记录不同")
    checked = baseline.project_path(record["arguments"]["check_dir"])
    checked_record = legacy.read_json(checked / "report.json")
    legacy.accepted(checked_record, "check")
    gl.require(bc.file_hash(checked / "report.json") == manifest["check_report_sha256"] and
               legacy.read_json(checked / "design.json") == plan, "原真实干预check身份不同")
    files += [directory / "report.json", directory / "evaluation_manifest.json"]
    for group in (manifest["artifacts"], manifest["summary_artifacts"]):
        for name, digest in group.items():
            path = (directory / name).resolve()
            gl.require(directory.resolve() in path.parents and path.is_file() and bc.file_hash(path) == digest,
                       f"原干预真实产物SHA256不同: {name}")
            files.append(path)
    for index, row in enumerate(rows):
        trial = Path(row["artifact_dir"]).resolve()
        gl.require(directory.resolve() in trial.parents and row["trial_index"] == index and
                   row["artifact_dir"] == manifest["rows"][index]["artifact_dir"], "原trial路径或顺序不同")
        for name in ("trajectory.npz", "events.json", "control_trace.json", "history_check.json",
                     "start.json", "metrics.json", "frame_metrics.json", "video.mp4"):
            gl.require(str((trial / name).relative_to(directory.resolve())) in manifest["artifacts"],
                       "原完整真实轨迹/事件/视频没有SHA256绑定")
    files += [checked / name for name in ("report.json", "design.json", "targets.npz")]
    protected += [directory, checked]
    if args.command == "evaluate":
        current = baseline.project_path(args.check_dir)
        protected.append(current)
        files += [current / name for name in ("report.json", "design.json", "targets.npz")]
    files += [ROOT / name for name in code_identity()]
    source.update(intervention_record=record, intervention_manifest=manifest, intervention_dir=directory)
    return source, protected, list(dict.fromkeys(path.resolve() for path in files))


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_closed_loop_continuation as intervention

    loader = SimpleNamespace(**source["record"]["arguments"])
    loader.command = "check"
    gl.require(args.device == loader.device, "必须使用原CUDA设备，不切换推理后端")
    cache, runtime, previous, bank = confirmed.load_inputs(loader, report)
    gl.require(previous == source["manifest"]["design"], "必须保持原worker300/目标/设备/初始化流程")
    fields = ("input_identity", "scenario", "visual_preprocessing", "action_names", "warmup_steps", "prefix_actions",
              "start_actions", "control_start_frame", "horizon", "worker_version", "repair_updates", "target_content_id", "target_telemetry")
    plan = {name: copy.deepcopy(previous[name]) for name in fields}
    plan.update(format=intervention.FORMAT, comparison_protocol=intervention.PROTOCOL, evaluation_code=code_identity(),
        environment_fingerprint=previous["environment_fingerprint"], focus_target=1, continuation_start_frame=80,
        continuation_horizon=4, actual_endpoint_frame=84, noop_action=previous["action_names"].index("noop"),
        turn_down_action=previous["action_names"].index("turn_down"), maximum_new_env_steps=840,
        feedback_rule=dict(input="current_real_visual_distances_only", margin_epsilon=intervention.MARGIN_EPSILON,
            correction="turn_down", correction_limit=1, hold="noop", stop_on_success=False),
        episode_split={str(i): intervention.split(i) for i in range(5)}, same_hidden_state=False, matched_start_comparison=False,
        behavior_accepted=False, t06_approved=False, original_numerical_gate=copy.deepcopy(source["record"]["numerical_gate"]),
        source_evaluation_id=source["manifest"]["evaluation_id"], source_design_id=previous["design_id"],
        source_intervention_id=source["intervention_manifest"]["evaluation_id"],
        source_intervention_report_sha256=bc.file_hash(source["intervention_dir"] / "report.json"),
        design=dict(seed=0, repetitions=5, randomization_seed=3, conditions=list(intervention.CONDITIONS),
                    execution_policy="mode", schedule=intervention.schedule()))
    intervention.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(comparison_protocol=intervention.PROTOCOL, evaluation_code=code_identity(), design_id=plan["design_id"],
        source_evaluation_id=plan["source_evaluation_id"], source_intervention_id=plan["source_intervention_id"],
        original_numerical_gate=plan["original_numerical_gate"], planned_trials=10, maximum_new_env_steps=840,
        randomization_seed=3, worker_action_queries=0, worker_control_actions=0, intervention_actions=0,
        correction_actions=0, actual_action_changes=0, control_actions=0, attempted_env_steps=0,
        behavior_accepted=False, t06_approved=False, labels_generated=False, expert_labels_generated=False)
    if args.command == "evaluate":
        directory = baseline.project_path(args.check_dir)
        checked = legacy.read_json(directory / "report.json")
        legacy.accepted(checked, "check")
        gl.require(legacy.read_json(directory / "design.json") == plan and checked.get("design_id") == plan["design_id"] and
            checked.get("evaluation_code") == code_identity() and checked.get("input_identity") == plan["input_identity"] and
            checked.get("targets_sha256") == bc.file_hash(directory / "targets.npz") and
            confirmed.passed_checks(checked, ("historical_identity", "trial_output_preflight", "causal_execution_interface",
                "worker_distribution_contract", "continuation_contracts", "predeclared_continuation", "no_training", "source_inputs_unchanged")),
            "需要与当前续段设计及代码完全一致的独立check")
        with np.load(directory / "targets.npz", allow_pickle=False) as saved:
            gl.require(set(saved.files) == set(bank) and all(np.array_equal(saved[n], bank[n]) for n in bank), "固定目标被修改")
        report.check("design_identity", "PASS", "原16步+新4步、视觉规则、两组各5局、整局划分/预算/代码与check一致")
    report.check("visual_feedback_contract", "PASS", "只用当前真实RGB/heatmap编码距离决定最多一次turn_down；其余noop；坐标/朝向仅诊断")
    return cache, runtime, plan, bank


def check(args, report, cache, runtime, plan, bank):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control
    import goal_closed_loop_continuation as intervention

    directory = pilot.prepare_trial_directory(report.directory / "trial_output_probe")
    marker = dict(output_probe_only=True, before_environment=True)
    baseline.write_json(directory / "probe.json", marker)
    gl.require(legacy.read_json(directory / "probe.json") == marker, "原子JSON读写不同")
    try:
        pilot.prepare_trial_directory(directory)
    except FileExistsError:
        pass
    else:
        raise ValueError("重复trial目录没有被拒绝")
    report.check("trial_output_preflight", "PASS", "启动环境前创建独立目录并验证原子写入；重复目录拒绝覆盖")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "原真实缓存/历史身份不同")
    rng = bc.capture_rng(args.device)
    for index in sorted({0, len(dataset.metadata["episodes"]) - 1}):
        episode = dataset.episode(index)
        stop = min(32, len(episode["image"]) - 1)
        expected = runtime.state_encoder.rollout(episode, stop)
        state, recovered = None, []
        for frame in range(stop + 1):
            obs = {name: episode[name][frame] for name in ctl.OBS_KEYS}
            state, feature = runtime.state_encoder.step(obs, episode["action"][frame], state)
            recovered.append(feature[0])
        t02.vector_difference(report, f"incremental_episode_{index}", torch.stack(recovered), expected, check=True)
        reset = {name: episode[name][0] for name in ctl.OBS_KEYS}
        _, restarted = runtime.state_encoder.step(reset, episode["action"][0], state)
        t02.vector_difference(report, f"reset_episode_{index}", restarted, expected[:1], check=True)
        t02.rejection(report, "missing_history_guard", lambda: runtime.state_encoder.step(dict(reset, is_first=False), episode["action"][1]), "本局历史")
        t02.rejection(report, "reset_action_guard", lambda: runtime.state_encoder.step(reset, episode["action"][1]), "必须为零")
        t02.rejection(report, "after_terminal_guard", lambda: runtime.step(dict(reset, is_last=True), episode["action"][0], bank["goals"][1], 1), "结束")
    features, _, _, _ = cache.worker_batch(cache.worker_rows["validation"][:1], args.device)
    for budget in (16, 4, 1):
        control.probabilities(runtime, features, bank["goals"], budget, 1, "goal")
    for invalid in (0, 17, 1.5):
        t02.rejection(report, "worker_remaining_guard", lambda invalid=invalid: control.probabilities(runtime, features, bank["goals"], invalid, 1, "goal"), "remaining")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "接口检查推进RNG")
    report.check("causal_execution_interface", "PASS", "两局真实历史增量/完整前缀、reset/incoming/结束守卫通过；不启动环境")
    report.check("worker_distribution_contract", "PASS", "原16步及新4步remaining16、4、1的冻结worker分布有效")
    contract_checks(report, plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    baseline.write_json(report.directory / "design.json", plan)
    pilot.target_gallery(report.directory / "targets.png", bank["images"])
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("predeclared_continuation", "PASS", "两组各5局；32noop+32前缀+16自主+4续段，最多840步；repeat0–2训练/3–4开发留出")
    report.check("scope", "WARN", "仅验新采集接口；纠偏/保持效果待真实执行；不构造优化器、训练标签或批准T06")


def contract_checks(report, plan):
    import numpy as np
    import goal_library as gl
    import goal_closed_loop_continuation as protocol

    original = copy.deepcopy(plan)
    frames, dimension = 85, 12
    p = np.eye(12)[2]
    telemetry = dict(pose=dict(x=0., y=64., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    for index, cell in enumerate(plan["design"]["schedule"]):
        used, decisions = False, []
        distances = [[.02, .5]] * 17 + [[.5, .02]] * 4
        for offset in range(20):
            item = protocol.decide(p, offset, cell["condition"], distances[offset], used, plan)
            decisions.append(item)
            used = used or item["correction_applied"]
        actual = [item["actual_action"] for item in decisions]
        arrays = dict(image=np.zeros((frames, 64, 64, 3), np.uint8), heatmap=np.zeros((frames, 64, 64), np.uint8),
            features=np.zeros((frames, 5120), np.float32), obs_reward=np.zeros((frames, 1), np.float32),
            action=np.concatenate((np.zeros((1, 12), np.float32), np.eye(12, dtype=np.float32)[plan["start_actions"] + actual])),
            is_first=np.arange(frames) == 0, is_last=np.zeros(frames, bool), is_terminal=np.zeros(frames, bool))
        events = [dict(frame=f, error=None, done=False, reward=0., telemetry=copy.deepcopy(telemetry),
                       native_actions=[] if f == 0 else [np.zeros(8, np.int64)]) for f in range(frames)]
        seed = protocol.action_seed(plan, index)
        trace = dict(comparison_protocol=protocol.PROTOCOL, design_id=plan["design_id"], model_id=plan["input_identity"]["model_id"],
            worker_version=300, repair_updates=100, condition=cell["condition"], issued_target=1,
            control_start_frame=64, continuation_start_frame=80, execution_policy="mode", episode_split=protocol.split(cell["repeat"]),
            action_ids=actual, worker_action_ids=[item["worker_action"] for item in decisions],
            probabilities=[p.tolist()] * 20, remaining=protocol.BUDGETS.copy(), action_seed=seed,
            uniforms=np.random.RandomState(seed).uniform(size=20).tolist(), distances_to_both_goals=distances)
        for field, saved in (("decision_source", "decision_sources"), ("phase", "phases"),
            ("intervention_applied", "intervention_applied"), ("action_differs_from_worker", "action_differs_from_worker"),
            ("correction_used_before", "correction_used_before"), ("correction_applied", "correction_applied")):
            trace[saved] = [item[field] for item in decisions]
        gl.require(protocol.validate_trace(arrays, events, trace, plan, index) == 20, "内存接口不一致")
    def reject_trace(name, change, message):
        a, e, t = copy.deepcopy(arrays), copy.deepcopy(events), copy.deepcopy(trace)
        change(a, e, t)
        t02.rejection(report, name, lambda: protocol.validate_trace(a, e, t, plan, index), message)
    reject_trace("incoming_guard", lambda a, e, t: a["action"].__setitem__(81, np.eye(12)[0]), "incoming")
    reject_trace("missing_native_guard", lambda a, e, t: e[81].update(native_actions=[]), "原生事件")
    reject_trace("missing_telemetry_guard", lambda a, e, t: e[81].update(telemetry={}), "遥测")
    reject_trace("fake_reset_guard", lambda a, e, t: a["is_first"].__setitem__(81, True), "reset")
    reject_trace("saved_terminal_guard", lambda a, e, t: a["is_last"].__setitem__(81, True), "结束")
    reject_trace("remaining_guard", lambda a, e, t: t["remaining"].__setitem__(16, 1), "remaining")
    reject_trace("worker_suggestion_guard", lambda a, e, t: t["worker_action_ids"].__setitem__(0, 0), "mode")
    reject_trace("probability_guard", lambda a, e, t: t["probabilities"].__setitem__(0, [0.] * 12), "概率")
    reject_trace("feedback_source_guard", lambda a, e, t: t["decision_sources"].__setitem__(16, "expert"), "来源")
    reject_trace("action_seed_guard", lambda a, e, t: t.update(action_seed=t["action_seed"] + 1), "随机数")
    reject_trace("action_uniform_guard", lambda a, e, t: t["uniforms"].__setitem__(0, 0.), "随机数")
    reject_trace("split_guard", lambda a, e, t: t.update(episode_split="train"), "划分")
    early = {name: value[:81].copy() for name, value in arrays.items()}
    early["is_last"][-1] = early["is_terminal"][-1] = True
    early_events = copy.deepcopy(events[:81])
    early_events[-1]["done"] = True
    early_trace = copy.deepcopy(trace)
    for name in ("action_ids", "worker_action_ids", "probabilities", "remaining", "uniforms", "decision_sources", "phases",
                 "intervention_applied", "action_differs_from_worker", "correction_used_before", "correction_applied"):
        early_trace[name] = early_trace[name][:16]
    early_trace["distances_to_both_goals"] = early_trace["distances_to_both_goals"][:17]
    gl.require(protocol.validate_trace(early, early_events, early_trace, plan, index) == 16, "真实提前结束不能单列")
    early["is_last"][-1] = early["is_terminal"][-1] = False
    t02.rejection(report, "incomplete_budget_guard", lambda: protocol.validate_trace(early, early_events, early_trace, plan, index), "remaining")
    for kind, used, action in (([.5, .02], False, 0), ([.02, .5], False, plan["turn_down_action"]),
                              ([.02, .5], True, 0), ([.5, .5], False, 0)):
        item = protocol.decide(p, 16, "visual_correct_hold", kind, used, plan)
        gl.require(item["actual_action"] == action, "纠偏/保持规则与预声明不同")
    rows = [dict(cell, trial_index=i, episode_split=protocol.split(cell["repeat"]), execution_valid=False)
            for i, cell in enumerate(plan["design"]["schedule"])]
    gl.require(protocol.summarize(rows, plan)["valid_histories"] == 0 and
               protocol.summarize(rows, plan)["groups"]["visual_correct_hold"]["continuation_mean_all_five"] is None,
               "缺测不能形成完整均值")
    t02.rejection(report, "duplicate_trial_guard", lambda: protocol.summarize(rows + rows[:1], plan), "重复")
    full_rows = [dict(cell, trial_index=i, episode_split=protocol.split(cell["repeat"]), execution_valid=True,
        saved_execution_verified=True, video_verified=True, actual_steps=20, returned_env_steps=84, returned_control_steps=20,
        worker_action_queries=20, fixed_horizon_completed=True, actual_terminal=False, end_reason="fixed_budget",
        autonomous_steps=16, continuation_steps=4,
        intervention_actions=4 if cell["condition"] == "visual_correct_hold" else 0,
        correction_actions=1 if cell["condition"] == "visual_correct_hold" else 0,
        actual_action_changes=4 if cell["condition"] == "visual_correct_hold" else 0,
        start_distances_to_both_goals=[.1, .7], boundary_distances_to_both_goals=[.02, .5],
        end_distances_to_both_goals=[.5, .02], start_distance=.7, end_distance=.02, distance_improvement=.68,
        target_preference_margin=.48, boundary_kind="closer_target0", continuation_progress=.48, autonomous_progress=.2)
        for i, cell in enumerate(plan["design"]["schedule"])]
    summary = protocol.summarize(full_rows, plan)
    gl.require(summary["valid_histories"] == 10 and
        np.isclose(summary["groups"]["visual_correct_hold"]["continuation_mean_all_five"], .48), "完整终点自身汇总不同")
    wrong = copy.deepcopy(full_rows)
    wrong[0]["continuation_progress"] += .1
    t02.rejection(report, "own_progress_guard", lambda: protocol.summarize(wrong, plan), "进展")
    wrong = copy.deepcopy(full_rows)
    wrong[0]["episode_split"] = "development_holdout"
    t02.rejection(report, "episode_split_guard", lambda: protocol.summarize(wrong, plan), "划分")
    # Failed correction still holds after one attempt, even if every later picture stays closer to target0.
    used, total = False, 0
    for offset in range(16, 20):
        item = protocol.decide(p, offset, "visual_correct_hold", [.02, .5], used, plan)
        used = used or item["correction_applied"]
        total += item["correction_applied"]
    gl.require(total == 1, "失败纠偏不能重复turn_down超过一次")
    changed = copy.deepcopy(plan)
    changed["continuation_horizon"] = 5
    t02.rejection(report, "schedule_guard", lambda: protocol.validate_plan(changed), "计划")
    gl.require(plan == original, "内存验收改变源计划")
    report.check("continuation_contracts", "PASS", "预算16–1/4–1、真实incoming/结束/反馈次数/种子/整局划分与缺测守卫通过；内存标记不写成轨迹")

def run_trial(report, runtime, plan, bank, cell, index):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl
    import goal_random_control as prefix
    import goal_residual_control as control
    import goal_closed_loop_continuation as intervention

    directory = pilot.prepare_trial_directory(report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_1_{cell['condition']}")
    first, seed = 64, intervention.action_seed(plan, index)
    uniforms = np.random.RandomState(seed).uniform(size=20)
    row = dict(cell, trial_index=index, artifact_dir=str(directory), execution_valid=False, saved_execution_verified=False,
        video_verified=False, actual_steps=0, fixed_horizon_completed=False, episode_split=intervention.split(cell["repeat"]),
        same_hidden_state=False, matched_start_comparison=False)
    trace = dict(comparison_protocol=intervention.PROTOCOL, design_id=plan["design_id"], model_id=runtime.policy.model_id,
        worker_version=300, repair_updates=100, condition=cell["condition"], issued_target=1, control_start_frame=64,
        continuation_start_frame=80, episode_split=row["episode_split"], execution_policy="mode", action_seed=seed,
        action_ids=[], worker_action_ids=[], probabilities=[], remaining=[], uniforms=[], phases=[],
        decision_sources=[], intervention_applied=[], action_differs_from_worker=[], distances_to_both_goals=[],
        correction_used_before=[], correction_applied=[])
    session, fatal = None, None
    returned_before = report.data["new_env_steps"]
    queries_before = report.data["worker_action_queries"]
    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
    def execute(action):
        gl.require(report.data["attempted_env_steps"] < 840, "真实执行尝试超过预声明840步")
        report.data["attempted_env_steps"] += 1
        session.step(action)
    try:
        print(f"[ENV] continuation repeat={cell['repeat']} condition={cell['condition']} target=1 "
              f"world_seed={plan['scenario']['world_seed']}; fresh reset", flush=True)
        session = ctl.Session(runtime, plan["scenario"], directory / "environment", on_step)
        gl.require(session.specs["visual_preprocessing"] == plan["visual_preprocessing"], "当前视觉处理与固定计划不同")
        for action in plan["start_actions"]:
            execute(action)
            gl.require(not session.done, "真实环境在初始化完成前结束；保留失败，不补跑")
        prefix.validate_prefix(session.arrays(), session.events, plan["start_actions"], 12)
        initial = ctl.encode_goal(runtime, session.observation)
        distances = ctl.distances(initial[None], bank["goals"])[0]
        trace["distances_to_both_goals"].append(distances.tolist())
        row["start_telemetry"] = copy.deepcopy(session.events[-1]["telemetry"])
        baseline.write_json(directory / "start.json", dict(control_start_frame=64, telemetry=row["start_telemetry"],
            initial_goal_feature=initial.tolist(), distances_to_both_goals=distances.tolist(),
            rssm_feature=session.features[0].cpu().tolist(), own_real_history_only=True, comparison_to_reference_required=False,
            condition_predeclared=True, no_state_or_distance_filter=True))
        correction_used = False
        for offset, remaining in enumerate(intervention.BUDGETS):
            p = control.probabilities(runtime, session.features, bank["goals"], remaining, 1, "goal")
            decision = intervention.decide(p, offset, cell["condition"], trace["distances_to_both_goals"][-1], correction_used, plan)
            correction_used = correction_used or decision["correction_applied"]
            report.data["worker_action_queries"] += 1
            trace["probabilities"].append(p.tolist())
            trace["worker_action_ids"].append(decision["worker_action"])
            trace["action_ids"].append(decision["actual_action"])
            trace["remaining"].append(remaining)
            trace["uniforms"].append(float(uniforms[offset]))
            for name, saved in (("decision_source", "decision_sources"), ("phase", "phases"),
                ("intervention_applied", "intervention_applied"), ("action_differs_from_worker", "action_differs_from_worker"),
                ("correction_used_before", "correction_used_before"), ("correction_applied", "correction_applied")):
                trace[saved].append(decision[name])
            # The next observed state always receives the ACTUAL action.
            execute(decision["actual_action"])
            trace["distances_to_both_goals"].append(ctl.distances(ctl.encode_goal(runtime, session.observation)[None], bank["goals"])[0].tolist())
            if session.done:
                break
        intervention.validate_trace(session.arrays(), session.events, trace, plan, index)
        recovered = runtime.state_encoder.rollout(session.arrays(), len(session.rows) - 1).cpu().numpy()
        actual = session.arrays()["features"]
        maximum = float(np.max(np.abs(recovered - actual)))
        gl.require(np.allclose(recovered, actual, atol=3e-5, rtol=3e-5), "实际incoming的本局完整因果历史不一致")
        baseline.write_json(directory / "history_check.json", dict(own_causal_history_passed=True, reset_is_real=True,
            maximum_absolute_error=maximum, start_frame=64, total_frames=len(actual), comparison_to_other_histories=False))
        row.update(execution_valid=True, own_history_max_error=maximum)
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}")
        if isinstance(error, OSError):
            fatal = error
        baseline.write_json(directory / "error.json", dict(error=row["error"], traceback=traceback.format_exc()))
        report.check(f"trial_{index}_execution", "FAIL", row["error"] + "；保留尝试，不重试")
    finally:
        row["returned_env_steps"] = report.data["new_env_steps"] - returned_before
        row["worker_action_queries"] = report.data["worker_action_queries"] - queries_before
        if session is not None:
            row.update(actual_steps=max(0, len(session.rows) - first - 1), actual_terminal=bool(session.done),
                end_reason="environment_done" if session.done else "fixed_budget",
                task_success=any(event["success"] for event in session.events[first + 1:]),
                task_success_during_prefix=any(event["success"] for event in session.events[:first + 1]),
                external_return=float(sum(event["reward"] for event in session.events[first + 1:])))
            try:
                before = session.arrays()
                row["video"] = ctl.save_session(session, directory, baseline.write_video)
                saved = ctl.read_trajectory(directory)
                gl.require(set(saved) == set(before) and all(np.array_equal(saved[name], before[name])
                    for name in before if name != "telemetry") and
                    stats.native_actions_equal(saved["telemetry"], before["telemetry"]) and
                    stats.native_actions_equal(legacy.read_json(directory / "events.json"), session.events), "真实保存轨迹/事件与本局历史不同")
                row["saved_history_roundtrip"] = True
                if row["execution_valid"]:
                    history = legacy.read_json(directory / "history_check.json")
                    history["saved_history_roundtrip"] = True
                    baseline.write_json(directory / "history_check.json", history)
            except Exception as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}")
                report.check(f"trial_{index}_save", "FAIL", row["error"] + "；原始尝试保留")
                if isinstance(error, OSError):
                    fatal = error
                baseline.write_json(directory / "save_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
            finally:
                try:
                    session.close()
                except Exception as error:
                    row.update(execution_valid=False, error=f"{type(error).__name__}: {error}")
                    fatal = error  # Do not start another environment after a failed close.
                    baseline.write_json(directory / "close_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
                    report.check(f"trial_{index}_close", "FAIL", row["error"] + "；停止后续环境启动，保留当前尝试")
                del session
                gc.collect()
        # Returned steps count even if storing the new observation failed.
        count = max(0, row["returned_env_steps"] - first)
        row["returned_control_steps"] = count
        report.data["control_actions"] += count
        applied = max(0, count - 16) if cell["condition"] == "visual_correct_hold" else 0
        report.data["intervention_actions"] += applied
        report.data["worker_control_actions"] += count - applied
        report.data["actual_action_changes"] += sum(trace["action_differs_from_worker"][:count])
        report.data["correction_actions"] += sum(trace["correction_applied"][:count])
        baseline.write_json(directory / "metrics.json", row)
        baseline.write_json(directory / "control_trace.json", trace)
    row["stop_after_trial"] = fatal is not None
    return row


def verify_saved_trial(runtime, plan, bank, row):
    import numpy as np
    import torch
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl
    import goal_reference_calibration as calibration
    import goal_residual_control as control
    import goal_closed_loop_continuation as intervention

    directory = Path(row["artifact_dir"])
    arrays = ctl.read_trajectory(directory)
    events = legacy.read_json(directory / "events.json")
    trace = legacy.read_json(directory / "control_trace.json")
    count = intervention.validate_trace(arrays, events, trace, plan, row["trial_index"])
    history = legacy.read_json(directory / "history_check.json")
    start = legacy.read_json(directory / "start.json")
    gl.require(row["saved_history_roundtrip"] is True and history.get("saved_history_roundtrip") is True and
               history.get("own_causal_history_passed") is True and history.get("reset_is_real") is True and
               history.get("comparison_to_other_histories") is False and history.get("total_frames") == 65 + count and
               history.get("start_frame") == 64 and row["video"]["frames"] == 65 + count and
               row["actual_steps"] == count and row["returned_env_steps"] == 64 + count and
               start.get("control_start_frame") == 64 and start.get("own_real_history_only") is True and
               start.get("no_state_or_distance_filter") is True and
               stats.native_actions_equal(start["telemetry"], events[64]["telemetry"]) and
               np.array_equal(np.asarray(start["rssm_feature"], dtype=np.float32), arrays["features"][64]),
               "保存的真实因果历史、起点或完整解码视频不同")
    device = next(runtime.parameters()).device
    measurements, maximum_error, correction_used = [], 0., False
    for offset in range(count + 1):
        frame = 64 + offset
        feature = ctl.encode_goal(runtime, {name: arrays[name][frame] for name in ("image", "heatmap")})
        measured = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                       events[frame]["telemetry"], bank, plan["target_telemetry"])
        measurement = dict(frame=frame, control_step=offset, goal_feature=feature.tolist(),
            phase="autonomous16" if offset < 16 else "boundary80" if offset == 16 else "continuation4",
            query_remaining=None, actual_next_action=None, worker_next_action=None, decision_source=None,
            intervention_applied=None, correction_applied=None, **measured)
        measurements.append(measurement)
        if offset < count:
            state = torch.as_tensor(arrays["features"][frame:frame + 1].copy(), device=device)
            p = control.probabilities(runtime, state, bank["goals"], intervention.BUDGETS[offset], 1, "goal")
            maximum_error = max(maximum_error, float(np.max(np.abs(p - np.asarray(trace["probabilities"][offset])))))
            real_distances = ctl.distances(feature[None], bank["goals"])[0]
            decision = intervention.decide(p, offset, row["condition"], real_distances, correction_used, plan)
            correction_used = correction_used or decision["correction_applied"]
            gl.require(np.allclose(p, trace["probabilities"][offset], atol=3e-5, rtol=3e-5) and
                       decision["worker_action"] == trace["worker_action_ids"][offset] and
                       decision["actual_action"] == trace["action_ids"][offset] and
                       decision["decision_source"] == trace["decision_sources"][offset], "真实保存画面的反馈及worker概率/建议/实际动作不能重现")
            measurement.update(query_remaining=intervention.BUDGETS[offset], actual_next_action=decision["actual_action"],
                worker_next_action=decision["worker_action"], decision_source=decision["decision_source"],
                intervention_applied=decision["intervention_applied"], correction_applied=decision["correction_applied"])
    distances = np.asarray([[item["distance_0"], item["distance_1"]] for item in measurements])
    gl.require(np.allclose(distances, trace["distances_to_both_goals"], atol=3e-5, rtol=3e-5) and
               np.allclose(start["distances_to_both_goals"], distances[0], atol=3e-5, rtol=3e-5) and
               np.allclose(start["initial_goal_feature"], measurements[0]["goal_feature"], atol=3e-5, rtol=3e-5),
               "保存真实画面/自身起点的目标测量不一致")
    final = measurements[-1]
    row.update(saved_execution_verified=True, video_verified=True, fixed_horizon_completed=count == 20,
        start_distances_to_both_goals=distances[0].tolist(), end_distances_to_both_goals=distances[-1].tolist(),
        start_distance=float(distances[0, 1]), end_distance=float(distances[-1, 1]),
        distance_improvement=float(distances[0, 1] - distances[-1, 1]), target_preference_margin=float(distances[-1, 0] - distances[-1, 1]),
        start_telemetry=events[64]["telemetry"], end_telemetry=events[-1]["telemetry"],
        pose_errors_at_end={name: value for name, value in final.items() if "_error_" in name},
        maximum_probability_error=maximum_error, probability_roundtrip_queries=count,
        measurement_roundtrip_frames=count + 1, autonomous_steps=min(16, count), continuation_steps=max(0, count - 16),
        intervention_actions=sum(trace["intervention_applied"]), correction_actions=sum(trace["correction_applied"]),
        actual_action_changes=sum(trace["action_differs_from_worker"]), boundary_kind=None,
        continuation_progress=None, autonomous_progress=None, pitch_error_at_end=final["pitch_error_1"])
    if count >= 16:
        boundary = measurements[16]
        row.update(boundary_distances_to_both_goals=distances[16].tolist(), boundary_telemetry=events[80]["telemetry"],
            boundary_kind=intervention.boundary_kind(distances[16]), boundary_distance=float(distances[16, 1]),
            autonomous_progress=float(distances[0, 1] - distances[16, 1]),
            continuation_progress=float(distances[16, 1] - distances[-1, 1]), pitch_error_at_boundary=boundary["pitch_error_1"])
    if count == 20:
        corrections = [offset for offset, applied in enumerate(trace["correction_applied"]) if applied]
        row.update(correction_kept_target1_preference=None, hold_kept_target1_preference=None)
        if row["condition"] == "visual_correct_hold":
            row["hold_kept_target1_preference"] = bool(not corrections and row["boundary_kind"] == "closer_target1" and
                all(intervention.boundary_kind(d) == "closer_target1" for d in distances[16:]))
            row["correction_kept_target1_preference"] = bool(corrections and
                all(intervention.boundary_kind(d) == "closer_target1" for d in distances[corrections[0] + 1:]))
        # Every complete endpoint is saved, including failures, using its own real RGB/heatmap.
        np.savez_compressed(directory / "continuation_endpoint.npz", image=arrays["image"][84], heatmap=arrays["heatmap"][84],
                            goal_feature=np.asarray(final["goal_feature"], dtype=np.float32))
        with np.load(directory / "continuation_endpoint.npz", allow_pickle=False) as endpoint:
            gl.require(set(endpoint.files) == {"image", "heatmap", "goal_feature"} and
                np.array_equal(endpoint["image"], arrays["image"][84]) and np.array_equal(endpoint["heatmap"], arrays["heatmap"][84]) and
                np.array_equal(endpoint["goal_feature"], np.asarray(final["goal_feature"], dtype=np.float32)), "本局真实终点保存不同")
        row["continuation_endpoint_frame"] = 84
    baseline.write_json(directory / "frame_metrics.json", measurements)
    return measurements


def continuation_data(rows, plan):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_closed_loop_continuation as protocol

    protocol.summarize(rows, plan)
    episodes = []
    for row in rows:
        item = dict(trial_index=row["trial_index"], repeat=row["repeat"], condition=row["condition"],
            episode_split=row["episode_split"], artifact_dir=row["artifact_dir"], execution_valid=row["execution_valid"],
            complete_continuation=bool(row["execution_valid"] and row["fixed_horizon_completed"]), queries=[],
            requested_target=1, expert_labels=False, approved_for_training=False)
        if item["complete_continuation"]:
            directory = Path(row["artifact_dir"])
            arrays = ctl.read_trajectory(directory)
            events = legacy.read_json(directory / "events.json")
            trace = legacy.read_json(directory / "control_trace.json")
            gl.require(protocol.validate_trace(arrays, events, trace, plan, row["trial_index"]) == 20 and
                       row["continuation_endpoint_frame"] == 84, "实际续段状态/动作/终点不同")
            item.update(endpoint_frame=84, endpoint_artifact="continuation_endpoint.npz",
                endpoint_sha256=bc.file_hash(directory / "continuation_endpoint.npz"),
                trajectory_sha256=bc.file_hash(directory / "trajectory.npz"),
                supervision_goal="this_episode_actual_frame84_rgb_heatmap", fixed_target1_is_not_a_label=True,
                endpoint_closer_to_requested_target=bool(row["target_preference_margin"] > protocol.MARGIN_EPSILON))
            item["queries"] = [dict(state_frame=80 + offset, incoming_action_frame=81 + offset,
                remaining=4 - offset, actual_action=trace["action_ids"][16 + offset],
                decision_source=trace["decision_sources"][16 + offset]) for offset in range(4)]
            gl.require(all(query["remaining"] == 84 - query["state_frame"] and
                query["incoming_action_frame"] == query["state_frame"] + 1 and
                query["actual_action"] == int(arrays["action"][query["incoming_action_frame"]].argmax())
                for query in item["queries"]), "真实obs[t]→incoming[t+1]/本局终点预算不同")
            with np.load(directory / "continuation_endpoint.npz", allow_pickle=False) as endpoint:
                gl.require(np.array_equal(endpoint["image"], arrays["image"][84]) and
                    np.array_equal(endpoint["heatmap"], arrays["heatmap"][84]) and np.isfinite(endpoint["goal_feature"]).all() and
                    np.isclose(np.linalg.norm(endpoint["goal_feature"]), 1, atol=1e-5), "索引实际目标不是本局真实终点")
        episodes.append(item)
    return dict(format="ls_imagine_real_continuation_index_v1", design_id=plan["design_id"],
        input_identity=plan["input_identity"], planned_trials=10, attempted_trials=len(rows), episodes=episodes,
        complete_episodes=sum(item["complete_continuation"] for item in episodes),
        real_action_records=sum(len(item["queries"]) for item in episodes),
        split_action_counts={name: sum(len(item["queries"]) for item in episodes if item["episode_split"] == name)
                             for name in ("train", "development_holdout")},
        expert_labels_generated=False, approved_for_training=False, optimizer_updates=0, t06_approved=False,
        interpretation="全部实际终点保留，不按接近目标1筛选；索引仅绑定真实监督事实，训练须另作独立check。")


def galleries(report, plan, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import goal_closed_loop_continuation as intervention

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for repeat in range(5):
        panel = Image.new("RGB", (820, 350), "white")
        draw = ImageDraw.Draw(panel)
        for x, title in zip((170, 330, 490, 650), ("fixed target1", "own frame64", "own frame80", "actual end84")):
            draw.text((x, 4), title, fill="black", font=font)
        for line, condition in enumerate(intervention.CONDITIONS):
            row = next((r for r in rows if r["repeat"] == repeat and r["condition"] == condition), None)
            y = 38 + line * 155
            draw.text((4, y + 12), condition.replace("_", "\n"), fill="black", font=font)
            panel.paste(Image.fromarray(bank["images"][1]).resize((128, 128)), (170, y))
            if row is None:
                draw.text((330, y + 40), "not attempted", fill="black", font=font)
                continue
            path = Path(row["artifact_dir"]) / "trajectory.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    images = saved["image"]
                    for x, frame in ((330, 64), (490, 80), (650, len(images) - 1)):
                        if frame < len(images):
                            panel.paste(Image.fromarray(images[frame]).resize((128, 128)), (x, y))
                        else:
                            draw.text((x, y + 40), "not reached", fill="black", font=font)
            label = (f"repeat={repeat} steps={row['actual_steps']} d1={row['end_distance']:.4f} continuation={row['continuation_progress']:.4f}"
                     if row["execution_valid"] and row["fixed_horizon_completed"] else
                     f"repeat={repeat} steps={row['actual_steps']} " + ("early terminal" if row["execution_valid"] else "interface FAIL"))
            draw.text((170, y - 18), label, fill="black", font=font)
        panel.save(report.directory / f"outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_closed_loop_continuation as intervention

    baseline.video_preflight(report)
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, 10 * 24 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("storage_preflight", "PASS", "预留10次完整真实轨迹/视频及实际终点空间；不复制模型或旧轨迹")
    report.check("independent_starts", "PASS", "10次独立fresh reset；16步自主后追加4步；反馈规则提前固定，不按边界状态筛选")
    rows, frames, balance = [], [], []
    first_start = None
    def persist():
        summary = intervention.summarize(rows, plan)
        summary["start_differences_diagnostic_only"] = balance
        verified = [row for row in rows if row["execution_valid"]]
        report.data.update(probability_roundtrip_queries=sum(row["probability_roundtrip_queries"] for row in verified),
            measurement_roundtrip_frames=sum(row["measurement_roundtrip_frames"] for row in verified),
            maximum_probability_error=max((row["maximum_probability_error"] for row in verified), default=None))
        summary.update(probability_roundtrip_queries=report.data["probability_roundtrip_queries"],
            measurement_roundtrip_frames=report.data["measurement_roundtrip_frames"],
            maximum_probability_error=report.data["maximum_probability_error"])
        baseline.write_json(report.directory / "trials.json", rows)
        baseline.write_json(report.directory / "diagnostics.json", summary)
        baseline.write_csv(report.directory / "trials.csv", [{name: row.get(name) for name in (
            "trial_index", "seed", "repeat", "target", "condition", "execution_valid", "saved_execution_verified", "video_verified",
            "actual_steps", "fixed_horizon_completed", "returned_env_steps", "returned_control_steps", "end_reason", "worker_action_queries",
            "episode_split", "autonomous_steps", "continuation_steps", "start_distance", "end_distance", "distance_improvement",
            "target_preference_margin", "boundary_kind", "boundary_distance", "autonomous_progress", "continuation_progress",
            "intervention_actions", "correction_actions", "actual_action_changes", "pitch_error_at_boundary", "pitch_error_at_end",
            "correction_kept_target1_preference", "hold_kept_target1_preference",
            "maximum_probability_error", "task_success", "external_return", "error", "artifact_dir")} for row in rows])
        baseline.write_csv(report.directory / "frame_metrics.csv", frames)
        baseline.write_json(report.directory / "initial_balance.json", balance)
        manual = [r for r in rows if r["condition"] == "visual_correct_hold" and
                  r["execution_valid"] and r["fixed_horizon_completed"]]
        coverage = dict(planned_manual_episodes=5, complete_manual_episodes=len(manual),
            correction_episodes=sum(r["correction_actions"] == 1 for r in manual),
            holding_episodes=sum(r["correction_actions"] == 0 and r["boundary_kind"] == "closer_target1" for r in manual),
            correction_kept_target1_preference=sum(r["correction_kept_target1_preference"] for r in manual),
            hold_kept_target1_preference=sum(r["hold_kept_target1_preference"] for r in manual),
            diagnostic_only=True, approved_for_training=False, t06_approved=False)
        coverage["both_branches_observed"] = coverage["correction_episodes"] > 0 and coverage["holding_episodes"] > 0
        baseline.write_json(report.directory / "coverage.json", coverage)
        summary["manual_branch_coverage"] = coverage
        baseline.write_json(report.directory / "diagnostics.json", summary)
        data = continuation_data(rows, plan)
        baseline.write_json(report.directory / "continuation_data.json", data)
        gl.require(legacy.read_json(report.directory / "continuation_data.json") == data, "实际续段索引保存不一致")
        report.data.update(continuation_episodes=data["complete_episodes"], continuation_action_records=data["real_action_records"])
        saved = dict(format=intervention.EVALUATION_FORMAT, comparison_protocol=intervention.PROTOCOL,
            design=plan, design_id=plan["design_id"], rows=rows, input_identity=plan["input_identity"],
            new_env_steps=report.data["new_env_steps"], attempted_env_steps=report.data["attempted_env_steps"],
            worker_action_queries=report.data["worker_action_queries"], worker_control_actions=report.data["worker_control_actions"],
            intervention_actions=report.data["intervention_actions"], control_actions=report.data["control_actions"],
            actual_action_changes=report.data["actual_action_changes"],
            correction_actions=report.data["correction_actions"],
            continuation_episodes=data["complete_episodes"], continuation_action_records=data["real_action_records"],
            probability_roundtrip_queries=report.data["probability_roundtrip_queries"],
            measurement_roundtrip_frames=report.data["measurement_roundtrip_frames"],
            maximum_probability_error=report.data["maximum_probability_error"],
            check_report_sha256=bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
            artifacts={str(path.relative_to(report.directory)): bc.file_hash(path)
                       for row in rows for path in Path(row["artifact_dir"]).iterdir() if path.is_file()},
            summary_artifacts={name: bc.file_hash(report.directory / name) for name in
                ("design.json", "targets.npz", "trials.json", "trials.csv", "diagnostics.json", "initial_balance.json", "continuation_data.json", "coverage.json",
                 "frame_metrics.csv", "attempts.jsonl", "trials.jsonl", *[f"outcomes_repeat_{repeat}.png" for repeat in range(5)])
                if (report.directory / name).is_file()},
            full_design_completed=len(rows) == 10, labels_generated=False, behavior_accepted=False, t06_approved=False)
        saved["evaluation_id"] = gl.tensor_digest({}, saved)
        baseline.write_json(report.directory / "evaluation_manifest.json", saved)
        report.data.update(evaluation_id=saved["evaluation_id"], completed_trial_records=len(rows),
                           full_design_completed=len(rows) == 10, execution_valid_trials=summary["valid_histories"])
        report.save()
        return summary
    for index, cell in enumerate(plan["design"]["schedule"]):
        report.require_writable()
        t04.append_json(report.directory / "attempts.jsonl", dict(trial_index=index, **cell, status="started"))
        report.data["attempted_trials"] = index + 1
        report.save()
        row = run_trial(report, runtime, plan, bank, cell, index)
        if row["execution_valid"]:
            try:
                measurements = verify_saved_trial(runtime, plan, bank, row)
                frames.extend(dict(trial_index=index, repeat=cell["repeat"], condition=cell["condition"], target=1,
                    **{name: value for name, value in measurement.items() if name != "goal_feature"}) for measurement in measurements)
                arrays = ctl.read_trajectory(Path(row["artifact_dir"]))
                if first_start is None:
                    first_start = arrays
                rgb = np.abs(arrays["image"][64].astype(np.float32) - first_start["image"][64].astype(np.float32))
                balance.append(dict(trial_index=index, repeat=cell["repeat"], condition=cell["condition"],
                    rgb_mae_to_first=float(rgb.mean()), pose_difference_to_first=ctl.pose_difference(arrays["telemetry"][64]["pose"], first_start["telemetry"][64]["pose"]),
                    rssm_relative_l2_to_first=float(np.linalg.norm(arrays["features"][64] - first_start["features"][64]) /
                        max(float(np.linalg.norm(first_start["features"][64])), 1e-8)), diagnostic_only=True))
            except Exception as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}", stop_after_trial=isinstance(error, OSError))
                baseline.write_json(Path(row["artifact_dir"]) / "verification_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
                report.check(f"trial_{index}_saved_execution", "FAIL", row["error"] + "；该尝试保留，不补样")
        rows.append(row)
        baseline.write_json(Path(row["artifact_dir"]) / "metrics.json", row)
        t04.append_json(report.directory / "trials.jsonl", row)
        summary = persist()
        detail = (f"steps={row['actual_steps']} d1={row['end_distance']:.5f} "
                  f"boundary={row['boundary_kind']} continuation={row['continuation_progress']} corrections={row['correction_actions']}"
                  if row["execution_valid"] else row.get("error", "interface failure"))
        print(f"[CONTINUATION {index + 1}/10] repeat={cell['repeat']} condition={cell['condition']} {detail}", flush=True)
        if row["stop_after_trial"]:
            raise RuntimeError("存储或环境关闭失败，停止后续环境启动；保留当前尝试/汇总，不支持续跑")
    galleries(report, plan, bank, rows)
    summary = persist()
    gl.require(report.data["new_env_steps"] <= report.data["attempted_env_steps"] <= 840 and
               report.data["worker_control_actions"] + report.data["intervention_actions"] == report.data["control_actions"], "实际预算或动作计数不一致")
    valid = summary["valid_histories"]
    report.check("planned_attempts", "PASS", "按预声明新顺序尝试10/10，全部保留，无重试/补样/起点筛选/中间最佳帧")
    report.check("real_control_interface", "PASS" if valid == 10 else "FAIL", f"自身真实历史/建议与实际动作/原生遥测/保存概率及视频有效{valid}/10")
    complete = sum(row["execution_valid"] and row["fixed_horizon_completed"] for row in rows)
    if complete == 10:
        expected = dict(new_env_steps=840, attempted_env_steps=840, evaluation_env_steps=840,
                        worker_action_queries=200, worker_control_actions=180, intervention_actions=20, control_actions=200,
                        probability_roundtrip_queries=200, measurement_roundtrip_frames=210,
                        continuation_episodes=10, continuation_action_records=40)
        gl.require(all(report.data[name] == count for name, count in expected.items()) and
                   0 <= report.data["actual_action_changes"] <= 20 and 0 <= report.data["correction_actions"] <= 5,
                   "完整10局的环境/查询/实际动作/干预计数异常")
        report.check("execution_counters", "PASS", "完整10局：env.step840、查询200、worker执行180、视觉干预20；最多5次纠偏，实际动作改变另计")
    report.check("fixed_horizon_endpoints", "PASS" if complete == 10 else "WARN", f"完整16+4步实际终点{complete}/10；原第80帧另存，提前结束/缺测单列")
    report.check("actual_endpoint_data", "PASS" if complete else "WARN",
        f"已索引{report.data['continuation_episodes']}条完整真实续段/{report.data['continuation_action_records']}个实际动作；"
        "绑定本局第84帧目标及80–83→81–84索引，失败不标成目标1，无专家标签")
    coverage = summary["manual_branch_coverage"]
    report.check("manual_branch_coverage", "PASS" if coverage["both_branches_observed"] else "WARN",
        f"纠偏分支{coverage['correction_episodes']}局、保持分支{coverage['holding_episodes']}局；"
        f"全过程保留目标1视觉偏好分别{coverage['correction_kept_target1_preference']}/{coverage['hold_kept_target1_preference']}；"
        "缺覆盖仍保留，不补样；效果另按全部轨迹判断")
    for condition, group in summary["groups"].items():
        print(f"[RESULT {condition}] full={group['full_horizon']}/5 closer_target1={group['endpoint_closer_to_target1']}/5 "
              f"mean_continuation_progress={group['continuation_mean_all_five']} corrections={group['corrections']} "
              f"actual_action_changes={group['actual_action_changes']}", flush=True)
    report.check("scope", "WARN", "仅真实纠偏/保持开发数据；手工控制不等于新worker确认，真实终点索引不是专家标签；不训练、不改旧门槛或批准T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        command.add_argument("--intervention-dir", required=True, help="已完整通过的terminal_intervention_evaluate目录")
        command.add_argument("--output-dir")
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        source, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "closed_loop_continuation_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出须独立于原诊断/确认/模型/缓存/验收目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0, attempted_env_steps=0,
        worker_action_queries=0, worker_control_actions=0, intervention_actions=0, control_actions=0,
        correction_actions=0, actual_action_changes=0, attempted_trials=0, full_design_completed=False,
        behavior_accepted=False, t06_approved=False, labels_generated=False, expert_labels_generated=False)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng, runtime = None, None, None
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
        report.require_writable()
        rng = bc.capture_rng(args.device)
        report.check("historical_identity", "PASS", "绑定完整10局干预/末步诊断/原30局与模型/verify及全部真实产物SHA256；旧行为失败保持")
        cache, runtime, plan, bank = load_inputs(args, report, source)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            {"check": check, "evaluate": evaluate}[args.command](args, report, cache, runtime, plan, bank)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()), "续段采集改变冻结参数或产生梯度")
        report.check("no_training", "PASS", f"更新0；worker/底座/WM/目标参数未变；worker查询={report.data['worker_action_queries']}，"
                     f"worker执行={report.data['worker_control_actions']}，视觉干预={report.data['intervention_actions']}，env.step={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型/缓存/诊断/确认/目标/验收/旧代码大小与修改时间未变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
