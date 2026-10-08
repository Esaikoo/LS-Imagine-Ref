"""T05 last-action intervention: offline check, then ten fresh development trials.

Version300/goal1/mode controls the first15 actions. The last action follows
the predeclared worker-mode or fixed-forward condition. No training, retries,
start filtering, checkpoint choice or T06 approval. Evaluate needs headless.
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
import t05_terminal_action_diagnose as terminal


class Report(pilot.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_TERMINAL_INTERVENTION_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(confirmed.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (
        *terminal.DIAGNOSIS_CODE_FILES, "goal_terminal_intervention.py", "scripts/t05_terminal_intervention.py")})


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_terminal_action_diagnose as diagnosis

    directory = baseline.project_path(args.diagnosis_dir)
    diagnosed = legacy.read_json(directory / "report.json")
    legacy.accepted(diagnosed, "analyze")
    required = ("historical_identity", "schedule_seed_contract", "diagnostic_contracts", "strict_frozen_load",
                "fixed_visual_targets", "real_query_alignment", "recorded_forward_roundtrip",
                "recorded_measurement_roundtrip", "same_state_terminal_comparison",
                "reference_context_comparison", "no_training", "source_inputs_unchanged")
    gl.require(confirmed.passed_checks(diagnosed, required) and
               diagnosed.get("worker_version") == 300 and diagnosed.get("repair_updates") == 100 and
               diagnosed.get("diagnosed_trials") == 30 and diagnosed.get("query_frames") == 60 and
               diagnosed.get("probability_roundtrip_frames") == 480 and diagnosed.get("measurement_roundtrip_frames") == 510 and
               diagnosed.get("reference_queries") == 8 and diagnosed.get("maximum_probability_error") == 0 and
               diagnosed.get("optimizer_updates") == diagnosed.get("new_env_steps") == 0 and
               diagnosed.get("behavior_accepted") is False and diagnosed.get("t06_approved") is False and
               diagnosed.get("diagnosis_code") == {name: bc.file_hash(ROOT / name) for name in terminal.DIAGNOSIS_CODE_FILES},
               "需要种子修复后完整通过的末步诊断和原代码；旧失败或子集不能进入真实干预")
    source = terminal.historical_inputs(SimpleNamespace(eval_dir=diagnosed["arguments"]["eval_dir"]))
    record, manifest, _, _, _, calibration, protected, files = source
    diagnostics = legacy.read_json(directory / "diagnostics.json")
    queries = legacy.read_json(directory / "action_queries.json")
    gl.require(diagnosed["source_evaluation_id"] == diagnostics["source_evaluation_id"] == manifest["evaluation_id"] and
               diagnosed["source_design_id"] == manifest["design_id"] and
               diagnosed["input_identity"] == manifest["input_identity"] and
               diagnosed["target_content_id"] == manifest["design"]["target_content_id"] and
               diagnosed["original_numerical_gate"] == diagnostics["original_numerical_gate"] == record["numerical_gate"] and
               diagnostics["format"] == queries["format"] == diagnosis.FORMAT and
               diagnostics["real_histories"] == 30 and diagnostics["query_frames"] == 60 and
               len(queries["frames"]) == 60 and len({(row["trial_index"], row["remaining"]) for row in queries["frames"]}) == 60 and
               queries["observed_actions_are_expert_labels"] is False and queries["counterfactual_rollouts_executed"] is False,
               "末步诊断与完整原确认/真实目标/预算或模型身份不同")
    names = ("report.json", "diagnostics.json", "action_queries.json", "trials.csv", "frame_metrics.csv",
             "reference_context.json", "target1_last_action_probabilities.png")
    hashes = {name: bc.file_hash(directory / name) for name in names}
    files += [directory / name for name in names] + [ROOT / name for name in code_identity()]
    protected.append(directory)
    if args.command == "evaluate":
        checked = baseline.project_path(args.check_dir)
        protected.append(checked)
        files += [checked / name for name in ("report.json", "design.json", "targets.npz")]
    gl.require(all(path.is_file() for path in files), "原诊断/模型/缓存/轨迹/check依赖缺失")
    return dict(record=record, manifest=manifest, calibration=calibration, diagnosed=diagnosed,
                diagnosis_dir=directory, diagnosis_hashes=hashes), protected, list(dict.fromkeys(path.resolve() for path in files))


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_terminal_intervention as intervention

    loader = SimpleNamespace(**source["record"]["arguments"])
    loader.command = "check"  # Reuse strict frozen loading; this call does not run a check or an environment.
    gl.require(args.device == loader.device, "必须使用原CUDA设备；本入口不切换推理后端")
    cache, runtime, previous, bank = confirmed.load_inputs(loader, report)
    gl.require(previous == source["manifest"]["design"],
               "必须使用原worker300/同一CUDA设备和初始化；不能改选模型或目标")
    forward = previous["action_names"].index("forward")
    for run in source["calibration"]["rows"]:
        if run["script"] == 1:
            real = ctl.read_trajectory(Path(run["artifact_dir"]))
            gl.require(int(real["action"][-1].argmax()) == forward, "固定forward必须是两条目标1已标定真实脚本的最后动作")
    fields = ("input_identity", "scenario", "visual_preprocessing", "action_names", "warmup_steps", "prefix_actions",
              "start_actions", "control_start_frame", "horizon", "worker_version", "repair_updates", "target_content_id", "target_telemetry")
    plan = {name: copy.deepcopy(previous[name]) for name in fields}
    plan.update(format=intervention.FORMAT, comparison_protocol=intervention.PROTOCOL, evaluation_code=code_identity(),
        environment_fingerprint=previous["environment_fingerprint"], focus_target=1, autonomous_steps_before_intervention=15,
        forward_action=forward, maximum_new_env_steps=800, same_hidden_state=False, matched_start_comparison=False,
        behavior_accepted=False, t06_approved=False, original_numerical_gate=copy.deepcopy(source["record"]["numerical_gate"]),
        source_evaluation_id=source["manifest"]["evaluation_id"], source_design_id=previous["design_id"],
        source_diagnosis=str(source["diagnosis_dir"]), source_diagnosis_sha256=source["diagnosis_hashes"],
        design=dict(seed=0, repetitions=5, randomization_seed=intervention.RANDOMIZATION_SEED,
                    conditions=list(intervention.CONDITIONS), execution_policy="mode", schedule=intervention.schedule()))
    intervention.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(comparison_protocol=intervention.PROTOCOL, evaluation_code=code_identity(), design_id=plan["design_id"],
        source_evaluation_id=plan["source_evaluation_id"], source_diagnosis_sha256=plan["source_diagnosis_sha256"],
        original_numerical_gate=plan["original_numerical_gate"], planned_trials=10, maximum_new_env_steps=800,
        randomization_seed=intervention.RANDOMIZATION_SEED, worker_action_queries=0, worker_control_actions=0,
        intervention_actions=0, actual_action_changes=0, control_actions=0, attempted_env_steps=0,
        behavior_accepted=False, t06_approved=False, labels_generated=False)
    if args.command == "evaluate":
        directory = baseline.project_path(args.check_dir)
        checked = legacy.read_json(directory / "report.json")
        legacy.accepted(checked, "check")
        gl.require(legacy.read_json(directory / "design.json") == plan and checked.get("design_id") == plan["design_id"] and
                   checked.get("comparison_protocol") == intervention.PROTOCOL and
                   checked.get("evaluation_code") == code_identity() and checked.get("input_identity") == plan["input_identity"] and
                   checked.get("targets_sha256") == bc.file_hash(directory / "targets.npz") and
                   confirmed.passed_checks(checked, ("historical_identity", "forward_source_contract", "trial_output_preflight",
                       "causal_execution_interface", "worker_distribution_contract", "intervention_contracts",
                       "predeclared_intervention", "no_training", "source_inputs_unchanged")),
                   "需要匹配当前10局干预的独立check；原确认check或修改计划不能执行")
        with np.load(directory / "targets.npz", allow_pickle=False) as saved:
            gl.require(set(saved.files) == set(bank) and all(np.array_equal(saved[name], bank[name]) for name in bank), "固定视觉目标被修改")
        report.check("design_identity", "PASS", "worker300/目标1/前15步mode/末步两条件各5次/新顺序与check完全一致")
    report.check("forward_source_contract", "PASS", "forward来自原目标1真实标定末步；固定干预不等于专家最优标签或已学会的策略")
    return cache, runtime, plan, bank


def check(args, report, cache, runtime, plan, bank):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control
    import goal_terminal_intervention as intervention

    directory = pilot.prepare_trial_directory(report.directory / "trial_output_probe")
    for name in ("start.json", "control_trace.json", "history_check.json", "metrics.json"):
        marker = dict(output_probe_only=True, before_environment=True, artifact=name)
        baseline.write_json(directory / name, marker)
        gl.require(legacy.read_json(directory / name) == marker, "trial原子JSON读写不一致")
    try:
        pilot.prepare_trial_directory(directory)
    except FileExistsError:
        pass
    else:
        raise ValueError("重复trial目录没有被拒绝")
    report.check("trial_output_preflight", "PASS", "启动环境前创建独立trial目录并验证原子JSON；重复目录拒绝覆盖")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "原真实历史数据集不同")
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
    for budget in (16, 2, 1):
        p = control.probabilities(runtime, features, bank["goals"], budget, 1, "goal")
        intervention.decide(p, 16 - budget, "fixed_forward", plan)
    for invalid in (0, 17, 1.5):
        t02.rejection(report, "worker_remaining_guard", lambda invalid=invalid: control.probabilities(runtime, features, bank["goals"], invalid, 1, "goal"), "remaining")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "因果或动作分布接口检查推进RNG")
    report.check("causal_execution_interface", "PASS", "复用两局真实因果增量/完整前缀及reset/结束/incoming守卫；未启动环境")
    report.check("worker_distribution_contract", "PASS", "真实缓存状态上的worker300/目标1/remaining16、2、1分布与末步决策接口有效")
    contract_checks(report, plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    baseline.write_json(report.directory / "design.json", plan)
    pilot.target_gallery(report.directory / "targets.png", bank["images"])
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("predeclared_intervention", "PASS", "目标1两末步条件各5局；32noop+32前缀+15自主+1预声明动作，最多800步，无起点筛选/补样")
    report.check("scope", "WARN", "离线check只验接口和固定设计；真实末步干预尚未运行，不训练或批准T06")


def contract_checks(report, plan):
    import numpy as np
    import goal_library as gl
    import goal_terminal_intervention as intervention

    # Memory-only records mirror real field placement, never saved as data.
    original = copy.deepcopy(plan)
    first, frames, dimension = 64, 81, 12
    probability = np.zeros(dimension)
    probability[[0, 2, plan["forward_action"]]] = [.1, .7, .2]
    telemetry = dict(pose=dict(x=0., y=64., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    for index, cell in enumerate(plan["design"]["schedule"]):
        decisions = [intervention.decide(probability, offset, cell["condition"], plan) for offset in range(16)]
        actual = [row["actual_action"] for row in decisions]
        arrays = dict(image=np.zeros((frames, 64, 64, 3), np.uint8), heatmap=np.zeros((frames, 64, 64), np.uint8),
            features=np.zeros((frames, 5120), np.float32), obs_reward=np.zeros((frames, 1), np.float32),
            action=np.concatenate((np.zeros((1, dimension), np.float32), np.eye(dimension, dtype=np.float32)[plan["start_actions"] + actual])),
            is_first=np.arange(frames) == 0, is_last=np.zeros(frames, bool), is_terminal=np.zeros(frames, bool))
        events = [dict(frame=frame, error=None, done=False, reward=0., telemetry=copy.deepcopy(telemetry),
                       native_actions=[] if frame == 0 else [np.zeros(8, np.int64)]) for frame in range(frames)]
        seed = intervention.action_seed(plan, index)
        trace = dict(comparison_protocol=intervention.PROTOCOL, design_id=plan["design_id"], model_id=plan["input_identity"]["model_id"],
            worker_version=300, repair_updates=100, condition=cell["condition"], issued_target=1, control_start_frame=64,
            execution_policy="mode", action_ids=actual, worker_action_ids=[row["worker_action"] for row in decisions],
            decision_sources=[row["decision_source"] for row in decisions], intervention_applied=[row["intervention_applied"] for row in decisions],
            action_differs_from_worker=[row["action_differs_from_worker"] for row in decisions], probabilities=[probability.tolist()] * 16,
            remaining=list(range(16, 0, -1)), action_seed=seed, uniforms=np.random.RandomState(seed).uniform(size=16).tolist(),
            distances_to_both_goals=[[.4, .2]] * 17)
        gl.require(intervention.validate_trace(arrays, events, trace, plan, index) == 16, "内存干预接口不一致")
    early_arrays = {name: value[:80].copy() for name, value in arrays.items()}
    early_arrays["is_terminal"][-1] = early_arrays["is_last"][-1] = True
    early_events = copy.deepcopy(events[:80])
    early_events[-1]["done"] = True
    early_trace = copy.deepcopy(trace)
    for name in ("action_ids", "worker_action_ids", "decision_sources", "intervention_applied",
                 "action_differs_from_worker", "probabilities", "remaining", "uniforms"):
        early_trace[name] = early_trace[name][:15]
    early_trace["distances_to_both_goals"] = early_trace["distances_to_both_goals"][:16]
    gl.require(intervention.validate_trace(early_arrays, early_events, early_trace, plan, index) == 15 and
               not any(early_trace["intervention_applied"]), "真实提前结束不能执行或记作最后干预")
    early_arrays["is_terminal"][-1] = early_arrays["is_last"][-1] = False
    t02.rejection(report, "incomplete_history_guard", lambda: intervention.validate_trace(
        early_arrays, early_events, early_trace, plan, index), "结束标志")
    # Both conditions occur; use the final loop's actual cell for mutations.
    wrong = copy.deepcopy(trace)
    wrong["worker_action_ids"][-1] = plan["forward_action"]
    t02.rejection(report, "worker_suggestion_guard", lambda: intervention.validate_trace(arrays, events, wrong, plan, index), "建议动作")
    wrong = copy.deepcopy(arrays)
    wrong["action"][-1] = np.eye(dimension)[0]
    t02.rejection(report, "incoming_guard", lambda: intervention.validate_trace(wrong, events, trace, plan, index), "incoming")
    wrong = copy.deepcopy(trace)
    wrong["decision_sources"][-2] = "fixed_forward"
    t02.rejection(report, "early_intervention_guard", lambda: intervention.validate_trace(arrays, events, wrong, plan, index), "只允许最后")
    wrong = copy.deepcopy(trace)
    wrong["remaining"][-1] = 2
    t02.rejection(report, "remaining_guard", lambda: intervention.validate_trace(arrays, events, wrong, plan, index), "remaining")
    wrong = copy.deepcopy(events)
    del wrong[-1]["native_actions"]
    t02.rejection(report, "missing_native_guard", lambda: intervention.validate_trace(arrays, wrong, trace, plan, index), "原生事件")
    wrong = copy.deepcopy(events)
    del wrong[-1]["telemetry"]["pose"]
    t02.rejection(report, "missing_telemetry_guard", lambda: intervention.validate_trace(arrays, wrong, trace, plan, index), "遥测")
    wrong = copy.deepcopy(arrays)
    wrong["is_first"][-1] = True
    t02.rejection(report, "fake_reset_guard", lambda: intervention.validate_trace(wrong, events, trace, plan, index), "伪reset")
    wrong = copy.deepcopy(arrays)
    wrong["is_terminal"][-2] = True
    t02.rejection(report, "saved_terminal_guard", lambda: intervention.validate_trace(wrong, events, trace, plan, index), "结束后")
    wrong = copy.deepcopy(trace)
    wrong["action_seed"] += 1
    t02.rejection(report, "action_seed_guard", lambda: intervention.validate_trace(arrays, events, wrong, plan, index), "随机数")
    wrong = copy.deepcopy(trace)
    wrong["uniforms"][-1] = (wrong["uniforms"][-1] + .5) % 1
    t02.rejection(report, "action_uniform_guard", lambda: intervention.validate_trace(arrays, events, wrong, plan, index), "随机数")
    changed = copy.deepcopy(plan)
    changed["design"]["schedule"][0]["condition"] = "undeclared"
    t02.rejection(report, "schedule_guard", lambda: intervention.validate_plan(changed), "计划")
    markers = [dict(cell, execution_valid=False, error="memory-only interface failure") for cell in plan["design"]["schedule"]]
    summary = intervention.summarize(markers, plan)
    gl.require(summary["attempted_trials"] == 10 and len(summary["errors"]) == 10 and
               all(group["whole_progress_mean_all_five"] is None for group in summary["groups"].values()), "错误被隐藏或缺测被计入五次均值")
    t02.rejection(report, "duplicate_trial_guard", lambda: intervention.summarize(markers + markers[:1], plan), "重复")
    partial = dict(markers[0], execution_valid=True, saved_execution_verified=True, video_verified=True,
        actual_steps=15, fixed_horizon_completed=False, returned_env_steps=79, returned_control_steps=15,
        worker_action_queries=15, actual_terminal=True, end_reason="environment_done",
        start_distances_to_both_goals=[.4, .2], end_distances_to_both_goals=[.4, .2],
        start_distance=.2, end_distance=.2, distance_improvement=0., target_preference_margin=.2,
        start_telemetry=telemetry, intervention_applied=False, last_action=None, last_worker_action=None,
        last_step_progress=None)
    group = intervention.summarize([partial], plan)["groups"][partial["condition"]]
    gl.require(group["early_terminal"] == 1 and group["full_horizon"] == 0 and
               group["whole_progress_mean_all_five"] is None, "真实提前结束被计作完整干预终点")
    complete = []
    for cell in plan["design"]["schedule"]:
        applied = cell["condition"] == "fixed_forward"
        end = [.5, .1] if applied else [.4, .2]
        complete.append(dict(cell, execution_valid=True, saved_execution_verified=True, video_verified=True,
            actual_steps=16, fixed_horizon_completed=True, returned_env_steps=80, returned_control_steps=16,
            worker_action_queries=16, actual_terminal=False, end_reason="fixed_budget",
            start_distances_to_both_goals=[.4, .2], end_distances_to_both_goals=end,
            start_distance=.2, end_distance=end[1], distance_improvement=.2 - end[1], target_preference_margin=end[0] - end[1],
            start_telemetry=telemetry, pre_last_telemetry=telemetry, intervention_applied=applied,
            last_worker_action=2, last_action=plan["forward_action"] if applied else 2,
            last_action_differs_from_worker=applied, distance_before_last=.2, last_step_progress=.2 - end[1]))
    summary = intervention.summarize(complete, plan)
    gl.require(summary["valid_histories"] == 10 and summary["groups"]["fixed_forward"]["actual_action_changes"] == 5 and
               np.isclose(summary["fixed_forward_minus_worker_mode"]["last_step_progress_mean_all_five"], .1),
               "完整固定预算的汇总与真实自身进展不同")
    wrong_rows = copy.deepcopy(complete)
    wrong_rows[0]["distance_improvement"] += .1
    t02.rejection(report, "own_progress_guard", lambda: intervention.summarize(wrong_rows, plan), "自身进展")
    equal = np.eye(12)[plan["forward_action"]]
    decision = intervention.decide(equal, 15, "fixed_forward", plan)
    gl.require(decision["intervention_applied"] and not decision["action_differs_from_worker"] and
               all(not intervention.decide(equal, offset, "fixed_forward", plan)["intervention_applied"] for offset in range(15)) and
               plan == original, "干预条件不能由worker动作或状态筛选决定；不能修改源计划")
    report.check("intervention_contracts", "PASS", "前15步mode/末步两条件/建议与实际动作分开；种子来自design，真实incoming/结束/原生遥测/重复/缺测守卫通过；内存标记不写轨迹")


def run_trial(report, runtime, plan, bank, cell, index):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl
    import goal_random_control as prefix
    import goal_residual_control as control
    import goal_terminal_intervention as intervention

    directory = pilot.prepare_trial_directory(report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_1_{cell['condition']}")
    first, seed = 64, intervention.action_seed(plan, index)
    uniforms = np.random.RandomState(seed).uniform(size=16)
    row = dict(cell, trial_index=index, artifact_dir=str(directory), execution_valid=False, saved_execution_verified=False,
        video_verified=False, actual_steps=0, fixed_horizon_completed=False, intervention_applied=False,
        same_hidden_state=False, matched_start_comparison=False)
    trace = dict(comparison_protocol=intervention.PROTOCOL, design_id=plan["design_id"], model_id=runtime.policy.model_id,
        worker_version=300, repair_updates=100, condition=cell["condition"], issued_target=1, control_start_frame=64,
        execution_policy="mode", action_seed=seed, action_ids=[], worker_action_ids=[], probabilities=[], remaining=[], uniforms=[],
        decision_sources=[], intervention_applied=[], action_differs_from_worker=[], distances_to_both_goals=[])
    session, fatal = None, None
    returned_before = report.data["new_env_steps"]
    queries_before = report.data["worker_action_queries"]
    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
    def execute(action):
        gl.require(report.data["attempted_env_steps"] < 800, "真实执行尝试超过预声明800步")
        report.data["attempted_env_steps"] += 1
        session.step(action)
    try:
        print(f"[ENV] intervention repeat={cell['repeat']} condition={cell['condition']} target=1 "
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
        for offset in range(16):
            p = control.probabilities(runtime, session.features, bank["goals"], 16 - offset, 1, "goal")
            decision = intervention.decide(p, offset, cell["condition"], plan)
            report.data["worker_action_queries"] += 1
            trace["probabilities"].append(p.tolist())
            trace["worker_action_ids"].append(decision["worker_action"])
            trace["action_ids"].append(decision["actual_action"])
            trace["remaining"].append(16 - offset)
            trace["uniforms"].append(float(uniforms[offset]))
            for name in ("decision_source", "intervention_applied", "action_differs_from_worker"):
                trace["decision_sources" if name == "decision_source" else name].append(decision[name])
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
        applied = count == 16 and cell["condition"] == "fixed_forward"
        report.data["intervention_actions"] += int(applied)
        report.data["worker_control_actions"] += count - int(applied)
        if applied and len(trace["action_differs_from_worker"]) == 16:
            report.data["actual_action_changes"] += int(trace["action_differs_from_worker"][-1])
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
    import goal_terminal_intervention as intervention

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
    measurements, maximum_error = [], 0.
    for offset in range(count + 1):
        frame = 64 + offset
        feature = ctl.encode_goal(runtime, {name: arrays[name][frame] for name in ("image", "heatmap")})
        measured = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                       events[frame]["telemetry"], bank, plan["target_telemetry"])
        measurements.append(dict(frame=frame, control_step=offset, goal_feature=feature.tolist(), **measured))
        if offset < count:
            state = torch.as_tensor(arrays["features"][frame:frame + 1].copy(), device=device)
            p = control.probabilities(runtime, state, bank["goals"], 16 - offset, 1, "goal")
            maximum_error = max(maximum_error, float(np.max(np.abs(p - np.asarray(trace["probabilities"][offset])))))
            decision = intervention.decide(p, offset, row["condition"], plan)
            gl.require(np.allclose(p, trace["probabilities"][offset], atol=3e-5, rtol=3e-5) and
                       decision["worker_action"] == trace["worker_action_ids"][offset] and
                       decision["actual_action"] == trace["action_ids"][offset], "保存状态的worker概率/建议/实际动作不能重现")
    distances = np.asarray([[item["distance_0"], item["distance_1"]] for item in measurements])
    gl.require(np.allclose(distances, trace["distances_to_both_goals"], atol=3e-5, rtol=3e-5) and
               np.allclose(start["distances_to_both_goals"], distances[0], atol=3e-5, rtol=3e-5) and
               np.allclose(start["initial_goal_feature"], measurements[0]["goal_feature"], atol=3e-5, rtol=3e-5),
               "保存真实画面/自身起点的目标测量不一致")
    final = measurements[-1]
    row.update(saved_execution_verified=True, video_verified=True, fixed_horizon_completed=count == 16,
        start_distances_to_both_goals=distances[0].tolist(), end_distances_to_both_goals=distances[-1].tolist(),
        start_distance=float(distances[0, 1]), end_distance=float(distances[-1, 1]),
        distance_improvement=float(distances[0, 1] - distances[-1, 1]), target_preference_margin=float(distances[-1, 0] - distances[-1, 1]),
        start_telemetry=events[64]["telemetry"], end_telemetry=events[-1]["telemetry"],
        pose_errors_at_end={name: value for name, value in final.items() if "_error_" in name},
        maximum_probability_error=maximum_error, probability_roundtrip_queries=count,
        measurement_roundtrip_frames=count + 1,
        last_step_progress=None, distance_before_last=None, last_action=None, last_worker_action=None,
        intervention_applied=False, last_action_differs_from_worker=False)
    if count == 16:
        before = measurements[-2]
        row.update(pre_last_telemetry=events[79]["telemetry"], last_step_progress=before["distance_1"] - final["distance_1"],
            distance_before_last=before["distance_1"], preference_margin_before_last=before["distance_0"] - before["distance_1"],
            distances_before_last_to_both_goals=distances[-2].tolist(),
            last_preference_reversed_away=bool(before["distance_1"] < before["distance_0"] and final["distance_1"] > final["distance_0"]),
            last_action=trace["action_ids"][-1], last_worker_action=trace["worker_action_ids"][-1],
            last_actual_action_name=plan["action_names"][trace["action_ids"][-1]], last_worker_action_name=plan["action_names"][trace["worker_action_ids"][-1]],
            intervention_applied=trace["intervention_applied"][-1], last_action_differs_from_worker=trace["action_differs_from_worker"][-1],
            pitch_error_before_last=before["pitch_error_1"], pitch_error_at_end=final["pitch_error_1"],
            worker_probability_actual_last=float(trace["probabilities"][-1][trace["action_ids"][-1]]),
            worker_probability_suggested_last=float(trace["probabilities"][-1][trace["worker_action_ids"][-1]]))
    baseline.write_json(directory / "frame_metrics.json", measurements)
    return measurements


def galleries(report, plan, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import goal_terminal_intervention as intervention

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for repeat in range(5):
        panel = Image.new("RGB", (820, 350), "white")
        draw = ImageDraw.Draw(panel)
        for x, title in zip((170, 330, 490, 650), ("fixed target1", "own frame64", "own frame79", "actual end")):
            draw.text((x, 4), title, fill="black", font=font)
        for line, condition in enumerate(intervention.CONDITIONS):
            row = next((r for r in rows if r["repeat"] == repeat and r["condition"] == condition), None)
            y = 38 + line * 155
            draw.text((4, y + 12), condition, fill="black", font=font)
            panel.paste(Image.fromarray(bank["images"][1]).resize((128, 128)), (170, y))
            if row is None:
                draw.text((330, y + 40), "not attempted", fill="black", font=font)
                continue
            path = Path(row["artifact_dir"]) / "trajectory.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    images = saved["image"]
                    for x, frame in ((330, 64), (490, 79), (650, len(images) - 1)):
                        if frame < len(images):
                            panel.paste(Image.fromarray(images[frame]).resize((128, 128)), (x, y))
                        else:
                            draw.text((x, y + 40), "not reached", fill="black", font=font)
            label = (f"repeat={repeat} steps={row['actual_steps']} d1={row['end_distance']:.4f} last_progress={row['last_step_progress']:.4f}"
                     if row["execution_valid"] and row["fixed_horizon_completed"] else
                     f"repeat={repeat} steps={row['actual_steps']} " + ("early terminal" if row["execution_valid"] else "interface FAIL"))
            draw.text((170, y - 18), label, fill="black", font=font)
        panel.save(report.directory / f"outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_terminal_intervention as intervention

    baseline.video_preflight(report)
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, 10 * 20 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("storage_preflight", "PASS", "预留10次完整真实轨迹/视频空间；不复制模型或原30局")
    report.check("independent_starts", "PASS", "10次独立fresh reset；两组前15步自主mode，末步条件提前固定，不按状态筛选")
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
            "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "distance_before_last", "last_step_progress",
            "last_worker_action_name", "last_actual_action_name", "intervention_applied", "last_action_differs_from_worker",
            "last_preference_reversed_away",
            "worker_probability_suggested_last", "worker_probability_actual_last", "pitch_error_before_last", "pitch_error_at_end",
            "maximum_probability_error", "task_success", "external_return", "error", "artifact_dir")} for row in rows])
        baseline.write_csv(report.directory / "frame_metrics.csv", frames)
        baseline.write_json(report.directory / "initial_balance.json", balance)
        saved = dict(format=intervention.EVALUATION_FORMAT, comparison_protocol=intervention.PROTOCOL,
            design=plan, design_id=plan["design_id"], rows=rows, input_identity=plan["input_identity"],
            new_env_steps=report.data["new_env_steps"], attempted_env_steps=report.data["attempted_env_steps"],
            worker_action_queries=report.data["worker_action_queries"], worker_control_actions=report.data["worker_control_actions"],
            intervention_actions=report.data["intervention_actions"], control_actions=report.data["control_actions"],
            actual_action_changes=report.data["actual_action_changes"],
            probability_roundtrip_queries=report.data["probability_roundtrip_queries"],
            measurement_roundtrip_frames=report.data["measurement_roundtrip_frames"],
            maximum_probability_error=report.data["maximum_probability_error"],
            check_report_sha256=bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
            artifacts={str(path.relative_to(report.directory)): bc.file_hash(path)
                       for row in rows for path in Path(row["artifact_dir"]).iterdir() if path.is_file()},
            summary_artifacts={name: bc.file_hash(report.directory / name) for name in
                ("design.json", "targets.npz", "trials.json", "trials.csv", "diagnostics.json", "initial_balance.json",
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
                  f"last={row.get('last_actual_action_name', 'not_reached')} intervention={int(row['intervention_applied'])}"
                  if row["execution_valid"] else row.get("error", "interface failure"))
        print(f"[INTERVENTION {index + 1}/10] repeat={cell['repeat']} condition={cell['condition']} {detail}", flush=True)
        if row["stop_after_trial"]:
            raise RuntimeError("存储或环境关闭失败，停止后续环境启动；保留当前尝试/汇总，不支持续跑")
    galleries(report, plan, bank, rows)
    summary = persist()
    gl.require(report.data["new_env_steps"] <= report.data["attempted_env_steps"] <= 800 and
               report.data["worker_control_actions"] + report.data["intervention_actions"] == report.data["control_actions"], "实际预算或动作计数不一致")
    valid = summary["valid_histories"]
    report.check("planned_attempts", "PASS", "按预声明新顺序尝试10/10，全部保留，无重试/补样/起点筛选/中间最佳帧")
    report.check("real_control_interface", "PASS" if valid == 10 else "FAIL", f"自身真实历史/建议与实际动作/原生遥测/保存概率及视频有效{valid}/10")
    complete = sum(row["execution_valid"] and row["fixed_horizon_completed"] for row in rows)
    if complete == 10:
        expected = dict(new_env_steps=800, attempted_env_steps=800, evaluation_env_steps=800,
                        worker_action_queries=160, worker_control_actions=155, intervention_actions=5, control_actions=160,
                        probability_roundtrip_queries=160, measurement_roundtrip_frames=170)
        gl.require(all(report.data[name] == count for name, count in expected.items()) and
                   0 <= report.data["actual_action_changes"] <= 5, "完整10局的环境/查询/实际动作/干预计数异常")
        report.check("execution_counters", "PASS", "完整10局：env.step800、查询160、worker执行155、固定干预5；干预与实际动作改变分别计数")
    report.check("fixed_horizon_endpoints", "PASS" if complete == 10 else "WARN", f"完整16步终点{complete}/10；提前结束和缺测单列，全五次均值不足不输出")
    for condition, group in summary["groups"].items():
        print(f"[RESULT {condition}] full={group['full_horizon']}/5 direction={group['positive_progress_and_correct_preference']}/5 "
              f"mean_progress={group['whole_progress_mean_all_five']} mean_last_step={group['last_step_progress_mean_all_five']} "
              f"actual_action_changes={group['actual_action_changes']}", flush=True)
    report.check("scope", "WARN", "这是固定末步干预开发实验，不是新worker确认；不生成训练标签，不修改旧门槛或批准T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        command.add_argument("--diagnosis-dir", required=True, help="已完整通过的terminal_action_diagnose_seed_fixed目录")
        command.add_argument("--output-dir")
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        source, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "terminal_intervention_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出须独立于原诊断/确认/模型/缓存/验收目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0, attempted_env_steps=0,
        worker_action_queries=0, worker_control_actions=0, intervention_actions=0, control_actions=0,
        attempted_trials=0, full_design_completed=False, behavior_accepted=False, t06_approved=False, labels_generated=False)
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
        report.check("historical_identity", "PASS", "绑定完整末步诊断/原30局与模型/verify/旧代码及真实产物SHA256；原行为失败保持")
        cache, runtime, plan, bank = load_inputs(args, report, source)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            {"check": check, "evaluate": evaluate}[args.command](args, report, cache, runtime, plan, bank)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()), "末步干预改变冻结参数或产生梯度")
        report.check("no_training", "PASS", f"更新0；worker/底座/WM/目标参数未变；worker查询={report.data['worker_action_queries']}，"
                     f"worker执行={report.data['worker_control_actions']}，固定干预={report.data['intervention_actions']}，env.step={report.data['new_env_steps']}")
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
