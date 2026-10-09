"""Check and collect dual-goal visual feedback inside worker350's original horizon.

This development collection adds no updates and does not change any confirmation gate.
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
import t05_continuation_control as deployed
import t05_continuation_action_diagnose as diagnostic

CODE = ("goal_within_horizon_feedback.py", "scripts/t05_within_horizon_feedback.py")


class Report(pilot.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_WITHIN_HORIZON_FEEDBACK_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(deployed.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (*diagnostic.CODE, *CODE)})


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_action_diagnose as diagnosis
    import goal_continuation_control as confirmation

    directory = baseline.project_path(args.diagnosis_dir)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "analyze")
    manifest = legacy.read_json(directory / "diagnosis_manifest.json")
    eval_dir = baseline.project_path(record["arguments"]["eval_dir"])
    source, protected, files = diagnostic.historical_inputs(SimpleNamespace(eval_dir=str(eval_dir), device=args.device))
    gl.require(record["arguments"]["device"] == args.device and
        manifest.get("format") == diagnosis.FORMAT and manifest.get("diagnosis_id") == record.get("diagnosis_id") ==
        gl.tensor_digest({}, {k: v for k, v in manifest.items() if k != "diagnosis_id"}) and
        manifest.get("source_evaluation_id") == source["manifest"]["evaluation_id"] and
        manifest.get("source_design_id") == source["manifest"]["design_id"] and
        manifest.get("source_manifest_sha256") == bc.file_hash(eval_dir / "evaluation_manifest.json") and
        manifest.get("diagnostic_code") == record.get("diagnostic_code") ==
        {name: bc.file_hash(ROOT / name) for name in diagnostic.CODE} and
        manifest.get("model_ids") == record.get("model_ids") and
        record["model_ids"]["350"] == source["manifest"]["input_identity"]["model_id"] and
        record.get("source_inputs_before") == record.get("source_inputs_after") and
        record.get("maximum_probability_error") == 0 and
        all(record.get(k) == v for k, v in dict(diagnosed_trials=30, query_frames=480,
            saved_history_records_verified=30, diagnostic_distribution_rows=4800,
            probability_roundtrip_frames=480, measurement_roundtrip_frames=510,
            optimizer_updates=0, new_env_steps=0, worker_control_actions=0,
            behavior_accepted=False, t06_approved=False).items()) and
        manifest.get("query_frames") == 480 and manifest.get("distributions_saved") == 4800 and
        manifest.get("optimizer_updates") == manifest.get("new_env_steps") == 0 and
        manifest.get("behavior_accepted") is False and manifest.get("t06_approved") is False and
        confirmation.passed(record, ("source_saved_history_schema", "strict_frozen_versions", "fixed_visual_targets",
            "saved_history_schema_contract", "diagnostic_contracts", "real_query_alignment", "recorded_forward_roundtrip",
            "recorded_measurement_roundtrip", "same_state_goal_and_version_comparison", "no_training", "source_inputs_unchanged")),
        "需要完整通过的30局/480状态同状态诊断及原worker350生产来源；不修改旧FAIL")
    required = {"queries.json", "frame_metrics.csv", "trials.json", "trials.csv", "diagnostics.json",
                "target0_last_action_probabilities.png", "target1_last_action_probabilities.png"}
    gl.require(required <= set(manifest["artifacts"]), "完整诊断产物缺失")
    for name, digest in manifest["artifacts"].items():
        path = (directory / name).resolve()
        gl.require(not Path(name).is_absolute() and directory in path.parents and path.is_file() and
                   bc.file_hash(path) == digest, f"诊断产物SHA256不同：{name}")
        files.append(path)
    source.update(diagnosis_directory=directory, diagnosis_record=record, diagnosis_manifest=manifest, eval_dir=eval_dir)
    protected.append(directory)
    files += [directory / "report.json", directory / "diagnosis_manifest.json", *[ROOT / name for name in code_identity()]]
    if args.command == "evaluate":
        checked_dir = baseline.project_path(args.check_dir)
        checked = legacy.read_json(checked_dir / "report.json")
        legacy.accepted(checked, "check")
        protected.append(checked_dir)
        files += [checked_dir / name for name in ("report.json", "design.json", "targets.npz")]
    return source, list(dict.fromkeys(p.resolve() for p in protected)), list(dict.fromkeys(p.resolve() for p in files))


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_within_horizon_feedback as protocol
    import goal_continuation_control as confirmation

    # Reuse the strict production350 loader, without emitting its separate 30-trial confirmation messages.
    class SourceReport:
        def __init__(self):
            self.data = {}
        def check(self, name, level, detail):
            gl.require(level == "PASS", f"原350加载接口失败：{name}: {detail}")
    audit = SourceReport()
    cache, runtime, previous, bank = deployed.load_inputs(SimpleNamespace(command="check", device=args.device), audit, source["context"])
    gl.require(previous == source["manifest"]["design"], "固定worker350/目标/初始化与原确认不同")
    fields = ("input_identity", "scenario", "visual_preprocessing", "action_names", "warmup_steps", "prefix_actions",
              "start_actions", "control_start_frame", "horizon", "worker_version", "source_worker_version", "additional_updates",
              "target_content_id", "target_telemetry", "environment_fingerprint")
    plan = {name: copy.deepcopy(previous[name]) for name in fields}
    plan.update(format=protocol.FORMAT, comparison_protocol=protocol.PROTOCOL, evaluation_code=code_identity(),
        feedback_start_frame=76, feedback_horizon=4, actual_endpoint_frame=80, maximum_new_env_steps=1600,
        noop_action=0, turn_up_action=previous["action_names"].index("turn_up"), turn_down_action=previous["action_names"].index("turn_down"),
        feedback_rule=protocol.feedback_rule(), episode_split={str(i): protocol.split(i) for i in range(5)},
        same_hidden_state=False, matched_start_comparison=False, behavior_accepted=False, t06_approved=False,
        original_numerical_gate=copy.deepcopy(source["record"]["numerical_gate"]), old_failures_unchanged=True,
        source_evaluation_id=source["manifest"]["evaluation_id"], source_design_id=previous["design_id"],
        source_diagnosis_id=source["diagnosis_manifest"]["diagnosis_id"],
        source_diagnosis_report_sha256=bc.file_hash(source["diagnosis_directory"] / "report.json"),
        design=dict(seed=0, repetitions=5, randomization_seed=protocol.RANDOMIZATION_SEED,
                    conditions=list(protocol.CONDITIONS), execution_policy="mode", schedule=protocol.schedule()))
    protocol.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update({name: audit.data[name] for name in ("input_identity", "model_id", "worker_version",
        "source_worker_version", "additional_updates", "counters", "environment_fingerprint", "environment_protocol", "visual_preprocessing_policy")})
    report.data.update(comparison_protocol=protocol.PROTOCOL, evaluation_code=code_identity(), design_id=plan["design_id"],
        source_evaluation_id=plan["source_evaluation_id"], source_diagnosis_id=plan["source_diagnosis_id"],
        original_numerical_gate=plan["original_numerical_gate"], planned_trials=20, maximum_new_env_steps=1600,
        randomization_seed=5, old_failures_unchanged=True, approved_for_training=False,
        matched_start_comparison=False, same_hidden_state=False, labels_generated=False, expert_labels_generated=False)
    report.check("strict_frozen_worker350", "PASS", "原生产300+固定50=350及独立verify；共用冻结底座/WM/两真实视觉目标，无优化器或候选模型")
    report.check("fixed_visual_targets", "PASS", "原两RGB/heatmap终点及初始化脚本不变并重编码通过；坐标/朝向仅诊断")
    report.check("visual_feedback_contract", "PASS", "原16步内前12步自主/末4步两目标反馈；只看当前真实视觉距离，目标0上转/目标1下转最多一次，其余noop")
    if args.command == "evaluate":
        directory = baseline.project_path(args.check_dir)
        checked = legacy.read_json(directory / "report.json")
        gl.require(legacy.read_json(directory / "design.json") == plan and checked.get("design_id") == plan["design_id"] and
            checked.get("input_identity") == plan["input_identity"] and checked.get("evaluation_code") == code_identity() and
            checked.get("targets_sha256") == bc.file_hash(directory / "targets.npz") and
            checked.get("comparison_protocol") == protocol.PROTOCOL and
            checked.get("source_diagnosis_id") == plan["source_diagnosis_id"] and
            checked.get("source_inputs_before") == checked.get("source_inputs_after") and
            all(checked.get(k) == 0 for k in ("new_env_steps", "attempted_env_steps", "optimizer_updates",
                "worker_action_queries", "worker_control_actions", "intervention_actions", "control_actions")) and
            confirmation.passed(checked, ("historical_identity", "strict_frozen_worker350",
                "fixed_visual_targets", "trial_output_preflight", "causal_execution_interface", "worker_distribution_contract",
                "feedback_contracts", "endpoint_index_contracts", "predeclared_collection", "no_training", "source_inputs_unchanged")),
            "需要同worker350/诊断/种子5/完整20局计划/代码的独立check")
        with np.load(directory / "targets.npz", allow_pickle=False) as saved:
            gl.require(set(saved.files) == set(bank) and all(np.array_equal(saved[k], bank[k]) for k in bank), "check固定目标改变")
        report.check("design_identity", "PASS", "worker350/两目标/种子5/20局/原16步/末4步规则及整局划分与check一致")
    return cache, runtime, plan, bank


def check(args, report, cache, runtime, plan, bank):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control
    directory = pilot.prepare_trial_directory(report.directory / "trial_output_probe")
    marker = dict(output_probe_only=True, before_environment=True)
    baseline.write_json(directory / "probe.json", marker)
    gl.require(legacy.read_json(directory / "probe.json") == marker, "原子JSON写入读取不同")
    try:
        pilot.prepare_trial_directory(directory)
    except FileExistsError:
        pass
    else:
        raise ValueError("重复trial目录未被拒绝")
    report.check("trial_output_preflight", "PASS", "启动环境前验证独立trial目录/原子写入；拒绝重复目录覆盖")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "真实缓存历史身份不同")
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
    for target in (0, 1):
        for budget in (16, 4, 1):
            control.probabilities(runtime, features, bank["goals"], budget, target, "goal")
    for invalid in (0, 17, 1.5):
        t02.rejection(report, "worker_remaining_guard", lambda invalid=invalid: control.probabilities(runtime, features, bank["goals"], invalid, 1, "goal"), "remaining")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "接口check推进查询RNG")
    report.check("causal_execution_interface", "PASS", "两局真实因果增量/完整前缀及reset/incoming/结束守卫；未启动环境")
    report.check("worker_distribution_contract", "PASS", "两固定目标remaining16、4、1分布有效；worker查询不执行")
    contract_checks(report, plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    baseline.write_json(report.directory / "design.json", plan)
    pilot.target_gallery(report.directory / "targets.png", bank["images"])
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("predeclared_collection", "PASS", "两目标×两条件×5=20局；32noop+32前缀+原16步，最多1600步；repeat0–2训练来源/3–4开发留出，不补样")
    report.check("scope", "WARN", "仅离线采集接口check，未启动环境；真实效果/双目标纠偏覆盖待采集，不训练或批准T06")


def run_trial(report, runtime, plan, bank, cell, index):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl
    import goal_random_control as prefix
    import goal_residual_control as control
    import goal_within_horizon_feedback as intervention

    directory = pilot.prepare_trial_directory(report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['condition']}")
    first, seed = 64, intervention.action_seed(plan, index)
    uniforms = np.random.RandomState(seed).uniform(size=16)
    row = dict(cell, trial_index=index, artifact_dir=str(directory), execution_valid=False, saved_execution_verified=False,
        video_verified=False, actual_steps=0, fixed_horizon_completed=False, episode_split=intervention.split(cell["repeat"]),
        same_hidden_state=False, matched_start_comparison=False)
    trace = dict(intervention.trace_identity(plan, index), action_seed=seed,
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
        gl.require(report.data["attempted_env_steps"] < 1600, "真实执行尝试超过预声明1600步")
        report.data["attempted_env_steps"] += 1
        session.step(action)
    try:
        print(f"[ENV] within_horizon repeat={cell['repeat']} condition={cell['condition']} target={cell['target']} "
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
            p = control.probabilities(runtime, session.features, bank["goals"], remaining, cell["target"], "goal")
            decision = intervention.decide(p, offset, cell, trace["distances_to_both_goals"][-1], correction_used, plan)
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
    except (Exception, KeyboardInterrupt) as error:
        row.update(error=f"{type(error).__name__}: {error}")
        if isinstance(error, (OSError, KeyboardInterrupt)):
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
            except (Exception, KeyboardInterrupt) as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}")
                report.check(f"trial_{index}_save", "FAIL", row["error"] + "；原始尝试保留")
                if isinstance(error, (OSError, KeyboardInterrupt)):
                    fatal = error
                baseline.write_json(directory / "save_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
            finally:
                try:
                    session.close()
                except (Exception, KeyboardInterrupt) as error:
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
        applied = max(0, count - 12) if cell["condition"] == "visual_correct_hold" else 0
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
    import goal_within_horizon_feedback as intervention

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
    gl.require(type(history.get("maximum_absolute_error")) in (int, float) and
        np.isfinite(history["maximum_absolute_error"]) and history["maximum_absolute_error"] >= 0 and
        history["maximum_absolute_error"] == row["own_history_max_error"], "真实因果误差缺失/非有限或不同")
    device = next(runtime.parameters()).device
    measurements, maximum_error, correction_used = [], 0., False
    for offset in range(count + 1):
        frame = 64 + offset
        feature = ctl.encode_goal(runtime, {name: arrays[name][frame] for name in ("image", "heatmap")})
        measured = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                       events[frame]["telemetry"], bank, plan["target_telemetry"])
        measurement = dict(frame=frame, control_step=offset, goal_feature=feature.tolist(),
            phase="autonomous12" if offset < 12 else "boundary76" if offset == 12 else "feedback_window4",
            query_remaining=None, actual_next_action=None, worker_next_action=None, decision_source=None,
            intervention_applied=None, correction_applied=None, **measured)
        measurements.append(measurement)
        if offset < count:
            state = torch.as_tensor(arrays["features"][frame:frame + 1].copy(), device=device)
            p = control.probabilities(runtime, state, bank["goals"], intervention.BUDGETS[offset], row["target"], "goal")
            maximum_error = max(maximum_error, float(np.max(np.abs(p - np.asarray(trace["probabilities"][offset])))))
            real_distances = ctl.distances(feature[None], bank["goals"])[0]
            decision = intervention.decide(p, offset, plan["design"]["schedule"][row["trial_index"]], real_distances, correction_used, plan)
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
    target, other = row["target"], 1 - row["target"]
    row.update(saved_execution_verified=True, video_verified=True, fixed_horizon_completed=count == 16,
        start_distances_to_both_goals=distances[0].tolist(), end_distances_to_both_goals=distances[-1].tolist(),
        start_distance=float(distances[0, target]), end_distance=float(distances[-1, target]),
        distance_improvement=float(distances[0, target] - distances[-1, target]), target_preference_margin=float(distances[-1, other] - distances[-1, target]),
        start_telemetry=events[64]["telemetry"], end_telemetry=events[-1]["telemetry"],
        pose_errors_at_end={name: value for name, value in final.items() if "_error_" in name},
        maximum_probability_error=maximum_error, probability_roundtrip_queries=count,
        measurement_roundtrip_frames=count + 1, autonomous_steps=min(12, count), window_steps=max(0, count - 12),
        intervention_actions=sum(trace["intervention_applied"]), correction_actions=sum(trace["correction_applied"]),
        actual_action_changes=sum(trace["action_differs_from_worker"]), boundary_kind=None,
        window_progress=None, autonomous_progress=None, pitch_error_at_end=final[f"pitch_error_{target}"])
    if count >= 12:
        boundary = measurements[12]
        row.update(boundary_distances_to_both_goals=distances[12].tolist(), boundary_telemetry=events[76]["telemetry"],
            boundary_kind=intervention.boundary_kind(distances[12]), boundary_distance=float(distances[12, target]),
            autonomous_progress=float(distances[0, target] - distances[12, target]),
            window_progress=float(distances[12, target] - distances[-1, target]), pitch_error_at_boundary=boundary[f"pitch_error_{target}"])
    if count == 16:
        corrections = [offset for offset, applied in enumerate(trace["correction_applied"]) if applied]
        row.update(correction_step_progress=None, correction_kept_requested_preference=None,
                   holding_requested_rows=0, hold_kept_requested_preference=None)
        if row["condition"] == "visual_correct_hold":
            preferred = f"closer_target{target}"
            holds = [o for o in range(12, 16) if not trace["correction_applied"][o] and
                     intervention.boundary_kind(distances[o]) == preferred]
            row["holding_requested_rows"] = len(holds)
            row["hold_kept_requested_preference"] = bool(holds and
                all(intervention.boundary_kind(d) == preferred for d in distances[holds[0]:])) if holds else None
            if corrections:
                o = corrections[0]
                row["correction_step_progress"] = float(distances[o, target] - distances[o + 1, target])
                row["correction_kept_requested_preference"] = bool(
                    all(intervention.boundary_kind(d) == preferred for d in distances[o + 1:]))
    # Save the real endpoint for every valid history, including early termination and behavior failures.
    endpoint_frame = 64 + count
    np.savez_compressed(directory / "actual_endpoint.npz", image=arrays["image"][endpoint_frame],
                        heatmap=arrays["heatmap"][endpoint_frame], goal_feature=np.asarray(final["goal_feature"], dtype=np.float32))
    with np.load(directory / "actual_endpoint.npz", allow_pickle=False) as endpoint:
        gl.require(set(endpoint.files) == {"image", "heatmap", "goal_feature"} and
            np.array_equal(endpoint["image"], arrays["image"][endpoint_frame]) and
            np.array_equal(endpoint["heatmap"], arrays["heatmap"][endpoint_frame]) and
            np.array_equal(endpoint["goal_feature"], np.asarray(final["goal_feature"], dtype=np.float32)), "本局真实终点保存不同")
    row["actual_endpoint_frame"] = endpoint_frame
    baseline.write_json(directory / "frame_metrics.json", measurements)
    return measurements


def contract_checks(report, plan):
    """In-memory interface markers; never save them as collected histories."""
    import numpy as np
    import goal_library as gl
    import goal_within_horizon_feedback as protocol
    original, rows, selected = copy.deepcopy(plan), [], None
    telemetry = dict(pose=dict(x=0., y=64., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    p = np.eye(12)[plan["action_names"].index("forward")]
    for index, cell in enumerate(plan["design"]["schedule"]):
        target = cell["target"]
        wrong = [.6, .03] if target == 0 else [.03, .6]
        right = wrong[::-1]
        distances = [wrong.copy() for _ in range(13)] + [right.copy() for _ in range(4)]
        used, decisions = False, []
        for offset in range(16):
            item = protocol.decide(p, offset, cell, distances[offset], used, plan)
            decisions.append(item)
            used = used or item["correction_applied"]
        actual = [item["actual_action"] for item in decisions]
        arrays = dict(image=np.zeros((81, 64, 64, 3), np.uint8), heatmap=np.zeros((81, 64, 64), np.uint8),
            features=np.zeros((81, 5120), np.float32), obs_reward=np.zeros((81, 1), np.float32),
            action=np.concatenate((np.zeros((1, 12), np.float32), np.eye(12, dtype=np.float32)[plan["start_actions"] + actual])),
            is_first=np.arange(81) == 0, is_last=np.zeros(81, bool), is_terminal=np.zeros(81, bool))
        events = [dict(frame=f, error=None, done=False, reward=0., telemetry=copy.deepcopy(telemetry),
                       native_actions=[] if f == 0 else [np.zeros(8, np.int64)]) for f in range(81)]
        seed = protocol.action_seed(plan, index)
        trace = dict(protocol.trace_identity(plan, index), action_seed=seed, probabilities=[p.tolist()] * 16,
            remaining=protocol.BUDGETS.copy(), uniforms=np.random.RandomState(seed).uniform(size=16).tolist(),
            distances_to_both_goals=distances)
        for field, saved in protocol.DECISION_FIELDS.items():
            trace[saved] = [item[field] for item in decisions]
        gl.require(protocol.validate_trace(arrays, events, trace, plan, index) == 16, "内存反馈接口不一致")
        manual = cell["condition"] == "visual_correct_hold"
        rows.append(dict(cell, trial_index=index, episode_split=protocol.split(cell["repeat"]), execution_valid=True,
            saved_execution_verified=True, video_verified=True, actual_steps=16, returned_env_steps=80, returned_control_steps=16,
            worker_action_queries=16, fixed_horizon_completed=True, actual_terminal=False, end_reason="fixed_budget",
            intervention_actions=4 if manual else 0, correction_actions=int(manual), actual_action_changes=4 if manual else 0,
            start_distances_to_both_goals=wrong, end_distances_to_both_goals=right, boundary_distances_to_both_goals=wrong,
            start_distance=.6, end_distance=.03, distance_improvement=.57, target_preference_margin=.57,
            boundary_kind=protocol.boundary_kind(wrong), window_progress=.57, autonomous_progress=0.,
            holding_requested_rows=3 if manual else 0, correction_step_progress=.57 if manual else None,
            correction_kept_requested_preference=True if manual else None, hold_kept_requested_preference=True if manual else None))
        if manual:
            selected = (index, arrays, events, trace)
    index, arrays, events, trace = selected
    def reject_trace(name, change, message):
        a, e, t = copy.deepcopy(arrays), copy.deepcopy(events), copy.deepcopy(trace)
        change(a, e, t)
        t02.rejection(report, name, lambda: protocol.validate_trace(a, e, t, plan, index), message)
    reject_trace("incoming_guard", lambda a, e, t: a["action"].__setitem__(77, np.eye(12)[0]), "incoming")
    reject_trace("missing_native_guard", lambda a, e, t: e[77].update(native_actions=[]), "原生事件")
    reject_trace("missing_telemetry_guard", lambda a, e, t: e[77].update(telemetry={}), "遥测")
    reject_trace("fake_reset_guard", lambda a, e, t: a["is_first"].__setitem__(77, True), "reset")
    reject_trace("terminal_guard", lambda a, e, t: a["is_last"].__setitem__(77, True), "结束")
    reject_trace("remaining_guard", lambda a, e, t: t["remaining"].__setitem__(12, 16), "remaining")
    reject_trace("worker_suggestion_guard", lambda a, e, t: t["worker_action_ids"].__setitem__(0, 0), "mode")
    reject_trace("probability_guard", lambda a, e, t: t["probabilities"].__setitem__(0, [0.] * 12), "概率")
    reject_trace("early_feedback_guard", lambda a, e, t: t["intervention_applied"].__setitem__(11, True), "来源")
    reject_trace("second_correction_guard", lambda a, e, t: t["correction_applied"].__setitem__(13, True), "次数")
    reject_trace("feedback_source_guard", lambda a, e, t: t["decision_sources"].__setitem__(12, "expert"), "来源")
    reject_trace("target_guard", lambda a, e, t: t.update(issued_target=1 - t["issued_target"]), "目标")
    reject_trace("worker_version_guard", lambda a, e, t: t.update(worker_version=300), "模型")
    reject_trace("action_seed_guard", lambda a, e, t: t.update(action_seed=t["action_seed"] + 1), "随机数")
    reject_trace("action_uniform_guard", lambda a, e, t: t["uniforms"].__setitem__(0, 0.), "随机数")
    reject_trace("split_guard", lambda a, e, t: t.update(episode_split="train" if t["episode_split"] != "train" else "development_holdout"), "划分")
    def wrong_action(a, e, t):
        t["action_ids"][12] = 0
        a["action"][77] = np.eye(12)[0]
    reject_trace("correction_action_guard", wrong_action, "实际动作")
    early = {name: value[:78].copy() for name, value in arrays.items()}
    early["is_last"][-1] = early["is_terminal"][-1] = True
    early_events = copy.deepcopy(events[:78])
    early_events[-1]["done"] = True
    early_trace = copy.deepcopy(trace)
    for name in (*protocol.DECISION_FIELDS.values(), "probabilities", "remaining", "uniforms"):
        early_trace[name] = early_trace[name][:13]
    early_trace["distances_to_both_goals"] = early_trace["distances_to_both_goals"][:14]
    gl.require(protocol.validate_trace(early, early_events, early_trace, plan, index) == 13, "真实提前结束必须单列")
    early["is_last"][-1] = early["is_terminal"][-1] = False
    t02.rejection(report, "incomplete_budget_guard", lambda: protocol.validate_trace(early, early_events, early_trace, plan, index), "remaining")
    for target in (0, 1):
        cell = dict(target=target, condition="visual_correct_hold")
        wrong = [.6, .03] if target == 0 else [.03, .6]
        expected = plan["turn_up_action" if target == 0 else "turn_down_action"]
        gl.require(protocol.decide(p, 12, cell, wrong, False, plan)["actual_action"] == expected and
            all(protocol.decide(p, 12, cell, d, used, plan)["actual_action"] == 0
                for d, used in ((wrong, True), (wrong[::-1], False), ([.4, .4], False))), "双目标方向/已纠偏/保持/持平规则不同")
        used, total = False, 0
        for offset in range(12, 16):
            item = protocol.decide(p, offset, cell, wrong, used, plan)
            used = used or item["correction_applied"]
            total += item["correction_applied"]
        gl.require(total == 1, "纠偏失败也不能再次纠偏")
    gl.require(protocol.summarize(rows, plan)["valid_histories"] == 20 and
        protocol.coverage(rows, plan)["all_target_split_branches_observed"], "完整双目标独立局覆盖统计不同")
    t02.rejection(report, "duplicate_trial_guard", lambda: protocol.summarize(rows + rows[:1], plan), "重复")
    bad = copy.deepcopy(rows)
    bad[0]["window_progress"] += .1
    t02.rejection(report, "own_progress_guard", lambda: protocol.summarize(bad, plan), "进展")
    absent = [dict(r, execution_valid=False) for r in rows]
    gl.require(all(g["window_mean_all_five"] is None for g in protocol.summarize(absent, plan)["groups"].values()) and
        not protocol.coverage(absent, plan)["all_target_split_branches_observed"], "缺测不能生成五次均值或覆盖证据")
    changed = copy.deepcopy(plan)
    changed["design"].pop("randomization_seed")
    t02.rejection(report, "missing_design_seed_guard", lambda: protocol.validate_plan(changed), "种子")
    changed = copy.deepcopy(plan)
    changed["feedback_start_frame"] = 80
    t02.rejection(report, "after_horizon_feedback_guard", lambda: protocol.validate_plan(changed), "计划")
    endpoint = dict(image=arrays["image"][80].copy(), heatmap=arrays["heatmap"][80].copy(), goal_feature=np.array([1., 0.], np.float32))
    queries = [dict(state_frame=76 + o, incoming_action_frame=77 + o, remaining=4 - o,
                   actual_action=trace["action_ids"][12 + o], decision_source=trace["decision_sources"][12 + o]) for o in range(4)]
    protocol.validate_endpoint_queries(arrays, trace, endpoint, queries)
    bad_queries = copy.deepcopy(queries)
    bad_queries[-1]["state_frame"] = 80
    t02.rejection(report, "after_endpoint_query_guard", lambda: protocol.validate_endpoint_queries(arrays, trace, endpoint, bad_queries), "终点后")
    bad_queries = copy.deepcopy(queries)
    bad_queries[0]["actual_action"] = 0
    t02.rejection(report, "factual_action_index_guard", lambda: protocol.validate_endpoint_queries(arrays, trace, endpoint, bad_queries), "事实索引")
    bad_endpoint = copy.deepcopy(endpoint)
    bad_endpoint["image"][0, 0, 0] = 255
    t02.rejection(report, "own_endpoint_guard", lambda: protocol.validate_endpoint_queries(arrays, trace, bad_endpoint, queries), "本局真实")
    gl.require(plan == original, "内存契约验收修改原计划")
    report.check("feedback_contracts", "PASS", "20个计划条目、真实incoming/结束/remaining16–1/种子/目标/最多一次纠偏/整局划分与缺测守卫；内存标记不写轨迹")
    report.check("endpoint_index_contracts", "PASS", "本局第80帧目标与76–79→incoming77–80/remaining4–1事实索引；失败不标意向目标，无专家标签")


def endpoint_data(rows, plan):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_within_horizon_feedback as protocol
    protocol.summarize(rows, plan)
    episodes = []
    for row in rows:
        complete = bool(row["execution_valid"] and row["fixed_horizon_completed"])
        item = dict(trial_index=row["trial_index"], repeat=row["repeat"], requested_target=row["target"], condition=row["condition"],
            episode_split=row["episode_split"], artifact_dir=row["artifact_dir"], execution_valid=row["execution_valid"],
            complete_window=complete, queries=[], expert_labels=False, approved_for_training=False,
            factual_pool="manual_" + row["episode_split"] if row["condition"] == "visual_correct_hold" else "worker_facts_only")
        if complete:
            directory = Path(row["artifact_dir"])
            arrays = ctl.read_trajectory(directory)
            events = legacy.read_json(directory / "events.json")
            trace = legacy.read_json(directory / "control_trace.json")
            gl.require(protocol.validate_trace(arrays, events, trace, plan, row["trial_index"]) == 16 and
                       row["actual_endpoint_frame"] == 80, "实际状态/动作/本局终点不同")
            with np.load(directory / "actual_endpoint.npz", allow_pickle=False) as saved:
                endpoint = {k: saved[k].copy() for k in saved.files}
            measurements = legacy.read_json(directory / "frame_metrics.json")
            gl.require(np.array_equal(endpoint["goal_feature"], np.asarray(measurements[-1]["goal_feature"], np.float32)),
                       "终点目标编码与本局真实测量不同")
            queries = [dict(state_frame=76 + o, incoming_action_frame=77 + o, remaining=4 - o,
                actual_action=trace["action_ids"][12 + o], decision_source=trace["decision_sources"][12 + o]) for o in range(4)]
            item["queries"] = protocol.validate_endpoint_queries(arrays, trace, endpoint, queries)
            item.update(endpoint_frame=80, endpoint_artifact="actual_endpoint.npz", supervision_goal="this_episode_actual_frame80_rgb_heatmap",
                intended_fixed_target_is_not_a_label=True, endpoint_closer_to_requested_target=bool(row["target_preference_margin"] > protocol.MARGIN_EPSILON),
                model_id=plan["input_identity"]["model_id"], worker_version=350,
                artifacts_sha256={name: bc.file_hash(directory / name) for name in
                    ("trajectory.npz", "events.json", "control_trace.json", "history_check.json", "frame_metrics.json", "actual_endpoint.npz", "video.mp4")})
        episodes.append(item)
    pools = ("manual_train", "manual_development_holdout", "worker_facts_only")
    return dict(format=protocol.DATA_FORMAT, design_id=plan["design_id"], input_identity=plan["input_identity"],
        planned_trials=20, attempted_trials=len(rows), episodes=episodes,
        complete_episodes=sum(e["complete_window"] for e in episodes), real_action_records=sum(len(e["queries"]) for e in episodes),
        pool_action_counts={pool: sum(len(e["queries"]) for e in episodes if e["factual_pool"] == pool) for pool in pools},
        intended_fixed_target_is_not_a_label=True, expert_labels_generated=False, approved_for_training=False,
        optimizer_updates=0, t06_approved=False)


def galleries(report, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import goal_within_horizon_feedback as protocol
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for target in (0, 1):
        for repeat in range(5):
            panel = Image.new("RGB", (820, 350), "white")
            draw = ImageDraw.Draw(panel)
            for x, title in zip((170, 330, 490, 650), (f"fixed target{target}", "own frame64", "own frame76", "actual end80")):
                draw.text((x, 4), title, fill="black", font=font)
            for line, condition in enumerate(protocol.CONDITIONS):
                row = next((r for r in rows if r["target"] == target and r["repeat"] == repeat and r["condition"] == condition), None)
                y = 38 + line * 155
                draw.text((4, y + 12), condition.replace("_", "\n"), fill="black", font=font)
                panel.paste(Image.fromarray(bank["images"][target]).resize((128, 128)), (170, y))
                if row is None:
                    draw.text((330, y + 40), "not attempted", fill="black", font=font)
                    continue
                path = Path(row["artifact_dir"]) / "trajectory.npz"
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        images = saved["image"]
                        for x, frame in ((330, 64), (490, 76), (650, len(images) - 1)):
                            if frame < len(images):
                                panel.paste(Image.fromarray(images[frame]).resize((128, 128)), (x, y))
                            else:
                                draw.text((x, y + 40), "not reached", fill="black", font=font)
                label = (f"steps={row['actual_steps']} d={row['end_distance']:.4f} window={row['window_progress']:.4f} corrections={row['correction_actions']}"
                         if row["execution_valid"] and row["fixed_horizon_completed"] else
                         f"steps={row['actual_steps']} " + ("early terminal; own actual end" if row["execution_valid"] else "interface FAIL"))
                draw.text((170, y - 18), label, fill="black", font=font)
            panel.save(report.directory / f"target{target}_outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_within_horizon_feedback as protocol
    baseline.video_preflight(report)
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, 20 * 24 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("storage_preflight", "PASS", "预留20次完整真实历史/视频/实际终点空间；不复制模型或旧轨迹")
    report.check("independent_starts", "PASS", "20次独立fresh reset；原16步前12自主/末4固定条件，边界状态不筛选、全保留")
    rows, frames, balance, first_start = [], [], [], None

    def persist():
        summary = protocol.summarize(rows, plan)
        branch_coverage = protocol.coverage(rows, plan)
        data = endpoint_data(rows, plan)
        verified = [r for r in rows if r["execution_valid"]]
        report.data.update(probability_roundtrip_queries=sum(r["probability_roundtrip_queries"] for r in verified),
            measurement_roundtrip_frames=sum(r["measurement_roundtrip_frames"] for r in verified),
            maximum_probability_error=max((r["maximum_probability_error"] for r in verified), default=None),
            endpoint_episodes=data["complete_episodes"], endpoint_action_records=data["real_action_records"],
            pool_action_counts=data["pool_action_counts"])
        summary.update(manual_branch_coverage=branch_coverage,
            original_numerical_gate=plan["original_numerical_gate"],
            **{k: report.data[k] for k in ("probability_roundtrip_queries", "measurement_roundtrip_frames", "maximum_probability_error")})
        for name, content in (("trials.json", rows), ("diagnostics.json", summary), ("initial_balance.json", balance),
                              ("coverage.json", branch_coverage), ("endpoint_data.json", data)):
            baseline.write_json(report.directory / name, content)
        gl.require(legacy.read_json(report.directory / "endpoint_data.json") == data, "事实索引原子保存不同")
        columns = ("trial_index", "seed", "repeat", "target", "condition", "episode_split", "execution_valid",
            "saved_execution_verified", "video_verified", "actual_steps", "fixed_horizon_completed", "actual_endpoint_frame",
            "returned_env_steps", "returned_control_steps", "worker_action_queries", "end_reason", "actual_terminal",
            "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "boundary_kind",
            "boundary_distance", "autonomous_progress", "window_progress", "intervention_actions", "correction_actions",
            "actual_action_changes", "correction_step_progress", "correction_kept_requested_preference", "holding_requested_rows",
            "hold_kept_requested_preference", "pitch_error_at_boundary", "pitch_error_at_end", "own_history_max_error",
            "maximum_probability_error", "task_success", "external_return", "error", "artifact_dir")
        baseline.write_csv(report.directory / "trials.csv", [{k: r.get(k) for k in columns} for r in rows])
        baseline.write_csv(report.directory / "frame_metrics.csv", frames)
        galleries(report, bank, rows)
        manifest = dict(format=protocol.EVALUATION_FORMAT, comparison_protocol=protocol.PROTOCOL, design=plan,
            design_id=plan["design_id"], rows=rows, input_identity=plan["input_identity"],
            source_diagnosis_id=plan["source_diagnosis_id"],
            check_report_sha256=bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
            artifacts={str(p.relative_to(report.directory)): bc.file_hash(p)
                       for r in rows for p in Path(r["artifact_dir"]).iterdir() if p.is_file()},
            summary_artifacts={p.name: bc.file_hash(p) for p in report.directory.iterdir()
                if p.is_file() and p.name not in ("report.json", "evaluation_manifest.json")},
            **{k: report.data[k] for k in ("new_env_steps", "attempted_env_steps", "worker_action_queries", "worker_control_actions",
                "intervention_actions", "control_actions", "actual_action_changes", "correction_actions", "probability_roundtrip_queries",
                "measurement_roundtrip_frames", "maximum_probability_error", "endpoint_episodes", "endpoint_action_records", "pool_action_counts")},
            full_design_completed=len(rows) == 20, optimizer_updates=0, expert_labels_generated=False,
            approved_for_training=False, behavior_accepted=False, t06_approved=False)
        manifest["evaluation_id"] = gl.tensor_digest({}, manifest)
        baseline.write_json(report.directory / "evaluation_manifest.json", manifest)
        report.data.update(evaluation_id=manifest["evaluation_id"], completed_trial_records=len(rows),
                           full_design_completed=len(rows) == 20, execution_valid_trials=summary["valid_histories"])
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
                frames.extend(dict(trial_index=index, repeat=cell["repeat"], condition=cell["condition"], target=cell["target"],
                    **{k: v for k, v in m.items() if k != "goal_feature"}) for m in measurements)
                arrays = ctl.read_trajectory(Path(row["artifact_dir"]))
                if first_start is None:
                    first_start = {k: arrays[k][64].copy() for k in ("image", "features")}
                    first_start["pose"] = copy.deepcopy(arrays["telemetry"][64]["pose"])
                balance.append(dict(trial_index=index, repeat=cell["repeat"], target=cell["target"], condition=cell["condition"],
                    rgb_mae_to_first=float(np.abs(arrays["image"][64].astype(np.float32) - first_start["image"].astype(np.float32)).mean()),
                    pose_difference_to_first=ctl.pose_difference(arrays["telemetry"][64]["pose"], first_start["pose"]),
                    rssm_relative_l2_to_first=float(np.linalg.norm(arrays["features"][64] - first_start["features"]) /
                        max(float(np.linalg.norm(first_start["features"])), 1e-8)),
                    start_distances_to_both_goals=row["start_distances_to_both_goals"], diagnostic_only=True))
            except (Exception, KeyboardInterrupt) as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}",
                           stop_after_trial=bool(row["stop_after_trial"] or isinstance(error, (OSError, KeyboardInterrupt))))
                baseline.write_json(Path(row["artifact_dir"]) / "verification_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
                report.check(f"trial_{index}_saved_execution", "FAIL", row["error"] + "；保留，不重试/补样")
        rows.append(row)
        baseline.write_json(Path(row["artifact_dir"]) / "metrics.json", row)
        t04.append_json(report.directory / "trials.jsonl", row)
        summary = persist()
        detail = (f"steps={row['actual_steps']} d={row['end_distance']:.5f} window={row['window_progress']} corrections={row['correction_actions']}"
                  if row["execution_valid"] else row.get("error", "interface failure"))
        print(f"[WITHIN HORIZON {index + 1}/20] repeat={cell['repeat']} target={cell['target']} condition={cell['condition']} {detail}", flush=True)
        if row["stop_after_trial"]:
            raise RuntimeError("存储/关闭失败或用户中断；保留当前尝试/汇总，停止环境启动，禁止续跑补样")
    gl.require(report.data["new_env_steps"] <= report.data["attempted_env_steps"] <= 1600 and
        report.data["worker_control_actions"] + report.data["intervention_actions"] == report.data["control_actions"], "预算或动作计数不同")
    valid = summary["valid_histories"]
    complete = sum(r["execution_valid"] and r["fixed_horizon_completed"] for r in rows)
    report.check("planned_attempts", "PASS", "按预声明顺序尝试20/20，全部保留，无重试/补样/起点筛选/最佳帧替换")
    report.check("real_control_interface", "PASS" if valid == 20 else "FAIL", f"真实历史/动作/遥测/概率/视频有效{valid}/20")
    if complete == 20:
        expected = dict(new_env_steps=1600, attempted_env_steps=1600, evaluation_env_steps=1600,
            worker_action_queries=320, worker_control_actions=280, intervention_actions=40, control_actions=320,
            probability_roundtrip_queries=320, measurement_roundtrip_frames=340, endpoint_episodes=20, endpoint_action_records=80)
        gl.require(all(report.data[k] == v for k, v in expected.items()) and
            report.data["pool_action_counts"] == dict(manual_train=24, manual_development_holdout=16, worker_facts_only=40) and
            0 <= report.data["actual_action_changes"] <= 40 and 0 <= report.data["correction_actions"] <= 10, "完整20局计数异常")
        report.check("execution_counters", "PASS", "完整20局：env.step1600/查询320/worker实际280/手工40；实际动作改变与最多10次纠偏另计")
    report.check("fixed_horizon_endpoints", "PASS" if complete == 20 else "WARN", f"原16步真实终点{complete}/20；第76帧边界另存，提前结束/缺测单列")
    report.check("actual_endpoint_data", "PASS" if complete else "WARN",
        f"完整事实索引{report.data['endpoint_episodes']}局/{report.data['endpoint_action_records']}动作；本局80帧目标，手工训练/开发与worker事实分开，无专家标签")
    coverage = summary["manual_branch_coverage"]
    report.check("manual_branch_coverage", "PASS" if coverage["all_target_split_branches_observed"] else "WARN",
        "分别报告两目标×训练/开发纠偏及保持独立局数、纠偏实际收益/持续偏好；缺覆盖明确保留，不补样，重采样不算新证据")
    for name, group in summary["groups"].items():
        print(f"[RESULT {name}] full={group['full_horizon']}/5 closer={group['endpoint_closer_to_requested_target']}/5 "
              f"mean_window={group['window_mean_all_five']} corrections={group['corrections']}", flush=True)
    report.check("scope", "WARN", "预算内双目标反馈开发采集；手工控制不等于新worker确认/专家标签，不训练、不改旧门槛或批准T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        command.add_argument("--diagnosis-dir", required=True)
        command.add_argument("--output-dir")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--feedback-dir", help="fresh independent export directory; otherwise auto-generated")
    args = parser.parse_args()
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    args.diagnosis_dir = str(baseline.project_path(args.diagnosis_dir))
    if args.command == "evaluate":
        args.check_dir = str(baseline.project_path(args.check_dir))
    try:
        source, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "within_horizon_feedback_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == p or p in directory.parents or directory in p.parents for p in protected):
            parser.error("输出须独立于旧模型/缓存/轨迹/验收目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0, attempted_env_steps=0,
        worker_action_queries=0, worker_control_actions=0, intervention_actions=0, control_actions=0,
        correction_actions=0, actual_action_changes=0, attempted_trials=0, full_design_completed=False,
        probability_roundtrip_queries=0, measurement_roundtrip_frames=0,
        behavior_accepted=False, t06_approved=False, labels_generated=False, expert_labels_generated=False,
        feedback_policy="automatic lossless single JSON/images bundle after final report; failures retained")
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng, runtime = None, None, None
    os.chdir(ROOT)
    try:
        import numpy as np
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        before = {str(p): baseline.file_signature(p) for p in files}
        report.data["source_inputs_before"] = before
        bc.require_disk_space(directory, 160 * 1024**2)
        report.require_writable()
        rng = bc.capture_rng(args.device)
        report.check("historical_identity", "PASS", "完整350同状态诊断/30局确认/check/固定latest/verify及全部真实SHA绑定；旧行为FAIL保持")
        cache, runtime, plan, bank = load_inputs(args, report, source)
        versions = {name: p._version for name, p in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        bank_before = {k: v.copy() for k, v in bank.items()}
        with torch.no_grad():
            {"check": check, "evaluate": evaluate}[args.command](args, report, cache, runtime, plan, bank)
        gl.require(versions == {name: p._version for name, p in runtime.named_parameters()} and
            digest == gl.tensor_digest(runtime.state_dict(), {}) and all(not p.requires_grad and p.grad is None for p in runtime.parameters()) and
            all(np.array_equal(bank[k], bank_before[k]) for k in bank), "采集改变冻结参数/版本/目标或产生梯度")
        report.check("no_training", "PASS", f"更新0；冻结worker350/底座/WM/目标未变；查询={report.data['worker_action_queries']}，"
            f"worker实际={report.data['worker_control_actions']}，手工={report.data['intervention_actions']}，env.step={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(p): baseline.file_signature(p) for p in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                    "原模型/缓存/轨迹/目标/诊断/验收及43个已验收代码大小和修改时间不变；只写独立新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    result = report.finish()
    # Export only after finalizing the report. Do not mutate any hashed result afterwards.
    try:
        import t05_result_bundle as feedback
        sources = [("run", directory)]
        if args.command == "evaluate":
            sources.append(("check", baseline.project_path(args.check_dir)))
        sources.extend((("diagnosis", source["diagnosis_directory"]), ("source_evaluate", source["eval_dir"]),
                        ("repair_verify", source["context"]["verify_dir"])))
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT / "relevance_map/t05_feedback" / (
            directory.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in protected):
            raise ValueError("汇总输出必须独立于旧保护目录")
        index = feedback.build_bundle(sources, export, directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export / 'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"[FAIL] feedback_export: {type(error).__name__}: {error}；原运行结果已保留", file=sys.stderr, flush=True)
        result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())


