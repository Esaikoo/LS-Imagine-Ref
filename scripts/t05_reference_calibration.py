"""T05 fixed reference-action calibration: offline check, then four real runs.

Probe requires MINEDOJO_HEADLESS=1. No worker action selection, training,
new goals, retries or matched-start gate. Old results remain unchanged.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_action_choice_diagnose as history
import t05_goal_control as legacy
import t05_random_control as pilot
import t05_residual_control as original


class Report(baseline.Report):
    def check(self, name, level, detail):
        if name == "video_preflight":
            detail = "2帧MP4编码及完整解码通过；尚未启动MineDojo"
        return super().check(name, level, detail)

    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_REFERENCE_CALIBRATION_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(pilot.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (
        "goal_reference_calibration.py", "scripts/t05_reference_calibration.py",
        "scripts/t05_action_choice_diagnose.py")})


def source_paths(args, historical):
    _, manifest, _, protected, files = historical
    reading = copy.copy(args)
    reading.source_prepare_dir = manifest["design"]["source_benchmark"]
    extra_protected, extra_files, source = original.source_paths(reading)
    if args.command == "probe":
        checked_dir = baseline.project_path(args.check_dir)
        extra_files.extend(checked_dir / name for name in ("design.json", "targets.npz"))
    files = list(dict.fromkeys(path.resolve() for path in [*files, *extra_files]))
    return list(dict.fromkeys([*protected, *extra_protected])), files, source


def load_inputs(args, report, historical):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_calibration as calibration

    _, previous_manifest, previous_check, _, _ = historical
    previous = previous_manifest["design"]
    gl.require(previous["design"]["execution_policy"] == "sample" and previous["warmup_steps"] == 32 and
               previous["horizon"] == 16 and previous["control_start_frame"] == 64 and
               len(previous["prefix_actions"]) == 32 and
               previous["start_actions"] == [0] * 32 + previous["prefix_actions"],
               "需要当前32noop+32前缀+16步的完整sample试跑；不重新选择初始化")
    source = baseline.project_path(previous["source_benchmark"])
    gl.require(bc.file_hash(source / "benchmark.json") == previous["source_manifest_sha256"] and
               bc.file_hash(source / "report.json") == previous["source_report_sha256"], "原真实目标来源改变")
    reading = copy.copy(args)
    reading.command = "check"
    cache, runtime = original.load_inputs(reading, report)
    gl.require(report.data["input_identity"] == previous["input_identity"] and
               report.data["environment_fingerprint"] == previous["environment_fingerprint"],
               "必须沿用完整sample试跑的模型、缓存、推理后端和环境")
    reading.benchmark_dir = str(source)
    _, source_manifest = original.load_benchmark(reading, report)
    case = next((item for item in source_manifest["cases"] if item["seed"] == previous["scenario"]["seed"]), None)
    gl.require(case is not None and case["scenario"] == previous["scenario"] and
               case["prefix_actions"] == previous["prefix_actions"] and
               source_manifest["prefix_steps"] == 32 and source_manifest["horizon"] == 16,
               "原参考的世界种子、前缀或预算不同")
    ctl.validate_scenario(case["scenario"])
    with np.load(source / case["directory"] / "goals.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    with np.load(previous_check / "targets.npz", allow_pickle=False) as saved:
        gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "不能替换sample的两个固定目标")
    gl.require(bank["goals"].shape == (2, cache.metadata["goal_dim"]) and
               bank["images"].shape == (2, 64, 64, 3) and bank["heatmaps"].shape == (2, 64, 64) and
               bank["images"].dtype == bank["heatmaps"].dtype == np.uint8 and
               np.isfinite(bank["goals"]).all() and
               np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5), "固定目标形状、类型或数值异常")
    content = gl.tensor_digest({}, {name: bank[name].tolist() for name in bank})
    gl.require(content == previous["target_content_id"], "固定目标内容ID不同")
    references, scripts, target_telemetry, native = [], [], [], []
    for branch in (0, 1):
        directory = source / case["directory"] / f"reference_{branch}"
        reference = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        gl.require(len(reference["image"]) == len(events) == 49 and
                   reference["action"][1:33].argmax(-1).tolist() == previous["prefix_actions"] and
                   np.array_equal(bank["images"][branch], reference["image"][-1]) and
                   np.array_equal(bank["heatmaps"][branch], reference["heatmap"][-1]),
                   "必须使用完整真实参考的action[33:49]和原终点，不能用失败prepare")
        script = reference["action"][33:49].argmax(-1).tolist()
        calibration.validate_execution(reference, events, previous["prefix_actions"], script, 12)
        encoded = ctl.encode_goal(runtime, dict(image=reference["image"][-1], heatmap=reference["heatmap"][-1]))
        gl.require(np.allclose(encoded, bank["goals"][branch], atol=3e-5, rtol=3e-5), "原视觉终点重编码不同")
        references.append(reference)
        scripts.append(script)
        target_telemetry.append(events[-1]["telemetry"])
        native.append([event["native_actions"] for event in events[33:49]])
    visual = legacy.read_json(source / case["directory"] / "visual_preprocessing.json")
    gl.require(visual["reference_0"] == visual["reference_1"] == previous["visual_preprocessing"] and
               pilot.action_names() == previous["action_names"] and
               len(previous["action_names"]) == runtime.bundle["config"]["num_actions"], "视觉处理或原动作接口不同")
    plan = dict(format=calibration.FORMAT, comparison_protocol=calibration.PROTOCOL,
        source_evaluation=str(baseline.project_path(args.eval_dir)),
        source_evaluation_id=previous_manifest["evaluation_id"], source_design_id=previous["design_id"],
        source_evaluation_manifest_sha256=bc.file_hash(baseline.project_path(args.eval_dir) / "evaluation_manifest.json"),
        source_evaluation_report_sha256=bc.file_hash(baseline.project_path(args.eval_dir) / "report.json"),
        source_benchmark=str(source), source_benchmark_id=source_manifest["benchmark_id"],
        source_manifest_sha256=previous["source_manifest_sha256"], source_report_sha256=previous["source_report_sha256"],
        input_identity=report.data["input_identity"], evaluation_code=code_identity(),
        environment_fingerprint=report.data["environment_fingerprint"], scenario=previous["scenario"],
        visual_preprocessing=previous["visual_preprocessing"], action_names=previous["action_names"],
        warmup_steps=32, prefix_actions=previous["prefix_actions"], start_actions=previous["start_actions"],
        control_start_frame=64, horizon=16, scripts=scripts, reference_native_events=native,
        old_reference_control_start_frame=32, target_telemetry=target_telemetry,
        target_content_id=content, goal_source="unchanged real RGB/heatmap endpoints from accepted adopt",
        repetitions=2, randomization_seed=args.randomization_seed,
        schedule=calibration.schedule(case["seed"], args.randomization_seed), maximum_new_env_steps=320,
        execution_policy="fixed_script", worker_control_actions=0,
        same_hidden_state=False, matched_start_comparison=False, old_failures_unchanged=True)
    calibration.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(comparison_protocol=calibration.PROTOCOL, execution_policy="fixed_script",
        evaluation_code=plan["evaluation_code"], design_id=plan["design_id"], source_evaluation_id=previous_manifest["evaluation_id"],
        source_benchmark_id=source_manifest["benchmark_id"], planned_runs=4, maximum_new_env_steps=320,
        attempted_env_steps=0, worker_control_actions=0, script_actions=0, model_loaded=True,
        old_failures_unchanged=True, goal_relabeling=False)
    report.check("historical_identity", "PASS", "完整sample试跑及全部真实产物SHA256、原模型/代码/check一致；原结果不改写")
    report.check("fixed_reference_scripts", "PASS", "action[33:49]两个真实16步脚本；在新64帧起点执行；原目标不替换")
    report.check("fixed_visual_targets", "PASS", "两固定RGB/heatmap终点内容及重编码一致；原终点位置仅用于诊断")
    if args.command == "probe":
        checked_dir = baseline.project_path(args.check_dir)
        checked = legacy.read_json(checked_dir / "report.json")
        legacy.accepted(checked, "check")
        gl.require(checked.get("comparison_protocol") == calibration.PROTOCOL and
                   checked.get("design_id") == plan["design_id"] and
                   legacy.read_json(checked_dir / "design.json") == plan and
                   history.passed_checks(checked, ("historical_identity", "fixed_reference_scripts", "fixed_visual_targets",
                        "causal_history_interface", "calibration_contracts", "trial_output_preflight", "predeclared_design",
                        "no_training", "source_inputs_unchanged")) and
                   checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz"),
                   "需要同一模型、脚本、目标、预算、顺序及当前代码的标定check")
        with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
            gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "标定check的目标改变")
        report.check("design_identity", "PASS", "四次固定顺序、脚本、目标、模型/缓存/环境与当前check一致")
    return cache, runtime, plan, bank, references


def check(args, report, cache, runtime, plan, bank, references):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    import goal_reference_calibration as calibration

    output = pilot.prepare_trial_directory(report.directory / "trial_output_probe")
    marker = dict(output_probe_only=True, before_environment=True)
    for name in ("start.json", "history_check.json", "metrics.json", "script_trace.json"):
        baseline.write_json(output / name, marker)
        gl.require(legacy.read_json(output / name) == marker, "原子JSON写入/读取不一致")
    try:
        pilot.prepare_trial_directory(output)
    except FileExistsError:
        pass
    else:
        raise ValueError("已有trial目录没有拒绝覆盖")
    report.check("trial_output_preflight", "PASS", "启动环境前创建trial目录并验证原子JSON读写；已有目录拒绝覆盖")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "因果历史数据集身份不同")
    rng = bc.capture_rng(args.device)
    for index in sorted({0, len(dataset.metadata["episodes"]) - 1}):
        ep = dataset.episode(index)
        stop = min(32, len(ep["image"]) - 1)
        expected = runtime.state_encoder.rollout(ep, stop)
        state, recovered = None, []
        for frame in range(stop + 1):
            obs = {name: ep[name][frame] for name in ctl.OBS_KEYS}
            state, feature = runtime.state_encoder.step(obs, ep["action"][frame], state)
            recovered.append(feature[0])
        t02.vector_difference(report, f"incremental_episode_{index}", torch.stack(recovered), expected, check=True)
        reset = {name: ep[name][0] for name in ctl.OBS_KEYS}
        _, restarted = runtime.state_encoder.step(reset, ep["action"][0], state)
        t02.vector_difference(report, f"reset_episode_{index}", restarted, expected[:1], check=True)
        t02.rejection(report, "missing_history_guard", lambda: runtime.state_encoder.step(dict(reset, is_first=False), ep["action"][1]), "本局历史")
        t02.rejection(report, "reset_action_guard", lambda: runtime.state_encoder.step(reset, ep["action"][1]), "必须为零")
    for branch, reference in enumerate(references):
        expected = runtime.state_encoder.rollout(reference, len(reference["image"]) - 1)
        t02.vector_difference(report, f"old_reference_history_{branch}", expected,
                              torch.as_tensor(reference["features"], device=args.device), check=True)
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "因果历史验收推进随机数")
    report.check("causal_history_interface", "PASS", "两局增量/reset及两个原真实参考完整历史一致；不执行worker动作")

    first, dimension, script = 64, len(plan["action_names"]), plan["scripts"][0]
    size = first + len(script) + 1
    actions = plan["start_actions"] + script
    arrays = dict(image=np.zeros((size, 64, 64, 3), np.uint8), heatmap=np.zeros((size, 64, 64), np.uint8),
        features=np.zeros((size, runtime.bundle["feature_dim"]), np.float32),
        action=np.concatenate((np.zeros((1, dimension), np.float32), np.eye(dimension, dtype=np.float32)[actions])),
        obs_reward=np.zeros((size, 1), np.float32), is_first=np.arange(size) == 0,
        is_last=np.zeros(size, bool), is_terminal=np.zeros(size, bool))
    telemetry = copy.deepcopy(plan["target_telemetry"][0])
    events = [dict(frame=frame, reward=0., done=False, error=None, telemetry=copy.deepcopy(telemetry),
                   native_actions=[] if frame == 0 else [dict(camera=np.zeros(2), forward=np.array(0))])
              for frame in range(size)]
    random_control.validate_prefix(arrays, events, plan["start_actions"], dimension)
    gl.require(calibration.validate_execution(arrays, events, plan["start_actions"], script, dimension) == 16,
               "完整脚本接口不一致")
    wrong = copy.deepcopy(arrays)
    wrong["action"][65] = np.roll(wrong["action"][65], 1)
    t02.rejection(report, "next_action_guard", lambda: calibration.validate_execution(wrong, events, plan["start_actions"], script, dimension), "incoming")
    bad_events = copy.deepcopy(events)
    del bad_events[65]["native_actions"]
    t02.rejection(report, "missing_native_guard", lambda: calibration.validate_execution(arrays, bad_events, plan["start_actions"], script, dimension), "原生事件")
    missing_pose = copy.deepcopy(events)
    del missing_pose[65]["telemetry"]["pose"]
    t02.rejection(report, "missing_telemetry_guard", lambda: calibration.validate_execution(arrays, missing_pose, plan["start_actions"], script, dimension), "遥测缺失")
    fake_reset = copy.deepcopy(arrays)
    fake_reset["is_first"][65] = True
    t02.rejection(report, "fake_reset_guard", lambda: calibration.validate_execution(fake_reset, events, plan["start_actions"], script, dimension), "伪reset")
    early = {name: value[:67].copy() for name, value in arrays.items()}
    early_events = copy.deepcopy(events[:67])
    early["is_last"][-1], early_events[-1]["done"] = True, True
    gl.require(calibration.validate_execution(early, early_events, plan["start_actions"], script, dimension) == 2,
               "真实提前结束不应补步")
    wrong = copy.deepcopy(arrays)
    wrong["is_last"][66] = True
    t02.rejection(report, "after_terminal_guard", lambda: calibration.validate_execution(wrong, events, plan["start_actions"], script, dimension), "真实结束")
    moved = copy.deepcopy(events)
    moved[-1]["telemetry"]["pose"]["x"] += 10.
    different = copy.deepcopy(arrays)
    different["image"] += 100
    different["features"] += 10
    calibration.validate_execution(different, moved, plan["start_actions"], script, dimension)
    gl.require(calibration.native_actions_equal(events[65]["native_actions"], calibration.plain(events[65]["native_actions"])),
               "实时NumPy/JSON原生事件不一致")
    for target in (0, 1):
        own = calibration.measure(bank["images"][target], bank["heatmaps"][target], bank["goals"][target],
                                  plan["target_telemetry"][target], bank, plan["target_telemetry"])
        gl.require(own[f"distance_{target}"] < 1e-5 and own[f"rgb_mae_{target}"] == 0 and
                   own[f"position_error_{target}"] == 0, "目标自身的视觉/物理误差不为零")
    drift = copy.deepcopy(plan)
    drift["scripts"][0][0] = (script[0] + 1) % dimension
    gl.require(gl.tensor_digest({}, {key: value for key, value in drift.items() if key != "design_id"}) != plan["design_id"],
               "脚本改变没有改变计划身份")
    bad_plan = copy.deepcopy(plan)
    bad_plan["schedule"] = list(reversed(plan["schedule"]))
    t02.rejection(report, "schedule_guard", lambda: calibration.validate_plan(bad_plan), "顺序")
    rows = [dict(cell, execution_valid=False, error="in-memory guard only") for cell in plan["schedule"]]
    gl.require(calibration.summarize(rows, plan)["complete_scripts"] == 0, "失败被当作有效终点")
    t02.rejection(report, "duplicate_run_guard", lambda: calibration.summarize(rows + rows[:1], plan), "重复")
    report.check("calibration_contracts", "PASS", "固定下一动作/真实结束/原生事件/计划守卫通过；跨局漂移仅诊断；内存标记不写成轨迹")
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("predeclared_design", "PASS", "两脚本×两重复=4次；32noop+32前缀+16固定动作，最多320步，无worker选择")
    report.check("scope", "WARN", "仅离线标定接口验收；真实终点尚未执行，不批准目标可达或T06")


def run_reference(report, runtime, plan, bank, cell, directory):
    import numpy as np
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    import goal_reference_calibration as calibration
    import goal_segments as segments

    pilot.prepare_trial_directory(directory)
    session, frames, issued, in_script = None, [], [], False
    first, script_index = plan["control_start_frame"], cell["script"]
    script = plan["scripts"][script_index]
    row = dict(cell, artifact_dir=str(directory), execution_valid=False, script_completed=False,
               actual_script_steps=0, worker_control_actions=0, error=None)
    trace = dict(execution_policy="fixed_script", script=script_index, start_frame=first,
                 planned_action_ids=script, sampled_actions=False, policy_queried=False)
    before = report.data["new_env_steps"]

    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
        if in_script:
            report.data["script_actions"] += 1

    def execute(action):
        report.data["attempted_env_steps"] += 1
        session.step(action)

    def record_measurement():
        feature = ctl.encode_goal(runtime, session.observation)
        result = calibration.measure(session.observation["image"], session.observation["heatmap"], feature,
                                     session.events[-1]["telemetry"], bank, plan["target_telemetry"])
        result.update(frame=len(session.rows) - 1, script_step=len(session.rows) - first - 1,
                      incoming_action=None if len(session.rows) == first + 1 else script[len(session.rows) - first - 2],
                      own_pose=calibration.plain(session.events[-1]["telemetry"]["pose"]),
                      goal_feature=feature.tolist())
        frames.append(result)

    try:
        print(f"[ENV] calibration repeat={cell['repeat']} script={script_index} "
              f"world_seed={plan['scenario']['world_seed']}; fresh reset", flush=True)
        session = ctl.Session(runtime, plan["scenario"], directory / "environment", on_step)
        gl.require(session.specs["visual_preprocessing"] == plan["visual_preprocessing"], "环境视觉处理不同")
        for action in plan["start_actions"]:
            execute(action)
            gl.require(not session.done, "真实环境在预热/前缀完成前结束；保留该尝试，不补跑")
        random_control.validate_prefix(session.arrays(), session.events, plan["start_actions"], len(plan["action_names"]))
        record_measurement()
        row["start"] = {name: value for name, value in frames[0].items() if name != "goal_feature"}
        baseline.write_json(directory / "start.json", dict(start_frame=first, own_real_history=True,
            comparison_to_reference_required=False, telemetry=calibration.plain(session.events[-1]["telemetry"]),
            measurement=frames[0]))
        in_script = True
        for action in script:
            issued.append(action)
            execute(action)
            record_measurement()
            if session.done:
                break
        arrays = session.arrays()
        segments.validate_episode(arrays, len(plan["action_names"]))
        count = calibration.validate_execution(arrays, session.events, plan["start_actions"], script, len(plan["action_names"]))
        recovered = runtime.state_encoder.rollout(arrays, len(session.rows) - 1).cpu().numpy()
        maximum_error = float(np.max(np.abs(recovered - arrays["features"])))
        gl.require(np.allclose(recovered, arrays["features"], atol=3e-5, rtol=3e-5), "本局完整真实因果历史重算不一致")
        baseline.write_json(directory / "history_check.json", dict(own_causal_history_passed=True,
            maximum_absolute_error=maximum_error, total_frames=len(session.rows), start_frame=first,
            comparison_to_other_histories=False, reset_is_real=True))
        # Native actions depend on the wrapper's real prior history (including
        # sticky actions). Preserve their differences; macro scripts stay fixed.
        native = [event["native_actions"] for event in session.events[first + 1:]]
        native_equal = [calibration.native_actions_equal(current, old) for current, old in
                        zip(native, plan["reference_native_events"][script_index])]
        end = {name: value for name, value in frames[-1].items() if name != "goal_feature"}
        margin = end[f"distance_{1 - script_index}"] - end[f"distance_{script_index}"]
        initial_margin = frames[0][f"distance_{1 - script_index}"] - frames[0][f"distance_{script_index}"]
        row.update(execution_valid=True, script_completed=count == len(script), actual_script_steps=count,
            endpoint=end, distance_improvement=frames[0][f"distance_{script_index}"] - end[f"distance_{script_index}"],
            target_preference_margin=margin, preference_improvement=margin - initial_margin,
            minimum_assigned_distance_diagnostic=min(frame[f"distance_{script_index}"] for frame in frames),
            minimum_distance_is_not_endpoint=True, own_history_max_error=maximum_error,
            native_steps_equal_to_old_reference=native_equal,
            movement_from_start=calibration.pose_difference(session.events[-1]["telemetry"]["pose"], session.events[first]["telemetry"]["pose"]),
            end_reason="environment_done" if session.done else "fixed_budget")
        gl.require(len(frames) == count + 1, "逐帧标定遗漏真实脚本观测")
    except Exception as error:
        row.update(execution_valid=False, script_completed=False, error=f"{type(error).__name__}: {error}")
        baseline.write_json(directory / "error.json", dict(error=row["error"], traceback=traceback.format_exc()))
        if isinstance(error, OSError):
            raise
        report.check(f"run_{cell['repeat']}_{script_index}_execution", "FAIL", row["error"] + "；保留该尝试，继续固定顺序，不重试")
    finally:
        row["returned_env_steps"] = report.data["new_env_steps"] - before
        trace.update(attempted_action_ids=issued, measurements=frames)
        if session is not None:
            row.update(actual_script_steps=max(0, len(session.rows) - first - 1),
                task_success_during_script=any(event["success"] for event in session.events[first + 1:]),
                external_return_during_script=float(sum(event["reward"] for event in session.events[first + 1:])))
            trace["actual_native_events"] = calibration.plain([event["native_actions"] for event in session.events[first + 1:]])
            try:
                row["video"] = ctl.save_session(session, directory, baseline.write_video)
                with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
                    actual = session.arrays()
                    gl.require(set(saved.files) == set(actual) - {"telemetry"} and
                               all(np.array_equal(saved[name], actual[name]) for name in saved.files),
                               "保存的真实历史与实时数组不同")
                gl.require(calibration.native_actions_equal(session.events, legacy.read_json(directory / "events.json")),
                           "保存的JSON事件与实时NumPy事件不同")
                row["saved_history_roundtrip"] = True
                if row["execution_valid"]:
                    saved_check = legacy.read_json(directory / "history_check.json")
                    saved_check["saved_history_roundtrip"] = True
                    baseline.write_json(directory / "history_check.json", saved_check)
            finally:
                session.close()
                del session
                gc.collect()
        baseline.write_json(directory / "metrics.json", calibration.plain(row))
        baseline.write_json(directory / "script_trace.json", trace)
    return row, frames


def galleries(directory, plan, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 720, 175
    grid = Image.new("RGB", (width, (len(rows) + 1) * height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((5, 5), "Fixed old targets; all attempts kept; start and end are each run's own history", fill="black")
    for target in (0, 1):
        x = 270 + target * 175
        draw.text((x, 25), f"old visual target {target}", fill="black")
        grid.paste(Image.fromarray(bank["images"][target]).resize((128, 128)), (x, 40))
    matrix = Image.new("RGB", (720, 55 + len(rows) * 75), "white")
    md = ImageDraw.Draw(matrix)
    md.text((5, 5), "Final cosine distances to fixed targets (lower is closer); no success threshold", fill="black")
    md.text((240, 28), "target 0", fill="black")
    md.text((415, 28), "target 1", fill="black")
    for index, row in enumerate(rows):
        y = (index + 1) * height
        label = f"repeat {row['repeat']} / script {row['script']}"
        draw.text((5, y + 5), label, fill="black")
        artifact = Path(row["artifact_dir"])
        if (artifact / "trajectory.npz").is_file():
            # A failed run can contain useful partial frames with invalid
            # execution metadata. Display pixels without accepting that history.
            with np.load(artifact / "trajectory.npz", allow_pickle=False) as saved:
                images = saved["image"].copy()
            first = plan["control_start_frame"]
            if len(images) > first:
                draw.text((270, y + 5), "own script start", fill="black")
                grid.paste(Image.fromarray(images[first]).resize((128, 128)), (270, y + 25))
            draw.text((445, y + 5), "actual end", fill="black")
            grid.paste(Image.fromarray(images[-1]).resize((128, 128)), (445, y + 25))
        status = "complete 16-step endpoint" if row["script_completed"] else "early terminal" if row["execution_valid"] else "execution error"
        draw.text((5, y + 30), status, fill="black")
        md.text((5, 65 + index * 75), label, fill="black")
        for target in (0, 1):
            x, yy = 235 + target * 175, 55 + index * 75
            value = row.get("endpoint", {}).get(f"distance_{target}")
            shade = int(np.clip(value / 2 * 200, 0, 200)) if value is not None else 0
            md.rectangle((x, yy, x + 155, yy + 60), fill=(255, 255 - shade, 255 - shade))
            md.text((x + 5, yy + 8), f"{value:.6f}" if value is not None else "missing", fill="black")
            md.text((x + 5, yy + 30), "diagnostic only" if not row["script_completed"] else "fixed endpoint", fill="black")
    grid.save(directory / "starts_and_outcomes.png")
    matrix.save(directory / "endpoint_matrix.png")


def probe(args, report, cache, runtime, plan, bank, references):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_calibration as calibration

    baseline.write_json(report.directory / "design.json", plan)
    baseline.video_preflight(report)
    bc.require_disk_space(report.directory, 4 * 32 * 1024**2 + 16 * 1024**2)
    report.check("storage_preflight", "PASS", "预留4次真实历史/录像空间；不复制模型、缓存或旧视频")
    report.check("predeclared_runs", "PASS", "两脚本各两次，最多320步；独立fresh reset，无worker选择或起点配对筛选")
    rows, all_frames = [], []
    for index, cell in enumerate(plan["schedule"]):
        report.require_writable()
        output = report.directory / f"run_{index:03d}_repeat_{cell['repeat']}_script_{cell['script']}"
        row, frames = run_reference(report, runtime, plan, bank, cell, output)
        rows.append(row)
        all_frames += [dict(cell, **{name: value for name, value in frame.items() if name not in ("goal_feature", "own_pose")})
                       for frame in frames]
        t04.append_json(report.directory / "runs.jsonl", row)
        baseline.write_json(report.directory / "runs.json", rows)
        baseline.write_json(report.directory / "diagnostics.json", calibration.summarize(rows, plan))
        report.data["attempted_runs"] = len(rows)
        report.save()
        detail = (f"steps={row['actual_script_steps']} d0={row['endpoint']['distance_0']:.5f} "
                  f"d1={row['endpoint']['distance_1']:.5f}") if row["execution_valid"] else row["error"]
        print(f"[REPLAY {index + 1}/4] repeat={cell['repeat']} script={cell['script']} {detail}", flush=True)
    summary = calibration.summarize(rows, plan)
    summary["old_targets_pose_difference"] = calibration.pose_difference(plan["target_telemetry"][0]["pose"], plan["target_telemetry"][1]["pose"])
    summary["old_targets_visual_separation"] = float(1 - bank["goals"][0] @ bank["goals"][1])
    summary["start_measurements"] = [dict(seed=row["seed"], repeat=row["repeat"], script=row["script"], **row["start"])
                                     for row in rows if "start" in row]
    summary["cross_script_endpoints"] = []
    for repeat in range(2):
        pair = [row for row in rows if row["repeat"] == repeat and row["execution_valid"] and row["script_completed"]]
        if len(pair) == 2:
            vectors = [legacy.read_json(Path(row["artifact_dir"]) / "script_trace.json")["measurements"][-1]["goal_feature"] for row in pair]
            summary["cross_script_endpoints"].append(dict(repeat=repeat, script_order=[row["script"] for row in pair],
                visual_separation=float(np.clip(1 - np.asarray(vectors[0]) @ np.asarray(vectors[1]), 0, 2)),
                pose_difference=calibration.pose_difference(pair[0]["endpoint"]["own_pose"], pair[1]["endpoint"]["own_pose"]),
                comparison="different real starts, diagnostic only"))
    baseline.write_json(report.directory / "diagnostics.json", summary)
    baseline.write_csv(report.directory / "frame_metrics.csv", all_frames)
    table = []
    for row in rows:
        entry = {name: row.get(name) for name in (
            "seed", "repeat", "script", "execution_valid", "script_completed", "actual_script_steps", "returned_env_steps",
            "distance_improvement", "target_preference_margin", "preference_improvement",
            "task_success_during_script", "external_return_during_script", "error", "artifact_dir")}
        entry.update({f"end_distance_{target}": row.get("endpoint", {}).get(f"distance_{target}") for target in (0, 1)})
        table.append(entry)
    baseline.write_csv(report.directory / "runs.csv", table)
    galleries(report.directory, plan, bank, rows)
    saved = dict(format=calibration.EVALUATION_FORMAT, comparison_protocol=calibration.PROTOCOL, design=plan,
        design_id=plan["design_id"], rows=rows, new_env_steps=report.data["new_env_steps"],
        attempted_env_steps=report.data["attempted_env_steps"], script_actions=report.data["script_actions"], worker_control_actions=0,
        # Simulator logs remain available for debugging, outside the immutable
        # measurement manifest. Do not hash changing process/world internals.
        artifacts={str(path.relative_to(report.directory)): bc.file_hash(path)
                   for parent in [report.directory, *(Path(row["artifact_dir"]) for row in rows)]
                   for path in parent.iterdir() if path.is_file() and path.name not in ("report.json", "calibration_manifest.json")})
    saved["evaluation_id"] = gl.tensor_digest({}, saved)
    baseline.write_json(report.directory / "calibration_manifest.json", saved)
    report.data.update(evaluation_id=saved["evaluation_id"], valid_histories=summary["valid_histories"],
        complete_scripts=summary["complete_scripts"], full_design_completed=summary["full_design_completed"])
    report.check("planned_attempts", "PASS", "固定随机顺序尝试4/4；全部保留，无重试、替换或补样")
    report.check("real_script_interface", "PASS" if summary["valid_histories"] == 4 else "FAIL",
                 f"自身真实动作/原生事件/遥测/因果历史有效{summary['valid_histories']}/4；不要求跨局相等")
    report.check("fixed_horizon_endpoints", "PASS" if summary["complete_scripts"] == 4 else "WARN",
                 f"完整16步终点{summary['complete_scripts']}/4；提前结束单列，不用最佳中间帧替代终点")
    report.check("behavior_acceptance", "WARN", "标定数值需结合视频、位置/朝向及两个脚本方向分析；工程PASS不批准worker控制或T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, probing = commands.add_parser("check"), commands.add_parser("probe")
    probing.add_argument("--check-dir", required=True)
    for command in (checking, probing):
        command.add_argument("--eval-dir", required=True, help="完整已通过的sample试跑目录")
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--residual-verify-dir", required=True)
        command.add_argument("--randomization-seed", type=int, default=0)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
    args = parser.parse_args()
    if not 0 <= args.randomization_seed < 2**31 - 10000:
        parser.error("随机顺序种子无效")
    if args.command == "probe" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        historical = history.historical_inputs(args)
        protected, files, source = source_paths(args, historical)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"输入身份/清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "reference_calibration_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、缓存、回放、目标来源、sample评估和check")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    os.chdir(ROOT)
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        gl.require(device.type == "cuda" and torch.cuda.is_available(), "沿用sample的CUDA推理环境；真实MineCLIP路径需要CUDA")
        torch.cuda.set_device(device)
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化模型变化")
        for episode in source["episodes"]:
            gl.require(baseline.file_signature(Path(episode["path"])) == episode["signature"], "原真实回放变化")
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            cache, runtime, plan, bank, references = load_inputs(args, report, historical)
            versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
            digest = gl.tensor_digest(runtime.state_dict(), {})
            {"check": check, "probe": probe}[args.command](args, report, cache, runtime, plan, bank, references)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()),
                   "标定改变冻结参数或产生梯度")
        report.check("no_training", "PASS", f"optimizer_updates=0；参数未变；worker控制动作0；"
                     f"env.step={report.data['new_env_steps']}（含预热/前缀/固定脚本）")
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
                             "源模型/回放/目标/原sample/check大小及修改时间未变；历史另按SHA256验真；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
