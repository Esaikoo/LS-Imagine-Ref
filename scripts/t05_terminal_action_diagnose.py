"""Diagnose the final two actions of all 30 repaired-worker real histories.

Uses saved causal states and the same frozen version 300, never a new
rollout, optimizer, RSSM constructor, goal, budget or success criterion.
"""

import argparse
import copy
from datetime import datetime
import gc
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t05_action_choice_diagnose as history
import t05_goal_control as legacy
import t05_repaired_control as confirmed


DIAGNOSIS_CODE_FILES = ("goal_terminal_action_diagnose.py", "scripts/t05_terminal_action_diagnose.py",
                        "goal_action_choice_diagnose.py", "scripts/t05_action_choice_diagnose.py")


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_TERMINAL_ACTION_DIAGNOSE; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_repair as repair
    import goal_repaired_control as confirmation

    directory = baseline.project_path(args.eval_dir)
    record = legacy.read_json(directory / "report.json")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    legacy.accepted(record, "evaluate")
    plan, rows = manifest["design"], manifest["rows"]
    confirmation.validate_plan(plan)
    summary = confirmation.summarize(rows, plan)
    gl.require(manifest.get("format") == confirmation.EVALUATION_FORMAT and
               manifest.get("comparison_protocol") == record.get("comparison_protocol") == confirmation.PROTOCOL and
               manifest.get("evaluation_id") == record.get("evaluation_id") == gl.tensor_digest({}, {
                   key: value for key, value in manifest.items() if key != "evaluation_id"}) and
               manifest.get("design_id") == record.get("design_id") == plan.get("design_id") == gl.tensor_digest({}, {
                   key: value for key, value in plan.items() if key != "design_id"}) and
               legacy.read_json(directory / "design.json") == plan and
               record.get("input_identity") == manifest.get("input_identity") == plan["input_identity"] and
               record.get("evaluation_code") == plan["evaluation_code"] and
               record.get("environment_fingerprint") == plan["environment_fingerprint"],
               "需要原完整修复自主确认及未改变的模型/计划/代码身份")
    gl.require(record.get("full_design_completed") is True and len(rows) == 30 and
               record.get("planned_trials") == record.get("attempted_trials") == record.get("execution_valid_trials") == 30 and
               all(row["execution_valid"] and row.get("fixed_horizon_completed") is True and
                   row["actual_steps"] == 16 and row["returned_env_steps"] == 80 for row in rows) and
               manifest.get("worker_control_actions") == record.get("worker_control_actions") == 480 and
               manifest.get("new_env_steps") == record.get("new_env_steps") == 2400 and
               record.get("optimizer_updates") == 0 and
               record.get("numerical_gate") == summary["numerical_gate"] == legacy.read_json(directory / "gate.json") and
               legacy.read_json(directory / "diagnostics.json")["numerical_gate"] == summary["numerical_gate"] and
               manifest.get("behavior_accepted") is False and manifest.get("t06_approved") is False and
               confirmed.passed_checks(record, ("strict_repaired_inference_load", "design_identity",
                   "planned_attempts", "real_control_interface", "no_training", "source_inputs_unchanged")),
               "需要全部30次工程有效完整历史；数值失败可诊断，但不能筛选子集或改写门槛")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(manifest["check_report_sha256"] == bc.file_hash(checked_dir / "report.json") and
               checked.get("design_id") == plan["design_id"] and checked.get("input_identity") == plan["input_identity"] and
               checked.get("evaluation_code") == plan["evaluation_code"] and
               legacy.read_json(checked_dir / "design.json") == plan and
               checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz") and
               confirmed.passed_checks(checked, ("strict_repaired_inference_load", "fixed_visual_targets",
                   "causal_execution_interface", "confirmation_gate_contract", "saved_execution_contract",
                   "predeclared_design", "no_training", "source_inputs_unchanged")), "原确认check或固定目标被修改")
    code_files = []
    for name, digest in plan["evaluation_code"].items():
        path = (ROOT / name).resolve()
        gl.require(ROOT in path.parents and path.is_file() and bc.file_hash(path) == digest,
                   f"原推理代码被修改：{name}；新诊断只新增文件，不改旧指纹")
        code_files.append(path)
    required = set()
    for index, row in enumerate(rows):
        name = f"trial_{index:03d}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}"
        gl.require(baseline.project_path(row["artifact_dir"]) == directory / name, "trial不属于本次完整确认")
        required.update(f"{name}/{item}" for item in ("trajectory.npz", "events.json", "metrics.json",
            "control_trace.json", "frame_metrics.json", "start.json", "history_check.json", "video.mp4"))
    gl.require(required <= set(manifest["artifacts"]) and
               {"design.json", "targets.npz", "trials.json", "trials.csv", "diagnostics.json", "gate.json"}
               <= set(manifest["summary_artifacts"]) and legacy.read_json(directory / "trials.json") == rows,
               "真实历史、状态、动作、视频或原汇总缺少SHA256绑定")
    files = [directory / "report.json", directory / "evaluation_manifest.json"]
    for table in (manifest["artifacts"], manifest["summary_artifacts"]):
        for name, digest in table.items():
            path = (directory / name).resolve()
            gl.require(directory in path.parents and not Path(name).is_absolute() and path.is_file() and
                       bc.file_hash(path) == digest, f"原真实产物缺失或SHA256改变：{name}")
            files.append(path)
    identity = plan["input_identity"]
    checkpoint = baseline.project_path(record["arguments"]["checkpoint"])
    cache_dir = baseline.project_path(record["arguments"]["cache_dir"])
    verify_dir = baseline.project_path(record["arguments"]["repair_verify_dir"])
    gl.require(identity["checkpoint"] == baseline.file_signature(checkpoint) and
               identity["checkpoint_sha256"] == bc.file_hash(checkpoint), "原确认checkpoint已被修改")
    payload, dependencies = repair.read_checkpoint(checkpoint)
    verification = legacy.read_json(verify_dir / "report.json")
    legacy.accepted(verification, "verify")
    gl.require(identity["checkpoint_format"] == verification.get("repair_format") == repair.FORMAT and
               identity["repair_verify_sha256"] == bc.file_hash(verify_dir / "report.json") and
               identity["input_identity"] == verification.get("repair_identity") == verification.get("input_identity") == payload["input_identity"] and
               identity["counters"] == verification.get("counters") == payload["counters"] and
               identity["dependencies"] == payload["dependencies"] and
               identity["model_id"] == verification.get("model_id") and
               identity["worker_version"] == verification.get("worker_version") == 300 and identity["repair_updates"] == 100 and
               verification.get("backend") == identity["inference_backend"] and
               verification.get("source_inputs_before", {}).get(str(checkpoint)) == identity["checkpoint"] and
               confirmed.passed_checks(verification, ("strict_load", "inference_artifact_guard", "roundtrip",
                    "next_update_equivalence", "frozen_dependencies", "source_inputs_unchanged")),
               "需要原worker300和同一独立修复verify，不能选旧版本或验收快照")
    files += [checkpoint, verify_dir / "report.json", *dependencies.values(), *code_files]
    files += [cache_dir / name for name in ("cache_manifest.json", "tables.npz", "states.npy", "frozen_bundle.pt", "report.json")]
    files += [checked_dir / name for name in ("report.json", "design.json", "targets.npz")]
    files += [ROOT / name for name in DIAGNOSIS_CODE_FILES]
    calibration = legacy.read_json(dependencies["calibration_manifest"])
    import goal_reference_calibration as calibration_rules
    calibration_rules.validate_plan(calibration["design"])
    reference_summary = calibration_rules.summarize(calibration["rows"], calibration["design"])
    gl.require(calibration.get("format") == calibration_rules.EVALUATION_FORMAT and
               calibration.get("evaluation_id") == gl.tensor_digest({}, {
                   name: value for name, value in calibration.items() if name != "evaluation_id"}) and
               len(calibration["rows"]) == 4 and
               reference_summary["valid_histories"] == reference_summary["complete_scripts"] == 4 and
               calibration["design"]["target_content_id"] == plan["target_content_id"] and
               calibration["design"]["start_actions"] == plan["start_actions"] and
               calibration["design"]["input_identity"]["input_identity"]["bundle_id"] == identity["input_identity"]["bundle_id"],
               "修复依赖的四条参考历史、固定目标或初始化来源不同")
    calibration_dir = dependencies["calibration_manifest"].parent
    for index, run in enumerate(calibration["rows"]):
        name = f"run_{index:03d}_repeat_{run['repeat']}_script_{run['script']}"
        gl.require(run["execution_valid"] is True and run["script_completed"] is True and
                   run["actual_script_steps"] == 16 and baseline.project_path(run["artifact_dir"]) == calibration_dir / name,
                   "参考对照必须保留四条完整真实固定脚本历史")
        for item in ("trajectory.npz", "events.json", "script_trace.json", "history_check.json", "metrics.json"):
            relative = f"{name}/{item}"
            path = (calibration_dir / relative).resolve()
            gl.require(relative in calibration["artifacts"] and path.is_file() and
                       bc.file_hash(path) == calibration["artifacts"][relative], "参考真实状态、事件或动作SHA256不同")
            files.append(path)
    protected = [directory, checked_dir, cache_dir, checkpoint.parent, verify_dir, Path(plan["source_benchmark"])]
    protected += [path.parent for path in dependencies.values()]
    files = list(dict.fromkeys(path.resolve() for path in files))
    gl.require(all(path.is_file() for path in files), "保留服务器原缓存、模型、verify和真实轨迹，输入依赖缺失")
    return record, manifest, checked_dir, checkpoint, cache_dir, calibration, protected, files


def contract_checks(report):
    import numpy as np
    import goal_action_choice_diagnose as choices
    import goal_library as gl
    import goal_terminal_action_diagnose as diagnosis

    p = {name: np.array([.1, .8, .1]) for name in choices.CONDITIONS}
    p["goal1"] = np.array([.2, .7, .1])
    observed = diagnosis.action_metrics(p, 1, 1)
    gl.require(observed["correct_mode"] == observed["swapped_mode"] == 1 and
               observed["correct_observed_probability"] < observed["swapped_observed_probability"] and
               not choices.distribution_difference(p["goal0"], p["goal1"])["mode_changed"],
               "概率作用与mode改变不能混为一谈")
    pre = dict(distance_0=.43, distance_1=.03, pitch_error_1=0., position_error_1=.15)
    post = dict(distance_0=.02, distance_1=.49, pitch_error_1=10., position_error_1=.26)
    step = diagnosis.transition(pre, post, 1)
    gl.require(step["preference_reversed_away"] and step["actual_step_progress"] < 0 and
               step["pitch_error_before"] == 0 and step["pitch_error_after"] == 10,
               "末步实际偏好翻转必须与事实测量一致")
    # Explicit memory-only markers test indexing; never saved as observations.
    first, horizon, dimension, index = 1, 16, 3, 0
    cell = dict(mode="goal", target=1, repeat=0, randomization_seed=1)
    plan = dict(control_start_frame=first, horizon=horizon, design_id="memory-only",
                comparison_protocol="memory-only", input_identity=dict(model_id="memory-only"))
    arrays = dict(features=np.zeros((18, 5120), np.float32),
        action=np.vstack((np.zeros(3), np.eye(3)[[0] + [1] * 16])).astype(np.float32),
        is_last=np.zeros(18, bool), is_terminal=np.zeros(18, bool))
    trace = dict(action_ids=[1] * 16, remaining=list(range(16, 0, -1)), condition="goal",
        evaluated_target=1, issued_target=1, action_seed=702,
        uniforms=np.random.RandomState(702).uniform(size=16).tolist(),
        probabilities=[p["goal0"].tolist()] * 16, distances_to_both_goals=[[.43, .03]] * 17,
        execution_policy="mode", design_id="memory-only", comparison_protocol="memory-only",
        model_id="memory-only", worker_version=300, repair_updates=100, control_start_frame=first)
    measurements = [dict(frame=first + i, control_step=i, distance_0=.43, distance_1=.03) for i in range(17)]
    def query(a=arrays, t=trace, m=measurements):
        return diagnosis.validate_queries(a, t, m, plan, cell, index, dimension)
    gl.require([(row["frame"], row["next_action_frame"], row["remaining"]) for row in query()] == [(15, 16, 2), (16, 17, 1)],
               "实际末尾两动作的状态/下一动作/预算边界异常")
    wrong = {name: value.copy() for name, value in arrays.items()}
    wrong["action"][-1] = np.eye(3)[0]
    t02.rejection(report, "next_action_guard", lambda: query(a=wrong), "下一动作")
    terminal = {name: value.copy() for name, value in arrays.items()}
    terminal["is_terminal"][-2] = True
    t02.rejection(report, "terminal_guard", lambda: query(a=terminal), "结束")
    bad_budget = copy.deepcopy(trace)
    bad_budget["remaining"][-1] = 2
    t02.rejection(report, "remaining_guard", lambda: query(t=bad_budget), "预算")
    bad_measurement = copy.deepcopy(measurements)
    bad_measurement[-1]["frame"] += 1
    t02.rejection(report, "measurement_guard", lambda: query(m=bad_measurement), "测量")
    bad_mode = copy.deepcopy(trace)
    bad_mode["probabilities"][-1] = [.8, .1, .1]
    t02.rejection(report, "recorded_mode_guard", lambda: query(t=bad_mode), "mode")
    marker_plan = dict(design=dict(schedule=[{}] * 30))
    t02.rejection(report, "missing_query_guard", lambda: diagnosis.summarize([], marker_plan), "遗漏")
    duplicates = [dict(trial_index=0, remaining=1)] * 60
    t02.rejection(report, "duplicate_query_guard", lambda: diagnosis.summarize(duplicates, marker_plan), "重复")
    refs = [dict(assigned_target=target, remaining=1, run_index=target * 2 + repeat, repeat=repeat,
                 observed_script_action_name="memory-only", state=np.ones(5120, dtype=np.float32))
            for target in (0, 1) for repeat in (0, 1)]
    context = diagnosis.reference_context(np.ones(5120, dtype=np.float32), refs, 1)
    gl.require(context["reference_target0_nearest_state_cosine_distance"] < 1e-12 and
               context["reference_target1_nearest_state_cosine_distance"] < 1e-12,
               "同预算状态余弦比较异常")
    t02.rejection(report, "reference_budget_guard", lambda: diagnosis.reference_context(np.ones(5120), refs, 2), "同预算")
    report.check("diagnostic_contracts", "PASS", "事实动作概率/排名、末步偏好翻转与frame→incoming/remaining2–1守卫通过；内存标记不写入数据")


def probability_gallery(path, rows, names):
    from PIL import Image, ImageDraw, ImageFont
    # Five readable rows, all goal-condition target1 repeats, no selection.
    chosen = sorted((row for row in rows if row["factual_condition"] == "goal" and
                     row["evaluated_target"] == 1 and row["remaining"] == 1), key=lambda row: row["repeat"])
    panel = Image.new("RGB", (1160, 155 * len(chosen) + 35), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), "Same saved state, remaining=1 | blue=goal0 orange=goal1 gray=no_goal | observed action is not an expert label", fill="black", font=font)
    for index, row in enumerate(chosen):
        y = 35 + index * 155
        draw.text((8, y), f"repeat={row['repeat']} trial={row['trial_index']} actual={row['observed_action_name']} "
                  f"d1 {row['distance_before']:.4f}->{row['distance_after']:.4f} "
                  f"pitch_error {row['pitch_error_before']:.0f}->{row['pitch_error_after']:.0f} "
                  f"goal_mode_switch={row['goal_mode_changed']}", fill="black", font=font)
        for column, name in enumerate(names):
            x = 12 + column * 95
            draw.line((x, y + 112, x + 76, y + 112), fill="#bbbbbb")
            for offset, condition, color in ((0, "goal0", "#2166ac"), (20, "goal1", "#d95f02"), (40, "no_goal", "#666666")):
                value = row["probabilities"][condition][column]
                height = round(75 * value)
                if height:
                    draw.rectangle((x + offset, y + 112 - height, x + offset + 16, y + 111), fill=color)
            draw.text((x, y + 119), name, fill="black", font=font)
    panel.save(path)


def analyze(args, report, record, manifest, checked_dir, checkpoint, cache_dir, calibration_record):
    import numpy as np
    import torch
    import goal_action_choice_diagnose as choices
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_calibration as calibration
    import goal_reference_repair as repair
    import goal_terminal_action_diagnose as diagnosis

    contract_checks(report)
    plan, identity = manifest["design"], manifest["input_identity"]
    gl.require(bc.backend_info(args.device) == identity["inference_backend"],
               "原概率重现需要同一推理后端；请在原CUDA环境运行，不以另一后端替代")
    cache = bc.TrainingCache(cache_dir)
    policy = repair.load_policy(checkpoint, cache, args.device)
    gl.require(policy.identity == identity["input_identity"] and policy.model_id == identity["model_id"] and
               policy.worker_version == 300 and policy.repair_updates == 100, "需要原真实确认的同一worker300")
    loader_rng = bc.capture_rng(args.device)
    try:
        library = gl.GoalLibrary.from_payload(cache.bundle["library"], args.device)
    finally:
        bc.restore_rng(loader_rng, args.device)
    modules = (policy, library)
    versions = [{name: parameter._version for name, parameter in module.named_parameters()} for module in modules]
    digests = [gl.tensor_digest(module.state_dict(), {}) for module in modules]
    query_rng = bc.capture_rng(args.device)
    with np.load(Path(args.eval_dir) / "targets.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
        gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "原确认与check的目标内容不同")
    gl.require(gl.tensor_digest({}, {name: bank[name].tolist() for name in bank}) == plan["target_content_id"] and
               bank["goals"].shape == (2, policy.identity["goal_dim"]) and bank["goals"].dtype == np.float32,
               "两固定目标内容或维度不同")
    for target in (0, 1):
        obs = {name: torch.as_tensor(bank[key][target:target + 1], device=args.device)
               for name, key in (("image", "images"), ("heatmap", "heatmaps"))}
        gl.require(np.allclose(library(obs)[0].cpu().numpy(), bank["goals"][target], atol=choices.ATOL, rtol=choices.RTOL),
                   "原真实终点目标重编码不同")
    report.data.update(source_evaluation_id=manifest["evaluation_id"], source_design_id=plan["design_id"],
        input_identity=identity, worker_version=300, repair_updates=100, target_content_id=plan["target_content_id"],
        original_numerical_gate=record["numerical_gate"], inference_backend=bc.backend_info(args.device),
        state_source="each SHA256-bound autonomous history; no recomputed or copied RSSM",
        diagnosis_code={name: bc.file_hash(ROOT / name) for name in DIAGNOSIS_CODE_FILES})
    report.check("strict_frozen_load", "PASS", "同一worker300及目标编码器；读取已保存5120维因果状态，无RSSM/候选模型或优化器构造")
    report.check("fixed_visual_targets", "PASS", "原两真实视觉终点与check逐值/内容ID一致并重编码通过；不替换目标或输入坐标")
    references = []
    import goal_reference_action_diagnose as reference_queries
    reference_plan = calibration_record["design"]
    for run_index, run in enumerate(calibration_record["rows"]):
        directory = Path(run["artifact_dir"])
        arrays = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "script_trace.json")
        saved_history = legacy.read_json(directory / "history_check.json")
        script = reference_plan["scripts"][run["script"]]
        gl.require(legacy.read_json(directory / "metrics.json") == run and
                   saved_history.get("own_causal_history_passed") is True and saved_history.get("saved_history_roundtrip") is True and
                   saved_history.get("reset_is_real") is True and saved_history.get("comparison_to_other_histories") is False and
                   saved_history.get("start_frame") == 64 and saved_history.get("total_frames") == 81 and
                   arrays["features"].shape == (81, 5120) and
                   calibration.native_actions_equal(trace["actual_native_events"], [event["native_actions"] for event in events[65:]]) and
                   calibration.validate_execution(arrays, events, reference_plan["start_actions"], script, 12) == 16,
                   "参考对照必须为本局已验收的完整真实因果历史")
        queries = reference_queries.validate_queries(arrays, trace, 64, 16, script, 12)
        for query in queries:
            if query["remaining"] not in diagnosis.FOCUS_REMAINING:
                continue
            state = arrays["features"][query["frame"]:query["frame"] + 1].copy()
            p, raw = history.query(policy, torch.as_tensor(state, device=args.device), bank["goals"], query["remaining"])
            references.append(dict(run_index=run_index, repeat=run["repeat"], assigned_target=run["script"],
                frame=query["frame"], remaining=query["remaining"], state=state.reshape(-1),
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                observed_script_action=query["script_action"], observed_script_action_name=plan["action_names"][query["script_action"]],
                development_role="repair_train" if run["repeat"] == 0 else "inspected_development_validation",
                **choices.frame_metrics(p, raw),
                **diagnosis.action_metrics(p, query["script_action"], run["script"]),
                probabilities={name: values.tolist() for name, values in p.items()}))
    gl.require(len(references) == 8, "末尾参考查询数量异常")
    frames, trials = [], []
    max_probability_error = 0.
    first, horizon, dimension = plan["control_start_frame"], plan["horizon"], len(plan["action_names"])
    for index, row in enumerate(manifest["rows"]):
        directory = Path(row["artifact_dir"])
        gl.require(legacy.read_json(directory / "metrics.json") == row, "原trial指标与manifest不同")
        arrays = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "control_trace.json")
        measurements = legacy.read_json(directory / "frame_metrics.json")
        checked_history = legacy.read_json(directory / "history_check.json")
        gl.require(checked_history.get("own_causal_history_passed") is True and checked_history.get("reset_is_real") is True and
                   checked_history.get("comparison_to_other_histories") is False and
                   checked_history.get("start_frame") == first and checked_history.get("total_frames") == 81 and
                   confirmed.validate_trace(arrays, events, trace, plan, dimension) == 16,
                   "必须使用原SHA256绑定的本局完整因果历史和真实遥测/动作")
        queries = diagnosis.validate_queries(arrays, trace, measurements, plan, plan["design"]["schedule"][index], index, dimension)
        start = legacy.read_json(directory / "start.json")
        gl.require(start.get("control_start_frame") == first and start.get("own_real_history_only") is True and
                   start.get("comparison_to_reference_required") is False and
                   start["telemetry"] == events[first]["telemetry"] and
                   np.array_equal(np.asarray(start["rssm_feature"], dtype=np.float32), arrays["features"][first]) and
                   np.allclose(start["distances_to_both_goals"], trace["distances_to_both_goals"][0],
                               atol=choices.ATOL, rtol=choices.RTOL), "原本局真实起点被修改")
        for step, saved in enumerate(measurements):
            frame = first + step
            obs = {name: torch.as_tensor(arrays[name][frame:frame + 1], device=args.device) for name in ("image", "heatmap")}
            feature = library(obs)[0].cpu().numpy()
            measured = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                            events[frame]["telemetry"], bank, plan["target_telemetry"])
            gl.require(np.allclose(feature, saved["goal_feature"], atol=choices.ATOL, rtol=choices.RTOL) and all(
                saved[name] == value if name.startswith("closest_") else
                np.isclose(saved[name], value, atol=choices.ATOL, rtol=choices.RTOL) for name, value in measured.items()),
                "原逐帧视觉/物理测量不能重现")
        queried = {}
        for offset in range(horizon):
            state = arrays["features"][first + offset:first + offset + 1].copy()
            tensor = torch.as_tensor(state, device=args.device)
            budget = horizon - offset
            factual = "no_goal" if row["mode"] == "no_goal" else f"goal{trace['issued_target']}"
            if budget in diagnosis.FOCUS_REMAINING:
                before = tensor.clone()
                p, raw = history.query(policy, tensor, bank["goals"], budget)
                gl.require(torch.equal(before, tensor), "目标替换改变了本局真实状态")
                queried[budget] = (state, p, raw)
                actual = p[factual]
            else:
                target = 0 if row["mode"] == "no_goal" else trace["issued_target"]
                goal = torch.as_tensor(bank["goals"][target:target + 1], device=args.device)
                remaining = torch.tensor([budget], dtype=torch.int64, device=args.device)
                with gl.encoder_precision(args.device):
                    actual = policy.action_dist(tensor, goal, remaining, "no_goal" if row["mode"] == "no_goal" else "goal").probs[0].cpu().numpy()
            error = float(np.max(np.abs(actual - np.asarray(trace["probabilities"][offset]))))
            max_probability_error = max(max_probability_error, error)
            gl.require(np.allclose(actual, trace["probabilities"][offset], atol=choices.ATOL, rtol=choices.RTOL) and
                       ctl.select_action(actual, "mode", trace["uniforms"][offset]) == trace["action_ids"][offset],
                       "原worker逐步概率或真实mode动作不能重现")
        for query in queries:
            offset = query["control_step"] - 1
            state, p, raw = queried[query["remaining"]]
            entry = dict(trial_index=index, repeat=row["repeat"], evaluated_target=row["target"],
                factual_condition=row["mode"], issued_target=trace["issued_target"], **query,
                observed_action_name=plan["action_names"][query["observed_action"]],
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                **choices.frame_metrics(p, raw), **diagnosis.action_metrics(p, query["observed_action"], row["target"]),
                **diagnosis.transition(measurements[offset], measurements[offset + 1], row["target"]),
                **diagnosis.reference_context(state, references, query["remaining"]),
                own_pose_before=events[query["frame"]]["telemetry"]["pose"],
                own_pose_after=events[query["next_action_frame"]]["telemetry"]["pose"],
                probabilities={name: values.tolist() for name, values in p.items()},
                raw_preferences={name: values.tolist() for name, values in raw.items()})
            for name in choices.CONDITIONS:
                entry[f"{name}_action_name"] = plan["action_names"][entry[f"{name}_mode"]]
                entry[f"{name}_runner_up_name"] = plan["action_names"][entry[f"{name}_runner_up"]]
            frames.append(entry)
        final = frames[-1]
        trial = dict(trial_index=index, repeat=row["repeat"], target=row["target"], mode=row["mode"],
            start_distance=row["start_distance"], end_distance=row["end_distance"],
            distance_improvement=row["distance_improvement"], original_direction_passed=bool(
                row["distance_improvement"] > 0 and row["target_preference_margin"] > 0),
            last_action=final["observed_action_name"], correct_last_mode=plan["action_names"][final["correct_mode"]],
            swapped_last_mode=plan["action_names"][final["swapped_mode"]],
            last_goal_mode_changed=final["goal_mode_changed"], last_goal_total_variation=final["goal_total_variation"],
            last_correct_top2_gap=final["correct_gap"], last_preference_reversed_away=final["preference_reversed_away"],
            last_step_progress=final["actual_step_progress"], distance_before_last=final["distance_before"],
            pitch_error_before_last=final["pitch_error_before"], pitch_error_at_end=final["pitch_error_after"])
        trials.append(trial)
        print(f"[TERMINAL] trial={index + 1}/30 repeat={row['repeat']} target={row['target']} condition={row['mode']} "
              f"actual={trial['last_action']} correct={trial['correct_last_mode']} swapped={trial['swapped_last_mode']} "
              f"mode_switch={int(trial['last_goal_mode_changed'])} reversal={int(trial['last_preference_reversed_away'])}", flush=True)
        report.require_writable()
    result = diagnosis.summarize(frames, plan)
    result.update(source_evaluation_id=manifest["evaluation_id"], action_names=plan["action_names"],
                  original_numerical_gate=record["numerical_gate"], probability_roundtrip_frames=480,
                  measurement_roundtrip_frames=510, reference_queries=8,
                  maximum_probability_error=max_probability_error)
    baseline.write_json(report.directory / "diagnostics.json", result)
    baseline.write_json(report.directory / "action_queries.json", dict(format=diagnosis.FORMAT, frames=frames,
        observed_actions_are_expert_labels=False, counterfactual_rollouts_executed=False, changed_budget_queries=False))
    baseline.write_csv(report.directory / "frame_metrics.csv", [{key: value for key, value in row.items()
        if key not in ("probabilities", "raw_preferences", "own_pose_before", "own_pose_after")} for row in frames])
    baseline.write_csv(report.directory / "trials.csv", trials)
    baseline.write_json(report.directory / "reference_context.json", dict(
        source_calibration_evaluation_id=calibration_record["evaluation_id"], worker_version=300,
        queries=[{key: value for key, value in row.items() if key != "state"} for row in references],
        same_remaining_comparison=True, reference_actions_transferred_as_labels=False,
        reference_distance_has_ood_threshold=False, references_are_independent_test=False))
    probability_gallery(report.directory / "target1_last_action_probabilities.png", frames, plan["action_names"])
    for module, expected, digest in zip(modules, versions, digests):
        gl.require({name: p._version for name, p in module.named_parameters()} == expected and
                   gl.tensor_digest(module.state_dict(), {}) == digest and
                   all(not p.requires_grad and p.grad is None for p in module.parameters()), "诊断改变冻结参数或产生梯度")
    gl.require(t01.same(query_rng, bc.capture_rng(args.device)), "只读动作查询推进了RNG")
    gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or name == "mineclip" or name.startswith("mineclip.")
                       for name in sys.modules), "只读诊断不能导入MineDojo/MineCLIP")
    report.data.update(query_frames=60, diagnosed_trials=30, probability_roundtrip_frames=480,
        measurement_roundtrip_frames=510, reference_queries=8, maximum_probability_error=max_probability_error,
        same_state_within_query=True, same_state_across_trials=False)
    report.check("real_query_alignment", "PASS", "全部30局、60个末尾真实动作前状态；frame78/79→incoming79/80、remaining2/1，无终点后查询")
    report.check("recorded_forward_roundtrip", "PASS", f"480个原概率与mode动作重现；最大概率误差={max_probability_error:.6g}；不重算RSSM")
    report.check("recorded_measurement_roundtrip", "PASS", "510个原逐帧视觉/物理测量重现，完整保留实际16步终点")
    report.check("same_state_terminal_comparison", "PASS", "同状态/真实预算两目标与no_goal/zero/base比较；概率变化、mode改变及事实末步收益分别记录")
    report.check("reference_context_comparison", "PASS", "四条SHA256绑定参考历史的8个同预算查询；只报告当前worker动作支持和状态余弦，不移植标签或判定分布外")
    report.check("no_training", "PASS", "权重/版本/梯度/查询RNG未变，无优化器/RSSM构造，无MineDojo/MineCLIP；new_env_steps=0")
    report.check("scope", "WARN", "只解释原自主轨迹末尾动作，不执行反事实、不生成专家标签、不改门槛或批准T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="完整repaired_control_evaluate目录，包含原manifest及30局轨迹")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    args.command = "analyze"
    try:
        record, manifest, checked_dir, checkpoint, cache_dir, calibration_record, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "terminal_action_diagnose_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于原评估/check、缓存、模型和共享依赖目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, worker_control_actions=0,
                       counterfactual_env_steps=0, behavior_accepted=False, t06_approved=False)
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
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        bc.require_disk_space(directory, 16 * 1024**2)
        report.require_writable()
        report.check("historical_identity", "PASS", "完整30次确认/check/修复verify、模型/旧代码及全部真实产物SHA256一致；行为失败保持")
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            analyze(args, report, record, manifest, checked_dir, checkpoint, cache_dir, calibration_record)
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
                             "原轨迹/目标/模型/依赖/check与旧代码大小及修改时间未变；只写独立新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
