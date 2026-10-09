"""Read all worker350 saved states; compare frozen 300/350 and both fixed goals.

No optimizer, RSSM, environment, rollout, label transfer or gate changes.
After the final report, automatically export one consolidated feedback file.
"""
import argparse
import copy
from datetime import datetime
import gc
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t05_action_choice_diagnose as querying
import t05_goal_control as legacy
import t05_continuation_control as deployed

CODE = ("goal_continuation_action_diagnose.py", "scripts/t05_continuation_action_diagnose.py",
        "scripts/t05_result_bundle.py")


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        saved = self.save()
        failed = not saved or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_CONTINUATION_ACTION_DIAGNOSE; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_control as confirmation
    import goal_continuation_action_diagnose as diagnosis
    directory = baseline.project_path(args.eval_dir)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "evaluate")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    plan, rows = manifest["design"], manifest["rows"]
    confirmation.validate_plan(plan)
    summary = confirmation.summarize(rows, plan)
    gl.require(manifest.get("format") == confirmation.EVALUATION_FORMAT and
        manifest.get("comparison_protocol") == record.get("comparison_protocol") == confirmation.PROTOCOL and
        manifest.get("evaluation_id") == record.get("evaluation_id") == gl.tensor_digest({}, {
            k: v for k, v in manifest.items() if k != "evaluation_id"}) and
        manifest.get("design_id") == record.get("design_id") == plan["design_id"] == gl.tensor_digest({}, {
            k: v for k, v in plan.items() if k != "design_id"}) and
        legacy.read_json(directory / "design.json") == plan and
        record["input_identity"] == manifest["input_identity"] == plan["input_identity"] and
        record["evaluation_code"] == plan["evaluation_code"] == deployed.code_identity(),
        "需要本次worker350完整确认及未改变的计划/模型/40个代码身份")
    expected = dict(planned_trials=30, attempted_trials=30, execution_valid_trials=30,
        worker_control_actions=480, worker_action_queries=480, probability_roundtrip_queries=480,
        measurement_roundtrip_frames=510, new_env_steps=2400, optimizer_updates=0, intervention_actions=0,
        worker_version=350, source_worker_version=300, additional_updates=50)
    gl.require(all(record.get(k) == v for k, v in expected.items()) and
        record.get("full_design_completed") is True and manifest.get("full_design_completed") is True and
        record.get("numerical_gate") == summary["numerical_gate"] == legacy.read_json(directory / "gate.json") and
        legacy.read_json(directory / "diagnostics.json")["numerical_gate"] == summary["numerical_gate"] and
        record.get("behavior_accepted") is False and record.get("t06_approved") is False and
        manifest.get("behavior_accepted") is False and manifest.get("t06_approved") is False and
        record.get("source_inputs_before") == record.get("source_inputs_after") and
        confirmation.passed(record, ("strict_continuation_inference_load", "design_identity", "planned_attempts",
            "real_control_interface", "execution_counters", "no_training", "source_inputs_unchanged")) and
        all(r["execution_valid"] and r["fixed_horizon_completed"] and r["actual_steps"] == 16 and
            r["returned_env_steps"] == 80 and r["actual_endpoint_frame"] == 80 for r in rows),
        "需要原30次完整工程有效历史；数值失败保留，不能筛选或改门槛")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(manifest["check_report_sha256"] == bc.file_hash(checked_dir / "report.json") and
        checked["input_identity"] == plan["input_identity"] and checked["design_id"] == plan["design_id"] and
        checked["evaluation_code"] == plan["evaluation_code"] and
        legacy.read_json(checked_dir / "design.json") == plan and
        checked["targets_sha256"] == bc.file_hash(checked_dir / "targets.npz") and
        confirmation.passed(checked, ("strict_continuation_inference_load", "fixed_visual_targets",
            "causal_execution_interface", "saved_execution_contract", "worker350_source_contract",
            "confirmation_gate_contract", "no_training", "source_inputs_unchanged")),
        "原确认check或固定目标身份不同")
    gl.require(args.device == record["arguments"]["device"], "诊断需要原确认设备与推理后端")
    context, protected, files = deployed.source_paths(SimpleNamespace(command="analyze", device=args.device,
        repair_verify_dir=record["arguments"]["repair_verify_dir"]))
    identity = plan["input_identity"]
    gl.require(identity["checkpoint"] == baseline.file_signature(context["checkpoint"]) and
        identity["checkpoint_sha256"] == bc.file_hash(context["checkpoint"]) and
        identity["input_identity"] == context["payload"]["input_identity"] == context["verified"]["repair_identity"] and
        identity["model_id"] == context["verified"]["model_id"] and
        identity["counters"] == context["payload"]["counters"] == context["verified"]["counters"] and
        identity["repair_verify_sha256"] == bc.file_hash(context["verify_dir"] / "report.json") and
        identity["dependencies"] == context["payload"]["dependencies"], "确认与固定50步生产latest/独立verify不同")
    required = set()
    for index, row in enumerate(rows):
        name = f"trial_{index:03d}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}"
        gl.require(baseline.project_path(row["artifact_dir"]) == directory / name, "真实trial路径属于另一确认")
        required.update(f"{name}/{item}" for item in ("trajectory.npz", "events.json", "control_trace.json",
            "frame_metrics.json", "metrics.json", "history_check.json", "start.json", "video.mp4"))
    gl.require(required <= set(manifest["artifacts"]) and
        {"design.json", "targets.npz", "trials.json", "trials.csv", "frame_metrics.csv", "gate.json", "diagnostics.json"}
        <= set(manifest["summary_artifacts"]) and legacy.read_json(directory / "trials.json") == rows,
        "原真实状态/动作/视频或汇总缺少SHA256绑定")
    for table in (manifest["artifacts"], manifest["summary_artifacts"]):
        for name, digest in table.items():
            path = (directory / name).resolve()
            gl.require(not Path(name).is_absolute() and directory in path.parents and path.is_file() and
                       bc.file_hash(path) == digest, f"原真实产物缺失或SHA256不同：{name}")
            files.append(path)
    # Check the real recorder schema before loading either inference model.
    for row in rows:
        trial_dir = baseline.project_path(row["artifact_dir"])
        diagnosis.validate_saved_history(legacy.read_json(trial_dir / "metrics.json"),
            legacy.read_json(trial_dir / "history_check.json"), row, plan["control_start_frame"],
            plan["control_start_frame"] + plan["horizon"] + 1)
    files += [directory / "report.json", directory / "evaluation_manifest.json",
              *[checked_dir / n for n in ("report.json", "design.json", "targets.npz")], *[ROOT / n for n in CODE]]
    return dict(record=record, manifest=manifest, checked_dir=checked_dir, context=context), \
        list(dict.fromkeys([*protected, directory, checked_dir])), list(dict.fromkeys(p.resolve() for p in files))


def load_policies(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_information_probe as probe
    import goal_reference_repair as original
    import goal_continuation_repair as repaired
    context, plan = source["context"], source["manifest"]["design"]
    cache = bc.TrainingCache(context["cache_dir"])
    policy350 = repaired.load_policy(context["checkpoint"], cache, args.device)
    path300 = context["refs"]["source_checkpoint"]
    verify300 = legacy.read_json(context["refs"]["source_verify_report"])
    legacy.accepted(verify300, "verify")
    payload300, refs300 = original.read_checkpoint(path300)
    original.validate_payload(payload300, payload300["input_identity"], payload300["options"])
    policy300 = original.load_policy(path300, cache, args.device)
    gl.require(path300.name == "latest.pt" and policy300.worker_version == verify300.get("worker_version") == 300 and
        policy300.repair_updates == payload300["counters"]["step"] == 100 and
        policy300.model_id == verify300.get("model_id") and policy300.identity == verify300.get("repair_identity") and
        baseline.project_path(verify300["arguments"]["checkpoint"]) == path300 and
        verify300.get("counters") == payload300["counters"] and
        policy350.worker_version == 350 and policy350.continuation_updates == 50 and
        policy350.model_id == plan["input_identity"]["model_id"] and policy350.identity == plan["input_identity"]["input_identity"] and
        context["verified"]["backend"] == verify300["backend"] == bc.backend_info(args.device) and
        policy300.base_hash == policy350.base_hash and
        all(policy300.identity[k] == policy350.identity[k] for k in
            ("base_id", "bundle_id", "cache_id", "feature_dim", "goal_dim", "action_dim", "horizon", "fixed_goals_id")) and
        refs300["frozen_bundle"] == context["refs"]["frozen_bundle"] == context["cache_dir"] / "frozen_bundle.pt",
        "冻结300/350来源、固定底座/WM/目标表示或推理后端不同")
    with np.load(Path(args.eval_dir) / "targets.npz", allow_pickle=False) as saved:
        bank = {k: saved[k].copy() for k in ("goals", "images", "heatmaps")}
    with np.load(source["checked_dir"] / "targets.npz", allow_pickle=False) as saved:
        gl.require(all(np.array_equal(saved[k], bank[k]) for k in bank), "原check/evaluate固定目标不同")
    gl.require(bank["goals"].dtype == np.float32 and bank["goals"].shape == (2, policy350.identity["goal_dim"]) and
        np.isfinite(bank["goals"]).all() and np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5) and
        gl.tensor_digest({}, {k: v.tolist() for k, v in bank.items()}) == plan["target_content_id"] and
        probe.array_digest({"goals": bank["goals"]}) == policy350.identity["fixed_goals_id"],
        "原真实RGB/heatmap目标内容或编码不同")
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], args.device).requires_grad_(False).eval()
    report.data.update(model_ids={"300": policy300.model_id, "350": policy350.model_id},
        worker_versions=[300, 350], source_evaluation_id=source["manifest"]["evaluation_id"],
        source_design_id=plan["design_id"], input_identity=plan["input_identity"],
        original_numerical_gate=source["record"]["numerical_gate"], target_content_id=plan["target_content_id"],
        inference_backend=bc.backend_info(args.device), source_checkpoint300_sha256=bc.file_hash(path300),
        source_verify300_sha256=bc.file_hash(context["refs"]["source_verify_report"]),
        source_checkpoint350_sha256=bc.file_hash(context["checkpoint"]),
        diagnostic_code={name: bc.file_hash(ROOT / name) for name in CODE},
        state_source="each SHA256-bound worker350 real history; saved causal features only; no RSSM reconstruction")
    report.check("strict_frozen_versions", "PASS", "生产300与350严格独立推理；共用底座/WM/目标表示，无优化器或RSSM构造")
    return {"300": policy300, "350": policy350}, library, bank


def saved_history_contract_checks(report, metrics, history, row, first, total_frames):
    import goal_continuation_action_diagnose as diagnosis

    # The real autonomous recorder never writes saved_history_roundtrip.
    # Acceptance comes from its SHA256-bound own-history fields and matching error.
    original_schema = copy.deepcopy(history)
    original_schema.pop("saved_history_roundtrip", None)
    diagnosis.validate_saved_history(metrics, original_schema, row, first, total_frames)
    for key, value in (("own_causal_history_passed", False), ("reset_is_real", False),
                       ("comparison_to_other_histories", True), ("start_frame", first + 1),
                       ("total_frames", total_frames - 1)):
        changed = dict(original_schema, **{key: value}, saved_history_roundtrip=True)
        t02.rejection(report, f"history_{key}_guard", lambda h=changed: diagnosis.validate_saved_history(
            metrics, h, row, first, total_frames), "因果历史")
    for name, changed in (("missing", {k: v for k, v in original_schema.items() if k != "maximum_absolute_error"}),
                          ("mismatch", dict(original_schema, maximum_absolute_error=history["maximum_absolute_error"] + 1)),
                          ("nonfinite", dict(original_schema, maximum_absolute_error=float("nan")))):
        t02.rejection(report, f"history_error_{name}_guard", lambda h=changed: diagnosis.validate_saved_history(
            metrics, h, row, first, total_frames), "误差")
    changed = copy.deepcopy(row)
    changed["video"]["frames"] = total_frames - 1
    t02.rejection(report, "history_video_frames_guard", lambda: diagnosis.validate_saved_history(
        changed, original_schema, changed, first, total_frames), "视频")
    changed = dict(metrics, own_history_max_error=row["own_history_max_error"] + 1)
    t02.rejection(report, "history_metrics_guard", lambda: diagnosis.validate_saved_history(
        changed, original_schema, row, first, total_frames), "指标")
    report.check("saved_history_schema_contract", "PASS", "原自主历史无需参考历史专用字段；真实重置/自身历史/边界/误差/视频仍严格核对，内存负守卫不写轨迹")


def contract_checks(report, arrays, events, trace, measured, plan):
    import numpy as np
    import goal_continuation_action_diagnose as diagnosis
    import goal_action_choice_diagnose as choices
    def reject(name, message, *, a=arrays, e=events, t=trace, m=measured, p=plan):
        t02.rejection(report, name, lambda: diagnosis.validate_queries(a, e, t, m, p, 0), message)
    queries = diagnosis.validate_queries(arrays, events, trace, measured, plan, 0)
    choices.require(len(queries) == 16 and [(q["frame"], q["next_action_frame"], q["remaining"]) for q in queries[-2:]] ==
                    [(78, 79, 2), (79, 80, 1)], "末两步真实索引异常")
    changed = copy.deepcopy(arrays)
    changed["action"][65] = np.eye(12, dtype=np.float32)[(trace["action_ids"][0] + 1) % 12]
    reject("next_action_guard", "incoming", a=changed)
    changed = copy.deepcopy(arrays)
    changed["is_terminal"][79] = True
    reject("terminal_guard", "结束", a=changed)
    changed = copy.deepcopy(trace)
    changed["remaining"][-1] = 2
    reject("remaining_guard", "预算", t=changed)
    changed = copy.deepcopy(trace)
    changed["worker_version"] = 300
    reject("saved_worker_version_guard", "worker350", t=changed)
    changed = copy.deepcopy(trace)
    changed["uniforms"][-1] = (changed["uniforms"][-1] + .5) % 1
    reject("action_uniform_guard", "随机数", t=changed)
    changed = copy.deepcopy(plan)
    del changed["design"]["randomization_seed"]
    reject("missing_design_seed_guard", "种子", p=changed)
    changed = copy.deepcopy(measured)
    changed[-1]["next_action_id"] = 0
    reject("measurement_terminal_guard", "测量", m=changed)
    changed = copy.deepcopy(trace)
    changed["probabilities"][-1] = np.eye(12)[(trace["action_ids"][-1] + 1) % 12].tolist()
    reject("recorded_mode_guard", "mode", t=changed)
    row = dict(queries[-1], trial_index=0, repeat=plan["design"]["schedule"][0]["repeat"],
        evaluated_target=plan["design"]["schedule"][0]["target"], factual_condition=plan["design"]["schedule"][0]["mode"])
    changed = dict(row, control_step=16, frame=80, next_action_frame=81, remaining=0)
    t02.rejection(report, "after_endpoint_query_guard", lambda: diagnosis.validate_query_row(changed, plan), "预算")
    t02.rejection(report, "missing_query_guard", lambda: diagnosis.summarize([], [], plan), "遗漏")
    p = {name: np.array([.6, .3, .1]) for name in choices.CONDITIONS}
    other = dict(p, goal1=np.array([.5, .4, .1]))
    difference = diagnosis.version_difference(p, other, 0, 1)
    choices.require(not difference["goal1_mode_changed"] and difference["goal1_total_variation"] > 0,
                    "版本概率变化与mode改变未分开")
    changed = dict(other, base=np.array([.5, .4, .1]))
    t02.rejection(report, "base_version_guard", lambda: diagnosis.version_difference(p, changed, 0, 1), "底座")
    report.check("diagnostic_contracts", "PASS", "真实历史内存副本的incoming/终止/remaining/种子/mode/版本/末步守卫通过；无伪轨迹写入")


def gallery(path, rows, names, target):
    from PIL import Image, ImageDraw
    selected = sorted((r for r in rows if r["factual_condition"] == "goal" and
        r["evaluated_target"] == target and r["remaining"] == 1), key=lambda r: r["repeat"])
    image = Image.new("RGB", (1200, 5 * 172 + 48), "white")
    draw = ImageDraw.Draw(image)
    draw.text((5, 5), "Same saved state / remaining=1 | blue=300 goal0 cyan=300 goal1 green=350 goal0 orange=350 goal1 gray=350 no_goal", fill="black")
    draw.text((5, 21), "Observed actions are facts, not expert labels; alternative actions were not executed.", fill="black")
    colors = (("300", "goal0", "#2166ac"), ("300", "goal1", "#67a9cf"),
              ("350", "goal0", "#1b7837"), ("350", "goal1", "#d95f02"), ("350", "no_goal", "#666666"))
    for index, row in enumerate(selected):
        top, bottom = 48 + index * 172, 48 + index * 172 + 122
        draw.text((5, top), f"repeat={row['repeat']} trial={row['trial_index']} actual={row['observed_action_name']} "
                  f"d{target} {row['distance_before']:.5f}->{row['distance_after']:.5f} reversal={row['preference_reversed_away']}", fill="black")
        for tick in (0., .5, 1.):
            y = bottom - round(tick * 90)
            draw.line((32, y, 1194, y), fill="#dddddd")
            draw.text((3, y - 5), str(tick), fill="black")
        for action, name in enumerate(names):
            x = 34 + action * 96
            draw.text((x, bottom + 5), name, fill="black")
            for column, (version, condition, color) in enumerate(colors):
                height = round(row["probabilities"][version][condition][action] * 90)
                if height:
                    draw.rectangle((x + column * 17, bottom - height, x + column * 17 + 13, bottom), fill=color)
    image.save(path)


def analyze(args, report, source):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_control as ctl
    import goal_reference_calibration as calibration
    import goal_action_choice_diagnose as choices
    import goal_terminal_action_diagnose as terminal
    import goal_continuation_action_diagnose as diagnosis
    loader_rng = bc.capture_rng(args.device)
    try:
        policies, library, bank = load_policies(args, report, source)
    finally:
        bc.restore_rng(loader_rng, args.device)
    modules = (*policies.values(), library)
    versions = [{name: parameter._version for name, parameter in module.named_parameters()} for module in modules]
    digests = [gl.tensor_digest(module.state_dict(), {}) for module in modules]
    query_rng = bc.capture_rng(args.device)
    bank_before = {k: v.copy() for k, v in bank.items()}
    plan, rows = source["manifest"]["design"], source["manifest"]["rows"]
    for target in (0, 1):
        obs = {name: torch.as_tensor(bank[key][target:target + 1], device=args.device)
               for name, key in (("image", "images"), ("heatmap", "heatmaps"))}
        gl.require(np.allclose(library(obs)[0].cpu().numpy(), bank["goals"][target], atol=choices.ATOL, rtol=choices.RTOL),
                   "固定真实终点目标重编码不同")
    report.check("fixed_visual_targets", "PASS", "原两RGB/heatmap终点及内容ID重编码通过；坐标仅测量，无目标替换")
    frames, trials, flat = [], [], []
    maximum_error = 0.
    for index, row in enumerate(rows):
        directory = Path(row["artifact_dir"])
        arrays = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "control_trace.json")
        measured = legacy.read_json(directory / "frame_metrics.json")
        history = legacy.read_json(directory / "history_check.json")
        metrics_record = legacy.read_json(directory / "metrics.json")
        diagnosis.validate_saved_history(metrics_record, history, row, plan["control_start_frame"], len(arrays["image"]))
        queries = diagnosis.validate_queries(arrays, events, trace, measured, plan, index)
        if index == 0:
            saved_history_contract_checks(report, metrics_record, history, row,
                                          plan["control_start_frame"], len(arrays["image"]))
            contract_checks(report, arrays, events, trace, measured, plan)
        start = legacy.read_json(directory / "start.json")
        gl.require(start.get("control_start_frame") == 64 and start.get("own_real_history_only") is True and
            start.get("comparison_to_reference_required") is False and start["telemetry"] == events[64]["telemetry"] and
            np.array_equal(np.asarray(start["rssm_feature"], dtype=np.float32), arrays["features"][64]) and
            np.allclose(start["distances_to_both_goals"], trace["distances_to_both_goals"][0], atol=choices.ATOL, rtol=choices.RTOL),
            "本局原真实起点被修改")
        for offset, saved in enumerate(measured):
            frame = 64 + offset
            obs = {name: torch.as_tensor(arrays[name][frame:frame + 1], device=args.device) for name in ("image", "heatmap")}
            feature = library(obs)[0].cpu().numpy()
            values = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                         events[frame]["telemetry"], bank, plan["target_telemetry"])
            gl.require(np.allclose(feature, saved["goal_feature"], atol=choices.ATOL, rtol=choices.RTOL) and all(
                saved[k] == v if k.startswith("closest_") else np.isclose(saved[k], v, atol=choices.ATOL, rtol=choices.RTOL)
                for k, v in values.items()), "原逐帧视觉/物理测量不能重现")
        for query in queries:
            offset = query["control_step"]
            state = arrays["features"][query["frame"]:query["frame"] + 1].copy()
            tensor = torch.as_tensor(state, device=args.device)
            unchanged = tensor.clone()
            probabilities, raw, metrics = {}, {}, {}
            for version, policy in policies.items():
                p, preferences = querying.query(policy, tensor, bank["goals"], query["remaining"])
                gl.require(torch.equal(unchanged, tensor), "目标/版本查询改变本局真实状态")
                probabilities[version], raw[version] = p, preferences
                metrics[version] = diagnosis.version_metrics(p, preferences, query["observed_action"], row["target"])
            factual = "no_goal" if row["mode"] == "no_goal" else f"goal{trace['issued_target']}"
            actual = probabilities["350"][factual]
            error = float(np.max(np.abs(actual - np.asarray(trace["probabilities"][offset]))))
            maximum_error = max(maximum_error, error)
            gl.require(np.allclose(actual, trace["probabilities"][offset], atol=choices.ATOL, rtol=choices.RTOL) and
                int(actual.argmax()) == query["observed_action"], "原350保存概率或实际mode不能重现")
            change = diagnosis.version_difference(probabilities["300"], probabilities["350"], query["observed_action"], row["target"])
            entry = dict(trial_index=index, repeat=row["repeat"], evaluated_target=row["target"], factual_condition=row["mode"],
                issued_target=trace["issued_target"], **query, observed_action_name=plan["action_names"][query["observed_action"]],
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                **terminal.transition(measured[offset], measured[offset + 1], row["target"]),
                own_pose_before=events[query["frame"]]["telemetry"]["pose"],
                own_pose_after=events[query["next_action_frame"]]["telemetry"]["pose"],
                version_metrics=metrics, version_change=change,
                probabilities={v: {k: p.tolist() for k, p in values.items()} for v, values in probabilities.items()},
                raw_preferences={v: {k: p.tolist() for k, p in values.items()} for v, values in raw.items()})
            frames.append(entry)
            scalar = {k: v for k, v in entry.items() if k not in
                      ("version_metrics", "version_change", "probabilities", "raw_preferences", "own_pose_before", "own_pose_after")}
            for version in diagnosis.VERSIONS:
                scalar.update({f"worker{version}_{k}": v for k, v in metrics[version].items()})
                scalar.update({f"worker{version}_{condition}_mode_name": plan["action_names"][metrics[version][f"{condition}_mode"]]
                               for condition in choices.CONDITIONS})
            scalar.update({f"version_change_{k}": v for k, v in change.items()})
            flat.append(scalar)
        final = frames[-1]
        trial = dict(trial_index=index, repeat=row["repeat"], target=row["target"], mode=row["mode"],
            start_distance=row["start_distance"], end_distance=row["end_distance"], distance_improvement=row["distance_improvement"],
            original_direction_passed=bool(row["distance_improvement"] > 0 and row["target_preference_margin"] > 0),
            last_action=final["observed_action_name"], last_step_progress=final["actual_step_progress"],
            last_preference_reversed_away=final["preference_reversed_away"],
            worker350_last_correct_mode=plan["action_names"][metrics["350"]["correct_mode"]],
            worker300_last_correct_mode=plan["action_names"][metrics["300"]["correct_mode"]],
            last_correct_version_mode_changed=change["correct_mode_changed"],
            worker350_last_goal_mode_changed=metrics["350"]["goal_mode_changed"])
        trials.append(trial)
        print(f"[ACTION DIAGNOSE {index + 1}/30] repeat={row['repeat']} target={row['target']} condition={row['mode']} "
              f"actual={trial['last_action']} correct300={trial['worker300_last_correct_mode']} "
              f"correct350={trial['worker350_last_correct_mode']} reversal={trial['last_preference_reversed_away']}", flush=True)
    diagnostics = diagnosis.summarize(frames, trials, plan)
    diagnostics["original_numerical_gate"] = source["record"]["numerical_gate"]
    t02.rejection(report, "duplicate_query_guard", lambda: diagnosis.summarize(frames + frames[:1], trials, plan), "重复")
    baseline.write_json(report.directory / "queries.json", frames)
    baseline.write_csv(report.directory / "frame_metrics.csv", flat)
    baseline.write_json(report.directory / "trials.json", trials)
    baseline.write_csv(report.directory / "trials.csv", trials)
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    for target in (0, 1):
        gallery(report.directory / f"target{target}_last_action_probabilities.png", frames, plan["action_names"], target)
    for module, expected_versions, digest in zip(modules, versions, digests):
        gl.require({name: p._version for name, p in module.named_parameters()} == expected_versions and
            gl.tensor_digest(module.state_dict(), {}) == digest and all(not p.requires_grad and p.grad is None for p in module.parameters()),
            "只读查询改变参数/版本或产生梯度")
    gl.require(all(np.array_equal(bank[k], bank_before[k]) for k in bank) and t01.same(query_rng, bc.capture_rng(args.device)),
               "只读查询改变固定目标或推进RNG")
    gl.require(not any(name in ("minedojo", "mineclip") or name.startswith(("minedojo.", "mineclip.")) for name in sys.modules),
               "只读诊断不能导入MineDojo/MineCLIP")
    output_manifest = dict(format=diagnosis.FORMAT, source_evaluation_id=source["manifest"]["evaluation_id"],
        source_design_id=plan["design_id"], model_ids=report.data["model_ids"], target_content_id=plan["target_content_id"],
        diagnostic_code=report.data["diagnostic_code"], query_frames=480, distributions_saved=4800,
        source_manifest_sha256=bc.file_hash(Path(args.eval_dir) / "evaluation_manifest.json"),
        artifacts={path.name: bc.file_hash(path) for path in report.directory.iterdir()
                   if path.is_file() and path.name != "report.json"}, optimizer_updates=0, new_env_steps=0,
        behavior_accepted=False, t06_approved=False)
    output_manifest["diagnosis_id"] = gl.tensor_digest({}, output_manifest)
    baseline.write_json(report.directory / "diagnosis_manifest.json", output_manifest)
    report.data.update(diagnosis_id=output_manifest["diagnosis_id"], diagnosed_trials=30, query_frames=480,
        saved_history_records_verified=30,
        diagnostic_distribution_rows=4800, terminal_query_frames=60, probability_roundtrip_frames=480,
        measurement_roundtrip_frames=510, maximum_probability_error=maximum_error,
        same_state_within_query=True, same_state_across_trials=False)
    report.check("real_query_alignment", "PASS", "全部30局480动作前状态，remaining16–1；末尾78/79→79/80，无终点后查询")
    report.check("recorded_forward_roundtrip", "PASS", f"原350的480概率与mode重现，最大误差={maximum_error:.6g}；不重算RSSM")
    report.check("recorded_measurement_roundtrip", "PASS", "全部510视觉/物理测量重现，原真实16步终点保留")
    report.check("same_state_goal_and_version_comparison", "PASS", "480状态×两版本×五条件完整概率/偏好/事实排名与NLL；底座逐值同，无目标严格目标不变")
    report.check("no_training", "PASS", "参数/版本/梯度/目标/查询RNG不变，无优化器/RSSM/环境，更新与新环境步0")
    report.check("scope", "WARN", "事实动作不是专家标签，版本概率变化不是实际控制因果收益，不移植续段标签、不改gate或批准T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--feedback-dir", help="fresh independent export directory; otherwise auto-generated")
    args = parser.parse_args()
    args.command = "analyze"
    args.eval_dir = str(baseline.project_path(args.eval_dir))
    try:
        source, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "continuation_action_diagnose_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == p or p in directory.parents or directory in p.parents for p in protected):
            parser.error("输出须独立于模型/缓存/旧轨迹/验收目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, worker_control_actions=0,
        source_saved_history_records_verified=30,
        counterfactual_env_steps=0, behavior_accepted=False, t06_approved=False,
        feedback_policy="automatic single lossless metadata/images JSON after final report; no raw data/models/videos")
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
        before = {str(p): baseline.file_signature(p) for p in files}
        report.data["source_inputs_before"] = before
        bc.require_disk_space(directory, 160 * 1024**2)
        report.require_writable()
        report.check("historical_identity", "PASS", "完整350确认/check/固定50步latest/verify及真实产物和40个旧代码SHA绑定；旧行为失败保持")
        report.check("source_saved_history_schema", "PASS", "全部30局原自主历史格式、逐局指标、误差及81帧视频验收标志一致；不要求参考历史专用字段")
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            analyze(args, report, source)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(p): baseline.file_signature(p) for p in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if after == before else "FAIL",
                             "原模型/缓存/轨迹/目标/验收/check及40个旧代码大小/修改时间不变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    result = report.finish()
    # Do not modify the finished report after hashing/exporting it; failures are exported too.
    try:
        import t05_result_bundle as feedback
        sources, missing = feedback.sources_for_run(directory)
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT / "relevance_map/t05_feedback" / (
            directory.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in protected):
            raise ValueError("汇总输出必须独立于原保护目录")
        index = feedback.build_bundle(sources, export, directory.name, missing)
        print(f"FEEDBACK_FILE={index['json_path']}", flush=True)
        print(f"FEEDBACK_INDEX={export / 'bundle_index.json'}", flush=True)
        print(f"FEEDBACK_ZIP={index['zip_path']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"[FAIL] result_bundle: {type(error).__name__}: {error}; 原诊断报告保留", file=sys.stderr, flush=True)
        return 2
    return result


if __name__ == "__main__":
    sys.exit(main())
