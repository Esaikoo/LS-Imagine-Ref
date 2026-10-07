"""Read-only worker action support on all four complete calibration runs.

No environment, RSSM constructor, optimizer or new training labels. All
queries use the saved real state before its actual next script action.
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


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_REFERENCE_ACTION_DIAGNOSE; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_calibration as calibration

    directory = baseline.project_path(args.calibration_dir)
    record = legacy.read_json(directory / "report.json")
    manifest = legacy.read_json(directory / "calibration_manifest.json")
    legacy.accepted(record, "probe")
    plan = manifest["design"]
    calibration.validate_plan(plan)
    gl.require(manifest.get("format") == calibration.EVALUATION_FORMAT and
               manifest.get("comparison_protocol") == record.get("comparison_protocol") == calibration.PROTOCOL and
               manifest.get("evaluation_id") == record.get("evaluation_id") == gl.tensor_digest({}, {
                   name: value for name, value in manifest.items() if name != "evaluation_id"}) and
               manifest.get("design_id") == record.get("design_id") == plan.get("design_id") == gl.tensor_digest({}, {
                   name: value for name, value in plan.items() if name != "design_id"}) and
               legacy.read_json(directory / "design.json") == plan and record.get("input_identity") == plan["input_identity"] and
               record.get("evaluation_code") == plan["evaluation_code"] and
               record.get("environment_fingerprint") == plan["environment_fingerprint"],
               "需要原标定manifest/计划/模型身份；不能分析失败或改变后的标定")
    rows = manifest["rows"]
    summary = calibration.summarize(rows, plan)
    gl.require(record.get("full_design_completed") is True and len(rows) == 4 and
               record.get("planned_runs") == record.get("attempted_runs") == record.get("valid_histories") ==
               record.get("complete_scripts") == summary["valid_histories"] == summary["complete_scripts"] == 4 and
               manifest.get("new_env_steps") == manifest.get("attempted_env_steps") == record.get("new_env_steps") == 320 and
               manifest.get("script_actions") == record.get("script_actions") == 64 and
               manifest.get("worker_control_actions") == record.get("worker_control_actions") == record.get("optimizer_updates") == 0 and
               history.passed_checks(record, ("historical_identity", "design_identity", "planned_attempts",
                    "real_script_interface", "fixed_horizon_endpoints", "no_training", "source_inputs_unchanged")),
               "需要全部4次完整有效固定脚本；不筛选成功子集或使用worker执行数据")
    for name, digest in plan["evaluation_code"].items():
        path = (ROOT / name).resolve()
        gl.require(ROOT in path.parents and path.is_file() and bc.file_hash(path) == digest,
                   f"原标定推理代码改变：{name}；保持旧指纹并只新增诊断入口")
    previous_args = copy.copy(args)
    previous_args.eval_dir = plan["source_evaluation"]
    _, previous, previous_check, protected, files = history.historical_inputs(previous_args)
    gl.require(previous["evaluation_id"] == plan["source_evaluation_id"] and
               previous["design_id"] == plan["source_design_id"] and
               previous["input_identity"] == plan["input_identity"] and
               previous["design"]["target_content_id"] == plan["target_content_id"] and
               previous["design"]["start_actions"] == plan["start_actions"] and
               bc.file_hash(Path(plan["source_evaluation"]) / "evaluation_manifest.json") == plan["source_evaluation_manifest_sha256"] and
               bc.file_hash(Path(plan["source_evaluation"]) / "report.json") == plan["source_evaluation_report_sha256"],
               "标定与原完整sample的真实模型、目标或历史来源不同")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(checked.get("design_id") == plan["design_id"] and
               checked.get("input_identity") == plan["input_identity"] and
               checked.get("evaluation_code") == plan["evaluation_code"] and
               checked.get("environment_fingerprint") == plan["environment_fingerprint"] and
               legacy.read_json(checked_dir / "design.json") == plan and
               checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz") and
               history.passed_checks(checked, ("historical_identity", "fixed_reference_scripts", "fixed_visual_targets",
                    "causal_history_interface", "calibration_contracts", "trial_output_preflight", "predeclared_design",
                    "no_training", "source_inputs_unchanged")), "原标定check或固定目标改变")
    required = {"design.json", "runs.json", "diagnostics.json", "runs.csv", "frame_metrics.csv"}
    for index, row in enumerate(rows):
        name = f"run_{index:03d}_repeat_{row['repeat']}_script_{row['script']}"
        gl.require(baseline.project_path(row["artifact_dir"]) == directory / name and
                   row.get("saved_history_roundtrip") is True and row.get("actual_script_steps") == 16 and
                   row.get("returned_env_steps") == 80 and row.get("worker_control_actions") == 0,
                   "完整标定历史路径、步数或保存往返验收不同")
        required.update(f"{name}/{item}" for item in (
            "trajectory.npz", "events.json", "metrics.json", "script_trace.json", "start.json", "history_check.json", "video.mp4"))
    gl.require(required <= set(manifest["artifacts"]), "标定真实轨迹、状态或事件缺少SHA256")
    for name, digest in manifest["artifacts"].items():
        path = (directory / name).resolve()
        gl.require(directory in path.parents and not Path(name).is_absolute() and path.is_file() and
                   bc.file_hash(path) == digest, f"标定真实产物缺失或改变：{name}")
        files.append(path)
    files += [directory / "report.json", directory / "calibration_manifest.json"]
    files += [checked_dir / name for name in ("report.json", "design.json", "targets.npz")]
    files += [ROOT / name for name in plan["evaluation_code"]]
    protected += [directory, checked_dir]
    return manifest, checked_dir, previous_check, list(dict.fromkeys(protected)), list(dict.fromkeys(files))


def contract_checks(report):
    import numpy as np
    import goal_action_choice_diagnose as choices
    import goal_reference_action_diagnose as diagnosis

    p = {name: np.array([.6, .3, .1]) for name in choices.CONDITIONS}
    result = diagnosis.label_metrics(p, 1, 0)
    diagnosis.require(result["correct_script_rank"] == 2 and not result["correct_script_mode_match"] and
                      np.isclose(result["correct_script_nll"], -np.log(.3)) and
                      result["swapped_minus_correct_nll"] == 0 and result["no_goal_minus_correct_nll"] == 0,
                      "已知真实动作的概率/排序/NLL或条件别名不一致")
    equal_mode = dict(p, goal1=np.array([.5, .4, .1]))
    diagnosis.require(not choices.distribution_difference(p["goal0"], equal_mode["goal1"])["mode_changed"] and
                      diagnosis.label_metrics(equal_mode, 1, 0)["swapped_minus_correct_nll"] < 0 and
                      diagnosis.common_prefix([[0, 0, 1], [0, 0, 2]]) == 2,
                      "同mode的真实动作支持差异或共同脚本前缀被忽略")
    # Tiny memory markers cover indexing only; never serialized as real data.
    first, horizon, dimension, script = 1, 2, 3, [1, 2]
    arrays = dict(features=np.zeros((4, 2), np.float32),
                  action=np.vstack((np.zeros(3), np.eye(3)[[0, 1, 2]])).astype(np.float32),
                  is_last=np.zeros(4, bool), is_terminal=np.zeros(4, bool))
    trace = dict(execution_policy="fixed_script", start_frame=first, planned_action_ids=script,
        attempted_action_ids=script, sampled_actions=False, policy_queried=False,
        measurements=[dict(frame=first + step, script_step=step, incoming_action=None if step == 0 else script[step - 1])
                      for step in range(horizon + 1)])
    queries = diagnosis.validate_queries(arrays, trace, first, horizon, script, dimension)
    diagnosis.require([(row["frame"], row["next_action_frame"], row["remaining"]) for row in queries] == [(1, 2, 2), (2, 3, 1)],
                      "动作前状态/真实下一动作/剩余步数索引异常")
    wrong = {name: value.copy() for name, value in arrays.items()}
    wrong["action"][2] = np.eye(3)[0]
    t02.rejection(report, "next_action_guard", lambda: diagnosis.validate_queries(wrong, trace, first, horizon, script, dimension), "incoming")
    terminal = {name: value.copy() for name, value in arrays.items()}
    terminal["is_terminal"][2] = True
    t02.rejection(report, "terminal_guard", lambda: diagnosis.validate_queries(terminal, trace, first, horizon, script, dimension), "结束")
    shifted = copy.deepcopy(trace)
    shifted["measurements"][1]["frame"] += 1
    t02.rejection(report, "measurement_guard", lambda: diagnosis.validate_queries(arrays, shifted, first, horizon, script, dimension), "索引")
    t02.rejection(report, "action_label_guard", lambda: diagnosis.label_metrics(p, 12, 0), "越界")
    marker_plan = dict(horizon=2, control_start_frame=1, action_names=["noop", "left", "right"],
        scripts=[[1, 2], [1, 0]], schedule=[dict(repeat=repeat, script=target) for repeat in range(2) for target in (0, 1)])
    marker_rows = []
    for index, cell in enumerate(marker_plan["schedule"]):
        for step, label in enumerate(marker_plan["scripts"][cell["script"]]):
            marker_rows.append(dict(run_index=index, repeat=cell["repeat"], assigned_target=cell["script"],
                script_step=step + 1, frame=first + step, next_action_frame=first + step + 1,
                remaining=2 - step, script_action=label, goal_mode_changed=False, goal_total_variation=0.,
                **diagnosis.label_metrics(p, label, cell["script"])))
    diagnosis.require(diagnosis.summarize(marker_rows, marker_plan)["first_script_divergence_step"] == 2,
                      "共同前缀与分叉后汇总边界异常")
    bad_budget = copy.deepcopy(marker_rows)
    bad_budget[0]["remaining"] = 1
    t02.rejection(report, "remaining_guard", lambda: diagnosis.summarize(bad_budget, marker_plan), "预算")
    duplicate = marker_rows[:-1] + marker_rows[:1]
    t02.rejection(report, "duplicate_query_guard", lambda: diagnosis.summarize(duplicate, marker_plan), "重复")
    report.check("diagnostic_contracts", "PASS", "真实脚本动作的概率/排序/NLL、共同前缀与下一动作索引守卫通过；内存marker不写入产物")


def probability_gallery(path, frames, names):
    from PIL import Image, ImageDraw

    # Every observed step, preserving all runs rather than choosing examples.
    panel = Image.new("RGB", (1150, 85 * len(frames) + 30), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), "Support for ACTUAL script action on its own saved state | blue=correct orange=swapped gray=no-goal", fill="black")
    for index, row in enumerate(frames):
        y = 30 + index * 85
        draw.text((5, y), f"run{row['run_index']} repeat{row['repeat']} target{row['assigned_target']} "
                  f"step{row['script_step']:02d} frame{row['frame']} rem{row['remaining']} action={names[row['script_action']]}", fill="black")
        for column, condition in enumerate(("correct", "swapped", "no_goal")):
            x = 10 + column * 380
            value = row[f"{condition}_script_probability"]
            draw.rectangle((x, y + 24, x + 340, y + 39), outline="#bbbbbb")
            width = round(value * 340)
            if width:
                draw.rectangle((x, y + 24, x + width, y + 39), fill=("#2166ac", "#d95f02", "#666666")[column])
            draw.text((x, y + 46), f"p={value:.5f} rank={row[f'{condition}_script_rank']} "
                      f"mode={names[row[f'{condition}_mode']]}", fill="black")
    panel.save(path)


def analyze(args, report, manifest, checked_dir, previous_check):
    import numpy as np
    import torch
    import goal_action_choice_diagnose as choices
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_action_diagnose as diagnosis
    import goal_reference_calibration as calibration
    import goal_residual_worker as residual

    contract_checks(report)
    plan, identity = manifest["design"], manifest["design"]["input_identity"]
    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    policy = residual.load_policy(baseline.project_path(args.checkpoint), cache, args.device)
    gl.require(policy.identity == identity["input_identity"] and policy.model_id == identity["model_id"] and
               policy.worker_version == identity["worker_version"], "需要本次标定使用的同一冻结worker与缓存")
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], args.device)
    modules = (policy, library)
    versions = [{name: p._version for name, p in module.named_parameters()} for module in modules]
    digests = [gl.tensor_digest(module.state_dict(), {}) for module in modules]
    rng = bc.capture_rng(args.device)
    with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    with np.load(previous_check / "targets.npz", allow_pickle=False) as saved:
        gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "标定目标不是原sample固定目标")
    gl.require(gl.tensor_digest({}, {name: bank[name].tolist() for name in bank}) == plan["target_content_id"] and
               bank["goals"].shape == (2, policy.identity["goal_dim"]) and bank["goals"].dtype == np.float32,
               "固定目标内容或向量结构不同")
    for target in (0, 1):
        obs = {name: torch.as_tensor(bank[key][target:target + 1], device=args.device)
               for name, key in (("image", "images"), ("heatmap", "heatmaps"))}
        gl.require(np.allclose(library(obs)[0].cpu().numpy(), bank["goals"][target], atol=choices.ATOL, rtol=choices.RTOL),
                   "原固定视觉目标重编码不同")
    report.data.update(input_identity=identity, calibration_evaluation_id=manifest["evaluation_id"],
        calibration_design_id=plan["design_id"], target_content_id=plan["target_content_id"],
        diagnosis_code={name: bc.file_hash(ROOT / name) for name in (
            "goal_reference_action_diagnose.py", "scripts/t05_reference_action_diagnose.py")},
        inference_backend=bc.backend_info(args.device), state_source="original SHA256-bound calibration features",
        source_goal_description=plan["goal_source"], teacher_forced=True)
    report.check("strict_frozen_load", "PASS", f"同一第{policy.worker_version}步残差worker/目标编码器；直接读各局真实因果状态，无RSSM构造或优化器")
    report.check("fixed_visual_targets", "PASS", "原两目标与sample/标定check逐值及内容ID一致；目标替换不改变实际历史或下一动作标签")
    if report.data["inference_backend"] != identity["inference_backend"]:
        report.check("inference_backend", "WARN", "当前推理后端与标定不同；本次概率为当前后端的离线查询，未声称原worker概率往返")
    frames = []
    first, horizon, dimension = plan["control_start_frame"], plan["horizon"], len(plan["action_names"])
    shared = diagnosis.common_prefix(plan["scripts"])
    for index, run in enumerate(manifest["rows"]):
        directory = Path(run["artifact_dir"])
        gl.require(legacy.read_json(directory / "metrics.json") == run, "标定run指标与manifest不同")
        arrays = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "script_trace.json")
        checked_history = legacy.read_json(directory / "history_check.json")
        gl.require(checked_history.get("own_causal_history_passed") is True and
                   checked_history.get("saved_history_roundtrip") is True and checked_history.get("reset_is_real") is True and
                   checked_history.get("comparison_to_other_histories") is False and
                   checked_history.get("start_frame") == first and checked_history.get("total_frames") == 81 and
                   checked_history.get("maximum_absolute_error") == run["own_history_max_error"] and
                   arrays["features"].shape == (81, policy.identity["feature_dim"]) and trace["script"] == run["script"],
                   "必须读该局已验收并按SHA256固定的完整因果历史")
        script = plan["scripts"][run["script"]]
        count = calibration.validate_execution(arrays, events, plan["start_actions"], script, dimension)
        queries = diagnosis.validate_queries(arrays, trace, first, horizon, script, dimension)
        gl.require(count == len(queries) == 16 and calibration.native_actions_equal(
            trace["actual_native_events"], [event["native_actions"] for event in events[first + 1:]]),
            "实际脚本步数或真实原生动作不同")
        start = legacy.read_json(directory / "start.json")
        gl.require(start.get("own_real_history") is True and start.get("start_frame") == first and
                   start.get("comparison_to_reference_required") is False and start["telemetry"] == events[first]["telemetry"] and
                   start["measurement"] == trace["measurements"][0], "真实脚本起点被修改")
        for step, measurement in enumerate(trace["measurements"]):
            frame = first + step
            obs = {name: torch.as_tensor(arrays[name][frame:frame + 1], device=args.device) for name in ("image", "heatmap")}
            encoded = library(obs)[0].cpu().numpy()
            actual = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], encoded,
                                         events[frame]["telemetry"], bank, plan["target_telemetry"])
            gl.require(np.allclose(encoded, measurement["goal_feature"], atol=choices.ATOL, rtol=choices.RTOL) and
                       all(np.isclose(value, measurement[name], atol=choices.ATOL, rtol=choices.RTOL)
                           if name not in ("closest_visual_target", "closest_rgb_target", "closest_position_target")
                           else value == measurement[name] for name, value in actual.items()) and
                       measurement["own_pose"] == events[frame]["telemetry"]["pose"], "逐帧视觉/物理测量不能重现原记录")
        for query in queries:
            frame = query["frame"]
            state = arrays["features"][frame:frame + 1].copy()
            features = torch.as_tensor(state, device=args.device)
            before = features.clone()
            p, raw = history.query(policy, features, bank["goals"], query["remaining"])
            gl.require(torch.equal(before, features), "目标替换改变了本局真实状态")
            entry = dict(run_index=index, seed=run["seed"], repeat=run["repeat"], assigned_target=run["script"],
                **query, script_action_name=plan["action_names"][query["script_action"]],
                phase="common_script_prefix" if query["script_step"] <= shared else "after_script_divergence",
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                distance_to_goal0=trace["measurements"][query["script_step"] - 1]["distance_0"],
                distance_to_goal1=trace["measurements"][query["script_step"] - 1]["distance_1"],
                **choices.frame_metrics(p, raw), **diagnosis.label_metrics(p, query["script_action"], run["script"]),
                probabilities={name: value.tolist() for name, value in p.items()},
                raw_preferences={name: value.tolist() for name, value in raw.items()})
            entry.update(correct_mode=entry[f"goal{run['script']}_mode"], swapped_mode=entry[f"goal{1 - run['script']}_mode"])
            frames.append(entry)
        own = diagnosis.aggregate(frames[-horizon:])["means"]
        print(f"[SCRIPT ACTIONS] run={index + 1}/4 target={run['script']} correct_nll={own['correct_script_nll']:.6f} "
              f"no_goal_gap={own['no_goal_minus_correct_nll']:.6f} swapped_gap={own['swapped_minus_correct_nll']:.6f} "
              f"mode_match={own['correct_script_mode_match']:.4f}", flush=True)
    result = diagnosis.summarize(frames, plan)
    result.update(calibration_evaluation_id=manifest["evaluation_id"], target_content_id=plan["target_content_id"],
                  action_names=plan["action_names"], state_source=report.data["state_source"])
    baseline.write_json(report.directory / "diagnostics.json", result)
    baseline.write_json(report.directory / "action_queries.json", dict(format=diagnosis.FORMAT, frames=frames,
        hypothetical_actions_executed=False, synthetic_training_labels=False, teacher_forced_real_histories=True))
    baseline.write_csv(report.directory / "frame_metrics.csv", [
        {name: value for name, value in row.items() if name not in ("probabilities", "raw_preferences")} for row in frames])
    baseline.write_csv(report.directory / "runs.csv", [dict(
        run_index=row["run_index"], repeat=row["repeat"], assigned_target=row["assigned_target"],
        query_frames=row["query_frames"], first_teacher_forced_mode_mismatch_step=row["first_teacher_forced_mode_mismatch_step"],
        **row["means"]) for row in result["per_run"]])
    probability_gallery(report.directory / "script_action_probabilities.png", frames, plan["action_names"])
    for module, expected, digest in zip(modules, versions, digests):
        gl.require({name: p._version for name, p in module.named_parameters()} == expected and
                   gl.tensor_digest(module.state_dict(), {}) == digest and
                   all(not p.requires_grad and p.grad is None for p in module.parameters()), "诊断改变冻结参数或产生梯度")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "只读动作诊断推进了随机数")
    gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or
                       name == "mineclip" or name.startswith("mineclip.") for name in sys.modules), "只读诊断不应导入MineDojo/MineCLIP")
    report.data.update(query_frames=len(frames), diagnosed_runs=4, same_state_within_query=True,
                       same_state_across_runs=False, first_script_divergence_step=result["first_script_divergence_step"])
    report.check("real_query_alignment", "PASS", "4条完整历史、64个真实动作前状态；frame64–79对应incoming65–80，remaining16–1，无终点后查询")
    report.check("recorded_measurement_roundtrip", "PASS", "68个原逐帧视觉/物理测量重现；原状态按SHA256及已通过history_check固定，没有重算或拼接RSSM")
    report.check("same_state_script_support", "PASS", "同状态/预算的两目标及独立无目标/零目标/底座；真实脚本动作概率/排序/NLL与mode分别报告，不强制目标动作不同")
    report.check("no_training", "PASS", "权重/版本/梯度/RNG未变；无优化器/RSSM构造，无MineDojo/MineCLIP，new_env_steps=0")
    report.check("scope", "WARN", "64个轨迹内查询来自4条历史、一个世界；脚本动作非唯一最优标签，无反事实执行或worker控制验收；T06未推进")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, help="已完整通过的reference_calibration_probe目录")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--residual-verify-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    args.command = "analyze"
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "reference_action_diagnose_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    try:
        manifest, checked_dir, previous_check, protected, files = historical_inputs(args)
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于标定、sample、check、模型、缓存和共享依赖")
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
        bc.require_disk_space(directory, 12 * 1024**2)
        report.require_writable()
        report.check("historical_identity", "PASS", "完整标定/sample/旧check/全部真实产物SHA256与原推理代码一致；绑定同一残差worker和独立verify")
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            analyze(args, report, manifest, checked_dir, previous_check)
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
                             "源轨迹/目标/模型/缓存/验收及旧代码大小和修改时间未变；历史另按原SHA256验真；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
