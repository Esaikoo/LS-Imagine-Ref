"""T05 warmed control: check offline; prepare/evaluate start MineDojo.

Always use MINEDOJO_HEADLESS=1 for prepare/evaluate. No training, original
replay writes, copied old endpoint goals, or automatic retry occurs here.
"""

import argparse
import copy
from datetime import datetime
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_residual_control as original
import t05_start_stability as stability

TRIAL_FILES = ("trajectory.npz", "events.json", "video.mp4", "metrics.json", "control_trace.json",
               "prefix_comparison.json", "start_comparison.json", "start_frame_statistics.csv")


def probe_files(directory):
    return [directory / name for name in ("report.json", "diagnostics.json", "probe_plan.json", "visual_preprocessing.json",
                                          "strict_history_comparison.json")] + [
        directory / f"run_{branch}" / name for branch in (0, 1) for name in ("trajectory.npz", "events.json", "video.mp4")]


class Report(baseline.Report):
    def check(self, name, level, detail):
        if name == "video_preflight":
            detail = "2帧MP4编码及完整解码通过；尚未启动MineDojo"
        return super().check(name, level, detail)

    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_WARMED_CONTROL_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    paths = set(original.code_identity()) | set(stability.code_identity()) | {
        "goal_warmed_control.py", "scripts/t05_warmed_control.py", "scripts/t05_goal_control.py"}
    return {name: bc.file_hash(ROOT / name) for name in sorted(paths)}


def compare(reference, actual, events_reference, events_actual, plan):
    import goal_control as ctl
    import goal_start_stability as starts
    import goal_warmed_control as warmed

    left, ev_left = warmed.prefix(reference, events_reference, plan)
    right, ev_right = warmed.prefix(actual, events_actual, plan)
    rows = stability.history_statistics(left, right, ev_left, ev_right)
    summary = starts.summarize_history(rows, plan["stability_plan"]["tail_frames"])
    strict = ctl.prefix_comparison(left, right)
    return warmed.start_result(summary, strict), strict, rows, summary


def load_inputs(args, report):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control
    import goal_start_stability as starts
    import goal_warmed_control as warmed

    loading = copy.copy(args)
    loading.command = "check"
    cache, runtime = original.load_inputs(loading, report)
    gl.require(runtime.policy.worker_version == 200, "本轮主模型固定残差第200步，不能临时替换模型")
    loading.benchmark_dir = args.source_benchmark_dir
    source_dir, source = original.load_benchmark(loading, report)
    probe_dir = baseline.project_path(args.stability_probe_dir)
    accepted, diagnostic = (legacy.read_json(probe_dir / name) for name in ("report.json", "diagnostics.json"))
    legacy.accepted(accepted, "probe")
    plan = diagnostic["plan"]
    starts.validate_plan(plan)
    required = {"full_causal_history_run_0", "full_causal_history_run_1", "physical_history", "recent_rgb",
                "two_real_histories", "no_training", "source_inputs_unchanged"}
    gl.require(accepted.get("stability_protocol") == starts.PROTOCOL and accepted.get("stability_code") == stability.code_identity() and
               accepted.get("input_identity") == report.data["input_identity"] and
               accepted.get("environment_fingerprint") == report.data["environment_fingerprint"] and
               accepted.get("plan") == plan and diagnostic.get("plan_id") == accepted.get("plan_id") == plan["plan_id"] and
               plan["benchmark_id"] == source["benchmark_id"] and plan["warmup_steps"] == 32 and
               plan["prefix_steps"] == 32 and plan["tail_frames"] == 4 and accepted.get("completed_runs") == 2 and
               accepted.get("new_env_steps") == plan["maximum_new_env_steps"] and accepted.get("optimizer_updates") == 0 and
               accepted.get("worker_control_actions") == 0 and accepted.get("candidate_preflight_passed") is True and
               required <= {row["name"] for row in accepted["checks"] if row["level"] == "PASS"},
               "需要同模型/环境/代码、固定32步预热和4帧窗口的已通过双次probe；不重跑旧预检")
    envelope = legacy.read_json(probe_dir / "probe_plan.json")
    gl.require(envelope["plan"] == plan and envelope["input_identity"] == report.data["input_identity"] and
               envelope["environment_fingerprint"] == report.data["environment_fingerprint"], "probe预声明计划与验收不同")
    histories, events = [], []
    for branch in (0, 1):
        arrays, ev = stability.pixels.read_reference(probe_dir / f"run_{branch}", plan["control_start_frame"] + 1)
        starts.validate_real_history(arrays, ev, plan)
        if args.command == "check":
            expected = runtime.state_encoder.rollout(arrays, plan["control_start_frame"])
            t02.vector_difference(report, f"accepted_probe_history_{branch}", expected, arrays["features"], check=True)
        histories.append(arrays)
        events.append(ev)
    measured = starts.summarize_history(stability.history_statistics(histories[0], histories[1], events[0], events[1]), plan["tail_frames"])
    gl.require(measured == diagnostic["summary"] and measured["candidate_preflight_passed"] and
               diagnostic.get("model_id") == runtime.policy.model_id and diagnostic.get("worker_version") == 200 and
               diagnostic.get("completed_runs") == 2 and diagnostic.get("new_env_steps") == 128 and
               diagnostic.get("optimizer_updates") == diagnostic.get("worker_control_actions") == 0,
               "probe真实数据与稳定性结论或模型计数不同")
    case = next(case for case in source["cases"] if case["seed"] == plan["seed"])
    gl.require(case["scenario"] == plan["scenario"] and case["prefix_actions"] == plan["prefix_actions"] and
               source["horizon"] == 16 and source["horizon"] <= runtime.bundle["horizon"], "旧动作来源的场景/前缀/16步预算不同")
    branches = []
    for branch in (0, 1):
        trajectory = ctl.read_trajectory(source_dir / case["directory"] / f"reference_{branch}")
        action = trajectory["action"]
        first = source["prefix_steps"]
        gl.require(len(action) == first + source["horizon"] + 1 and np.all(action[0] == 0) and
                   np.all((action[1:] == 0) | (action[1:] == 1)) and np.all(action[1:].sum(-1) == 1) and
                   action[1:first + 1].argmax(-1).tolist() == plan["prefix_actions"], "旧参考真实动作脚本不正确")
        branches.append(action[first + 1:].argmax(-1).tolist())
    design = dict(seeds=[plan["seed"]], repetitions=args.repetitions, randomization_seed=args.randomization_seed,
                  execution_policy=args.execution_policy, modes=list(control.MODES),
                  schedule=control.schedule([plan["seed"]], args.repetitions, args.randomization_seed))
    control.validate_design(design)
    reference_plan = warmed.make_reference_plan(plan, branches, source["horizon"], source["min_goal_distance"], design)
    provenance = dict(source_benchmark_id=source["benchmark_id"], source_directory=str(source_dir),
        source_report_sha256=bc.file_hash(source_dir / "report.json"), source_benchmark_sha256=bc.file_hash(source_dir / "benchmark.json"),
        stability_probe_directory=str(probe_dir), stability_report_sha256=bc.file_hash(probe_dir / "report.json"),
        stability_diagnostics_sha256=bc.file_hash(probe_dir / "diagnostics.json"),
        probe_files={str(path.relative_to(probe_dir)): bc.file_hash(path) for path in probe_files(probe_dir)})
    report.data.update(evaluation_code=code_identity(), comparison_protocol=warmed.PROTOCOL,
                       reference_plan=reference_plan, reference_plan_id=reference_plan["reference_plan_id"],
                       provenance=provenance, old_goal_vectors_reused=False, approved_benchmark=False)
    if args.command != "check":
        checked = legacy.read_json(baseline.project_path(args.check_dir) / "report.json")
        legacy.accepted(checked, "check")
        gl.require(checked.get("comparison_protocol") == warmed.PROTOCOL and checked.get("evaluation_code") == code_identity() and
                   checked.get("input_identity") == report.data["input_identity"] and checked.get("reference_plan") == reference_plan and
                   checked.get("provenance") == provenance and checked.get("environment_fingerprint") == report.data["environment_fingerprint"] and
                   {"new_reference_contract", "residual_execution_contract", "no_training", "source_inputs_unchanged"} <= {
                       row["name"] for row in checked["checks"] if row["level"] == "PASS"}, "需要同一新预热设计的check；旧check不能代替")
    report.check("accepted_stability_probe", "PASS", "复用已通过双次probe，核对真实历史和当前模型/环境；不新增环境预检")
    report.check("fixed_reference_scripts", "PASS", "仅复用两个原真实16步动作脚本；新目标必须来自新预热执行的真实终点")
    return cache, runtime, reference_plan


def check(args, report, cache, runtime, plan):
    import numpy as np
    import goal_start_stability as starts
    import goal_warmed_control as warmed

    # Reuse accepted causal/distribution and complete-block contracts.
    original.check(args, report, cache, runtime)
    changed = copy.deepcopy(plan)
    changed["old_goal_vectors_reused"] = True
    t02.rejection(report, "old_goal_guard", lambda: warmed.validate_reference_plan(changed), "身份")
    changed = copy.deepcopy(plan)
    changed["branch_actions"][0][0] = (changed["branch_actions"][0][0] + 1) % len(plan["stability_plan"]["action_names"])
    t02.rejection(report, "reference_script_guard", lambda: warmed.validate_reference_plan(changed), "身份")
    markers = [dict(frame=frame, rgb_mae=0., rgb_p99=0., heatmap_mae=0., state_relative_l2=0., position=0., angle=0.,
        reward=0., missing_telemetry=False, position_delta=dict(x=0, y=0, z=0), **{name: True for name in starts.FLAGS})
        for frame in range(plan["control_start_frame"] + 1)]
    markers[0].update(rgb_mae=20., rgb_p99=40.)
    summary = starts.summarize_history(markers, plan["stability_plan"]["tail_frames"])
    starts.require(warmed.start_result(summary, dict(passed=False))["passed"], "早期RGB错误决定新近期门槛")
    markers[0]["position"] = 0.1
    starts.require(not warmed.start_result(starts.summarize_history(markers, 4), dict(passed=False))["passed"], "预热期间物理偏差未拒绝")
    # Index markers isolate the new 64->65 action boundary; never persisted as data.
    size, count = plan["control_start_frame"] + plan["horizon"] + 1, len(plan["stability_plan"]["action_names"])
    arrays = {name: np.zeros(size, np.bool_) for name in ("is_first", "is_last", "is_terminal")}
    arrays["is_first"][0] = True
    actions = plan["stability_plan"]["actions"] + plan["branch_actions"][0]
    arrays.update(image=np.zeros((size, 64, 64, 3), np.uint8), heatmap=np.zeros((size, 64, 64), np.uint8),
        features=np.zeros((size, 1), np.float32), obs_reward=np.zeros((size, 1), np.float32),
        action=np.concatenate((np.zeros((1, count), np.float32), np.eye(count, dtype=np.float32)[actions])))
    events = [dict(frame=frame, native_actions=[], error=None, telemetry={}) for frame in range(size)]
    warmed.validate_reference_history(arrays, events, plan, 0)
    bad = copy.deepcopy(arrays)
    bad["action"][plan["control_start_frame"] + 1] = np.roll(bad["action"][plan["control_start_frame"] + 1], 1)
    t02.rejection(report, "control_boundary_guard", lambda: warmed.validate_reference_history(bad, events, plan, 0), "下一动作")
    bad["action"] = arrays["action"].copy()
    bad["is_first"][32] = True
    t02.rejection(report, "warmup_reset_guard", lambda: warmed.validate_reference_history(bad, events, plan, 0), "伪reset")
    baseline.write_json(report.directory / "reference_plan.json", plan)
    report.check("new_reference_contract", "PASS", "预热64->65真实下一动作、固定脚本和新终点来源正确；内存marker不写入数据")
    report.check("scope", "WARN", "离线新入口验收；新参考和真实控制尚未执行，仍为T05")


def save_environment(session, directory):
    import goal_control as ctl
    if session is not None:
        try:
            ctl.save_session(session, directory, baseline.write_video)
        finally:
            session.close()
    gc.collect()


def own_history(report, runtime, trajectory, name):
    expected = runtime.state_encoder.rollout(trajectory, len(trajectory["image"]) - 1)
    t02.vector_difference(report, name, expected, trajectory["features"], check=True)


def prepare(args, report, cache, runtime, plan):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_warmed_control as warmed
    from PIL import Image, ImageDraw

    baseline.video_preflight(report)
    bc.require_disk_space(report.directory, 48 * 1024**2)
    baseline.write_json(report.directory / "reference_plan.json", dict(plan=plan, evaluation_code=code_identity(),
        input_identity=report.data["input_identity"], environment_fingerprint=report.data["environment_fingerprint"], provenance=report.data["provenance"]))
    baseline.write_json(report.directory / "schedule.json", plan["design"])
    report.check("predeclared_references", "PASS", f"两条固定新参考，最多{plan['maximum_reference_env_steps']}步；"
                 f"随机{len(plan['design']['schedule'])}次顺序已保存，无补跑")
    preflight, first = plan["stability_plan"], plan["control_start_frame"]
    seed, case_dir = preflight["seed"], report.directory / f"case_{preflight['seed']}"
    case_dir.mkdir()
    baseline.write_json(case_dir / "scenario.json", preflight["scenario"])
    old_visual = legacy.read_json(baseline.project_path(args.source_benchmark_dir) / f"case_{seed}/visual_preprocessing.json")["reference_0"]
    references, events, settings = [], [], {}
    start_result = None
    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
    for branch in (0, 1):
        session, output = None, case_dir / f"reference_{branch}"
        print(f"[ENV] warmed reference={branch} seed={seed} world_seed={preflight['scenario']['world_seed']}; fresh reset", flush=True)
        try:
            session = ctl.Session(runtime, preflight["scenario"], output / "environment", on_step)
            gl.require(session.specs["visual_preprocessing"] == old_visual, "新参考改变原任务RGB/HUD设置")
            settings[f"reference_{branch}"] = session.specs["visual_preprocessing"]
            for action in preflight["actions"]:
                session.step(action)
                gl.require(not session.done, "新参考预热/前缀提前结束；不补跑")
            warmed.prefix(session.arrays(), session.events, plan)
            if branch == 1:
                start_result, strict, rows, summary = compare(references[0], session.arrays(), events[0], session.events, plan)
                baseline.write_json(case_dir / "prefix_comparison.json", strict)
                baseline.write_json(case_dir / "start_comparison.json", start_result)
                left, _ = warmed.prefix(references[0], events[0], plan)
                stability.export_gallery(left, session.arrays(), rows, summary, case_dir / "start_diagnostics")
                gl.require(start_result["passed"], "新参考起点不满足完整物理/近期RGB门槛；保留失败，不重试或替换脚本")
            for action in plan["branch_actions"][branch]:
                session.step(action)
                if session.done:
                    break
        finally:
            save_environment(session, output)
        trajectory = ctl.read_trajectory(output)
        ev = legacy.read_json(output / "events.json")
        warmed.validate_reference_history(trajectory, ev, plan, branch)
        own_history(report, runtime, trajectory, f"new_reference_history_{branch}")
        references.append(trajectory)
        events.append(ev)
        report.data["completed_references"] = len(references)
        report.save()
        print(f"[REFERENCE {branch + 1}/2] frames={len(trajectory['image'])}; endpoint from new real execution", flush=True)
    images = np.stack([reference["image"][-1] for reference in references])
    heatmaps = np.stack([reference["heatmap"][-1] for reference in references])
    goals = np.stack([ctl.encode_goal(runtime, dict(image=image, heatmap=heat)) for image, heat in zip(images, heatmaps)])
    gl.require(goals.shape == (2, cache.metadata["goal_dim"]) and np.isfinite(goals).all() and
               np.allclose(np.linalg.norm(goals, axis=-1), 1, atol=1e-5), "新真实目标编码异常")
    separation = float(ctl.distances(goals[:1], goals[1:])[0, 0])
    np.savez_compressed(case_dir / "goals.npz", goals=goals, images=images, heatmaps=heatmaps)
    baseline.write_json(case_dir / "visual_preprocessing.json", settings)
    baseline.write_json(case_dir / "reference_diagnostics.json", dict(goal_separation=separation,
        minimum=plan["min_goal_distance"], start_comparison=start_result, goal_source="new_real_endpoint_only",
        old_goal_vectors_reused=False, endpoint_frames=[first + plan["horizon"]] * 2))
    panel = Image.new("RGB", (660, 160), "white")
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(((references[0]["image"][first], "new control start"),
                                           (images[0], "new real target 0"), (images[1], "new real target 1"))):
        draw.text((index * 220 + 3, 3), label, fill="black")
        panel.paste(Image.fromarray(image).resize((128, 128)), (index * 220, 22))
    panel.save(report.directory / "targets.png")
    gl.require(separation >= plan["min_goal_distance"], "两个新目标未满足原可区分门槛；保存真实参考，不挑选更有利终点")
    gl.require(report.data["new_env_steps"] == plan["maximum_reference_env_steps"] and start_result["passed"], "新参考完整计数/起点验收异常")
    manifest = dict(format=warmed.FORMAT, comparison_protocol=warmed.PROTOCOL, reference_plan=plan,
        reference_plan_id=plan["reference_plan_id"], provenance=report.data["provenance"],
        protocol=ctl.ENVIRONMENT_PROTOCOL, environment_fingerprint=report.data["environment_fingerprint"],
        evaluation_code=code_identity(), input_identity=report.data["input_identity"], model_id=runtime.policy.model_id,
        worker_version=runtime.policy.worker_version, design=plan["design"], prefix_steps=first, horizon=plan["horizon"],
        pair_limits=ctl.PAIR_LIMITS, same_hidden_state=False, controlled_reachability_verified=False,
        old_goal_vectors_reused=False, goal_source="new_real_endpoint_only", cases=[dict(seed=seed, directory=case_dir.name,
            scenario=preflight["scenario"], prefix_actions=preflight["actions"], goal_separation=separation,
            files={str(path.relative_to(report.directory)): bc.file_hash(path) for path in sorted(case_dir.rglob("*")) if path.is_file()})])
    manifest["benchmark_id"] = legacy.benchmark_id(manifest)
    baseline.write_json(report.directory / "benchmark.json", manifest)
    report.data.update(benchmark_id=manifest["benchmark_id"], approved_benchmark=True, completed_references=2, goal_separation=separation)
    report.check("new_real_endpoints", "PASS", f"两个新81帧参考/终点目标已保存和编码；目标距离={separation:.5f}，未复制旧目标")
    report.check("reference_start_acceptance", "PASS", "新参考完整物理历史和最后4帧RGB通过；原严格历史结果保留")
    report.check("predeclared_design", "PASS", f"新benchmark绑定预热、当前模型/代码和六组{len(plan['design']['schedule'])}次随机顺序；先执行一个12次块")
    report.check("scope", "WARN", "新参考准备工程验收，不代表worker会到达；尚未执行目标控制，仍为T05")


def load_benchmark(args, report, runtime, plan):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_warmed_control as warmed

    directory = baseline.project_path(args.benchmark_dir)
    manifest, record = (legacy.read_json(directory / name) for name in ("benchmark.json", "report.json"))
    legacy.accepted(record, "prepare")
    gl.require(manifest.get("format") == warmed.FORMAT and manifest.get("comparison_protocol") == warmed.PROTOCOL and
               manifest.get("reference_plan") == plan and manifest.get("reference_plan_id") == plan["reference_plan_id"] and
               manifest.get("benchmark_id") == legacy.benchmark_id(manifest) == record.get("benchmark_id") and
               manifest.get("input_identity") == report.data["input_identity"] and manifest.get("provenance") == report.data["provenance"] and
               manifest.get("evaluation_code") == code_identity() and manifest.get("environment_fingerprint") == report.data["environment_fingerprint"] and
               manifest.get("protocol") == ctl.ENVIRONMENT_PROTOCOL and manifest.get("model_id") == runtime.policy.model_id and
               manifest.get("worker_version") == runtime.policy.worker_version and manifest.get("horizon") == plan["horizon"] and
               manifest.get("old_goal_vectors_reused") is False and manifest.get("same_hidden_state") is False and
               manifest.get("goal_source") == "new_real_endpoint_only" and record.get("approved_benchmark") is True and
               record.get("new_env_steps") == plan["maximum_reference_env_steps"] and record.get("optimizer_updates") == 0 and
               {"new_real_endpoints", "reference_start_acceptance", "no_training", "source_inputs_unchanged"} <= {
                   row["name"] for row in record["checks"] if row["level"] == "PASS"}, "需要当前新预热prepare已通过的benchmark；旧adopt不能替代")
    gl.require(len(manifest["cases"]) == 1 and manifest["prefix_steps"] == plan["control_start_frame"] and
               manifest["design"] == plan["design"] and manifest["pair_limits"] == ctl.PAIR_LIMITS, "新benchmark时间边界/门槛/随机设计异常")
    case = manifest["cases"][0]
    gl.require(case["seed"] == plan["stability_plan"]["seed"] and case["directory"] == f"case_{case['seed']}" and
               case["scenario"] == plan["stability_plan"]["scenario"] and case["prefix_actions"] == plan["stability_plan"]["actions"], "新case前缀或场景不同")
    required = {str(Path(case["directory"]) / name) for name in ("scenario.json", "goals.npz", "reference_diagnostics.json",
                                                              "visual_preprocessing.json", "prefix_comparison.json", "start_comparison.json")}
    required |= {str(Path(case["directory"]) / f"reference_{branch}" / name) for branch in (0, 1) for name in ("trajectory.npz", "events.json", "video.mp4")}
    gl.require(required <= set(case["files"]), "新目标/参考产物声明缺失")
    for relative, digest in case["files"].items():
        path = (directory / relative).resolve()
        gl.require(directory in path.parents and path.is_file() and bc.file_hash(path) == digest, "新真实参考文件SHA256不同")
    references, events = [], []
    for branch in (0, 1):
        reference = ctl.read_trajectory(directory / case["directory"] / f"reference_{branch}")
        ev = legacy.read_json(directory / case["directory"] / f"reference_{branch}/events.json")
        warmed.validate_reference_history(reference, ev, plan, branch)
        references.append(reference)
        events.append(ev)
    start, strict, _, _ = compare(references[0], references[1], events[0], events[1], plan)
    gl.require(start["passed"] and start == legacy.read_json(directory / case["directory"] / "start_comparison.json") and
               strict == legacy.read_json(directory / case["directory"] / "prefix_comparison.json"), "新参考起点统计与保存结果不同")
    with np.load(directory / case["directory"] / "goals.npz", allow_pickle=False) as data:
        goals = data["goals"].copy()
        gl.require(goals.shape == (2, runtime.policy.identity["goal_dim"]) and np.isfinite(goals).all() and
                   np.allclose(np.linalg.norm(goals, axis=-1), 1, atol=1e-5), "新目标向量异常")
        for branch, reference in enumerate(references):
            gl.require(np.array_equal(data["images"][branch], reference["image"][-1]) and
                       np.array_equal(data["heatmaps"][branch], reference["heatmap"][-1]) and
                       np.allclose(ctl.encode_goal(runtime, dict(image=reference["image"][-1], heatmap=reference["heatmap"][-1])),
                                   goals[branch], atol=3e-5, rtol=3e-5), "目标必须逐值匹配新真实终点和冻结编码器")
    gl.require(float(ctl.distances(goals[:1], goals[1:])[0, 0]) >= plan["min_goal_distance"], "新目标不再可区分")
    report.data["benchmark_id"] = manifest["benchmark_id"]
    report.check("new_benchmark_identity", "PASS", "新参考SHA256、真实终点重编码、物理/近期RGB起点和预声明计划一致")
    return directory, manifest, goals


def run_trial(report, runtime, benchmark_dir, manifest, goals, cell, output):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as repeated
    import goal_library as gl
    import goal_residual_control as control
    import goal_warmed_control as warmed

    plan, case = manifest["reference_plan"], manifest["cases"][0]
    first, horizon = plan["control_start_frame"], plan["horizon"]
    target, mode, repeat = cell["target"], cell["mode"], cell["repeat"]
    assigned = 1 - target if mode == "swapped_goal" else target
    reference_dir = benchmark_dir / case["directory"]
    reference = ctl.read_trajectory(reference_dir / "reference_0")
    reference_events = legacy.read_json(reference_dir / "reference_0/events.json")
    endpoint = ctl.read_trajectory(reference_dir / f"reference_{target}")
    uniforms_seed = (case["seed"] + manifest["design"]["randomization_seed"] + repeat * 1009 + 701) % (2**31 - 1)
    uniforms = np.random.RandomState(uniforms_seed).uniform(size=horizon)
    session, strict, start, rows = None, None, None, None
    def on_step():
        report.data["evaluation_env_steps"] += 1
        report.data["new_env_steps"] += 1
    print(f"[ENV] warmed seed={case['seed']} world_seed={case['scenario']['world_seed']} repeat={repeat} "
          f"target={target} mode={mode}; fresh reset", flush=True)
    try:
        session = ctl.Session(runtime, case["scenario"], output / "environment", on_step)
        visual = legacy.read_json(reference_dir / "visual_preprocessing.json")["reference_0"]
        gl.require(session.specs["visual_preprocessing"] == visual, "真实评估RGB/HUD协议不同")
        for action in case["prefix_actions"]:
            session.step(action)
            gl.require(not session.done, "真实预热/前缀提前结束；保留数据，不补跑")
        start, strict, rows, _ = compare(reference, session.arrays(), reference_events, session.events, plan)
        own_history(report, runtime, session.arrays(), f"trial_history_{output.name}")
        row = dict(cell, directory=output.name, artifact_dir=str(output), same_hidden_state=False,
                   start_eligible=bool(start["passed"]), execution_valid=False, start_comparison=start,
                   control_start_frame=first, warmup_steps=plan["stability_plan"]["warmup_steps"],
                   horizon=horizon, model_id=runtime.policy.model_id, reference_plan_id=plan["reference_plan_id"])
        if not start["passed"]:
            row.update(reason="full physical/recent RGB/action prefix mismatch; no retry", actual_steps=0)
            return row, dict(scope="failed measured start; no worker control", model_id=runtime.policy.model_id)
        actions, distributions, frames = [], [], []
        for step in range(horizon):
            remaining = horizon - step
            if mode == "reference_replay":
                action = int(endpoint["action"][first + step + 1].argmax())
                probabilities = ctl.onehot(action, runtime.bundle["config"]["num_actions"])
            else:
                probabilities = control.probabilities(runtime, session.features, goals, remaining, target, mode)
                action = ctl.select_action(probabilities, manifest["design"]["execution_policy"], float(uniforms[step]))
            actions.append(action)
            distributions.append(probabilities)
            frames.append(len(session.rows) - 1)
            session.step(action)
            if session.done:
                break
        trajectory = session.arrays()
        own_history(report, runtime, trajectory, f"trial_control_history_{output.name}")
        trajectory["goal_features"] = np.stack([ctl.encode_goal(runtime, observation) for observation in session.rows])
        metrics = ctl.trial_metrics(trajectory, goals, target, first, endpoint["telemetry"][-1])
        distances = ctl.distances(trajectory["goal_features"][first:], goals)
        start_margin = float(distances[0, 1 - target] - distances[0, target])
        control_events = session.events[first + 1:]
        row.update(metrics, execution_valid=True, actual_steps=len(actions),
                   issued_goal=assigned if mode in ("goal", "swapped_goal") else None,
                   goal_input="zero" if mode == "zero_goal" else "unused" if mode in ("no_goal", "base", "reference_replay") else "new_real_endpoint",
                   end_reason="environment_done" if session.done else "fixed_budget",
                   task_success=any(event["success"] for event in control_events),
                   task_success_during_prefix=any(event["success"] for event in session.events[:first + 1]),
                   external_return=float(sum(event["reward"] for event in control_events)),
                   start_target_preference_margin=start_margin, preference_improvement=metrics["target_preference_margin"] - start_margin,
                   end_distances_to_both_goals=distances[-1].tolist(),
                   issued_target_margin=float(distances[-1, 1 - assigned] - distances[-1, assigned]) if mode in ("goal", "swapped_goal") else None,
                   actions=actions, action_histogram=np.bincount(actions, minlength=runtime.bundle["config"]["num_actions"]).tolist())
        if mode == "reference_replay":
            replay = ctl.prefix_comparison(endpoint, trajectory) if len(endpoint["image"]) == len(trajectory["image"]) else dict(passed=False, reason="real ending shortened replay")
            expected_events = legacy.read_json(reference_dir / f"reference_{target}/events.json")
            expected_actions = endpoint["action"][first + 1:].argmax(-1).tolist()
            row.update(reference_replay_valid=bool(replay["passed"]), reference_replay_check=replay,
                       reference_macro_actions_equal=actions == expected_actions[:len(actions)], reference_budget_completed=len(actions) == horizon,
                       reference_native_actions_equal=repeated.native_actions_equal(
                           [event["native_actions"] for event in control_events],
                           [event["native_actions"] for event in expected_events[first + 1:first + 1 + len(actions)]]))
            row["execution_valid"] = row["reference_macro_actions_equal"]
        trace = dict(model_id=runtime.policy.model_id, worker_version=runtime.policy.worker_version,
            reference_plan_id=plan["reference_plan_id"], condition=mode, evaluated_target=target, issued_target=row["issued_goal"],
            assigned_goal=goals[assigned].tolist() if row["issued_goal"] is not None else None,
            execution_policy="recorded_actions" if mode == "reference_replay" else manifest["design"]["execution_policy"],
            action_seed=uniforms_seed, uniforms=[] if mode == "reference_replay" else uniforms[:len(actions)].tolist(),
            control_start_frame=first, frames=frames, action_ids=actions, probabilities=np.asarray(distributions).tolist(),
            remaining=list(range(horizon, horizon - len(actions), -1)), distances_to_both_goals=distances.tolist())
        return row, trace
    finally:
        if session is not None:
            try:
                ctl.save_session(session, output, baseline.write_video)
                if strict is not None:
                    baseline.write_json(output / "prefix_comparison.json", strict)
                    baseline.write_json(output / "start_comparison.json", start)
                    baseline.write_csv(output / "start_frame_statistics.csv", rows)
            finally:
                session.close()
        gc.collect()


def previous_rows(args, report, manifest):
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_control as control
    import goal_warmed_control as warmed

    if not args.previous_eval_dir:
        return [], []
    directory = baseline.project_path(args.previous_eval_dir)
    record, saved = (legacy.read_json(directory / name) for name in ("report.json", "evaluation_manifest.json"))
    legacy.accepted(record, "evaluate")
    gl.require(saved.get("format") == warmed.EVALUATION_FORMAT and saved.get("comparison_protocol") == warmed.PROTOCOL and
               saved.get("evaluation_id") == original.evaluation_id(saved) == record.get("evaluation_id") and
               saved.get("benchmark_id") == manifest["benchmark_id"] and saved.get("reference_plan_id") == manifest["reference_plan_id"] and
               saved.get("input_identity") == report.data["input_identity"] and saved.get("evaluation_code") == code_identity() and
               saved.get("environment_fingerprint") == report.data["environment_fingerprint"] and
               saved.get("execution_policy") == manifest["design"]["execution_policy"], "已有评估不是同一新预热计划；禁止续跑旧24次")
    gl.require({"stage_execution", "comparable_blocks", "no_training", "source_inputs_unchanged"} <= {
                   row["name"] for row in record["checks"] if row["level"] == "PASS"} and
               record.get("completed_trials") == saved["completed_trials"] == len(saved["rows"]) and
               record.get("cumulative_evaluation_env_steps") == saved["cumulative_evaluation_env_steps"] and
               record.get("optimizer_updates") == 0 and bool(saved["rows"]), "前段未通过完整块、冻结参数或计数验收")
    control.pending_cells(manifest["design"], saved["rows"], args.blocks)
    origins = saved["source_evaluations"] + [dict(directory=str(directory), evaluation_id=saved["evaluation_id"],
        manifest_sha256=bc.file_hash(directory / "evaluation_manifest.json"), report_sha256=bc.file_hash(directory / "report.json"))]
    owners = {Path(item["directory"]).resolve() for item in origins}
    for row in saved["rows"]:
        artifact = Path(row["artifact_dir"]).resolve()
        gl.require(artifact.parent in owners and artifact.name == row["directory"] ==
                   f"seed_{row['seed']}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}", "已有trial目录身份异常")
        for name in TRIAL_FILES:
            path = artifact / name
            gl.require(path.is_file() and bc.file_hash(path) == saved["artifacts"].get(str(path)), "已有轨迹/起点报告/视频SHA256不同")
        gl.require(legacy.read_json(artifact / "metrics.json") == row, "已有逐trial与manifest不同")
    for origin in saved["source_evaluations"]:
        root = Path(origin["directory"])
        gl.require(bc.file_hash(root / "evaluation_manifest.json") == origin["manifest_sha256"] and
                   bc.file_hash(root / "report.json") == origin["report_sha256"], "已有分段来源改变")
    report.data.update(previous_evaluation_env_steps=saved["cumulative_evaluation_env_steps"], inherited_trials=len(saved["rows"]))
    report.check("complete_block_continuation", "PASS", f"核对已有{len(saved['rows'])}条新预热trial内容；只执行剩余完整块")
    return saved["rows"], origins


def evaluate(args, report, cache, runtime, plan):
    import goal_bc as bc
    import goal_residual_control as control
    import goal_warmed_control as warmed

    directory, manifest, goals = load_benchmark(args, report, runtime, plan)
    rows, origins = previous_rows(args, report, manifest)
    pending = control.pending_cells(manifest["design"], rows, args.blocks)
    inherited = len(rows)
    baseline.video_preflight(report)
    bc.require_disk_space(report.directory, len(pending) * 16 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "schedule.json", manifest["design"])
    baseline.write_json(report.directory / "stage_schedule.json", dict(inherited_trials=inherited, cells=pending,
        maximum_new_env_steps=len(pending) * (plan["control_start_frame"] + plan["horizon"])))
    report.check("storage_preflight", "PASS", f"检查本段{len(pending)}次轨迹/视频余量；不复制大型模型或缓存")
    report.check("control_groups", "PASS", "六组使用新真实目标；固定32步noop+32步共同前缀，完整物理历史/最后4帧RGB门槛")
    for cell in pending:
        report.require_writable()
        name = f"seed_{cell['seed']}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['mode']}"
        output = report.directory / name
        row, trace = run_trial(report, runtime, directory, manifest, goals, cell, output)
        baseline.write_json(output / "metrics.json", row)
        baseline.write_json(output / "control_trace.json", trace)
        rows.append(row)
        t04.append_json(report.directory / "trials.jsonl", row)
        baseline.write_json(report.directory / "diagnostics.json", warmed.summarize(rows, manifest["design"], plan))
        report.data.update(completed_trials=len(rows), newly_completed_trials=len(rows) - inherited)
        report.save()
        if row["start_eligible"]:
            detail = f"d={row['end_distance']:.5f} margin={row['target_preference_margin']:.5f} steps={row['actual_steps']}"
        else:
            start = row["start_comparison"]
            detail = f"START MISMATCH checks={start['failed_checks']} recent_mae={start['recent_rgb_window']['rgb_mae']:.4f} "
            detail += f"p99={start['recent_rgb_window']['rgb_p99']:.1f}; block excluded; no retry"
        print(f"[TRIAL {len(rows)}/{len(manifest['design']['schedule'])}] repeat={cell['repeat']} target={cell['target']} mode={cell['mode']} {detail}", flush=True)
    summary = warmed.summarize(rows, manifest["design"], plan)
    baseline.write_json(report.directory / "diagnostics.json", summary)
    baseline.write_json(report.directory / "cumulative_trials.json", rows)
    baseline.write_csv(report.directory / "trials.csv", [{key: row.get(key) for key in (
        "seed", "repeat", "target", "mode", "start_eligible", "execution_valid", "actual_steps", "task_success", "control_start_frame",
        "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "preference_improvement",
        "reference_replay_valid", "reference_native_actions_equal", "artifact_dir")} for row in rows])
    original.gallery(report, manifest, rows)
    saved = dict(format=warmed.EVALUATION_FORMAT, comparison_protocol=warmed.PROTOCOL,
        benchmark_id=manifest["benchmark_id"], reference_plan_id=plan["reference_plan_id"], input_identity=report.data["input_identity"],
        evaluation_code=code_identity(), environment_fingerprint=report.data["environment_fingerprint"],
        execution_policy=manifest["design"]["execution_policy"], completed_trials=len(rows), rows=rows, source_evaluations=origins,
        cumulative_evaluation_env_steps=report.data.get("previous_evaluation_env_steps", 0) + report.data["evaluation_env_steps"],
        artifacts={str(Path(row["artifact_dir"]) / name): bc.file_hash(Path(row["artifact_dir"]) / name) for row in rows for name in TRIAL_FILES})
    saved["evaluation_id"] = original.evaluation_id(saved)
    baseline.write_json(report.directory / "evaluation_manifest.json", saved)
    report.data.update(evaluation_id=saved["evaluation_id"], completed_trials=len(rows), eligible_trials=summary["eligible_trials"],
        planned_trials=summary["planned_trials"], full_design_completed=len(rows) == summary["planned_trials"],
        cumulative_evaluation_env_steps=saved["cumulative_evaluation_env_steps"], task_success_evaluated=True)
    report.check("stage_execution", "PASS", f"按新随机顺序完成{len(pending)}次，本计划累计{len(rows)}/{summary['planned_trials']}；无重试")
    report.check("comparable_blocks", "PASS" if summary["eligible_trials"] == len(rows) else "FAIL", f"完整可比块纳入{summary['eligible_trials']}/{len(rows)}；失败全保留")
    report.check("calibration", "PASS", "参考重放的原生动作/终点/严格状态波动为诊断，不按未来结果排除或补样")
    report.check("design_completion", "PASS" if report.data["full_design_completed"] else "WARN", "完整设计已完成" if report.data["full_design_completed"] else "首段12次仅用于接口和行为诊断；分析后决定剩余重复")
    report.check("behavior_acceptance", "WARN", "工程PASS不证明目标控制；需分析无目标优势、目标切换方向和参考波动；T06未开始")


def source_paths(args):
    import goal_library as gl

    protected, files, source = original.source_paths(args)
    old = baseline.project_path(args.source_benchmark_dir)
    probe_dir = baseline.project_path(args.stability_probe_dir)
    protected.extend((old, probe_dir))
    files.extend(path for path in old.rglob("*") if path.is_file())
    files.extend(probe_files(probe_dir))
    if args.command != "check":
        files.append(baseline.project_path(args.check_dir) / "reference_plan.json")
    if getattr(args, "previous_eval_dir", None):
        previous = baseline.project_path(args.previous_eval_dir)
        manifest = legacy.read_json(previous / "evaluation_manifest.json")
        for row in manifest["rows"]:
            files.extend(Path(row["artifact_dir"]) / name for name in TRIAL_FILES)
    gl.require(all(path.is_file() for path in files), "依赖缺失；请保留模型/缓存/旧动作来源、已通过probe及前段新评估")
    return list(dict.fromkeys(path.resolve() for path in protected)), list(dict.fromkeys(path.resolve() for path in files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "evaluate"):
        command = commands.add_parser(name)
        for option in ("cache-dir", "checkpoint", "residual-verify-dir", "source-benchmark-dir", "stability-probe-dir"):
            command.add_argument("--" + option, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
        command.add_argument("--repetitions", type=int, default=3)
        command.add_argument("--randomization-seed", type=int, default=0)
        command.add_argument("--execution-policy", choices=("mode", "sample"), default="mode")
        if name == "check":
            command.add_argument("--prefix-steps", type=int, default=32, help="只读真实回放接口检查长度，不改变64步环境前缀")
        else:
            command.add_argument("--check-dir", required=True)
        if name == "evaluate":
            command.add_argument("--benchmark-dir", required=True)
            command.add_argument("--previous-eval-dir")
            command.add_argument("--blocks", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 3 or not 0 <= args.randomization_seed < 2**31 - 10000:
        parser.error("至少3次重复；随机种子必须为合法非负整数")
    if args.command == "check" and args.prefix_steps < 1:
        parser.error("只读检查prefix-steps至少1")
    if args.command == "evaluate" and args.blocks < 1:
        parser.error("blocks至少1")
    if args.command in ("prepare", "evaluate") and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = source_paths(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "warmed_control_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、缓存、旧参考、probe和前段新评估")
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
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        else:
            gl.require(args.command == "check", "真实MineCLIP路径需要CUDA")
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化模型变化")
        gl.require(all(baseline.file_signature(Path(episode["path"])) == episode["signature"] for episode in source["episodes"]), "原真实回放变化")
        rng = bc.capture_rng(args.device)
        cache, runtime, plan = load_inputs(args, report)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            {"check": check, "prepare": prepare, "evaluate": evaluate}[args.command](args, report, cache, runtime, plan)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()), "评估改变冻结模型或产生梯度")
        report.check("no_training", "PASS", f"optimizer_updates=0；全部模型/目标库参数未变；本段真实env.step={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型/回放/旧参考/probe/前段产物未变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
