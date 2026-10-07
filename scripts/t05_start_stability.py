"""T05 audit/check are offline; probe runs exactly two warmed-up replays.

Use MINEDOJO_HEADLESS=1 for probe. This tool does not train, issue worker
control actions, change old acceptance, or create a control benchmark.
"""

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t05_prefix_diagnose as pixels


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        print(f"[{'FAIL' if failed else 'PASS'}] T05_START_STABILITY_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_start_stability as starts
    return {name: starts.file_hash(ROOT / name) for name in (
        "goal_start_stability.py", "scripts/t05_start_stability.py", "scripts/t05_prefix_diagnose.py",
        "scripts/t00_baseline.py", "scripts/t05_residual_control.py", "goal_control.py", "goal_control_stats.py")}


def history_statistics(left, right, events_left, events_right):
    import goal_start_stability as starts
    return starts.extend_statistics(pixels.frame_statistics(left, right, events_left, events_right),
                                    left, right, events_left, events_right)


def export_gallery(left, right, rows, summary, directory):
    import numpy as np
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    selected = sorted({0, len(rows) // 2, len(rows) - 1,
                       max(rows, key=lambda row: row["rgb_mae"])["frame"],
                       max(rows, key=lambda row: row["rgb_p99"])["frame"],
                       summary["recent_window"]["first_frame"]})
    panel = Image.new("RGB", (640, 155 * len(selected)), "white")
    draw = ImageDraw.Draw(panel)
    for index, frame in enumerate(selected):
        row, y = rows[frame], index * 155
        position = "missing" if row["position"] is None else f"{row['position']:.5f}"
        draw.text((3, y + 2), f"frame {frame} | MAE {row['rgb_mae']:.3f} P99 {row['rgb_p99']:.1f} "
                  f"position {position} state L2 {row['state_relative_l2']:.3f}", fill="black")
        diff = np.abs(left["image"][frame].astype(np.int16) - right["image"][frame].astype(np.int16))
        images = (left["image"][frame], right["image"][frame], np.clip(diff * 4, 0, 255).astype(np.uint8),
                  left["heatmap"][frame], right["heatmap"][frame])
        for column, (image, label) in enumerate(zip(images, ("reference / run 0", "actual / run 1", "abs diff x4", "heatmap left", "heatmap right"))):
            draw.text((column * 128 + 2, y + 17), label, fill="black")
            panel.paste(Image.fromarray(image).convert("RGB").resize((128, 128)), (column * 128, y + 27))
    panel.save(directory / "prefix_pairs.png")
    baseline.write_csv(directory / "frame_statistics.csv", rows)
    baseline.write_json(directory / "diagnostics.json", dict(summary=summary, selected_frames=selected, per_frame=rows))


def historical_inputs(directory):
    """Read only saved artifacts, including a FAILED historical evaluation."""
    import goal_start_stability as starts
    record = read_json(directory / "report.json")
    manifest = read_json(directory / "evaluation_manifest.json")
    starts.require(record.get("command") == "evaluate" and
                   manifest.get("format") == "ls_imagine_residual_control_evaluation_v1" and
                   record.get("evaluation_id") == manifest.get("evaluation_id") == starts.content_id({
                       name: value for name, value in manifest.items() if name != "evaluation_id"}), "历史评估内容身份异常")
    rows = manifest["rows"]
    starts.require(rows and record.get("completed_trials") == manifest.get("completed_trials") == len(rows) and
                   len({(row["seed"], row["repeat"], row["target"], row["mode"]) for row in rows}) == len(rows),
                   "历史trial计数为空、不同或重复")
    benchmark_dir = baseline.project_path(record["arguments"]["benchmark_dir"])
    benchmark, adopted = read_json(benchmark_dir / "benchmark.json"), read_json(benchmark_dir / "report.json")
    starts.require(benchmark["benchmark_id"] == manifest["benchmark_id"] == adopted.get("benchmark_id") == starts.content_id({
                       name: value for name, value in benchmark.items() if name != "benchmark_id"}) and
                   adopted.get("status") in ("passed", "passed_with_warnings"), "原目标/参考benchmark身份异常")
    planned = {(cell["seed"], cell["repeat"], cell["target"], cell["mode"]) for cell in benchmark["design"]["schedule"]}
    starts.require(all((row["seed"], row["repeat"], row["target"], row["mode"]) in planned for row in rows), "历史trial超出原计划")
    files = [directory / name for name in ("report.json", "evaluation_manifest.json")]
    files += [benchmark_dir / name for name in ("benchmark.json", "report.json")]
    protected = [directory, benchmark_dir]
    for row in rows:
        artifact = baseline.project_path(row["artifact_dir"])
        starts.require(artifact.name == row["directory"] == f"seed_{row['seed']}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}",
                       "历史trial目录身份不同")
        protected.append(artifact)
        files += [artifact / name for name in ("trajectory.npz", "events.json", "metrics.json", "prefix_comparison.json", "control_trace.json", "video.mp4")]
    for case in benchmark["cases"]:
        files += [benchmark_dir / case["directory"] / "reference_0" / name for name in ("trajectory.npz", "events.json")]
    # Do not accidentally write inside any original model/replay directory.
    protected += [Path(path).resolve().parent for path in record.get("source_inputs_before", {})]
    return record, manifest, benchmark, benchmark_dir, protected, files


def audit(args, report):
    import numpy as np
    from PIL import Image, ImageDraw
    import goal_start_stability as starts

    directory = baseline.project_path(args.eval_dir)
    record, manifest, benchmark, benchmark_dir, _, _ = historical_inputs(directory)
    prefix = benchmark["prefix_steps"]
    starts.require(1 <= args.tail_frames <= prefix + 1 and benchmark["pair_limits"] == starts.LIMITS,
                   "原门槛或审计窗口异常")
    cases = {case["seed"]: case for case in benchmark["cases"]}
    references = {}
    for seed, case in cases.items():
        reference_dir = benchmark_dir / case["directory"] / "reference_0"
        for name in ("trajectory.npz", "events.json"):
            relative = str((reference_dir / name).relative_to(benchmark_dir))
            starts.require(relative in case["files"] and starts.file_hash(reference_dir / name) == case["files"][relative],
                           "已保存参考内容SHA256不同")
        references[seed] = pixels.read_reference(reference_dir, prefix + 1)
    summary_rows, detail_rows, physical, old_failures = [], [], [], Counter()
    overview = Image.new("RGB", (780, 160 * len(manifest["rows"])), "white")
    draw = ImageDraw.Draw(overview)
    for index, trial in enumerate(manifest["rows"]):
        artifact = baseline.project_path(trial["artifact_dir"])
        for name in ("trajectory.npz", "events.json", "metrics.json", "prefix_comparison.json", "control_trace.json", "video.mp4"):
            path = artifact / name
            starts.require(manifest["artifacts"].get(str(path)) == starts.file_hash(path), "原trial产物SHA256不同")
        starts.require(read_json(artifact / "metrics.json") == trial, "原trial与manifest记录不同")
        left, events_left = references[trial["seed"]]
        right, events_right = pixels.read_reference(artifact, prefix + 1)
        rows = history_statistics(left, right, events_left, events_right)
        saved = read_json(artifact / "prefix_comparison.json")
        starts.require(saved["limits"] == starts.LIMITS and len(saved.get("per_frame", [])) == len(rows), "原逐帧报告缺失或门槛不同")
        for row, previous in zip(rows, saved["per_frame"]):
            starts.require(row["frame"] == previous["frame"] and all(np.isclose(row[name], previous[name], atol=1e-6, rtol=1e-6)
                           for name in ("rgb_mae", "rgb_p99", "heatmap_mae", "state_relative_l2", "reward")), "重算与原逐帧误差不一致")
            for name in ("position", "angle"):
                starts.require((row[name] is None and previous["missing_telemetry"]) or
                               (row[name] is not None and np.isclose(row[name], previous[name], atol=1e-6, rtol=1e-6)), "物理位置/朝向重算不同")
            starts.require(all(row[name] == previous[name] for name in ("inventory_equal", "health_equal", "flags_equal", "actions_equal")),
                           "原物品/健康/边界/宏动作检查不同")
        summary = starts.summarize_history(rows, args.tail_frames)
        starts.require(summary["full_history"]["passed"] == trial["start_eligible"] and
                       all(row["native_actions_equal"] is True for row in rows) == trial["start_comparison"]["native_actions_equal"],
                       "全历史重算与原起点判定不一致")
        old_failures.update(trial["start_comparison"]["failed_checks"])
        output = report.directory / trial["directory"]
        export_gallery(left, right, rows, summary, output)
        current, recent, full = (summary[name] for name in ("control_start", "recent_window", "full_history"))
        flat = dict(seed=trial["seed"], repeat=trial["repeat"], target=trial["target"], mode=trial["mode"],
                    historical_start_eligible=trial["start_eligible"], historical_failed_checks="|".join(trial["start_comparison"]["failed_checks"]),
                    full_rgb_mae=full["rgb_mae"], full_rgb_p99=full["rgb_p99"], physical_history_passed=full["physical_passed"],
                    current_rgb_mae=current["rgb_mae"], current_rgb_p99=current["rgb_p99"], current_passed=current["passed"],
                    current_state_relative_l2=current["state_relative_l2"],
                    tail_frames=args.tail_frames, tail_rgb_mae=recent["rgb_mae"], tail_rgb_p99=recent["rgb_p99"],
                    candidate_preflight_passed=summary["candidate_preflight_passed"],
                    early_rgb_only_failure=not full["rgb_passed"] and recent["rgb_passed"] and full["physical_passed"])
        summary_rows.append(flat)
        detail_rows.append(dict(trial={name: trial[name] for name in ("seed", "repeat", "target", "mode", "directory")},
                               historical_start_comparison=trial["start_comparison"], diagnostics=summary))
        physical += [dict(directory=trial["directory"], **item) for item in summary["physical_failure_frames"]]
        y = index * 160
        draw.text((3, y + 3), f"target {trial['target']} {trial['mode']} | OLD eligible={trial['start_eligible']} "
                  f"current={current['passed']} recent={summary['candidate_preflight_passed']} | diagnostic only", fill="black")
        peak = max(rows, key=lambda row: row["rgb_p99"])["frame"]
        for column, (image, label) in enumerate(((left["image"][-1], "reference control start"), (right["image"][-1], "actual control start"),
                                                 (left["image"][peak], f"reference peak frame {peak}"), (right["image"][peak], "actual peak"))):
            draw.text((column * 195 + 2, y + 18), label, fill="black")
            overview.paste(Image.fromarray(image).resize((128, 128)), (column * 195, y + 31))
        print(f"[AUDIT] {trial['directory']} old={int(trial['start_eligible'])} "
              f"current={int(current['passed'])} recent={int(summary['candidate_preflight_passed'])} "
              f"current_mae={current['rgb_mae']:.3f} p99={current['rgb_p99']:.1f}", flush=True)
    diagnostics = dict(protocol=starts.PROTOCOL, source_evaluation=str(directory), source_evaluation_id=manifest["evaluation_id"],
        source_manifest_sha256=starts.file_hash(directory / "evaluation_manifest.json"), benchmark_id=benchmark["benchmark_id"],
        historical_status=record["status"], historical_eligible_trials=record.get("eligible_trials"),
        trials=len(summary_rows), historical_failure_counts=dict(old_failures), tail_frames=args.tail_frames,
        physical_history_pass_count=sum(row["physical_history_passed"] for row in summary_rows),
        current_frame_pass_count=sum(row["current_passed"] for row in summary_rows),
        recent_candidate_pass_count=sum(row["candidate_preflight_passed"] for row in summary_rows),
        early_rgb_only_failure_count=sum(row["early_rgb_only_failure"] for row in summary_rows),
        summaries=summary_rows, trial_diagnostics=detail_rows, model_loaded=False,
        new_env_steps=0, optimizer_updates=0, approved_benchmark=False, historical_acceptance_unchanged=True)
    baseline.write_csv(report.directory / "summary.csv", summary_rows)
    baseline.write_json(report.directory / "physical_failures.json", physical)
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    overview.save(report.directory / "audit_overview.png")
    report.data.update({name: diagnostics[name] for name in ("source_evaluation", "source_evaluation_id", "source_manifest_sha256",
        "benchmark_id", "historical_status", "historical_eligible_trials", "trials", "tail_frames", "historical_failure_counts")})
    report.check("historical_identity", "PASS", "原失败报告/manifest/目标参考及全部trial内容SHA256一致；未要求失败结果变PASS")
    report.check("recorded_statistics", "PASS", "全量重算RGB/heatmap/状态、物理/宏动作/原生动作，与原逐帧检查和起点判定一致")
    report.check("artifacts", "PASS", "保存全12条当前/近期/全历史对比、分区/颜色偏差、位置失败明细和图集；未改观测")
    report.check("scope", "WARN", "窗口统计是反事实诊断，不重新验收旧trial；无模型/MineCLIP/MineDojo，无新环境步")


def action_names():
    """Read the literal action keys without importing an environment."""
    import goal_start_stability as starts
    tree = ast.parse((ROOT / "envs/tasks/base/ls_imagine_wrapper.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "BASIC_ACTIONS" for target in node.targets)]
    starts.require(len(nodes) == 1 and isinstance(nodes[0].value, ast.Dict) and all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in nodes[0].value.keys), "不能静态识别真实动作名称")
    return [key.value for key in nodes[0].value.keys]


def load_runtime(args, report):
    import goal_library as gl
    import goal_start_stability as starts
    import t05_goal_control as legacy
    import t05_residual_control as residual_control

    # Reuse the dedicated strict residual verifier. It does not construct
    # an optimizer; the original args/report command remains check/probe.
    loading = copy.copy(args)
    loading.command = "check"
    cache, runtime = residual_control.load_inputs(loading, report)
    _, benchmark = residual_control.load_benchmark(args, report)
    directory = baseline.project_path(args.audit_dir)
    audited, diagnostics = read_json(directory / "report.json"), read_json(directory / "diagnostics.json")
    legacy.accepted(audited, "audit")
    required = {"historical_identity", "recorded_statistics", "no_training", "source_inputs_unchanged"}
    gl.require(audited.get("stability_protocol") == starts.PROTOCOL and audited.get("stability_code") == code_identity() and
               audited.get("benchmark_id") == benchmark["benchmark_id"] and
               diagnostics.get("benchmark_id") == benchmark["benchmark_id"] and
               diagnostics.get("source_evaluation_id") == audited.get("source_evaluation_id") and
               args.tail_frames == audited.get("tail_frames") == diagnostics.get("tail_frames") and
               required <= {row["name"] for row in audited["checks"] if row["level"] == "PASS"},
               "需要当前代码、同一旧benchmark的已通过离线audit；audit不改变旧失败结果")
    case = next((case for case in benchmark["cases"] if case["seed"] == args.case_seed), None)
    gl.require(case is not None, "case-seed不在已保存真实参考中")
    plan = starts.make_plan(case, benchmark, args.warmup_steps, args.tail_frames,
                           runtime.bundle["config"]["num_actions"], action_names())
    starts.validate_plan(plan)
    audit_identity = dict(directory=str(directory), report_sha256=starts.file_hash(directory / "report.json"),
                          diagnostics_sha256=starts.file_hash(directory / "diagnostics.json"),
                          source_evaluation_id=audited["source_evaluation_id"],
                          source_manifest_sha256=audited["source_manifest_sha256"])
    report.data.update(stability_code=code_identity(), plan_id=plan["plan_id"], plan=plan, audit_identity=audit_identity,
                       same_hidden_state=False, approved_benchmark=False, historical_acceptance_unchanged=True,
                       worker_control_actions=0, model_loaded=True)
    if args.command == "probe":
        checked = read_json(baseline.project_path(args.check_dir) / "report.json")
        legacy.accepted(checked, "check")
        gl.require(checked.get("stability_protocol") == starts.PROTOCOL and checked.get("plan") == plan and
                   checked.get("audit_identity") == audit_identity and checked.get("stability_code") == code_identity() and
                   checked.get("input_identity") == report.data["input_identity"] and
                   checked.get("environment_fingerprint") == report.data["environment_fingerprint"] and
                   {"warmup_contracts", "source_inputs_unchanged", "no_training"} <= {
                       row["name"] for row in checked["checks"] if row["level"] == "PASS"},
                   "当前预热/近期窗口/模型/环境与稳定性check不同；不允许边跑边调整直到通过")
    return runtime, plan


def check(args, report, runtime, plan):
    import numpy as np
    import goal_start_stability as starts
    import t02_goal_library as t02

    markers = [dict(frame=index, rgb_mae=0.0, rgb_p99=0.0, heatmap_mae=0.0, state_relative_l2=0.0,
                    position=0.0, angle=0.0, reward=0.0, missing_telemetry=False, position_delta=dict(x=0, y=0, z=0),
                    **{name: True for name in starts.FLAGS}) for index in range(plan["tail_frames"] + 1)]
    markers[0].update(rgb_mae=10.0, rgb_p99=40.0)
    summary = starts.summarize_history(markers, plan["tail_frames"])
    starts.require(not summary["full_history"]["rgb_passed"] and summary["candidate_preflight_passed"], "早期RGB仍错误地覆盖近期窗口")
    physical = copy.deepcopy(markers)
    physical[0]["position"] = 0.1
    starts.require(not starts.summarize_history(physical, plan["tail_frames"])["candidate_preflight_passed"], "物理历史超限被近期RGB掩盖")
    physical[0]["position"] = 0.0
    physical[0]["native_actions_equal"] = False
    starts.require(not starts.summarize_history(physical, plan["tail_frames"])["candidate_preflight_passed"], "原生动作不一致未拒绝")
    t02.rejection(report, "tail_guard", lambda: starts.summarize_history(markers, 0), "tail-frames")
    changed = copy.deepcopy(plan)
    changed["actions"][0] = 1
    t02.rejection(report, "plan_guard", lambda: starts.validate_plan(changed), "身份")
    # Only in-memory markers: check the new warmup/incoming/reset interface,
    # never save these arrays as trajectories or insert them into replay.
    size, count = plan["control_start_frame"] + 1, len(plan["action_names"])
    arrays = {name: np.zeros(size, dtype=np.bool_) for name in ("is_first", "is_last", "is_terminal")}
    arrays["is_first"][0] = True
    arrays.update(image=np.zeros((size, 64, 64, 3), np.uint8), heatmap=np.zeros((size, 64, 64), np.uint8),
                  features=np.zeros((size, 1), np.float32), obs_reward=np.zeros((size, 1), np.float32),
                  action=np.concatenate((np.zeros((1, count), np.float32), np.eye(count, dtype=np.float32)[plan["actions"]])))
    events = [dict(frame=frame, native_actions=[], error=None) for frame in range(size)]
    starts.validate_real_history(arrays, events, plan)
    bad = {name: value[plan["warmup_steps"]:] for name, value in arrays.items()}
    t02.rejection(report, "dropped_warmup_guard", lambda: starts.validate_real_history(bad, events[plan["warmup_steps"]:], plan), "完整真实")
    bad = copy.deepcopy(arrays)
    bad["is_first"][plan["warmup_steps"]] = True
    t02.rejection(report, "fake_reset_guard", lambda: starts.validate_real_history(bad, events, plan), "伪reset")
    bad = copy.deepcopy(arrays)
    bad["action"][1] = 0
    t02.rejection(report, "incoming_guard", lambda: starts.validate_real_history(bad, events, plan), "incoming")
    baseline.write_json(report.directory / "probe_plan.json", plan)
    report.check("warmup_contracts", "PASS", "静态识别0号noop；固定预热/前缀/两次重放和窗口；物理/原生动作全历史门槛保留")
    report.check("probe_scope", "PASS", "内存标记仅检查历史/窗口接口，不写成真实轨迹；无环境交互，不丢预热或伪reset")
    report.check("scope", "WARN", "这是离线接口验收；尚未确认近期RGB稳定、状态相同或目标控制有效")


def probe(args, report, runtime, plan):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_start_stability as starts
    import t02_goal_library as t02

    baseline.video_preflight(report)
    bc.require_disk_space(report.directory, 2 * 16 * 1024**2 + 8 * 1024**2)
    # The complete plan is persisted BEFORE either new environment starts.
    baseline.write_json(report.directory / "probe_plan.json", dict(plan=plan, input_identity=report.data["input_identity"],
        environment_fingerprint=report.data["environment_fingerprint"], stability_code=code_identity(), audit_identity=report.data["audit_identity"]))
    report.check("predeclared_two_runs", "PASS", f"固定{plan['warmup_steps']}步noop+{plan['prefix_steps']}步原前缀，"
                 f"两次fresh reset；总预算{plan['maximum_new_env_steps']}；不执行worker控制动作")
    recorded_visual = read_json(baseline.project_path(args.benchmark_dir) / f"case_{plan['seed']}" /
                                "visual_preprocessing.json")["reference_0"]
    histories, event_histories, visual = [], [], []
    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
    for run in range(2):
        output = report.directory / f"run_{run}"
        session = None
        print(f"[ENV] stability run={run} seed={plan['seed']} world_seed={plan['scenario']['world_seed']} "
              f"warmup={plan['warmup_steps']} prefix={plan['prefix_steps']}; fresh reset", flush=True)
        try:
            session = ctl.Session(runtime, plan["scenario"], output / "environment", on_step)
            starts.require(session.specs["visual_preprocessing"] == recorded_visual, "新重放改变了原任务RGB/HUD预处理")
            visual.append(session.specs["visual_preprocessing"])
            for action in plan["actions"]:
                session.step(action)
                starts.require(not session.done, "预热/共同前缀提前结束；保存真实数据但不补跑")
        finally:
            if session is not None:
                try:
                    ctl.save_session(session, output, baseline.write_video)
                finally:
                    session.close()
            gc.collect()
        arrays, events = pixels.read_reference(output, plan["control_start_frame"] + 1)
        starts.validate_real_history(arrays, events, plan)
        # Verify against each run's OWN full real history, not the old
        # reference state or a reset invented after the warmup.
        expected = runtime.state_encoder.rollout(arrays, plan["control_start_frame"])
        t02.vector_difference(report, f"full_causal_history_run_{run}", expected,
                              np.asarray(arrays["features"]), check=True)
        histories.append(arrays)
        event_histories.append(events)
        report.data["completed_runs"] = run + 1
        report.save()
        print(f"[REPLAY {run + 1}/2] saved={output.name} frames={len(arrays['image'])} worker_control_actions=0", flush=True)
    starts.require(visual[0] == visual[1], "两次视觉协议不同")
    baseline.write_json(report.directory / "visual_preprocessing.json", visual)
    rows = history_statistics(histories[0], histories[1], event_histories[0], event_histories[1])
    summary = starts.summarize_history(rows, plan["tail_frames"])
    # Preserve the previous strict definition unchanged as a diagnostic.
    strict = ctl.prefix_comparison(dict(histories[0], telemetry=[event["telemetry"] for event in event_histories[0]]),
                                   dict(histories[1], telemetry=[event["telemetry"] for event in event_histories[1]]))
    baseline.write_json(report.directory / "strict_history_comparison.json", strict)
    export_gallery(histories[0], histories[1], rows, summary, report.directory)
    diagnostics = read_json(report.directory / "diagnostics.json")
    diagnostics.update(protocol=starts.PROTOCOL, plan_id=plan["plan_id"], plan=plan, strict_history_comparison=strict,
                       completed_runs=2, new_env_steps=report.data["new_env_steps"], optimizer_updates=0, worker_control_actions=0,
                       model_id=runtime.policy.model_id, worker_version=runtime.policy.worker_version,
                       original_benchmark_reusable=False, approved_benchmark=False, controlled_reachability_verified=False)
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    report.data.update(candidate_preflight_passed=summary["candidate_preflight_passed"],
                       strict_history_passed=strict["passed"], completed_runs=2,
                       original_benchmark_reusable=False, worker_control_actions=0)
    starts.require(report.data["new_env_steps"] == plan["maximum_new_env_steps"], "两次真实环境步计数不同")
    full, recent, current = (summary[name] for name in ("full_history", "recent_window", "control_start"))
    report.check("two_real_histories", "PASS", "两次完整真实reset/预热/前缀逐帧保留；重算本局RSSM一致，没有复制旧状态或伪reset")
    report.check("physical_history", "PASS" if full["physical_passed"] else "FAIL",
                 f"全历史物理/原生动作门槛；失败项={full['physical_failed_checks']}")
    report.check("recent_rgb", "PASS" if recent["rgb_passed"] else "FAIL",
                 f"最后固定{plan['tail_frames']}帧max MAE={recent['rgb_mae']:.4f}/1 P99={recent['rgb_p99']:.1f}/8；"
                 f"控制起点MAE={current['rgb_mae']:.4f} P99={current['rgb_p99']:.1f}")
    report.check("strict_history_retained", "PASS", "原全历史RGB/heatmap/RSSM严格比较原样保存；近期通过不等于隐藏状态相同")
    report.check("scope", "WARN", "两次重放仅用于稳定性预检；旧失败保持，旧参考/目标不能直接沿用，不自动重建36次或进入T06")


def input_paths(args):
    import goal_start_stability as starts
    if args.command == "audit":
        _, _, _, _, protected, files = historical_inputs(baseline.project_path(args.eval_dir))
        source = None
    else:
        import t05_residual_control as residual_control
        protected, files, source = residual_control.source_paths(args)
        audited = baseline.project_path(args.audit_dir)
        protected.append(audited)
        files += [audited / name for name in ("report.json", "diagnostics.json", "summary.csv", "physical_failures.json", "audit_overview.png")]
        if args.command == "probe":
            files.append(baseline.project_path(args.check_dir) / "probe_plan.json")
    starts.require(all(path.is_file() for path in files), "输入/缓存/底座/参考/验收文件缺失，请保留原服务器产物")
    return list(dict.fromkeys(path.resolve() for path in protected)), list(dict.fromkeys(path.resolve() for path in files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    auditing = commands.add_parser("audit")
    auditing.add_argument("--eval-dir", required=True)
    for name in ("check", "probe"):
        command = commands.add_parser(name)
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--residual-verify-dir", required=True)
        command.add_argument("--benchmark-dir", required=True, help="原residual_control_adopt目录；仅提供场景/共同前缀")
        command.add_argument("--audit-dir", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--case-seed", type=int, default=0)
        command.add_argument("--warmup-steps", type=int, default=32)
        if name == "probe":
            command.add_argument("--check-dir", required=True)
    for command in commands.choices.values():
        command.add_argument("--tail-frames", type=int, default=4)
        command.add_argument("--output-dir")
    args = parser.parse_args()
    if not 2 <= args.tail_frames <= 8:
        parser.error("tail-frames必须在[2,8]；audit/check/probe保持同一预声明窗口")
    if args.command != "audit" and (not 1 <= args.warmup_steps <= 256 or args.case_seed < 0):
        parser.error("warmup-steps必须在[1,256]，case-seed必须非负")
    if args.command == "probe" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = input_paths(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "start_stability_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、缓存、回放、参考、失败评估和前段验收")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    before, rng = None, None
    try:
        import goal_start_stability as starts
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data.update(stability_protocol=starts.PROTOCOL, stability_code=code_identity(),
                           source_inputs_before=before, optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0,
                           model_loaded=False, approved_benchmark=False, same_hidden_state=False,
                           historical_acceptance_unchanged=True, worker_control_actions=0,
                           controlled_reachability_verified=False)
        report.save()
        report.require_writable()
        if args.command == "audit":
            audit(args, report)
            report.check("no_training", "PASS", "纯CPU读取现有文件；未导入模型/torch/MineCLIP/MineDojo；优化器/新增环境步为0")
        else:
            import torch
            import goal_bc as bc
            import goal_library as gl
            device = torch.device(args.device)
            if device.type == "cuda":
                starts.require(torch.cuda.is_available(), "CUDA不可用")
                torch.cuda.set_device(device)
            else:
                starts.require(args.command != "probe", "真实MineCLIP路径需要CUDA")
            starts.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化checkpoint变化")
            starts.require(all(baseline.file_signature(Path(episode["path"])) == episode["signature"]
                               for episode in source["episodes"]), "原真实回放变化")
            rng = bc.capture_rng(args.device)
            runtime, plan = load_runtime(args, report)
            versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
            digest = gl.tensor_digest(runtime.state_dict(), {})
            with torch.no_grad():
                {"check": check, "probe": probe}[args.command](args, report, runtime, plan)
            starts.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                           gl.tensor_digest(runtime.state_dict(), {}) == digest and
                           all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()),
                           "冻结参数/梯度所有权变化")
            report.check("no_training", "PASS", f"无优化器/参数更新；worker控制动作为0；新env.step={report.data['new_env_steps']}（包含预热）")
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
                             "源产物大小/修改时间未变；读取的历史trial另按SHA256验真，只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
