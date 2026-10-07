"""T05 independent-start pilot: offline check/design, then 12 real trials.

Requires MINEDOJO_HEADLESS=1 for evaluate. No training, retries or matched
reference-state gate. Old failed runs retain their original interpretation.
"""

import argparse
import ast
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
import t05_goal_control as legacy
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
        print(f"[{'FAIL' if failed else 'PASS'}] T05_RANDOM_CONTROL_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return dict(original.code_identity(), **{name: bc.file_hash(ROOT / name) for name in (
        "goal_random_control.py", "scripts/t05_random_control.py", "goal_control.py",
        "scripts/t05_goal_control.py")})


def action_names():
    import goal_library as gl
    tree = ast.parse((ROOT / "envs/tasks/base/ls_imagine_wrapper.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "BASIC_ACTIONS" for target in node.targets)]
    gl.require(len(nodes) == 1 and isinstance(nodes[0].value, ast.Dict) and all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in nodes[0].value.keys),
        "不能静态识别真实动作名称")
    return [key.value for key in nodes[0].value.keys]


def load_inputs(args, report):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control

    # Reuse immutable accepted model/reference readers. The old source is
    # validated under its own format before declaring the new experiment.
    reading = copy.copy(args)
    reading.command = "check"
    cache, runtime = original.load_inputs(reading, report)
    reading.benchmark_dir = args.source_benchmark_dir
    source, old = original.load_benchmark(reading, report)
    case = next((case for case in old["cases"] if case["seed"] == args.case_seed), None)
    gl.require(case is not None, "case-seed不在完整真实目标来源中")
    names = action_names()
    gl.require(len(names) == runtime.bundle["config"]["num_actions"] and names[0] == "noop",
               "动作接口或0号noop不同")
    prefix = list(case["prefix_actions"])
    gl.require(all(type(action) is int and 0 <= action < len(names) for action in prefix), "固定前缀动作异常")
    first = args.warmup_steps + len(prefix)
    design = dict(seed=args.case_seed, repetitions=args.repetitions, randomization_seed=args.randomization_seed,
        execution_policy=args.execution_policy, modes=list(random_control.MODES),
        schedule=random_control.schedule(args.case_seed, args.repetitions, args.randomization_seed))
    random_control.validate_design(design)
    with np.load(source / case["directory"] / "goals.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    gl.require(bank["goals"].shape == (2, cache.metadata["goal_dim"]) and
               bank["images"].shape == (2, 64, 64, 3) and bank["heatmaps"].shape == (2, 64, 64) and
               bank["images"].dtype == bank["heatmaps"].dtype == np.uint8 and
               np.isfinite(bank["goals"]).all() and
               np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5), "真实目标库形状、类型或数值不同")
    references = [ctl.read_trajectory(source / case["directory"] / f"reference_{branch}") for branch in (0, 1)]
    for target, reference in enumerate(references):
        gl.require(len(reference["image"]) == old["prefix_steps"] + old["horizon"] + 1 and
                   np.array_equal(bank["images"][target], reference["image"][-1]) and
                   np.array_equal(bank["heatmaps"][target], reference["heatmap"][-1]),
                   "需要两个完整真实终点；不能采用半途失败的新prepare")
        actual = ctl.encode_goal(runtime, dict(image=bank["images"][target], heatmap=bank["heatmaps"][target]))
        gl.require(np.allclose(actual, bank["goals"][target], atol=3e-5, rtol=3e-5), "固定视觉目标重编码不同")
    visual = legacy.read_json(source / case["directory"] / "visual_preprocessing.json")["reference_0"]
    plan = dict(format=random_control.FORMAT, comparison_protocol=random_control.PROTOCOL,
        source_benchmark_id=old["benchmark_id"], source_benchmark=str(source),
        source_manifest_sha256=bc.file_hash(source / "benchmark.json"),
        source_report_sha256=bc.file_hash(source / "report.json"),
        input_identity=report.data["input_identity"], evaluation_code=code_identity(),
        environment_fingerprint=report.data["environment_fingerprint"],
        scenario=case["scenario"], visual_preprocessing=visual, action_names=names,
        warmup_steps=args.warmup_steps, prefix_actions=prefix,
        start_actions=[0] * args.warmup_steps + prefix, control_start_frame=first, horizon=old["horizon"],
        design=design, goal_source="fixed real visual endpoints, not current-start branches",
        target_content_id=gl.tensor_digest({}, {name: bank[name].tolist() for name in bank}),
        same_hidden_state=False, matched_start_comparison=False, old_failures_unchanged=True)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(comparison_protocol=random_control.PROTOCOL, evaluation_code=code_identity(),
        design_id=plan["design_id"], planned_trials=len(design["schedule"]),
        maximum_new_env_steps=len(design["schedule"]) * (first + old["horizon"]),
        matched_start_comparison=False, old_failures_unchanged=True,
        benchmark_id=None, source_benchmark_id=old["benchmark_id"], model_loaded=True)
    if args.command == "evaluate":
        checked_dir = baseline.project_path(args.check_dir)
        checked = legacy.read_json(checked_dir / "report.json")
        legacy.accepted(checked, "check")
        saved_plan = legacy.read_json(checked_dir / "design.json")
        required = {"causal_execution_interface", "independent_start_contract", "fixed_visual_targets",
                    "predeclared_design", "no_training", "source_inputs_unchanged"}
        gl.require(saved_plan == plan and checked.get("comparison_protocol") == random_control.PROTOCOL and
                   checked.get("design_id") == plan["design_id"] and checked.get("input_identity") == plan["input_identity"] and
                   checked.get("evaluation_code") == code_identity() and
                   required <= {row["name"] for row in checked["checks"] if row["level"] == "PASS"},
                   "需要当前新协议check；旧失败、旧预热入口或被修改计划不能用于执行")
        with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
            gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "已声明视觉目标改变")
        gl.require(checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz"), "已声明目标文件SHA256不同")
        report.check("design_identity", "PASS", "固定目标、随机顺序、预算、模型/缓存/环境与离线check逐项一致")
    report.check("fixed_visual_targets", "PASS", "只读两个已有完整真实终点并重编码；不采集新参考，不要求本轮起点等于参考")
    return cache, runtime, plan, bank


def check(args, report, cache, runtime, plan, bank):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    import goal_residual_control as control

    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "真实历史数据集身份不同")
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
        t02.rejection(report, "after_terminal_guard", lambda: runtime.step(dict(reset, is_last=True), ep["action"][0], bank["goals"][0], 1), "结束")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "因果状态检查推进了随机数")
    features, _, _, _ = cache.worker_batch(cache.worker_rows["validation"][:1], args.device)
    pair, horizon = bank["goals"], plan["horizon"]
    for target in (0, 1):
        swapped = control.probabilities(runtime, features, pair, horizon, target, "swapped_goal")
        other = control.probabilities(runtime, features, pair, horizon, 1 - target, "goal")
        gl.require(np.array_equal(swapped, other), "交换目标没有使用另一个固定目标")
    gl.require(np.array_equal(control.probabilities(runtime, features, pair, horizon, 0, "no_goal"),
                             control.probabilities(runtime, features, pair, horizon, 1, "no_goal")), "独立无目标分支依赖目标")
    for invalid in (0, horizon + 1, 1.5):
        t02.rejection(report, "remaining_guard", lambda invalid=invalid: control.probabilities(runtime, features, pair, invalid, 0, "goal"), "remaining")
    report.check("causal_execution_interface", "PASS", "两局真实历史增量/完整前缀一致；reset/结束/incoming/remaining守卫和三组部署分布正确")
    dimension = runtime.bundle["config"]["num_actions"]
    size = len(plan["start_actions"]) + 1
    markers = dict(image=np.zeros((size, 64, 64, 3), np.uint8), heatmap=np.zeros((size, 64, 64), np.uint8),
        features=np.zeros((size, runtime.bundle["feature_dim"]), np.float32),
        action=np.concatenate((np.zeros((1, dimension), np.float32), np.eye(dimension, dtype=np.float32)[plan["start_actions"]])),
        obs_reward=np.zeros((size, 1), np.float32), is_first=np.arange(size) == 0,
        is_last=np.zeros(size, bool), is_terminal=np.zeros(size, bool))
    telemetry = dict(pose=dict(x=0., y=64., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    events = [dict(frame=frame, error=None, done=False, telemetry=copy.deepcopy(telemetry),
                   native_actions=[] if frame == 0 else [np.zeros(8, np.int64)]) for frame in range(size)]
    random_control.validate_prefix(markers, events, plan["start_actions"], dimension)
    drift = copy.deepcopy(events)
    drift[-1]["telemetry"]["pose"]["x"] = 10.
    different = copy.deepcopy(markers)
    different["image"] += 100
    different["features"] += 10
    random_control.validate_prefix(different, drift, plan["start_actions"], dimension)
    missing = copy.deepcopy(events)
    del missing[-1]["native_actions"]
    t02.rejection(report, "missing_native_guard", lambda: random_control.validate_prefix(markers, missing, plan["start_actions"], dimension), "原生事件")
    wrong = copy.deepcopy(markers)
    wrong["action"][1] = np.roll(wrong["action"][1], 1)
    t02.rejection(report, "incoming_guard", lambda: random_control.validate_prefix(wrong, events, plan["start_actions"], dimension), "incoming")
    wrong = copy.deepcopy(markers)
    wrong["is_first"][-1] = True
    t02.rejection(report, "fake_reset_guard", lambda: random_control.validate_prefix(wrong, events, plan["start_actions"], dimension), "伪reset")
    fake_rows = [dict(cell, execution_valid=True, task_success=False, start_telemetry=telemetry,
                      prefix_movement=dict(position=10.), **{name: .1 for name in random_control.METRICS})
                 for cell in plan["design"]["schedule"]]
    fake_rows[0].update(execution_valid=False, error="in-memory interface marker")
    summary = random_control.summarize(fake_rows, plan["design"])
    gl.require(summary["attempted_trials"] == len(fake_rows) and len(summary["execution_errors"]) == 1 and
               sum(value["measured"] for value in summary["modes"].values()) == len(fake_rows) - 1,
               "错误被隐藏或导致其它独立起点整块删除")
    t02.rejection(report, "duplicate_trial_guard", lambda: random_control.summarize(fake_rows + fake_rows[:1], plan["design"]), "重复")
    changed = copy.deepcopy(plan["design"])
    changed["schedule"][0]["target"] = 1 - changed["schedule"][0]["target"]
    t02.rejection(report, "schedule_guard", lambda: random_control.validate_design(changed), "随机计划")
    report.check("independent_start_contract", "PASS", "不同位置/RGB/RSSM无需与另一局相等；真实动作/遥测/历史异常仍拒绝，计划错误和缺测单列；内存marker不写入产物")
    np.savez_compressed(report.directory / "targets.npz", **bank)
    baseline.write_json(report.directory / "design.json", plan)
    target_gallery(report.directory / "targets.png", bank["images"])
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.check("predeclared_design", "PASS", f"三组×两目标×{args.repetitions}重复={len(fake_rows)}trial；"
                 f"每局{plan['warmup_steps']}noop+{len(plan['prefix_actions'])}真实前缀+最多{horizon}控制；只需下一步evaluate")
    report.check("scope", "WARN", "设计已固定；未启动环境。独立起点统计是新协议，旧FAIL保持，工程PASS不证明控制有效")


def target_gallery(path, images):
    from PIL import Image, ImageDraw
    panel = Image.new("RGB", (420, 155), "white")
    draw = ImageDraw.Draw(panel)
    for target in (0, 1):
        draw.text((210 * target + 3, 3), f"fixed visual target {target}", fill="black")
        panel.paste(Image.fromarray(images[target]).resize((128, 128)), (210 * target + 3, 22))
    panel.save(path)


def run_trial(report, runtime, plan, bank, cell, directory):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    import goal_residual_control as control
    import goal_segments as segments

    first, horizon, goals = plan["control_start_frame"], plan["horizon"], bank["goals"]
    target, mode = cell["target"], cell["mode"]
    assigned = 1 - target if mode == "swapped_goal" else target
    # Action draws are private to each scheduled attempt, never consumed by
    # simulator setup or used to retry a favorable start.
    index = next(index for index, scheduled in enumerate(plan["design"]["schedule"]) if scheduled == cell)
    action_seed = (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)
    uniforms = np.random.RandomState(action_seed).uniform(size=horizon)
    session, actions, probabilities, row = None, [], [], dict(cell, artifact_dir=str(directory), execution_valid=False,
        same_hidden_state=False, matched_start_comparison=False, actual_steps=0,
        issued_goal=assigned if mode != "no_goal" else None, task_success=False)
    trace = dict(model_id=runtime.policy.model_id, condition=mode, evaluated_target=target,
                 issued_target=row["issued_goal"], action_seed=action_seed, execution_policy=plan["design"]["execution_policy"])
    before_steps = report.data["new_env_steps"]
    def on_step():
        report.data["new_env_steps"] += 1
        report.data["evaluation_env_steps"] += 1
    def execute(action):
        report.data["attempted_env_steps"] += 1
        session.step(action)
    try:
        print(f"[ENV] independent repeat={cell['repeat']} target={target} mode={mode} "
              f"world_seed={plan['scenario']['world_seed']}; fresh reset", flush=True)
        session = ctl.Session(runtime, plan["scenario"], directory / "environment", on_step)
        gl.require(session.specs["visual_preprocessing"] == plan["visual_preprocessing"], "当前环境视觉处理不同")
        for action in plan["start_actions"]:
            execute(action)
            gl.require(not session.done, "真实环境在共同前缀完成前结束；该尝试原样记录，不补跑")
        random_control.validate_prefix(session.arrays(), session.events, plan["start_actions"], runtime.bundle["config"]["num_actions"])
        initial = ctl.encode_goal(runtime, session.observation)
        initial_distances = ctl.distances(initial[None], goals)[0]
        initial_margin = float(initial_distances[1 - target] - initial_distances[target])
        row.update(start_telemetry=session.events[-1]["telemetry"],
                   start_distance=float(initial_distances[target]),
                   start_target_preference_margin=initial_margin,
                   prefix_movement=ctl.pose_difference(session.events[-1]["telemetry"]["pose"], session.events[0]["telemetry"]["pose"]))
        baseline.write_json(directory / "start.json", dict(control_start_frame=first,
            telemetry=session.events[-1]["telemetry"], initial_goal_feature=initial.tolist(),
            distances_to_both_goals=initial_distances.tolist(), rssm_feature=session.features[0].cpu().tolist(),
            comparison_to_reference_required=False, own_real_history_only=True))
        distances = [initial_distances.tolist()]
        for step in range(horizon):
            p = control.probabilities(runtime, session.features, goals, horizon - step, target, mode)
            action = ctl.select_action(p, plan["design"]["execution_policy"], float(uniforms[step]))
            probabilities.append(p.tolist())
            actions.append(action)
            execute(action)
            distances.append(ctl.distances(ctl.encode_goal(runtime, session.observation)[None], goals)[0].tolist())
            if session.done:
                break
        arrays = session.arrays()
        segments.validate_episode(arrays, runtime.bundle["config"]["num_actions"])
        expected = np.eye(runtime.bundle["config"]["num_actions"], dtype=np.float32)[actions]
        gl.require(np.array_equal(arrays["action"][first + 1:], expected) and
                   len(session.events) == first + len(actions) + 1 and all(
                       event.get("frame") == frame and not event.get("error") and
                       isinstance(event.get("native_actions"), (list, tuple)) and
                       (frame == 0 or len(event["native_actions"]) > 0) for frame, event in enumerate(session.events)),
                   "真实控制incoming或原生事件异常")
        recovered = runtime.state_encoder.rollout(arrays, len(session.rows) - 1).cpu().numpy()
        maximum_error = float(np.max(np.abs(recovered - arrays["features"])))
        gl.require(np.allclose(recovered, arrays["features"], atol=3e-5, rtol=3e-5), "本局完整真实因果历史重算不一致")
        baseline.write_json(directory / "history_check.json", dict(own_causal_history_passed=True,
            maximum_absolute_error=maximum_error, total_frames=len(session.rows), start_frame=first,
            comparison_to_other_histories=False, reset_is_real=True))
        end = np.asarray(distances[-1])
        margin = float(end[1 - target] - end[target])
        start_telemetry, end_telemetry = session.events[first]["telemetry"], session.events[-1]["telemetry"]
        row.update(execution_valid=True, actual_steps=len(actions), start_telemetry=start_telemetry,
            end_telemetry=end_telemetry, prefix_movement=ctl.pose_difference(start_telemetry["pose"], session.events[0]["telemetry"]["pose"]),
            movement_from_start=ctl.pose_difference(end_telemetry["pose"], start_telemetry["pose"]),
            start_distance=float(initial_distances[target]), end_distance=float(end[target]),
            distance_improvement=float(initial_distances[target] - end[target]), target_preference_margin=margin,
            start_target_preference_margin=initial_margin, preference_improvement=margin - initial_margin,
            end_distances_to_both_goals=end.tolist(),
            issued_target_margin=float(end[1 - assigned] - end[assigned]) if mode != "no_goal" else None,
            task_success=any(event["success"] for event in session.events[first + 1:]),
            task_success_during_prefix=any(event["success"] for event in session.events[:first + 1]),
            external_return=float(sum(event["reward"] for event in session.events[first + 1:])),
            end_reason="environment_done" if session.done else "fixed_budget", own_history_max_error=maximum_error,
            action_histogram=np.bincount(actions, minlength=runtime.bundle["config"]["num_actions"]).tolist())
        trace["distances_to_both_goals"] = distances
    except Exception as error:
        if isinstance(error, OSError):
            raise
        row.update(error=f"{type(error).__name__}: {error}", actual_steps=max(0, len(session.rows) - first - 1) if session else 0)
        directory.mkdir(parents=True, exist_ok=True)
        baseline.write_json(directory / "error.json", dict(error=row["error"], traceback=traceback.format_exc()))
        report.check(f"trial_{index}_execution", "FAIL", row["error"] + "；保留该尝试，继续预声明顺序，不重试")
    finally:
        row["returned_env_steps"] = report.data["new_env_steps"] - before_steps
        trace.update(action_ids=actions, probabilities=probabilities, remaining=list(range(horizon, horizon - len(actions), -1)),
                     uniforms=uniforms[:len(actions)].tolist())
        if session is not None:
            row.update(task_success=any(event["success"] for event in session.events[first + 1:]),
                       task_success_during_prefix=any(event["success"] for event in session.events[:first + 1]),
                       external_return=float(sum(event["reward"] for event in session.events[first + 1:])))
            try:
                status = ctl.save_session(session, directory, baseline.write_video)
                row["video"] = status
            finally:
                session.close()
                del session
                gc.collect()
        directory.mkdir(parents=True, exist_ok=True)
        baseline.write_json(directory / "metrics.json", row)
        baseline.write_json(directory / "control_trace.json", trace)
    return row


def outcome_gallery(report, plan, bank, rows):
    from PIL import Image, ImageDraw
    import goal_control as ctl
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
                directory = Path(row["artifact_dir"])
                if (directory / "trajectory.npz").is_file():
                    panel.paste(Image.fromarray(ctl.read_trajectory(directory)["image"][-1]).resize((128, 128)), (x, y + 20))
        panel.save(report.directory / f"outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    from PIL import Image

    baseline.video_preflight(report)
    cells = plan["design"]["schedule"]
    bc.require_disk_space(report.directory, len(cells) * 20 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    report.data["attempted_env_steps"] = 0
    report.check("storage_preflight", "PASS", f"预留{len(cells)}次真实轨迹/视频空间；不复制冻结模型或缓存")
    report.check("independent_starts", "PASS", "每次fresh reset保留自身完整真实预热/前缀，然后直接执行预声明条件；跨局位置/RGB/RSSM差异只做诊断")
    rows = []
    for index, cell in enumerate(cells):
        report.require_writable()
        directory = report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['mode']}"
        row = run_trial(report, runtime, plan, bank, cell, directory)
        rows.append(row)
        t04.append_json(report.directory / "trials.jsonl", row)
        baseline.write_json(report.directory / "diagnostics.json", random_control.summarize(rows, plan["design"]))
        baseline.write_json(report.directory / "trials.json", rows)
        report.data["attempted_trials"] = len(rows)
        report.save()
        detail = f"steps={row['actual_steps']} d={row['end_distance']:.5f} progress={row['distance_improvement']:.5f}" if row["execution_valid"] else row.get("error", "execution error")
        print(f"[TRIAL {index + 1}/{len(cells)}] repeat={cell['repeat']} target={cell['target']} mode={cell['mode']} {detail}", flush=True)
    summary = random_control.summarize(rows, plan["design"])
    # Start images/state distances are diagnostic measurements; there is no
    # cutoff and no sample rejection based on proximity to the first start.
    observed = [row for row in rows if (Path(row["artifact_dir"]) / "start.json").exists()]
    balance = []
    if observed:
        reference = ctl.read_trajectory(Path(observed[0]["artifact_dir"]))
        first = plan["control_start_frame"]
        grid = Image.new("RGB", (256 * len(observed), 128), "white")
        for index, row in enumerate(observed):
            trajectory = ctl.read_trajectory(Path(row["artifact_dir"]))
            image = trajectory["image"][first]
            rgb = np.abs(image.astype(np.float32) - reference["image"][first].astype(np.float32))
            heat = np.abs(trajectory["heatmap"][first].astype(np.float32) - reference["heatmap"][first].astype(np.float32))
            balance.append(dict(trial=row["artifact_dir"], target=row["target"], mode=row["mode"], repeat=row["repeat"],
                rgb_mae_to_first_start=float(rgb.mean()), rgb_p99_to_first_start=float(np.percentile(rgb, 99)),
                heatmap_mae_to_first_start=float(heat.mean()),
                pose_difference_to_first_start=ctl.pose_difference(trajectory["telemetry"][first]["pose"], reference["telemetry"][first]["pose"]),
                rssm_relative_l2_to_first_start=float(np.linalg.norm(trajectory["features"][first] - reference["features"][first]) /
                    max(float(np.linalg.norm(reference["features"][first])), 1e-8)), diagnostic_only=True))
            grid.paste(Image.fromarray(image).resize((128, 128)), (index * 256, 0))
            grid.paste(Image.fromarray(trajectory["image"][-1]).resize((128, 128)), (index * 256 + 128, 0))
        grid.save(report.directory / "starts_and_outcomes.png")
    summary["start_differences_diagnostic_only"] = balance
    baseline.write_json(report.directory / "diagnostics.json", summary)
    baseline.write_csv(report.directory / "trials.csv", [{name: row.get(name) for name in (
        "seed", "repeat", "target", "mode", "execution_valid", "actual_steps", "returned_env_steps", "task_success",
        "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "preference_improvement", "error", "artifact_dir")} for row in rows])
    outcome_gallery(report, plan, bank, rows)
    saved = dict(format=random_control.EVALUATION_FORMAT, comparison_protocol=random_control.PROTOCOL,
        design=plan, design_id=plan["design_id"], rows=rows, new_env_steps=report.data["new_env_steps"],
        attempted_env_steps=report.data["attempted_env_steps"], input_identity=report.data["input_identity"],
        artifacts={str(path.relative_to(report.directory)): bc.file_hash(path) for row in rows
                   for path in Path(row["artifact_dir"]).iterdir() if path.is_file()})
    saved["evaluation_id"] = gl.tensor_digest({}, saved)
    baseline.write_json(report.directory / "evaluation_manifest.json", saved)
    valid = sum(row["execution_valid"] for row in rows)
    report.data.update(evaluation_id=saved["evaluation_id"], execution_valid_trials=valid,
        task_success_evaluated=True, full_design_completed=summary["full_design_completed"],
        controlled_reachability_verified=False)
    report.check("planned_attempts", "PASS", f"按固定随机顺序尝试{len(rows)}/{len(cells)}，全部保留；无补样、重试或旧失败续跑")
    report.check("real_control_interface", "PASS" if valid == len(rows) else "FAIL", f"真实因果历史/动作接口有效{valid}/{len(rows)}；跨局起点差异不改变此判定；错误数量另列")
    report.check("behavior_acceptance", "WARN", "只完成一个世界的描述性试跑；须比较进展、目标切换、初始平衡与视频。工程PASS不批准T06或任务成功提升")


def source_paths(args):
    reading = copy.copy(args)
    reading.source_prepare_dir = args.source_benchmark_dir
    protected, files, source = original.source_paths(reading)
    if getattr(args, "check_dir", None):
        directory = baseline.project_path(args.check_dir)
        files.extend(directory / name for name in ("design.json", "targets.npz"))
    return protected, list(dict.fromkeys(files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--residual-verify-dir", required=True)
        command.add_argument("--source-benchmark-dir", required=True, help="完整旧adopt，仅复用固定视觉目标及初始化脚本")
        command.add_argument("--case-seed", type=int, default=0)
        command.add_argument("--warmup-steps", type=int, default=32)
        command.add_argument("--repetitions", type=int, default=2)
        command.add_argument("--randomization-seed", type=int, default=0)
        command.add_argument("--execution-policy", choices=("mode", "sample"), default="mode")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
    args = parser.parse_args()
    if not (0 <= args.warmup_steps <= 128 and 2 <= args.repetitions <= 10 and
            0 <= args.case_seed < 2**31 - 10000 and 0 <= args.randomization_seed < 2**31 - 10000):
        parser.error("预热/重复次数或随机种子无效")
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = source_paths(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "random_control_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、缓存、回放、参考和离线check")
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
            gl.require(args.command != "evaluate", "真实MineCLIP路径需要CUDA")
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化模型变化")
        for episode in source["episodes"]:
            gl.require(baseline.file_signature(Path(episode["path"])) == episode["signature"], "原真实回放变化")
        rng = bc.capture_rng(args.device)
        cache, runtime, plan, bank = load_inputs(args, report)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            {"check": check, "evaluate": evaluate}[args.command](args, report, cache, runtime, plan, bank)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()),
                   "评估改变冻结模型或产生梯度")
        report.check("no_training", "PASS", f"optimizer_updates=0；全部模型/目标库参数未变；本轮env.step={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "源模型/回放/目标来源与check大小/修改时间未变；旧失败不改写")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
