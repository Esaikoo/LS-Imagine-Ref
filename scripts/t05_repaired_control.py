"""Fixed repaired-worker confirmation: offline check, then 30 autonomous trials.

No training, checkpoint selection, start matching, retries or partial resume.
Evaluate requires MINEDOJO_HEADLESS=1. Numerical gates never authorize T06
without video and physical-direction review.
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
import t05_reference_repair as repairing
import t05_residual_control as original


class Report(pilot.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_REPAIRED_CONTROL_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    names = (*repairing.CODE_FILES, "goal_repaired_control.py", "scripts/t05_repaired_control.py")
    return dict(pilot.code_identity(), **{name: bc.file_hash(ROOT / name) for name in names})


def passed_checks(record, names):
    return set(names) <= {row["name"] for row in record["checks"] if row["level"] == "PASS"}


def load_inputs(args, report):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_information_probe as probe
    import goal_library as gl
    import goal_random_control as random_control
    import goal_reference_repair as repair
    import goal_repaired_control as confirmation
    import goal_residual_worker as residual

    path = baseline.project_path(args.checkpoint)
    payload, dependencies = repair.read_checkpoint(path)
    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    policy = repair.load_policy(path, cache, args.device)
    verification_path = baseline.project_path(args.repair_verify_dir) / "report.json"
    verification = legacy.read_json(verification_path)
    legacy.accepted(verification, "verify")
    identity = dict(checkpoint=baseline.file_signature(path), checkpoint_sha256=bc.file_hash(path),
        checkpoint_format=repair.FORMAT, input_identity=policy.identity, counters=payload["counters"],
        model_id=policy.model_id, worker_version=policy.worker_version, repair_updates=policy.repair_updates,
        inference_backend=bc.backend_info(args.device), dependencies=payload["dependencies"],
        repair_verify_sha256=bc.file_hash(verification_path))
    gl.require(policy.worker_version == 300 and policy.repair_updates == 100 and
               verification.get("repair_format") == repair.FORMAT and
               verification.get("repair_identity") == policy.identity and
               verification.get("input_identity") == policy.identity and
               verification.get("counters") == payload["counters"] and
               verification.get("model_id") == policy.model_id and
               verification.get("worker_version") == policy.worker_version and
               verification.get("backend") == identity["inference_backend"] and
               verification.get("source_inputs_before", {}).get(str(path)) == identity["checkpoint"] and
               verification.get("repair_code") == legacy.read_json(dependencies["repair_check_report"])["repair_code"] and
               passed_checks(verification, ("strict_load", "inference_artifact_guard", "roundtrip",
                   "next_update_equivalence", "frozen_dependencies", "source_inputs_unchanged")),
               "需要同一固定100步修复latest和缓存的独立verify，不能采用旧残差或验收快照")
    runtime = residual.OnlineRuntime(policy, args.device)
    fingerprint = ctl.environment_fingerprint(ROOT)
    report.data.update(input_identity=identity, model_id=policy.model_id, worker_version=300,
        repair_updates=100, evaluation_code=code_identity(), environment_fingerprint=fingerprint,
        comparison_protocol=confirmation.PROTOCOL, environment_protocol=ctl.ENVIRONMENT_PROTOCOL,
        visual_preprocessing_policy=ctl.VISUAL_PREPROCESSING_POLICY, optimizer_updates=0,
        new_env_steps=0, evaluation_env_steps=0, worker_control_actions=0,
        task_success_evaluated=False, candidate_used_for_selection=False,
        same_hidden_state=False, controlled_reachability_verified=False,
        behavior_accepted=False, t06_approved=False)
    report.check("strict_repaired_inference_load", "PASS", "绑定修复独立verify；源200+追加100=version300；"
                 "只构造冻结底座/WM/目标库及两修正分支，无优化器或候选模型")

    # The old benchmark remains an accepted artifact of its old model. Validate
    # it in that historical context, then use only its targets/initialization.
    source = baseline.project_path(args.source_benchmark_dir)
    old = legacy.read_json(source / "benchmark.json")
    calibration = legacy.read_json(dependencies["calibration_manifest"])
    calibrated = calibration["design"]
    gl.require(old["input_identity"]["checkpoint_sha256"] == policy.identity["lineage"]["source_checkpoint_sha256"] and
               calibrated["source_benchmark_id"] == old["benchmark_id"] and
               calibrated["source_manifest_sha256"] == bc.file_hash(source / "benchmark.json") and
               calibrated["source_report_sha256"] == bc.file_hash(source / "report.json") and
               calibrated["environment_fingerprint"] == fingerprint,
               "目标来源/已标定场景与修复模型的真实来源不同")
    context = SimpleNamespace(data=dict(environment_fingerprint=fingerprint,
        input_identity=old["input_identity"], model_id=old["model_id"]), check=report.check)
    original.load_benchmark(SimpleNamespace(benchmark_dir=str(source)), context)
    case = next((item for item in old["cases"] if item["seed"] == 0), None)
    gl.require(case is not None and case["scenario"] == calibrated["scenario"] and
               case["prefix_actions"] == calibrated["prefix_actions"] and old["horizon"] == 16 and
               old["prefix_steps"] == 32, "当前确认仅接受已标定的seed0世界和32步真实前缀")
    names = pilot.action_names()
    visual = legacy.read_json(source / case["directory"] / "visual_preprocessing.json")
    gl.require(names == calibrated["action_names"] and len(names) == runtime.bundle["config"]["num_actions"] and
               visual["reference_0"] == visual["reference_1"] == calibrated["visual_preprocessing"],
               "动作接口或真实视觉预处理不同")
    with np.load(source / case["directory"] / "goals.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    gl.require(bank["goals"].shape == (2, cache.metadata["goal_dim"]) and bank["goals"].dtype == np.float32 and
               bank["images"].shape == (2, 64, 64, 3) and bank["heatmaps"].shape == (2, 64, 64) and
               bank["images"].dtype == bank["heatmaps"].dtype == np.uint8 and
               np.isfinite(bank["goals"]).all() and
               np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5) and
               probe.array_digest({"goals": bank["goals"]}) == policy.identity["fixed_goals_id"],
               "固定视觉目标与修复实验不同")
    content = gl.tensor_digest({}, {name: bank[name].tolist() for name in bank})
    gl.require(content == calibrated["target_content_id"], "固定目标内容ID不同")
    for target in (0, 1):
        reference = ctl.read_trajectory(source / case["directory"] / f"reference_{target}")
        gl.require(len(reference["image"]) == 49 and
                   np.array_equal(reference["image"][-1], bank["images"][target]) and
                   np.array_equal(reference["heatmap"][-1], bank["heatmaps"][target]) and
                   reference["telemetry"][-1] == calibrated["target_telemetry"][target] and
                   np.allclose(ctl.encode_goal(runtime, dict(image=bank["images"][target],
                       heatmap=bank["heatmaps"][target])), bank["goals"][target], atol=3e-5, rtol=3e-5),
                   "两个固定目标必须是原完整真实终点，不能采用失败prepare或替换坐标")
    design = dict(seed=0, repetitions=confirmation.REPETITIONS, randomization_seed=confirmation.RANDOMIZATION_SEED,
        execution_policy="mode", modes=list(random_control.MODES),
        schedule=random_control.schedule(0, confirmation.REPETITIONS, confirmation.RANDOMIZATION_SEED))
    plan = dict(format=confirmation.FORMAT, comparison_protocol=confirmation.PROTOCOL,
        source_benchmark=str(source), source_benchmark_id=old["benchmark_id"],
        source_manifest_sha256=bc.file_hash(source / "benchmark.json"),
        source_report_sha256=bc.file_hash(source / "report.json"),
        input_identity=identity, evaluation_code=code_identity(), environment_fingerprint=fingerprint,
        scenario=case["scenario"], visual_preprocessing=visual["reference_0"], action_names=names,
        warmup_steps=32, prefix_actions=case["prefix_actions"], start_actions=[0] * 32 + case["prefix_actions"],
        control_start_frame=64, horizon=16, design=design, gate=copy.deepcopy(confirmation.GATE),
        maximum_new_env_steps=2400, worker_version=300, repair_updates=100,
        target_content_id=content, target_telemetry=calibrated["target_telemetry"],
        goal_source="unchanged real RGB/heatmap endpoints; no coordinate input",
        same_hidden_state=False, matched_start_comparison=False, old_failures_unchanged=True)
    confirmation.validate_plan(plan)
    plan["design_id"] = gl.tensor_digest({}, plan)
    report.data.update(design_id=plan["design_id"], planned_trials=30, maximum_new_env_steps=2400,
        source_benchmark_id=old["benchmark_id"], execution_policy="mode", randomization_seed=1,
        matched_start_comparison=False, old_failures_unchanged=True)
    if args.command == "evaluate":
        checked_dir = baseline.project_path(args.check_dir)
        checked = legacy.read_json(checked_dir / "report.json")
        legacy.accepted(checked, "check")
        gl.require(checked.get("comparison_protocol") == confirmation.PROTOCOL and
                   checked.get("input_identity") == identity and checked.get("design_id") == plan["design_id"] and
                   checked.get("evaluation_code") == code_identity() and
                   legacy.read_json(checked_dir / "design.json") == plan and
                   checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz") and
                   passed_checks(checked, ("strict_repaired_inference_load", "fixed_visual_targets",
                       "trial_output_preflight", "causal_execution_interface", "independent_start_contract",
                       "confirmation_gate_contract", "saved_execution_contract", "predeclared_design",
                       "no_training", "source_inputs_unchanged")),
                   "需要当前修复确认check；旧试跑check或修改后的计划不能用于执行")
        with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
            gl.require(all(np.array_equal(saved[name], bank[name]) for name in bank), "check的目标内容改变")
        report.check("design_identity", "PASS", "worker300、mode、新随机顺序、30次预算、目标、门槛和代码与check一致")
    report.check("fixed_visual_targets", "PASS", "两个已标定真实视觉终点及完整初始化脚本不变；不采集新目标或要求跨局起点相等")
    del payload
    gc.collect()
    return cache, runtime, plan, bank


def validate_trace(arrays, events, trace, plan, dimension):
    import numpy as np
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_calibration as calibration

    first, horizon = plan["control_start_frame"], plan["horizon"]
    actions = trace["action_ids"]
    gl.require(all(type(action) is int and 0 <= action < dimension for action in actions) and
               1 <= len(actions) <= horizon and
               trace["remaining"] == list(range(horizon, horizon - len(actions), -1)) and
               trace["execution_policy"] == "mode", "动作或剩余预算与预声明mode不同")
    count = calibration.validate_execution(arrays, events, plan["start_actions"], actions, dimension)
    gl.require(count == len(actions) and (count == horizon or bool(arrays["is_last"][-1] or arrays["is_terminal"][-1])),
               "未执行完整16步且没有真实结束")
    p = np.asarray(trace["probabilities"], dtype=np.float64)
    u = np.asarray(trace["uniforms"], dtype=np.float64)
    gl.require(p.shape == (count, dimension) and u.shape == (count,) and np.isfinite(p).all() and
               np.isfinite(u).all() and np.all((u >= 0) & (u < 1)) and np.all(p >= 0) and
               np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0), "保存的动作概率或随机数异常")
    gl.require([ctl.select_action(values, "mode", float(uniform)) for values, uniform in zip(p, u)] == actions,
               "真实下一动作不是已记录分布的mode")
    gl.require(len(arrays["image"]) == first + count + 1, "控制起点或真实下一动作索引不同")
    return count


def check(args, report, cache, runtime, plan, bank):
    import numpy as np
    import goal_library as gl
    import goal_repaired_control as confirmation

    # Reuse accepted causal-state/distribution/output guards without changing
    # their fingerprinted implementations or the deployment policy.
    pilot.check(SimpleNamespace(device=args.device, repetitions=5), report, cache, runtime, plan, bank)
    first, dimension = plan["control_start_frame"], len(plan["action_names"])
    frames = first + plan["horizon"] + 1
    markers = dict(image=np.zeros((frames, 64, 64, 3), np.uint8), heatmap=np.zeros((frames, 64, 64), np.uint8),
        features=np.zeros((frames, 2), np.float32), obs_reward=np.zeros((frames, 1), np.float32),
        action=np.concatenate([np.zeros((1, dimension), np.float32),
            np.eye(dimension, dtype=np.float32)[plan["start_actions"] + [0] * 16]]),
        is_first=np.arange(frames) == 0, is_last=np.zeros(frames, bool), is_terminal=np.zeros(frames, bool))
    telemetry = dict(pose=dict(x=0., y=0., z=0., yaw=0., pitch=0.), inventory={}, health=20.)
    events = [dict(frame=frame, reward=0., done=False, error=None,
                   native_actions=[] if frame == 0 else [[0]], telemetry=telemetry) for frame in range(frames)]
    trace = dict(action_ids=[0] * 16, remaining=list(range(16, 0, -1)), execution_policy="mode",
        probabilities=np.eye(dimension)[[0] * 16].tolist(), uniforms=[.5] * 16)
    gl.require(validate_trace(markers, events, trace, plan, dimension) == 16, "保存接口标记不一致")
    wrong = copy.deepcopy(trace)
    wrong["action_ids"][0] = 1
    t02.rejection(report, "saved_next_action_guard", lambda: validate_trace(markers, events, wrong, plan, dimension), "incoming")
    missing = copy.deepcopy(events)
    del missing[-1]["telemetry"]
    t02.rejection(report, "saved_telemetry_guard", lambda: validate_trace(markers, missing, trace, plan, dimension), "遥测")
    wrong = copy.deepcopy(trace)
    wrong["probabilities"][0] = np.eye(dimension)[1].tolist()
    t02.rejection(report, "recorded_mode_guard", lambda: validate_trace(markers, events, wrong, plan, dimension), "mode")
    report.check("saved_execution_contract", "PASS", "真实保存接口检查全历史遥测/原生事件/下一动作/mode/remaining；内存标记不写成轨迹")

    rows = []
    for cell in plan["design"]["schedule"]:
        target, mode = cell["target"], cell["mode"]
        start, end = [1., 1.], [1., 1.]
        progress = .5 if mode == "goal" else .1
        end[target] -= progress
        rows.append(dict(cell, execution_valid=True, saved_execution_verified=True, video_verified=True,
            actual_steps=16, end_reason="fixed_budget", task_success=False, start_telemetry=telemetry,
            prefix_movement=dict(position=0.), start_distance=1., end_distance=1. - progress,
            distance_improvement=progress, target_preference_margin=progress, preference_improvement=progress,
            start_distances_to_both_goals=start, end_distances_to_both_goals=end))
    summary = confirmation.summarize(rows, plan)
    gl.require(summary["numerical_gate"]["passed"] and not summary["t06_approved"], "数值与人工门槛未分开")
    boundary = copy.deepcopy(rows)
    failing = [row for row in boundary if row["target"] == 1 and row["mode"] == "goal"]
    for row in failing[:1]:
        row.update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                   preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(confirmation.summarize(boundary, plan)["numerical_gate"]["passed"], "4/5边界被错误拒绝")
    failing[1].update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                      preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(not confirmation.summarize(boundary, plan)["numerical_gate"]["passed"], "3/5被错误通过")
    weak = copy.deepcopy(rows)
    for row in weak:
        if row["target"] == 1 and row["mode"] == "goal":
            row.update(end_distance=1., distance_improvement=0., target_preference_margin=0.,
                       preference_improvement=0., end_distances_to_both_goals=[1., 1.])
    gl.require(not confirmation.summarize(weak, plan)["numerical_gate"]["passed"], "pooled结果掩盖一个目标失败")
    missing = copy.deepcopy(rows)
    missing[0].update(execution_valid=False, error="in-memory missing-measurement marker")
    gl.require(not confirmation.summarize(missing, plan)["numerical_gate"]["passed"] and
               not confirmation.summarize(rows[:-1], plan)["numerical_gate"]["passed"], "缺测/未完成被通过")
    equal = copy.deepcopy(rows)
    for row in equal:
        if row["mode"] != "goal":
            target = row["target"]
            end = [1., 1.]
            end[target] = .5
            row.update(end_distance=.5, distance_improvement=.5, target_preference_margin=.5,
                       preference_improvement=.5, end_distances_to_both_goals=end)
    gl.require(not confirmation.summarize(equal, plan)["numerical_gate"]["passed"], "与对照持平被当作优势")
    t02.rejection(report, "confirmation_duplicate_guard", lambda: confirmation.summarize(rows + rows[:1], plan), "重复")
    changed = copy.deepcopy(plan)
    changed["gate"]["minimum_directional_trials_per_target"] = 3
    t02.rejection(report, "gate_change_guard", lambda: confirmation.validate_plan(changed), "门槛")
    changed = copy.deepcopy(plan)
    changed["design"]["execution_policy"] = "sample"
    t02.rejection(report, "execution_policy_guard", lambda: confirmation.validate_plan(changed), "执行方式")
    wrong = copy.deepcopy(rows)
    wrong[0]["distance_improvement"] += .1
    t02.rejection(report, "own_progress_guard", lambda: confirmation.summarize(wrong, plan), "自身进展")
    report.check("confirmation_gate_contract", "PASS", "两个目标分别4/5正进展且正确偏好、五次均值胜两对照；缺测/持平/未完成不通过；数值达标仍需人工核对")
    gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or name == "mineclip" or
                       name.startswith("mineclip.") for name in sys.modules), "离线check不应导入环境/MineCLIP")


def verify_saved_trial(runtime, plan, bank, row):
    import numpy as np
    import torch
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_calibration as calibration
    import goal_residual_control as control

    directory = Path(row["artifact_dir"])
    arrays = ctl.read_trajectory(directory)
    events = legacy.read_json(directory / "events.json")
    trace = legacy.read_json(directory / "control_trace.json")
    history = legacy.read_json(directory / "history_check.json")
    dimension = runtime.bundle["config"]["num_actions"]
    count = validate_trace(arrays, events, trace, plan, dimension)
    first, target, mode = plan["control_start_frame"], row["target"], row["mode"]
    issued = None if mode == "no_goal" else 1 - target if mode == "swapped_goal" else target
    index = next(index for index, cell in enumerate(plan["design"]["schedule"])
                 if all(cell[name] == row[name] for name in ("seed", "repeat", "target", "mode")))
    action_seed = (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)
    gl.require(trace["model_id"] == runtime.policy.model_id and trace["condition"] == mode and
               trace["evaluated_target"] == target and trace["issued_target"] == issued and
               trace["action_seed"] == action_seed and
               np.array_equal(trace["uniforms"], np.random.RandomState(action_seed).uniform(size=plan["horizon"])[:count]) and
               history.get("own_causal_history_passed") is True and history.get("reset_is_real") is True and
               history.get("comparison_to_other_histories") is False and history.get("start_frame") == first and
               history.get("total_frames") == len(arrays["image"]) and row["actual_steps"] == count and
               row["video"]["frames"] == len(arrays["image"]), "保存的模型/条件/因果历史或完整解码视频不一致")
    device = next(runtime.parameters()).device
    measurements = []
    for offset in range(count + 1):
        frame = first + offset
        feature = ctl.encode_goal(runtime, {name: arrays[name][frame] for name in ("image", "heatmap")})
        values = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                                     events[frame]["telemetry"], bank, plan["target_telemetry"])
        measurements.append(dict(frame=frame, control_step=offset, goal_feature=feature.tolist(), **values))
        if offset < count:
            state = torch.as_tensor(arrays["features"][frame:frame + 1].copy(), device=device)
            p = control.probabilities(runtime, state, bank["goals"], plan["horizon"] - offset, target, mode)
            gl.require(np.allclose(p, trace["probabilities"][offset], atol=3e-5, rtol=3e-5) and
                       ctl.select_action(p, "mode", trace["uniforms"][offset]) == trace["action_ids"][offset],
                       "保存真实状态的控制概率或mode动作重现不一致")
    distances = [[item[f"distance_{target}"] for target in (0, 1)] for item in measurements]
    gl.require(np.allclose(distances, trace["distances_to_both_goals"], atol=3e-5, rtol=3e-5),
               "保存真实画面的目标距离重现不一致")
    start, end = np.asarray(distances[0]), np.asarray(distances[-1])
    margin = float(end[1 - target] - end[target])
    # Persist measurements computed from saved, real frames and actual endpoints.
    row.update(start_distances_to_both_goals=start.tolist(), end_distances_to_both_goals=end.tolist(),
        start_distance=float(start[target]), end_distance=float(end[target]),
        distance_improvement=float(start[target] - end[target]), target_preference_margin=margin,
        preference_improvement=margin - float(start[1 - target] - start[target]),
        saved_execution_verified=True, video_verified=True,
        fixed_horizon_completed=count == plan["horizon"],
        target_pose_errors_at_start={str(target): {key: value for key, value in measurements[0].items()
            if key.endswith(f"_error_{target}")} for target in (0, 1)},
        target_pose_errors_at_end={str(target): {key: value for key, value in measurements[-1].items()
            if key.endswith(f"_error_{target}")} for target in (0, 1)})
    trace.update(comparison_protocol=plan["comparison_protocol"], design_id=plan["design_id"],
                 worker_version=300, repair_updates=100, control_start_frame=first)
    baseline.write_json(directory / "frame_metrics.json", measurements)
    baseline.write_json(directory / "control_trace.json", trace)
    return measurements


def outcome_gallery(report, plan, bank, rows):
    import numpy as np
    from PIL import Image, ImageDraw
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
                path = Path(row["artifact_dir"]) / "trajectory.npz"
                if path.is_file():
                    # An invalid attempt's raw last image is a diagnostic only;
                    # previewing it must not revalidate or discard its failure.
                    try:
                        with np.load(path, allow_pickle=False) as saved:
                            image = saved["image"][-1].copy()
                        if image.shape == (64, 64, 3) and image.dtype == np.uint8:
                            panel.paste(Image.fromarray(image).resize((128, 128)), (x, y + 20))
                    except (ValueError, KeyError, IndexError):
                        draw.text((x, y + 40), "raw image unavailable", fill="black")
        panel.save(report.directory / f"outcomes_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime, plan, bank):
    import numpy as np
    from PIL import Image
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_repaired_control as confirmation

    baseline.video_preflight(report)
    cells = plan["design"]["schedule"]
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, len(cells) * 20 * 1024**2 + 16 * 1024**2)
    baseline.write_json(report.directory / "design.json", plan)
    np.savez_compressed(report.directory / "targets.npz", **bank)
    report.data["targets_sha256"] = bc.file_hash(report.directory / "targets.npz")
    report.data["attempted_env_steps"] = 0
    report.check("storage_preflight", "PASS", "预留30次完整真实历史/视频空间；不复制模型或旧轨迹")
    report.check("independent_starts", "PASS", "各局fresh reset并保留自身32noop+32前缀；直接按固定mode条件执行，不比较或筛选跨局起点")
    rows, frames, balance = [], [], []
    first_start = None
    for index, cell in enumerate(cells):
        report.require_writable()
        directory = report.directory / f"trial_{index:03d}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['mode']}"
        row = pilot.run_trial(report, runtime, plan, bank, cell, directory)
        report.data["worker_control_actions"] += row["actual_steps"]
        row.update(saved_execution_verified=False, video_verified=False)
        if row["execution_valid"]:
            try:
                measurements = verify_saved_trial(runtime, plan, bank, row)
                frames.extend(dict(trial=index, target=cell["target"], mode=cell["mode"], repeat=cell["repeat"],
                    **{key: value for key, value in item.items() if key != "goal_feature"}) for item in measurements)
            except OSError:
                raise
            except Exception as error:
                row.update(execution_valid=False, error=f"{type(error).__name__}: {error}")
                baseline.write_json(directory / "saved_execution_error.json", dict(error=row["error"], traceback=traceback.format_exc()))
                report.check(f"trial_{index}_saved_execution", "FAIL", row["error"] + "；保留尝试，不补样")
        if row["execution_valid"] and (directory / "start.json").is_file() and (directory / "trajectory.npz").is_file():
            trajectory = ctl.read_trajectory(directory)
            first = plan["control_start_frame"]
            if len(trajectory["image"]) > first:
                if first_start is None:
                    first_start = trajectory
                rgb = np.abs(trajectory["image"][first].astype(np.float32) - first_start["image"][first].astype(np.float32))
                balance.append(dict(trial=index, target=cell["target"], mode=cell["mode"], repeat=cell["repeat"],
                    rgb_mae_to_first=float(rgb.mean()), rgb_p99_to_first=float(np.percentile(rgb, 99)),
                    pose_difference_to_first=ctl.pose_difference(trajectory["telemetry"][first]["pose"], first_start["telemetry"][first]["pose"]),
                    rssm_relative_l2_to_first=float(np.linalg.norm(trajectory["features"][first] - first_start["features"][first]) /
                        max(float(np.linalg.norm(first_start["features"][first])), 1e-8)), diagnostic_only=True))
                pair = Image.new("RGB", (256, 128), "white")
                pair.paste(Image.fromarray(trajectory["image"][first]).resize((128, 128)), (0, 0))
                pair.paste(Image.fromarray(trajectory["image"][-1]).resize((128, 128)), (128, 0))
                pair.save(directory / "start_and_outcome.png")
        rows.append(row)
        baseline.write_json(directory / "metrics.json", row)
        t04.append_json(report.directory / "trials.jsonl", row)
        summary = confirmation.summarize(rows, plan)
        summary["start_differences_diagnostic_only"] = balance
        baseline.write_json(report.directory / "diagnostics.json", summary)
        baseline.write_json(report.directory / "gate.json", summary["numerical_gate"])
        baseline.write_json(report.directory / "trials.json", rows)
        report.data.update(attempted_trials=len(rows), numerical_gate=summary["numerical_gate"])
        report.save()
        report.require_writable()
        detail = f"steps={row['actual_steps']} d={row['end_distance']:.5f} progress={row['distance_improvement']:.5f}" if row["execution_valid"] else row.get("error", "execution error")
        print(f"[CONFIRM {index + 1}/30] repeat={cell['repeat']} target={cell['target']} mode={cell['mode']} {detail}", flush=True)
    baseline.write_csv(report.directory / "frame_metrics.csv", frames)
    baseline.write_csv(report.directory / "trials.csv", [{name: row.get(name) for name in (
        "seed", "repeat", "target", "mode", "execution_valid", "saved_execution_verified", "video_verified",
        "actual_steps", "fixed_horizon_completed", "returned_env_steps", "end_reason", "task_success",
        "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "preference_improvement",
        "error", "artifact_dir")} for row in rows])
    outcome_gallery(report, plan, bank, rows)
    # Five modest six-trial strips avoid an unreadable 7680-pixel thumbnail.
    for repeat in range(5):
        panel = Image.new("RGB", (1536, 150), "white")
        for index, row in enumerate(item for item in rows if item["repeat"] == repeat):
            path = Path(row["artifact_dir"]) / "start_and_outcome.png"
            if path.is_file():
                with Image.open(path) as pair:
                    panel.paste(pair, (index * 256, 22))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(panel)
        for index, row in enumerate(item for item in rows if item["repeat"] == repeat):
            draw.text((index * 256 + 2, 3), f"target{row['target']} {row['mode']} | own start / end", fill="black")
        panel.save(report.directory / f"starts_and_outcomes_repeat_{repeat}.png")
    saved = dict(format=confirmation.EVALUATION_FORMAT, comparison_protocol=confirmation.PROTOCOL,
        design=plan, design_id=plan["design_id"], rows=rows, input_identity=report.data["input_identity"],
        new_env_steps=report.data["new_env_steps"], attempted_env_steps=report.data["attempted_env_steps"],
        worker_control_actions=report.data["worker_control_actions"],
        check_report_sha256=bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
        artifacts={str(path.relative_to(report.directory)): bc.file_hash(path)
                   for row in rows for path in Path(row["artifact_dir"]).iterdir() if path.is_file()},
        summary_artifacts={name: bc.file_hash(report.directory / name) for name in
            ("design.json", "targets.npz", "trials.json", "trials.csv", "diagnostics.json", "gate.json")},
        behavior_accepted=False, t06_approved=False)
    saved["evaluation_id"] = gl.tensor_digest({}, saved)
    baseline.write_json(report.directory / "evaluation_manifest.json", saved)
    valid = sum(row["execution_valid"] for row in rows)
    gl.require(report.data["new_env_steps"] <= plan["maximum_new_env_steps"], "真实执行超过预声明预算")
    report.data.update(evaluation_id=saved["evaluation_id"], execution_valid_trials=valid,
                       task_success_evaluated=True, full_design_completed=True)
    report.check("planned_attempts", "PASS", "按固定新随机顺序尝试30/30，全部保留，无重试、补样、best终点或失败续跑")
    report.check("real_control_interface", "PASS" if valid == 30 else "FAIL", f"完整保存、真实动作/遥测/概率及视频有效{valid}/30")
    gate = summary["numerical_gate"]
    print(f"[NUMERICAL GATE] passed={int(gate['passed'])}; human_review_pending=1; t06_approved=0", flush=True)
    for target, values in gate["by_target"].items():
        print(f"[TARGET {target}] direction={values['positive_progress_and_correct_preference']}/5 "
              f"mean_progress={values['mean_progress']} advantages={values['progress_advantages']}", flush=True)
    report.check("numerical_behavior_gate", "WARN", "数值门槛" + ("达到" if gate["passed"] else "未达到") +
                 "；逐目标结果见gate.json；该结论与工程PASS分别报告")
    report.check("behavior_acceptance", "WARN", "须结合全部视频、位置/朝向和起点平衡人工核对；仅当前世界局部目标，T06未自动批准")


def source_paths(args):
    import goal_library as gl
    import goal_reference_repair as repair

    protected, files, source = legacy.source_paths(baseline.project_path(args.cache_dir), args)
    _, dependencies = repair.read_checkpoint(baseline.project_path(args.checkpoint))
    for path in dependencies.values():
        protected.append(path.parent)
        files.append(path)
    for value in (args.repair_verify_dir, args.source_benchmark_dir):
        directory = baseline.project_path(value)
        protected.append(directory)
        files += [path for path in directory.rglob("*") if path.is_file() and path.suffix in (".json", ".npz", ".mp4", ".png")]
    if args.command == "evaluate":
        files += [baseline.project_path(args.check_dir) / name for name in ("design.json", "targets.npz")]
    files += [ROOT / name for name in code_identity()]
    files = list(dict.fromkeys(path.resolve() for path in files))
    gl.require(all(path.is_file() for path in files), "输入依赖缺失，请保留服务器底座/验收/check和目标来源")
    return list(dict.fromkeys(path.resolve() for path in protected)), files, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking, evaluation = commands.add_parser("check"), commands.add_parser("evaluate")
    evaluation.add_argument("--check-dir", required=True)
    for command in (checking, evaluation):
        for name in ("cache-dir", "checkpoint", "repair-verify-dir", "source-benchmark-dir"):
            command.add_argument("--" + name, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
    args = parser.parse_args()
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = source_paths(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "repaired_control_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于模型、依赖、缓存、回放、旧参考和check")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    report.data.update(new_env_steps=0, evaluation_env_steps=0, worker_control_actions=0,
                       behavior_accepted=False, t06_approved=False)
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
                   "确认改变冻结模型或产生梯度")
        report.check("no_training", "PASS", f"optimizer_updates=0；冻结worker/底座/WM/目标内容未变；"
                     f"worker动作={report.data['worker_control_actions']}，新env.step={report.data['new_env_steps']}")
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
                             "源模型/依赖/回放/目标/验收/check与旧代码大小及修改时间未变；旧FAIL不改写")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
