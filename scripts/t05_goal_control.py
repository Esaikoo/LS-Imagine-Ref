"""T05: check offline, prepare real reference goals, evaluate paired control.

Only prepare/evaluate start MineDojo. Always run those commands with
MINEDOJO_HEADLESS=1. No training or original replay writes occur here.
"""

import argparse
import copy
from datetime import datetime
import gc
import json
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


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        label = {"check": "T05_OFFLINE_CHECK", "prepare": "T05_BENCHMARK_PREPARE", "evaluate": "T05_CONTROL_EVAL"}[self.data["command"]]
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def accepted(record, command):
    import goal_library as gl
    gl.require(record.get("command") == command and record.get("status") in ("passed", "passed_with_warnings"),
               f"需要已通过的 {command} 验收记录")


def load_inputs(args, report):
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl

    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    path = baseline.project_path(args.checkpoint)
    payload = t04.training_checkpoint(path)
    identity = {"checkpoint": baseline.file_signature(path), "checkpoint_sha256": bc.file_hash(path),
                "cache_id": cache.cache_id, "bundle_id": cache.bundle["bundle_id"],
                "counters": payload["counters"], "conditioning": payload["options"]["conditioning"],
                "inference_backend": bc.backend_info(args.device)}
    if args.command == "check":
        checked = read_json(baseline.project_path(args.t04_verify_dir) / "report.json")
        accepted(checked, "verify")
        gl.require(checked.get("cache_id") == cache.cache_id and checked.get("bundle_id") == cache.bundle["bundle_id"] and
                   checked.get("counters") == payload["counters"] and
                   checked.get("source_inputs_before", {}).get(str(path)) == identity["checkpoint"],
                   "T04 verify 记录与指定 checkpoint/缓存不匹配；请先独立 verify 此文件")
    else:
        checked = read_json(baseline.project_path(args.check_dir) / "report.json")
        accepted(checked, "check")
        gl.require(checked.get("input_identity") == identity, "T05 check 记录与当前 checkpoint 内容不匹配")
    model = ctl.load_model(payload, cache, args.device)
    gl.require(model.options["conditioning"] == "goal", "主 checkpoint 必须是 goal BC；no_goal 只能作为独立对照")
    identity["model_id"] = ctl.model_identity(model)
    if args.command != "check":
        # check stored model_id as well as file identity above
        gl.require(checked["model_id"] == identity["model_id"], "T05 模型加载结果与 offline check 不同")
    report.data.update(input_identity={k: v for k, v in identity.items() if k != "model_id"},
                       model_id=identity["model_id"], state_policy=cache.bundle["state_policy"],
                       checkpoint_selection=payload.get("selection"),
                       optimizer_updates=0, new_env_steps=0, reference_env_steps=0, evaluation_env_steps=0,
                       candidate_used_for_selection=False, controlled_reachability_verified=False)
    report.check("strict_inference_load", "PASS", f"严格加载已验收 T04 模型，step={payload['counters']['step']}；不构造或恢复优化器")
    del payload
    gc.collect()
    return cache, model


def check(args, report, cache, model):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl

    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "T03 与缓存来源不同")
    rng = bc.capture_rng(args.device)
    maximum = 0.0
    for index in sorted({0, len(dataset.metadata["episodes"]) - 1}):
        ep = dataset.episode(index)
        stop = min(args.prefix_steps, len(ep["image"]) - 1)
        expected = model.state_encoder.rollout(ep, stop)
        state, recovered = None, []
        for frame in range(stop + 1):
            obs = {key: ep[key][frame] for key in ctl.OBS_KEYS}
            state, features = model.state_encoder.step(obs, ep["action"][frame], state)
            recovered.append(features[0])
        actual = torch.stack(recovered)
        maximum = max(maximum, float((actual - expected).abs().max()))
        t02.vector_difference(report, f"incremental_episode_{index}", actual, expected, check=True)
        reset = {key: ep[key][0] for key in ctl.OBS_KEYS}
        _, restarted = model.state_encoder.step(reset, ep["action"][0], state)
        t02.vector_difference(report, f"reset_episode_{index}", restarted, expected[:1], check=True)
        bad = dict(reset, is_first=False)
        t02.rejection(report, "missing_history_guard", lambda: model.state_encoder.step(bad, ep["action"][1]), "本局历史")
        t02.rejection(report, "reset_action_guard", lambda: model.state_encoder.step(reset, ep["action"][1]), "必须为零")
        # Full reset prefix remains causal when later pixels/actions change.
        changed = copy.deepcopy(ep)
        if stop + 1 < len(ep["image"]):
            changed["image"][stop + 1:] = 0
            changed["action"][stop + 1:] = 0
            t02.vector_difference(report, "future_invariance", model.state_encoder.rollout(changed, stop), expected, check=True)
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "在线状态推进改变了调用方 RNG")
    report.data["incremental_max_error"] = maximum
    report.check("incremental_causal_state", "PASS", "逐帧真实状态与 T03 完整前缀一致；reset 清空旧历史；未来不影响当前状态；RNG 未推进")
    features, goals, remaining, _ = cache.worker_batch(cache.worker_rows["validation"][:32], args.device)
    with torch.no_grad():
        for mode in ("trained", "zero_goal", "original"):
            probs = model.action_dist(features, goals, remaining, mode).probs
            gl.require(probs.shape == (len(features), cache.metadata["action_dim"]) and bool(torch.isfinite(probs).all()) and
                       bool(torch.allclose(probs.sum(-1), torch.ones(len(features), device=probs.device), atol=1e-5)), "在线动作接口异常")
        for invalid in (0, model.bundle["horizon"] + 1):
            t02.rejection(report, "remaining_guard", lambda invalid=invalid: model.action_dist(features, goals, torch.full_like(remaining, invalid)), "remaining 必须")
    # Counterfactual pairing rejects real-history differences. Markers stay
    # in RAM, not in benchmark/goal/training data.
    prefix = {key: ep[key][:2].copy() for key in ctl.OBS_KEYS}
    prefix.update(action=ep["action"][:2].copy(), features=expected[:2].cpu().numpy().copy(),
                  telemetry=[{"pose": dict(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0), "inventory": {}, "health": 20.0} for _ in range(2)])
    gl.require(ctl.prefix_comparison(prefix, prefix)["passed"], "同一历史未能通过配对检查")
    for key in ("position", "action", "reward", "state"):
        bad = copy.deepcopy(prefix)
        if key == "position":
            bad["telemetry"][0]["pose"]["x"] = 10
        elif key == "action":
            bad["action"][1] = np.roll(bad["action"][1], 1)
        elif key == "reward":
            bad["obs_reward"][1] += 1
        else:
            bad["features"] += 10
        gl.require(not ctl.prefix_comparison(prefix, bad)["passed"], f"{key} 变更未被配对检查识别")
    for policy in ("mode", "sample"):
        p = np.array([0.2, 0.8])
        gl.require(ctl.select_action(p, policy, 0.5) == 1, "真实动作选择错误")
    gl.require(ctl.angle_error(179, -179) == 2, "朝向跨 180 度计算错误")
    finished_session = object.__new__(ctl.Session)
    finished_session.done = True
    t02.rejection(report, "after_terminal_guard", lambda: finished_session.step(0), "真实结束后")
    report.check("execution_contract", "PASS", "12 维 onehot/incoming 接口和剩余步数边界正确；不同历史拒绝配对；动作随机数独立；内存探针不写入训练数据")
    report.check("scope", "WARN", "只完成在线接口离线检查；尚未启动 MineDojo/MineCLIP，目标控制待真实评估")


def count_step(report, category):
    report.data["new_env_steps"] += 1
    report.data[category + "_env_steps"] += 1


def run_reference(args, report, model, scenario, directory, prefix_actions=None, branch=None):
    import numpy as np
    import goal_control as ctl
    import goal_library as gl

    session = None
    try:
        print(f"[ENV] reference seed={scenario['seed']} branch={branch}; fresh reset", flush=True)
        session = ctl.Session(model, scenario, directory / "environment", lambda: count_step(report, "reference"))
        uniforms = np.random.RandomState(scenario["seed"] + 101 + (branch or 0)).uniform(size=args.horizon)
        actions = []
        for step in range(args.prefix_steps):
            if prefix_actions is None:
                probs = ctl.probabilities(model, session.features, np.zeros(model.library.goals.shape[1], np.float32), 1, "original")
                action = ctl.select_action(probs, "mode", 0.0)
            else:
                action = int(prefix_actions[step])
            actions.append(action)
            session.step(action)
            gl.require(not session.done, "参考预热在控制起点前结束；请减少 --prefix-steps 或使用其他预先指定种子")
        prefix = session.arrays()
        if branch is not None:
            for step in range(args.horizon):
                if args.reference_mode == "turn_pair":
                    names = baseline.source_action_names()
                    action = names.index("turn_left" if branch == 0 else "turn_right") if step < max(1, args.horizon // 2) else names.index("noop")
                else:
                    probs = ctl.probabilities(model, session.features, np.zeros(model.library.goals.shape[1], np.float32), 1, "original")
                    action = ctl.select_action(probs, "sample", float(uniforms[step]))
                session.step(action)
                if session.done:
                    break
        trajectory = session.arrays()
        trajectory["goal_features"] = np.stack([ctl.encode_goal(model, row) for row in session.rows])
        return {"prefix": prefix, "trajectory": trajectory, "prefix_actions": actions,
                "endpoint_goal": trajectory["goal_features"][-1], "steps": len(session.rows) - 1 - args.prefix_steps,
                "success": any(event["success"] for event in session.events), "specs": session.specs}
    finally:
        if session is not None:
            try:
                ctl.save_session(session, directory, baseline.write_video)
            finally:
                session.close()
        gc.collect()


def benchmark_id(manifest):
    import goal_library as gl
    return gl.tensor_digest({}, {k: v for k, v in manifest.items() if k != "benchmark_id"})


def prepare(args, report, cache, model):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    from PIL import Image, ImageDraw

    gl.require(args.horizon <= model.bundle["horizon"], "参考执行长度超过已训练低层剩余步数范围")
    manifest = {"format": ctl.FORMAT, "bundle_id": cache.bundle["bundle_id"], "library_id": cache.bundle["library"]["library_id"],
                "encoder_id": cache.bundle["library"]["encoder_id"], "state_policy": cache.bundle["state_policy"],
                "inference_backend": bc.backend_info(args.device),
                "environment_fingerprint": ctl.environment_fingerprint(ROOT), "horizon": args.horizon,
                "prefix_steps": args.prefix_steps, "reference_mode": args.reference_mode,
                "min_goal_distance": args.min_goal_distance, "pair_limits": ctl.PAIR_LIMITS,
                "requested_seeds": args.seeds, "cases": [], "reference_env_steps": 0,
                "protocol": "fresh_task_fixed_pose_seed_clear_weather_frozen_time_no_fast_reset",
                "scope": "real diagnostic goals, not training data or task-success labels"}
    report.data["environment_protocol"] = manifest["protocol"]
    tiles = []
    for seed in args.seeds:
        case_dir = report.directory / f"case_{seed}"
        case_dir.mkdir()
        # First discover a valid spawn pose. All reference/evaluation arms
        # subsequently start from this same explicitly specified pose.
        discovery = None
        try:
            print(f"[ENV] discover seed={seed}; fresh reset", flush=True)
            discovery = ctl.Session(model, {"seed": seed}, case_dir / "discovery_environment", lambda: count_step(report, "reference"))
            start = discovery.tap.current["pose"]
            gl.require(start is not None, "MineDojo 未提供位置/朝向，无法建立可控起点")
            baseline.write_json(case_dir / "spawn.json", discovery.tap.current)
        finally:
            if discovery is not None:
                try:
                    ctl.save_session(discovery, case_dir / "discovery", baseline.write_video)
                finally:
                    discovery.close()
            discovery = None
            gc.collect()
        scenario = {"seed": seed, "start_position": start}
        left = run_reference(args, report, model, scenario, case_dir / "reference_0", branch=0)
        right = run_reference(args, report, model, scenario, case_dir / "reference_1", left["prefix_actions"], branch=1)
        comparison = ctl.prefix_comparison(left["prefix"], right["prefix"])
        baseline.write_json(case_dir / "prefix_comparison.json", comparison)
        gl.require(comparison["passed"], f"seed={seed} 参考历史不一致；见 prefix_comparison.json；不能用该起点做配对结论")
        gl.require(left["steps"] == right["steps"] == args.horizon, "参考分支提前终止；保留视频，但不建立不同执行预算的配对目标")
        goals = np.stack([left["endpoint_goal"], right["endpoint_goal"]])
        separation = float(ctl.distances(goals[:1], goals[1:])[0, 0])
        current_goal = ctl.encode_goal(model, {k: left["prefix"][k][-1] for k in ctl.OBS_KEYS})
        initial_distances = ctl.distances(current_goal[None], goals)[0]
        baseline.write_json(case_dir / "goal_diagnostics.json", {"goal_separation": separation,
            "initial_distances": initial_distances.tolist(), "minimum": args.min_goal_distance,
            "reference_endpoint_pose_difference": ctl.pose_difference(left["trajectory"]["telemetry"][-1]["pose"], right["trajectory"]["telemetry"][-1]["pose"])})
        gl.require(separation >= args.min_goal_distance and bool(np.all(initial_distances >= args.min_goal_distance)),
                   f"seed={seed} 两目标过近或控制起点已接近目标；见 goal_diagnostics.json；不能据此验收目标切换")
        classes = model.library.assign(torch.as_tensor(goals, device=args.device)).cpu().tolist()
        center_distances = ctl.distances(goals, model.library.centers.cpu().numpy())
        np.savez_compressed(case_dir / "goals.npz", goals=goals,
            images=np.stack([ref["trajectory"]["image"][-1] for ref in (left, right)]),
            heatmaps=np.stack([ref["trajectory"]["heatmap"][-1] for ref in (left, right)]))
        files = [case_dir / "goals.npz", case_dir / "spawn.json"]
        for branch in (0, 1):
            files += [case_dir / f"reference_{branch}" / name for name in ("trajectory.npz", "events.json", "video.mp4")]
        manifest["cases"].append({"seed": seed, "directory": case_dir.name, "scenario": scenario,
            "prefix_actions": left["prefix_actions"], "goal_classes": classes, "goal_separation": separation,
            "nearest_center_distances": center_distances.min(1).tolist(),
            "initial_distances": initial_distances.tolist(), "files": {str(path.relative_to(report.directory)): bc.file_hash(path) for path in files}})
        for branch, ref in enumerate((left, right)):
            tile = Image.new("RGB", (384, 148), "white")
            ImageDraw.Draw(tile).text((4, 4), f"seed {seed} goal {branch} | class {classes[branch]} (display only)", fill="black")
            tile.paste(Image.fromarray(left["prefix"]["image"][-1]).resize((128, 128)), (0, 20))
            tile.paste(Image.fromarray(ref["trajectory"]["image"][-1]).resize((128, 128)), (128, 20))
            tile.paste(Image.fromarray(ref["trajectory"]["heatmap"][-1]).convert("RGB").resize((128, 128)), (256, 20))
            tiles.append(tile)
        print(f"[CASE] seed={seed} matched_prefix=1 goal_separation={separation:.5f} classes={classes}", flush=True)
    manifest["reference_env_steps"] = report.data["reference_env_steps"]
    manifest["benchmark_id"] = benchmark_id(manifest)
    baseline.write_json(report.directory / "benchmark.json", manifest)
    gallery = Image.new("RGB", (384, 148 * len(tiles)), "white")
    for index, tile in enumerate(tiles):
        gallery.paste(tile, (0, index * 148))
    gallery.save(report.directory / "targets.png")
    report.data.update(benchmark_id=manifest["benchmark_id"], cases=len(manifest["cases"]),
                       reference_goals=len(manifest["cases"]) * 2)
    report.check("paired_references", "PASS", "同一起点的完整真实历史、位置/朝向/物品与 RSSM 状态通过配对核对；两目标来自同预算真实执行")
    report.check("real_targets", "PASS", "冻结目标编码器编码真实终点；目标图集、参考视频及 SHA256 已保存；类别不作为成功标签")
    report.check("scope", "WARN", "已建立诊断目标；还需 evaluate 和查看视频。原 actor 参考使用 T03 posterior mode；该受控环境不是原任务成功率评测")


def load_benchmark(args, cache, report):
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl

    directory = baseline.project_path(args.benchmark_dir)
    manifest = read_json(directory / "benchmark.json")
    build = read_json(directory / "report.json")
    accepted(build, "prepare")
    gl.require(manifest["format"] == ctl.FORMAT and manifest["benchmark_id"] == benchmark_id(manifest) and
               build.get("benchmark_id") == manifest["benchmark_id"] and
               manifest["bundle_id"] == cache.bundle["bundle_id"] and manifest["library_id"] == cache.bundle["library"]["library_id"] and
               manifest["encoder_id"] == cache.bundle["library"]["encoder_id"] and manifest["state_policy"] == cache.bundle["state_policy"] and
               manifest["inference_backend"] == bc.backend_info(args.device) and
               manifest["environment_fingerprint"] == ctl.environment_fingerprint(ROOT) and manifest["pair_limits"] == ctl.PAIR_LIMITS,
               "benchmark 格式/内容、冻结依赖或真实 heatmap 环境代码/权重不匹配，请重新 prepare")
    gl.require(1 <= manifest["horizon"] <= cache.bundle["horizon"] and len(manifest["cases"]) > 0, "benchmark 长度或样本数错误")
    gl.require(manifest["requested_seeds"] == [case["seed"] for case in manifest["cases"]], "benchmark 丢弃了预先指定种子")
    for case in manifest["cases"]:
        gl.require(len(case["prefix_actions"]) == manifest["prefix_steps"] and case["scenario"]["seed"] == case["seed"] and
                   case["goal_separation"] >= manifest["min_goal_distance"] and
                   min(case["initial_distances"]) >= manifest["min_goal_distance"], "benchmark 起点、执行预算或目标区分度异常")
        for relative, digest in case["files"].items():
            path = (directory / relative).resolve()
            gl.require(directory in path.parents and bc.file_hash(path) == digest, "benchmark 真实参考文件 SHA256 不匹配")
    report.data.update(benchmark_id=manifest["benchmark_id"], cases=len(manifest["cases"]))
    report.check("benchmark_identity", "PASS", "原目标库/WM、实际热力图代码和 MineCLIP 权重、真实参考文件 SHA256 均匹配")
    return directory, manifest


def run_trial(args, report, model, case, benchmark, goals, target, mode, directory):
    import numpy as np
    import goal_control as ctl
    import goal_library as gl

    prefix_steps, horizon = benchmark["prefix_steps"], benchmark["horizon"]
    session = None
    comparison = None
    try:
        print(f"[ENV] eval seed={case['seed']} target={target} mode={mode}; fresh reset", flush=True)
        session = ctl.Session(model, case["scenario"], directory / "environment", lambda: count_step(report, "evaluation"))
        for action in case["prefix_actions"]:
            session.step(int(action))
            gl.require(not session.done, "重放预热在控制起点前结束")
        reference = ctl.read_trajectory(baseline.project_path(args.benchmark_dir) / case["directory"] / "reference_0")
        reference = {key: value[:prefix_steps + 1] for key, value in reference.items()}
        comparison = ctl.prefix_comparison(reference, session.arrays(), benchmark["pair_limits"])
        report.data.setdefault("pair_checks", []).append(dict(seed=case["seed"], target=target, mode=mode, **comparison))
        report.save()
        if not comparison["passed"]:
            return {"seed": case["seed"], "target": target, "mode": mode, "paired": False, "pair_check": comparison,
                    "reason": "actual reset/prefix mismatch; no behavioral comparison"}
        assigned = goals[1 - target] if mode == "shuffled_goal" else goals[target]
        goal_reference = ctl.read_trajectory(baseline.project_path(args.benchmark_dir) / case["directory"] / f"reference_{target}")
        # Same predeclared uniform sequence across conditions AND target
        # switches at a start. Environment RNG cannot perturb action draws.
        uniforms = np.random.RandomState(case["seed"] + args.action_seed + 701).uniform(size=horizon)
        distributions, actions, frames = [], [], []
        for step in range(horizon):
            if mode == "reference_replay":
                # Positive control: reissue the actual recorded next action,
                # never ask the worker to choose it or infer it from a goal.
                action = int(goal_reference["action"][prefix_steps + step + 1].argmax())
                p = ctl.onehot(action, model.bundle["config"]["num_actions"])
            else:
                p = ctl.probabilities(model, session.features, assigned, horizon - step, mode)
                action = ctl.select_action(p, args.execution_policy, float(uniforms[step]))
            distributions.append(p)
            actions.append(action)
            frames.append(len(session.rows) - 1)
            session.step(action)
            if session.done:
                break
        trajectory = session.arrays()
        trajectory["goal_features"] = np.stack([ctl.encode_goal(model, row) for row in session.rows])
        metrics = ctl.trial_metrics(trajectory, goals, target, prefix_steps, goal_reference["telemetry"][-1])
        control_events = session.events[prefix_steps + 1:]
        metrics.update(seed=case["seed"], target=target, mode=mode, paired=True,
            assigned_target_class=case["goal_classes"][target],
            actual_end_class=int(model.library.assign(trajectory["goal_features"][-1]).item()),
            issued_goal=1 - target if mode == "shuffled_goal" else target,
            goal_input="zero" if mode in ("zero_goal", "no_goal_bc") else "unused" if mode in ("original", "reference_replay") else "real_endpoint",
            actual_steps=len(actions), end_reason="environment_done" if session.done else "fixed_budget",
            task_success=any(event["success"] for event in control_events),
            task_success_during_prefix=any(event["success"] for event in session.events[:prefix_steps + 1]),
            external_return=float(sum(event["reward"] for event in control_events)),
            initial_action_probabilities=np.asarray(distributions[0]).tolist(),
            actions=actions, action_histogram=np.bincount(actions, minlength=model.bundle["config"]["num_actions"]).tolist(),
            directory=directory.name, pair_check=comparison)
        if mode == "reference_replay":
            if len(trajectory["image"]) == len(goal_reference["image"]):
                replay_check = ctl.prefix_comparison(goal_reference, trajectory, benchmark["pair_limits"])
            else:
                replay_check = {"passed": False, "reason": "reference execution length changed"}
            metrics.update(reference_replay_check=replay_check, reference_replay_valid=bool(replay_check["passed"]))
        trace = {"model_id": ctl.model_identity(model), "target_index": target, "assigned_goal": assigned.tolist(),
                 "execution_policy": "recorded_actions" if mode == "reference_replay" else args.execution_policy,
                 "probability_semantics": "prescribed_action" if mode == "reference_replay" else "policy_distribution",
                 "uniforms": [] if mode == "reference_replay" else uniforms[:len(actions)].tolist(),
                 "frames": frames, "remaining": list(range(horizon, horizon - len(actions), -1)),
                 "probabilities": np.asarray(distributions).tolist(), "action_ids": actions,
                 "distances_to_both_goals": ctl.distances(trajectory["goal_features"][prefix_steps:], goals).tolist()}
        # Save alongside trajectory after save_session creates the directory.
        return metrics, trace
    finally:
        if session is not None:
            try:
                ctl.save_session(session, directory, baseline.write_video)
                if comparison is not None:
                    baseline.write_json(directory / "prefix_comparison.json", comparison)
            finally:
                session.close()
        gc.collect()


def summarize(rows):
    import numpy as np
    result = {"engineering_only": True, "controlled_reachability_verified": False,
              "trials": len(rows), "matched_trials": sum(row["paired"] for row in rows),
              "unmatched_trials": sum(not row["paired"] for row in rows), "modes": {}, "paired_differences": [], "goal_switches": []}
    calibration = {(row["seed"], row["target"]): bool(row.get("reference_replay_valid", False))
                   for row in rows if row["mode"] == "reference_replay"}
    result["reference_replay"] = [{"seed": seed, "target": target, "passed": passed}
                                  for (seed, target), passed in sorted(calibration.items())]
    # A matched start is insufficient if replaying the known reference no
    # longer produces its endpoint. Exclude that entire target comparison.
    valid = [row for row in rows if row["paired"] and calibration.get((row["seed"], row["target"]), False)]
    result["eligible_trials"] = len(valid)
    for mode in sorted({row["mode"] for row in valid}):
        selected = [row for row in valid if row["mode"] == mode]
        result["modes"][mode] = {"trials": len(selected), "task_success_rate": float(np.mean([r["task_success"] for r in selected])),
            **{key: float(np.mean([r[key] for r in selected])) for key in ("distance_improvement", "relative_distance_improvement", "target_preference_margin", "end_distance")}}
    lookup = {(r["seed"], r["target"], r["mode"]): r for r in valid}
    for seed in sorted({r["seed"] for r in rows}):
        for target in (0, 1):
            goal = lookup.get((seed, target, "goal"))
            if goal:
                for mode in ("zero_goal", "shuffled_goal", "original", "no_goal_bc"):
                    control = lookup.get((seed, target, mode))
                    if control:
                        result["paired_differences"].append({"seed": seed, "target": target, "control": mode,
                            "distance_improvement_advantage": goal["distance_improvement"] - control["distance_improvement"],
                            "end_distance_advantage": control["end_distance"] - goal["end_distance"],
                            "target_margin_advantage": goal["target_preference_margin"] - control["target_preference_margin"]})
        for mode in sorted(result["modes"]):
            left, right = lookup.get((seed, 0, mode)), lookup.get((seed, 1, mode))
            if left and right:
                n = min(len(left["actions"]), len(right["actions"]))
                result["goal_switches"].append({"seed": seed, "mode": mode,
                    "common_steps": n, "action_disagreement": float(np.mean(np.asarray(left["actions"][:n]) != np.asarray(right["actions"][:n]))) if n else None,
                    "initial_probability_l1": float(np.abs(np.asarray(left["initial_action_probabilities"]) - np.asarray(right["initial_action_probabilities"])).sum()),
                    "lengths_equal": len(left["actions"]) == len(right["actions"]),
                    "both_prefer_assigned_target": left["target_preference_margin"] > 0 and right["target_preference_margin"] > 0})
    result["interpretation"] = ["动作改变本身不足以证明控制；需同时看各自目标进展、对照差异、物理遥测和视频。",
        "目标距离/类别不代表任务成功；参考位姿匹配是额外诊断，也不是通用语义成功标签。",
        "零目标是推理消融；只有独立 no_goal BC 才控制继续模仿学习的影响。",
        "这是受控短程诊断，不能与 T00 完整任务成功率直接比较；不自动通过 T05 科学验收。"]
    return result


def outcome_gallery(report, benchmark_directory, manifest, rows):
    """Start/targets above the actual paired endpoints, for quick review."""
    import numpy as np
    import goal_control as ctl
    from PIL import Image, ImageDraw

    panels = []
    modes = ["goal", "zero_goal", "shuffled_goal", "original", "no_goal_bc", "reference_replay"]
    for case in manifest["cases"]:
        reference = ctl.read_trajectory(benchmark_directory / case["directory"] / "reference_0")
        with np.load(benchmark_directory / case["directory"] / "goals.npz", allow_pickle=False) as data:
            images = [reference["image"][manifest["prefix_steps"]], *data["images"]]
        panel = Image.new("RGB", (600, 156), "white")
        draw = ImageDraw.Draw(panel)
        for index, (image, label) in enumerate(zip(images, (f"seed {case['seed']} start", "real target 0", "real target 1"))):
            draw.text((index * 200 + 4, 4), label, fill="black")
            panel.paste(Image.fromarray(image).resize((128, 128)), (index * 200, 24))
        panels.append(panel)
        for mode in modes:
            selected = [r for r in rows if r["seed"] == case["seed"] and r["mode"] == mode]
            if not selected:
                continue
            panel = Image.new("RGB", (600, 156), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((4, 24), mode, fill="black")
            for row in selected:
                x = (row["target"] + 1) * 200
                directory = report.directory / row.get("directory", f"seed_{row['seed']}_target_{row['target']}_{mode}")
                trajectory = ctl.read_trajectory(directory)
                label = f"target {row['target']} end | delta {row['distance_improvement']:.3f}" if row["paired"] else "UNPAIRED: excluded"
                if mode == "reference_replay" and not row.get("reference_replay_valid", False):
                    label = "REFERENCE REPLAY FAILED"
                draw.text((x + 4, 4), label, fill="black")
                panel.paste(Image.fromarray(trajectory["image"][-1]).resize((128, 128)), (x, 24))
            panels.append(panel)
    gallery = Image.new("RGB", (600, 156 * len(panels)), "white")
    for index, panel in enumerate(panels):
        gallery.paste(panel, (0, index * 156))
    gallery.save(report.directory / "outcomes.png")


def evaluate(args, report, cache, model):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl

    directory, manifest = load_benchmark(args, cache, report)
    modes = [*ctl.MODES, "reference_replay"]
    no_goal = None
    if args.no_goal_checkpoint:
        payload = t04.training_checkpoint(baseline.project_path(args.no_goal_checkpoint))
        record = read_json(baseline.project_path(args.no_goal_verify_dir) / "report.json")
        accepted(record, "verify")
        path = baseline.project_path(args.no_goal_checkpoint)
        gl.require(record.get("cache_id") == cache.cache_id and record.get("bundle_id") == cache.bundle["bundle_id"] and
                   record.get("counters") == payload["counters"] and
                   record.get("source_inputs_before", {}).get(str(path)) == baseline.file_signature(path), "no_goal BC 验收记录不匹配")
        no_goal = ctl.load_model(payload, cache, args.device)
        gl.require(no_goal.options["conditioning"] == "no_goal", "独立无目标对照必须使用 --conditioning no_goal 训练")
        other_options = dict(no_goal.options, conditioning="goal")
        gl.require(other_options == model.options and payload["counters"]["step"] == report.data["input_identity"]["counters"]["step"],
                   "公平 no_goal BC 对照需要相同训练配置和更新预算")
        report.data["no_goal_identity"] = {"checkpoint": baseline.file_signature(path), "sha256": bc.file_hash(path), "model_id": ctl.model_identity(no_goal)}
        del payload
        modes.append("no_goal_bc")
    else:
        report.check("no_goal_control", "WARN", "本次只有目标置零推理消融；未运行独立 no_goal BC，不能排除继续 BC 的影响")
    rows = []
    for case in manifest["cases"]:
        with np.load(directory / case["directory"] / "goals.npz", allow_pickle=False) as data:
            goals = data["goals"].copy()
            for target in (0, 1):
                encoded = ctl.encode_goal(model, {"image": data["images"][target], "heatmap": data["heatmaps"][target]})
                gl.require(np.allclose(encoded, goals[target], atol=3e-5, rtol=3e-5), "参考目标保存/重编码不一致")
        for target in (0, 1):
            # Counterbalance condition order instead of always running goal first.
            offset = (case["seed"] + target) % len(modes)
            order = modes[offset:] + modes[:offset]
            for mode in order:
                output = report.directory / f"seed_{case['seed']}_target_{target}_{mode}"
                result = run_trial(args, report, no_goal if mode == "no_goal_bc" else model, case, manifest, goals, target, mode, output)
                if isinstance(result, tuple):
                    row, trace = result
                    baseline.write_json(output / "control_trace.json", trace)
                    print(f"[TRIAL] seed={case['seed']} target={target} mode={mode} delta={row['distance_improvement']:.5f} margin={row['target_preference_margin']:.5f} steps={row['actual_steps']} success={int(row['task_success'])}", flush=True)
                else:
                    row = result
                    print(f"[UNPAIRED] seed={case['seed']} target={target} mode={mode}; excluded from effects", flush=True)
                rows.append(row)
                baseline.write_json(output / "metrics.json", row)
                t04.append_json(report.directory / "trials.jsonl", row)
                baseline.write_json(report.directory / "diagnostics.json", summarize(rows))
    diagnostics = summarize(rows)
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    flat = [{key: row.get(key) for key in ("seed", "target", "mode", "paired", "actual_steps", "task_success",
             "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "reference_pose_match")} for row in rows]
    baseline.write_csv(report.directory / "trials.csv", flat)
    outcome_gallery(report, directory, manifest, rows)
    report.data.update(trials=len(rows), matched_trials=diagnostics["matched_trials"], unmatched_trials=diagnostics["unmatched_trials"],
                       summary=diagnostics["modes"], execution_policy=args.execution_policy)
    if no_goal is not None:
        gl.require(ctl.model_identity(no_goal) == report.data["no_goal_identity"]["model_id"], "评估改变了 no_goal 模型")
    report.check("pairing", "PASS" if not diagnostics["unmatched_trials"] else "FAIL",
                 f"匹配 {diagnostics['matched_trials']}/{len(rows)}；不匹配保留录像和诊断并排除效果比较；不能据此断言低层失败")
    replay_passed = len(diagnostics["reference_replay"]) == len(manifest["cases"]) * 2 and all(r["passed"] for r in diagnostics["reference_replay"])
    report.check("reference_replay", "PASS" if replay_passed else "FAIL", "正对照重放参考动作并核对整段真实观测/状态/遥测；不通过的目标组排除效果比较")
    report.check("real_execution", "PASS", "固定真实终点目标、逐步更新因果 RSSM、剩余步数递减；真实结束即停止；未使用候选模型、虚拟 zoom 或 imagined 终点")
    report.check("artifacts", "PASS", "保存真实视频、RGB/heatmap/动作/状态、原生执行动作、位置/物品事件、每步目标距离和配对汇总")
    report.check("behavior_acceptance", "WARN", "PASS 只表示工程执行完成。需查看视频及对照：目标切换是否带来对应终点进展；本工具不自动批准 T06")


def source_paths(cache_dir, args):
    meta = read_json(cache_dir / "cache_manifest.json")
    source = meta["source_metadata"]
    protected = [cache_dir, Path(meta["dataset_dir"]), Path(meta["t03_verify_dir"]), Path(source["baseline_dir"]),
                 Path(source["checkpoint"]["path"]).parent, Path(source["library_path"]).parent]
    files = [cache_dir / name for name in ("states.npy", "tables.npz", "cache_manifest.json", "frozen_bundle.pt", "report.json")]
    files += [Path(meta["dataset_dir"]) / name for name in ("segments_manifest.json", "segments.npz", "report.json")]
    files += [Path(meta["t03_verify_dir"]) / "report.json", Path(source["baseline_dir"]) / "report.json",
              Path(source["baseline_dir"]) / "resolved_config.json", Path(source["checkpoint"]["path"]), Path(source["library_path"])]
    for entry in source["episodes"]:
        files.append(Path(entry["path"]))
        protected.append(Path(entry["path"]).parent)
    files.append(baseline.project_path(args.checkpoint))
    protected.append(baseline.project_path(args.checkpoint).parent)
    for name in ("t04_verify_dir", "check_dir", "no_goal_verify_dir"):
        value = getattr(args, name, None)
        if value:
            path = baseline.project_path(value)
            protected.append(path)
            files.append(path / "report.json")
    if getattr(args, "no_goal_checkpoint", None):
        path = baseline.project_path(args.no_goal_checkpoint)
        protected.append(path.parent)
        files.append(path)
    if getattr(args, "benchmark_dir", None):
        path = baseline.project_path(args.benchmark_dir)
        protected.append(path)
        files.extend(file for file in path.rglob("*") if file.is_file())
    return protected, list(dict.fromkeys(files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--t04-verify-dir", required=True)
    check_parser.add_argument("--prefix-steps", type=int, default=32)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--check-dir", required=True)
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    prepare_parser.add_argument("--prefix-steps", type=int, default=32)
    prepare_parser.add_argument("--horizon", type=int, default=16)
    prepare_parser.add_argument("--reference-mode", choices=("original_sample", "turn_pair"), default="original_sample")
    prepare_parser.add_argument("--min-goal-distance", type=float, default=0.01)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--check-dir", required=True)
    eval_parser.add_argument("--benchmark-dir", required=True)
    eval_parser.add_argument("--execution-policy", choices=("mode", "sample"), default="mode")
    eval_parser.add_argument("--action-seed", type=int, default=0)
    eval_parser.add_argument("--no-goal-checkpoint")
    eval_parser.add_argument("--no-goal-verify-dir")
    for command in (check_parser, prepare_parser, eval_parser):
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True, help="T04 的真实 latest.pt 或 best_worker.pt")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
        command.add_argument("--output-root", default="relevance_map/t05_outputs")
    args = parser.parse_args()
    if args.command in ("check", "prepare") and args.prefix_steps < (1 if args.command == "check" else 0):
        parser.error("prefix-steps 必须非负；离线检查至少一步")
    if args.command == "prepare" and (args.horizon < 1 or not 0 < args.min_goal_distance <= 2 or
            len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s < 2**31 - 10000 for s in args.seeds)):
        parser.error("horizon/目标距离/种子参数无效；种子必须互异且为非负有限整数")
    if args.command == "evaluate" and (bool(args.no_goal_checkpoint) != bool(args.no_goal_verify_dir) or not 0 <= args.action_seed < 2**31 - 10000):
        parser.error("no_goal checkpoint 与 verify 目录必须一起提供；action-seed 必须为非负有限整数")
    if args.command != "check" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动 MineDojo 必须显式使用 MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = source_paths(baseline.project_path(args.cache_dir), args)
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"无法读取输入清单：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else baseline.project_path(args.output_root) / (args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出目录必须独立于原运行、回放、缓存、输入 checkpoint/验收/参考目标目录")
    directory.mkdir(parents=True, exist_ok=False)
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    before, model, rng = None, None, None
    try:
        import torch
        import goal_bc as bc
        import goal_control as ctl
        import goal_library as gl
        if torch.device(args.device).type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA 不可用")
            torch.cuda.set_device(torch.device(args.device))
        elif args.command != "check":
            raise ValueError("当前真实 heatmap/MineCLIP 路径需要 CUDA；CPU 只用于 offline check")
        baseline.runtime_info(report)
        report.data["runtime"]["torch"] = str(torch.__version__)
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        for entry in source["episodes"]:
            gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "T03 原 replay 大小/修改时间改变")
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化 checkpoint 已改变")
        rng = bc.capture_rng(args.device)
        if args.command != "check":
            baseline.video_preflight(report)
        cache, model = load_inputs(args, report)
        versions = {id(p): p._version for p in model.parameters()}
        model_id = ctl.model_identity(model)
        {"check": check, "prepare": prepare, "evaluate": evaluate}[args.command](args, report, cache, model)
        gl.require(versions == {id(p): p._version for p in model.parameters()} and ctl.model_identity(model) == model_id, "T05 修改了模型参数或目标库")
        report.check("no_training", "PASS", f"低层/原 actor/WM/目标库内容及参数版本未变；optimizer_updates=0；新增真实动作步={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.check("execution", "FAIL", f"{type(error).__name__}: {error}；已完成轨迹尽量保留，见 error.txt")
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原 checkpoint/回放及输入产物大小/修改时间未变（非内容哈希）；新结果只在独立目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
