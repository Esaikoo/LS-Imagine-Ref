"""T05 residual control: check/adopt offline; evaluate starts MineDojo.

Evaluate in predeclared complete blocks, keeping every failed trial.
Use MINEDOJO_HEADLESS=1 for evaluate. No training or replay writes.
"""

import argparse
import copy
from datetime import datetime
import gc
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_RESIDUAL_CONTROL_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return {name: bc.file_hash(ROOT / name) for name in (
        "goal_residual_control.py", "goal_residual_worker.py", "goal_information_probe.py",
        "goal_control_stats.py", "goal_segments.py", "goal_library.py", "goal_bc.py", "long_horizon.py",
        "networks.py", "tools.py", "goal_sampling.py", "scripts/t00_baseline.py",
        "scripts/t01_checkpoint_check.py", "scripts/t02_goal_library.py", "scripts/t04_goal_bc.py",
        "scripts/t05_residual_control.py")}


def load_inputs(args, report):
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control
    import goal_residual_worker as residual

    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    path = baseline.project_path(args.checkpoint)
    payload, dependencies = residual.read_checkpoint(path)
    policy = residual.load_policy(path, cache, args.device)
    identity = dict(checkpoint=baseline.file_signature(path), checkpoint_sha256=bc.file_hash(path),
                    input_identity=policy.identity, counters=payload["counters"], model_id=policy.model_id,
                    worker_version=policy.worker_version, inference_backend=bc.backend_info(args.device),
                    dependencies=payload["dependencies"])
    if args.command == "check":
        record = legacy.read_json(baseline.project_path(args.residual_verify_dir) / "report.json")
        legacy.accepted(record, "verify")
        required = {"strict_load", "strict_inference_load", "incremental_causal_state",
                    "next_update_equivalence", "frozen_dependencies", "source_inputs_unchanged"}
        gl.require(record.get("architecture") == residual.ARCHITECTURE and
                   record.get("input_identity") == policy.identity and record.get("counters") == payload["counters"] and
                   record.get("model_id") == policy.model_id and record.get("worker_version") == policy.worker_version and
                   record.get("source_inputs_before", {}).get(str(path)) == identity["checkpoint"] and
                   required <= {row["name"] for row in record["checks"] if row["level"] == "PASS"},
                   "需要同一残差latest及缓存的已通过独立verify；旧T04或探针verify不能替代")
    else:
        record = legacy.read_json(baseline.project_path(args.check_dir) / "report.json")
        legacy.accepted(record, "check")
        gl.require(record.get("comparison_protocol") == control.PROTOCOL and record.get("input_identity") == identity and
                   record.get("evaluation_code") == code_identity() and
                   record.get("environment_fingerprint") == ctl.environment_fingerprint(ROOT),
                   "残差控制check与模型/缓存/代码/环境不同；请先运行当前入口check")
    runtime = residual.OnlineRuntime(policy, args.device)
    report.data.update(input_identity=identity, model_id=policy.model_id, worker_version=policy.worker_version,
                       evaluation_code=code_identity(), environment_fingerprint=ctl.environment_fingerprint(ROOT),
                       comparison_protocol=control.PROTOCOL, environment_protocol=ctl.ENVIRONMENT_PROTOCOL,
                       visual_preprocessing_policy=ctl.VISUAL_PREPROCESSING_POLICY,
                       optimizer_updates=0, new_env_steps=0, evaluation_env_steps=0,
                       same_hidden_state=False, controlled_reachability_verified=False,
                       task_success_evaluated=False, candidate_used_for_selection=False)
    report.check("strict_residual_inference_load", "PASS", f"匹配残差独立verify；step={policy.worker_version}；"
                 "只构造冻结底座/WM/目标库和两修正分支，无优化器或候选模型")
    del payload
    gc.collect()
    return cache, runtime


def check(args, report, cache, runtime):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_control_stats as repeated
    import goal_library as gl
    import goal_residual_control as control

    task = runtime.bundle["config"]["task"].partition("_")[2]
    original = baseline.yaml_read(ROOT / "envs/tasks/task_specs.yaml")[task].get("screenshot_specs")
    visual = ctl.visual_preprocessing_specs({"screenshot_specs": original})
    gl.require(bool(visual) == bool(original) and (not original or all(
        visual[k] == value for k, value in original.items() if k not in ("reset_flag", "step_flag"))), "任务视觉预处理不同")
    report.check("visual_preprocessing_contract", "PASS", "保留任务ScreenshotWrapper/HUD，只关闭截图文件输出；不启动环境")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "真实历史数据集身份不同")
    rng = bc.capture_rng(args.device)
    prefix = None
    for index in sorted({0, len(dataset.metadata["episodes"]) - 1}):
        ep = dataset.episode(index)
        stop = min(args.prefix_steps, len(ep["image"]) - 1)
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
        t02.rejection(report, "after_terminal_guard", lambda: runtime.step(dict(reset, is_last=True), ep["action"][0],
            np.zeros(cache.metadata["goal_dim"], dtype=np.float32), 1), "结束")
        prefix = {name: ep[name][:2].copy() for name in ctl.OBS_KEYS}
        prefix.update(action=ep["action"][:2].copy(), features=expected[:2].cpu().numpy(), telemetry=[
            {"pose": dict(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0), "inventory": {}, "health": 20.0} for _ in range(2)])
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "因果状态检查推进了随机数")
    features, _, remaining, _ = cache.worker_batch(cache.worker_rows["validation"][:1], args.device)
    validation_goals = cache.tables["goals"][cache.candidate_rows["validation"]]
    distinct = np.flatnonzero(np.linalg.norm(validation_goals - validation_goals[0], axis=1) > 1e-5)
    gl.require(len(distinct) > 0, "目标替换接口检查需要两个不同的真实留出目标")
    pair = validation_goals[[0, int(distinct[0])]].copy()
    horizon = runtime.bundle["horizon"]
    for mode in control.MODES[:-1]:
        actual = control.probabilities(runtime, features[:1], pair, horizon, 0, mode)
        assigned = 1 if mode == "swapped_goal" else 0
        branch = "goal" if mode == "swapped_goal" else mode
        expected_goal = torch.as_tensor(pair[assigned:assigned + 1], device=args.device)
        expected = runtime.action_dist(features[:1], expected_goal, torch.full_like(remaining[:1], horizon), branch).probs[0].cpu().numpy()
        gl.require(np.array_equal(actual, expected), "评估与已验收部署分布不同")
    for mode in ("no_goal", "base"):
        gl.require(np.array_equal(control.probabilities(runtime, features[:1], pair, horizon, 0, mode),
                                 control.probabilities(runtime, features[:1], pair, horizon, 1, mode)), "独立无目标/底座依赖目标")
    for invalid in (0, runtime.bundle["horizon"] + 1, 1.5):
        t02.rejection(report, "remaining_guard", lambda invalid=invalid: control.probabilities(runtime, features[:1], pair, invalid, 0, "goal"), "remaining")
    gl.require(ctl.validate_scenario(ctl.make_scenario(0)) == "1", "非零世界种子映射错误")
    drift = copy.deepcopy(prefix)
    drift["features"] += 10
    drift["heatmap"] = ((drift["heatmap"].astype(np.uint16) + 1) % 256).astype(np.uint8)
    strict = ctl.prefix_comparison(prefix, drift)
    gl.require(not strict["passed"] and repeated.start_check(strict, True)["passed"], "严格失败被丢弃或错误声称同状态")
    for name in ("image", "action", "pose", "native"):
        bad = copy.deepcopy(drift)
        if name == "image":
            bad[name] = ((bad[name].astype(np.uint16) + 64) % 256).astype(np.uint8)
        elif name == "action":
            bad[name][1] = np.roll(bad[name][1], 1)
        elif name == "pose":
            bad["telemetry"][1]["pose"]["x"] += 10
        gl.require(not repeated.start_check(ctl.prefix_comparison(prefix, bad), name != "native")["passed"], "物理/RGB/动作漂移未拒绝")
    design = dict(seeds=[0], repetitions=3, randomization_seed=0, execution_policy="mode", modes=list(control.MODES),
                  schedule=control.schedule([0], 3, 0))
    planned = control.pending_cells(design, [], 1)
    rows = [dict(cell, start_eligible=True, execution_valid=True, task_success=False, start_distance=.2,
                 end_distance=.1, distance_improvement=.1, relative_distance_improvement=.5,
                 target_preference_margin=.03, preference_improvement=.01) for cell in planned]
    gl.require(len(planned) == 12 and control.summarize(rows, design)["eligible_trials"] == 12 and
               len(control.pending_cells(design, rows, 2)) == 24, "完整块分段/六组覆盖异常")
    t02.rejection(report, "partial_block_guard", lambda: control.pending_cells(design, rows[:-1], 2), "完整")
    t02.rejection(report, "duplicate_trial_guard", lambda: control.summarize(rows + rows[:1], design), "重复")
    rows[0]["start_eligible"] = False
    gl.require(control.summarize(rows, design)["eligible_trials"] == 0, "失败块没有整体排除")
    t02.rejection(report, "failed_block_resume_guard", lambda: control.pending_cells(design, rows, 2), "失败")
    report.check("residual_execution_contract", "PASS", "五种策略共用已验收分布；独立无目标/底座不依赖目标；真实incoming、reset/结束/remaining正确")
    report.check("randomized_block_contract", "PASS", "六组×两目标×三重复=36；首块12/余两块24；禁止不完整恢复、重复trial或隐藏失败；保留严格检查")
    report.check("scope", "WARN", "纯离线接口检查；内存标记不写入数据，无MineDojo/MineCLIP，无控制效果声明")


def adopt(args, report, cache, runtime):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_control_stats as repeated
    import goal_library as gl
    import goal_residual_control as control
    from PIL import Image, ImageDraw

    source = baseline.project_path(args.source_prepare_dir)
    original = legacy.read_json(source / "report.json")
    source_identity = original.get("input_identity", {})
    gl.require(original.get("command") == "prepare" and original.get("environment_protocol") == ctl.ENVIRONMENT_PROTOCOL and
               source_identity.get("cache_id") == cache.cache_id and
               source_identity.get("bundle_id") == runtime.bundle["bundle_id"],
               "需要相同冻结WM/目标库/缓存及已修复视觉协议的原始prepare参考；不能采用旧HUD参考")
    gl.require(any(row["name"] == "source_inputs_unchanged" and row["level"] == "PASS" for row in original.get("checks", [])),
               "原参考缺少源输入未变记录")
    recorded_fingerprint = original.get("environment_fingerprint")
    if recorded_fingerprint is not None:
        gl.require(recorded_fingerprint == report.data["environment_fingerprint"], "原参考环境代码或MineCLIP权重指纹不同")
    else:
        report.check("reference_provenance", "WARN", "旧prepare未保存完整环境指纹；只采用其真实动作/视觉终点，"
                     "重算当前冻结状态并核对已记录视觉协议。生成策略与新残差worker不同，不声称其表现可直接比较")
    seeds = original["arguments"]["seeds"]
    prefix, horizon = (original["arguments"][name] for name in ("prefix_steps", "horizon"))
    minimum = original["arguments"]["min_goal_distance"]
    gl.require(type(prefix) is int and prefix >= 1 and type(horizon) is int and
               1 <= horizon <= runtime.bundle["horizon"] and 0 < minimum <= 2, "原参考预算/目标距离异常")
    design = dict(seeds=seeds, repetitions=args.repetitions, randomization_seed=args.randomization_seed,
                  execution_policy=args.execution_policy, modes=list(control.MODES),
                  schedule=control.schedule(seeds, args.repetitions, args.randomization_seed))
    control.validate_design(design)
    manifest = dict(format=control.FORMAT, comparison_protocol=control.PROTOCOL,
        protocol=ctl.ENVIRONMENT_PROTOCOL, visual_preprocessing_policy=ctl.VISUAL_PREPROCESSING_POLICY,
        world_seed_policy=ctl.WORLD_SEED_POLICY, environment_fingerprint=report.data["environment_fingerprint"],
        evaluation_code=code_identity(), input_identity=report.data["input_identity"], model_id=runtime.policy.model_id,
        source_prepare=str(source), source_report_sha256=bc.file_hash(source / "report.json"),
        reference_generation_model_id=original.get("model_id"),
        reference_generation_environment_fingerprint=recorded_fingerprint,
        prefix_steps=prefix, horizon=horizon, pair_limits=ctl.PAIR_LIMITS, min_goal_distance=minimum,
        design=design, cases=[], same_hidden_state=False, controlled_reachability_verified=False)
    total_copy = sum((source / f"case_{seed}/reference_{branch}" / name).stat().st_size
                     for seed in seeds for branch in (0, 1) for name in ("trajectory.npz", "events.json", "video.mp4"))
    bc.require_disk_space(report.directory, total_copy + 8 * 1024**2)
    task = runtime.bundle["config"]["task"].partition("_")[2]
    expected_specs = ctl.visual_preprocessing_specs(baseline.yaml_read(ROOT / "envs/tasks/task_specs.yaml")[task])
    panels = []
    for seed in seeds:
        original_case = source / f"case_{seed}"
        scenario = legacy.read_json(original_case / "scenario.json")
        ctl.validate_scenario(scenario)
        gl.require(scenario["seed"] == seed and scenario.get("start_position") is not None, "原参考没有明确固定起点")
        visual = legacy.read_json(original_case / "visual_preprocessing.json")
        gl.require(set(visual) == {"reference_0", "reference_1"} and visual["reference_0"] == visual["reference_1"], "两分支视觉设置不同")
        for settings in visual.values():
            gl.require(settings["policy"] == ctl.VISUAL_PREPROCESSING_POLICY and not settings["screenshot_file_output"] and
                       settings["screenshot_specs"] == expected_specs and settings["screenshot_wrapper"] == bool(expected_specs) and
                       settings["remove_hud"] == (bool(expected_specs) and not bool(expected_specs.get("HUD", False))), "任务RGB/HUD协议不同")
        references = [ctl.read_trajectory(original_case / f"reference_{branch}") for branch in (0, 1)]
        gl.require(all(len(reference["image"]) == prefix + horizon + 1 and
                       not bool((reference["is_last"][:prefix + 1] | reference["is_terminal"][:prefix + 1]).any())
                       for reference in references), "参考长度/真实控制起点不正确；不静默改变预算")
        for branch, reference in enumerate(references):
            actual = runtime.state_encoder.rollout(reference, len(reference["image"]) - 1).cpu().numpy()
            gl.require(actual.shape == reference["features"].shape and
                       np.allclose(actual, reference["features"], atol=3e-5, rtol=3e-5), "原参考因果状态与当前冻结WM不同")
        prefixes = [{name: value[:prefix + 1] for name, value in reference.items()} for reference in references]
        strict = ctl.prefix_comparison(*prefixes)
        events = [legacy.read_json(original_case / f"reference_{branch}/events.json") for branch in (0, 1)]
        gl.require(all(len(ev) == prefix + horizon + 1 and all("native_actions" in event and not event.get("error") for event in ev)
                       for ev in events), "原参考缺少真实原生动作或包含执行错误")
        native = repeated.native_actions_equal([event["native_actions"] for event in events[0][:prefix + 1]],
                                               [event["native_actions"] for event in events[1][:prefix + 1]])
        start = repeated.start_check(strict, native)
        gl.require(start["passed"], f"seed={seed} 原参考物理/RGB/动作不可比：{start['failed_checks']}")
        goals = np.stack([ctl.encode_goal(runtime, {name: reference[name][-1] for name in ("image", "heatmap")}) for reference in references])
        current = np.stack([ctl.encode_goal(runtime, {name: reference[name][prefix] for name in ("image", "heatmap")}) for reference in references])
        separation, initial = float(ctl.distances(goals[:1], goals[1:])[0, 0]), ctl.distances(current, goals)
        case_dir = report.directory / f"case_{seed}"
        case_dir.mkdir()
        baseline.write_json(case_dir / "goal_diagnostics.json", dict(goal_separation=separation, initial_distances=initial.tolist(), minimum=minimum))
        gl.require(separation >= minimum and np.all(initial >= minimum), "两个视觉目标或目标/起点过近；不筛除种子重试")
        for name in ("scenario.json", "spawn.json", "visual_preprocessing.json"):
            shutil.copyfile(original_case / name, case_dir / name)
        if (original_case / "prefix_comparison.json").exists():
            shutil.copyfile(original_case / "prefix_comparison.json", case_dir / "source_prefix_comparison.json")
        for branch in (0, 1):
            destination = case_dir / f"reference_{branch}"
            destination.mkdir()
            for name in ("trajectory.npz", "events.json", "video.mp4"):
                shutil.copyfile(original_case / f"reference_{branch}" / name, destination / name)
        baseline.write_json(case_dir / "prefix_comparison.json", strict)
        baseline.write_json(case_dir / "start_comparison.json", start)
        np.savez_compressed(case_dir / "goals.npz", goals=goals,
            images=np.stack([reference["image"][-1] for reference in references]),
            heatmaps=np.stack([reference["heatmap"][-1] for reference in references]))
        classes = runtime.library.assign(torch.as_tensor(goals, device=args.device)).cpu().tolist()
        manifest["cases"].append(dict(seed=seed, directory=case_dir.name, scenario=scenario, goal_classes=classes,
            prefix_actions=references[0]["action"][1:prefix + 1].argmax(-1).tolist(), goal_separation=separation,
            initial_distances=initial.tolist(), files={str(path.relative_to(report.directory)): bc.file_hash(path)
                for path in sorted(case_dir.rglob("*")) if path.is_file()}))
        panel = Image.new("RGB", (660, 150), "white")
        draw = ImageDraw.Draw(panel)
        for index, image in enumerate((references[0]["image"][prefix], references[0]["image"][-1], references[1]["image"][-1])):
            draw.text((index * 220 + 4, 3), f"seed {seed} | " + ("start" if index == 0 else f"real target {index - 1}"), fill="black")
            panel.paste(Image.fromarray(image).resize((128, 128)), (index * 220, 20))
        panels.append(panel)
        print(f"[ADOPT] seed={seed} world_seed={scenario['world_seed']} comparable_start=1 "
              f"strict_pair={int(strict['passed'])} goal_separation={separation:.5f}", flush=True)
    manifest["benchmark_id"] = legacy.benchmark_id(manifest)
    baseline.write_json(report.directory / "benchmark.json", manifest)
    baseline.write_json(report.directory / "schedule.json", design)
    targets_panel = Image.new("RGB", (660, 150 * len(panels)), "white")
    for index, panel in enumerate(panels):
        targets_panel.paste(panel, (0, 150 * index))
    targets_panel.save(report.directory / "targets.png")
    shutil.copyfile(source / "report.json", report.directory / "source_prepare_report.json")
    report.data.update(benchmark_id=manifest["benchmark_id"], planned_trials=len(design["schedule"]),
                       execution_policy=design["execution_policy"], maximum_new_eval_steps=len(design["schedule"]) * (prefix + horizon))
    report.check("real_reference_adoption", "PASS", "两个真实终点/动作/完整历史离线复用并重编码；原严格失败保留；未复载旧生成策略或启动环境")
    report.check("predeclared_design", "PASS", f"六组、两个终点、{args.repetitions}次随机重复，共{len(design['schedule'])}trial；执行方式提前固定")
    report.check("scope", "WARN", "新模型身份和重复协议已绑定；旧参考生成策略不同，仅提供视觉目标/参考动作。尚未证明真实控制")


def load_benchmark(args, report):
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control

    directory = baseline.project_path(args.benchmark_dir)
    manifest = legacy.read_json(directory / "benchmark.json")
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "adopt")
    gl.require(manifest["format"] == control.FORMAT and manifest["comparison_protocol"] == control.PROTOCOL and
               manifest["protocol"] == ctl.ENVIRONMENT_PROTOCOL and manifest["world_seed_policy"] == ctl.WORLD_SEED_POLICY and
               manifest["visual_preprocessing_policy"] == ctl.VISUAL_PREPROCESSING_POLICY and
               manifest["pair_limits"] == ctl.PAIR_LIMITS and manifest["evaluation_code"] == code_identity() and
               manifest["environment_fingerprint"] == report.data["environment_fingerprint"] and
               manifest["input_identity"] == report.data["input_identity"] and manifest["model_id"] == report.data["model_id"] and
               manifest["benchmark_id"] == legacy.benchmark_id(manifest) == record["benchmark_id"], "残差benchmark/代码/环境/模型身份不同")
    control.validate_design(manifest["design"])
    gl.require([case["seed"] for case in manifest["cases"]] == manifest["design"]["seeds"], "丢弃或修改了计划种子")
    gl.require(type(manifest["prefix_steps"]) is int and manifest["prefix_steps"] >= 1 and
               type(manifest["horizon"]) is int and 1 <= manifest["horizon"] <= report.data["input_identity"]["input_identity"]["horizon"], "benchmark预算异常")
    for case in manifest["cases"]:
        ctl.validate_scenario(case["scenario"])
        gl.require(case["directory"] == f"case_{case['seed']}" and case["scenario"]["seed"] == case["seed"] and
                   len(case["prefix_actions"]) == manifest["prefix_steps"] and
                   legacy.read_json(directory / case["directory"] / "scenario.json") == case["scenario"], "case起点或动作历史异常")
        required = {str(Path(case["directory"]) / name) for name in ("scenario.json", "goals.npz")}
        required.update(str(Path(case["directory"]) / f"reference_{branch}" / name)
                        for branch in (0, 1) for name in ("trajectory.npz", "events.json", "video.mp4"))
        gl.require(required <= set(case["files"]), "真实目标/参考文件声明缺失")
        for relative, digest in case["files"].items():
            path = (directory / relative).resolve()
            gl.require(directory in path.parents and path.is_file() and bc.file_hash(path) == digest, "真实参考/目标内容SHA256不同")
    report.data.update(benchmark_id=manifest["benchmark_id"], execution_policy=manifest["design"]["execution_policy"])
    report.check("benchmark_identity", "PASS", "六组预声明计划、真实参考、模型/依赖、视觉环境和代码指纹一致")
    return directory, manifest


def run_trial(report, runtime, benchmark_dir, manifest, case, goals, cell, directory):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as repeated
    import goal_library as gl
    import goal_residual_control as control

    target, mode, repeat = (cell[name] for name in ("target", "mode", "repeat"))
    prefix, horizon = manifest["prefix_steps"], manifest["horizon"]
    session, strict = None, None
    # Common action draws are independent of environment RNG and trial order.
    action_seed = (case["seed"] + manifest["design"]["randomization_seed"] + repeat * 1009 + 701) % (2**31 - 1)
    uniforms = np.random.RandomState(action_seed).uniform(size=horizon)
    assigned = 1 - target if mode == "swapped_goal" else target
    events = legacy.read_json(benchmark_dir / case["directory"] / "reference_0/events.json")
    goal_reference = ctl.read_trajectory(benchmark_dir / case["directory"] / f"reference_{target}")
    reference = ctl.read_trajectory(benchmark_dir / case["directory"] / "reference_0")
    reference = {name: value[:prefix + 1] for name, value in reference.items()}
    def on_step():
        report.data["evaluation_env_steps"] += 1
        report.data["new_env_steps"] += 1
    try:
        print(f"[ENV] residual seed={case['seed']} world_seed={case['scenario']['world_seed']} "
              f"repeat={repeat} target={target} mode={mode}; fresh reset", flush=True)
        session = ctl.Session(runtime, case["scenario"], directory / "environment", on_step)
        recorded_visual = legacy.read_json(benchmark_dir / case["directory"] / "visual_preprocessing.json")["reference_0"]
        gl.require(session.specs["visual_preprocessing"] == recorded_visual, "当前环境视觉处理与参考不同")
        for action in case["prefix_actions"]:
            session.step(int(action))
            gl.require(not session.done, "重放前缀在控制起点前结束")
        gl.require(not bool(session.observation["is_last"]) and not bool(session.observation["is_terminal"]), "真实控制起点已经结束")
        strict = ctl.prefix_comparison(reference, session.arrays(), manifest["pair_limits"])
        native = repeated.native_actions_equal([event["native_actions"] for event in events[:prefix + 1]],
                                               [event["native_actions"] for event in session.events])
        start = repeated.start_check(strict, native)
        row = dict(cell, directory=directory.name, artifact_dir=str(directory), same_hidden_state=False,
                   start_eligible=bool(start["passed"]), execution_valid=False, start_comparison=start)
        if not start["passed"]:
            row.update(reason="physical/RGB/action prefix mismatch; no retry", actual_steps=0)
            return row, dict(scope="failed start; no control actions", model_id=runtime.policy.model_id)
        distributions, actions, frames = [], [], []
        for step in range(horizon):
            remaining = horizon - step
            if mode == "reference_replay":
                action = int(goal_reference["action"][prefix + step + 1].argmax())
                probabilities = ctl.onehot(action, runtime.bundle["config"]["num_actions"])
            else:
                probabilities = control.probabilities(runtime, session.features, goals, remaining, target, mode)
                action = ctl.select_action(probabilities, manifest["design"]["execution_policy"], float(uniforms[step]))
            distributions.append(probabilities)
            actions.append(action)
            frames.append(len(session.rows) - 1)
            session.step(action)
            if session.done:
                break
        trajectory = session.arrays()
        trajectory["goal_features"] = np.stack([ctl.encode_goal(runtime, observation) for observation in session.rows])
        metrics = ctl.trial_metrics(trajectory, goals, target, prefix, goal_reference["telemetry"][-1])
        distances = ctl.distances(trajectory["goal_features"][prefix:], goals)
        start_margin = float(distances[0, 1 - target] - distances[0, target])
        control_events = session.events[prefix + 1:]
        row.update(metrics, execution_valid=True, issued_goal=assigned if mode in ("goal", "swapped_goal") else None,
                   goal_input="zero" if mode == "zero_goal" else "unused" if mode in ("no_goal", "base", "reference_replay") else "real_endpoint",
                   actual_steps=len(actions), end_reason="environment_done" if session.done else "fixed_budget",
                   task_success=any(event["success"] for event in control_events),
                   task_success_during_prefix=any(event["success"] for event in session.events[:prefix + 1]),
                   external_return=float(sum(event["reward"] for event in control_events)),
                   start_target_preference_margin=start_margin,
                   preference_improvement=metrics["target_preference_margin"] - start_margin,
                   end_distances_to_both_goals=distances[-1].tolist(),
                   issued_target_margin=float(distances[-1, 1 - assigned] - distances[-1, assigned]) if mode in ("goal", "swapped_goal") else None,
                   actions=actions, action_histogram=np.bincount(actions, minlength=runtime.bundle["config"]["num_actions"]).tolist())
        if mode == "reference_replay":
            replay = ctl.prefix_comparison(goal_reference, trajectory, manifest["pair_limits"]) if len(trajectory["image"]) == len(goal_reference["image"]) else dict(passed=False, reason="reference length differs")
            reference_events = legacy.read_json(benchmark_dir / case["directory"] / f"reference_{target}/events.json")
            expected = goal_reference["action"][prefix + 1:prefix + horizon + 1].argmax(-1).tolist()
            row.update(reference_replay_valid=bool(replay["passed"]), reference_replay_check=replay,
                       reference_macro_actions_equal=actions == expected[:len(actions)],
                       reference_native_actions_equal=repeated.native_actions_equal(
                           [event["native_actions"] for event in session.events[prefix + 1:]],
                           [event["native_actions"] for event in reference_events[prefix + 1:prefix + 1 + len(actions)]]),
                       reference_budget_completed=len(actions) == horizon)
            # Preserve future/native differences as calibration, not outcome-based exclusions.
            row["execution_valid"] = row["reference_macro_actions_equal"]
        trace = dict(model_id=runtime.policy.model_id, worker_version=runtime.policy.worker_version,
                     condition=mode, evaluated_target=target, issued_target=row["issued_goal"],
                     assigned_goal=goals[assigned].tolist() if row["issued_goal"] is not None else None,
                     execution_policy="recorded_actions" if mode == "reference_replay" else manifest["design"]["execution_policy"],
                     action_seed=action_seed, uniforms=[] if mode == "reference_replay" else uniforms[:len(actions)].tolist(),
                     probabilities=np.asarray(distributions).tolist(), action_ids=actions, frames=frames,
                     remaining=list(range(horizon, horizon - len(actions), -1)), distances_to_both_goals=distances.tolist())
        return row, trace
    finally:
        if session is not None:
            try:
                ctl.save_session(session, directory, baseline.write_video)
                if strict is not None:
                    baseline.write_json(directory / "prefix_comparison.json", strict)
            finally:
                session.close()
        gc.collect()


def evaluation_id(manifest):
    import goal_library as gl
    return gl.tensor_digest({}, {name: value for name, value in manifest.items() if name != "evaluation_id"})


def previous_rows(args, report, manifest):
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_control as control

    if not args.previous_eval_dir:
        return [], []
    directory = baseline.project_path(args.previous_eval_dir)
    record = legacy.read_json(directory / "report.json")
    saved = legacy.read_json(directory / "evaluation_manifest.json")
    legacy.accepted(record, "evaluate")
    gl.require(saved.get("format") == "ls_imagine_residual_control_evaluation_v1" and
               saved["evaluation_id"] == evaluation_id(saved) == record.get("evaluation_id") and
               saved["benchmark_id"] == manifest["benchmark_id"] and
               saved["input_identity"] == report.data["input_identity"] and saved["evaluation_code"] == code_identity() and
               saved["environment_fingerprint"] == report.data["environment_fingerprint"] and
               saved["execution_policy"] == manifest["design"]["execution_policy"], "已有评估的模型/代码/执行方式/计划身份不同")
    required = {"stage_execution", "comparable_blocks", "no_training", "source_inputs_unchanged"}
    gl.require(required <= {row["name"] for row in record["checks"] if row["level"] == "PASS"} and
               record.get("completed_trials") == saved["completed_trials"] and
               record.get("cumulative_evaluation_env_steps") == saved["cumulative_evaluation_env_steps"] and
               record.get("optimizer_updates") == 0, "前段缺少完整起点/执行/冻结/计数验收")
    rows = saved["rows"]
    gl.require(rows and saved["completed_trials"] == len(rows), "已有评估没有完整trial记录")
    control.pending_cells(manifest["design"], rows, args.blocks)
    origins = saved["source_evaluations"] + [dict(directory=str(directory), evaluation_id=saved["evaluation_id"],
        manifest_sha256=bc.file_hash(directory / "evaluation_manifest.json"), report_sha256=bc.file_hash(directory / "report.json"))]
    owners = {Path(item["directory"]).resolve() for item in origins}
    for row in rows:
        artifact = Path(row["artifact_dir"]).resolve()
        gl.require(artifact.parent in owners and artifact.name == row["directory"] and
                   row["directory"] == f"seed_{row['seed']}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}", "已有trial目录身份不同")
        for name in ("trajectory.npz", "events.json", "video.mp4", "metrics.json", "control_trace.json", "prefix_comparison.json"):
            path = artifact / name
            gl.require(str(path) in saved["artifacts"] and path.is_file() and bc.file_hash(path) == saved["artifacts"][str(path)], "已有真实trial文件改变或缺失")
        gl.require(legacy.read_json(artifact / "metrics.json") == row, "已有逐trial与汇总记录不同")
    for origin in saved["source_evaluations"]:
        root = Path(origin["directory"])
        gl.require(bc.file_hash(root / "evaluation_manifest.json") == origin["manifest_sha256"] and
                   bc.file_hash(root / "report.json") == origin["report_sha256"], "已有分段评估来源记录改变")
    report.data.update(previous_evaluation_dir=str(directory), inherited_trials=len(rows),
                       previous_evaluation_env_steps=saved["cumulative_evaluation_env_steps"])
    report.check("complete_block_continuation", "PASS", f"核对{len(rows)}个已有trial及真实轨迹SHA256；仅执行剩余计划，无重试或重跑")
    return rows, origins


def gallery(report, manifest, rows):
    import numpy as np
    from PIL import Image, ImageDraw
    import goal_control as ctl
    import goal_residual_control as control

    for seed in manifest["design"]["seeds"]:
        for repeat in range(manifest["design"]["repetitions"]):
            chosen = [row for row in rows if row["seed"] == seed and row["repeat"] == repeat]
            if not chosen:
                continue
            case = next(case for case in manifest["cases"] if case["seed"] == seed)
            with np.load(baseline.project_path(report.data["arguments"]["benchmark_dir"]) / case["directory"] / "goals.npz", allow_pickle=False) as data:
                targets = data["images"].copy()
            panel = Image.new("RGB", (680, 150 * (len(control.MODES) + 1)), "white")
            draw = ImageDraw.Draw(panel)
            for target in (0, 1):
                x = 240 + 220 * target
                draw.text((x, 3), f"real target {target}", fill="black")
                panel.paste(Image.fromarray(targets[target]).resize((128, 128)), (x, 20))
            eligible = len(chosen) == 12 and all(row["start_eligible"] and row["execution_valid"] for row in chosen)
            for index, mode in enumerate(control.MODES, 1):
                y = index * 150
                draw.text((3, y + 20), f"{mode} | repeat {repeat}", fill="black")
                for row in chosen:
                    if row["mode"] != mode:
                        continue
                    x = 240 + 220 * row["target"]
                    label = f"d={row['end_distance']:.4f} margin={row['target_preference_margin']:.4f}" if row["start_eligible"] else "START MISMATCH"
                    if not eligible:
                        label = "BLOCK EXCLUDED | " + label
                    draw.text((x, y + 3), label, fill="black")
                    panel.paste(Image.fromarray(ctl.read_trajectory(Path(row["artifact_dir"]))["image"][-1]).resize((128, 128)), (x, y + 20))
            panel.save(report.directory / f"outcomes_seed_{seed}_repeat_{repeat}.png")


def evaluate(args, report, cache, runtime):
    import numpy as np
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_residual_control as control

    directory, manifest = load_benchmark(args, report)
    rows, origins = previous_rows(args, report, manifest)
    pending = control.pending_cells(manifest["design"], rows, args.blocks)
    inherited = len(rows)
    goals_by_seed = {}
    for case in manifest["cases"]:
        with np.load(directory / case["directory"] / "goals.npz", allow_pickle=False) as data:
            goals = data["goals"].copy()
            gl.require(goals.shape == (2, cache.metadata["goal_dim"]) and np.isfinite(goals).all() and
                       np.allclose(np.linalg.norm(goals, axis=-1), 1, atol=1e-5), "目标向量尺寸/数值异常")
            for target in (0, 1):
                encoded = ctl.encode_goal(runtime, {"image": data["images"][target], "heatmap": data["heatmaps"][target]})
                gl.require(np.allclose(encoded, goals[target], atol=3e-5, rtol=3e-5), "真实终点目标重编码不同")
        goals_by_seed[case["seed"]] = goals
    baseline.video_preflight(report)
    baseline.write_json(report.directory / "schedule.json", manifest["design"])
    baseline.write_json(report.directory / "stage_schedule.json", dict(inherited_trials=inherited, cells=pending))
    # Do not duplicate weights/states/videos from earlier completed blocks.
    # A conservative reserve covers compact traces and ordinary videos;
    # arbitrary simulator log growth cannot be guaranteed in advance.
    bc.require_disk_space(report.directory, len(pending) * 16 * 1024**2 + 16 * 1024**2)
    report.check("storage_preflight", "PASS", f"检查本段{len(pending)}次轨迹/视频空间；不复制checkpoint、缓存或前段录像")
    report.check("control_groups", "PASS", "有目标/独立无目标修正/冻结底座/零目标/交换目标/参考动作六组；计划和执行方式提前固定")
    cases = {case["seed"]: case for case in manifest["cases"]}
    for cell in pending:
        report.require_writable()
        name = f"seed_{cell['seed']}_repeat_{cell['repeat']}_target_{cell['target']}_{cell['mode']}"
        output = report.directory / name
        row, trace = run_trial(report, runtime, directory, manifest, cases[cell["seed"]], goals_by_seed[cell["seed"]], cell, output)
        baseline.write_json(output / "metrics.json", row)
        baseline.write_json(output / "control_trace.json", trace)
        rows.append(row)
        t04.append_json(report.directory / "trials.jsonl", row)
        baseline.write_json(report.directory / "diagnostics.json", control.summarize(rows, manifest["design"]))
        report.data.update(completed_trials=len(rows), newly_completed_trials=len(rows) - inherited)
        report.save()
        message = f"d={row['end_distance']:.5f} margin={row['target_preference_margin']:.5f} steps={row['actual_steps']}" if row["start_eligible"] else "START MISMATCH; block excluded; no retry"
        print(f"[TRIAL {len(rows)}/{len(manifest['design']['schedule'])}] repeat={cell['repeat']} target={cell['target']} mode={cell['mode']} {message}", flush=True)
    summary = control.summarize(rows, manifest["design"])
    baseline.write_json(report.directory / "diagnostics.json", summary)
    # The complete cumulative table is separate from this invocation's
    # append-only trials.jsonl, avoiding implicit re-execution on continuation.
    baseline.write_json(report.directory / "cumulative_trials.json", rows)
    baseline.write_csv(report.directory / "trials.csv", [{name: row.get(name) for name in (
        "seed", "repeat", "target", "mode", "start_eligible", "execution_valid", "actual_steps", "task_success",
        "start_distance", "end_distance", "distance_improvement", "target_preference_margin", "preference_improvement",
        "reference_replay_valid", "reference_native_actions_equal", "artifact_dir")} for row in rows])
    gallery(report, manifest, rows)
    saved = dict(format="ls_imagine_residual_control_evaluation_v1", benchmark_id=manifest["benchmark_id"],
        input_identity=report.data["input_identity"], evaluation_code=code_identity(),
        environment_fingerprint=report.data["environment_fingerprint"], execution_policy=manifest["design"]["execution_policy"],
        completed_trials=len(rows), rows=rows, source_evaluations=origins,
        cumulative_evaluation_env_steps=report.data.get("previous_evaluation_env_steps", 0) + report.data["evaluation_env_steps"],
        artifacts={str(Path(row["artifact_dir"]) / name): bc.file_hash(Path(row["artifact_dir"]) / name)
                   for row in rows for name in ("trajectory.npz", "events.json", "video.mp4", "metrics.json", "control_trace.json", "prefix_comparison.json")})
    saved["evaluation_id"] = evaluation_id(saved)
    baseline.write_json(report.directory / "evaluation_manifest.json", saved)
    report.data.update(evaluation_id=saved["evaluation_id"], planned_trials=summary["planned_trials"],
                       completed_trials=len(rows), eligible_trials=summary["eligible_trials"],
                       newly_completed_trials=len(rows) - inherited,
                       cumulative_evaluation_env_steps=saved["cumulative_evaluation_env_steps"],
                       task_success_evaluated=True, full_design_completed=len(rows) == summary["planned_trials"])
    report.check("stage_execution", "PASS", f"按预声明顺序完成本段{len(pending)}个trial，累计{len(rows)}/{summary['planned_trials']}；没有重试或按终点补样")
    report.check("comparable_blocks", "PASS" if summary["eligible_trials"] == len(rows) else "FAIL",
                 f"完整可比块纳入{summary['eligible_trials']}/{len(rows)}；失败轨迹原样保留，不自动批准继续")
    report.check("calibration", "PASS", "参考重放原生动作/终点/严格状态差异保留为波动诊断；不按其未来结果删目标组")
    report.check("design_completion", "PASS" if report.data["full_design_completed"] else "WARN",
                 "全部预声明重复已完成" if report.data["full_design_completed"] else "这是首段接口验收；尚未执行完全部重复，后续只续跑剩余完整块")
    report.check("behavior_acceptance", "WARN", "工程PASS不等于控制效果通过。对照优势、目标切换方向、参考波动及视频需分析；仍在T05，不能自动推进T06")


def source_paths(args):
    import goal_residual_worker as residual
    import goal_library as gl

    cache_dir = baseline.project_path(args.cache_dir)
    protected, files, source = legacy.source_paths(cache_dir, args)
    _, dependencies = residual.read_checkpoint(baseline.project_path(args.checkpoint))
    for path in dependencies.values():
        protected.append(path.parent)
        files.append(path)
    for name in ("residual_verify_dir", "source_prepare_dir", "previous_eval_dir"):
        value = getattr(args, name, None)
        if not value:
            continue
        directory = baseline.project_path(value)
        protected.append(directory)
        files.append(directory / "report.json")
        if name == "source_prepare_dir":
            files += [path for path in directory.rglob("*") if path.is_file() and path.suffix in (".json", ".npz", ".mp4", ".png")]
        if name == "previous_eval_dir":
            files.append(directory / "evaluation_manifest.json")
            saved = legacy.read_json(directory / "evaluation_manifest.json")
            for row in saved["rows"]:
                artifact = Path(row["artifact_dir"]).resolve()
                protected.append(artifact)
                files += [artifact / item for item in ("trajectory.npz", "events.json", "video.mp4", "metrics.json", "control_trace.json", "prefix_comparison.json")]
            for origin in saved["source_evaluations"]:
                root = Path(origin["directory"]).resolve()
                protected.append(root)
                files.extend([root / "evaluation_manifest.json", root / "report.json"])
    gl.require(all(path.is_file() for path in files), "输入产物或依赖缺失，请保留服务器缓存/底座/验收/参考与已有评估")
    return list(dict.fromkeys(path.resolve() for path in protected)), list(dict.fromkeys(path.resolve() for path in files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--residual-verify-dir", required=True)
    checking.add_argument("--prefix-steps", type=int, default=32)
    adoption = commands.add_parser("adopt")
    adoption.add_argument("--source-prepare-dir", required=True)
    adoption.add_argument("--repetitions", type=int, default=3)
    adoption.add_argument("--randomization-seed", type=int, default=0)
    adoption.add_argument("--execution-policy", choices=("mode", "sample"), default="mode")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--benchmark-dir", required=True)
    evaluation.add_argument("--previous-eval-dir")
    evaluation.add_argument("--blocks", type=int, default=1, help="本段新执行完整块数；每块12trial，不是累计值")
    for command in (checking, adoption, evaluation):
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output-dir")
        command.add_argument("--device", default="cuda:0")
    for command in (adoption, evaluation):
        command.add_argument("--check-dir", required=True)
    args = parser.parse_args()
    if args.command == "check" and args.prefix_steps < 1:
        parser.error("prefix-steps至少1")
    if args.command == "adopt" and (args.repetitions < 3 or not 0 <= args.randomization_seed < 2**31 - 10000):
        parser.error("至少3次重复，随机种子必须是合法非负整数")
    if args.command == "evaluate" and (args.blocks < 1 or os.environ.get("MINEDOJO_HEADLESS") != "1"):
        parser.error("blocks至少1；启动MineDojo必须显式使用MINEDOJO_HEADLESS=1")
    try:
        protected, files, source = source_paths(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "residual_control_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、缓存、回放、参考及前段评估")
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
        cache, runtime = load_inputs(args, report)
        versions = {name: parameter._version for name, parameter in runtime.named_parameters()}
        digest = gl.tensor_digest(runtime.state_dict(), {})
        with torch.no_grad():
            {"check": check, "adopt": adopt, "evaluate": evaluate}[args.command](args, report, cache, runtime)
        gl.require(versions == {name: parameter._version for name, parameter in runtime.named_parameters()} and
                   gl.tensor_digest(runtime.state_dict(), {}) == digest and
                   all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.parameters()), "真实评估改变冻结模型或产生梯度")
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
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型/回放/参考/前段产物大小及修改时间未变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
