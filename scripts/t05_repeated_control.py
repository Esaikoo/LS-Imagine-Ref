"""T05 repeated behavioral diagnostic; only evaluate starts MineDojo.

check validates offline interfaces. adopt uses existing real references
from a visual-preprocessing-correct prepare, including a failed strict
pair. evaluate runs a predeclared randomized repeated schedule. It never
claims identical RSSM states, trains models, or approves scientific gains.
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
import t05_goal_control as t05

FORMAT = "ls_imagine_repeated_goal_control_v1"


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        label = {"check": "T05_REPEATED_CHECK", "adopt": "T05_REPEATED_ADOPT", "evaluate": "T05_REPEATED_EVAL"}[self.data["command"]]
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return {name: bc.file_hash(ROOT / name) for name in ("goal_control_stats.py", "scripts/t05_repeated_control.py")}


def check(args, report, cache, model):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl

    t05.check(args, report, cache, model)
    prefix = {"image": np.zeros((2, 64, 64, 3), np.uint8), "heatmap": np.zeros((2, 64, 64), np.uint8),
              "features": np.ones((2, 8), np.float32), "obs_reward": np.zeros((2, 1), np.float32),
              "action": np.array([[0, 0], [1, 0]], np.float32),
              "is_first": np.array([True, False]), "is_last": np.zeros(2, bool), "is_terminal": np.zeros(2, bool),
              "telemetry": [{"pose": dict(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0),
                             "inventory": {}, "health": 20.0} for _ in range(2)]}
    drift = copy.deepcopy(prefix)
    drift["heatmap"] += 1
    drift["features"] += 10
    comparison = ctl.prefix_comparison(prefix, drift)
    gl.require(not comparison["passed"] and stats.start_check(comparison, True)["passed"] and
               not stats.start_check(comparison, True)["same_hidden_state"], "重复协议错误宣称隐藏状态相同")
    for field in ("rgb", "position", "action", "inventory", "reward"):
        bad = copy.deepcopy(drift)
        if field == "rgb":
            bad["image"] += 32
        elif field == "position":
            bad["telemetry"][1]["pose"]["x"] = 10.0
        elif field == "action":
            bad["action"][1] = [0, 1]
        elif field == "inventory":
            bad["telemetry"][1]["inventory"] = {"log": 1}
        else:
            bad["obs_reward"][1] = 1.0
        gl.require(not stats.start_check(ctl.prefix_comparison(prefix, bad), True)["passed"], f"{field} 不一致未被拒绝")
    gl.require(not stats.start_check(comparison, False)["passed"], "底层动作不一致未被拒绝")
    saved_native = [{"camera": [0.0, 10.0], "attack": 1}]
    live_native = [{"camera": np.array([0.0, 10.0]), "attack": np.array(1)}]
    gl.require(stats.native_actions_equal(saved_native, live_native) and
               not stats.native_actions_equal(saved_native, [{"camera": np.array([0.0, -10.0]), "attack": np.array(1)}]),
               "底层动作JSON/NumPy表示比较异常")
    design = {"seeds": [0], "repetitions": 3, "randomization_seed": 0, "schedule": stats.schedule([0], 3, 0)}
    gl.require(design["schedule"] == stats.schedule([0], 3, 0) and len(design["schedule"]) == 30 and
               len({(c["seed"], c["repeat"], c["target"], c["mode"]) for c in design["schedule"]}) == 30,
               "预先随机顺序或重复覆盖异常")
    rows = [dict(cell, start_eligible=True, execution_valid=True, task_success=False,
                 start_distance=0.2, end_distance=0.1, distance_improvement=0.1,
                 relative_distance_improvement=0.5, target_preference_margin=0.05) for cell in design["schedule"]]
    gl.require(stats.summarize(rows, design)["eligible_trials"] == 30, "完整重复块被误删")
    rows[0]["start_eligible"] = False
    gl.require(stats.summarize(rows, design)["eligible_trials"] == 20 and
               stats.summarize(rows[:9], design)["eligible_trials"] == 0, "未整体排除不匹配/不完整重复块")
    report.data["comparison_protocol"] = stats.PROTOCOL
    report.check("repeated_protocol", "PASS", "严格配对仍拒绝状态/heatmap漂移；新协议保留物理/RGB/底层动作边界，必须使用预先随机的完整重复块")
    report.check("probe_scope", "PASS", "合成边界与汇总探针仅在内存，无环境交互或训练数据写入")


def prefix_events(directory, count):
    events = t05.read_json(directory / "events.json")
    if len(events) < count or not all("native_actions" in e for e in events[:count]):
        raise ValueError("真实参考缺少完整底层动作事件")
    return events[:count]


def goal_probe(model, references, goals, prefix_steps, horizon):
    import numpy as np
    import torch
    import goal_control as ctl
    import goal_control_stats as stats

    rows = []
    device = next(model.parameters()).device
    for branch, reference in enumerate(references):
        for step in range(horizon):
            frame, remaining = prefix_steps + step, horizon - step
            features = torch.as_tensor(reference["features"][frame:frame + 1], device=device)
            left = ctl.probabilities(model, features, goals[0], remaining, "goal")
            right = ctl.probabilities(model, features, goals[1], remaining, "goal")
            zero = ctl.probabilities(model, features, goals[0], remaining, "zero_goal")
            rows.append({"reference_branch": branch, "frame": frame, "remaining": remaining, "same_state": True,
                "goal_0_probabilities": left.tolist(), "goal_1_probabilities": right.tolist(), "zero_goal_probabilities": zero.tolist(),
                "goal_switch_probability_l1": float(np.abs(left - right).sum()),
                "goal_0_vs_zero_l1": float(np.abs(left - zero).sum()),
                "goal_1_vs_zero_l1": float(np.abs(right - zero).sum()),
                "goal_0_mode_action": int(left.argmax()), "goal_1_mode_action": int(right.argmax())})
    return {"scope": "each pair shares one cached causal state; action sensitivity only, not successful control",
            "summary": {"states": len(rows),
                "goal_switch_l1": stats.statistics([r["goal_switch_probability_l1"] for r in rows]),
                "mode_action_change_fraction": float(np.mean([r["goal_0_mode_action"] != r["goal_1_mode_action"] for r in rows]))},
            "per_state": rows}


def adopt(args, report, cache, model):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl

    source = baseline.project_path(args.source_prepare_dir)
    original = t05.read_json(source / "report.json")
    gl.require(original.get("command") == "prepare" and original.get("environment_protocol") == ctl.ENVIRONMENT_PROTOCOL and
               original.get("input_identity") == report.data["input_identity"] and original.get("model_id") == report.data["model_id"],
               "需要当前视觉预处理协议、相同模型/缓存的真实 prepare 产物；旧 HUD 输入不能复用")
    gl.require(original.get("source_inputs_before") and any(c["name"] == "source_inputs_unchanged" and c["level"] == "PASS"
               for c in original.get("checks", [])), "原 prepare 没有源文件未变验收记录")
    for path, signature in original["source_inputs_before"].items():
        gl.require(baseline.file_signature(Path(path)) == signature, "原 prepare 的冻结模型、缓存或验收依赖已变化")
    seeds = original["arguments"]["seeds"]
    prefix_steps, horizon = original["arguments"]["prefix_steps"], original["arguments"]["horizon"]
    minimum = original["arguments"]["min_goal_distance"]
    gl.require(isinstance(seeds, list) and seeds and all(type(s) is int and 0 <= s < 2**31 - 10000 for s in seeds) and
               len(set(seeds)) == len(seeds) and type(prefix_steps) is int and prefix_steps >= 0 and
               type(horizon) is int and 1 <= horizon <= model.bundle["horizon"] and 0 < minimum <= 2,
               "原参考种子、长度或目标区分参数异常")
    design = {"seeds": seeds, "repetitions": args.repetitions, "randomization_seed": args.randomization_seed,
              "modes": list(stats.MODES), "schedule": stats.schedule(seeds, args.repetitions, args.randomization_seed)}
    manifest = {"format": FORMAT, "comparison_protocol": stats.PROTOCOL, "protocol": ctl.ENVIRONMENT_PROTOCOL,
                "visual_preprocessing_policy": ctl.VISUAL_PREPROCESSING_POLICY, "world_seed_policy": ctl.WORLD_SEED_POLICY,
                "environment_fingerprint": ctl.environment_fingerprint(ROOT), "evaluation_code": code_identity(),
                "reference_generation_environment_fingerprint": original.get("environment_fingerprint"),
                "source_prepare": str(source), "source_report_sha256": bc.file_hash(source / "report.json"),
                "input_identity": report.data["input_identity"], "model_id": report.data["model_id"],
                "prefix_steps": prefix_steps, "horizon": horizon, "pair_limits": ctl.PAIR_LIMITS,
                "min_goal_distance": minimum, "design": design, "cases": [],
                "same_hidden_state": False, "controlled_reachability_verified": False}
    if original.get("environment_fingerprint") is None:
        report.check("reference_provenance", "WARN", "旧失败报告未持久化完整环境指纹；核对已记录视觉协议/模型/底层动作及真实历史。当前环境指纹固定用于重复执行，原参考仅作为真实视觉目标与重放诊断")
    else:
        gl.require(original["environment_fingerprint"] == manifest["environment_fingerprint"], "原参考环境代码/权重指纹已变化，不能复用")
    for seed in seeds:
        source_case = source / f"case_{seed}"
        scenario = t05.read_json(source_case / "scenario.json")
        ctl.validate_scenario(scenario)
        gl.require(scenario["seed"] == seed and scenario.get("start_position") is not None, "原参考缺少固定起点")
        visual = t05.read_json(source_case / "visual_preprocessing.json")
        for item in visual.values():
            gl.require(item["policy"] == ctl.VISUAL_PREPROCESSING_POLICY and not item["screenshot_file_output"], "原参考视觉处理不匹配")
        gl.require(set(visual) == {"reference_0", "reference_1"} and visual["reference_0"] == visual["reference_1"],
                   "两参考分支视觉处理设置不同")
        task = model.bundle["config"]["task"].partition("_")[2]
        catalogue = baseline.yaml_read(ROOT / "envs/tasks/task_specs.yaml")
        expected_specs = ctl.visual_preprocessing_specs(catalogue.get(task, {}))
        expected_hud = bool(expected_specs) and not bool(expected_specs.get("HUD", False))
        gl.require(visual["reference_0"]["screenshot_specs"] == expected_specs and
                   visual["reference_0"]["screenshot_wrapper"] == bool(expected_specs) and
                   visual["reference_0"]["remove_hud"] == expected_hud, "原参考不匹配当前任务 HUD 处理")
        references = [ctl.read_trajectory(source_case / f"reference_{branch}") for branch in (0, 1)]
        gl.require(all(len(r["image"]) == prefix_steps + horizon + 1 for r in references), "原参考未完成相同 horizon；不能静默改变预算")
        for branch, reference in enumerate(references):
            recovered = model.state_encoder.rollout(reference, len(reference["image"]) - 1).cpu().numpy()
            gl.require(recovered.shape == reference["features"].shape and
                       np.allclose(recovered, reference["features"], atol=3e-5, rtol=3e-5),
                       "原参考状态不符合当前冻结 WM 的因果状态约定")
        report.check(f"real_state_reencode_{seed}", "PASS", "从两段真实 reset 历史重算状态，与保存的因果状态一致；没有环境交互")
        prefixes = [{key: value[:prefix_steps + 1] for key, value in ref.items()} for ref in references]
        strict = ctl.prefix_comparison(*prefixes)
        events = [prefix_events(source_case / f"reference_{branch}", prefix_steps + 1) for branch in (0, 1)]
        native_equal = stats.native_actions_equal([e["native_actions"] for e in events[0]], [e["native_actions"] for e in events[1]])
        physical = stats.start_check(strict, native_equal)
        gl.require(physical["passed"], f"seed={seed} 原参考物理/RGB/动作仍不可比：{physical['failed_checks']}；不重试或丢弃此种子")
        goals = np.stack([ctl.encode_goal(model, {key: ref[key][-1] for key in ("image", "heatmap")}) for ref in references])
        current = np.stack([ctl.encode_goal(model, {key: ref[key][prefix_steps] for key in ("image", "heatmap")}) for ref in references])
        separation = float(ctl.distances(goals[:1], goals[1:])[0, 0])
        initial = ctl.distances(current, goals)
        case_dir = report.directory / f"case_{seed}"
        case_dir.mkdir()
        baseline.write_json(case_dir / "goal_diagnostics.json", {"goal_separation": separation, "initial_distances": initial.tolist(), "minimum": minimum})
        gl.require(separation >= minimum and bool(np.all(initial >= minimum)), "真实目标过近或起点已接近目标；保留诊断，不能据此检验目标切换")
        for name in ("scenario.json", "spawn.json", "visual_preprocessing.json"):
            shutil.copyfile(source_case / name, case_dir / name)
        for branch in (0, 1):
            target_dir = case_dir / f"reference_{branch}"
            target_dir.mkdir()
            for name in ("trajectory.npz", "events.json", "video.mp4"):
                shutil.copyfile(source_case / f"reference_{branch}" / name, target_dir / name)
        baseline.write_json(case_dir / "prefix_comparison.json", strict)
        baseline.write_json(case_dir / "start_comparison.json", physical)
        probe = goal_probe(model, references, goals, prefix_steps, horizon)
        baseline.write_json(case_dir / "same_state_goal_probe.json", probe)
        print(f"[GOAL PROBE] seed={seed} states={probe['summary']['states']} "
              f"mean_l1={probe['summary']['goal_switch_l1']['mean']:.6f} "
              f"mode_change_fraction={probe['summary']['mode_action_change_fraction']:.4f}", flush=True)
        np.savez_compressed(case_dir / "goals.npz", goals=goals,
            images=np.stack([r["image"][-1] for r in references]), heatmaps=np.stack([r["heatmap"][-1] for r in references]))
        classes = model.library.assign(torch.as_tensor(goals, device=args.device)).cpu().tolist()
        files = sorted(p for p in case_dir.rglob("*") if p.is_file())
        manifest["cases"].append({"seed": seed, "directory": case_dir.name, "scenario": scenario, "goal_classes": classes,
            "prefix_actions": references[0]["action"][1:prefix_steps + 1].argmax(-1).tolist(),
            "goal_separation": separation, "initial_distances": initial.tolist(),
            "files": {str(p.relative_to(report.directory)): bc.file_hash(p) for p in files}})
        print(f"[ADOPT] seed={seed} start_comparable=1 strict_pair={int(strict['passed'])} goal_separation={separation:.5f}", flush=True)
    manifest["benchmark_id"] = t05.benchmark_id(manifest)
    baseline.write_json(report.directory / "benchmark.json", manifest)
    report.data.update(benchmark_id=manifest["benchmark_id"], adopted_reference_env_steps=original.get("new_env_steps"),
                       comparison_protocol=stats.PROTOCOL, same_hidden_state=False, planned_trials=len(design["schedule"]),
                       maximum_new_eval_steps=len(design["schedule"]) * (prefix_steps + horizon))
    report.check("real_reference_adoption", "PASS", "只读并复制完整真实参考；重新编码目标、核对物理/RGB/底层动作；严格失败原值保留，没有重跑环境")
    report.check("predeclared_design", "PASS", f"已保存随机顺序及 {args.repetitions} 次重复，共 {len(design['schedule'])} 个 trial；无隐藏状态相等声明")
    report.check("scope", "WARN", "采用新重复评估设计，原严格 prepare 仍未通过；同状态目标探针只检查动作敏感度，不证明控制有效")


def load_benchmark(args, report):
    import goal_bc as bc
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl

    directory = baseline.project_path(args.benchmark_dir)
    manifest = t05.read_json(directory / "benchmark.json")
    record = t05.read_json(directory / "report.json")
    t05.accepted(record, "adopt")
    gl.require(manifest["format"] == FORMAT and manifest["comparison_protocol"] == stats.PROTOCOL and
               manifest["protocol"] == ctl.ENVIRONMENT_PROTOCOL and manifest["pair_limits"] == ctl.PAIR_LIMITS and
               manifest["environment_fingerprint"] == ctl.environment_fingerprint(ROOT) and
               manifest["evaluation_code"] == code_identity() and manifest["input_identity"] == report.data["input_identity"] and
               manifest["model_id"] == report.data["model_id"] and
               manifest["benchmark_id"] == t05.benchmark_id(manifest) == record["benchmark_id"], "重复 benchmark、模型或执行代码不匹配")
    design = manifest["design"]
    gl.require(design["repetitions"] >= 3 and design["modes"] == list(stats.MODES) and
               design["schedule"] == stats.schedule(design["seeds"], design["repetitions"], design["randomization_seed"]) and
               [c["seed"] for c in manifest["cases"]] == design["seeds"], "随机设计或种子覆盖被修改")
    for case in manifest["cases"]:
        ctl.validate_scenario(case["scenario"])
        for relative, digest in case["files"].items():
            path = (directory / relative).resolve()
            gl.require(directory in path.parents and bc.file_hash(path) == digest, "真实参考或目标文件内容发生变化")
    report.data.update(benchmark_id=manifest["benchmark_id"], comparison_protocol=stats.PROTOCOL, same_hidden_state=False)
    report.check("benchmark_identity", "PASS", "真实参考文件、随机顺序、冻结模型、环境及评估代码指纹一致")
    return directory, manifest


def gallery(report, benchmark_dir, manifest, rows):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    from PIL import Image, ImageDraw

    summary = stats.summarize(rows, manifest["design"])
    eligible_blocks = {(b["seed"], b["repeat"]) for b in summary["blocks"] if b["eligible"]}
    for case in manifest["cases"]:
        with np.load(benchmark_dir / case["directory"] / "goals.npz", allow_pickle=False) as data:
            targets = data["images"].copy()
        for repeat in range(manifest["design"]["repetitions"]):
            panel = Image.new("RGB", (600, 156 * (len(stats.MODES) + 1)), "white")
            draw = ImageDraw.Draw(panel)
            for target in (0, 1):
                x = 200 * (target + 1)
                draw.text((x + 2, 3), f"real target {target}", fill="black")
                panel.paste(Image.fromarray(targets[target]).resize((128, 128)), (x, 24))
            for index, mode in enumerate(stats.MODES, 1):
                y = index * 156
                draw.text((3, y + 24), f"{mode} | repeat {repeat}", fill="black")
                for row in rows:
                    if row["seed"] != case["seed"] or row["repeat"] != repeat or row["mode"] != mode:
                        continue
                    x = 200 * (row["target"] + 1)
                    trajectory = ctl.read_trajectory(report.directory / row["directory"])
                    label = f"end d={row['end_distance']:.4f}" if row["start_eligible"] else "START MISMATCH"
                    if row["start_eligible"] and (row["seed"], row["repeat"]) not in eligible_blocks:
                        label = "BLOCK EXCLUDED"
                    draw.text((x + 2, y + 3), label, fill="black")
                    panel.paste(Image.fromarray(trajectory["image"][-1]).resize((128, 128)), (x, y + 24))
            panel.save(report.directory / f"outcomes_seed_{case['seed']}_repeat_{repeat}.png")


def evaluate(args, report, cache, model):
    import numpy as np
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl

    directory, manifest = load_benchmark(args, report)
    baseline.video_preflight(report)
    cases = {case["seed"]: case for case in manifest["cases"]}
    goals_by_seed = {}
    for seed, case in cases.items():
        with np.load(directory / case["directory"] / "goals.npz", allow_pickle=False) as data:
            goals = data["goals"].copy()
            for target in (0, 1):
                encoded = ctl.encode_goal(model, {"image": data["images"][target], "heatmap": data["heatmaps"][target]})
                gl.require(np.allclose(encoded, goals[target], atol=3e-5, rtol=3e-5), "真实目标重编码不同")
        goals_by_seed[seed] = goals
    rows = []
    baseline.write_json(report.directory / "schedule.json", manifest["design"])
    report.check("no_goal_control", "WARN", "包含原actor及零/交换目标消融；未包含独立no_goal BC，同一世界重复仅用于波动诊断")
    for cell in manifest["design"]["schedule"]:
        seed, repeat, target, mode = (cell[key] for key in ("seed", "repeat", "target", "mode"))
        trial_args = copy.copy(args)
        trial_args.repeat = repeat
        # A fixed draw sequence shared across all cells in this repeat.
        # Session's world seed is unchanged; this only drives action sampling.
        trial_args.action_seed = (manifest["design"]["randomization_seed"] + repeat * 1009) % (2**31 - 10000)
        output = report.directory / f"seed_{seed}_repeat_{repeat}_target_{target}_{mode}"
        result = t05.run_trial(trial_args, report, model, cases[seed], manifest, goals_by_seed[seed], target, mode, output)
        if isinstance(result, tuple):
            row, trace = result
            baseline.write_json(output / "control_trace.json", trace)
            print(f"[REPEAT] seed={seed} repeat={repeat} target={target} mode={mode} end_distance={row['end_distance']:.5f} steps={row['actual_steps']}", flush=True)
        else:
            row = result
            print(f"[START MISMATCH] seed={seed} repeat={repeat} target={target} mode={mode}; entire block excluded, no retry", flush=True)
        row.update(repeat=repeat, comparison_protocol=stats.PROTOCOL, same_hidden_state=False)
        rows.append(row)
        baseline.write_json(output / "metrics.json", row)
        t05.t04.append_json(report.directory / "trials.jsonl", row)
        baseline.write_json(report.directory / "diagnostics.json", stats.summarize(rows, manifest["design"]))
    diagnostics = stats.summarize(rows, manifest["design"])
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    baseline.write_csv(report.directory / "trials.csv", [{key: row.get(key) for key in
        ("seed", "repeat", "target", "mode", "start_eligible", "execution_valid", "actual_steps", "task_success", "start_distance",
         "end_distance", "distance_improvement", "target_preference_margin", "reference_replay_valid")} for row in rows])
    gallery(report, directory, manifest, rows)
    complete = len(rows) == len(manifest["design"]["schedule"])
    report.data.update(planned_trials=len(manifest["design"]["schedule"]), completed_trials=len(rows),
                       eligible_trials=diagnostics["eligible_trials"], same_hidden_state=False,
                       execution_policy=args.execution_policy)
    report.check("repeated_execution", "PASS" if complete else "FAIL", "按预先顺序执行全部重复，没有失败重试、补样或按终点表现选择样本")
    report.check("comparable_blocks", "PASS" if diagnostics["eligible_trials"] == len(rows) else "FAIL",
                 f"完整可比重复块纳入 {diagnostics['eligible_trials']}/{len(rows)} 个trial；原始失败数据保留")
    report.check("calibration", "PASS", "参考动作重放的真实终点波动已保存；其heatmap/RSSM严格复现失败不删除结果，也不声称参考完全到达")
    report.check("behavior_acceptance", "WARN", "这是重复行为诊断的工程验收。查看同状态目标敏感度、参考波动、对照优势和视频；单世界3次重复不批准T06或论文结论")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--t04-verify-dir", required=True)
    check_parser.add_argument("--prefix-steps", type=int, default=32)
    adopt_parser = commands.add_parser("adopt")
    adopt_parser.add_argument("--check-dir", required=True)
    adopt_parser.add_argument("--source-prepare-dir", required=True)
    adopt_parser.add_argument("--repetitions", type=int, default=3)
    adopt_parser.add_argument("--randomization-seed", type=int, default=0)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--check-dir", required=True)
    eval_parser.add_argument("--benchmark-dir", required=True)
    eval_parser.add_argument("--execution-policy", choices=("mode", "sample"), default="mode")
    for command in (check_parser, adopt_parser, eval_parser):
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
    args = parser.parse_args()
    if args.command == "check" and args.prefix_steps < 1:
        parser.error("prefix-steps 至少1")
    if args.command == "adopt" and (args.repetitions < 3 or not 0 <= args.randomization_seed < 2**31 - 10000):
        parser.error("至少3次重复，随机顺序种子必须为合法非负整数")
    if args.command == "evaluate" and os.environ.get("MINEDOJO_HEADLESS") != "1":
        parser.error("启动 MineDojo 必须显式使用 MINEDOJO_HEADLESS=1")
    try:
        protected, files, source_metadata = t05.source_paths(baseline.project_path(args.cache_dir), args)
        if args.command == "adopt":
            source = baseline.project_path(args.source_prepare_dir)
            protected.append(source)
            files.extend(path for path in source.rglob("*") if path.is_file() and path.suffix in (".json", ".npz", ".mp4"))
            original = t05.read_json(source / "report.json")
            for path in original.get("source_inputs_before", {}):
                files.append(Path(path))
                protected.append(Path(path).parent)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / ("repeated_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、回放、缓存和输入产物")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    before, rng, model = None, None, None
    try:
        import torch
        import goal_bc as bc
        import goal_control as ctl
        import goal_control_stats as stats
        import goal_library as gl

        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA 不可用")
            torch.cuda.set_device(device)
        else:
            gl.require(args.command != "evaluate", "真实 MineCLIP 路径需要 CUDA")
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        for entry in source_metadata["episodes"]:
            gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "原回放大小/修改时间变化")
        gl.require(baseline.file_signature(Path(source_metadata["checkpoint"]["path"])) == source_metadata["checkpoint"], "原初始化模型变化")
        if args.command != "check":
            checked = t05.read_json(baseline.project_path(args.check_dir) / "report.json")
            gl.require(checked.get("comparison_protocol") == stats.PROTOCOL, "请先运行新的 repeated check；旧严格check不替代新协议验收")
        rng = bc.capture_rng(args.device)
        cache, model = t05.load_inputs(args, report)
        versions = {id(p): p._version for p in model.parameters()}
        model_id = ctl.model_identity(model)
        {"check": check, "adopt": adopt, "evaluate": evaluate}[args.command](args, report, cache, model)
        gl.require(versions == {id(p): p._version for p in model.parameters()} and ctl.model_identity(model) == model_id,
                   "重复诊断修改了冻结模型")
        report.check("no_training", "PASS", f"optimizer_updates=0；参数未变；新增真实环境步={report.data['new_env_steps']}")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型/回放/参考产物大小及修改时间未变；只写独立目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        del model
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
