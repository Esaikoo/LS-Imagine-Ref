"""Analyze an accepted T05 random-control run offline; no new trials/training.

For every factual pre-action state, hold history/state/remaining fixed and
query goal0, goal1 and independent no-goal controls. Do not instantiate an
environment, MineCLIP, RSSM, candidate model or optimizer.
"""

import argparse
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
import t05_goal_control as legacy


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_ACTION_CHOICE_DIAGNOSE; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def passed_checks(record, names):
    return set(names) <= {row["name"] for row in record["checks"] if row["level"] == "PASS"}


def historical_inputs(args):
    """Bind to the completed experiment without rerunning its old check."""
    import goal_bc as bc
    import goal_library as gl
    import goal_random_control as random_control
    import goal_residual_worker as residual

    directory = baseline.project_path(args.eval_dir)
    record = legacy.read_json(directory / "report.json")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    legacy.accepted(record, "evaluate")
    plan = manifest["design"]
    gl.require(manifest.get("format") == random_control.EVALUATION_FORMAT and
               manifest.get("comparison_protocol") == random_control.PROTOCOL and
               plan.get("format") == random_control.FORMAT and
               plan.get("comparison_protocol") == random_control.PROTOCOL and
               manifest.get("evaluation_id") == record.get("evaluation_id") == gl.tensor_digest({}, {
                   key: value for key, value in manifest.items() if key != "evaluation_id"}) and
               manifest.get("design_id") == record.get("design_id") == plan.get("design_id") == gl.tensor_digest({}, {
                   key: value for key, value in plan.items() if key != "design_id"}),
               "需要完整已通过的独立起点评估及原内容身份；不能用旧FAIL或其它协议")
    random_control.validate_design(plan["design"])
    rows, schedule = manifest["rows"], plan["design"]["schedule"]
    gl.require(record.get("full_design_completed") is True and len(rows) == len(schedule) == record.get("attempted_trials") ==
               record.get("planned_trials") == record.get("execution_valid_trials") and
               all(row.get("execution_valid") is True for row in rows) and
               [random_control.key(row) for row in rows] == [random_control.key(cell) for cell in schedule] and
               passed_checks(record, ("planned_attempts", "real_control_interface", "no_training", "source_inputs_unchanged")),
               "需要全部计划完成且全部接口有效；不分析挑选后的子集")
    gl.require(record.get("input_identity") == manifest.get("input_identity") == plan.get("input_identity") and
               legacy.read_json(directory / "design.json") == plan and
               type(plan["horizon"]) is int and 1 <= plan["horizon"] <= plan["input_identity"]["input_identity"]["horizon"] and
               plan["control_start_frame"] == len(plan["start_actions"]), "历史模型/计划或控制边界不同")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(checked.get("design_id") == plan["design_id"] and checked.get("input_identity") == plan["input_identity"] and
               checked.get("evaluation_code") == plan["evaluation_code"] and
               legacy.read_json(checked_dir / "design.json") == plan and
               passed_checks(checked, ("fixed_visual_targets", "causal_execution_interface", "predeclared_design", "no_training")) and
               checked.get("targets_sha256") == bc.file_hash(checked_dir / "targets.npz"), "历史check或固定目标改变")
    # New analysis files are deliberately outside the old evaluator's identity.
    # A new diagnose entry point must not invalidate a completed experiment.
    for name, digest in plan["evaluation_code"].items():
        path = (ROOT / name).resolve()
        gl.require(ROOT in path.parents and path.is_file() and bc.file_hash(path) == digest,
                   f"历史推理代码改变：{name}；不能把新代码前向冒充原模型前向")
    checkpoint = baseline.project_path(args.checkpoint)
    gl.require(baseline.file_signature(checkpoint) == plan["input_identity"]["checkpoint"] and
               bc.file_hash(checkpoint) == plan["input_identity"]["checkpoint_sha256"], "指定模型不是本次真实控制模型")
    payload, dependencies = residual.read_checkpoint(checkpoint)
    verify_dir = baseline.project_path(args.residual_verify_dir)
    verify = legacy.read_json(verify_dir / "report.json")
    legacy.accepted(verify, "verify")
    gl.require(verify.get("architecture") == residual.ARCHITECTURE and
               verify.get("input_identity") == plan["input_identity"]["input_identity"] and
               verify.get("counters") == payload["counters"] == plan["input_identity"]["counters"] and
               verify.get("model_id") == plan["input_identity"]["model_id"] and
               verify.get("worker_version") == plan["input_identity"]["worker_version"] and
               verify.get("source_inputs_before", {}).get(str(checkpoint)) == baseline.file_signature(checkpoint) and
               payload.get("dependencies") == plan["input_identity"]["dependencies"] and
               passed_checks(verify, ("strict_load", "strict_inference_load", "incremental_causal_state",
                                     "next_update_equivalence", "frozen_dependencies", "source_inputs_unchanged")),
               "需要真实控制所用同一残差worker的独立verify")
    del payload
    files = [directory / name for name in ("report.json", "evaluation_manifest.json", "design.json")]
    files += [checked_dir / name for name in ("report.json", "design.json", "targets.npz")]
    files += [checkpoint, verify_dir / "report.json", *dependencies.values()]
    cache_dir = baseline.project_path(args.cache_dir)
    files += [cache_dir / name for name in ("cache_manifest.json", "tables.npz", "states.npy", "frozen_bundle.pt", "report.json")]
    required = set()
    for index, row in enumerate(rows):
        name = f"trial_{index:03d}_repeat_{row['repeat']}_target_{row['target']}_{row['mode']}"
        gl.require(baseline.project_path(row["artifact_dir"]) == directory / name, "真实trial路径不属于本次评估")
        required.update(f"{name}/{item}" for item in (
            "trajectory.npz", "events.json", "metrics.json", "control_trace.json", "start.json", "history_check.json"))
    gl.require(required <= set(manifest["artifacts"]), "真实轨迹、状态或动作记录缺少SHA256")
    for name, digest in manifest["artifacts"].items():
        path = (directory / name).resolve()
        gl.require(directory in path.parents and not Path(name).is_absolute() and path.is_file() and
                   bc.file_hash(path) == digest, f"真实产物缺失或内容改变：{name}")
        files.append(path)
    protected = [directory, checked_dir, cache_dir, verify_dir, checkpoint.parent]
    protected += [path.parent for path in dependencies.values()]
    protected += [Path(path).resolve().parent for path in record.get("source_inputs_before", {})]
    return record, manifest, checked_dir, protected, list(dict.fromkeys(files))


def contract_checks(report):
    """Tiny in-memory guards; no fake histories are written as real data."""
    import copy
    import numpy as np
    import goal_action_choice_diagnose as diagnosis

    left, same_winner, other_winner = np.array([.6, .3, .1]), np.array([.5, .4, .1]), np.array([.2, .7, .1])
    same = diagnosis.distribution_difference(left, same_winner)
    changed = diagnosis.distribution_difference(left, other_winner)
    diagnosis.require(not same["mode_changed"] and same["total_variation"] > 0 and
                      np.isclose(same["shared_uniform_sampling_disagreement"], .1) and changed["mode_changed"] and
                      np.isclose(diagnosis.distribution_difference(left, left)["shared_uniform_sampling_disagreement"], 0),
                      "分布变化/argmax/共同均匀随机数的解析区间检查不一致")
    p = {name: left for name in diagnosis.CONDITIONS}
    raw = {name: np.log(left) for name in diagnosis.CONDITIONS}
    raw["goal1"] = raw["goal0"] + 3.
    diagnosis.require(np.isclose(diagnosis.frame_metrics(p, raw)["goal_centered_raw_preference_delta_rms"], 0),
                      "共同偏好平移被误记为有效目标作用")
    for invalid in ([.5, float("nan"), .5], [.5, -.1, .6]):
        t02.rejection(report, "probability_guard", lambda invalid=invalid: diagnosis.probability(invalid), "数值")
    dimension, first, horizon = 3, 1, 2
    marker = dict(features=np.zeros((4, 2), np.float32),
        action=np.vstack((np.zeros(3), np.eye(3)[[0, 1, 2]])),
        is_last=np.zeros(4, bool), is_terminal=np.zeros(4, bool))
    trace = dict(condition="goal", evaluated_target=0, issued_target=0, action_seed=701,
        action_ids=[1, 2], remaining=[2, 1], probabilities=[left.tolist()] * 2,
        uniforms=np.random.RandomState(701).uniform(size=horizon).tolist(), distances_to_both_goals=[[.2, .4]] * 3)
    cell = dict(mode="goal", target=0, randomization_seed=0)
    diagnosis.validate_trace(marker, trace, first, horizon, dimension, cell, 0)
    wrong = copy.deepcopy(marker)
    wrong["action"][first + 1:] = wrong["action"][first:first + 2]
    t02.rejection(report, "next_action_guard", lambda: diagnosis.validate_trace(wrong, trace, first, horizon, dimension, cell, 0), "下一动作")
    wrong_trace = dict(trace, remaining=[1, 2])
    t02.rejection(report, "remaining_guard", lambda: diagnosis.validate_trace(marker, wrong_trace, first, horizon, dimension, cell, 0), "预算")
    wrong = copy.deepcopy(marker)
    wrong["is_terminal"][first] = True
    t02.rejection(report, "terminal_guard", lambda: diagnosis.validate_trace(wrong, trace, first, horizon, dimension, cell, 0), "结束")
    report.check("diagnostic_contracts", "PASS", "概率改变与最大概率动作改变分别计数；共享随机数分歧解析计算；obs[t]→action[t+1]和remaining守卫通过。内存marker不写入真实数据")


def query(policy, features, goals, remaining):
    import torch
    import goal_action_choice_diagnose as diagnosis
    import goal_library as gl

    probabilities, preferences = {}, {}
    steps = torch.tensor([remaining], dtype=torch.int64, device=features.device)
    pair = torch.as_tensor(goals, dtype=torch.float32, device=features.device)
    with torch.no_grad(), gl.encoder_precision(features.device):
        base = policy.base_preferences(features, pair[:1], steps)
        for name in diagnosis.CONDITIONS:
            target = 1 if name == "goal1" else 0
            goal = pair[target:target + 1]
            branch = "goal" if name in ("goal0", "goal1") else name
            correction_goal = torch.zeros_like(goal) if name == "zero_goal" else goal
            raw = base if name == "base" else base + policy.correction(
                "goal" if branch == "zero_goal" else branch, features, correction_goal, steps)
            dist = policy.action_dist(features, goal, steps, branch)
            gl.require(torch.equal(policy.distribution(raw).probs, dist.probs), "偏好分解与真实部署分布不同")
            probabilities[name], preferences[name] = dist.probs[0].cpu().numpy(), raw[0].cpu().numpy()
        no_goal_other = policy.action_dist(features, pair[1:], steps, "no_goal").probs[0].cpu().numpy()
        gl.require((probabilities["no_goal"] == no_goal_other).all(), "独立无目标分支受替换目标影响")
    return probabilities, preferences


def probability_gallery(path, frames, names):
    """All trial starts, in original order; no selection by outcome."""
    from PIL import Image, ImageDraw

    starts = [row for row in frames if row["control_step"] == 0]
    panel = Image.new("RGB", (1080, 185 * len(starts) + 30), "white")
    draw = ImageDraw.Draw(panel)
    colors = dict(goal0="#2166ac", goal1="#d95f02", no_goal="#666666")
    draw.text((5, 5), "Saved control starts only | blue=goal0 orange=goal1 gray=independent no-goal | same state/remaining", fill="black")
    for index, row in enumerate(starts):
        top, bottom = 32 + index * 185, 32 + index * 185 + 137
        draw.text((5, top), f"trial {row['trial_index']:02d} factual={row['factual_condition']} target={row['evaluated_target']} "
                  f"| TV={row['goal_total_variation']:.5f} goal mode switch={row['goal_mode_changed']}", fill="black")
        y0 = top + 22
        for tick in (0., .5, 1.):
            y = bottom - round(tick * (bottom - y0))
            draw.line((55, y, 1070, y), fill="#dddddd")
            draw.text((5, y - 5), str(tick), fill="black")
        for action, name in enumerate(names):
            x = 57 + action * 84
            draw.text((x, bottom + 4), name, fill="black")
            for column, condition in enumerate(colors):
                p = row["probabilities"][condition][action]
                height = round(p * (bottom - y0))
                if height:
                    draw.rectangle((x + column * 19, bottom - height, x + column * 19 + 15, bottom), fill=colors[condition])
    panel.save(path)


def analyze(args, report, manifest, checked_dir):
    import numpy as np
    import torch
    import goal_action_choice_diagnose as diagnosis
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_random_control as random_control
    import goal_residual_worker as residual

    contract_checks(report)
    plan, dimension = manifest["design"], manifest["input_identity"]["input_identity"]["action_dim"]
    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    policy = residual.load_policy(baseline.project_path(args.checkpoint), cache, args.device)
    gl.require(policy.identity == manifest["input_identity"]["input_identity"] and
               policy.model_id == manifest["input_identity"]["model_id"] and
               policy.worker_version == manifest["input_identity"]["worker_version"], "冻结模型或缓存与真实评估身份不同")
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], args.device)
    modules = (policy, library)
    versions = [{name: p._version for name, p in module.named_parameters()} for module in modules]
    digests = [gl.tensor_digest(module.state_dict(), {}) for module in modules]
    rng = bc.capture_rng(args.device)
    with np.load(checked_dir / "targets.npz", allow_pickle=False) as saved:
        bank = {name: saved[name].copy() for name in ("goals", "images", "heatmaps")}
    gl.require(gl.tensor_digest({}, {name: bank[name].tolist() for name in bank}) == plan["target_content_id"] and
               bank["goals"].shape == (2, policy.identity["goal_dim"]) and bank["goals"].dtype == np.float32,
               "固定目标内容或向量结构不同")
    for target in (0, 1):
        observation = {name: torch.as_tensor(bank[key][target:target + 1], device=args.device)
                       for name, key in (("image", "images"), ("heatmap", "heatmaps"))}
        encoded = library(observation)[0].cpu().numpy()
        gl.require(np.allclose(encoded, bank["goals"][target], atol=diagnosis.ATOL, rtol=diagnosis.RTOL), "原真实目标重编码不同")
    report.data.update(input_identity=manifest["input_identity"], evaluation_id=manifest["evaluation_id"],
        design_id=plan["design_id"], target_content_id=plan["target_content_id"],
        source_goal_description=plan["goal_source"], inference_backend=bc.backend_info(args.device),
        diagnosis_code={name: bc.file_hash(ROOT / name) for name in (
            "goal_action_choice_diagnose.py", "scripts/t05_action_choice_diagnose.py")},
        probability_roundtrip_tolerance=dict(atol=diagnosis.ATOL, rtol=diagnosis.RTOL))
    report.check("strict_frozen_load", "PASS", f"同一残差worker第{policy.worker_version}步和目标编码器；只读已有5120维真实因果状态，不构造RSSM、候选模型或优化器")
    report.check("fixed_visual_targets", "PASS", "两个目标与已验收check的SHA256/内容ID一致并重编码通过；目标是原真实RGB/heatmap终点，不是地图坐标")
    if report.data["inference_backend"] != manifest["input_identity"]["inference_backend"]:
        report.check("inference_backend", "WARN", "当前后端与原评估不同；必须逐步通过概率和动作重现，数值容差已记录")
    frames, trial_summary, maximum_error = [], [], 0.
    first, horizon = plan["control_start_frame"], plan["horizon"]
    with torch.no_grad():
        for index, row in enumerate(manifest["rows"]):
            directory = Path(row["artifact_dir"])
            gl.require(legacy.read_json(directory / "metrics.json") == row, "逐trial指标与原manifest不同")
            arrays, events = ctl.read_trajectory(directory), legacy.read_json(directory / "events.json")
            trace, history = legacy.read_json(directory / "control_trace.json"), legacy.read_json(directory / "history_check.json")
            random_control.validate_prefix(arrays, events, plan["start_actions"], dimension)
            gl.require(arrays["features"].dtype == np.float32 and arrays["features"].shape[1] == policy.identity["feature_dim"] and
                       history.get("own_causal_history_passed") is True and history.get("reset_is_real") is True and
                       history.get("comparison_to_other_histories") is False and history.get("start_frame") == first and
                       history.get("total_frames") == len(arrays["features"]) and
                       history.get("maximum_absolute_error") == row["own_history_max_error"],
                       "必须使用原验收通过并按SHA256固定的真实状态；不能重置或复制其它局状态")
            cell = dict(row, randomization_seed=plan["design"]["randomization_seed"])
            n = diagnosis.validate_trace(arrays, trace, first, horizon, dimension, cell, index)
            gl.require(n == row["actual_steps"] and trace.get("model_id") == policy.model_id and
                       trace.get("issued_target") == row["issued_goal"] and
                       trace.get("execution_policy") == plan["design"]["execution_policy"], "动作记录模型、执行策略或步数不同")
            distances = []
            for frame in range(first, first + n + 1):
                observation = {name: torch.as_tensor(arrays[name][frame:frame + 1], device=args.device)
                               for name in ("image", "heatmap")}
                distances.append(ctl.distances(library(observation).cpu().numpy(), bank["goals"])[0])
            gl.require(np.allclose(distances, trace["distances_to_both_goals"], atol=diagnosis.ATOL, rtol=diagnosis.RTOL),
                       "真实逐步画面的目标距离与原记录不同")
            for step in range(n):
                frame, remaining = first + step, horizon - step
                factual_state = arrays["features"][frame:frame + 1].copy()
                features = torch.as_tensor(factual_state, device=args.device)
                before = features.clone()
                p, raw = query(policy, features, bank["goals"], remaining)
                gl.require(torch.equal(features, before), "目标替换改变了真实当前状态")
                factual = "no_goal" if row["mode"] == "no_goal" else f"goal{row['issued_goal']}"
                recorded = np.asarray(trace["probabilities"][step])
                maximum_error = max(maximum_error, float(np.abs(p[factual] - recorded).max()))
                gl.require(np.allclose(p[factual], recorded, atol=diagnosis.ATOL, rtol=diagnosis.RTOL) and
                           ctl.select_action(p[factual], trace["execution_policy"], trace["uniforms"][step]) == trace["action_ids"][step] ==
                           ctl.select_action(recorded, trace["execution_policy"], trace["uniforms"][step]), "冻结前向不能重现原动作概率或原动作")
                entry = dict(trial_index=index, seed=row["seed"], repeat=row["repeat"], evaluated_target=row["target"],
                    factual_condition=row["mode"], factual_issued_goal=row["issued_goal"], control_step=step,
                    frame=frame, next_action_frame=frame + 1, remaining=remaining,
                    factual_action=trace["action_ids"][step], factual_action_name=plan["action_names"][trace["action_ids"][step]],
                    state_sha256=hashlib.sha256(factual_state.tobytes()).hexdigest(),
                    distance_to_goal0=float(distances[step][0]), distance_to_goal1=float(distances[step][1]),
                    **diagnosis.frame_metrics(p, raw), probabilities={name: value.tolist() for name, value in p.items()},
                    raw_preferences={name: value.tolist() for name, value in raw.items()})
                for condition in diagnosis.CONDITIONS:
                    entry[f"{condition}_action_name"] = plan["action_names"][entry[f"{condition}_mode"]]
                frames.append(entry)
            trial_summary.append(dict(trial_index=index, repeat=row["repeat"], target=row["target"], condition=row["mode"],
                actual_action_ids=trace["action_ids"], start_telemetry=row["start_telemetry"], end_telemetry=row["end_telemetry"],
                **diagnosis.aggregate(frames[-n:], dimension)))
            print(f"[ACTION DIAGNOSE] trial={index + 1}/{len(manifest['rows'])} states={n} "
                  f"goal_mode_changes={trial_summary[-1]['goal_mode_changes']} "
                  f"mean_tv={trial_summary[-1]['means']['goal_total_variation']:.6f}", flush=True)
    gl.require(len(frames) == sum(row["actual_steps"] for row in manifest["rows"]), "真实控制状态被丢弃或重复")
    report.data.update(query_frames=len(frames), diagnosed_trials=len(trial_summary), max_recorded_probability_error=maximum_error)
    report.check("real_query_alignment", "PASS", f"全{len(trial_summary)}局、{len(frames)}个真实动作前状态；incoming/下一动作/终止/预算一致；目标替换只改变目标输入")
    report.check("recorded_forward_roundtrip", "PASS", f"逐步概率、原随机数与真实动作全部重现；最大概率误差={maximum_error:.6g}；原状态历史由原SHA256及已通过history_check固定")
    result = diagnosis.summarize(frames, dimension)
    result.update(evaluation_id=manifest["evaluation_id"], target_content_id=plan["target_content_id"], action_names=plan["action_names"],
                  per_trial=trial_summary, goal_separation=float(ctl.distances(bank["goals"][:1], bank["goals"][1:])[0, 0]))
    sequence_comparisons = []
    for repeat in range(plan["design"]["repetitions"]):
        for mode in random_control.MODES:
            pair = [next(row for row in trial_summary if row["repeat"] == repeat and row["condition"] == mode and
                         row["target"] == target) for target in (0, 1)]
            left, right = pair[0]["actual_action_ids"], pair[1]["actual_action_ids"]
            common = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            sequence_comparisons.append(dict(repeat=repeat, condition=mode,
                trial_indices=[row["trial_index"] for row in pair], actual_sequences_equal=left == right,
                common_action_prefix_steps=common, action_counts_equal=bool(np.array_equal(
                    np.bincount(left, minlength=dimension), np.bincount(right, minlength=dimension))),
                end_poses_equal=pair[0]["end_telemetry"]["pose"] == pair[1]["end_telemetry"]["pose"],
                start_pose_difference=ctl.pose_difference(pair[0]["start_telemetry"]["pose"], pair[1]["start_telemetry"]["pose"]),
                descriptive_cross_trial_only=True, identical_history_claimed=False))
    result["factual_sequence_comparisons"] = sequence_comparisons
    baseline.write_json(report.directory / "diagnostics.json", result)
    baseline.write_json(report.directory / "counterfactual_frames.json", dict(format=diagnosis.FORMAT, frames=frames,
        factual_states_only=True, hypothetical_actions_executed=False, synthetic_training_labels=False))
    baseline.write_csv(report.directory / "frame_metrics.csv", [
        {name: value for name, value in row.items() if name not in ("probabilities", "raw_preferences")} for row in frames])
    probability_gallery(report.directory / "start_action_probabilities.png", frames, plan["action_names"])
    overall = result["frame_weighted"]
    print(f"[ACTION SUMMARY] states={len(frames)} goal_mode_change_fraction={overall['goal_mode_change_fraction']:.6f} "
          f"mean_tv={overall['means']['goal_total_variation']:.6f} "
          f"goal0_top2_gap={overall['means']['goal0_gap']:.6f} goal1_top2_gap={overall['means']['goal1_gap']:.6f} "
          f"analytic_shared_uniform_disagreement={overall['means']['shared_uniform_sampling_disagreement']:.6f}", flush=True)
    for comparison in sequence_comparisons:
        print(f"[ACTUAL SEQUENCES] repeat={comparison['repeat']} condition={comparison['condition']} "
              f"equal={int(comparison['actual_sequences_equal'])} counts_equal={int(comparison['action_counts_equal'])} "
              f"common_prefix={comparison['common_action_prefix_steps']} (different real histories)", flush=True)
    for module, expected_versions, digest in zip(modules, versions, digests):
        gl.require({name: p._version for name, p in module.named_parameters()} == expected_versions and
                   gl.tensor_digest(module.state_dict(), {}) == digest and
                   all(not p.requires_grad and p.grad is None for p in module.parameters()), "离线诊断改变冻结参数或产生梯度")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "离线诊断推进了调用方随机数")
    gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or
                       name == "mineclip" or name.startswith("mineclip.") for name in sys.modules), "离线诊断不应导入MineDojo/MineCLIP")
    report.check("same_state_goal_comparison", "PASS", "逐帧两目标、独立无目标、零目标、冻结底座五分布；独立无目标替换不变；概率差、top2差距、mode变化及解析采样分歧分别保存")
    report.check("no_training", "PASS", "冻结参数/版本/梯度/RNG未变；不构造优化器/RSSM，不导入MineDojo/MineCLIP；new_env_steps=0")
    report.check("scope", "WARN", "只诊断同一真实状态的动作选择；没有执行反事实动作，不能证明目标可达、延长horizon有效或sample执行更好；仍为T05，T06未推进")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="已完整通过的random_control_evaluate目录")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--residual-verify-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    args.command = "analyze"
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "action_choice_diagnose_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    try:
        record, manifest, checked_dir, protected, files = historical_inputs(args)
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于真实轨迹、check、缓存、模型和共享依赖")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, counterfactual_env_steps=0,
                       controlled_reachability_verified=False, task_success_evaluated=False)
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
        report.check("historical_identity", "PASS", "原完整随机评估/check/manifest/全部trial SHA256及模型/代码/验收身份匹配；原失败和真实结果不改写")
        bc.require_disk_space(directory, 8 * 1024**2)
        report.require_writable()
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            analyze(args, report, manifest, checked_dir)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原轨迹/模型/缓存/验收产物大小和修改时间未变；读取的真实产物另按原SHA256验真；只写新诊断目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
