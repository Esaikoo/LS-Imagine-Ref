"""Check factual within-horizon supervision with frozen350; no training or rollout.

Read all twenty saved episodes, retain whole-episode splits, and compare each
real frame80 endpoint goal with the original fixed goals on frames76..79.
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
import t05_goal_control as legacy
import t05_within_horizon_feedback as collection

CODE = ("goal_within_horizon_supervision.py", "scripts/t05_within_horizon_supervision.py")


class Report(baseline.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        saved = self.save()
        failed = not saved or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_WITHIN_HORIZON_SUPERVISION_CHECK; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def historical_inputs(args):
    import goal_bc as bc
    import goal_continuation_control as confirmation
    import goal_library as gl
    import goal_within_horizon_feedback as feedback
    directory = baseline.project_path(args.eval_dir)
    record = legacy.read_json(directory / "report.json")
    legacy.accepted(record, "evaluate")
    manifest = legacy.read_json(directory / "evaluation_manifest.json")
    plan, rows = manifest["design"], manifest["rows"]
    feedback.validate_plan(plan)
    summary = feedback.summarize(rows, plan)
    coverage = feedback.coverage(rows, plan)
    index = legacy.read_json(directory / "endpoint_data.json")
    gl.require(record["arguments"]["device"] == args.device and
        manifest.get("format") == feedback.EVALUATION_FORMAT and
        manifest.get("comparison_protocol") == record.get("comparison_protocol") == feedback.PROTOCOL and
        manifest.get("evaluation_id") == record.get("evaluation_id") == gl.tensor_digest({}, {
            k: v for k, v in manifest.items() if k != "evaluation_id"}) and
        manifest.get("design_id") == record.get("design_id") == plan["design_id"] == gl.tensor_digest({}, {
            k: v for k, v in plan.items() if k != "design_id"}) and
        record["input_identity"] == manifest["input_identity"] == plan["input_identity"] and
        record["evaluation_code"] == plan["evaluation_code"] == collection.code_identity() and
        legacy.read_json(directory / "design.json") == plan and legacy.read_json(directory / "trials.json") == rows and
        legacy.read_json(directory / "coverage.json") == coverage,
        "需要原双目标预算内20局的完整身份/计划/45个已验收代码和真实事实索引")
    expected = dict(planned_trials=20, attempted_trials=20, execution_valid_trials=20, completed_trial_records=20,
        new_env_steps=1600, attempted_env_steps=1600, evaluation_env_steps=1600,
        worker_action_queries=320, worker_control_actions=280, intervention_actions=40, control_actions=320,
        probability_roundtrip_queries=320, measurement_roundtrip_frames=340,
        maximum_probability_error=0, endpoint_episodes=20, endpoint_action_records=80,
        optimizer_updates=0, worker_version=350, source_worker_version=300, additional_updates=50,
        expert_labels_generated=False, labels_generated=False, approved_for_training=False,
        behavior_accepted=False, t06_approved=False)
    gl.require(all(record.get(k) == v for k, v in expected.items()) and
        all(manifest.get(k) == expected[k] for k in ("new_env_steps", "attempted_env_steps", "worker_action_queries",
            "worker_control_actions", "intervention_actions", "control_actions", "probability_roundtrip_queries",
            "measurement_roundtrip_frames", "maximum_probability_error", "endpoint_episodes", "endpoint_action_records",
            "optimizer_updates", "expert_labels_generated", "approved_for_training", "behavior_accepted", "t06_approved")) and
        record.get("full_design_completed") is True and manifest.get("full_design_completed") is True and
        record.get("source_inputs_before") == record.get("source_inputs_after") and
        summary["valid_histories"] == len(rows) == 20 and
        all(r["execution_valid"] and r["fixed_horizon_completed"] for r in rows) and
        record["pool_action_counts"] == manifest["pool_action_counts"] == index["pool_action_counts"] ==
        dict(manual_train=24, manual_development_holdout=16, worker_facts_only=40) and
        index.get("format") == feedback.DATA_FORMAT and index.get("design_id") == plan["design_id"] and
        index.get("input_identity") == plan["input_identity"] and index.get("complete_episodes") == 20 and
        index.get("real_action_records") == 80 and len(index["episodes"]) == 20 and
        index.get("planned_trials") == index.get("attempted_trials") == 20 and
        index.get("intended_fixed_target_is_not_a_label") is True and
        index.get("approved_for_training") is False and index.get("expert_labels_generated") is False and
        index.get("optimizer_updates") == 0 and index.get("t06_approved") is False and
        confirmation.passed(record, ("historical_identity", "strict_frozen_worker350", "fixed_visual_targets",
            "design_identity", "planned_attempts", "real_control_interface", "execution_counters",
            "fixed_horizon_endpoints", "actual_endpoint_data", "no_training", "source_inputs_unchanged")),
        "需要全部20局/80事实动作与完整工程验收；缺覆盖保留，不接受筛选/补样或训练标签")
    source, protected, files = collection.historical_inputs(SimpleNamespace(command="check", device=args.device,
        diagnosis_dir=record["arguments"]["diagnosis_dir"]))
    gl.require(plan["source_diagnosis_id"] == source["diagnosis_manifest"]["diagnosis_id"] and
        plan["input_identity"] == source["manifest"]["input_identity"] and
        plan["original_numerical_gate"] == record["original_numerical_gate"] == source["record"]["numerical_gate"] and
        plan["source_evaluation_id"] == source["manifest"]["evaluation_id"], "旧诊断/生产350/固定目标或原行为失败被修改")
    checked_dir = baseline.project_path(record["arguments"]["check_dir"])
    checked = legacy.read_json(checked_dir / "report.json")
    legacy.accepted(checked, "check")
    gl.require(manifest["check_report_sha256"] == bc.file_hash(checked_dir / "report.json") and
        checked["design_id"] == plan["design_id"] and checked["input_identity"] == plan["input_identity"] and
        checked["evaluation_code"] == plan["evaluation_code"] and
        legacy.read_json(checked_dir / "design.json") == plan and
        checked["targets_sha256"] == bc.file_hash(checked_dir / "targets.npz") and
        record["targets_sha256"] == bc.file_hash(directory / "targets.npz") and
        checked.get("source_inputs_before") == checked.get("source_inputs_after") and
        all(checked.get(k) == 0 for k in ("optimizer_updates", "new_env_steps", "worker_action_queries", "control_actions")) and
        confirmation.passed(checked, ("historical_identity", "strict_frozen_worker350", "fixed_visual_targets",
            "causal_execution_interface", "feedback_contracts", "endpoint_index_contracts", "no_training", "source_inputs_unchanged")),
        "需要原双目标预算内计划的独立check及真实目标文件")
    required = set()
    for i, row in enumerate(rows):
        name = f"trial_{i:03d}_repeat_{row['repeat']}_target_{row['target']}_{row['condition']}"
        gl.require(baseline.project_path(row["artifact_dir"]) == directory / name, "本局轨迹属于另一采集")
        required.update(f"{name}/{n}" for n in ("trajectory.npz", "events.json", "control_trace.json", "history_check.json",
            "frame_metrics.json", "metrics.json", "start.json", "actual_endpoint.npz", "video.mp4"))
        for n, digest in index["episodes"][i]["artifacts_sha256"].items():
            gl.require(manifest["artifacts"].get(f"{name}/{n}") == digest, "事实索引与本局真实产物SHA不同")
    gl.require(required <= set(manifest["artifacts"]) and
        {"design.json", "targets.npz", "trials.json", "trials.csv", "frame_metrics.csv", "diagnostics.json",
         "coverage.json", "endpoint_data.json", "initial_balance.json"} <= set(manifest["summary_artifacts"]),
        "完整真实历史/视频/终点或汇总未绑定SHA")
    for table in (manifest["artifacts"], manifest["summary_artifacts"]):
        for name, digest in table.items():
            path = (directory / name).resolve()
            gl.require(not Path(name).is_absolute() and directory in path.parents and path.is_file() and
                bc.file_hash(path) == digest, f"预算内真实产物SHA256不同：{name}")
            files.append(path)
    saved_summary = legacy.read_json(directory / "diagnostics.json")
    gl.require(all(saved_summary[k] == v for k, v in summary.items()) and
        saved_summary["manual_branch_coverage"] == coverage and
        saved_summary["original_numerical_gate"] == plan["original_numerical_gate"], "原全部轨迹统计/覆盖或旧gate不同")
    files += [directory / "report.json", directory / "evaluation_manifest.json",
        *[checked_dir / n for n in ("report.json", "design.json", "targets.npz")], *[ROOT / n for n in CODE]]
    source.update(feedback_dir=directory, feedback_record=record, feedback_manifest=manifest,
        feedback_checked_dir=checked_dir, feedback_index=index, feedback_coverage=coverage)
    return source, list(dict.fromkeys([*protected, directory, checked_dir])), list(dict.fromkeys(p.resolve() for p in files))


def load_policy(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_continuation_repair as repaired
    import goal_information_probe as probe
    import goal_library as gl
    context, plan = source["context"], source["feedback_manifest"]["design"]
    cache = bc.TrainingCache(context["cache_dir"])
    policy = repaired.load_policy(context["checkpoint"], cache, args.device)
    gl.require(context["checkpoint"].name == "latest.pt" and policy.worker_version == 350 and
        policy.continuation_updates == 50 and policy.model_id == plan["input_identity"]["model_id"] and
        policy.identity == plan["input_identity"]["input_identity"] and
        context["verified"]["backend"] == bc.backend_info(args.device), "需要同后端/固定50步生产latest的冻结worker350")
    paths = (source["feedback_dir"] / "targets.npz", source["feedback_checked_dir"] / "targets.npz",
             source["eval_dir"] / "targets.npz", source["checked_dir"] / "targets.npz")
    banks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            gl.require(set(saved.files) == {"goals", "images", "heatmaps"}, "固定视觉目标字段不同")
            banks.append({k: saved[k].copy() for k in saved.files})
    bank = banks[0]
    gl.require(all(np.array_equal(b[k], bank[k]) for b in banks for k in bank) and
        bank["goals"].dtype == np.float32 and bank["goals"].shape == (2, policy.identity["goal_dim"]) and
        np.isfinite(bank["goals"]).all() and np.allclose(np.linalg.norm(bank["goals"], axis=-1), 1, atol=1e-5) and
        bank["images"].shape == (2, 64, 64, 3) and bank["heatmaps"].shape == (2, 64, 64) and
        bank["images"].dtype == bank["heatmaps"].dtype == np.uint8 and
        policy.identity["feature_dim"] == 5120 and policy.identity["action_dim"] == 12 and policy.identity["horizon"] == 16 and
        gl.tensor_digest({}, {k: v.tolist() for k, v in bank.items()}) == plan["target_content_id"] and
        probe.array_digest({"goals": bank["goals"]}) == policy.identity["fixed_goals_id"], "原check/确认/预算内两视觉目标不同")
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], args.device).requires_grad_(False).eval()
    report.data.update(worker_version=350, model_id=policy.model_id, input_identity=plan["input_identity"],
        source_worker_version=300, additional_updates=50, inference_backend=bc.backend_info(args.device),
        source_checkpoint_sha256=bc.file_hash(context["checkpoint"]), target_content_id=plan["target_content_id"])
    report.check("strict_frozen_worker350", "PASS", "固定生产300+50=350及独立verify，同后端独立推理；无优化器/RSSM/候选模型")
    return policy, library, bank


def distribution(policy, state, goal, remaining, mode):
    import torch
    import goal_library as gl
    steps = torch.tensor([remaining], dtype=torch.int64, device=state.device)
    target = torch.as_tensor(goal[None].copy(), dtype=torch.float32, device=state.device)
    with torch.no_grad(), gl.encoder_precision(state.device):
        base = policy.base_preferences(state, target, steps)
        correction_goal = torch.zeros_like(target) if mode == "zero_goal" else target
        raw = base if mode == "base" else base + policy.correction("goal" if mode == "zero_goal" else mode,
                                                                  state, correction_goal, steps)
        dist = policy.action_dist(state, target, steps, mode)
        gl.require(torch.equal(policy.distribution(raw).probs, dist.probs), "raw偏好分解/归一化或unimix与部署不同")
        return dist.probs[0].cpu().numpy().copy(), raw[0].cpu().numpy().copy()


def contract_checks(report, arrays, events, trace, measured, endpoint, episode, metrics, history, row, plan):
    import numpy as np
    import goal_within_horizon_feedback as feedback
    import goal_within_horizon_supervision as supervision
    i = row["trial_index"]
    def episode_check(candidate):
        return supervision.episode_binding(candidate, row, plan, i)
    for name, changes in (("holdout_pool_guard", dict(episode_split="development_holdout")),
        ("worker_pool_guard", dict(factual_pool="worker_facts_only")), ("expert_guard", dict(expert_labels=True)),
        ("endpoint_source_guard", dict(supervision_goal="intended_fixed_target")), ("version_guard", dict(worker_version=300))):
        t02.rejection(report, name, lambda c=dict(episode, **changes): episode_check(c), "事实池")
    expected = supervision.query_identity(episode, row, trace, measured, 0)
    for name, changes in (("after_endpoint_query_guard", dict(state_frame=80, incoming_action_frame=81, remaining=0)),
        ("remaining_guard", dict(remaining=1)), ("incoming_index_guard", dict(incoming_action_frame=76)),
        ("factual_action_guard", dict(actual_action=(expected["actual_action"] + 1) % 12)),
        ("query_pool_guard", dict(factual_pool="manual_development_holdout")),
        ("borrowed_goal_guard", dict(supervision_goal="another_episode_endpoint")),
        ("training_approval_guard", dict(approved_for_training=True))):
        t02.rejection(report, name, lambda c=dict(expected, **changes): supervision.validate_query(c, expected), "查询状态")
    broken = copy.deepcopy(episode["queries"])
    broken[0]["actual_action"] = (broken[0]["actual_action"] + 1) % 12
    t02.rejection(report, "endpoint_action_alignment_guard", lambda: feedback.validate_endpoint_queries(arrays, trace, endpoint, broken), "事实索引")
    altered = {k: v.copy() for k, v in endpoint.items()}
    altered["image"] = arrays["image"][64].copy()
    if np.array_equal(altered["image"], endpoint["image"]):
        altered["image"].flat[0] ^= 1
    t02.rejection(report, "own_endpoint_guard", lambda: feedback.validate_endpoint_queries(arrays, trace, altered, episode["queries"]), "监督目标")
    broken_history = dict(history, reset_is_real=False)
    t02.rejection(report, "real_history_guard", lambda: supervision.validate_saved_trial(arrays, events, trace, measured,
        endpoint, episode, metrics, broken_history, row, plan, i), "真实因果历史")
    broken_arrays = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    broken_arrays["action"][77] = np.roll(broken_arrays["action"][77], 1)
    t02.rejection(report, "real_incoming_guard", lambda: feedback.validate_trace(broken_arrays, events, trace, plan, i), "incoming")
    report.check("supervision_contracts", "PASS", "真实incoming/本局80帧/末4步预算/划分/事实池与负守卫；内存副本不写作数据")


def check(args, report, source):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_control_stats as stats
    import goal_library as gl
    import goal_reference_calibration as calibration
    import goal_action_choice_diagnose as choices
    import goal_within_horizon_supervision as supervision
    loader_rng = bc.capture_rng(args.device)
    try:
        policy, library, bank = load_policy(args, report, source)
    finally:
        bc.restore_rng(loader_rng, args.device)
    modules = (policy, library)
    versions = [{n: p._version for n, p in m.named_parameters()} for m in modules]
    digests = [gl.tensor_digest(m.state_dict(), {}) for m in modules]
    rng = bc.capture_rng(args.device)
    bank_before = {k: v.copy() for k, v in bank.items()}
    plan, original = source["feedback_manifest"]["design"], source["feedback_manifest"]["rows"]
    report.data.update(comparison_protocol=supervision.PROTOCOL, source_evaluation_id=source["feedback_manifest"]["evaluation_id"],
        source_design_id=plan["design_id"], original_numerical_gate=plan["original_numerical_gate"],
        source_manifest_sha256=bc.file_hash(source["feedback_dir"] / "evaluation_manifest.json"),
        read_only_code={n: bc.file_hash(ROOT / n) for n in CODE},
        state_source="SHA-bound saved causal features; no RSSM reconstruction", diagnostic_conditions=list(supervision.CONDITIONS))
    for target in (0, 1):
        obs = {n: torch.as_tensor(bank[k][target:target + 1], device=args.device)
               for n, k in (("image", "images"), ("heatmap", "heatmaps"))}
        gl.require(np.allclose(library(obs)[0].cpu().numpy(), bank["goals"][target], atol=choices.ATOL, rtol=choices.RTOL), "原固定目标重编码不同")
    report.check("fixed_visual_targets", "PASS", "原两RGB/heatmap目标重编码与四来源内容一致；真实终点另编码，不替换意向目标")
    queries, trials, flat, maximum_error = [], [], [], 0.
    checked_contracts = False
    for i, row in enumerate(original):
        directory = baseline.project_path(row["artifact_dir"])
        arrays = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "control_trace.json")
        measured = legacy.read_json(directory / "frame_metrics.json")
        history = legacy.read_json(directory / "history_check.json")
        metrics = legacy.read_json(directory / "metrics.json")
        start = legacy.read_json(directory / "start.json")
        episode = source["feedback_index"]["episodes"][i]
        with np.load(directory / "actual_endpoint.npz", allow_pickle=False) as saved:
            endpoint = {k: saved[k].copy() for k in saved.files}
        supervision.validate_saved_trial(arrays, events, trace, measured, endpoint, episode, metrics, history, row, plan, i)
        gl.require(start.get("control_start_frame") == 64 and start.get("own_real_history_only") is True and
            start.get("no_state_or_distance_filter") is True and stats.native_actions_equal(start["telemetry"], events[64]["telemetry"]) and
            np.array_equal(np.asarray(start["rssm_feature"], np.float32), arrays["features"][64]), "本局真实起点/历史被替换")
        if not checked_contracts and episode["factual_pool"] == "manual_train":
            contract_checks(report, arrays, events, trace, measured, endpoint, episode, metrics, history, row, plan)
            checked_contracts = True
        endpoint_goal = None
        for offset, saved in enumerate(measured):
            frame = 64 + offset
            obs = {n: torch.as_tensor(arrays[n][frame:frame + 1], device=args.device) for n in ("image", "heatmap")}
            feature = library(obs)[0].cpu().numpy()
            values = calibration.measure(arrays["image"][frame], arrays["heatmap"][frame], feature,
                events[frame]["telemetry"], bank, plan["target_telemetry"])
            gl.require(np.allclose(feature, saved["goal_feature"], atol=choices.ATOL, rtol=choices.RTOL) and
                all(saved[k] == v if k.startswith("closest_") else np.isclose(saved[k], v, atol=choices.ATOL, rtol=choices.RTOL)
                    for k, v in values.items()), "原真实视觉/物理测量不能重现")
            report.data["measurement_roundtrip_frames"] += 1
            if frame == 80:
                endpoint_goal = feature.copy()
        gl.require(endpoint_goal is not None and np.allclose(endpoint_goal, endpoint["goal_feature"],
            atol=choices.ATOL, rtol=choices.RTOL), "本局真实80帧终点编码不能重现")
        # Use the SHA-bound stored goal after its independent re-encoding check.
        endpoint_goal = endpoint["goal_feature"].copy()
        fixed, swapped = bank["goals"][row["target"]], bank["goals"][1 - row["target"]]
        trial_queries = []
        for offset in range(16):
            frame, remaining = 64 + offset, 16 - offset
            state_array = arrays["features"][frame:frame + 1].copy()
            state = torch.as_tensor(state_array, device=args.device)
            unchanged = state.clone()
            fixed_probability, fixed_raw = distribution(policy, state, fixed, remaining, "goal")
            error = float(np.max(np.abs(fixed_probability - np.asarray(trace["probabilities"][offset]))))
            maximum_error = max(maximum_error, error)
            gl.require(np.allclose(fixed_probability, trace["probabilities"][offset], atol=choices.ATOL, rtol=choices.RTOL) and
                int(fixed_probability.argmax()) == trace["worker_action_ids"][offset], "原350概率或worker建议mode不能重现")
            # The actual manual action can differ from the worker's recorded mode.
            report.data["probability_roundtrip_queries"] += 1
            if offset >= 12:
                identity = supervision.query_identity(episode, row, trace, measured, offset - 12)
                probabilities, raw = dict(fixed_goal=fixed_probability), dict(fixed_goal=fixed_raw)
                inputs = {"actual_endpoint": (endpoint_goal, "goal"), "swapped_goal": (swapped, "goal"),
                          "no_goal": (fixed, "no_goal"), "zero_goal": (fixed, "zero_goal"), "base": (fixed, "base")}
                for name, (goal, mode) in inputs.items():
                    probabilities[name], raw[name] = distribution(policy, state, goal, remaining, mode)
                for name in ("no_goal", "zero_goal", "base"):
                    for alternative in (endpoint_goal, swapped):
                        p, pref = distribution(policy, state, alternative, remaining, name)
                        gl.require(np.array_equal(p, probabilities[name]) and np.array_equal(pref, raw[name]),
                                   "无目标/置零/底座受实际或交换目标影响")
                entry = dict(identity, actual_action_name=plan["action_names"][identity["actual_action"]],
                    worker_suggestion_name=plan["action_names"][identity["worker_suggestion"]],
                    state_sha256=hashlib.sha256(state_array.tobytes()).hexdigest(),
                    endpoint_goal_sha256=hashlib.sha256(endpoint_goal.tobytes()).hexdigest(),
                    endpoint_artifact_sha256=episode["artifacts_sha256"]["actual_endpoint.npz"],
                    fixed_goal_sha256=hashlib.sha256(fixed.tobytes()).hexdigest(),
                    metrics=supervision.factual_metrics(probabilities, raw, identity["actual_action"]),
                    probabilities={k: v.tolist() for k, v in probabilities.items()},
                    raw_preferences={k: v.tolist() for k, v in raw.items()})
                supervision.validate_query(entry, identity)
                queries.append(entry)
                trial_queries.append(entry)
                flat.append(dict({k: v for k, v in entry.items() if k not in ("metrics", "probabilities", "raw_preferences")},
                                 **entry["metrics"]))
            gl.require(torch.equal(state, unchanged), "只读目标查询改变本局状态")
        trial = dict(trial_index=i, repeat=row["repeat"], target=row["target"], condition=row["condition"],
            episode_split=row["episode_split"], factual_pool=episode["factual_pool"],
            actual_endpoint_frame=80, endpoint_distance_to_fixed_goal=row["end_distance"],
            endpoint_pose_errors=row["pose_errors_at_end"], corrections=row["correction_actions"],
            source_endpoint_artifact_sha256=episode["artifacts_sha256"]["actual_endpoint.npz"],
            fixed_goal_is_not_supervision_label=True, **supervision.aggregate(trial_queries))
        trials.append(trial)
        report.data.update(checked_episodes=len(trials), query_rows=len(queries), diagnostic_distribution_rows=len(queries) * 6)
        report.save()
        print(f"[SUPERVISION {i + 1}/20] target={row['target']} split={row['episode_split']} pool={episode['factual_pool']} "
              f"facts=4 actual_nll={trial['metrics']['actual_endpoint_factual_nll']:.6f} "
              f"fixed_nll={trial['metrics']['fixed_goal_factual_nll']:.6f} corrections={row['correction_actions']}", flush=True)
    summary = supervision.summarize(queries, trials, plan)
    for target in (0, 1):
        for split in ("train", "development_holdout"):
            previous = source["feedback_coverage"]["groups"][f"target{target}_{split}"]
            for branch in ("holding", "correction"):
                group = summary["by_target_split_branch"][f"target{target}_{split}_{branch}"]
                gl.require(group["rows"] == previous[f"{branch}_rows"] and
                    group["independent_episodes"] == previous[f"{branch}_episodes"], "事实分支覆盖与原采集不同")
    gl.require(summary["all_manual_branches_observed"] == source["feedback_coverage"]["all_target_split_branches_observed"],
               "缺覆盖被监督检查改写")
    summary["original_numerical_gate"] = plan["original_numerical_gate"]
    summary["source_manual_branch_coverage"] = source["feedback_coverage"]
    review = supervision.review(summary, trials, plan["action_names"])
    for name, candidate in (("duplicate_query_guard", queries + queries[:1]), ("missing_query_guard", queries[:-1]),
                            ("query_order_guard", [queries[1], queries[0], *queries[2:]])):
        t02.rejection(report, name, lambda c=candidate: supervision.summarize(c, trials, plan), "80查询")
    baseline.write_json(report.directory / "queries.json", queries)
    baseline.write_csv(report.directory / "frame_metrics.csv", flat)
    baseline.write_json(report.directory / "trials.json", trials)
    baseline.write_json(report.directory / "diagnostics.json", summary)
    baseline.write_json(report.directory / "training_review.json", review)
    gl.require(legacy.read_json(report.directory / "queries.json") == queries and
               legacy.read_json(report.directory / "training_review.json") == review, "监督检查原子JSON往返不同")
    for module, expected_versions, digest in zip(modules, versions, digests):
        gl.require({n: p._version for n, p in module.named_parameters()} == expected_versions and
            gl.tensor_digest(module.state_dict(), {}) == digest and all(not p.requires_grad and p.grad is None for p in module.parameters()),
            "监督检查改变模型参数/版本/梯度")
    gl.require(all(np.array_equal(bank[k], bank_before[k]) for k in bank) and t01.same(rng, bc.capture_rng(args.device)),
               "只读监督查询改变固定目标或RNG")
    gl.require(not any(n in ("minedojo", "mineclip") or n.startswith(("minedojo.", "mineclip.")) for n in sys.modules),
               "监督check不能导入MineDojo/MineCLIP")
    gl.require(report.data["probability_roundtrip_queries"] == 320 and report.data["measurement_roundtrip_frames"] == 340 and
        report.data["query_rows"] == 80 and report.data["diagnostic_distribution_rows"] == 480, "完整20局监督查询计数不同")
    report.data.update(maximum_probability_error=maximum_error, missing_manual_branches=summary["missing_manual_branches"],
        dual_goal_correction_train_coverage=review["dual_goal_correction_train_coverage"],
        pool_action_counts={p: summary["by_pool"][p]["rows"] for p in supervision.POOLS})
    output = dict(format=supervision.FORMAT, comparison_protocol=supervision.PROTOCOL,
        source_evaluation_id=source["feedback_manifest"]["evaluation_id"], source_design_id=plan["design_id"],
        source_manifest_sha256=report.data["source_manifest_sha256"], input_identity=plan["input_identity"],
        target_content_id=plan["target_content_id"], read_only_code=report.data["read_only_code"],
        query_rows=80, diagnostic_distribution_rows=480, checked_episodes=20,
        probability_roundtrip_queries=320, measurement_roundtrip_frames=340, maximum_probability_error=maximum_error,
        artifacts={p.name: bc.file_hash(p) for p in report.directory.iterdir() if p.is_file() and p.name != "report.json"},
        optimizer_updates=0, new_env_steps=0, labels_generated=False, expert_labels_generated=False,
        approved_for_training=False, behavior_accepted=False, t06_approved=False)
    output["supervision_id"] = gl.tensor_digest({}, output)
    baseline.write_json(report.directory / "supervision_manifest.json", output)
    report.data["supervision_id"] = output["supervision_id"]
    report.check("real_query_alignment", "PASS", "完整20局/80事实查询，frame76–79→incoming77–80/remaining4–1；实际终点只作事后评价")
    report.check("recorded_forward_roundtrip", "PASS", f"原350全部320概率/worker建议重现，最大误差={maximum_error:.6g}；手工实际动作另核对")
    report.check("recorded_measurement_roundtrip", "PASS", "全部340视觉/物理测量与20个本局80帧终点编码重现，不重算RSSM")
    report.check("same_state_goal_comparison", "PASS", "80状态×实际终点/固定/交换/无目标/置零/底座六分布及raw偏好，后三区严格目标不变")
    report.check("manual_branch_coverage", "WARN" if summary["missing_manual_branches"] else "PASS",
        "缺覆盖=" + str(summary["missing_manual_branches"]) + "；空组统计为不可用，不重划分/补样或重复纠偏")
    report.check("no_training", "PASS", "参数/版本/梯度/目标/RNG未变；无优化器/RSSM/环境，更新/新环境步/动作执行0")
    report.check("scope", "WARN", "事实拟合与保持/纠偏分别报告；实际终点是事后目标，非部署输入或专家标签；不批准训练、改gate或T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--eval-dir", required=True)
    checking.add_argument("--device", default="cuda:0")
    checking.add_argument("--output-dir")
    checking.add_argument("--feedback-dir", help="fresh independent export directory; otherwise generated automatically")
    args = parser.parse_args()
    args.eval_dir = str(baseline.project_path(args.eval_dir))
    try:
        source, protected, files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            "within_horizon_supervision_check_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == p or p in directory.parents or directory in p.parents for p in protected):
            parser.error("输出须独立于原模型/缓存/轨迹/验收目录")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, worker_control_actions=0, intervention_actions=0,
        labels_generated=False, expert_labels_generated=False, approved_for_training=False,
        behavior_accepted=False, t06_approved=False, checked_episodes=0, query_rows=0,
        probability_roundtrip_queries=0, measurement_roundtrip_frames=0, diagnostic_distribution_rows=0,
        feedback_policy="automatic lossless result bundle after final report; failures retained")
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
        bc.require_disk_space(directory, 64 * 1024**2)
        baseline.write_json(directory / "output_probe.json", dict(atomic_output_probe_only=True, no_model_or_environment=True))
        gl.require(legacy.read_json(directory / "output_probe.json")["atomic_output_probe_only"] is True, "独立目录原子写入不同")
        report.check("output_preflight", "PASS", "独立新目录拒绝覆盖，原子JSON/空间预检通过，尚未构造推理模块")
        report.check("historical_identity", "PASS", "完整20局/80事实动作/独立check/原诊断/350确认/生产latest/verify及45个已验收代码SHA；旧FAIL保持")
        rng = bc.capture_rng(args.device)
        with torch.no_grad():
            check(args, report, source)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(p): baseline.file_signature(p) for p in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                    "原模型/依赖/轨迹/终点/验收及45个旧代码大小/修改时间未变；只写独立新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    result = report.finish()
    try:
        import t05_result_bundle as feedback
        sources = [("run", directory), ("feedback_evaluate", source["feedback_dir"]),
            ("feedback_check", source["feedback_checked_dir"]), ("diagnosis", source["diagnosis_directory"]),
            ("source_evaluate", source["eval_dir"]), ("repair_verify", source["context"]["verify_dir"])]
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT / "relevance_map/t05_feedback" / (
            directory.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in [*protected, directory]):
            raise ValueError("汇总输出必须独立于旧目录与新check目录")
        index = feedback.build_bundle(sources, export, directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export / 'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"[FAIL] feedback_export: {type(error).__name__}: {error}；原报告已保留", file=sys.stderr, flush=True)
        result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())
