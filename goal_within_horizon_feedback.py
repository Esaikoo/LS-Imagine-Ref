"""Fixed worker350 development collection inside the original 16-step budget."""
import numpy as np
import goal_control as ctl
import goal_library as gl
import goal_reference_calibration as calibration
from goal_action_choice_diagnose import probability
from goal_control_stats import statistics

FORMAT = "ls_imagine_within_horizon_feedback_design_v1"
EVALUATION_FORMAT = "ls_imagine_within_horizon_feedback_evaluation_v1"
DATA_FORMAT = "ls_imagine_within_horizon_feedback_facts_v1"
PROTOCOL = "worker350_dual_goal_12_mode_4_worker_or_visual_correct_once_hold_v1"
CONDITIONS = ("worker_mode", "visual_correct_hold")
RANDOMIZATION_SEED = 5
BUDGETS = list(range(16, 0, -1))
MARGIN_EPSILON = 1e-7
DECISION_FIELDS = {
    "actual_action": "action_ids", "worker_action": "worker_action_ids",
    "decision_source": "decision_sources", "phase": "phases",
    "intervention_applied": "intervention_applied", "action_differs_from_worker": "action_differs_from_worker",
    "correction_used_before": "correction_used_before", "correction_applied": "correction_applied",
}


def schedule():
    rng = np.random.RandomState(RANDOMIZATION_SEED)
    cells = [(target, condition) for target in (0, 1) for condition in CONDITIONS]
    return [dict(seed=0, repeat=repeat, target=target, condition=condition)
            for repeat in range(5)
            for index in rng.permutation(4)
            for target, condition in [cells[int(index)]]]


def split(repeat):
    gl.require(type(repeat) is int and 0 <= repeat < 5, "整局repeat划分异常")
    return "train" if repeat < 3 else "development_holdout"


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "target", "condition"))


def feedback_rule():
    return dict(input="current_real_rgb_heatmap_distances_only", margin_epsilon=MARGIN_EPSILON,
                correction_by_target={"0": "turn_up", "1": "turn_down"},
                correction_limit=1, hold="noop", tie="noop", stop_on_success=False)


def validate_plan(plan):
    d = plan["design"]
    # The accepted MineDojo scenario stores the world seed as a string. Keep it unchanged.
    gl.require(plan.get("scenario", {}).get("world_seed") == "1",
        "原场景world_seed必须保持字符串'1'，不能转换类型或更换世界")
    gl.require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL and
        plan.get("worker_version") == 350 and plan.get("source_worker_version") == 300 and
        plan.get("additional_updates") == 50 and plan.get("control_start_frame") == 64 and
        plan.get("horizon") == 16 and plan.get("feedback_start_frame") == 76 and
        plan.get("feedback_horizon") == 4 and plan.get("actual_endpoint_frame") == 80 and
        plan.get("maximum_new_env_steps") == 1600 and plan.get("warmup_steps") == 32 and
        len(plan["prefix_actions"]) == 32 and plan["start_actions"] == [0] * 32 + plan["prefix_actions"] and
        plan["feedback_rule"] == feedback_rule() and
        plan.get("episode_split") == {str(i): split(i) for i in range(5)} and
        plan.get("same_hidden_state") is False and plan.get("matched_start_comparison") is False and
        plan.get("behavior_accepted") is False and plan.get("t06_approved") is False and
        d == dict(seed=0, repetitions=5, randomization_seed=RANDOMIZATION_SEED,
                  conditions=list(CONDITIONS), execution_policy="mode", schedule=schedule()),
        "双目标预算内反馈计划/规则/种子/划分被修改")
    gl.require(len(plan["action_names"]) == 12 and plan["action_names"][0] == "noop" and
        plan["noop_action"] == 0 and
        all(plan[name + "_action"] == plan["action_names"].index(name) for name in ("turn_up", "turn_down")),
        "预声明宏动作接口不同")


def action_seed(plan, index):
    validate_plan(plan)
    gl.require(type(index) is int and 0 <= index < 20, "计划trial索引异常")
    return (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)


def boundary_kind(distances):
    d = np.asarray(distances, dtype=np.float64)
    gl.require(d.shape == (2,) and np.isfinite(d).all() and np.all((0 <= d) & (d <= 2)), "真实视觉距离异常")
    margin = d[0] - d[1]
    return "closer_target1" if margin > MARGIN_EPSILON else "closer_target0" if margin < -MARGIN_EPSILON else "tie"


def decide(probabilities, offset, cell, distances, correction_used, plan):
    gl.require(type(offset) is int and 0 <= offset < 16 and cell["condition"] in CONDITIONS and
               type(cell["target"]) is int and cell["target"] in (0, 1) and type(correction_used) is bool,
               "反馈动作查询边界异常")
    p = probability(probabilities)
    gl.require(p.shape == (12,), "宏动作概率维度异常")
    worker = int(ctl.select_action(p, "mode", .5))
    manual = cell["condition"] == "visual_correct_hold" and offset >= 12
    kind = boundary_kind(distances)
    wrong = kind == f"closer_target{1 - cell['target']}"
    correction = bool(manual and wrong and not correction_used)
    action = plan["turn_up_action" if cell["target"] == 0 else "turn_down_action"] if correction else 0 if manual else worker
    return dict(worker_action=worker, actual_action=int(action),
        decision_source="visual_correction_once" if correction else "visual_noop" if manual else "worker_mode",
        intervention_applied=manual, action_differs_from_worker=bool(action != worker),
        correction_used_before=correction_used, correction_applied=correction,
        phase="feedback_window4" if offset >= 12 else "autonomous12")


def trace_identity(plan, index):
    cell = plan["design"]["schedule"][index]
    return dict(comparison_protocol=PROTOCOL, design_id=plan["design_id"], model_id=plan["input_identity"]["model_id"],
        worker_version=350, source_worker_version=300, additional_updates=50, trial_index=index,
        condition=cell["condition"], issued_target=cell["target"], repeat=cell["repeat"], seed=cell["seed"],
        control_start_frame=64, feedback_start_frame=76, episode_split=split(cell["repeat"]), execution_policy="mode")


def validate_trace(arrays, events, trace, plan, index):
    validate_plan(plan)
    cell = plan["design"]["schedule"][index]
    gl.require(all(trace.get(k) == v for k, v in trace_identity(plan, index).items()), "保存模型/trial/目标/整局划分不同")
    n = len(trace["action_ids"])
    gl.require(1 <= n <= 16 and len(arrays["image"]) == 65 + n and
        arrays["features"].shape == (65 + n, 5120) and arrays["features"].dtype == np.float32 and
        np.isfinite(arrays["features"]).all() and
        (n == 16 or bool(arrays["is_last"][-1] or arrays["is_terminal"][-1])) and
        trace["remaining"] == BUDGETS[:n], "真实控制长度、结束标志或remaining异常")
    gl.require(all(type(a) is int and 0 <= a < 12 for a in trace["action_ids"]), "真实动作异常")
    calibration.validate_execution(arrays, events, plan["start_actions"], trace["action_ids"], 12)
    p, d = np.asarray(trace["probabilities"]), np.asarray(trace["distances_to_both_goals"])
    gl.require(p.shape == (n, 12) and d.shape == (n + 1, 2), "保存概率或距离形状异常")
    used, decisions = False, []
    for offset in range(n):
        item = decide(p[offset], offset, cell, d[offset], used, plan)
        decisions.append(item)
        used = used or item["correction_applied"]
    boundary_kind(d[-1])
    gl.require(trace["worker_action_ids"] == [item["worker_action"] for item in decisions], "worker建议动作与概率mode不同")
    for field, saved in DECISION_FIELDS.items():
        gl.require(trace[saved] == [item[field] for item in decisions], "实际动作/反馈来源/纠偏次数/阶段不同")
    gl.require(trace["action_seed"] == action_seed(plan, index) and np.array_equal(trace["uniforms"],
        np.random.RandomState(action_seed(plan, index)).uniform(size=16)[:n]), "计划动作种子或随机数不同")
    return n


def summarize(rows, plan):
    validate_plan(plan)
    gl.require(len(rows) <= 20 and len({key(r) for r in rows}) == len(rows) and
        [key(r) for r in rows] == [key(c) for c in plan["design"]["schedule"][:len(rows)]] and
        [r["trial_index"] for r in rows] == list(range(len(rows))), "记录重复、遗漏、计划外trial或顺序不同；禁止续跑补样")
    for row in rows:
        gl.require(type(row["execution_valid"]) is bool and row["episode_split"] == split(row["repeat"]), "整局划分或有效标志异常")
        if not row["execution_valid"]:
            continue
        n, target = row["actual_steps"], row["target"]
        manual = max(0, n - 12) if row["condition"] == "visual_correct_hold" else 0
        gl.require(type(n) is int and 1 <= n <= 16 and row["saved_execution_verified"] is True and
            row["video_verified"] is True and row["fixed_horizon_completed"] == (n == 16) and
            row["returned_env_steps"] == 64 + n and row["returned_control_steps"] == n and
            row["worker_action_queries"] == n and (n == 16 or row["actual_terminal"]) and
            row["end_reason"] == ("environment_done" if row["actual_terminal"] else "fixed_budget") and
            row["intervention_actions"] == manual and type(row["correction_actions"]) is int and
            0 <= row["correction_actions"] <= min(1, manual) and type(row["actual_action_changes"]) is int and
            0 <= row["actual_action_changes"] <= manual, "真实保存接口或终点/干预算异常")
        start, end = np.asarray(row["start_distances_to_both_goals"]), np.asarray(row["end_distances_to_both_goals"])
        boundary_kind(start)
        boundary_kind(end)
        gl.require(np.allclose([row["start_distance"], row["end_distance"], row["distance_improvement"], row["target_preference_margin"]],
            [start[target], end[target], start[target] - end[target], end[1 - target] - end[target]], atol=1e-6, rtol=1e-6),
            "自身进展或真实终点不同")
        if n >= 12:
            boundary = np.asarray(row["boundary_distances_to_both_goals"])
            gl.require(row["boundary_kind"] == boundary_kind(boundary) and
                np.isclose(row["window_progress"], boundary[target] - end[target]) and
                np.isclose(row["autonomous_progress"], start[target] - boundary[target]), "自身窗口进展或边界不同")
    groups = {}
    for target in (0, 1):
        for condition in CONDITIONS:
            observed = [r for r in rows if r["target"] == target and r["condition"] == condition]
            valid = [r for r in observed if r["execution_valid"]]
            full = [r for r in valid if r["fixed_horizon_completed"]]
            groups[f"target{target}_{condition}"] = dict(target=target, condition=condition, planned=5,
                attempted=len(observed), valid=len(valid), full_horizon=len(full), errors=len(observed) - len(valid),
                early_terminal=len(valid) - len(full), endpoint_closer_to_requested_target=sum(r["target_preference_margin"] > MARGIN_EPSILON for r in full),
                window_progress=statistics([r["window_progress"] for r in full]),
                window_mean_all_five=float(np.mean([r["window_progress"] for r in full])) if len(full) == 5 else None,
                whole_progress=statistics([r["distance_improvement"] for r in full]),
                whole_mean_all_five=float(np.mean([r["distance_improvement"] for r in full])) if len(full) == 5 else None,
                corrections=sum(r["correction_actions"] for r in full), actual_action_changes=sum(r["actual_action_changes"] for r in full),
                boundary_strata_diagnostic_only={kind: dict(count=sum(r["boundary_kind"] == kind for r in full),
                    window_progress=statistics([r["window_progress"] for r in full if r["boundary_kind"] == kind]))
                    for kind in ("closer_target0", "closer_target1", "tie")})
    return dict(format=EVALUATION_FORMAT, protocol=PROTOCOL, planned_trials=20, attempted_trials=len(rows),
        valid_histories=sum(r["execution_valid"] for r in rows), full_design_completed=len(rows) == 20, groups=groups,
        errors=[dict(trial_index=r["trial_index"], error=r.get("error")) for r in rows if not r["execution_valid"]],
        independent_world_seeds=1, same_hidden_state=False, matched_start_comparison=False,
        endpoint="actual frame80; feedback boundary frame76; original horizon16 unchanged",
        expert_labels_generated=False, approved_for_training=False, behavior_accepted=False, t06_approved=False)


def coverage(rows, plan):
    summarize(rows, plan)
    result = {}
    for target in (0, 1):
        for part in ("train", "development_holdout"):
            manual = [r for r in rows if r["target"] == target and r["episode_split"] == part and
                      r["condition"] == "visual_correct_hold" and r["execution_valid"] and r["fixed_horizon_completed"]]
            corrections = [r for r in manual if r["correction_actions"] == 1]
            holds = [r for r in manual if r["holding_requested_rows"] > 0]
            result[f"target{target}_{part}"] = dict(target=target, episode_split=part, planned_episodes=3 if part == "train" else 2,
                complete_episodes=len(manual), correction_episodes=len(corrections), correction_rows=len(corrections),
                holding_episodes=len(holds), holding_rows=sum(r["holding_requested_rows"] for r in holds),
                correction_step_progress=statistics([r["correction_step_progress"] for r in corrections]),
                corrective_effect_observed=bool(corrections), holding_effect_observed=bool(holds),
                corrections_then_kept_requested_preference=sum(r["correction_kept_requested_preference"] for r in corrections),
                holding_rows_then_kept_requested_preference=sum(r["hold_kept_requested_preference"] for r in holds),
                both_branches_observed=bool(corrections and holds))
    return dict(groups=result, all_target_split_branches_observed=all(g["both_branches_observed"] for g in result.values()),
        branch_counts_are_independent_episodes=True, overlapping_branches_in_one_episode=True,
        missing_effect_statistics_are_unavailable=True, resampling_creates_no_new_evidence=True,
        expert_labels_generated=False, approved_for_training=False, t06_approved=False)


def validate_endpoint_queries(arrays, trace, endpoint, queries):
    expected = [dict(state_frame=76 + offset, incoming_action_frame=77 + offset, remaining=4 - offset,
        actual_action=trace["action_ids"][12 + offset], decision_source=trace["decision_sources"][12 + offset])
        for offset in range(4)]
    gl.require(queries == expected and all(q["remaining"] == 80 - q["state_frame"] and
        q["actual_action"] == int(arrays["action"][q["incoming_action_frame"]].argmax()) for q in queries),
        "事实索引必须为76–79→incoming77–80/remaining4–1，不能移植标签或终点后查询")
    gl.require(set(endpoint) == {"image", "heatmap", "goal_feature"} and
        np.array_equal(endpoint["image"], arrays["image"][80]) and
        np.array_equal(endpoint["heatmap"], arrays["heatmap"][80]) and
        endpoint["goal_feature"].dtype == np.float32 and endpoint["goal_feature"].ndim == 1 and
        np.isfinite(endpoint["goal_feature"]).all() and np.isclose(np.linalg.norm(endpoint["goal_feature"]), 1, atol=1e-5),
        "监督目标必须绑定本局真实第80帧，失败不能替换为意向目标")
    return expected
