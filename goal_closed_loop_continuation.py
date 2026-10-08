"""Fixed-budget visual feedback continuations; no expert or success labels."""

import numpy as np

import goal_action_choice_diagnose as choices
import goal_control as ctl
import goal_library as gl
import goal_reference_calibration as calibration
from goal_control_stats import statistics

FORMAT = "ls_imagine_closed_loop_continuation_design_v1"
EVALUATION_FORMAT = "ls_imagine_closed_loop_continuation_evaluation_v1"
PROTOCOL = "worker300_goal1_full16_then4_worker_or_visual_down_once_noop_v1"
CONDITIONS = ("worker_continuation", "visual_correct_hold")
REPETITIONS = 5
RANDOMIZATION_SEED = 3
BUDGETS = list(range(16, 0, -1)) + list(range(4, 0, -1))
MARGIN_EPSILON = 1e-7


def schedule():
    rng = np.random.RandomState(RANDOMIZATION_SEED)
    return [dict(seed=0, repeat=repeat, target=1, condition=CONDITIONS[int(index)])
            for repeat in range(REPETITIONS) for index in rng.permutation(2)]


def split(repeat):
    gl.require(type(repeat) is int and 0 <= repeat < 5, "整局划分repeat异常")
    return "train" if repeat < 3 else "development_holdout"


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "target", "condition"))


def validate_plan(plan):
    gl.require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL and
        plan["design"] == dict(seed=0, repetitions=5, randomization_seed=3, conditions=list(CONDITIONS),
                               execution_policy="mode", schedule=schedule()) and
        plan["worker_version"] == 300 and plan["repair_updates"] == 100 and
        plan["warmup_steps"] == 32 and len(plan["prefix_actions"]) == 32 and
        plan["start_actions"] == [0] * 32 + plan["prefix_actions"] and
        all(type(a) is int and 0 <= a < 12 for a in plan["start_actions"]) and
        plan["control_start_frame"] == 64 and plan["horizon"] == 16 and
        plan["continuation_start_frame"] == 80 and plan["continuation_horizon"] == 4 and
        plan["actual_endpoint_frame"] == 84 and plan["focus_target"] == 1 and
        len(plan["action_names"]) == 12 and
        plan["feedback_rule"] == dict(input="current_real_visual_distances_only", margin_epsilon=MARGIN_EPSILON,
            correction="turn_down", correction_limit=1, hold="noop", stop_on_success=False) and
        plan["action_names"][plan["noop_action"]] == "noop" and plan["noop_action"] == 0 and
        type(plan["turn_down_action"]) is int and 0 <= plan["turn_down_action"] < 12 and
        plan["action_names"][plan["turn_down_action"]] == "turn_down" and
        plan["episode_split"] == {str(i): split(i) for i in range(5)} and
        plan["maximum_new_env_steps"] == 840 and plan["same_hidden_state"] is False and
        plan["matched_start_comparison"] is False and plan["behavior_accepted"] is False and
        plan["t06_approved"] is False, "续段计划、反馈规则、划分或预算被修改")


def action_seed(plan, index):
    validate_plan(plan)
    gl.require(type(index) is int and 0 <= index < 10, "计划trial索引异常")
    return (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)


def boundary_kind(distances):
    d = np.asarray(distances, dtype=np.float64)
    gl.require(d.shape == (2,) and np.isfinite(d).all() and np.all((d >= 0) & (d <= 2)), "当前真实视觉距离异常")
    margin = float(d[0] - d[1])
    return "closer_target1" if margin > MARGIN_EPSILON else "closer_target0" if margin < -MARGIN_EPSILON else "tie"


def decide(probabilities, offset, condition, distances, correction_used, plan):
    gl.require(type(offset) is int and 0 <= offset < 20 and condition in CONDITIONS and
               type(correction_used) is bool, "续段动作条件或预算异常")
    p = choices.probability(probabilities)
    gl.require(p.shape == (12,), "worker动作概率维度异常")
    kind = boundary_kind(distances)
    worker = ctl.select_action(p, "mode", .5)
    manual = offset >= 16 and condition == "visual_correct_hold"
    correction = manual and not correction_used and kind == "closer_target0"
    actual = plan["turn_down_action"] if correction else plan["noop_action"] if manual else worker
    return dict(worker_action=worker, actual_action=actual,
        decision_source="visual_turn_down_once" if correction else "visual_noop_hold" if manual else "worker_mode",
        intervention_applied=manual, action_differs_from_worker=bool(actual != worker),
        correction_used_before=correction_used, correction_applied=bool(correction),
        phase="autonomous16" if offset < 16 else "continuation4")


def validate_trace(arrays, events, trace, plan, index):
    validate_plan(plan)
    cell = plan["design"]["schedule"][index]
    n = len(trace["action_ids"])
    gl.require(trace.get("comparison_protocol") == PROTOCOL and trace.get("design_id") == plan["design_id"] and
        trace.get("model_id") == plan["input_identity"]["model_id"] and trace.get("worker_version") == 300 and
        trace.get("repair_updates") == 100 and trace.get("condition") == cell["condition"] and
        trace.get("issued_target") == 1 and trace.get("control_start_frame") == 64 and
        trace.get("continuation_start_frame") == 80 and trace.get("execution_policy") == "mode" and
        trace.get("episode_split") == split(cell["repeat"]), "保存的续段/模型/trial身份或划分不同")
    gl.require(1 <= n <= 20 and len(arrays["image"]) == 65 + n and
        arrays["features"].shape == (65 + n, 5120) and arrays["features"].dtype == np.float32 and
        (n == 20 or bool(arrays["is_last"][-1] or arrays["is_terminal"][-1])) and
        trace["remaining"] == BUDGETS[:n], "真实控制长度、结束标志或两阶段remaining异常")
    gl.require(all(type(a) is int and 0 <= a < 12 for a in trace["action_ids"]), "实际动作标签异常")
    calibration.validate_execution(arrays, events, plan["start_actions"], trace["action_ids"], 12)
    p = np.asarray(trace["probabilities"], dtype=np.float64)
    d = np.asarray(trace["distances_to_both_goals"], dtype=np.float64)
    gl.require(p.shape == (n, 12) and d.shape == (n + 1, 2) and np.isfinite(d).all() and
               np.all((d >= 0) & (d <= 2)), "保存概率或真实距离形状异常")
    used, decisions = False, []
    for offset in range(n):
        item = decide(p[offset], offset, cell["condition"], d[offset], used, plan)
        decisions.append(item)
        used = used or item["correction_applied"]
    gl.require(trace["worker_action_ids"] == [item["worker_action"] for item in decisions], "worker建议动作与概率mode不同")
    for field, saved in (("actual_action", "action_ids"), ("decision_source", "decision_sources"),
        ("intervention_applied", "intervention_applied"), ("action_differs_from_worker", "action_differs_from_worker"),
        ("correction_used_before", "correction_used_before"), ("correction_applied", "correction_applied"), ("phase", "phases")):
        gl.require(trace[saved] == [item[field] for item in decisions], "实际续段动作/反馈来源/纠偏次数或阶段不同")
    seed = action_seed(plan, index)
    gl.require(trace["action_seed"] == seed and np.array_equal(trace["uniforms"],
               np.random.RandomState(seed).uniform(size=20)[:n]), "原动作种子或随机数记录不同")
    return n


def summarize(rows, plan):
    validate_plan(plan)
    gl.require(len(rows) <= 10 and len({key(row) for row in rows}) == len(rows) and
        [key(row) for row in rows] == [key(cell) for cell in plan["design"]["schedule"][:len(rows)]],
        "续段记录重复、遗漏或计划外trial；禁止续跑和补样")
    for row in rows:
        gl.require(type(row["execution_valid"]) is bool and row["episode_split"] == split(row["repeat"]), "整局划分或有效标志异常")
        if not row["execution_valid"]:
            continue
        n = row["actual_steps"]
        gl.require(type(n) is int and 1 <= n <= 20 and row["saved_execution_verified"] is True and
            row["video_verified"] is True and row["fixed_horizon_completed"] == (n == 20) and
            row["returned_env_steps"] == 64 + n and row["returned_control_steps"] == n and
            row["worker_action_queries"] == n and (n == 20 or row["actual_terminal"]) and
            row["end_reason"] == ("environment_done" if row["actual_terminal"] else "fixed_budget"), "真实保存接口或终点预算异常")
        manual = max(0, n - 16) if row["condition"] == "visual_correct_hold" else 0
        gl.require(row["autonomous_steps"] == min(16, n) and row["continuation_steps"] == max(0, n - 16) and
            row["intervention_actions"] == manual and type(row["correction_actions"]) is int and
            0 <= row["correction_actions"] <= min(1, manual) and type(row["actual_action_changes"]) is int and
            0 <= row["actual_action_changes"] <= manual, "两阶段动作或实际纠偏计数异常")
        start, end = np.asarray(row["start_distances_to_both_goals"]), np.asarray(row["end_distances_to_both_goals"])
        gl.require(start.shape == end.shape == (2,) and np.isfinite([start, end]).all() and
            np.allclose([row["start_distance"], row["end_distance"], row["distance_improvement"], row["target_preference_margin"]],
                        [start[1], end[1], start[1] - end[1], end[0] - end[1]], atol=1e-6, rtol=1e-6), "自身进展或终点测量不同")
        if n >= 16:
            boundary = np.asarray(row["boundary_distances_to_both_goals"])
            gl.require(boundary.shape == (2,) and row["boundary_kind"] == boundary_kind(boundary) and
                np.isclose(row["continuation_progress"], boundary[1] - end[1]) and
                np.isclose(row["autonomous_progress"], start[1] - boundary[1]), "两阶段自身进展或边界不同")
    groups = {}
    for condition in CONDITIONS:
        observed = [r for r in rows if r["condition"] == condition]
        valid = [r for r in observed if r["execution_valid"]]
        full = [r for r in valid if r["fixed_horizon_completed"]]
        groups[condition] = dict(planned=5, attempted=len(observed), valid=len(valid), full_horizon=len(full),
            errors=len(observed) - len(valid), early_terminal=len(valid) - len(full),
            endpoint_closer_to_target1=sum(r["target_preference_margin"] > MARGIN_EPSILON for r in full),
            positive_continuation_progress=sum(r["continuation_progress"] > 0 for r in full),
            continuation_progress=statistics([r["continuation_progress"] for r in full]),
            continuation_mean_all_five=float(np.mean([r["continuation_progress"] for r in full])) if len(full) == 5 else None,
            whole_progress=statistics([r["distance_improvement"] for r in full]),
            corrections=sum(r["correction_actions"] for r in full),
            intervention_actions=sum(r["intervention_actions"] for r in full),
            actual_action_changes=sum(r["actual_action_changes"] for r in full),
            boundary_strata_diagnostic_only={kind: dict(count=sum(r["boundary_kind"] == kind for r in full),
                end_distance=statistics([r["end_distance"] for r in full if r["boundary_kind"] == kind]),
                continuation_progress=statistics([r["continuation_progress"] for r in full if r["boundary_kind"] == kind]),
                closer_target1_at_end=sum(r["target_preference_margin"] > MARGIN_EPSILON for r in full if r["boundary_kind"] == kind))
                for kind in ("closer_target0", "closer_target1", "tie")})
    return dict(format=EVALUATION_FORMAT, protocol=PROTOCOL, planned_trials=10, attempted_trials=len(rows),
        full_design_completed=len(rows) == 10, valid_histories=sum(r["execution_valid"] for r in rows), groups=groups,
        errors=[dict(trial_index=r["trial_index"], error=r.get("error")) for r in rows if not r["execution_valid"]],
        independent_world_seeds=1, same_hidden_state=False, matched_start_comparison=False,
        endpoint="actual frame84; original autonomous16 boundary is separately retained at frame80",
        labels_generated=False, expert_labels_generated=False, intervention_is_learned_policy=False,
        p_value=None, confidence_interval=None,
        behavior_accepted=False, t06_approved=False,
        interpretation=["两组各五次独立fresh reset，先完成相同worker300/目标1的16步；续段是新的4步预算。",
            "手工视觉反馈最多一次turn_down，其余noop；规则只读当前真实画面对两目标的距离，不使用位置/朝向。",
            "视觉更近不等于已经到达；noop也不保证画面/物理状态不变，必须核对真实终点与视频。",
            "所有尝试保留；边界分类只用于事后诊断，不能筛选起点、补样或形成配对反事实结论。",
            "整局repeat0–2训练、3–4开发留出，未来监督只能绑定实际动作及各局实际终点。",
            "新增手工数据不是专家最优标签、新worker确认或T06准入证据；不修改原16步门槛。"])
