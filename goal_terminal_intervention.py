"""Predeclared last-action development experiment on fresh autonomous histories."""

import numpy as np

import goal_action_choice_diagnose as choices
import goal_control as ctl
import goal_library as gl
import goal_reference_calibration as calibration
from goal_control_stats import statistics


FORMAT = "ls_imagine_terminal_intervention_design_v1"
EVALUATION_FORMAT = "ls_imagine_terminal_intervention_evaluation_v1"
PROTOCOL = "worker300_goal1_first15_mode_last_mode_or_forward_v1"
CONDITIONS = ("worker_mode", "fixed_forward")
REPETITIONS = 5
RANDOMIZATION_SEED = 2


def schedule():
    rng = np.random.RandomState(RANDOMIZATION_SEED)
    return [dict(seed=0, repeat=repeat, target=1, condition=CONDITIONS[int(index)])
            for repeat in range(REPETITIONS) for index in rng.permutation(2)]


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "target", "condition"))


def validate_plan(plan):
    design = plan["design"]
    forward = plan.get("forward_action")
    gl.require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL and
               design == dict(seed=0, repetitions=REPETITIONS, randomization_seed=RANDOMIZATION_SEED,
                              conditions=list(CONDITIONS), execution_policy="mode", schedule=schedule()) and
               plan["worker_version"] == 300 and plan["repair_updates"] == 100 and
               plan["warmup_steps"] == 32 and len(plan["prefix_actions"]) == 32 and
               plan["start_actions"] == [0] * 32 + plan["prefix_actions"] and
               plan["control_start_frame"] == 64 and plan["horizon"] == 16 and
               plan["autonomous_steps_before_intervention"] == 15 and plan["focus_target"] == 1 and
               len(plan["action_names"]) == 12 and plan["action_names"][0] == "noop" and
               type(forward) is int and 0 <= forward < 12 and plan["action_names"][forward] == "forward" and
               plan["maximum_new_env_steps"] == 800 and plan["same_hidden_state"] is False and
               plan["matched_start_comparison"] is False and plan["behavior_accepted"] is False and
               plan["t06_approved"] is False and all(type(action) is int and 0 <= action < 12
                                                    for action in plan["start_actions"]),
               "末步干预计划、worker/动作/目标/随机顺序或预算被修改")


def action_seed(plan, index):
    validate_plan(plan)
    gl.require(type(index) is int and 0 <= index < len(plan["design"]["schedule"]), "计划trial索引异常")
    return (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)


def decide(probabilities, offset, condition, plan):
    gl.require(type(offset) is int and 0 <= offset < 16 and condition in CONDITIONS, "末步动作条件或预算异常")
    p = choices.probability(probabilities)
    gl.require(p.shape == (12,), "worker动作概率维度异常")
    worker = ctl.select_action(probabilities, "mode", .5)
    intervention = condition == "fixed_forward" and offset == 15
    return dict(worker_action=worker, actual_action=plan["forward_action"] if intervention else worker,
                decision_source="fixed_forward" if intervention else "worker_mode",
                intervention_applied=intervention,
                action_differs_from_worker=bool(intervention and worker != plan["forward_action"]))


def validate_trace(arrays, events, trace, plan, index):
    validate_plan(plan)
    seed = action_seed(plan, index)
    cell = plan["design"]["schedule"][index]
    n = len(trace["action_ids"])
    gl.require(trace.get("comparison_protocol") == PROTOCOL and trace.get("design_id") == plan["design_id"] and
               trace.get("model_id") == plan["input_identity"]["model_id"] and
               trace.get("worker_version") == 300 and trace.get("repair_updates") == 100 and
               trace.get("condition") == cell["condition"] and trace.get("issued_target") == 1 and
               trace.get("control_start_frame") == 64 and trace.get("execution_policy") == "mode",
               "保存的干预/目标/模型或trial身份不同")
    gl.require(1 <= n <= 16 and len(arrays["image"]) == 65 + n and
               arrays["features"].shape == (65 + n, 5120) and arrays["features"].dtype == np.float32 and
               (n == 16 or bool(arrays["is_last"][-1] or arrays["is_terminal"][-1])) and
               trace["remaining"] == list(range(16, 16 - n, -1)), "真实控制长度、结束标志或remaining预算异常")
    gl.require(all(type(action) is int and 0 <= action < 12 for action in trace["action_ids"]), "实际动作标签异常")
    calibration.validate_execution(arrays, events, plan["start_actions"], trace["action_ids"], 12)
    p = np.asarray(trace["probabilities"], dtype=np.float64)
    gl.require(p.shape == (n, 12), "保存worker概率形状异常")
    decisions = [decide(value, offset, cell["condition"], plan) for offset, value in enumerate(p)]
    gl.require(trace["worker_action_ids"] == [item["worker_action"] for item in decisions], "worker建议动作与概率mode不同")
    gl.require(trace["action_ids"] == [item["actual_action"] for item in decisions] and
               trace["decision_sources"] == [item["decision_source"] for item in decisions] and
               trace["intervention_applied"] == [item["intervention_applied"] for item in decisions] and
               trace["action_differs_from_worker"] == [item["action_differs_from_worker"] for item in decisions],
               "实际动作或干预来源不同；只允许最后remaining1执行固定forward")
    gl.require(trace["action_seed"] == seed and np.array_equal(trace["uniforms"],
               np.random.RandomState(seed).uniform(size=16)[:n]), "原动作种子或随机数记录不同")
    distances = np.asarray(trace["distances_to_both_goals"], dtype=np.float64)
    gl.require(distances.shape == (n + 1, 2) and np.isfinite(distances).all() and
               np.all((distances >= 0) & (distances <= 2)), "保存真实距离异常")
    return n


def summarize(rows, plan):
    validate_plan(plan)
    cells = plan["design"]["schedule"]
    gl.require(len(rows) <= 10 and len({key(row) for row in rows}) == len(rows) and
               [key(row) for row in rows] == [key(cell) for cell in cells[:len(rows)]],
               "干预记录重复、遗漏或计划外trial；禁止续跑和补样")
    for row in rows:
        gl.require(type(row["execution_valid"]) is bool, "真实执行有效标志异常")
        if not row["execution_valid"]:
            continue
        n = row["actual_steps"]
        gl.require(row["saved_execution_verified"] is True and row["video_verified"] is True and
                   type(n) is int and 1 <= n <= 16 and row["fixed_horizon_completed"] == (n == 16) and
                   row["returned_env_steps"] == 64 + n and
                   row["worker_action_queries"] == n and row["returned_control_steps"] == n and
                   row["end_reason"] == ("environment_done" if row["actual_terminal"] else "fixed_budget") and
                   (n == 16 or row["actual_terminal"]), "真实保存接口、终点、视频或步数异常")
        start, end = np.asarray(row["start_distances_to_both_goals"]), np.asarray(row["end_distances_to_both_goals"])
        gl.require(start.shape == end.shape == (2,) and np.isfinite(start).all() and np.isfinite(end).all() and
                   np.allclose([row["start_distance"], row["end_distance"], row["distance_improvement"], row["target_preference_margin"]],
                               [start[1], end[1], start[1] - end[1], end[0] - end[1]], atol=1e-6, rtol=1e-6),
                   "自身进展或真实终点测量不同")
        if n < 16:
            gl.require(row["intervention_applied"] is False and row["last_action"] is None and
                       row["last_worker_action"] is None and row["last_step_progress"] is None,
                       "提前结束不能声称已执行第16步干预")
        else:
            gl.require(row["last_action"] == (plan["forward_action"] if row["condition"] == "fixed_forward" else row["last_worker_action"]) and
                       row["intervention_applied"] == (row["condition"] == "fixed_forward") and
                       row["last_action_differs_from_worker"] == (row["last_action"] != row["last_worker_action"]) and
                       np.isfinite([row["distance_before_last"], row["last_step_progress"]]).all() and
                       np.isclose(row["last_step_progress"], row["distance_before_last"] - end[1]), "事实最后一步或干预标志不同")
    groups = {}
    for condition in CONDITIONS:
        observed = [row for row in rows if row["condition"] == condition]
        valid = [row for row in observed if row["execution_valid"]]
        complete = [row for row in valid if row["fixed_horizon_completed"]]
        groups[condition] = dict(planned=5, attempted=len(observed), valid=len(valid), full_horizon=len(complete),
            early_terminal=sum(not row["fixed_horizon_completed"] for row in valid), errors=len(observed) - len(valid),
            positive_progress_and_correct_preference=sum(row["distance_improvement"] > 0 and row["target_preference_margin"] > 0 for row in complete),
            endpoint_closer_to_target1=sum(row["target_preference_margin"] > 0 for row in complete),
            actual_action_changes=sum(row["last_action_differs_from_worker"] for row in complete),
            start_distance=statistics([row["start_distance"] for row in valid]),
            end_distance=statistics([row["end_distance"] for row in complete]),
            whole_progress=statistics([row["distance_improvement"] for row in complete]),
            last_step_progress=statistics([row["last_step_progress"] for row in complete]),
            whole_progress_mean_all_five=float(np.mean([row["distance_improvement"] for row in complete])) if len(complete) == 5 else None,
            last_step_progress_mean_all_five=float(np.mean([row["last_step_progress"] for row in complete])) if len(complete) == 5 else None,
            initial_poses=[dict(repeat=row["repeat"], **row["start_telemetry"]["pose"]) for row in valid],
            pre_last_poses=[dict(repeat=row["repeat"], **row["pre_last_telemetry"]["pose"]) for row in complete])
    differences = {name: groups["fixed_forward"][name] - groups["worker_mode"][name]
                   if all(groups[c][name] is not None for c in CONDITIONS) else None
                   for name in ("whole_progress_mean_all_five", "last_step_progress_mean_all_five")}
    return dict(format=EVALUATION_FORMAT, protocol=PROTOCOL, planned_trials=10, attempted_trials=len(rows), full_design_completed=len(rows) == 10,
        valid_histories=sum(row["execution_valid"] for row in rows), groups=groups,
        fixed_forward_minus_worker_mode=differences,
        errors=[dict(cell={name: row[name] for name in ("seed", "repeat", "target", "condition")}, error=row.get("error"))
                for row in rows if not row["execution_valid"]],
        independent_world_seeds=1, same_hidden_state=False, matched_start_comparison=False,
        endpoint="actual frame80 after fixed16 actions; real early terminal recorded separately",
        intervention_is_learned_policy=False, labels_generated=False, behavior_accepted=False, t06_approved=False,
        p_value=None, confidence_interval=None,
        interpretation=["全部独立fresh reset尝试保留，不按第15步画面/距离/历史选择条件或保留结果。",
            "两组前15步均由同一worker300/目标1/mode执行；末步仍查询worker并分别保存建议/实际动作。",
            "固定forward是预声明干预，即使恰等于worker mode也保留干预标志；不把它声明为已学会的控制。",
            "两组不共享环境/RSSM快照；组间均值差是描述性比较，不能绑定为同状态反事实收益。",
            "缺测/失败/提前结束单列，不能以少于五个完整终点形成全五次均值；不改旧30局或T06门槛。"])
