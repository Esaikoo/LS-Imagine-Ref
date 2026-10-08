"""Read-only terminal action comparisons on each autonomous real history.

An observed autonomous action is a factual label, never an expert target.
All comparisons keep that history's state and actual remaining budget.
"""

import numpy as np

import goal_action_choice_diagnose as choices


FORMAT = "ls_imagine_repaired_terminal_action_diagnosis_v1"
FOCUS_REMAINING = (2, 1)


def validate_queries(arrays, trace, measurements, plan, cell, index, dimension):
    first, horizon = plan["control_start_frame"], plan["horizon"]
    choices.require(horizon == 16 and len(trace["action_ids"]) == horizon and
                    arrays["features"].shape == (first + horizon + 1, 5120) and
                    arrays["features"].dtype == np.float32 and np.isfinite(arrays["features"]).all(),
                    "需要全部完整16步及本局真实5120维状态")
    choices.require(trace.get("execution_policy") == "mode" and
                    trace.get("comparison_protocol") == plan["comparison_protocol"] and
                    trace.get("design_id") == plan["design_id"] and
                    trace.get("worker_version") == 300 and trace.get("repair_updates") == 100 and
                    trace.get("control_start_frame") == first and
                    trace.get("model_id") == plan["input_identity"]["model_id"],
                    "真实控制模型/计划或mode执行身份不同")
    choices.validate_trace(arrays, trace, first, horizon, dimension, cell, index)
    choices.require(len(measurements) == horizon + 1 and all(
        row["frame"] == first + step and row["control_step"] == step
        for step, row in enumerate(measurements)), "保存测量与真实动作边界不同")
    distances = np.asarray([[row[f"distance_{goal}"] for goal in (0, 1)] for row in measurements])
    choices.require(np.allclose(distances, trace["distances_to_both_goals"], atol=choices.ATOL, rtol=choices.RTOL),
                    "保存视觉测量与原控制距离不同")
    p = np.asarray(trace["probabilities"])
    choices.require(np.argmax(p, axis=1).tolist() == trace["action_ids"], "真实下一动作不是保存分布的mode")
    return [dict(control_step=step + 1, frame=first + step, next_action_frame=first + step + 1,
                 remaining=horizon - step, observed_action=action)
            for step, action in enumerate(trace["action_ids"]) if horizon - step in FOCUS_REMAINING]


def action_metrics(probabilities, observed_action, evaluated_target):
    choices.require(set(probabilities) == set(choices.CONDITIONS) and
                    type(observed_action) is int and evaluated_target in (0, 1), "动作查询条件或标签异常")
    result = {}
    for name in choices.CONDITIONS:
        p = choices.probability(probabilities[name])
        choices.require(0 <= observed_action < len(p), "真实自主动作标签越界")
        rank = int(np.flatnonzero(np.argsort(-p, kind="stable") == observed_action)[0]) + 1
        result[f"{name}_observed_probability"] = float(p[observed_action])
        result[f"{name}_observed_rank"] = rank
    for alias, name in (("correct", f"goal{evaluated_target}"), ("swapped", f"goal{1 - evaluated_target}")):
        result.update({f"{alias}_{key}": result[f"{name}_{key}"]
                       for key in ("observed_probability", "observed_rank")})
        top = choices.top_two(probabilities[name])
        result.update({f"{alias}_{key}": top[key] for key in ("mode", "runner_up", "gap")})
    result["correct_mode_matches_observed"] = result["correct_mode"] == observed_action
    return result


def transition(before, after, target):
    choices.require(target in (0, 1), "评价目标异常")
    values = np.asarray([row[f"distance_{goal}"] for row in (before, after) for goal in (0, 1)])
    choices.require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 2)), "实际两帧距离异常")
    pre_margin = float(before[f"distance_{1 - target}"] - before[f"distance_{target}"])
    post_margin = float(after[f"distance_{1 - target}"] - after[f"distance_{target}"])
    # A descriptive strict sign change, not a new success threshold.
    return dict(distance_before=float(before[f"distance_{target}"]),
                distance_after=float(after[f"distance_{target}"]),
                actual_step_progress=float(before[f"distance_{target}"] - after[f"distance_{target}"]),
                preference_margin_before=pre_margin, preference_margin_after=post_margin,
                preference_reversed_away=bool(pre_margin > 0 and post_margin < 0),
                pitch_error_before=float(before[f"pitch_error_{target}"]),
                pitch_error_after=float(after[f"pitch_error_{target}"]),
                position_error_before=float(before[f"position_error_{target}"]),
                position_error_after=float(after[f"position_error_{target}"]))


def reference_context(state, references, remaining):
    own = np.asarray(state, dtype=np.float64).reshape(-1)
    choices.require(own.shape == (5120,) and np.isfinite(own).all() and np.linalg.norm(own) > 0,
                    "自主当前状态形状或范数异常")
    result = {}
    for target in (0, 1):
        matches = [row for row in references if row["assigned_target"] == target and row["remaining"] == remaining]
        choices.require(len(matches) == 2, "同预算参考历史缺失或重复")
        distances = []
        for row in matches:
            reference = np.asarray(row["state"], dtype=np.float64).reshape(-1)
            choices.require(reference.shape == own.shape and np.isfinite(reference).all() and np.linalg.norm(reference) > 0,
                            "参考状态形状或范数异常")
            distances.append(float(np.clip(1 - own @ reference / (np.linalg.norm(own) * np.linalg.norm(reference)), 0, 2)))
        nearest = int(np.argmin(distances))
        result.update({f"reference_target{target}_nearest_state_cosine_distance": distances[nearest],
                       f"reference_target{target}_nearest_run": matches[nearest]["run_index"],
                       f"reference_target{target}_nearest_repeat": matches[nearest]["repeat"],
                       f"reference_target{target}_observed_script_action": matches[nearest]["observed_script_action_name"]})
    return result


def summarize(rows, plan):
    schedule = plan["design"]["schedule"]
    choices.require(len(rows) == 2 * len(schedule) == 60 and
                    len({(row["trial_index"], row["remaining"]) for row in rows}) == 60,
                    "末尾查询有遗漏、重复或计划外trial")
    for index, cell in enumerate(schedule):
        own = [row for row in rows if row["trial_index"] == index]
        choices.require([row["remaining"] for row in own] == list(FOCUS_REMAINING) and all(
            row["repeat"] == cell["repeat"] and row["evaluated_target"] == cell["target"] and
            row["factual_condition"] == cell["mode"] and
            row["frame"] == plan["control_start_frame"] + plan["horizon"] - row["remaining"] and
            row["next_action_frame"] == row["frame"] + 1 for row in own),
            "末尾查询的状态/目标/预算/下一动作不同")
    last = [row for row in rows if row["remaining"] == 1]
    return dict(format=FORMAT, real_histories=30, query_frames=60, independent_world_seeds=1,
        same_state_within_query=True, same_state_across_trials=False,
        by_remaining={str(budget): choices.aggregate([row for row in rows if row["remaining"] == budget], 12)
                      for budget in FOCUS_REMAINING},
        by_factual_condition={mode: choices.aggregate([row for row in rows if row["factual_condition"] == mode], 12)
                              for mode in ("goal", "no_goal", "swapped_goal")},
        last_step_by_target={str(target): dict(
            goal_mode_changes=sum(row["goal_mode_changed"] for row in last
                                  if row["factual_condition"] == "goal" and row["evaluated_target"] == target),
            real_preference_reversals=sum(row["preference_reversed_away"] for row in last
                                  if row["factual_condition"] == "goal" and row["evaluated_target"] == target),
            trials=[row for row in last if row["factual_condition"] == "goal" and row["evaluated_target"] == target])
                             for target in (0, 1)},
        observed_actions_are_expert_labels=False, reference_actions_transferred_as_labels=False,
        reference_distance_has_ood_threshold=False, changed_budget_queries=False,
        counterfactual_rollouts_executed=False, new_env_steps=0, optimizer_updates=0,
        behavior_accepted=False, t06_approved=False,
        interpretation=["全部30局保留；60次末尾查询不是60个独立样本。",
            "概率/排名针对已执行自主动作，不把该动作当专家标签，也不推断另一个目标的正确动作。",
            "只有事实动作的前后观测是实际结果；换目标后的mode没有执行，不能给它绑定实际收益。",
            "remaining2/1严格采用原记录；不改成15步，不选最佳中间帧或重分类数值门槛。"])
