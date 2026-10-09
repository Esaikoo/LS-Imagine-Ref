"""Same-state goal/version comparisons; observed actions are facts, not experts."""
import numpy as np

import goal_action_choice_diagnose as choices
import goal_continuation_control as confirmation
import goal_terminal_action_diagnose as terminal

FORMAT = "ls_imagine_continuation_action_diagnosis_v1"
VERSIONS = ("300", "350")


def validate_saved_history(metrics, history, row, first, total_frames):
    """Validate the autonomous recorder's schema, not the reference-history schema."""
    choices.require(type(first) is int and type(total_frames) is int and 0 <= first < total_frames,
                    "真实因果历史边界异常")
    choices.require(metrics == row, "逐trial指标与原manifest不同")
    choices.require(isinstance(history, dict) and
                    history.get("own_causal_history_passed") is True and
                    history.get("reset_is_real") is True and
                    history.get("comparison_to_other_histories") is False and
                    history.get("start_frame") == first and history.get("total_frames") == total_frames,
                    "必须保留本局已验收真实因果历史/重置/边界，不能复制另一局状态")
    error = history.get("maximum_absolute_error")
    choices.require(type(error) in (int, float) and np.isfinite(error) and error >= 0 and
                    type(row.get("own_history_max_error")) in (int, float) and
                    error == row["own_history_max_error"], "真实因果历史误差缺失、无效或与原指标不同")
    choices.require(row.get("saved_execution_verified") is True and row.get("video_verified") is True and
                    isinstance(row.get("video"), dict) and row["video"].get("frames") == total_frames,
                    "必须保留本局已验收执行与完整解码视频")


def validate_queries(arrays, events, trace, measured, plan, index):
    choices.require(type(plan.get("design", {}).get("randomization_seed")) is int and
                    plan["design"]["randomization_seed"] == 4, "真实计划随机种子缺失或不同")
    choices.require(type(index) is int and 0 <= index < 30, "查询trial索引异常")
    cell = plan["design"]["schedule"][index]
    count = confirmation.validate_trace(arrays, events, trace, plan, cell, index, 12)
    choices.require(count == 16 and arrays["features"].shape == (81, 5120) and
                    arrays["features"].dtype == np.float32, "需要完整16步真实5120维因果状态")
    # Legacy field adaptation is in a fresh copy; preserve source schedule bytes.
    adapted = dict(cell, randomization_seed=plan["design"]["randomization_seed"])
    choices.validate_trace(arrays, trace, 64, 16, 12, adapted, index)
    choices.require(len(measured) == 17 and all(
        row["frame"] == 64 + offset and row["control_step"] == offset and
        row["incoming_action_id"] == int(np.argmax(arrays["action"][64 + offset])) and
        row["next_action_id"] == (trace["action_ids"][offset] if offset < 16 else None) and
        row["next_action_name"] == (plan["action_names"][trace["action_ids"][offset]] if offset < 16 else None) and
        row["remaining"] == (16 - offset if offset < 16 else None) and
        row["actual_next_action_probability"] == (trace["probabilities"][offset][trace["action_ids"][offset]]
                                                  if offset < 16 else None)
        for offset, row in enumerate(measured)), "保存测量/下一动作/remaining或真实终点后查询不同")
    choices.require(np.allclose([[m[f"distance_{g}"] for g in (0, 1)] for m in measured],
                               trace["distances_to_both_goals"], atol=choices.ATOL, rtol=choices.RTOL),
                    "保存测量与真实距离不同")
    return [dict(control_step=offset, frame=64 + offset, next_action_frame=65 + offset,
                 remaining=16 - offset, observed_action=action)
            for offset, action in enumerate(trace["action_ids"])]


def validate_query_row(row, plan):
    index = row.get("trial_index")
    choices.require(type(index) is int and 0 <= index < 30, "查询trial索引异常")
    cell = plan["design"]["schedule"][index]
    step = row.get("control_step")
    choices.require(type(step) is int and 0 <= step < 16 and row.get("frame") == 64 + step and
                    row.get("next_action_frame") == 65 + step and row.get("remaining") == 16 - step and
                    row.get("repeat") == cell["repeat"] and row.get("evaluated_target") == cell["target"] and
                    row.get("factual_condition") == cell["mode"], "查询真实状态/目标/预算/下一动作身份不同")


def version_metrics(probabilities, raw_preferences, action, target):
    result = dict(choices.frame_metrics(probabilities, raw_preferences),
                  **terminal.action_metrics(probabilities, action, target))
    for name in choices.CONDITIONS:
        p = choices.probability(probabilities[name])
        result[f"{name}_observed_nll"] = float(-np.log(max(p[action], np.finfo(float).tiny)))
    for alias, name in (("correct", f"goal{target}"), ("swapped", f"goal{1 - target}")):
        result[f"{alias}_observed_nll"] = result[f"{name}_observed_nll"]
    result["swapped_minus_correct_observed_nll"] = result["swapped_observed_nll"] - result["correct_observed_nll"]
    result["no_goal_minus_correct_observed_nll"] = result["no_goal_observed_nll"] - result["correct_observed_nll"]
    return result


def version_difference(old, new, observed, target):
    choices.require(set(old) == set(new) == set(choices.CONDITIONS), "版本比较条件缺失")
    result = {}
    for name in choices.CONDITIONS:
        before, after = choices.probability(old[name]), choices.probability(new[name])
        difference = choices.distribution_difference(before, after)
        result[f"{name}_mode_changed"] = difference["mode_changed"]
        result[f"{name}_total_variation"] = difference["total_variation"]
        result[f"{name}_observed_probability_delta"] = float(after[observed] - before[observed])
        result[f"{name}_observed_nll_delta"] = float(-np.log(after[observed]) + np.log(before[observed]))
    choices.require(np.array_equal(old["base"], new["base"]), "同状态两个版本的冻结底座概率不同")
    result["correct_mode_changed"] = result[f"goal{target}_mode_changed"]
    result["swapped_mode_changed"] = result[f"goal{1 - target}_mode_changed"]
    return result


def aggregate(rows):
    choices.require(bool(rows), "缺失覆盖不能填成零")
    result = dict(query_frames=len(rows), real_histories=len({r["trial_index"] for r in rows}),
                  actual_preference_reversals=sum(r["preference_reversed_away"] for r in rows))
    for version in VERSIONS:
        first = rows[0]["version_metrics"][version]
        keys = [k for k in first if k.endswith(("observed_nll", "observed_probability", "observed_rank")) or
                k in ("goal_total_variation", "goal_mode_changed", "no_goal_minus_correct_observed_nll",
                      "swapped_minus_correct_observed_nll")]
        result[f"worker{version}"] = dict(
            means={key: float(np.mean([row["version_metrics"][version][key] for row in rows])) for key in keys},
            goal_mode_changes=sum(row["version_metrics"][version]["goal_mode_changed"] for row in rows),
            condition_mode_histograms={condition: np.bincount([
                row["version_metrics"][version][f"{condition}_mode"] for row in rows], minlength=12).tolist()
                for condition in choices.CONDITIONS})
    result["version_change_means"] = {key: float(np.mean([row["version_change"][key] for row in rows]))
                                     for key in rows[0]["version_change"]}
    return result


def summarize(rows, trials, plan):
    choices.require(len(rows) == 480 and len(trials) == 30 and
                    [(r["trial_index"], r["control_step"]) for r in rows] ==
                    [(trial, step) for trial in range(30) for step in range(16)] and
                    [t["trial_index"] for t in trials] == list(range(30)),
                    "全部查询有遗漏、重复、计划外trial或顺序不同")
    for row in rows:
        validate_query_row(row, plan)
    group = lambda selected: aggregate(selected) if selected else dict(available=False, query_frames=0)
    last = [r for r in rows if r["remaining"] == 1]
    goal_last = [r for r in last if r["factual_condition"] == "goal"]
    success = {t["trial_index"]: t["original_direction_passed"] for t in trials}
    return dict(format=FORMAT, query_frames=480, real_histories=30, distributions_saved=4800,
        terminal_query_frames=60, independent_world_seeds=1, all_frames=aggregate(rows),
        per_trial=[dict(trial_index=index, **aggregate([r for r in rows if r["trial_index"] == index]))
                   for index in range(30)],
        by_remaining={str(b): aggregate([r for r in rows if r["remaining"] == b]) for b in range(16, 0, -1)},
        by_target_condition={f"{target}_{condition}": aggregate([r for r in rows if
            r["evaluated_target"] == target and r["factual_condition"] == condition])
            for target in (0, 1) for condition in ("goal", "no_goal", "swapped_goal")},
        goal_last_step_by_outcome={f"{target}_{'pass' if passed else 'fail'}": group([r for r in goal_last if
            r["evaluated_target"] == target and success[r["trial_index"]] == passed])
            for target in (0, 1) for passed in (False, True)},
        last_step_rows=last, same_state_within_query=True, same_state_across_trials=False,
        versions_compared_on_worker350_generated_states=True, observed_actions_are_expert_labels=False,
        version_probability_difference_is_causal_control_effect=False, counterfactual_rollouts_executed=False,
        independently_sampled_frames=False, corrective_generalization_verified=False,
        optimizer_updates=0, new_env_steps=0, behavior_accepted=False, t06_approved=False,
        interpretation=["事实动作NLL描述已执行动作的支持，不把失败动作当正确标签。",
                        "300与350使用同一批350生成的状态；概率差不等于实际终点收益。",
                        "全部480状态与原remaining16–1保留；末步观察只作诊断，不改变16步门槛。"])
