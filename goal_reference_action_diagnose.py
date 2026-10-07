"""Teacher-forced action support on complete real calibration histories.

The observed script action is a known feasible label on its own history,
not a unique optimal action or a counterfactual label for the other goal.
"""

import numpy as np

from goal_action_choice_diagnose import CONDITIONS, probability, require


FORMAT = "ls_imagine_reference_action_diagnosis_v1"


def common_prefix(scripts):
    require(len(scripts) == 2 and len(scripts[0]) == len(scripts[1]) > 0, "两脚本长度异常")
    return next((step for step, pair in enumerate(zip(*scripts)) if pair[0] != pair[1]), len(scripts[0]))


def validate_queries(arrays, trace, first, horizon, script, dimension):
    require(type(first) is int and type(horizon) is int and first >= 0 and horizon > 0 and
            len(script) == horizon and len(arrays["features"]) == first + horizon + 1 and
            arrays["features"].dtype == np.float32 and arrays["features"].ndim == 2 and
            np.isfinite(arrays["features"]).all(), "需要完整真实脚本状态与有效控制边界")
    require(all(type(action) is int and 0 <= action < dimension for action in script) and
            trace.get("execution_policy") == "fixed_script" and trace.get("start_frame") == first and
            trace.get("planned_action_ids") == trace.get("attempted_action_ids") == script and
            trace.get("sampled_actions") is False and trace.get("policy_queried") is False,
            "实际固定脚本或动作记录不同")
    expected = np.eye(dimension, dtype=np.float32)[script]
    require(arrays["action"].shape == (first + horizon + 1, dimension) and
            np.array_equal(arrays["action"][first + 1:], expected), "真实下一脚本动作与incoming不同")
    require(not (arrays["is_last"][first:first + horizon] |
                 arrays["is_terminal"][first:first + horizon]).any(), "不能查询真实结束后的动作")
    measurements = trace["measurements"]
    require(len(measurements) == horizon + 1 and
            all(row["frame"] == first + step and row["script_step"] == step and
                row["incoming_action"] == (None if step == 0 else script[step - 1])
                for step, row in enumerate(measurements)), "逐帧测量或下一动作索引不同")
    return [dict(script_step=step + 1, frame=first + step, next_action_frame=first + step + 1,
                 remaining=horizon - step, script_action=action) for step, action in enumerate(script)]


def label_metrics(probabilities, label, assigned):
    require(type(label) is int and type(assigned) is int and assigned in (0, 1) and
            set(probabilities) == set(CONDITIONS), "真实动作标签或诊断条件异常")
    result = {}
    for name in CONDITIONS:
        p = probability(probabilities[name])
        require(0 <= label < len(p), "真实动作标签越界")
        order = np.argsort(-p, kind="stable")
        rank = int(np.flatnonzero(order == label)[0]) + 1
        result.update({f"{name}_script_probability": float(p[label]),
                       f"{name}_script_nll": float(-np.log(max(p[label], np.finfo(float).tiny))),
                       f"{name}_script_rank": rank, f"{name}_script_mode_match": bool(rank == 1),
                       f"{name}_script_top3_match": bool(rank <= 3)})
    for alias, name in (("correct", f"goal{assigned}"), ("swapped", f"goal{1 - assigned}")):
        for suffix in ("probability", "nll", "rank", "mode_match", "top3_match"):
            result[f"{alias}_script_{suffix}"] = result[f"{name}_script_{suffix}"]
    result["swapped_minus_correct_nll"] = result["swapped_script_nll"] - result["correct_script_nll"]
    result["no_goal_minus_correct_nll"] = result["no_goal_script_nll"] - result["correct_script_nll"]
    return result


def aggregate(rows):
    require(bool(rows), "没有真实脚本状态可供汇总")
    names = [name for name in rows[0] if "_script_" in name or name in (
        "swapped_minus_correct_nll", "no_goal_minus_correct_nll", "goal_total_variation", "goal_mode_changed")]
    return dict(query_frames=len(rows), means={name: float(np.mean([row[name] for row in rows])) for name in names},
                goal_mode_changes=sum(row["goal_mode_changed"] for row in rows))


def summarize(rows, plan):
    require(len(rows) == 4 * plan["horizon"] and
            len({(row["run_index"], row["script_step"]) for row in rows}) == len(rows),
            "脚本查询缺失或重复")
    runs = []
    for index, cell in enumerate(plan["schedule"]):
        own = [row for row in rows if row["run_index"] == index]
        require(len(own) == plan["horizon"] and
                [row["script_step"] for row in own] == list(range(1, plan["horizon"] + 1)) and
                all(row["repeat"] == cell["repeat"] and row["assigned_target"] == cell["script"] and
                    row["frame"] == plan["control_start_frame"] + row["script_step"] - 1 and
                    row["next_action_frame"] == row["frame"] + 1 and
                    row["remaining"] == plan["horizon"] - row["script_step"] + 1 and
                    row["script_action"] == plan["scripts"][cell["script"]][row["script_step"] - 1]
                    for row in own), "查询状态/预算/下一动作与原计划不同")
        mismatch = next((row["script_step"] for row in own if not row["correct_script_mode_match"]), None)
        runs.append(dict(run_index=index, repeat=cell["repeat"], assigned_target=cell["script"],
                         first_teacher_forced_mode_mismatch_step=mismatch, **aggregate(own)))
    shared = common_prefix(plan["scripts"])
    phases = {"common_script_prefix": [row for row in rows if row["script_step"] <= shared],
              "after_script_divergence": [row for row in rows if row["script_step"] > shared]}
    return dict(format=FORMAT, query_frames=len(rows), real_histories=len(runs), independent_world_seeds=1,
        script_common_prefix_steps=shared, first_script_divergence_step=shared + 1 if shared < plan["horizon"] else None,
        frame_weighted=aggregate(rows), per_run=runs,
        equal_run_means={name: float(np.mean([run["means"][name] for run in runs])) for name in runs[0]["means"]},
        by_assigned_target={str(target): aggregate([row for row in rows if row["assigned_target"] == target]) for target in (0, 1)},
        by_phase={name: aggregate(group) for name, group in phases.items() if group},
        by_script_step={str(step): aggregate([row for row in rows if row["script_step"] == step])
                        for step in range(1, plan["horizon"] + 1)},
        by_observed_action={name: aggregate([row for row in rows if row["script_action"] == action])
                            for action, name in enumerate(plan["action_names"])
                            if any(row["script_action"] == action for row in rows)},
        teacher_forced_real_histories=True, same_state_within_query=True, same_state_across_runs=False,
        observed_script_is_unique_optimal_label=False, swapped_goal_has_counterfactual_label=False,
        counterfactual_rollouts_executed=False, independent_query_samples_claimed=False,
        behavior_accepted=False, t06_approved=False,
        interpretation=["按真实脚本历史查询下一动作支持；改变目标不改变该局状态或remaining。",
                        "NLL收益为对该局已知可行动作的支持差异；不是另一目标的反事实动作正确率。",
                        "共同动作前缀允许相同mode；首次不匹配仅是轨迹内预测指标，不是实际闭环分歧。",
                        "64次查询来自4条历史、一个世界；不独立计样、不证明目标控制或批准T06。"])
