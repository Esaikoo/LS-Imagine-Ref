"""Read-only fitting diagnostics for factual actions inside the original horizon."""
import numpy as np

import goal_action_choice_diagnose as choices
import goal_continuation_action_diagnose as history_schema
import goal_library as gl
import goal_within_horizon_feedback as feedback

FORMAT = "ls_imagine_within_horizon_supervision_check_v1"
PROTOCOL = "worker350_saved_frames76_79_own_endpoint80_vs_fixed_goals_readonly_v1"
CONDITIONS = ("actual_endpoint", "fixed_goal", "swapped_goal", "no_goal", "zero_goal", "base")
POOLS = ("manual_train", "manual_development_holdout", "worker_facts_only")
BRANCHES = ("correction", "holding", "manual_noop_without_requested_preference", "worker_fact")


def episode_binding(episode, row, plan, index):
    feedback.validate_plan(plan)
    cell = plan["design"]["schedule"][index]
    pool = "manual_" + feedback.split(cell["repeat"]) if cell["condition"] == "visual_correct_hold" else "worker_facts_only"
    expected = dict(trial_index=index, repeat=cell["repeat"], requested_target=cell["target"],
        condition=cell["condition"], episode_split=feedback.split(cell["repeat"]),
        artifact_dir=row["artifact_dir"], execution_valid=True, complete_window=True, factual_pool=pool,
        endpoint_frame=80, endpoint_artifact="actual_endpoint.npz",
        supervision_goal="this_episode_actual_frame80_rgb_heatmap", intended_fixed_target_is_not_a_label=True,
        endpoint_closer_to_requested_target=bool(row["target_preference_margin"] > feedback.MARGIN_EPSILON),
        model_id=plan["input_identity"]["model_id"], worker_version=350, expert_labels=False, approved_for_training=False)
    gl.require(all(episode.get(k) == v for k, v in expected.items()) and
        all(row[k] == cell[k] for k in ("seed", "repeat", "target", "condition")) and
        row["trial_index"] == index and row["episode_split"] == expected["episode_split"],
        "本局事实池/整局划分/实际终点或模型身份不同；开发及worker事实不能混成训练")
    return pool


def validate_saved_trial(arrays, events, trace, measured, endpoint, episode, metrics, history, row, plan, index):
    pool = episode_binding(episode, row, plan, index)
    gl.require(feedback.validate_trace(arrays, events, trace, plan, index) == 16,
               "监督检查需要原16步完整真实历史")
    history_schema.validate_saved_history(metrics, history, row, 64, 81)
    gl.require(row.get("saved_history_roundtrip") is True and history.get("saved_history_roundtrip") is True and
        row["actual_endpoint_frame"] == 80 and row["actual_steps"] == 16,
        "需要本局已验收完整历史及真实第80帧终点")
    feedback.validate_endpoint_queries(arrays, trace, endpoint, episode["queries"])
    gl.require(len(measured) == 17 and np.array_equal(endpoint["goal_feature"],
        np.asarray(measured[-1]["goal_feature"], np.float32)), "本局实际终点编码与原测量不同")
    for offset, value in enumerate(measured):
        gl.require(value["frame"] == 64 + offset and value["control_step"] == offset,
                   "逐帧测量/状态边界不同")
        gl.require(np.allclose([value[f"distance_{t}"] for t in (0, 1)],
            trace["distances_to_both_goals"][offset], atol=choices.ATOL, rtol=choices.RTOL), "真实测量距离与保存反馈不同")
        if offset < 16:
            expected = dict(query_remaining=16 - offset, actual_next_action=trace["action_ids"][offset],
                worker_next_action=trace["worker_action_ids"][offset], decision_source=trace["decision_sources"][offset],
                intervention_applied=trace["intervention_applied"][offset], correction_applied=trace["correction_applied"][offset])
        else:
            expected = {k: None for k in ("query_remaining", "actual_next_action", "worker_next_action",
                                          "decision_source", "intervention_applied", "correction_applied")}
        gl.require(all(value.get(k) == v for k, v in expected.items()), "真实测量/建议/实际动作/预算或终点后查询不同")
    return pool


def query_identity(episode, row, trace, measured, offset):
    gl.require(type(offset) is int and 0 <= offset < 4, "只能查询真实frame76–79，禁止终点后查询")
    step, frame = 12 + offset, 76 + offset
    manual = row["condition"] == "visual_correct_hold"
    correction = bool(trace["correction_applied"][step])
    preferred = feedback.boundary_kind(trace["distances_to_both_goals"][step]) == f"closer_target{row['target']}"
    branch = "worker_fact" if not manual else "correction" if correction else "holding" if preferred else "manual_noop_without_requested_preference"
    return dict(trial_index=row["trial_index"], repeat=row["repeat"], target=row["target"],
        condition=row["condition"], episode_split=row["episode_split"], factual_pool=episode["factual_pool"],
        state_frame=frame, incoming_action_frame=frame + 1, remaining=4 - offset,
        actual_action=trace["action_ids"][step], worker_suggestion=trace["worker_action_ids"][step],
        decision_source=trace["decision_sources"][step], correction_applied=correction, branch=branch,
        endpoint_frame=80, supervision_goal=episode["supervision_goal"], worker_version=350,
        model_id=episode["model_id"], intended_fixed_target_is_not_a_label=True,
        expert_labels=False, approved_for_training=False,
        fixed_distance_before=float(measured[step][f"distance_{row['target']}"]),
        fixed_distance_after=float(measured[step + 1][f"distance_{row['target']}"]),
        fixed_step_progress=float(measured[step][f"distance_{row['target']}"] - measured[step + 1][f"distance_{row['target']}"]))


def validate_query(row, expected):
    gl.require(all(row.get(k) == v for k, v in expected.items()) and
        type(row.get("remaining")) is int and row["remaining"] == 80 - row["state_frame"] and
        76 <= row["state_frame"] < 80 and row["incoming_action_frame"] == row["state_frame"] + 1 and
        row["endpoint_frame"] == 80 and row["expert_labels"] is False and row["approved_for_training"] is False,
        "查询状态/真实下一动作/remaining/实际终点/整局划分或来源不同；不能移植标签")


def factual_metrics(probabilities, raw_preferences, action):
    gl.require(set(probabilities) == set(raw_preferences) == set(CONDITIONS) and
        type(action) is int and 0 <= action < 12, "事实动作或六种诊断分布缺失")
    result = {}
    for condition in CONDITIONS:
        p = choices.probability(probabilities[condition])
        raw = np.asarray(raw_preferences[condition])
        gl.require(p.shape == raw.shape == (12,) and np.isfinite(raw).all(), "动作概率或raw偏好异常")
        rank = int(np.flatnonzero(np.argsort(-p, kind="stable") == action)[0]) + 1
        result.update({f"{condition}_factual_probability": float(p[action]),
            f"{condition}_factual_nll": float(-np.log(max(p[action], np.finfo(float).tiny))),
            f"{condition}_factual_rank": rank, f"{condition}_mode": int(p.argmax()),
            f"{condition}_mode_matches_factual": bool(int(p.argmax()) == action)})
    for control in CONDITIONS[1:]:
        result[f"{control}_minus_actual_endpoint_nll"] = result[f"{control}_factual_nll"] - result["actual_endpoint_factual_nll"]
    for goal in ("actual_endpoint", "fixed_goal"):
        for control in ("no_goal", "zero_goal", "swapped_goal"):
            result[f"{goal}_vs_{control}_tv"] = choices.distribution_difference(probabilities[goal], probabilities[control])["total_variation"]
    result["actual_vs_fixed_mode_changed"] = bool(result["actual_endpoint_mode"] != result["fixed_goal_mode"])
    result["actual_vs_fixed_tv"] = choices.distribution_difference(probabilities["actual_endpoint"], probabilities["fixed_goal"])["total_variation"]
    return result


def aggregate(rows):
    episodes = sorted({r["trial_index"] for r in rows})
    if not rows:
        return dict(rows=0, independent_episodes=0, trial_indices=[], metrics=None, mode_matches=None,
                    action_counts=None, mean_fixed_step_progress=None, equal_episode_means=None)
    names = [k for k, v in rows[0]["metrics"].items() if type(v) in (int, float, bool) and not k.endswith("_mode")]
    return dict(rows=len(rows), independent_episodes=len(episodes), trial_indices=episodes,
        metrics={k: float(np.mean([r["metrics"][k] for r in rows])) for k in names},
        equal_episode_means={k: float(np.mean([np.mean([r["metrics"][k] for r in rows if r["trial_index"] == i])
                                             for i in episodes])) for k in names},
        mode_matches={c: sum(r["metrics"][f"{c}_mode_matches_factual"] for r in rows) for c in CONDITIONS},
        action_counts=np.bincount([r["actual_action"] for r in rows], minlength=12).tolist(),
        mean_fixed_step_progress=float(np.mean([r["fixed_step_progress"] for r in rows])))


def summarize(rows, episodes, plan):
    feedback.validate_plan(plan)
    expected = [(i, 76 + j) for i in range(20) for j in range(4)]
    gl.require([(r["trial_index"], r["state_frame"]) for r in rows] == expected and
        [e["trial_index"] for e in episodes] == list(range(20)), "80查询/20局有遗漏、重复、计划外记录或顺序不同")
    for row in rows:
        cell = plan["design"]["schedule"][row["trial_index"]]
        pool = "manual_" + feedback.split(cell["repeat"]) if cell["condition"] == "visual_correct_hold" else "worker_facts_only"
        gl.require(row["repeat"] == cell["repeat"] and row["target"] == cell["target"] and
            row["condition"] == cell["condition"] and row["episode_split"] == feedback.split(cell["repeat"]) and
            row["factual_pool"] == pool and row["branch"] in BRANCHES and row["worker_version"] == 350 and
            row["model_id"] == plan["input_identity"]["model_id"] and row["remaining"] == 80 - row["state_frame"] and
            row["incoming_action_frame"] == row["state_frame"] + 1 and row["endpoint_frame"] == 80 and
            row["expert_labels"] is False and row["approved_for_training"] is False,
            "事实池/模型/整局划分或监督查询身份不同")
    grid = {f"target{target}_{split}_{branch}": aggregate([r for r in rows if r["target"] == target and
        r["episode_split"] == split and r["branch"] == branch])
        for target in (0, 1) for split in ("train", "development_holdout") for branch in BRANCHES}
    missing = [k for k, v in grid.items() if k.endswith(("_correction", "_holding")) and not v["rows"]]
    return dict(format=FORMAT, comparison_protocol=PROTOCOL, query_rows=80, complete_episodes=20,
        by_pool={p: aggregate([r for r in rows if r["factual_pool"] == p]) for p in POOLS},
        by_target_split_branch=grid,
        by_remaining={str(n): aggregate([r for r in rows if r["remaining"] == n]) for n in (4, 3, 2, 1)},
        by_trial=[dict(trial_index=i, **aggregate([r for r in rows if r["trial_index"] == i])) for i in range(20)],
        missing_manual_branches=missing, all_manual_branches_observed=not missing,
        factual_fitting_is_not_expert_validation=True, missing_statistics_are_unavailable=True,
        repeated_rows_are_not_independent_corrections=True, endpoint_is_future_hindsight_not_deployment_input=True,
        optimizer_updates=0, new_env_steps=0, expert_labels_generated=False, approved_for_training=False,
        behavior_accepted=False, t06_approved=False)


def review(summary, episodes, names):
    grid = summary["by_target_split_branch"]
    coverage = {str(t): {split: {b: dict(rows=grid[f"target{t}_{split}_{b}"]["rows"],
        independent_episodes=grid[f"target{t}_{split}_{b}"]["independent_episodes"])
        for b in ("holding", "correction")} for split in ("train", "development_holdout")} for t in (0, 1)}
    train = summary["by_pool"]["manual_train"]
    return dict(format=FORMAT, scope="read_only_scope_review_no_training_authorization", coverage=coverage,
        manual_train_action_counts=dict(zip(names, train["action_counts"])),
        missing_manual_branches=summary["missing_manual_branches"],
        train_correction_targets_observed=[t for t in (0, 1) if grid[f"target{t}_train_correction"]["rows"]],
        hold_targets_with_train_and_development_data=[t for t in (0, 1) if all(
            grid[f"target{t}_{split}_holding"]["rows"] for split in ("train", "development_holdout"))],
        residual_yaw_episodes=[e["trial_index"] for e in episodes if e["condition"] == "visual_correct_hold" and
                              e["endpoint_pose_errors"][f"yaw_error_{e['target']}"] > feedback.MARGIN_EPSILON],
        residual_yaw_is_diagnostic_not_a_training_filter=True,
        dual_goal_correction_train_coverage=all(grid[f"target{t}_train_correction"]["rows"] for t in (0, 1)),
        original_holdout_split_preserved=True, worker_facts_excluded_from_training=True,
        fitting_metrics_define_no_new_gate=True, all_records_retained=True,
        training_budget=None, approved_for_training=False, behavior_accepted=False, t06_approved=False,
        next_step="Review holding fit and goal ablations separately; any training needs a separate fixed protocol. "
                  "Missing correction branches need an independent preregistered design, never holdout reassignment or resampling.")
