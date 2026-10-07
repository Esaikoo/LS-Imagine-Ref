"""Offline action sensitivity on each saved factual state, never a rollout.

Probability distances and shared-uniform sampling disagreement describe an
action distribution. Neither establishes progress, reachability or success.
"""

import numpy as np


FORMAT = "ls_imagine_same_state_action_choice_diagnosis_v1"
CONDITIONS = ("goal0", "goal1", "no_goal", "zero_goal", "base")
ATOL = 3e-5
RTOL = 3e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 1 and len(result) >= 2 and np.isfinite(result).all() and
            (result >= 0).all() and np.isclose(result.sum(), 1, atol=1e-5, rtol=0),
            "动作概率形状或数值异常")
    return result / result.sum()


def top_two(value):
    p = probability(value)
    # Stable ordering agrees with np.argmax for an exact tie.
    first, second = np.argsort(-p, kind="stable")[:2]
    return dict(mode=int(first), runner_up=int(second), top_probability=float(p[first]),
                runner_up_probability=float(p[second]), gap=float(p[first] - p[second]),
                entropy=float(-(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum()))


def distribution_difference(left, right):
    p, q = probability(left), probability(right)
    require(p.shape == q.shape, "动作分布维度不同")
    midpoint = (p + q) / 2
    tiny = np.finfo(float).tiny
    js = .5 * ((p * np.log(np.maximum(p, tiny) / np.maximum(midpoint, tiny))).sum() +
               (q * np.log(np.maximum(q, tiny) / np.maximum(midpoint, tiny))).sum())
    # Exact interval intersection for inverse-CDF draws with the SAME U[0,1).
    # This does not draw actions or simulate either hypothetical continuation.
    pc, qc = np.cumsum(p), np.cumsum(q)
    pc[-1] = qc[-1] = 1.
    low_p, low_q = np.r_[0., pc[:-1]], np.r_[0., qc[:-1]]
    agreement = np.maximum(0., np.minimum(pc, qc) - np.maximum(low_p, low_q)).sum()
    return dict(total_variation=float(np.abs(p - q).sum() / 2),
                max_probability_difference=float(np.abs(p - q).max()),
                js_divergence=float(max(0., js)), mode_changed=bool(p.argmax() != q.argmax()),
                shared_uniform_sampling_disagreement=float(np.clip(1 - agreement, 0, 1)))


def frame_metrics(probabilities, preferences):
    require(set(probabilities) == set(preferences) == set(CONDITIONS), "诊断条件缺失")
    tops = {name: top_two(probabilities[name]) for name in CONDITIONS}
    raw = {name: np.asarray(preferences[name], dtype=np.float64) for name in CONDITIONS}
    dimension = len(probability(probabilities["goal0"]))
    require(all(value.shape == (dimension,) and np.isfinite(value).all() for value in raw.values()),
            "原始动作偏好形状或数值异常")
    pair = distribution_difference(probabilities["goal0"], probabilities["goal1"])
    result = dict(goal_total_variation=pair["total_variation"],
        goal_max_probability_difference=pair["max_probability_difference"],
        goal_js_divergence=pair["js_divergence"], goal_mode_changed=pair["mode_changed"],
        shared_uniform_sampling_disagreement=pair["shared_uniform_sampling_disagreement"])
    delta = raw["goal0"] - raw["goal1"]
    result["goal_centered_raw_preference_delta_rms"] = float(np.sqrt(np.mean((delta - delta.mean())**2)))
    for name in CONDITIONS:
        result.update({f"{name}_{key}": value for key, value in tops[name].items()})
        if name in ("goal0", "goal1"):
            for control in ("no_goal", "zero_goal", "base"):
                comparison = distribution_difference(probabilities[name], probabilities[control])
                result[f"{name}_vs_{control}_tv"] = comparison["total_variation"]
                result[f"{name}_vs_{control}_mode_changed"] = comparison["mode_changed"]
    for name, other in (("goal0", "goal1"), ("goal1", "goal0")):
        top, runner = tops[name]["mode"], tops[name]["runner_up"]
        # Does the other goal reverse this goal's winner/runner-up margin?
        p = probability(probabilities[other])
        result[f"{name}_top2_margin_under_{other}"] = float(p[top] - p[runner])
    return result


def validate_trace(arrays, trace, first, horizon, dimension, cell, index):
    """Keep pre-action frame t separate from the observed incoming t+1."""
    n = len(trace["action_ids"])
    require(1 <= n <= horizon and len(arrays["features"]) == first + n + 1,
            "真实控制轨迹长度或步数异常")
    require(trace.get("condition") == cell["mode"] and trace.get("evaluated_target") == cell["target"],
            "动作记录条件或目标不同")
    issued = None if cell["mode"] == "no_goal" else 1 - cell["target"] if cell["mode"] == "swapped_goal" else cell["target"]
    require(trace.get("issued_target") == issued, "实际发出目标不同")
    require(trace["remaining"] == list(range(horizon, horizon - n, -1)), "记录的剩余预算异常")
    require(all(type(action) is int and 0 <= action < dimension for action in trace["action_ids"]),
            "动作记录编号异常")
    expected = np.eye(dimension, dtype=np.float32)[trace["action_ids"]]
    require(np.array_equal(arrays["action"][first + 1:], expected), "真实下一动作与控制记录不同")
    require(not (arrays["is_last"][first:first + n] | arrays["is_terminal"][first:first + n]).any(),
            "不能把真实结束后的状态用于动作查询")
    require(np.asarray(trace["probabilities"]).shape == (n, dimension), "记录动作概率长度异常")
    for value in trace["probabilities"]:
        probability(value)
    distances = np.asarray(trace["distances_to_both_goals"], dtype=np.float64)
    require(distances.shape == (n + 1, 2) and np.isfinite(distances).all(), "目标距离记录异常")
    seed = trace.get("action_seed")
    expected_seed = (index * 1009 + 701 + cell["randomization_seed"]) % (2**31 - 1)
    require(seed == expected_seed and np.array_equal(np.asarray(trace["uniforms"]),
            np.random.RandomState(seed).uniform(size=horizon)[:n]), "原动作随机数记录不同")
    return n


def aggregate(rows, dimension):
    require(rows, "没有真实控制状态可供诊断")
    prefixes = tuple(f"{name}_" for name in CONDITIONS) + ("goal_", "shared_uniform_")
    metrics = [name for name in rows[0] if name.startswith(prefixes)
               and not name.endswith(("mode", "runner_up", "action_name", "runner_up_name"))]
    return dict(frames=len(rows),
        means={name: float(np.mean([row[name] for row in rows])) for name in metrics},
        goal_mode_changes=sum(row["goal_mode_changed"] for row in rows),
        goal_mode_change_fraction=float(np.mean([row["goal_mode_changed"] for row in rows])),
        goal0_mode_histogram=np.bincount([row["goal0_mode"] for row in rows], minlength=dimension).tolist(),
        goal1_mode_histogram=np.bincount([row["goal1_mode"] for row in rows], minlength=dimension).tolist(),
        condition_mode_histograms={name: np.bincount([row[f"{name}_mode"] for row in rows], minlength=dimension).tolist()
                                   for name in CONDITIONS},
        max_goal_total_variation=float(max(row["goal_total_variation"] for row in rows)))


def summarize(rows, dimension):
    trials = sorted({row["trial_index"] for row in rows})
    per_trial = [dict(trial_index=index, **aggregate([row for row in rows if row["trial_index"] == index], dimension))
                 for index in trials]
    return dict(format=FORMAT, query_frames=len(rows), trials=len(trials),
        frame_weighted=aggregate(rows, dimension), per_trial=per_trial,
        equal_trial_means={name: float(np.mean([trial["means"][name] for trial in per_trial]))
                           for name in per_trial[0]["means"]},
        by_factual_condition={mode: aggregate([row for row in rows if row["factual_condition"] == mode], dimension)
                              for mode in sorted({row["factual_condition"] for row in rows})},
        by_remaining={str(budget): aggregate([row for row in rows if row["remaining"] == budget], dimension)
                      for budget in sorted({row["remaining"] for row in rows}, reverse=True)},
        by_repeat={str(repeat): aggregate([row for row in rows if row["repeat"] == repeat], dimension)
                   for repeat in sorted({row["repeat"] for row in rows})},
        by_evaluated_target={str(target): aggregate([row for row in rows if row["evaluated_target"] == target], dimension)
                             for target in sorted({row["evaluated_target"] for row in rows})},
        control_starts_only=aggregate([row for row in rows if row["control_step"] == 0], dimension),
        same_state_within_query=True, same_state_across_trials=False,
        counterfactual_rollouts_executed=False, reachability_verified=False,
        behavior_accepted=False, independent_samples_claimed=False,
        scope="同一真实状态的离线动作分布比较；不推断换目标后的轨迹、任务成功或目标可达性")
