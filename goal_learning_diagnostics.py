"""Read-only T05 goal-learning diagnostics; no environment or optimizer.

Cross-episode neighbours are observational candidates, not the same
physical state or demonstrations that an alternative goal is reachable.
"""

import numpy as np

import goal_library as gl


FORMAT = "ls_imagine_goal_learning_diagnosis_v1"


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    gl.require(np.isfinite(values).all(), "诊断统计含非有限值")
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "min": float(values.min()), "max": float(values.max())}


def state_index(cache):
    count = len(cache.states)
    episodes = np.full(count, -1, np.int64)
    frames = np.full(count, -1, np.int64)
    for block in cache.metadata["episode_states"]:
        offset, indices = int(block["offset"]), np.asarray(block["frames"], np.int64)
        gl.require(offset >= 0 and offset + len(indices) <= count and
                   np.all(np.diff(indices) > 0) and np.all(episodes[offset:offset + len(indices)] == -1),
                   "状态缓存块重叠、越界或帧次序错误")
        episodes[offset:offset + len(indices)] = int(block["episode_index"])
        frames[offset:offset + len(indices)] = indices
    gl.require(np.all(episodes >= 0) and np.all(frames >= 0), "状态缓存块覆盖不完整")
    return episodes, frames


def stratified_rows(rows, strata, count, seed):
    """Round-robin duration groups, without changing the training sampler."""
    rng = np.random.RandomState(seed)
    rows = np.asarray(rows, np.int64)
    pools = [rng.permutation(rows[strata[rows] == value]).tolist() for value in np.unique(strata[rows])]
    chosen = []
    while len(chosen) < count and any(pools):
        for index in rng.permutation(len(pools)):
            if pools[index]:
                chosen.append(pools[index].pop())
                if len(chosen) == count:
                    break
    return np.asarray(sorted(chosen), np.int64)


def probe_rows(cache, state_episodes, train_per_episode, validation_per_episode, seed):
    t = cache.tables
    selected = []
    for index, entry in enumerate(cache.metadata["source_metadata"]["episodes"]):
        rows = np.flatnonzero(state_episodes[t["worker_state"]] == index)
        # Repeated overlapping segments can expose the same current state,
        # remaining duration and endpoint. Count this query only once.
        keys = np.stack((t["worker_state"][rows], t["remaining"][rows]), axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        unique = rows[np.sort(first)]
        count = train_per_episode if entry["split"] == "train" else validation_per_episode
        selected.extend(stratified_rows(unique, t["remaining"], count, (seed + index * 1009) % 2**32))
    return np.asarray(selected, np.int64)


def alternate_rows(cache, rows, state_episodes, seed, candidates=8):
    """Choose distant REAL goals, same duration/split, another episode.

    Selection depends on input geometry, never model output or endpoint
    control results. The original next-action label is only a diagnostic
    likelihood query; it is not supervision for the swapped goal.
    """
    t, rng = cache.tables, np.random.RandomState(seed)
    output = np.full(len(rows), -1, np.int64)
    groups = {}
    for row in rows:
        key = (bool(t["worker_train"][row]), int(t["remaining"][row]))
        if key not in groups:
            group = np.flatnonzero((t["worker_train"] == key[0]) & (t["remaining"] == key[1]))
            # One donor per segment avoids weighting an overlapping endpoint
            # by the number of expanded action entries it contributed.
            _, first = np.unique(t["worker_segment"][group], return_index=True)
            groups[key] = group[np.sort(first)]
    for index, row in enumerate(rows):
        group = groups[(bool(t["worker_train"][row]), int(t["remaining"][row]))]
        group = group[state_episodes[t["worker_state"][group]] != state_episodes[t["worker_state"][row]]]
        if not len(group):
            continue
        pool = rng.choice(group, min(candidates, len(group)), replace=False)
        goal = t["goals"][t["worker_segment"][row]]
        distances = 1 - np.clip(t["goals"][t["worker_segment"][pool]] @ goal, -1, 1)
        output[index] = int(pool[int(distances.argmax())])
    return output


def supervision_redundancy(cache, state_episodes):
    t = cache.tables
    actions_min = np.full(len(cache.states), cache.metadata["action_dim"], np.int64)
    actions_max = np.full(len(cache.states), -1, np.int64)
    np.minimum.at(actions_min, t["worker_state"], t["action_id"])
    np.maximum.at(actions_max, t["worker_state"], t["action_id"])
    gl.require(np.array_equal(actions_min, actions_max), "同一真实当前状态存在冲突下一动作标签")
    pairs = np.unique(np.stack((t["worker_state"], t["remaining"]), axis=1), axis=0)
    result = {}
    for split, train in (("train", True), ("validation", False)):
        rows = np.flatnonzero(t["worker_train"] == train)
        counts = np.bincount(t["worker_state"][rows], minlength=len(cache.states))
        pair_counts = np.bincount(pairs[:, 0], minlength=len(cache.states))
        states = np.flatnonzero(counts)
        result[split] = {"expanded_labels": len(rows), "unique_states": len(states),
            "unique_state_remaining_queries": int(pair_counts[states].sum()),
            "states_with_multiple_remaining_values": int((pair_counts[states] > 1).sum()),
            "states_with_different_observed_next_actions": 0,
            "remaining_label_histogram": np.bincount(t["remaining"][rows], minlength=cache.bundle["horizon"] + 1).tolist(),
            "action_label_histogram": np.bincount(t["action_id"][rows], minlength=cache.metadata["action_dim"]).tolist(),
            "episodes": len(np.unique(state_episodes[states])),
            "interpretation": "one recorded continuation per exact history state; this is structural, not a BC impossibility proof"}
    return result


def summarize_probes(records):
    def summarize(rows):
        keys = ("correct_nll", "zero_nll", "original_nll", "zero_minus_correct_nll",
                "correct_vs_zero_l1", "correct_vs_zero_mode_changed")
        result = {key: describe([row[key] for row in rows]) for key in keys}
        supported = [row for row in rows if row["alternate_row"] >= 0]
        result.update(rows=len(rows), alternate_supported_rows=len(supported),
                      episodes=len({row["episode"] for row in rows}))
        for key in ("swap_nll", "swap_minus_correct_nll", "correct_vs_swap_l1",
                    "correct_vs_swap_mode_changed", "swap_goal_cosine_distance"):
            result[key] = describe([row[key] for row in supported])
        result["correct_accuracy"] = describe([row["correct_action"] == row["label_action"] for row in rows])
        if rows and "no_goal_bc_nll" in rows[0]:
            for key in ("no_goal_bc_nll", "no_goal_bc_minus_correct_nll", "correct_vs_no_goal_bc_l1"):
                result[key] = describe([row[key] for row in rows])
        return result

    result = {}
    for split in ("train", "validation"):
        rows = [row for row in records if row["split"] == split]
        per_episode = [{"episode": index, **summarize([row for row in rows if row["episode"] == index])}
                       for index in sorted({row["episode"] for row in rows})]
        macro = {}
        for key in ("zero_minus_correct_nll", "swap_minus_correct_nll", "correct_vs_swap_l1",
                    "correct_vs_swap_mode_changed", "correct_vs_zero_mode_changed", "no_goal_bc_minus_correct_nll"):
            values = [entry[key]["mean"] for entry in per_episode if key in entry and entry[key]["mean"] is not None]
            if values:
                macro[key] = describe(values)
        result[split] = {"row_weighted": summarize(rows), "episode_mean_descriptive": macro,
                         "per_episode": per_episode,
                         "per_remaining": [{"remaining": value, **summarize([row for row in rows if row["remaining"] == value])}
                                           for value in sorted({row["remaining"] for row in rows})]}
    result["scope"] = "held-out episodes, correlated queries; no seed-generalization, control or significance claim"
    return result


def nearest_start_pairs(items, visual, states, neighbours):
    """Rank only by start geometry; never choose on future action/goal."""
    gl.require(len(items) == len(visual) == len(states), "近邻输入长度不同")
    visual = np.asarray(visual, np.float32)
    states = np.asarray(states, np.float32)
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    gl.require(np.isfinite(states).all() and np.all(norms > 1e-8), "近邻状态范数异常")
    states = states / norms
    pairs, seen = [], set()
    for split in ("train", "validation"):
        for duration in sorted({item["duration"] for item in items}):
            indices = np.array([i for i, item in enumerate(items)
                                if item["split"] == split and item["duration"] == duration], np.int64)
            if len(indices) < 2:
                continue
            vg, sg = visual[indices], states[indices]
            visual_distance = np.clip(1 - vg @ vg.T, 0, 2)
            state_distance = np.clip(1 - sg @ sg.T, 0, 2)
            score = (visual_distance + state_distance) / 2
            for local, index in enumerate(indices):
                order = sorted((float(score[local, j]), int(indices[j]), j) for j in range(len(indices))
                               if items[int(indices[j])]["episode"] != items[int(index)]["episode"])
                for value, other, j in order[:neighbours]:
                    key = tuple(sorted((int(index), other)))
                    if key in seen:
                        continue
                    seen.add(key)
                    left, right = items[key[0]], items[key[1]]
                    sequence_left, sequence_right = np.asarray(left["actions"]), np.asarray(right["actions"])
                    changes = np.flatnonzero(sequence_left != sequence_right)
                    pairs.append({"left_index": key[0], "right_index": key[1], "split": split,
                        "duration": duration, "start_score": value,
                        "start_visual_cosine_distance": float(visual_distance[local, j]),
                        "start_state_cosine_distance": float(state_distance[local, j]),
                        "goal_cosine_distance": float(np.clip(1 - np.dot(left["goal"], right["goal"]), 0, 2)),
                        "first_actions_differ": bool(len(changes) and changes[0] == 0),
                        "first_action_divergence_step": int(changes[0] + 1) if len(changes) else None,
                        "action_sequence_difference_fraction": float((sequence_left != sequence_right).mean()),
                        "same_physical_state": False})
    return sorted(pairs, key=lambda row: (row["split"], row["start_score"], row["left_index"], row["right_index"]))


def summarize_pairs(pairs):
    result = {}
    for split in ("train", "validation"):
        rows = [row for row in pairs if row["split"] == split]
        buckets = []
        # These are fixed descriptive bins, not T05 start-acceptance limits.
        for visual_limit, state_limit in ((0.02, 0.05), (0.05, 0.10), (0.10, 0.20)):
            selected = [row for row in rows if row["start_visual_cosine_distance"] <= visual_limit and
                        row["start_state_cosine_distance"] <= state_limit]
            separated = [row for row in selected if row["goal_cosine_distance"] >= 0.10]
            buckets.append({"visual_limit": visual_limit, "state_limit": state_limit, "goal_distance_min": 0.10,
                "pairs": len(selected), "separated_goal_pairs": len(separated),
                "separated_goals_first_action_differs": sum(row["first_actions_differ"] for row in separated),
                "separated_goals_sequence_differs": sum(row["first_action_divergence_step"] is not None for row in separated)})
        result[split] = {"candidate_pairs": len(rows), "descriptive_bins": buckets,
            "start_score": describe([row["start_score"] for row in rows]),
            "goal_distance": describe([row["goal_cosine_distance"] for row in rows]),
            "sequence_difference": describe([row["action_sequence_difference_fraction"] for row in rows])}
    result["scope"] = "bounded sampled cross-episode neighbours, same duration and split; no full-data absence or reachability proof"
    return result
