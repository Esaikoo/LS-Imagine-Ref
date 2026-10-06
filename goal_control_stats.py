"""Descriptive repeated T05 comparisons, without identical-latent claims."""

import numpy as np


PROTOCOL = "randomized_repeated_comparable_start_v1"
MODES = ("goal", "zero_goal", "shuffled_goal", "original", "reference_replay")
START_FIELDS = ("rgb_mae", "rgb_p99", "position", "angle", "reward")
START_FLAGS = ("inventory_equal", "health_equal", "flags_equal", "actions_equal")


def native_actions_equal(left, right):
    """Compare saved JSON actions with live NumPy actions safely."""
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value
    return plain(left) == plain(right)


def start_check(comparison, native_actions_equal):
    """Keep physical/RGB limits; latent/heatmap drift remains diagnostic.

    This accepts comparable measured histories, not cloned environment or
    hidden states. Callers must use randomized repeated execution and must
    not label these runs as identical-state pairs.
    """
    failed = [key for key in START_FIELDS if not np.isfinite(comparison[key]) or comparison[key] > comparison["limits"][key]]
    failed += [key for key in START_FLAGS if not comparison[key]]
    if comparison["missing_telemetry"]:
        failed.append("missing_telemetry")
    if native_actions_equal is not True:
        failed.append("native_actions_equal")
    return {"protocol": PROTOCOL, "passed": not failed, "failed_checks": failed,
            "same_hidden_state": False, "strict_pair_passed": comparison["passed"],
            "native_actions_equal": native_actions_equal,
            "acceptance_limits": {key: comparison["limits"][key] for key in START_FIELDS},
            "diagnostic_only": {key: comparison[key] for key in ("heatmap_mae", "state_relative_l2")}}


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max())}


def schedule(seeds, repetitions, randomization_seed):
    rng = np.random.RandomState(randomization_seed)
    result = []
    for seed in seeds:
        for repeat in range(repetitions):
            cells = [(target, mode) for target in (0, 1) for mode in MODES]
            for index in rng.permutation(len(cells)):
                target, mode = cells[int(index)]
                result.append({"seed": seed, "repeat": repeat, "target": target, "mode": mode})
    return result


def summarize(rows, design):
    """Drop incomplete start blocks, never select by endpoint outcome.

    Same-world repeats are descriptive variability measurements. They are
    not additional independent world seeds and do not support a p-value.
    """
    planned = {(cell["seed"], cell["repeat"]): set() for cell in design["schedule"]}
    for cell in design["schedule"]:
        planned[(cell["seed"], cell["repeat"])].add((cell["target"], cell["mode"]))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["repeat"]), []).append(row)
    blocks, valid = [], []
    for key, cells in planned.items():
        observed = grouped.get(key, [])
        completed = len(observed) == len(cells) and {(r["target"], r["mode"]) for r in observed} == cells
        eligible = completed and all(r["start_eligible"] and r.get("execution_valid", False) for r in observed)
        blocks.append({"seed": key[0], "repeat": key[1], "planned": len(cells),
                       "completed": len(observed), "eligible": eligible})
        if eligible:
            valid.extend(observed)
    output = {"comparison_protocol": PROTOCOL, "same_hidden_state": False,
              "controlled_reachability_verified": False, "engineering_only": True,
              "planned_trials": len(design["schedule"]), "completed_trials": len(rows),
              "eligible_trials": len(valid), "blocks": blocks, "modes": {},
              "randomized_block_differences": [], "reference_replay_variability": [],
              "independent_world_seeds": len({r["seed"] for r in valid}),
              "confidence_interval": None, "p_value": None}
    metric_keys = ("start_distance", "end_distance", "distance_improvement",
                   "relative_distance_improvement", "target_preference_margin")
    for mode in MODES:
        selected = [r for r in valid if r["mode"] == mode]
        output["modes"][mode] = {"trials": len(selected),
            "task_success_rate": float(np.mean([r["task_success"] for r in selected])) if selected else None,
            **{key: statistics([r[key] for r in selected]) for key in metric_keys}}
    lookup = {(r["seed"], r["repeat"], r["target"], r["mode"]): r for r in valid}
    for seed, repeat in planned:
        for target in (0, 1):
            goal = lookup.get((seed, repeat, target, "goal"))
            if goal is None:
                continue
            for mode in ("zero_goal", "shuffled_goal", "original"):
                control = lookup[(seed, repeat, target, mode)]
                output["randomized_block_differences"].append({"seed": seed, "repeat": repeat,
                    "target": target, "control": mode,
                    "end_distance_advantage": control["end_distance"] - goal["end_distance"],
                    "improvement_advantage": goal["distance_improvement"] - control["distance_improvement"],
                    "initial_distance_difference": goal["start_distance"] - control["start_distance"]})
    for seed in design["seeds"]:
        for target in (0, 1):
            replay = [r for r in valid if r["seed"] == seed and r["target"] == target and r["mode"] == "reference_replay"]
            output["reference_replay_variability"].append({"seed": seed, "target": target,
                "end_distance": statistics([r["end_distance"] for r in replay]),
                "improvement": statistics([r["distance_improvement"] for r in replay]),
                "reference_pose_errors": [r.get("reference_pose_error") for r in replay],
                "native_action_change_count": sum(r.get("reference_native_actions_equal") is False for r in replay),
                "strict_replay_pass_count": sum(bool(r.get("reference_replay_valid")) for r in replay)})
    output["interpretation"] = [
        "通过只表示有随机顺序和完整重复的可比起点执行；没有证明隐藏状态相同或目标控制有效。",
        "heatmap/RSSM差异始终保留；物理遥测、RGB、宏动作与底层动作仍需通过原有边界。",
        "参考动作重放的终点分布用于测量波动；不按其终点结果挑选或删除目标组。",
        "整个seed/repeat块完整且起点/执行检查通过才纳入汇总；不匹配不重试，原数据保留。",
        "同一世界重复不等于新增独立种子；小样本只报描述统计，不报显著性或自动验收。",
        "零目标是推理消融，未包含独立no_goal BC；不能排除继续模仿学习的影响。"]
    return output
