"""Predeclared repeated residual control; never claims cloned hidden states."""

import copy

import numpy as np
import torch

import goal_control as ctl
import goal_control_stats as repeated
import goal_library as gl


FORMAT = "ls_imagine_residual_control_benchmark_v1"
PROTOCOL = "residual_six_conditions_randomized_complete_blocks_v1"
MODES = ("goal", "no_goal", "base", "zero_goal", "swapped_goal", "reference_replay")
CONTROLS = ("no_goal", "base", "zero_goal", "swapped_goal")
METRICS = ("start_distance", "end_distance", "distance_improvement",
           "relative_distance_improvement", "target_preference_margin", "preference_improvement")


def schedule(seeds, repetitions, randomization_seed):
    rng = np.random.RandomState(randomization_seed)
    result = []
    for seed in seeds:
        for repeat in range(repetitions):
            cells = [(target, mode) for target in (0, 1) for mode in MODES]
            for index in rng.permutation(len(cells)):
                target, mode = cells[int(index)]
                result.append(dict(seed=seed, repeat=repeat, target=target, mode=mode))
    return result


def validate_design(design):
    seeds, repetitions, randomization = (design[key] for key in ("seeds", "repetitions", "randomization_seed"))
    gl.require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds) and
               all(type(seed) is int and 0 <= seed < 2**31 - 10000 for seed in seeds), "重复计划种子异常")
    gl.require(type(repetitions) is int and repetitions >= 3 and type(randomization) is int and
               0 <= randomization < 2**31 - 10000, "重复次数或随机顺序种子异常")
    gl.require(design["modes"] == list(MODES) and design["execution_policy"] in ("mode", "sample") and
               design["schedule"] == schedule(seeds, repetitions, randomization), "六组条件、执行方式或随机计划被修改")


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "target", "mode"))


def pending_cells(design, completed, blocks):
    """Only continue an exact, complete-block prefix. Never retry a trial."""
    validate_design(design)
    plan = design["schedule"]
    size = 2 * len(MODES)
    gl.require(type(blocks) is int and blocks >= 1, "blocks至少为1")
    gl.require(len(completed) % size == 0 and len(completed) <= len(plan) and
               [key(row) for row in completed] == [key(cell) for cell in plan[:len(completed)]],
               "已有执行不是完整随机计划前缀；禁止重试、不完整块恢复或重复trial")
    gl.require(all(row.get("start_eligible") and row.get("execution_valid") for row in completed),
               "已有块未通过起点/执行接口验收，不能隐藏失败后继续")
    stop = len(completed) + blocks * size
    gl.require(stop <= len(plan), "blocks超过剩余预声明预算")
    return copy.deepcopy(plan[len(completed):stop])


@torch.no_grad()
def probabilities(runtime, features, goals, remaining, target, mode):
    gl.require(mode in MODES and mode != "reference_replay", "参考重放必须使用记录动作")
    gl.require(type(target) is int and target in (0, 1), "目标编号异常")
    gl.require(type(remaining) is int and 1 <= remaining <= runtime.bundle["horizon"], "remaining必须为有效整数预算")
    assigned = 1 - target if mode == "swapped_goal" else target
    goal = torch.as_tensor(np.asarray(goals[assigned])[None], dtype=torch.float32, device=features.device)
    steps = torch.tensor([remaining], dtype=torch.int64, device=features.device)
    branch = "goal" if mode == "swapped_goal" else mode
    p = runtime.action_dist(features, goal, steps, branch).probs[0].cpu().numpy()
    gl.require(p.shape == (runtime.bundle["config"]["num_actions"],) and np.isfinite(p).all() and
               np.all(p >= 0) and np.isclose(p.sum(), 1, atol=1e-5), "残差部署动作分布异常")
    return p


def summarize(rows, design):
    validate_design(design)
    gl.require(len({key(row) for row in rows}) == len(rows) and
               {key(row) for row in rows} <= {key(cell) for cell in design["schedule"]}, "汇总有重复或计划外trial")
    blocks, valid = [], []
    for seed in design["seeds"]:
        for repeat in range(design["repetitions"]):
            planned = [cell for cell in design["schedule"] if cell["seed"] == seed and cell["repeat"] == repeat]
            observed = [row for row in rows if row["seed"] == seed and row["repeat"] == repeat]
            complete = len(observed) == len(planned) and {key(row) for row in observed} == {key(cell) for cell in planned}
            eligible = complete and all(row.get("start_eligible") and row.get("execution_valid") for row in observed)
            blocks.append(dict(seed=seed, repeat=repeat, planned=len(planned), completed=len(observed), eligible=eligible))
            if eligible:
                valid.extend(observed)
    result = dict(comparison_protocol=PROTOCOL, same_hidden_state=False,
                  controlled_reachability_verified=False, engineering_only=True,
                  planned_trials=len(design["schedule"]), completed_trials=len(rows), eligible_trials=len(valid),
                  blocks=blocks, modes={}, per_seed_target=[], randomized_block_differences=[],
                  target_switches=[], reference_replay_variability=[],
                  independent_world_seeds=len({row["seed"] for row in valid}), p_value=None, confidence_interval=None)
    for mode in MODES:
        chosen = [row for row in valid if row["mode"] == mode]
        result["modes"][mode] = dict(trials=len(chosen),
            task_success_rate=float(np.mean([row["task_success"] for row in chosen])) if chosen else None,
            **{name: repeated.statistics([row[name] for row in chosen]) for name in METRICS})
    lookup = {key(row): row for row in valid}
    for block in blocks:
        if not block["eligible"]:
            continue
        seed, repeat = block["seed"], block["repeat"]
        for target in (0, 1):
            goal = lookup[(seed, repeat, target, "goal")]
            for mode in CONTROLS:
                control = lookup[(seed, repeat, target, mode)]
                result["randomized_block_differences"].append(dict(seed=seed, repeat=repeat, target=target, control=mode,
                    end_distance_advantage=control["end_distance"] - goal["end_distance"],
                    improvement_advantage=goal["distance_improvement"] - control["distance_improvement"],
                    target_margin_advantage=goal["target_preference_margin"] - control["target_preference_margin"],
                    preference_improvement_advantage=goal["preference_improvement"] - control["preference_improvement"],
                    initial_distance_difference=goal["start_distance"] - control["start_distance"]))
        for mode in MODES:
            left, right = (lookup[(seed, repeat, target, mode)] for target in (0, 1))
            result["target_switches"].append(dict(seed=seed, repeat=repeat, mode=mode,
                target_0_margin=left["target_preference_margin"], target_1_margin=right["target_preference_margin"],
                switch_contrast=left["target_preference_margin"] + right["target_preference_margin"],
                switch_change=left["preference_improvement"] + right["preference_improvement"],
                both_prefer_evaluated_target=left["target_preference_margin"] > 0 and right["target_preference_margin"] > 0))
    result["control_advantages"] = {mode: {metric: repeated.statistics([
        row[metric] for row in result["randomized_block_differences"] if row["control"] == mode])
        for metric in ("end_distance_advantage", "improvement_advantage", "target_margin_advantage",
                       "preference_improvement_advantage", "initial_distance_difference")} for mode in CONTROLS}
    for seed in design["seeds"]:
        for target in (0, 1):
            for mode in MODES:
                chosen = [row for row in valid if row["seed"] == seed and row["target"] == target and row["mode"] == mode]
                result["per_seed_target"].append(dict(seed=seed, target=target, mode=mode,
                    **{name: repeated.statistics([row[name] for row in chosen]) for name in METRICS}))
            replay = [row for row in valid if row["seed"] == seed and row["target"] == target and row["mode"] == "reference_replay"]
            result["reference_replay_variability"].append(dict(seed=seed, target=target,
                end_distance=repeated.statistics([row["end_distance"] for row in replay]),
                improvement=repeated.statistics([row["distance_improvement"] for row in replay]),
                target_margin=repeated.statistics([row["target_preference_margin"] for row in replay]),
                reference_pose_errors=[row.get("reference_pose_error") for row in replay],
                native_action_change_count=sum(row.get("reference_native_actions_equal") is False for row in replay),
                strict_replay_pass_count=sum(bool(row.get("reference_replay_valid")) for row in replay)))
    result["interpretation"] = [
        "仅完整且起点/执行接口可比的随机块纳入比较；失败不重试，原轨迹保留。",
        "六组包含独立同预算无目标修正和冻结BC底座；零目标不能代替独立无目标对照。",
        "参考重放终点及原生动作波动是诊断，不按未来结果挑选/排除目标组。",
        "起点heatmap/RSSM严格失败保留，不声称隐藏状态相同，不放宽物理/RGB边界。",
        "一个世界多次重复不是多个独立世界；不自动批准目标控制、T06或任务成功提升。"]
    return result
