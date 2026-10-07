"""Independent-start randomized T05 pilot. No matched-start acceptance rule."""

import numpy as np

import goal_library as gl
from goal_control_stats import statistics


FORMAT = "ls_imagine_independent_start_design_v1"
EVALUATION_FORMAT = "ls_imagine_independent_start_evaluation_v1"
PROTOCOL = "independent_starts_three_conditions_fixed_visual_targets_v1"
MODES = ("goal", "no_goal", "swapped_goal")
METRICS = ("start_distance", "end_distance", "distance_improvement",
           "target_preference_margin", "preference_improvement")


def schedule(seed, repetitions, randomization_seed):
    rng = np.random.RandomState(randomization_seed)
    result = []
    for repeat in range(repetitions):
        cells = [(target, mode) for target in (0, 1) for mode in MODES]
        for index in rng.permutation(len(cells)):
            target, mode = cells[int(index)]
            result.append(dict(seed=seed, repeat=repeat, target=target, mode=mode))
    return result


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "target", "mode"))


def validate_design(design):
    for name in ("seed", "randomization_seed"):
        gl.require(type(design[name]) is int and 0 <= design[name] < 2**31 - 10000,
                   "随机设计种子异常")
    gl.require(type(design["repetitions"]) is int and 2 <= design["repetitions"] <= 10 and
               design["modes"] == list(MODES) and design["execution_policy"] in ("mode", "sample") and
               design["schedule"] == schedule(design["seed"], design["repetitions"], design["randomization_seed"]),
               "随机计划、三组条件或预算被修改")


def validate_prefix(arrays, events, actions, dimension):
    """Validate this trial's real history, without comparison to another run."""
    size = len(actions) + 1
    names = ("image", "heatmap", "features", "action", "obs_reward", "is_first", "is_last", "is_terminal")
    gl.require(all(len(arrays[name]) >= size for name in names) and len(events) >= size,
               "缺少完整真实reset/预热/前缀")
    gl.require(bool(arrays["is_first"][0]) and not arrays["is_first"][1:size].any() and
               not (arrays["is_last"][:size] | arrays["is_terminal"][:size]).any(),
               "控制前缀含伪reset或真实结束")
    expected = np.eye(dimension, dtype=np.float32)[actions]
    gl.require(np.all(arrays["action"][0] == 0) and
               np.array_equal(arrays["action"][1:size], expected), "真实incoming与固定预热/前缀不同")
    for frame, event in enumerate(events[:size]):
        telemetry = event.get("telemetry", {})
        gl.require(event.get("frame") == frame and not event.get("error") and not event.get("done") and
                   isinstance(event.get("native_actions"), (list, tuple)) and
                   (frame == 0 or len(event["native_actions"]) > 0), "真实原生事件缺失或环境执行错误")
        gl.require(telemetry.get("pose") is not None and telemetry.get("inventory") is not None and
                   telemetry.get("health") is not None, "真实起点遥测缺失")
        values = [telemetry["pose"][name] for name in ("x", "y", "z", "yaw", "pitch")]
        gl.require(np.isfinite(values + [telemetry["health"]]).all(), "真实起点遥测非有限")
    gl.require(np.isfinite(arrays["features"][:size]).all(), "真实当前状态非有限")


def summarize(rows, design):
    validate_design(design)
    plan = design["schedule"]
    gl.require(len({key(row) for row in rows}) == len(rows) and
               [key(row) for row in rows] == [key(cell) for cell in plan[:len(rows)]],
               "汇总有重复或计划外trial；必须保留执行顺序")
    result = dict(protocol=PROTOCOL, planned_trials=len(plan), attempted_trials=len(rows),
                  full_design_completed=len(rows) == len(plan), independent_world_seeds=1,
                  same_hidden_state=False, matched_start_comparison=False,
                  controlled_reachability_verified=False, descriptive_pilot_only=True,
                  p_value=None, confidence_interval=None, modes={}, contrasts=[], initial_balance={},
                  execution_errors=[dict(cell={name: row[name] for name in ("seed", "repeat", "target", "mode")},
                                         error=row.get("error")) for row in rows if not row["execution_valid"]])
    for mode in MODES:
        observed = [row for row in rows if row["mode"] == mode]
        measured = [row for row in observed if row["execution_valid"]]
        started = [row for row in observed if "start_telemetry" in row]
        # Failures remain in the planned/attempted denominator. Never drop a
        # complete block because one independent start differs or has an error.
        result["modes"][mode] = dict(planned=2 * design["repetitions"], attempted=len(observed),
            measured=len(measured), errors=len(observed) - len(measured),
            task_successes=sum(bool(row.get("task_success")) for row in observed),
            **{name: statistics([row[name] for row in measured]) for name in METRICS})
        result["initial_balance"][mode] = dict(
            recorded_starts=len(started), start_distance=statistics([row["start_distance"] for row in started]),
            start_poses=[dict(target=row["target"], repeat=row["repeat"], **row["start_telemetry"]["pose"])
                         for row in started],
            start_telemetry=[row["start_telemetry"] for row in started],
            prefix_movement=statistics([row["prefix_movement"]["position"] for row in started]))
    for target in (0, 1):
        chosen = {mode: [row for row in rows if row["mode"] == mode and row["target"] == target and
                        row["execution_valid"]] for mode in MODES}
        for mode in ("no_goal", "swapped_goal"):
            goal, control = chosen["goal"], chosen[mode]
            difference = lambda name: (float(np.mean([row[name] for row in goal]) -
                                               np.mean([row[name] for row in control])) if goal and control else None)
            result["contrasts"].append(dict(target=target, control=mode, goal_measured=len(goal),
                control_measured=len(control), initial_distance_difference=difference("start_distance"),
                end_distance_difference=difference("end_distance"),
                progress_advantage=difference("distance_improvement"),
                evaluated_target_margin_advantage=difference("target_preference_margin"),
                preference_change_advantage=difference("preference_improvement"),
                comparison="means of independent starts; not a matched-state pair"))
    result["interpretation"] = [
        "所有预声明尝试及错误均保留；有效测量数量另列，不按位置/RGB/终点筛选起点。",
        "目标是提前固定的真实视觉图像；不保证从每个实际起点都可达。",
        "有目标与交换目标在不同真实历史上执行；须同时检查初始距离/位置平衡和视频。",
        "一个世界两次重复仅为描述性试跑；工程PASS不批准目标控制、论文结论或T06。"]
    return result
