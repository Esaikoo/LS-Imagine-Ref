"""Fixed-action T05 calibration measurements; no policy or environment code."""

import numpy as np

from goal_control_stats import native_actions_equal, statistics


FORMAT = "ls_imagine_reference_calibration_design_v1"
EVALUATION_FORMAT = "ls_imagine_reference_calibration_evaluation_v1"
PROTOCOL = "fixed_reference_scripts_current_initialization_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def schedule(seed, randomization_seed):
    rng = np.random.RandomState(randomization_seed)
    return [dict(seed=seed, repeat=repeat, script=int(script))
            for repeat in range(2) for script in rng.permutation(2)]


def key(cell):
    return tuple(cell[name] for name in ("seed", "repeat", "script"))


def validate_plan(plan):
    require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL,
            "标定计划格式或协议不同")
    seed, random_seed = plan["scenario"]["seed"], plan["randomization_seed"]
    require(type(seed) is int and type(random_seed) is int and
            0 <= seed < 2**31 - 10000 and 0 <= random_seed < 2**31 - 10000,
            "标定计划种子异常")
    dimension = len(plan["action_names"])
    require(dimension == 12 and plan["action_names"][0] == "noop" and
            plan["warmup_steps"] == 32 and len(plan["prefix_actions"]) == 32 and
            plan["start_actions"] == [0] * 32 + plan["prefix_actions"] and
            plan["control_start_frame"] == 64 and plan["horizon"] == 16 and
            len(plan["scripts"]) == 2 and all(len(actions) == 16 for actions in plan["scripts"]) and
            all(type(action) is int and 0 <= action < dimension
                for actions in [plan["start_actions"], *plan["scripts"]] for action in actions),
            "固定预热、前缀、16步脚本或动作接口被修改")
    require(plan["repetitions"] == 2 and plan["maximum_new_env_steps"] == 320 and
            plan["schedule"] == schedule(seed, random_seed) and
            plan.get("execution_policy") == "fixed_script" and
            plan.get("same_hidden_state") is False and plan.get("matched_start_comparison") is False,
            "四次标定顺序、预算或独立起点协议被修改")


def validate_execution(arrays, events, start_actions, script, dimension):
    """Check actual incoming labels, including valid early terminal histories."""
    first, frames = len(start_actions), len(arrays["image"])
    count = frames - first - 1
    require(0 <= count <= len(script) and len(events) == frames and
            all(len(arrays[name]) == frames for name in (
                "heatmap", "action", "features", "obs_reward", "is_first", "is_last", "is_terminal")),
            "真实历史长度或脚本预算异常")
    require(bool(arrays["is_first"][0]) and not arrays["is_first"][1:].any(), "真实历史含伪reset")
    ending = np.asarray(arrays["is_last"] | arrays["is_terminal"]).reshape(-1)
    require(not ending[:-1].any() and not ending[:first + 1].any(), "真实结束后继续执行或前缀提前结束")
    require(count == len(script) or bool(ending[-1]), "脚本未完整执行且没有真实结束")
    expected = np.eye(dimension, dtype=np.float32)[start_actions + script[:count]]
    require(arrays["action"].shape == (frames, dimension) and
            np.all(arrays["action"][0] == 0) and np.array_equal(arrays["action"][1:], expected),
            "真实incoming与固定前缀/下一脚本动作不同")
    require(np.isfinite(arrays["features"]).all() and np.isfinite(arrays["obs_reward"]).all(),
            "真实状态或奖励非有限")
    for frame, event in enumerate(events):
        native = event.get("native_actions")
        telemetry = event.get("telemetry") or {}
        require(event.get("frame") == frame and not event.get("error") and
                bool(event.get("done")) == bool(ending[frame]) and
                isinstance(native, (list, tuple)) and (frame == 0 or len(native) > 0),
                "真实原生事件缺失、终止标志或环境执行错误")
        require(telemetry.get("pose") is not None and telemetry.get("inventory") is not None and
                telemetry.get("health") is not None, "真实遥测缺失")
        pose = telemetry["pose"]
        require(all(name in pose for name in ("x", "y", "z", "yaw", "pitch")) and
                np.isfinite([pose[name] for name in ("x", "y", "z", "yaw", "pitch")] +
                            [telemetry["health"]]).all(), "真实遥测非有限")
        require(np.isclose(float(np.asarray(arrays["obs_reward"][frame]).reshape(-1)[0]),
                           event["reward"], atol=1e-6, rtol=1e-6), "真实奖励与事件不一致")
    return count


def pose_difference(left, right):
    values = [pose[name] for pose in (left, right) for name in ("x", "y", "z", "yaw", "pitch")]
    require(np.isfinite(values).all(), "位置或角度非有限")
    return dict(position=float(np.linalg.norm([left[name] - right[name] for name in ("x", "y", "z")])),
                yaw=float(abs((left["yaw"] - right["yaw"] + 180) % 360 - 180)),
                pitch=float(abs(left["pitch"] - right["pitch"])))


def nearest(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (2,) and np.isfinite(values).all(), "两目标距离形状或数值异常")
    # Avoid treating numerically equal measurements as a target preference.
    return None if np.isclose(values[0], values[1], atol=1e-7, rtol=0) else int(values.argmin())


def measure(image, heatmap, feature, telemetry, bank, target_telemetry):
    require(image.shape == (64, 64, 3) and heatmap.shape == (64, 64) and
            image.dtype == heatmap.dtype == np.uint8, "真实RGB/heatmap形状或类型异常")
    feature = np.asarray(feature)
    require(feature.shape == bank["goals"].shape[1:] and np.isfinite(feature).all() and
            np.isclose(np.linalg.norm(feature), 1, atol=1e-5), "真实目标编码形状或数值异常")
    distances = np.clip(1 - feature @ bank["goals"].T, 0, 2)
    result = {}
    for target in (0, 1):
        rgb = np.abs(image.astype(np.float32) - bank["images"][target].astype(np.float32))
        heat = np.abs(heatmap.astype(np.float32) - bank["heatmaps"][target].astype(np.float32))
        pose = pose_difference(telemetry["pose"], target_telemetry[target]["pose"])
        result.update({f"distance_{target}": float(distances[target]), f"rgb_mae_{target}": float(rgb.mean()),
                       f"rgb_p99_{target}": float(np.percentile(rgb, 99)), f"heatmap_mae_{target}": float(heat.mean()),
                       **{f"{name}_error_{target}": value for name, value in pose.items()}})
    result.update(closest_visual_target=nearest(distances),
                  closest_rgb_target=nearest([result[f"rgb_mae_{target}"] for target in (0, 1)]),
                  closest_position_target=nearest([result[f"position_error_{target}"] for target in (0, 1)]))
    return result


def summarize(rows, plan):
    validate_plan(plan)
    require(len(rows) <= len(plan["schedule"]) and len({key(row) for row in rows}) == len(rows) and
            [key(row) for row in rows] == [key(cell) for cell in plan["schedule"][:len(rows)]],
            "重复或计划外重放；必须保留全部尝试顺序")
    valid = [row for row in rows if row["execution_valid"]]
    complete = [row for row in valid if row["script_completed"]]
    result = dict(protocol=PROTOCOL, planned_runs=4, attempted_runs=len(rows), valid_histories=len(valid),
                  complete_scripts=len(complete), full_design_completed=len(rows) == 4,
                  early_terminal_runs=[plain(row) for row in valid if not row["script_completed"]],
                  execution_errors=[plain(row) for row in rows if not row["execution_valid"]],
                  endpoint_matrix=[dict(seed=row["seed"], repeat=row["repeat"], script=row["script"],
                                        **row["endpoint"]) for row in complete], scripts={},
                  independent_world_seeds=1, same_hidden_state=False, matched_start_comparison=False,
                  controlled_reachability_verified=False, task_success_improvement_verified=False,
                  p_value=None, confidence_interval=None)
    for script in (0, 1):
        observed = [row for row in rows if row["script"] == script]
        measured = [row for row in complete if row["script"] == script]
        result["scripts"][str(script)] = dict(planned=2, attempted=len(observed), complete=len(measured),
            assigned_target_closest_count=sum(row["endpoint"]["closest_visual_target"] == script for row in measured),
            assigned_distance=statistics([row["endpoint"][f"distance_{script}"] for row in measured]),
            assigned_progress=statistics([row["distance_improvement"] for row in measured]),
            assigned_margin=statistics([row["target_preference_margin"] for row in measured]))
    result["interpretation"] = [
        "固定脚本在每局自身真实历史上执行；所有尝试保留，不因跨局位置/RGB/RSSM差异筛选。",
        "完整16步终点是主测量；提前结束单列，逐帧最小距离只作诊断，不替换终点。",
        "视觉距离、RGB差和位置/朝向差是不同测量；物理位置不自动等于视觉目标真值。",
        "两个重复、一个世界仅作可达性/度量定位；工程PASS不批准worker控制、任务成功或T06。"]
    return result
