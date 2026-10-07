"""Measured start stability, separate from historical/control acceptance.

CPU-only diagnostics; no torch, environment or model imports. An endpoint
or recent-window comparison never establishes equal recurrent states.
"""

import hashlib
import json

import numpy as np


PROTOCOL = "fixed_noop_warmup_full_physical_recent_rgb_preflight_v1"
LIMITS = dict(rgb_mae=1.0, rgb_p99=8.0, heatmap_mae=0.25,
              state_relative_l2=1e-4, position=0.05, angle=0.25, reward=1e-6)
PHYSICAL = ("position", "angle", "reward")
FLAGS = ("inventory_equal", "health_equal", "flags_equal", "actions_equal", "native_actions_equal")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def content_id(value):
    # Same JSON encoding as the existing benchmark/evaluation digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extend_statistics(rows, left, right, events_left, events_right):
    """Extend existing pixel statistics with the unchanged physical checks."""
    require(len(rows) == len(events_left) == len(events_right), "统计/事件长度不同")
    result = []
    for frame, source in enumerate(rows):
        a, b = events_left[frame]["telemetry"], events_right[frame]["telemetry"]
        row = dict(source)
        position, yaw, pitch = (row[name] for name in ("position_difference", "yaw_difference", "pitch_difference"))
        missing = position is None or yaw is None or pitch is None or any(
            value.get(name) is None for value in (a, b) for name in ("inventory", "health"))
        row.update(position=position, angle=None if yaw is None or pitch is None else max(yaw, pitch),
                   reward=row["reward_difference"], missing_telemetry=missing,
                   inventory_equal=a.get("inventory") == b.get("inventory"), health_equal=a.get("health") == b.get("health"),
                   flags_equal=all(np.array_equal(left[name][frame], right[name][frame])
                                   for name in ("is_first", "is_last", "is_terminal")),
                   actions_equal=row["incoming_actions_equal"],
                   position_delta=None if a.get("pose") is None or b.get("pose") is None else {
                       name: float(b["pose"][name] - a["pose"][name]) for name in ("x", "y", "z")})
        result.append(row)
    return result


def physical_failures(row, limits):
    failed = [name for name in PHYSICAL if row[name] is None or not np.isfinite(row[name]) or row[name] > limits[name]]
    failed += [name for name in FLAGS if row[name] is not True]
    if row["missing_telemetry"]:
        failed.append("missing_telemetry")
    return failed


def rgb_failures(row, limits):
    return [name for name in ("rgb_mae", "rgb_p99") if not np.isfinite(row[name]) or row[name] > limits[name]]


def window_summary(rows, limits):
    require(bool(rows), "比较窗口不能为空")
    physical = sorted({name for row in rows for name in physical_failures(row, limits)})
    rgb = sorted({name for row in rows for name in rgb_failures(row, limits)})
    return dict(first_frame=rows[0]["frame"], last_frame=rows[-1]["frame"], frames=len(rows),
                physical_passed=not physical, physical_failed_checks=physical,
                rgb_passed=not rgb, rgb_failed_checks=rgb,
                passed=not physical and not rgb,
                **{name: max(float(row[name]) for row in rows) for name in
                   ("rgb_mae", "rgb_p99", "heatmap_mae", "state_relative_l2")},
                position=None if any(row["position"] is None for row in rows) else max(row["position"] for row in rows),
                angle=None if any(row["angle"] is None for row in rows) else max(row["angle"] for row in rows))


def summarize_history(rows, tail_frames, limits=LIMITS):
    require(type(tail_frames) is int and 1 <= tail_frames <= len(rows), "tail-frames超出真实前缀范围")
    require([row["frame"] for row in rows] == list(range(len(rows))), "历史帧编号不连续")
    require(limits == LIMITS, "数值阈值必须沿用原门槛；不根据本轮结果放宽")
    full, current, recent = (window_summary(selected, limits) for selected in (rows, rows[-1:], rows[-tail_frames:]))
    return dict(limits=dict(limits), full_history=full, control_start=current, recent_window=recent,
                candidate_preflight_passed=bool(full["physical_passed"] and recent["rgb_passed"]),
                physical_failure_frames=[dict(frame=row["frame"], failed_checks=physical_failures(row, limits),
                    position=row["position"], angle=row["angle"], reward=row["reward"], position_delta=row["position_delta"])
                    for row in rows if physical_failures(row, limits)],
                rgb_failure_frames=[row["frame"] for row in rows if rgb_failures(row, limits)],
                same_hidden_state=False, approved_benchmark=False,
                scope="window diagnostics/preflight only; never reclassify historical trials or prove control")


def make_plan(case, benchmark, warmup_steps, tail_frames, num_actions, action_names):
    require(type(warmup_steps) is int and 1 <= warmup_steps <= 256, "warmup-steps必须在[1,256]")
    require(type(tail_frames) is int and 2 <= tail_frames <= 8, "tail-frames必须在[2,8]")
    prefix = case["prefix_actions"]
    require(len(prefix) == benchmark["prefix_steps"] and len(prefix) >= tail_frames - 1,
            "共同前缀长度与窗口不兼容")
    require(len(action_names) == num_actions and action_names[0] == "noop" and len(set(action_names)) == num_actions,
            "真实动作空间的0号动作必须为noop")
    require(all(type(action) is int and 0 <= action < num_actions for action in prefix), "共同前缀动作越界")
    plan = dict(protocol=PROTOCOL, seed=case["seed"], scenario=case["scenario"], benchmark_id=benchmark["benchmark_id"],
                warmup_steps=warmup_steps, noop_action=0, action_names=action_names, prefix_steps=len(prefix),
                prefix_actions=prefix, actions=[0] * warmup_steps + prefix, runs=2,
                control_start_frame=warmup_steps + len(prefix), tail_frames=tail_frames, limits=dict(LIMITS),
                maximum_new_env_steps=2 * (warmup_steps + len(prefix)),
                acceptance=dict(physical="all real reset/warmup/prefix frames, including native actions",
                                rgb="last fixed tail_frames ending at the proposed control start",
                                heatmap_rssm="complete history retained as diagnostics; no equal-state claim"),
                original_benchmark_reusable=False, approved_benchmark=False,
                worker_control_actions=0, controlled_reachability_verified=False)
    plan["plan_id"] = content_id(plan)
    return plan


def validate_plan(plan):
    names = plan.get("action_names")
    require(plan.get("protocol") == PROTOCOL and plan.get("runs") == 2 and plan.get("limits") == LIMITS and
            plan.get("noop_action") == 0 and isinstance(names, list) and bool(names) and names[0] == "noop" and
            all(isinstance(name, str) for name in names) and len(set(names)) == len(names) and
            plan.get("plan_id") == content_id({name: value for name, value in plan.items() if name != "plan_id"}),
            "预热/两次重放计划身份异常")
    warmup, prefix, tail = (plan[name] for name in ("warmup_steps", "prefix_steps", "tail_frames"))
    require(type(warmup) is int and 1 <= warmup <= 256 and type(prefix) is int and prefix >= 1 and
            type(tail) is int and 2 <= tail <= 8 and tail <= prefix + 1 and
            len(plan["prefix_actions"]) == prefix and plan["actions"] == [0] * warmup + plan["prefix_actions"] and
            all(type(action) is int and 0 <= action < len(names) for action in plan["prefix_actions"]) and
            plan["control_start_frame"] == warmup + prefix and plan["maximum_new_env_steps"] == 2 * (warmup + prefix),
            "预热或前缀动作/窗口/计数被修改")


def validate_real_history(arrays, events, plan):
    """Never discard warmup, fake reset, or fill actions from a reference."""
    validate_plan(plan)
    size = plan["control_start_frame"] + 1
    require(all(len(arrays[name]) == size for name in ("image", "heatmap", "features", "action", "is_first",
                "is_last", "is_terminal", "obs_reward")) and len(events) == size, "必须保留完整真实reset/预热/前缀历史")
    require(bool(arrays["is_first"][0]) and not arrays["is_first"][1:].any() and
            not arrays["is_last"].any() and not arrays["is_terminal"].any(), "历史含伪reset或提前终止")
    count = len(plan["action_names"])
    require(arrays["action"].shape == (size, count) and np.all(arrays["action"][0] == 0) and
            np.array_equal(arrays["action"][1:], np.eye(count, dtype=np.float32)[plan["actions"]]),
            "预热noop或真实incoming动作没有逐步保留")
    require(all(event.get("frame") == frame and "native_actions" in event and not event.get("error")
                for frame, event in enumerate(events)), "真实原生动作/事件缺失或环境执行错误")
