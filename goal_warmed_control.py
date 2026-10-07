"""Fixed warmed-up reference scripts and recent-RGB start acceptance.

Original stability/control modules stay unchanged so accepted artifacts
remain readable. Only freshly collected endpoints become control goals.
"""

import numpy as np

import goal_start_stability as starts


FORMAT = "ls_imagine_warmed_residual_control_benchmark_v1"
EVALUATION_FORMAT = "ls_imagine_warmed_residual_control_evaluation_v1"
PROTOCOL = "warmed_residual_six_conditions_full_physical_recent_rgb_v1"


def comparison_events(events):
    """Copy live/saved events into save_session's JSON value representation.

    Lists containing NumPy arrays cannot be compared with Python ==.
    Preserve values, action ordering and missing fields; do not alter the
    real events, observations or arrays used by the policy.
    """
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    starts.require(isinstance(events, (list, tuple)) and all(isinstance(event, dict) for event in events),
                   "比较需要逐帧真实事件列表")
    return plain(events)


def make_reference_plan(stability_plan, branch_actions, horizon, minimum, design):
    starts.validate_plan(stability_plan)
    plan = dict(protocol=PROTOCOL, stability_plan=stability_plan,
                branch_actions=branch_actions, horizon=horizon, min_goal_distance=minimum,
                design=design, control_start_frame=stability_plan["control_start_frame"],
                maximum_reference_env_steps=2 * (stability_plan["control_start_frame"] + horizon),
                goal_source="new_real_endpoint_only", old_goal_vectors_reused=False)
    plan["reference_plan_id"] = starts.content_id(plan)
    validate_reference_plan(plan)
    return plan


def validate_reference_plan(plan):
    starts.require(plan.get("protocol") == PROTOCOL and plan.get("goal_source") == "new_real_endpoint_only" and
                   plan.get("old_goal_vectors_reused") is False and plan.get("reference_plan_id") == starts.content_id({
                       key: value for key, value in plan.items() if key != "reference_plan_id"}), "新参考计划身份异常")
    preflight = plan["stability_plan"]
    starts.validate_plan(preflight)
    horizon, branches = plan["horizon"], plan["branch_actions"]
    starts.require(type(horizon) is int and 1 <= horizon <= 16 and isinstance(branches, list) and len(branches) == 2 and
                   all(isinstance(actions, list) and len(actions) == horizon and all(
                       type(action) is int and 0 <= action < len(preflight["action_names"]) for action in actions) for actions in branches) and
                   branches[0] != branches[1] and 0 < plan["min_goal_distance"] <= 2 and
                   plan["control_start_frame"] == preflight["control_start_frame"] and
                   plan["maximum_reference_env_steps"] == 2 * (preflight["control_start_frame"] + horizon),
                   "参考脚本长度/动作/控制起点或预算异常")


def prefix(arrays, events, plan):
    """Validate the actual entire warmup/prefix; never discard old history."""
    validate_reference_plan(plan)
    size = plan["control_start_frame"] + 1
    subset = {name: value[:size] for name, value in arrays.items() if name != "telemetry"}
    actual_events = events[:size]
    starts.validate_real_history(subset, actual_events, plan["stability_plan"])
    subset["telemetry"] = [event["telemetry"] for event in actual_events]
    return subset, actual_events


def validate_reference_history(arrays, events, plan, branch):
    prefix(arrays, events, plan)
    starts.require(type(branch) is int and branch in (0, 1), "参考分支编号异常")
    first, horizon = plan["control_start_frame"], plan["horizon"]
    size = first + horizon + 1
    starts.require(len(events) == size and all(len(arrays[name]) == size for name in (
        "image", "heatmap", "features", "action", "obs_reward", "is_first", "is_last", "is_terminal")),
        "新参考必须完整执行预声明脚本；不静默缩短或补跑")
    starts.require(not arrays["is_first"][1:].any() and not (arrays["is_last"][:-1] | arrays["is_terminal"][:-1]).any(),
                   "新参考含伪reset或跨真实结束")
    expected = np.eye(len(plan["stability_plan"]["action_names"]), dtype=np.float32)[plan["branch_actions"][branch]]
    starts.require(np.array_equal(arrays["action"][first + 1:], expected) and all(
        event.get("frame") == frame and "native_actions" in event and not event.get("error") for frame, event in enumerate(events)),
        "新参考真实下一动作/原生事件与固定脚本不同")


def start_result(summary, strict):
    full, recent = summary["full_history"], summary["recent_window"]
    return dict(protocol=PROTOCOL, passed=summary["candidate_preflight_passed"], same_hidden_state=False,
                strict_pair_passed=strict["passed"],
                failed_checks=sorted(set(full["physical_failed_checks"] + recent["rgb_failed_checks"])),
                acceptance_limits=dict(starts.LIMITS), full_physical_history=full,
                recent_rgb_window=recent, control_start=summary["control_start"],
                physical_failure_frames=summary["physical_failure_frames"], rgb_failure_frames=summary["rgb_failure_frames"],
                diagnostic_only={key: full[key] for key in ("heatmap_mae", "state_relative_l2", "rgb_mae", "rgb_p99")})


def summarize(rows, design, plan):
    import goal_residual_control as residual

    result = residual.summarize(rows, design)
    result.update(comparison_protocol=PROTOCOL, reference_plan_id=plan["reference_plan_id"],
                  warmup_steps=plan["stability_plan"]["warmup_steps"],
                  original_prefix_steps=plan["stability_plan"]["prefix_steps"],
                  control_start_frame=plan["control_start_frame"], tail_frames=plan["stability_plan"]["tail_frames"],
                  goal_source=plan["goal_source"], old_goal_vectors_reused=False)
    result["interpretation"][3] = "完整物理/动作历史及预声明近期RGB门槛决定起点；原严格历史保留，不声称隐藏状态相同。"
    return result
