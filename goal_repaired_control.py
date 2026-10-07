"""Predeclared local confirmation of the fixed, repaired T05 worker.

Numerical criteria identify a candidate for video/physical review. They do
not authorize T06 or establish general navigation or task success.
"""

import copy

import numpy as np

import goal_library as gl
import goal_random_control as pilot


FORMAT = "ls_imagine_repaired_control_design_v1"
EVALUATION_FORMAT = "ls_imagine_repaired_control_evaluation_v1"
PROTOCOL = "fixed_repair_300_local_autonomous_confirmation_v1"
RANDOMIZATION_SEED = 1
REPETITIONS = 5
GATE = dict(version="t05_local_macro_collection_gate_v1", planned_trials=30,
            repetitions_per_target_condition=5, minimum_directional_trials_per_target=4,
            positive_progress_strict=True, closer_to_evaluated_target_strict=True,
            positive_goal_mean_per_target=True, beat_both_control_means_per_target=True,
            missing_measurements_fail=True, endpoint="fixed budget or actual early terminal observation",
            human_video_and_pose_review_required=True, scope="one calibrated world and two local visual targets")


def validate_plan(plan):
    gl.require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL and
               plan.get("gate") == GATE, "确认协议或预声明行为门槛被修改")
    design = plan["design"]
    pilot.validate_design(design)
    gl.require(design["seed"] == 0 and design["randomization_seed"] == RANDOMIZATION_SEED and
               design["repetitions"] == REPETITIONS and design["execution_policy"] == "mode" and
               plan["warmup_steps"] == 32 and len(plan["prefix_actions"]) == 32 and
               plan["start_actions"] == [0] * 32 + plan["prefix_actions"] and
               plan["control_start_frame"] == 64 and plan["horizon"] == 16 and
               len(plan["action_names"]) == 12 and plan["action_names"][0] == "noop" and
               all(type(action) is int and 0 <= action < 12 for action in plan["start_actions"]) and
               plan["worker_version"] == 300 and plan["repair_updates"] == 100 and
               plan["maximum_new_env_steps"] == 2400 and
               plan["same_hidden_state"] is False and plan["matched_start_comparison"] is False,
               "固定worker/执行方式/30次顺序/预热或16步预算被修改")


def validate_rows(rows, plan):
    validate_plan(plan)
    schedule = plan["design"]["schedule"]
    gl.require(len(rows) <= len(schedule) and len({pilot.key(row) for row in rows}) == len(rows) and
               [pilot.key(row) for row in rows] == [pilot.key(cell) for cell in schedule[:len(rows)]],
               "确认记录有重复、遗漏或计划外trial；禁止续跑和补样")
    for row in rows:
        gl.require(type(row["execution_valid"]) is bool, "真实执行有效标志异常")
        if not row["execution_valid"]:
            continue
        gl.require(row.get("saved_execution_verified") is True and row.get("video_verified") is True and
                   type(row["actual_steps"]) is int and 1 <= row["actual_steps"] <= plan["horizon"] and
                   (row["end_reason"] == "fixed_budget" and row["actual_steps"] == plan["horizon"] or
                    row["end_reason"] == "environment_done"), "真实终点、录像或保存接口未通过")
        values = [row[name] for name in pilot.METRICS]
        start = np.asarray(row["start_distances_to_both_goals"], dtype=np.float64)
        end = np.asarray(row["end_distances_to_both_goals"], dtype=np.float64)
        target = row["target"]
        gl.require(start.shape == end.shape == (2,) and np.isfinite(values).all() and
                   np.isfinite(start).all() and np.isfinite(end).all() and
                   np.all((start >= 0) & (start <= 2)) and np.all((end >= 0) & (end <= 2)) and
                   np.allclose([row["start_distance"], row["end_distance"], row["distance_improvement"],
                                row["target_preference_margin"], row["preference_improvement"]],
                               [start[target], end[target], start[target] - end[target],
                                end[1 - target] - end[target],
                                end[1 - target] - end[target] - start[1 - target] + start[target]],
                               atol=1e-6, rtol=1e-6), "确认终点测量缺失、非有限或与自身进展不同")


def summarize(rows, plan):
    validate_rows(rows, plan)
    result = pilot.summarize(rows, plan["design"])
    result.update(protocol=PROTOCOL, descriptive_pilot_only=False, local_confirmation_only=True,
                  worker_version=plan["worker_version"], gate=copy.deepcopy(GATE),
                  behavior_accepted=False, t06_approved=False, manual_review_required=True,
                  full_horizon_trials=sum(row["execution_valid"] and row["actual_steps"] == plan["horizon"]
                                          for row in rows),
                  early_terminal_trials=[dict(cell={name: row[name] for name in ("seed", "repeat", "target", "mode")},
                                               actual_steps=row["actual_steps"])
                                         for row in rows if row["execution_valid"] and
                                         row["end_reason"] == "environment_done"])
    valid = [row for row in rows if row["execution_valid"]]
    complete = len(rows) == GATE["planned_trials"]
    all_valid = complete and len(valid) == len(rows)
    reasons = []
    if not complete:
        reasons.append("complete_30_attempts")
    if not all_valid:
        reasons.append("all_real_interfaces_measurements_and_videos")
    targets = {}
    for target in (0, 1):
        cells = {mode: [row for row in rows if row["target"] == target and row["mode"] == mode]
                 for mode in pilot.MODES}
        measurements = {mode: [row for row in group if row["execution_valid"]] for mode, group in cells.items()}
        # A missing attempt is not silently removed from a five-trial mean.
        means = {mode: float(np.mean([row["distance_improvement"] for row in group]))
                 if len(group) == REPETITIONS else None for mode, group in measurements.items()}
        direction = sum(row["distance_improvement"] > 0 and row["target_preference_margin"] > 0
                        for row in measurements["goal"])
        advances = {mode: means["goal"] - means[mode] if means["goal"] is not None and means[mode] is not None
                    else None for mode in ("no_goal", "swapped_goal")}
        positive = means["goal"] is not None and means["goal"] > 0
        beats = all(value is not None and value > 0 for value in advances.values())
        directional = direction >= GATE["minimum_directional_trials_per_target"]
        passed = all(len(group) == REPETITIONS for group in measurements.values()) and positive and beats and directional
        targets[str(target)] = dict(planned_per_condition=REPETITIONS,
            attempted={mode: len(group) for mode, group in cells.items()},
            measured={mode: len(group) for mode, group in measurements.items()},
            positive_progress_and_correct_preference=direction, minimum_required=4,
            mean_progress=means, progress_advantages=advances,
            positive_goal_mean=positive, beats_both_controls=beats,
            directional_rule_passed=directional, numerical_rules_passed=passed)
        if not passed:
            reasons.append(f"target_{target}_direction_and_control_advantages")
    result["numerical_gate"] = dict(passed=all_valid and all(item["numerical_rules_passed"] for item in targets.values()),
        complete_attempts=complete, all_interfaces_valid=all_valid, by_target=targets, failed_rules=reasons,
        human_review_pending=True, behavior_accepted=False, t06_approved=False)
    result["interpretation"] = [
        "每目标/条件五条独立真实历史；全部尝试保留，不按起点差异筛选、不重试或补样。",
        "主结果为第16步真实终点；真实提前终止用最后观测并单列，不挑选最佳中间帧。",
        "数值门槛按两个目标分别判定，缺测不进入五次均值，不用pooled结果放行。",
        "数值达标后仍需核对视频、位置/朝向、起点平衡；本入口不自动批准T06。",
        "只涉及已标定世界和两个局部视觉目标，不宣称显著性、通用导航或采木成功提升。"]
    return result
