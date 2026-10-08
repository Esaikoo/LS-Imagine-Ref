"""Fixed worker350 confirmation; the original gate and 16-step budget remain."""
import copy
from pathlib import Path

import numpy as np

import goal_library as gl
import goal_random_control as pilot
import goal_repaired_control as original

FORMAT = "ls_imagine_continuation_control_design_v1"
EVALUATION_FORMAT = "ls_imagine_continuation_control_evaluation_v1"
PROTOCOL = "fixed_continuation_repair_350_local_autonomous_confirmation_v1"
RANDOMIZATION_SEED = 4
REPETITIONS = 5
GATE = copy.deepcopy(original.GATE)
LOCKED_FIELDS = ("worker_version", "execution_policy", "maximum_new_env_steps", "gate",
                 "target_content_id", "scenario", "visual_preprocessing", "environment_fingerprint",
                 "action_names", "warmup_steps", "prefix_actions", "start_actions", "control_start_frame", "horizon")


def validate_predeclared(saved):
    gl.require(type(saved.get("randomization_seed")) is int and saved["randomization_seed"] == 4,
               "预声明计划随机种子缺失或不同")
    gl.require(saved.get("schedule") == pilot.schedule(0, REPETITIONS, RANDOMIZATION_SEED) and
        saved.get("worker_version") == 350 and saved.get("execution_policy") == "mode" and
        saved.get("planned_trials") == 30 and saved.get("maximum_new_env_steps") == 2400 and
        saved.get("gate") == GATE and saved.get("warmup_steps") == 32 and
        len(saved.get("prefix_actions", [])) == 32 and saved.get("start_actions") == [0] * 32 + saved["prefix_actions"] and
        saved.get("control_start_frame") == 64 and saved.get("horizon") == 16 and
        saved.get("actual_endpoint_frame") == 80 and saved.get("independent_fresh_resets") is True and
        saved.get("matched_start_comparison") is False and saved.get("no_retries_or_replacements") is True and
        saved.get("implementation_pending") is True and saved.get("t06_approved") is False and
        saved.get("checkpoint_binding") == "fixed step50 latest plus new independent verify; no best selection",
        "预声明worker350/30次顺序/原16步预算或门槛不同")
    gl.require(saved.get("scenario", {}).get("seed") == 0 and
        str(saved["scenario"].get("world_seed")) == "1" and
        len(saved.get("action_names", [])) == 12 and saved["action_names"][0] == "noop" and
        all(type(a) is int and 0 <= a < 12 for a in saved["start_actions"]),
        "预声明场景或动作接口不同")


def validate_plan(plan):
    gl.require(plan.get("format") == FORMAT and plan.get("comparison_protocol") == PROTOCOL and
               plan.get("gate") == GATE, "确认协议或预声明行为门槛被修改")
    locked = plan["preregistered_confirmation"]
    validate_predeclared(locked)
    design = plan["design"]
    pilot.validate_design(design)
    gl.require(all(plan.get(k) == locked[k] for k in LOCKED_FIELDS) and
        design["seed"] == 0 and design["randomization_seed"] == RANDOMIZATION_SEED and
        design["repetitions"] == REPETITIONS and design["execution_policy"] == "mode" and
        design["schedule"] == locked["schedule"] and plan.get("source_worker_version") == 300 and
        plan.get("additional_updates") == 50 and plan.get("same_hidden_state") is False and
        plan.get("matched_start_comparison") is False and plan.get("old_failures_unchanged") is True and
        plan.get("input_identity", {}).get("worker_version") == 350 and
        plan["input_identity"].get("additional_updates") == 50,
        "固定worker/执行方式/30次顺序/预热或16步预算被修改")


def passed(record, names):
    return set(names) <= {r["name"] for r in record.get("checks", []) if r.get("level") == "PASS"}


def validate_source_records(checkpoint, check_dir, checked, trained, verified, payload):
    """Check only accepted records/counters; no optimizer or policy is built."""
    checkpoint, check_dir = Path(checkpoint).resolve(), Path(check_dir).resolve()
    gl.require(checkpoint.name == "latest.pt" and
        checkpoint.parent == Path(trained["arguments"]["output_dir"]).resolve() and
        Path(verified["arguments"]["checkpoint"]).resolve() == checkpoint,
        "只接受本次固定50步生产latest，不接受best或验收快照")
    for record, command in ((checked, "check"), (trained, "train"), (verified, "verify")):
        gl.require(record.get("command") == command and record.get("status") in ("passed", "passed_with_warnings"),
                   f"需要已通过的续段修复{command}记录")
        gl.require(record.get("repair_format") == "ls_imagine_continuation_repair_worker_v1" and
            record.get("repair_identity") == record.get("input_identity") == payload["input_identity"] and
            record.get("options") == payload["options"] and record.get("repair_code") == checked["repair_code"] and
            record.get("source_worker_version") == 300 and record.get("planned_repair_updates") == 50 and
            record.get("confirmation_plan") == checked["confirmation_plan"] and
            record.get("retention_plan") == checked["retention_plan"] and
            record.get("source_inputs_before") == record.get("source_inputs_after") and
            record.get("new_env_steps") == 0 and record.get("worker_control_actions") == 0 and
            record.get("behavior_accepted") is False and record.get("t06_approved") is False,
            "续段修复报告/代码/计划/来源或冻结计数不同")
    gl.require(Path(checked["arguments"]["output_dir"]).resolve() == check_dir and
        all(Path(r["arguments"]["check_dir"]).resolve() == check_dir for r in (trained, verified)) and
        checked["counters"]["step"] == 0 and checked.get("optimizer_updates") == 0 and
        payload["counters"] == trained.get("counters") == verified.get("counters") and
        payload["counters"]["step"] == 50 and trained.get("additional_updates") == 50 and
        trained.get("worker_version") == verified.get("worker_version") == 350 and
        trained.get("optimizer_updates") == 100 and verified.get("verification_only") is True and
        verified.get("verification_updates_per_copy") == 1 and
        verified.get("verification_updates_not_saved_to_training") is True and verified.get("optimizer_updates") == 4,
        "需要生产源300加50步worker350，验收副本更新不能混入生产计数")
    gl.require(passed(checked, ("real_continuation_supervision", "warm_start", "sampler_contract",
            "training_split_guard", "sample_holdout_guard", "sample_worker_guard", "distribution_contract",
            "diagnostic_contracts", "initial_metrics", "no_updates", "source_inputs_unchanged")) and
        passed(trained, ("paired_training", "retention_review", "training", "frozen_dependencies", "source_inputs_unchanged")) and
        passed(verified, ("strict_load", "inference_artifact_guard", "roundtrip", "next_update_equivalence",
                          "frozen_dependencies", "source_inputs_unchanged")),
        "需要本轮check/固定训练及独立verify完整通过")
    validate_predeclared(checked["confirmation_plan"])


def validate_trace(arrays, events, trace, plan, cell, index, dimension):
    from scripts.t05_repaired_control import validate_trace as original_trace

    validate_plan(plan)
    gl.require(type(index) is int and 0 <= index < 30 and plan["design"]["schedule"][index] == cell,
               "保存trial索引或顺序不同")
    count = original_trace(arrays, events, trace, plan, dimension)
    seed = (plan["design"]["randomization_seed"] + index * 1009 + 701) % (2**31 - 1)
    issued = None if cell["mode"] == "no_goal" else 1 - cell["target"] if cell["mode"] == "swapped_goal" else cell["target"]
    gl.require(trace.get("worker_version") == 350 and trace.get("source_worker_version") == 300 and
        trace.get("additional_updates") == 50 and trace.get("model_id") == plan["input_identity"]["model_id"] and
        trace.get("comparison_protocol") == PROTOCOL and trace.get("design_id") == plan["design_id"] and
        trace.get("trial_index") == index and trace.get("control_start_frame") == 64 and
        trace.get("condition") == cell["mode"] and trace.get("evaluated_target") == cell["target"] and
        trace.get("issued_target") == issued, "保存worker350/模型/目标条件身份不同")
    gl.require(trace.get("action_seed") == seed and
        np.array_equal(trace["uniforms"], np.random.RandomState(seed).uniform(size=16)[:count]),
        "保存动作种子或随机数与design不同")
    gl.require(arrays["features"].shape == (64 + count + 1, plan["input_identity"]["input_identity"]["feature_dim"]) and
        np.isfinite(arrays["features"]).all(), "保存因果状态形状或数值不同")
    return count

def validate_rows(rows, plan):
    validate_plan(plan)
    schedule = plan["design"]["schedule"]
    gl.require(len(rows) <= len(schedule) and len({pilot.key(row) for row in rows}) == len(rows) and
               [pilot.key(row) for row in rows] == [pilot.key(cell) for cell in schedule[:len(rows)]],
               "确认记录有重复、遗漏或计划外trial；禁止续跑和补样")
    for index, row in enumerate(rows):
        gl.require(row.get("trial_index") == index and row.get("worker_version") == 350 and
            row.get("source_worker_version") == 300 and row.get("additional_updates") == 50 and
            row.get("model_id") == plan["input_identity"]["model_id"], "确认汇总worker350或trial身份不同")
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
                  worker_version=plan["worker_version"], source_worker_version=300, additional_updates=50,
                  gate=copy.deepcopy(GATE),
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


