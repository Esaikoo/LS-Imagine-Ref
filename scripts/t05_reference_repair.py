"""Offline real-reference residual repair: check, train, verify; no environment."""

import argparse
import copy
import gc
import hashlib
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_goal_information_probe as information
import t05_reference_action_diagnose as historical

PLANNED_UPDATES = 100
CODE_FILES = ("goal_reference_repair.py", "scripts/t05_reference_repair.py", "goal_residual_worker.py",
              "goal_information_probe.py", "goal_bc.py", "goal_library.py", "long_horizon.py", "goal_sampling.py",
              "goal_reference_action_diagnose.py", "goal_reference_calibration.py", "goal_control.py",
              "scripts/t05_reference_action_diagnose.py", "scripts/t05_goal_information_probe.py")


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        written = self.save()
        failed = not written or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_REFERENCE_REPAIR_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def input_records(args):
    if args.command == "check":
        return dict(cache_dir=args.cache_dir, source_checkpoint=args.source_checkpoint,
                    residual_verify_dir=args.residual_verify_dir, calibration_dir=args.calibration_dir,
                    diagnosis_dir=args.diagnosis_dir)
    checked = legacy.read_json(baseline.project_path(args.check_dir) / "report.json")
    legacy.accepted(checked, "check")
    return {key: checked["arguments"][key] for key in
            ("cache_dir", "source_checkpoint", "residual_verify_dir", "calibration_dir", "diagnosis_dir")}


def historical_inputs(records):
    import goal_bc as bc
    import goal_library as gl

    args = argparse.Namespace(**records)
    args.checkpoint = records["source_checkpoint"]
    manifest, checked_dir, _, protected, files = historical.historical_inputs(args)
    diagnosis_dir = baseline.project_path(records["diagnosis_dir"])
    diagnosis = legacy.read_json(diagnosis_dir / "report.json")
    legacy.accepted(diagnosis, "analyze")
    required = ("historical_identity", "real_query_alignment", "recorded_measurement_roundtrip",
                "same_state_script_support", "no_training", "source_inputs_unchanged")
    gl.require(diagnosis.get("calibration_evaluation_id") == manifest["evaluation_id"] and
               diagnosis.get("input_identity") == manifest["design"]["input_identity"] and
               diagnosis.get("query_frames") == 64 and diagnosis.get("diagnosed_runs") == 4 and
               historical.history.passed_checks(diagnosis, required), "需要同一完整标定的已通过动作诊断")
    for name, digest in diagnosis["diagnosis_code"].items():
        gl.require(bc.file_hash(ROOT / name) == digest, "原动作诊断代码指纹改变")
    files += [diagnosis_dir / "report.json", diagnosis_dir / "diagnostics.json"]
    protected += [diagnosis_dir]
    return manifest, checked_dir, protected, files


def references_from_histories(manifest, checked_dir, cache, device):
    import numpy as np
    import torch
    import goal_action_choice_diagnose as choices
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl
    import goal_reference_action_diagnose as diagnosis
    import goal_reference_calibration as calibration

    plan = manifest["design"]
    library = gl.GoalLibrary.from_payload(cache.bundle["library"], device)
    before = gl.tensor_digest(library.state_dict(), {})
    arrays = {key: [] for key in ("features", "goals", "remaining", "labels", "train", "run", "target", "frame")}
    runs = []
    first, horizon, dimension = plan["control_start_frame"], plan["horizon"], cache.metadata["action_dim"]
    gl.require(first == 64 and horizon == 16 and diagnosis.common_prefix(plan["scripts"]) == 4,
               "当前局部修复只接受已诊断的64帧起点/16步/4步共同前缀")
    with np.load(checked_dir / "targets.npz", allow_pickle=False) as file:
        fixed = {key: file[key].copy() for key in ("goals", "images", "heatmaps")}
    gl.require(gl.tensor_digest({}, {key: value.tolist() for key, value in fixed.items()}) == plan["target_content_id"],
               "原固定视觉目标改变")
    for index, run in enumerate(manifest["rows"]):
        directory = baseline.project_path(run["artifact_dir"])
        trajectory = ctl.read_trajectory(directory)
        events = legacy.read_json(directory / "events.json")
        trace = legacy.read_json(directory / "script_trace.json")
        history_check = legacy.read_json(directory / "history_check.json")
        gl.require(history_check.get("own_causal_history_passed") is True and
                   history_check.get("saved_history_roundtrip") is True and history_check.get("reset_is_real") is True and
                   history_check.get("comparison_to_other_histories") is False and history_check.get("total_frames") == 81 and
                   history_check.get("start_frame") == first and
                   history_check.get("maximum_absolute_error") == run["own_history_max_error"] and
                   trajectory["features"].shape == (81, cache.states.shape[1]) and trace["script"] == run["script"],
                   "必须使用本局已验收的完整真实因果状态")
        script = plan["scripts"][run["script"]]
        gl.require(calibration.validate_execution(trajectory, events, plan["start_actions"], script, dimension) == horizon and
                   calibration.native_actions_equal(trace["actual_native_events"],
                       [event["native_actions"] for event in events[first + 1:]]), "真实原生动作/脚本不同")
        queries = diagnosis.validate_queries(trajectory, trace, first, horizon, script, dimension)
        endpoint = first + horizon
        observation = {key: torch.as_tensor(trajectory[key][endpoint:endpoint + 1], device=device)
                       for key in ("image", "heatmap")}
        own_goal = library(observation)[0].cpu().numpy().astype(np.float32)
        gl.require(np.allclose(own_goal, trace["measurements"][-1]["goal_feature"],
                   atol=choices.ATOL, rtol=choices.RTOL), "本局真实终点目标重编码不一致")
        for query in queries:
            arrays["features"].append(trajectory["features"][query["frame"]].copy())
            arrays["goals"].append(own_goal.copy())
            for key, value in dict(remaining=query["remaining"], labels=query["script_action"],
                    train=run["repeat"] == 0, run=index, target=run["script"], frame=query["frame"]).items():
                arrays[key].append(value)
        runs.append(dict(run=index, repeat=run["repeat"], target=run["script"], start_frame=first,
            endpoint_frame=endpoint, action_ids=script, trajectory_path=str(directory / "trajectory.npz"),
            trajectory_sha256=bc.file_hash(directory / "trajectory.npz"),
            events_sha256=bc.file_hash(directory / "events.json"),
            own_endpoint_rgb_sha256=hashlib.sha256(trajectory["image"][endpoint].tobytes()).hexdigest(),
            goal_source="this episode's actual frame-80 RGB/heatmap endpoint; not old target replacement"))
    arrays = {key: np.asarray(value, dtype=np.float32 if key in ("features", "goals") else
                             np.bool_ if key == "train" else np.int64) for key, value in arrays.items()}
    gl.require(gl.tensor_digest(library.state_dict(), {}) == before and
               all(not p.requires_grad and p.grad is None for p in library.parameters()), "目标编码器发生改变")
    return arrays, runs, fixed["goals"]


def load_inputs(args, report, records, manifest, checked_dir):
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_worker as residual
    import goal_reference_repair as repair

    source_path = baseline.project_path(records["source_checkpoint"])
    source, paths = residual.read_checkpoint(source_path)
    options = copy.deepcopy(source["options"])
    options["learning_rate"] = args.learning_rate if args.command == "check" else legacy.read_json(
        baseline.project_path(args.check_dir) / "report.json")["options"]["learning_rate"]
    cache_args = argparse.Namespace(command="check", cache_dir=records["cache_dir"],
        t04_verify_dir=str(paths["base_verify_report"].parent), **options)
    original, options = information.load_data(cache_args, report)
    base, references = residual.load_base(paths["base_checkpoint"], paths["base_verify_report"], original.cache)
    residual.validate_payload(source, residual.ResidualData(original.cache, source["options"],
                              residual.base_identity(base)).identity(), source["options"])
    gl.require(source["counters"]["step"] == 200, "本次修复必须从已诊断的第200步残差双分支开始")
    arrays, runs, fixed = references_from_histories(manifest, checked_dir, original.cache, args.device)
    lineage = dict(source_checkpoint_sha256=bc.file_hash(source_path), source_branch_updates=200,
        source_verify_sha256=bc.file_hash(baseline.project_path(records["residual_verify_dir"]) / "report.json"),
        calibration_evaluation_id=manifest["evaluation_id"], calibration_manifest_sha256=bc.file_hash(
            baseline.project_path(records["calibration_dir"]) / "calibration_manifest.json"),
        diagnosis_report_sha256=bc.file_hash(baseline.project_path(records["diagnosis_dir"]) / "report.json"),
        optimizer_initialization="fresh Adam; source weights only; not exact source training resume",
        reference_role="inspected development train/validation; no independent test or control claim")
    data = repair.RepairData(original.cache, options, residual.base_identity(base), arrays, runs, lineage, fixed)
    check_path = report.directory / "report.json" if args.command == "check" else baseline.project_path(args.check_dir) / "report.json"
    additional = dict(source_checkpoint=source_path,
        source_verify_report=baseline.project_path(records["residual_verify_dir"]) / "report.json",
        calibration_manifest=baseline.project_path(records["calibration_dir"]) / "calibration_manifest.json",
        diagnosis_report=baseline.project_path(records["diagnosis_dir"]) / "report.json", repair_check_report=check_path)
    references.update({name: dict(source_path=path.resolve(), sha256=bc.file_hash(path))
                       for name, path in additional.items() if name != "repair_check_report"})
    # The check report is finalized after all guards and input-unchanged checks.
    references["repair_check_report"] = dict(source_path=check_path.resolve(), sha256=None)
    code = {name: bc.file_hash(ROOT / name) for name in CODE_FILES}
    if args.command != "check":
        checked = legacy.read_json(check_path)
        gl.require(checked.get("repair_format") == repair.FORMAT and checked.get("repair_code") == code and
                   checked.get("repair_identity") == data.identity() and checked.get("options") == options and
                   checked.get("planned_repair_updates") == PLANNED_UPDATES and
                   checked.get("backend") == bc.backend_info(args.device), "修复check配置/数据/代码/后端不同")
        gl.require(historical.history.passed_checks(checked, ("real_reference_supervision", "warm_start",
            "sampler_contract", "training_split_guard", "source_inputs_unchanged", "no_updates")), "修复check缺少必要验收")
        gl.require(bc.file_hash(check_path.parent / "reference_supervision.npz") == checked["reference_supervision_sha256"],
                   "check保存的真实监督产物改变")
        references["repair_check_report"]["sha256"] = bc.file_hash(check_path)
    base = {key: value for key, value in base.items() if key not in ("optimizers", "candidate")}
    base["frozen_bundle"] = original.cache.bundle
    report.data.update(repair_format=repair.FORMAT, architecture=residual.ARCHITECTURE, semantics=repair.SEMANTICS,
        repair_identity=data.identity(), input_identity=data.identity(), options=options,
        repair_code=code, backend=bc.backend_info(args.device), planned_repair_updates=PLANNED_UPDATES,
        sampling_plan=repair.sampling_plan(options["batch_size"]), lineage=lineage)
    report.check("real_reference_supervision", "PASS", "4条完整真实局固定repeat0训练/repeat1留出；各32标签；状态64–79→真实incoming65–80；训练目标来自各局真实第80帧终点")
    return data, options, base, references, source


def make_trainer(inputs, device):
    import numpy as np
    import torch
    import goal_reference_repair as repair

    data, options, base, references, source = inputs
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return repair.Trainer(data, base, references, options, source, device)


def check(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_repair as repair
    import goal_residual_worker as residual

    trainer = make_trainer(inputs, args.device)
    data, _, _, _, source = inputs
    for name in residual.NAMES:
        gl.require(t01.same(trainer.model.heads[name].state_dict(), source["heads"][name]) and
                   not trainer.optimizers[name].state, "修复初始化未严格继承源权重或恢复了旧优化器")
    report.check("warm_start", "PASS", "分别继承源goal/no_goal第200步权重；新优化器步0；相同追加预算，不声称两分支初始权重相同")
    rng = trainer.sampler.get_state()
    first = trainer.draw()
    trainer.sampler.set_state(rng)
    second = trainer.draw()
    gl.require(t01.same(first, second), "修复采样RNG恢复后批次不同")
    trainer.sampler.set_state(rng)
    plan = trainer.sampling_plan()
    report.check("sampler_contract", "PASS", f"每批{plan['original_rows_per_batch']}原训练+"
                 f"{plan['reference_rows_per_batch']}参考训练（每目标{plan['reference_rows_per_target_per_batch']}），"
                 "有放回；同批双分支；保存恢复采样RNG一致")
    t02.rejection(report, "sampling_budget_guard", lambda: repair.sampling_plan(255), "8的")
    bad = {key: value.copy() for key, value in data.reference_arrays.items()}
    bad["train"][0] = not bad["train"][0]
    t02.rejection(report, "training_split_guard", lambda: repair.validate_reference(bad, data.runs,
        data.identity()["horizon"], data.identity()["feature_dim"], data.identity()["goal_dim"], data.identity()["action_dim"]), "整局划分")
    payload = trainer.payload()
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "protocol_guard", lambda: trainer.restore(dict(payload, sampling_protocol="wrong")), "采样")
    x, g, r, _ = data.reference_batch(data.reference_rows["validation"], args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        raw = trainer.model.base_preferences(x, g, r) + trainer.model.correction("goal", x, g, r)
        gl.require(torch.equal(trainer.model.distribution(raw).probs, trainer.model.action_dist(x, g, r).probs),
                   "修复部署分布重复unimix或偏好相加不同")
        gl.require(torch.equal(trainer.model.action_dist(x, g, r, "no_goal").probs,
            trainer.model.action_dist(x, torch.flip(g, [0]), r, "no_goal").probs), "无目标分支依赖目标")
    report.check("distribution_contract", "PASS", "保留原宏动作、归一化与raw偏好相加；只执行一次原onehot unimix；no_goal严格目标不变")
    np.savez_compressed(report.directory / "reference_supervision.npz", **data.reference_arrays)
    report.data["reference_supervision_sha256"] = bc.file_hash(report.directory / "reference_supervision.npz")
    baseline.write_json(report.directory / "coverage.json", repair.coverage(data, args.device))
    report.check("real_training_coverage", "PASS", "保存原池动作/remaining与去重动作状态计数，及同预算训练起点32近邻动作支持；近邻不看未来且不用于训练选择")
    trainer.model.check_frozen(content=True)
    report.data["counters"] = trainer.counters()
    report.data["optimizer_updates"] = 0
    report.check("no_updates", "PASS", "仅初始化、采样和前向检查；两分支更新0、新环境步0；无RSSM构造或MineDojo/MineCLIP")


def evaluation_outputs(trainer, report):
    import goal_reference_repair as repair

    metrics, frames = repair.evaluate(trainer)
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    baseline.write_csv(report.directory / "frame_metrics.csv", [dict(step=trainer.step, **row) for row in frames])
    local = metrics["reference"]["fixed_old_targets"]["by_split"]["validation"]["all_rows"]
    original = metrics["original_cache"]["validation_full"]["row_weighted"]
    print(f"[REPAIR] step={trainer.step} local_goal_nll={local['goal_nll']:.6f} "
          f"no_goal_gap={local['no_goal_minus_goal_nll']:.6f} zero_gap={local['zero_minus_goal_nll']:.6f} "
          f"swap_gap={local['swapped_minus_goal_nll']:.6f} original_goal_nll={original['goal_nll']:.6f}", flush=True)
    return metrics


def train(args, report, inputs):
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_repair as repair

    trainer = make_trainer(inputs, args.device)
    digest = trainer.reference["repair_check_report"]["sha256"]
    if args.resume:
        saved, _ = repair.read_checkpoint(baseline.project_path(args.resume), trainer, digest)
        trainer.restore(saved)
        report.check("resume", "PASS", f"精确恢复新修复实验，追加步={trainer.step}；不恢复旧训练优化器")
    gl.require(trainer.step < PLANNED_UPDATES, "固定100步修复预算已用完；不能反复延长或覆盖源结果")
    estimate = 3 * bc.tensor_storage_bytes(trainer.model.heads.state_dict()) + 1024**2
    report.data["storage_preflight"] = bc.require_disk_space(report.directory, 3 * estimate)
    report.check("storage_preflight", "PASS", f"完整双头快照约{estimate / 2**20:.1f}MiB；只存新头/优化器和共享依赖引用")
    initial = evaluation_outputs(trainer, report)
    baseline.write_json(report.directory / "initial_metrics.json", initial)
    report.check("paired_training", "PASS", f"固定追加100步；新Adam学习率{trainer.options['learning_rate']:g}；每批相同真实标签更新两分支；不训练交换标签或强制动作差异")
    start_step = trainer.step
    report.require_writable()
    while trainer.step < PLANNED_UPDATES:
        values = trainer.update()
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % args.log_every == 0 or trainer.step == PLANNED_UPDATES:
            print(f"[TRAIN REPAIR] step={trainer.step}/{PLANNED_UPDATES} goal_nll={values['goal_nll']:.4f} "
                  f"no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
        if trainer.step % args.eval_every == 0 or trainer.step == PLANNED_UPDATES:
            final = evaluation_outputs(trainer, report)
        if trainer.step % args.save_every == 0 or trainer.step == PLANNED_UPDATES:
            repair.save_checkpoint(trainer, report.directory / "latest.pt", digest)
            baseline.write_json(report.directory / "sampling.json", dict(plan=trainer.sampling_plan(),
                start_step=start_step, end_step=trainer.step,
                cumulative_counts={key: value.tolist() for key, value in trainer.counts.items()},
                labels_per_branch=trainer.counters()["labels_seen_per_branch"], cumulative_including_resume=True))
    trainer.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle内容改变")
    report.data.update(counters=trainer.counters(), initial_metrics=initial, final_metrics=final,
                       optimizer_updates=trainer.counters()["optimizer_updates"],
                       source_branch_updates=200, total_branch_updates=200 + trainer.step)
    baseline.write_json(report.directory / "diagnostics.json", dict(initial=initial, final=final,
        fixed_budget_latest_primary=True, checkpoint_selected_by_validation=False,
        source_branch_updates=200, additional_branch_updates=trainer.step,
        total_branch_updates=200 + trainer.step, behavior_accepted=False, t06_approved=False,
        scope="one-world inspected development histories; offline support repair, not autonomous control"))
    report.check("training", "PASS", "两分支各追加100次更新；有限NLL/梯度；真实动作和本局终点监督；底座/WM更新0，新环境步0")
    report.check("frozen_dependencies", "PASS", "原底座/WM/目标bundle内容不变；独立新格式保存完整修复状态；固定预算latest为主，无best筛选")


def verify(args, report, inputs):
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_reference_repair as repair
    import goal_residual_worker as residual

    trainer = make_trainer(inputs, args.device)
    path = baseline.project_path(args.checkpoint)
    digest = trainer.reference["repair_check_report"]["sha256"]
    payload, _ = repair.read_checkpoint(path, trainer, digest)
    trainer.restore(payload)
    gl.require(trainer.step == PLANNED_UPDATES, "本轮验收需要固定100步latest，不用中途或best快照")
    report.check("strict_load", "PASS", "新格式双头/新优化器/修复计数/混合采样统计/RNG严格恢复；源200步与追加100步分开")
    t02.rejection(report, "legacy_loader_guard", lambda: residual.read_checkpoint(path), "需要残差worker")
    t02.rejection(report, "counter_guard", lambda: trainer.restore(dict(payload,
        counters=dict(payload["counters"], base_updates=1))), "计数")
    t02.rejection(report, "identity_guard", lambda: trainer.restore(dict(payload,
        input_identity=dict(payload["input_identity"], reference_id="wrong"))), "身份")
    counts = copy.deepcopy(payload["sampling_counts"])
    counts["reference_target"][0] += 1
    t02.rejection(report, "sample_count_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "计数")
    heads = copy.deepcopy(payload["heads"])
    heads["goal"]["state_projection.weight"] = heads["goal"]["state_projection.weight"][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, heads=heads)), "权重不兼容")
    artifact_path = report.directory / "roundtrip_verification.pt"
    artifact = repair.save_checkpoint(trainer, artifact_path, digest, verification_artifact=True)
    other = make_trainer(inputs, args.device)
    saved, _ = repair.read_checkpoint(artifact_path, other, digest)
    other.restore(saved, allow_verification=True)
    again = other.payload(True)
    for key in ("heads", "optimizers", "counters", "sampler_state", "rng_state", "sampling_counts", "input_identity", "options"):
        gl.require(t01.same(artifact[key], again[key]), f"修复保存恢复{key}不同")
    inference = repair.load_policy(path, trainer.data.cache, args.device)
    x, g, r, _ = trainer.data.reference_batch(trainer.data.reference_rows["validation"], args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            gl.require(torch.equal(inference.action_dist(x, g, r, mode).probs,
                       trainer.model.action_dist(x, g, r, mode).probs) and
                       torch.equal(trainer.model.action_dist(x, g, r, mode).probs,
                       other.model.action_dist(x, g, r, mode).probs), "修复训练/恢复/推理分布不同")
        gl.require(torch.equal(inference.action_dist(x, g, r, "no_goal").probs,
                   inference.action_dist(x, torch.flip(g, [0]), r, "no_goal").probs), "推理无目标分支依赖目标")
    t02.rejection(report, "inference_artifact_guard", lambda: repair.load_policy(artifact_path, trainer.data.cache,
                  args.device), "合成验收")
    report.data.update(model_id=inference.model_id, worker_version=inference.worker_version)
    report.check("roundtrip", "PASS", "双头/优化器/计数/混合采样与RNG往返一致；独立推理无优化器，与训练四种分布逐值相同")
    metrics = evaluation_outputs(trainer, report)
    rng = bc.capture_rng(args.device)
    left = trainer.update()
    bc.restore_rng(rng, args.device)
    right = other.update()
    gl.require(left["shared_batch_id"] == right["shared_batch_id"], "恢复后下一真实混合batch不同")
    for name in ("goal", "no_goal"):
        gl.require(t01.same(trainer.model.heads[name].state_dict(), other.model.heads[name].state_dict()) and
                   t01.same(trainer.optimizers[name].state_dict(), other.optimizers[name].state_dict()), "下一修复更新不同")
    gl.require(t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and
               t01.same(trainer.counts, other.counts) and trainer.counters() == other.counters(), "下一更新后计数或随机数不同")
    trainer.model.check_frozen(content=True)
    other.model.check_frozen(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle改变")
    report.data.update(counters=payload["counters"], metrics=metrics, verification_updates_per_copy=1,
                       optimizer_updates=4, verification_only=True)
    report.check("next_update_equivalence", "PASS", "两份内存副本下一次真实混合BC更新/优化器/采样计数一致；源latest不写回")
    report.check("frozen_dependencies", "PASS", "底座/WM/目标bundle内容不变；验收快照禁止resume/控制；未构造RSSM或启动环境")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    for name in ("cache-dir", "source-checkpoint", "residual-verify-dir", "calibration-dir", "diagnosis-dir"):
        checking.add_argument("--" + name, required=True)
    checking.add_argument("--learning-rate", type=float, default=1e-4)
    training = commands.add_parser("train")
    training.add_argument("--check-dir", required=True)
    training.add_argument("--resume", help="仅精确恢复新修复快照；输出使用另一个新目录")
    for name in ("eval-every", "log-every", "save-every"):
        training.add_argument("--" + name, type=int, default=25)
    verification = commands.add_parser("verify")
    verification.add_argument("--check-dir", required=True)
    verification.add_argument("--checkpoint", required=True)
    for command in (checking, training, verification):
        command.add_argument("--output-dir", required=True)
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "train" and min(args.eval_every, args.log_every, args.save_every) < 1:
        parser.error("输出周期必须为正整数")
    directory = baseline.project_path(args.output_dir)
    try:
        records = input_records(args)
        manifest, checked_dir, protected, files = historical_inputs(records)
        import goal_bc as bc
        cache_manifest = legacy.read_json(baseline.project_path(records["cache_dir"]) / "cache_manifest.json")
        source = cache_manifest["source_metadata"]
        files += [Path(source["checkpoint"]["path"]), Path(source["library_path"])]
        files += [Path(entry["path"]) for entry in source["episodes"]]
        files += [Path(cache_manifest["dataset_dir"]) / name for name in ("segments_manifest.json", "segments.npz", "report.json")]
        files += [Path(cache_manifest["t03_verify_dir"]) / "report.json", *[ROOT / name for name in CODE_FILES]]
        if args.command != "check":
            files += [baseline.project_path(args.check_dir) / name for name in ("report.json", "reference_supervision.npz")]
            protected += [baseline.project_path(args.check_dir)]
            checkpoint = args.resume if args.command == "train" else args.checkpoint
            if checkpoint:
                files.append(baseline.project_path(checkpoint))
                protected.append(baseline.project_path(checkpoint).parent)
        protected += [path.parent.resolve() for path in files if ROOT not in path.resolve().parents]
        protected += [baseline.project_path(records["cache_dir"]), Path(cache_manifest["dataset_dir"])]
        if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
            parser.error("输出必须独立于旧结果、模型、缓存、源回放、check和输入checkpoint")
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    report.data.update(new_env_steps=0, worker_control_actions=0, behavior_accepted=False, t06_approved=False)
    try:
        import torch
        import goal_library as gl
        before = {str(path): baseline.file_signature(path) for path in dict.fromkeys(files)}
        report.data["source_inputs_before"] = before
        report.require_writable()
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        rng = bc.capture_rng(args.device)
        report.check("historical_identity", "PASS", "绑定完整标定/sample/源残差验收及成功动作诊断SHA256；旧代码/失败不改写")
        inputs = load_inputs(args, report, records, manifest, checked_dir)
        {"check": check, "train": train, "verify": verify}[args.command](args, report, inputs)
        gl.require(not any(name == "minedojo" or name.startswith("minedojo.") or
                   name == "mineclip" or name.startswith("mineclip.") for name in sys.modules), "离线修复不应导入环境/MineCLIP")
        report.check("scope", "WARN", "同世界、已检查的开发轨迹修复；报告旧固定目标与本局终点两组指标及原整局留出；离线收益不批准T06，下一步需新自主确认")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {name: baseline.file_signature(Path(name)) for name in before}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                             "源模型/回放/标定/诊断/check与旧失败产物大小、修改时间不变；只写独立修复目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
