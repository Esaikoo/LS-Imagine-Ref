"""T05 paired small action predictors: check, train, verify; no environment."""

import argparse
import copy
from datetime import datetime
import gc
import json
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


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        label = {"check": "T05_INFORMATION_CHECK", "train": "T05_INFORMATION_TRAIN", "verify": "T05_INFORMATION_VERIFY"}[self.data["command"]]
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def options_from(args, base):
    import goal_information_probe as probe

    options = copy.deepcopy(base)
    for key in probe.DEFAULTS:
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    probe.validate_options(options)
    return options


def load_data(args, report):
    import numpy as np
    import goal_bc as bc
    import goal_information_probe as probe
    import goal_library as gl
    import goal_learning_diagnostics as diag

    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    checked = read_json(baseline.project_path(args.check_dir) / "report.json") if args.command != "check" else None
    options = options_from(args, checked["options"] if checked else probe.DEFAULTS)
    if checked is not None:
        gl.require(checked.get("command") == "check" and checked.get("status") in ("passed", "passed_with_warnings") and
                   checked.get("architecture") == probe.ARCHITECTURE and checked["options"] == options,
                   "需要同架构/配置的已通过探针check")
        verification = baseline.project_path(checked["arguments"]["t04_verify_dir"])
    else:
        verification = baseline.project_path(args.t04_verify_dir)
    record = read_json(verification / "report.json")
    gl.require(record.get("command") == "verify" and record.get("status") in ("passed", "passed_with_warnings") and
               record.get("cache_id") == cache.cache_id and record.get("bundle_id") == cache.bundle["bundle_id"],
               "需要同缓存/冻结依赖的已通过T04 verify")
    required = {"real_supervision", "causal_cache", "next_update_equivalence", "frozen_ownership"}
    gl.require(required <= {row["name"] for row in record["checks"] if row["level"] == "PASS"}, "T04验收缺少真实监督/因果缓存/恢复检查")
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"] and dataset.metadata == cache.metadata["source_metadata"], "探针缓存与T03数据不同")
    data = probe.ProbeData(cache, options)
    t, a = cache.tables, dataset.arrays
    segments, states = t["worker_segment"], t["worker_state"]
    gl.require(np.array_equal(data.episodes[states], a["episode_index"][segments]) and
               np.array_equal(data.frames[states], a["end"][segments] - t["remaining"]) and
               np.all(data.frames[states] >= a["start"][segments]) and
               np.array_equal(t["goals"], a["goals"]), "探针当前帧/remaining/真实终点目标不一致")
    flags = np.array([entry["split"] == "train" for entry in dataset.metadata["episodes"]])
    gl.require(np.array_equal(t["worker_train"], flags[data.episodes[states]]) and
               np.array_equal(t["candidate_train"], flags[a["episode_index"]]), "探针跨整局训练/留出划分")
    diag.supervision_redundancy(cache, data.episodes)
    identity = data.identity()
    if checked is not None:
        gl.require(checked["input_identity"] == identity and
                   checked["t04_verify_sha256"] == bc.file_hash(verification / "report.json"), "探针缓存/查询或验收记录改变")
    report.data.update(input_identity=identity, options=options, architecture=probe.ARCHITECTURE,
                       t04_verify_dir=str(verification), t04_verify_sha256=bc.file_hash(verification / "report.json"))
    report.check("accepted_real_cache", "PASS", "完整校验T04缓存与T03片段；沿用真实下一动作、终点目标及整局划分；不构造encoder/RSSM/actor或MineCLIP")
    flags = t["worker_train"][data.rows]
    report.data["query_counts"] = dict(train=int(flags.sum()), validation=int((~flags).sum()),
                                      supported_goal_swaps=int((data.donors >= 0).sum()))
    report.check("query_contracts", "PASS", f"固定去重查询：训练{int(flags.sum())}、留出{int((~flags).sum())}；替换目标同split/remaining且跨局，仅用于评价")
    if args.command == "check":
        # Recheck labels against two source episodes without any WM rollout.
        episode_ids = [int(data.episodes[t["worker_state"][cache.worker_rows[split][0]]]) for split in ("train", "validation")]
        for index in episode_ids:
            rows = np.flatnonzero(data.episodes[states] == index)
            rows = rows[np.unique(np.linspace(0, len(rows) - 1, min(32, len(rows)), dtype=int))]
            episode = dataset.episode(index)
            action = episode["action"][data.frames[states[rows]] + 1]
            gl.require(np.array_equal(action.argmax(-1), t["action_id"][rows]) and
                       np.allclose(action, np.eye(cache.metadata["action_dim"], dtype=np.float32)[t["action_id"][rows]], atol=1e-6),
                       "探针下一动作与真实回放不一致")
        report.check("source_action_alignment", "PASS", f"抽查训练/留出{len(episode_ids)}局真实obs[t]→action[t+1]，不执行策略")
    return data, options


def make_trainer(data, options, device):
    import numpy as np
    import torch
    import goal_information_probe as probe

    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return probe.Trainer(data, options, device)


def check(args, report, data, options):
    import numpy as np
    import torch
    import goal_control as ctl
    import goal_library as gl

    trainer = make_trainer(data, options, args.device)
    gl.require(t01.same(trainer.heads["goal"].state_dict(), trainer.heads["no_goal"].state_dict()), "两组初始权重不同")
    rows = data.cache.worker_rows["validation"][:32]
    features, goals, remaining, _ = data.cache.worker_batch(rows, args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        left = trainer.heads["goal"](features, goals, remaining)
        right = trainer.heads["no_goal"](features, goals, remaining)
        gl.require(torch.equal(left, right), "同一输入的初始前向不同")
        baseline_logits = trainer.logits("no_goal", features, goals, remaining)
        gl.require(torch.equal(baseline_logits, trainer.logits("no_goal", features, torch.flip(goals, [0]), remaining)), "无目标探针依赖目标")
        state, goal, _ = trainer.heads["goal"].normalized_inputs(features, goals, remaining)
        scales = {"state_rms": float(state.square().mean().sqrt()), "goal_rms": float(goal.square().mean().sqrt())}
        gl.require(all(np.isfinite(value) and 0.9 < value < 1.1 for value in scales.values()), "独立归一化未产生接近单位尺度的输入")
        zero = trainer.heads["goal"].normalized_inputs(features, torch.zeros_like(goals), remaining)[1]
        gl.require(torch.count_nonzero(zero) == 0, "无目标归一化产生非零输入")
        probabilities = left.softmax(-1)
        gl.require(probabilities.shape == (len(rows), data.cache.metadata["action_dim"]) and
                   torch.allclose(probabilities.sum(-1), torch.ones(len(rows), device=args.device), atol=1e-6), "探针类别概率异常")
    count = sum(p.numel() for p in trainer.heads["goal"].parameters())
    report.data.update(parameters_per_head=count, normalized_inputs=scales, sampling=trainer.sampling_plan())
    report.check("paired_initialization", "PASS", f"两组相同初始权重及同输入前向；每头{count}参数；原actor未克隆")
    report.check("normalized_input_paths", "PASS", f"state_rms={scales['state_rms']:.4f} goal_rms={scales['goal_rms']:.4f}；目标/状态独立归一化，无目标输入恒零")
    report.check("goal_free_invariance", "PASS", "独立无目标组不依赖目标；归一化目标无可训练偏置；相同采样序列供两头更新")
    t02.rejection(report, "remaining_guard", lambda: trainer.heads["goal"](features, goals, remaining * 0), "remaining")
    t02.rejection(report, "shape_guard", lambda: trainer.heads["goal"](features, goals[:, :-1], remaining), "形状")
    # This format must be refused before the real worker loader constructs modules.
    t02.rejection(report, "worker_use_guard", lambda: ctl.load_model(trainer.payload(), data.cache, args.device), "真实 T04")
    report.check("scope", "WARN", "小型动作信息探针，不是可执行worker；尚未训练，不能据此确认目标信息或控制效果")
    report.data.update(counters=trainer.counters(), trainable_modules=["goal_head", "no_goal_head"])
    gl.require(trainer.step == 0 and all(not optimizer.state for optimizer in trainer.optimizers.values()), "离线check更新了探针")
    report.check("no_training", "PASS", "optimizer_updates=0；new_env_steps=0；只初始化新头并检查真实输入/接口")


def evaluation_outputs(trainer, report, final=False):
    import numpy as np
    import goal_information_probe as probe

    metrics, prediction = probe.evaluate(trainer)
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    result = metrics["validation_queries"]["row_weighted"]
    print(f"[INFORMATION] step={trainer.step} queries={result['rows']} goal_nll={result['goal_nll']:.6f} "
          f"no_goal_nll={result['no_goal_nll']:.6f} gap={result['no_goal_minus_goal_nll']:.6f} "
          f"swap_modes={result['goal_vs_swapped_goal_mode_change_fraction']}", flush=True)
    long = metrics["validation_queries"].get("remaining_13_plus")
    if long:
        print(f"[LONG BUDGET] step={trainer.step} rows={long['rows']} gap={long['no_goal_minus_goal_nll']:.6f}", flush=True)
    if final:
        diagnostic = dict(format=probe.FORMAT, architecture=probe.ARCHITECTURE, semantics=probe.SEMANTICS,
                          options=trainer.options, counters=trainer.counters(), metrics=metrics,
                          controlled_reachability_verified=False, task_success_evaluated=False,
                          interpretation="predictive information under this model capacity; not causal controllability")
        baseline.write_json(report.directory / "diagnostics.json", diagnostic)
        baseline.write_csv(report.directory / "query_metrics.csv", probe.query_records(trainer.data, prediction))
        arrays = dict(rows=prediction["rows"], donors=prediction["donors"])
        arrays.update({name: value for name, value in prediction["probabilities"].items()})
        np.savez_compressed(report.directory / "query_probabilities.npz", **arrays)
    return metrics


def checked_payload(trainer, args, verification_artifact=False):
    import goal_bc as bc

    payload = trainer.payload(verification_artifact=verification_artifact)
    payload["accepted_check_sha256"] = bc.file_hash(baseline.project_path(args.check_dir) / "report.json")
    return payload


def restore_input(trainer, args, path):
    import goal_bc as bc
    import goal_library as gl

    payload = bc.torch_load(path)
    gl.require(payload.get("accepted_check_sha256") == bc.file_hash(baseline.project_path(args.check_dir) / "report.json"),
               "探针checkpoint与独立check记录不同")
    trainer.restore(payload)
    return payload


def train(args, report, data, options):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl

    trainer = make_trainer(data, options, args.device)
    if args.resume:
        restore_input(trainer, args, baseline.project_path(args.resume))
        report.check("resume", "PASS", f"两头、优化器、采样器及RNG完整恢复，step={trainer.step}")
    gl.require(args.steps > trainer.step, "steps必须大于当前累计探针更新数")
    start_step = trainer.step
    mutable = bc.tensor_storage_bytes([head.state_dict() for head in trainer.heads.values()])
    estimated = 3 * mutable + 1024**2
    plan = bc.require_disk_space(report.directory, 3 * estimated)
    report.data.update(start_step=start_step, checkpoint_estimated_bytes=estimated, storage_preflight=plan)
    report.check("storage_preflight", "PASS", f"每份双头完整快照约{estimated / 2**20:.1f}MiB；仅存新头和优化器，不复制冻结bundle")
    counts = np.zeros(data.cache.bundle["horizon"], np.int64)
    best = {}

    def save_sampling():
        labels = (trainer.step - start_step) * options["batch_size"]
        gl.require(int(counts.sum()) == labels, "探针本轮采样预算不同")
        baseline.write_json(report.directory / "sampling.json", dict(trainer.sampling_plan(),
            start_step=start_step, end_step=trainer.step, observed_labels_per_head_this_invocation=labels,
            observed_counts=counts.tolist(), observed_fractions=(counts / labels).tolist() if labels else None,
            paired_batches_identical=True, observation_scope="this invocation only", candidate_model_trained=False))

    def save_best(metrics):
        full = metrics["validation_full"]["row_weighted"]
        value = (full["goal_nll"] + full["no_goal_nll"]) / 2
        if not best or value < best["value"]:
            best.update(step=trainer.step, value=value, metric="mean_full_validation_nll_of_both_heads",
                        selection_start_step=start_step, scope="secondary_diagnostic; fixed-budget latest is primary")
            payload = checked_payload(trainer, args)
            payload["selection"] = copy.deepcopy(best)
            bc.save_atomic(payload, report.directory / "best_pair.pt", overwrite=True)
            baseline.write_json(report.directory / "best_pair.json", best)

    save_sampling()
    report.check("paired_training", "PASS", "同一真实batch同时供两头训练；uniform_rows固定；只有两头优化器，无候选模型或原actor")
    report.require_writable()
    initial = evaluation_outputs(trainer, report)
    save_best(initial)
    while trainer.step < args.steps:
        values = trainer.update()
        counts += np.asarray(values["remaining_histogram"], np.int64)
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % args.log_every == 0 or trainer.step == args.steps:
            print(f"[TRAIN PROBE] step={trainer.step}/{args.steps} goal_nll={values['goal_nll']:.4f} no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
        if trainer.step % args.eval_every == 0 or trainer.step == args.steps:
            metrics = evaluation_outputs(trainer, report, final=trainer.step == args.steps)
            save_best(metrics)
        if trainer.step % args.save_every == 0 or trainer.step == args.steps:
            save_sampling()
            bc.save_atomic(checked_payload(trainer, args), report.directory / "latest.pt", overwrite=True)
    gl.require(bc.bundle_id(data.cache.bundle) == data.cache.metadata["bundle_id"], "探针修改冻结依赖")
    report.data.update(counters=trainer.counters(), initial_metrics=initial, final_metrics=metrics,
                       best_pair=best, controlled_reachability_verified=False)
    report.check("training", "PASS", f"两头各更新{trainer.step}次；原始真实动作CE，无目标替换训练或强制动作差异loss；new_env_steps=0")
    report.check("frozen_dependencies", "PASS", "冻结bundle内容未变；未实例化原encoder/RSSM/actor；仅两小头可更新")
    report.check("checkpoint", "PASS", "latest/best_pair保存同更新数的两个头及完整训练状态；verify检查下一次更新；禁止作为worker加载")
    report.check("scope", "WARN", "只检验动作预测信息；无环境控制、任务成功或信息不存在的结论；主比较用固定预算latest")


def verify(args, report, data, options):
    import torch
    import goal_bc as bc
    import goal_control as ctl
    import goal_library as gl

    trainer = make_trainer(data, options, args.device)
    path = baseline.project_path(args.checkpoint)
    payload = restore_input(trainer, args, path)
    report.check("strict_load", "PASS", f"双头/优化器/计数/采样器/RNG严格恢复，step={trainer.step}")
    for name, bad, message in (
        ("cache_guard", dict(payload, input_identity=dict(payload["input_identity"], cache_id="wrong")), "身份"),
        ("counter_guard", dict(payload, counters=dict(payload["counters"], new_env_steps=1)), "计数"),
        ("sampling_guard", dict(payload, sampling_protocol="wrong"), "采样"),
        ("artifact_guard", dict(payload, verification_artifact=True), "合成验收")):
        t02.rejection(report, name, lambda bad=bad: trainer.restore(bad), message)
    bad_heads = copy.deepcopy(payload["heads"])
    bad_heads["goal"]["state_projection.weight"] = bad_heads["goal"]["state_projection.weight"][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, heads=bad_heads)), "权重不兼容")
    t02.rejection(report, "worker_use_guard", lambda: ctl.load_model(payload, data.cache, args.device), "真实 T04")
    artifact = checked_payload(trainer, args, verification_artifact=True)
    bc.save_atomic(artifact, report.directory / "roundtrip_verification.pt")
    other = make_trainer(data, options, args.device)
    other.restore(bc.torch_load(report.directory / "roundtrip_verification.pt"), allow_verification=True)
    again = other.payload(verification_artifact=True)
    for key in ("heads", "optimizers", "counters", "sampler_state", "rng_state", "input_identity", "options"):
        gl.require(t01.same(artifact[key], again[key]), f"探针保存恢复{key}不同")
    features, goals, remaining, _ = data.cache.worker_batch(data.cache.worker_rows["validation"][:32], args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for name in ("goal", "no_goal"):
            gl.require(torch.equal(trainer.logits(name, features, goals, remaining), other.logits(name, features, goals, remaining)), "探针恢复前向不同")
    report.check("roundtrip", "PASS", "双头权重、完整优化器、计数、采样及RNG保存恢复一致；前向相同")
    before = {name: bc.cpu_tree(head.state_dict()) for name, head in trainer.heads.items()}
    metrics = evaluation_outputs(trainer, report, final=True)
    gl.require(all(t01.same(before[name], head.state_dict()) for name, head in trainer.heads.items()), "评价改变探针权重")
    rng = bc.capture_rng(args.device)
    left = trainer.update()
    bc.restore_rng(rng, args.device)
    right = other.update()
    gl.require(left["shared_batch_id"] == right["shared_batch_id"] and left["remaining_histogram"] == right["remaining_histogram"], "恢复后下一batch不同")
    for name in ("goal", "no_goal"):
        gl.require(t01.same(trainer.heads[name].state_dict(), other.heads[name].state_dict()) and
                   t01.same(trainer.optimizers[name].state_dict(), other.optimizers[name].state_dict()), "恢复后下一次探针更新不同")
    gl.require(t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and trainer.counters() == other.counters(), "恢复后采样器/计数不同")
    gl.require(bc.bundle_id(data.cache.bundle) == data.cache.metadata["bundle_id"], "验收修改冻结依赖")
    report.data.update(counters=payload["counters"], metrics=metrics, verification_probe_updates_per_copy=1,
                       controlled_reachability_verified=False)
    report.check("next_update_equivalence", "PASS", "两份内存模型下一次真实batch、梯度更新、优化器及计数一致")
    report.check("frozen_dependencies", "PASS", "冻结依赖内容不变，源checkpoint未写回；仅内存探针各更新一次")
    report.check("scope", "WARN", "roundtrip_verification.pt禁止真实resume或控制；工程PASS不要求额外目标收益达到阈值")


def source_paths(args):
    cache_dir = baseline.project_path(args.cache_dir)
    manifest = read_json(cache_dir / "cache_manifest.json")
    source, dataset = manifest["source_metadata"], Path(manifest["dataset_dir"])
    files = [cache_dir / name for name in ("cache_manifest.json", "tables.npz", "states.npy", "frozen_bundle.pt", "report.json")]
    files += [dataset / name for name in ("segments_manifest.json", "segments.npz", "report.json")]
    files += [Path(manifest["t03_verify_dir"]) / "report.json", Path(source["checkpoint"]["path"]),
              Path(source["library_path"]), Path(source["baseline_dir"]) / "report.json",
              Path(source["baseline_dir"]) / "resolved_config.json"]
    files += [Path(entry["path"]) for entry in source["episodes"]]
    if args.command == "check":
        verification = baseline.project_path(args.t04_verify_dir)
    else:
        checked = baseline.project_path(args.check_dir) / "report.json"
        files.append(checked)
        verification = baseline.project_path(read_json(checked)["arguments"]["t04_verify_dir"])
        checkpoint = args.resume if args.command == "train" else args.checkpoint
        if checkpoint:
            files.append(baseline.project_path(checkpoint))
    files.append(verification / "report.json")
    return list(dict.fromkeys(files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--t04-verify-dir", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--check-dir", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--steps", type=int, default=400, help="每个头的累计更新次数，不是环境步数")
    for name in ("eval-every", "log-every", "save-every"):
        train_parser.add_argument("--" + name, type=int, default=100)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--check-dir", required=True)
    verify_parser.add_argument("--checkpoint", required=True)
    for command in (check_parser, train_parser, verify_parser):
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
        for name in ("seed", "batch-size", "state-dim", "hidden", "train-per-episode", "validation-per-episode", "donor-candidates"):
            command.add_argument("--" + name, type=int)
        for name in ("learning-rate", "grad-clip"):
            command.add_argument("--" + name, type=float)
    args = parser.parse_args()
    if args.command == "train" and min(args.steps, args.eval_every, args.log_every, args.save_every) < 1:
        parser.error("更新数和输出周期必须为正")
    try:
        files, source = source_paths(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"无法读取输入清单：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / ("information_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    protected = {path.parent.resolve() for path in files}
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于输入缓存、源模型/回放及验收目录")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl

        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        rng = bc.capture_rng(args.device)
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化checkpoint改变")
        for entry in source["episodes"]:
            gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "原真实回放改变")
        data, options = load_data(args, report)
        {"check": check, "train": train, "verify": verify}[args.command](args, report, data, options)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "源输入大小/修改时间未变；只写独立探针目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
