"""T05 frozen-base residual worker: offline check, paired train, verify."""

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
import t05_goal_information_probe as information


class Report(baseline.Report):
    def finish(self):
        levels = {row["level"] for row in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        label = "T05_RESIDUAL_WORKER_" + self.data["command"].upper()
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_records(args):
    if args.command == "check":
        return None, baseline.project_path(args.base_checkpoint), baseline.project_path(args.base_verify_dir)
    checked = read_json(baseline.project_path(args.check_dir) / "report.json")
    return checked, baseline.project_path(checked["arguments"]["base_checkpoint"]), baseline.project_path(checked["arguments"]["base_verify_dir"])


def load_inputs(args, report):
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_worker as residual

    checked, base_path, verification = input_records(args)
    options = copy.deepcopy(checked["options"] if checked else residual.DEFAULTS)
    for key in options:
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if checked is not None:
        gl.require(checked.get("command") == "check" and checked.get("status") in ("passed", "passed_with_warnings") and
                   checked.get("architecture") == residual.ARCHITECTURE and checked["options"] == options,
                   "需要相同配置且通过的残差worker check；不能复用小探针check")
    # Share the accepted T03/T04 mapping and true action checks with the
    # information tool, without sharing its model or checkpoint protocol.
    request = copy.copy(args)
    request.command, request.t04_verify_dir = "check", str(verification)
    for key, value in options.items():
        setattr(request, key, value)
    source_data, _ = information.load_data(request, report)
    base, reference = residual.load_base(base_path, verification / "report.json", source_data.cache)
    data = residual.ResidualData(source_data.cache, options, residual.base_identity(base))
    # Reuse the cache's already validated immutable CPU object. No optimizer
    # or candidate from the T04 checkpoint is constructed/restored.
    base = {key: value for key, value in base.items() if key not in ("optimizers", "candidate")}
    base["frozen_bundle"] = data.cache.bundle
    identity = data.identity()
    dependencies = {name: dict(path=str(item["source_path"]), sha256=item["sha256"])
                    for name, item in reference.items()}
    if checked is not None:
        gl.require(checked["input_identity"] == identity and checked["dependencies"] == dependencies,
                   "残差check的缓存/底座/查询或依赖文件改变")
    report.data.update(architecture=residual.ARCHITECTURE, semantics=residual.SEMANTICS,
                       input_identity=identity, options=options, dependencies=dependencies,
                       base_counters=base["counters"], base_conditioning=base["options"]["conditioning"],
                       controlled_reachability_verified=False, task_success_evaluated=False)
    report.check("accepted_no_goal_base", "PASS", f"匹配独立T04 verify的no_goal底座，step={base['counters']['step']}；固定uniform_rows；不恢复旧优化器或候选模型")
    return data, options, base, reference


def make_trainer(data, options, base, reference, device):
    import numpy as np
    import torch
    import goal_residual_worker as residual

    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return residual.Trainer(data, base, reference, options, device)


def check(args, report, data, options, base, reference):
    import numpy as np
    import torch
    import goal_library as gl
    import tools

    trainer = make_trainer(data, options, base, reference, args.device)
    model = trainer.model
    gl.require(t01.same(model.heads["goal"].state_dict(), model.heads["no_goal"].state_dict()), "两分支初始权重不同")
    errors = []
    scales = None
    with torch.no_grad(), gl.encoder_precision(args.device):
        for split in ("train", "validation"):
            rows = data.cache.worker_rows[split][:32]
            features, goals, remaining, labels = data.cache.worker_batch(rows, args.device)
            expected = model.base(features, torch.zeros_like(goals), remaining)
            for mode in ("base", "goal", "no_goal", "zero_goal"):
                dist = model.action_dist(features, goals, remaining, mode)
                errors.append(float((expected.probs - dist.probs).abs().max()))
                gl.require(torch.equal(expected.probs, dist.probs) and torch.equal(expected.logits, dist.logits),
                           "零修正初始分布与已训练底座不同")
                target = torch.nn.functional.one_hot(labels, data.cache.metadata["action_dim"]).float()
                gl.require(torch.equal(expected.log_prob(target), dist.log_prob(target)), "初始动作NLL与底座不同")
                mode_action = dist.mode()
                gl.require(np.allclose(mode_action.cpu().numpy(), np.eye(data.cache.metadata["action_dim"], dtype=np.float32)[dist.probs.argmax(-1).cpu().numpy()], atol=1e-6, rtol=0),
                           "残差worker mode不是有效onehot动作")
            gl.require(torch.count_nonzero(model.correction("goal", features, goals, remaining)) == 0 and
                       torch.count_nonzero(model.correction("no_goal", features, goals, remaining)) == 0,
                       "修正最后一层未置零")
            state, goal, _ = model.heads["goal"].normalized_inputs(features, goals, remaining)
            scales = dict(state_rms=float(state.square().mean().sqrt()), goal_rms=float(goal.square().mean().sqrt()))
            gl.require(all(0.9 < value < 1.1 for value in scales.values()), "归一化输入尺度异常")
            zero = model.heads["goal"].normalized_inputs(features, torch.zeros_like(goals), remaining)[1]
            gl.require(torch.count_nonzero(zero) == 0, "无目标归一化产生非零输入")
            # Verify the correction precedes the SAME single mixture step,
            # including nonzero correction, without changing any parameter.
            raw = model.base_preferences(features, goals, remaining)
            offset = torch.linspace(-0.2, 0.2, raw.shape[-1], device=raw.device)
            corrected = raw + offset
            direct = tools.OneHotDist(corrected, unimix_ratio=model.base.actor._unimix_ratio)
            gl.require(torch.equal(model.distribution(corrected).probs, direct.probs), "残差动作混合协议不同")
    trainer.check_ownership()
    model.check_frozen(content=True)
    for invalid in (0, data.cache.bundle["horizon"] + 1):
        t02.rejection(report, "remaining_guard", lambda invalid=invalid: model.action_dist(features, goals, torch.full_like(remaining, invalid)), "remaining 必须")
    t02.rejection(report, "shape_guard", lambda: model.action_dist(features, goals[:, :-1], remaining), "形状")
    t02.rejection(report, "format_guard", lambda: trainer.restore(dict(trainer.payload(), checkpoint_format="ls_imagine_goal_information_probe_v1")), "格式")
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(trainer.payload(True)), "合成验收")
    count = sum(p.numel() for p in model.heads["goal"].parameters())
    report.data.update(initialization_max_error=max(errors), normalized_inputs=scales,
                       parameters_per_branch=count, sampling=trainer.sampling_plan(), counters=trainer.counters())
    report.check("zero_residual_initialization", "PASS", f"两套同权重修正分支；train/validation多条件的概率、logits和动作NLL均与底座逐值一致；最大误差{max(errors):.3g}")
    report.check("distribution_contract", "PASS", "原始动作偏好加修正后只执行一次原onehot unimix；mode为有效onehot；训练与部署共用分布")
    report.check("normalized_input_paths", "PASS", f"状态/目标独立无偏置归一化；每分支{count}参数；无目标输入恒零")
    report.check("frozen_ownership", "PASS", "优化器只含两套独立修正分支；完整状态底座冻结、无梯度、内容一致；无WM优化器")
    gl.require(trainer.step == 0 and all(not opt.state for opt in trainer.optimizers.values()), "check发生训练更新")
    report.check("no_training", "PASS", "optimizer_updates=0；new_env_steps=0；没有MineDojo/MineCLIP交互")
    report.check("scope", "WARN", "初始接口工程验收；尚未训练修正分支，也未证明目标预测收益或控制效果")


def evaluation_outputs(trainer, report, final=False):
    import numpy as np
    import goal_information_probe as probe
    import goal_residual_worker as residual

    metrics, queries, full = residual.evaluate(trainer)
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    values = metrics["validation_full"]["row_weighted"]
    print(f"[RESIDUAL] step={trainer.step} full_rows={values['rows']} base_nll={values['base_nll']:.6f} "
          f"goal_nll={values['goal_nll']:.6f} no_goal_nll={values['no_goal_nll']:.6f} "
          f"goal_gain={values['no_goal_minus_goal_nll']:.6f} base_gain={values['base_minus_goal_nll']:.6f}", flush=True)
    long = metrics["validation_full"].get("remaining_13_plus")
    if long:
        print(f"[LONG BUDGET FULL] rows={long['rows']} goal_gain={long['no_goal_minus_goal_nll']:.6f} "
              f"base_gain={long['base_minus_goal_nll']:.6f} goal_accuracy={long['goal_accuracy']:.4f} "
              f"no_goal_accuracy={long['no_goal_accuracy']:.4f}", flush=True)
    if final:
        baseline.write_json(report.directory / "diagnostics.json", dict(
            architecture=residual.ARCHITECTURE, semantics=residual.SEMANTICS, options=trainer.options,
            counters=trainer.counters(), metrics=metrics, controlled_reachability_verified=False,
            task_success_evaluated=False, interpretation="goal contribution beyond frozen base and equally trained no-goal residual; offline only"))
        baseline.write_csv(report.directory / "query_metrics.csv", probe.query_records(trainer.data, queries))
        for label, prediction in (("queries", queries), ("validation_full", full)):
            arrays = dict(rows=prediction["rows"])
            if prediction["donors"] is not None:
                arrays["donors"] = prediction["donors"]
            for key in ("nll", "probabilities", "correction_rms"):
                arrays.update({key + "_" + name: value for name, value in prediction[key].items()})
            np.savez_compressed(report.directory / (label + "_predictions.npz"), **arrays)
    return metrics


def check_hash(args):
    import goal_bc as bc
    return bc.file_hash(baseline.project_path(args.check_dir) / "report.json")


def restore_input(trainer, args, path):
    import goal_residual_worker as residual
    payload, _ = residual.read_checkpoint(path, trainer, check_hash(args))
    trainer.restore(payload)
    return payload


def train(args, report, data, options, base, reference):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_worker as residual

    trainer = make_trainer(data, options, base, reference, args.device)
    if args.resume:
        restore_input(trainer, args, baseline.project_path(args.resume))
        report.check("resume", "PASS", f"仅恢复两修正分支/优化器、计数、抽样和RNG，step={trainer.step}；底座保持冻结")
    gl.require(args.steps > trainer.step, "steps必须大于当前累计残差更新数")
    start_step = trainer.step
    mutable = bc.tensor_storage_bytes(trainer.model.heads.state_dict())
    estimate = 3 * mutable + 1024**2
    storage = bc.require_disk_space(report.directory, 3 * estimate)
    report.data.update(start_step=start_step, checkpoint_estimated_bytes=estimate, storage_preflight=storage)
    report.check("storage_preflight", "PASS", f"每份完整双分支快照约{estimate / 2**20:.1f}MiB；检查latest/best及临时写入；底座和WM使用SHA256共享引用")
    counts = np.zeros(data.cache.bundle["horizon"], np.int64)
    best, digest = {}, check_hash(args)

    def save_sampling():
        labels = (trainer.step - start_step) * options["batch_size"]
        gl.require(int(counts.sum()) == labels, "残差采样预算不同")
        baseline.write_json(report.directory / "sampling.json", dict(trainer.sampling_plan(),
            start_step=start_step, end_step=trainer.step, observed_labels_per_branch_this_invocation=labels,
            observed_counts=counts.tolist(), observed_fractions=(counts / labels).tolist() if labels else None,
            observation_scope="this invocation only", candidate_model_trained=False, base_updates=0))

    def save_best(metrics):
        full = metrics["validation_full"]["row_weighted"]
        value = (full["goal_nll"] + full["no_goal_nll"]) / 2
        if not best or value < best["value"]:
            best.update(step=trainer.step, value=value, metric="mean_full_validation_nll_of_both_residual_branches",
                        selection_start_step=start_step, scope="secondary diagnostic; fixed-budget latest is primary")
            residual.save_checkpoint(trainer, report.directory / "best_pair.pt", digest, selection=best, overwrite=True)
            baseline.write_json(report.directory / "best_pair.json", best)

    save_sampling()
    report.check("paired_training", "PASS", "同一真实batch训练有目标/无目标两修正分支；uniform_rows；真实动作混合分布NLL；底座、WM、目标编码器均不训练")
    report.require_writable()
    initial = evaluation_outputs(trainer, report)
    save_best(initial)
    while trainer.step < args.steps:
        values = trainer.update()
        counts += np.asarray(values["remaining_histogram"], np.int64)
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % args.log_every == 0 or trainer.step == args.steps:
            print(f"[TRAIN RESIDUAL] step={trainer.step}/{args.steps} goal_nll={values['goal_nll']:.4f} "
                  f"no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
        if trainer.step % args.eval_every == 0 or trainer.step == args.steps:
            metrics = evaluation_outputs(trainer, report, final=trainer.step == args.steps)
            save_best(metrics)
        if trainer.step % args.save_every == 0 or trainer.step == args.steps:
            save_sampling()
            residual.save_checkpoint(trainer, report.directory / "latest.pt", digest, overwrite=True)
    trainer.model.check_frozen(content=True)
    gl.require(bc.bundle_id(data.cache.bundle) == data.cache.metadata["bundle_id"], "冻结WM/目标依赖改变")
    report.data.update(counters=trainer.counters(), initial_metrics=initial, final_metrics=metrics, best_pair=best)
    report.check("training", "PASS", f"两修正分支各更新{trainer.step}次；有限损失/梯度；new_env_steps=0；底座未更新")
    report.check("frozen_dependencies", "PASS", "底座参数内容/冻结标志/梯度及WM/目标库内容均未改变；仅两小分支优化器")
    report.check("checkpoint", "PASS", "latest/best_pair保存双分支完整训练状态；底座/验收报告/冻结bundle外部共享并校验；verify检查恢复和推理")
    report.check("scope", "WARN", "这是可执行策略的离线BC对照；固定预算latest为主；收益、真实目标控制和任务成功仍需分析/验证")


def verify(args, report, data, options, base, reference):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_residual_worker as residual

    trainer = make_trainer(data, options, base, reference, args.device)
    path = baseline.project_path(args.checkpoint)
    payload = restore_input(trainer, args, path)
    report.check("strict_load", "PASS", f"两修正分支、独立优化器、计数/抽样/RNG恢复，step={trainer.step}；底座/缓存身份一致")
    for name, bad, message in (
        ("base_identity_guard", dict(payload, input_identity=dict(payload["input_identity"], base_id="wrong")), "身份"),
        ("counter_guard", dict(payload, counters=dict(payload["counters"], base_updates=1)), "计数"),
        ("sampling_guard", dict(payload, sampling_protocol="wrong"), "采样"),
        ("artifact_guard", dict(payload, verification_artifact=True), "合成验收")):
        t02.rejection(report, name, lambda bad=bad: trainer.restore(bad), message)
    bad = copy.deepcopy(payload["heads"])
    bad["goal"]["state_projection.weight"] = bad["goal"]["state_projection.weight"][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, heads=bad)), "权重不兼容")
    missing = dict(payload, dependencies={})
    t02.rejection(report, "dependency_guard", lambda: residual.checkpoint_dependencies(missing, path), "依赖")
    changed = copy.deepcopy(payload["dependencies"])
    changed["base_checkpoint"]["sha256"] = "wrong"
    t02.rejection(report, "dependency_hash_guard", lambda: residual.checkpoint_dependencies(dict(payload, dependencies=changed), path), "SHA256")

    digest = check_hash(args)
    artifact = residual.save_checkpoint(trainer, report.directory / "roundtrip_verification.pt", digest, verification_artifact=True)
    other = make_trainer(data, options, base, reference, args.device)
    saved, _ = residual.read_checkpoint(report.directory / "roundtrip_verification.pt", other, digest)
    other.restore(saved, allow_verification=True)
    again = other.payload(True)
    for key in ("heads", "optimizers", "counters", "sampler_state", "rng_state", "input_identity", "options"):
        gl.require(t01.same(artifact[key], again[key]), f"残差保存恢复{key}不同")
    rows = data.cache.worker_rows["validation"][:32]
    features, goals, remaining, _ = data.cache.worker_batch(rows, args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            gl.require(torch.equal(trainer.model.action_dist(features, goals, remaining, mode).probs,
                                   other.model.action_dist(features, goals, remaining, mode).probs), "残差恢复动作分布不同")
    report.check("roundtrip", "PASS", "双分支权重、完整优化器、计数、抽样、RNG及四种行为分布保存恢复一致")

    # The dedicated inference loader constructs neither optimizer nor
    # candidate; its distribution is the same one optimized by BC.
    inference = residual.load_policy(path, data.cache, args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            dist = inference.action_dist(features, goals, remaining, mode)
            gl.require(torch.equal(dist.probs, trainer.model.action_dist(features, goals, remaining, mode).probs),
                       "残差训练/推理动作分布不同")
        gl.require(torch.equal(inference.action_dist(features, goals, remaining, "no_goal").probs,
                               inference.action_dist(features, torch.flip(goals, [0]), remaining, "no_goal").probs),
                   "无目标修正策略依赖目标")
    t02.rejection(report, "inference_artifact_guard", lambda: residual.load_policy(report.directory / "roundtrip_verification.pt", data.cache, args.device), "合成验收")
    report.data.update(model_id=inference.model_id, worker_version=inference.worker_version)
    report.check("strict_inference_load", "PASS", "独立推理加载只构造冻结底座和两修正分支；与训练分布逐值一致；无目标分支不依赖目标")

    runtime = residual.OnlineRuntime(inference, args.device)
    dataset, _ = t04.accepted_dataset(Path(data.cache.metadata["dataset_dir"]), Path(data.cache.metadata["t03_verify_dir"]))
    rng = bc.capture_rng(args.device)
    for index in sorted({0, len(dataset.metadata["episodes"]) - 1}):
        episode = dataset.episode(index)
        stop = min(args.prefix_steps, len(episode["image"]) - 1)
        expected = runtime.state_encoder.rollout(episode, stop)
        state, actual = None, []
        keys = ("image", "heatmap", "obs_reward", "is_first", "is_last", "is_terminal")
        for frame in range(stop + 1):
            obs = {key: episode[key][frame] for key in keys}
            state, feature = runtime.state_encoder.step(obs, episode["action"][frame], state)
            actual.append(feature[0])
        t02.vector_difference(report, f"incremental_episode_{index}", torch.stack(actual), expected, check=True)
        reset = {key: episode[key][0] for key in keys}
        _, reset_dist = runtime.step(reset, episode["action"][0], goals[0].cpu().numpy(), data.cache.bundle["horizon"], state)
        with torch.no_grad():
            direct = inference.action_dist(expected[:1], goals[:1], torch.full_like(remaining[:1], data.cache.bundle["horizon"]))
            gl.require(torch.equal(reset_dist.probs, direct.probs), "reset在线动作与因果前向不同")
        t02.rejection(report, "missing_history_guard", lambda: runtime.state_encoder.step(dict(reset, is_first=False), episode["action"][1]), "本局历史")
        t02.rejection(report, "reset_action_guard", lambda: runtime.state_encoder.step(reset, episode["action"][1]), "必须为零")
    for invalid in (0, data.cache.bundle["horizon"] + 1):
        t02.rejection(report, "remaining_guard", lambda invalid=invalid: inference.action_dist(features, goals, torch.full_like(remaining, invalid)), "remaining 必须")
    t02.rejection(report, "after_terminal_guard", lambda: runtime.step(dict(is_last=True, is_terminal=True), np.zeros(data.cache.metadata["action_dim"]), goals[0].cpu().numpy(), 1), "真实结束")
    gl.require(t01.same(rng, bc.capture_rng(args.device)), "残差在线状态检查推进调用方RNG")
    report.check("incremental_causal_state", "PASS", "真实回放逐帧状态与完整reset前缀一致；reset清空旧历史；动作/remaining/结束边界保持原协议；无环境交互")
    del runtime, inference
    gc.collect()

    metrics = evaluation_outputs(trainer, report, final=True)
    rng = bc.capture_rng(args.device)
    left = trainer.update()
    bc.restore_rng(rng, args.device)
    right = other.update()
    gl.require(left["shared_batch_id"] == right["shared_batch_id"] and
               left["remaining_histogram"] == right["remaining_histogram"], "恢复后下一真实batch不同")
    for name in ("goal", "no_goal"):
        gl.require(t01.same(trainer.model.heads[name].state_dict(), other.model.heads[name].state_dict()) and
                   t01.same(trainer.optimizers[name].state_dict(), other.optimizers[name].state_dict()), "残差恢复后下一更新不同")
    gl.require(t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and trainer.counters() == other.counters(),
               "残差恢复后抽样/计数不同")
    trainer.model.check_frozen(content=True)
    other.model.check_frozen(content=True)
    gl.require(bc.bundle_id(data.cache.bundle) == data.cache.metadata["bundle_id"], "冻结WM/目标依赖改变")
    report.data.update(counters=payload["counters"], metrics=metrics, verification_updates_per_copy=1)
    report.check("next_update_equivalence", "PASS", "两份内存模型下一真实batch更新后分支参数、优化器、抽样和计数一致")
    report.check("frozen_dependencies", "PASS", "冻结底座、WM/目标依赖内容未改；源checkpoint不写回；仅验收内存分支各更新一次")
    report.check("scope", "WARN", "roundtrip_verification.pt禁止resume/控制；工程PASS不要求预测收益为正；尚未启动真实环境或推进T06")


def source_paths(args):
    cache_dir = baseline.project_path(args.cache_dir)
    manifest = read_json(cache_dir / "cache_manifest.json")
    source, dataset = manifest["source_metadata"], Path(manifest["dataset_dir"])
    checked, base_path, verification = input_records(args)
    files = [cache_dir / name for name in ("cache_manifest.json", "tables.npz", "states.npy", "frozen_bundle.pt", "report.json")]
    files += [dataset / name for name in ("segments_manifest.json", "segments.npz", "report.json")]
    files += [Path(manifest["t03_verify_dir"]) / "report.json", Path(source["checkpoint"]["path"]),
              Path(source["library_path"]), Path(source["baseline_dir"]) / "report.json",
              Path(source["baseline_dir"]) / "resolved_config.json", base_path, verification / "report.json"]
    files += [Path(entry["path"]) for entry in source["episodes"]]
    if checked is not None:
        files.append(baseline.project_path(args.check_dir) / "report.json")
        checkpoint = args.resume if args.command == "train" else args.checkpoint
        if checkpoint:
            files.append(baseline.project_path(checkpoint))
    return list(dict.fromkeys(files)), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--base-checkpoint", required=True)
    checking.add_argument("--base-verify-dir", required=True)
    training = commands.add_parser("train")
    training.add_argument("--check-dir", required=True)
    training.add_argument("--resume")
    training.add_argument("--steps", type=int, default=200, help="每分支累计更新数，不是新环境步")
    for name in ("eval-every", "log-every", "save-every"):
        training.add_argument("--" + name, type=int, default=50)
    verification = commands.add_parser("verify")
    verification.add_argument("--check-dir", required=True)
    verification.add_argument("--checkpoint", required=True)
    verification.add_argument("--prefix-steps", type=int, default=32)
    for command in (checking, training, verification):
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
    if args.command == "verify" and args.prefix_steps < 1:
        parser.error("prefix-steps必须为正")
    try:
        files, source = source_paths(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"无法读取输入清单：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
        "residual_" + args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    protected = {path.parent.resolve() for path in files}
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于底座、缓存、源模型/回放及验收目录")
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
        data, options, base, reference = load_inputs(args, report)
        {"check": check, "train": train, "verify": verify}[args.command](args, report, data, options, base, reference)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "底座、源模型/回放及输入验收产物大小/修改时间未变；只写独立残差目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
