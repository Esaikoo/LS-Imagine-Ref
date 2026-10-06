"""T04 offline cache, goal-conditioned BC and checkpoint acceptance.

No MineDojo, MineCLIP, old optimizer or flat agent training is started here.
"""

import argparse
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        labels = {"prepare": "T04_CACHE_PREPARE", "train": "T04_BC_TRAIN", "verify": "T04_BC_VERIFY"}
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] {labels[self.data['command']]}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def accepted_dataset(directory, verification):
    import goal_library as gl
    import goal_segments as gs

    dataset = gs.SegmentDataset(directory)
    build = read_json(dataset.directory / "report.json")
    checked = read_json(Path(verification) / "report.json")
    for record, command in ((build, "build"), (checked, "verify")):
        gl.require(record.get("command") == command and record.get("status") in ("passed", "passed_with_warnings") and
                   record.get("content_id") == dataset.content_id and record.get("library_id") == dataset.metadata["library_id"],
                   "需要对应同一数据集的已通过 T03 build/verify 记录")
    gl.require(checked.get("state_encoder", {}).get("state_policy") == gs.STATE_POLICY, "T03 因果状态约定不匹配")
    return dataset, checked


def prepare(args, report):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_segments as gs
    import long_horizon as lh

    dataset, checked = accepted_dataset(baseline.project_path(args.dataset_dir), baseline.project_path(args.t03_verify_dir))
    meta, a = dataset.metadata, dataset.arrays
    source = copy.copy(args)
    source.checkpoint, source.baseline_dir = meta["checkpoint"]["path"], meta["baseline_dir"]
    config, shapes = t02.resolve_baseline(source, report)
    gl.require(baseline.file_signature(Path(source.checkpoint)) == meta["checkpoint"], "T03 原初始化 checkpoint 已变更")
    library = gl.load_library(meta["library_path"], "cpu").payload()
    gl.require(library["library_id"] == meta["library_id"] and library["encoder_id"] == meta["encoder_id"], "T03 目标库身份改变")
    checkpoint = lh.read_checkpoint(Path(source.checkpoint))
    gl.require(gl.tensor_digest(gl.cnn_weights(checkpoint["agent_state_dict"])) == library["source_encoder_hash"], "WM 与目标编码器视觉权重不同")
    frozen = gs.state_encoder_from_checkpoint(checkpoint, config, shapes, args.device)
    frozen_id = gl.tensor_digest(frozen.state_dict())
    gl.require(frozen_id == checked["state_encoder"]["state_encoder_id"], "当前 encoder/RSSM 与 T03 因果验收不同")
    original = {key[len("_task_behavior.actor."):]: value for key, value in lh.normalize(checkpoint["agent_state_dict"]).items()
                if key.startswith("_task_behavior.actor.")}
    feature_dim = int(original["layers.Actor_linear0.weight"].shape[1])
    bundle = {"format": bc.BUNDLE_FORMAT, "state_policy": gs.STATE_POLICY, "config": config, "shapes": shapes,
              "feature_dim": feature_dim, "horizon": meta["horizon"], "state_encoder": bc.cpu_tree(frozen.state_dict()),
              "original_actor": bc.cpu_tree(original), "library": library}
    del original, checkpoint
    gc.collect()
    bundle["bundle_id"] = bc.bundle_id(bundle)
    bc.validate_bundle(bundle)
    versions = {key: p._version for key, p in frozen.named_parameters()}
    report.check("t03_input", "PASS", "真实片段、整局划分及已验收的 encoder/RSSM 一致；无环境或 MineCLIP 交互")

    blocks, offset = [], 0
    for index, entry in enumerate(meta["episodes"]):
        segments = np.flatnonzero(a["episode_index"] == index)
        frames = np.unique(np.concatenate([np.arange(a["start"][row], a["end"][row], dtype=np.int64) for row in segments]))
        blocks.append({"episode_index": index, "offset": offset, "frames": frames.tolist(), "split": entry["split"]})
        offset += len(frames)
    state_shape = (offset, feature_dim)
    states = np.lib.format.open_memmap(report.directory / "states.npy", mode="w+", dtype=np.float32, shape=state_shape)
    print(f"[CACHE] unique_states={offset} feature_dim={feature_dim} size_mib={offset * feature_dim * 4 / 2**20:.1f}", flush=True)
    m, n = len(dataset), int((a["end"] - a["start"]).sum())
    tables = {key: np.empty(n, dtype=np.int64) for key in ("worker_state", "worker_segment", "remaining", "action_id")}
    tables.update(worker_train=np.empty(n, dtype=bool), candidate_state=np.empty(m, dtype=np.int64),
                  candidate_goal=a["goal_id"].copy(), candidate_train=np.empty(m, dtype=bool), goals=a["goals"].copy())
    # Stable row ordering: one contiguous worker block per original segment.
    worker_offsets = np.concatenate(([0], np.cumsum(a["end"] - a["start"])))
    for block in blocks:
        index, offset = block["episode_index"], block["offset"]
        episode = dataset.episode(index)
        frames = np.asarray(block["frames"], dtype=np.int64)
        # A single sequential prefix per episode, preserving T03's one-frame
        # numeric convention; unused prefix states receive no loss.
        features = frozen.rollout(episode, int(frames[-1]))
        states[offset:offset + len(frames)] = features.cpu().numpy()[frames]
        del features
        lookup = {int(frame): offset + row for row, frame in enumerate(frames)}
        is_train = block["split"] == "train"
        for row in np.flatnonzero(a["episode_index"] == index):
            start, end = int(a["start"][row]), int(a["end"][row])
            selection = slice(int(worker_offsets[row]), int(worker_offsets[row + 1]))
            tables["worker_state"][selection] = [lookup[t] for t in range(start, end)]
            tables["worker_segment"][selection] = row
            tables["remaining"][selection] = np.arange(end - start, 0, -1)
            tables["action_id"][selection] = episode["action"][start + 1:end + 1].argmax(-1)
            tables["worker_train"][selection] = is_train
            tables["candidate_state"][row] = lookup[start]
            tables["candidate_train"][row] = is_train
        print(f"[STATES] episodes={index + 1}/{len(blocks)} prefix_frames={frames[-1] + 1} retained={len(frames)}", flush=True)
    states.flush()
    del states
    gl.require(versions == {key: p._version for key, p in frozen.named_parameters()} and
               gl.tensor_digest(frozen.state_dict()) == frozen_id, "缓存构建更新了冻结模型")
    np.savez_compressed(report.directory / "tables.npz", **tables)
    bc.save_atomic(bundle, report.directory / "frozen_bundle.pt")
    cache_meta = {"format": bc.CACHE_FORMAT, "state_policy": gs.STATE_POLICY, "bundle_id": bundle["bundle_id"],
                  "state_encoder_id": frozen_id, "state_shape": list(state_shape), "states_sha256": bc.file_hash(report.directory / "states.npy"),
                  "goal_dim": meta["goal_dim"], "goal_count": meta["goal_count"], "action_dim": meta["action_dim"],
                  "dataset_dir": str(dataset.directory), "dataset_id": dataset.content_id,
                  "t03_verify_dir": str(baseline.project_path(args.t03_verify_dir)), "source_metadata": meta,
                  "episode_states": blocks, "worker_sampling": "uniform expanded action-label rows with replacement",
                  "candidate_sampling": "uniform segment-start rows with replacement", "new_env_steps": 0}
    identifier = gs.content_id(tables, cache_meta)
    baseline.write_json(report.directory / "cache_manifest.json", dict(cache_meta, cache_id=identifier))
    with np.load(report.directory / "tables.npz", allow_pickle=False) as reloaded:
        gl.require(all(np.array_equal(tables[key], reloaded[key]) for key in tables), "监督表保存恢复不一致")
    restored = bc.torch_load(report.directory / "frozen_bundle.pt")
    bc.validate_bundle(restored)
    gl.require(restored["bundle_id"] == bundle["bundle_id"], "冻结 bundle 保存恢复不一致")
    stats = {"episodes": len(blocks), "unique_states": state_shape[0], "feature_dim": feature_dim, "segments": m,
             "worker_labels": n, "train_worker_labels": int(tables["worker_train"].sum()),
             "validation_worker_labels": int((~tables["worker_train"]).sum()), "train_segments": int(tables["candidate_train"].sum()),
             "validation_segments": int((~tables["candidate_train"]).sum()), "state_size_mib": state_shape[0] * feature_dim * 4 / 2**20,
             "new_env_steps": 0, "optimizer_updates": 0, "controlled_reachability_verified": False}
    baseline.write_json(report.directory / "diagnostics.json", stats)
    report.data.update(cache_id=identifier, bundle_id=bundle["bundle_id"], diagnostics=stats)
    report.check("causal_state_cache", "PASS", f"{state_shape[0]} 个真实当前状态；每局从 reset 恢复一次；维度 {feature_dim}；无未来目标输入")
    report.check("action_supervision", "PASS", f"{n} 个 obs[t]→action[t+1] 标签；目标固定为真实终点，剩余步数递减")
    report.check("sampling", "PASS", "低层均匀抽取展开的动作条目；候选模型均匀抽取片段起点；重叠条目不计作新环境步")
    report.check("serialization", "PASS", "状态文件 SHA256、监督表/元数据 ID 与冻结 bundle ID 已保存；加载后逐项一致")
    report.check("frozen_no_training", "PASS", "仅计算冻结 encoder/RSSM 状态；无梯度、优化器或参数更新")
    report.check("scope", "WARN", "沿用 episode 留出；缓存和 BC 误差不能证明目标控制有效，真实行为需 T05 验收")


def training_options(args, resume):
    defaults = dict(seed=0, batch_size=256, candidate_hidden=512, learning_rate=3e-5,
                    candidate_learning_rate=3e-4, grad_clip=100.0, conditioning="goal")
    result = dict(resume["options"] if resume is not None else defaults)
    for key in defaults:
        value = getattr(args, key)
        if value is not None:
            result[key] = value
    return result


def training_checkpoint(path):
    import goal_bc as bc
    import goal_library as gl

    payload = bc.torch_load(path)
    gl.require(isinstance(payload, dict) and payload.get("checkpoint_format") == bc.CHECKPOINT_FORMAT,
               "需要 T04 训练输出的 checkpoint，不能传入原完整 agent latest.pt")
    return payload


def evaluate_and_save(trainer, report):
    import goal_bc as bc

    metrics = {"step": trainer.step, "train": bc.evaluate(trainer.model, trainer.cache, "train", max_worker_rows=2048),
               "validation": bc.evaluate(trainer.model, trainer.cache)}
    append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    v = metrics["validation"]
    print(f"[VALIDATION] step={trainer.step} worker_nll={v['worker_nll']:.4f} original_nll={v['original_nll']:.4f} "
          f"zero_goal_nll={v['zero_goal_nll']:.4f} shuffled_goal_nll={v['shuffled_goal_nll']:.4f} "
          f"candidate_top1={v['candidate_accuracy']:.4f} top4={v['candidate_top4_recall']:.4f}", flush=True)
    return metrics


def train(args, report):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl

    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    resume = training_checkpoint(baseline.project_path(args.resume)) if args.resume else None
    options = training_options(args, resume)
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    trainer = bc.Trainer(cache, options, args.device)
    if resume is not None:
        trainer.restore(resume)
        del resume
        gc.collect()
        report.check("resume", "PASS", f"权重、新模块优化器、计数及 RNG 完整恢复；step={trainer.step}")
    else:
        rows = cache.worker_rows["train"][:32]
        features, goals, remaining, _ = cache.worker_batch(rows, args.device)
        with torch.no_grad():
            reference = trainer.model.action_dist(features, goals, remaining, "original").probs
            for goal in (goals, torch.zeros_like(goals), goals.flip(0)):
                t02.vector_difference(report, "worker_initialization", trainer.model.action_dist(features, goal, remaining).probs, reference, check=True)
        report.check("original_initialization", "PASS", "目标/剩余步数输入权重为零；3 组目标条件初始动作分布与原 actor 一致")
    gl.require(args.steps > trainer.step, "--steps 是累计更新目标，必须大于已保存 step")
    versions = {id(p): p._version for module in (trainer.model.state_encoder, trainer.model.library, trainer.model.original_actor)
                for p in module.parameters()}
    frozen_id = trainer.model.frozen_identity()
    gl.require(frozen_id == cache.bundle["bundle_id"], "初始化冻结模块与源 bundle 不一致")
    report.data.update(cache_id=cache.cache_id, bundle_id=frozen_id, options=options, start_step=trainer.step)
    report.check("optimizers", "PASS", "只有独立 worker/candidate 优化器；原 WM、目标编码器及原 actor 冻结")
    report.check("scope", "WARN", "离线真实回放训练；new_env_steps=0；候选输出是未来类别概率，真实目标控制待 T05")
    reference = bc.frozen_bundle_reference(cache)
    # Budget three complete mutable snapshots plus one atomic replacement.
    # Adam will allocate two moments per parameter after the first update.
    mutable_bytes = bc.tensor_storage_bytes([trainer.model.worker.state_dict(), trainer.model.candidate.state_dict()])
    snapshot_bytes = 3 * mutable_bytes + bc.tensor_storage_bytes([bc.capture_rng(args.device), trainer.sampler.get_state()]) + 1024**2
    storage_plan = bc.require_disk_space(report.directory, 4 * snapshot_bytes)
    report.data["checkpoint_storage"] = dict(storage_plan, format=bc.CHECKPOINT_STORAGE,
        snapshot_estimated_bytes=snapshot_bytes, shared_bundle_path=str(reference["source_path"]),
        shared_bundle_sha256=reference["sha256"])
    report.check("storage_preflight", "PASS", f"预计单份快照 {snapshot_bytes / 2**20:.1f} MiB；已检查三份快照及一次临时写入空间；共享缓存冻结模型")
    report.require_writable()
    initial_metrics = evaluate_and_save(trainer, report)
    # Select the two modules independently. These are complete, resumable T04
    # snapshots, not mixtures of weights/optimizers from different updates.
    # A resumed invocation establishes its own selection window, including
    # the starting checkpoint; older runs did not preserve intermediate files.
    best = {}

    def save_best(metrics):
        for name, metric in (("worker", "worker_nll"), ("candidate", "candidate_ce")):
            value = float(metrics["validation"][metric])
            if name not in best or value < best[name]["value"]:
                selection = {"module": name, "metric": "validation." + metric,
                             "value": value, "step": trainer.step,
                             "selection_start_step": report.data["start_step"],
                             "scope": "evaluations_in_this_invocation"}
                payload = trainer.payload()
                payload["selection"] = selection
                path = report.directory / ("best_" + name + ".pt")
                bc.save_checkpoint(payload, path, reference, overwrite=True)
                best[name] = dict(selection, path=str(path))
                baseline.write_json(report.directory / "best_checkpoints.json", best)

    save_best(initial_metrics)
    while trainer.step < args.steps:
        values = trainer.update()
        append_json(report.directory / "training.jsonl", values)
        if trainer.step % args.log_every == 0 or trainer.step == args.steps:
            print(f"[TRAIN] step={trainer.step}/{args.steps} worker_nll={values['worker_nll']:.4f} candidate_ce={values['candidate_ce']:.4f}", flush=True)
        if trainer.step % args.eval_every == 0 or trainer.step == args.steps:
            final_metrics = evaluate_and_save(trainer, report)
            save_best(final_metrics)
        if trainer.step % args.save_every == 0 or trainer.step == args.steps:
            bc.save_checkpoint(trainer.payload(), report.directory / "latest.pt", reference, overwrite=True)
    gl.require(versions == {id(p): p._version for module in (trainer.model.state_encoder, trainer.model.library, trainer.model.original_actor)
                           for p in module.parameters()} and trainer.model.frozen_identity() == frozen_id, "训练更新了冻结模块")
    report.data.update(counters=trainer.counters(), checkpoint=str(report.directory / "latest.pt"),
                       initial_metrics=initial_metrics, final_metrics=final_metrics, best_checkpoints=best)
    report.check("training", "PASS", f"worker/candidate 各更新 {trainer.step} 次；损失与梯度有限；新环境步数 0")
    report.check("frozen_modules", "PASS", "原 encoder/RSSM、目标库和原 actor 参数版本及内容哈希未变")
    report.check("checkpoint", "PASS", "独立 T04 latest.pt 保存新优化器、计数、采样器、RNG；冻结依赖共享缓存文件并校验 SHA256/内容 ID；完整恢复请运行 verify")
    report.check("best_checkpoints", "PASS", "分别按留出 worker NLL/candidate CE 保存完整最佳快照；选择窗口及更新步数见 best_checkpoints.json")


def verify(args, report):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl

    cache = bc.TrainingCache(baseline.project_path(args.cache_dir))
    dataset, _ = accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"] and dataset.metadata == cache.metadata["source_metadata"], "缓存与 T03 源片段不同")
    report.data.update(cache_id=cache.cache_id, bundle_id=cache.bundle["bundle_id"])
    report.check("content_identity", "PASS", "状态文件、监督表、冻结模型/目标库及 T03 数据集内容 ID 匹配")

    # Check all frame/segment/split mappings independently of batch sampling.
    a, t = dataset.arrays, cache.tables
    state_episode = np.empty(len(cache.states), dtype=np.int64)
    state_frame = np.empty(len(cache.states), dtype=np.int64)
    cursor = 0
    gl.require(len(cache.metadata["episode_states"]) == len(dataset.metadata["episodes"]), "状态块 episode 数量错误")
    for index, block in enumerate(cache.metadata["episode_states"]):
        frames = np.asarray(block["frames"], dtype=np.int64)
        gl.require(block["episode_index"] == index and block["offset"] == cursor and len(frames) > 0 and
                   cursor + len(frames) <= len(cache.states) and np.all(np.diff(frames) > 0) and
                   frames[0] >= 0 and frames[-1] < dataset.metadata["episodes"][index]["usable_length"] - 1 and
                   block["split"] == dataset.metadata["episodes"][index]["split"], "状态块边界/episode/划分错误")
        state_episode[cursor:cursor + len(frames)] = index
        state_frame[cursor:cursor + len(frames)] = frames
        cursor += len(frames)
    gl.require(cursor == len(cache.states), "状态块未覆盖整个缓存")
    gl.require(np.array_equal(state_episode[t["candidate_state"]], a["episode_index"]) and
               np.array_equal(state_frame[t["candidate_state"]], a["start"]) and
               np.array_equal(t["candidate_goal"], a["goal_id"]) and np.array_equal(t["goals"], a["goals"]), "候选监督含终点状态或目标不一致")
    segments = t["worker_segment"]
    ep = a["episode_index"][segments]
    current = state_frame[t["worker_state"]]
    gl.require(np.array_equal(state_episode[t["worker_state"]], ep) and
               np.array_equal(t["remaining"], a["end"][segments] - current) and
               np.all((current >= a["start"][segments]) & (current < a["end"][segments])), "动作状态/目标/剩余步数对齐错误")
    expected_segments = np.repeat(np.arange(len(dataset)), a["end"] - a["start"])
    expected_current = np.concatenate([np.arange(start, end) for start, end in zip(a["start"], a["end"])])
    gl.require(np.array_equal(segments, expected_segments) and np.array_equal(current, expected_current), "动作监督有重复、遗漏或乱序")
    train = np.array([entry["split"] == "train" for entry in dataset.metadata["episodes"]])
    gl.require(np.array_equal(t["candidate_train"], train[a["episode_index"]]) and np.array_equal(t["worker_train"], train[ep]), "监督表发生 episode 分割泄漏")
    selected = np.sort(np.random.RandomState(0).choice(len(dataset), min(args.samples, len(dataset)), replace=False))
    for row in selected:
        sample = dataset[int(row)]
        actual_rows = np.flatnonzero(segments == row)
        gl.require(np.array_equal(t["action_id"][actual_rows], sample["next_actions"].argmax(-1)), "缓存动作未取 action[t+1]")
    report.check("real_supervision", "PASS", f"全量当前帧/片段/目标/递减步数/整局划分一致；抽查 {len(selected)} 段真实下一动作标签")

    payload = training_checkpoint(baseline.project_path(args.checkpoint))
    trainer = bc.Trainer(cache, payload["options"], args.device)
    trainer.restore(payload)
    gl.require(trainer.step > 0, "验收需要至少一次真实回放 BC 更新")
    saved = trainer.payload()
    for key in ("worker", "candidate", "optimizers", "counters", "sampler_state", "rng_state"):
        gl.require(t01.same(saved[key], payload[key]), f"加载 checkpoint 后 {key} 不一致")
    frozen_id = trainer.model.frozen_identity()
    gl.require(frozen_id == cache.bundle["bundle_id"], "加载后的冻结依赖发生变化")
    report.check("strict_load", "PASS", f"T04 权重、新优化器、计数及 RNG 完整加载；step={trainer.step}")
    blocks = cache.metadata["episode_states"]
    probes = np.linspace(0, len(blocks) - 1, min(args.state_probes, len(blocks)), dtype=int)
    for index in probes:
        block = blocks[int(index)]
        frames = np.asarray(block["frames"], dtype=np.int64)
        recovered = trainer.model.state_encoder.rollout(dataset.episode(int(index)), int(frames[-1]))
        positions = np.unique(np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int))
        t02.vector_difference(report, f"causal_cache_episode_{index}", recovered.cpu().numpy()[frames[positions]],
                              np.array(cache.states[block["offset"] + positions], copy=True), check=True)
    report.check("causal_cache", "PASS", f"抽查 {len(probes)} 局：从真实 reset 重算完整历史，与当前状态缓存一致")

    for name, bad, message in (("cache_guard", dict(payload, cache_id="wrong"), "缓存 ID"),
                               ("counter_guard", dict(payload, counters=dict(payload["counters"], new_env_steps=1)), "训练计数"),
                               ("artifact_guard", dict(payload, verification_artifact=True), "合成验收")):
        t02.rejection(report, name, lambda bad=bad: trainer.restore(bad), message)
    bad_worker = dict(payload["worker"])
    first = "actor.layers.Actor_linear0.weight"
    bad_worker[first] = bad_worker[first][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, worker=bad_worker)), "权重不兼容")
    roundtrip = trainer.payload(verification_artifact=True)
    reference = bc.frozen_bundle_reference(cache)
    bc.save_checkpoint(roundtrip, report.directory / "roundtrip_verification.pt", reference)
    reloaded = bc.torch_load(report.directory / "roundtrip_verification.pt")
    gl.require(reloaded.get("checkpoint_storage") == bc.CHECKPOINT_STORAGE and
               reloaded["frozen_bundle"]["bundle_id"] == cache.bundle["bundle_id"], "共享冻结依赖加载不一致")
    report.check("shared_frozen_bundle", "PASS", "新快照不重复保存冻结权重；相对路径、文件 SHA256 与 bundle ID 验证后加载；旧内嵌格式仍可读取")
    other = bc.Trainer(cache, payload["options"], args.device)
    other.restore(reloaded, allow_verification=True)
    again = other.payload(verification_artifact=True)
    for key in ("worker", "candidate", "optimizers", "counters", "sampler_state", "rng_state"):
        gl.require(t01.same(roundtrip[key], again[key]), f"保存/恢复 {key} 不一致")
    rows = cache.worker_rows["validation"][:32]
    features, goals, remaining, _ = cache.worker_batch(rows, args.device)
    with torch.no_grad():
        t02.vector_difference(report, "worker_roundtrip", trainer.model.action_dist(features, goals, remaining).probs,
                              other.model.action_dist(features, goals, remaining).probs, check=True)
        t02.vector_difference(report, "candidate_roundtrip", trainer.model.candidate(features).softmax(-1),
                              other.model.candidate(features).softmax(-1), check=True)
        proposals = trainer.model.propose(features, min(4, cache.metadata["goal_count"]))
        gl.require(t01.same(proposals, other.model.propose(features, min(4, cache.metadata["goal_count"]))), "候选前向保存恢复不一致")
        gl.require(t01.same(proposals["goals"], trainer.model.library.goal(proposals["goal_ids"])), "候选目标与正式目标库不一致")
    report.check("roundtrip", "PASS", "保存恢复的权重、新优化器、计数、采样器、RNG、低层及候选前向一致")

    # Only RAM copies advance once to prove optimizer/sampler resume works.
    # Neither the real training checkpoint nor roundtrip artifact is updated.
    versions = {id(p): p._version for model in (trainer.model, other.model)
                for module in (model.state_encoder, model.library, model.original_actor) for p in module.parameters()}
    rng = bc.capture_rng(args.device)
    trainer.update()
    bc.restore_rng(rng, args.device)
    other.update()
    for name, left, right in (("worker", trainer.model.worker, other.model.worker), ("candidate", trainer.model.candidate, other.model.candidate)):
        gl.require(t01.same(left.state_dict(), right.state_dict()) and
                   t01.same(trainer.optimizers[name].state_dict(), other.optimizers[name].state_dict()), f"恢复后下一次 {name} 更新不一致")
    gl.require(t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and trainer.counters() == other.counters(), "恢复后采样器/计数不同")
    gl.require(versions == {id(p): p._version for model in (trainer.model, other.model)
                           for module in (model.state_encoder, model.library, model.original_actor) for p in module.parameters()} and
               trainer.model.frozen_identity() == frozen_id and other.model.frozen_identity() == frozen_id, "验收探针更新冻结参数")
    report.check("next_update_equivalence", "PASS", "两份内存模型恢复后下一次真实回放 BC 更新的参数/优化器/采样器/计数一致")
    report.check("frozen_ownership", "PASS", "仅新 worker/candidate 可更新；冻结模型与目标库版本/内容未变")
    report.data.update(counters=payload["counters"], verification_probe_updates_per_copy=1,
                       controlled_reachability_verified=False)
    report.check("probe_scope", "PASS", "内存更新仅用于恢复验收；roundtrip_verification.pt 禁止真实 resume；原训练 checkpoint 未写入")
    report.check("scope", "WARN", "这是数据/训练/恢复工程验收，T05 之前不能宣称目标控制或任务成功率提升")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset-dir", required=True)
    prepare_parser.add_argument("--t03-verify-dir", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--cache-dir", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--steps", type=int, default=2000, help="累计 BC 更新数，不是环境步数")
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--log-every", type=int, default=20)
    train_parser.add_argument("--save-every", type=int, default=100)
    for name in ("seed", "batch-size", "candidate-hidden"):
        train_parser.add_argument("--" + name, type=int)
    for name in ("learning-rate", "candidate-learning-rate", "grad-clip"):
        train_parser.add_argument("--" + name, type=float)
    train_parser.add_argument("--conditioning", choices=("goal", "no_goal"))
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--cache-dir", required=True)
    verify_parser.add_argument("--checkpoint", required=True, help="T04 训练输出的 latest.pt")
    verify_parser.add_argument("--samples", type=int, default=64)
    verify_parser.add_argument("--state-probes", type=int, default=2, help="重新恢复完整真实历史的 episode 数量")
    for command in (prepare_parser, train_parser, verify_parser):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-root", default="relevance_map/t04_outputs")
        command.add_argument("--output-dir", help="新的独立目录，拒绝覆盖")
    args = parser.parse_args()
    if args.command == "train" and (min(args.steps, args.eval_every, args.log_every, args.save_every) < 1 or
            (args.seed is not None and not 0 <= args.seed < 2**32)):
        parser.error("训练计数需要正数、种子需要合法非负数")
    if args.command == "verify" and min(args.samples, args.state_probes) < 1:
        parser.error("至少抽查一段和一局状态")
    try:
        if args.command == "prepare":
            dataset_dir = baseline.project_path(args.dataset_dir)
            meta = read_json(dataset_dir / "segments_manifest.json")
            protected = [dataset_dir, baseline.project_path(args.t03_verify_dir)]
            source_files = [dataset_dir / "segments_manifest.json", dataset_dir / "segments.npz", dataset_dir / "report.json",
                            baseline.project_path(args.t03_verify_dir) / "report.json"]
        else:
            cache_dir = baseline.project_path(args.cache_dir)
            cache_meta = read_json(cache_dir / "cache_manifest.json")
            meta = cache_meta["source_metadata"]
            dataset_dir = Path(cache_meta["dataset_dir"])
            protected = [cache_dir, dataset_dir, Path(cache_meta["t03_verify_dir"])]
            source_files = [cache_dir / name for name in ("states.npy", "tables.npz", "cache_manifest.json", "frozen_bundle.pt", "report.json")]
            source_files += [dataset_dir / "segments_manifest.json", dataset_dir / "segments.npz", dataset_dir / "report.json",
                             Path(cache_meta["t03_verify_dir"]) / "report.json"]
            input_checkpoint = args.resume if args.command == "train" else args.checkpoint
            if input_checkpoint:
                source_files.append(baseline.project_path(input_checkpoint))
                protected.append(baseline.project_path(input_checkpoint).parent)
        protected += [Path(meta["checkpoint"]["path"]).parent, Path(meta["library_path"]).parent, Path(meta["baseline_dir"])]
        source_files += [Path(meta["checkpoint"]["path"]), Path(meta["library_path"]),
                         Path(meta["baseline_dir"]) / "resolved_config.json", Path(meta["baseline_dir"]) / "report.json"]
        source_files += [Path(entry["path"]) for entry in meta["episodes"]]
        protected += [Path(entry["path"]).parent for entry in meta["episodes"]]
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"无法读取输入清单：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else baseline.project_path(args.output_root) / (args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出目录必须独立于原运行、源回放、T00/T02/T03、缓存及恢复输入目录")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: 无法创建 {directory}: {error}；请检查磁盘可用空间及 inode，或指定其他文件系统上的 --output-dir", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    try:
        import torch
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请检查 --device")
        baseline.runtime_info(report)
        report.data["runtime"]["torch"] = str(torch.__version__)
        before = {str(path): baseline.file_signature(path) for path in source_files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        for entry in meta["episodes"]:
            if baseline.file_signature(Path(entry["path"])) != entry["signature"]:
                raise ValueError("T03 源 replay 大小/修改时间改变")
        if baseline.file_signature(Path(meta["checkpoint"]["path"])) != meta["checkpoint"]:
            raise ValueError("T03 原初始化 checkpoint 大小/修改时间改变")
        {"prepare": prepare, "train": train, "verify": verify}[args.command](args, report)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if report.data.get("source_inputs_before"):
            try:
                after = {str(path): baseline.file_signature(path) for path in source_files}
                report.data["source_inputs_after"] = after
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
            else:
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型、回放及输入产物大小/修改时间未变（非内容哈希）；输出在独立目录")
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
