"""T03 offline hindsight-segment build/acceptance; never starts MineDojo."""

import argparse
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t02_goal_library as t02


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        label = "T03_SEGMENTS_BUILD" if self.data["command"] == "build" else "T03_SEGMENTS_VERIFY"
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def accepted_library(args, report):
    import goal_library as gl

    artifact = baseline.project_path(args.library)
    library = gl.load_library(artifact, args.device)
    build = json.loads((artifact.parent / "report.json").read_text(encoding="utf-8"))
    verification = json.loads((baseline.project_path(args.t02_verify_dir) / "report.json").read_text(encoding="utf-8"))
    for record, command in ((build, "build"), (verification, "verify")):
        gl.require(record.get("command") == command and record.get("status") in ("passed", "passed_with_warnings"), "需要通过 T02 build/verify 的目标库")
        gl.require(record.get("library_id") == library.payload()["library_id"], "T02 验收记录与本次目标库 ID 不同")
    selection = json.loads((artifact.parent / "replay_selection.json").read_text(encoding="utf-8"))
    report.check("t02_input", "PASS", "目标库已通过 T02 构建/独立验收；沿用完整 episode 的训练/检查划分")
    return artifact, library, selection


def summarize(arrays, metadata):
    import numpy as np

    rows = []
    for split in ("train", "validation"):
        mask = np.array([metadata["episodes"][int(i)]["split"] == split for i in arrays["episode_index"]])
        for goal in range(metadata["goal_count"]):
            selected = mask & (arrays["goal_id"] == goal)
            rows.append({"split": split, "goal_id": goal, "segments": int(selected.sum()),
                         "episodes": len(set(arrays["episode_index"][selected].tolist())),
                         "action_labels": int((arrays["end"][selected] - arrays["start"][selected]).sum())})
    return rows


def build(args, report):
    import numpy as np
    import goal_library as gl
    import goal_segments as gs

    artifact, library, selection = accepted_library(args, report)
    config, _ = t02.resolve_baseline(args, report)
    gl.require(config["task"] == library.metadata["task"], "T00 与目标库任务不同")
    gl.require(baseline.file_signature(baseline.project_path(args.checkpoint)) == library.metadata["checkpoint"], "目标库与当前 checkpoint 来源不同")
    gl.require(args.horizon <= config.get("macro_max_steps", 16), "片段长度超过当前低层 horizon")
    episodes = sorted(copy.deepcopy(selection["selected_episodes"]), key=lambda entry: entry["path"])
    columns = {key: [] for key in ("episode_index", "start", "end", "goal_id", "goals")}
    examples, horizon_counts = [], {str(h): 0 for h in range(1, args.horizon + 1)}
    for index, entry in enumerate(episodes):
        path = Path(entry["path"])
        gl.require(baseline.file_signature(path) == entry["signature"], "T02 源 replay 已变更")
        episode = gs.read_episode(path, library.metadata["max_steps"], config["num_actions"])
        gl.require(episode["usable_length"] == entry["usable_length"], "T02 源 episode 边界改变")
        pairs = gs.sample_pairs(episode["usable_length"], args.horizon, args.segments_per_episode, args.seed + index)
        endpoints = np.array(sorted({end for _, end in pairs}), dtype=np.int64)
        vectors = []
        for start in range(0, len(endpoints), args.batch_size):
            positions = endpoints[start:start + args.batch_size]
            vectors.append(library({key: episode[key][positions] for key in ("image", "heatmap")}).cpu().numpy())
        vectors = np.concatenate(vectors)
        classes = library.assign(vectors).cpu().numpy()
        endpoint_rows = {int(endpoint): row for row, endpoint in enumerate(endpoints)}
        entry["sampling_seed"] = args.seed + index
        entry["segment_count"] = len(pairs)
        # Do not retain T02's sparse frame indices as if they were a trajectory.
        entry.pop("indices", None)
        for start, end in pairs:
            row = endpoint_rows[end]
            for key, value in (("episode_index", index), ("start", start), ("end", end), ("goal_id", int(classes[row])), ("goals", vectors[row])):
                columns[key].append(value)
            horizon_counts[str(end - start)] += 1
        if len(examples) < 8:
            start, end = pairs[0]
            view = gs.segment_view(episode, start, end, vectors[endpoint_rows[end]], int(classes[endpoint_rows[end]]), args.horizon)
            examples.append({"episode": path.name, "split": entry["split"], "start": start, "end": end,
                             "current_indices": view["current_indices"].tolist(), "label_indices": view["label_indices"].tolist(),
                             "action_argmax": view["next_actions"].argmax(-1).tolist(), "remaining": view["remaining"].tolist(),
                             "goal_id": view["goal_id"], "causal_history": [0, end - 1], "endpoint_in_state_input": False})
        gl.require(baseline.file_signature(path) == entry["signature"], "构建过程中源 replay 已变更")
        if (index + 1) % 16 == 0 or index + 1 == len(episodes):
            print(f"[SEGMENTS] episodes={index + 1}/{len(episodes)} segments={len(columns['start'])}", flush=True)
    arrays = {key: np.asarray(values, dtype=np.float32 if key == "goals" else np.int64) for key, values in columns.items()}
    metadata = {"format": gs.FORMAT, "state_policy": gs.STATE_POLICY,
                "action_alignment": "obs[t] -> action[t+1]; incoming action[t]",
                "goal_definition": "continuous frozen encoding of the real endpoint, not a cluster prototype",
                "encoder_id": library.payload()["encoder_id"], "library_id": library.payload()["library_id"],
                "library_path": str(artifact), "task": config["task"], "goal_dim": int(library.goals.shape[1]),
                "goal_count": int(len(library.goals)), "action_dim": int(config["num_actions"]),
                "horizon": args.horizon, "max_steps": library.metadata["max_steps"], "seed": args.seed,
                "step_unit": "recorded_env_step", "action_repeat": config.get("action_repeat", 1),
                "segments_per_episode": args.segments_per_episode, "batch_size": args.batch_size,
                "checkpoint": baseline.file_signature(baseline.project_path(args.checkpoint)),
                "baseline_dir": str(baseline.project_path(args.baseline_dir)), "episodes": episodes,
                "seed_holdout": "unavailable_in_legacy_replay", "trained_worker": False}
    gs.validate_dataset(arrays, metadata)
    identifier = gs.content_id(arrays, metadata)
    np.savez_compressed(report.directory / "segments.npz", **arrays)
    baseline.write_json(report.directory / "segments_manifest.json", dict(metadata, content_id=identifier))
    baseline.write_json(report.directory / "alignment_examples.json", examples)
    rows = summarize(arrays, metadata)
    baseline.write_csv(report.directory / "coverage.csv", rows)
    stats = {"episodes": len(episodes), "segments": len(arrays["start"]),
             "train_episodes": sum(entry["split"] == "train" for entry in episodes),
             "validation_episodes": sum(entry["split"] == "validation" for entry in episodes),
             "action_labels": int((arrays["end"] - arrays["start"]).sum()), "horizon_histogram": horizon_counts,
             "discarded_tail_frames": sum(entry["discarded_tail"] for entry in episodes),
             "seed_holdout_available": False, "controlled_reachability_verified": False}
    baseline.write_json(report.directory / "diagnostics.json", stats)
    reloaded = gs.SegmentDataset(report.directory)
    gl.require(reloaded.content_id == identifier, "片段保存恢复改变内容 ID")
    for row in (0, len(reloaded) - 1):
        sample = reloaded[row]
        gl.require(np.array_equal(sample["remaining"], np.arange(len(sample["next_actions"]), 0, -1)), "剩余步数错误")
    report.data.update(content_id=identifier, library_id=metadata["library_id"], diagnostics=stats)
    report.check("real_segments", "PASS", f"{stats['episodes']} 局、{stats['segments']} 段、{stats['action_labels']} 个下一动作标签；均在同一真实 episode 内")
    report.check("hindsight_targets", "PASS", "真实终点 RGB/heatmap 的连续编码作为统一目标；类别仅用于候选模型监督，不是到达成功标签")
    report.check("serialization", "PASS", "保存后独立加载、内容 ID 和样本结构一致；源 replay 未写入")
    report.check("seed_holdout", "WARN", "旧 replay 未记录每局环境种子，只能按 episode 留出；不宣称种子泛化")


def structural_probes(report, action_dim, horizon, goal_dim):
    import numpy as np
    import goal_library as gl
    import goal_segments as gs

    length = max(8, min(action_dim, 12))
    episode = {"image": np.zeros((length, 4, 4, 3), dtype=np.uint8), "heatmap": np.zeros((length, 4, 4), dtype=np.uint8),
               "obs_reward": np.arange(length, dtype=np.float32)[:, None],
               "action": np.zeros((length, action_dim), dtype=np.float32),
               "is_first": np.arange(length) == 0, "is_last": np.arange(length) == length - 1,
               "is_terminal": np.zeros(length, dtype=bool)}
    episode["action"][1:] = np.eye(action_dim, dtype=np.float32)[np.arange(1, length) % action_dim]
    goal = np.zeros(goal_dim, dtype=np.float32)
    goal[0] = 1
    start, end = 2, min(length - 1, 2 + horizon)
    view = gs.segment_view(episode, start, end, goal, 0, horizon)
    gl.require(np.array_equal(view["next_actions"], episode["action"][start + 1:end + 1]), "动作 marker 对齐错误")
    gl.require(not np.array_equal(view["next_actions"], episode["action"][start:end]), "动作 marker 不能区分 off-by-one")
    gl.require(len(view["history"]["image"]) == end and set(view["history"]) == set(gs.HISTORY_KEYS), "当前历史混入终点或监督字段")
    gl.require(np.array_equal(view["loss_mask"], np.arange(end) >= start) and np.array_equal(view["remaining"], np.arange(end - start, 0, -1)), "预热 mask 或递减步数错误")
    gl.require(np.array_equal(view["goals"], np.repeat(goal[None], end - start, axis=0)), "同段目标不统一")
    bad = copy.deepcopy(episode)
    bad["is_first"][start + 1] = True
    t02.rejection(report, "reset_guard", lambda: gs.segment_view(bad, start, end, goal, 0, horizon), "reset")
    bad = copy.deepcopy(episode)
    bad["is_terminal"][start] = True
    t02.rejection(report, "terminal_guard", lambda: gs.segment_view(bad, start, end, goal, 0, horizon), "终止")
    bad = copy.deepcopy(episode)
    bad["is_virtual"] = np.ones(length, dtype=bool)
    t02.rejection(report, "virtual_guard", lambda: gs.segment_view(bad, start, end, goal, 0, horizon), "虚拟")
    t02.rejection(report, "endpoint_guard", lambda: gs.segment_view(episode, start, length, goal, 0, horizon), "越界")
    t02.rejection(report, "horizon_guard", lambda: gs.segment_view(episode, start, end, goal, 0, 0), "最大执行步数")
    report.check("alignment_markers", "PASS", "可区分的动作 marker 确认 obs[t]→action[t+1]，真实 incoming action[t] 保留；统一目标、剩余步数及预热 mask 正确")
    report.check("probe_scope", "PASS", "合成 marker 仅在内存中验收边界/对齐，不写入训练片段")


def verify(args, report):
    import numpy as np
    import torch
    import goal_library as gl
    import goal_segments as gs
    import long_horizon as lh

    dataset = gs.SegmentDataset(baseline.project_path(args.dataset_dir))
    metadata = dataset.metadata
    build_report = json.loads((dataset.directory / "report.json").read_text(encoding="utf-8"))
    gl.require(build_report.get("command") == "build" and build_report.get("status") in ("passed", "passed_with_warnings") and
               build_report.get("content_id") == dataset.content_id, "需要通过 T03 build 的片段目录")
    library = gl.load_library(args.library or metadata["library_path"], args.device)
    payload = library.payload()
    gl.require(payload["library_id"] == metadata["library_id"] and payload["encoder_id"] == metadata["encoder_id"], "片段与目标库内容 ID 不匹配")
    report.data.update(content_id=dataset.content_id, library_id=metadata["library_id"])
    report.check("content_identity", "PASS", "片段内容哈希、目标库/编码器 ID 和状态约定匹配")
    structural_probes(report, metadata["action_dim"], metadata["horizon"], metadata["goal_dim"])
    split_paths = {split: {entry["path"] for entry in metadata["episodes"] if entry["split"] == split} for split in ("train", "validation")}
    gl.require(not split_paths["train"] & split_paths["validation"], "训练/检查 episode 重叠")
    for index, entry in enumerate(metadata["episodes"]):
        episode = dataset.episode(index)
        expected = gs.sample_pairs(episode["usable_length"], metadata["horizon"], metadata["segments_per_episode"], entry["sampling_seed"])
        positions = np.flatnonzero(dataset.arrays["episode_index"] == index)
        actual = list(zip(dataset.arrays["start"][positions].tolist(), dataset.arrays["end"][positions].tolist()))
        gl.require(actual == expected, "相同种子不能重现片段编号")
    report.check("episode_split_and_boundaries", "PASS", f"{len(split_paths['train'])}/{len(split_paths['validation'])} 局分开；全量源 episode 的动作、边界、后缀和同种子采样检查通过")
    indices = np.sort(np.random.RandomState(0).choice(len(dataset), min(args.samples, len(dataset)), replace=False))
    for row in indices:
        sample = dataset[int(row)]
        ep_index = int(dataset.arrays["episode_index"][row])
        episode = dataset.episode(ep_index)
        start, end = int(dataset.arrays["start"][row]), int(dataset.arrays["end"][row])
        gl.require(np.array_equal(sample["next_actions"], episode["action"][start + 1:end + 1]), "真实动作标签错位")
        gl.require(np.array_equal(sample["history"]["action"], episode["action"][:end]), "当前状态的历史动作错位")
        gl.require(len(sample["history"]["image"]) == end and not sample["history"]["is_first"][1:].any(), "历史从错误起点恢复或含未来帧")
        endpoint = {key: episode[key][end:end + 1] for key in ("image", "heatmap")}
        target = library(endpoint)
        t02.vector_difference(report, f"endpoint_goal_{row}", target, dataset.arrays["goals"][row:row + 1], check=True)
        gl.require(int(library.assign(target)[0]) == sample["goal_id"], "终点目标类别改变")
        gl.require(np.array_equal(sample["goals"], np.repeat(dataset.arrays["goals"][row:row + 1], end - start, axis=0)), "目标在片段中变化")
        gl.require(np.array_equal(sample["remaining"], np.arange(end - start, 0, -1)), "剩余步数错误")
    report.check("real_action_alignment", "PASS", f"抽查 {len(indices)} 段：下一动作、当前历史动作、真实终点目标和逐步剩余步数正确")
    # Only a short real prefix is used for the actual frozen RSSM causality
    # probe, not thousands of full-episode training passes.
    source_args = copy.copy(args)
    source_args.checkpoint = args.checkpoint or metadata["checkpoint"]["path"]
    source_args.baseline_dir = metadata["baseline_dir"]
    config, shapes = t02.resolve_baseline(source_args, report)
    gl.require(baseline.file_signature(baseline.project_path(source_args.checkpoint)) == metadata["checkpoint"], "片段初始化 checkpoint 已变更")
    checkpoint = lh.read_checkpoint(baseline.project_path(source_args.checkpoint))
    gl.require(gl.tensor_digest(gl.cnn_weights(checkpoint["agent_state_dict"])) == payload["source_encoder_hash"], "目标编码器与当前 WM 视觉权重不同")
    frozen = gs.state_encoder_from_checkpoint(checkpoint, config, shapes, args.device)
    del checkpoint
    gc.collect()
    versions = {key: value._version for key, value in frozen.named_parameters()}
    index = next((i for i, entry in enumerate(metadata["episodes"]) if entry["usable_length"] >= 8), None)
    gl.require(index is not None, "缺少至少 8 帧的 episode，无法检查真实 RSSM 因果性")
    episode = dataset.episode(index)
    stop = min(args.prefix_steps, len(episode["image"]) - 2)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(args.device).clone() if torch.device(args.device).type == "cuda" else None
    reference = frozen.rollout(episode, stop)
    gl.require(torch.equal(cpu_rng, torch.get_rng_state()) and (cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(args.device))), "冻结状态恢复改变了调用方 RNG")
    changed = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in episode.items()}
    for key in ("image", "heatmap"):
        changed[key][stop + 1:] = 255 - changed[key][stop + 1:]
    changed["obs_reward"][stop + 1:] += 123
    changed["action"][stop + 1:] = np.roll(changed["action"][stop + 1:], 1, axis=-1)
    t02.vector_difference(report, "future_invariance", frozen.rollout(changed, stop), reference, check=True)
    prefix = {key: value[:stop + 1].copy() for key, value in episode.items() if isinstance(value, np.ndarray)}
    t02.vector_difference(report, "prefix_equivalence", frozen.rollout(prefix, stop), reference, check=True)
    history = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in episode.items()}
    history["action"][1:stop + 1] = np.roll(history["action"][1:stop + 1], 1, axis=-1)
    altered = frozen.rollout(history, stop)
    difference = float(torch.max(torch.abs(altered[1:] - reference[1:])))
    gl.require(not torch.allclose(altered[1:], reference[1:], atol=1e-6, rtol=1e-6), "RSSM 未响应过去动作，不能证明历史动作路径有效")
    gl.require(versions == {key: value._version for key, value in frozen.named_parameters()} and not any(p.requires_grad for p in frozen.parameters()), "验收更新了冻结 WM 参数")
    report.data["state_encoder"] = {"state_policy": gs.STATE_POLICY, "feature_dim": reference.shape[-1],
                                    "state_encoder_id": gl.tensor_digest(frozen.state_dict()), "probe_frames": stop + 1,
                                    "past_action_max_difference": difference, "optimizer_updates": 0}
    report.check("past_action_sensitivity", "PASS", f"仅改变过去真实动作会改变 RSSM 状态；最大差异 {difference:.6g}")
    report.check("causal_state", "PASS", f"严格加载原 encoder/RSSM，{reference.shape[-1]} 维；未来扰动与完整前缀等价检查通过；从 reset 逐步恢复")
    report.check("no_training", "PASS", "无优化器/参数更新；保留调用方 RNG；没有环境或 MineCLIP 交互")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--checkpoint", required=True)
    build_parser.add_argument("--baseline-dir", required=True)
    build_parser.add_argument("--library", required=True)
    build_parser.add_argument("--t02-verify-dir", required=True)
    build_parser.add_argument("--horizon", type=int, default=16)
    build_parser.add_argument("--segments-per-episode", type=int, default=64)
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--seed", type=int, default=0)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--dataset-dir", required=True)
    verify_parser.add_argument("--library", help="默认使用片段清单中的同一份目标库")
    verify_parser.add_argument("--checkpoint", help="默认使用片段清单中的原初始化模型")
    verify_parser.add_argument("--samples", type=int, default=64)
    verify_parser.add_argument("--prefix-steps", type=int, default=24)
    for command in (build_parser, verify_parser):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-root", default="relevance_map/t03_outputs")
        command.add_argument("--output-dir", help="新的独立输出目录，拒绝覆盖")
    args = parser.parse_args()
    if args.command == "build":
        if min(args.horizon, args.segments_per_episode, args.batch_size) < 1 or not 0 <= args.seed < 2**32 - 10000:
            parser.error("需要正长度/采样参数和合法非负种子")
        protected = [baseline.project_path(args.checkpoint).parent, baseline.project_path(args.baseline_dir),
                     baseline.project_path(args.library).parent, baseline.project_path(args.t02_verify_dir)]
        try:
            selection = json.loads((baseline.project_path(args.library).parent / "replay_selection.json").read_text(encoding="utf-8"))
            source_files = [baseline.project_path(args.checkpoint), baseline.project_path(args.library)]
            source_files += [Path(entry["path"]) for entry in selection["selected_episodes"]]
            protected += [Path(entry["path"]).parent for entry in selection["selected_episodes"]]
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"无法读取 T02 输入清单：{error}")
    else:
        if args.samples < 1 or args.prefix_steps < 4:
            parser.error("至少抽查一段，因果状态检查至少 4 步")
        protected = [baseline.project_path(args.dataset_dir)]
        if args.checkpoint:
            protected.append(baseline.project_path(args.checkpoint).parent)
        if args.library:
            protected.append(baseline.project_path(args.library).parent)
        try:
            metadata = json.loads((baseline.project_path(args.dataset_dir) / "segments_manifest.json").read_text(encoding="utf-8"))
            source_files = [baseline.project_path(args.checkpoint or metadata["checkpoint"]["path"]),
                            baseline.project_path(args.library or metadata["library_path"]),
                            baseline.project_path(args.dataset_dir) / "segments_manifest.json",
                            baseline.project_path(args.dataset_dir) / "segments.npz"]
            source_files += [Path(entry["path"]) for entry in metadata["episodes"]]
            protected += [source_files[0].parent, source_files[1].parent]
            protected += [Path(entry["path"]).parent for entry in metadata["episodes"]]
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"无法读取 T03 输入清单：{error}")
    output_root = baseline.project_path(args.output_root)
    directory = baseline.project_path(args.output_dir) if args.output_dir else output_root / (args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents for path in protected):
        parser.error("输出目录必须独立于原运行、T00/T02 和输入片段目录")
    directory.mkdir(parents=True, exist_ok=False)
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    try:
        import torch
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请检查 --device")
        baseline.runtime_info(report)
        report.data["runtime"]["torch"] = torch.__version__
        before = {str(path): baseline.file_signature(path) for path in source_files}
        report.data["source_inputs_before"] = before
        if args.command == "build":
            build(args, report)
        else:
            verify(args, report)
    except (Exception, KeyboardInterrupt) as error:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.check("execution", "FAIL", f"{type(error).__name__}: {error}；见 error.txt")
    finally:
        if report.data.get("source_inputs_before"):
            try:
                after = {str(path): baseline.file_signature(path) for path in source_files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原 checkpoint、目标库及源 replay 大小/修改时间未变（非内容哈希）；输出在独立目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
