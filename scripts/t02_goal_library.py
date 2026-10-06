"""Build/verify a T02 goal library from real replay; never starts MineDojo."""

import argparse
import copy
from datetime import datetime
import gc
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        label = "T02_GOAL_LIBRARY_BUILD" if self.data["command"] == "build" else "T02_GOAL_LIBRARY_VERIFY"
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def rejection(report, name, operation, message):
    try:
        operation()
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f"{name} 未命中预期规则: {error}")
        report.check(name, "PASS", str(error))
    else:
        raise AssertionError(f"{name} 没有拒绝错误输入")


def vector_difference(report, name, actual, expected, *, check=False):
    """Persist the error magnitude before failing, rather than just allclose."""
    import numpy as np
    import torch
    import goal_library as gl

    actual = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    expected = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    gl.require(actual.shape == expected.shape and actual.ndim == 2,
               f"{name} 对照形状错误: {actual.shape} / {expected.shape}")
    gl.require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{name} 特征非有限")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    actual_norm = np.linalg.norm(actual.astype(np.float64), axis=1)
    expected_norm = np.linalg.norm(expected.astype(np.float64), axis=1)
    cosine = np.sum(actual.astype(np.float64) * expected.astype(np.float64), axis=1) / np.maximum(actual_norm * expected_norm, 1e-20)
    passed = bool(np.allclose(actual, expected, atol=3e-5, rtol=3e-5))
    details = {"shape": list(actual.shape), "max_abs_error": float(delta.max()),
               "mean_abs_error": float(delta.mean()), "max_cos_distance": float(np.max(1 - np.clip(cosine, -1, 1))),
               "min_actual_norm": float(actual_norm.min()), "min_expected_norm": float(expected_norm.min()),
               "per_row_max_abs_error": delta.max(axis=1).tolist(),
               "atol": 3e-5, "rtol": 3e-5, "allclose": passed}
    report.data.setdefault("numerical_checks", {})[name] = details
    baseline.write_json(report.directory / "numerical_checks.json", report.data["numerical_checks"])
    message = f"最大绝对误差 {details['max_abs_error']:.6g}；最大 cosine 距离 {details['max_cos_distance']:.6g}"
    if check:
        report.check(name, "PASS" if passed else "FAIL", message)
        gl.require(passed, f"{name} 特征不一致：{message}；见 numerical_checks.json")
    else:
        print(f"[NUMERICS] {name}: {message}", flush=True)
    return details


def resolve_baseline(args, report):
    source = baseline.project_path(args.baseline_dir)
    accepted = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if accepted.get("command") != "evaluate" or accepted.get("status") not in ("passed", "passed_with_warnings"):
        raise ValueError("需要已通过 T00 的 evaluate 输出目录")
    checkpoint = baseline.project_path(args.checkpoint)
    if baseline.project_path(accepted["baseline_summary"]["checkpoint"]) != checkpoint:
        raise ValueError("T00 报告与本次 checkpoint 路径不一致")
    if accepted["checkpoint_file_after"] != baseline.file_signature(checkpoint):
        raise ValueError("checkpoint 大小/修改时间已偏离 T00 验收记录")
    config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    report.data["configuration"] = {"baseline_dir": str(source), "task": config["task"],
                                    "historical_training_config_verified": False}
    report.check("baseline_input", "PASS", "复用通过 T00 的当前配置、观测空间与 checkpoint 文件记录")
    return config, accepted["observation_shapes"]


def sample_records(args, config, report):
    import numpy as np
    import goal_library as gl

    directories = [baseline.project_path(value) for value in args.replay_dir] if args.replay_dir else [baseline.project_path(args.checkpoint).parent / "train_eps"]
    files = sorted({file.resolve() for directory in directories for file in directory.glob("*.npz")})
    # Legacy names end in -<length>.npz. Copies or multiple saved suffixes of
    # the same UUID must not appear on both sides of the episode split.
    episodes, duplicate_files = {}, []
    for path in files:
        match = re.fullmatch(r"(.+)-(\d+)", path.stem)
        episode_id = match[1] if match else path.stem
        advertised_length = int(match[2]) if match else 0
        previous = episodes.get(episode_id)
        if previous is None:
            episodes[episode_id] = (advertised_length, path)
        elif advertised_length > previous[0]:
            duplicate_files.append(str(previous[1]))
            episodes[episode_id] = (advertised_length, path)
        else:
            duplicate_files.append(str(path))
    files = sorted(value[1] for value in episodes.values())
    gl.require(len(files) >= 5, "至少需要 5 个真实 episode，才能按 episode 留出检查集")
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(files))
    accepted, rejected = [], []
    for position in order:
        path = files[int(position)]
        try:
            before = baseline.file_signature(path)
            episode = gl.read_episode(path, config["episode_max_steps"])
            gl.require(baseline.file_signature(path) == before, "读取过程中 replay 被改写")
            candidates = np.arange(0, episode["usable_length"], args.stride, dtype=np.int64)
            candidates = np.unique(np.append(candidates, episode["usable_length"] - 1))
            if len(candidates) > args.frames_per_episode:
                candidates = np.sort(rng.choice(candidates, args.frames_per_episode, replace=False))
            accepted.append({"path": str(path), "signature": before, "indices": candidates.tolist(),
                             "original_length": episode["original_length"], "usable_length": episode["usable_length"],
                             "discarded_tail": episode["discarded_tail"]})
            del episode
        except (ValueError, OSError, KeyError, EOFError) as error:
            rejected.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
        if len(accepted) >= args.max_episodes:
            break
    if len(accepted) < 5:
        baseline.write_json(report.directory / "replay_selection.json", {"selected_episodes": accepted, "rejected_episodes": rejected})
        raise ValueError("有效 replay 不足；见 replay_selection.json 中的排除原因")
    # Split whole episodes BEFORE extracting features/fitting PCA or clusters.
    validation_count = max(1, int(round(len(accepted) * args.validation_fraction)))
    validation_paths = {entry["path"] for entry in accepted[:validation_count]}
    records = []
    for entry in sorted(accepted, key=lambda item: item["path"]):
        entry["split"] = "validation" if entry["path"] in validation_paths else "train"
        for index in entry["indices"]:
            records.append({"row": len(records), "path": entry["path"], "frame": index, "split": entry["split"]})
    selection = {"directories": [str(path) for path in directories], "available_episodes": len(files),
                 "selected_episodes": accepted, "rejected_episodes": rejected, "records": records,
                 "duplicate_files_excluded": duplicate_files,
                 "seed": args.seed, "real_image_policy": "image/heatmap only; first episode until first ending or max_steps",
                 "validation_paths": sorted(validation_paths)}
    baseline.write_json(report.directory / "replay_selection.json", selection)
    gl.require(not ({entry["path"] for entry in accepted if entry["split"] == "train"} & validation_paths), "训练/检查 episode 发生重叠")
    report.data["sampling"] = {"available_episodes": len(files), "selected_episodes": len(accepted), "frames": len(records),
                               "validation_episodes": validation_count, "discarded_tail_frames": sum(entry["discarded_tail"] for entry in accepted),
                               "rejected_episodes": len(rejected)}
    report.check("real_replay", "PASS", f"选择 {len(accepted)} 局、{len(records)} 帧；保留真实 RGB，排除终止后后缀；检查集 {validation_count} 局")
    if rejected:
        report.check("replay_rejections", "WARN", f"排除 {len(rejected)} 个不合规文件，原因已记录；没有静默补造观测")
    return selection


def extract_features(encoder, selection, config, args, report):
    import numpy as np
    import torch
    import goal_library as gl

    path = report.directory / "raw_features.npy"
    features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                         shape=(len(selection["records"]), encoder.spec["raw_dim"]))
    # Per-episode loading limits decoded RGB memory; feature cache is mmap.
    records_by_path = {}
    for record in selection["records"]:
        records_by_path.setdefault(record["path"], []).append(record)
    for number, entry in enumerate(sorted(selection["selected_episodes"], key=lambda item: item["path"]), 1):
        episode = gl.read_episode(entry["path"], config["episode_max_steps"])
        gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "replay 在选样后被改写")
        records = records_by_path[entry["path"]]
        for start in range(0, len(records), args.batch_size):
            group = records[start:start + args.batch_size]
            indices = [record["frame"] for record in group]
            observation = {key: episode[key][indices] for key in ("image", "heatmap")}
            raw = encoder.raw_features(observation).cpu().numpy()
            gl.require(np.isfinite(raw).all(), "冻结 CNN 产生非有限特征")
            features[[record["row"] for record in group]] = raw
        if number % 16 == 0 or number == len(selection["selected_episodes"]):
            print(f"[FEATURES] episodes={number}/{len(selection['selected_episodes'])}", flush=True)
    features.flush()
    del features
    report.check("frozen_encoder", "PASS", "严格加载原视觉 CNN；只读真实 RGB/heatmap，粗空间池化；没有梯度或优化器更新")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def read_representatives(records, indices, max_steps):
    import numpy as np
    import goal_library as gl

    references = [copy.deepcopy(records[int(index)]) for index in indices]
    images, heatmaps = [], []
    cache_path = None
    for reference in references:
        if reference["path"] != cache_path:
            episode = gl.read_episode(reference["path"], max_steps)
            cache_path = reference["path"]
        images.append(episode["image"][reference["frame"]].copy())
        heatmaps.append(episode["heatmap"][reference["frame"]].copy())
    return references, np.stack(images), np.stack(heatmaps)


def coverage(features, labels, records, centers):
    import numpy as np

    rows = []
    for goal in range(len(centers)):
        row = {"goal_id": goal}
        for split in ("train", "validation"):
            indices = [index for index, reference in enumerate(records) if reference["split"] == split and labels[index] == goal]
            count = sum(reference["split"] == split for reference in records)
            row[f"{split}_frames"] = len(indices)
            row[f"{split}_fraction"] = len(indices) / max(1, count)
            row[f"{split}_episodes"] = len({records[index]["path"] for index in indices})
            row[f"{split}_mean_cos_distance"] = float(np.mean(1 - features[indices] @ centers[goal])) if indices else None
        rows.append(row)
    return rows


def gallery(library, rows, records, features, labels, max_steps, directory):
    import numpy as np
    from PIL import Image, ImageDraw

    images = library.images.cpu().numpy()
    heatmaps = library.heatmaps.cpu().numpy()
    tile_w, tile_h, side = 404, 252, 192
    atlas = Image.new("RGB", (tile_w * 4, tile_h * ((len(images) + 3) // 4)), "white")
    draw = ImageDraw.Draw(atlas)
    representatives_dir = directory / "representatives"
    representatives_dir.mkdir()
    for goal, image in enumerate(images):
        x, y = (goal % 4) * tile_w, (goal // 4) * tile_h
        rgb = Image.fromarray(image)
        heat = Image.fromarray(heatmaps[goal], "L")
        rgb.save(representatives_dir / f"goal_{goal:02d}_rgb.png")
        heat.save(representatives_dir / f"goal_{goal:02d}_heatmap.png")
        atlas.paste(rgb.resize((side, side), Image.Resampling.NEAREST), (x + 4, y + 24))
        atlas.paste(heat.convert("RGB").resize((side, side), Image.Resampling.NEAREST), (x + side + 12, y + 24))
        draw.text((x + 4, y + 3), f"goal {goal:02d} | RGB + spatial heatmap", fill="black")
        draw.text((x + 4, y + 220), f"train {rows[goal]['train_frames']} frames / {rows[goal]['train_episodes']} eps", fill="black")
        draw.text((x + 4, y + 235), f"validation {rows[goal]['validation_frames']} frames / {rows[goal]['validation_episodes']} eps", fill="black")
    atlas.save(directory / "representatives.png")
    # Three members from different training episodes help reveal classes that
    # mostly identify terrain/lighting rather than useful local configurations.
    examples = Image.new("RGB", (side * 3 + 24, (side + 24) * len(images)), "white")
    draw = ImageDraw.Draw(examples)
    references = []
    centers = library.centers.cpu().numpy()
    for goal in range(len(images)):
        candidates = np.array([index for index, record in enumerate(records) if record["split"] == "train" and labels[index] == goal])
        order = candidates[np.argsort(-(features[candidates] @ centers[goal]), kind="stable")]
        chosen, paths = [], set()
        for index in order:
            if records[int(index)]["path"] not in paths:
                chosen.append(int(index))
                paths.add(records[int(index)]["path"])
            if len(chosen) == 3:
                break
        refs, rgb, _ = read_representatives(records, chosen, max_steps)
        references.append({"goal_id": goal, "examples": refs})
        y = goal * (side + 24)
        draw.text((4, y + 3), f"goal {goal:02d}: nearest members from different train episodes", fill="black")
        for column, image in enumerate(rgb):
            examples.paste(Image.fromarray(image).resize((side, side), Image.Resampling.NEAREST), (4 + column * (side + 4), y + 21))
    examples.save(directory / "examples.png")
    baseline.write_json(directory / "gallery_references.json", references)


def build(args, report):
    import numpy as np
    import torch
    import goal_library as gl
    import long_horizon as lh

    config, shapes = resolve_baseline(args, report)
    selection = sample_records(args, config, report)
    spec = gl.encoder_spec(config, shapes, args.goal_dim, args.pool_size)
    checkpoint = lh.read_checkpoint(baseline.project_path(args.checkpoint))
    gl.require(not checkpoint.get("verification_artifact"), "不能用 T01 合成验收模型建立正式目标库")
    encoder = gl.encoder_from_checkpoint(checkpoint, spec).to(args.device)
    del checkpoint
    gc.collect()
    versions = {name: parameter._version for name, parameter in encoder.named_parameters()}
    raw = extract_features(encoder, selection, config, args, report)
    train = np.array([record["row"] for record in selection["records"] if record["split"] == "train"], dtype=np.int64)
    mean, components, pca_stats = gl.fit_pca(raw, train, args.goal_dim, args.seed, args.pca_iterations)
    features = gl.project_array(raw, mean, components)
    clustering = gl.fit_spherical_kmeans(features[train], args.goal_count, args.seed, args.restarts, args.cluster_iterations)
    labels = (features @ clustering["centers"].T).argmax(1)
    indices = train[clustering["representatives"]]
    references, images, heatmaps = read_representatives(selection["records"], indices, config["episode_max_steps"])
    # Projection buffers are fitted once; the CNN itself has not changed.
    encoder.mean.copy_(mean.to(encoder.mean.device))
    encoder.components.copy_(components.to(encoder.components.device))
    gl.require(versions == {name: parameter._version for name, parameter in encoder.named_parameters()}, "CNN 权重意外变化")
    metadata = {"task": config["task"], "max_steps": config["episode_max_steps"], "representatives": references,
                "pca": pca_stats, "fit_options": {"seed": args.seed, "pca_iterations": args.pca_iterations,
                "restarts": args.restarts, "cluster_iterations": args.cluster_iterations, "goal_count": args.goal_count},
                "feature_extraction": {"device": str(torch.device(args.device)), "batch_size": args.batch_size,
                                       "numeric_policy": gl.NUMERIC_POLICY},
                "checkpoint": baseline.file_signature(baseline.project_path(args.checkpoint)),
                "policy": "real observation anchors; no action/reward/RSSM/future input", "trained_worker": False}
    library = gl.GoalLibrary(encoder.cpu(), clustering["centers"], features[indices], images, heatmaps, metadata)
    artifact = report.directory / "goal_library.pt"
    payload = gl.save_library(library, artifact)
    np.savez_compressed(report.directory / "features.npz", features=features, labels=labels, train_indices=train,
                        representative_rows=indices, encoder_id=payload["encoder_id"], library_id=payload["library_id"])
    rows = coverage(features, labels, selection["records"], clustering["centers"])
    baseline.write_csv(report.directory / "coverage.csv", rows)
    counts = np.bincount(labels[train], minlength=args.goal_count)
    fractions = counts / counts.sum()
    diagnostics = {"pca": pca_stats, "train_cluster_cos_distance": clustering["loss"], "goal_count": args.goal_count,
                   "train_coverage": int((counts > 0).sum()), "validation_coverage": sum(row["validation_frames"] > 0 for row in rows),
                   "max_train_cluster_fraction": float(fractions.max()),
                   "normalized_train_entropy": float(-np.sum(fractions * np.log(np.maximum(fractions, 1e-12))) / np.log(args.goal_count)),
                   "effective_goals": float(np.exp(-np.sum(fractions * np.log(np.maximum(fractions, 1e-12))))),
                   "semantic_quality_accepted": False, "note": "statistics do not prove semantic usefulness or controllability"}
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    baseline.write_json(report.directory / "library_manifest.json", {"artifact": str(artifact), "encoder_id": payload["encoder_id"],
                                                                  "library_id": payload["library_id"], "encoder_spec": spec, "metadata": metadata})
    gallery(library, rows, selection["records"], features, labels, config["episode_max_steps"], report.directory)
    report.data.update(artifact=str(artifact), encoder_id=payload["encoder_id"], library_id=payload["library_id"], diagnostics=diagnostics)
    report.check("pca_train_only", "PASS", f"仅训练 episode 拟合 {spec['raw_dim']} -> {args.goal_dim} 维 PCA；保留方差 {pca_stats['retained_variance']:.3f}")
    report.check("real_anchors", "PASS", f"{args.goal_count} 类均有真实训练帧代表；保存 CNN、PCA、聚类中心及真实目标向量")
    if diagnostics["max_train_cluster_fraction"] > 0.4 or diagnostics["normalized_train_entropy"] < 0.7:
        report.check("cluster_balance", "WARN", "目标类别明显不均衡；请结合图集检查是否被背景或视角主导")
    if diagnostics["validation_coverage"] < args.goal_count:
        report.check("validation_coverage", "WARN", f"留出 episode 覆盖 {diagnostics['validation_coverage']}/{args.goal_count} 类；检查稀有类")
    # Actual serialization round-trip, not an in-memory state_dict alias.
    # Serialization is checked on the extraction device. Comparing a CUDA
    # cache to a CPU convolution is a separate cross-backend comparison.
    restored = gl.load_library(artifact, args.device)
    anchor_obs = {"image": images, "heatmap": heatmaps}
    actual_raw = restored.encoder.raw_features(anchor_obs)
    vector_difference(report, "anchor_raw_features", actual_raw, raw[indices])
    projected_cache = restored.encoder.project(np.array(raw[indices], copy=True))
    vector_difference(report, "anchor_cached_projection", projected_cache, features[indices])
    actual = restored.encoder.project(actual_raw)
    comparison = vector_difference(report, "roundtrip_features", actual, features[indices], check=True)
    gl.require(torch.equal(restored.assign(actual), torch.arange(args.goal_count, device=args.device)), "代表帧的类别编号没有保持一致")
    gl.require(restored.payload()["library_id"] == payload["library_id"], "保存/加载后的目标库 ID 改变")
    report.check("roundtrip", "PASS", f"保存/加载后真实代表帧特征与编号一致；设备 {args.device}；最大误差 {comparison['max_abs_error']:.3g}")
    report.check("manual_review", "WARN", "程序构建完成；请检查 representatives.png 和 examples.png，语义质量尚未验收")


def verify(args, report):
    import numpy as np
    import torch
    import goal_library as gl
    from types import SimpleNamespace

    artifact = baseline.project_path(args.library)
    library = gl.load_library(artifact, args.device)
    payload = library.payload()
    report.data.update(artifact=str(artifact), encoder_id=payload["encoder_id"], library_id=payload["library_id"])
    report.check("content_identity", "PASS", "结构、数值、CNN/PCA/目标库内容哈希匹配；编号无需重新聚类")
    second = gl.GoalLibrary.from_payload(payload, args.device)
    observation = {"image": library.images, "heatmap": library.heatmaps}
    features = library(observation)
    vector_difference(report, "representative_features", features, library.goals, check=True)
    gl.require(torch.equal(library.assign(features), torch.arange(len(library.goals), device=args.device)), "代表帧编号不一致")
    gl.require(torch.allclose(features, second(observation), atol=1e-6, rtol=1e-6), "两次独立加载的目标编码不同")
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        if library.encoder.mean.device.type == "cuda":
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
        ambient_matmul = torch.backends.cuda.matmul.allow_tf32
        ambient_precision = torch.get_float32_matmul_precision()
        ambient_cudnn = torch.backends.cudnn.allow_tf32
        with torch.autocast(device_type=library.encoder.mean.device.type, enabled=True):
            amp_features = library(observation)
            amp_labels = library.assign(amp_features)
        gl.require(amp_features.dtype == torch.float32 and torch.equal(features, amp_features),
                   "外层 AMP/TF32 改变了冻结目标特征")
        gl.require(torch.equal(amp_labels, library.assign(features)), "外层 AMP/TF32 改变了目标编号")
        gl.require(torch.backends.cuda.matmul.allow_tf32 == ambient_matmul and torch.backends.cudnn.allow_tf32 == ambient_cudnn and
                   torch.get_float32_matmul_precision() == ambient_precision, "冻结目标编码未恢复调用方精度设置")
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn
    report.check("numeric_policy", "PASS", "CNN/PCA/类别分配使用 FP32；外层 AMP/TF32 不改变目标特征与编号；调用方设置恢复")
    library.train(True)
    gl.require(not library.training and not any(parameter.requires_grad for parameter in library.parameters()), "目标库被意外改为可训练")
    extra = dict(observation, action=torch.randn(len(library.goals), 12, device=args.device),
                 reward=torch.randn(len(library.goals), device=args.device), obs_reward=torch.ones(len(library.goals), 1, device=args.device),
                 future_rssm=torch.randn(len(library.goals), 32, device=args.device), zoomed_image=torch.zeros_like(library.images))
    gl.require(torch.equal(features, library(extra)), "非观测字段改变了目标编码")
    rejection(report, "zoom_only_guard", lambda: library({"zoomed_image": library.images}), "真实 image")
    rejection(report, "goal_id_guard", lambda: library.goal(len(library.goals)), "越界")
    damaged = copy.deepcopy(payload)
    damaged["state_dict"]["encoder.cnn.layers.0.weight"].flatten()[0] += 0.01
    rejection(report, "content_guard", lambda: gl.GoalLibrary.from_payload(damaged), "哈希")
    report.check("observation_only_frozen", "PASS", "目标编码只依赖真实 RGB/heatmap；训练模式无法解冻；未来/动作/奖励/zoom 辅助字段不影响输出")
    # Check integration with T01 metadata without constructing a 235M agent.
    agent = SimpleNamespace(experiment_mode="goal_worker", _config=SimpleNamespace(goal_dim=library.goals.shape[1],
                            goal_count=len(library.goals), device=args.device, task=library.metadata["task"]),
                            _goal_library=None, _goal_encoder_id=None, _goal_library_id=None)
    gl.attach(agent, library)
    restored = gl.from_agent(agent)
    gl.require(restored.payload()["library_id"] == payload["library_id"], "agent 的目标库元数据加载改变了编号")
    # Same encoder with permuted goal numbers must be rejected independently
    # of the encoder ID, rather than silently swapping worker goal meanings.
    for key in ("centers", "goals", "images", "heatmaps"):
        value = getattr(second, key)
        value.copy_(value.flip(0))
    second.metadata["representatives"].reverse()
    gl.require(second.payload()["encoder_id"] == payload["encoder_id"], "交换目标编号意外改变编码器")
    rejection(report, "library_id_guard", lambda: gl.attach(agent, second), "目标库编号")
    report.check("agent_metadata", "PASS", "正式目标库可嵌入 T01 元数据并独立恢复；不新增原模型参数或优化器")

    directory = artifact.parent
    selection = json.loads((directory / "replay_selection.json").read_text(encoding="utf-8"))
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        cached = {key: data[key].copy() for key in data.files}
    gl.require(str(cached["encoder_id"]) == payload["encoder_id"] and str(cached["library_id"]) == payload["library_id"], "特征缓存不属于本目标库")
    records = selection["records"]
    expected_train = np.array([index for index, record in enumerate(records) if record["split"] == "train"], dtype=np.int64)
    gl.require(len(records) == len(cached["features"]) and np.array_equal(expected_train, cached["train_indices"]),
               "特征缓存的训练行与 episode 划分不一致")
    train_paths = {record["path"] for record in records if record["split"] == "train"}
    validation_paths = {record["path"] for record in records if record["split"] == "validation"}
    gl.require(train_paths and validation_paths and not train_paths & validation_paths, "训练/检查 episode 重叠或为空")
    cached_labels = library.assign(cached["features"]).cpu().numpy()
    gl.require(np.array_equal(cached_labels, cached["labels"]), "已保存特征的类别编号不稳定")
    report.check("episode_split", "PASS", f"训练 {len(train_paths)} 局、留出 {len(validation_paths)} 局，无重叠；缓存编号一致")
    if args.refit:
        raw = np.load(directory / "raw_features.npy", mmap_mode="r", allow_pickle=False)
        options = library.metadata["fit_options"]
        mean, components, _ = gl.fit_pca(raw, cached["train_indices"], library.goals.shape[1], options["seed"], options["pca_iterations"])
        gl.require(np.allclose(mean.numpy(), library.encoder.mean.cpu().numpy(), atol=1e-6, rtol=1e-6), "重新拟合 PCA 均值不一致")
        gl.require(np.allclose(components.numpy(), library.encoder.components.cpu().numpy(), atol=1e-4, rtol=1e-4), "相同数据/种子的 PCA 投影不能重现")
        projected = gl.project_array(raw, mean, components)
        fit = gl.fit_spherical_kmeans(projected[cached["train_indices"]], options["goal_count"], options["seed"], options["restarts"], options["cluster_iterations"])
        gl.require(np.allclose(fit["centers"], library.centers.cpu().numpy(), atol=1e-4, rtol=1e-4), "相同数据/种子的聚类中心不能重现")
        rebuilt = cached["train_indices"][fit["representatives"]]
        gl.require(np.array_equal(rebuilt, cached["representative_rows"]), "重新拟合改变了目标编号或代表帧")
        report.check("refit_reproducibility", "PASS", "相同缓存/种子重新拟合 PCA 与聚类，中心及代表帧编号一致（当前运行环境）")
    if args.replay_samples:
        rng = np.random.RandomState(0)
        indices = np.sort(rng.choice(len(records), min(args.replay_samples, len(records)), replace=False))
        max_error = 0.0
        for entry in selection["selected_episodes"]:
            gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "源 replay 已变更，无法对照原缓存")
            chosen = [int(index) for index in indices if records[int(index)]["path"] == entry["path"]]
            if not chosen:
                continue
            episode = gl.read_episode(entry["path"], library.metadata["max_steps"])
            positions = [records[index]["frame"] for index in chosen]
            observations = {key: episode[key][positions] for key in ("image", "heatmap")}
            actual_raw = library.encoder.raw_features(observations)
            actual = library.encoder.project(actual_raw).cpu().numpy()
            expected = cached["features"][chosen]
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            vector_difference(report, f"replay_features_{chosen[0]}", actual, expected, check=True)
        report.check("real_replay_reencode", "PASS", f"抽查 {len(indices)} 个真实帧，缓存与重编码一致；最大误差 {max_error:.3g}")
    report.check("scope", "PASS", "目标库工程验收；没有环境交互或策略训练，语义用途仍需检查图集")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--checkpoint", required=True)
    build_parser.add_argument("--baseline-dir", required=True)
    build_parser.add_argument("--replay-dir", action="append", default=[], help="默认 checkpoint 同级 train_eps；可重复指定")
    build_parser.add_argument("--max-episodes", type=int, default=128)
    build_parser.add_argument("--frames-per-episode", type=int, default=64)
    build_parser.add_argument("--stride", type=int, default=8)
    build_parser.add_argument("--validation-fraction", type=float, default=0.2)
    build_parser.add_argument("--goal-dim", type=int, default=128)
    build_parser.add_argument("--goal-count", type=int, default=16)
    build_parser.add_argument("--pool-size", type=int, default=2)
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--pca-iterations", type=int, default=2)
    build_parser.add_argument("--restarts", type=int, default=3)
    build_parser.add_argument("--cluster-iterations", type=int, default=100)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--library", required=True)
    verify_parser.add_argument("--refit", action="store_true", help="用缓存重新拟合降维与聚类，验证同种子重现")
    verify_parser.add_argument("--replay-samples", type=int, default=64)
    for entry in (build_parser, verify_parser):
        entry.add_argument("--device", default="cuda:0")
        entry.add_argument("--output-root", default="relevance_map/t02_outputs")
        entry.add_argument("--output-dir", help="可显式指定新的独立输出目录；拒绝覆盖")
    args = parser.parse_args()
    if args.command == "build":
        if (min(args.max_episodes, args.frames_per_episode, args.stride, args.goal_dim, args.pool_size, args.batch_size,
                args.restarts, args.cluster_iterations) < 1 or args.max_episodes < 5 or args.goal_count < 2 or
                args.pca_iterations < 0 or not 0 < args.validation_fraction < 0.5):
            parser.error("需要正采样/维度参数、至少 5 局、至少 2 类，以及 (0, 0.5) 的检查集比例")
        protected = [baseline.project_path(args.checkpoint).parent, baseline.project_path(args.baseline_dir)]
        protected += [baseline.project_path(value) for value in args.replay_dir]
    else:
        if args.replay_samples < 0:
            parser.error("--replay-samples 不能为负")
        protected = [baseline.project_path(args.library).parent]
    output_root = baseline.project_path(args.output_root)
    if any(output_root == path or path in output_root.parents for path in protected):
        parser.error("输出目录必须独立于 checkpoint、源 replay、T00 和目标库输入目录")
    directory = (baseline.project_path(args.output_dir) if args.output_dir else
                 output_root / (args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f")))
    if any(directory == path or path in directory.parents for path in protected):
        parser.error("指定输出目录不能位于源运行或目标库输入目录中")
    directory.mkdir(parents=True, exist_ok=False)
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    try:
        import torch
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用；请检查设备，或使用 --device cpu（仅目标 CNN，无完整 agent）")
        baseline.runtime_info(report)
        report.data["runtime"]["torch"] = torch.__version__
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        input_file = baseline.project_path(args.checkpoint if args.command == "build" else args.library)
        report.data["source_file_before"] = baseline.file_signature(input_file)
        if args.command == "build":
            build(args, report)
        else:
            verify(args, report)
    except (Exception, KeyboardInterrupt) as error:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.check("execution", "FAIL", f"{type(error).__name__}: {error}；见 error.txt")
    finally:
        before = report.data.get("source_file_before")
        if before:
            try:
                after = baseline.file_signature(input_file)
                report.data["source_file_after"] = after
                report.check("source_file_unchanged", "PASS" if before == after else "FAIL", "源文件大小/修改时间未变（非内容哈希）；输出在独立目录")
            except OSError as error:
                report.check("source_file_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
