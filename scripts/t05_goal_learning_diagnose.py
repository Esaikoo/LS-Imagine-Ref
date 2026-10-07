"""T05 offline goal-learning diagnosis using accepted T03/T04 artifacts.

No MineDojo/MineCLIP, optimizer, checkpoint write or training is performed.
Input gradients describe a frozen model at its current weights; they are
not optimizer updates or a measurement of historical training gradients.
"""

import argparse
import copy
from datetime import datetime
import gc
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t04_goal_bc as t04
import t05_goal_control as t05


class Report(baseline.Report):
    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_GOAL_LEARNING_DIAGNOSE; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def contract_probes(report):
    import numpy as np
    from types import SimpleNamespace
    import goal_learning_diagnostics as diag
    import goal_library as gl

    episodes = np.arange(4, dtype=np.int64)
    table = {"worker_state": np.repeat(episodes, 2), "worker_segment": np.arange(8, dtype=np.int64),
             "remaining": np.tile([1, 2], 4), "worker_train": np.repeat([True, True, False, False], 2),
             "goals": np.tile(np.eye(2, dtype=np.float32), (4, 1))}
    cache = SimpleNamespace(tables=table)
    rows = np.arange(8, dtype=np.int64)
    swapped = diag.alternate_rows(cache, rows, episodes, 0)
    gl.require(np.all(swapped >= 0) and np.array_equal(table["remaining"][rows], table["remaining"][swapped]) and
               np.array_equal(table["worker_train"][rows], table["worker_train"][swapped]) and
               np.all(episodes[table["worker_state"][rows]] != episodes[table["worker_state"][swapped]]),
               "目标替换混入其他 split、剩余步数或同局数据")
    cache.tables["worker_train"] = np.repeat([True, False, False, False], 2)
    gl.require(np.all(diag.alternate_rows(cache, rows[:2], episodes, 0) == -1), "没有跨局目标时必须标记不支持")
    items = [{"episode": i, "duration": 2, "split": "train" if i < 2 else "validation",
              "goal": np.array([1., 0.]), "actions": [i % 2, 0]} for i in range(4)]
    visual, state = np.tile([1., 0.], (4, 1)), np.tile([1., 0., 0.], (4, 1))
    pairs = diag.nearest_start_pairs(items, visual, state, 1)
    changed = copy.deepcopy(items)
    changed[1]["goal"], changed[1]["actions"] = np.array([0., 1.]), [0, 1]
    other = diag.nearest_start_pairs(changed, visual, state, 1)
    keys = lambda values: [(row["left_index"], row["right_index"], row["start_score"]) for row in values]
    gl.require(keys(pairs) == keys(other) and len(pairs) == 2 and all(not row["same_physical_state"] for row in pairs),
               "近邻选择依赖未来结果或混入其他 split")
    sample = diag.stratified_rows(rows, np.tile([1, 2], 4), 6, 0)
    gl.require(len(sample) == len(np.unique(sample)) == 6 and
               np.array_equal(sample, diag.stratified_rows(rows, np.tile([1, 2], 4), 6, 0)), "诊断采样重复或不可复现")
    report.check("diagnostic_contracts", "PASS", "内存探针确认目标替换保持 split/步数且跨局；近邻选择不看未来；采样可复现。合成数据不写入产物")


def map_and_validate(cache, dataset, report):
    import numpy as np
    import goal_learning_diagnostics as diag
    import goal_library as gl

    episodes, frames = diag.state_index(cache)
    t, a = cache.tables, dataset.arrays
    segments = t["worker_segment"]
    gl.require(np.array_equal(episodes[t["worker_state"]], a["episode_index"][segments]) and
               np.array_equal(frames[t["worker_state"]], a["end"][segments] - t["remaining"]) and
               np.all(frames[t["worker_state"]] >= a["start"][segments]), "监督表的当前帧/剩余步数不匹配")
    gl.require(np.array_equal(episodes[t["candidate_state"]], a["episode_index"]) and
               np.array_equal(frames[t["candidate_state"]], a["start"]) and
               np.array_equal(t["goals"], a["goals"]), "真实目标或片段起点与 T03 不同")
    split_flags = np.array([entry["split"] == "train" for entry in dataset.metadata["episodes"]])
    gl.require(np.array_equal(t["worker_train"], split_flags[episodes[t["worker_state"]]]) and
               np.array_equal(t["candidate_train"], split_flags[a["episode_index"]]), "缓存与整局留出划分不同")
    report.check("real_supervision_mapping", "PASS", "全量当前帧/真实终点/剩余步数/episode 划分匹配；沿用已验收因果状态，不重算 WM")
    return episodes, frames


def load_no_goal(args, cache, model, report):
    import goal_bc as bc
    import goal_library as gl
    import long_horizon as lh
    import torch

    if not args.no_goal_checkpoint:
        report.check("independent_no_goal", "WARN", "本次没有独立无目标 BC；先完成只读诊断，必要时再补相同更新预算的对照")
        return None
    path = baseline.project_path(args.no_goal_checkpoint)
    payload = t04.training_checkpoint(path)
    bc.validate_checkpoint_sampling(payload)
    verified = t05.read_json(baseline.project_path(args.no_goal_verify_dir) / "report.json")
    t05.accepted(verified, "verify")
    step = report.data["input_identity"]["counters"]["step"]
    gl.require(not payload.get("verification_artifact") and payload.get("stage") == "t04_offline_bc" and
               payload.get("experiment_mode") == "goal_worker" and payload.get("worker_architecture") == "expanded_actor_v1" and
               payload.get("cache_id") == cache.cache_id and payload["frozen_bundle"]["bundle_id"] == cache.bundle["bundle_id"],
               "独立无目标对照结构、来源或缓存不同")
    gl.require(payload["options"]["conditioning"] == "no_goal" and
               bc.normalize_training_options(dict(payload["options"], conditioning="goal")) ==
               bc.normalize_training_options(model.options) and
               payload["counters"] == report.data["input_identity"]["counters"] and
               verified.get("cache_id") == cache.cache_id and verified.get("bundle_id") == cache.bundle["bundle_id"] and
               verified.get("counters") == payload["counters"] and
               verified.get("source_inputs_before", {}).get(str(path)) == baseline.file_signature(path),
               "独立 no_goal 需要同一数据/初始化/配置/累计更新数及其独立 verify")
    # Reuse main model's frozen dependencies, loading only ~37 MiB worker
    # weights for the comparison rather than another entire frozen WM.
    worker = lh.GoalWorker(model.original_actor, model.worker.goal_dim, model.worker.horizon).to(args.device)
    lh.require_compatible(worker.state_dict(), payload["worker"])
    gl.require(all(bool(torch.isfinite(value).all()) for value in payload["worker"].values()), "no_goal worker 权重非有限")
    worker.load_state_dict(payload["worker"], strict=True)
    worker.requires_grad_(False).eval()
    report.data["no_goal_identity"] = {"checkpoint": baseline.file_signature(path), "sha256": bc.file_hash(path),
        "worker_hash": gl.tensor_digest(worker.state_dict()), "step": step, "selection": payload.get("selection"),
        "comparison_rule": "matched update count; no inference goal input; inspect selection metadata separately"}
    report.check("independent_no_goal", "PASS", f"严格加载同配置/预算的独立无目标低层，step={step}；只比较同一组诊断状态，不恢复优化器")
    del payload
    gc.collect()
    return worker


def query_model(args, report, cache, model, episodes, frames, no_goal):
    import numpy as np
    import torch
    import goal_learning_diagnostics as diag
    import goal_library as gl

    t = cache.tables
    rows = diag.probe_rows(cache, episodes, args.train_per_episode, args.validation_per_episode, args.seed)
    donors = diag.alternate_rows(cache, rows, episodes, (args.seed + 17) % 2**32, args.donor_candidates)
    records, probability_blocks = [], []
    for start in range(0, len(rows), args.batch_size):
        chosen, alternate = rows[start:start + args.batch_size], donors[start:start + args.batch_size]
        features, goals, remaining, actions = cache.worker_batch(chosen, args.device)
        swaps = t["goals"][t["worker_segment"][np.maximum(alternate, 0)]].copy()
        swaps[alternate < 0] = goals.detach().cpu().numpy()[alternate < 0]
        with torch.no_grad(), gl.encoder_precision(args.device):
            distributions = {"correct": model.action_dist(features, goals, remaining),
                "zero": model.action_dist(features, goals, remaining, "zero_goal"),
                "swap": model.action_dist(features, torch.as_tensor(swaps, device=args.device), remaining),
                "original": model.action_dist(features, goals, remaining, "original")}
            if no_goal is not None:
                distributions["no_goal_bc"] = no_goal(features, torch.zeros_like(goals), remaining)
            probabilities = {name: dist.probs.detach().cpu().numpy() for name, dist in distributions.items()}
        gl.require(all(np.isfinite(value).all() and np.all(value >= 0) and np.allclose(value.sum(1), 1, atol=1e-5)
                       for value in probabilities.values()), "诊断动作分布异常")
        labels = actions.cpu().numpy()
        nll = {name: -np.log(value[np.arange(len(chosen)), labels].clip(1e-30)) for name, value in probabilities.items()}
        modes = {name: value.argmax(1) for name, value in probabilities.items()}
        for local, row in enumerate(chosen):
            state, segment = int(t["worker_state"][row]), int(t["worker_segment"][row])
            record = {"worker_row": int(row), "state_row": state, "segment": segment,
                "episode": int(episodes[state]), "frame": int(frames[state]), "remaining": int(t["remaining"][row]),
                "split": "train" if t["worker_train"][row] else "validation", "label_action": int(labels[local]),
                "alternate_row": int(alternate[local]), "alternate_episode": None,
                "correct_nll": float(nll["correct"][local]), "zero_nll": float(nll["zero"][local]),
                "original_nll": float(nll["original"][local]), "zero_minus_correct_nll": float(nll["zero"][local] - nll["correct"][local]),
                "correct_action": int(modes["correct"][local]), "zero_action": int(modes["zero"][local]),
                "correct_vs_zero_l1": float(np.abs(probabilities["correct"][local] - probabilities["zero"][local]).sum()),
                "correct_vs_zero_mode_changed": bool(modes["correct"][local] != modes["zero"][local]),
                "swap_nll": None, "swap_minus_correct_nll": None, "correct_vs_swap_l1": None,
                "correct_vs_swap_mode_changed": None, "swap_goal_cosine_distance": None}
            if alternate[local] >= 0:
                record.update(alternate_episode=int(episodes[t["worker_state"][alternate[local]]]),
                    swap_nll=float(nll["swap"][local]), swap_minus_correct_nll=float(nll["swap"][local] - nll["correct"][local]),
                    correct_vs_swap_l1=float(np.abs(probabilities["correct"][local] - probabilities["swap"][local]).sum()),
                    correct_vs_swap_mode_changed=bool(modes["correct"][local] != modes["swap"][local]),
                    swap_goal_cosine_distance=float(np.clip(1 - np.dot(t["goals"][segment], swaps[local]), 0, 2)))
            if no_goal is not None:
                record.update(no_goal_bc_nll=float(nll["no_goal_bc"][local]),
                    no_goal_bc_minus_correct_nll=float(nll["no_goal_bc"][local] - nll["correct"][local]),
                    correct_vs_no_goal_bc_l1=float(np.abs(probabilities["correct"][local] - probabilities["no_goal_bc"][local]).sum()))
            records.append(record)
        probability_blocks.append(probabilities)
    raw = {name: np.concatenate([block[name] for block in probability_blocks]) for name in probability_blocks[0]}
    raw.update(worker_rows=rows, alternate_rows=donors)
    np.savez_compressed(report.directory / "probe_probabilities.npz", **raw)
    baseline.write_csv(report.directory / "probe_rows.csv", records)
    summary = diag.summarize_probes(records)
    baseline.write_json(report.directory / "goal_sensitivity.json", summary)
    for split in ("train", "validation"):
        stats = summary[split]["row_weighted"]
        changed = stats["correct_vs_swap_mode_changed"]["mean"]
        print(f"[GOAL SENSITIVITY] split={split} episodes={stats['episodes']} queries={stats['rows']} "
              f"swap_mode_change={changed} zero_nll_gap={stats['zero_minus_correct_nll']['mean']:.6f}", flush=True)
        if no_goal is not None:
            print(f"[NO GOAL BC] split={split} correct_nll={stats['correct_nll']['mean']:.6f} "
                  f"independent_no_goal_nll={stats['no_goal_bc_nll']['mean']:.6f} "
                  f"no_goal_minus_correct_nll={stats['no_goal_bc_minus_correct_nll']['mean']:.6f}", flush=True)
    report.check("heldout_goal_queries", "PASS", f"{len(rows)} 个去重真实查询；替换目标同 split/相同步数/不同 episode；按局与按条目分别汇总，不训练替换标签")
    return rows, summary


def input_path_probe(args, report, cache, model, rows):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import goal_learning_diagnostics as diag
    import goal_library as gl

    validation = rows[~cache.tables["worker_train"][rows]]
    positions = np.linspace(0, len(validation) - 1, min(args.gradient_rows, len(validation)), dtype=np.int64)
    validation = validation[positions]
    gl.require(len(validation) > 0 and not model.worker.actor._symlog_inputs, "输入贡献探针需要留出条目及原非 symlog actor")
    features, goals, remaining, actions = cache.worker_batch(validation, args.device)
    features, goals = features.detach().requires_grad_(True), goals.detach().requires_grad_(True)
    layer, captured = model.worker.actor.layers.Actor_linear0, []
    hook = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.enable_grad(), gl.encoder_precision(args.device):
            distribution = model.action_dist(features, goals, remaining)
            loss = -distribution.log_prob(F.one_hot(actions, cache.metadata["action_dim"]).float()).mean()
            feature_grad, goal_grad, output_grad = torch.autograd.grad(loss, (features, goals, captured[0]))
    finally:
        hook.remove()
    with torch.no_grad(), gl.encoder_precision(args.device):
        width = model.worker.feature_dim
        state_part = F.linear(features, layer.weight[:, :width])
        goal_part = F.linear(goals, layer.weight[:, width:width + model.worker.goal_dim])
        time_input = remaining.float().unsqueeze(-1) / model.worker.horizon
        time_part = F.linear(time_input, layer.weight[:, -1:])
        reconstructed = state_part + goal_part + time_part
        if layer.bias is not None:
            reconstructed = reconstructed + layer.bias
        error = float((reconstructed - captured[0]).abs().max())
        reconstruction_scale = max(1.0, float(captured[0].abs().max()))
        gl.require(error <= 3e-5 * reconstruction_scale, "第一层输入贡献分解与真实前向不一致")
        # Analytical dLoss/dW at the CURRENT frozen weights. No Parameter
        # requires_grad flag, .grad slot or optimizer is touched.
        state_weight_grad = output_grad.T @ features
        goal_weight_grad = output_grad.T @ goals
        rms = lambda value: float(value.detach().square().mean().sqrt())
        result = {"queries": len(validation), "loss": float(loss.detach()), "first_layer_reconstruction_max_error": error,
            "first_layer_reconstruction_scale": reconstruction_scale, "first_layer_reconstruction_relative_limit": 3e-5,
            "state_input_rms": rms(features), "goal_input_rms": rms(goals),
            "state_weight_rms": rms(layer.weight[:, :width]), "goal_weight_rms": rms(layer.weight[:, width:width + model.worker.goal_dim]),
            "state_contribution_rms": rms(state_part), "goal_contribution_rms": rms(goal_part), "remaining_contribution_rms": rms(time_part),
            "goal_to_state_contribution_rms_ratio": rms(goal_part) / max(rms(state_part), 1e-30),
            "loss_gradient_state_input_rms": rms(feature_grad), "loss_gradient_goal_input_rms": rms(goal_grad),
            "hypothetical_first_layer_state_weight_gradient_rms": rms(state_weight_grad),
            "hypothetical_first_layer_goal_weight_gradient_rms": rms(goal_weight_grad),
            "goal_contribution_per_query_rms": diag.describe(goal_part.square().mean(1).sqrt().cpu().numpy()),
            "scope": "current held-out BC loss and pre-LayerNorm contribution, not historical gradients or optimal scale; no model update"}
    gl.require(all(np.isfinite(value) for value in result.values() if isinstance(value, (int, float))), "输入/梯度探针非有限")
    gl.require(all(parameter.grad is None for parameter in model.parameters()), "诊断写入了参数梯度")
    baseline.write_json(report.directory / "input_path.json", result)
    report.check("frozen_input_gradient_probe", "PASS", "仅局部输入求导及解析第一层梯度；真实前向分解一致；没有参数 .grad、优化器或权重更新")
    print(f"[INPUT PATH] goal/state_contribution_ratio={result['goal_to_state_contribution_rms_ratio']:.6g} "
          f"goal_input_gradient_rms={result['loss_gradient_goal_input_rms']:.6g}", flush=True)
    return result


def real_branch_candidates(args, report, cache, model, dataset, episodes, frames):
    import numpy as np
    import torch
    import goal_learning_diagnostics as diag
    import goal_library as gl

    a, t, items, descriptors, state_rows, images = dataset.arrays, cache.tables, [], [], [], []
    durations = a["end"] - a["start"]
    for index, entry in enumerate(dataset.metadata["episodes"]):
        ep = dataset.episode(index)
        worker = np.flatnonzero(episodes[t["worker_state"]] == index)
        current = frames[t["worker_state"][worker]]
        gl.require(np.array_equal(t["action_id"][worker], ep["action"][current + 1].argmax(-1)), "真实回放的下一动作与缓存不同")
        available = np.flatnonzero(a["episode_index"] == index)
        chosen = diag.stratified_rows(available, durations, args.segments_per_episode, (args.seed + index * 1013) % 2**32)
        for segment in chosen:
            start, end = int(a["start"][segment]), int(a["end"][segment])
            obs = {"image": torch.as_tensor(ep["image"][start:start + 1], device=args.device),
                   "heatmap": torch.as_tensor(ep["heatmap"][start:start + 1], device=args.device)}
            descriptor = model.library(obs)[0].cpu().numpy()
            items.append({"segment": int(segment), "episode": index, "split": entry["split"], "start": start,
                "end": end, "duration": end - start, "goal": a["goals"][segment].tolist(),
                "actions": ep["action"][start + 1:end + 1].argmax(-1).tolist()})
            descriptors.append(descriptor)
            state_rows.append(int(t["candidate_state"][segment]))
            images.append((ep["image"][start].copy(), ep["image"][end].copy()))
        if (index + 1) % 16 == 0 or index + 1 == len(dataset.metadata["episodes"]):
            print(f"[BRANCH CANDIDATES] episodes={index + 1}/{len(dataset.metadata['episodes'])} real_segments={len(items)}", flush=True)
    states = np.array(cache.states[np.asarray(state_rows)], copy=True)
    pairs = diag.nearest_start_pairs(items, np.asarray(descriptors), states, args.neighbours)
    del states
    for pair in pairs:
        left, right = items[pair["left_index"]], items[pair["right_index"]]
        li, ri = pair["left_index"], pair["right_index"]
        pair.update(left_segment=left["segment"], right_segment=right["segment"],
            left_episode=left["episode"], right_episode=right["episode"],
            start_rgb_mae=float(np.abs(images[li][0].astype(float) - images[ri][0].astype(float)).mean()))
    baseline.write_csv(report.directory / "branch_pairs.csv", pairs)
    baseline.write_json(report.directory / "branch_segments.json", {"segments": items,
        "selection": "duration-stratified, bounded per episode; never selected by future action, goal distance or model outcome"})
    summary = diag.summarize_pairs(pairs)
    baseline.write_json(report.directory / "branch_coverage.json", summary)
    gallery(args, report.directory, pairs, items, images)
    report.check("real_action_labels", "PASS", "全量缓存下一动作与原真实回放一致；近邻按起点、同 split/相同持续时间/不同局选取")
    report.check("branch_candidates", "PASS", f"{len(items)} 个真实片段、{len(pairs)} 个去重近邻候选；保存原 RGB/终点/动作与距离，不宣称同一物理状态")
    return summary


def gallery(args, directory, pairs, items, images):
    from PIL import Image, ImageDraw

    for split in ("train", "validation"):
        # Display short and long durations rather than letting one-step
        # neighbours occupy the entire gallery. Within each duration band
        # retain the nearest start candidates, without inspecting outcomes.
        rows = [row for row in pairs if row["split"] == split]
        bands = [[row for row in rows if low <= row["duration"] <= high]
                 for low, high in ((1, 4), (5, 8), (9, 12), (13, 16))]
        selected, cursor = [], 0
        while len(selected) < args.gallery_pairs and any(len(band) > cursor for band in bands):
            for band in bands:
                if len(band) > cursor and len(selected) < args.gallery_pairs:
                    selected.append(band[cursor])
            cursor += 1
        if not selected:
            continue
        canvas = Image.new("RGB", (800, len(selected) * 190), "white")
        draw = ImageDraw.Draw(canvas)
        for index, pair in enumerate(selected):
            top = index * 190
            left, right = items[pair["left_index"]], items[pair["right_index"]]
            draw.text((4, top + 2), f"{split} | episodes {left['episode']}/{right['episode']} | duration {pair['duration']} | "
                      f"visual {pair['start_visual_cosine_distance']:.3f} state {pair['start_state_cosine_distance']:.3f} "
                      f"goal {pair['goal_cosine_distance']:.3f} | first divergence {pair['first_action_divergence_step']}", fill="black")
            for column, (item_index, image_index, label) in enumerate(((pair["left_index"], 0, "A start"),
                    (pair["right_index"], 0, "B start"), (pair["left_index"], 1, "A real endpoint"), (pair["right_index"], 1, "B real endpoint"))):
                x = column * 196 + 4
                draw.text((x, top + 19), label, fill="black")
                resampling = getattr(Image, "Resampling", Image)
                canvas.paste(Image.fromarray(images[item_index][image_index]).resize((128, 128), resampling.NEAREST), (x, top + 35))
            draw.text((4, top + 166), f"A actions: {left['actions']} | B actions: {right['actions']}", fill="black")
        canvas.save(directory / f"branch_examples_{split}.png")


def analyze(args, report, cache, model):
    import goal_bc as bc
    import goal_learning_diagnostics as diag
    import goal_library as gl

    contract_probes(report)
    dataset, _ = t04.accepted_dataset(Path(cache.metadata["dataset_dir"]), Path(cache.metadata["t03_verify_dir"]))
    gl.require(dataset.content_id == cache.metadata["dataset_id"], "T03 与缓存身份不同")
    episodes, frames = map_and_validate(cache, dataset, report)
    redundancy = diag.supervision_redundancy(cache, episodes)
    baseline.write_json(report.directory / "supervision_structure.json", redundancy)
    no_goal = load_no_goal(args, cache, model, report)
    rows, sensitivity = query_model(args, report, cache, model, episodes, frames, no_goal)
    contributions = input_path_probe(args, report, cache, model, rows)
    coverage = real_branch_candidates(args, report, cache, model, dataset, episodes, frames)
    if no_goal is not None:
        gl.require(gl.tensor_digest(no_goal.state_dict()) == report.data["no_goal_identity"]["worker_hash"], "诊断更新了无目标对照")
    diagnostics = {"format": diag.FORMAT, "cache_id": cache.cache_id, "model_id": report.data["model_id"],
        "goal_sensitivity": sensitivity, "supervision_structure": redundancy,
        "input_path": contributions, "branch_coverage": coverage,
        "new_env_steps": 0, "optimizer_updates": 0, "controlled_reachability_verified": False,
        "scope": "T05 mechanism diagnosis only; T06 remains pending; no fixed sensitivity acceptance threshold"}
    baseline.write_json(report.directory / "diagnostics.json", diagnostics)
    report.data.update(diagnostic_format=diag.FORMAT, query_counts={key: sensitivity[key]["row_weighted"]["rows"]
        for key in ("train", "validation")}, diagnostic_code={name: bc.file_hash(ROOT / name)
        for name in ("goal_learning_diagnostics.py", "scripts/t05_goal_learning_diagnose.py")})
    report.check("scope", "WARN", "工程 PASS 不代表目标控制有效；相似历史不是同一物理状态；梯度只描述当前模型。暂缓旧30 trial及T06")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True).add_parser("analyze")
    command.add_argument("--cache-dir", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--t04-verify-dir", required=True)
    command.add_argument("--no-goal-checkpoint")
    command.add_argument("--no-goal-verify-dir")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--output-dir")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--train-per-episode", type=int, default=8)
    command.add_argument("--validation-per-episode", type=int, default=32)
    command.add_argument("--segments-per-episode", type=int, default=16)
    command.add_argument("--donor-candidates", type=int, default=8)
    command.add_argument("--neighbours", type=int, default=2)
    command.add_argument("--gradient-rows", type=int, default=64)
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--gallery-pairs", type=int, default=8)
    args = parser.parse_args()
    if bool(args.no_goal_checkpoint) != bool(args.no_goal_verify_dir):
        parser.error("no-goal-checkpoint 与 no-goal-verify-dir 必须一起提供")
    if not 0 <= args.seed < 2**32 or min(args.train_per_episode, args.validation_per_episode,
            args.segments_per_episode, args.donor_candidates, args.neighbours, args.gradient_rows,
            args.batch_size, args.gallery_pairs) < 1:
        parser.error("诊断计数需为正数，种子需合法非负整数")
    try:
        protected, files, source = t05.source_paths(baseline.project_path(args.cache_dir), args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"输入清单异常：{error}")
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / ("goal_learning_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        parser.error("输出必须独立于原模型、回放、缓存和输入产物")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng, model = None, None, None
    try:
        import torch
        import goal_bc as bc
        import goal_control as ctl
        import goal_library as gl

        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA 不可用")
            torch.cuda.set_device(device)
        before = {str(path): baseline.file_signature(path) for path in files}
        report.data["source_inputs_before"] = before
        report.save()
        report.require_writable()
        for entry in source["episodes"]:
            gl.require(baseline.file_signature(Path(entry["path"])) == entry["signature"], "原回放大小/修改时间改变")
        gl.require(baseline.file_signature(Path(source["checkpoint"]["path"])) == source["checkpoint"], "原初始化 checkpoint 改变")
        rng = bc.capture_rng(args.device)
        loader_args = copy.copy(args)
        loader_args.command = "check"  # Require the exact independent T04 verify record.
        cache, model = t05.load_inputs(loader_args, report)
        versions = {id(parameter): parameter._version for parameter in model.parameters()}
        identity = report.data["model_id"]
        analyze(args, report, cache, model)
        gl.require(versions == {id(parameter): parameter._version for parameter in model.parameters()} and
                   identity == ctl.model_identity(model) and
                   not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()),
                   "只读诊断改变了权重、训练标志或参数梯度")
        report.check("no_training", "PASS", "optimizer_updates=0；new_env_steps=0；权重/参数版本/冻结标志未变；无 MineDojo/MineCLIP")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {str(path): baseline.file_signature(path) for path in files}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原模型、源回放和验收产物大小/修改时间未变；只写独立诊断目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        del model
        gc.collect()
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
