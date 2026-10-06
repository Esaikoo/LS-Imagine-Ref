"""Inspect existing T05 reference pixels; no model, GPU or environment.

This command exports diagnostic images and statistics from already saved
trajectories. A successful export does not approve the failed benchmark.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

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
        print(f"[{'FAIL' if failed else 'PASS'}] T05_PREFIX_DIAGNOSE; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_reference(directory, frames):
    import numpy as np

    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        keys = ("image", "heatmap", "features", "obs_reward", "action", "is_first", "is_last", "is_terminal")
        require(set(keys) <= set(data.files), "参考轨迹缺少真实观测、状态或动作字段")
        arrays = {key: data[key][:frames].copy() for key in keys}
    events = read_json(directory / "events.json")[:frames]
    require(all(len(value) == frames for value in arrays.values()) and len(events) == frames,
            "参考轨迹不足指定 prefix 长度，请核对 --prefix-steps")
    require(arrays["image"].dtype == np.uint8 and arrays["image"].shape == (frames, 64, 64, 3) and
            arrays["heatmap"].dtype == np.uint8 and arrays["heatmap"].shape == (frames, 64, 64),
            "需要 T05 保存的真实 uint8 RGB/heatmap")
    require(arrays["features"].ndim == 2 and np.isfinite(arrays["features"]).all() and
            np.isfinite(arrays["obs_reward"]).all() and arrays["action"].ndim == 2 and
            np.isfinite(arrays["action"]).all(), "状态、奖励或动作数组异常")
    require(bool(arrays["is_first"][0]) and not arrays["is_first"][1:].any() and
            np.all(arrays["action"][0] == 0) and
            all(event.get("frame") == index for index, event in enumerate(events)), "参考轨迹 reset/动作/事件索引异常")
    return arrays, events


def frame_statistics(left, right, events_left, events_right):
    import numpy as np

    rows = []
    for frame in range(len(left["image"])):
        a, b = left["image"][frame].astype(np.float64), right["image"][frame].astype(np.float64)
        signed, absolute = a - b, np.abs(a - b)
        bias = signed.mean(axis=(0, 1))
        # This correction is diagnostic only. Neither pixels nor WM inputs
        # are changed, and these numbers never decide pairing acceptance.
        residual = np.abs(signed - bias[None, None, :])
        state_a, state_b = left["features"][frame], right["features"][frame]
        event_a, event_b = events_left[frame], events_right[frame]
        pose_a = event_a.get("telemetry", {}).get("pose")
        pose_b = event_b.get("telemetry", {}).get("pose")
        position = yaw = pitch = None
        if pose_a is not None and pose_b is not None:
            position = float(np.linalg.norm([pose_a[key] - pose_b[key] for key in ("x", "y", "z")]))
            yaw = float(abs((pose_a["yaw"] - pose_b["yaw"] + 180) % 360 - 180))
            pitch = float(abs(pose_a["pitch"] - pose_b["pitch"]))
        row = {"frame": frame, "rgb_mae": float(absolute.mean()), "rgb_p99": float(np.percentile(absolute, 99)),
               "rgb_max": float(absolute.max()), "pixel_fraction_over_8": float((absolute.max(axis=-1) > 8).mean()),
               "signed_channel_bias_r": float(bias[0]), "signed_channel_bias_g": float(bias[1]), "signed_channel_bias_b": float(bias[2]),
               "channel_bias_removed_mae": float(residual.mean()),
               "top_third_rgb_mae": float(absolute[:21].mean()), "middle_third_rgb_mae": float(absolute[21:43].mean()),
               "bottom_third_rgb_mae": float(absolute[43:].mean()),
               "reference_0_rgb_mean": float(a.mean()), "reference_1_rgb_mean": float(b.mean()),
               "heatmap_mae": float(np.abs(left["heatmap"][frame].astype(float) - right["heatmap"][frame].astype(float)).mean()),
               "state_relative_l2": float(np.linalg.norm(state_a - state_b) / max(float(np.linalg.norm(state_a)), 1e-8)),
               "reward_difference": float(np.abs(left["obs_reward"][frame] - right["obs_reward"][frame]).max()),
               "position_difference": position, "yaw_difference": yaw, "pitch_difference": pitch,
               "incoming_actions_equal": bool(np.array_equal(left["action"][frame], right["action"][frame])),
               "native_actions_equal": event_a.get("native_actions") == event_b.get("native_actions") if
                   "native_actions" in event_a and "native_actions" in event_b else None}
        rows.append(row)
    return rows


def export_images(left, right, rows, directory):
    import numpy as np
    from PIL import Image, ImageDraw

    frame_dir = directory / "frames"
    frame_dir.mkdir()
    peak_mae = max(rows, key=lambda row: row["rgb_mae"])["frame"]
    peak_p99 = max(rows, key=lambda row: row["rgb_p99"])["frame"]
    selected = sorted({frame for frame in (0, 1, 2, 4, 7, 8, peak_mae, peak_p99, len(rows) - 1) if frame < len(rows)})
    gallery = Image.new("RGB", (640, 160 * len(selected)), "white")
    for frame, stats in enumerate(rows):
        tile = Image.new("RGB", (640, 160), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((4, 2), f"frame {frame} | MAE {stats['rgb_mae']:.3f} | P99 {stats['rgb_p99']:.1f} | state L2 {stats['state_relative_l2']:.4f}", fill="black")
        diff = np.abs(left["image"][frame].astype(np.int16) - right["image"][frame].astype(np.int16))
        panels = (left["image"][frame], right["image"][frame], np.clip(diff * 4, 0, 255).astype(np.uint8),
                  left["heatmap"][frame], right["heatmap"][frame])
        labels = ("RGB ref 0", "RGB ref 1", "abs diff x4", "heatmap ref 0", "heatmap ref 1")
        for column, (panel, label) in enumerate(zip(panels, labels)):
            draw.text((column * 128 + 3, 17), label, fill="black")
            tile.paste(Image.fromarray(panel).convert("RGB").resize((128, 128), Image.Resampling.NEAREST
                       if hasattr(Image, "Resampling") else Image.NEAREST), (column * 128, 32))
        tile.save(frame_dir / f"frame_{frame:03d}.png")
        if frame in selected:
            gallery.paste(tile, (0, selected.index(frame) * 160))
    gallery.save(directory / "prefix_pairs.png")
    return selected


def diagnose(args, report, source):
    import numpy as np

    case = source / f"case_{args.case_seed}"
    original_report = read_json(source / "report.json")
    require(original_report.get("command") == "prepare", "输入目录必须是 T05 prepare 的输出")
    original_steps = original_report["arguments"]["prefix_steps"]
    steps = args.prefix_steps if args.prefix_steps is not None else original_steps
    require(type(original_steps) is int and type(steps) is int and 0 <= steps <= original_steps,
            "prefix-steps 必须在原 prepare 的共同前缀范围内，不能包含分支后的目标执行")
    left, events_left = read_reference(case / "reference_0", steps + 1)
    right, events_right = read_reference(case / "reference_1", steps + 1)
    require(left["features"].shape == right["features"].shape and left["action"].shape == right["action"].shape,
            "两参考分支状态或动作维度不同")
    report.check("saved_references", "PASS", f"读取已有参考分支的 {steps + 1} 帧；未启动环境或加载模型")
    rows = frame_statistics(left, right, events_left, events_right)
    comparison = read_json(case / "prefix_comparison.json") if (case / "prefix_comparison.json").is_file() else None
    if comparison is not None and "per_frame" in comparison:
        recorded = comparison["per_frame"][:steps + 1]
        require(len(recorded) == len(rows), "原逐帧配对报告与参考前缀长度不一致")
        for row, previous in zip(rows, recorded):
            require(row["frame"] == previous["frame"] and all(
                np.isclose(row[key], previous[key], atol=1e-6, rtol=1e-6)
                for key in ("rgb_mae", "rgb_p99", "heatmap_mae", "state_relative_l2")),
                "导出画面统计与原配对报告不同，请核对目录是否来自同一次 prepare")
        report.check("recorded_statistics", "PASS", "已有 RGB/heatmap/状态误差与原逐帧配对报告一致")
    selected = export_images(left, right, rows, report.directory)
    scenario = read_json(case / "scenario.json") if (case / "scenario.json").is_file() else None
    diagnostics = {"source_prepare": str(source), "source_prepare_status": original_report.get("status"),
                   "case_seed": args.case_seed, "scenario": scenario, "prefix_steps": steps,
                   "original_pair_comparison": comparison, "selected_gallery_frames": selected,
                   "per_frame": rows, "model_loaded": False, "new_env_steps": 0, "optimizer_updates": 0,
                   "approved_benchmark": False,
                   "interpretation": ["亮度、分区和差异面积仅帮助定位画面来源，不是配对通过条件。",
                       "看左右RGB是否同一地形；差异在大面积颜色还是局部对象/天空/手部。",
                       "仅凭误差下降不能确认初始光照原因；也不能据此放宽RSSM或像素阈值。",
                       "导出成功不代表T05 prepare、环境复现或目标控制验收通过。"]}
    baseline.write_json(report.directory / "prefix_diagnostics.json", diagnostics)
    baseline.write_csv(report.directory / "frame_statistics.csv", rows)
    report.data.update(model_loaded=False, new_env_steps=0, optimizer_updates=0, approved_benchmark=False,
                       source_prepare_status=original_report.get("status"), selected_gallery_frames=selected)
    report.check("artifacts", "PASS", "保存 prefix_pairs.png、每帧PNG、分区/颜色偏差统计及原配对报告")
    report.check("scope", "WARN", "仅离线定位现有失败参考；不改变图像/模型输入，不批准配对或目标控制结论")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", required=True)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, help="默认读取原 prepare 参数")
    parser.add_argument("--output-dir", help="新的独立目录，拒绝覆盖")
    parser.set_defaults(command="diagnose")
    args = parser.parse_args()
    source = baseline.project_path(args.prepare_dir)
    case = source / f"case_{args.case_seed}"
    files = [source / "report.json"] + [case / f"reference_{branch}" / name for branch in (0, 1)
                                          for name in ("trajectory.npz", "events.json")]
    files += [case / name for name in ("scenario.json", "prefix_comparison.json") if (case / name).is_file()]
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / ("diagnose_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    try:
        original_report = read_json(source / "report.json")
        protected = [source] + [Path(path).parent for path in original_report.get("source_inputs_before", {})]
        require(not any(directory == path or path in directory.parents or directory in path.parents for path in protected),
                "诊断输出必须独立于原 prepare、checkpoint、缓存和回放目录")
        before = {str(path): baseline.file_signature(path) for path in files}
        directory.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(f"[FAIL] diagnosis_input: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data["source_inputs_before"] = before
    print(f"OUTPUT_DIR={directory}", flush=True)
    try:
        report.save()
        report.require_writable()
        diagnose(args, report, source)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        try:
            after = {str(path): baseline.file_signature(path) for path in files}
            report.data["source_inputs_after"] = after
            report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "原参考文件大小/修改时间未变（非内容哈希）；只写新诊断目录")
        except OSError as error:
            report.check("source_inputs_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
