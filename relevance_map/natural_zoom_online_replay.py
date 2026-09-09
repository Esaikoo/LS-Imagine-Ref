#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
natural_zoom_online_replay.py

按时间顺序重放 real_approach_test.py 保存的 trajectory.npz，
模拟正式运行时的 Natural Zoom 因果决策。

不会：
- 从整条轨迹里 np.argmax 选未来最优帧
- 强制 check_threshold = -1
- 使用未来 observation 决定当前是否 zoom

会：
- 只创建一个持久的 MinedojoConcentrationReward
- frame 0 -> 1 -> 2 -> ... 顺序处理
- 每帧 get_reward()
- 立即 generate_zoom_in_frame()
- 只有自然触发时才 compute_reward_on_zoomed_image()
- 记录 zoom_prob / adaptive threshold / trigger / zoom 后 relevance
- 检查当前帧 Vision 只跑 1 次，Natural Zoom 新图只额外跑 1 次

注意：
这是 online-equivalent causal replay，不是完整 closed-loop。
trajectory 的未来 RGB 已经固定，因此 Zoom 不会反过来改变未来 Actor 动作或环境轨迹。
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_SPECS_PATH = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS_PATH.exists():
    raise FileNotFoundError(f"Cannot find task_specs.yaml: {TASK_SPECS_PATH}")

os.chdir(PROJECT_ROOT)

import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps

from envs.tasks.minedojo.wrappers import (
    MinedojoClipReward,
    MinedojoConcentrationReward,
)

# ============================================================
# PyCharm 配置
# ============================================================

NPZ_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_approach_runs"
    / "harvest_water_with_bucket"
    / "world_49_taskseed_0_20260905_222039"
    / "trajectory.npz"
)

PROMPT = "Obtain water"

RELEVANCE_THRESHOLD = 0.3
RELEVANCE_TEMPERATURE = 0.02
GAUSSIAN_SIGMA_WEIGHT = 0.5

RUN_GLOBAL_PARITY_TEST = False
GIF_DURATION_MS = 180
SAVE_EACH_ZOOM_EVENT = True
SAVE_FULL_RELEVANCE_NPY = True
PRINT_EVERY_FRAME = True

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    SCRIPT_DIR
    / "relevance_test"
    / f"natural_zoom_online_replay_{TIMESTAMP}"
)


def frame_to_obs(frame_hwc):
    rgb = np.transpose(frame_hwc, (2, 0, 1))
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return {"rgb": rgb}


def make_overlay(frame, relevance, alpha=0.5):
    relevance = np.clip(relevance, 0.0, 1.0)
    heatmap = (colormaps["turbo"](relevance)[..., :3] * 255.0).astype(np.float32)
    frame_float = frame.astype(np.float32)
    overlay = (1.0 - alpha) * frame_float + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gif(frames, path, duration=180):
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def largest_cc_fraction(relevance, threshold=0.5):
    binary = (relevance > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if num_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(np.max(areas) / binary.size)


def cosine_metrics(sim):
    sim = np.asarray(sim, dtype=np.float32).reshape(-1)
    p50 = float(np.percentile(sim, 50))
    p90 = float(np.percentile(sim, 90))
    p95 = float(np.percentile(sim, 95))
    return {
        "cos_min": float(sim.min()),
        "cos_mean": float(sim.mean()),
        "cos_max": float(sim.max()),
        "cos_p50": p50,
        "cos_p90": p90,
        "cos_p95": p95,
        "cos_range": float(sim.max() - sim.min()),
        "cos_p95_p50": float(p95 - p50),
    }


def relevance_metrics(relevance):
    relevance = np.asarray(relevance, dtype=np.float32)
    return {
        "rel_mean": float(relevance.mean()),
        "rel_max": float(relevance.max()),
        "rel_area_03": float(np.mean(relevance > 0.3)),
        "rel_area_05": float(np.mean(relevance > 0.5)),
        "rel_area_07": float(np.mean(relevance > 0.7)),
        "rel_area_09": float(np.mean(relevance > 0.9)),
        "rel_cc05": largest_cc_fraction(relevance, threshold=0.5),
    }


def get_trajectory_metadata(data, num_frames):
    action_commands = None
    if "action_commands" in data:
        action_commands = np.asarray(data["action_commands"]).astype(str)

    reward = np.zeros(num_frames, dtype=np.float32)
    done = np.zeros(num_frames, dtype=np.uint8)
    inventory = np.zeros(num_frames, dtype=np.float32)

    if "reward" in data:
        values = np.asarray(data["reward"]).reshape(-1)
        reward[:min(num_frames, len(values))] = values[:num_frames]

    if "done" in data:
        values = np.asarray(data["done"]).reshape(-1)
        done[:min(num_frames, len(values))] = values[:num_frames]

    if "target_inventory_quantity" in data:
        values = np.asarray(data["target_inventory_quantity"]).reshape(-1)
        inventory[:min(num_frames, len(values))] = values[:num_frames]

    return {
        "action_commands": action_commands,
        "reward": reward,
        "done": done,
        "inventory": inventory,
    }


@torch.no_grad()
def test_global_feature_parity(clip, frame):
    print("\n======================================")
    print("OPTIONAL TEST: Global feature parity")
    print("======================================")

    obs = frame_to_obs(frame)
    curr_frame = clip._get_curr_frame(obs)

    new_global, patch, grid = clip.forward_image_and_patch(curr_frame)

    official_global = clip.model.forward_image_features(
        curr_frame.unsqueeze(0).to(clip.device)
    )

    error = torch.max(torch.abs(new_global - official_global)).item()

    print("max absolute error:", error)

    torch.testing.assert_close(
        new_global,
        official_global,
        rtol=1e-5,
        atol=1e-5,
    )

    print("[PASS] Global feature parity.")


def make_timeline_frame(
    frame,
    relevance,
    frame_index,
    zoom_prob,
    adaptive_threshold,
    natural_zoom_triggered,
    zoomed_image=None,
    zoomed_relevance=None,
):
    H, W, _ = frame.shape
    current_overlay = make_overlay(frame, relevance)

    canvas = Image.new(
        "RGB",
        (W * 3, H + 64),
        "black",
    )
    draw = ImageDraw.Draw(canvas)

    canvas.paste(Image.fromarray(frame), (0, 64))
    canvas.paste(Image.fromarray(current_overlay), (W, 64))

    if natural_zoom_triggered and zoomed_image is not None:
        if zoomed_relevance is not None:
            zoom_overlay = make_overlay(zoomed_image, zoomed_relevance)
            canvas.paste(Image.fromarray(zoom_overlay), (W * 2, 64))
        else:
            canvas.paste(Image.fromarray(zoomed_image), (W * 2, 64))
    else:
        draw.rectangle(
            (W * 2, 64, W * 3 - 1, H + 63),
            outline="white",
            width=1,
        )
        draw.text(
            (W * 2 + 35, 64 + H // 2),
            "NO NATURAL ZOOM",
            fill="white",
        )

    draw.text((5, 7), "RGB", fill="white")
    draw.text((W + 5, 7), "Current relevance", fill="white")
    draw.text((W * 2 + 5, 7), "Natural Zoom", fill="white")

    draw.text(
        (5, 30),
        (
            f"step={frame_index} "
            f"zoom_prob={zoom_prob:.4f} "
            f"adaptive_threshold={adaptive_threshold:.4f} "
            f"trigger={int(natural_zoom_triggered)}"
        ),
        fill="white",
    )

    return canvas


def save_rows_csv(rows, path):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    zoom_dir = OUTPUT_DIR / "zoom_events"
    if SAVE_EACH_ZOOM_EVENT:
        zoom_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(
        NPZ_PATH,
        allow_pickle=False,
    )

    if "frames" not in data:
        raise KeyError("'frames' not found in trajectory.npz")

    frames = np.asarray(data["frames"])
    assert frames.ndim == 4
    assert frames.shape[-1] == 3

    num_frames = frames.shape[0]
    metadata = get_trajectory_metadata(data, num_frames)

    print("\n============================================================")
    print("NATURAL ZOOM ONLINE-EQUIVALENT REPLAY")
    print("============================================================")
    print("NPZ:", NPZ_PATH)
    print("frames:", frames.shape, frames.dtype)
    print("prompt:", PROMPT)
    print("tau:", RELEVANCE_THRESHOLD)
    print("T:", RELEVANCE_TEMPERATURE)
    print("Important:")
    print("  - no future-frame argmax")
    print("  - no forced threshold")
    print("  - one persistent ConcentrationReward object")
    print("  - frames replayed strictly t=0 -> T-1")

    print("\nLoading shared MineCLIP...")

    clip = MinedojoClipReward()

    concentration = MinedojoConcentrationReward(
        clip_reward=clip,
        gaussian_sigma_weight=GAUSSIAN_SIGMA_WEIGHT,
        relevance_threshold=RELEVANCE_THRESHOLD,
        relevance_temperature=RELEVANCE_TEMPERATURE,
        output_dir=str(OUTPUT_DIR),
    )

    assert concentration.model is clip.model

    if hasattr(concentration, "unet"):
        assert concentration.unet is None

    if RUN_GLOBAL_PARITY_TEST:
        test_global_feature_parity(
            clip,
            frames[0],
        )

    if hasattr(clip, "vision_forward_count"):
        clip.vision_forward_count = 0

    rows = []
    timeline_gif_frames = []
    current_relevance_maps = []

    zoom_event_steps = []
    zoom_event_probs = []
    zoom_event_thresholds = []

    total_original_vision = 0
    total_zoom_vision = 0

    print("\n============================================================")
    print("CAUSAL NATURAL ZOOM REPLAY")
    print("============================================================")

    for frame_index, frame in enumerate(frames):
        obs = frame_to_obs(frame)

        # ----------------------------------------------------
        # A. 当前真实帧：Vision 只跑一次
        # ----------------------------------------------------
        before_current = getattr(
            clip,
            "vision_forward_count",
            0,
        )

        bundle = clip.get_frame_bundle(obs)

        after_bundle = getattr(
            clip,
            "vision_forward_count",
            0,
        )

        similarity = clip.get_patch_similarity(
            bundle,
            [PROMPT],
        )

        after_similarity = getattr(
            clip,
            "vision_forward_count",
            0,
        )

        sim = (
            similarity[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        raw_metrics = cosine_metrics(sim)

        # ----------------------------------------------------
        # B. 正式 concentration reward
        # 同一个 concentration 对象持续整条 trajectory
        # ----------------------------------------------------
        score, zoom_prob, adaptive_threshold = concentration.get_reward(
            obs,
            [PROMPT],
            episode_num=0,
            step_num=frame_index,
        )

        after_reward = getattr(
            clip,
            "vision_forward_count",
            0,
        )

        current_map = (
            np.asarray(
                concentration.mask,
                dtype=np.float32,
            )
            / 255.0
        )

        current_metrics = relevance_metrics(current_map)

        current_relevance_maps.append(
            current_map.copy()
        )

        if hasattr(clip, "vision_forward_count"):
            current_delta = after_bundle - before_current

            assert current_delta == 1, (
                f"Frame {frame_index}: current RGB Vision delta "
                f"must be 1, got {current_delta}"
            )

            assert after_similarity == after_bundle, (
                f"Frame {frame_index}: patch similarity reran Vision."
            )

            assert after_reward == after_bundle, (
                f"Frame {frame_index}: get_reward reran Vision."
            )

            total_original_vision += current_delta

        # ----------------------------------------------------
        # C. Natural Zoom
        # 不改 threshold，不看未来，直接当前帧决策
        # ----------------------------------------------------
        zoomed_image, is_check = concentration.generate_zoom_in_frame()

        natural_trigger = bool(is_check)

        zoomed_relevance_map = None
        zoom_result = None
        zoom_vision_delta = 0

        zoom_rel_metrics = {
            "zoom_rel_mean": np.nan,
            "zoom_rel_max": np.nan,
            "zoom_rel_area_03": np.nan,
            "zoom_rel_area_05": np.nan,
            "zoom_rel_area_07": np.nan,
            "zoom_rel_area_09": np.nan,
            "zoom_rel_cc05": np.nan,
        }

        if natural_trigger:
            zoom_event_steps.append(frame_index)
            zoom_event_probs.append(float(zoom_prob))
            zoom_event_thresholds.append(float(adaptive_threshold))

            before_zoom = getattr(
                clip,
                "vision_forward_count",
                0,
            )

            zoom_result = concentration.compute_reward_on_zoomed_image()

            after_zoom = getattr(
                clip,
                "vision_forward_count",
                0,
            )

            if hasattr(clip, "vision_forward_count"):
                zoom_vision_delta = after_zoom - before_zoom

                assert zoom_vision_delta == 1, (
                    f"Frame {frame_index}: Natural Zoom should add exactly "
                    f"1 Vision forward, got {zoom_vision_delta}"
                )

                total_zoom_vision += zoom_vision_delta

            if hasattr(
                concentration,
                "mask_on_zoomed_image",
            ):
                zoomed_relevance_map = np.asarray(
                    concentration.mask_on_zoomed_image,
                    dtype=np.float32,
                )

                if zoomed_relevance_map.max() > 1.5:
                    zoomed_relevance_map = (
                        zoomed_relevance_map
                        / 255.0
                    )

                metrics = relevance_metrics(
                    zoomed_relevance_map
                )

                zoom_rel_metrics = {
                    f"zoom_{key}": value
                    for key, value in metrics.items()
                }

            if SAVE_EACH_ZOOM_EVENT:
                prefix = f"step_{frame_index:04d}"

                Image.fromarray(
                    frame
                ).save(
                    zoom_dir
                    / f"{prefix}_original_rgb.png"
                )

                Image.fromarray(
                    make_overlay(
                        frame,
                        current_map,
                    )
                ).save(
                    zoom_dir
                    / f"{prefix}_before_zoom.png"
                )

                zoom_rgb = np.asarray(
                    zoomed_image,
                    dtype=np.uint8,
                )

                Image.fromarray(
                    zoom_rgb
                ).save(
                    zoom_dir
                    / f"{prefix}_zoom_rgb.png"
                )

                if zoomed_relevance_map is not None:
                    Image.fromarray(
                        make_overlay(
                            zoom_rgb,
                            zoomed_relevance_map,
                        )
                    ).save(
                        zoom_dir
                        / f"{prefix}_after_zoom.png"
                    )

        # ----------------------------------------------------
        # D. trajectory metadata：只做日志，不参与 zoom 决策
        # ----------------------------------------------------
        action_command = ""

        if (
            metadata["action_commands"] is not None
            and frame_index < len(metadata["action_commands"])
        ):
            action_command = str(
                metadata["action_commands"][frame_index]
            )

        reward = float(
            metadata["reward"][frame_index]
        )

        done = int(
            metadata["done"][frame_index]
        )

        inventory = float(
            metadata["inventory"][frame_index]
        )

        row = {
            "step": frame_index,
            "action_command": action_command,
            "reward": reward,
            "done": done,
            "target_inventory_quantity": inventory,

            "concentration_score": float(score),
            "zoom_prob": float(zoom_prob),
            "adaptive_threshold": float(adaptive_threshold),
            "natural_zoom_triggered": int(natural_trigger),

            "current_vision_delta": 1,
            "zoom_vision_delta": int(zoom_vision_delta),

            **raw_metrics,
            **current_metrics,
            **zoom_rel_metrics,
        }

        if zoom_result is not None:
            if isinstance(
                zoom_result,
                (tuple, list),
            ):
                for j, value in enumerate(zoom_result):
                    if isinstance(
                        value,
                        (
                            bool,
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    ):
                        row[f"zoom_result_{j}"] = value

            elif isinstance(
                zoom_result,
                (
                    bool,
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                row["zoom_result"] = zoom_result

        rows.append(row)

        zoom_rgb_for_gif = None
        if natural_trigger:
            zoom_rgb_for_gif = np.asarray(
                zoomed_image,
                dtype=np.uint8,
            )

        timeline_gif_frames.append(
            make_timeline_frame(
                frame=frame,
                relevance=current_map,
                frame_index=frame_index,
                zoom_prob=float(zoom_prob),
                adaptive_threshold=float(adaptive_threshold),
                natural_zoom_triggered=natural_trigger,
                zoomed_image=zoom_rgb_for_gif,
                zoomed_relevance=zoomed_relevance_map,
            )
        )

        if PRINT_EVERY_FRAME:
            marker = (
                "<<< NATURAL ZOOM"
                if natural_trigger
                else ""
            )

            print(
                f"step={frame_index:04d} | "
                f"cos mean={raw_metrics['cos_mean']:.4f} "
                f"P95={raw_metrics['cos_p95']:.4f} "
                f"P95-P50={raw_metrics['cos_p95_p50']:.4f} | "
                f"rel mean={current_metrics['rel_mean']:.4f} "
                f">.5={current_metrics['rel_area_05']:.3f} "
                f"CC={current_metrics['rel_cc05']:.3f} | "
                f"zoom_prob={float(zoom_prob):.4f} "
                f"threshold={float(adaptive_threshold):.4f} | "
                f"trigger={int(natural_trigger)} "
                f"{marker}"
            )

            if natural_trigger:
                print(
                    "    zoom_result:",
                    zoom_result,
                )
                print(
                    "    zoom Vision delta:",
                    zoom_vision_delta,
                )

                if zoomed_relevance_map is not None:
                    print(
                        "    zoom relevance: "
                        f"mean={zoom_rel_metrics['zoom_rel_mean']:.4f} "
                        f">.5={zoom_rel_metrics['zoom_rel_area_05']:.3f} "
                        f"CC={zoom_rel_metrics['zoom_rel_cc05']:.3f}"
                    )

    # ========================================================
    # 保存
    # ========================================================
    csv_path = (
        OUTPUT_DIR
        / "natural_zoom_timeline.csv"
    )

    save_rows_csv(
        rows,
        csv_path,
    )

    current_relevance_maps = np.stack(
        current_relevance_maps,
        axis=0,
    )

    if SAVE_FULL_RELEVANCE_NPY:
        np.save(
            OUTPUT_DIR
            / "current_relevance_maps.npy",
            current_relevance_maps,
        )

    np.save(
        OUTPUT_DIR
        / "natural_zoom_steps.npy",
        np.asarray(
            zoom_event_steps,
            dtype=np.int64,
        ),
    )

    np.save(
        OUTPUT_DIR
        / "natural_zoom_probs.npy",
        np.asarray(
            zoom_event_probs,
            dtype=np.float32,
        ),
    )

    np.save(
        OUTPUT_DIR
        / "natural_zoom_thresholds.npy",
        np.asarray(
            zoom_event_thresholds,
            dtype=np.float32,
        ),
    )

    gif_path = (
        OUTPUT_DIR
        / "natural_zoom_timeline.gif"
    )

    save_gif(
        timeline_gif_frames,
        gif_path,
        duration=GIF_DURATION_MS,
    )

    zoom_event_count = len(
        zoom_event_steps
    )

    trigger_rate = (
        zoom_event_count
        / num_frames
        if num_frames
        else 0.0
    )

    summary = {
        "npz_path": str(NPZ_PATH),
        "prompt": PROMPT,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "relevance_temperature": RELEVANCE_TEMPERATURE,

        "num_frames": int(num_frames),
        "natural_zoom_count": int(zoom_event_count),
        "natural_zoom_rate": float(trigger_rate),
        "natural_zoom_steps": zoom_event_steps,

        "total_original_vision_forwards": int(
            total_original_vision
        ),

        "total_zoom_vision_forwards": int(
            total_zoom_vision
        ),

        "expected_total_vision_forwards": int(
            num_frames
            + zoom_event_count
        ),

        "actual_total_vision_forwards": int(
            getattr(
                clip,
                "vision_forward_count",
                num_frames
                + zoom_event_count,
            )
        ),
    }

    with (
        OUTPUT_DIR
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n============================================================")
    print("NATURAL ZOOM REPLAY FINISHED")
    print("============================================================")

    print("frames:", num_frames)
    print("Natural Zoom count:", zoom_event_count)
    print("Natural Zoom rate:", f"{trigger_rate:.4f}")
    print("Natural Zoom steps:", zoom_event_steps)

    print(
        "Original-frame Vision forwards:",
        total_original_vision,
    )

    print(
        "Zoom-frame Vision forwards:",
        total_zoom_vision,
    )

    print(
        "Expected total Vision forwards:",
        num_frames
        + zoom_event_count,
    )

    print(
        "Actual total Vision forwards:",
        getattr(
            clip,
            "vision_forward_count",
            "N/A",
        ),
    )

    print("\nCSV:", csv_path)
    print("GIF:", gif_path)
    print(
        "Summary:",
        OUTPUT_DIR
        / "summary.json",
    )
    print(
        "Zoom event images:",
        zoom_dir,
    )


if __name__ == "__main__":
    main()
