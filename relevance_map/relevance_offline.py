import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np


# ============================================================
# 路径初始化
#
# 当前文件：
# LS-Imagine-Ref/relevance_map/relevance_offline.py
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# LS-Imagine-Ref/
PROJECT_ROOT = SCRIPT_DIR.parent


# 让 Python 可以找到：
# LS-Imagine-Ref/envs
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# 先检查项目根目录找得对不对
TASK_SPECS_PATH = (
    PROJECT_ROOT
    / "envs"
    / "tasks"
    / "task_specs.yaml"
)

if not TASK_SPECS_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find task_specs.yaml: "
        f"{TASK_SPECS_PATH}"
    )


# ============================================================
# 很重要
#
# envs/tasks/__init__.py 内部目前写的是：
#
# OmegaConf.load("envs/tasks/task_specs.yaml")
#
# MineCLIP 默认 checkpoint 又是：
#
# weights/mineclip_attn.pth
#
# 所以这里把 cwd 固定回项目根目录。
# ============================================================

os.chdir(PROJECT_ROOT)


# ============================================================
# 下面才 import 项目代码
# ============================================================

import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from matplotlib import colormaps

from envs.tasks.minedojo.wrappers import (
    MinedojoClipReward,
    MinedojoConcentrationReward,
)

# ============================================================
# 配置
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

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "relevance_test"
    / f"relevance_offline_test_{timestamp}"
)

# 和你接入 LS-Imagine 时保持一致
RELEVANCE_THRESHOLD = 0.285
RELEVANCE_TEMPERATURE = 0.015

RUN_GLOBAL_PARITY_TEST = True
RUN_ZOOM_TEST = True

# ============================================================
# Pillow 兼容
# ============================================================

RESAMPLE_BILINEAR = getattr(
    Image,
    "Resampling",
    Image,
).BILINEAR


# ============================================================
# NPZ -> MineDojo obs
# ============================================================

def frame_to_obs(frame_hwc):
    """
    frame:
        [H,W,3] uint8

    MineDojo obs["rgb"]:
        [3,H,W]
    """

    rgb = np.transpose(
        frame_hwc,
        (2, 0, 1),
    )

    rgb = np.ascontiguousarray(
        rgb,
        dtype=np.uint8,
    )

    return {
        "rgb": rgb,
    }


# ============================================================
# resize 10x16 map -> 160x256
# ============================================================

def resize_map(
        small_map,
        output_size=(160, 256),
):
    """
    small_map:
        [10,16]

    Returns:
        [160,256]
    """

    x = torch.from_numpy(
        small_map
    ).float()

    x = x.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    x = F.interpolate(
        x,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    return (
        x[0, 0]
        .cpu()
        .numpy()
    )


# ============================================================
# heatmap overlay
# ============================================================

def make_overlay(
        frame,
        relevance,
        alpha=0.5,
):
    """
    frame:
        [H,W,3]

    relevance:
        [H,W], 0~1
    """

    relevance = np.clip(
        relevance,
        0.0,
        1.0,
    )

    cmap = colormaps["turbo"]

    heatmap = cmap(
        relevance
    )[..., :3]

    heatmap = (
            heatmap * 255
    ).astype(np.float32)

    frame_float = frame.astype(
        np.float32
    )

    overlay = (
            (1.0 - alpha)
            * frame_float
            + alpha
            * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return overlay


# ============================================================
# 横向三图：
#
# RGB | Fixed scale | Per-frame min-max
# ============================================================

def make_comparison_frame(
        frame,
        fixed_map,
        minmax_map,
):
    fixed_overlay = make_overlay(
        frame,
        fixed_map,
    )

    minmax_overlay = make_overlay(
        frame,
        minmax_map,
    )

    H, W, _ = frame.shape

    canvas = Image.new(
        "RGB",
        (
            W * 3,
            H + 25,
        ),
    )

    canvas.paste(
        Image.fromarray(frame),
        (0, 25),
    )

    canvas.paste(
        Image.fromarray(
            fixed_overlay
        ),
        (W, 25),
    )

    canvas.paste(
        Image.fromarray(
            minmax_overlay
        ),
        (W * 2, 25),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (5, 5),
        "RGB",
        fill="white",
    )

    draw.text(
        (W + 5, 5),
        "Fixed scale",
        fill="white",
    )

    draw.text(
        (W * 2 + 5, 5),
        "Per-frame min-max",
        fill="white",
    )

    return canvas


# ============================================================
# 保存 GIF
# ============================================================

def save_gif(
        frames,
        path,
        duration=200,
):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


# ============================================================
# Global feature 一致性测试
# ============================================================

@torch.no_grad()
def test_global_feature_parity(
        clip,
        frame,
):
    print(
        "\n"
        "======================================"
    )
    print(
        "TEST 1: Global feature parity"
    )
    print(
        "======================================"
    )

    obs = frame_to_obs(
        frame
    )

    curr_frame = (
        clip._get_curr_frame(
            obs
        )
    )

    # --------------------------------------------------------
    # 新方法
    # --------------------------------------------------------

    new_global, patch, grid = (
        clip.forward_image_and_patch(
            curr_frame
        )
    )

    # --------------------------------------------------------
    # MineCLIP 官方方法
    #
    # 注意：
    # 这里只是测试，所以故意再运行一遍 ViT。
    # 正式训练绝对不要这么做。
    # --------------------------------------------------------

    official_global = (
        clip.model
        .forward_image_features(
            curr_frame
            .unsqueeze(0)
            .to(clip.device)
        )
    )

    error = torch.max(
        torch.abs(
            new_global
            - official_global
        )
    ).item()

    print(
        "new global:",
        tuple(new_global.shape),
    )

    print(
        "official global:",
        tuple(official_global.shape),
    )

    print(
        "patch:",
        tuple(patch.shape),
    )

    print(
        "grid:",
        grid,
    )

    print(
        "max absolute error:",
        error,
    )

    torch.testing.assert_close(
        new_global,
        official_global,
        rtol=1e-5,
        atol=1e-5,
    )

    print(
        "[PASS] Global feature is correct."
    )


# ============================================================
# main
# ============================================================

def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. load NPZ
    # ========================================================

    data = np.load(
        NPZ_PATH
    )

    if "frames" not in data:
        raise KeyError(
            "'frames' not found in npz."
        )

    frames = data[
        "frames"
    ]

    print(
        "NPZ:",
        NPZ_PATH
    )

    print(
        "frames:",
        frames.shape,
        frames.dtype,
    )

    assert frames.ndim == 4
    assert frames.shape[-1] == 3

    # ========================================================
    # 2. Shared MineCLIP
    # ========================================================

    print(
        "\n"
        "Loading shared MineCLIP..."
    )

    clip = (
        MinedojoClipReward()
    )

    concentration = (
        MinedojoConcentrationReward(
            clip_reward=clip,
            gaussian_sigma_weight=0.5,
            relevance_threshold=(
                RELEVANCE_THRESHOLD
            ),
            relevance_temperature=(
                RELEVANCE_TEMPERATURE
            ),
            output_dir=OUTPUT_DIR,
        )
    )

    # ========================================================
    # 必须共享同一个模型
    # ========================================================

    assert (
            concentration.model
            is clip.model
    )

    print(
        "[PASS] ConcentrationReward "
        "and ClipReward share MineCLIP."
    )

    # 不应该再存在真正工作的 U-Net
    if hasattr(
            concentration,
            "unet",
    ):
        assert (
                concentration.unet
                is None
        )

    print(
        "[PASS] Swin-U-Net is not loaded."
    )

    # ========================================================
    # 3. Global parity
    # ========================================================

    if RUN_GLOBAL_PARITY_TEST:
        test_global_feature_parity(
            clip,
            frames[0],
        )

    # parity 测试会额外跑 ViT，
    # 这里归零，后面专门统计共享 forward。
    clip.vision_forward_count = 0

    # ========================================================
    # 保存结果
    # ========================================================

    raw_cosines = []

    fixed_maps = []
    minmax_maps = []

    comparison_frames = []

    zoom_probs = []

    natural_zoom_count = 0

    # ========================================================
    # 4. 逐帧测试
    # ========================================================

    print(
        "\n"
        "======================================"
    )
    print(
        "TEST 2: Relevance map"
    )
    print(
        "======================================"
    )

    for i, frame in enumerate(
            frames
    ):

        obs = frame_to_obs(
            frame
        )

        before = (
            clip.vision_forward_count
        )

        # ====================================================
        # 第一次：
        # 当前帧 Vision 运行一次
        # ====================================================

        bundle = (
            clip.get_frame_bundle(
                obs
            )
        )

        after_bundle = (
            clip.vision_forward_count
        )

        # ====================================================
        # Patch Temporal + cosine
        #
        # 不允许重新运行 Vision
        # ====================================================

        similarity = (
            clip.get_patch_similarity(
                bundle,
                [PROMPT],
            )
        )

        after_similarity = (
            clip.vision_forward_count
        )

        # [1,1,160]
        sim = (
            similarity[
                0,
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        raw_cosines.append(
            sim
        )

        grid_h, grid_w = (
            bundle["grid_size"]
        )

        sim_grid = sim.reshape(
            grid_h,
            grid_w,
        )

        # ====================================================
        # 参考：
        # 固定尺度
        #
        # sigmoid((cosine - tau) / T)
        # ====================================================

        fixed_small = (
                1.0
                /
                (
                        1.0
                        +
                        np.exp(
                            -(
                                    sim_grid
                                    - RELEVANCE_THRESHOLD
                            )
                            /
                            RELEVANCE_TEMPERATURE
                        )
                )
        )

        fixed_map_reference = (
            resize_map(
                fixed_small,
                output_size=(
                    160,
                    256,
                ),
            )
        )

        # ====================================================
        # 旧可视化方式：
        # 每一帧自己 min-max
        #
        # 这里只为了比较，不能用于 RL。
        # ====================================================

        sim_min = sim_grid.min()
        sim_max = sim_grid.max()

        minmax_small = (
                               sim_grid
                               - sim_min
                       ) / (
                               sim_max
                               - sim_min
                               + 1e-8
                       )

        minmax_map = resize_map(
            minmax_small,
            output_size=(
                160,
                256,
            ),
        )

        # ====================================================
        # 真正生产代码
        #
        # get_reward()
        # 内部会再次调用 get_frame_bundle(obs)
        #
        # 但是因为 obs 已经缓存 bundle，
        # Vision count 不允许增加。
        # ====================================================

        score, zoom_prob, threshold = (
            concentration.get_reward(
                obs,
                [PROMPT],
                episode_num=0,
                step_num=i,
            )
        )

        after_reward = (
            clip.vision_forward_count
        )

        # production map:
        # self.mask 是 0~255
        production_map = (
                concentration.mask
                / 255.0
        )

        # ====================================================
        # 关键检查：
        # 生产 map == 我们算的固定尺度 map
        # ====================================================

        map_error = np.max(
            np.abs(
                production_map
                - fixed_map_reference
            )
        )

        # ====================================================
        # Vision 必须只 +1
        # ====================================================

        assert (
                after_bundle
                - before
                == 1
        ), (
            "Vision did not run exactly once."
        )

        assert (
                after_similarity
                == after_bundle
        ), (
            "Patch similarity reran Vision!"
        )

        assert (
                after_reward
                == after_bundle
        ), (
            "ConcentrationReward reran Vision!"
        )

        if map_error > 1e-4:
            raise AssertionError(
                f"Frame {i}: "
                f"production relevance map "
                f"does not match fixed-scale map. "
                f"error={map_error}"
            )

        fixed_maps.append(
            production_map.copy()
        )

        minmax_maps.append(
            minmax_map.copy()
        )

        zoom_probs.append(
            zoom_prob
        )

        comparison_frames.append(
            make_comparison_frame(
                frame,
                production_map,
                minmax_map,
            )
        )

        # ====================================================
        # 自然 Zoom 测试
        # ====================================================

        zoomed_image, is_check = (
            concentration
            .generate_zoom_in_frame()
        )

        if is_check:
            natural_zoom_count += 1

            before_zoom = (
                clip.vision_forward_count
            )

            zoom_result = (
                concentration
                .compute_reward_on_zoomed_image()
            )

            after_zoom = (
                clip.vision_forward_count
            )

            # zoomed_image 是新图，
            # 所以允许额外 +1 Vision
            assert (
                    after_zoom
                    - before_zoom
                    == 1
            )

            print(
                f"Frame {i:02d}: "
                f"NATURAL ZOOM triggered, "
                f"result={zoom_result}"
            )

        print(
            f"Frame {i:02d} | "
            f"cos min={sim.min():.4f} "
            f"mean={sim.mean():.4f} "
            f"max={sim.max():.4f} | "
            f"fixed mean="
            f"{production_map.mean():.4f} "
            f"max="
            f"{production_map.max():.4f} | "
            f"area>0.5="
            f"{np.mean(production_map > 0.5):.3f} | "
            f"zoom_prob={zoom_prob:.4f} | "
            f"threshold={threshold:.4f}"
        )

    # ========================================================
    # 5. 保存数据
    # ========================================================

    raw_cosines = np.stack(
        raw_cosines,
        axis=0,
    )

    fixed_maps = np.stack(
        fixed_maps,
        axis=0,
    )

    minmax_maps = np.stack(
        minmax_maps,
        axis=0,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "raw_cosine.npy",
        ),
        raw_cosines,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "fixed_relevance.npy",
        ),
        fixed_maps,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "minmax_visualization.npy",
        ),
        minmax_maps,
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "fixed_vs_minmax.gif",
    )

    save_gif(
        comparison_frames,
        comparison_path,
    )

    # ========================================================
    # cosine 全局统计
    # ========================================================

    print(
        "\n"
        "======================================"
    )
    print(
        "Cosine statistics"
    )
    print(
        "======================================"
    )

    for q in [
        1,
        5,
        25,
        50,
        75,
        90,
        95,
        99,
    ]:
        value = np.percentile(
            raw_cosines,
            q,
        )

        print(
            f"P{q:02d}: "
            f"{value:.4f}"
        )

    print(
        "global min:",
        raw_cosines.min(),
    )

    print(
        "global max:",
        raw_cosines.max(),
    )

    print(
        "global mean:",
        raw_cosines.mean(),
    )

    print(
        "\nNatural zoom count:",
        natural_zoom_count,
    )

    print(
        "comparison GIF:",
        comparison_path,
    )

    # ========================================================
    # 6. 强制测试 Zoom 完整链
    # ========================================================

    if RUN_ZOOM_TEST:

        print(
            "\n"
            "======================================"
        )
        print(
            "TEST 3: Forced zoom"
        )
        print(
            "======================================"
        )

        # 选择 zoom_prob 最大的那一帧
        best_idx = int(
            np.argmax(
                zoom_probs
            )
        )

        print(
            "best frame:",
            best_idx,
        )

        print(
            "zoom_prob:",
            zoom_probs[
                best_idx
            ],
        )

        # 新对象，避免之前 ThresholdBuffer
        # 影响 forced test
        concentration_zoom = (
            MinedojoConcentrationReward(
                clip_reward=clip,
                gaussian_sigma_weight=0.5,
                relevance_threshold=(
                    RELEVANCE_THRESHOLD
                ),
                relevance_temperature=(
                    RELEVANCE_TEMPERATURE
                ),
                output_dir=OUTPUT_DIR,
            )
        )

        obs = frame_to_obs(
            frames[
                best_idx
            ]
        )

        concentration_zoom.get_reward(
            obs,
            [PROMPT],
            episode_num=0,
            step_num=best_idx,
        )

        # ====================================================
        # 强行通过第一道 adaptive threshold
        #
        # 只用于 smoke test。
        # 正式训练绝对不要这么写。
        # ====================================================

        concentration_zoom.check_threshold = (
            -1.0
        )

        zoomed_image, is_check = (
            concentration_zoom
            .generate_zoom_in_frame()
        )

        print(
            "generate_zoom_in_frame:",
            is_check,
        )

        # 保存当前原图
        Image.fromarray(
            frames[
                best_idx
            ]
        ).save(
            os.path.join(
                OUTPUT_DIR,
                "forced_zoom_original.png",
            )
        )

        # 保存当前 heatmap
        current_overlay = make_overlay(
            frames[
                best_idx
            ],
            concentration_zoom.mask
            / 255.0,
        )

        Image.fromarray(
            current_overlay
        ).save(
            os.path.join(
                OUTPUT_DIR,
                "forced_zoom_before.png",
            )
        )

        if not is_check:

            print(
                "[WARNING] Forced threshold passed, "
                "but generate_zoom_in_frame() "
                "did not find a valid contour."
            )

            print(
                "This does not necessarily mean the "
                "code is broken; the selected heatmap "
                "may not form a valid crop region."
            )

        else:

            before_zoom = (
                clip.vision_forward_count
            )

            result = (
                concentration_zoom
                .compute_reward_on_zoomed_image()
            )

            after_zoom = (
                clip.vision_forward_count
            )

            print(
                "compute_reward_on_zoomed_image:",
                result,
            )

            print(
                "zoom Vision delta:",
                after_zoom
                - before_zoom,
            )

            assert (
                    after_zoom
                    - before_zoom
                    == 1
            )

            assert (
                    concentration_zoom
                    .mask_on_zoomed_image
                    .shape
                    ==
                    (
                        160,
                        256,
                    )
            )

            Image.fromarray(
                zoomed_image
            ).save(
                os.path.join(
                    OUTPUT_DIR,
                    "forced_zoom_rgb.png",
                )
            )

            zoom_overlay = make_overlay(
                zoomed_image,
                concentration_zoom
                .mask_on_zoomed_image,
            )

            Image.fromarray(
                zoom_overlay
            ).save(
                os.path.join(
                    OUTPUT_DIR,
                    "forced_zoom_after.png",
                )
            )

            print(
                "[PASS] Complete zoom pipeline works."
            )

    # ========================================================
    # 最终统计
    # ========================================================

    print(
        "\n"
        "======================================"
    )
    print(
        "ALL OFFLINE TESTS FINISHED"
    )
    print(
        "======================================"
    )

    print(
        "Total current-frame Vision forwards:",
        clip.vision_forward_count,
    )


if __name__ == "__main__":
    main()
