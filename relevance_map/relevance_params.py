import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from matplotlib import colormaps


# ============================================================
# 路径
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


# ============================================================
# 配置
# ============================================================

# 原始 16 帧
NPZ_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "maskclip_test"
    / "20260901_170642_050808_mode_A"
    / "raw_16frames.npz"
)

# 上一个测试程序保存的原始 cosine
COSINE_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "relevance_test"
    / "relevance_offline_test_20260903_181155"
    / "raw_cosine.npy"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "relevance_test"
    / "relevance_offline_test_20260903_181155"
)


# ============================================================
# 要测试的 threshold / temperature
# ============================================================

SETTINGS = [

    # 当前版本，作为 baseline
    {
        "name": "tau025_T005",
        "threshold": 0.250,
        "temperature": 0.050,
    },

    # 稍微严格
    {
        "name": "tau0285_T003",
        "threshold": 0.285,
        "temperature": 0.030,
    },

    # 推荐重点看
    {
        "name": "tau0300_T003",
        "threshold": 0.300,
        "temperature": 0.030,
    },

    # 更严格
    {
        "name": "tau0315_T003",
        "threshold": 0.315,
        "temperature": 0.030,
    },

    # 再严格一点
    {
        "name": "tau0330_T003",
        "threshold": 0.330,
        "temperature": 0.030,
    },
]


# ============================================================
# Patch grid
# ============================================================

GRID_H = 10
GRID_W = 16

TARGET_H = 160
TARGET_W = 256

ALPHA = 0.50

GIF_DURATION = 180


# ============================================================
# Pillow 兼容
# ============================================================

if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
else:
    BILINEAR = Image.BILINEAR


# ============================================================
# 加载数据
# ============================================================

def load_data():

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"NPZ not found: {NPZ_PATH}"
        )

    if not os.path.exists(COSINE_PATH):
        raise FileNotFoundError(
            f"cosine file not found: {COSINE_PATH}"
        )

    data = np.load(NPZ_PATH)

    if "frames" not in data:
        raise KeyError(
            f"'frames' not found in {NPZ_PATH}"
        )

    frames = data["frames"]

    cosine = np.load(
        COSINE_PATH
    )

    print(
        "frames:",
        frames.shape,
        frames.dtype,
    )

    print(
        "cosine:",
        cosine.shape,
        cosine.dtype,
    )

    assert frames.ndim == 4
    assert frames.shape[-1] == 3

    assert cosine.ndim == 2

    assert cosine.shape[0] == frames.shape[0]

    assert cosine.shape[1] == (
        GRID_H * GRID_W
    )

    return frames, cosine


# ============================================================
# sigmoid fixed-scale
# ============================================================

def cosine_to_relevance(
    cosine,
    threshold,
    temperature,
):
    """
    cosine:
        [T,160]

    return:
        [T,160]
    """

    if temperature <= 0:
        raise ValueError(
            "temperature must > 0"
        )

    x = (
        cosine
        - threshold
    ) / temperature

    # 防止 exp overflow
    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    relevance = (
        1.0
        /
        (
            1.0
            + np.exp(-x)
        )
    )

    return relevance.astype(
        np.float32
    )


# ============================================================
# Patch map -> image map
# ============================================================

def resize_relevance_maps(
    relevance,
):
    """
    relevance:
        [T,160]

    return:
        [T,160,256]
    """

    T = relevance.shape[0]

    x = torch.from_numpy(
        relevance
    ).float()

    x = x.reshape(
        T,
        1,
        GRID_H,
        GRID_W,
    )

    x = F.interpolate(
        x,
        size=(
            TARGET_H,
            TARGET_W,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return (
        x[:, 0]
        .cpu()
        .numpy()
    )


# ============================================================
# 旧的 min-max
# 仅用于视觉参考
# ============================================================

def make_global_minmax(
    cosine,
):
    """
    和你之前 GIF 类似：
    对全部 16 帧统一 min/max。

    注意：
    只用于可视化对比，
    不作为 RL 输入。
    """

    value_min = cosine.min()
    value_max = cosine.max()

    normalized = (
        cosine
        - value_min
    ) / (
        value_max
        - value_min
        + 1e-8
    )

    return resize_relevance_maps(
        normalized.astype(
            np.float32
        )
    )


# ============================================================
# overlay
# ============================================================

def make_overlay(
    frame,
    relevance,
    alpha=0.50,
):

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
        heatmap
        * 255
    ).astype(
        np.float32
    )

    frame = frame.astype(
        np.float32
    )

    overlay = (
        (1.0 - alpha)
        * frame
        +
        alpha
        * heatmap
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# 保存单组 GIF
# ============================================================

def save_single_gif(
    frames,
    maps,
    output_path,
    title,
):

    gif_frames = []

    for i in range(
        len(frames)
    ):

        overlay = make_overlay(
            frames[i],
            maps[i],
            ALPHA,
        )

        H, W, _ = overlay.shape

        canvas = Image.new(
            "RGB",
            (
                W,
                H + 28,
            ),
            "black",
        )

        canvas.paste(
            Image.fromarray(
                overlay
            ),
            (0, 28),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        draw.text(
            (5, 7),
            f"{title} | frame={i}",
            fill="white",
        )

        gif_frames.append(
            canvas
        )

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=GIF_DURATION,
        loop=0,
    )


# ============================================================
# 保存大型横向比较 GIF
# ============================================================

def save_comparison_gif(
    frames,
    setting_maps,
    minmax_maps,
    output_path,
):
    """
    每一帧：

    RGB
    |
    min-max
    |
    tau025
    |
    tau0285
    |
    tau030
    |
    ...
    """

    gif_frames = []

    labels = [
        "RGB",
        "Global min-max",
    ]

    labels += [
        setting["name"]
        for setting in SETTINGS
    ]

    column_num = (
        2
        + len(SETTINGS)
    )

    for frame_idx in range(
        len(frames)
    ):

        H, W, _ = frames[
            frame_idx
        ].shape

        canvas = Image.new(
            "RGB",
            (
                W * column_num,
                H + 32,
            ),
            "black",
        )

        # ====================================================
        # RGB
        # ====================================================

        canvas.paste(
            Image.fromarray(
                frames[
                    frame_idx
                ]
            ),
            (
                0,
                32,
            ),
        )

        # ====================================================
        # min-max
        # ====================================================

        minmax_overlay = (
            make_overlay(
                frames[
                    frame_idx
                ],
                minmax_maps[
                    frame_idx
                ],
                ALPHA,
            )
        )

        canvas.paste(
            Image.fromarray(
                minmax_overlay
            ),
            (
                W,
                32,
            ),
        )

        # ====================================================
        # Fixed-scale settings
        # ====================================================

        for idx, setting in enumerate(
            SETTINGS
        ):

            maps = setting_maps[
                setting["name"]
            ]

            overlay = make_overlay(
                frames[
                    frame_idx
                ],
                maps[
                    frame_idx
                ],
                ALPHA,
            )

            x = (
                W
                * (
                    idx
                    + 2
                )
            )

            canvas.paste(
                Image.fromarray(
                    overlay
                ),
                (
                    x,
                    32,
                ),
            )

        # ====================================================
        # labels
        # ====================================================

        draw = ImageDraw.Draw(
            canvas
        )

        for idx, label in enumerate(
            labels
        ):

            draw.text(
                (
                    idx * W + 5,
                    8,
                ),
                label,
                fill="white",
            )

        gif_frames.append(
            canvas
        )

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=GIF_DURATION,
        loop=0,
    )


# ============================================================
# 统计
# ============================================================

def calculate_metrics(
    maps,
):
    """
    maps:
        [T,H,W]
    """

    metrics = {
        "mean": float(
            maps.mean()
        ),

        "max": float(
            maps.max()
        ),

        "area_05": float(
            np.mean(
                maps > 0.5
            )
        ),

        "area_07": float(
            np.mean(
                maps > 0.7
            )
        ),

        "area_08": float(
            np.mean(
                maps > 0.8
            )
        ),

        "area_09": float(
            np.mean(
                maps > 0.9
            )
        ),
    }

    return metrics


# ============================================================
# 每帧统计
# ============================================================

def print_frame_metrics(
    setting,
    maps,
):

    print(
        "\n--------------------------------------"
    )

    print(
        setting["name"]
    )

    print(
        "threshold:",
        setting[
            "threshold"
        ],
    )

    print(
        "temperature:",
        setting[
            "temperature"
        ],
    )

    print(
        "--------------------------------------"
    )

    for i in range(
        maps.shape[0]
    ):

        current = maps[i]

        print(
            f"Frame {i:02d} | "
            f"mean={current.mean():.4f} | "
            f"max={current.max():.4f} | "
            f">0.5={np.mean(current > 0.5):.3f} | "
            f">0.7={np.mean(current > 0.7):.3f} | "
            f">0.9={np.mean(current > 0.9):.3f}"
        )


# ============================================================
# 保存 summary csv
# ============================================================

def save_summary_csv(
    summary,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "name,"
            "threshold,"
            "temperature,"
            "mean,"
            "max,"
            "area_gt_05,"
            "area_gt_07,"
            "area_gt_08,"
            "area_gt_09\n"
        )

        for row in summary:

            f.write(
                f"{row['name']},"
                f"{row['threshold']},"
                f"{row['temperature']},"
                f"{row['mean']:.6f},"
                f"{row['max']:.6f},"
                f"{row['area_05']:.6f},"
                f"{row['area_07']:.6f},"
                f"{row['area_08']:.6f},"
                f"{row['area_09']:.6f}\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # load
    # ========================================================

    frames, cosine = (
        load_data()
    )

    # ========================================================
    # 原始 cosine 信息
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "Raw cosine statistics"
    )

    print(
        "======================================"
    )

    print(
        "min:",
        cosine.min(),
    )

    print(
        "mean:",
        cosine.mean(),
    )

    print(
        "max:",
        cosine.max(),
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

        print(
            f"P{q:02d}: "
            f"{np.percentile(cosine, q):.4f}"
        )

    # ========================================================
    # min-max
    # ========================================================

    minmax_maps = (
        make_global_minmax(
            cosine
        )
    )

    # ========================================================
    # 所有 fixed-scale
    # ========================================================

    setting_maps = {}

    summary = []

    for setting in SETTINGS:

        print(
            "\n======================================"
        )

        print(
            "Processing:",
            setting["name"],
        )

        print(
            "======================================"
        )

        relevance = (
            cosine_to_relevance(
                cosine,
                threshold=(
                    setting[
                        "threshold"
                    ]
                ),
                temperature=(
                    setting[
                        "temperature"
                    ]
                ),
            )
        )

        maps = (
            resize_relevance_maps(
                relevance
            )
        )

        setting_maps[
            setting["name"]
        ] = maps

        # ====================================================
        # save npy
        # ====================================================

        npy_path = os.path.join(
            OUTPUT_DIR,
            f"{setting['name']}.npy",
        )

        np.save(
            npy_path,
            maps,
        )

        # ====================================================
        # save gif
        # ====================================================

        gif_path = os.path.join(
            OUTPUT_DIR,
            f"{setting['name']}.gif",
        )

        save_single_gif(
            frames,
            maps,
            gif_path,
            (
                f"tau="
                f"{setting['threshold']}, "
                f"T="
                f"{setting['temperature']}"
            ),
        )

        # ====================================================
        # metrics
        # ====================================================

        metrics = calculate_metrics(
            maps
        )

        summary.append(
            {
                "name": setting[
                    "name"
                ],
                "threshold": setting[
                    "threshold"
                ],
                "temperature": setting[
                    "temperature"
                ],
                **metrics,
            }
        )

        print(
            "global mean:",
            f"{metrics['mean']:.4f}",
        )

        print(
            "global max:",
            f"{metrics['max']:.4f}",
        )

        print(
            "area > 0.5:",
            f"{metrics['area_05']:.4f}",
        )

        print(
            "area > 0.7:",
            f"{metrics['area_07']:.4f}",
        )

        print(
            "area > 0.8:",
            f"{metrics['area_08']:.4f}",
        )

        print(
            "area > 0.9:",
            f"{metrics['area_09']:.4f}",
        )

        # 每帧也打印
        print_frame_metrics(
            setting,
            maps,
        )

    # ========================================================
    # 横向 comparison GIF
    # ========================================================

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "all_settings_comparison.gif",
    )

    save_comparison_gif(
        frames,
        setting_maps,
        minmax_maps,
        comparison_path,
    )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "summary.csv",
    )

    save_summary_csv(
        summary,
        csv_path,
    )

    # ========================================================
    # 最终 summary
    # ========================================================

    print(
        "\n\n"
        "======================================"
    )

    print(
        "SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"{'name':<18} "
        f"{'tau':>7} "
        f"{'T':>7} "
        f"{'mean':>8} "
        f"{'>0.5':>8} "
        f"{'>0.7':>8} "
        f"{'>0.9':>8} "
        f"{'max':>8}"
    )

    for row in summary:

        print(
            f"{row['name']:<18} "
            f"{row['threshold']:>7.3f} "
            f"{row['temperature']:>7.3f} "
            f"{row['mean']:>8.4f} "
            f"{row['area_05']:>8.4f} "
            f"{row['area_07']:>8.4f} "
            f"{row['area_09']:>8.4f} "
            f"{row['max']:>8.4f}"
        )

    print(
        "\nOutput directory:",
        OUTPUT_DIR,
    )

    print(
        "Comparison GIF:",
        comparison_path,
    )

    print(
        "Summary CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()