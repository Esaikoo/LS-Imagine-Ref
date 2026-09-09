import os
import sys
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from matplotlib import colormaps


# ============================================================
# 路径初始化
#
# 当前文件：
# LS-Imagine-Ref/relevance_map/zoom_diagnostic.py
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

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

# LS-Imagine 中有相对路径，
# 所以 import envs 前固定 cwd 到项目根目录
os.chdir(PROJECT_ROOT)


# ============================================================
# 现在才导入项目代码
# ============================================================

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
    / "maskclip_test"
    / "20260901_170642_050808_mode_A"
    / "raw_16frames.npz"
)

PROMPT = "Cut a tree"

# 你前面 Forced Zoom 使用的是 Frame 07
FRAME_INDEX = 7

# 当前正式 relevance 参数
RELEVANCE_THRESHOLD = 0.300
RELEVANCE_TEMPERATURE = 0.030

# 控制变量：
# 相同中心，不同 zoom factor
ZOOM_FACTORS = [
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
]

INTERPOLATION_MODES = [
    "bilinear",
    "nearest",
]

ALPHA = 0.50

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "relevance_test"
    / f"zoom_diagnostic_{timestamp}"
)


# ============================================================
# MineDojo obs
# ============================================================

def frame_to_obs(
    frame_hwc,
):
    """
    [H,W,3]
        ->
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
# sigmoid fixed relevance
# ============================================================

def cosine_to_relevance(
    similarity,
    grid_size,
):
    """
    similarity:
        [N]

    Returns:
        small_map: [10,16]
        big_map:   [160,256]
    """

    grid_h, grid_w = grid_size

    similarity = similarity.reshape(
        grid_h,
        grid_w,
    )

    x = (
        similarity
        - RELEVANCE_THRESHOLD
    ) / RELEVANCE_TEMPERATURE

    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    small_map = (
        1.0
        /
        (
            1.0
            + np.exp(-x)
        )
    ).astype(
        np.float32
    )

    tensor = torch.from_numpy(
        small_map
    ).float()

    tensor = tensor[
        None,
        None,
    ]

    big_map = F.interpolate(
        tensor,
        size=(
            160,
            256,
        ),
        mode="bilinear",
        align_corners=False,
    )

    big_map = (
        big_map[
            0,
            0,
        ]
        .cpu()
        .numpy()
    )

    return (
        small_map,
        big_map,
    )


# ============================================================
# 单张图 -> MineCLIP cosine
# ============================================================

@torch.no_grad()
def compute_similarity(
    clip,
    frame_hwc,
):
    obs = frame_to_obs(
        frame_hwc
    )

    bundle = clip.get_frame_bundle(
        obs
    )

    similarity = (
        clip.get_patch_similarity(
            bundle,
            [PROMPT],
        )
    )

    # [1,1,160]
    similarity = (
        similarity[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        similarity,
        bundle["grid_size"],
    )


# ============================================================
# cosine statistics
# ============================================================

def get_metrics(
    similarity,
    relevance,
):
    metrics = {
        "cos_min": float(
            similarity.min()
        ),

        "cos_mean": float(
            similarity.mean()
        ),

        "cos_max": float(
            similarity.max()
        ),

        "cos_p50": float(
            np.percentile(
                similarity,
                50,
            )
        ),

        "cos_p90": float(
            np.percentile(
                similarity,
                90,
            )
        ),

        "cos_p95": float(
            np.percentile(
                similarity,
                95,
            )
        ),

        "cos_range": float(
            similarity.max()
            - similarity.min()
        ),

        # P95-P50：
        # 可以很好反映“高相关 Patch 是否
        # 和普通 Patch 拉开距离”
        "cos_p95_p50": float(
            np.percentile(
                similarity,
                95,
            )
            -
            np.percentile(
                similarity,
                50,
            )
        ),

        "rel_mean": float(
            relevance.mean()
        ),

        "rel_max": float(
            relevance.max()
        ),

        "rel_area_05": float(
            np.mean(
                relevance > 0.5
            )
        ),

        "rel_area_07": float(
            np.mean(
                relevance > 0.7
            )
        ),
    }

    return metrics


# ============================================================
# overlay
# ============================================================

def make_overlay(
    frame,
    relevance,
):
    relevance = np.clip(
        relevance,
        0.0,
        1.0,
    )

    heatmap = (
        colormaps["turbo"](
            relevance
        )[..., :3]
        * 255.0
    ).astype(
        np.float32
    )

    frame = frame.astype(
        np.float32
    )

    overlay = (
        (1.0 - ALPHA)
        * frame
        +
        ALPHA
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
# 复制 LS-Imagine 原 Zoom box 选择逻辑
#
# 不改变正式代码，只用于诊断。
# ============================================================

def get_ls_imagine_crop_box(
    mask_01,
    gaussian,
    device,
    stride=1,
):
    """
    Returns:
        (x1, y1, x2, y2)
        extra_info
    """

    H, W = mask_01.shape

    # --------------------------------------------------------
    # 和 ConcentrationReward.generate_zoom_in_frame 一样
    # --------------------------------------------------------

    threshold_value = (
        (
            np.max(mask_01)
            +
            np.min(mask_01)
        )
        / 2.0
        +
        np.std(mask_01)
    )

    _, binary_image = cv2.threshold(
        mask_01,
        threshold_value,
        1,
        cv2.THRESH_BINARY,
    )

    open_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                16,
                10,
            ),
        )
    )

    binary_image = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    contours, _ = cv2.findContours(
        binary_image.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        raise RuntimeError(
            "No contour found in relevance map."
        )

    max_mean_value = -1.0
    max_area_ratio = 0.0

    centroid_x = 0
    centroid_y = 0

    for contour in contours:

        contour_mask = np.zeros_like(
            mask_01,
            dtype=np.uint8,
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            1,
            thickness=cv2.FILLED,
        )

        values = mask_01[
            contour_mask == 1
        ]

        if len(values) == 0:
            continue

        mean_val = float(
            np.mean(values)
        )

        M = cv2.moments(
            contour
        )

        if M["m00"] == 0:
            continue

        cx = int(
            M["m10"]
            /
            M["m00"]
        )

        cy = int(
            M["m01"]
            /
            M["m00"]
        )

        if mean_val > max_mean_value:

            max_mean_value = (
                mean_val
            )

            max_area_ratio = (
                np.sum(
                    contour_mask
                )
                /
                contour_mask.size
            )

            centroid_x = cx
            centroid_y = cy

    # --------------------------------------------------------
    # LS-Imagine 原代码：
    #
    # area / gaussian(center)
    # --------------------------------------------------------

    gaussian_value = float(
        gaussian[
            centroid_y,
            centroid_x,
        ]
    )

    gaussian_value = max(
        gaussian_value,
        1e-6,
    )

    num_above_threshold = (
        max_area_ratio
        /
        gaussian_value
    )

    proportion = torch.tensor(
        [
            num_above_threshold
        ],
        device=device,
        dtype=torch.float32,
    )

    proportion = torch.clamp(
        proportion,
        0.01,
        1.0,
    )

    sqrt_proportion = (
        torch.sqrt(
            proportion
        )
    )

    window_width = int(
        (
            sqrt_proportion
            * W
        ).int()[0].item()
    )

    window_height = int(
        (
            sqrt_proportion
            * H
        ).int()[0].item()
    )

    window_width = max(
        1,
        min(
            W,
            window_width,
        ),
    )

    window_height = max(
        1,
        min(
            H,
            window_height,
        ),
    )

    # --------------------------------------------------------
    # 在整个 relevance map 上寻找
    # 平均值最高的窗口
    # --------------------------------------------------------

    mask_tensor = (
        torch.from_numpy(
            mask_01
        )
        .float()
        .to(device)
        [None, None]
    )

    kernel = torch.ones(
        (
            1,
            1,
            window_height,
            window_width,
        ),
        device=device,
    )

    conv_result = F.conv2d(
        mask_tensor,
        kernel,
        stride=stride,
    )

    best_value, best_idx = (
        torch.max(
            conv_result.view(-1),
            dim=0,
        )
    )

    best_y, best_x = divmod(
        best_idx.item(),
        conv_result.shape[-1],
    )

    x1 = int(
        best_x
        * stride
    )

    y1 = int(
        best_y
        * stride
    )

    x2 = x1 + window_width
    y2 = y1 + window_height

    best_mean = float(
        (
            best_value
            /
            (
                window_width
                *
                window_height
            )
        ).item()
    )

    extra_info = {
        "threshold_value": (
            threshold_value
        ),

        "contour_mean": (
            max_mean_value
        ),

        "max_area_ratio": (
            max_area_ratio
        ),

        "gaussian_value": (
            gaussian_value
        ),

        "num_above_threshold": (
            num_above_threshold
        ),

        "window_width": (
            window_width
        ),

        "window_height": (
            window_height
        ),

        "best_window_mean": (
            best_mean
        ),
    }

    return (
        (
            x1,
            y1,
            x2,
            y2,
        ),
        extra_info,
    )


# ============================================================
# 根据固定中心 + zoom factor 生成 crop box
# ============================================================

def make_centered_box(
    center_x,
    center_y,
    zoom_factor,
    H,
    W,
):
    """
    zoom_factor = 1:
        crop 整张图

    zoom_factor = 2:
        crop H/2 × W/2

    zoom_factor = 3:
        crop H/3 × W/3
    """

    crop_w = int(
        round(
            W
            /
            zoom_factor
        )
    )

    crop_h = int(
        round(
            H
            /
            zoom_factor
        )
    )

    crop_w = max(
        1,
        min(
            crop_w,
            W,
        ),
    )

    crop_h = max(
        1,
        min(
            crop_h,
            H,
        ),
    )

    x1 = int(
        round(
            center_x
            -
            crop_w / 2
        )
    )

    y1 = int(
        round(
            center_y
            -
            crop_h / 2
        )
    )

    # --------------------------------------------------------
    # 保证 crop 不超出边界，
    # 同时尽可能保持尺寸
    # --------------------------------------------------------

    x1 = max(
        0,
        min(
            x1,
            W - crop_w,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            H - crop_h,
        ),
    )

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# crop + resize
# ============================================================

def crop_and_resize(
    frame,
    box,
    mode,
):
    x1, y1, x2, y2 = box

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                crop
            )
        )
        .float()
        .permute(
            2,
            0,
            1,
        )
        [None]
    )

    if mode == "bilinear":

        output = F.interpolate(
            tensor,
            size=(
                160,
                256,
            ),
            mode="bilinear",
            align_corners=False,
        )

    elif mode == "nearest":

        output = F.interpolate(
            tensor,
            size=(
                160,
                256,
            ),
            mode="nearest",
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    output = (
        output[
            0
        ]
        .permute(
            1,
            2,
            0,
        )
        .clamp(
            0,
            255,
        )
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    return output


# ============================================================
# 保存单个 case
# ============================================================

def save_case_image(
    rgb,
    relevance,
    metrics,
    title,
    path,
):
    overlay = make_overlay(
        rgb,
        relevance,
    )

    H, W, _ = rgb.shape

    canvas = Image.new(
        "RGB",
        (
            W * 2,
            H + 55,
        ),
        "black",
    )

    canvas.paste(
        Image.fromarray(
            rgb
        ),
        (
            0,
            55,
        ),
    )

    canvas.paste(
        Image.fromarray(
            overlay
        ),
        (
            W,
            55,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            5,
            5,
        ),
        title,
        fill="white",
    )

    text = (
        f"mean={metrics['cos_mean']:.4f} "
        f"max={metrics['cos_max']:.4f} "
        f"P90={metrics['cos_p90']:.4f} "
        f"P95={metrics['cos_p95']:.4f} "
        f"range={metrics['cos_range']:.4f}"
    )

    draw.text(
        (
            5,
            27,
        ),
        text,
        fill="white",
    )

    canvas.save(
        path
    )


# ============================================================
# 每种 interpolation 做一个大对比图
# ============================================================

def save_mode_comparison(
    results,
    mode,
    path,
):
    selected = [
        x
        for x in results
        if x["mode"] == mode
    ]

    selected = sorted(
        selected,
        key=lambda x: x[
            "zoom_factor"
        ],
    )

    if not selected:
        return

    W = 256
    H = 160

    canvas = Image.new(
        "RGB",
        (
            W * len(selected),
            H * 2 + 60,
        ),
        "black",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for col, item in enumerate(
        selected
    ):

        x = col * W

        canvas.paste(
            Image.fromarray(
                item["rgb"]
            ),
            (
                x,
                60,
            ),
        )

        canvas.paste(
            Image.fromarray(
                item["overlay"]
            ),
            (
                x,
                H + 60,
            ),
        )

        label = (
            f"{item['zoom_factor']:.2f}x "
            f"max={item['metrics']['cos_max']:.3f} "
            f"P95={item['metrics']['cos_p95']:.3f}"
        )

        draw.text(
            (
                x + 4,
                7,
            ),
            label,
            fill="white",
        )

        label2 = (
            f"mean={item['metrics']['cos_mean']:.3f} "
            f"range={item['metrics']['cos_range']:.3f}"
        )

        draw.text(
            (
                x + 4,
                30,
            ),
            label2,
            fill="white",
        )

    canvas.save(
        path
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows,
    path,
):
    fieldnames = [
        "mode",
        "zoom_factor",

        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_width",
        "crop_height",

        "cos_min",
        "cos_mean",
        "cos_max",
        "cos_p50",
        "cos_p90",
        "cos_p95",
        "cos_range",
        "cos_p95_p50",

        "rel_mean",
        "rel_max",
        "rel_area_05",
        "rel_area_07",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key: row[key]
                    for key in fieldnames
                }
            )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. load frames
    # ========================================================

    data = np.load(
        NPZ_PATH
    )

    frames = data[
        "frames"
    ]

    frame = np.ascontiguousarray(
        frames[
            FRAME_INDEX
        ],
        dtype=np.uint8,
    )

    H, W, _ = frame.shape

    print(
        "frame:",
        FRAME_INDEX
    )

    print(
        "shape:",
        frame.shape
    )

    # ========================================================
    # 2. Shared MineCLIP
    # ========================================================

    print(
        "\nLoading shared MineCLIP..."
    )

    clip = MinedojoClipReward()

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
            output_dir=str(
                OUTPUT_DIR
            ),
        )
    )

    assert (
        concentration.model
        is clip.model
    )

    # ========================================================
    # 3. 正常原图
    # ========================================================

    obs = frame_to_obs(
        frame
    )

    score, zoom_prob, threshold = (
        concentration.get_reward(
            obs,
            [PROMPT],
            episode_num=0,
            step_num=FRAME_INDEX,
        )
    )

    original_mask = (
        concentration.mask
        /
        255.0
    )

    original_similarity, grid_size = (
        compute_similarity(
            clip,
            frame,
        )
    )

    _, original_relevance = (
        cosine_to_relevance(
            original_similarity,
            grid_size,
        )
    )

    original_metrics = (
        get_metrics(
            original_similarity,
            original_relevance,
        )
    )

    print(
        "\n"
        "======================================"
    )

    print(
        "ORIGINAL"
    )

    print(
        "======================================"
    )

    print(
        f"cos min="
        f"{original_metrics['cos_min']:.4f}, "
        f"mean="
        f"{original_metrics['cos_mean']:.4f}, "
        f"max="
        f"{original_metrics['cos_max']:.4f}"
    )

    print(
        f"P50="
        f"{original_metrics['cos_p50']:.4f}, "
        f"P90="
        f"{original_metrics['cos_p90']:.4f}, "
        f"P95="
        f"{original_metrics['cos_p95']:.4f}"
    )

    print(
        f"range="
        f"{original_metrics['cos_range']:.4f}, "
        f"P95-P50="
        f"{original_metrics['cos_p95_p50']:.4f}"
    )

    # ========================================================
    # 4. 复制当前 LS-Imagine Zoom box
    # ========================================================

    ls_box, ls_info = (
        get_ls_imagine_crop_box(
            original_mask,
            concentration.gaussian,
            clip.device,
            stride=concentration.stride,
        )
    )

    (
        ls_x1,
        ls_y1,
        ls_x2,
        ls_y2,
    ) = ls_box

    ls_crop_w = (
        ls_x2 - ls_x1
    )

    ls_crop_h = (
        ls_y2 - ls_y1
    )

    center_x = (
        ls_x1
        +
        ls_x2
    ) / 2.0

    center_y = (
        ls_y1
        +
        ls_y2
    ) / 2.0

    zoom_x = (
        W
        /
        ls_crop_w
    )

    zoom_y = (
        H
        /
        ls_crop_h
    )

    current_zoom_factor = (
        zoom_x + zoom_y
    ) / 2.0

    print(
        "\n"
        "======================================"
    )

    print(
        "CURRENT LS-IMAGINE CROP"
    )

    print(
        "======================================"
    )

    print(
        "box:",
        ls_box,
    )

    print(
        "crop size:",
        (
            ls_crop_h,
            ls_crop_w,
        ),
    )

    print(
        "center:",
        (
            center_x,
            center_y,
        ),
    )

    print(
        "effective zoom x:",
        zoom_x,
    )

    print(
        "effective zoom y:",
        zoom_y,
    )

    print(
        "effective zoom mean:",
        current_zoom_factor,
    )

    print(
        "num_above_threshold:",
        ls_info[
            "num_above_threshold"
        ],
    )

    print(
        "best_window_mean:",
        ls_info[
            "best_window_mean"
        ],
    )

    # ========================================================
    # 验证复制的 crop 是否和正式
    # generate_zoom_in_frame() 一致
    # ========================================================

    concentration.check_threshold = -1.0

    actual_zoom, is_check = (
        concentration
        .generate_zoom_in_frame()
    )

    if is_check:

        reproduced_zoom = (
            crop_and_resize(
                frame,
                ls_box,
                mode="bilinear",
            )
        )

        reproduce_mae = float(
            np.mean(
                np.abs(
                    actual_zoom.astype(
                        np.float32
                    )
                    -
                    reproduced_zoom.astype(
                        np.float32
                    )
                )
            )
        )

        print(
            "LS crop reproduction MAE:",
            reproduce_mae,
        )

        Image.fromarray(
            actual_zoom
        ).save(
            OUTPUT_DIR
            /
            "actual_ls_imagine_zoom.png"
        )

    # ========================================================
    # 5. 加入当前 LS Zoom factor
    # ========================================================

    factors = list(
        ZOOM_FACTORS
    )

    # 与已有 factor 差异较大时才加入
    if all(
        abs(
            current_zoom_factor
            - x
        ) > 0.05
        for x in factors
    ):
        factors.append(
            current_zoom_factor
        )

    factors = sorted(
        factors
    )

    print(
        "\nTest factors:",
        factors,
    )

    # ========================================================
    # 6. 保存原图
    # ========================================================

    Image.fromarray(
        frame
    ).save(
        OUTPUT_DIR
        /
        "original_rgb.png"
    )

    Image.fromarray(
        make_overlay(
            frame,
            original_relevance,
        )
    ).save(
        OUTPUT_DIR
        /
        "original_relevance.png"
    )

    # ========================================================
    # 7. 逐 Zoom factor + interpolation 测试
    # ========================================================

    rows = []
    visual_results = []

    clip.vision_forward_count = 0

    print(
        "\n"
        "======================================"
    )

    print(
        "ZOOM DIAGNOSTIC"
    )

    print(
        "======================================"
    )

    for mode in INTERPOLATION_MODES:

        print(
            f"\n---------- {mode} ----------"
        )

        for zoom_factor in factors:

            box = make_centered_box(
                center_x=center_x,
                center_y=center_y,
                zoom_factor=zoom_factor,
                H=H,
                W=W,
            )

            (
                x1,
                y1,
                x2,
                y2,
            ) = box

            zoomed = crop_and_resize(
                frame,
                box,
                mode=mode,
            )

            before = (
                clip.vision_forward_count
            )

            similarity, grid_size = (
                compute_similarity(
                    clip,
                    zoomed,
                )
            )

            after = (
                clip.vision_forward_count
            )

            assert (
                after - before
                == 1
            )

            _, relevance = (
                cosine_to_relevance(
                    similarity,
                    grid_size,
                )
            )

            metrics = get_metrics(
                similarity,
                relevance,
            )

            overlay = make_overlay(
                zoomed,
                relevance,
            )

            print(
                f"{zoom_factor:>5.2f}x | "
                f"crop="
                f"{y2-y1:>3}x{x2-x1:<3} | "
                f"mean="
                f"{metrics['cos_mean']:.4f} | "
                f"max="
                f"{metrics['cos_max']:.4f} | "
                f"P90="
                f"{metrics['cos_p90']:.4f} | "
                f"P95="
                f"{metrics['cos_p95']:.4f} | "
                f"range="
                f"{metrics['cos_range']:.4f} | "
                f"P95-P50="
                f"{metrics['cos_p95_p50']:.4f} | "
                f"rel>0.5="
                f"{metrics['rel_area_05']:.3f}"
            )

            row = {
                "mode": mode,
                "zoom_factor": (
                    zoom_factor
                ),

                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,

                "crop_width": (
                    x2 - x1
                ),

                "crop_height": (
                    y2 - y1
                ),

                **metrics,
            }

            rows.append(
                row
            )

            visual_results.append(
                {
                    "mode": mode,
                    "zoom_factor": (
                        zoom_factor
                    ),
                    "rgb": zoomed,
                    "overlay": overlay,
                    "metrics": metrics,
                }
            )

            factor_name = (
                f"{zoom_factor:.2f}"
                .replace(
                    ".",
                    "p",
                )
            )

            save_case_image(
                rgb=zoomed,
                relevance=relevance,
                metrics=metrics,
                title=(
                    f"{mode} | "
                    f"zoom={zoom_factor:.2f}x | "
                    f"crop="
                    f"{y2-y1}x{x2-x1}"
                ),
                path=(
                    OUTPUT_DIR
                    /
                    (
                        f"{mode}_"
                        f"{factor_name}x.png"
                    )
                ),
            )

    # ========================================================
    # 8. 保存大对比图
    # ========================================================

    for mode in INTERPOLATION_MODES:

        save_mode_comparison(
            visual_results,
            mode,
            OUTPUT_DIR
            /
            f"comparison_{mode}.png",
        )

    # ========================================================
    # 9. 保存 CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        /
        "zoom_diagnostic.csv"
    )

    save_csv(
        rows,
        csv_path,
    )

    # ========================================================
    # 10. 打印相对原图变化
    # ========================================================

    print(
        "\n"
        "======================================"
    )

    print(
        "CHANGE RELATIVE TO ORIGINAL"
    )

    print(
        "======================================"
    )

    print(
        "Original:"
    )

    print(
        f"mean="
        f"{original_metrics['cos_mean']:.4f}, "
        f"max="
        f"{original_metrics['cos_max']:.4f}, "
        f"P95="
        f"{original_metrics['cos_p95']:.4f}, "
        f"range="
        f"{original_metrics['cos_range']:.4f}"
    )

    for row in rows:

        delta_mean = (
            row["cos_mean"]
            -
            original_metrics[
                "cos_mean"
            ]
        )

        delta_max = (
            row["cos_max"]
            -
            original_metrics[
                "cos_max"
            ]
        )

        delta_p95 = (
            row["cos_p95"]
            -
            original_metrics[
                "cos_p95"
            ]
        )

        delta_range = (
            row["cos_range"]
            -
            original_metrics[
                "cos_range"
            ]
        )

        print(
            f"{row['mode']:<8} "
            f"{row['zoom_factor']:>5.2f}x | "
            f"dMean={delta_mean:+.4f} "
            f"dMax={delta_max:+.4f} "
            f"dP95={delta_p95:+.4f} "
            f"dRange={delta_range:+.4f}"
        )

    print(
        "\n"
        "======================================"
    )

    print(
        "DONE"
    )

    print(
        "======================================"
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    print(
        "CSV:",
        csv_path,
    )

    print(
        "Vision forwards during diagnostic:",
        clip.vision_forward_count,
    )


if __name__ == "__main__":
    main()