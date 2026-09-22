#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NACLIP + Temporal(L=1) + raw cosine P95
Natural-Zoom / ScoreStorage offline test

目的
----
1) Natural Zoom 的“触发条件 / crop 方式 / post-zoom Gaussian acceptance”
   尽量按当前 concentration_reward.py 保持不变。

2) 不改当前用于 Natural Zoom 的 map 分支：
      Value patch
        -> Temporal(L=1)
        -> raw cosine
        -> sigmoid((sim - 0.30) / 0.03)
        -> 10x16 -> 160x256
        -> zoom_in_prob / contour / crop / Gaussian post-check

3) 单独增加新的 progress 分支：
      NACLIP K-K spatial refinement
        -> Temporal(L=1)
        -> raw cosine
        -> P95
        -> Progress Score

4) 对每个真正 is_zoomed=True 的 zoom 事件：
      score_on_zoomed(t) = zoomed image 的 NACLIP+Temporal raw P95

   然后在未来真实 trajectory 中搜索：
      第一个 k > t，使
      score_current(k) > score_on_zoomed(t)

   jumping_steps = k - t

5) 一条 trajectory 可以出现多个 accepted zoom，
   每个 zoom 独立搜索 crossing。

输出
----
- zoom_events.csv
    每一个 actual crop attempt 的完整记录：
    accepted/rejected、crop、zoom factor、Gaussian check、
    P95 score_on_zoomed、crossing frame、jumping_steps 等。

- current_progress.csv
    每一帧真实 trajectory 的：
    NACLIP+Temporal raw P95、
    当前 zoom driver Gaussian score / zoom probability / adaptive threshold。

- accepted_zoom_comparisons/
    每个 accepted zoom 一张大图：
      上排：
        source RGB + crop rectangle
        zoomed RGB
        first crossing RGB
        event summary
      下排：
        source zoom-driver sigmoid relevance map
        source NACLIP raw-cosine map
        zoomed NACLIP raw-cosine map
        crossing NACLIP raw-cosine map

- zoom_timeline.png
    trajectory current P95 + accepted zoom score_on_zoomed +
    first crossing 关系。

- jumping_steps.png
    accepted zoom 的 t -> jumping_steps。

说明
----
这个脚本的关键设计是“map 和 progress score 解耦”：
Natural Zoom 仍由当前 production map 决定；
新的 P95 只负责 ScoreStorage-style progress comparison。

放置位置建议：
    <LS-Imagine>/relevance_map/naclip_zoom_p95_test.py

PyCharm 直接运行；无 argparse。
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis


# =====================================================================
# 0. PROJECT
# =====================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

# 如果脚本不放在 <project>/relevance_map/，手工改：
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

TASK_SPECS = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS.exists():
    raise FileNotFoundError(f"PROJECT_ROOT wrong: {TASK_SPECS}")

# 复用前面 relevance_variants.py 中已经验证过的 MineCLIP 实现：
# - Value patch
# - Temporal(L=1)
# - NACLIP K-K
# - cosine
import relevance_map.relevance_variants as rv


# =====================================================================
# 1. GLOBAL CONFIG —— 主要改这里
# =====================================================================

TASK_NAME = "harvest_log_in_plains"

NPZ_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_approach_runs"
    / TASK_NAME
    / "world_42_taskseed_0_20260904_183014"
    / "trajectory.npz"
)

MINECLIP_CKPT = PROJECT_ROOT / "weights" / "mineclip_attn.pth"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "naclip_zoom_p95_test_outputs"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_H = 160
IMAGE_W = 256

TARGET_PROMPTS = ["Cut a tree"]

# ---------------------------------------------------------------------
# 1.1 当前 production map 的 calibration
#     这是 Natural Zoom 分支，不是新的 P95 score。
# ---------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016

# ---------------------------------------------------------------------
# 1.2 当前 ConcentrationReward Natural Zoom 配置
#     默认值与用户当前 concentration_reward.py 对齐。
# ---------------------------------------------------------------------

GAUSSIAN_SIGMA_WEIGHT = 0.5

ZOOM_COOLDOWN_STEPS = 5

SEMANTIC_GATE_ENABLED = False
SEMANTIC_MIN_P95_P50 = 0.025
SEMANTIC_MIN_CC_FRACTION = 0.02

POST_ZOOM_SEMANTIC_GATE_ENABLED = False
POST_ZOOM_MIN_P95_GAIN = 0.0
POST_ZOOM_MIN_CONTRAST_GAIN = -0.005
POST_ZOOM_MIN_CC_RETENTION = 0.50

MAX_ZOOM_FACTOR = 3.5

# 原代码
MORPH_KERNEL = (16, 10)
CROP_STRIDE = 1

# ---------------------------------------------------------------------
# 1.3 新 Progress 分支
# ---------------------------------------------------------------------

# raw P95：对每帧 10x16=160 个 patch 的 raw cosine 取 95th percentile。
P95_PERCENTILE = 95.0

# NACLIP 超参数。
# 不是每个 task 单独调；这里按前面 ablation 使用的全局设置。
NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True

# ScoreStorage 原类默认 max_steps=1000。
SCORESTORAGE_MAX_STEPS = 1000

# crossing 是否要求 k > t。
# ScoreStorage 的未来兑现逻辑应当是未来帧，所以保持 True。
SEARCH_FROM_NEXT_FRAME = True

# ---------------------------------------------------------------------
# 1.4 trajectory range
# ---------------------------------------------------------------------

FRAME_START = 0
FRAME_END = None      # None = 到最后
FRAME_STRIDE = 1

# ---------------------------------------------------------------------
# 1.5 保存 / 可视化
# ---------------------------------------------------------------------

SAVE_ACCEPTED_COMPARISONS = True

# rejected crop attempt 也可以保存，用于看 post-zoom Gaussian check
# 为什么拒绝。默认 False，避免图太多。
SAVE_REJECTED_ATTEMPTS = False

VIS_DPI = 150

# raw cosine 各图统一色标，避免每帧 min-max 拉伸。
RAW_COSINE_VMIN_PERCENTILE = 1.0
RAW_COSINE_VMAX_PERCENTILE = 99.0

# console 不刷屏。
PRINT_EVERY_N_FRAMES = 20


# =====================================================================
# 2. Utility
# =====================================================================

def sigmoid_scalar(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def upsample_patch_grid(grid: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(
        np.asarray(grid, dtype=np.float32)
    )[None, None]

    y = F.interpolate(
        x,
        size=(IMAGE_H, IMAGE_W),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return y.numpy().astype(np.float32)


def make_center_gaussian() -> np.ndarray:
    sigma_x = IMAGE_W * GAUSSIAN_SIGMA_WEIGHT
    sigma_y = IMAGE_H * GAUSSIAN_SIGMA_WEIGHT

    x = np.linspace(-IMAGE_W // 2, IMAGE_W // 2, IMAGE_W)
    y = np.linspace(-IMAGE_H // 2, IMAGE_H // 2, IMAGE_H)

    xx, yy = np.meshgrid(x, y)

    g = np.exp(
        -(
            xx ** 2 / (2.0 * sigma_x ** 2)
            + yy ** 2 / (2.0 * sigma_y ** 2)
        )
    )

    return g.astype(np.float32)


CENTER_GAUSSIAN = make_center_gaussian()
CENTER_GAUSSIAN_MEAN = float(np.mean(CENTER_GAUSSIAN))


class RunningStats:
    """
    与 concentration_reward.py ThresholdBuffer 一致：
    population std = sqrt(M2 / n)
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, number: float):
        number = float(number)
        self.n += 1
        delta = number - self.mean
        self.mean += delta / self.n
        delta2 = number - self.mean
        self.M2 += delta * delta2

    def std_dev(self) -> float:
        if self.n > 1:
            return float(math.sqrt(self.M2 / self.n))
        return 0.0

    def threshold(self) -> float:
        if self.n > 0:
            return float(self.mean + self.std_dev())
        return 1.0


def largest_connected_component_fraction(
    relevance_grid: np.ndarray,
    threshold: float = 0.5,
) -> float:
    binary = (
        np.asarray(relevance_grid) > threshold
    ).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(np.max(areas) / binary.size)


# =====================================================================
# 3. MineCLIP branches
# =====================================================================

@dataclass
class FrameFeatures:
    grid_h: int
    grid_w: int

    # current production map branch:
    # [P,N] raw cosine after Value + Temporal(L=1)
    current_similarity: np.ndarray

    # new progress branch:
    # [P,N] raw cosine after NACLIP + Temporal(L=1)
    progress_similarity: np.ndarray


class BranchTester:
    def __init__(self, ckpt: Path):
        # relevance_variants.MineCLIPTester 的 NACLIP 读取模块级参数，
        # 所以先覆盖为本脚本明确配置。
        rv.NACLIP_GAUSSIAN_STD = float(NACLIP_GAUSSIAN_STD)
        rv.NACLIP_GAUSSIAN_WEIGHT = float(NACLIP_GAUSSIAN_WEIGHT)
        rv.NACLIP_INCLUDE_CLS = bool(NACLIP_INCLUDE_CLS)

        self.tester = rv.MineCLIPTester(ckpt)
        self.tester.text_feats(TARGET_PROMPTS)

    @torch.inference_mode()
    def extract(self, frame: np.ndarray) -> FrameFeatures:
        x, last, gh, gw = self.tester.encode_to_last(frame)

        # -------------------------------------------------------------
        # Map branch:
        # Value -> Temporal(L=1) -> cosine
        # -------------------------------------------------------------
        value = self.tester.raw_value_patch(x, last)
        value_temporal = self.tester.temporal_l1(value)

        current_sim = self.tester.cosine(
            value_temporal,
            TARGET_PROMPTS,
        )[0]  # [P,N]

        # -------------------------------------------------------------
        # Progress branch:
        # NACLIP -> Temporal(L=1) -> cosine
        # -------------------------------------------------------------
        naclip = self.tester.naclip_patch(
            x,
            last,
            gh,
            gw,
        )

        naclip_temporal = self.tester.temporal_l1(
            naclip
        )

        progress_sim = self.tester.cosine(
            naclip_temporal,
            TARGET_PROMPTS,
        )[0]  # [P,N]

        return FrameFeatures(
            grid_h=int(gh),
            grid_w=int(gw),
            current_similarity=(
                current_sim.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
            progress_similarity=(
                progress_sim.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
        )


# =====================================================================
# 4. Map branch + Progress branch
# =====================================================================

@dataclass
class MapBranchResult:
    masks: np.ndarray             # [P,H,W]
    max_map: np.ndarray           # [H,W]
    gaussian_score: float
    zoom_in_prob: float

    raw_p50: float
    raw_p95: float
    raw_p95_p50: float

    largest_cc_fraction: float

    relevance_grid_max: np.ndarray  # [gh,gw]


@dataclass
class ProgressBranchResult:
    p95: float
    raw_grid_max: np.ndarray        # [gh,gw]
    raw_map: np.ndarray             # [H,W]


def compute_map_branch(
    current_similarity: np.ndarray,
    gh: int,
    gw: int,
) -> MapBranchResult:

    sim = np.asarray(
        current_similarity,
        dtype=np.float32,
    )  # [P,N]

    sim_max = np.max(
        sim,
        axis=0,
    )

    raw_p50 = float(
        np.percentile(
            sim_max,
            50,
        )
    )

    raw_p95 = float(
        np.percentile(
            sim_max,
            95,
        )
    )

    relevance = stable_sigmoid_np(
        (
            sim
            - RELEVANCE_THRESHOLD
        )
        / RELEVANCE_TEMPERATURE
    )  # [P,N]

    P, N = relevance.shape

    if N != gh * gw:
        raise RuntimeError(
            f"patch count mismatch: {N} != {gh}*{gw}"
        )

    masks = []

    for p in range(P):
        grid = relevance[p].reshape(
            gh,
            gw,
        )

        masks.append(
            upsample_patch_grid(
                grid
            )
        )

    masks = np.stack(
        masks,
        axis=0,
    ).astype(np.float32)

    max_map = np.max(
        masks,
        axis=0,
    )

    gaussian_score = 0.0

    for mask in masks:
        gaussian_score += (
            float(
                np.mean(
                    mask * CENTER_GAUSSIAN
                )
            )
            / CENTER_GAUSSIAN_MEAN
        )

    kurt = float(
        kurtosis(
            max_map.flatten()
        )
    )

    normalized_kurtosis = sigmoid_scalar(
        kurt
    )

    zoom_in_prob = (
        normalized_kurtosis
        * (
            float(np.max(max_map))
            - float(np.mean(max_map))
        )
    )

    relevance_grid_max = np.max(
        relevance,
        axis=0,
    ).reshape(
        gh,
        gw,
    )

    cc_fraction = largest_connected_component_fraction(
        relevance_grid_max,
        threshold=0.5,
    )

    return MapBranchResult(
        masks=masks,
        max_map=max_map,
        gaussian_score=float(
            gaussian_score
        ),
        zoom_in_prob=float(
            zoom_in_prob
        ),
        raw_p50=raw_p50,
        raw_p95=raw_p95,
        raw_p95_p50=float(
            raw_p95 - raw_p50
        ),
        largest_cc_fraction=float(
            cc_fraction
        ),
        relevance_grid_max=(
            relevance_grid_max.astype(
                np.float32
            )
        ),
    )


def compute_progress_branch(
    progress_similarity: np.ndarray,
    gh: int,
    gw: int,
) -> ProgressBranchResult:

    sim = np.asarray(
        progress_similarity,
        dtype=np.float32,
    )

    # 多 prompt 时，和之前 ablation 一样：
    # 每个 patch 取最相关 prompt，再计算全图 P95。
    sim_max = np.max(
        sim,
        axis=0,
    )

    p95 = float(
        np.percentile(
            sim_max,
            P95_PERCENTILE,
        )
    )

    raw_grid = sim_max.reshape(
        gh,
        gw,
    ).astype(np.float32)

    raw_map = upsample_patch_grid(
        raw_grid
    )

    return ProgressBranchResult(
        p95=p95,
        raw_grid_max=raw_grid,
        raw_map=raw_map,
    )


# =====================================================================
# 5. Exact Natural Zoom logic
# =====================================================================

@dataclass
class ZoomCandidate:
    produced_crop: bool
    gate_reason: str

    zoomed_frame: Optional[np.ndarray]

    have_center: bool = False

    threshold_value: float = np.nan
    max_area_ratio: float = np.nan
    num_above_threshold: float = np.nan

    raw_zoom_factor: float = 1.0
    actual_zoom_factor: float = 1.0

    crop_x1: int = -1
    crop_y1: int = -1
    crop_x2: int = -1
    crop_y2: int = -1

    best_value_on_mask: float = np.nan


class NaturalZoomSimulator:
    """
    mirror:
      ConcentrationReward.get_reward()
      ConcentrationReward.generate_zoom_in_frame()
      ConcentrationReward.compute_reward_on_zoomed_image()

    Map branch 仍是 current production map。
    """

    def __init__(self):
        self.check_threshold_buffer = RunningStats()
        self.gaussian_buffer = RunningStats()

        self.last_zoom_attempt_step = None

    def update_current_statistics(
        self,
        map_result: MapBranchResult,
    ) -> float:

        self.gaussian_buffer.add(
            map_result.gaussian_score
        )

        self.check_threshold_buffer.add(
            map_result.zoom_in_prob
        )

        return self.check_threshold_buffer.threshold()

    def generate_candidate(
        self,
        frame: np.ndarray,
        step: int,
        map_result: MapBranchResult,
        check_threshold: float,
    ) -> ZoomCandidate:

        zoom_in_prob = map_result.zoom_in_prob

        # Gate 0
        if check_threshold >= zoom_in_prob:
            return ZoomCandidate(
                produced_crop=False,
                gate_reason="adaptive_threshold",
                zoomed_frame=None,
            )

        # Gate 1
        if (
            ZOOM_COOLDOWN_STEPS > 0
            and self.last_zoom_attempt_step is not None
            and (
                step
                - self.last_zoom_attempt_step
                <= ZOOM_COOLDOWN_STEPS
            )
        ):
            return ZoomCandidate(
                produced_crop=False,
                gate_reason="cooldown",
                zoomed_frame=None,
            )

        # Gate 2
        semantic_ok = (
            map_result.raw_p95_p50
            >= SEMANTIC_MIN_P95_P50
            and map_result.largest_cc_fraction
            >= SEMANTIC_MIN_CC_FRACTION
        )

        if (
            SEMANTIC_GATE_ENABLED
            and not semantic_ok
        ):
            return ZoomCandidate(
                produced_crop=False,
                gate_reason="semantic_gate",
                zoomed_frame=None,
            )

        heatmap = np.asarray(
            map_result.max_map,
            dtype=np.float32,
        )

        threshold_value = (
            (
                float(np.max(heatmap))
                + float(np.min(heatmap))
            )
            / 2.0
            + float(np.std(heatmap))
        )

        _, binary_image = cv2.threshold(
            heatmap,
            threshold_value,
            1,
            cv2.THRESH_BINARY,
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            MORPH_KERNEL,
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
            return ZoomCandidate(
                produced_crop=False,
                gate_reason="no_contour",
                zoomed_frame=None,
                threshold_value=threshold_value,
            )

        max_mean_value = 0.0
        max_area_ratio = 0.0
        centroid_x = 0
        centroid_y = 0

        # 与当前代码一致：
        # have_center 对“任意 contour”检测，而不是只对最终选中的 contour。
        have_center = False

        for contour in contours:
            contour_mask = np.zeros_like(
                heatmap,
                dtype=np.float32,
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                1,
                thickness=cv2.FILLED,
            )

            mean_val = float(
                np.mean(
                    heatmap[
                        contour_mask == 1
                    ]
                )
            )

            M = cv2.moments(
                contour
            )

            if M["m00"] == 0:
                continue

            cx = int(
                M["m10"]
                / M["m00"]
            )

            cy = int(
                M["m01"]
                / M["m00"]
            )

            if (
                CENTER_GAUSSIAN[
                    cy,
                    cx
                ]
                >= CENTER_GAUSSIAN_MEAN
            ):
                have_center = True

            if mean_val > max_mean_value:
                max_mean_value = mean_val

                max_area_ratio = float(
                    np.sum(
                        contour_mask
                    )
                    / contour_mask.size
                )

                centroid_x = cx
                centroid_y = cy

        if max_area_ratio <= 0:
            return ZoomCandidate(
                produced_crop=False,
                gate_reason="invalid_contour",
                zoomed_frame=None,
                threshold_value=threshold_value,
            )

        num_above_threshold = (
            max_area_ratio
            / float(
                CENTER_GAUSSIAN[
                    centroid_y,
                    centroid_x
                ]
            )
        )

        proportion = float(
            np.clip(
                num_above_threshold,
                0.01,
                1.0,
            )
        )

        raw_zoom_factor = (
            1.0
            / math.sqrt(
                proportion
            )
        )

        if MAX_ZOOM_FACTOR is not None:
            min_proportion = (
                1.0
                / (
                    float(
                        MAX_ZOOM_FACTOR
                    ) ** 2
                )
            )

            proportion = float(
                np.clip(
                    proportion,
                    min_proportion,
                    1.0,
                )
            )

        sqrt_proportion = math.sqrt(
            proportion
        )

        H, W, _ = frame.shape

        cur_window_width = int(
            math.ceil(
                sqrt_proportion
                * W
            )
        )

        cur_window_height = int(
            math.ceil(
                sqrt_proportion
                * H
            )
        )

        cur_window_width = max(
            1,
            min(
                cur_window_width,
                W,
            ),
        )

        cur_window_height = max(
            1,
            min(
                cur_window_height,
                H,
            ),
        )

        actual_zoom_factor = max(
            W / float(
                cur_window_width
            ),
            H / float(
                cur_window_height
            ),
        )

        relevance_tensor = (
            torch.from_numpy(
                heatmap
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        kernel = torch.ones(
            (
                1,
                1,
                cur_window_height,
                cur_window_width,
            ),
            dtype=torch.float32,
        )

        conv_result = F.conv2d(
            relevance_tensor,
            kernel,
            stride=CROP_STRIDE,
        )

        best_value, best_idx = torch.max(
            conv_result.view(-1),
            dim=0,
        )

        kernel_size = (
            cur_window_height
            * cur_window_width
        )

        best_value_on_mask = float(
            best_value.item()
            / kernel_size
        )

        out_w = int(
            conv_result.shape[-1]
        )

        best_y, best_x = divmod(
            int(
                best_idx.item()
            ),
            out_w,
        )

        x1 = int(
            best_x
            * CROP_STRIDE
        )

        y1 = int(
            best_y
            * CROP_STRIDE
        )

        x2 = int(
            x1
            + cur_window_width
        )

        y2 = int(
            y1
            + cur_window_height
        )

        # 与 concentration_reward.py 一样使用 torch nearest resize，
        # 避免 cv2 INTER_NEAREST 在坐标映射上产生细微差异。
        crop = torch.from_numpy(
            frame[y1:y2, x1:x2, :]
        ).unsqueeze(0).float().permute(0, 3, 1, 2)

        zoomed = (
            F.interpolate(
                crop,
                size=(H, W),
                mode="nearest",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
            .numpy()
            .astype(np.uint8)
        )

        # 与当前 concentration_reward.py 一样：
        # 只要真正生成过 crop，就开始 cooldown；
        # 即使后面 post-zoom Gaussian check 拒绝。
        self.last_zoom_attempt_step = int(
            step
        )

        return ZoomCandidate(
            produced_crop=True,
            gate_reason="natural_zoom_candidate",
            zoomed_frame=zoomed,
            have_center=bool(
                have_center
            ),
            threshold_value=float(
                threshold_value
            ),
            max_area_ratio=float(
                max_area_ratio
            ),
            num_above_threshold=float(
                num_above_threshold
            ),
            raw_zoom_factor=float(
                raw_zoom_factor
            ),
            actual_zoom_factor=float(
                actual_zoom_factor
            ),
            crop_x1=x1,
            crop_y1=y1,
            crop_x2=x2,
            crop_y2=y2,
            best_value_on_mask=float(
                best_value_on_mask
            ),
        )

    def post_zoom_accept(
        self,
        current_map: MapBranchResult,
        zoomed_map: MapBranchResult,
        candidate: ZoomCandidate,
    ) -> Dict[str, float | bool]:

        gaussian_std = self.gaussian_buffer.std_dev()

        gaussian_required = (
            current_map.gaussian_score
            + 2.0 * gaussian_std
        )

        gaussian_gain_ok = (
            zoomed_map.gaussian_score
            >= gaussian_required
        )

        zoom_p95_gain = (
            zoomed_map.raw_p95
            - current_map.raw_p95
        )

        zoom_contrast_gain = (
            zoomed_map.raw_p95_p50
            - current_map.raw_p95_p50
        )

        if (
            current_map.largest_cc_fraction
            > 1e-8
        ):
            cc_retention = (
                zoomed_map.largest_cc_fraction
                / current_map.largest_cc_fraction
            )
        else:
            cc_retention = (
                1.0
                if zoomed_map.largest_cc_fraction
                > 0
                else 0.0
            )

        semantic_gain_ok = (
            zoom_p95_gain
            >= POST_ZOOM_MIN_P95_GAIN
            and zoom_contrast_gain
            >= POST_ZOOM_MIN_CONTRAST_GAIN
            and cc_retention
            >= POST_ZOOM_MIN_CC_RETENTION
        )

        if POST_ZOOM_SEMANTIC_GATE_ENABLED:
            is_zoomed = (
                gaussian_gain_ok
                and semantic_gain_ok
            )
        else:
            is_zoomed = (
                gaussian_gain_ok
            )

        jump = (
            bool(is_zoomed)
            and bool(
                candidate.have_center
            )
        )

        return {
            "gaussian_buffer_std": float(
                gaussian_std
            ),
            "gaussian_required": float(
                gaussian_required
            ),
            "gaussian_gain_ok": bool(
                gaussian_gain_ok
            ),
            "zoom_p95_gain": float(
                zoom_p95_gain
            ),
            "zoom_contrast_gain": float(
                zoom_contrast_gain
            ),
            "cc_retention": float(
                cc_retention
            ),
            "semantic_gain_ok": bool(
                semantic_gain_ok
            ),
            "is_zoomed": bool(
                is_zoomed
            ),
            "jump": bool(
                jump
            ),
        }


# =====================================================================
# 6. Visualization
# =====================================================================

def annotated_source(
    frame: np.ndarray,
    candidate: ZoomCandidate,
) -> np.ndarray:
    out = frame.copy()

    if candidate.produced_crop:
        cv2.rectangle(
            out,
            (
                candidate.crop_x1,
                candidate.crop_y1,
            ),
            (
                candidate.crop_x2 - 1,
                candidate.crop_y2 - 1,
            ),
            (
                255,
                255,
                255,
            ),
            2,
        )

    return out


def save_accepted_comparison(
    path: Path,
    event: Dict,
    source_frame: np.ndarray,
    zoomed_frame: np.ndarray,
    crossing_frame: Optional[np.ndarray],
    source_zoom_driver_map: np.ndarray,
    source_progress_map: np.ndarray,
    zoomed_progress_map: np.ndarray,
    crossing_progress_map: Optional[np.ndarray],
    raw_vmin: float,
    raw_vmax: float,
):
    """
    2 x 4，大图，避免之前过多小 panel。

    top:
      source + crop
      zoomed
      crossing
      summary text

    bottom:
      production zoom-driver map
      source progress raw cosine
      zoomed progress raw cosine
      crossing progress raw cosine
    """
    fig, ax = plt.subplots(
        2,
        4,
        figsize=(20, 9),
        squeeze=False,
    )

    # ---------------- top row ----------------
    ax[0, 0].imshow(
        annotated_source(
            source_frame,
            ZoomCandidate(
                produced_crop=True,
                gate_reason="",
                zoomed_frame=None,
                crop_x1=int(event["crop_x1"]),
                crop_y1=int(event["crop_y1"]),
                crop_x2=int(event["crop_x2"]),
                crop_y2=int(event["crop_y2"]),
            ),
        )
    )

    ax[0, 0].set_title(
        f"source frame t={event['zoom_frame']}\n"
        f"current P95={event['score_current_at_t']:.4f}"
    )

    ax[0, 1].imshow(
        zoomed_frame
    )

    ax[0, 1].set_title(
        "accepted zoomed frame\n"
        f"score_on_zoomed={event['score_on_zoomed']:.4f}"
    )

    if crossing_frame is not None:
        ax[0, 2].imshow(
            crossing_frame
        )

        ax[0, 2].set_title(
            f"first crossing k={event['crossing_frame']}\n"
            f"score_current={event['crossing_score']:.4f} "
            f"> {event['score_on_zoomed']:.4f}"
        )
    else:
        ax[0, 2].imshow(
            np.zeros_like(
                source_frame
            )
        )

        ax[0, 2].set_title(
            "no crossing before trajectory end"
        )

    ax[0, 3].axis(
        "off"
    )

    jumping_text = (
        "unresolved"
        if not np.isfinite(
            event["jumping_steps"]
        )
        else str(
            int(
                event["jumping_steps"]
            )
        )
    )

    summary = (
        f"jumping_steps = {jumping_text}\n\n"
        f"zoom_prob = {event['zoom_in_prob']:.4f}\n"
        f"adaptive_threshold = {event['check_threshold']:.4f}\n\n"
        f"Gaussian:\n"
        f"  current = {event['current_gaussian']:.4f}\n"
        f"  zoomed  = {event['zoomed_gaussian']:.4f}\n"
        f"  required= {event['gaussian_required']:.4f}\n\n"
        f"zoom factor:\n"
        f"  raw    = {event['raw_zoom_factor']:.2f}x\n"
        f"  actual = {event['actual_zoom_factor']:.2f}x\n\n"
        f"have_center={event['have_center']}\n"
        f"jump={event['jump']}"
    )

    ax[0, 3].text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=11,
        transform=ax[0, 3].transAxes,
        family="monospace",
    )

    # ---------------- bottom row ----------------
    ax[1, 0].imshow(
        source_zoom_driver_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 0].set_title(
        "source production map\n"
        "Value + Temporal + sigmoid"
    )

    ax[1, 1].imshow(
        source_progress_map,
        cmap="viridis",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    ax[1, 1].set_title(
        "source progress raw cosine\n"
        "NACLIP + Temporal"
    )

    ax[1, 2].imshow(
        zoomed_progress_map,
        cmap="viridis",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    ax[1, 2].set_title(
        "zoomed progress raw cosine\n"
        f"P95={event['score_on_zoomed']:.4f}"
    )

    if crossing_progress_map is not None:
        ax[1, 3].imshow(
            crossing_progress_map,
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 3].set_title(
            "crossing progress raw cosine\n"
            f"P95={event['crossing_score']:.4f}"
        )
    else:
        ax[1, 3].imshow(
            np.zeros_like(
                source_progress_map
            ),
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 3].set_title(
            "no crossing"
        )

    for a in ax.reshape(-1):
        a.axis(
            "off"
        )

    fig.suptitle(
        "Natural Zoom accepted event + ScoreStorage-style first crossing",
        fontsize=15,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_rejected_attempt(
    path: Path,
    frame: np.ndarray,
    zoomed: np.ndarray,
    current_map: np.ndarray,
    zoomed_map: np.ndarray,
    event: Dict,
):
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    ax[0, 0].imshow(
        frame
    )

    ax[0, 0].set_title(
        f"source t={event['zoom_frame']}"
    )

    ax[0, 1].imshow(
        zoomed
    )

    ax[0, 1].set_title(
        "candidate crop"
    )

    ax[1, 0].imshow(
        current_map,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    ax[1, 0].set_title(
        f"current G={event['current_gaussian']:.3f}"
    )

    ax[1, 1].imshow(
        zoomed_map,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    ax[1, 1].set_title(
        f"zoomed G={event['zoomed_gaussian']:.3f}\n"
        f"required={event['gaussian_required']:.3f}"
    )

    for a in ax.reshape(-1):
        a.axis(
            "off"
        )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_timeline(
    path: Path,
    frame_indices: np.ndarray,
    current_scores: np.ndarray,
    accepted_events: List[Dict],
):
    fig, ax = plt.subplots(
        figsize=(14, 7),
    )

    ax.plot(
        frame_indices,
        current_scores,
        label="current NACLIP+Temporal raw P95",
        linewidth=1.8,
    )

    first_label = True

    for e in accepted_events:
        t = int(
            e["zoom_frame"]
        )

        z = float(
            e["score_on_zoomed"]
        )

        ax.scatter(
            [t],
            [z],
            marker="^",
            s=70,
            label=(
                "accepted zoom score_on_zoomed"
                if first_label
                else None
            ),
        )

        if np.isfinite(
            e["crossing_frame"]
        ):
            k = int(
                e["crossing_frame"]
            )

            cs = float(
                e["crossing_score"]
            )

            ax.scatter(
                [k],
                [cs],
                marker="o",
                s=45,
            )

            ax.plot(
                [t, k],
                [z, cs],
                linestyle="--",
                alpha=0.45,
            )

            ax.text(
                (t + k) / 2.0,
                max(
                    z,
                    cs,
                ),
                f"J={int(e['jumping_steps'])}",
                fontsize=8,
            )

        first_label = False

    ax.set_xlabel(
        "Trajectory frame index"
    )

    ax.set_ylabel(
        "raw P95 progress score"
    )

    ax.set_title(
        "NACLIP + Temporal raw P95: accepted zoom and first crossing"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_jumping_steps_plot(
    path: Path,
    accepted_events: List[Dict],
):
    resolved = [
        e
        for e in accepted_events
        if np.isfinite(
            e["jumping_steps"]
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5.5),
    )

    if resolved:
        x = np.asarray(
            [
                int(
                    e["zoom_frame"]
                )
                for e in resolved
            ]
        )

        y = np.asarray(
            [
                int(
                    e["jumping_steps"]
                )
                for e in resolved
            ]
        )

        ax.plot(
            x,
            y,
            marker="o",
        )

        for xx, yy in zip(
            x,
            y,
        ):
            ax.text(
                xx,
                yy,
                str(
                    int(
                        yy
                    )
                ),
                fontsize=8,
            )

    ax.set_xlabel(
        "accepted zoom frame t"
    )

    ax.set_ylabel(
        "jumping_steps = k - t"
    )

    ax.set_title(
        "Implied jumping steps from raw P95 first crossing"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# 7. Main
# =====================================================================

def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outdir = (
        OUTPUT_ROOT
        / f"{TASK_NAME}_{timestamp}"
    )

    compare_dir = (
        outdir
        / "accepted_zoom_comparisons"
    )

    rejected_dir = (
        outdir
        / "rejected_zoom_attempts"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if SAVE_ACCEPTED_COMPARISONS:
        compare_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_REJECTED_ATTEMPTS:
        rejected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "=" * 92
    )

    print(
        "NACLIP + Temporal(L=1) + raw P95 | Natural Zoom offline test"
    )

    print(
        "=" * 92
    )

    print(
        "OUTPUT:",
        outdir,
    )

    print(
        "DEVICE:",
        DEVICE,
    )

    print(
        "NPZ:",
        NPZ_PATH,
    )

    # -------------------------------------------------------------
    # Save config
    # -------------------------------------------------------------
    config = {
        "TASK_NAME": TASK_NAME,
        "NPZ_PATH": str(
            NPZ_PATH
        ),
        "TARGET_PROMPTS": TARGET_PROMPTS,

        "map_branch": (
            "Value -> Temporal(L=1) -> raw cosine "
            "-> sigmoid -> zoom logic"
        ),

        "progress_branch": (
            "NACLIP K-K -> Temporal(L=1) -> "
            f"raw cosine -> P{P95_PERCENTILE:g}"
        ),

        "RELEVANCE_THRESHOLD": RELEVANCE_THRESHOLD,
        "RELEVANCE_TEMPERATURE": RELEVANCE_TEMPERATURE,

        "GAUSSIAN_SIGMA_WEIGHT": GAUSSIAN_SIGMA_WEIGHT,
        "ZOOM_COOLDOWN_STEPS": ZOOM_COOLDOWN_STEPS,
        "SEMANTIC_GATE_ENABLED": SEMANTIC_GATE_ENABLED,
        "POST_ZOOM_SEMANTIC_GATE_ENABLED": POST_ZOOM_SEMANTIC_GATE_ENABLED,
        "MAX_ZOOM_FACTOR": MAX_ZOOM_FACTOR,

        "NACLIP_GAUSSIAN_STD": NACLIP_GAUSSIAN_STD,
        "NACLIP_GAUSSIAN_WEIGHT": NACLIP_GAUSSIAN_WEIGHT,
        "NACLIP_INCLUDE_CLS": NACLIP_INCLUDE_CLS,

        "P95_PERCENTILE": P95_PERCENTILE,
        "SCORESTORAGE_MAX_STEPS": SCORESTORAGE_MAX_STEPS,
    }

    (outdir / "run_config.json").write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Load trajectory
    # -------------------------------------------------------------
    frames, rgb_key, key_info = rv.load_frames(
        NPZ_PATH
    )

    end = (
        len(frames)
        if FRAME_END is None
        else min(
            len(frames),
            int(
                FRAME_END
            ),
        )
    )

    frame_indices = np.arange(
        max(
            0,
            int(
                FRAME_START
            ),
        ),
        end,
        max(
            1,
            int(
                FRAME_STRIDE
            ),
        ),
        dtype=np.int64,
    )

    if len(
        frame_indices
    ) == 0:
        raise RuntimeError(
            "No selected frames."
        )

    print(
        f"RGB key={rgb_key}; "
        f"total={len(frames)}; "
        f"tested={len(frame_indices)}"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    if not MINECLIP_CKPT.exists():
        raise FileNotFoundError(
            MINECLIP_CKPT
        )

    tester = BranchTester(
        MINECLIP_CKPT
    )

    zoom_simulator = NaturalZoomSimulator()

    # -------------------------------------------------------------
    # Caches for true trajectory frames
    # -------------------------------------------------------------
    current_progress_scores: List[float] = []
    source_progress_maps: List[np.ndarray] = []
    source_zoom_maps: List[np.ndarray] = []

    frame_rows: List[Dict] = []
    candidate_events: List[Dict] = []

    # Accepted zoom data that need later crossing search / visualization.
    accepted_payloads: List[Dict] = []

    all_progress_raw_values: List[np.ndarray] = []

    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # Sequential pass
    # -------------------------------------------------------------
    for local_i, fi in enumerate(
        frame_indices
    ):
        frame = frames[
            int(
                fi
            )
        ]

        # One ViT pass -> both map branch and progress branch.
        features = tester.extract(
            frame
        )

        map_result = compute_map_branch(
            features.current_similarity,
            features.grid_h,
            features.grid_w,
        )

        progress_result = compute_progress_branch(
            features.progress_similarity,
            features.grid_h,
            features.grid_w,
        )

        current_progress_scores.append(
            progress_result.p95
        )

        source_progress_maps.append(
            progress_result.raw_map
        )

        source_zoom_maps.append(
            map_result.max_map
        )

        all_progress_raw_values.append(
            progress_result.raw_grid_max.reshape(
                -1
            )
        )

        # 与 get_reward 一样：
        # 先把 current score 加入 running buffer，
        # 再计算当前 adaptive threshold。
        check_threshold = (
            zoom_simulator
            .update_current_statistics(
                map_result
            )
        )

        candidate = (
            zoom_simulator
            .generate_candidate(
                frame=frame,
                step=int(
                    fi
                ),
                map_result=map_result,
                check_threshold=check_threshold,
            )
        )

        frame_row = {
            "frame_index": int(
                fi
            ),

            "current_progress_p95": float(
                progress_result.p95
            ),

            "current_gaussian": float(
                map_result.gaussian_score
            ),

            "zoom_in_prob": float(
                map_result.zoom_in_prob
            ),

            "check_threshold": float(
                check_threshold
            ),

            "zoom_gate_reason": (
                candidate.gate_reason
            ),

            "produced_crop": int(
                candidate.produced_crop
            ),
        }

        frame_rows.append(
            frame_row
        )

        # Gate 0/1/no contour 没有实际 crop，不进入 post check。
        if not candidate.produced_crop:
            if (
                local_i == 0
                or (
                    local_i + 1
                ) % PRINT_EVERY_N_FRAMES
                == 0
                or local_i + 1
                == len(
                    frame_indices
                )
            ):
                print(
                    f"[{local_i+1:4d}/{len(frame_indices):4d}] "
                    f"frame={fi} "
                    f"P95={progress_result.p95:.4f} "
                    f"zoom={candidate.gate_reason}"
                )

            continue

        # ---------------------------------------------------------
        # Actual crop attempt -> run zoomed frame
        # ---------------------------------------------------------
        zoomed_features = tester.extract(
            candidate.zoomed_frame
        )

        zoomed_map_result = compute_map_branch(
            zoomed_features.current_similarity,
            zoomed_features.grid_h,
            zoomed_features.grid_w,
        )

        zoomed_progress_result = compute_progress_branch(
            zoomed_features.progress_similarity,
            zoomed_features.grid_h,
            zoomed_features.grid_w,
        )

        all_progress_raw_values.append(
            zoomed_progress_result
            .raw_grid_max
            .reshape(-1)
        )

        post = (
            zoom_simulator
            .post_zoom_accept(
                current_map=map_result,
                zoomed_map=zoomed_map_result,
                candidate=candidate,
            )
        )

        event = {
            "zoom_frame": int(
                fi
            ),

            "produced_crop": 1,
            "is_zoomed": int(
                bool(
                    post["is_zoomed"]
                )
            ),
            "jump": int(
                bool(
                    post["jump"]
                )
            ),

            "zoom_gate_reason": (
                candidate.gate_reason
            ),

            "score_current_at_t": float(
                progress_result.p95
            ),

            "score_on_zoomed": float(
                zoomed_progress_result.p95
            ),

            "crossing_frame": np.nan,
            "crossing_score": np.nan,
            "jumping_steps": np.nan,

            "current_gaussian": float(
                map_result.gaussian_score
            ),

            "zoomed_gaussian": float(
                zoomed_map_result.gaussian_score
            ),

            "gaussian_buffer_std": float(
                post["gaussian_buffer_std"]
            ),

            "gaussian_required": float(
                post["gaussian_required"]
            ),

            "gaussian_gain_ok": int(
                bool(
                    post["gaussian_gain_ok"]
                )
            ),

            "zoom_in_prob": float(
                map_result.zoom_in_prob
            ),

            "check_threshold": float(
                check_threshold
            ),

            "current_map_raw_p95": float(
                map_result.raw_p95
            ),

            "zoomed_map_raw_p95": float(
                zoomed_map_result.raw_p95
            ),

            "zoom_map_p95_gain": float(
                post["zoom_p95_gain"]
            ),

            "current_map_contrast": float(
                map_result.raw_p95_p50
            ),

            "zoomed_map_contrast": float(
                zoomed_map_result.raw_p95_p50
            ),

            "zoom_contrast_gain": float(
                post["zoom_contrast_gain"]
            ),

            "current_cc_fraction": float(
                map_result.largest_cc_fraction
            ),

            "zoomed_cc_fraction": float(
                zoomed_map_result.largest_cc_fraction
            ),

            "cc_retention": float(
                post["cc_retention"]
            ),

            "semantic_gain_ok": int(
                bool(
                    post["semantic_gain_ok"]
                )
            ),

            "have_center": int(
                candidate.have_center
            ),

            "raw_zoom_factor": float(
                candidate.raw_zoom_factor
            ),

            "actual_zoom_factor": float(
                candidate.actual_zoom_factor
            ),

            "crop_x1": int(
                candidate.crop_x1
            ),
            "crop_y1": int(
                candidate.crop_y1
            ),
            "crop_x2": int(
                candidate.crop_x2
            ),
            "crop_y2": int(
                candidate.crop_y2
            ),

            "best_value_on_mask": float(
                candidate.best_value_on_mask
            ),

            "max_area_ratio": float(
                candidate.max_area_ratio
            ),

            "num_above_threshold": float(
                candidate.num_above_threshold
            ),
        }

        candidate_events.append(
            event
        )

        # ScoreStorage 只会在 obs['is_zoomed'] 为 True 时加入。
        if bool(
            post["is_zoomed"]
        ):
            accepted_payloads.append(
                {
                    "event": event,
                    "source_local_index": int(
                        local_i
                    ),
                    "source_frame": frame.copy(),
                    "zoomed_frame": (
                        candidate.zoomed_frame.copy()
                    ),
                    "source_zoom_driver_map": (
                        map_result.max_map.copy()
                    ),
                    "source_progress_map": (
                        progress_result.raw_map.copy()
                    ),
                    "zoomed_progress_map": (
                        zoomed_progress_result.raw_map.copy()
                    ),
                }
            )
        elif SAVE_REJECTED_ATTEMPTS:
            rejected_path = (
                rejected_dir
                / f"rejected_t{int(fi):06d}.png"
            )

            save_rejected_attempt(
                rejected_path,
                frame,
                candidate.zoomed_frame,
                map_result.max_map,
                zoomed_map_result.max_map,
                event,
            )

        if (
            local_i == 0
            or (
                local_i + 1
            ) % PRINT_EVERY_N_FRAMES
            == 0
            or local_i + 1
            == len(
                frame_indices
            )
        ):
            print(
                f"[{local_i+1:4d}/{len(frame_indices):4d}] "
                f"frame={fi} "
                f"P95={progress_result.p95:.4f} "
                f"crop=1 accepted={int(post['is_zoomed'])}"
            )

    current_progress_scores_np = np.asarray(
        current_progress_scores,
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # First-crossing search
    # -------------------------------------------------------------
    frame_to_local = {
        int(
            fi
        ): int(
            i
        )
        for i, fi in enumerate(
            frame_indices
        )
    }

    accepted_events = []

    for payload in accepted_payloads:
        event = payload["event"]

        t = int(
            event["zoom_frame"]
        )

        local_t = int(
            payload[
                "source_local_index"
            ]
        )

        score_on_zoomed = float(
            event[
                "score_on_zoomed"
            ]
        )

        start_local = (
            local_t + 1
            if SEARCH_FROM_NEXT_FRAME
            else local_t
        )

        max_frame = (
            t
            + int(
                SCORESTORAGE_MAX_STEPS
            )
        )

        crossing_local = None

        for j in range(
            start_local,
            len(
                frame_indices
            ),
        ):
            k = int(
                frame_indices[j]
            )

            if k - t > SCORESTORAGE_MAX_STEPS:
                break

            if (
                current_progress_scores_np[j]
                > score_on_zoomed
            ):
                crossing_local = j
                break

        if crossing_local is not None:
            k = int(
                frame_indices[
                    crossing_local
                ]
            )

            crossing_score = float(
                current_progress_scores_np[
                    crossing_local
                ]
            )

            event[
                "crossing_frame"
            ] = k

            event[
                "crossing_score"
            ] = crossing_score

            event[
                "jumping_steps"
            ] = int(
                k - t
            )

            payload[
                "crossing_frame_rgb"
            ] = frames[k].copy()

            payload[
                "crossing_progress_map"
            ] = (
                source_progress_maps[
                    crossing_local
                ].copy()
            )
        else:
            event[
                "crossing_frame"
            ] = np.nan

            event[
                "crossing_score"
            ] = np.nan

            event[
                "jumping_steps"
            ] = np.nan

            payload[
                "crossing_frame_rgb"
            ] = None

            payload[
                "crossing_progress_map"
            ] = None

        accepted_events.append(
            event
        )

    # -------------------------------------------------------------
    # Shared raw-cosine colormap range
    # -------------------------------------------------------------
    if all_progress_raw_values:
        all_raw = np.concatenate(
            all_progress_raw_values,
            axis=0,
        )

        raw_vmin = float(
            np.percentile(
                all_raw,
                RAW_COSINE_VMIN_PERCENTILE,
            )
        )

        raw_vmax = float(
            np.percentile(
                all_raw,
                RAW_COSINE_VMAX_PERCENTILE,
            )
        )
    else:
        raw_vmin = 0.0
        raw_vmax = 1.0

    if raw_vmax <= raw_vmin:
        raw_vmax = (
            raw_vmin
            + 1e-3
        )

    # -------------------------------------------------------------
    # Save accepted comparison images
    # -------------------------------------------------------------
    if SAVE_ACCEPTED_COMPARISONS:
        for payload in accepted_payloads:
            event = payload[
                "event"
            ]

            t = int(
                event[
                    "zoom_frame"
                ]
            )

            if np.isfinite(
                event[
                    "crossing_frame"
                ]
            ):
                k = int(
                    event[
                        "crossing_frame"
                    ]
                )

                jump = int(
                    event[
                        "jumping_steps"
                    ]
                )

                name = (
                    f"zoom_t{t:06d}"
                    f"_to_k{k:06d}"
                    f"_jump{jump:04d}.png"
                )
            else:
                name = (
                    f"zoom_t{t:06d}"
                    "_unresolved.png"
                )

            save_accepted_comparison(
                path=(
                    compare_dir
                    / name
                ),
                event=event,
                source_frame=(
                    payload[
                        "source_frame"
                    ]
                ),
                zoomed_frame=(
                    payload[
                        "zoomed_frame"
                    ]
                ),
                crossing_frame=(
                    payload.get(
                        "crossing_frame_rgb"
                    )
                ),
                source_zoom_driver_map=(
                    payload[
                        "source_zoom_driver_map"
                    ]
                ),
                source_progress_map=(
                    payload[
                        "source_progress_map"
                    ]
                ),
                zoomed_progress_map=(
                    payload[
                        "zoomed_progress_map"
                    ]
                ),
                crossing_progress_map=(
                    payload.get(
                        "crossing_progress_map"
                    )
                ),
                raw_vmin=raw_vmin,
                raw_vmax=raw_vmax,
            )

    # -------------------------------------------------------------
    # Save concise CSVs
    # -------------------------------------------------------------
    current_rows = []

    for i, fi in enumerate(
        frame_indices
    ):
        row = dict(
            frame_rows[i]
        )

        row[
            "current_progress_p95"
        ] = float(
            current_progress_scores_np[
                i
            ]
        )

        current_rows.append(
            row
        )

    # candidate_events 已经包含 accepted + post-check rejected crop attempts。
    # crossing 更新发生在相同 dict object 上，所以这里直接保存即可。
    save_csv(
        outdir
        / "zoom_events.csv",
        candidate_events,
    )

    save_csv(
        outdir
        / "current_progress.csv",
        current_rows,
    )

    # -------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------
    save_timeline(
        outdir
        / "zoom_timeline.png",
        frame_indices,
        current_progress_scores_np,
        accepted_events,
    )

    save_jumping_steps_plot(
        outdir
        / "jumping_steps.png",
        accepted_events,
    )

    # -------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------
    n_attempt = len(
        candidate_events
    )

    n_accepted = len(
        accepted_events
    )

    n_resolved = sum(
        np.isfinite(
            e[
                "jumping_steps"
            ]
        )
        for e in accepted_events
    )

    jumps = [
        int(
            e[
                "jumping_steps"
            ]
        )
        for e in accepted_events
        if np.isfinite(
            e[
                "jumping_steps"
            ]
        )
    ]

    summary = {
        "crop_attempts": int(
            n_attempt
        ),
        "accepted_is_zoomed": int(
            n_accepted
        ),
        "resolved_crossing": int(
            n_resolved
        ),
        "unresolved_crossing": int(
            n_accepted
            - n_resolved
        ),
        "jumping_steps_mean": (
            float(
                np.mean(
                    jumps
                )
            )
            if jumps
            else None
        ),
        "jumping_steps_median": (
            float(
                np.median(
                    jumps
                )
            )
            if jumps
            else None
        ),
        "jumping_steps_min": (
            int(
                np.min(
                    jumps
                )
            )
            if jumps
            else None
        ),
        "jumping_steps_max": (
            int(
                np.max(
                    jumps
                )
            )
            if jumps
            else None
        ),
        "elapsed_seconds": float(
            time.perf_counter()
            - t0
        ),
    }

    (outdir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 92
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 92
    )

    print(
        "crop attempts       :",
        n_attempt,
    )

    print(
        "accepted is_zoomed  :",
        n_accepted,
    )

    print(
        "resolved crossing   :",
        n_resolved,
    )

    print(
        "unresolved crossing :",
        n_accepted
        - n_resolved,
    )

    if jumps:
        print(
            "jumping_steps mean :",
            f"{np.mean(jumps):.2f}",
        )

        print(
            "jumping_steps median:",
            f"{np.median(jumps):.2f}",
        )

        print(
            "jumping_steps range:",
            f"{min(jumps)} .. {max(jumps)}",
        )

    print()
    print(
        "重点看："
    )

    print(
        "  zoom_events.csv"
    )

    print(
        "  zoom_timeline.png"
    )

    print(
        "  jumping_steps.png"
    )

    print(
        "  accepted_zoom_comparisons/"
    )

    print(
        "  current_progress.csv"
    )

    print()
    print(
        "Output:",
        outdir,
    )

    print(
        "=" * 92
    )


if __name__ == "__main__":
    main()
