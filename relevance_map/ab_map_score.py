#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A/B ablation for Natural Zoom map vs progress score
====================================================

实验 A：只测试 MAP 是否值得换成 NACLIP
---------------------------------------
A1) Current-map + NACLIP-P85
    MAP:
        Value -> Temporal(L=1) -> raw cosine -> sigmoid map
    Progress score:
        NACLIP K-K -> Temporal(L=1) -> raw cosine -> P85

A2) NACLIP-map + NACLIP-P85
    MAP:
        NACLIP K-K -> Temporal(L=1) -> raw cosine -> sigmoid map
    Progress score:
        SAME NACLIP raw cosine -> P85

A1 vs A2:
    Progress score 完全相同，只改变 Natural-Zoom map。
    用于回答：
        NACLIP spatial refinement 是否真的改善 zoom，
        还是只是增加 zoom 次数 / 后期重复触发？

实验 B：只测试 SCORE 是否需要 NACLIP / P85
------------------------------------------
B 的 Natural Zoom 完全固定为 Current-map，而且只运行一次。
每一个 accepted zoom 的 source/crop/zoomed RGB 完全相同。

然后比较三个 crossing score：
B1) Current-Gaussian
    Value+Temporal sigmoid map 的 Gaussian-weighted score

B2) Current-P85
    Value+Temporal raw cosine -> P85

B3) NACLIP-P85
    NACLIP+Temporal raw cosine -> P85

B1/B2/B3 共用相同 accepted zoom event，
所以差异只来自 score readout。

额外针对你观察到的后期 NACLIP zoom
--------------------------------
默认：
    LATE_ZOOM_START = 90

会统计：
    - 90 步后 accepted zoom 数量
    - late zoom fraction
    - late zoom frame
    - 相邻 zoom 间隔
并生成 late zoom contact sheet，
方便重点看类似 96 / 109 / 117 这种后期触发。

核心原则
--------
- Natural Zoom 是否触发 / crop 哪里：
    由 map 决定，不由 P85 决定。
- P85：
    只用于 ScoreStorage-style first crossing：
        first k > t such that
        score_current(k) > score_on_zoomed(t)
- 当前 map 和 NACLIP map 都使用同一个 tau/T，
  这样实验 A 只比较 representation/map source。

PyCharm 直接运行，无 argparse。

建议保存为：
    <LS-Imagine>/relevance_map/ab_map_score_test.py
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
from typing import Dict, List, Optional, Tuple

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

# 如脚本不在 <project>/relevance_map/ 下，手工指定：
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

TASK_SPECS = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS.exists():
    raise FileNotFoundError(f"PROJECT_ROOT wrong: {TASK_SPECS}")

# 复用前面已经验证过的 MineCLIP / NACLIP 实现。
import relevance_map.relevance_variants as rv


# =====================================================================
# 1. GLOBAL CONFIG
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
    / "ab_map_score_test_outputs"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_H = 160
IMAGE_W = 256

TARGET_PROMPTS = ["Cut a tree"]


# ---------------------------------------------------------------------
# 1.1 固定 P85
# ---------------------------------------------------------------------

PROGRESS_PERCENTILE = 85.0


# ---------------------------------------------------------------------
# 1.2 map calibration
#
# Current-map 和 NACLIP-map 都使用同一组参数。
# 注意：tau/T 会影响 zoom map / zoom trigger / crop，
# 不直接进入 raw P85 的计算。
# ---------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016


# ---------------------------------------------------------------------
# 1.3 NACLIP 全局参数
# ---------------------------------------------------------------------

NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True


# ---------------------------------------------------------------------
# 1.4 Natural Zoom 参数
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

MORPH_KERNEL = (16, 10)
CROP_STRIDE = 1

SCORESTORAGE_MAX_STEPS = 1000


# ---------------------------------------------------------------------
# 1.5 late zoom diagnostics
#
# 你已经观察到 NACLIP 在 96 / 109 / 117 一类后期 frame 的 zoom
# 视觉效果不佳，所以单独统计 90 步之后的触发。
# ---------------------------------------------------------------------

LATE_ZOOM_START = 90


# ---------------------------------------------------------------------
# 1.6 trajectory
# ---------------------------------------------------------------------

FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1


# ---------------------------------------------------------------------
# 1.7 output
# ---------------------------------------------------------------------

SAVE_EXPERIMENT_A_EVENT_PANELS = True
SAVE_EXPERIMENT_B_EVENT_PANELS = True
SAVE_LATE_ZOOM_CONTACT_SHEETS = True

VIS_DPI = 150
PRINT_EVERY_N_FRAMES = 20

RAW_COSINE_LOW_PERCENTILE = 1.0
RAW_COSINE_HIGH_PERCENTILE = 99.0


# =====================================================================
# 2. Helpers
# =====================================================================

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

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def sigmoid_scalar(x: float) -> float:
    x = float(
        np.clip(
            x,
            -30.0,
            30.0,
        )
    )

    return float(
        1.0
        / (
            1.0
            + math.exp(-x)
        )
    )


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(
        x,
        dtype=np.float64,
    )

    z = np.clip(
        z,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-z)
        )
    ).astype(np.float32)


def upsample_patch_grid(
    grid: np.ndarray,
) -> np.ndarray:
    x = torch.from_numpy(
        np.asarray(
            grid,
            dtype=np.float32,
        )
    )[None, None]

    y = F.interpolate(
        x,
        size=(IMAGE_H, IMAGE_W),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return (
        y.numpy()
        .astype(np.float32)
    )


def make_center_gaussian() -> np.ndarray:
    sigma_x = (
        IMAGE_W
        * GAUSSIAN_SIGMA_WEIGHT
    )

    sigma_y = (
        IMAGE_H
        * GAUSSIAN_SIGMA_WEIGHT
    )

    x = np.linspace(
        -IMAGE_W // 2,
        IMAGE_W // 2,
        IMAGE_W,
    )

    y = np.linspace(
        -IMAGE_H // 2,
        IMAGE_H // 2,
        IMAGE_H,
    )

    xx, yy = np.meshgrid(x, y)

    g = np.exp(
        -(
            xx ** 2
            / (
                2.0
                * sigma_x ** 2
            )
            +
            yy ** 2
            / (
                2.0
                * sigma_y ** 2
            )
        )
    )

    return g.astype(np.float32)


CENTER_GAUSSIAN = make_center_gaussian()
CENTER_GAUSSIAN_MEAN = float(
    np.mean(
        CENTER_GAUSSIAN
    )
)


class RunningStats:
    """
    与 current ThresholdBuffer 保持 population std：
        std = sqrt(M2 / n)
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x: float):
        x = float(x)

        self.n += 1

        delta = x - self.mean
        self.mean += delta / self.n

        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std_dev(self) -> float:
        if self.n <= 1:
            return 0.0

        return float(
            math.sqrt(
                self.M2
                / self.n
            )
        )

    def threshold(self) -> float:
        if self.n <= 0:
            return 1.0

        return float(
            self.mean
            + self.std_dev()
        )


def largest_cc_fraction(
    grid: np.ndarray,
    threshold: float = 0.5,
) -> float:
    binary = (
        np.asarray(grid)
        > threshold
    ).astype(np.uint8)

    n_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if n_labels <= 1:
        return 0.0

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    return float(
        np.max(areas)
        / binary.size
    )


def gaussian_score(
    map01: np.ndarray,
) -> float:
    return float(
        np.mean(
            map01
            * CENTER_GAUSSIAN
        )
        / CENTER_GAUSSIAN_MEAN
    )


def zoom_probability(
    map01: np.ndarray,
) -> float:
    k = float(
        kurtosis(
            map01.flatten()
        )
    )

    normalized_kurtosis = sigmoid_scalar(
        k
    )

    return float(
        normalized_kurtosis
        * (
            float(np.max(map01))
            - float(np.mean(map01))
        )
    )


# =====================================================================
# 3. MineCLIP / NACLIP feature extraction
# =====================================================================

@dataclass
class FeatureBundle:
    gh: int
    gw: int

    # [P,N]
    current_similarity: np.ndarray

    # [P,N]
    naclip_similarity: np.ndarray


class FeatureExtractor:
    def __init__(self):
        if not MINECLIP_CKPT.exists():
            raise FileNotFoundError(
                MINECLIP_CKPT
            )

        # relevance_variants.py 读取模块级 NACLIP 参数。
        rv.NACLIP_GAUSSIAN_STD = float(
            NACLIP_GAUSSIAN_STD
        )

        rv.NACLIP_GAUSSIAN_WEIGHT = float(
            NACLIP_GAUSSIAN_WEIGHT
        )

        rv.NACLIP_INCLUDE_CLS = bool(
            NACLIP_INCLUDE_CLS
        )

        self.tester = rv.MineCLIPTester(
            MINECLIP_CKPT
        )

        self.tester.text_feats(
            TARGET_PROMPTS
        )

    @torch.inference_mode()
    def extract(
        self,
        frame: np.ndarray,
    ) -> FeatureBundle:

        x, last, gh, gw = (
            self.tester.encode_to_last(
                frame
            )
        )

        # ---------------- Current ----------------
        value = (
            self.tester.raw_value_patch(
                x,
                last,
            )
        )

        value_temporal = (
            self.tester.temporal_l1(
                value
            )
        )

        current_sim = (
            self.tester.cosine(
                value_temporal,
                TARGET_PROMPTS,
            )[0]
        )

        # ---------------- NACLIP ----------------
        naclip = (
            self.tester.naclip_patch(
                x,
                last,
                gh,
                gw,
            )
        )

        naclip_temporal = (
            self.tester.temporal_l1(
                naclip
            )
        )

        naclip_sim = (
            self.tester.cosine(
                naclip_temporal,
                TARGET_PROMPTS,
            )[0]
        )

        return FeatureBundle(
            gh=int(gh),
            gw=int(gw),

            current_similarity=(
                current_sim
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),

            naclip_similarity=(
                naclip_sim
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
        )


# =====================================================================
# 4. Representation
# =====================================================================

@dataclass
class RepresentationResult:
    source_name: str

    # multi-prompt max raw cosine [N]
    raw_patch_scores: np.ndarray

    # [H,W]
    raw_cosine_map: np.ndarray

    # sigmoid spatial map [H,W]
    zoom_map: np.ndarray

    gaussian_score: float
    zoom_in_prob: float

    raw_p50: float
    raw_p95: float
    raw_p95_p50: float

    largest_cc_fraction: float

    def percentile_score(
        self,
        q: float = PROGRESS_PERCENTILE,
    ) -> float:
        return float(
            np.percentile(
                self.raw_patch_scores,
                q,
            )
        )


def representation_from_similarity(
    source_name: str,
    similarity: np.ndarray,
    gh: int,
    gw: int,
) -> RepresentationResult:

    sim = np.asarray(
        similarity,
        dtype=np.float32,
    )

    # 多 prompt 时，对每个 patch 取最高 prompt cosine。
    raw_patch_scores = np.max(
        sim,
        axis=0,
    ).astype(np.float32)

    raw_p50 = float(
        np.percentile(
            raw_patch_scores,
            50,
        )
    )

    raw_p95 = float(
        np.percentile(
            raw_patch_scores,
            95,
        )
    )

    relevance = stable_sigmoid_np(
        (
            sim
            - RELEVANCE_THRESHOLD
        )
        / RELEVANCE_TEMPERATURE
    )

    relevance_max = np.max(
        relevance,
        axis=0,
    ).reshape(
        gh,
        gw,
    )

    raw_grid = raw_patch_scores.reshape(
        gh,
        gw,
    )

    zoom_map = upsample_patch_grid(
        relevance_max
    )

    raw_cosine_map = upsample_patch_grid(
        raw_grid
    )

    return RepresentationResult(
        source_name=source_name,

        raw_patch_scores=raw_patch_scores,

        raw_cosine_map=raw_cosine_map,

        zoom_map=zoom_map,

        gaussian_score=gaussian_score(
            zoom_map
        ),

        zoom_in_prob=zoom_probability(
            zoom_map
        ),

        raw_p50=raw_p50,
        raw_p95=raw_p95,

        raw_p95_p50=float(
            raw_p95
            - raw_p50
        ),

        largest_cc_fraction=(
            largest_cc_fraction(
                relevance_max,
                threshold=0.5,
            )
        ),
    )


@dataclass
class FrameResults:
    current: RepresentationResult
    naclip: RepresentationResult


class Evaluator:
    def __init__(self):
        self.extractor = FeatureExtractor()

    @torch.inference_mode()
    def evaluate(
        self,
        frame: np.ndarray,
    ) -> FrameResults:

        features = self.extractor.extract(
            frame
        )

        current = representation_from_similarity(
            "current",
            features.current_similarity,
            features.gh,
            features.gw,
        )

        naclip = representation_from_similarity(
            "naclip",
            features.naclip_similarity,
            features.gh,
            features.gw,
        )

        return FrameResults(
            current=current,
            naclip=naclip,
        )


# =====================================================================
# 5. Natural Zoom state machine
# =====================================================================

@dataclass
class CropCandidate:
    produced_crop: bool
    reason: str

    zoomed_frame: Optional[np.ndarray]

    have_center: bool = False

    raw_zoom_factor: float = 1.0
    actual_zoom_factor: float = 1.0

    crop_x1: int = -1
    crop_y1: int = -1
    crop_x2: int = -1
    crop_y2: int = -1

    max_area_ratio: float = np.nan
    num_above_threshold: float = np.nan

    best_value_on_mask: float = np.nan


class NaturalZoomEngine:
    """
    Current-map 与 NACLIP-map 各有独立 state machine。
    """

    def __init__(self):
        self.check_buffer = RunningStats()
        self.gaussian_buffer = RunningStats()

        self.last_zoom_attempt_step = None

    def update_current(
        self,
        result: RepresentationResult,
    ) -> float:

        self.gaussian_buffer.add(
            result.gaussian_score
        )

        self.check_buffer.add(
            result.zoom_in_prob
        )

        return self.check_buffer.threshold()

    def propose(
        self,
        frame: np.ndarray,
        step: int,
        result: RepresentationResult,
        check_threshold: float,
    ) -> CropCandidate:

        # Gate 0
        if (
            check_threshold
            >= result.zoom_in_prob
        ):
            return CropCandidate(
                produced_crop=False,
                reason="adaptive_threshold",
                zoomed_frame=None,
            )

        # Gate 1
        if (
            ZOOM_COOLDOWN_STEPS > 0
            and self.last_zoom_attempt_step
            is not None
            and (
                step
                - self.last_zoom_attempt_step
                <= ZOOM_COOLDOWN_STEPS
            )
        ):
            return CropCandidate(
                produced_crop=False,
                reason="cooldown",
                zoomed_frame=None,
            )

        # Gate 2
        semantic_ok = (
            result.raw_p95_p50
            >= SEMANTIC_MIN_P95_P50
            and result.largest_cc_fraction
            >= SEMANTIC_MIN_CC_FRACTION
        )

        if (
            SEMANTIC_GATE_ENABLED
            and not semantic_ok
        ):
            return CropCandidate(
                produced_crop=False,
                reason="semantic_gate",
                zoomed_frame=None,
            )

        heatmap = np.asarray(
            result.zoom_map,
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

        _, binary = cv2.threshold(
            heatmap,
            threshold_value,
            1,
            cv2.THRESH_BINARY,
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            MORPH_KERNEL,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return CropCandidate(
                produced_crop=False,
                reason="no_contour",
                zoomed_frame=None,
            )

        max_mean_value = 0.0
        max_area_ratio = 0.0

        centroid_x = 0
        centroid_y = 0

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

            values = heatmap[
                contour_mask == 1
            ]

            if values.size <= 0:
                continue

            mean_value = float(
                np.mean(values)
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
                    cx,
                ]
                >= CENTER_GAUSSIAN_MEAN
            ):
                have_center = True

            if mean_value > max_mean_value:
                max_mean_value = mean_value

                max_area_ratio = float(
                    np.sum(
                        contour_mask
                    )
                    / contour_mask.size
                )

                centroid_x = cx
                centroid_y = cy

        if max_area_ratio <= 0:
            return CropCandidate(
                produced_crop=False,
                reason="invalid_contour",
                zoomed_frame=None,
            )

        num_above_threshold = (
            max_area_ratio
            / float(
                CENTER_GAUSSIAN[
                    centroid_y,
                    centroid_x,
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
                / float(
                    MAX_ZOOM_FACTOR
                    ** 2
                )
            )

            proportion = float(
                np.clip(
                    proportion,
                    min_proportion,
                    1.0,
                )
            )

        sqrt_prop = math.sqrt(
            proportion
        )

        H, W, _ = frame.shape

        win_w = int(
            math.ceil(
                sqrt_prop
                * W
            )
        )

        win_h = int(
            math.ceil(
                sqrt_prop
                * H
            )
        )

        win_w = max(
            1,
            min(
                win_w,
                W,
            ),
        )

        win_h = max(
            1,
            min(
                win_h,
                H,
            ),
        )

        actual_zoom_factor = max(
            W
            / float(win_w),
            H
            / float(win_h),
        )

        relevance_tensor = (
            torch.from_numpy(
                heatmap
            )
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kernel = torch.ones(
            (
                1,
                1,
                win_h,
                win_w,
            ),
            dtype=torch.float32,
        )

        conv = F.conv2d(
            relevance_tensor,
            kernel,
            stride=CROP_STRIDE,
        )

        best_value, best_idx = torch.max(
            conv.view(-1),
            dim=0,
        )

        best_y, best_x = divmod(
            int(
                best_idx.item()
            ),
            int(
                conv.shape[-1]
            ),
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
            + win_w
        )

        y2 = int(
            y1
            + win_h
        )

        crop = frame[
            y1:y2,
            x1:x2,
            :
        ]

        zoomed = cv2.resize(
            crop,
            (
                W,
                H,
            ),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        # 当前代码语义：
        # 只要实际生成 crop，就开始 cooldown，
        # 即使 post-zoom acceptance 最后失败。
        self.last_zoom_attempt_step = int(
            step
        )

        return CropCandidate(
            produced_crop=True,
            reason="natural_zoom_candidate",
            zoomed_frame=zoomed,

            have_center=bool(
                have_center
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

            max_area_ratio=float(
                max_area_ratio
            ),

            num_above_threshold=float(
                num_above_threshold
            ),

            best_value_on_mask=float(
                best_value.item()
                / (
                    win_h
                    * win_w
                )
            ),
        )

    def accept(
        self,
        current_result: RepresentationResult,
        zoomed_result: RepresentationResult,
        candidate: CropCandidate,
    ) -> Dict:

        gaussian_std = (
            self.gaussian_buffer
            .std_dev()
        )

        gaussian_required = (
            current_result.gaussian_score
            + 2.0
            * gaussian_std
        )

        gaussian_gain_ok = (
            zoomed_result.gaussian_score
            >= gaussian_required
        )

        p95_gain = (
            zoomed_result.raw_p95
            - current_result.raw_p95
        )

        contrast_gain = (
            zoomed_result.raw_p95_p50
            - current_result.raw_p95_p50
        )

        if (
            current_result.largest_cc_fraction
            > 1e-8
        ):
            cc_retention = (
                zoomed_result.largest_cc_fraction
                / current_result.largest_cc_fraction
            )
        else:
            cc_retention = (
                1.0
                if zoomed_result.largest_cc_fraction
                > 0
                else 0.0
            )

        semantic_gain_ok = (
            p95_gain
            >= POST_ZOOM_MIN_P95_GAIN
            and contrast_gain
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
            bool(
                is_zoomed
            )
            and bool(
                candidate.have_center
            )
        )

        return {
            "gaussian_std": float(
                gaussian_std
            ),

            "gaussian_required": float(
                gaussian_required
            ),

            "gaussian_gain_ok": int(
                bool(
                    gaussian_gain_ok
                )
            ),

            "semantic_gain_ok": int(
                bool(
                    semantic_gain_ok
                )
            ),

            "p95_gain": float(
                p95_gain
            ),

            "contrast_gain": float(
                contrast_gain
            ),

            "cc_retention": float(
                cc_retention
            ),

            "is_zoomed": int(
                bool(
                    is_zoomed
                )
            ),

            "jump": int(
                bool(
                    jump
                )
            ),
        }


# =====================================================================
# 6. Accepted event
# =====================================================================

@dataclass
class AcceptedZoomEvent:
    map_source: str

    zoom_frame: int
    source_local_index: int

    source_frame: np.ndarray
    zoomed_frame: np.ndarray

    source_results: FrameResults
    zoomed_results: FrameResults

    map_source_result: RepresentationResult
    zoomed_map_source_result: RepresentationResult

    candidate: CropCandidate
    post: Dict


# =====================================================================
# 7. crossing
# =====================================================================

def first_crossing(
    frame_indices: np.ndarray,
    score_curve: np.ndarray,
    source_local_index: int,
    score_on_zoomed: float,
) -> Tuple[
    Optional[int],
    Optional[int],
    Optional[float],
]:

    t = int(
        frame_indices[
            source_local_index
        ]
    )

    for j in range(
        source_local_index + 1,
        len(
            frame_indices
        ),
    ):
        k = int(
            frame_indices[
                j
            ]
        )

        if (
            k - t
            > SCORESTORAGE_MAX_STEPS
        ):
            break

        if (
            float(
                score_curve[j]
            )
            > float(
                score_on_zoomed
            )
        ):
            return (
                int(j),
                k,
                float(
                    score_curve[j]
                ),
            )

    return None, None, None


# =====================================================================
# 8. Visualization helpers
# =====================================================================

def crop_box_image(
    frame: np.ndarray,
    candidate: CropCandidate,
) -> np.ndarray:

    out = frame.copy()

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


def save_experiment_a_event_panel(
    path: Path,
    event: AcceptedZoomEvent,
    event_row: Dict,
    frames: np.ndarray,
    naclip_results: List[RepresentationResult],
    raw_vmin: float,
    raw_vmax: float,
):
    """
    A: NACLIP-P85 score fixed.
    Only map source differs across separate events.
    """

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        squeeze=False,
    )

    ax[0, 0].imshow(
        crop_box_image(
            event.source_frame,
            event.candidate,
        )
    )

    ax[0, 0].set_title(
        f"{event.map_source} map\n"
        f"source t={event.zoom_frame}\n"
        f"NACLIP P85={event_row['score_current_at_t']:.4f}"
    )

    ax[0, 1].imshow(
        event.zoomed_frame
    )

    ax[0, 1].set_title(
        "accepted zoomed image\n"
        f"zoomed NACLIP P85={event_row['score_on_zoomed']:.4f}"
    )

    if np.isfinite(
        event_row[
            "crossing_frame"
        ]
    ):
        k = int(
            event_row[
                "crossing_frame"
            ]
        )

        ax[0, 2].imshow(
            frames[k]
        )

        ax[0, 2].set_title(
            f"first crossing k={k}\n"
            f"J={int(event_row['jumping_steps'])}"
        )
    else:
        ax[0, 2].imshow(
            np.zeros_like(
                event.source_frame
            )
        )

        ax[0, 2].set_title(
            "unresolved"
        )

    # -------------------------------------------------------------
    # maps
    # -------------------------------------------------------------
    ax[1, 0].imshow(
        event.map_source_result.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 0].set_title(
        f"source {event.map_source} zoom map"
    )

    ax[1, 1].imshow(
        event.zoomed_results.naclip.raw_cosine_map,
        cmap="viridis",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    ax[1, 1].set_title(
        "zoomed NACLIP raw cosine"
    )

    if np.isfinite(
        event_row[
            "crossing_local_index"
        ]
    ):
        crossing_local = int(
            event_row[
                "crossing_local_index"
            ]
        )

        ax[1, 2].imshow(
            naclip_results[
                crossing_local
            ].raw_cosine_map,
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 2].set_title(
            "crossing NACLIP raw cosine"
        )
    else:
        ax[1, 2].imshow(
            np.zeros_like(
                event.zoomed_results.naclip.raw_cosine_map
            ),
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 2].set_title(
            "no crossing"
        )

    for a in ax.reshape(-1):
        a.axis(
            "off"
        )

    fig.suptitle(
        "Experiment A: map ablation | score fixed = NACLIP P85",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_experiment_b_event_panel(
    path: Path,
    event: AcceptedZoomEvent,
    score_rows: List[Dict],
    frames: np.ndarray,
):
    """
    Same source / crop / zoomed RGB for all B scores.

    One large horizontal comparison:
        source | zoomed | Gaussian crossing | Current-P85 crossing | NACLIP-P85 crossing
    """

    by_name = {
        r["score_name"]: r
        for r in score_rows
    }

    order = [
        "current_gaussian",
        "current_p85",
        "naclip_p85",
    ]

    cols = 2 + len(order)

    fig, ax = plt.subplots(
        1,
        cols,
        figsize=(25, 5.8),
        squeeze=False,
    )

    ax = ax[0]

    ax[0].imshow(
        crop_box_image(
            event.source_frame,
            event.candidate,
        )
    )

    ax[0].set_title(
        f"Current-map source\n"
        f"t={event.zoom_frame}"
    )

    ax[1].imshow(
        event.zoomed_frame
    )

    ax[1].set_title(
        "SAME accepted zoomed image\n"
        "used by all three scores"
    )

    display = {
        "current_gaussian": "Current-Gaussian",
        "current_p85": "Current-P85",
        "naclip_p85": "NACLIP-P85",
    }

    for i, name in enumerate(
        order
    ):
        row = by_name[name]
        a = ax[
            2 + i
        ]

        if np.isfinite(
            row[
                "crossing_frame"
            ]
        ):
            k = int(
                row[
                    "crossing_frame"
                ]
            )

            a.imshow(
                frames[k]
            )

            a.set_title(
                f"{display[name]}\n"
                f"crossing k={k}\n"
                f"J={int(row['jumping_steps'])}\n"
                f"{row['crossing_score']:.4f} > "
                f"{row['score_on_zoomed']:.4f}"
            )
        else:
            a.imshow(
                np.zeros_like(
                    event.source_frame
                )
            )

            a.set_title(
                f"{display[name]}\n"
                "unresolved"
            )

    for a in ax:
        a.axis(
            "off"
        )

    fig.suptitle(
        "Experiment B: score ablation | SAME Current-map Natural Zoom",
        fontsize=15,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_experiment_a_event_raster(
    path: Path,
    current_map_events: List[AcceptedZoomEvent],
    naclip_map_events: List[AcceptedZoomEvent],
):
    fig, ax = plt.subplots(
        figsize=(14, 4.8),
    )

    current_frames = [
        e.zoom_frame
        for e in current_map_events
    ]

    naclip_frames = [
        e.zoom_frame
        for e in naclip_map_events
    ]

    ax.scatter(
        current_frames,
        [1] * len(current_frames),
        s=70,
        label="Current-map + NACLIP-P85",
    )

    ax.scatter(
        naclip_frames,
        [0] * len(naclip_frames),
        s=70,
        label="NACLIP-map + NACLIP-P85",
    )

    ax.axvline(
        LATE_ZOOM_START,
        linestyle="--",
        alpha=0.65,
        label=f"late start={LATE_ZOOM_START}",
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "NACLIP map",
            "Current map",
        ]
    )

    ax.set_xlabel(
        "Trajectory frame index"
    )

    ax.set_title(
        "Experiment A: accepted Natural-Zoom events"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_experiment_b_jumping_plot(
    path: Path,
    rows: List[Dict],
):
    fig, ax = plt.subplots(
        figsize=(13, 6),
    )

    display = {
        "current_gaussian": "Current-Gaussian",
        "current_p85": "Current-P85",
        "naclip_p85": "NACLIP-P85",
    }

    for score_name in [
        "current_gaussian",
        "current_p85",
        "naclip_p85",
    ]:
        subset = [
            r
            for r in rows
            if r["score_name"] == score_name
            and np.isfinite(
                r[
                    "jumping_steps"
                ]
            )
        ]

        if not subset:
            continue

        x = [
            int(
                r[
                    "zoom_frame"
                ]
            )
            for r in subset
        ]

        y = [
            int(
                r[
                    "jumping_steps"
                ]
            )
            for r in subset
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=display[
                score_name
            ],
        )

    ax.set_xlabel(
        "SAME Current-map accepted zoom frame t"
    )

    ax.set_ylabel(
        "jumping_steps"
    )

    ax.set_title(
        "Experiment B: score readout only"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_late_zoom_contact_sheet(
    path: Path,
    title: str,
    events: List[AcceptedZoomEvent],
):
    late = [
        e
        for e in events
        if e.zoom_frame
        >= LATE_ZOOM_START
    ]

    if not late:
        return

    n = len(
        late
    )

    fig, ax = plt.subplots(
        2,
        n,
        figsize=(
            max(
                4 * n,
                8,
            ),
            7,
        ),
        squeeze=False,
    )

    for i, event in enumerate(
        late
    ):
        ax[0, i].imshow(
            crop_box_image(
                event.source_frame,
                event.candidate,
            )
        )

        ax[0, i].set_title(
            f"source t={event.zoom_frame}"
        )

        ax[1, i].imshow(
            event.zoomed_frame
        )

        ax[1, i].set_title(
            f"zoomed\n"
            f"{event.candidate.actual_zoom_factor:.2f}x"
        )

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    fig.suptitle(
        f"{title}\n"
        f"late accepted zooms (t >= {LATE_ZOOM_START})",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 9. Summary helpers
# =====================================================================

def summarize_zoom_branch(
    name: str,
    events: List[AcceptedZoomEvent],
    event_rows: List[Dict],
) -> Dict:

    jumps = [
        int(
            r[
                "jumping_steps"
            ]
        )
        for r in event_rows
        if np.isfinite(
            r[
                "jumping_steps"
            ]
        )
    ]

    late_events = [
        e
        for e in events
        if e.zoom_frame
        >= LATE_ZOOM_START
    ]

    frames = sorted(
        [
            e.zoom_frame
            for e in events
        ]
    )

    gaps = [
        frames[i]
        - frames[i - 1]
        for i in range(
            1,
            len(
                frames
            ),
        )
    ]

    return {
        "method": name,

        "accepted_zoom": int(
            len(
                events
            )
        ),

        "accepted_zoom_frames": ",".join(
            str(
                x
            )
            for x in frames
        ),

        "late_zoom_start": int(
            LATE_ZOOM_START
        ),

        "late_zoom_count": int(
            len(
                late_events
            )
        ),

        "late_zoom_frames": ",".join(
            str(
                e.zoom_frame
            )
            for e in late_events
        ),

        "late_zoom_fraction": (
            float(
                len(
                    late_events
                )
                / len(
                    events
                )
            )
            if events
            else np.nan
        ),

        "mean_inter_zoom_gap": (
            float(
                np.mean(
                    gaps
                )
            )
            if gaps
            else np.nan
        ),

        "min_inter_zoom_gap": (
            int(
                np.min(
                    gaps
                )
            )
            if gaps
            else np.nan
        ),

        "resolved_crossing": int(
            len(
                jumps
            )
        ),

        "resolve_rate": (
            float(
                len(
                    jumps
                )
                / len(
                    event_rows
                )
            )
            if event_rows
            else np.nan
        ),

        "jumping_steps_mean": (
            float(
                np.mean(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_median": (
            float(
                np.median(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_min": (
            int(
                np.min(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_max": (
            int(
                np.max(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),
    }


def summarize_score_method(
    score_name: str,
    rows: List[Dict],
) -> Dict:

    subset = [
        r
        for r in rows
        if r[
            "score_name"
        ] == score_name
    ]

    jumps = [
        int(
            r[
                "jumping_steps"
            ]
        )
        for r in subset
        if np.isfinite(
            r[
                "jumping_steps"
            ]
        )
    ]

    return {
        "score_name": score_name,

        "shared_current_map_zoom_events": int(
            len(
                subset
            )
        ),

        "resolved_crossing": int(
            len(
                jumps
            )
        ),

        "resolve_rate": (
            float(
                len(
                    jumps
                )
                / len(
                    subset
                )
            )
            if subset
            else np.nan
        ),

        "jumping_steps_mean": (
            float(
                np.mean(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_median": (
            float(
                np.median(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_min": (
            int(
                np.min(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jumping_steps_max": (
            int(
                np.max(
                    jumps
                )
            )
            if jumps
            else np.nan
        ),

        "jump_le_5_fraction": (
            float(
                np.mean(
                    np.asarray(
                        jumps
                    )
                    <= 5
                )
            )
            if jumps
            else np.nan
        ),

        "jump_le_20_fraction": (
            float(
                np.mean(
                    np.asarray(
                        jumps
                    )
                    <= 20
                )
            )
            if jumps
            else np.nan
        ),
    }


# =====================================================================
# 10. Main
# =====================================================================

def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outdir = (
        OUTPUT_ROOT
        / f"{TASK_NAME}_{timestamp}"
    )

    a_current_dir = (
        outdir
        / "experiment_A"
        / "current_map_naclip_p85"
    )

    a_naclip_dir = (
        outdir
        / "experiment_A"
        / "naclip_map_naclip_p85"
    )

    b_dir = (
        outdir
        / "experiment_B"
        / "shared_current_map"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if SAVE_EXPERIMENT_A_EVENT_PANELS:
        a_current_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        a_naclip_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_EXPERIMENT_B_EVENT_PANELS:
        b_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT A/B: MAP vs SCORE ABLATION"
    )

    print(
        "=" * 100
    )

    print(
        "A1: Current map + NACLIP P85"
    )

    print(
        "A2: NACLIP map + NACLIP P85"
    )

    print(
        "B1/B2/B3 share the SAME Current-map zoom:"
    )

    print(
        "    Current-Gaussian / Current-P85 / NACLIP-P85"
    )

    print(
        "P85 =",
        PROGRESS_PERCENTILE,
    )

    print(
        "OUTPUT:",
        outdir,
    )

    print(
        "DEVICE:",
        DEVICE,
    )

    # -------------------------------------------------------------
    # config
    # -------------------------------------------------------------
    config = {
        "TASK_NAME": TASK_NAME,
        "NPZ_PATH": str(
            NPZ_PATH
        ),

        "TARGET_PROMPTS": TARGET_PROMPTS,

        "PROGRESS_PERCENTILE": PROGRESS_PERCENTILE,

        "RELEVANCE_THRESHOLD": RELEVANCE_THRESHOLD,
        "RELEVANCE_TEMPERATURE": RELEVANCE_TEMPERATURE,

        "NACLIP_GAUSSIAN_STD": NACLIP_GAUSSIAN_STD,
        "NACLIP_GAUSSIAN_WEIGHT": NACLIP_GAUSSIAN_WEIGHT,
        "NACLIP_INCLUDE_CLS": NACLIP_INCLUDE_CLS,

        "ZOOM_COOLDOWN_STEPS": ZOOM_COOLDOWN_STEPS,
        "MAX_ZOOM_FACTOR": MAX_ZOOM_FACTOR,

        "LATE_ZOOM_START": LATE_ZOOM_START,

        "experiment_A": {
            "A1": "Current map + NACLIP P85",
            "A2": "NACLIP map + NACLIP P85",
        },

        "experiment_B": {
            "shared_map": "Current map",
            "B1": "Current Gaussian",
            "B2": "Current P85",
            "B3": "NACLIP P85",
        },
    }

    (
        outdir
        / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # trajectory
    # -------------------------------------------------------------
    frames, rgb_key, _ = rv.load_frames(
        NPZ_PATH
    )

    end = (
        len(
            frames
        )
        if FRAME_END is None
        else min(
            len(
                frames
            ),
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

    evaluator = Evaluator()

    # A1/B share Current-map state machine.
    current_map_engine = NaturalZoomEngine()

    # A2 independent NACLIP-map state machine.
    naclip_map_engine = NaturalZoomEngine()

    current_results_all: List[
        RepresentationResult
    ] = []

    naclip_results_all: List[
        RepresentationResult
    ] = []

    current_map_events: List[
        AcceptedZoomEvent
    ] = []

    naclip_map_events: List[
        AcceptedZoomEvent
    ] = []

    all_raw_maps = []

    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # sequential replay
    # -------------------------------------------------------------
    for local_i, fi in enumerate(
        frame_indices
    ):
        frame = frames[
            int(
                fi
            )
        ]

        results = evaluator.evaluate(
            frame
        )

        current_results_all.append(
            results.current
        )

        naclip_results_all.append(
            results.naclip
        )

        all_raw_maps.append(
            results.current
            .raw_cosine_map
            .reshape(-1)
        )

        all_raw_maps.append(
            results.naclip
            .raw_cosine_map
            .reshape(-1)
        )

        # =========================================================
        # Current-map
        # Used by:
        #   Experiment A1
        #   Experiment B1/B2/B3
        # =========================================================
        current_check_threshold = (
            current_map_engine
            .update_current(
                results.current
            )
        )

        current_candidate = (
            current_map_engine
            .propose(
                frame=frame,
                step=int(
                    fi
                ),
                result=results.current,
                check_threshold=(
                    current_check_threshold
                ),
            )
        )

        if current_candidate.produced_crop:
            zoomed_results = evaluator.evaluate(
                current_candidate
                .zoomed_frame
            )

            post = (
                current_map_engine
                .accept(
                    current_result=(
                        results.current
                    ),
                    zoomed_result=(
                        zoomed_results.current
                    ),
                    candidate=(
                        current_candidate
                    ),
                )
            )

            if post[
                "is_zoomed"
            ]:
                current_map_events.append(
                    AcceptedZoomEvent(
                        map_source="current",

                        zoom_frame=int(
                            fi
                        ),

                        source_local_index=int(
                            local_i
                        ),

                        source_frame=(
                            frame.copy()
                        ),

                        zoomed_frame=(
                            current_candidate
                            .zoomed_frame
                            .copy()
                        ),

                        source_results=results,
                        zoomed_results=(
                            zoomed_results
                        ),

                        map_source_result=(
                            results.current
                        ),

                        zoomed_map_source_result=(
                            zoomed_results.current
                        ),

                        candidate=(
                            current_candidate
                        ),

                        post=post,
                    )
                )

                all_raw_maps.append(
                    zoomed_results.current
                    .raw_cosine_map
                    .reshape(-1)
                )

                all_raw_maps.append(
                    zoomed_results.naclip
                    .raw_cosine_map
                    .reshape(-1)
                )

        # =========================================================
        # NACLIP-map
        # Used by Experiment A2 only.
        # =========================================================
        naclip_check_threshold = (
            naclip_map_engine
            .update_current(
                results.naclip
            )
        )

        naclip_candidate = (
            naclip_map_engine
            .propose(
                frame=frame,
                step=int(
                    fi
                ),
                result=results.naclip,
                check_threshold=(
                    naclip_check_threshold
                ),
            )
        )

        if naclip_candidate.produced_crop:
            zoomed_results = evaluator.evaluate(
                naclip_candidate
                .zoomed_frame
            )

            post = (
                naclip_map_engine
                .accept(
                    current_result=(
                        results.naclip
                    ),
                    zoomed_result=(
                        zoomed_results.naclip
                    ),
                    candidate=(
                        naclip_candidate
                    ),
                )
            )

            if post[
                "is_zoomed"
            ]:
                naclip_map_events.append(
                    AcceptedZoomEvent(
                        map_source="naclip",

                        zoom_frame=int(
                            fi
                        ),

                        source_local_index=int(
                            local_i
                        ),

                        source_frame=(
                            frame.copy()
                        ),

                        zoomed_frame=(
                            naclip_candidate
                            .zoomed_frame
                            .copy()
                        ),

                        source_results=results,
                        zoomed_results=(
                            zoomed_results
                        ),

                        map_source_result=(
                            results.naclip
                        ),

                        zoomed_map_source_result=(
                            zoomed_results.naclip
                        ),

                        candidate=(
                            naclip_candidate
                        ),

                        post=post,
                    )
                )

                all_raw_maps.append(
                    zoomed_results.naclip
                    .raw_cosine_map
                    .reshape(-1)
                )

        if (
            local_i == 0
            or (
                local_i + 1
            )
            % PRINT_EVERY_N_FRAMES
            == 0
            or (
                local_i + 1
            )
            == len(
                frame_indices
            )
        ):
            print(
                f"[{local_i+1:4d}/{len(frame_indices):4d}] "
                f"frame={fi} | "
                f"Current-G={results.current.gaussian_score:.3f} | "
                f"Current-P85={results.current.percentile_score():.4f} | "
                f"NACLIP-P85={results.naclip.percentile_score():.4f}"
            )

    # -------------------------------------------------------------
    # Current true-trajectory score curves
    # -------------------------------------------------------------
    current_gaussian_curve = np.asarray(
        [
            r.gaussian_score
            for r in current_results_all
        ],
        dtype=np.float64,
    )

    current_p85_curve = np.asarray(
        [
            r.percentile_score()
            for r in current_results_all
        ],
        dtype=np.float64,
    )

    naclip_p85_curve = np.asarray(
        [
            r.percentile_score()
            for r in naclip_results_all
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Experiment A
    #
    # score always NACLIP P85;
    # only map/crop differs.
    # -------------------------------------------------------------
    experiment_a_rows: List[Dict] = []

    for method_name, events in [
        (
            "current_map_naclip_p85",
            current_map_events,
        ),
        (
            "naclip_map_naclip_p85",
            naclip_map_events,
        ),
    ]:
        for event_index, event in enumerate(
            events
        ):
            score_current_at_t = float(
                event.source_results.naclip
                .percentile_score()
            )

            score_on_zoomed = float(
                event.zoomed_results.naclip
                .percentile_score()
            )

            (
                crossing_local,
                crossing_frame,
                crossing_score,
            ) = first_crossing(
                frame_indices=frame_indices,
                score_curve=naclip_p85_curve,
                source_local_index=(
                    event.source_local_index
                ),
                score_on_zoomed=(
                    score_on_zoomed
                ),
            )

            if crossing_frame is None:
                jumping_steps = np.nan
            else:
                jumping_steps = int(
                    crossing_frame
                    - event.zoom_frame
                )

            row = {
                "experiment": "A",

                "method": method_name,

                "map_source": (
                    event.map_source
                ),

                "score_source": "naclip_p85",

                "event_index": int(
                    event_index
                ),

                "zoom_frame": int(
                    event.zoom_frame
                ),

                "late_zoom": int(
                    event.zoom_frame
                    >= LATE_ZOOM_START
                ),

                "score_current_at_t": float(
                    score_current_at_t
                ),

                "score_on_zoomed": float(
                    score_on_zoomed
                ),

                "crossing_local_index": (
                    np.nan
                    if crossing_local is None
                    else int(
                        crossing_local
                    )
                ),

                "crossing_frame": (
                    np.nan
                    if crossing_frame is None
                    else int(
                        crossing_frame
                    )
                ),

                "crossing_score": (
                    np.nan
                    if crossing_score is None
                    else float(
                        crossing_score
                    )
                ),

                "jumping_steps": (
                    jumping_steps
                ),

                "actual_zoom_factor": float(
                    event.candidate
                    .actual_zoom_factor
                ),

                "raw_zoom_factor": float(
                    event.candidate
                    .raw_zoom_factor
                ),

                "map_gaussian_current": float(
                    event.map_source_result
                    .gaussian_score
                ),

                "map_gaussian_zoomed": float(
                    event.zoomed_map_source_result
                    .gaussian_score
                ),

                "gaussian_required": float(
                    event.post[
                        "gaussian_required"
                    ]
                ),

                "crop_x1": int(
                    event.candidate.crop_x1
                ),
                "crop_y1": int(
                    event.candidate.crop_y1
                ),
                "crop_x2": int(
                    event.candidate.crop_x2
                ),
                "crop_y2": int(
                    event.candidate.crop_y2
                ),
            }

            experiment_a_rows.append(
                row
            )

    # -------------------------------------------------------------
    # Experiment B
    #
    # all scores share SAME Current-map accepted zoom events.
    # -------------------------------------------------------------
    experiment_b_rows: List[Dict] = []

    score_defs = {
        "current_gaussian": {
            "curve": current_gaussian_curve,
        },

        "current_p85": {
            "curve": current_p85_curve,
        },

        "naclip_p85": {
            "curve": naclip_p85_curve,
        },
    }

    b_rows_by_event: Dict[
        int,
        List[Dict]
    ] = {}

    for event_index, event in enumerate(
        current_map_events
    ):
        rows_for_event = []

        for score_name, score_def in (
            score_defs.items()
        ):
            if score_name == "current_gaussian":
                score_current_at_t = float(
                    event.source_results.current
                    .gaussian_score
                )

                score_on_zoomed = float(
                    event.zoomed_results.current
                    .gaussian_score
                )

            elif score_name == "current_p85":
                score_current_at_t = float(
                    event.source_results.current
                    .percentile_score()
                )

                score_on_zoomed = float(
                    event.zoomed_results.current
                    .percentile_score()
                )

            elif score_name == "naclip_p85":
                score_current_at_t = float(
                    event.source_results.naclip
                    .percentile_score()
                )

                score_on_zoomed = float(
                    event.zoomed_results.naclip
                    .percentile_score()
                )

            else:
                raise RuntimeError(
                    score_name
                )

            (
                crossing_local,
                crossing_frame,
                crossing_score,
            ) = first_crossing(
                frame_indices=frame_indices,
                score_curve=score_def[
                    "curve"
                ],
                source_local_index=(
                    event.source_local_index
                ),
                score_on_zoomed=(
                    score_on_zoomed
                ),
            )

            if crossing_frame is None:
                jumping_steps = np.nan
            else:
                jumping_steps = int(
                    crossing_frame
                    - event.zoom_frame
                )

            row = {
                "experiment": "B",

                "event_index": int(
                    event_index
                ),

                "zoom_frame": int(
                    event.zoom_frame
                ),

                "score_name": score_name,

                "score_current_at_t": float(
                    score_current_at_t
                ),

                "score_on_zoomed": float(
                    score_on_zoomed
                ),

                "crossing_local_index": (
                    np.nan
                    if crossing_local is None
                    else int(
                        crossing_local
                    )
                ),

                "crossing_frame": (
                    np.nan
                    if crossing_frame is None
                    else int(
                        crossing_frame
                    )
                ),

                "crossing_score": (
                    np.nan
                    if crossing_score is None
                    else float(
                        crossing_score
                    )
                ),

                "jumping_steps": (
                    jumping_steps
                ),

                "actual_zoom_factor": float(
                    event.candidate
                    .actual_zoom_factor
                ),
            }

            experiment_b_rows.append(
                row
            )

            rows_for_event.append(
                row
            )

        b_rows_by_event[
            event_index
        ] = rows_for_event

    # -------------------------------------------------------------
    # common raw cosine range
    # -------------------------------------------------------------
    all_raw = np.concatenate(
        all_raw_maps,
        axis=0,
    )

    raw_vmin = float(
        np.percentile(
            all_raw,
            RAW_COSINE_LOW_PERCENTILE,
        )
    )

    raw_vmax = float(
        np.percentile(
            all_raw,
            RAW_COSINE_HIGH_PERCENTILE,
        )
    )

    if raw_vmax <= raw_vmin:
        raw_vmax = (
            raw_vmin
            + 1e-3
        )

    # -------------------------------------------------------------
    # Experiment A panels
    # -------------------------------------------------------------
    if SAVE_EXPERIMENT_A_EVENT_PANELS:
        for method_name, events, output_dir in [
            (
                "current_map_naclip_p85",
                current_map_events,
                a_current_dir,
            ),
            (
                "naclip_map_naclip_p85",
                naclip_map_events,
                a_naclip_dir,
            ),
        ]:
            rows = [
                r
                for r in experiment_a_rows
                if r[
                    "method"
                ] == method_name
            ]

            for event, row in zip(
                events,
                rows,
            ):
                if np.isfinite(
                    row[
                        "crossing_frame"
                    ]
                ):
                    name = (
                        f"t{event.zoom_frame:06d}"
                        f"_k{int(row['crossing_frame']):06d}"
                        f"_J{int(row['jumping_steps']):04d}.png"
                    )
                else:
                    name = (
                        f"t{event.zoom_frame:06d}"
                        "_unresolved.png"
                    )

                save_experiment_a_event_panel(
                    path=(
                        output_dir
                        / name
                    ),

                    event=event,

                    event_row=row,

                    frames=frames,

                    naclip_results=(
                        naclip_results_all
                    ),

                    raw_vmin=raw_vmin,
                    raw_vmax=raw_vmax,
                )

    # -------------------------------------------------------------
    # Experiment B panels
    # -------------------------------------------------------------
    if SAVE_EXPERIMENT_B_EVENT_PANELS:
        for event_index, event in enumerate(
            current_map_events
        ):
            save_experiment_b_event_panel(
                path=(
                    b_dir
                    / f"t{event.zoom_frame:06d}.png"
                ),

                event=event,

                score_rows=(
                    b_rows_by_event[
                        event_index
                    ]
                ),

                frames=frames,
            )

    # -------------------------------------------------------------
    # late zoom contact sheets
    # -------------------------------------------------------------
    if SAVE_LATE_ZOOM_CONTACT_SHEETS:
        save_late_zoom_contact_sheet(
            outdir
            / "experiment_A"
            / "late_zoom_current_map.png",

            "Experiment A1: Current map",

            current_map_events,
        )

        save_late_zoom_contact_sheet(
            outdir
            / "experiment_A"
            / "late_zoom_naclip_map.png",

            "Experiment A2: NACLIP map",

            naclip_map_events,
        )

    # -------------------------------------------------------------
    # summaries
    # -------------------------------------------------------------
    a_current_rows = [
        r
        for r in experiment_a_rows
        if r[
            "method"
        ]
        == "current_map_naclip_p85"
    ]

    a_naclip_rows = [
        r
        for r in experiment_a_rows
        if r[
            "method"
        ]
        == "naclip_map_naclip_p85"
    ]

    experiment_a_summary = [
        summarize_zoom_branch(
            "Current-map + NACLIP-P85",
            current_map_events,
            a_current_rows,
        ),

        summarize_zoom_branch(
            "NACLIP-map + NACLIP-P85",
            naclip_map_events,
            a_naclip_rows,
        ),
    ]

    experiment_b_summary = [
        summarize_score_method(
            "current_gaussian",
            experiment_b_rows,
        ),

        summarize_score_method(
            "current_p85",
            experiment_b_rows,
        ),

        summarize_score_method(
            "naclip_p85",
            experiment_b_rows,
        ),
    ]

    # -------------------------------------------------------------
    # concise outputs
    # -------------------------------------------------------------
    save_csv(
        outdir
        / "experiment_A_events.csv",
        experiment_a_rows,
    )

    save_csv(
        outdir
        / "experiment_A_summary.csv",
        experiment_a_summary,
    )

    save_csv(
        outdir
        / "experiment_B_events.csv",
        experiment_b_rows,
    )

    save_csv(
        outdir
        / "experiment_B_summary.csv",
        experiment_b_summary,
    )

    save_experiment_a_event_raster(
        outdir
        / "experiment_A_event_raster.png",

        current_map_events,
        naclip_map_events,
    )

    save_experiment_b_jumping_plot(
        outdir
        / "experiment_B_jumping_steps.png",

        experiment_b_rows,
    )

    # -------------------------------------------------------------
    # console
    # -------------------------------------------------------------
    print()
    print(
        "=" * 100
    )

    print(
        "EXPERIMENT A SUMMARY"
    )

    print(
        "=" * 100
    )

    for row in experiment_a_summary:
        print()
        print(
            f"[{row['method']}]"
        )

        print(
            f"  accepted zoom      : {row['accepted_zoom']}"
        )

        print(
            f"  frames             : {row['accepted_zoom_frames']}"
        )

        print(
            f"  late zoom count    : {row['late_zoom_count']}"
        )

        print(
            f"  late zoom frames   : {row['late_zoom_frames']}"
        )

        print(
            f"  late zoom fraction : {row['late_zoom_fraction']}"
        )

        print(
            f"  mean inter-zoom gap: {row['mean_inter_zoom_gap']}"
        )

        print(
            f"  median jumping     : {row['jumping_steps_median']}"
        )

    print()
    print(
        "=" * 100
    )

    print(
        "EXPERIMENT B SUMMARY"
    )

    print(
        "=" * 100
    )

    for row in experiment_b_summary:
        print()
        print(
            f"[{row['score_name']}]"
        )

        print(
            f"  shared zoom events : {row['shared_current_map_zoom_events']}"
        )

        print(
            f"  resolve rate       : {row['resolve_rate']}"
        )

        print(
            f"  jump mean          : {row['jumping_steps_mean']}"
        )

        print(
            f"  jump median        : {row['jumping_steps_median']}"
        )

        print(
            f"  J<=5 fraction      : {row['jump_le_5_fraction']}"
        )

        print(
            f"  J<=20 fraction     : {row['jump_le_20_fraction']}"
        )

    print()
    print(
        f"Elapsed: {time.perf_counter()-t0:.2f}s"
    )

    print()
    print(
        "重点看："
    )

    print(
        "  experiment_A_summary.csv"
    )

    print(
        "  experiment_A_event_raster.png"
    )

    print(
        "  experiment_A/late_zoom_current_map.png"
    )

    print(
        "  experiment_A/late_zoom_naclip_map.png"
    )

    print(
        "  experiment_A/current_map_naclip_p85/"
    )

    print(
        "  experiment_A/naclip_map_naclip_p85/"
    )

    print(
        "  experiment_B_summary.csv"
    )

    print(
        "  experiment_B_jumping_steps.png"
    )

    print(
        "  experiment_B/shared_current_map/"
    )

    print()
    print(
        "Output:",
        outdir,
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
