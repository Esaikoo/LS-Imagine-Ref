#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter-free relevance-map test v2 for Natural Zoom
==================================================

Purpose
-------
Test whether the Current Value+Temporal(L=1) map can remove the manually
calibrated:

    RELEVANCE_THRESHOLD
    RELEVANCE_TEMPERATURE

without hurting Natural Zoom quality.

Map variants
------------
1) current_sigmoid  [baseline]
    Value
      -> Temporal(L=1)
      -> raw cosine
      -> sigmoid((cos - tau) / T)

2) current_raw_shifted  [parameter-free candidate]
    Value
      -> Temporal(L=1)
      -> raw cosine
      -> (cos + 1) / 2

    No relevance threshold.
    No relevance temperature.
    This is a positive affine transform of cosine and keeps the map in [0,1].

3) original_unet  [optional reference]
    Original LS-Imagine task-specific U-Net heatmap

Controlled progress score
-------------------------
To isolate MAP quality, all map variants use the SAME progress score:

    NACLIP K-K
      -> Temporal(L=1)
      -> raw cosine
      -> P85

Therefore the experiment mainly answers:

    Does removing sigmoid calibration change:
      - zoom trigger frequency?
      - crop position / crop size?
      - late repeated zoom?
      - accepted zoom quality?
      - first crossing under the same NACLIP-P85 score?

Important
---------
P85 does NOT determine crop location.
It is only used for ScoreStorage-style first crossing:

    first k > t such that:
        score_current(k) > score_on_zoomed(t)

The raw-shifted map branch never uses tau/T.

PyCharm usage
-------------
No argparse. Edit GLOBAL CONFIG and run directly.

Suggested location:
    <LS-Imagine>/relevance_map/parameter_free_map_test.py
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

# If needed:
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

TASK_SPECS = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS.exists():
    raise FileNotFoundError(
        f"PROJECT_ROOT looks wrong: {TASK_SPECS}"
    )

import relevance_map.relevance_variants as rv


# =====================================================================
# 1. GLOBAL CONFIG
# =====================================================================

TASK_NAME = "harvest_sand"

NPZ_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_approach_runs"
    / TASK_NAME
    / "world_60_taskseed_0_20260908_223444"
    / "trajectory.npz"
)


# ---------------------------------------------------------------------
# Optional no-target negative-control trajectory
#
# 建议另外手动保存一条 100~200 step 的 trajectory：
#   - 尽量不让目标树进入视野
#   - 看草地 / 天空 / 山坡 / 背对目标
#
# 然后设置：
# NEGATIVE_NPZ_PATH = PROJECT_ROOT / "..." / "trajectory.npz"
#
# 暂时没有就保持 None，主实验仍可运行。
# ---------------------------------------------------------------------

NEGATIVE_NPZ_PATH = None

NEGATIVE_FRAME_START = 0
NEGATIVE_FRAME_END = None
NEGATIVE_FRAME_STRIDE = 1

MINECLIP_CKPT = PROJECT_ROOT / "weights" / "mineclip_attn.pth"

UNET_CKPT = (
    PROJECT_ROOT
    / "affordance_map"
    / "finetune_unet"
    / "finetune_checkpoints"
    / TASK_NAME
    / "swin_unet_checkpoint.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "parameter_free_map_test_v2_outputs"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_H = 160
IMAGE_W = 256

TARGET_PROMPTS = ["Obtain sand"]


# ---------------------------------------------------------------------
# 1.1 What to test
# ---------------------------------------------------------------------

RUN_CURRENT_SIGMOID = True
RUN_CURRENT_RAW_SHIFTED = True

# Optional LS-Imagine reference.
RUN_ORIGINAL_UNET = True


# ---------------------------------------------------------------------
# 1.2 Baseline sigmoid calibration
#
# ONLY current_sigmoid uses these.
# current_raw_shifted does not use either value.
# ---------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016


# ---------------------------------------------------------------------
# 1.3 Fixed progress score for ALL branches
# ---------------------------------------------------------------------

PROGRESS_PERCENTILE = 85.0


# ---------------------------------------------------------------------
# 1.4 NACLIP parameters
#
# NACLIP is used only as the common P85 progress score in this test.
# It is NOT used to generate current_sigmoid/current_raw_shifted maps.
# ---------------------------------------------------------------------

NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True


# ---------------------------------------------------------------------
# 1.5 Natural Zoom settings
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
# 1.6 Late zoom diagnostics
# ---------------------------------------------------------------------

LATE_ZOOM_START = 90


# ---------------------------------------------------------------------
# 1.7 Trajectory
# ---------------------------------------------------------------------

FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1


# ---------------------------------------------------------------------
# 1.8 Visualization
# ---------------------------------------------------------------------

# Saves RGB + all maps + overlays for every frame.
# This is useful for checking whether raw-shifted keeps distance variation.
SAVE_EVERY_FRAME_MAP_COMPARISON = True

# Accepted zoom event panels.
SAVE_ACCEPTED_ZOOM_PANELS = True

# Contact sheet of late accepted zooms (t >= LATE_ZOOM_START).
SAVE_LATE_ZOOM_CONTACT_SHEETS = True

VIS_DPI = 150
PRINT_EVERY_N_FRAMES = 20

RAW_COSINE_LOW_PERCENTILE = 1.0
RAW_COSINE_HIGH_PERCENTILE = 99.0


# Raw-Shifted 的 contrast view 仅用于可视化；绝不进入任何算法计算。
CONTRAST_VIEW_LOW_PERCENTILE = 2.0
CONTRAST_VIEW_HIGH_PERCENTILE = 98.0

# Negative-control outputs
SAVE_NEGATIVE_EVERY_FRAME_MAP_COMPARISON = True
SAVE_NEGATIVE_ACCEPTED_ZOOM_PANELS = True


# =====================================================================
# 2. Branch names
# =====================================================================

BRANCH_CURRENT_SIGMOID = "current_sigmoid"
BRANCH_CURRENT_RAW = "current_raw_shifted"
BRANCH_UNET = "original_unet"

DISPLAY_NAME = {
    BRANCH_CURRENT_SIGMOID:
        "Current sigmoid: Value+Temporal -> sigmoid(tau,T)",

    BRANCH_CURRENT_RAW:
        "Current raw-shifted: Value+Temporal -> (cos+1)/2",

    BRANCH_UNET:
        "Original LS-Imagine U-Net",
}


# =====================================================================
# 3. Generic helpers
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
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
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

    xx, yy = np.meshgrid(
        x,
        y,
    )

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
    Same population-standard-deviation convention as current code.
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

    normalized_kurtosis = sigmoid_scalar(k)

    return float(
        normalized_kurtosis
        * (
            float(
                np.max(
                    map01
                )
            )
            -
            float(
                np.mean(
                    map01
                )
            )
        )
    )


# =====================================================================
# 4. MineCLIP / NACLIP feature extraction
# =====================================================================

@dataclass
class FeatureBundle:
    gh: int
    gw: int

    # Current Value + Temporal(L=1), [P,N]
    current_similarity: np.ndarray

    # NACLIP K-K + Temporal(L=1), [P,N]
    naclip_similarity: np.ndarray


class FeatureExtractor:
    def __init__(self):
        if not MINECLIP_CKPT.exists():
            raise FileNotFoundError(
                MINECLIP_CKPT
            )

        # relevance_variants reads module globals.
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

        # ---------------------------------------------------------
        # Current Value + Temporal(L=1)
        # ---------------------------------------------------------
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

        current_similarity = (
            self.tester.cosine(
                value_temporal,
                TARGET_PROMPTS,
            )[0]
        )

        # ---------------------------------------------------------
        # NACLIP + Temporal(L=1)
        # ---------------------------------------------------------
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

        naclip_similarity = (
            self.tester.cosine(
                naclip_temporal,
                TARGET_PROMPTS,
            )[0]
        )

        return FeatureBundle(
            gh=int(gh),
            gw=int(gw),

            current_similarity=(
                current_similarity
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),

            naclip_similarity=(
                naclip_similarity
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
        )


# =====================================================================
# 5. Map representations
# =====================================================================

@dataclass
class RawFeatureResult:
    """
    Raw features shared by map variants.
    """

    gh: int
    gw: int

    current_raw_patch_scores: np.ndarray
    current_raw_cosine_map: np.ndarray

    naclip_raw_patch_scores: np.ndarray
    naclip_raw_cosine_map: np.ndarray

    def naclip_percentile(
        self,
        q: float = PROGRESS_PERCENTILE,
    ) -> float:

        return float(
            np.percentile(
                self.naclip_raw_patch_scores,
                q,
            )
        )


@dataclass
class MapResult:
    branch: str

    # [H,W], what Natural Zoom actually sees.
    zoom_map: np.ndarray

    gaussian_score: float
    zoom_in_prob: float

    # Diagnostics.
    map_mean: float
    map_std: float
    map_min: float
    map_max: float
    map_p50: float
    map_p95: float
    map_p95_p50: float
    map_kurtosis: float
    map_max_minus_mean: float

    largest_cc_fraction: float

    # Underlying Current raw cosine, only for display.
    raw_cosine_map: Optional[np.ndarray]


def max_prompt_raw_scores(
    similarity: np.ndarray,
) -> np.ndarray:

    sim = np.asarray(
        similarity,
        dtype=np.float32,
    )

    return (
        np.max(
            sim,
            axis=0,
        )
        .astype(np.float32)
    )


def extract_raw_feature_result(
    features: FeatureBundle,
) -> RawFeatureResult:

    current_raw = max_prompt_raw_scores(
        features.current_similarity
    )

    naclip_raw = max_prompt_raw_scores(
        features.naclip_similarity
    )

    current_raw_map = upsample_patch_grid(
        current_raw.reshape(
            features.gh,
            features.gw,
        )
    )

    naclip_raw_map = upsample_patch_grid(
        naclip_raw.reshape(
            features.gh,
            features.gw,
        )
    )

    return RawFeatureResult(
        gh=features.gh,
        gw=features.gw,

        current_raw_patch_scores=(
            current_raw
        ),

        current_raw_cosine_map=(
            current_raw_map
        ),

        naclip_raw_patch_scores=(
            naclip_raw
        ),

        naclip_raw_cosine_map=(
            naclip_raw_map
        ),
    )


def build_current_sigmoid_map(
    features: FeatureBundle,
) -> np.ndarray:
    """
    Baseline:
        sigmoid((cos - tau) / T)
    """

    sim = np.asarray(
        features.current_similarity,
        dtype=np.float32,
    )

    relevance = stable_sigmoid_np(
        (
            sim
            - RELEVANCE_THRESHOLD
        )
        / RELEVANCE_TEMPERATURE
    )

    # max over prompts after monotonic transform
    relevance_max = np.max(
        relevance,
        axis=0,
    ).reshape(
        features.gh,
        features.gw,
    )

    return upsample_patch_grid(
        relevance_max
    )


def build_current_raw_shifted_map(
    features: FeatureBundle,
) -> np.ndarray:
    """
    Parameter-free map:

        raw cosine in [-1,1]
            -> (cos + 1) / 2

    No threshold.
    No temperature.

    Multi-prompt:
        max cosine over prompts first.

    Because (x+1)/2 is monotonic, max-before-transform and
    max-after-transform are equivalent.
    """

    raw = max_prompt_raw_scores(
        features.current_similarity
    )

    shifted = (
        raw
        + 1.0
    ) / 2.0

    # Only numerical safety; cosine should already lie in [-1,1].
    shifted = np.clip(
        shifted,
        0.0,
        1.0,
    ).astype(np.float32)

    shifted_grid = shifted.reshape(
        features.gh,
        features.gw,
    )

    return upsample_patch_grid(
        shifted_grid
    )


def make_map_result(
    branch: str,
    zoom_map: np.ndarray,
    raw_cosine_map: Optional[np.ndarray],
) -> MapResult:

    m = np.asarray(
        zoom_map,
        dtype=np.float32,
    )

    p50 = float(
        np.percentile(
            m,
            50,
        )
    )

    p95 = float(
        np.percentile(
            m,
            95,
        )
    )

    k = float(
        kurtosis(
            m.flatten()
        )
    )

    cc = largest_cc_fraction(
        m,
        threshold=float(
            (
                np.max(m)
                + np.min(m)
            )
            / 2.0
        ),
    )

    return MapResult(
        branch=branch,

        zoom_map=m,

        gaussian_score=gaussian_score(
            m
        ),

        zoom_in_prob=zoom_probability(
            m
        ),

        map_mean=float(
            np.mean(m)
        ),

        map_std=float(
            np.std(m)
        ),

        map_min=float(
            np.min(m)
        ),

        map_max=float(
            np.max(m)
        ),

        map_p50=p50,
        map_p95=p95,

        map_p95_p50=float(
            p95 - p50
        ),

        map_kurtosis=k,

        map_max_minus_mean=float(
            np.max(m)
            - np.mean(m)
        ),

        largest_cc_fraction=cc,

        raw_cosine_map=raw_cosine_map,
    )


@dataclass
class FrameEvaluation:
    raw: RawFeatureResult

    maps: Dict[str, MapResult]


class Evaluator:
    def __init__(self):
        self.extractor = FeatureExtractor()

        self.unet = None

        if RUN_ORIGINAL_UNET:
            if not UNET_CKPT.exists():
                print(
                    "[WARN] U-Net checkpoint missing; "
                    "continue without U-Net:",
                    UNET_CKPT,
                )
            else:
                try:
                    self.unet = rv.OriginalUNet(
                        UNET_CKPT,
                        self.extractor.tester,
                    )

                    print(
                        "[OK] Original U-Net loaded"
                    )
                except Exception as e:
                    print(
                        "[WARN] U-Net load failed; "
                        "continue without it:",
                        repr(e),
                    )

    @torch.inference_mode()
    def evaluate(
        self,
        frame: np.ndarray,
    ) -> FrameEvaluation:

        features = self.extractor.extract(
            frame
        )

        raw = extract_raw_feature_result(
            features
        )

        maps: Dict[
            str,
            MapResult
        ] = {}

        if RUN_CURRENT_SIGMOID:
            m = build_current_sigmoid_map(
                features
            )

            maps[
                BRANCH_CURRENT_SIGMOID
            ] = make_map_result(
                branch=BRANCH_CURRENT_SIGMOID,
                zoom_map=m,
                raw_cosine_map=(
                    raw.current_raw_cosine_map
                ),
            )

        if RUN_CURRENT_RAW_SHIFTED:
            m = build_current_raw_shifted_map(
                features
            )

            maps[
                BRANCH_CURRENT_RAW
            ] = make_map_result(
                branch=BRANCH_CURRENT_RAW,
                zoom_map=m,
                raw_cosine_map=(
                    raw.current_raw_cosine_map
                ),
            )

        if self.unet is not None:
            m = self.unet.generate(
                frame,
                TARGET_PROMPTS,
            )

            m = np.asarray(
                m,
                dtype=np.float32,
            )

            # Original U-Net is treated as its own map.
            maps[
                BRANCH_UNET
            ] = make_map_result(
                branch=BRANCH_UNET,
                zoom_map=m,
                raw_cosine_map=None,
            )

        return FrameEvaluation(
            raw=raw,
            maps=maps,
        )


# =====================================================================
# 6. Natural Zoom state machine
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
    Every map branch has its own:
        check buffer
        Gaussian buffer
        cooldown state
    """

    def __init__(self):
        self.check_buffer = RunningStats()
        self.gaussian_buffer = RunningStats()

        self.last_zoom_attempt_step = None

    def update_current(
        self,
        result: MapResult,
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
        result: MapResult,
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

        # Current semantic gates are normally disabled.
        semantic_ok = (
            result.map_p95_p50
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
                np.mean(
                    values
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
            / float(
                win_w
            ),
            H
            / float(
                win_h
            ),
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
            x1 + win_w
        )

        y2 = int(
            y1 + win_h
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

        # Current implementation starts cooldown after a real crop attempt.
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
        current_result: MapResult,
        zoomed_result: MapResult,
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

        contrast_gain = (
            zoomed_result.map_p95_p50
            - current_result.map_p95_p50
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
            contrast_gain
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
            is_zoomed = gaussian_gain_ok

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
        }


# =====================================================================
# 7. Accepted event
# =====================================================================

@dataclass
class AcceptedZoomEvent:
    branch: str

    zoom_frame: int
    source_local_index: int

    source_frame: np.ndarray
    zoomed_frame: np.ndarray

    source_eval: FrameEvaluation
    zoomed_eval: FrameEvaluation

    source_map: MapResult
    zoomed_map: MapResult

    candidate: CropCandidate
    post: Dict


# =====================================================================
# 8. Crossing
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
# 9. Visualization
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


def _contrast_limits_for_display(
    map2d: np.ndarray,
):
    """
    DISPLAY ONLY.

    用当前帧自己的百分位拉伸颜色，帮助观察热点位置。
    这个范围永远不会被 Natural Zoom / threshold / score 使用。
    """
    arr = np.asarray(
        map2d,
        dtype=np.float32,
    )

    lo = float(
        np.percentile(
            arr,
            CONTRAST_VIEW_LOW_PERCENTILE,
        )
    )

    hi = float(
        np.percentile(
            arr,
            CONTRAST_VIEW_HIGH_PERCENTILE,
        )
    )

    if not np.isfinite(lo):
        lo = float(np.min(arr))

    if not np.isfinite(hi):
        hi = float(np.max(arr))

    if hi <= lo:
        hi = lo + 1e-6

    return lo, hi


def save_frame_map_comparison(
    path: Path,
    frame: np.ndarray,
    frame_index: int,
    maps: Dict[str, MapResult],
):
    """
    V2 visualisation.

    重点把 Raw-Shifted 分成两种显示：

    1) ABSOLUTE [0,1]
       固定 vmin=0 / vmax=1。
       用来判断绝对数值是否真的高。

    2) CONTRAST (DISPLAY ONLY)
       当前帧内部局部拉伸。
       只用来观察热点位置。

    两种图读取的是完全相同的 result.zoom_map；
    第二种只改变 imshow 的 vmin/vmax，不改变算法数据。
    """

    has_sigmoid = BRANCH_CURRENT_SIGMOID in maps
    has_raw = BRANCH_CURRENT_RAW in maps
    has_unet = BRANCH_UNET in maps

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(18, 8),
        squeeze=False,
    )

    # -------------------------------------------------------------
    # Row 0
    # -------------------------------------------------------------
    ax[0, 0].imshow(frame)
    ax[0, 0].set_title(
        f"RGB | frame={frame_index}"
    )
    ax[0, 0].axis("off")

    if has_sigmoid:
        r = maps[BRANCH_CURRENT_SIGMOID]
        ax[0, 1].imshow(
            r.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        ax[0, 1].set_title(
            "Current-Sigmoid | ABSOLUTE [0,1]\n"
            f"range={r.map_min:.4f}..{r.map_max:.4f}\n"
            f"std={r.map_std:.5f} | Zprob={r.zoom_in_prob:.5f}"
        )
        ax[0, 1].axis("off")
    else:
        ax[0, 1].axis("off")

    if has_raw:
        r = maps[BRANCH_CURRENT_RAW]

        # Raw absolute view: fixed [0,1]
        ax[0, 2].imshow(
            r.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        ax[0, 2].set_title(
            "Raw-Shifted | ABSOLUTE [0,1]\n"
            f"true range={r.map_min:.4f}..{r.map_max:.4f}\n"
            f"std={r.map_std:.5f} | Zprob={r.zoom_in_prob:.5f}"
        )
        ax[0, 2].axis("off")

        # Raw contrast view: display only
        lo, hi = _contrast_limits_for_display(
            r.zoom_map
        )
        ax[0, 3].imshow(
            r.zoom_map,
            cmap="jet",
            vmin=lo,
            vmax=hi,
        )
        ax[0, 3].set_title(
            "Raw-Shifted | CONTRAST\n"
            f"display range={lo:.4f}..{hi:.4f}\n"
            "DISPLAY ONLY — NOT USED BY ZOOM"
        )
        ax[0, 3].axis("off")
    else:
        ax[0, 2].axis("off")
        ax[0, 3].axis("off")

    # -------------------------------------------------------------
    # Row 1
    # -------------------------------------------------------------
    if has_unet:
        r = maps[BRANCH_UNET]
        ax[1, 0].imshow(
            r.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        ax[1, 0].set_title(
            "Original U-Net | ABSOLUTE [0,1]\n"
            f"range={r.map_min:.4f}..{r.map_max:.4f}"
        )
        ax[1, 0].axis("off")
    else:
        ax[1, 0].imshow(frame)
        ax[1, 0].set_title("RGB reference")
        ax[1, 0].axis("off")

    if has_sigmoid:
        r = maps[BRANCH_CURRENT_SIGMOID]
        ax[1, 1].imshow(frame)
        ax[1, 1].imshow(
            r.zoom_map,
            cmap="jet",
            alpha=0.50,
            vmin=0.0,
            vmax=1.0,
        )
        ax[1, 1].set_title(
            "Sigmoid overlay | absolute scale"
        )
        ax[1, 1].axis("off")
    else:
        ax[1, 1].axis("off")

    if has_raw:
        r = maps[BRANCH_CURRENT_RAW]

        ax[1, 2].imshow(frame)
        ax[1, 2].imshow(
            r.zoom_map,
            cmap="jet",
            alpha=0.50,
            vmin=0.0,
            vmax=1.0,
        )
        ax[1, 2].set_title(
            "Raw overlay | ABSOLUTE [0,1]"
        )
        ax[1, 2].axis("off")

        lo, hi = _contrast_limits_for_display(
            r.zoom_map
        )
        ax[1, 3].imshow(frame)
        ax[1, 3].imshow(
            r.zoom_map,
            cmap="jet",
            alpha=0.50,
            vmin=lo,
            vmax=hi,
        )
        ax[1, 3].set_title(
            "Raw overlay | CONTRAST\n"
            "DISPLAY ONLY"
        )
        ax[1, 3].axis("off")
    else:
        ax[1, 2].axis("off")
        ax[1, 3].axis("off")

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

def save_event_panel(
    path: Path,
    event: AcceptedZoomEvent,
    event_row: Dict,
    frames: np.ndarray,
    naclip_results: List[RawFeatureResult],
    raw_vmin: float,
    raw_vmax: float,
):
    """
    Per accepted zoom:
        source+crop | zoomed | first crossing
        source map  | zoomed map | crossing NACLIP cosine
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
        f"{DISPLAY_NAME[event.branch]}\n"
        f"source t={event.zoom_frame}\n"
        f"NACLIP P85={event_row['score_current_at_t']:.4f}"
    )

    ax[0, 1].imshow(
        event.zoomed_frame
    )

    ax[0, 1].set_title(
        f"accepted zoomed image\n"
        f"zoom={event.candidate.actual_zoom_factor:.2f}x\n"
        f"zoomed P85={event_row['score_on_zoomed']:.4f}"
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

    # source/zoom maps
    if event.branch == BRANCH_CURRENT_RAW:
        ax[1, 0].imshow(
            event.source_map.zoom_map,
            cmap="jet",
        )

        ax[1, 1].imshow(
            event.zoomed_map.zoom_map,
            cmap="jet",
        )
    else:
        ax[1, 0].imshow(
            event.source_map.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )

        ax[1, 1].imshow(
            event.zoomed_map.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )

    ax[1, 0].set_title(
        f"source zoom map\n"
        f"range={event.source_map.map_min:.3f}.."
        f"{event.source_map.map_max:.3f}"
    )

    ax[1, 1].set_title(
        f"zoomed zoom map\n"
        f"range={event.zoomed_map.map_min:.3f}.."
        f"{event.zoomed_map.map_max:.3f}"
    )

    if np.isfinite(
        event_row[
            "crossing_local_index"
        ]
    ):
        local_k = int(
            event_row[
                "crossing_local_index"
            ]
        )

        ax[1, 2].imshow(
            naclip_results[
                local_k
            ].naclip_raw_cosine_map,
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 2].set_title(
            "crossing NACLIP raw cosine"
        )
    else:
        ax[1, 2].imshow(
            np.zeros(
                (
                    IMAGE_H,
                    IMAGE_W,
                ),
                dtype=np.float32,
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
        "Map calibration ablation | progress score fixed = NACLIP P85",
        fontsize=14,
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


def save_event_raster(
    path: Path,
    events_by_branch: Dict[
        str,
        List[AcceptedZoomEvent]
    ],
):
    fig, ax = plt.subplots(
        figsize=(14, 5),
    )

    active = list(
        events_by_branch.keys()
    )

    for y, branch in enumerate(
        reversed(
            active
        )
    ):
        xs = [
            event.zoom_frame
            for event in events_by_branch[
                branch
            ]
        ]

        ax.scatter(
            xs,
            [y] * len(xs),
            s=70,
            label=DISPLAY_NAME[
                branch
            ],
        )

    ax.axvline(
        LATE_ZOOM_START,
        linestyle="--",
        alpha=0.65,
        label=f"late start={LATE_ZOOM_START}",
    )

    ax.set_yticks(
        list(
            range(
                len(
                    active
                )
            )
        )
    )

    ax.set_yticklabels(
        [
            DISPLAY_NAME[b]
            for b in reversed(
                active
            )
        ]
    )

    ax.set_xlabel(
        "Trajectory frame index"
    )

    ax.set_title(
        "Accepted Natural-Zoom events"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_map_statistics_plot(
    path: Path,
    frame_indices: np.ndarray,
    per_frame_rows: List[Dict],
    active_branches: List[str],
):
    """
    One file with two separate figures is avoided.
    Here we make a single clean 2-row figure:
        zoom probability
        map max - mean
    """

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        squeeze=False,
    )

    ax0 = ax[0, 0]
    ax1 = ax[1, 0]

    for branch in active_branches:
        zoom_prob = np.asarray(
            [
                row[
                    f"{branch}_zoom_prob"
                ]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        contrast = np.asarray(
            [
                row[
                    f"{branch}_max_minus_mean"
                ]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        ax0.plot(
            frame_indices,
            zoom_prob,
            label=DISPLAY_NAME[
                branch
            ],
        )

        ax1.plot(
            frame_indices,
            contrast,
            label=DISPLAY_NAME[
                branch
            ],
        )

    ax0.set_ylabel(
        "zoom_in_prob"
    )

    ax0.set_title(
        "Natural-Zoom trigger statistic"
    )

    ax0.grid(
        alpha=0.25
    )

    ax0.legend(
        fontsize=8,
    )

    ax1.set_ylabel(
        "map max - mean"
    )

    ax1.set_xlabel(
        "Trajectory frame index"
    )

    ax1.set_title(
        "Map contrast used inside zoom probability"
    )

    ax1.grid(
        alpha=0.25
    )

    ax1.legend(
        fontsize=8,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_jumping_steps_plot(
    path: Path,
    event_rows: List[Dict],
    active_branches: List[str],
):
    fig, ax = plt.subplots(
        figsize=(14, 6),
    )

    for branch in active_branches:
        rows = [
            row
            for row in event_rows
            if row[
                "branch"
            ] == branch
            and np.isfinite(
                row[
                    "jumping_steps"
                ]
            )
        ]

        if not rows:
            continue

        x = [
            int(
                row[
                    "zoom_frame"
                ]
            )
            for row in rows
        ]

        y = [
            int(
                row[
                    "jumping_steps"
                ]
            )
            for row in rows
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=DISPLAY_NAME[
                branch
            ],
        )

    ax.set_xlabel(
        "accepted zoom frame t"
    )

    ax.set_ylabel(
        "jumping_steps"
    )

    ax.set_title(
        "Same NACLIP-P85 progress score, different map generation"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=175,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_late_zoom_contact_sheet(
    path: Path,
    branch: str,
    events: List[AcceptedZoomEvent],
):
    late = [
        event
        for event in events
        if event.zoom_frame
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
        f"{DISPLAY_NAME[branch]}\n"
        f"late accepted zooms (t >= {LATE_ZOOM_START})",
        fontsize=14,
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


# =====================================================================
# 10. Summary
# =====================================================================

def summarize_branch(
    branch: str,
    events: List[AcceptedZoomEvent],
    rows: List[Dict],
) -> Dict:

    frames = sorted(
        [
            event.zoom_frame
            for event in events
        ]
    )

    late_frames = [
        x
        for x in frames
        if x >= LATE_ZOOM_START
    ]

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

    jumps = [
        int(
            row[
                "jumping_steps"
            ]
        )
        for row in rows
        if np.isfinite(
            row[
                "jumping_steps"
            ]
        )
    ]

    return {
        "branch": branch,

        "display_name": DISPLAY_NAME[
            branch
        ],

        "uses_relevance_threshold": int(
            branch
            == BRANCH_CURRENT_SIGMOID
        ),

        "uses_relevance_temperature": int(
            branch
            == BRANCH_CURRENT_SIGMOID
        ),

        "accepted_zoom": int(
            len(
                events
            )
        ),

        "accepted_zoom_frames": ",".join(
            str(x)
            for x in frames
        ),

        "late_zoom_count": int(
            len(
                late_frames
            )
        ),

        "late_zoom_frames": ",".join(
            str(x)
            for x in late_frames
        ),

        "late_zoom_fraction": (
            float(
                len(
                    late_frames
                )
                / len(
                    frames
                )
            )
            if frames
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
                    rows
                )
            )
            if rows
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


# =====================================================================
# 11. Main
# =====================================================================


# =====================================================================
# Negative-control helpers (V2)
# =====================================================================

def _select_negative_indices(n_frames: int):
    end = (
        n_frames
        if NEGATIVE_FRAME_END is None
        else min(
            n_frames,
            int(NEGATIVE_FRAME_END),
        )
    )

    indices = np.arange(
        max(0, int(NEGATIVE_FRAME_START)),
        end,
        max(1, int(NEGATIVE_FRAME_STRIDE)),
        dtype=np.int64,
    )

    if len(indices) == 0:
        raise RuntimeError(
            "No negative-control frames selected."
        )

    return indices


def _save_negative_zoom_panel(
    path: Path,
    branch: str,
    source_frame: np.ndarray,
    zoomed_frame: np.ndarray,
    source_map: MapResult,
    zoomed_map: MapResult,
    candidate: CropCandidate,
    post: Dict,
):
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        squeeze=False,
    )

    source_box = crop_box_image(
        source_frame,
        candidate,
    )

    ax[0, 0].imshow(source_box)
    ax[0, 0].set_title(
        f"{DISPLAY_NAME[branch]}\nNO-TARGET source + crop"
    )

    ax[0, 1].imshow(zoomed_frame)
    ax[0, 1].set_title(
        "NO-TARGET accepted zoom"
    )

    ax[0, 2].axis("off")
    ax[0, 2].text(
        0.02,
        0.98,
        (
            f"map range={source_map.map_min:.4f}..{source_map.map_max:.4f}\n"
            f"map std={source_map.map_std:.6f}\n"
            f"max-mean={source_map.map_max_minus_mean:.6f}\n"
            f"zoom_prob={source_map.zoom_in_prob:.6f}\n\n"
            f"G current={source_map.gaussian_score:.4f}\n"
            f"G zoomed={zoomed_map.gaussian_score:.4f}\n"
            f"G required={post['gaussian_required']:.4f}"
        ),
        va="top",
        transform=ax[0, 2].transAxes,
        family="monospace",
        fontsize=10,
    )

    # Absolute source map
    ax[1, 0].imshow(
        source_map.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    ax[1, 0].set_title(
        "Source map | ABSOLUTE [0,1]\n"
        f"true range={source_map.map_min:.4f}..{source_map.map_max:.4f}"
    )

    # Contrast source map — display only
    lo, hi = _contrast_limits_for_display(
        source_map.zoom_map
    )
    ax[1, 1].imshow(
        source_map.zoom_map,
        cmap="jet",
        vmin=lo,
        vmax=hi,
    )
    ax[1, 1].set_title(
        "Source map | CONTRAST\n"
        f"display range={lo:.4f}..{hi:.4f}\n"
        "DISPLAY ONLY"
    )

    # Absolute zoomed map
    ax[1, 2].imshow(
        zoomed_map.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    ax[1, 2].set_title(
        "Zoomed map | ABSOLUTE [0,1]"
    )

    for a in ax.reshape(-1):
        if a is not ax[0, 2]:
            a.axis("off")

    plt.tight_layout()
    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_negative_control(
    evaluator: Evaluator,
    outdir: Path,
):
    """
    Optional no-target trajectory test.

    这里不评价 first crossing，因为目标本来就应该不存在。
    每一个 accepted zoom 都视作可疑 false-positive zoom。
    """

    if NEGATIVE_NPZ_PATH is None:
        print()
        print(
            "[INFO] NEGATIVE_NPZ_PATH=None; "
            "skip no-target negative-control test."
        )
        return None

    negative_path = Path(
        NEGATIVE_NPZ_PATH
    )

    if not negative_path.exists():
        print()
        print(
            "[WARN] Negative-control NPZ not found; skip:"
        )
        print(negative_path)
        return None

    print()
    print("=" * 100)
    print("NO-TARGET NEGATIVE CONTROL")
    print("=" * 100)

    frames, rgb_key, _ = rv.load_frames(
        negative_path
    )

    frame_indices = _select_negative_indices(
        len(frames)
    )

    print(
        f"RGB key={rgb_key}; total={len(frames)}; "
        f"tested={len(frame_indices)}"
    )

    # Determine branch set from first frame.
    first_eval = evaluator.evaluate(
        frames[int(frame_indices[0])]
    )
    active_branches = list(
        first_eval.maps.keys()
    )

    engines = {
        branch: NaturalZoomEngine()
        for branch in active_branches
    }

    negative_dir = (
        outdir
        / "negative_control"
    )
    negative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_dir = (
        negative_dir
        / "frame_map_comparisons"
    )
    if SAVE_NEGATIVE_EVERY_FRAME_MAP_COMPARISON:
        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    panel_root = (
        negative_dir
        / "accepted_zoom_panels"
    )
    if SAVE_NEGATIVE_ACCEPTED_ZOOM_PANELS:
        panel_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    per_frame_rows = []
    accepted_by_branch = {
        branch: []
        for branch in active_branches
    }

    for local_i, fi in enumerate(frame_indices):
        frame = frames[int(fi)]
        evaluation = evaluator.evaluate(frame)

        row = {
            "frame_index": int(fi),
        }

        for branch in active_branches:
            result = evaluation.maps[branch]

            check_threshold = engines[branch].update_current(
                result
            )

            row[f"{branch}_map_min"] = float(result.map_min)
            row[f"{branch}_map_max"] = float(result.map_max)
            row[f"{branch}_map_mean"] = float(result.map_mean)
            row[f"{branch}_map_std"] = float(result.map_std)
            row[f"{branch}_map_range"] = float(
                result.map_max - result.map_min
            )
            row[f"{branch}_max_minus_mean"] = float(
                result.map_max_minus_mean
            )
            row[f"{branch}_zoom_prob"] = float(
                result.zoom_in_prob
            )
            row[f"{branch}_check_threshold"] = float(
                check_threshold
            )

            candidate = engines[branch].propose(
                frame=frame,
                step=int(fi),
                result=result,
                check_threshold=check_threshold,
            )

            row[f"{branch}_crop_attempt"] = int(
                candidate.produced_crop
            )

            accepted = 0

            if candidate.produced_crop:
                zoomed_eval = evaluator.evaluate(
                    candidate.zoomed_frame
                )
                zoomed_map = zoomed_eval.maps[branch]

                post = engines[branch].accept(
                    current_result=result,
                    zoomed_result=zoomed_map,
                    candidate=candidate,
                )

                if post["is_zoomed"]:
                    accepted = 1

                    event = {
                        "frame_index": int(fi),
                        "source_frame": frame.copy(),
                        "zoomed_frame": candidate.zoomed_frame.copy(),
                        "source_map": result,
                        "zoomed_map": zoomed_map,
                        "candidate": candidate,
                        "post": post,
                    }
                    accepted_by_branch[branch].append(event)

                    if SAVE_NEGATIVE_ACCEPTED_ZOOM_PANELS:
                        branch_dir = panel_root / branch
                        branch_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        _save_negative_zoom_panel(
                            path=(
                                branch_dir
                                / f"false_zoom_t{int(fi):06d}.png"
                            ),
                            branch=branch,
                            source_frame=frame,
                            zoomed_frame=candidate.zoomed_frame,
                            source_map=result,
                            zoomed_map=zoomed_map,
                            candidate=candidate,
                            post=post,
                        )

            row[f"{branch}_accepted_zoom"] = int(accepted)

        per_frame_rows.append(row)

        if SAVE_NEGATIVE_EVERY_FRAME_MAP_COMPARISON:
            save_frame_map_comparison(
                path=(
                    frame_dir
                    / f"frame_{int(fi):06d}.png"
                ),
                frame=frame,
                frame_index=int(fi),
                maps=evaluation.maps,
            )

        if (
            local_i == 0
            or (local_i + 1) % PRINT_EVERY_N_FRAMES == 0
            or local_i + 1 == len(frame_indices)
        ):
            print(
                f"[negative {local_i+1:4d}/{len(frame_indices):4d}] "
                f"frame={fi}"
            )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    summary_rows = []
    n_frames = len(frame_indices)

    for branch in active_branches:
        zoom_prob = np.asarray(
            [
                row[f"{branch}_zoom_prob"]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        max_minus_mean = np.asarray(
            [
                row[f"{branch}_max_minus_mean"]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        map_std = np.asarray(
            [
                row[f"{branch}_map_std"]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        map_range = np.asarray(
            [
                row[f"{branch}_map_range"]
                for row in per_frame_rows
            ],
            dtype=np.float64,
        )

        attempts = int(
            np.sum(
                [
                    row[f"{branch}_crop_attempt"]
                    for row in per_frame_rows
                ]
            )
        )

        accepted_frames = [
            int(event["frame_index"])
            for event in accepted_by_branch[branch]
        ]
        accepted = len(accepted_frames)

        summary_rows.append(
            {
                "branch": branch,
                "display_name": DISPLAY_NAME[branch],
                "tested_frames": int(n_frames),
                "crop_attempts": attempts,
                "accepted_zoom": int(accepted),
                "accepted_zoom_frames": ",".join(
                    str(x)
                    for x in accepted_frames
                ),
                "crop_attempts_per_100_frames": float(
                    attempts / max(1, n_frames) * 100.0
                ),
                "accepted_zoom_per_100_frames": float(
                    accepted / max(1, n_frames) * 100.0
                ),
                "zoom_prob_mean": float(np.mean(zoom_prob)),
                "zoom_prob_p95": float(np.percentile(zoom_prob, 95)),
                "zoom_prob_max": float(np.max(zoom_prob)),
                "max_minus_mean_mean": float(np.mean(max_minus_mean)),
                "max_minus_mean_p95": float(
                    np.percentile(max_minus_mean, 95)
                ),
                "max_minus_mean_max": float(np.max(max_minus_mean)),
                "map_std_mean": float(np.mean(map_std)),
                "map_std_p95": float(np.percentile(map_std, 95)),
                "map_range_mean": float(np.mean(map_range)),
                "map_range_p95": float(np.percentile(map_range, 95)),
            }
        )

    save_csv(
        negative_dir / "negative_per_frame_stats.csv",
        per_frame_rows,
    )
    save_csv(
        negative_dir / "negative_summary.csv",
        summary_rows,
    )

    # -------------------------------------------------------------
    # Simple false-zoom bar chart
    # -------------------------------------------------------------
    names = [
        row["display_name"]
        for row in summary_rows
    ]
    attempt_rates = [
        row["crop_attempts_per_100_frames"]
        for row in summary_rows
    ]
    accepted_rates = [
        row["accepted_zoom_per_100_frames"]
        for row in summary_rows
    ]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )
    ax.bar(
        x - width / 2,
        attempt_rates,
        width,
        label="crop attempts / 100 frames",
    )
    ax.bar(
        x + width / 2,
        accepted_rates,
        width,
        label="accepted zoom / 100 frames",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=10,
        ha="right",
    )
    ax.set_ylabel(
        "Count per 100 no-target frames"
    )
    ax.set_title(
        "No-target negative control: false zoom activity"
    )
    ax.grid(
        axis="y",
        alpha=0.25,
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        negative_dir / "negative_false_zoom_summary.png",
        dpi=175,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print("Negative-control summary:")
    for row in summary_rows:
        print(
            f"  [{row['display_name']}] "
            f"attempts/100={row['crop_attempts_per_100_frames']:.2f}, "
            f"accepted/100={row['accepted_zoom_per_100_frames']:.2f}, "
            f"frames={row['accepted_zoom_frames']}"
        )

    return {
        "summary_rows": summary_rows,
        "per_frame_rows": per_frame_rows,
        "accepted_by_branch": accepted_by_branch,
    }

def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outdir = (
        OUTPUT_ROOT
        / f"{TASK_NAME}_{timestamp}"
    )

    frame_dir = (
        outdir
        / "frame_map_comparisons"
    )

    event_root = (
        outdir
        / "accepted_zoom_panels"
    )

    late_root = (
        outdir
        / "late_zoom_contact_sheets"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if SAVE_EVERY_FRAME_MAP_COMPARISON:
        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_ACCEPTED_ZOOM_PANELS:
        event_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_LATE_ZOOM_CONTACT_SHEETS:
        late_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "=" * 100
    )

    print(
        "PARAMETER-FREE MAP TEST V2"
    )

    print(
        "=" * 100
    )

    print(
        "Baseline map:"
    )

    print(
        f"  sigmoid((cos - {RELEVANCE_THRESHOLD}) / "
        f"{RELEVANCE_TEMPERATURE})"
    )

    print(
        "Parameter-free candidate:"
    )

    print(
        "  (cosine + 1) / 2"
    )

    print(
        f"All branches use NACLIP-P{PROGRESS_PERCENTILE:g} "
        "for first crossing."
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
    # run config
    # -------------------------------------------------------------
    config = {
        "TASK_NAME": TASK_NAME,

        "NPZ_PATH": str(
            NPZ_PATH
        ),

        "TARGET_PROMPTS": TARGET_PROMPTS,

        "RUN_CURRENT_SIGMOID": RUN_CURRENT_SIGMOID,
        "RUN_CURRENT_RAW_SHIFTED": RUN_CURRENT_RAW_SHIFTED,
        "RUN_ORIGINAL_UNET": RUN_ORIGINAL_UNET,

        "current_sigmoid": {
            "formula":
                "sigmoid((cosine - tau) / T)",

            "RELEVANCE_THRESHOLD":
                RELEVANCE_THRESHOLD,

            "RELEVANCE_TEMPERATURE":
                RELEVANCE_TEMPERATURE,
        },

        "current_raw_shifted": {
            "formula":
                "(cosine + 1) / 2",

            "uses_threshold": False,
            "uses_temperature": False,
        },

        "progress_score": {
            "source":
                "NACLIP K-K + Temporal(L=1)",

            "percentile":
                PROGRESS_PERCENTILE,
        },

        "NACLIP_GAUSSIAN_STD":
            NACLIP_GAUSSIAN_STD,

        "NACLIP_GAUSSIAN_WEIGHT":
            NACLIP_GAUSSIAN_WEIGHT,

        "NACLIP_INCLUDE_CLS":
            NACLIP_INCLUDE_CLS,

        "MAX_ZOOM_FACTOR":
            MAX_ZOOM_FACTOR,

        "ZOOM_COOLDOWN_STEPS":
            ZOOM_COOLDOWN_STEPS,

        "LATE_ZOOM_START":
            LATE_ZOOM_START,
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

    # Determine active branches after possible U-Net load failure.
    active_branches = []

    if RUN_CURRENT_SIGMOID:
        active_branches.append(
            BRANCH_CURRENT_SIGMOID
        )

    if RUN_CURRENT_RAW_SHIFTED:
        active_branches.append(
            BRANCH_CURRENT_RAW
        )

    if evaluator.unet is not None:
        active_branches.append(
            BRANCH_UNET
        )

    print(
        "Active branches:",
        active_branches,
    )

    engines = {
        branch: NaturalZoomEngine()
        for branch in active_branches
    }

    events_by_branch: Dict[
        str,
        List[AcceptedZoomEvent]
    ] = {
        branch: []
        for branch in active_branches
    }

    raw_results_all: List[
        RawFeatureResult
    ] = []

    # Per-frame map results kept for visualization/statistics.
    frame_maps_all: List[
        Dict[str, MapResult]
    ] = []

    per_frame_rows = []

    all_naclip_raw = []

    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # sequential trajectory replay
    # -------------------------------------------------------------
    for local_i, fi in enumerate(
        frame_indices
    ):
        frame = frames[
            int(
                fi
            )
        ]

        evaluation = evaluator.evaluate(
            frame
        )

        raw_results_all.append(
            evaluation.raw
        )

        frame_maps_all.append(
            evaluation.maps
        )

        all_naclip_raw.append(
            evaluation.raw
            .naclip_raw_cosine_map
            .reshape(-1)
        )

        row = {
            "frame_index": int(
                fi
            ),

            "naclip_p85": float(
                evaluation.raw
                .naclip_percentile()
            ),
        }

        # ---------------------------------------------------------
        # Every map branch gets its own Natural Zoom state.
        # ---------------------------------------------------------
        for branch in active_branches:
            map_result = evaluation.maps[
                branch
            ]

            check_threshold = (
                engines[
                    branch
                ].update_current(
                    map_result
                )
            )

            row[
                f"{branch}_gaussian"
            ] = float(
                map_result.gaussian_score
            )

            row[
                f"{branch}_zoom_prob"
            ] = float(
                map_result.zoom_in_prob
            )

            row[
                f"{branch}_check_threshold"
            ] = float(
                check_threshold
            )

            row[
                f"{branch}_max_minus_mean"
            ] = float(
                map_result.map_max_minus_mean
            )

            row[
                f"{branch}_map_std"
            ] = float(
                map_result.map_std
            )

            candidate = (
                engines[
                    branch
                ].propose(
                    frame=frame,
                    step=int(
                        fi
                    ),
                    result=map_result,
                    check_threshold=(
                        check_threshold
                    ),
                )
            )

            accepted = 0

            if candidate.produced_crop:
                zoomed_eval = evaluator.evaluate(
                    candidate.zoomed_frame
                )

                zoomed_map = zoomed_eval.maps[
                    branch
                ]

                post = (
                    engines[
                        branch
                    ].accept(
                        current_result=(
                            map_result
                        ),
                        zoomed_result=(
                            zoomed_map
                        ),
                    )
                )

                if post[
                    "is_zoomed"
                ]:
                    accepted = 1

                    events_by_branch[
                        branch
                    ].append(
                        AcceptedZoomEvent(
                            branch=branch,

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
                                candidate
                                .zoomed_frame
                                .copy()
                            ),

                            source_eval=(
                                evaluation
                            ),

                            zoomed_eval=(
                                zoomed_eval
                            ),

                            source_map=(
                                map_result
                            ),

                            zoomed_map=(
                                zoomed_map
                            ),

                            candidate=(
                                candidate
                            ),

                            post=post,
                        )
                    )

                    all_naclip_raw.append(
                        zoomed_eval.raw
                        .naclip_raw_cosine_map
                        .reshape(-1)
                    )

            row[
                f"{branch}_accepted_zoom"
            ] = int(
                accepted
            )

        per_frame_rows.append(
            row
        )

        # ---------------------------------------------------------
        # Every-frame visual comparison
        # ---------------------------------------------------------
        if SAVE_EVERY_FRAME_MAP_COMPARISON:
            save_frame_map_comparison(
                path=(
                    frame_dir
                    / f"frame_{int(fi):06d}.png"
                ),

                frame=frame,

                frame_index=int(
                    fi
                ),

                maps=evaluation.maps,
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
                f"NACLIP-P85="
                f"{evaluation.raw.naclip_percentile():.4f}"
            )

    # -------------------------------------------------------------
    # True trajectory NACLIP-P85 score curve
    # -------------------------------------------------------------
    naclip_p85_curve = np.asarray(
        [
            raw.naclip_percentile()
            for raw in raw_results_all
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Common NACLIP raw-cosine display range
    # -------------------------------------------------------------
    all_raw = np.concatenate(
        all_naclip_raw,
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
    # First crossings, same score definition for every branch.
    # -------------------------------------------------------------
    event_rows = []

    rows_by_branch: Dict[
        str,
        List[Dict]
    ] = {
        branch: []
        for branch in active_branches
    }

    for branch in active_branches:
        if SAVE_ACCEPTED_ZOOM_PANELS:
            (
                event_root
                / branch
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        for event_index, event in enumerate(
            events_by_branch[
                branch
            ]
        ):
            score_current_at_t = float(
                event.source_eval.raw
                .naclip_percentile()
            )

            score_on_zoomed = float(
                event.zoomed_eval.raw
                .naclip_percentile()
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
                "branch": branch,

                "display_name": DISPLAY_NAME[
                    branch
                ],

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

                "score_type":
                    f"NACLIP raw P{PROGRESS_PERCENTILE:g}",

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

                "map_current_gaussian": float(
                    event.source_map
                    .gaussian_score
                ),

                "map_zoomed_gaussian": float(
                    event.zoomed_map
                    .gaussian_score
                ),

                "gaussian_required": float(
                    event.post[
                        "gaussian_required"
                    ]
                ),

                "map_current_min": float(
                    event.source_map.map_min
                ),

                "map_current_max": float(
                    event.source_map.map_max
                ),

                "map_current_std": float(
                    event.source_map.map_std
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

            event_rows.append(
                row
            )

            rows_by_branch[
                branch
            ].append(
                row
            )

            if SAVE_ACCEPTED_ZOOM_PANELS:
                if crossing_frame is None:
                    filename = (
                        f"t{event.zoom_frame:06d}"
                        "_unresolved.png"
                    )
                else:
                    filename = (
                        f"t{event.zoom_frame:06d}"
                        f"_k{crossing_frame:06d}"
                        f"_J{int(jumping_steps):04d}.png"
                    )

                save_event_panel(
                    path=(
                        event_root
                        / branch
                        / filename
                    ),

                    event=event,

                    event_row=row,

                    frames=frames,

                    naclip_results=(
                        raw_results_all
                    ),

                    raw_vmin=raw_vmin,
                    raw_vmax=raw_vmax,
                )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    summary_rows = []

    for branch in active_branches:
        summary_rows.append(
            summarize_branch(
                branch=branch,

                events=events_by_branch[
                    branch
                ],

                rows=rows_by_branch[
                    branch
                ],
            )
        )

    # -------------------------------------------------------------
    # Save concise data
    # -------------------------------------------------------------
    save_csv(
        outdir
        / "map_ablation_summary.csv",
        summary_rows,
    )

    save_csv(
        outdir
        / "map_ablation_events.csv",
        event_rows,
    )

    save_csv(
        outdir
        / "per_frame_map_stats.csv",
        per_frame_rows,
    )

    save_event_raster(
        outdir
        / "accepted_zoom_event_raster.png",

        events_by_branch,
    )

    save_map_statistics_plot(
        outdir
        / "map_statistics_along_trajectory.png",

        frame_indices,

        per_frame_rows,

        active_branches,
    )

    save_jumping_steps_plot(
        outdir
        / "jumping_steps_comparison.png",

        event_rows,

        active_branches,
    )

    # -------------------------------------------------------------
    # Late zoom contact sheets
    # -------------------------------------------------------------
    if SAVE_LATE_ZOOM_CONTACT_SHEETS:
        for branch in active_branches:
            save_late_zoom_contact_sheet(
                path=(
                    late_root
                    / f"{branch}.png"
                ),

                branch=branch,

                events=events_by_branch[
                    branch
                ],
            )

    # -------------------------------------------------------------
    # Optional no-target negative control (V2)
    # -------------------------------------------------------------
    negative_result = run_negative_control(
        evaluator=evaluator,
        outdir=outdir,
    )

    # -------------------------------------------------------------
    # Console
    # -------------------------------------------------------------
    print()
    print(
        "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 100
    )

    for row in summary_rows:
        print()
        print(
            f"[{row['display_name']}]"
        )

        print(
            f"  uses tau/T        : "
            f"{bool(row['uses_relevance_threshold'])}"
        )

        print(
            f"  accepted zoom     : "
            f"{row['accepted_zoom']}"
        )

        print(
            f"  zoom frames       : "
            f"{row['accepted_zoom_frames']}"
        )

        print(
            f"  late zoom count   : "
            f"{row['late_zoom_count']}"
        )

        print(
            f"  late zoom frames  : "
            f"{row['late_zoom_frames']}"
        )

        print(
            f"  resolve rate      : "
            f"{row['resolve_rate']}"
        )

        print(
            f"  jump median       : "
            f"{row['jumping_steps_median']}"
        )

        print(
            f"  jump mean         : "
            f"{row['jumping_steps_mean']}"
        )

    print()
    print(
        f"Elapsed: "
        f"{time.perf_counter()-t0:.2f}s"
    )

    print()
    print(
        "重点看："
    )

    print(
        "  map_ablation_summary.csv"
    )

    print(
        "  accepted_zoom_event_raster.png"
    )

    print(
        "  map_statistics_along_trajectory.png"
    )

    print(
        "  jumping_steps_comparison.png"
    )

    print(
        "  late_zoom_contact_sheets/"
    )

    print(
        "  accepted_zoom_panels/"
    )

    if SAVE_EVERY_FRAME_MAP_COMPARISON:
        print(
            "  frame_map_comparisons/  "
            "(每一步一张)"
        )

    print()
    print(
        "最关键的判断："
    )

    print(
        "  如果 current_raw_shifted 的 zoom/crop/late-zoom "
        "不比 current_sigmoid 差，"
    )

    print(
        "  就有理由删除 RELEVANCE_THRESHOLD / "
        "RELEVANCE_TEMPERATURE。"
    )

    if negative_result is not None:
        print()
        print(
            "Negative-control 重点看："
        )
        print(
            "  negative_control/negative_summary.csv"
        )
        print(
            "  negative_control/negative_false_zoom_summary.png"
        )
        print(
            "  negative_control/frame_map_comparisons/"
        )
        print(
            "  negative_control/accepted_zoom_panels/"
        )

    print()
    print(
        "V2 可视化解释："
    )
    print(
        "  Raw-Shifted ABSOLUTE 固定 [0,1]，用于看绝对相关性。"
    )
    print(
        "  Raw-Shifted CONTRAST 只是显示拉伸，绝不参与 Natural Zoom。"
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
