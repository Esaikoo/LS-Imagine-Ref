#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-way Natural Zoom comparison
================================

Compare four independent test branches on the same saved trajectory:

1) U-Net all
   map   : task-specific U-Net heatmap
   score : U-Net Gaussian score

2) Hybrid: Current map + NACLIP score
   map   : Value patch -> Temporal(L=1) -> raw cosine -> sigmoid map
   score : NACLIP K-K -> Temporal(L=1) -> raw cosine -> P95
   This reproduces the previous mixed test: keep current zoom map, change only progress score.

3) Current all: Value + Temporal(L=1) for BOTH map and score
   map   : Value patch -> Temporal(L=1) -> raw cosine -> sigmoid map
   score : the SAME Value+Temporal raw cosine -> P95

4) NACLIP all
   map   : NACLIP K-K -> Temporal(L=1) -> raw cosine -> sigmoid map
   score : the SAME NACLIP+Temporal raw cosine -> P95

Important
---------
- P95 is a scalar progress score; it never locates the crop.
- Natural Zoom location / crop / post-zoom Gaussian acceptance always use the branch's MAP source.
- Each test branch has its own adaptive-threshold buffer, Gaussian buffer and cooldown.
- For each accepted zoom at t, search the first future real frame k where:
      score_current(k) > score_on_zoomed(t)
  and record:
      jumping_steps = k - t

PyCharm: edit GLOBAL CONFIG and run directly. No argparse.
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
        f"PROJECT_ROOT looks wrong: cannot find {TASK_SPECS}"
    )

import relevance_map.relevance_variants as rv


# =====================================================================
# 1. GLOBAL CONFIG
# =====================================================================

TASK_NAME = "shear_sheep"

NPZ_PATH = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_approach_runs"
    / TASK_NAME
    / "world_63_taskseed_0_20260909_113724"
    / "trajectory.npz"
)

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
    / "zoom_branch_compare_outputs"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_PROMPTS = ["Obtain wool"]

IMAGE_H = 160
IMAGE_W = 256

# ---------------------------------------------------------------------
# Current / NACLIP sigmoid-map calibration.
# This ONLY controls the spatial map used for Natural Zoom.
# P95 progress score itself does not use these two parameters.
# ---------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016

# ---------------------------------------------------------------------
# NACLIP global spatial hyperparameters
# ---------------------------------------------------------------------

NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True

# ---------------------------------------------------------------------
# P95 progress score
# ---------------------------------------------------------------------

P95_PERCENTILE = 85

# ---------------------------------------------------------------------
# Natural Zoom settings.
# Defaults mirror the uploaded current concentration_reward.py.
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

MAX_ZOOM_FACTOR = 3

MORPH_KERNEL = (16, 10)
CROP_STRIDE = 1

SCORESTORAGE_MAX_STEPS = 1000

# ---------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------

FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

SAVE_EVENT_PANELS = True

# rejected post-check attempts can be numerous.
SAVE_REJECTED_ATTEMPTS = False

VIS_DPI = 150
PRINT_EVERY_N_FRAMES = 20

# raw cosine visualization only
RAW_COSINE_LOW_PERCENTILE = 1.0
RAW_COSINE_HIGH_PERCENTILE = 99.0


BRANCHES = [
    "unet_all",
    "hybrid_currentmap_naclipscore",
    "current_gaussian",
    "current_p95",
    "naclip_all",
]

DISPLAY_NAME = {
    "unet_all": "U-Net all: U-Net map + U-Net Gaussian score",
    "hybrid_currentmap_naclipscore": "Hybrid: Current map + NACLIP P95 score",
    "current_gaussian": "Current-Gaussian: Value+Temporal map + Gaussian score",
    "current_p95": "Current-P95: Value+Temporal map + Value+Temporal P95",
    "naclip_all": "NACLIP all: NACLIP map + NACLIP P95",
}

# Base representations produced by evaluator:
#   unet    : U-Net heatmap + U-Net Gaussian
#   current : Value + Temporal(L=1) sigmoid map + raw P95
#   naclip  : NACLIP + Temporal(L=1) sigmoid map + raw P95
#
# Test branches below choose:
#   - MAP source: controls zoom trigger / crop / post-zoom Gaussian check
#   - SCORE source + SCORE field: controls ScoreStorage-style crossing
#
# score_field:
#   "gaussian_score" -> use Gaussian-weighted map score (NO P95)
#   "progress_score" -> use the base branch raw P95
BRANCH_SPECS = {
    "unet_all": {
        "map_source": "unet",
        "score_source": "unet",
        "score_field": "gaussian_score",
        "score_type": "unet_gaussian",
    },
    "hybrid_currentmap_naclipscore": {
        "map_source": "current",
        "score_source": "naclip",
        "score_field": "progress_score",
        "score_type": "naclip_raw_p95",
    },
    "current_gaussian": {
        "map_source": "current",
        "score_source": "current",
        "score_field": "gaussian_score",
        "score_type": "current_sigmoid_gaussian",
    },
    "current_p95": {
        "map_source": "current",
        "score_source": "current",
        "score_field": "progress_score",
        "score_type": "current_raw_p95",
    },
    "naclip_all": {
        "map_source": "naclip",
        "score_source": "naclip",
        "score_field": "progress_score",
        "score_type": "naclip_raw_p95",
    },
}


# =====================================================================
# 2. Helpers
# =====================================================================

def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

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


def stable_sigmoid_np(x) -> np.ndarray:
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
    ).astype(
        np.float32
    )


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
        size=(
            IMAGE_H,
            IMAGE_W,
        ),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return (
        y.numpy()
        .astype(
            np.float32
        )
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

    return g.astype(
        np.float32
    )


CENTER_GAUSSIAN = make_center_gaussian()
CENTER_GAUSSIAN_MEAN = float(
    np.mean(
        CENTER_GAUSSIAN
    )
)


class RunningStats:
    """
    Same population-std convention as ThresholdBuffer in current code.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x: float):
        x = float(x)

        self.n += 1

        delta = x - self.mean

        self.mean += (
            delta
            / self.n
        )

        delta2 = (
            x
            - self.mean
        )

        self.M2 += (
            delta
            * delta2
        )

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
        np.asarray(
            grid
        )
        > threshold
    ).astype(
        np.uint8
    )

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
        np.max(
            areas
        )
        / binary.size
    )


# =====================================================================
# 3. MineCLIP feature extraction
# =====================================================================

@dataclass
class MineCLIPFrame:
    gh: int
    gw: int

    current_similarity: np.ndarray  # [P,N]
    naclip_similarity: np.ndarray   # [P,N]


class MineCLIPBranches:
    def __init__(
        self,
        ckpt: Path,
    ):
        # relevance_variants reads these globals.
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
            ckpt
        )

        self.tester.text_feats(
            TARGET_PROMPTS
        )

    @torch.inference_mode()
    def extract(
        self,
        frame: np.ndarray,
    ) -> MineCLIPFrame:

        x, last, gh, gw = (
            self.tester.encode_to_last(
                frame
            )
        )

        # ---------------- current branch ----------------
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

        # ---------------- NACLIP branch ----------------
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

        return MineCLIPFrame(
            gh=int(gh),
            gw=int(gw),

            current_similarity=(
                current_sim
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            ),

            naclip_similarity=(
                naclip_sim
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            ),
        )


# =====================================================================
# 4. Branch outputs
# =====================================================================

@dataclass
class BranchResult:
    branch: str

    # Natural Zoom spatial map in [0,1]
    zoom_map: np.ndarray

    # scalar used by post-zoom Gaussian acceptance
    gaussian_score: float

    # scalar used in ScoreStorage-style crossing
    progress_score: float

    zoom_in_prob: float

    # diagnostics for semantic gates
    raw_p50: float
    raw_p95: float
    raw_p95_p50: float
    largest_cc_fraction: float

    # only Current / NACLIP have raw cosine maps
    raw_cosine_map: Optional[np.ndarray]


def map_gaussian_score(
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

    nk = sigmoid_scalar(k)

    return float(
        nk
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


def result_from_similarity(
    branch: str,
    similarity: np.ndarray,
    gh: int,
    gw: int,
) -> BranchResult:
    """
    Current and NACLIP:
      same raw cosine serves two roles:

      spatial map:
          sigmoid -> 10x16 -> 160x256

      progress scalar:
          raw P95, NO sigmoid, NO tau/T
    """

    sim = np.asarray(
        similarity,
        dtype=np.float32,
    )

    # multi-prompt max, same convention as prior tests
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
            P95_PERCENTILE,
        )
    )

    rel = stable_sigmoid_np(
        (
            sim
            - RELEVANCE_THRESHOLD
        )
        / RELEVANCE_TEMPERATURE
    )

    # max over prompts for final spatial map
    rel_max = np.max(
        rel,
        axis=0,
    ).reshape(
        gh,
        gw,
    )

    zoom_map = upsample_patch_grid(
        rel_max
    )

    raw_grid = sim_max.reshape(
        gh,
        gw,
    )

    raw_cosine_map = upsample_patch_grid(
        raw_grid
    )

    return BranchResult(
        branch=branch,

        zoom_map=zoom_map,

        gaussian_score=map_gaussian_score(
            zoom_map
        ),

        progress_score=raw_p95,

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
                rel_max,
                threshold=0.5,
            )
        ),

        raw_cosine_map=raw_cosine_map,
    )


def result_from_unet(
    unet_map: np.ndarray,
) -> BranchResult:
    """
    U-Net does not have a comparable patch raw-cosine P95.

    Therefore:
      zoom map       = original U-Net heatmap
      progress score = original U-Net Gaussian score

    This keeps the U-Net branch closest to original LS-Imagine semantics.
    """

    m = np.asarray(
        unet_map,
        dtype=np.float32,
    )

    # U-Net output should already be [0,1].
    # Only clip tiny numerical overshoot for safety.
    m = np.clip(
        m,
        0.0,
        1.0,
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

    gaussian = map_gaussian_score(
        m
    )

    # semantic gates are off by default.
    # If user turns them on, this uses full-resolution U-Net map.
    cc = largest_cc_fraction(
        m,
        threshold=0.5,
    )

    return BranchResult(
        branch="unet",

        zoom_map=m,

        gaussian_score=gaussian,

        progress_score=gaussian,

        zoom_in_prob=zoom_probability(
            m
        ),

        raw_p50=p50,
        raw_p95=p95,
        raw_p95_p50=float(
            p95
            - p50
        ),

        largest_cc_fraction=cc,

        raw_cosine_map=None,
    )


# =====================================================================
# 5. Generic Natural Zoom engine
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
    Each branch gets one independent instance.
    """

    def __init__(
        self,
        branch: str,
    ):
        self.branch = branch

        self.check_buffer = RunningStats()
        self.gaussian_buffer = RunningStats()

        self.last_zoom_attempt_step = None

    def update_current(
        self,
        result: BranchResult,
    ) -> float:
        """
        Same order as current get_reward():
          add gaussian
          add zoom_in_prob
          threshold = mean + std
        """

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
        result: BranchResult,
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
                float(
                    np.max(
                        heatmap
                    )
                )
                +
                float(
                    np.min(
                        heatmap
                    )
                )
            )
            / 2.0
            +
            float(
                np.std(
                    heatmap
                )
            )
        )

        _, binary = cv2.threshold(
            heatmap,
            threshold_value,
            1,
            cv2.THRESH_BINARY,
        )

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                MORPH_KERNEL,
            )
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        contours, _ = cv2.findContours(
            binary.astype(
                np.uint8
            ),
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

        # Reproduce current-code behaviour:
        # have_center becomes True if ANY contour center is central enough.
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

            vals = heatmap[
                contour_mask == 1
            ]

            if vals.size <= 0:
                continue

            mean_val = float(
                np.mean(
                    vals
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

        heat_tensor = (
            torch.from_numpy(
                heatmap
            )
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        conv_kernel = torch.ones(
            (
                1,
                1,
                win_h,
                win_w,
            ),
            dtype=torch.float32,
        )

        conv = F.conv2d(
            heat_tensor,
            conv_kernel,
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
        ).astype(
            np.uint8
        )

        # Same current-code semantics:
        # any real crop attempt starts cooldown even if post-check rejects.
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
        current_result: BranchResult,
        zoomed_result: BranchResult,
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
# 6. Base evaluator + four composite test branches
# =====================================================================

class BaseRepresentationEvaluator:
    def __init__(self):
        if not MINECLIP_CKPT.exists():
            raise FileNotFoundError(
                MINECLIP_CKPT
            )

        self.mineclip = MineCLIPBranches(
            MINECLIP_CKPT
        )

        if not UNET_CKPT.exists():
            raise FileNotFoundError(
                "U-Net checkpoint not found: "
                + str(
                    UNET_CKPT
                )
            )

        # Reuse the already-validated U-Net wrapper.
        self.unet = rv.OriginalUNet(
            UNET_CKPT,
            self.mineclip.tester,
        )

    def evaluate_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, BranchResult]:

        mc = self.mineclip.extract(
            frame
        )

        current = result_from_similarity(
            "current",
            mc.current_similarity,
            mc.gh,
            mc.gw,
        )

        naclip = result_from_similarity(
            "naclip",
            mc.naclip_similarity,
            mc.gh,
            mc.gw,
        )

        unet_map = self.unet.generate(
            frame,
            TARGET_PROMPTS,
        )

        unet = result_from_unet(
            unet_map
        )

        return {
            "unet": unet,
            "current": current,
            "naclip": naclip,
        }


def compose_test_branches(
    base: Dict[str, BranchResult],
) -> Dict[str, BranchResult]:
    """
    Build the four test branches without recomputing features.

    A composite BranchResult takes:
      - zoom_map / Gaussian / zoom probability / semantic diagnostics
        from map_source;
      - progress_score / raw cosine visualization
        from score_source.

    This makes the score ablations explicit:
      Hybrid           = Current map + NACLIP P95
      Current-Gaussian = Current map + Current Gaussian (NO P95)
      Current-P95      = Current map + Current P95

These three branches have the SAME Natural-Zoom map. Therefore their
crop attempts / accepted zoom frames should match exactly. If they do not,
there is a state-machine bug.
    """
    out: Dict[str, BranchResult] = {}

    for branch, spec in BRANCH_SPECS.items():
        map_result = base[spec["map_source"]]
        score_result = base[spec["score_source"]]

        if spec["score_field"] == "gaussian_score":
            crossing_score = float(score_result.gaussian_score)
        elif spec["score_field"] == "progress_score":
            crossing_score = float(score_result.progress_score)
        else:
            raise ValueError(
                f"Unknown score_field={spec['score_field']} for branch={branch}"
            )

        out[branch] = BranchResult(
            branch=branch,

            # Everything used by Natural Zoom comes from MAP source.
            zoom_map=map_result.zoom_map,
            gaussian_score=float(map_result.gaussian_score),
            zoom_in_prob=float(map_result.zoom_in_prob),
            raw_p50=float(map_result.raw_p50),
            raw_p95=float(map_result.raw_p95),
            raw_p95_p50=float(map_result.raw_p95_p50),
            largest_cc_fraction=float(map_result.largest_cc_fraction),

            # Everything used by ScoreStorage-style crossing comes from SCORE source.
            # current_gaussian is the requested "Value+Temporal without P95" branch.
            progress_score=crossing_score,
            raw_cosine_map=score_result.raw_cosine_map,
        )

    return out


# =====================================================================
# 7. Visualization
# =====================================================================

def crop_box_image(
    frame: np.ndarray,
    candidate: CropCandidate,
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


def save_event_panel(
    path: Path,
    branch: str,
    event: Dict,

    source_frame: np.ndarray,
    zoomed_frame: np.ndarray,
    crossing_frame: Optional[np.ndarray],

    source_result: BranchResult,
    zoomed_result: BranchResult,
    crossing_result: Optional[BranchResult],

    candidate: CropCandidate,

    raw_vmin: float,
    raw_vmax: float,
):
    """
    Compact 2x4 panel:

    row 1:
      source+crop | zoomed | crossing | text

    row 2:
      source zoom map | zoomed zoom map |
      source raw cosine/corresponding map |
      crossing raw cosine/corresponding map

    For U-Net there is no raw cosine;
    bottom-right maps are U-Net heatmaps.
    """

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(20, 9),
        squeeze=False,
    )

    # -------------------------------------------------------------
    # top row
    # -------------------------------------------------------------
    ax[0, 0].imshow(
        crop_box_image(
            source_frame,
            candidate,
        )
    )

    ax[0, 0].set_title(
        f"{DISPLAY_NAME[branch]}\n"
        f"source t={event['zoom_frame']}\n"
        f"current score={event['score_current_at_t']:.4f}"
    )

    ax[0, 1].imshow(
        zoomed_frame
    )

    ax[0, 1].set_title(
        "accepted zoomed image\n"
        f"score_on_zoomed={event['score_on_zoomed']:.4f}"
    )

    if crossing_frame is not None:
        ax[0, 2].imshow(
            crossing_frame
        )

        ax[0, 2].set_title(
            f"first crossing k={int(event['crossing_frame'])}\n"
            f"current={event['crossing_score']:.4f} "
            f"> zoom={event['score_on_zoomed']:.4f}"
        )
    else:
        ax[0, 2].imshow(
            np.zeros_like(
                source_frame
            )
        )

        ax[0, 2].set_title(
            "no crossing before end"
        )

    ax[0, 3].axis(
        "off"
    )

    if np.isfinite(
        event[
            "jumping_steps"
        ]
    ):
        jump_text = str(
            int(
                event[
                    "jumping_steps"
                ]
            )
        )
    else:
        jump_text = "unresolved"

    score_type = BRANCH_SPECS[branch]["score_type"]
    if score_type == "unet_gaussian":
        score_name = "U-Net Gaussian"
    elif score_type == "current_sigmoid_gaussian":
        score_name = "Current sigmoid-map Gaussian"
    else:
        score_name = "raw P95"

    text = (
        f"progress score: {score_name}\n"
        f"jumping_steps: {jump_text}\n\n"

        f"zoom probability : {event['zoom_in_prob']:.4f}\n"
        f"adaptive threshold: {event['check_threshold']:.4f}\n\n"

        f"Gaussian acceptance\n"
        f"current : {event['current_gaussian']:.4f}\n"
        f"zoomed  : {event['zoomed_gaussian']:.4f}\n"
        f"required: {event['gaussian_required']:.4f}\n\n"

        f"raw zoom   : {event['raw_zoom_factor']:.2f}x\n"
        f"actual zoom: {event['actual_zoom_factor']:.2f}x\n"
        f"have_center: {event['have_center']}\n"
        f"jump: {event['jump']}"
    )

    ax[0, 3].text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        transform=ax[0, 3].transAxes,
        family="monospace",
        fontsize=10,
    )

    # -------------------------------------------------------------
    # bottom row
    # -------------------------------------------------------------
    ax[1, 0].imshow(
        source_result.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 0].set_title(
        "source map used by Natural Zoom"
    )

    ax[1, 1].imshow(
        zoomed_result.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 1].set_title(
        "zoomed map used by post-check"
    )

    if (
        source_result.raw_cosine_map
        is not None
    ):
        ax[1, 2].imshow(
            source_result.raw_cosine_map,
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )

        ax[1, 2].set_title(
            "source raw cosine correlation map"
        )
    else:
        ax[1, 2].imshow(
            source_result.zoom_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )

        ax[1, 2].set_title(
            "U-Net source heatmap\n(no raw cosine)"
        )

    if crossing_result is not None:
        if (
            crossing_result.raw_cosine_map
            is not None
        ):
            ax[1, 3].imshow(
                crossing_result.raw_cosine_map,
                cmap="viridis",
                vmin=raw_vmin,
                vmax=raw_vmax,
            )

            ax[1, 3].set_title(
                "first-crossing raw cosine map"
            )
        else:
            ax[1, 3].imshow(
                crossing_result.zoom_map,
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

            ax[1, 3].set_title(
                "U-Net first-crossing heatmap"
            )
    else:
        ax[1, 3].imshow(
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

        ax[1, 3].set_title(
            "no crossing"
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


def save_timeline_plot(
    path: Path,
    frame_indices: np.ndarray,
    current_scores: Dict[str, np.ndarray],
    accepted_events: Dict[str, List[Dict]],
):
    fig, axes = plt.subplots(
        len(BRANCHES),
        1,
        figsize=(15, 4 * len(BRANCHES)),
        sharex=True,
        squeeze=False,
    )

    for row, branch in enumerate(
        BRANCHES
    ):
        ax = axes[row, 0]

        ax.plot(
            frame_indices,
            current_scores[branch],
            linewidth=1.8,
            label="current score",
        )

        first_zoom = True

        for event in accepted_events[
            branch
        ]:
            t = int(
                event[
                    "zoom_frame"
                ]
            )

            z = float(
                event[
                    "score_on_zoomed"
                ]
            )

            ax.scatter(
                [t],
                [z],
                marker="^",
                s=65,
                label=(
                    "score_on_zoomed"
                    if first_zoom
                    else None
                ),
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

                cs = float(
                    event[
                        "crossing_score"
                    ]
                )

                ax.scatter(
                    [k],
                    [cs],
                    marker="o",
                    s=35,
                )

                ax.plot(
                    [t, k],
                    [z, cs],
                    linestyle="--",
                    alpha=0.45,
                )

                ax.text(
                    (
                        t
                        + k
                    )
                    / 2.0,
                    max(
                        z,
                        cs,
                    ),
                    f"J={int(event['jumping_steps'])}",
                    fontsize=8,
                )

            first_zoom = False

        score_type = BRANCH_SPECS[branch]["score_type"]
        if score_type == "unet_gaussian":
            score_name = "U-Net Gaussian"
        elif score_type == "current_sigmoid_gaussian":
            score_name = "Current sigmoid-map Gaussian"
        else:
            score_name = "raw P95"

        ax.set_ylabel(
            score_name
        )

        ax.set_title(
            DISPLAY_NAME[
                branch
            ]
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            fontsize=8
        )

    axes[-1, 0].set_xlabel(
        "Trajectory frame index"
    )

    fig.suptitle(
        "Four-way zoom / ScoreStorage first-crossing comparison",
        fontsize=15,
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


def save_jump_comparison_plot(
    path: Path,
    accepted_events: Dict[str, List[Dict]],
):
    fig, axes = plt.subplots(
        len(BRANCHES),
        1,
        figsize=(14, 3.2 * len(BRANCHES)),
        sharex=True,
        squeeze=False,
    )

    for row, branch in enumerate(
        BRANCHES
    ):
        ax = axes[
            row,
            0,
        ]

        resolved = [
            e
            for e in accepted_events[
                branch
            ]
            if np.isfinite(
                e[
                    "jumping_steps"
                ]
            )
        ]

        if resolved:
            x = np.asarray(
                [
                    int(
                        e[
                            "zoom_frame"
                        ]
                    )
                    for e in resolved
                ]
            )

            y = np.asarray(
                [
                    int(
                        e[
                            "jumping_steps"
                        ]
                    )
                    for e in resolved
                ]
            )

            ax.plot(
                x,
                y,
                marker="o",
            )

        ax.set_ylabel(
            "jumping_steps"
        )

        ax.set_title(
            DISPLAY_NAME[
                branch
            ]
        )

        ax.grid(
            alpha=0.25
        )

    axes[-1, 0].set_xlabel(
        "accepted zoom frame t"
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


def save_zoom_event_raster(
    path: Path,
    frame_indices: np.ndarray,
    events: Dict[str, List[Dict]],
):
    """
    Simple view: where each branch accepted zoom.
    """

    fig, ax = plt.subplots(
        figsize=(14, 4.5),
    )

    ymap = {
        "unet_all": 4,
        "hybrid_currentmap_naclipscore": 3,
        "current_gaussian": 2,
        "current_p95": 1,
        "naclip_all": 0,
    }

    for branch in BRANCHES:
        xs = [
            int(
                e[
                    "zoom_frame"
                ]
            )
            for e in events[
                branch
            ]
        ]

        ys = [
            ymap[
                branch
            ]
        ] * len(
            xs
        )

        ax.scatter(
            xs,
            ys,
            s=65,
            label=DISPLAY_NAME[
                branch
            ],
        )

    ax.set_yticks(
        [0, 1, 2, 3, 4]
    )

    ax.set_yticklabels(
        [
            "NACLIP all",
            "Current-P95",
            "Current-Gaussian",
            "Hybrid",
            "U-Net all",
        ]
    )

    ax.set_xlim(
        int(
            frame_indices[0]
        )
        - 1,
        int(
            frame_indices[-1]
        )
        + 1,
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
# 8. Main
# =====================================================================

def main():
    if len(
        TARGET_PROMPTS
    ) != 1:
        print(
            "[WARN] This comparison is cleanest with one prompt. "
            "For multi-prompt U-Net, OriginalUNet.generate returns max map "
            "while original Gaussian score historically sums prompt maps."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outdir = (
        OUTPUT_ROOT
        / f"{TASK_NAME}_{timestamp}"
    )

    panel_root = (
        outdir
        / "accepted_zoom_comparisons"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if SAVE_EVENT_PANELS:
        for branch in BRANCHES:
            (
                panel_root
                / branch
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    print(
        "=" * 96
    )

    print(
        "FIVE-WAY NATURAL ZOOM COMPARISON"
    )

    print(
        "=" * 96
    )

    print("U-Net all        : U-Net map -> zoom; U-Net Gaussian -> crossing")
    print("Hybrid           : Current Value+Temporal map -> zoom; NACLIP P95 -> crossing")
    print("Current-Gaussian : Value+Temporal map -> zoom; SAME map Gaussian -> crossing (NO P95)")
    print("Current-P95      : Value+Temporal map -> zoom; Value+Temporal P95 -> crossing")
    print("NACLIP all       : NACLIP+Temporal map -> zoom; NACLIP P95 -> crossing")

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

        "RELEVANCE_THRESHOLD": RELEVANCE_THRESHOLD,
        "RELEVANCE_TEMPERATURE": RELEVANCE_TEMPERATURE,

        "P95_PERCENTILE": P95_PERCENTILE,

        "NACLIP_GAUSSIAN_STD": NACLIP_GAUSSIAN_STD,
        "NACLIP_GAUSSIAN_WEIGHT": NACLIP_GAUSSIAN_WEIGHT,
        "NACLIP_INCLUDE_CLS": NACLIP_INCLUDE_CLS,

        "ZOOM_COOLDOWN_STEPS": ZOOM_COOLDOWN_STEPS,
        "SEMANTIC_GATE_ENABLED": SEMANTIC_GATE_ENABLED,
        "POST_ZOOM_SEMANTIC_GATE_ENABLED": POST_ZOOM_SEMANTIC_GATE_ENABLED,
        "MAX_ZOOM_FACTOR": MAX_ZOOM_FACTOR,

        "branch_definition": {
            "unet_all": {
                "map_source": "U-Net",
                "zoom_accept": "U-Net Gaussian post-check",
                "progress_score": "U-Net Gaussian",
            },
            "hybrid_currentmap_naclipscore": {
                "map_source": "Value + Temporal(L=1) sigmoid map",
                "zoom_accept": "Current-map Gaussian post-check",
                "progress_score": "NACLIP + Temporal(L=1) raw P95",
            },
            "current_gaussian": {
                "map_source": "Value + Temporal(L=1) sigmoid map",
                "zoom_accept": "Current-map Gaussian post-check",
                "progress_score": "Gaussian-weighted score of the SAME current sigmoid map (NO P95)",
            },
            "current_p95": {
                "map_source": "Value + Temporal(L=1) sigmoid map",
                "zoom_accept": "Current-map Gaussian post-check",
                "progress_score": "Value + Temporal(L=1) raw P95",
            },
            "naclip_all": {
                "map_source": "NACLIP K-K + Temporal(L=1) sigmoid map",
                "zoom_accept": "NACLIP-map Gaussian post-check",
                "progress_score": "NACLIP + Temporal(L=1) raw P95",
            },
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

    print(
        f"RGB key={rgb_key}; "
        f"total={len(frames)}; "
        f"tested={len(frame_indices)}"
    )

    # -------------------------------------------------------------
    # models
    # -------------------------------------------------------------
    evaluator = BaseRepresentationEvaluator()

    engines = {
        branch: NaturalZoomEngine(
            branch
        )
        for branch in BRANCHES
    }

    # -------------------------------------------------------------
    # per-frame caches
    # -------------------------------------------------------------
    current_results: Dict[
        str,
        List[BranchResult]
    ] = {
        b: []
        for b in BRANCHES
    }

    current_scores: Dict[
        str,
        List[float]
    ] = {
        b: []
        for b in BRANCHES
    }

    current_rows: List[Dict] = []

    # all actual crop attempts
    event_rows: List[Dict] = []

    # accepted payloads by branch
    accepted_payloads: Dict[
        str,
        List[Dict]
    ] = {
        b: []
        for b in BRANCHES
    }

    # raw cosine values only for Current / NACLIP shared vis range
    all_raw_cosine = []

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

        source_base = evaluator.evaluate_frame(
            frame
        )
        source_results = compose_test_branches(
            source_base
        )

        row = {
            "frame_index": int(
                fi
            ),
        }

        # cache true trajectory results FIRST.
        for branch in BRANCHES:
            result = source_results[
                branch
            ]

            current_results[
                branch
            ].append(
                result
            )

            current_scores[
                branch
            ].append(
                result.progress_score
            )

            row[
                f"{branch}_progress_score"
            ] = float(
                result.progress_score
            )

            row[
                f"{branch}_gaussian"
            ] = float(
                result.gaussian_score
            )

            row[
                f"{branch}_zoom_prob"
            ] = float(
                result.zoom_in_prob
            )

            if (
                result.raw_cosine_map
                is not None
            ):
                all_raw_cosine.append(
                    result.raw_cosine_map
                    .reshape(-1)
                )

        current_rows.append(
            row
        )

        # ---------------------------------------------------------
        # process each branch independently
        # ---------------------------------------------------------
        for branch in BRANCHES:
            result = source_results[
                branch
            ]

            engine = engines[
                branch
            ]

            check_threshold = (
                engine.update_current(
                    result
                )
            )

            candidate = engine.propose(
                frame=frame,
                step=int(
                    fi
                ),
                result=result,
                check_threshold=check_threshold,
            )

            # no real crop
            if not candidate.produced_crop:
                continue

            # Evaluate the ACTUAL cropped pixels under all models once,
            # then pick same branch's result for post-check / score.
            zoomed_base = evaluator.evaluate_frame(
                candidate.zoomed_frame
            )
            zoomed_all = compose_test_branches(
                zoomed_base
            )

            zoomed_result = zoomed_all[branch]

            if (
                zoomed_result.raw_cosine_map
                is not None
            ):
                all_raw_cosine.append(
                    zoomed_result.raw_cosine_map
                    .reshape(-1)
                )

            post = engine.accept(
                current_result=result,
                zoomed_result=zoomed_result,
                candidate=candidate,
            )

            event = {
                "branch": branch,
                "display_name": DISPLAY_NAME[
                    branch
                ],

                "zoom_frame": int(
                    fi
                ),

                "produced_crop": 1,

                "is_zoomed": int(
                    post[
                        "is_zoomed"
                    ]
                ),

                "jump": int(
                    post[
                        "jump"
                    ]
                ),

                "map_source": BRANCH_SPECS[branch]["map_source"],
                "score_source": BRANCH_SPECS[branch]["score_source"],
                "score_type": BRANCH_SPECS[branch]["score_type"],

                "score_current_at_t": float(
                    result.progress_score
                ),

                "score_on_zoomed": float(
                    zoomed_result.progress_score
                ),

                "crossing_frame": np.nan,
                "crossing_score": np.nan,
                "jumping_steps": np.nan,

                "current_gaussian": float(
                    result.gaussian_score
                ),

                "zoomed_gaussian": float(
                    zoomed_result.gaussian_score
                ),

                "gaussian_std": float(
                    post[
                        "gaussian_std"
                    ]
                ),

                "gaussian_required": float(
                    post[
                        "gaussian_required"
                    ]
                ),

                "gaussian_gain_ok": int(
                    post[
                        "gaussian_gain_ok"
                    ]
                ),

                "zoom_in_prob": float(
                    result.zoom_in_prob
                ),

                "check_threshold": float(
                    check_threshold
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
            }

            event_rows.append(
                event
            )

            if post[
                "is_zoomed"
            ]:
                accepted_payloads[
                    branch
                ].append(
                    {
                        "event": event,

                        "source_local_index": int(
                            local_i
                        ),

                        "source_frame": frame.copy(),

                        "zoomed_frame": (
                            candidate.zoomed_frame
                            .copy()
                        ),

                        "source_result": result,

                        "zoomed_result": zoomed_result,

                        "candidate": candidate,
                    }
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
                f"U={source_results['unet_all'].progress_score:.3f} | "
                f"H-NP95={source_results['hybrid_currentmap_naclipscore'].progress_score:.4f} | "
                f"C-G={source_results['current_gaussian'].progress_score:.3f} | "
                f"C-P95={source_results['current_p95'].progress_score:.4f} | "
                f"N-P95={source_results['naclip_all'].progress_score:.4f}"
            )

    # numpy score arrays
    current_score_np = {
        branch: np.asarray(
            current_scores[
                branch
            ],
            dtype=np.float64,
        )
        for branch in BRANCHES
    }

    # -------------------------------------------------------------
    # first-crossing search, separately per branch
    # -------------------------------------------------------------
    accepted_events: Dict[
        str,
        List[Dict]
    ] = {
        b: []
        for b in BRANCHES
    }

    for branch in BRANCHES:
        for payload in accepted_payloads[
            branch
        ]:
            event = payload[
                "event"
            ]

            local_t = int(
                payload[
                    "source_local_index"
                ]
            )

            t = int(
                event[
                    "zoom_frame"
                ]
            )

            threshold_score = float(
                event[
                    "score_on_zoomed"
                ]
            )

            crossing_local = None

            for j in range(
                local_t + 1,
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
                    current_score_np[
                        branch
                    ][j]
                    > threshold_score
                ):
                    crossing_local = j
                    break

            if crossing_local is not None:
                k = int(
                    frame_indices[
                        crossing_local
                    ]
                )

                event[
                    "crossing_frame"
                ] = k

                event[
                    "crossing_score"
                ] = float(
                    current_score_np[
                        branch
                    ][
                        crossing_local
                    ]
                )

                event[
                    "jumping_steps"
                ] = int(
                    k - t
                )

                payload[
                    "crossing_frame_rgb"
                ] = frames[
                    k
                ].copy()

                payload[
                    "crossing_result"
                ] = current_results[
                    branch
                ][
                    crossing_local
                ]
            else:
                payload[
                    "crossing_frame_rgb"
                ] = None

                payload[
                    "crossing_result"
                ] = None

            accepted_events[
                branch
            ].append(
                event
            )

    # -------------------------------------------------------------
    # raw cosine vis range
    # -------------------------------------------------------------
    if all_raw_cosine:
        all_raw = np.concatenate(
            all_raw_cosine,
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
    else:
        raw_vmin = 0.0
        raw_vmax = 1.0

    # -------------------------------------------------------------
    # event panels
    # -------------------------------------------------------------
    if SAVE_EVENT_PANELS:
        for branch in BRANCHES:
            branch_dir = (
                panel_root
                / branch
            )

            for payload in accepted_payloads[
                branch
            ]:
                e = payload[
                    "event"
                ]

                t = int(
                    e[
                        "zoom_frame"
                    ]
                )

                if np.isfinite(
                    e[
                        "crossing_frame"
                    ]
                ):
                    k = int(
                        e[
                            "crossing_frame"
                        ]
                    )

                    j = int(
                        e[
                            "jumping_steps"
                        ]
                    )

                    filename = (
                        f"t{t:06d}"
                        f"_k{k:06d}"
                        f"_jump{j:04d}.png"
                    )
                else:
                    filename = (
                        f"t{t:06d}"
                        "_unresolved.png"
                    )

                save_event_panel(
                    path=(
                        branch_dir
                        / filename
                    ),

                    branch=branch,

                    event=e,

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

                    source_result=(
                        payload[
                            "source_result"
                        ]
                    ),

                    zoomed_result=(
                        payload[
                            "zoomed_result"
                        ]
                    ),

                    crossing_result=(
                        payload.get(
                            "crossing_result"
                        )
                    ),

                    candidate=(
                        payload[
                            "candidate"
                        ]
                    ),

                    raw_vmin=raw_vmin,
                    raw_vmax=raw_vmax,
                )

    # -------------------------------------------------------------
    # save concise data
    # -------------------------------------------------------------
    save_csv(
        outdir
        / "current_scores.csv",
        current_rows,
    )

    save_csv(
        outdir
        / "zoom_events.csv",
        event_rows,
    )

    save_timeline_plot(
        outdir
        / "zoom_score_timeline.png",
        frame_indices,
        current_score_np,
        accepted_events,
    )

    save_jump_comparison_plot(
        outdir
        / "jumping_steps_comparison.png",
        accepted_events,
    )

    save_zoom_event_raster(
        outdir
        / "accepted_zoom_event_raster.png",
        frame_indices,
        accepted_events,
    )

    # -------------------------------------------------------------
    # summary
    # -------------------------------------------------------------
    summary_rows = []

    for branch in BRANCHES:
        events = accepted_events[
            branch
        ]

        jumps = [
            int(
                e[
                    "jumping_steps"
                ]
            )
            for e in events
            if np.isfinite(
                e[
                    "jumping_steps"
                ]
            )
        ]

        attempt_count = sum(
            1
            for e in event_rows
            if e[
                "branch"
            ] == branch
        )

        resolved = len(
            jumps
        )

        summary_rows.append(
            {
                "branch": branch,

                "display_name": DISPLAY_NAME[
                    branch
                ],

                "map_source": BRANCH_SPECS[branch]["map_source"],
                "score_source": BRANCH_SPECS[branch]["score_source"],
                "score_type": BRANCH_SPECS[branch]["score_type"],

                "crop_attempts": int(
                    attempt_count
                ),

                "accepted_zoom": int(
                    len(
                        events
                    )
                ),

                "resolved_crossing": int(
                    resolved
                ),

                "unresolved_crossing": int(
                    len(
                        events
                    )
                    - resolved
                ),

                "resolve_rate": (
                    float(
                        resolved
                        / len(
                            events
                        )
                    )
                    if events
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
        )

    save_csv(
        outdir
        / "branch_summary.csv",
        summary_rows,
    )

    # -------------------------------------------------------------
    # console
    # -------------------------------------------------------------
    print()
    print(
        "=" * 96
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 96
    )

    for row in summary_rows:
        print()
        print(
            f"[{row['display_name']}]"
        )

        print(
            f"  score type       : {row['score_type']}"
        )

        print(
            f"  crop attempts    : {row['crop_attempts']}"
        )

        print(
            f"  accepted zoom    : {row['accepted_zoom']}"
        )

        print(
            f"  resolved crossing: {row['resolved_crossing']}"
        )

        print(
            f"  resolve rate     : {row['resolve_rate']}"
        )

        print(
            f"  jumping mean     : {row['jumping_steps_mean']}"
        )

        print(
            f"  jumping median   : {row['jumping_steps_median']}"
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
        "  branch_summary.csv"
    )

    print(
        "  zoom_events.csv"
    )

    print(
        "  zoom_score_timeline.png"
    )

    print(
        "  jumping_steps_comparison.png"
    )

    print(
        "  accepted_zoom_event_raster.png"
    )

    print(
        "  accepted_zoom_comparisons/"
    )

    print()
    print(
        "Output:",
        outdir,
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
