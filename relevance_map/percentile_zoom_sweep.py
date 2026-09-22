#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Percentile sweep for Natural-Zoom / ScoreStorage crossing
=========================================================

Goal
----
Test whether P80/P85/P90/P95 (or any configured percentile) gives a more
reasonable first-crossing state than P95, while keeping one clean baseline:

Baseline
--------
Current-Gaussian:
    Value patch
      -> Temporal(L=1)
      -> raw cosine
      -> sigmoid relevance map
      -> Natural Zoom
      -> Gaussian-weighted map score

Percentile branch
-----------------
Default map source = NACLIP:
    NACLIP K-K
      -> Temporal(L=1)
      -> raw cosine
      -> sigmoid relevance map
      -> Natural Zoom

For EVERY accepted zoom from that SAME NACLIP map:
    the zoomed image is fixed
    the crop is fixed
    the accepted zoom event is fixed

Then only the scalar readout changes:
    P80 / P85 / P90 / P95

For each percentile q:
    score_on_zoomed_q(t)
    first k > t where score_current_q(k) > score_on_zoomed_q(t)
    jumping_steps_q = k - t

This is a controlled percentile experiment:
changing PERCENTILES does NOT change the Natural-Zoom crop.

Optional
--------
Set PERCENTILE_MAP_SOURCE = "current" if you want:
    Current map + Current raw percentile
instead of:
    NACLIP map + NACLIP raw percentile

PyCharm usage
-------------
No argparse. Edit GLOBAL CONFIG and run directly.

Suggested location:
    <LS-Imagine>/relevance_map/percentile_zoom_sweep_test.py
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

# If the file is not inside <project>/relevance_map/, set manually:
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

TASK_SPECS = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS.exists():
    raise FileNotFoundError(f"PROJECT_ROOT wrong: {TASK_SPECS}")

# Reuse the already-tested MineCLIP / NACLIP implementation.
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
    / "world_63_taskseed_0_20260909_113238"
    / "trajectory.npz"
)

MINECLIP_CKPT = PROJECT_ROOT / "weights" / "mineclip_attn.pth"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "percentile_zoom_sweep_outputs"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_H = 160
IMAGE_W = 256

TARGET_PROMPTS = ["Shear sheep"]


# ---------------------------------------------------------------------
# 1.1 Percentiles to test
# ---------------------------------------------------------------------

# Main experimental variable.
PERCENTILES = [60, 65, 70, 75, 80, 85, 90, 95]

# Which representation generates the percentile branch's Natural-Zoom map
# and raw-cosine percentile scores.
#
# "naclip":
#     NACLIP K-K -> Temporal(L=1)
#
# "current":
#     Value -> Temporal(L=1)
#
# Default is NACLIP because this test is intended to study why
# NACLIP P95 crosses too early and whether P85/P90 is more reasonable.
PERCENTILE_MAP_SOURCE = "naclip"

if PERCENTILE_MAP_SOURCE not in {"current", "naclip"}:
    raise ValueError(
        "PERCENTILE_MAP_SOURCE must be 'current' or 'naclip'"
    )


# ---------------------------------------------------------------------
# 1.2 Spatial map calibration
#
# IMPORTANT:
# These parameters affect Natural Zoom map/crop/acceptance,
# but DO NOT directly affect raw percentile scores.
# ---------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016


# ---------------------------------------------------------------------
# 1.3 NACLIP parameters
# ---------------------------------------------------------------------

NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True


# ---------------------------------------------------------------------
# 1.4 Natural Zoom settings
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
# 1.5 Trajectory
# ---------------------------------------------------------------------

FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1


# ---------------------------------------------------------------------
# 1.6 Output / visualization
# ---------------------------------------------------------------------

SAVE_PERCENTILE_EVENT_PANELS = True
SAVE_BASELINE_EVENT_PANELS = True

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
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


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
            xx ** 2 / (2.0 * sigma_x ** 2)
            + yy ** 2 / (2.0 * sigma_y ** 2)
        )
    )

    return g.astype(np.float32)


CENTER_GAUSSIAN = make_center_gaussian()
CENTER_GAUSSIAN_MEAN = float(np.mean(CENTER_GAUSSIAN))


class RunningStats:
    """
    Same population-std convention as current ThresholdBuffer:
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
                self.M2 / self.n
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
        np.asarray(grid) > threshold
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


# =====================================================================
# 3. MineCLIP extraction
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

        # relevance_variants reads these module globals.
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

        # ---------------- current ----------------
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
# 4. Representation result
# =====================================================================

@dataclass
class RepresentationResult:
    """
    One representation (current or NACLIP).

    raw_patch_scores:
        multi-prompt max raw cosine, shape [N]

    zoom_map:
        sigmoid map, shape [H,W], used by Natural Zoom

    gaussian_score:
        Gaussian-weighted score of zoom_map
    """

    source_name: str

    raw_patch_scores: np.ndarray

    raw_cosine_map: np.ndarray
    zoom_map: np.ndarray

    gaussian_score: float
    zoom_in_prob: float

    raw_p50: float
    raw_p95: float
    raw_p95_p50: float

    largest_cc_fraction: float

    def percentile_score(
        self,
        q: float,
    ) -> float:
        return float(
            np.percentile(
                self.raw_patch_scores,
                q,
            )
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

    nk = sigmoid_scalar(k)

    return float(
        nk
        * (
            float(np.max(map01))
            - float(np.mean(map01))
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

    # multi-prompt max
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

    rel = stable_sigmoid_np(
        (
            sim
            - RELEVANCE_THRESHOLD
        )
        / RELEVANCE_TEMPERATURE
    )

    rel_max = np.max(
        rel,
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
        rel_max
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
                rel_max,
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

        bundle = self.extractor.extract(
            frame
        )

        current = representation_from_similarity(
            "current",
            bundle.current_similarity,
            bundle.gh,
            bundle.gw,
        )

        naclip = representation_from_similarity(
            "naclip",
            bundle.naclip_similarity,
            bundle.gh,
            bundle.gw,
        )

        return FrameResults(
            current=current,
            naclip=naclip,
        )


# =====================================================================
# 5. Natural Zoom engine
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
    Generic Natural Zoom state machine.

    We instantiate TWO engines:
        baseline Current-Gaussian engine
        percentile-branch engine

    Percentiles themselves DO NOT get separate engines because
    all percentiles must share exactly the same zoom/crop.
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
            and self.last_zoom_attempt_step is not None
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

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            MORPH_KERNEL,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
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

            vals = heatmap[
                contour_mask == 1
            ]

            if vals.size == 0:
                continue

            mean_val = float(
                np.mean(vals)
            )

            M = cv2.moments(contour)

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

            if mean_val > max_mean_value:
                max_mean_value = mean_val

                max_area_ratio = float(
                    np.sum(contour_mask)
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
            / math.sqrt(proportion)
        )

        if MAX_ZOOM_FACTOR is not None:
            min_proportion = (
                1.0
                / float(
                    MAX_ZOOM_FACTOR ** 2
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
                sqrt_prop * W
            )
        )

        win_h = int(
            math.ceil(
                sqrt_prop * H
            )
        )

        win_w = max(
            1,
            min(win_w, W),
        )

        win_h = max(
            1,
            min(win_h, H),
        )

        actual_zoom_factor = max(
            W / float(win_w),
            H / float(win_h),
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
            int(best_idx.item()),
            int(conv.shape[-1]),
        )

        x1 = int(
            best_x * CROP_STRIDE
        )

        y1 = int(
            best_y * CROP_STRIDE
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
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        # Same current-code semantics:
        # actual crop attempt starts cooldown even if post-check rejects.
        self.last_zoom_attempt_step = int(step)

        return CropCandidate(
            produced_crop=True,
            reason="natural_zoom_candidate",
            zoomed_frame=zoomed,

            have_center=bool(have_center),

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
                if zoomed_result.largest_cc_fraction > 0
                else 0.0
            )

        semantic_gain_ok = (
            p95_gain >= POST_ZOOM_MIN_P95_GAIN
            and contrast_gain >= POST_ZOOM_MIN_CONTRAST_GAIN
            and cc_retention >= POST_ZOOM_MIN_CC_RETENTION
        )

        if POST_ZOOM_SEMANTIC_GATE_ENABLED:
            is_zoomed = (
                gaussian_gain_ok
                and semantic_gain_ok
            )
        else:
            is_zoomed = gaussian_gain_ok

        jump = (
            bool(is_zoomed)
            and bool(candidate.have_center)
        )

        return {
            "gaussian_std": float(
                gaussian_std
            ),

            "gaussian_required": float(
                gaussian_required
            ),

            "gaussian_gain_ok": int(
                bool(gaussian_gain_ok)
            ),

            "semantic_gain_ok": int(
                bool(semantic_gain_ok)
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
                bool(is_zoomed)
            ),

            "jump": int(
                bool(jump)
            ),
        }


# =====================================================================
# 6. Event data
# =====================================================================

@dataclass
class AcceptedEvent:
    branch: str
    zoom_frame: int
    source_local_index: int

    source_frame: np.ndarray
    zoomed_frame: np.ndarray

    source_result: RepresentationResult
    zoomed_result: RepresentationResult

    candidate: CropCandidate

    post: Dict


# =====================================================================
# 7. First-crossing
# =====================================================================

def first_crossing(
    frame_indices: np.ndarray,
    score_curve: np.ndarray,
    local_t: int,
    score_on_zoomed: float,
) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Returns:
        crossing_local_idx
        crossing_frame
        crossing_score
    """

    t = int(
        frame_indices[
            local_t
        ]
    )

    for j in range(
        local_t + 1,
        len(frame_indices),
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
            float(score_curve[j])
            > float(score_on_zoomed)
        ):
            return (
                int(j),
                k,
                float(score_curve[j]),
            )

    return None, None, None


# =====================================================================
# 8. Visualization
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


def save_percentile_event_panel(
    path: Path,

    event: AcceptedEvent,

    percentile_rows: List[Dict],

    frames: np.ndarray,
    current_results_for_score_source: List[RepresentationResult],

    raw_vmin: float,
    raw_vmax: float,
):
    """
    One panel for ONE accepted percentile-map zoom.

    Row 1:
        source+crop | zoomed | P80 crossing | P85 crossing | P90 crossing | P95 crossing
        (number of columns adapts to PERCENTILES)

    Row 2:
        source raw cosine | zoomed raw cosine |
        crossing raw cosine for each percentile

    This lets you visually decide which percentile's first crossing
    best matches the zoomed image.
    """

    n_q = len(
        percentile_rows
    )

    cols = 2 + n_q

    fig, ax = plt.subplots(
        2,
        cols,
        figsize=(
            max(
                16,
                3.7 * cols,
            ),
            8.5,
        ),
        squeeze=False,
    )

    # -------------------------------------------------------------
    # source
    # -------------------------------------------------------------
    ax[0, 0].imshow(
        crop_box_image(
            event.source_frame,
            event.candidate,
        )
    )

    ax[0, 0].set_title(
        f"source t={event.zoom_frame}\n"
        f"zoom={event.candidate.actual_zoom_factor:.2f}x"
    )

    ax[1, 0].imshow(
        event.source_result.raw_cosine_map,
        cmap="viridis",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    ax[1, 0].set_title(
        f"source raw cosine\n"
        f"{event.source_result.source_name}"
    )

    # -------------------------------------------------------------
    # zoomed
    # -------------------------------------------------------------
    ax[0, 1].imshow(
        event.zoomed_frame
    )

    ax[0, 1].set_title(
        "accepted zoomed image"
    )

    ax[1, 1].imshow(
        event.zoomed_result.raw_cosine_map,
        cmap="viridis",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    ax[1, 1].set_title(
        "zoomed raw cosine"
    )

    # -------------------------------------------------------------
    # each percentile
    # -------------------------------------------------------------
    for qi, row in enumerate(
        percentile_rows
    ):
        c = 2 + qi

        q = float(
            row["percentile"]
        )

        crossing_frame = row[
            "crossing_frame"
        ]

        if np.isfinite(
            crossing_frame
        ):
            k = int(
                crossing_frame
            )

            crossing_rgb = frames[
                k
            ]

            crossing_result = (
                current_results_for_score_source[
                    int(
                        row[
                            "crossing_local_index"
                        ]
                    )
                ]
            )

            ax[0, c].imshow(
                crossing_rgb
            )

            ax[0, c].set_title(
                f"P{q:g} crossing k={k}\n"
                f"J={int(row['jumping_steps'])}"
            )

            ax[1, c].imshow(
                crossing_result.raw_cosine_map,
                cmap="viridis",
                vmin=raw_vmin,
                vmax=raw_vmax,
            )

            ax[1, c].set_title(
                f"P{q:g} raw cosine\n"
                f"current={row['crossing_score']:.4f}\n"
                f"zoom={row['score_on_zoomed']:.4f}"
            )
        else:
            ax[0, c].imshow(
                np.zeros_like(
                    event.source_frame
                )
            )

            ax[0, c].set_title(
                f"P{q:g}: unresolved"
            )

            ax[1, c].imshow(
                np.zeros_like(
                    event.source_result.raw_cosine_map
                ),
                cmap="viridis",
                vmin=raw_vmin,
                vmax=raw_vmax,
            )

            ax[1, c].set_title(
                f"P{q:g}: unresolved"
            )

    for a in ax.reshape(-1):
        a.axis(
            "off"
        )

    fig.suptitle(
        f"Percentile first-crossing sweep | "
        f"map source={PERCENTILE_MAP_SOURCE} | "
        f"accepted zoom t={event.zoom_frame}",
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


def save_baseline_event_panel(
    path: Path,

    event: AcceptedEvent,

    crossing_frame: Optional[int],
    crossing_score: Optional[float],
    jumping_steps: Optional[int],

    frames: np.ndarray,
    baseline_current_results: List[RepresentationResult],
):
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        squeeze=False,
    )

    ax[0, 0].imshow(
        crop_box_image(
            event.source_frame,
            event.candidate,
        )
    )

    ax[0, 0].set_title(
        f"Current-Gaussian source t={event.zoom_frame}\n"
        f"current G={event.source_result.gaussian_score:.3f}"
    )

    ax[0, 1].imshow(
        event.zoomed_frame
    )

    ax[0, 1].set_title(
        f"zoomed\n"
        f"score_on_zoomed={event.zoomed_result.gaussian_score:.3f}"
    )

    if crossing_frame is not None:
        ax[0, 2].imshow(
            frames[
                crossing_frame
            ]
        )

        ax[0, 2].set_title(
            f"first crossing k={crossing_frame}\n"
            f"J={jumping_steps}"
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

    ax[1, 0].imshow(
        event.source_result.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 0].set_title(
        "source current sigmoid map"
    )

    ax[1, 1].imshow(
        event.zoomed_result.zoom_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax[1, 1].set_title(
        "zoomed current sigmoid map"
    )

    if crossing_frame is not None:
        # map frame index -> local index
        local_idx = None

        # baseline_current_results aligns with frame_indices, and
        # this script uses stride=1 by default. We avoid assuming
        # frame==local if user changes FRAME_START/STRIDE by finding it later.
        ax[1, 2].axis("off")
        ax[1, 2].text(
            0.02,
            0.95,
            f"Gaussian crossing\n"
            f"crossing_score={crossing_score:.4f}\n"
            f"zoomed_score={event.zoomed_result.gaussian_score:.4f}\n"
            f"jumping_steps={jumping_steps}",
            va="top",
            transform=ax[1, 2].transAxes,
            family="monospace",
        )
    else:
        ax[1, 2].axis("off")
        ax[1, 2].text(
            0.02,
            0.95,
            "No future current Gaussian\n"
            "exceeded score_on_zoomed.",
            va="top",
            transform=ax[1, 2].transAxes,
        )

    for a in ax[0]:
        a.axis(
            "off"
        )

    ax[1, 0].axis("off")
    ax[1, 1].axis("off")

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=VIS_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_jumping_steps_by_percentile(
    path: Path,
    percentile_event_rows: List[Dict],
):
    fig, ax = plt.subplots(
        figsize=(13, 6.5),
    )

    for q in PERCENTILES:
        rows = [
            r
            for r in percentile_event_rows
            if float(
                r["percentile"]
            ) == float(q)
            and np.isfinite(
                r["jumping_steps"]
            )
        ]

        if not rows:
            continue

        x = np.asarray(
            [
                int(
                    r["zoom_frame"]
                )
                for r in rows
            ]
        )

        y = np.asarray(
            [
                int(
                    r["jumping_steps"]
                )
                for r in rows
            ]
        )

        ax.plot(
            x,
            y,
            marker="o",
            label=f"P{q:g}",
        )

    ax.set_xlabel(
        "accepted zoom frame t"
    )

    ax.set_ylabel(
        "jumping_steps"
    )

    ax.set_title(
        "Percentile sweep: implied jumping steps\n"
        f"Natural-Zoom map source = {PERCENTILE_MAP_SOURCE}"
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

    plt.close(
        fig
    )


def save_median_jump_curve(
    path: Path,
    percentile_summary: List[Dict],
    baseline_summary: Dict,
):
    qs = []
    medians = []
    means = []

    for row in percentile_summary:
        qs.append(
            float(
                row["percentile"]
            )
        )

        medians.append(
            float(
                row["jumping_steps_median"]
            )
        )

        means.append(
            float(
                row["jumping_steps_mean"]
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 5.5),
    )

    ax.plot(
        qs,
        medians,
        marker="o",
        label="median jumping_steps",
    )

    ax.plot(
        qs,
        means,
        marker="s",
        linestyle="--",
        label="mean jumping_steps",
    )

    if np.isfinite(
        baseline_summary.get(
            "jumping_steps_median",
            np.nan,
        )
    ):
        ax.axhline(
            baseline_summary[
                "jumping_steps_median"
            ],
            linestyle=":",
            label=(
                "Current-Gaussian baseline median"
            ),
        )

    ax.set_xlabel(
        "percentile q"
    )

    ax.set_ylabel(
        "jumping_steps"
    )

    ax.set_title(
        "How percentile changes first-crossing horizon"
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

    plt.close(
        fig
    )


def save_crossing_timeline(
    path: Path,

    frame_indices: np.ndarray,
    score_curves: Dict[float, np.ndarray],

    percentile_event_rows: List[Dict],
):
    n = len(
        PERCENTILES
    )

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(
            15,
            max(
                4.5 * n,
                8,
            ),
        ),
        sharex=True,
        squeeze=False,
    )

    for row_i, q in enumerate(
        PERCENTILES
    ):
        ax = axes[
            row_i,
            0,
        ]

        curve = score_curves[
            float(q)
        ]

        ax.plot(
            frame_indices,
            curve,
            linewidth=1.7,
            label=f"current P{q:g}",
        )

        rows = [
            r
            for r in percentile_event_rows
            if float(
                r["percentile"]
            ) == float(q)
        ]

        first_zoom = True

        for r in rows:
            t = int(
                r["zoom_frame"]
            )

            z = float(
                r["score_on_zoomed"]
            )

            ax.scatter(
                [t],
                [z],
                marker="^",
                s=60,
                label=(
                    "score_on_zoomed"
                    if first_zoom
                    else None
                ),
            )

            if np.isfinite(
                r["crossing_frame"]
            ):
                k = int(
                    r["crossing_frame"]
                )

                cs = float(
                    r["crossing_score"]
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
                    alpha=0.40,
                )

                ax.text(
                    (t + k) / 2.0,
                    max(z, cs),
                    f"J={int(r['jumping_steps'])}",
                    fontsize=8,
                )

            first_zoom = False

        ax.set_ylabel(
            f"P{q:g}"
        )

        ax.set_title(
            f"P{q:g} first crossings"
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
        f"Percentile crossing timeline | map={PERCENTILE_MAP_SOURCE}",
        fontsize=15,
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


# =====================================================================
# 9. Main
# =====================================================================

def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outdir = (
        OUTPUT_ROOT
        / f"{TASK_NAME}_{timestamp}"
    )

    percentile_panel_dir = (
        outdir
        / "percentile_event_panels"
    )

    baseline_panel_dir = (
        outdir
        / "baseline_current_gaussian_panels"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if SAVE_PERCENTILE_EVENT_PANELS:
        percentile_panel_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if SAVE_BASELINE_EVENT_PANELS:
        baseline_panel_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "=" * 96
    )

    print(
        "PERCENTILE NATURAL-ZOOM FIRST-CROSSING SWEEP"
    )

    print(
        "=" * 96
    )

    print(
        "Baseline : Current Value+Temporal map + Current Gaussian score"
    )

    print(
        f"Sweep    : {PERCENTILE_MAP_SOURCE} map + "
        f"{PERCENTILE_MAP_SOURCE} raw percentile score"
    )

    print(
        "PERCENTILES:",
        PERCENTILES,
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
        "NPZ_PATH": str(NPZ_PATH),
        "TARGET_PROMPTS": TARGET_PROMPTS,

        "PERCENTILES": PERCENTILES,
        "PERCENTILE_MAP_SOURCE": PERCENTILE_MAP_SOURCE,

        "baseline": (
            "Value + Temporal(L=1) sigmoid map "
            "+ Gaussian crossing score"
        ),

        "percentile_branch": (
            f"{PERCENTILE_MAP_SOURCE} sigmoid map "
            f"+ same-branch raw percentile score"
        ),

        "RELEVANCE_THRESHOLD": RELEVANCE_THRESHOLD,
        "RELEVANCE_TEMPERATURE": RELEVANCE_TEMPERATURE,

        "NACLIP_GAUSSIAN_STD": NACLIP_GAUSSIAN_STD,
        "NACLIP_GAUSSIAN_WEIGHT": NACLIP_GAUSSIAN_WEIGHT,
        "NACLIP_INCLUDE_CLS": NACLIP_INCLUDE_CLS,

        "ZOOM_COOLDOWN_STEPS": ZOOM_COOLDOWN_STEPS,
        "MAX_ZOOM_FACTOR": MAX_ZOOM_FACTOR,

        "SEMANTIC_GATE_ENABLED": SEMANTIC_GATE_ENABLED,
        "POST_ZOOM_SEMANTIC_GATE_ENABLED": (
            POST_ZOOM_SEMANTIC_GATE_ENABLED
        ),
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
        len(frames)
        if FRAME_END is None
        else min(
            len(frames),
            int(FRAME_END),
        )
    )

    frame_indices = np.arange(
        max(
            0,
            int(FRAME_START),
        ),
        end,
        max(
            1,
            int(FRAME_STRIDE),
        ),
        dtype=np.int64,
    )

    if len(frame_indices) == 0:
        raise RuntimeError(
            "No selected frames."
        )

    print(
        f"RGB key={rgb_key}; "
        f"total={len(frames)}; "
        f"tested={len(frame_indices)}"
    )

    evaluator = Evaluator()

    # Independent zoom state machines.
    baseline_engine = NaturalZoomEngine()
    percentile_engine = NaturalZoomEngine()

    # -------------------------------------------------------------
    # caches
    # -------------------------------------------------------------
    current_results: List[RepresentationResult] = []
    naclip_results: List[RepresentationResult] = []

    baseline_events: List[AcceptedEvent] = []
    percentile_events: List[AcceptedEvent] = []

    all_raw_maps: List[np.ndarray] = []

    per_frame_rows = []

    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # sequential replay
    # -------------------------------------------------------------
    for local_i, fi in enumerate(
        frame_indices
    ):
        frame = frames[
            int(fi)
        ]

        results = evaluator.evaluate(
            frame
        )

        current_results.append(
            results.current
        )

        naclip_results.append(
            results.naclip
        )

        all_raw_maps.append(
            results.current.raw_cosine_map.reshape(-1)
        )

        all_raw_maps.append(
            results.naclip.raw_cosine_map.reshape(-1)
        )

        # =========================================================
        # A) baseline Current-Gaussian
        # =========================================================
        baseline_check_threshold = (
            baseline_engine.update_current(
                results.current
            )
        )

        baseline_candidate = (
            baseline_engine.propose(
                frame=frame,
                step=int(fi),
                result=results.current,
                check_threshold=(
                    baseline_check_threshold
                ),
            )
        )

        baseline_is_zoomed = 0

        if baseline_candidate.produced_crop:
            zoomed_results = evaluator.evaluate(
                baseline_candidate.zoomed_frame
            )

            baseline_post = (
                baseline_engine.accept(
                    current_result=results.current,
                    zoomed_result=zoomed_results.current,
                    candidate=baseline_candidate,
                )
            )

            baseline_is_zoomed = int(
                baseline_post[
                    "is_zoomed"
                ]
            )

            if baseline_is_zoomed:
                baseline_events.append(
                    AcceptedEvent(
                        branch="current_gaussian",
                        zoom_frame=int(fi),
                        source_local_index=int(local_i),

                        source_frame=frame.copy(),
                        zoomed_frame=(
                            baseline_candidate
                            .zoomed_frame
                            .copy()
                        ),

                        source_result=results.current,
                        zoomed_result=(
                            zoomed_results.current
                        ),

                        candidate=baseline_candidate,
                        post=baseline_post,
                    )
                )

        # =========================================================
        # B) controlled percentile branch
        # =========================================================
        score_source_result = (
            results.naclip
            if PERCENTILE_MAP_SOURCE == "naclip"
            else results.current
        )

        percentile_check_threshold = (
            percentile_engine.update_current(
                score_source_result
            )
        )

        percentile_candidate = (
            percentile_engine.propose(
                frame=frame,
                step=int(fi),
                result=score_source_result,
                check_threshold=(
                    percentile_check_threshold
                ),
            )
        )

        percentile_is_zoomed = 0

        if percentile_candidate.produced_crop:
            zoomed_results = evaluator.evaluate(
                percentile_candidate.zoomed_frame
            )

            zoomed_source_result = (
                zoomed_results.naclip
                if PERCENTILE_MAP_SOURCE == "naclip"
                else zoomed_results.current
            )

            percentile_post = (
                percentile_engine.accept(
                    current_result=(
                        score_source_result
                    ),
                    zoomed_result=(
                        zoomed_source_result
                    ),
                    candidate=(
                        percentile_candidate
                    ),
                )
            )

            percentile_is_zoomed = int(
                percentile_post[
                    "is_zoomed"
                ]
            )

            if percentile_is_zoomed:
                all_raw_maps.append(
                    zoomed_source_result
                    .raw_cosine_map
                    .reshape(-1)
                )

                percentile_events.append(
                    AcceptedEvent(
                        branch=(
                            f"{PERCENTILE_MAP_SOURCE}_percentile"
                        ),

                        zoom_frame=int(fi),
                        source_local_index=int(local_i),

                        source_frame=frame.copy(),
                        zoomed_frame=(
                            percentile_candidate
                            .zoomed_frame
                            .copy()
                        ),

                        source_result=(
                            score_source_result
                        ),

                        zoomed_result=(
                            zoomed_source_result
                        ),

                        candidate=(
                            percentile_candidate
                        ),

                        post=percentile_post,
                    )
                )

        # =========================================================
        # per-frame data
        # =========================================================
        row = {
            "frame_index": int(fi),

            "current_gaussian": float(
                results.current.gaussian_score
            ),

            "baseline_zoom_prob": float(
                results.current.zoom_in_prob
            ),

            "baseline_check_threshold": float(
                baseline_check_threshold
            ),

            "baseline_is_zoomed": int(
                baseline_is_zoomed
            ),

            "percentile_branch_zoom_prob": float(
                score_source_result.zoom_in_prob
            ),

            "percentile_branch_check_threshold": float(
                percentile_check_threshold
            ),

            "percentile_branch_is_zoomed": int(
                percentile_is_zoomed
            ),
        }

        for q in PERCENTILES:
            row[
                f"current_P{q:g}"
            ] = float(
                score_source_result
                .percentile_score(q)
            )

        per_frame_rows.append(
            row
        )

        if (
            local_i == 0
            or (
                local_i + 1
            ) % PRINT_EVERY_N_FRAMES == 0
            or local_i + 1 == len(frame_indices)
        ):
            scores_txt = " ".join(
                [
                    f"P{q:g}="
                    f"{score_source_result.percentile_score(q):.4f}"
                    for q in PERCENTILES
                ]
            )

            print(
                f"[{local_i+1:4d}/{len(frame_indices):4d}] "
                f"frame={fi} | "
                f"G={results.current.gaussian_score:.3f} | "
                f"{scores_txt}"
            )

    # -------------------------------------------------------------
    # score curves
    # -------------------------------------------------------------
    percentile_current_results = (
        naclip_results
        if PERCENTILE_MAP_SOURCE == "naclip"
        else current_results
    )

    percentile_score_curves: Dict[
        float,
        np.ndarray
    ] = {}

    for q in PERCENTILES:
        percentile_score_curves[
            float(q)
        ] = np.asarray(
            [
                r.percentile_score(q)
                for r in percentile_current_results
            ],
            dtype=np.float64,
        )

    baseline_score_curve = np.asarray(
        [
            r.gaussian_score
            for r in current_results
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Baseline crossings
    # -------------------------------------------------------------
    baseline_event_rows: List[Dict] = []

    for event in baseline_events:
        score_on_zoomed = float(
            event.zoomed_result.gaussian_score
        )

        (
            crossing_local,
            crossing_frame,
            crossing_score,
        ) = first_crossing(
            frame_indices=frame_indices,
            score_curve=baseline_score_curve,
            local_t=event.source_local_index,
            score_on_zoomed=score_on_zoomed,
        )

        if crossing_frame is None:
            jumping_steps = np.nan
        else:
            jumping_steps = int(
                crossing_frame
                - event.zoom_frame
            )

        row = {
            "zoom_frame": int(
                event.zoom_frame
            ),

            "score_type": "current_gaussian",

            "score_current_at_t": float(
                event.source_result.gaussian_score
            ),

            "score_on_zoomed": float(
                score_on_zoomed
            ),

            "crossing_local_index": (
                np.nan
                if crossing_local is None
                else int(crossing_local)
            ),

            "crossing_frame": (
                np.nan
                if crossing_frame is None
                else int(crossing_frame)
            ),

            "crossing_score": (
                np.nan
                if crossing_score is None
                else float(crossing_score)
            ),

            "jumping_steps": jumping_steps,

            "actual_zoom_factor": float(
                event.candidate.actual_zoom_factor
            ),

            "current_gaussian": float(
                event.source_result.gaussian_score
            ),

            "zoomed_gaussian": float(
                event.zoomed_result.gaussian_score
            ),

            "gaussian_required": float(
                event.post["gaussian_required"]
            ),
        }

        baseline_event_rows.append(
            row
        )

        if SAVE_BASELINE_EVENT_PANELS:
            if crossing_frame is None:
                filename = (
                    f"baseline_t{event.zoom_frame:06d}"
                    "_unresolved.png"
                )
            else:
                filename = (
                    f"baseline_t{event.zoom_frame:06d}"
                    f"_k{crossing_frame:06d}"
                    f"_jump{int(jumping_steps):04d}.png"
                )

            save_baseline_event_panel(
                path=(
                    baseline_panel_dir
                    / filename
                ),

                event=event,

                crossing_frame=(
                    crossing_frame
                ),

                crossing_score=(
                    crossing_score
                ),

                jumping_steps=(
                    None
                    if crossing_frame is None
                    else int(jumping_steps)
                ),

                frames=frames,

                baseline_current_results=(
                    current_results
                ),
            )

    # -------------------------------------------------------------
    # Percentile crossings
    # -------------------------------------------------------------
    percentile_event_rows: List[Dict] = []

    # map event -> rows, for visual panel
    event_to_rows: Dict[
        int,
        List[Dict]
    ] = {}

    for event_idx, event in enumerate(
        percentile_events
    ):
        rows_for_event = []

        for q in PERCENTILES:
            q = float(q)

            score_on_zoomed = float(
                event.zoomed_result
                .percentile_score(q)
            )

            (
                crossing_local,
                crossing_frame,
                crossing_score,
            ) = first_crossing(
                frame_indices=frame_indices,
                score_curve=(
                    percentile_score_curves[q]
                ),
                local_t=event.source_local_index,
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
                "event_index": int(
                    event_idx
                ),

                "zoom_frame": int(
                    event.zoom_frame
                ),

                "percentile": float(
                    q
                ),

                "score_source": (
                    PERCENTILE_MAP_SOURCE
                ),

                "score_current_at_t": float(
                    event.source_result
                    .percentile_score(q)
                ),

                "score_on_zoomed": float(
                    score_on_zoomed
                ),

                "crossing_local_index": (
                    np.nan
                    if crossing_local is None
                    else int(crossing_local)
                ),

                "crossing_frame": (
                    np.nan
                    if crossing_frame is None
                    else int(crossing_frame)
                ),

                "crossing_score": (
                    np.nan
                    if crossing_score is None
                    else float(crossing_score)
                ),

                "jumping_steps": jumping_steps,

                "actual_zoom_factor": float(
                    event.candidate.actual_zoom_factor
                ),

                "map_gaussian_current": float(
                    event.source_result.gaussian_score
                ),

                "map_gaussian_zoomed": float(
                    event.zoomed_result.gaussian_score
                ),

                "gaussian_required": float(
                    event.post["gaussian_required"]
                ),
            }

            percentile_event_rows.append(
                row
            )

            rows_for_event.append(
                row
            )

        event_to_rows[
            event_idx
        ] = rows_for_event

    # -------------------------------------------------------------
    # raw cosine common range
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
        raw_vmax = raw_vmin + 1e-3

    # -------------------------------------------------------------
    # percentile event visual panels
    # -------------------------------------------------------------
    if SAVE_PERCENTILE_EVENT_PANELS:
        for event_idx, event in enumerate(
            percentile_events
        ):
            rows = event_to_rows[
                event_idx
            ]

            save_percentile_event_panel(
                path=(
                    percentile_panel_dir
                    / f"zoom_t{event.zoom_frame:06d}.png"
                ),

                event=event,

                percentile_rows=rows,

                frames=frames,

                current_results_for_score_source=(
                    percentile_current_results
                ),

                raw_vmin=raw_vmin,
                raw_vmax=raw_vmax,
            )

    # -------------------------------------------------------------
    # summaries
    # -------------------------------------------------------------
    percentile_summary = []

    for q in PERCENTILES:
        q = float(q)

        rows = [
            r
            for r in percentile_event_rows
            if float(r["percentile"]) == q
        ]

        resolved_jumps = [
            int(
                r["jumping_steps"]
            )
            for r in rows
            if np.isfinite(
                r["jumping_steps"]
            )
        ]

        resolved = len(
            resolved_jumps
        )

        total = len(
            rows
        )

        percentile_summary.append(
            {
                "percentile": q,

                "map_source": (
                    PERCENTILE_MAP_SOURCE
                ),

                "accepted_zoom_events": int(
                    total
                ),

                "resolved_crossing": int(
                    resolved
                ),

                "unresolved_crossing": int(
                    total - resolved
                ),

                "resolve_rate": (
                    float(
                        resolved
                        / total
                    )
                    if total
                    else np.nan
                ),

                "jumping_steps_mean": (
                    float(
                        np.mean(
                            resolved_jumps
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),

                "jumping_steps_median": (
                    float(
                        np.median(
                            resolved_jumps
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),

                "jumping_steps_min": (
                    int(
                        np.min(
                            resolved_jumps
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),

                "jumping_steps_max": (
                    int(
                        np.max(
                            resolved_jumps
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),

                "jump_le_5_fraction": (
                    float(
                        np.mean(
                            np.asarray(
                                resolved_jumps
                            )
                            <= 5
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),

                "jump_le_20_fraction": (
                    float(
                        np.mean(
                            np.asarray(
                                resolved_jumps
                            )
                            <= 20
                        )
                    )
                    if resolved_jumps
                    else np.nan
                ),
            }
        )

    baseline_jumps = [
        int(
            r["jumping_steps"]
        )
        for r in baseline_event_rows
        if np.isfinite(
            r["jumping_steps"]
        )
    ]

    baseline_total = len(
        baseline_event_rows
    )

    baseline_resolved = len(
        baseline_jumps
    )

    baseline_summary = {
        "baseline": "Current-Gaussian",

        "accepted_zoom_events": int(
            baseline_total
        ),

        "resolved_crossing": int(
            baseline_resolved
        ),

        "unresolved_crossing": int(
            baseline_total
            - baseline_resolved
        ),

        "resolve_rate": (
            float(
                baseline_resolved
                / baseline_total
            )
            if baseline_total
            else np.nan
        ),

        "jumping_steps_mean": (
            float(
                np.mean(
                    baseline_jumps
                )
            )
            if baseline_jumps
            else np.nan
        ),

        "jumping_steps_median": (
            float(
                np.median(
                    baseline_jumps
                )
            )
            if baseline_jumps
            else np.nan
        ),

        "jumping_steps_min": (
            int(
                np.min(
                    baseline_jumps
                )
            )
            if baseline_jumps
            else np.nan
        ),

        "jumping_steps_max": (
            int(
                np.max(
                    baseline_jumps
                )
            )
            if baseline_jumps
            else np.nan
        ),
    }

    # -------------------------------------------------------------
    # save concise files
    # -------------------------------------------------------------
    save_csv(
        outdir
        / "percentile_event_results.csv",
        percentile_event_rows,
    )

    save_csv(
        outdir
        / "percentile_summary.csv",
        percentile_summary,
    )

    save_csv(
        outdir
        / "baseline_current_gaussian_events.csv",
        baseline_event_rows,
    )

    save_csv(
        outdir
        / "per_frame_scores.csv",
        per_frame_rows,
    )

    (
        outdir
        / "baseline_summary.json"
    ).write_text(
        json.dumps(
            baseline_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # plots
    # -------------------------------------------------------------
    save_jumping_steps_by_percentile(
        outdir
        / "jumping_steps_by_percentile.png",
        percentile_event_rows,
    )

    save_median_jump_curve(
        outdir
        / "percentile_vs_jump_horizon.png",
        percentile_summary,
        baseline_summary,
    )

    save_crossing_timeline(
        outdir
        / "percentile_crossing_timelines.png",
        frame_indices,
        percentile_score_curves,
        percentile_event_rows,
    )

    # -------------------------------------------------------------
    # console summary
    # -------------------------------------------------------------
    print()
    print(
        "=" * 96
    )

    print(
        "BASELINE: CURRENT-GAUSSIAN"
    )

    print(
        "=" * 96
    )

    print(
        f"accepted={baseline_total} | "
        f"resolved={baseline_resolved} | "
        f"resolve_rate={baseline_summary['resolve_rate']}"
    )

    print(
        f"jump mean={baseline_summary['jumping_steps_mean']} | "
        f"median={baseline_summary['jumping_steps_median']} | "
        f"range={baseline_summary['jumping_steps_min']}.."
        f"{baseline_summary['jumping_steps_max']}"
    )

    print()
    print(
        "=" * 96
    )

    print(
        "PERCENTILE SWEEP"
    )

    print(
        "=" * 96
    )

    for row in percentile_summary:
        print(
            f"P{row['percentile']:g}: "
            f"accepted={row['accepted_zoom_events']} | "
            f"resolved={row['resolved_crossing']} | "
            f"mean={row['jumping_steps_mean']} | "
            f"median={row['jumping_steps_median']} | "
            f"range={row['jumping_steps_min']}..{row['jumping_steps_max']} | "
            f"J<=5={row['jump_le_5_fraction']} | "
            f"J<=20={row['jump_le_20_fraction']}"
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
        "  percentile_summary.csv"
    )

    print(
        "  percentile_event_results.csv"
    )

    print(
        "  baseline_current_gaussian_events.csv"
    )

    print(
        "  percentile_vs_jump_horizon.png"
    )

    print(
        "  jumping_steps_by_percentile.png"
    )

    print(
        "  percentile_crossing_timelines.png"
    )

    print(
        "  percentile_event_panels/"
    )

    print(
        "  baseline_current_gaussian_panels/"
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
