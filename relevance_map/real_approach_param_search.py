#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_approach_param_search_no_mark.py

Search RELEVANCE_THRESHOLD / RELEVANCE_TEMPERATURE from trajectories saved by
real_approach_test.py WITHOUT using any manually entered
    mark far / mark mid / mark near / mark contact
labels.

The script uses only objective data already stored in trajectory.npz:
    - raw MineCLIP patch cosine values
    - frame order
    - action_commands (attack/use when available)
    - reward / done / target inventory (success when available)

No MineDojo run and no MineCLIP forward are needed again.

Core idea
---------
For each trajectory we first derive automatic reference frames from RAW cosine,
before applying any candidate tau/T:

1. PRE-INTERACTION RANGE
   Frames before the last attack/use that precedes success.
   If no attack/use exists, use frames before success.
   If the trajectory has no success, use the whole trajectory.

2. WEAK-SEMANTIC FRAMES
   Bottom quantile of an automatic raw semantic-strength signal.

3. STRONG-SEMANTIC FRAMES
   Top quantile of the same raw semantic-strength signal.

The raw semantic-strength signal is based on:
    - raw P95-P50 patch cosine contrast
    - raw P95 patch cosine

This is independent of RELEVANCE_THRESHOLD / RELEVANCE_TEMPERATURE, so the
parameter search is not using manual stage labels.

The ranking prefers a fixed-scale relevance transform that:
    - suppresses weak/background-like frames
    - clearly activates on strong target-like frames
    - forms a larger connected semantic region on strong frames
    - does not saturate the whole image
    - retains useful response at the objective interaction frame
    - is not excessively temporally unstable

Success-frame suppression is RECORDED but is NOT part of the main score,
because it is task-dependent:
    tree/sand/ore may disappear after success,
    but water or sheep may remain visible.

Important limitation
--------------------
There is still no pixel-level ground-truth target mask, so "best" means
"best trajectory-calibrated fixed-scale parameters under these objective
automatic criteria", not a universal segmentation optimum.

For a final cross-task choice, add multiple successful trajectories from
different tasks/seeds to TRAJECTORY_PATHS.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from matplotlib import colormaps
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ============================================================
# PyCharm configuration
# ============================================================

# Add one or more trajectory.npz files here.
# No manual mark labels are required.
TRAJECTORY_PATHS = [
    PROJECT_ROOT
    / "relevance_map"
    / "real_approach_runs"
    / "shear_sheep"
    / "world_61_taskseed_0_20260909_103155"
    / "trajectory.npz",
]


# Current baseline for comparison.
BASELINE_THRESHOLD = 0.300
BASELINE_TEMPERATURE = 0.030


# Dense search grid.
THRESHOLDS = np.round(
    np.arange(
        0.250,
        0.336,
        0.005,
    ),
    3,
)

TEMPERATURES = np.round(
    np.arange(
        0.015,
        0.061,
        0.005,
    ),
    3,
)

REFERENCE_SETTINGS = [
    (0.250, 0.050),
    (0.285, 0.030),
    (0.300, 0.030),
    (0.315, 0.030),
    (0.330, 0.030),
]


# ============================================================
# Automatic reference-frame selection
# ============================================================

# Among frames before interaction/success:
# bottom 25% = weak semantic frames
# top    20% = strong semantic frames
WEAK_FRACTION = 0.25
STRONG_FRACTION = 0.20

# Always select at least this many if the trajectory is long enough.
MIN_REFERENCE_FRAMES = 8

# Raw semantic strength:
#   65% rank(P95-P50) + 35% rank(P95)
#
# P95-P50 measures patch separation.
# P95 preserves cases where a broad/large target raises many patches.
RAW_CONTRAST_WEIGHT = 0.65
RAW_P95_WEIGHT = 0.35

# Exclude a few frames immediately AFTER interaction from normal calibration.
# The attack/use transition can contain animation, breaking, pickup, etc.
POST_INTERACTION_TRANSITION_FRAMES = 3


# ============================================================
# Patch / output dimensions
# ============================================================

GRID_H = 10
GRID_W = 16

TARGET_H = 160
TARGET_W = 256

ALPHA = 0.50
GIF_DURATION = 120

TOP_K_VISUALIZE = 5


# ============================================================
# Scoring
# ============================================================

# No manual far/near/contact labels are used here.
SCORE_WEIGHTS = {
    # Main signal: strong automatically selected frames should have a
    # much larger >0.5 region than weak frames.
    "strong_weak_area05_sep": 25.0,

    # High-confidence relevance should separate strong / weak frames.
    "strong_weak_area07_sep": 15.0,

    # Average relevance separation.
    "strong_weak_mean_sep": 12.0,

    # Largest connected semantic region should be much larger in strong frames.
    "strong_weak_cc05_sep": 18.0,

    # Weak frames should not already light up broadly.
    "weak_non_saturation": 10.0,

    # Strong frames should contain a useful, but not globally saturated,
    # amount of target relevance.
    "strong_useful_occupancy": 8.0,

    # If objective attack/use action exists, relevance should not collapse
    # completely at the pre-interaction observation.
    "interaction_retention": 7.0,

    # Avoid temperature choices that create excessive frame-to-frame flicker.
    "temporal_stability": 5.0,
}


# Quantities mapped to a 0..1 score.
GOOD_AREA05_GAP = 0.15
GOOD_AREA07_GAP = 0.10
GOOD_MEAN_GAP = 0.10
GOOD_CC05_GAP = 0.15

# Weak/background-like frames should stay sparse.
WEAK_AREA05_SOFT_LIMIT = 0.08
WEAK_MEAN_SOFT_LIMIT = 0.32

# Strong frames: broad acceptable band. We do NOT force a tiny object size.
STRONG_AREA05_GOOD_LOW = 0.08
STRONG_AREA05_GOOD_HIGH = 0.45

# Interaction should retain at least this fraction of strong-frame >0.5 area.
MIN_GOOD_INTERACTION_RETENTION = 0.35

# Mean frame-to-frame relevance change scale.
STABILITY_SCALE = 0.04


# ============================================================
# Output
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = (
    SCRIPT_DIR
    / "relevance_test"
    / f"real_approach_param_search_no_mark_{timestamp}"
)


# ============================================================
# Generic helpers
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def clip01(value):
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def high_is_good(value, good_value):
    if not np.isfinite(value):
        return np.nan

    if good_value <= 0:
        return 1.0

    return clip01(
        value / good_value
    )


def low_is_good(value, soft_limit):
    """
    score=1 when value <= soft_limit,
    linearly falls to 0 at 2*soft_limit.
    """
    if not np.isfinite(value):
        return np.nan

    if value <= soft_limit:
        return 1.0

    return clip01(
        1.0
        - (
            value - soft_limit
        )
        / max(
            soft_limit,
            1e-8,
        )
    )


def range_score(
    value,
    good_low,
    good_high,
    soft_width,
):
    if not np.isfinite(value):
        return np.nan

    if good_low <= value <= good_high:
        return 1.0

    if value < good_low:
        return clip01(
            1.0
            - (
                good_low - value
            )
            / soft_width
        )

    return clip01(
        1.0
        - (
            value - good_high
        )
        / soft_width
    )


def percentile_rank(values):
    """
    Return rank in [0,1].
    Only relative order matters; robust to task-specific cosine scale.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n <= 1:
        return np.zeros_like(
            values,
            dtype=np.float64,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    ranks[
        order
    ] = np.arange(
        n,
        dtype=np.float64,
    )

    return ranks / (
        n - 1
    )


def largest_cc_fraction_patch(
    patch_values,
    threshold=0.5,
):
    """
    patch_values: [10,16]
    """

    binary = (
        patch_values
        > threshold
    ).astype(
        np.uint8
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if num_labels <= 1:
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


# ============================================================
# Load trajectory
# ============================================================

def normalize_command(command):
    return str(
        command
    ).strip().lower()


def load_trajectory(npz_path):

    npz_path = Path(
        npz_path
    ).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(
            f"trajectory not found: {npz_path}"
        )

    data = np.load(
        npz_path,
        allow_pickle=False,
    )

    if "frames" not in data:
        raise KeyError(
            f"'frames' not found in {npz_path}"
        )

    if "cosines" not in data:
        raise KeyError(
            f"'cosines' not found in {npz_path}"
        )

    frames = np.asarray(
        data[
            "frames"
        ]
    )

    cosine = np.asarray(
        data[
            "cosines"
        ],
        dtype=np.float32,
    )

    if cosine.ndim == 3:
        cosine = cosine.reshape(
            cosine.shape[0],
            -1,
        )

    if cosine.ndim != 2:
        raise ValueError(
            f"Unexpected cosine shape: {cosine.shape}"
        )

    if cosine.shape[1] != (
        GRID_H * GRID_W
    ):
        raise ValueError(
            f"Expected 160 patch cosine values, got {cosine.shape}"
        )

    if frames.shape[0] != cosine.shape[0]:
        raise ValueError(
            "frames/cosines length mismatch: "
            f"{frames.shape[0]} vs {cosine.shape[0]}"
        )

    T = frames.shape[0]

    # --------------------------------------------------------
    # Objective task-success information
    # --------------------------------------------------------

    reward = np.zeros(
        T,
        dtype=np.float32,
    )

    done = np.zeros(
        T,
        dtype=np.uint8,
    )

    inventory = np.zeros(
        T,
        dtype=np.float32,
    )

    if "reward" in data:
        x = np.asarray(
            data[
                "reward"
            ]
        ).reshape(-1)

        reward[
            :min(
                T,
                len(x),
            )
        ] = x[
            :T
        ]

    if "done" in data:
        x = np.asarray(
            data[
                "done"
            ]
        ).reshape(-1)

        done[
            :min(
                T,
                len(x),
            )
        ] = x[
            :T
        ]

    if "target_inventory_quantity" in data:
        x = np.asarray(
            data[
                "target_inventory_quantity"
            ]
        ).reshape(-1)

        inventory[
            :min(
                T,
                len(x),
            )
        ] = x[
            :T
        ]

    success_candidates = np.where(
        (reward > 0)
        | (done > 0)
        | (inventory >= 1)
    )[0]

    success_step = (
        int(
            success_candidates[
                0
            ]
        )
        if len(
            success_candidates
        )
        else None
    )

    # --------------------------------------------------------
    # Objective interaction information
    #
    # action_commands[i] is the command that transitions
    # observation i -> observation i+1.
    #
    # The useful "contact-like" observation is therefore i,
    # just BEFORE attack/use is executed.
    # --------------------------------------------------------

    action_commands = None
    interaction_action_indices = []

    if "action_commands" in data:

        action_commands = np.asarray(
            data[
                "action_commands"
            ]
        ).astype(str)

        for i, command in enumerate(
            action_commands
        ):

            command = normalize_command(
                command
            )

            if command in (
                "x",
                "attack",
                "e",
                "use",
            ):
                interaction_action_indices.append(
                    i
                )

    # Prefer the LAST interaction before objective success.
    interaction_step = None

    if interaction_action_indices:

        candidates = interaction_action_indices

        if success_step is not None:

            before_success = [
                i
                for i in candidates
                if i
                < success_step
            ]

            if before_success:
                candidates = (
                    before_success
                )

        if candidates:
            interaction_step = int(
                candidates[
                    -1
                ]
            )

    # If action commands were not saved but there is a success event,
    # use the frame just before success as a weak fallback.
    interaction_is_fallback = False

    if (
        interaction_step is None
        and success_step is not None
        and success_step > 0
    ):
        interaction_step = int(
            success_step - 1
        )

        interaction_is_fallback = True

    # --------------------------------------------------------
    # Automatic raw semantic strength
    # --------------------------------------------------------

    raw_p50 = np.percentile(
        cosine,
        50,
        axis=1,
    )

    raw_p90 = np.percentile(
        cosine,
        90,
        axis=1,
    )

    raw_p95 = np.percentile(
        cosine,
        95,
        axis=1,
    )

    raw_max = np.max(
        cosine,
        axis=1,
    )

    raw_contrast = (
        raw_p95
        - raw_p50
    )

    # Calibration horizon:
    # do not use post-interaction destruction/pickup frames as normal
    # target-approach examples.
    if interaction_step is not None:
        calibration_end = (
            interaction_step + 1
        )

    elif success_step is not None:
        calibration_end = (
            success_step
        )

    else:
        calibration_end = T

    calibration_end = int(
        np.clip(
            calibration_end,
            1,
            T,
        )
    )

    calibration_indices = np.arange(
        0,
        calibration_end,
        dtype=np.int64,
    )

    # Rank only within the objective calibration horizon.
    contrast_rank = percentile_rank(
        raw_contrast[
            calibration_indices
        ]
    )

    p95_rank = percentile_rank(
        raw_p95[
            calibration_indices
        ]
    )

    semantic_strength_local = (
        RAW_CONTRAST_WEIGHT
        * contrast_rank
        + RAW_P95_WEIGHT
        * p95_rank
    )

    semantic_strength = np.full(
        T,
        np.nan,
        dtype=np.float32,
    )

    semantic_strength[
        calibration_indices
    ] = semantic_strength_local

    n = len(
        calibration_indices
    )

    weak_count = max(
        1,
        min(
            n,
            max(
                MIN_REFERENCE_FRAMES,
                int(
                    round(
                        n
                        * WEAK_FRACTION
                    )
                ),
            ),
        ),
    )

    strong_count = max(
        1,
        min(
            n,
            max(
                MIN_REFERENCE_FRAMES,
                int(
                    round(
                        n
                        * STRONG_FRACTION
                    )
                ),
            ),
        ),
    )

    order = np.argsort(
        semantic_strength_local
    )

    weak_local = order[
        :weak_count
    ]

    strong_local = order[
        -strong_count:
    ]

    weak_indices = np.sort(
        calibration_indices[
            weak_local
        ]
    )

    strong_indices = np.sort(
        calibration_indices[
            strong_local
        ]
    )

    # Interaction evaluation: exactly the pre-action observation if available.
    interaction_indices = np.asarray(
        [],
        dtype=np.int64,
    )

    if interaction_step is not None:
        interaction_indices = np.asarray(
            [
                int(
                    np.clip(
                        interaction_step,
                        0,
                        T - 1,
                    )
                )
            ],
            dtype=np.int64,
        )

    success_indices = np.asarray(
        [],
        dtype=np.int64,
    )

    if success_step is not None:
        success_indices = np.asarray(
            [
                success_step
            ],
            dtype=np.int64,
        )

    task_name = (
        str(
            data[
                "task_name"
            ]
        )
        if "task_name" in data
        else npz_path.parent.parent.name
    )

    prompt = (
        str(
            data[
                "prompt"
            ]
        )
        if "prompt" in data
        else ""
    )

    return {
        "path": npz_path,
        "name": npz_path.parent.name,

        "task_name": task_name,
        "prompt": prompt,

        "frames": frames,
        "cosine": cosine,

        "reward": reward,
        "done": done,
        "inventory": inventory,

        "action_commands": action_commands,

        "success_step": success_step,

        "interaction_step": interaction_step,
        "interaction_is_fallback": interaction_is_fallback,

        "calibration_indices": calibration_indices,

        "weak_indices": weak_indices,
        "strong_indices": strong_indices,

        "interaction_indices": interaction_indices,
        "success_indices": success_indices,

        "raw_p50": raw_p50.astype(
            np.float32
        ),
        "raw_p90": raw_p90.astype(
            np.float32
        ),
        "raw_p95": raw_p95.astype(
            np.float32
        ),
        "raw_max": raw_max.astype(
            np.float32
        ),
        "raw_contrast": raw_contrast.astype(
            np.float32
        ),

        "semantic_strength": semantic_strength,
    }


# ============================================================
# Relevance transform
# ============================================================

def cosine_to_relevance(
    cosine,
    threshold,
    temperature,
):

    if temperature <= 0:
        raise ValueError(
            "temperature must > 0"
        )

    x = (
        cosine
        - threshold
    ) / temperature

    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -x
            )
        )
    ).astype(
        np.float32
    )


def resize_relevance_maps(
    relevance_patch,
):

    T = relevance_patch.shape[
        0
    ]

    x = torch.from_numpy(
        relevance_patch
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
        x[
            :,
            0,
        ]
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


# ============================================================
# Relevance metrics
# ============================================================

def frame_metrics_from_patch(
    relevance_patch,
):

    T = relevance_patch.shape[
        0
    ]

    mean = relevance_patch.mean(
        axis=1
    )

    p50 = np.percentile(
        relevance_patch,
        50,
        axis=1,
    )

    p90 = np.percentile(
        relevance_patch,
        90,
        axis=1,
    )

    p95 = np.percentile(
        relevance_patch,
        95,
        axis=1,
    )

    area05 = np.mean(
        relevance_patch > 0.5,
        axis=1,
    )

    area07 = np.mean(
        relevance_patch > 0.7,
        axis=1,
    )

    area09 = np.mean(
        relevance_patch > 0.9,
        axis=1,
    )

    cc05 = np.zeros(
        T,
        dtype=np.float32,
    )

    for t in range(
        T
    ):

        patch = relevance_patch[
            t
        ].reshape(
            GRID_H,
            GRID_W,
        )

        cc05[
            t
        ] = largest_cc_fraction_patch(
            patch,
            threshold=0.5,
        )

    return {
        "mean": mean.astype(
            np.float32
        ),

        "p50": p50.astype(
            np.float32
        ),

        "p90": p90.astype(
            np.float32
        ),

        "p95": p95.astype(
            np.float32
        ),

        "p95_p50": (
            p95 - p50
        ).astype(
            np.float32
        ),

        "area05": area05.astype(
            np.float32
        ),

        "area07": area07.astype(
            np.float32
        ),

        "area09": area09.astype(
            np.float32
        ),

        "cc05": cc05,
    }


def aggregate_indices(
    frame_metrics,
    indices,
):

    if (
        indices is None
        or len(
            indices
        ) == 0
    ):
        return None

    result = {}

    for key, values in (
        frame_metrics.items()
    ):

        selected = values[
            indices
        ]

        result[
            key
        ] = float(
            np.mean(
                selected
            )
        )

    return result


def temporal_stability_score(
    frame_metrics,
    calibration_indices,
):

    if len(
        calibration_indices
    ) < 2:
        return np.nan

    values = frame_metrics[
        "mean"
    ][
        calibration_indices
    ]

    mean_abs_delta = float(
        np.mean(
            np.abs(
                np.diff(
                    values
                )
            )
        )
    )

    return float(
        np.exp(
            -mean_abs_delta
            / STABILITY_SCALE
        )
    )


# ============================================================
# Score one trajectory
# ============================================================

def evaluate_setting_on_trajectory(
    trajectory,
    threshold,
    temperature,
):

    relevance = cosine_to_relevance(
        trajectory[
            "cosine"
        ],
        threshold,
        temperature,
    )

    frame_metrics = frame_metrics_from_patch(
        relevance
    )

    weak = aggregate_indices(
        frame_metrics,
        trajectory[
            "weak_indices"
        ],
    )

    strong = aggregate_indices(
        frame_metrics,
        trajectory[
            "strong_indices"
        ],
    )

    interaction = aggregate_indices(
        frame_metrics,
        trajectory[
            "interaction_indices"
        ],
    )

    success = aggregate_indices(
        frame_metrics,
        trajectory[
            "success_indices"
        ],
    )

    components = {}

    # --------------------------------------------------------
    # Strong vs weak automatic-frame separation
    # --------------------------------------------------------

    components[
        "strong_weak_area05_sep"
    ] = high_is_good(
        strong[
            "area05"
        ]
        - weak[
            "area05"
        ],
        GOOD_AREA05_GAP,
    )

    components[
        "strong_weak_area07_sep"
    ] = high_is_good(
        strong[
            "area07"
        ]
        - weak[
            "area07"
        ],
        GOOD_AREA07_GAP,
    )

    components[
        "strong_weak_mean_sep"
    ] = high_is_good(
        strong[
            "mean"
        ]
        - weak[
            "mean"
        ],
        GOOD_MEAN_GAP,
    )

    components[
        "strong_weak_cc05_sep"
    ] = high_is_good(
        strong[
            "cc05"
        ]
        - weak[
            "cc05"
        ],
        GOOD_CC05_GAP,
    )

    # --------------------------------------------------------
    # Weak-frame non-saturation
    # --------------------------------------------------------

    weak_area_score = low_is_good(
        weak[
            "area05"
        ],
        WEAK_AREA05_SOFT_LIMIT,
    )

    weak_mean_score = low_is_good(
        weak[
            "mean"
        ],
        WEAK_MEAN_SOFT_LIMIT,
    )

    components[
        "weak_non_saturation"
    ] = (
        0.65
        * weak_area_score
        + 0.35
        * weak_mean_score
    )

    # --------------------------------------------------------
    # Strong-frame useful occupancy
    # --------------------------------------------------------

    components[
        "strong_useful_occupancy"
    ] = range_score(
        strong[
            "area05"
        ],
        STRONG_AREA05_GOOD_LOW,
        STRONG_AREA05_GOOD_HIGH,
        soft_width=0.20,
    )

    # --------------------------------------------------------
    # Interaction retention
    # --------------------------------------------------------

    interaction_retention = np.nan

    if (
        interaction is not None
        and strong[
            "area05"
        ] > 1e-8
    ):

        interaction_retention = (
            interaction[
                "area05"
            ]
            / strong[
                "area05"
            ]
        )

        components[
            "interaction_retention"
        ] = high_is_good(
            interaction_retention,
            MIN_GOOD_INTERACTION_RETENTION,
        )

    else:

        components[
            "interaction_retention"
        ] = np.nan

    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    components[
        "temporal_stability"
    ] = temporal_stability_score(
        frame_metrics,
        trajectory[
            "calibration_indices"
        ],
    )

    # --------------------------------------------------------
    # Success suppression: diagnostic only, not score.
    # --------------------------------------------------------

    success_suppression = np.nan

    if (
        success is not None
        and interaction is not None
        and interaction[
            "area05"
        ] > 1e-8
    ):

        success_suppression = (
            interaction[
                "area05"
            ]
            - success[
                "area05"
            ]
        ) / interaction[
            "area05"
        ]

    # --------------------------------------------------------
    # Weighted final score
    # --------------------------------------------------------

    weighted_sum = 0.0
    weight_sum = 0.0

    for key, weight in (
        SCORE_WEIGHTS.items()
    ):

        value = components.get(
            key,
            np.nan,
        )

        if np.isfinite(
            value
        ):

            weighted_sum += (
                value
                * weight
            )

            weight_sum += weight

    score = (
        100.0
        * weighted_sum
        / weight_sum
        if weight_sum > 0
        else np.nan
    )

    result = {
        "score": float(
            score
        ),

        "threshold": float(
            threshold
        ),

        "temperature": float(
            temperature
        ),

        "interaction_retention_ratio": float(
            interaction_retention
        ),

        "success_suppression_diagnostic": float(
            success_suppression
        ),
    }

    # Score components.
    for key, value in (
        components.items()
    ):
        result[
            f"component_{key}"
        ] = float(
            value
        )

    # Flatten automatic groups.
    groups = {
        "weak": weak,
        "strong": strong,
        "interaction": interaction,
        "success": success,
    }

    for group_name, values in (
        groups.items()
    ):

        if values is None:
            continue

        for metric_name, metric_value in (
            values.items()
        ):

            result[
                f"{group_name}_{metric_name}"
            ] = float(
                metric_value
            )

    return result


# ============================================================
# Aggregate across trajectories
# ============================================================

def evaluate_setting(
    trajectories,
    threshold,
    temperature,
):

    per_trajectory = []

    for trajectory in (
        trajectories
    ):

        per_trajectory.append(
            evaluate_setting_on_trajectory(
                trajectory,
                threshold,
                temperature,
            )
        )

    all_keys = set()

    for row in (
        per_trajectory
    ):
        all_keys.update(
            row.keys()
        )

    result = {
        "threshold": float(
            threshold
        ),

        "temperature": float(
            temperature
        ),

        "num_trajectories": len(
            trajectories
        ),
    }

    for key in sorted(
        all_keys
    ):

        if key in (
            "threshold",
            "temperature",
        ):
            continue

        values = []

        for row in (
            per_trajectory
        ):

            if key not in row:
                continue

            value = row[
                key
            ]

            if np.isfinite(
                value
            ):
                values.append(
                    value
                )

        result[
            key
        ] = (
            float(
                np.mean(
                    values
                )
            )
            if values
            else np.nan
        )

    scores = [
        row[
            "score"
        ]
        for row in per_trajectory
        if np.isfinite(
            row[
                "score"
            ]
        )
    ]

    if scores:

        result[
            "score_mean"
        ] = float(
            np.mean(
                scores
            )
        )

        result[
            "score_min"
        ] = float(
            np.min(
                scores
            )
        )

        result[
            "score_std"
        ] = float(
            np.std(
                scores
            )
        )

        # Prefer both good average quality and robustness.
        result[
            "score"
        ] = (
            0.80
            * result[
                "score_mean"
            ]
            + 0.20
            * result[
                "score_min"
            ]
        )

    return result


# ============================================================
# Candidate grid
# ============================================================

def build_candidates():

    pairs = set()

    for threshold in (
        THRESHOLDS
    ):

        for temperature in (
            TEMPERATURES
        ):

            pairs.add(
                (
                    round(
                        float(
                            threshold
                        ),
                        6,
                    ),

                    round(
                        float(
                            temperature
                        ),
                        6,
                    ),
                )
            )

    for threshold, temperature in (
        REFERENCE_SETTINGS
    ):

        pairs.add(
            (
                round(
                    float(
                        threshold
                    ),
                    6,
                ),

                round(
                    float(
                        temperature
                    ),
                    6,
                ),
            )
        )

    pairs.add(
        (
            round(
                BASELINE_THRESHOLD,
                6,
            ),

            round(
                BASELINE_TEMPERATURE,
                6,
            ),
        )
    )

    return sorted(
        pairs
    )


# ============================================================
# CSV helpers
# ============================================================

def save_rows_csv(
    rows,
    path,
):

    if not rows:
        return

    all_keys = set()

    for row in rows:
        all_keys.update(
            row.keys()
        )

    preferred = [
        "rank",
        "score",
        "score_mean",
        "score_min",
        "score_std",
        "threshold",
        "temperature",
        "num_trajectories",

        "weak_mean",
        "weak_p95",
        "weak_area05",
        "weak_area07",
        "weak_cc05",

        "strong_mean",
        "strong_p95",
        "strong_area05",
        "strong_area07",
        "strong_cc05",

        "interaction_mean",
        "interaction_p95",
        "interaction_area05",
        "interaction_area07",
        "interaction_cc05",

        "success_mean",
        "success_p95",
        "success_area05",
        "success_area07",
        "success_cc05",

        "interaction_retention_ratio",
        "success_suppression_diagnostic",
    ]

    remaining = sorted(
        all_keys
        - set(
            preferred
        )
    )

    fieldnames = (
        [
            key
            for key in preferred
            if key in all_keys
        ]
        + remaining
    )

    with Path(
        path
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# ============================================================
# Save automatic frame selection for auditing
# ============================================================

def save_auto_selection_csv(
    trajectory,
    path,
):

    T = trajectory[
        "frames"
    ].shape[0]

    weak_set = set(
        trajectory[
            "weak_indices"
        ].tolist()
    )

    strong_set = set(
        trajectory[
            "strong_indices"
        ].tolist()
    )

    interaction_set = set(
        trajectory[
            "interaction_indices"
        ].tolist()
    )

    success_set = set(
        trajectory[
            "success_indices"
        ].tolist()
    )

    rows = []

    for step in range(
        T
    ):

        labels = []

        if step in weak_set:
            labels.append(
                "weak"
            )

        if step in strong_set:
            labels.append(
                "strong"
            )

        if step in interaction_set:
            labels.append(
                "interaction"
            )

        if step in success_set:
            labels.append(
                "success"
            )

        command = ""

        action_commands = trajectory[
            "action_commands"
        ]

        if (
            action_commands is not None
            and step < len(
                action_commands
            )
        ):
            command = str(
                action_commands[
                    step
                ]
            )

        rows.append(
            {
                "step": step,

                "auto_group": "|".join(
                    labels
                ),

                "raw_p50": float(
                    trajectory[
                        "raw_p50"
                    ][
                        step
                    ]
                ),

                "raw_p90": float(
                    trajectory[
                        "raw_p90"
                    ][
                        step
                    ]
                ),

                "raw_p95": float(
                    trajectory[
                        "raw_p95"
                    ][
                        step
                    ]
                ),

                "raw_max": float(
                    trajectory[
                        "raw_max"
                    ][
                        step
                    ]
                ),

                "raw_p95_p50": float(
                    trajectory[
                        "raw_contrast"
                    ][
                        step
                    ]
                ),

                "semantic_strength": float(
                    trajectory[
                        "semantic_strength"
                    ][
                        step
                    ]
                ),

                "action_command": command,

                "reward": float(
                    trajectory[
                        "reward"
                    ][
                        step
                    ]
                ),

                "done": int(
                    trajectory[
                        "done"
                    ][
                        step
                    ]
                ),

                "target_inventory_quantity": float(
                    trajectory[
                        "inventory"
                    ][
                        step
                    ]
                ),
            }
        )

    save_rows_csv(
        rows,
        path,
    )


# ============================================================
# Visualization helpers
# ============================================================

def make_overlay(
    frame,
    relevance,
    alpha=ALPHA,
):

    relevance = np.clip(
        relevance,
        0.0,
        1.0,
    )

    heatmap = (
        colormaps[
            "turbo"
        ](
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
        (
            1.0 - alpha
        )
        * frame
        + alpha
        * heatmap
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


def auto_group_for_step(
    trajectory,
    step,
):

    labels = []

    if step in set(
        trajectory[
            "weak_indices"
        ].tolist()
    ):
        labels.append(
            "WEAK"
        )

    if step in set(
        trajectory[
            "strong_indices"
        ].tolist()
    ):
        labels.append(
            "STRONG"
        )

    if step in set(
        trajectory[
            "interaction_indices"
        ].tolist()
    ):
        labels.append(
            "INTERACTION"
        )

    if step in set(
        trajectory[
            "success_indices"
        ].tolist()
    ):
        labels.append(
            "SUCCESS"
        )

    return "|".join(
        labels
    )


def save_setting_gif(
    trajectory,
    threshold,
    temperature,
    output_path,
):

    frames = trajectory[
        "frames"
    ]

    relevance_patch = cosine_to_relevance(
        trajectory[
            "cosine"
        ],
        threshold,
        temperature,
    )

    maps = resize_relevance_maps(
        relevance_patch
    )

    gif_frames = []

    for step in range(
        len(
            frames
        )
    ):

        overlay = make_overlay(
            frames[
                step
            ],
            maps[
                step
            ],
        )

        H, W, _ = overlay.shape

        canvas = Image.new(
            "RGB",
            (
                W,
                H + 44,
            ),
            "black",
        )

        canvas.paste(
            Image.fromarray(
                overlay
            ),
            (
                0,
                44,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        group = auto_group_for_step(
            trajectory,
            step,
        )

        current = relevance_patch[
            step
        ]

        draw.text(
            (
                5,
                5,
            ),
            (
                f"tau={threshold:.3f} "
                f"T={temperature:.3f} "
                f"step={step} "
                f"auto={group or '-'}"
            ),
            fill="white",
        )

        draw.text(
            (
                5,
                24,
            ),
            (
                f"mean={current.mean():.3f} "
                f">.5={np.mean(current > .5):.3f} "
                f">.7={np.mean(current > .7):.3f} "
                f"P95={np.percentile(current,95):.3f}"
            ),
            fill="white",
        )

        gif_frames.append(
            canvas
        )

    gif_frames[
        0
    ].save(
        output_path,
        save_all=True,
        append_images=gif_frames[
            1:
        ],
        duration=GIF_DURATION,
        loop=0,
    )


def make_global_minmax_maps(
    cosine,
):

    vmin = float(
        cosine.min()
    )

    vmax = float(
        cosine.max()
    )

    normalized = (
        cosine - vmin
    ) / (
        vmax - vmin
        + 1e-8
    )

    return resize_relevance_maps(
        normalized.astype(
            np.float32
        )
    )


def save_top_comparison_gif(
    trajectory,
    ranked_rows,
    output_path,
):

    frames = trajectory[
        "frames"
    ]

    cosine = trajectory[
        "cosine"
    ]

    minmax_maps = make_global_minmax_maps(
        cosine
    )

    settings = [
        (
            "baseline",
            BASELINE_THRESHOLD,
            BASELINE_TEMPERATURE,
        )
    ]

    for rank_i, row in enumerate(
        ranked_rows[
            :3
        ],
        start=1,
    ):

        settings.append(
            (
                f"top{rank_i}",
                row[
                    "threshold"
                ],
                row[
                    "temperature"
                ],
            )
        )

    maps = {}

    for name, threshold, temperature in (
        settings
    ):

        relevance = cosine_to_relevance(
            cosine,
            threshold,
            temperature,
        )

        maps[
            name
        ] = resize_relevance_maps(
            relevance
        )

    columns = (
        2
        + len(
            settings
        )
    )

    gif_frames = []

    for step in range(
        len(
            frames
        )
    ):

        H, W, _ = frames[
            step
        ].shape

        canvas = Image.new(
            "RGB",
            (
                W * columns,
                H + 48,
            ),
            "black",
        )

        draw = ImageDraw.Draw(
            canvas
        )

        # RGB
        canvas.paste(
            Image.fromarray(
                frames[
                    step
                ]
            ),
            (
                0,
                48,
            ),
        )

        draw.text(
            (
                5,
                7,
            ),
            "RGB",
            fill="white",
        )

        # global min-max reference only
        minmax_overlay = make_overlay(
            frames[
                step
            ],
            minmax_maps[
                step
            ],
        )

        canvas.paste(
            Image.fromarray(
                minmax_overlay
            ),
            (
                W,
                48,
            ),
        )

        draw.text(
            (
                W + 5,
                7,
            ),
            "global min-max",
            fill="white",
        )

        # fixed-scale settings
        for idx, (
            name,
            threshold,
            temperature,
        ) in enumerate(
            settings
        ):

            overlay = make_overlay(
                frames[
                    step
                ],
                maps[
                    name
                ][
                    step
                ],
            )

            x = (
                idx + 2
            ) * W

            canvas.paste(
                Image.fromarray(
                    overlay
                ),
                (
                    x,
                    48,
                ),
            )

            draw.text(
                (
                    x + 5,
                    7,
                ),
                (
                    f"{name} "
                    f"tau={threshold:.3f} "
                    f"T={temperature:.3f}"
                ),
                fill="white",
            )

        group = auto_group_for_step(
            trajectory,
            step,
        )

        draw.text(
            (
                5,
                28,
            ),
            (
                f"step={step} "
                f"auto={group or '-'}"
            ),
            fill="white",
        )

        gif_frames.append(
            canvas
        )

    gif_frames[
        0
    ].save(
        output_path,
        save_all=True,
        append_images=gif_frames[
            1:
        ],
        duration=GIF_DURATION,
        loop=0,
    )


def save_anchor_contact_sheet(
    trajectory,
    output_path,
    max_each=8,
):
    """
    Visual audit of frames selected automatically as WEAK / STRONG, plus
    objective interaction/success frames.
    """

    frames = trajectory[
        "frames"
    ]

    groups = []

    for name, indices in [
        (
            "WEAK",
            trajectory[
                "weak_indices"
            ],
        ),
        (
            "STRONG",
            trajectory[
                "strong_indices"
            ],
        ),
        (
            "INTERACTION",
            trajectory[
                "interaction_indices"
            ],
        ),
        (
            "SUCCESS",
            trajectory[
                "success_indices"
            ],
        ),
    ]:

        if len(
            indices
        ) == 0:
            continue

        if len(
            indices
        ) > max_each:

            pick_positions = np.linspace(
                0,
                len(
                    indices
                ) - 1,
                max_each,
            ).astype(
                int
            )

            selected = indices[
                pick_positions
            ]

        else:
            selected = indices

        groups.append(
            (
                name,
                selected,
            )
        )

    if not groups:
        return

    W = TARGET_W
    H = TARGET_H
    label_h = 28

    max_cols = max(
        len(
            indices
        )
        for _, indices in groups
    )

    canvas = Image.new(
        "RGB",
        (
            W * max_cols,
            (
                H + label_h
            )
            * len(
                groups
            ),
        ),
        "black",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for row_idx, (
        group_name,
        indices,
    ) in enumerate(
        groups
    ):

        y = row_idx * (
            H + label_h
        )

        for col_idx, step in enumerate(
            indices
        ):

            x = col_idx * W

            canvas.paste(
                Image.fromarray(
                    frames[
                        step
                    ]
                ),
                (
                    x,
                    y + label_h,
                ),
            )

            draw.text(
                (
                    x + 5,
                    y + 6,
                ),
                (
                    f"{group_name} "
                    f"step={int(step)} "
                    f"raw_gap="
                    f"{trajectory['raw_contrast'][step]:.3f}"
                ),
                fill="white",
            )

    canvas.save(
        output_path
    )


def save_score_heatmap(
    rows,
    output_path,
):

    thresholds = sorted(
        {
            row[
                "threshold"
            ]
            for row in rows
        }
    )

    temperatures = sorted(
        {
            row[
                "temperature"
            ]
            for row in rows
        }
    )

    matrix = np.full(
        (
            len(
                temperatures
            ),
            len(
                thresholds
            ),
        ),
        np.nan,
        dtype=np.float32,
    )

    t_to_i = {
        value: i
        for i, value in enumerate(
            temperatures
        )
    }

    tau_to_i = {
        value: i
        for i, value in enumerate(
            thresholds
        )
    }

    for row in (
        rows
    ):

        matrix[
            t_to_i[
                row[
                    "temperature"
                ]
            ],
            tau_to_i[
                row[
                    "threshold"
                ]
            ],
        ] = row[
            "score"
        ]

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )

    ax.set_xlabel(
        "RELEVANCE_THRESHOLD (tau)"
    )

    ax.set_ylabel(
        "RELEVANCE_TEMPERATURE (T)"
    )

    ax.set_xticks(
        np.arange(
            len(
                thresholds
            )
        )
    )

    ax.set_xticklabels(
        [
            f"{value:.3f}"
            for value in thresholds
        ],
        rotation=60,
    )

    ax.set_yticks(
        np.arange(
            len(
                temperatures
            )
        )
    )

    ax.set_yticklabels(
        [
            f"{value:.3f}"
            for value in temperatures
        ]
    )

    ax.set_title(
        "Automatic trajectory parameter score (no manual stage labels)"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="score",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )


# ============================================================
# Best detailed outputs
# ============================================================

def save_best_details(
    trajectory,
    best_row,
    output_dir,
):

    threshold = best_row[
        "threshold"
    ]

    temperature = best_row[
        "temperature"
    ]

    relevance_patch = cosine_to_relevance(
        trajectory[
            "cosine"
        ],
        threshold,
        temperature,
    )

    maps = resize_relevance_maps(
        relevance_patch
    )

    np.save(
        output_dir
        / "best_relevance_patch.npy",
        relevance_patch,
    )

    np.save(
        output_dir
        / "best_relevance_maps.npy",
        maps,
    )

    frame_metrics = frame_metrics_from_patch(
        relevance_patch
    )

    rows = []

    weak_set = set(
        trajectory[
            "weak_indices"
        ].tolist()
    )

    strong_set = set(
        trajectory[
            "strong_indices"
        ].tolist()
    )

    interaction_set = set(
        trajectory[
            "interaction_indices"
        ].tolist()
    )

    success_set = set(
        trajectory[
            "success_indices"
        ].tolist()
    )

    for step in range(
        relevance_patch.shape[
            0
        ]
    ):

        labels = []

        if step in weak_set:
            labels.append(
                "weak"
            )

        if step in strong_set:
            labels.append(
                "strong"
            )

        if step in interaction_set:
            labels.append(
                "interaction"
            )

        if step in success_set:
            labels.append(
                "success"
            )

        row = {
            "step": step,

            "auto_group": "|".join(
                labels
            ),
        }

        for key, values in (
            frame_metrics.items()
        ):

            row[
                key
            ] = float(
                values[
                    step
                ]
            )

        rows.append(
            row
        )

    save_rows_csv(
        rows,
        output_dir
        / "best_frame_metrics.csv",
    )


# ============================================================
# Text summary
# ============================================================

def save_best_summary(
    trajectories,
    ranked_rows,
    output_path,
):

    best = ranked_rows[
        0
    ]

    with Path(
        output_path
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "REAL-APPROACH PARAMETER SEARCH - NO MANUAL MARK LABELS\n"
        )

        f.write(
            "======================================================\n\n"
        )

        f.write(
            f"BEST RELEVANCE_THRESHOLD = "
            f"{best['threshold']:.6f}\n"
        )

        f.write(
            f"BEST RELEVANCE_TEMPERATURE = "
            f"{best['temperature']:.6f}\n"
        )

        f.write(
            f"SCORE = "
            f"{best['score']:.3f}\n\n"
        )

        f.write(
            "Manual mark labels were NOT used.\n\n"
        )

        f.write(
            "Automatic references:\n"
        )

        f.write(
            "- weak frames: bottom raw semantic-strength quantile\n"
        )

        f.write(
            "- strong frames: top raw semantic-strength quantile\n"
        )

        f.write(
            "- interaction: pre attack/use observation when available\n"
        )

        f.write(
            "- success: reward/done/inventory event when available\n\n"
        )

        f.write(
            "Raw semantic strength = "
            f"{RAW_CONTRAST_WEIGHT:.2f}*rank(P95-P50) + "
            f"{RAW_P95_WEIGHT:.2f}*rank(P95)\n\n"
        )

        f.write(
            "Trajectories:\n"
        )

        for trajectory in (
            trajectories
        ):

            f.write(
                f"  {trajectory['path']}\n"
            )

            f.write(
                f"    task={trajectory['task_name']}\n"
            )

            f.write(
                f"    prompt={trajectory['prompt']}\n"
            )

            f.write(
                f"    weak_n="
                f"{len(trajectory['weak_indices'])}, "
                f"strong_n="
                f"{len(trajectory['strong_indices'])}\n"
            )

            f.write(
                f"    interaction_step="
                f"{trajectory['interaction_step']} "
                f"(fallback="
                f"{trajectory['interaction_is_fallback']})\n"
            )

            f.write(
                f"    success_step="
                f"{trajectory['success_step']}\n"
            )

        f.write(
            "\nTop settings:\n"
        )

        for rank, row in enumerate(
            ranked_rows[
                :20
            ],
            start=1,
        ):

            f.write(
                f"{rank:02d}. "
                f"score={row['score']:.3f} "
                f"tau={row['threshold']:.3f} "
                f"T={row['temperature']:.3f} "
                f"weak>.5="
                f"{row.get('weak_area05', np.nan):.3f} "
                f"strong>.5="
                f"{row.get('strong_area05', np.nan):.3f} "
                f"strong>.7="
                f"{row.get('strong_area07', np.nan):.3f}"
            )

            if "interaction_area05" in row:
                f.write(
                    f" interaction>.5="
                    f"{row['interaction_area05']:.3f}"
                )

            if "success_area05" in row:
                f.write(
                    f" success>.5="
                    f"{row['success_area05']:.3f}"
                )

            f.write(
                "\n"
            )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "- tau controls the relevance=0.5 raw-cosine boundary.\n"
        )

        f.write(
            "- T controls softness/high-confidence values and temporal "
            "sensitivity, but does not change which raw patches are >0.5.\n"
        )

        f.write(
            "- success suppression is diagnostic only and is not scored, "
            "because target persistence differs by task.\n"
        )

        f.write(
            "- inspect automatic_anchor_contact_sheet.png before trusting "
            "the ranking; it shows exactly which frames were selected "
            "without manual labels.\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "REAL APPROACH PARAMETER SEARCH - NO MANUAL MARKS"
    )

    print(
        "============================================================"
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    trajectories = []

    # --------------------------------------------------------
    # Load + automatic frame selection
    # --------------------------------------------------------

    for trajectory_path in (
        TRAJECTORY_PATHS
    ):

        trajectory = load_trajectory(
            trajectory_path
        )

        trajectories.append(
            trajectory
        )

        print(
            "\nLoaded:",
            trajectory[
                "path"
            ],
        )

        print(
            "  frames:",
            trajectory[
                "frames"
            ].shape,
        )

        print(
            "  cosine:",
            trajectory[
                "cosine"
            ].shape,
        )

        print(
            "  task:",
            trajectory[
                "task_name"
            ],
        )

        print(
            "  prompt:",
            trajectory[
                "prompt"
            ],
        )

        print(
            "  manual marks used: NO"
        )

        print(
            "  calibration frames:",
            len(
                trajectory[
                    "calibration_indices"
                ]
            ),
        )

        print(
            "  auto weak frames:",
            len(
                trajectory[
                    "weak_indices"
                ]
            ),
            "range:",
            (
                int(
                    trajectory[
                        "weak_indices"
                    ].min()
                ),
                int(
                    trajectory[
                        "weak_indices"
                    ].max()
                ),
            ),
        )

        print(
            "  auto strong frames:",
            len(
                trajectory[
                    "strong_indices"
                ]
            ),
            "range:",
            (
                int(
                    trajectory[
                        "strong_indices"
                    ].min()
                ),
                int(
                    trajectory[
                        "strong_indices"
                    ].max()
                ),
            ),
        )

        print(
            "  interaction step:",
            trajectory[
                "interaction_step"
            ],
            (
                "(fallback)"
                if trajectory[
                    "interaction_is_fallback"
                ]
                else ""
            ),
        )

        print(
            "  success step:",
            trajectory[
                "success_step"
            ],
        )

        # Audit files for this trajectory.
        safe_name = (
            trajectory[
                "name"
            ]
            .replace(
                "/",
                "_",
            )
        )

        save_auto_selection_csv(
            trajectory,
            OUTPUT_DIR
            / (
                f"auto_frame_selection_"
                f"{safe_name}.csv"
            ),
        )

        save_anchor_contact_sheet(
            trajectory,
            OUTPUT_DIR
            / (
                f"automatic_anchor_contact_sheet_"
                f"{safe_name}.png"
            ),
        )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    candidates = build_candidates()

    print(
        "\nCandidates:",
        len(
            candidates
        ),
    )

    rows = []

    for i, (
        threshold,
        temperature,
    ) in enumerate(
        candidates,
        start=1,
    ):

        row = evaluate_setting(
            trajectories,
            threshold,
            temperature,
        )

        rows.append(
            row
        )

        if (
            i == 1
            or i % 20 == 0
            or i == len(
                candidates
            )
        ):

            print(
                f"[{i:>3}/{len(candidates)}] "
                f"tau={threshold:.3f} "
                f"T={temperature:.3f} "
                f"score={row['score']:.2f}"
            )

    rows = sorted(
        rows,
        key=lambda row: -row[
            "score"
        ],
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "rank"
        ] = rank

    ranking_path = (
        OUTPUT_DIR
        / "parameter_ranking.csv"
    )

    save_rows_csv(
        rows,
        ranking_path,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_row = None

    for row in (
        rows
    ):

        if (
            abs(
                row[
                    "threshold"
                ]
                - BASELINE_THRESHOLD
            ) < 1e-8

            and abs(
                row[
                    "temperature"
                ]
                - BASELINE_TEMPERATURE
            ) < 1e-8
        ):

            baseline_row = row
            break

    # --------------------------------------------------------
    # Print top 20
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "TOP 20"
    )

    print(
        "============================================================"
    )

    print(
        f"{'rank':>4} "
        f"{'score':>7} "
        f"{'tau':>7} "
        f"{'T':>7} "
        f"{'weak>.5':>9} "
        f"{'strong>.5':>10} "
        f"{'strong>.7':>10} "
        f"{'inter>.5':>9} "
        f"{'success>.5':>11}"
    )

    for row in rows[
        :20
    ]:

        print(
            f"{row['rank']:>4} "
            f"{row['score']:>7.2f} "
            f"{row['threshold']:>7.3f} "
            f"{row['temperature']:>7.3f} "
            f"{row.get('weak_area05', np.nan):>9.3f} "
            f"{row.get('strong_area05', np.nan):>10.3f} "
            f"{row.get('strong_area07', np.nan):>10.3f} "
            f"{row.get('interaction_area05', np.nan):>9.3f} "
            f"{row.get('success_area05', np.nan):>11.3f}"
        )

    best = rows[
        0
    ]

    print(
        "\nBEST:"
    )

    print(
        "  RELEVANCE_THRESHOLD =",
        best[
            "threshold"
        ],
    )

    print(
        "  RELEVANCE_TEMPERATURE =",
        best[
            "temperature"
        ],
    )

    print(
        "  score =",
        f"{best['score']:.3f}",
    )

    if baseline_row is not None:

        print(
            "\nBASELINE:"
        )

        print(
            "  tau / T =",
            BASELINE_THRESHOLD,
            "/",
            BASELINE_TEMPERATURE,
        )

        print(
            "  rank =",
            baseline_row[
                "rank"
            ],
        )

        print(
            "  score =",
            f"{baseline_row['score']:.3f}",
        )

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------

    save_best_summary(
        trajectories,
        rows,
        OUTPUT_DIR
        / "best_settings.txt",
    )

    save_score_heatmap(
        rows,
        OUTPUT_DIR
        / "parameter_score_heatmap.png",
    )

    # --------------------------------------------------------
    # Visualize using the first trajectory.
    # --------------------------------------------------------

    visual_trajectory = trajectories[
        0
    ]

    for rank_i, row in enumerate(
        rows[
            :TOP_K_VISUALIZE
        ],
        start=1,
    ):

        gif_path = (
            OUTPUT_DIR
            / (
                f"top{rank_i:02d}_"
                f"tau{row['threshold']:.3f}_"
                f"T{row['temperature']:.3f}.gif"
            )
        )

        print(
            "Saving:",
            gif_path.name,
        )

        save_setting_gif(
            visual_trajectory,
            row[
                "threshold"
            ],
            row[
                "temperature"
            ],
            gif_path,
        )

    comparison_path = (
        OUTPUT_DIR
        / "top_settings_comparison.gif"
    )

    print(
        "Saving:",
        comparison_path.name,
    )

    save_top_comparison_gif(
        visual_trajectory,
        rows,
        comparison_path,
    )

    save_best_details(
        visual_trajectory,
        best,
        OUTPUT_DIR,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )

    print(
        "Ranking:",
        ranking_path,
    )

    print(
        "Best settings:",
        OUTPUT_DIR
        / "best_settings.txt",
    )

    print(
        "Automatic anchor sheet:",
        OUTPUT_DIR
        / (
            "automatic_anchor_contact_sheet_"
            f"{visual_trajectory['name']}.png"
        ),
    )

    print(
        "Comparison GIF:",
        comparison_path,
    )


if __name__ == "__main__":
    main()
