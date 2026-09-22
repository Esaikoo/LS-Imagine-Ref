#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
final_pre1m_audit.py
====================

Final audit before a 1M run for:

    Natural-Zoom map:
        Value + Temporal(L=1)
        -> raw cosine
        -> (cos + 1) / 2

    ScoreStorage progress:
        NACLIP + Temporal(L=1)
        -> raw cosine
        -> P85

It compares TWO real training replay episodes:

    1) original LS-Imagine / U-Net
    2) modified LS-Imagine / relevance-map

The script answers four questions:

A. Does our understanding of the REAL ScoreStorage path exactly reproduce
   the saved jumping_steps / accumulated_reward?

B. How strong was the heatmap signal that the world model actually received
   in U-Net and relevance-map training?

C. If sigmoid relevance is replaced by (cos+1)/2, does the heatmap still have
   enough spatial/temporal variation after the REAL path:

       heatmap -> *255 -> uint8 -> replay -> /255 -> ConvEncoder

   and is its reconstruction signal still large enough compared with RGB?

D. On the SAME historical accepted zoom events, what jumping_steps would
   NACLIP-P85 generate?

Important limitation
--------------------
Saved LS-Imagine replay RGB is 64x64. MineCLIP online originally saw 160x256.
Therefore recomputation from saved RGB is APPROXIMATE:

    saved 64x64 RGB -> resize 160x256 -> MineCLIP

Exact parts:
    - stored fields
    - ScoreStorage replay
    - stored heatmap statistics
    - wrapper uint8 quantization
    - world-model /255 input
    - saved accumulated_reward validation

Approximate parts:
    - Value+Temporal raw-shifted map recomputed from cached 64x64 RGB
    - NACLIP-P85 recomputed from cached 64x64 RGB

This script does NOT regenerate Natural-Zoom trigger/crop events.
Use your previous 160x256 online trajectory tests for that.

PyCharm:
    no argparse
    edit GLOBAL CONFIG and run directly.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from bisect import insort
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# 0. GLOBAL CONFIG
# =====================================================================

THIS_FILE = Path(__file__).resolve()

# Recommended location:
#   <LS-Imagine>/relevance_map/final_pre1m_audit.py
PROJECT_ROOT = THIS_FILE.parents[1]

# If needed:
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------
# Two REAL 1M-training replay episodes
# ---------------------------------------------------------------------

UNET_NPZ = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_train_trajectory"
    / "unetmap_trajectory.npz"
)

RELEVANT_NPZ = (
    PROJECT_ROOT
    / "relevance_map"
    / "real_train_trajectory"
    / "relevantmap_trajectory.npz"
)


# ---------------------------------------------------------------------
# MineCLIP
# ---------------------------------------------------------------------

MINECLIP_CKPT = (
    PROJECT_ROOT
    / "weights"
    / "mineclip_attn.pth"
)

TARGET_PROMPTS = [
    "Cut a tree"
]

PROGRESS_PERCENTILE = 85.0

MINECLIP_H = 160
MINECLIP_W = 256

NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True


# ---------------------------------------------------------------------
# Public LS-Imagine ScoreStorage
# ---------------------------------------------------------------------

MAX_STEPS = 1000
GAMMA = 0.997


# ---------------------------------------------------------------------
# OLD relevance-map calibration
#
# Used ONLY for diagnostic inverse estimation:
#
#   stored sigmoid heatmap
#      -> estimate old cosine
#      -> estimate (cos+1)/2
#
# This is NOT the preferred final estimate.
# ---------------------------------------------------------------------

OLD_RELEVANCE_THRESHOLD = 0.288
OLD_RELEVANCE_TEMPERATURE = 0.016

INVERSE_SIGMOID_EPS = (
    0.5
    / 255.0
)


# ---------------------------------------------------------------------
# Candidate recomputation
# ---------------------------------------------------------------------

RUN_MINECLIP_RECOMPUTE = True


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "relevance_map"
    / "final_pre1m_audit_outputs"
)

SAVE_EVENT_PANELS = True
SAVE_HEATMAP_EXAMPLES = True

MAX_EVENT_PANELS_PER_RUN = 60
MAX_HEATMAP_EXAMPLES_PER_RUN = 16

PRINT_EVERY_N = 50


# ---------------------------------------------------------------------
# Diagnostic thresholds
#
# These are warning rules, not scientific truth.
# ---------------------------------------------------------------------

# If the candidate heatmap has less than ~2 uint8 levels of spatial std,
# the signal is extremely weak after replay quantization.
WARN_MEDIAN_SPATIAL_STD_UINT8 = 2.0

# Global P95-P05 span.
WARN_GLOBAL_P95_P05_UINT8 = 4.0

# Candidate heatmap reconstruction residual vs RGB residual.
# <1% means that after learning a simple average heatmap, the heatmap
# reconstruction target may become tiny relative to RGB.
WARN_HEATMAP_TO_RGB_MSE_RATIO = 0.01

WARN_SCORE_RESOLVE_RATE = 0.80
WARN_JUMP_LE5_FRACTION = 0.10


RUNS = {
    "unet": UNET_NPZ,
    "relevant": RELEVANT_NPZ,
}


# =====================================================================
# 1. I/O HELPERS
# =====================================================================

def save_csv(
    path: Path,
    rows: List[Dict],
):
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


def load_npz(
    path: Path,
) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=True,
    ) as f:
        return {
            key: f[key]
            for key in f.files
        }


def scalar_series(
    x,
    n: int,
    default=np.nan,
) -> np.ndarray:
    if x is None:
        return np.full(
            n,
            default,
            dtype=np.float64,
        )

    a = np.asarray(x)

    if a.shape[0] != n:
        raise ValueError(
            f"Expected first dimension={n}, got {a.shape}"
        )

    if a.ndim == 1:
        return a

    return a.reshape(
        n,
        -1,
    )[:, 0]


def bool_series(
    x,
    n: int,
) -> np.ndarray:
    if x is None:
        return np.zeros(
            n,
            dtype=bool,
        )

    return scalar_series(
        x,
        n,
        default=0,
    ).astype(bool)


def normalize_rgb(
    x: np.ndarray,
) -> np.ndarray:
    a = np.asarray(x)

    if (
        a.ndim != 4
        or a.shape[-1] != 3
    ):
        raise ValueError(
            f"Unexpected RGB shape: {a.shape}"
        )

    if a.dtype != np.uint8:
        if np.nanmax(a) <= 1.5:
            a = a * 255.0

        a = np.clip(
            a,
            0,
            255,
        ).astype(np.uint8)

    return a


def normalize_heatmap(
    x: np.ndarray,
) -> np.ndarray:
    """
    Convert replay heatmap to:
        [T,H,W] float32 in [0,1]
    """

    a = np.asarray(x)

    while (
        a.ndim > 3
        and a.shape[-1] == 1
    ):
        a = a[..., 0]

    if a.ndim != 3:
        raise ValueError(
            f"Unexpected heatmap shape: {a.shape}"
        )

    a = a.astype(np.float32)

    if np.nanmax(a) > 1.5:
        a = a / 255.0

    return np.clip(
        a,
        0.0,
        1.0,
    )


def quantize_like_ls_wrapper(
    map01: np.ndarray,
) -> np.ndarray:
    """
    REAL path:

    ConcentrationWrapper:
        heatmap float [0,1]

    LSImagineWrapper:
        heatmap = clip(heatmap * 255).astype(uint8)

    WorldModel.preprocess:
        heatmap = Tensor(heatmap) / 255
    """

    q = np.clip(
        np.asarray(
            map01,
            dtype=np.float32,
        )
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return (
        q.astype(np.float32)
        / 255.0
    )


# =====================================================================
# 2. EXACT SCORESTORAGE REPLAY
# =====================================================================

class ScoreStorageReplay:
    """
    Public LS-Imagine behavior:

        self.data = sorted list of:
            (score_on_zoomed, step)

        resolve if:
            score_on_zoomed < current_score

        or:
            current_step - step >= max_steps
    """

    def __init__(
        self,
        max_steps: int,
    ):
        self.max_steps = int(max_steps)

        self.data: List[
            Tuple[float, int]
        ] = []

    def add(
        self,
        step: int,
        score_on_zoomed: float,
    ):
        insort(
            self.data,
            (
                float(score_on_zoomed),
                int(step),
            ),
        )

    def resolve(
        self,
        current_step: int,
        current_score: float,
    ) -> List[Dict]:
        removable = []
        results = []

        for i, (
            score_on_zoomed,
            source_step,
        ) in enumerate(self.data):

            crossed = (
                float(score_on_zoomed)
                < float(current_score)
            )

            timeout = (
                int(current_step)
                - int(source_step)
                >= self.max_steps
            )

            if crossed or timeout:
                removable.append(i)

                results.append(
                    {
                        "source_step":
                            int(source_step),

                        "resolve_step":
                            int(current_step),

                        "jumping_steps":
                            int(
                                current_step
                                - source_step
                            ),

                        "score_on_zoomed":
                            float(score_on_zoomed),

                        "current_score_at_resolution":
                            float(current_score),

                        "reason":
                            (
                                "score_cross"
                                if crossed
                                else "timeout"
                            ),
                    }
                )

        for i in reversed(removable):
            del self.data[i]

        return results


def replay_scorestorage(
    current_scores: np.ndarray,
    zoomed_scores: np.ndarray,
    is_zoomed: np.ndarray,
    is_last: np.ndarray,
    max_steps: int,
):
    """
    Mirrors the ordering in public tools.simulate():

    reset frame:
        if is_zoomed:
            add()

    every following env transition:
        if is_zoomed and not done:
            add()

        resolve using transition["score"]
    """

    n = len(current_scores)

    storage = ScoreStorageReplay(
        max_steps=max_steps
    )

    results = []

    if (
        n > 0
        and bool(is_zoomed[0])
    ):
        storage.add(
            step=0,
            score_on_zoomed=zoomed_scores[0],
        )

    for t in range(
        1,
        n,
    ):
        if (
            bool(is_zoomed[t])
            and not bool(is_last[t])
        ):
            storage.add(
                step=t,
                score_on_zoomed=zoomed_scores[t],
            )

        results.extend(
            storage.resolve(
                current_step=t,
                current_score=current_scores[t],
            )
        )

    return (
        results,
        list(storage.data),
    )


def replay_accumulated_reward(
    reward: np.ndarray,
    intrinsic: np.ndarray,
    source_step: int,
    resolve_step: int,
    gamma: float,
) -> float:
    """
    Public LS-Imagine:

        reward[source+1 : resolve_step]
        intrinsic[source+1 : resolve_step]

    resolve_step itself is excluded.
    """

    lo = int(source_step) + 1
    hi = int(resolve_step)

    r = np.asarray(
        reward[lo:hi],
        dtype=np.float64,
    )

    ir = np.asarray(
        intrinsic[lo:hi],
        dtype=np.float64,
    )

    if len(r) == 0:
        return 0.0

    gammas = np.power(
        float(gamma),
        np.arange(
            len(r),
            dtype=np.float64,
        ),
    )

    return float(
        np.sum(
            (
                r
                + ir
            )
            * gammas
        )
        / np.sum(gammas)
    )


# =====================================================================
# 3. WORLD-MODEL INPUT / LOSS PROXIES
# =====================================================================

def rgb_loss_proxy(
    rgb_uint8: np.ndarray,
) -> Dict:
    """
    Proxy for RGB reconstruction difficulty.

    The actual decoder learns, so this is not the real training loss.
    It answers:
        how much pixel residual remains if the decoder predicts only
        a simple average template?

    This gives a useful scale for comparing heatmap vs RGB supervision.
    """

    x = (
        rgb_uint8.astype(np.float64)
        / 255.0
    )

    temporal_mean_template = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    fixed_template_sum_mse = float(
        np.mean(
            np.sum(
                (
                    x
                    - temporal_mean_template
                )
                ** 2,
                axis=(
                    1,
                    2,
                    3,
                ),
            )
        )
    )

    scalar_frame_mean = np.mean(
        x,
        axis=(
            1,
            2,
            3,
        ),
        keepdims=True,
    )

    spatial_sum_mse = float(
        np.mean(
            np.sum(
                (
                    x
                    - scalar_frame_mean
                )
                ** 2,
                axis=(
                    1,
                    2,
                    3,
                ),
            )
        )
    )

    return {
        "rgb_mean":
            float(np.mean(x)),

        "rgb_global_std":
            float(np.std(x)),

        "rgb_fixed_template_sum_mse_per_frame":
            fixed_template_sum_mse,

        "rgb_frame_scalar_mean_sum_mse_per_frame":
            spatial_sum_mse,
    }


def heatmap_summary(
    name: str,
    map01: np.ndarray,
    rgb_proxy: Optional[Dict] = None,
) -> Dict:
    """
    map01:
        [T,H,W] in the exact [0,1] scale that WorldModel sees.
    """

    x = np.asarray(
        map01,
        dtype=np.float64,
    )

    if x.ndim != 3:
        raise ValueError(x.shape)

    T = x.shape[0]

    ff = x.reshape(
        T,
        -1,
    )

    flat = ff.reshape(-1)

    frame_std = np.std(
        ff,
        axis=1,
    )

    frame_range = (
        np.max(
            ff,
            axis=1,
        )
        -
        np.min(
            ff,
            axis=1,
        )
    )

    frame_p05 = np.percentile(
        ff,
        5,
        axis=1,
    )

    frame_p95 = np.percentile(
        ff,
        95,
        axis=1,
    )

    p05, p50, p95 = np.percentile(
        flat,
        [
            5,
            50,
            95,
        ],
    )

    # -------------------------------------------------------------
    # Decoder-ignore proxy 1:
    # always predict one temporal mean heatmap.
    # -------------------------------------------------------------

    temporal_mean_template = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    fixed_template_sum_mse = float(
        np.mean(
            np.sum(
                (
                    x
                    - temporal_mean_template
                )
                ** 2,
                axis=(
                    1,
                    2,
                ),
            )
        )
    )

    # -------------------------------------------------------------
    # Decoder-ignore proxy 2:
    # know the per-frame scalar mean but ignore spatial pattern.
    # -------------------------------------------------------------

    scalar_frame_mean = np.mean(
        x,
        axis=(
            1,
            2,
        ),
        keepdims=True,
    )

    spatial_pattern_sum_mse = float(
        np.mean(
            np.sum(
                (
                    x
                    - scalar_frame_mean
                )
                ** 2,
                axis=(
                    1,
                    2,
                ),
            )
        )
    )

    # ConvEncoder does:
    #     obs -= 0.5
    centered = (
        x
        - 0.5
    )

    unique_per_frame = np.asarray(
        [
            len(
                np.unique(
                    np.rint(
                        ff[t]
                        * 255.0
                    ).astype(np.uint8)
                )
            )
            for t in range(T)
        ],
        dtype=np.float64,
    )

    row = {
        "name":
            name,

        "frames":
            int(T),

        "min":
            float(np.min(flat)),

        "max":
            float(np.max(flat)),

        "mean":
            float(np.mean(flat)),

        "global_std":
            float(np.std(flat)),

        "global_p05":
            float(p05),

        "global_p50":
            float(p50),

        "global_p95":
            float(p95),

        "global_p95_minus_p05":
            float(
                p95
                - p05
            ),

        "global_p95_minus_p05_uint8_levels":
            float(
                (
                    p95
                    - p05
                )
                * 255.0
            ),

        "median_frame_spatial_std":
            float(
                np.median(frame_std)
            ),

        "median_frame_spatial_std_uint8_levels":
            float(
                np.median(frame_std)
                * 255.0
            ),

        "median_frame_range":
            float(
                np.median(frame_range)
            ),

        "median_frame_range_uint8_levels":
            float(
                np.median(frame_range)
                * 255.0
            ),

        "median_frame_p95_minus_p05":
            float(
                np.median(
                    frame_p95
                    - frame_p05
                )
            ),

        "median_frame_p95_minus_p05_uint8_levels":
            float(
                np.median(
                    frame_p95
                    - frame_p05
                )
                * 255.0
            ),

        "median_unique_uint8_levels_per_frame":
            float(
                np.median(
                    unique_per_frame
                )
            ),

        "centered_rms_after_encoder_minus_0.5":
            float(
                np.sqrt(
                    np.mean(
                        centered
                        ** 2
                    )
                )
            ),

        "fixed_temporal_mean_template_sum_mse_per_frame":
            fixed_template_sum_mse,

        "frame_scalar_mean_only_sum_mse_per_frame":
            spatial_pattern_sum_mse,
    }

    if rgb_proxy is not None:
        rgb_fixed = (
            rgb_proxy[
                "rgb_fixed_template_sum_mse_per_frame"
            ]
        )

        rgb_spatial = (
            rgb_proxy[
                "rgb_frame_scalar_mean_sum_mse_per_frame"
            ]
        )

        row[
            "fixed_template_heatmap_to_rgb_mse_ratio"
        ] = (
            fixed_template_sum_mse
            / rgb_fixed
            if rgb_fixed > 0
            else np.nan
        )

        row[
            "spatial_heatmap_to_rgb_mse_ratio"
        ] = (
            spatial_pattern_sum_mse
            / rgb_spatial
            if rgb_spatial > 0
            else np.nan
        )

    return row


def quantization_summary(
    pre_quant_map: np.ndarray,
) -> Dict:
    q = quantize_like_ls_wrapper(
        pre_quant_map
    )

    e = (
        np.asarray(
            pre_quant_map,
            dtype=np.float64,
        )
        - q.astype(np.float64)
    )

    return {
        "quantization_mae":
            float(
                np.mean(
                    np.abs(e)
                )
            ),

        "quantization_rmse":
            float(
                np.sqrt(
                    np.mean(
                        e ** 2
                    )
                )
            ),

        "quantization_max_abs":
            float(
                np.max(
                    np.abs(e)
                )
            ),
    }


# =====================================================================
# 4. RELEVANCE MAP -> APPROX RAW-SHIFTED
# =====================================================================

def inverse_old_relevance_map(
    stored_sigmoid_map: np.ndarray,
) -> np.ndarray:
    """
    Diagnostic only.

        relevance =
            sigmoid(
                (cosine - tau)
                / temperature
            )

    invert:
        cosine =
            tau
            + temperature * logit(relevance)

        raw_shifted =
            (cosine + 1) / 2
    """

    p = np.clip(
        np.asarray(
            stored_sigmoid_map,
            dtype=np.float64,
        ),
        INVERSE_SIGMOID_EPS,
        1.0
        - INVERSE_SIGMOID_EPS,
    )

    logit = np.log(
        p
        / (
            1.0
            - p
        )
    )

    cosine = (
        OLD_RELEVANCE_THRESHOLD
        + OLD_RELEVANCE_TEMPERATURE
        * logit
    )

    return np.clip(
        (
            cosine
            + 1.0
        )
        / 2.0,
        0.0,
        1.0,
    ).astype(np.float32)


# =====================================================================
# 5. MINECLIP / NACLIP
# =====================================================================

class CachedRGBMineCLIP:
    """
    Approximate because replay RGB is 64x64.

    Uses the same helper implementation as the previous experiments:
        relevance_map.relevance_variants
    """

    def __init__(
        self,
    ):
        import relevance_map.relevance_variants as rv

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
    def evaluate(
        self,
        rgb64: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
    ]:
        # ---------------------------------------------------------
        # Saved replay RGB is 64x64.
        # Approximate original MineCLIP input.
        # ---------------------------------------------------------

        frame = cv2.resize(
            rgb64.astype(np.uint8),
            (
                MINECLIP_W,
                MINECLIP_H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        x, last, gh, gw = (
            self.tester.encode_to_last(
                frame
            )
        )

        # ---------------------------------------------------------
        # Candidate map:
        # Value -> Temporal(L=1) -> cosine -> (cos+1)/2
        # ---------------------------------------------------------

        value = self.tester.raw_value_patch(
            x,
            last,
        )

        value_t = self.tester.temporal_l1(
            value
        )

        current_sim = self.tester.cosine(
            value_t,
            TARGET_PROMPTS,
        )[0]

        current_sim = (
            current_sim
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        current_raw = np.max(
            current_sim,
            axis=0,
        )

        shifted = np.clip(
            (
                current_raw
                + 1.0
            )
            / 2.0,
            0.0,
            1.0,
        ).reshape(
            int(gh),
            int(gw),
        )

        shifted_t = torch.from_numpy(
            shifted.astype(np.float32)
        )[None, None]

        shifted_64 = (
            F.interpolate(
                shifted_t,
                size=(
                    64,
                    64,
                ),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ---------------------------------------------------------
        # Candidate progress:
        # NACLIP -> Temporal(L=1) -> raw P85
        # ---------------------------------------------------------

        naclip = self.tester.naclip_patch(
            x,
            last,
            gh,
            gw,
        )

        naclip_t = self.tester.temporal_l1(
            naclip
        )

        naclip_sim = self.tester.cosine(
            naclip_t,
            TARGET_PROMPTS,
        )[0]

        naclip_sim = (
            naclip_sim
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        naclip_raw = np.max(
            naclip_sim,
            axis=0,
        )

        p85 = float(
            np.percentile(
                naclip_raw,
                PROGRESS_PERCENTILE,
            )
        )

        return (
            shifted_64,
            p85,
        )


# =====================================================================
# 6. PLOTS
# =====================================================================

def plot_distribution(
    path: Path,
    title: str,
    series: Dict[
        str,
        np.ndarray,
    ],
):
    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    for label, values in series.items():
        ax.hist(
            np.asarray(
                values
            ).reshape(-1),
            bins=80,
            density=True,
            histtype="step",
            label=label,
        )

    ax.set_xlabel(
        "value"
    )

    ax.set_ylabel(
        "density"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.2
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


def plot_score_timeline(
    path: Path,
    old_current: np.ndarray,
    old_zoomed: np.ndarray,
    is_zoomed: np.ndarray,
    candidate_current:
        Optional[
            np.ndarray
        ],
    candidate_zoomed:
        Optional[
            np.ndarray
        ],
):
    fig, ax = plt.subplots(
        figsize=(
            15,
            6,
        )
    )

    t = np.arange(
        len(
            old_current
        )
    )

    ax.plot(
        t,
        old_current,
        label="stored current score",
    )

    z = np.where(
        is_zoomed
    )[0]

    if len(z):
        ax.scatter(
            z,
            old_zoomed[z],
            marker="^",
            s=48,
            label="stored score_on_zoomed",
        )

    ax.set_xlabel(
        "step"
    )

    ax.set_ylabel(
        "stored ScoreStorage score"
    )

    ax.grid(
        alpha=0.2
    )

    lines1, labels1 = (
        ax.get_legend_handles_labels()
    )

    if candidate_current is not None:
        ax2 = ax.twinx()

        ax2.plot(
            t,
            candidate_current,
            linestyle="--",
            label="current NACLIP-P85",
        )

        valid = (
            z[
                np.isfinite(
                    candidate_zoomed[z]
                )
            ]
        )

        if len(valid):
            ax2.scatter(
                valid,
                candidate_zoomed[
                    valid
                ],
                marker="s",
                s=38,
                label="zoomed NACLIP-P85",
            )

        ax2.set_ylabel(
            "raw NACLIP-P85"
        )

        lines2, labels2 = (
            ax2.get_legend_handles_labels()
        )
    else:
        lines2 = []
        labels2 = []

    ax.legend(
        lines1
        + lines2,
        labels1
        + labels2,
        loc="best",
    )

    ax.set_title(
        "Stored ScoreStorage vs proposed NACLIP-P85"
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
# 7. ONE RUN
# =====================================================================

def audit_one_run(
    run_name: str,
    npz_path: Path,
    output_root: Path,
    mineclip:
        Optional[
            CachedRGBMineCLIP
        ],
) -> Dict:

    run_dir = (
        output_root
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ep = load_npz(
        npz_path
    )

    # -------------------------------------------------------------
    # Inventory
    # -------------------------------------------------------------

    inventory = []

    for key, value in ep.items():
        a = np.asarray(
            value
        )

        row = {
            "key":
                key,

            "shape":
                str(
                    a.shape
                ),

            "dtype":
                str(
                    a.dtype
                ),
        }

        if (
            a.size
            and np.issubdtype(
                a.dtype,
                np.number,
            )
        ):
            row.update(
                {
                    "min":
                        float(
                            np.nanmin(
                                a
                            )
                        ),

                    "max":
                        float(
                            np.nanmax(
                                a
                            )
                        ),

                    "mean":
                        float(
                            np.nanmean(
                                a
                            )
                        ),
                }
            )

        inventory.append(
            row
        )

    save_csv(
        run_dir
        / "00_inventory.csv",
        inventory,
    )

    # -------------------------------------------------------------
    # Core arrays
    # -------------------------------------------------------------

    rgb = normalize_rgb(
        ep[
            "image"
        ]
    )

    heatmap = normalize_heatmap(
        ep[
            "heatmap"
        ]
    )

    n = len(
        rgb
    )

    current_score = (
        scalar_series(
            ep.get(
                "score"
            ),
            n,
        )
        .astype(
            np.float64
        )
    )

    score_on_zoomed = (
        scalar_series(
            ep.get(
                "score_on_zoomed"
            ),
            n,
        )
        .astype(
            np.float64
        )
    )

    is_zoomed = bool_series(
        ep.get(
            "is_zoomed"
        ),
        n,
    )

    is_calculated = bool_series(
        ep.get(
            "is_calculated"
        ),
        n,
    )

    is_last = bool_series(
        ep.get(
            "is_last"
        ),
        n,
    )

    jump_flag = bool_series(
        ep.get(
            "jump"
        ),
        n,
    )

    saved_jumping = (
        scalar_series(
            ep.get(
                "jumping_steps"
            ),
            n,
        )
        .astype(
            np.float64
        )
    )

    reward = (
        scalar_series(
            ep.get(
                "reward"
            ),
            n,
            default=0.0,
        )
        .astype(
            np.float64
        )
    )

    intrinsic = (
        scalar_series(
            ep.get(
                "intrinsic"
            ),
            n,
            default=0.0,
        )
        .astype(
            np.float64
        )
    )

    saved_accumulated = (
        scalar_series(
            ep.get(
                "accumulated_reward"
            ),
            n,
        )
        .astype(
            np.float64
        )
    )

    # -------------------------------------------------------------
    # Exact ScoreStorage replay
    # -------------------------------------------------------------

    exact_rows, exact_pending = (
        replay_scorestorage(
            current_scores=(
                current_score
            ),

            zoomed_scores=(
                score_on_zoomed
            ),

            is_zoomed=(
                is_zoomed
            ),

            is_last=(
                is_last
            ),

            max_steps=(
                MAX_STEPS
            ),
        )
    )

    exact_audit = []

    for row in exact_rows:
        ss = row[
            "source_step"
        ]

        out = dict(
            row
        )

        out[
            "saved_jumping_steps"
        ] = float(
            saved_jumping[
                ss
            ]
        )

        out[
            "saved_is_calculated"
        ] = int(
            bool(
                is_calculated[
                    ss
                ]
            )
        )

        out[
            "jumping_steps_match"
        ] = int(
            np.isfinite(
                saved_jumping[
                    ss
                ]
            )
            and abs(
                saved_jumping[
                    ss
                ]
                - row[
                    "jumping_steps"
                ]
            )
            < 1e-9
        )

        replay_acc = (
            replay_accumulated_reward(
                reward=(
                    reward
                ),

                intrinsic=(
                    intrinsic
                ),

                source_step=(
                    ss
                ),

                resolve_step=(
                    row[
                        "resolve_step"
                    ]
                ),

                gamma=(
                    GAMMA
                ),
            )
        )

        out[
            "replayed_accumulated_reward"
        ] = float(
            replay_acc
        )

        out[
            "saved_accumulated_reward"
        ] = float(
            saved_accumulated[
                ss
            ]
        )

        out[
            "accumulated_reward_abs_error"
        ] = (
            float(
                abs(
                    replay_acc
                    - saved_accumulated[
                        ss
                    ]
                )
            )
            if np.isfinite(
                saved_accumulated[
                    ss
                ]
            )
            else np.nan
        )

        exact_audit.append(
            out
        )

    save_csv(
        run_dir
        / "01_exact_scorestorage_replay.csv",
        exact_audit,
    )

    # -------------------------------------------------------------
    # Stored heatmap world-model input
    # -------------------------------------------------------------

    rgb_proxy = (
        rgb_loss_proxy(
            rgb
        )
    )

    stored_world_map = (
        quantize_like_ls_wrapper(
            heatmap
        )
    )

    stored_map_summary = (
        heatmap_summary(
            name=(
                f"{run_name}_stored_current_heatmap"
            ),

            map01=(
                stored_world_map
            ),

            rgb_proxy=(
                rgb_proxy
            ),
        )
    )

    # -------------------------------------------------------------
    # Stored zoomed heatmap
    # -------------------------------------------------------------

    zoomed_map_summary = None

    if (
        "heatmap_on_zoomed"
        in ep
        and np.any(
            is_zoomed
        )
    ):
        zoomed_maps_all = (
            normalize_heatmap(
                ep[
                    "heatmap_on_zoomed"
                ]
            )
        )

        zoomed_maps = (
            zoomed_maps_all[
                is_zoomed
            ]
        )

        zoomed_map_summary = (
            heatmap_summary(
                name=(
                    f"{run_name}_stored_zoomed_heatmap"
                ),

                map01=(
                    zoomed_maps
                ),

                rgb_proxy=None,
            )
        )

    # -------------------------------------------------------------
    # Inverse relevance estimate
    # -------------------------------------------------------------

    inverse_current = None
    inverse_zoomed = None
    inverse_summary = None
    inverse_zoomed_summary = None

    if run_name == "relevant":
        inverse_pre_quant = (
            inverse_old_relevance_map(
                stored_world_map
            )
        )

        inverse_current = (
            quantize_like_ls_wrapper(
                inverse_pre_quant
            )
        )

        inverse_summary = (
            heatmap_summary(
                name=(
                    "relevant_raw_shifted_inverse_current_ESTIMATE"
                ),

                map01=(
                    inverse_current
                ),

                rgb_proxy=(
                    rgb_proxy
                ),
            )
        )

        inverse_summary.update(
            quantization_summary(
                inverse_pre_quant
            )
        )

        if (
            "heatmap_on_zoomed"
            in ep
            and np.any(
                is_zoomed
            )
        ):
            stored_zoomed_all = (
                normalize_heatmap(
                    ep[
                        "heatmap_on_zoomed"
                    ]
                )
            )

            inv_zoom_pre = (
                inverse_old_relevance_map(
                    stored_zoomed_all[
                        is_zoomed
                    ]
                )
            )

            inverse_zoomed = (
                quantize_like_ls_wrapper(
                    inv_zoom_pre
                )
            )

            inverse_zoomed_summary = (
                heatmap_summary(
                    name=(
                        "relevant_raw_shifted_inverse_zoomed_ESTIMATE"
                    ),

                    map01=(
                        inverse_zoomed
                    ),

                    rgb_proxy=None,
                )
            )

    # -------------------------------------------------------------
    # Candidate MineCLIP recompute
    # -------------------------------------------------------------

    recomputed_map = None
    recomputed_summary = None

    candidate_current_p85 = None
    candidate_zoomed_p85 = None

    candidate_rows = []
    candidate_pending = []

    mineclip_error = None

    if mineclip is not None:
        try:
            recomputed_map = np.zeros(
                (
                    n,
                    64,
                    64,
                ),
                dtype=np.float32,
            )

            candidate_current_p85 = np.full(
                n,
                np.nan,
                dtype=np.float64,
            )

            candidate_zoomed_p85 = np.full(
                n,
                np.nan,
                dtype=np.float64,
            )

            zoomed_rgb = None

            if (
                "zoomed_image"
                in ep
            ):
                zoomed_rgb = normalize_rgb(
                    ep[
                        "zoomed_image"
                    ]
                )

            for t in range(
                n
            ):
                (
                    raw_map64,
                    p85,
                ) = mineclip.evaluate(
                    rgb[
                        t
                    ]
                )

                recomputed_map[
                    t
                ] = raw_map64

                candidate_current_p85[
                    t
                ] = p85

                if (
                    bool(
                        is_zoomed[
                            t
                        ]
                    )
                    and zoomed_rgb
                    is not None
                    and np.any(
                        zoomed_rgb[
                            t
                        ]
                    )
                ):
                    _, zp85 = (
                        mineclip.evaluate(
                            zoomed_rgb[
                                t
                            ]
                        )
                    )

                    candidate_zoomed_p85[
                        t
                    ] = zp85

                if (
                    t == 0
                    or (
                        t
                        + 1
                    )
                    % PRINT_EVERY_N
                    == 0
                    or (
                        t
                        + 1
                    )
                    == n
                ):
                    print(
                        f"[{run_name} "
                        f"{t+1:4d}/{n:4d}] "
                        f"NACLIP-P85="
                        f"{candidate_current_p85[t]:.4f}"
                    )

            recomputed_world = (
                quantize_like_ls_wrapper(
                    recomputed_map
                )
            )

            recomputed_summary = (
                heatmap_summary(
                    name=(
                        f"{run_name}_raw_shifted_cached64_RGB_APPROX"
                    ),

                    map01=(
                        recomputed_world
                    ),

                    rgb_proxy=(
                        rgb_proxy
                    ),
                )
            )

            recomputed_summary.update(
                quantization_summary(
                    recomputed_map
                )
            )

            # -----------------------------------------------------
            # Candidate ScoreStorage
            # -----------------------------------------------------

            valid_zoom = (
                is_zoomed
                & np.isfinite(
                    candidate_zoomed_p85
                )
            )

            (
                candidate_rows,
                candidate_pending,
            ) = replay_scorestorage(
                current_scores=(
                    candidate_current_p85
                ),

                zoomed_scores=(
                    np.where(
                        valid_zoom,
                        candidate_zoomed_p85,
                        np.inf,
                    )
                ),

                is_zoomed=(
                    valid_zoom
                ),

                is_last=(
                    is_last
                ),

                max_steps=(
                    MAX_STEPS
                ),
            )

            for row in candidate_rows:
                ss = row[
                    "source_step"
                ]

                row[
                    "historical_jumping_steps"
                ] = float(
                    saved_jumping[
                        ss
                    ]
                )

                row[
                    "candidate_minus_historical_J"
                ] = float(
                    row[
                        "jumping_steps"
                    ]
                    - saved_jumping[
                        ss
                    ]
                )

                row[
                    "current_P85_at_source"
                ] = float(
                    candidate_current_p85[
                        ss
                    ]
                )

                row[
                    "zoomed_P85"
                ] = float(
                    candidate_zoomed_p85[
                        ss
                    ]
                )

                row[
                    "historical_current_score_at_source"
                ] = float(
                    current_score[
                        ss
                    ]
                )

                row[
                    "historical_score_on_zoomed"
                ] = float(
                    score_on_zoomed[
                        ss
                    ]
                )

            save_csv(
                run_dir
                / "02_candidate_naclip_p85_scorestorage.csv",
                candidate_rows,
            )

        except Exception as e:
            mineclip_error = repr(
                e
            )

            print(
                f"[WARN] {run_name}: "
                f"MineCLIP/NACLIP recompute failed:"
            )

            print(
                mineclip_error
            )

    # -------------------------------------------------------------
    # Heatmap summary CSV
    # -------------------------------------------------------------

    hm_rows = [
        stored_map_summary
    ]

    if zoomed_map_summary is not None:
        hm_rows.append(
            zoomed_map_summary
        )

    if inverse_summary is not None:
        hm_rows.append(
            inverse_summary
        )

    if inverse_zoomed_summary is not None:
        hm_rows.append(
            inverse_zoomed_summary
        )

    if recomputed_summary is not None:
        hm_rows.append(
            recomputed_summary
        )

    save_csv(
        run_dir
        / "03_worldmodel_heatmap_summary.csv",
        hm_rows,
    )

    # -------------------------------------------------------------
    # Event summary
    # -------------------------------------------------------------

    zoom_steps = np.where(
        is_zoomed
    )[0]

    historical_J = (
        saved_jumping[
            zoom_steps
        ]
        .astype(
            np.float64
        )
    )

    exact_match_count = int(
        sum(
            row[
                "jumping_steps_match"
            ]
            for row
            in exact_audit
        )
    )

    max_acc_error = (
        float(
            np.max(
                [
                    row[
                        "accumulated_reward_abs_error"
                    ]
                    for row
                    in exact_audit
                ]
            )
        )
        if exact_audit
        else np.nan
    )

    event_summary = {
        "run":
            run_name,

        "frames":
            int(n),

        "is_first_steps":
            ",".join(
                str(
                    int(x)
                )
                for x
                in np.where(
                    bool_series(
                        ep.get(
                            "is_first"
                        ),
                        n,
                    )
                )[0]
            ),

        "is_last_steps":
            ",".join(
                str(
                    int(x)
                )
                for x
                in np.where(
                    is_last
                )[0]
            ),

        "is_zoomed_count":
            int(
                len(
                    zoom_steps
                )
            ),

        "zoom_steps":
            ",".join(
                str(
                    int(x)
                )
                for x
                in zoom_steps
            ),

        "jump_flag_count":
            int(
                np.sum(
                    jump_flag
                )
            ),

        "is_calculated_count":
            int(
                np.sum(
                    is_calculated
                )
            ),

        "historical_J_mean":
            (
                float(
                    np.mean(
                        historical_J
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "historical_J_median":
            (
                float(
                    np.median(
                        historical_J
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "historical_J_min":
            (
                float(
                    np.min(
                        historical_J
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "historical_J_max":
            (
                float(
                    np.max(
                        historical_J
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "historical_J_le5_fraction":
            (
                float(
                    np.mean(
                        historical_J
                        <= 5
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "historical_J_le20_fraction":
            (
                float(
                    np.mean(
                        historical_J
                        <= 20
                    )
                )
                if len(
                    historical_J
                )
                else np.nan
            ),

        "exact_replay_resolved":
            int(
                len(
                    exact_rows
                )
            ),

        "exact_replay_match_count":
            exact_match_count,

        "exact_replay_pending":
            int(
                len(
                    exact_pending
                )
            ),

        "max_accumulated_reward_abs_error":
            max_acc_error,

        "mineclip_error":
            mineclip_error,
    }

    # -------------------------------------------------------------
    # Candidate score summary
    # -------------------------------------------------------------

    candidate_score_summary = None

    if candidate_current_p85 is not None:
        valid_zoom = (
            is_zoomed
            & np.isfinite(
                candidate_zoomed_p85
            )
        )

        valid_count = int(
            np.sum(
                valid_zoom
            )
        )

        crossed_rows = [
            row
            for row
            in candidate_rows
            if row[
                "reason"
            ]
            == "score_cross"
        ]

        candidate_J = np.asarray(
            [
                row[
                    "jumping_steps"
                ]
                for row
                in crossed_rows
            ],
            dtype=np.float64,
        )

        candidate_score_summary = {
            "run":
                run_name,

            "historical_zoom_events_with_valid_zoomed_RGB":
                valid_count,

            "candidate_resolved_count":
                int(
                    len(
                        candidate_rows
                    )
                ),

            "candidate_score_cross_count":
                int(
                    len(
                        crossed_rows
                    )
                ),

            "candidate_pending_count":
                int(
                    len(
                        candidate_pending
                    )
                ),

            "candidate_resolve_rate":
                (
                    float(
                        len(
                            candidate_rows
                        )
                        / valid_count
                    )
                    if valid_count
                    else np.nan
                ),

            "candidate_J_mean":
                (
                    float(
                        np.mean(
                            candidate_J
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),

            "candidate_J_median":
                (
                    float(
                        np.median(
                            candidate_J
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),

            "candidate_J_min":
                (
                    float(
                        np.min(
                            candidate_J
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),

            "candidate_J_max":
                (
                    float(
                        np.max(
                            candidate_J
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),

            "candidate_J_le5_fraction":
                (
                    float(
                        np.mean(
                            candidate_J
                            <= 5
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),

            "candidate_J_le20_fraction":
                (
                    float(
                        np.mean(
                            candidate_J
                            <= 20
                        )
                    )
                    if len(
                        candidate_J
                    )
                    else np.nan
                ),
        }

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------

    maps_for_plot = {
        "stored current heatmap":
            stored_world_map
    }

    if inverse_current is not None:
        maps_for_plot[
            "raw-shifted inverse estimate"
        ] = inverse_current

    if recomputed_map is not None:
        maps_for_plot[
            "raw-shifted cached-RGB approx"
        ] = (
            quantize_like_ls_wrapper(
                recomputed_map
            )
        )

    plot_distribution(
        path=(
            run_dir
            / "04_heatmap_distribution.png"
        ),

        title=(
            f"{run_name}: "
            "world-model heatmap input"
        ),

        series=(
            maps_for_plot
        ),
    )

    plot_score_timeline(
        path=(
            run_dir
            / "05_score_timeline.png"
        ),

        old_current=(
            current_score
        ),

        old_zoomed=(
            score_on_zoomed
        ),

        is_zoomed=(
            is_zoomed
        ),

        candidate_current=(
            candidate_current_p85
        ),

        candidate_zoomed=(
            candidate_zoomed_p85
        ),
    )

    # -------------------------------------------------------------
    # Heatmap examples
    # -------------------------------------------------------------

    if SAVE_HEATMAP_EXAMPLES:
        example_dir = (
            run_dir
            / "heatmap_examples"
        )

        example_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = min(
            MAX_HEATMAP_EXAMPLES_PER_RUN,
            n,
        )

        frame_ids = np.linspace(
            0,
            n
            - 1,
            count,
            dtype=int,
        )

        for t in frame_ids:
            cols = 2

            if inverse_current is not None:
                cols += 1

            if recomputed_map is not None:
                cols += 1

            fig, ax = plt.subplots(
                1,
                cols,
                figsize=(
                    4.2
                    * cols,
                    4.2,
                ),
                squeeze=False,
            )

            ax = ax[
                0
            ]

            j = 0

            ax[j].imshow(
                rgb[
                    t
                ]
            )

            ax[j].set_title(
                f"RGB t={t}"
            )

            j += 1

            ax[j].imshow(
                stored_world_map[
                    t
                ],
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

            ax[j].set_title(
                "stored heatmap\n"
                f"{stored_world_map[t].min():.3f}"
                ".."
                f"{stored_world_map[t].max():.3f}"
            )

            j += 1

            if inverse_current is not None:
                ax[j].imshow(
                    inverse_current[
                        t
                    ],
                    cmap="jet",
                    vmin=0.0,
                    vmax=1.0,
                )

                ax[j].set_title(
                    "raw-shifted inverse EST\n"
                    f"{inverse_current[t].min():.3f}"
                    ".."
                    f"{inverse_current[t].max():.3f}"
                )

                j += 1

            if recomputed_map is not None:
                rm = (
                    quantize_like_ls_wrapper(
                        recomputed_map[
                            t:
                            t
                            + 1
                        ]
                    )[
                        0
                    ]
                )

                ax[j].imshow(
                    rm,
                    cmap="jet",
                    vmin=0.0,
                    vmax=1.0,
                )

                ax[j].set_title(
                    "raw-shifted cached RGB APPROX\n"
                    f"{rm.min():.3f}"
                    ".."
                    f"{rm.max():.3f}"
                )

            for aa in ax:
                aa.axis(
                    "off"
                )

            plt.tight_layout()

            fig.savefig(
                example_dir
                / f"frame_{t:06d}.png",
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    # -------------------------------------------------------------
    # Zoom event panels
    # -------------------------------------------------------------

    if (
        SAVE_EVENT_PANELS
        and "zoomed_image"
        in ep
    ):
        event_dir = (
            run_dir
            / "zoom_events"
        )

        event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        zoomed_rgb = normalize_rgb(
            ep[
                "zoomed_image"
            ]
        )

        exact_by_source = {
            row[
                "source_step"
            ]:
                row
            for row
            in exact_audit
        }

        candidate_by_source = {
            row[
                "source_step"
            ]:
                row
            for row
            in candidate_rows
        }

        for ss in (
            np.where(
                is_zoomed
            )[0][
                :
                MAX_EVENT_PANELS_PER_RUN
            ]
        ):
            fig, ax = plt.subplots(
                1,
                5,
                figsize=(
                    21,
                    4.5,
                ),
                squeeze=False,
            )

            ax = ax[
                0
            ]

            ax[0].imshow(
                rgb[
                    ss
                ]
            )

            ax[0].set_title(
                f"source t={ss}"
            )

            ax[1].imshow(
                stored_world_map[
                    ss
                ],
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

            ax[1].set_title(
                "stored heatmap"
            )

            ax[2].imshow(
                zoomed_rgb[
                    ss
                ]
            )

            ax[2].set_title(
                "zoomed RGB"
            )

            hist = exact_by_source.get(
                int(
                    ss
                )
            )

            cand = candidate_by_source.get(
                int(
                    ss
                )
            )

            if hist is not None:
                hk = int(
                    hist[
                        "resolve_step"
                    ]
                )

                ax[3].imshow(
                    rgb[
                        hk
                    ]
                )

                ax[3].set_title(
                    "historical crossing\n"
                    f"k={hk}, "
                    f"J={hist['jumping_steps']}"
                )
            else:
                ax[3].imshow(
                    np.zeros_like(
                        rgb[
                            ss
                        ]
                    )
                )

                ax[3].set_title(
                    "historical unresolved"
                )

            if cand is not None:
                ck = int(
                    cand[
                        "resolve_step"
                    ]
                )

                ax[4].imshow(
                    rgb[
                        ck
                    ]
                )

                ax[4].set_title(
                    "NACLIP-P85 crossing\n"
                    f"k={ck}, "
                    f"J={cand['jumping_steps']}"
                )
            else:
                ax[4].imshow(
                    np.zeros_like(
                        rgb[
                            ss
                        ]
                    )
                )

                ax[4].set_title(
                    "NACLIP-P85 unresolved"
                )

            for aa in ax:
                aa.axis(
                    "off"
                )

            plt.tight_layout()

            fig.savefig(
                event_dir
                / f"zoom_t{int(ss):06d}.png",
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    # -------------------------------------------------------------
    # Per-run report
    # -------------------------------------------------------------

    per_run_report = {
        "event_summary":
            event_summary,

        "stored_current_heatmap":
            stored_map_summary,

        "stored_zoomed_heatmap":
            zoomed_map_summary,

        "inverse_candidate_current":
            inverse_summary,

        "inverse_candidate_zoomed":
            inverse_zoomed_summary,

        "recomputed_candidate_heatmap":
            recomputed_summary,

        "candidate_score_summary":
            candidate_score_summary,

        "mineclip_error":
            mineclip_error,
    }

    (
        run_dir
        / "06_run_summary.json"
    ).write_text(
        json.dumps(
            per_run_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "run":
            run_name,

        "n":
            n,

        "rgb":
            rgb,

        "stored_map":
            stored_world_map,

        "event_summary":
            event_summary,

        "stored_map_summary":
            stored_map_summary,

        "zoomed_map_summary":
            zoomed_map_summary,

        "inverse_summary":
            inverse_summary,

        "inverse_zoomed_summary":
            inverse_zoomed_summary,

        "recomputed_summary":
            recomputed_summary,

        "candidate_score_summary":
            candidate_score_summary,

        "exact_rows":
            exact_audit,

        "candidate_rows":
            candidate_rows,

        "mineclip_error":
            mineclip_error,
    }


# =====================================================================
# 8. FINAL DECISION REPORT
# =====================================================================

def main():
    base = (
        OUTPUT_ROOT
        / "comparison"
    )

    outdir = base

    i = 1

    while outdir.exists():
        outdir = Path(
            str(
                base
            )
            + f"_{i}"
        )

        i += 1

    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "="
        * 100
    )

    print(
        "FINAL PRE-1M AUDIT"
    )

    print(
        "="
        * 100
    )

    print(
        "U-Net:",
        UNET_NPZ,
    )

    print(
        "Relevant-map:",
        RELEVANT_NPZ,
    )

    print(
        "Output:",
        outdir,
    )

    # -------------------------------------------------------------
    # MineCLIP initialization
    # -------------------------------------------------------------

    mineclip = None
    mineclip_init_error = None

    if RUN_MINECLIP_RECOMPUTE:
        try:
            mineclip = (
                CachedRGBMineCLIP()
            )

        except Exception as e:
            mineclip_init_error = repr(
                e
            )

            print(
                "[WARN] MineCLIP initialization failed:"
            )

            print(
                mineclip_init_error
            )

    # -------------------------------------------------------------
    # Audit both
    # -------------------------------------------------------------

    results = {}

    for run_name, path in RUNS.items():
        print()
        print(
            "-"
            * 100
        )

        print(
            "AUDIT:",
            run_name,
        )

        print(
            "-"
            * 100
        )

        results[
            run_name
        ] = audit_one_run(
            run_name=(
                run_name
            ),

            npz_path=(
                path
            ),

            output_root=(
                outdir
            ),

            mineclip=(
                mineclip
            ),
        )

    # -------------------------------------------------------------
    # Cross-run CSVs
    # -------------------------------------------------------------

    scorestorage_rows = [
        results[
            "unet"
        ][
            "event_summary"
        ],

        results[
            "relevant"
        ][
            "event_summary"
        ],
    ]

    save_csv(
        outdir
        / "10_historical_scorestorage_comparison.csv",
        scorestorage_rows,
    )

    heatmap_rows = []

    for name in [
        "unet",
        "relevant",
    ]:
        for source_key, source_type in [
            (
                "stored_map_summary",
                "stored_current",
            ),
            (
                "zoomed_map_summary",
                "stored_zoomed",
            ),
            (
                "inverse_summary",
                "raw_shifted_inverse_estimate",
            ),
            (
                "inverse_zoomed_summary",
                "raw_shifted_zoomed_inverse_estimate",
            ),
            (
                "recomputed_summary",
                "raw_shifted_cachedRGB_approx",
            ),
        ]:
            value = (
                results[
                    name
                ][
                    source_key
                ]
            )

            if value is not None:
                row = dict(
                    value
                )

                row[
                    "run"
                ] = name

                row[
                    "source_type"
                ] = source_type

                heatmap_rows.append(
                    row
                )

    save_csv(
        outdir
        / "11_worldmodel_heatmap_comparison.csv",
        heatmap_rows,
    )

    candidate_score_rows = []

    for name in [
        "unet",
        "relevant",
    ]:
        row = (
            results[
                name
            ][
                "candidate_score_summary"
            ]
        )

        if row is not None:
            candidate_score_rows.append(
                row
            )

    save_csv(
        outdir
        / "12_naclip_p85_candidate_comparison.csv",
        candidate_score_rows,
    )

    # -------------------------------------------------------------
    # Cross-run stored heatmap distribution
    # -------------------------------------------------------------

    plot_distribution(
        path=(
            outdir
            / "13_stored_heatmap_comparison.png"
        ),

        title=(
            "Actual heatmaps used by the two training runs"
        ),

        series={
            "U-Net stored heatmap":
                results[
                    "unet"
                ][
                    "stored_map"
                ],

            "Relevant stored heatmap":
                results[
                    "relevant"
                ][
                    "stored_map"
                ],
        },
    )

    # -------------------------------------------------------------
    # Automatic decision logic
    # -------------------------------------------------------------

    findings = []
    warnings = []
    reds = []

    # Exact ScoreStorage.
    for name in [
        "unet",
        "relevant",
    ]:
        e = (
            results[
                name
            ][
                "event_summary"
            ]
        )

        findings.append(
            f"{name}: exact ScoreStorage "
            f"{e['exact_replay_match_count']}/"
            f"{e['exact_replay_resolved']} "
            f"jump targets reproduced; "
            f"max accumulated_reward error="
            f"{e['max_accumulated_reward_abs_error']}."
        )

        if (
            e[
                "exact_replay_match_count"
            ]
            != e[
                "exact_replay_resolved"
            ]
        ):
            reds.append(
                f"{name}: ScoreStorage replay does not "
                "match all stored jumping_steps."
            )

    # Historical distributions.
    u = (
        results[
            "unet"
        ][
            "event_summary"
        ]
    )

    r = (
        results[
            "relevant"
        ][
            "event_summary"
        ]
    )

    findings.append(
        "The two npz files are different environment episodes, "
        "so zoom-count differences are descriptive, not a direct winner/loser comparison."
    )

    findings.append(
        f"U-Net historical: "
        f"{u['is_zoomed_count']} zooms, "
        f"median J={u['historical_J_median']}, "
        f"J<=5={u['historical_J_le5_fraction']:.3f}."
    )

    findings.append(
        f"Relevant historical: "
        f"{r['is_zoomed_count']} zooms, "
        f"median J={r['historical_J_median']}, "
        f"J<=5={r['historical_J_le5_fraction']:.3f}."
    )

    # Candidate raw map.
    raw_summary = (
        results[
            "relevant"
        ][
            "recomputed_summary"
        ]
    )

    raw_source = (
        "cached-RGB MineCLIP recomputation"
    )

    if raw_summary is None:
        raw_summary = (
            results[
                "relevant"
            ][
                "inverse_summary"
            ]
        )

        raw_source = (
            "inverse-sigmoid estimate"
        )

    if raw_summary is not None:
        spatial_levels = (
            raw_summary[
                "median_frame_spatial_std_uint8_levels"
            ]
        )

        span_levels = (
            raw_summary[
                "global_p95_minus_p05_uint8_levels"
            ]
        )

        mse_ratio = (
            raw_summary.get(
                "spatial_heatmap_to_rgb_mse_ratio",
                np.nan,
            )
        )

        findings.append(
            f"Proposed raw-shifted heatmap ({raw_source}): "
            f"median spatial std={spatial_levels:.3f} uint8 levels; "
            f"global P95-P05={span_levels:.3f} uint8 levels; "
            f"spatial heatmap/RGB loss-proxy ratio={mse_ratio}."
        )

        if (
            spatial_levels
            < WARN_MEDIAN_SPATIAL_STD_UINT8
        ):
            reds.append(
                "Raw-shifted heatmap has extremely weak "
                "per-frame spatial variation after uint8 replay."
            )

        elif (
            spatial_levels
            < 4.0
        ):
            warnings.append(
                "Raw-shifted heatmap spatial variation is usable but narrow."
            )

        if (
            span_levels
            < WARN_GLOBAL_P95_P05_UINT8
        ):
            reds.append(
                "Raw-shifted global dynamic range is too narrow."
            )

        if (
            np.isfinite(
                mse_ratio
            )
            and mse_ratio
            < WARN_HEATMAP_TO_RGB_MSE_RATIO
        ):
            warnings.append(
                "Raw-shifted heatmap reconstruction residual is "
                "<1% of RGB proxy. Decoder may learn a near-constant "
                "heatmap cheaply, weakening direct heatmap supervision."
            )

    else:
        warnings.append(
            "No raw-shifted heatmap candidate could be estimated."
        )

    # NACLIP P85.
    relevant_candidate = (
        results[
            "relevant"
        ][
            "candidate_score_summary"
        ]
    )

    if relevant_candidate is not None:
        rr = relevant_candidate

        findings.append(
            f"NACLIP-P85 on relevant-run historical zoom events: "
            f"resolve_rate={rr['candidate_resolve_rate']}, "
            f"median J={rr['candidate_J_median']}, "
            f"min J={rr['candidate_J_min']}, "
            f"J<=5={rr['candidate_J_le5_fraction']}."
        )

        if (
            np.isfinite(
                rr[
                    "candidate_resolve_rate"
                ]
            )
            and rr[
                "candidate_resolve_rate"
            ]
            < WARN_SCORE_RESOLVE_RATE
        ):
            reds.append(
                "NACLIP-P85 resolves too few historical zoom events."
            )

        if (
            np.isfinite(
                rr[
                    "candidate_J_le5_fraction"
                ]
            )
            and rr[
                "candidate_J_le5_fraction"
            ]
            > WARN_JUMP_LE5_FRACTION
        ):
            reds.append(
                "NACLIP-P85 still produces too many J<=5 crossings."
            )

    else:
        warnings.append(
            "NACLIP-P85 candidate was not recomputed. "
            "Run in the project environment with MineCLIP available."
        )

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------

    if reds:
        status = (
            "NOT_READY_FOR_1M"
        )

    elif warnings:
        status = (
            "CONDITIONALLY_READY"
        )

    else:
        status = (
            "READY_FOR_1M"
        )

    report = [
        "FINAL PRE-1M AUDIT",
        "=" * 90,
        "",
        f"STATUS: {status}",
        "",
        "Important:",
        "The U-Net and relevance-map npz files are different trajectories.",
        "Do not treat zoom count alone as a direct quality ranking.",
        "",
        "Findings:",
    ]

    report.extend(
        [
            f"- {x}"
            for x
            in findings
        ]
    )

    report.extend(
        [
            "",
            "Warnings:",
        ]
    )

    if warnings:
        report.extend(
            [
                f"- {x}"
                for x
                in warnings
            ]
        )
    else:
        report.append(
            "- none"
        )

    report.extend(
        [
            "",
            "Red flags:",
        ]
    )

    if reds:
        report.extend(
            [
                f"- {x}"
                for x
                in reds
            ]
        )
    else:
        report.append(
            "- none"
        )

    report.extend(
        [
            "",
            "Interpretation:",
            "1) ScoreStorage exact replay is the strongest sanity check.",
            "2) NACLIP-P85 scale itself does not need to match the old ~0.01 score scale;",
            "   ScoreStorage only needs the SAME monotonic metric for current and zoomed states.",
            "3) Raw-shifted map may be excellent for Natural Zoom even if its decoder reconstruction",
            "   signal is much weaker than the old sigmoid heatmap.",
            "4) If NACLIP-P85 passes but raw-shifted world-model supervision is very weak, consider",
            "   decoupling the two roles instead of discarding NACLIP-P85:",
            "       - raw-shifted Value+Temporal map for Natural Zoom",
            "       - a separate contrast-preserving heatmap representation for world-model input",
            "     or explicitly increase heatmap reconstruction importance.",
            "5) Do NOT use per-frame min-max for the progress score; it destroys absolute cross-frame scale.",
            "",
            "Files to inspect first:",
            "10_historical_scorestorage_comparison.csv",
            "11_worldmodel_heatmap_comparison.csv",
            "12_naclip_p85_candidate_comparison.csv",
            "unet/01_exact_scorestorage_replay.csv",
            "relevant/01_exact_scorestorage_replay.csv",
            "relevant/02_candidate_naclip_p85_scorestorage.csv",
            "relevant/zoom_events/",
        ]
    )

    (
        outdir
        / "14_READ_ME_FIRST.txt"
    ).write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    config = {
        "UNET_NPZ":
            str(
                UNET_NPZ
            ),

        "RELEVANT_NPZ":
            str(
                RELEVANT_NPZ
            ),

        "TARGET_PROMPTS":
            TARGET_PROMPTS,

        "PROGRESS_PERCENTILE":
            PROGRESS_PERCENTILE,

        "OLD_RELEVANCE_THRESHOLD":
            OLD_RELEVANCE_THRESHOLD,

        "OLD_RELEVANCE_TEMPERATURE":
            OLD_RELEVANCE_TEMPERATURE,

        "NACLIP_GAUSSIAN_STD":
            NACLIP_GAUSSIAN_STD,

        "NACLIP_GAUSSIAN_WEIGHT":
            NACLIP_GAUSSIAN_WEIGHT,

        "NACLIP_INCLUDE_CLS":
            NACLIP_INCLUDE_CLS,

        "RUN_MINECLIP_RECOMPUTE":
            RUN_MINECLIP_RECOMPUTE,

        "mineclip_initialization_error":
            mineclip_init_error,
    }

    (
        outdir
        / "15_run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "="
        * 100
    )

    print(
        "FINAL STATUS:",
        status,
    )

    print(
        "Read first:"
    )

    print(
        outdir
        / "14_READ_ME_FIRST.txt"
    )

    print(
        "="
        * 100
    )


if __name__ == "__main__":
    main()
