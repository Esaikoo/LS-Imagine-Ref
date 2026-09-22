#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LS-Imagine / MineCLIP progress-score diagnostic.

用途：
1) Current: Value + Temporal(L=1)
2) Value + NoTemporal
3) NACLIP spatial refinement + NoTemporal
4) NACLIP spatial refinement + Temporal(L=1)

不使用 Competitive。
不使用 Lidar 作为评价。
U-Net 正式纳入 progress 序列指标；但 U-Net 没有 tau/T，因此不参与 patch calibration 参数扫描。

建议把本文件放在：
    <LS-Imagine>/relevance_map/progress_score_test.py
并保证同目录存在你当前已经能运行的 relevance_variants.py。
PyCharm 直接 Run；没有 argparse。
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# =====================================================================
# 0. Project / import existing tested MineCLIP helper
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

# 复用你现有 relevance_variants.py 中已经验证过的：
# - trajectory loading
# - MineCLIP Value patch
# - Temporal(L=1)
# - NACLIP patch
# - Original U-Net
import relevance_map.relevance_variants as rv

# =====================================================================
# 1. GLOBAL CONFIG —— 主要改这里
# =====================================================================

TASK_NAME = "harvest_log_in_plains"

NPZ_PATH = (
    PROJECT_ROOT / "relevance_map" / "real_approach_runs"
    / TASK_NAME / "world_42_taskseed_0_20260904_183014"
    / "trajectory.npz"
)

MINECLIP_CKPT = PROJECT_ROOT / "weights" / "mineclip_attn.pth"

OUTPUT_ROOT = PROJECT_ROOT / "relevance_map" / "progress_score_test_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_H, IMAGE_W = 160, 256

TARGET_PROMPTS = ["Cut a tree"]

# ---- 四条分支 ----
RUN_CURRENT_TEMPORAL = True
RUN_VALUE_NO_TEMPORAL = True
RUN_NACLIP_NO_TEMPORAL = True
RUN_NACLIP_TEMPORAL = True

# ---- U-Net 仅用于参考，不作为优化目标 ----
RUN_ORIGINAL_UNET = True
UNET_CKPT = (
    PROJECT_ROOT / "affordance_map" / "finetune_unet" / "finetune_checkpoints"
    / TASK_NAME / "swin_unet_checkpoint.pth"
)

# ---- 每一步都测试 / 每一步都画 ----
FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1
MAX_FRAMES = None
SAVE_EVERY_FRAME_VISUALIZATION = True    # 每一步都保存可视化
SAVE_SELECTED_FRAME_VISUALIZATION = False
KEY_VIS_FRAMES = [0, 20, 60, 100, 130, 170, 196, 210]
VIS_DPI = 120
OVERLAY_ALPHA = 0.48

# ---- 输出精简 ----
SAVE_ONLY_KEY_OUTPUTS = True
SAVE_FULL_DETAILS = False   # True 时再额外保存完整 csv / heatmap / raw arrays / 全帧图

# ---- progress 区间 ----
# 这条 trajectory 已经手工确认：
# frame 0 开始接近目标树，约 frame 196 到达/开始破坏，
# frame 210 检测到掉落物进入背包。
# 因此本实验明确使用完整任务 progress 区间 [0, 210]。
PROGRESS_SEGMENTS = [(0, 210)]
AUTO_END_AT_FIRST_SUCCESS = False  # 手工区间存在时本来也不会使用自动截断
AUTO_PROGRESS_START_FRAME = None

# ---- 当前 Current calibration ----
# 这条 trajectory.npz 当时真实运行使用的是 tau=0.3, T=0.03。
# 因此 base visualization / base progress 也使用同一组参数，
# 保证与 trajectory.npz 中保存的 relevance_patch 可直接比较。
BASE_THRESHOLD = 0.300
BASE_TEMPERATURE = 0.030

# consistency 检查会优先读取 NPZ 自己保存的 threshold/temperature；
# 如果旧 NPZ 没保存，再退回下面的 fallback。
RECORDED_THRESHOLD_FALLBACK = 0.300
RECORDED_TEMPERATURE_FALLBACK = 0.030

# ---- 自动参数扫描 ----
RUN_PARAMETER_SWEEP = True
MANUAL_THRESHOLDS = [0.24, 0.26, 0.28, 0.288, 0.30, 0.32, 0.34, 0.36, 0.38]
TEMPERATURES = [0.010, 0.016, 0.020, 0.030, 0.040, 0.050, 0.080]

# 不要求你手工猜每个方法的 tau：把每个方法 cosine 的 quantile 也加入 sweep。
AUTO_ADD_QUANTILE_THRESHOLDS = True
AUTO_THRESHOLD_QUANTILES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
SWEEP_TOP_N = 20

# ---- NACLIP ----
NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True

# ---- score / visualization ----
GAUSSIAN_SIGMA_WEIGHT = 0.5
RAW_VIS_LOW_PERCENTILE = 2.0
RAW_VIS_HIGH_PERCENTILE = 98.0
SAVE_RAW_ARRAYS = False

METHOD_DISPLAY = OrderedDict([
    ("current_temporal_l1", "Current: Value + Temporal(L=1)"),
    ("value_no_temporal", "Value + NoTemporal"),
    ("naclip_no_temporal", "NACLIP + NoTemporal"),
    ("naclip_temporal_l1", "NACLIP + Temporal(L=1)"),
])

# 把本脚本配置同步给 relevance_variants.py 的 helper。
rv.DEVICE = DEVICE
rv.RELEVANCE_THRESHOLD = BASE_THRESHOLD
rv.RELEVANCE_TEMPERATURE = BASE_TEMPERATURE
rv.NACLIP_GAUSSIAN_STD = NACLIP_GAUSSIAN_STD
rv.NACLIP_GAUSSIAN_WEIGHT = NACLIP_GAUSSIAN_WEIGHT
rv.NACLIP_INCLUDE_CLS = NACLIP_INCLUDE_CLS
rv.IMAGE_H = IMAGE_H
rv.IMAGE_W = IMAGE_W
rv.UNET_CKPT = UNET_CKPT

# =====================================================================
# 2. Utils
# =====================================================================

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def safe_spearman(a, b) -> float:
    x = np.asarray(a, np.float64).reshape(-1)
    y = np.asarray(b, np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    r = spearmanr(x, y)
    if hasattr(r, "statistic"):
        return float(r.statistic)
    if hasattr(r, "correlation"):
        return float(r.correlation)
    return float(r[0])


def safe_pearson(a, b) -> float:
    x = np.asarray(a, np.float64).reshape(-1)
    y = np.asarray(b, np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def upsample_patch(patch: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(patch, np.float32))[None, None]
    return F.interpolate(
        x, size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False
    )[0, 0].numpy().astype(np.float32)


def load_saved_metadata(path: Path) -> Dict[str, np.ndarray]:
    # 明确不使用 Lidar。
    wanted = [
        "cosines", "relevance_patch", "actions", "action_commands",
        "stage_labels", "notes", "positions", "reward",
        "target_inventory_quantity", "prompt", "threshold", "temperature",
        "task_name",
    ]
    out = {}
    with np.load(path, allow_pickle=True) as data:
        for k in wanted:
            if k in data.files:
                out[k] = np.asarray(data[k])
    return out


def scalar_at(saved, key, fi, default=np.nan):
    arr = saved.get(key)
    if arr is None:
        return default
    if arr.ndim == 0:
        try:
            return arr.item()
        except Exception:
            return default
    if fi >= len(arr):
        return default
    x = np.asarray(arr[fi]).reshape(-1)
    if not len(x):
        return default
    try:
        return x[0].item()
    except Exception:
        return x[0]


def string_at(saved, key, fi, default=""):
    arr = saved.get(key)
    if arr is None or arr.ndim == 0 or fi >= len(arr):
        return default
    return str(arr[fi])


def metadata_for_frame(saved, fi):
    row = {
        "saved_reward": scalar_at(saved, "reward", fi),
        "saved_target_inventory_quantity": scalar_at(saved, "target_inventory_quantity", fi),
        "saved_stage_label": string_at(saved, "stage_labels", fi),
        "saved_note": string_at(saved, "notes", fi),
        "saved_action_command": string_at(saved, "action_commands", fi),
    }
    pos = saved.get("positions")
    if pos is not None and pos.ndim >= 2 and fi < pos.shape[0] and pos.shape[1] >= 3:
        row.update(
            saved_pos_x=float(pos[fi, 0]),
            saved_pos_y=float(pos[fi, 1]),
            saved_pos_z=float(pos[fi, 2]),
        )
    else:
        row.update(saved_pos_x=np.nan, saved_pos_y=np.nan, saved_pos_z=np.nan)
    return row


def select_indices(n):
    end = n if FRAME_END is None else min(n, int(FRAME_END))
    idx = list(range(max(0, int(FRAME_START)), end, max(1, int(FRAME_STRIDE))))
    if MAX_FRAMES is not None:
        idx = idx[:int(MAX_FRAMES)]
    if not idx:
        raise RuntimeError("No selected frames")
    return np.asarray(idx, np.int64)

# =====================================================================
# 3. Exact patch aggregation weights
# =====================================================================

def center_gaussian_full():
    sx = IMAGE_W * GAUSSIAN_SIGMA_WEIGHT
    sy = IMAGE_H * GAUSSIAN_SIGMA_WEIGHT
    x = np.linspace(-IMAGE_W // 2, IMAGE_W // 2, IMAGE_W)
    y = np.linspace(-IMAGE_H // 2, IMAGE_H // 2, IMAGE_H)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx**2 / (2*sx**2) + yy**2 / (2*sy**2))).astype(np.float32)


CENTER_GAUSSIAN = center_gaussian_full()


def exact_patch_weights(gh: int, gw: int) -> Tuple[np.ndarray, np.ndarray]:
    """精确等价于：patch map -> bilinear 160x256 -> Gaussian/full mean。"""
    n = gh * gw
    basis = torch.eye(n, dtype=torch.float32).reshape(n, 1, gh, gw)
    full = F.interpolate(basis, (IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)[:, 0]
    g = torch.from_numpy(CENTER_GAUSSIAN).float()
    wg = ((full * g[None]).sum((1, 2)) / g.sum()).numpy().astype(np.float64)
    wm = full.mean((1, 2)).numpy().astype(np.float64)
    return wg, wm

# =====================================================================
# 4. Progress window (no Lidar)
# =====================================================================

def first_success_frame(saved, selected) -> Optional[int]:
    cands = []
    reward = saved.get("reward")
    if reward is not None and reward.ndim > 0:
        for fi in selected:
            if fi >= len(reward):
                continue
            try:
                r = float(np.asarray(reward[fi]).reshape(-1)[0])
            except Exception:
                continue
            if np.isfinite(r) and r > 0:
                cands.append(int(fi))
                break

    inv = saved.get("target_inventory_quantity")
    if inv is not None and inv.ndim > 0:
        vals = []
        for fi in selected:
            try:
                vals.append(float(np.asarray(inv[fi]).reshape(-1)[0]) if fi < len(inv) else np.nan)
            except Exception:
                vals.append(np.nan)
        vals = np.asarray(vals, np.float64)
        finite = np.where(np.isfinite(vals))[0]
        if len(finite):
            base = vals[finite[0]]
            for i, v in enumerate(vals):
                if np.isfinite(v) and v > base:
                    cands.append(int(selected[i]))
                    break
    return min(cands) if cands else None


def make_progress_mask(selected, saved):
    if PROGRESS_SEGMENTS is not None:
        mask = np.zeros(len(selected), bool)
        segs = []
        for a, b in PROGRESS_SEGMENTS:
            lo, hi = min(int(a), int(b)), max(int(a), int(b))
            mask |= (selected >= lo) & (selected <= hi)
            segs.append((lo, hi))
        return mask, segs, "manual PROGRESS_SEGMENTS"

    start = int(selected[0] if AUTO_PROGRESS_START_FRAME is None else AUTO_PROGRESS_START_FRAME)
    end = int(selected[-1])
    reason = "all selected frames"
    if AUTO_END_AT_FIRST_SUCCESS:
        suc = first_success_frame(saved, selected)
        if suc is not None:
            end = int(suc)
            reason = "auto stop at first reward>0 or target inventory increase"
    mask = (selected >= start) & (selected <= end)
    return mask, [(start, end)], reason

# =====================================================================
# 5. Similarity / score definitions
# =====================================================================

def cosine_vector(tester, patch, prompts) -> np.ndarray:
    sim = tester.cosine(patch, prompts)  # [1,P,N]
    sim = sim.max(dim=1).values[0]
    return sim.detach().cpu().numpy().astype(np.float32)


def patch_stats(sim):
    x = np.asarray(sim, np.float64).reshape(-1)
    sx = np.sort(x)
    k10p = max(1, int(round(len(x) * 0.10)))
    p50, p95 = np.percentile(x, [50, 95])
    return dict(
        raw_mean=float(x.mean()), raw_std=float(x.std()),
        raw_min=float(x.min()), raw_max=float(x.max()),
        raw_p50=float(p50), raw_p75=float(np.percentile(x, 75)),
        raw_p90=float(np.percentile(x, 90)), raw_p95=float(p95),
        raw_p99=float(np.percentile(x, 99)),
        raw_p95_minus_p50=float(p95-p50),
        raw_top5_mean=float(sx[-min(5, len(x)):].mean()),
        raw_top10_mean=float(sx[-min(10, len(x)):].mean()),
        raw_top10pct_mean=float(sx[-k10p:].mean()),
    )


def progress_scores(sim, wg, wm, tau, temp):
    """
    新 progress candidates：

    sigmoid_gaussian:
        当前方式，可能饱和。

    excess_gaussian:
        max(sim-tau, 0)，超过阈值后继续线性增长，不会在 1 封顶。

    softplus_gaussian:
        T*softplus((sim-tau)/T)，excess 的平滑版。
    """
    x = np.asarray(sim, np.float64).reshape(-1)
    z = (x - float(tau)) / float(temp)
    rel = stable_sigmoid(z)
    excess = np.maximum(x - float(tau), 0.0)
    soft = float(temp) * softplus(z)
    out = patch_stats(x)
    out.update(
        raw_center_gaussian=float(x @ wg),
        raw_full_mean_equiv=float(x @ wm),
        sigmoid_gaussian=float(rel @ wg),
        sigmoid_mean=float(rel @ wm),
        sigmoid_patch_area_gt_03=float(np.mean(rel > 0.3)),
        sigmoid_patch_area_gt_05=float(np.mean(rel > 0.5)),
        sigmoid_patch_area_gt_07=float(np.mean(rel > 0.7)),
        sigmoid_patch_area_gt_09=float(np.mean(rel > 0.9)),
        sigmoid_patch_saturation_gt_098=float(np.mean(rel > 0.98)),
        sigmoid_patch_saturation_lt_002=float(np.mean(rel < 0.02)),
        excess_gaussian=float(excess @ wg),
        excess_mean=float(excess @ wm),
        softplus_gaussian=float(soft @ wg),
        softplus_mean=float(soft @ wm),
    )
    return out


PROGRESS_SCORE_NAMES = [
    "raw_top10_mean", "raw_top10pct_mean", "raw_p95", "raw_center_gaussian",
    "sigmoid_gaussian", "sigmoid_mean",
    "excess_gaussian", "excess_mean",
    "softplus_gaussian", "softplus_mean",
]

KEY_SCORE_NAMES = [
    "raw_p95",
    "sigmoid_gaussian",
    "excess_gaussian",
    "softplus_gaussian",
]

# =====================================================================
# 6. Progress metrics
# =====================================================================

def pair_order_accuracy(y):
    y = np.asarray(y, np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return float("nan")
    correct, total = 0.0, 0
    for i in range(len(y)-1):
        d = y[i+1:] - y[i]
        correct += float(np.sum(d > 0)) + 0.5 * float(np.sum(np.isclose(d, 0.0, atol=1e-12)))
        total += len(d)
    return float(correct / total)


def series_metrics(frames, scores):
    x = np.asarray(frames, np.float64)
    y = np.asarray(scores, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < 3:
        return {k: float("nan") for k in [
            "spearman_to_progress", "pearson_to_progress", "pair_order_accuracy",
            "positive_delta_ratio", "nondecrease_ratio", "slope_per_100_frames",
            "start_mean", "end_mean", "start_end_gain", "effect_size_gain_over_std",
            "p10", "p90", "p90_minus_p10", "std", "median_abs_delta"]} | {"n": len(y)}

    order = np.argsort(x)
    x, y = x[order], y[order]
    d = np.diff(y)
    tol = max(1e-10, float(np.std(y)) * 0.01)
    chunk = min(max(3, int(math.ceil(len(y)*0.10))), max(1, len(y)//2))
    start_mean = float(y[:chunk].mean())
    end_mean = float(y[-chunk:].mean())
    gain = end_mean - start_mean
    std = float(y.std())
    p10, p90 = np.percentile(y, [10, 90])
    slope = float(np.polyfit(x, y, 1)[0])
    return dict(
        n=int(len(y)),
        spearman_to_progress=safe_spearman(x, y),
        pearson_to_progress=safe_pearson(x, y),
        pair_order_accuracy=pair_order_accuracy(y),
        positive_delta_ratio=float(np.mean(d > 0)),
        nondecrease_ratio=float(np.mean(d >= -tol)),
        slope_per_100_frames=slope*100.0,
        start_mean=start_mean, end_mean=end_mean,
        start_end_gain=float(gain),
        effect_size_gain_over_std=float(gain/(std+1e-12)),
        p10=float(p10), p90=float(p90), p90_minus_p10=float(p90-p10),
        std=std, median_abs_delta=float(np.median(np.abs(d))),
    )

# =====================================================================
# 7. Parameter sweep
# =====================================================================

def thresholds_for_method(sim_matrix):
    vals = list(MANUAL_THRESHOLDS)
    if AUTO_ADD_QUANTILE_THRESHOLDS:
        flat = sim_matrix.reshape(-1)
        for q in AUTO_THRESHOLD_QUANTILES:
            vals.append(round(float(np.quantile(flat, q)), 4))
    return sorted(set(round(float(x), 4) for x in vals))


def sweep_method(method, sim_matrix, frames, mask, wg, wm):
    rows = []
    pf = frames[mask]
    for tau in thresholds_for_method(sim_matrix):
        for temp in TEMPERATURES:
            z = (sim_matrix - tau) / temp
            rel = stable_sigmoid(z)
            sg = rel @ wg
            sm = rel @ wm
            soft = temp * softplus(z)
            softg = soft @ wg
            mg = series_metrics(pf, sg[mask])
            mm = series_metrics(pf, sm[mask])
            ms = series_metrics(pf, softg[mask])
            row = dict(
                method=method,
                display_name=METHOD_DISPLAY.get(method, method),
                threshold=float(tau), temperature=float(temp),
                sigmoid_mean_patch_area_gt_05=float(np.mean(rel[mask] > 0.5)),
                sigmoid_mean_saturation_gt_098=float(np.mean(rel[mask] > 0.98)),
                sigmoid_mean_saturation_lt_002=float(np.mean(rel[mask] < 0.02)),
            )
            row.update({"sigmoid_"+k: v for k, v in mg.items()})
            row.update({"sigmoid_mean_"+k: v for k, v in mm.items()})
            row.update({"softplus_"+k: v for k, v in ms.items()})
            rows.append(row)
    return rows


def sweep_excess(method, sim_matrix, frames, mask, wg, wm):
    rows = []
    pf = frames[mask]
    for tau in thresholds_for_method(sim_matrix):
        ex = np.maximum(sim_matrix - tau, 0.0)
        eg, em = ex @ wg, ex @ wm
        a, b = series_metrics(pf, eg[mask]), series_metrics(pf, em[mask])
        row = dict(method=method, display_name=METHOD_DISPLAY.get(method, method), threshold=float(tau))
        row.update({"excess_gaussian_"+k: v for k, v in a.items()})
        row.update({"excess_mean_"+k: v for k, v in b.items()})
        rows.append(row)
    return rows


def top_sweep(rows):
    out = []
    for method in sorted(set(r["method"] for r in rows)):
        rr = [dict(r) for r in rows if r["method"] == method]
        def key(r):
            def f(name, bad):
                v = r.get(name, np.nan)
                return bad if not np.isfinite(v) else float(v)
            return (-f("sigmoid_spearman_to_progress", -999),
                    -f("sigmoid_pair_order_accuracy", -999),
                    -f("sigmoid_effect_size_gain_over_std", -999),
                     f("sigmoid_mean_saturation_gt_098", 999))
        rr.sort(key=key)
        for rank, r in enumerate(rr[:SWEEP_TOP_N], 1):
            r["rank_within_method"] = rank
            out.append(r)
    return out


def save_sweep_heatmaps(outdir, rows):
    d = outdir / "parameter_sweep_heatmaps"
    d.mkdir(exist_ok=True)
    for method in sorted(set(r["method"] for r in rows)):
        rr = [r for r in rows if r["method"] == method]
        ths = sorted(set(float(r["threshold"]) for r in rr))
        ts = sorted(set(float(r["temperature"]) for r in rr))
        pairs = [
            ("sigmoid_spearman_to_progress", "Sigmoid Gaussian: Spearman to progress", "spearman"),
            ("sigmoid_start_end_gain", "Sigmoid Gaussian: start-end gain", "gain"),
            ("sigmoid_mean_saturation_gt_098", "Patch saturation > 0.98", "saturation"),
            ("softplus_spearman_to_progress", "Softplus Gaussian: Spearman to progress", "softplus_spearman"),
        ]
        for metric, title, suffix in pairs:
            mat = np.full((len(ts), len(ths)), np.nan)
            ti, hi = {v:i for i,v in enumerate(ts)}, {v:i for i,v in enumerate(ths)}
            for r in rr:
                mat[ti[float(r["temperature"])], hi[float(r["threshold"])]] = float(r[metric])
            fig, ax = plt.subplots(figsize=(max(10, len(ths)*0.7), 5.5))
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_xticks(range(len(ths))); ax.set_xticklabels([f"{x:.3f}" for x in ths], rotation=45, ha="right")
            ax.set_yticks(range(len(ts))); ax.set_yticklabels([f"{x:.3f}" for x in ts])
            ax.set_xlabel("threshold tau"); ax.set_ylabel("temperature T")
            ax.set_title(METHOD_DISPLAY.get(method, method) + "\n" + title)
            fig.colorbar(im, ax=ax)
            plt.tight_layout(); fig.savefig(d/f"{method}_{suffix}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# =====================================================================
# 8. Summary / factorial ablation
# =====================================================================

def build_progress_summary(scores, frames, mask):
    rows = []
    pf = frames[mask]
    for method, dd in scores.items():
        for score_name in PROGRESS_SCORE_NAMES:
            m = series_metrics(pf, dd[score_name][mask])
            rows.append(dict(method=method, display_name=METHOD_DISPLAY.get(method, method), score_name=score_name, **m))
    return rows


def build_factorial(summary):
    lookup = {(r["method"], r["score_name"]): r for r in summary}
    pairs = [
        ("temporal_effect_on_value", "value_no_temporal", "current_temporal_l1"),
        ("spatial_effect_without_temporal", "value_no_temporal", "naclip_no_temporal"),
        ("spatial_effect_with_temporal", "current_temporal_l1", "naclip_temporal_l1"),
        ("temporal_effect_on_naclip", "naclip_no_temporal", "naclip_temporal_l1"),
    ]
    metrics = ["spearman_to_progress", "pair_order_accuracy", "positive_delta_ratio",
               "nondecrease_ratio", "start_end_gain", "effect_size_gain_over_std",
               "p90_minus_p10", "median_abs_delta"]
    rows = []
    for name, base, changed in pairs:
        for score_name in PROGRESS_SCORE_NAMES:
            a, b = lookup.get((base, score_name)), lookup.get((changed, score_name))
            if a is None or b is None:
                continue
            row = dict(comparison=name, baseline_method=base, changed_method=changed, score_name=score_name)
            for m in metrics:
                av, bv = float(a.get(m, np.nan)), float(b.get(m, np.nan))
                row["baseline_"+m] = av; row["changed_"+m] = bv
                row["delta_"+m] = bv-av if np.isfinite(av) and np.isfinite(bv) else np.nan
            rows.append(row)
    return rows

# =====================================================================
# 9. Visualization
# =====================================================================

def save_frame_panel(path, frame, fi, sims, frame_scores, unet_map, raw_vmin, raw_vmax, excess_vmax=None):
    """
    恢复较清晰的 3 行布局：
      row 1: sigmoid relevance heatmap
      row 2: RGB + sigmoid overlay
      row 3: raw cosine map

    注意：raw_p95 / excess_gaussian / softplus_gaussian 都是标量 progress score，
    不是另一张 spatial heatmap，因此只显示在标题中，不额外再占一整行。
    """
    methods = list(sims.keys())
    extra = 1 if unet_map is not None else 0
    cols, rows = 1 + extra + len(methods), 3
    fig, ax = plt.subplots(
        rows, cols,
        figsize=(max(18, cols * 4.2), 10.2),
        squeeze=False,
    )

    # RGB / text column
    for r in range(rows):
        ax[r, 0].axis("off")
    ax[0, 0].imshow(frame)
    ax[0, 0].set_title(f"RGB | frame={fi}", fontsize=11)
    ax[1, 0].imshow(frame)
    ax[1, 0].set_title("RGB reference", fontsize=11)
    ax[2, 0].text(
        0.02, 0.96,
        "Rows:\n"
        "1) sigmoid relevance map\n"
        "2) RGB + sigmoid overlay\n"
        "3) raw cosine map\n\n"
        "P95 / Gex / Soft are scalar progress scores\n"
        "and are shown in each title only.\n\n"
        f"base tau={BASE_THRESHOLD}\n"
        f"base T={BASE_TEMPERATURE}",
        va="top", ha="left",
        transform=ax[2, 0].transAxes,
        fontsize=10,
    )

    c = 1
    if unet_map is not None:
        unet_g_frame = float(
            (np.asarray(unet_map, np.float64) * CENTER_GAUSSIAN).sum()
            / CENTER_GAUSSIAN.sum()
        )
        ax[0, c].imshow(unet_map, cmap="jet", vmin=0.0, vmax=1.0)
        ax[0, c].set_title(
            f"Original LS-Imagine U-Net\nGaussian progress={unet_g_frame:.3f}",
            fontsize=10,
        )
        ax[1, c].imshow(frame)
        ax[1, c].imshow(
            unet_map,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            alpha=OVERLAY_ALPHA,
        )
        ax[1, c].set_title("U-Net overlay", fontsize=10)
        ax[2, c].axis("off")
        ax[2, c].text(
            0.03, 0.92,
            "U-Net has no raw cosine / excess map.\n"
            "Its progress scalar is unet_gaussian.",
            va="top",
            transform=ax[2, c].transAxes,
            fontsize=10,
        )
        ax[0, c].axis("off")
        ax[1, c].axis("off")
        c += 1

    for method in methods:
        sim = np.asarray(sims[method], np.float32).reshape(10, 16)
        rel = stable_sigmoid(
            (sim - BASE_THRESHOLD) / BASE_TEMPERATURE
        ).astype(np.float32)
        rel_big = upsample_patch(rel)
        raw_big = upsample_patch(sim)
        s = frame_scores[method]

        title = (
            METHOD_DISPLAY.get(method, method)
            + "\n"
            + f"P95={s['raw_p95']:.4f} | Gsig={s['sigmoid_gaussian']:.3f}\n"
            + f"Gex={s['excess_gaussian']:.4f} | Soft={s['softplus_gaussian']:.4f}"
        )

        ax[0, c].imshow(rel_big, cmap="jet", vmin=0.0, vmax=1.0)
        ax[0, c].set_title(title, fontsize=9)

        ax[1, c].imshow(frame)
        ax[1, c].imshow(
            rel_big,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            alpha=OVERLAY_ALPHA,
        )
        ax[1, c].set_title("sigmoid overlay", fontsize=9)

        ax[2, c].imshow(
            raw_big,
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )
        ax[2, c].set_title(
            f"raw cosine map\nlocal={sim.min():.3f}~{sim.max():.3f}",
            fontsize=9,
        )

        for r in range(rows):
            ax[r, c].axis("off")
        c += 1

    plt.tight_layout()
    fig.savefig(path, dpi=VIS_DPI, bbox_inches="tight")
    plt.close(fig)

def _plot_unet_reference(ax, frames, unet_gaussian, same_axis: bool):
    """Add U-Net reference without destroying the left-axis scale."""
    if unet_gaussian is None:
        return None
    if same_axis:
        line = ax.plot(
            frames,
            unet_gaussian,
            "--",
            linewidth=2.0,
            label="Original U-Net Gaussian",
        )[0]
        return None

    ax2 = ax.twinx()
    ax2.plot(
        frames,
        unet_gaussian,
        "--",
        linewidth=2.0,
        alpha=0.75,
        label="Original U-Net Gaussian (right axis)",
    )
    ax2.set_ylabel("U-Net Gaussian score (right axis)")
    return ax2


def _merge_legends(ax, ax2=None):
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        h1 += h2
        l1 += l2
    ax.legend(h1, l1, fontsize=8, ncol=2)


def save_curves(outdir, frames, scores, mask, unet_gaussian=None):
    """
    U-Net Gaussian 与 sigmoid_gaussian 是同一种 [0,1] map-score 量纲，可共用 y 轴。

    raw_p95/raw_top10 是 cosine；excess/softplus 是未饱和的 similarity excess，
    数值尺度与 U-Net Gaussian 完全不同。因此这些图中的 U-Net 使用右侧 y 轴，
    只比较时间趋势/形状，不比较绝对数值大小。
    """
    groups = [
        (["raw_top10_mean", "raw_p95"], "raw_similarity_progress.png", "Raw similarity progress", False),
        (["sigmoid_gaussian"], "sigmoid_gaussian_progress.png", "Sigmoid Gaussian progress", True),
        (["excess_gaussian"], "excess_gaussian_progress.png", "Non-saturating excess Gaussian progress", False),
        (["softplus_gaussian"], "softplus_gaussian_progress.png", "Softplus Gaussian progress", False),
    ]

    for names, fn, title, same_axis in groups:
        fig, ax = plt.subplots(figsize=(13, 6))
        for method, dd in scores.items():
            for name in names:
                label = METHOD_DISPLAY.get(method, method)
                if len(names) > 1:
                    label += " | " + name
                ax.plot(frames, dd[name], label=label)

        ax2 = _plot_unet_reference(
            ax,
            frames,
            unet_gaussian,
            same_axis=same_axis,
        )

        if np.any(mask):
            pf = frames[mask]
            ax.axvspan(
                pf.min(), pf.max(),
                alpha=0.08,
                label="progress eval window",
            )

        ax.set_xlabel("Trajectory frame index")
        if names == ["sigmoid_gaussian"]:
            ax.set_ylabel("Gaussian-weighted relevance score")
        elif names == ["excess_gaussian"]:
            ax.set_ylabel("Excess Gaussian score")
        elif names == ["softplus_gaussian"]:
            ax.set_ylabel("Softplus Gaussian score")
        else:
            ax.set_ylabel("Raw cosine similarity")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        _merge_legends(ax, ax2)
        plt.tight_layout()
        fig.savefig(outdir / fn, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 总览：把每条曲线各自 min-max 到 [0,1]，只比较 progress 形状。
    # 这样 U-Net / raw cosine / excess 不会因为单位不同而互相压扁。
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    flat_axes = axes.reshape(-1)
    for ax, (names, _, title, _) in zip(flat_axes, groups):
        for method, dd in scores.items():
            for name in names:
                y = np.asarray(dd[name], np.float64)
                ymin, ymax = np.nanmin(y), np.nanmax(y)
                yn = (y - ymin) / (ymax - ymin + 1e-12)
                label = METHOD_DISPLAY.get(method, method)
                if len(names) > 1:
                    label += " | " + name
                ax.plot(frames, yn, label=label)
        if unet_gaussian is not None:
            y = np.asarray(unet_gaussian, np.float64)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            yn = (y - ymin) / (ymax - ymin + 1e-12)
            ax.plot(
                frames,
                yn,
                "--",
                linewidth=2.0,
                label="Original U-Net Gaussian",
            )
        if np.any(mask):
            pf = frames[mask]
            ax.axvspan(pf.min(), pf.max(), alpha=0.08)
        ax.set_title(title + " (per-series min-max normalized)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized progress shape [0,1]")
        ax.grid(alpha=0.25)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        outdir / "progress_overview_normalized_2x2.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

def save_key_summary_csv(outdir, summary):
    rows = []
    for r in summary:
        if (r['method'] == 'unet_original' and r['score_name'] == 'unet_gaussian') or (r['method'] != 'unet_original' and r['score_name'] in KEY_SCORE_NAMES):
            rows.append(r)
    save_csv(outdir/"key_progress_summary.csv", rows)


def save_key_metric_bars(outdir, summary):
    rows = []
    for r in summary:
        if (r['method'] == 'unet_original' and r['score_name'] == 'unet_gaussian') or (r['method'] != 'unet_original' and r['score_name'] in ['raw_p95', 'sigmoid_gaussian', 'excess_gaussian']):
            rows.append(r)
    if not rows:
        return
    labels = []
    spearman = []
    pairacc = []
    effect = []
    for r in rows:
        short_method = 'U-Net' if r['method'] == 'unet_original' else METHOD_DISPLAY.get(r['method'], r['method']).replace('Current: ', '').replace('Value + ', 'Value|').replace('NACLIP + ', 'NACLIP|').replace(' + ', '|')
        short_score = 'unet_G' if r['score_name'] == 'unet_gaussian' else r['score_name'].replace('_gaussian','').replace('raw_','')
        labels.append(f"{short_method}\n{short_score}")
        spearman.append(float(r.get('spearman_to_progress', np.nan)))
        pairacc.append(float(r.get('pair_order_accuracy', np.nan)))
        effect.append(float(r.get('effect_size_gain_over_std', np.nan)))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(labels)*0.9), 10), squeeze=False)
    for ax, vals, title in zip(axes.reshape(-1), [spearman, pairacc, effect], ['Spearman to progress', 'Pair-order accuracy', 'Effect size (gain/std)']):
        x = np.arange(len(labels))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.25)
    plt.tight_layout()
    fig.savefig(outdir/'key_metrics_bars.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_compact_per_frame(outdir, idx, meta_rows, method_scores, unet_g=None):
    rows = []
    for pi, fi in enumerate(idx):
        row = dict(frame_index=int(fi), is_progress_eval_frame=int(meta_rows[pi]['is_progress_eval_frame']))
        for method, dd in method_scores.items():
            prefix = method
            row[prefix + '_raw_p95'] = float(dd['raw_p95'][pi])
            row[prefix + '_sigmoid_gaussian'] = float(dd['sigmoid_gaussian'][pi])
            row[prefix + '_excess_gaussian'] = float(dd['excess_gaussian'][pi])
            row[prefix + '_softplus_gaussian'] = float(dd['softplus_gaussian'][pi])
        if unet_g is not None:
            row['unet_gaussian'] = float(unet_g[pi])
        rows.append(row)
    save_csv(outdir/'per_frame_key_scores.csv', rows)


def save_selected_frame_indices(outdir, idx):
    selected = []
    idx_set = set(int(x) for x in idx.tolist())
    for fi in KEY_VIS_FRAMES:
        if int(fi) in idx_set:
            selected.append(int(fi))
    if not selected:
        selected = [int(idx[0]), int(idx[len(idx)//2]), int(idx[-1])]
    (outdir/'selected_visualization_frames.txt').write_text('\n'.join(map(str, selected)), encoding='utf-8')
    return selected


# =====================================================================
# 10. Docs
# =====================================================================

README_ZH = r"""
Progress Score 测试说明
======================

四条支路：
1) Current = Value + Temporal(L=1)
2) Value + NoTemporal
3) NACLIP + NoTemporal
4) NACLIP + Temporal(L=1)

注意：第4条不是 NACLIP 官方结构，而是专门做的 ablation。
Temporal(L=1) 没有真正时间关系，但仍是 MineCLIP 学到的 feature transformation。
NACLIP feature 的分布和原 Value/global feature 也不保证一致，所以必须实测。

核心 progress score：

raw_top10_mean / raw_p95
    直接使用 sigmoid 前 cosine，不会被 0~1 上限饱和。

sigmoid_gaussian
    当前方式：sigmoid((sim-tau)/T) 后做 center Gaussian 聚合。
    T 很小时会快速饱和。

excess_gaussian
    excess=max(sim-tau,0)，再 Gaussian 聚合。
    超过 tau 后仍线性增长，同时对目标面积和中心位置敏感。

softplus_gaussian
    T*softplus((sim-tau)/T)，是 excess 的平滑版本。
    低于 tau 不会硬截断，高于 tau 后近似线性，不会在 1 封顶。

Progress 指标：

spearman_to_progress
    和 approach frame 顺序的秩相关。越高代表越能表达“越接近目标分数越高”。

pair_order_accuracy
    对所有 earlier/later frame pair，看 later score 是否更高。
    0.5 约等于没有顺序信息，越接近 1 越好。

start_end_gain
    approach 最后10%均值 - 开始10%均值。表示动态范围。

effect_size_gain_over_std
    start_end_gain / 整段 std，避免单纯数值尺度大造成误判。

positive_delta_ratio / nondecrease_ratio
    相邻帧上升/不下降比例。真实视角会波动，所以只作辅助。

p90_minus_p10
    整段 score 的有效动态范围。

参数扫描：
- MANUAL_THRESHOLDS + 每个 method 自己的 cosine quantile thresholds
- TEMPERATURES
- calibration_sweep.csv
- sweep_top_candidates.csv
- parameter_sweep_heatmaps/

这些 top candidates 只用于筛选，不代表最终训练参数。
最好跨多条 approach trajectory / 多任务再决定。

判断 NACLIP 是否真正有帮助：
重点看 factorial_comparison.csv：
- spatial_effect_without_temporal: NACLIP-NoTemporal vs Value-NoTemporal
- spatial_effect_with_temporal: NACLIP-Temporal vs Current
如果两条都能稳定改善 progress 指标，才有较强证据说 spatial refinement 对 LS-Imagine 有帮助。
仅仅 map 更集中、更漂亮不够。

U-Net progress：
- U-Net 的 Gaussian score 会进入 progress_summary.csv；
- 使用和 patch progress 相同的 spearman / pair-order / gain / effect 等序列指标；
- U-Net 没有 threshold / temperature，所以不参与 calibration_sweep。

本程序不使用 Lidar 作为评价。
当前轨迹手工指定 progress 区间为 [0,210]。
默认只生成关键帧 visualizations/frame_xxxxxx.png；如果 SAVE_EVERY_FRAME_VISUALIZATION=True，则保存全部 tested frame。
"""

# =====================================================================
# 11. Main
# =====================================================================

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT_ROOT / f"{TASK_NAME}_{stamp}"
    visdir = outdir / "visualizations"
    outdir.mkdir(parents=True, exist_ok=False)
    if SAVE_EVERY_FRAME_VISUALIZATION or SAVE_SELECTED_FRAME_VISUALIZATION:
        visdir.mkdir()
    (outdir/"README_zh.txt").write_text(README_ZH, encoding="utf-8")

    cfg = dict(
        TASK_NAME=TASK_NAME, NPZ_PATH=str(NPZ_PATH), MINECLIP_CKPT=str(MINECLIP_CKPT),
        TARGET_PROMPTS=TARGET_PROMPTS, DEVICE=DEVICE,
        FRAME_START=FRAME_START, FRAME_END=FRAME_END, FRAME_STRIDE=FRAME_STRIDE, MAX_FRAMES=MAX_FRAMES,
        PROGRESS_SEGMENTS=PROGRESS_SEGMENTS, AUTO_END_AT_FIRST_SUCCESS=AUTO_END_AT_FIRST_SUCCESS,
        BASE_THRESHOLD=BASE_THRESHOLD, BASE_TEMPERATURE=BASE_TEMPERATURE,
        RECORDED_THRESHOLD_FALLBACK=RECORDED_THRESHOLD_FALLBACK,
        RECORDED_TEMPERATURE_FALLBACK=RECORDED_TEMPERATURE_FALLBACK,
        MANUAL_THRESHOLDS=MANUAL_THRESHOLDS, TEMPERATURES=TEMPERATURES,
        AUTO_ADD_QUANTILE_THRESHOLDS=AUTO_ADD_QUANTILE_THRESHOLDS,
        AUTO_THRESHOLD_QUANTILES=AUTO_THRESHOLD_QUANTILES,
        NACLIP_GAUSSIAN_STD=NACLIP_GAUSSIAN_STD, NACLIP_GAUSSIAN_WEIGHT=NACLIP_GAUSSIAN_WEIGHT,
        RUN_ORIGINAL_UNET=RUN_ORIGINAL_UNET, RUN_PARAMETER_SWEEP=RUN_PARAMETER_SWEEP,
        SAVE_ONLY_KEY_OUTPUTS=SAVE_ONLY_KEY_OUTPUTS, SAVE_FULL_DETAILS=SAVE_FULL_DETAILS,
        SAVE_EVERY_FRAME_VISUALIZATION=SAVE_EVERY_FRAME_VISUALIZATION,
        SAVE_SELECTED_FRAME_VISUALIZATION=SAVE_SELECTED_FRAME_VISUALIZATION,
        KEY_VIS_FRAMES=KEY_VIS_FRAMES,
    )
    (outdir/"run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("="*88); print("PROGRESS SCORE TEST"); print("OUTPUT:", outdir); print("DEVICE:", DEVICE)

    frames, rgb_key, key_info = rv.load_frames(NPZ_PATH)
    saved = load_saved_metadata(NPZ_PATH)

    # trajectory.npz 当时真正使用的 calibration。
    # consistency 必须用这一组参数重算，而不是盲目使用当前脚本 BASE 参数。
    recorded_tau = RECORDED_THRESHOLD_FALLBACK
    recorded_temp = RECORDED_TEMPERATURE_FALLBACK
    if "threshold" in saved:
        try:
            recorded_tau = float(np.asarray(saved["threshold"]).item())
        except Exception:
            pass
    if "temperature" in saved:
        try:
            recorded_temp = float(np.asarray(saved["temperature"]).item())
        except Exception:
            pass

    print(f"Recorded trajectory calibration: tau={recorded_tau}, T={recorded_temp}")
    (outdir/"recorded_trajectory_calibration.json").write_text(
        json.dumps({
            "threshold": recorded_tau,
            "temperature": recorded_temp,
            "source": "trajectory.npz if available, otherwise fallback"
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    idx = select_indices(len(frames))
    (outdir/"npz_keys.txt").write_text("Selected RGB key: "+rgb_key+"\n\n"+"\n".join(f"{k}: {v}" for k,v in key_info.items()), encoding="utf-8")
    print(f"RGB key={rgb_key}; total={len(frames)}; tested={len(idx)}")

    pmask, segments, reason = make_progress_mask(idx, saved)
    if not np.any(pmask):
        raise RuntimeError("Empty progress evaluation window")
    ptxt = f"Reason: {reason}\nSegments: {segments}\nFrames: {idx[pmask][0]}..{idx[pmask][-1]}\nN={pmask.sum()}\n"
    (outdir/"progress_window.txt").write_text(ptxt, encoding="utf-8")
    print("\n[Progress window]\n"+ptxt)

    if not MINECLIP_CKPT.exists():
        raise FileNotFoundError(MINECLIP_CKPT)
    tester = rv.MineCLIPTester(MINECLIP_CKPT)
    tester.text_feats(TARGET_PROMPTS)

    unet = None
    if RUN_ORIGINAL_UNET and UNET_CKPT.exists():
        try:
            unet = rv.OriginalUNet(UNET_CKPT, tester)
            print("[OK] Original U-Net loaded (reference only)")
        except Exception as e:
            print("[WARN] U-Net load failed:", type(e).__name__, e)

    # -------------------------------------------------------------
    # Pass 1: raw similarity extraction
    # -------------------------------------------------------------
    sim_store = defaultdict(list)
    unet_maps = []
    meta_rows = []
    gh = gw = None
    t0 = time.perf_counter()

    for pi, fi in enumerate(idx):
        frame = frames[int(fi)]
        meta_rows.append(dict(frame_index=int(fi), processed_index=int(pi), is_progress_eval_frame=int(pmask[pi]), **metadata_for_frame(saved, int(fi))))
        with torch.inference_mode():
            x, last, h, w = tester.encode_to_last(frame)
            if gh is None: gh, gw = h, w
            value = tester.raw_value_patch(x, last) if (RUN_CURRENT_TEMPORAL or RUN_VALUE_NO_TEMPORAL) else None

            if RUN_CURRENT_TEMPORAL:
                sim_store["current_temporal_l1"].append(cosine_vector(tester, tester.temporal_l1(value), TARGET_PROMPTS))
            if RUN_VALUE_NO_TEMPORAL:
                sim_store["value_no_temporal"].append(cosine_vector(tester, value, TARGET_PROMPTS))

            nac = tester.naclip_patch(x, last, h, w) if (RUN_NACLIP_NO_TEMPORAL or RUN_NACLIP_TEMPORAL) else None
            if RUN_NACLIP_NO_TEMPORAL:
                sim_store["naclip_no_temporal"].append(cosine_vector(tester, nac, TARGET_PROMPTS))
            if RUN_NACLIP_TEMPORAL:
                # 专门的 factorial ablation：NACLIP dense feature 再过 MineCLIP Temporal(L=1) adapter。
                sim_store["naclip_temporal_l1"].append(cosine_vector(tester, tester.temporal_l1(nac), TARGET_PROMPTS))

        if unet is not None:
            unet_maps.append(unet.generate(frame, TARGET_PROMPTS))
        if pi == 0 or (pi+1)%10 == 0 or pi+1 == len(idx):
            print(f"[extract {pi+1:4d}/{len(idx):4d}] frame={fi}")

    for m in list(sim_store):
        sim_store[m] = np.stack(sim_store[m], 0).astype(np.float32)
    unet_maps = np.stack(unet_maps, 0).astype(np.float32) if unet_maps else None
    print(f"Feature extraction elapsed: {time.perf_counter()-t0:.2f}s")

    wg, wm = exact_patch_weights(gh, gw)
    print(f"Patch weights sanity: gaussian_sum={wg.sum():.6f}, mean_sum={wm.sum():.6f}")

    allraw = np.concatenate([x.reshape(-1) for x in sim_store.values()])
    raw_vmin = float(np.percentile(allraw, RAW_VIS_LOW_PERCENTILE))
    raw_vmax = float(np.percentile(allraw, RAW_VIS_HIGH_PERCENTILE))
    if raw_vmax <= raw_vmin: raw_vmax = raw_vmin + 1e-3
    allexcess = np.maximum(allraw - BASE_THRESHOLD, 0.0)
    excess_vmax = float(np.percentile(allexcess, 99.0)) if np.any(np.isfinite(allexcess)) else 1e-3
    if excess_vmax <= 0: excess_vmax = 1e-3

    # -------------------------------------------------------------
    # Base calibration per-frame data
    # -------------------------------------------------------------
    per_frame = []
    method_scores = {}
    frame_scores = [dict() for _ in idx]
    base_rel = {}

    for method, sim3 in sim_store.items():
        sim2 = sim3.reshape(len(idx), -1).astype(np.float64)
        dd = {name: [] for name in PROGRESS_SCORE_NAMES}
        base_rel[method] = stable_sigmoid((sim2-BASE_THRESHOLD)/BASE_TEMPERATURE).reshape(len(idx), gh, gw).astype(np.float32)
        for pi, fi in enumerate(idx):
            s = progress_scores(sim2[pi], wg, wm, BASE_THRESHOLD, BASE_TEMPERATURE)
            frame_scores[pi][method] = s
            for name in PROGRESS_SCORE_NAMES: dd[name].append(s[name])
            per_frame.append(dict(**meta_rows[pi], method=method, display_name=METHOD_DISPLAY.get(method,method), base_threshold=BASE_THRESHOLD, base_temperature=BASE_TEMPERATURE, **s))
        method_scores[method] = {k: np.asarray(v, np.float64) for k,v in dd.items()}

    unet_g = None
    if unet_maps is not None:
        unet_g = (unet_maps * CENTER_GAUSSIAN[None]).sum((1,2)) / CENTER_GAUSSIAN.sum()
        for pi, fi in enumerate(idx):
            per_frame.append(dict(**meta_rows[pi], method="unet_original", display_name="Original LS-Imagine U-Net", unet_gaussian_score=float(unet_g[pi])))

    if not SAVE_ONLY_KEY_OUTPUTS or SAVE_FULL_DETAILS:
        save_csv(outdir/"per_frame_progress.csv", per_frame)
        save_csv(outdir/"trajectory_metadata.csv", meta_rows)

    summary = build_progress_summary(method_scores, idx, pmask)

    # U-Net 正式进入同一套 progress metrics。
    # 它只有自己的 Gaussian map score，因此 score_name=unet_gaussian，
    # 但序列评价完全复用 series_metrics：Spearman / PairAcc / Gain / Effect 等。
    if unet_g is not None:
        unet_metrics = series_metrics(idx[pmask], unet_g[pmask])
        summary.append(dict(
            method="unet_original",
            display_name="Original LS-Imagine U-Net",
            score_name="unet_gaussian",
            **unet_metrics,
        ))

    save_csv(outdir/"progress_summary.csv", summary)
    save_key_summary_csv(outdir, summary)
    # factorial 仍只比较四条 MineCLIP patch 分支；U-Net 没有对应 raw/excess score，
    # 不强行塞进 factorial comparison。
    save_csv(outdir/"factorial_comparison.csv", build_factorial(summary))
    save_key_metric_bars(outdir, summary)
    save_compact_per_frame(outdir, idx, meta_rows, method_scores, unet_g)

    # Stage summary 仅在 full details 模式保存。
    if not SAVE_ONLY_KEY_OUTPUTS or SAVE_FULL_DETAILS:
        stage_rows = []
        labels = np.asarray([str(r.get("saved_stage_label", "")) for r in meta_rows], dtype=object)
        for stage in [s for s in sorted(set(labels.tolist())) if s.strip()]:
            sm = labels == stage
            for method, dd in method_scores.items():
                for score_name in PROGRESS_SCORE_NAMES:
                    v = dd[score_name][sm]
                    if len(v):
                        stage_rows.append(dict(stage=stage, method=method, score_name=score_name, n=len(v), mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max())))
            if unet_g is not None:
                v = unet_g[sm]
                if len(v):
                    stage_rows.append(dict(
                        stage=stage, method="unet_original", score_name="unet_gaussian",
                        n=len(v), mean=float(v.mean()), std=float(v.std()),
                        min=float(v.min()), max=float(v.max())
                    ))
        save_csv(outdir/"stage_summary.csv", stage_rows)

    # Current recomputation consistency with trajectory.npz saved relevance_patch.
    # 关键：这里使用 trajectory 当时真实保存的 tau/T。
    consistency = []
    saved_rel = saved.get("relevance_patch")
    if (
        saved_rel is not None
        and "current_temporal_l1" in sim_store
        and saved_rel.ndim >= 3
    ):
        current_sim2 = sim_store["current_temporal_l1"].reshape(len(idx), -1).astype(np.float64)
        recorded_recomputed_rel = stable_sigmoid(
            (current_sim2 - recorded_tau) / recorded_temp
        ).reshape(len(idx), gh, gw)

        for pi, fi in enumerate(idx):
            if fi >= saved_rel.shape[0]:
                continue
            a = np.asarray(saved_rel[int(fi)], np.float64)
            b = np.asarray(recorded_recomputed_rel[pi], np.float64)
            if a.shape == b.shape:
                consistency.append(dict(
                    frame_index=int(fi),
                    recorded_threshold=float(recorded_tau),
                    recorded_temperature=float(recorded_temp),
                    mae=float(np.mean(np.abs(a-b))),
                    pearson=safe_pearson(a,b),
                    spearman=safe_spearman(a,b),
                ))
    save_csv(outdir/"recorded_current_consistency.csv", consistency)

    save_curves(outdir, idx, method_scores, pmask, unet_g)

    # -------------------------------------------------------------
    # Parameter sweep
    # -------------------------------------------------------------
    sweep_rows, excess_rows = [], []
    if RUN_PARAMETER_SWEEP:
        ts = time.perf_counter()
        for method, sim3 in sim_store.items():
            sim2 = sim3.reshape(len(idx), -1).astype(np.float64)
            print(f"[sweep] {method}: {len(thresholds_for_method(sim2))} thresholds x {len(TEMPERATURES)} temperatures")
            sweep_rows += sweep_method(method, sim2, idx, pmask, wg, wm)
            excess_rows += sweep_excess(method, sim2, idx, pmask, wg, wm)
        if not SAVE_ONLY_KEY_OUTPUTS or SAVE_FULL_DETAILS:
            save_csv(outdir/"calibration_sweep.csv", sweep_rows)
            save_csv(outdir/"excess_threshold_sweep.csv", excess_rows)
            save_sweep_heatmaps(outdir, sweep_rows)
        save_csv(outdir/"sweep_top_candidates.csv", top_sweep(sweep_rows))
        print(f"Parameter sweep elapsed: {time.perf_counter()-ts:.2f}s")

    # -------------------------------------------------------------
    # Every-frame visualization
    # -------------------------------------------------------------
    if SAVE_EVERY_FRAME_VISUALIZATION or SAVE_SELECTED_FRAME_VISUALIZATION:
        tv = time.perf_counter()
        order = [m for m in METHOD_DISPLAY if m in sim_store]
        selected_for_vis = set(save_selected_frame_indices(outdir, idx)) if SAVE_SELECTED_FRAME_VISUALIZATION else set()
        for pi, fi in enumerate(idx):
            if (not SAVE_EVERY_FRAME_VISUALIZATION) and (int(fi) not in selected_for_vis):
                continue
            sims = OrderedDict((m, sim_store[m][pi]) for m in order)
            ref = None if unet_maps is None else unet_maps[pi]
            save_frame_panel(visdir/f"frame_{int(fi):06d}.png", frames[int(fi)], int(fi), sims, frame_scores[pi], ref, raw_vmin, raw_vmax, excess_vmax)
            if pi == 0 or (pi + 1) % 20 == 0 or pi + 1 == len(idx):
                print(f"[visualize {pi + 1:4d}/{len(idx):4d}] frame={fi}")
        print(f"Visualization elapsed: {time.perf_counter()-tv:.2f}s")

    # -------------------------------------------------------------
    # Save raw arrays for future re-analysis without rerunning MineCLIP
    # -------------------------------------------------------------
    if SAVE_RAW_ARRAYS and (not SAVE_ONLY_KEY_OUTPUTS or SAVE_FULL_DETAILS):
        payload = dict(frame_indices=idx, progress_mask=pmask.astype(np.uint8))
        for m, a in sim_store.items(): payload[m+"_raw_similarity"] = a.astype(np.float32)
        np.savez_compressed(outdir/"raw_patch_similarity.npz", **payload)
        payload2 = dict(frame_indices=idx)
        for m, a in base_rel.items(): payload2[m+"_relevance"] = a.astype(np.float32)
        np.savez_compressed(outdir/"default_relevance_patch.npz", **payload2)

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    lookup = {(r["method"], r["score_name"]): r for r in summary}
    important = ["raw_p95", "sigmoid_gaussian", "excess_gaussian", "softplus_gaussian"]
    print("\n" + "="*88 + "\nBASE CALIBRATION PROGRESS SUMMARY\n" + "="*88)
    for method in METHOD_DISPLAY:
        if method not in method_scores: continue
        print("\n[" + METHOD_DISPLAY[method] + "]")
        for name in important:
            r = lookup[(method, name)]
            print(f"  {name:22s} Spearman={r['spearman_to_progress']:+.4f} | PairAcc={r['pair_order_accuracy']:.4f} | Gain={r['start_end_gain']:+.6f} | Effect={r['effect_size_gain_over_std']:+.4f}")

    if unet_g is not None:
        ur = lookup.get(("unet_original", "unet_gaussian"))
        if ur is not None:
            print("\n[Original LS-Imagine U-Net | unet_gaussian]")
            print(
                f"  Spearman={ur['spearman_to_progress']:+.4f} | "
                f"PairAcc={ur['pair_order_accuracy']:.4f} | "
                f"Gain={ur['start_end_gain']:+.6f} | "
                f"Effect={ur['effect_size_gain_over_std']:+.4f}"
            )

    if consistency:
        print("\n[Recorded-current consistency]")
        print("  MAE mean     :", float(np.mean([r["mae"] for r in consistency])))
        print("  Spearman mean:", float(np.nanmean([r["spearman"] for r in consistency])))

    print("\n主要看：")
    print("  progress_summary.csv")
    print("  factorial_comparison.csv")
    print("  per_frame_progress.csv")
    print("  calibration_sweep.csv")
    print("  sweep_top_candidates.csv")
    print("  excess_threshold_sweep.csv")
    print("  parameter_sweep_heatmaps/")
    print("  visualizations/  (每一步一张)")
    print("  raw_patch_similarity.npz")
    print("  README_zh.txt")
    print("\nOutput:", outdir)
    print("="*88)


if __name__ == "__main__":
    main()
