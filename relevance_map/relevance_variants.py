"""
Independent relevance-map ablation test for LS-Imagine / MineCLIP.

Put this file at <LS-Imagine>/relevance_map/test_relevance_variants.py and run it
normally in PyCharm. There is no argparse; edit the GLOBAL CONFIG section.

Compared methods:
  current_temporal_l1 : current code, Value -> Temporal(L=1) -> cosine
  no_temporal         : Value -> cosine
  competitive         : no_temporal + CLIP4MC-style target-vs-negative filtering
  naclip              : NACLIP-style reduced final block (K-K + Gaussian -> V)
  naclip_competitive  : optional combination
  unet_original       : original task-specific LS-Imagine U-Net reference
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# =====================================================================
# GLOBAL CONFIG -- edit here in PyCharm
# =====================================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # file is expected under <project>/relevance_map/
# If needed, replace the previous line manually, e.g.:
# PROJECT_ROOT = Path(r"D:\\work\\LS-Imagine")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LS-Imagine has modules that resolve files relative to the current working
# directory (for example envs/tasks/task_specs.yaml). PyCharm commonly starts
# this file from relevance_map/, so match real_approach_test.py and force cwd
# to the project root before importing envs.* modules.
os.chdir(PROJECT_ROOT)

TASK_SPECS_PATH = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS_PATH.exists():
    raise FileNotFoundError(
        f"PROJECT_ROOT looks wrong; cannot find: {TASK_SPECS_PATH}"
    )

TASK_NAME = "harvest_log_in_plains"
NPZ_PATH = (
    PROJECT_ROOT / "relevance_map" / "real_approach_runs"
    / TASK_NAME / "world_42_taskseed_0_20260904_183014"
    / "trajectory.npz"
)
MINECLIP_CKPT = PROJECT_ROOT / "weights" / "mineclip_attn.pth"
UNET_CKPT = (
    PROJECT_ROOT / "affordance_map" / "finetune_unet" / "finetune_checkpoints"
    / TASK_NAME / "swin_unet_checkpoint.pth"
)
OUTPUT_ROOT = PROJECT_ROOT / "relevance_map" / "relevance_variants_test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable/disable each ablation independently.
RUN_ORIGINAL_UNET = True
RUN_CURRENT_TEMPORAL_L1 = True
RUN_NO_TEMPORAL = True
RUN_COMPETITIVE = True
RUN_NACLIP = True
RUN_NACLIP_COMPETITIVE = True

# trajectory sampling
RGB_KEY_CANDIDATES = [
    "rgb", "image", "images", "obs_rgb", "observation_rgb",
    "frames", "frame", "observations", "obs",
]
FRAME_START = 0
FRAME_END = None
FRAME_STRIDE = 1
MAX_FRAMES = 1000

IMAGE_H, IMAGE_W = 160, 256

# Keep the same task prompt first, so each ablation is clean.
TARGET_PROMPTS = ["Cut a tree"]
COMPETITIVE_TARGET_PROMPTS = list(TARGET_PROMPTS)
# Second-round experiment suggestion:
# COMPETITIVE_TARGET_PROMPTS = ["water"]

COMPETITIVE_NEGATIVE_PROMPTS = [
    "sky", "cloud", "grass", "dirt", "stone", "sand", "gravel",
    "flower", "sheep", "cow", "pig",
    "chicken", "horse", "village", "mountain", "snow", "lava",
    "coal ore", "iron ore", "crafting table", "furnace", "torch",
    "player hand", "bucket",
]

# Current map calibration
RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016

# Competitive filtering
# soft_margin: R = R_abs * sigmoid((target - max_negative - margin_th) / margin_T)
# hard:        R = R_abs * 1[target > max_negative]
COMPETITIVE_MODE = "soft_margin"
COMPETITIVE_MARGIN_THRESHOLD = 0.0
COMPETITIVE_MARGIN_TEMPERATURE = 0.02

# NACLIP-style. Official NACLIP default gaussian_std is 5.0.
NACLIP_GAUSSIAN_STD = 5.0
NACLIP_GAUSSIAN_WEIGHT = 1.0
NACLIP_INCLUDE_CLS = True

# Original U-Net
UNET_MODULE = "envs.tasks.base.concentration_reward"
UNET_BLUR_KERNEL = (51, 79)

# Metrics/visualization
TOP_FRACTION = 0.10
GAUSSIAN_SIGMA_WEIGHT = 0.5
SAVE_VISUALIZATIONS = True
VIS_EVERY_N_PROCESSED_FRAMES = 5
MAX_VISUALIZATIONS = 30
VIS_INCLUDE_DIFF_TO_UNET = True
SAVE_RAW_MAPS = True
OVERLAY_ALPHA = 0.48

MINECLIP_CONFIG = dict(
    arch="vit_base_p16_fz.v2.t2",
    hidden_dim=512,
    image_feature_dim=512,
    mlp_adapter_spec="v0-2.t0",
    pool_type="attn.d2.nh8.glusw",
    resolution=[IMAGE_H, IMAGE_W],
)

DISPLAY_NAMES = {
    "unet_original": "Original LS-Imagine U-Net",
    "current_temporal_l1": "Current: Value + Temporal(L=1)",
    "no_temporal": "Ablation 1: No Temporal(L=1)",
    "competitive": "Ablation 2: Competitive",
    "naclip": "Ablation 3: NACLIP-style",
    "naclip_competitive": "Combo: NACLIP + Competitive",
}

# =====================================================================
# Trajectory loading
# =====================================================================
def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame)
    while x.ndim > 3 and 1 in x.shape:
        x = np.squeeze(x, axis=list(x.shape).index(1))
    if x.ndim != 3:
        raise ValueError(f"frame is not 3D after squeeze: {x.shape}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    if x.shape[-1] != 3:
        raise ValueError(f"cannot find RGB channel: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        if np.nanmax(x) <= 1.5:
            x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    if x.shape[:2] != (IMAGE_H, IMAGE_W):
        x = cv2.resize(x, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(x)


def object_array_frames(arr: np.ndarray) -> Optional[List[np.ndarray]]:
    frames = []
    for item in arr.reshape(-1):
        if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
            try:
                item = item.item()
            except Exception:
                pass
        if isinstance(item, dict):
            for key in ("rgb", "image", "frame", "obs_rgb"):
                if key in item:
                    frames.append(ensure_rgb(item[key]))
                    break
    return frames or None


def load_frames(path: Path) -> Tuple[List[np.ndarray], str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    info = OrderedDict()
    for key in data.files:
        a = data[key]
        info[key] = f"shape={getattr(a, 'shape', None)}, dtype={getattr(a, 'dtype', None)}"

    def try_key(key):
        if key not in data.files:
            return None
        a = data[key]
        if getattr(a, "dtype", None) == object:
            out = object_array_frames(a)
            if out:
                return out
        x = np.asarray(a)
        if x.ndim == 4 and (x.shape[-1] == 3 or x.shape[1] == 3):
            return [ensure_rgb(x[i]) for i in range(x.shape[0])]
        if x.ndim == 3 and (x.shape[-1] == 3 or x.shape[0] == 3):
            return [ensure_rgb(x)]
        if x.ndim == 5:
            y = np.squeeze(x)
            if y.ndim == 4 and (y.shape[-1] == 3 or y.shape[1] == 3):
                return [ensure_rgb(y[i]) for i in range(y.shape[0])]
        return None

    for key in RGB_KEY_CANDIDATES + [k for k in data.files if k not in RGB_KEY_CANDIDATES]:
        out = try_key(key)
        if out:
            return out, key, info
    raise RuntimeError("No RGB array found in npz. Keys:\n" + "\n".join(f"{k}: {v}" for k, v in info.items()))


def load_saved_diagnostics(path: Path) -> Dict[str, np.ndarray]:
    """Load optional fields written by real_approach_test.py."""
    wanted = [
        "cosines", "relevance_patch", "target_lidar_distance",
        "target_lidar_visible", "stage_labels", "positions", "reward",
        "target_inventory_quantity", "prompt", "threshold", "temperature",
    ]
    out = {}
    with np.load(path, allow_pickle=True) as data:
        for key in wanted:
            if key in data.files:
                out[key] = np.asarray(data[key])
    return out


def recorded_relevance_full(saved: Dict[str, np.ndarray], frame_index: int) -> Optional[np.ndarray]:
    """Upsample saved [10,16] relevance_patch to [160,256]."""
    arr = saved.get("relevance_patch")
    if arr is None or arr.ndim < 3 or frame_index >= arr.shape[0]:
        return None
    small = np.asarray(arr[frame_index], dtype=np.float32)
    if small.shape != (10, 16):
        return None
    big = F.interpolate(
        torch.from_numpy(small)[None, None],
        size=(IMAGE_H, IMAGE_W),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    return big.astype(np.float32)


def trajectory_meta(saved: Dict[str, np.ndarray], frame_index: int) -> Dict[str, object]:
    """Attach manual-trajectory distance/stage/position to output CSV rows."""
    meta: Dict[str, object] = {}

    def scalar_at(key, default=np.nan):
        a = saved.get(key)
        if a is None or a.ndim == 0 or frame_index >= len(a):
            return default
        v = np.asarray(a[frame_index]).reshape(-1)
        return v[0].item() if len(v) else default

    meta["saved_target_lidar_distance"] = scalar_at("target_lidar_distance")
    meta["saved_target_lidar_visible"] = scalar_at("target_lidar_visible")
    meta["saved_reward"] = scalar_at("reward")
    meta["saved_target_inventory_quantity"] = scalar_at("target_inventory_quantity")

    labels = saved.get("stage_labels")
    meta["saved_stage_label"] = (
        str(labels[frame_index])
        if labels is not None and labels.ndim > 0 and frame_index < len(labels)
        else ""
    )

    pos = saved.get("positions")
    if pos is not None and pos.ndim >= 2 and frame_index < pos.shape[0] and pos.shape[1] >= 3:
        meta["saved_pos_x"] = float(pos[frame_index, 0])
        meta["saved_pos_y"] = float(pos[frame_index, 1])
        meta["saved_pos_z"] = float(pos[frame_index, 2])
    else:
        meta["saved_pos_x"] = np.nan
        meta["saved_pos_y"] = np.nan
        meta["saved_pos_z"] = np.nan

    return meta


def selected_indices(n: int) -> List[int]:
    end = n if FRAME_END is None else min(n, int(FRAME_END))
    idx = list(range(max(0, int(FRAME_START)), end, max(1, int(FRAME_STRIDE))))
    if MAX_FRAMES is not None:
        idx = idx[:int(MAX_FRAMES)]
    if not idx:
        raise RuntimeError("No frame selected.")
    return idx

# =====================================================================
# Metrics
# =====================================================================
def make_center_gaussian() -> np.ndarray:
    sx, sy = IMAGE_W * GAUSSIAN_SIGMA_WEIGHT, IMAGE_H * GAUSSIAN_SIGMA_WEIGHT
    x = np.linspace(-IMAGE_W // 2, IMAGE_W // 2, IMAGE_W)
    y = np.linspace(-IMAGE_H // 2, IMAGE_H // 2, IMAGE_H)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx**2 / (2*sx**2) + yy**2 / (2*sy**2))).astype(np.float32)

CENTER_GAUSSIAN = make_center_gaussian()
CENTER_GAUSSIAN_MEAN = float(CENTER_GAUSSIAN.mean())


def gaussian_score(m):
    return float(np.mean(m * CENTER_GAUSSIAN) / CENTER_GAUSSIAN_MEAN)


def safe_pearson(a, b):
    x, y = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(a, b):
    x, y = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    # SciPy compatibility:
    # newer SciPy exposes .statistic; older SciPy exposes .correlation.
    result = spearmanr(x, y)
    if hasattr(result, "statistic"):
        rho = result.statistic
    elif hasattr(result, "correlation"):
        rho = result.correlation
    else:
        rho = result[0]
    return float(rho)


def top_mask(m, fraction=TOP_FRACTION):
    x = np.asarray(m).reshape(-1)
    k = min(x.size, max(1, int(round(x.size * fraction))))
    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros(x.size, np.uint8)
    out[idx] = 1
    return out.reshape(m.shape)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)


def weighted_center(m):
    x = np.asarray(m, np.float64)
    s = x.sum()
    if s <= 1e-12:
        return IMAGE_W / 2, IMAGE_H / 2
    ys, xs = np.indices(x.shape)
    return float((xs*x).sum()/s), float((ys*x).sum()/s)


def peak_xy(m):
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return float(x), float(y)


def norm_dist(p, q):
    return float(math.hypot(p[0]-q[0], p[1]-q[1]) / math.hypot(IMAGE_W, IMAGE_H))


def largest_cc(m, th=0.5):
    binary = (np.asarray(m) > th).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / binary.size)


def get_map_stats(m):
    x = np.asarray(m, np.float64)
    p50, p95 = np.percentile(x, [50, 95])
    return dict(
        mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
        p50=float(p50), p95=float(p95), p95_minus_p50=float(p95-p50),
        gaussian_score=gaussian_score(x), largest_cc_fraction_at_0_5=largest_cc(x, 0.5),
    )


def compare_ref(pred, ref):
    return dict(
        pearson_vs_unet=safe_pearson(pred, ref),
        spearman_vs_unet=safe_spearman(pred, ref),
        mae_vs_unet=float(np.mean(np.abs(pred-ref))),
        top10_iou_vs_unet=iou(top_mask(pred), top_mask(ref)),
        weighted_center_dist_norm_vs_unet=norm_dist(weighted_center(pred), weighted_center(ref)),
        peak_dist_norm_vs_unet=norm_dist(peak_xy(pred), peak_xy(ref)),
        gaussian_score_abs_diff_vs_unet=abs(gaussian_score(pred)-gaussian_score(ref)),
    )


def compare_current(pred, current):
    """Same metrics, but use the user's current Temporal(L=1) map as reference."""
    q = compare_ref(pred, current)
    return {k.replace("_vs_unet", "_vs_current"): v for k, v in q.items()}

# =====================================================================
# MineCLIP methods
# =====================================================================
class MineCLIPTester:
    def __init__(self, ckpt: Path):
        from mineclip import MineCLIP
        from mineclip.mineclip.base import MC_IMAGE_MEAN, MC_IMAGE_STD
        import mineclip.utils as U
        from omegaconf import OmegaConf
        self.device = torch.device(DEVICE)
        self.U, self.mean, self.std = U, MC_IMAGE_MEAN, MC_IMAGE_STD
        self.model = MineCLIP(**OmegaConf.create(MINECLIP_CONFIG)).to(self.device)
        self.model.load_ckpt(str(ckpt), strict=True)
        self.model.eval()
        self.vit = self.model.clip_model.vision_model
        self.text_cache = {}

    @torch.inference_mode()
    def text_feats(self, prompts: List[str]):
        missing = [p for p in prompts if p not in self.text_cache]
        if missing:
            feats = self.model.encode_text(missing)
            for p, f in zip(missing, feats):
                self.text_cache[p] = f.detach()
        return torch.stack([self.text_cache[p] for p in prompts], 0)

    @torch.inference_mode()
    def encode_to_last(self, frame):
        t = torch.from_numpy(frame).permute(2,0,1).contiguous().unsqueeze(0).to(self.device)
        t = self.U.basic_image_tensor_preprocess(t, mean=self.mean, std=self.std)
        x = self.vit.conv1(t)
        B, _, gh, gw = x.shape
        x = x.reshape(B, x.shape[1], -1).permute(0,2,1)
        x = torch.cat([self.vit.cls_token.repeat(B,1,1), x], 1)
        x = self.vit.ln_pre(x + self.vit.pos_embed).permute(1,0,2)
        blocks = list(self.vit.blocks.children())
        for block in blocks[:-1]:
            x = block(x)
        return x, blocks[-1], int(gh), int(gw)

    @torch.inference_mode()
    def raw_value_patch(self, x, last_block):
        z = last_block.ln_1(x)
        attn = last_block.attn
        D = z.shape[-1]
        vw = attn.in_proj_weight[2*D:3*D]
        vb = None if attn.in_proj_bias is None else attn.in_proj_bias[2*D:3*D]
        v = F.linear(z, vw, vb)
        v = F.linear(v, attn.out_proj.weight, attn.out_proj.bias)
        p = self.vit.ln_post(v[1:].permute(1,0,2))
        if self.vit.projection is not None:
            p = p @ self.vit.projection
        return p

    @torch.inference_mode()
    def temporal_l1(self, patch):
        B, N, D = patch.shape
        return self.model.forward_video_features(patch.reshape(B*N,1,D)).reshape(B,N,-1)

    def gaussian_addition(self, gh, gw, dtype, device, include_cls=True):
        ys, xs = torch.meshgrid(torch.arange(gh, device=device, dtype=torch.float32),
                                torch.arange(gw, device=device, dtype=torch.float32), indexing="ij")
        c = torch.stack([ys.reshape(-1), xs.reshape(-1)], 1)
        d = c[:,None,:] - c[None,:,:]
        d2 = (d*d).sum(-1)
        om = torch.exp(-d2 / (2 * float(NACLIP_GAUSSIAN_STD)**2))
        if not include_cls:
            return om.to(dtype)
        out = torch.zeros((gh*gw+1, gh*gw+1), device=device, dtype=torch.float32)
        out[1:,1:] = om
        return out.to(dtype)

    @torch.inference_mode()
    def naclip_patch(self, x, last_block, gh, gw):
        # Adaptation of official NACLIP final reduced block:
        # softmax(KK^T * d^-1/2 + Gaussian) @ V, then out_proj, no residual/FFN.
        z = last_block.ln_1(x)
        attn = last_block.attn
        L, B, D = z.shape
        H = attn.num_heads
        hd = D // H
        scale = hd ** -0.5
        q, k, v = F.linear(z, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
        k = k.contiguous().view(L, B*H, hd).transpose(0,1)
        v = v.contiguous().view(L, B*H, hd).transpose(0,1)

        if NACLIP_INCLUDE_CLS:
            w = torch.bmm(k, k.transpose(1,2)) * scale
            om = self.gaussian_addition(gh, gw, w.dtype, w.device, True)
            w = F.softmax(w + float(NACLIP_GAUSSIAN_WEIGHT)*om.unsqueeze(0), -1)
            out = torch.bmm(w, v).transpose(0,1).contiguous().view(L,B,D)
            out = attn.out_proj(out).permute(1,0,2)
            out = self.vit.ln_post(out)
            if self.vit.projection is not None:
                out = out @ self.vit.projection
            return out[:,1:,:]

        kp, vp = k[:,1:,:], v[:,1:,:]
        w = torch.bmm(kp, kp.transpose(1,2)) * scale
        om = self.gaussian_addition(gh, gw, w.dtype, w.device, False)
        w = F.softmax(w + float(NACLIP_GAUSSIAN_WEIGHT)*om.unsqueeze(0), -1)
        N = gh*gw
        out = torch.bmm(w, vp).transpose(0,1).contiguous().view(N,B,D)
        out = attn.out_proj(out).permute(1,0,2)
        out = self.vit.ln_post(out)
        if self.vit.projection is not None:
            out = out @ self.vit.projection
        return out

    @torch.inference_mode()
    def cosine(self, patch, prompts):
        p = F.normalize(patch, dim=-1)
        t = F.normalize(self.text_feats(prompts), dim=-1)
        return torch.einsum("bnd,pd->bpn", p, t)

    def upsample_vector(self, v, gh, gw):
        x = F.interpolate(v.reshape(1,1,gh,gw), (IMAGE_H,IMAGE_W), mode="bilinear", align_corners=False)
        return x[0,0].detach().cpu().numpy().astype(np.float32)

    def normal_map(self, patch, prompts, gh, gw):
        sim = self.cosine(patch, prompts)
        r = torch.sigmoid((sim-RELEVANCE_THRESHOLD)/RELEVANCE_TEMPERATURE).max(1).values[0]
        return self.upsample_vector(r, gh, gw)

    def competitive_map(self, patch, target_prompts, negative_prompts, gh, gw):
        negative_prompts = [p for p in negative_prompts if p not in target_prompts]
        ts = self.cosine(patch, target_prompts).max(1).values[0]
        ns = self.cosine(patch, negative_prompts).max(1).values[0]
        margin = ts - ns
        abs_r = torch.sigmoid((ts-RELEVANCE_THRESHOLD)/RELEVANCE_TEMPERATURE)
        if COMPETITIVE_MODE == "soft_margin":
            mr = torch.sigmoid((margin-COMPETITIVE_MARGIN_THRESHOLD)/COMPETITIVE_MARGIN_TEMPERATURE)
            r = abs_r * mr
        elif COMPETITIVE_MODE == "hard":
            r = abs_r * (ts > ns).to(abs_r.dtype)
        else:
            raise ValueError("COMPETITIVE_MODE must be 'soft_margin' or 'hard'")
        debug = {
            "target_similarity": self.upsample_vector(ts, gh, gw),
            "max_negative_similarity": self.upsample_vector(ns, gh, gw),
            "margin": self.upsample_vector(margin, gh, gw),
        }
        return self.upsample_vector(r, gh, gw), debug

# =====================================================================
# Original U-Net reference
# =====================================================================
class OriginalUNet:
    def __init__(self, ckpt: Path, mineclip: MineCLIPTester):
        module = importlib.import_module(UNET_MODULE)
        Config, MCUnet = getattr(module, "Config"), getattr(module, "MCUnet")
        cfg = Config()
        self.device = torch.device(DEVICE)
        self.model = MCUnet(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=1).to(self.device)
        state = torch.load(str(ckpt), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.mineclip = mineclip
        self.mean = np.array([0.3331,0.3245,0.3051], np.float32)
        self.std = np.array([0.2439,0.2493,0.2873], np.float32)

    @torch.inference_mode()
    def generate(self, frame, prompts):
        img = cv2.resize(frame, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        img = (img-self.mean)/self.std
        img = torch.from_numpy(img).permute(2,0,1).contiguous().unsqueeze(0).float().to(self.device)
        text = self.mineclip.text_feats(prompts).to(self.device).float()
        out = self.model(img.expand(text.shape[0],-1,-1,-1), text).squeeze(1).detach().cpu().numpy()
        maps = []
        for p in out:
            m = cv2.resize(p, (IMAGE_W,IMAGE_H), interpolation=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, UNET_BLUR_KERNEL, 0)
            maps.append(m.astype(np.float32))
        return np.max(np.stack(maps,0), axis=0).astype(np.float32)

# =====================================================================
# Output helpers
# =====================================================================
def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def save_panel(path, frame, maps: OrderedDict, frame_idx, frame_metrics):
    has_ref = "unet_original" in maps
    diff_row = has_ref and VIS_INCLUDE_DIFF_TO_UNET
    rows, cols = (3 if diff_row else 2), len(maps)+1
    fig, ax = plt.subplots(rows, cols, figsize=(4*cols,3.3*rows), squeeze=False)
    ax[0,0].imshow(frame); ax[0,0].set_title(f"RGB | frame={frame_idx}"); ax[0,0].axis("off")
    ax[1,0].imshow(frame); ax[1,0].set_title("RGB reference"); ax[1,0].axis("off")
    if diff_row: ax[2,0].axis("off"); ax[2,0].set_title("|Map - U-Net|")
    ref = maps.get("unet_original")
    for c,(name,m) in enumerate(maps.items(),1):
        s = get_map_stats(m)
        title = f"{DISPLAY_NAMES.get(name,name)}\nP95-P50={s['p95_minus_p50']:.3f} | G={s['gaussian_score']:.3f}"
        if name in frame_metrics:
            q = frame_metrics[name]
            title += f"\nSpearman={q['spearman_vs_unet']:.3f} | Top10IoU={q['top10_iou_vs_unet']:.3f}"
        ax[0,c].imshow(m,cmap="jet",vmin=0,vmax=1); ax[0,c].set_title(title); ax[0,c].axis("off")
        ax[1,c].imshow(frame); ax[1,c].imshow(m,cmap="jet",vmin=0,vmax=1,alpha=OVERLAY_ALPHA); ax[1,c].set_title("Overlay"); ax[1,c].axis("off")
        if diff_row:
            d = np.abs(m-ref)
            ax[2,c].imshow(d,cmap="magma",vmin=0,vmax=1); ax[2,c].set_title(f"Mean abs diff={d.mean():.3f}"); ax[2,c].axis("off")
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)


def agg(rows, method, key):
    vals = [float(r[key]) for r in rows if r.get("method")==method and key in r and r[key] not in (None,"")]
    if not vals or np.all(np.isnan(vals)):
        return float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def build_summary(ref_rows, stat_rows, temp_rows):
    methods = sorted(set(r["method"] for r in stat_rows))
    ref_keys = ["pearson_vs_unet","spearman_vs_unet","mae_vs_unet","top10_iou_vs_unet",
                "weighted_center_dist_norm_vs_unet","peak_dist_norm_vs_unet","gaussian_score_abs_diff_vs_unet"]
    stat_keys = ["p95_minus_p50","gaussian_score","largest_cc_fraction_at_0_5"]
    temp_keys = ["temporal_pearson","temporal_weighted_center_disp_norm"]
    out=[]
    for m in methods:
        row={"method":m,"display_name":DISPLAY_NAMES.get(m,m),"n_frames":sum(r["method"]==m for r in stat_rows)}
        for k in ref_keys:
            row[k+"_mean"],row[k+"_std"]=agg(ref_rows,m,k)
        for k in stat_keys:
            row[k+"_mean"],row[k+"_std"]=agg(stat_rows,m,k)
        for k in temp_keys:
            row[k+"_mean"],row[k+"_std"]=agg(temp_rows,m,k)
        out.append(row)
    return out


def summary_plots(outdir: Path, summary):
    rows=[r for r in summary if r["method"]!="unet_original" and not np.isnan(r.get("spearman_vs_unet_mean",np.nan))]
    if not rows: return
    names=[DISPLAY_NAMES.get(r["method"],r["method"]) for r in rows]; x=np.arange(len(rows)); width=.23
    fig,ax=plt.subplots(figsize=(max(10,2.8*len(rows)),5.8))
    for i,(k,label) in enumerate([("pearson_vs_unet_mean","Pearson"),("spearman_vs_unet_mean","Spearman"),("top10_iou_vs_unet_mean","Top-10% IoU")]):
        ax.bar(x+(i-1)*width,[r[k] for r in rows],width,label=label)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=18,ha="right"); ax.set_ylim(-.05,1.05); ax.set_ylabel("Higher is better"); ax.set_title("Agreement with original U-Net"); ax.legend(); ax.grid(axis="y",alpha=.25)
    plt.tight_layout(); fig.savefig(outdir/"summary_vs_unet_high_is_better.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(max(10,2.8*len(rows)),5.8))
    for i,(k,label) in enumerate([("mae_vs_unet_mean","MAE"),("weighted_center_dist_norm_vs_unet_mean","Weighted-center distance"),("peak_dist_norm_vs_unet_mean","Peak distance")]):
        ax.bar(x+(i-1)*width,[r[k] for r in rows],width,label=label)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=18,ha="right"); ax.set_ylabel("Lower is better"); ax.set_title("Distance / calibration error vs U-Net"); ax.legend(); ax.grid(axis="y",alpha=.25)
    plt.tight_layout(); fig.savefig(outdir/"summary_vs_unet_low_is_better.png",dpi=160,bbox_inches="tight"); plt.close(fig)


def curve_plots(outdir: Path, curves):
    for field,title,ylabel,filename in [
        ("gaussian_score","Gaussian score along trajectory","Gaussian-weighted map score","gaussian_score_curves.png"),
        ("p95_minus_p50","Map contrast along trajectory","P95 - P50","contrast_curves.png")]:
        fig,ax=plt.subplots(figsize=(12,6))
        for m,d in curves.items(): ax.plot(d["frame"],d[field],label=DISPLAY_NAMES.get(m,m))
        ax.set_xlabel("Trajectory frame index"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25); ax.legend()
        plt.tight_layout(); fig.savefig(outdir/filename,dpi=160,bbox_inches="tight"); plt.close(fig)

METRICS_ZH = r"""
指标中文说明
============

一、和原始 LS-Imagine task-specific U-Net 的对比（U-Net 是 reference，不是真值）

1) Pearson vs U-Net（越高越好）
   两张图逐像素数值的线性相关。受数值尺度/标定影响较大。

2) Spearman vs U-Net（越高越好，建议重点看）
   比较各区域“从低到高的排序”是否一致，不太在意绝对数值尺度。
   Natural Zoom 更关心哪个区域更值得看，所以这个指标比单纯 MAE 更重要。

3) MAE vs U-Net（越低越好）
   平均 |新图-U-Net图|。主要衡量数值标定接近程度，不等同于定位正确率。

4) Top-10% IoU vs U-Net（越高越好，建议重点看）
   分别取两张图最热的 10% 像素，计算交并比。
   直接回答“最值得 zoom 的区域是不是同一块”。

5) Weighted-center distance（越低越好）
   比较整张 relevance map 的热力重心，距离除以 160x256 图像对角线归一化。

6) Peak distance（越低越好）
   比较两张图最高热点 argmax 的位置，也按图像对角线归一化。
   对单点噪声较敏感，所以要结合 Weighted-center distance。

7) Gaussian-score absolute difference（越低越好，建议重点看）
   用 LS-Imagine 的中心 Gaussian 计算 map score，再比较新方法和 U-Net 的差值。
   它和后续 ConcentrationReward / intrinsic shaping signal 更直接相关。

二、每种图自己的诊断

8) P95-P50
   P95 是较高 relevance 区域水平，P50 是中位背景水平；差越大说明热点和背景更容易分开。
   但错误热点也会让它很大，因此必须结合可视化。

9) Largest CC Fraction @ 0.5
   relevance>0.5 后最大 8 连通区域占整张图比例。
   太碎可能表示 map 不稳定；太大可能表示整幅图泛激活。它受 calibration 影响，只作辅助。

10) Gaussian Score
    按 LS-Imagine 中心 Gaussian 加权后的 map score，用来观察整条 trajectory 的 shaping signal。

三、时序稳定性

11) Temporal Pearson
    相邻“被采样帧”的 relevance map 相关性。高通常表示更稳定，但视角真的改变时下降是正常的。

12) Temporal weighted-center displacement
    相邻采样帧热力重心移动距离（对角线归一化）。
    如果 RGB 变化不大但该值频繁很高，说明热点可能在闪烁/跳动。

推荐看结果的顺序：
A. 先看 visualizations：目标是不是 water？有没有 sky/grass/hand/bucket 假阳性？
B. 再看 Spearman、Top-10% IoU、Weighted-center distance、Gaussian-score diff。
C. 最后结合 Pearson/MAE、P95-P50 和 temporal stability。

四、vs 当前版本

程序还会额外生成 per_frame_vs_current.csv 和 summary_vs_current.csv。
其中 Spearman/Top-10% IoU/重心距离等定义完全相同，只是 reference 从 U-Net 换成了
你现在的 Value + Temporal(L=1) 版本。它适合回答“这个消融到底把当前 map 改了多少”。

注意：更像原 U-Net 不等于一定更好。最终仍要结合可视化、Natural Zoom crop 和 RL success rate。
"""

def build_vs_current_summary(rows):
    methods = sorted(set(r["method"] for r in rows))
    keys = ["pearson_vs_current","spearman_vs_current","mae_vs_current","top10_iou_vs_current",
            "weighted_center_dist_norm_vs_current","peak_dist_norm_vs_current","gaussian_score_abs_diff_vs_current"]
    out=[]
    for m in methods:
        row={"method":m,"display_name":DISPLAY_NAMES.get(m,m),"n_frames":sum(r["method"]==m for r in rows)}
        for k in keys:
            vals=np.asarray([float(r[k]) for r in rows if r["method"]==m and k in r],dtype=np.float64)
            row[k+"_mean"] = float(np.nanmean(vals)) if vals.size and not np.all(np.isnan(vals)) else float("nan")
            row[k+"_std"] = float(np.nanstd(vals)) if vals.size and not np.all(np.isnan(vals)) else float("nan")
        out.append(row)
    return out


# =====================================================================
# Main
# =====================================================================
def main():
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir=OUTPUT_ROOT/f"{TASK_NAME}_{timestamp}"
    visdir,rawdir=outdir/"visualizations",outdir/"raw_maps"
    outdir.mkdir(parents=True,exist_ok=False)
    if SAVE_VISUALIZATIONS: visdir.mkdir()
    if SAVE_RAW_MAPS: rawdir.mkdir()

    config={k:str(v) if isinstance(v,Path) else v for k,v in dict(
        PROJECT_ROOT=PROJECT_ROOT, NPZ_PATH=NPZ_PATH, MINECLIP_CKPT=MINECLIP_CKPT, UNET_CKPT=UNET_CKPT,
        TASK_NAME=TASK_NAME, DEVICE=DEVICE, TARGET_PROMPTS=TARGET_PROMPTS,
        COMPETITIVE_TARGET_PROMPTS=COMPETITIVE_TARGET_PROMPTS, COMPETITIVE_NEGATIVE_PROMPTS=COMPETITIVE_NEGATIVE_PROMPTS,
        RELEVANCE_THRESHOLD=RELEVANCE_THRESHOLD, RELEVANCE_TEMPERATURE=RELEVANCE_TEMPERATURE,
        COMPETITIVE_MODE=COMPETITIVE_MODE, COMPETITIVE_MARGIN_THRESHOLD=COMPETITIVE_MARGIN_THRESHOLD,
        COMPETITIVE_MARGIN_TEMPERATURE=COMPETITIVE_MARGIN_TEMPERATURE,
        NACLIP_GAUSSIAN_STD=NACLIP_GAUSSIAN_STD, NACLIP_GAUSSIAN_WEIGHT=NACLIP_GAUSSIAN_WEIGHT,
        FRAME_START=FRAME_START, FRAME_END=FRAME_END, FRAME_STRIDE=FRAME_STRIDE, MAX_FRAMES=MAX_FRAMES,
        RUN_ORIGINAL_UNET=RUN_ORIGINAL_UNET, RUN_CURRENT_TEMPORAL_L1=RUN_CURRENT_TEMPORAL_L1,
        RUN_NO_TEMPORAL=RUN_NO_TEMPORAL, RUN_COMPETITIVE=RUN_COMPETITIVE, RUN_NACLIP=RUN_NACLIP,
        RUN_NACLIP_COMPETITIVE=RUN_NACLIP_COMPETITIVE).items()}
    (outdir/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8")
    (outdir/"metrics_explained_zh.txt").write_text(METRICS_ZH,encoding="utf-8")

    print("="*80); print("Relevance-map ablation test"); print("OUTPUT:",outdir); print("DEVICE:",DEVICE)
    frames,rgb_key,keyinfo=load_frames(NPZ_PATH)
    saved_diag=load_saved_diagnostics(NPZ_PATH)
    (outdir/"npz_keys.txt").write_text("Selected RGB key: "+rgb_key+"\n\n"+"\n".join(f"{k}: {v}" for k,v in keyinfo.items()),encoding="utf-8")
    idxs=selected_indices(len(frames)); print(f"RGB key={rgb_key}; total={len(frames)}; tested={len(idxs)}")
    if saved_diag:
        print("Saved diagnostic fields:", ", ".join(sorted(saved_diag.keys())))
        if "prompt" in saved_diag:
            print("Saved prompt:", np.asarray(saved_diag["prompt"]).item())
        if "threshold" in saved_diag and "temperature" in saved_diag:
            print("Saved tau / T:", float(np.asarray(saved_diag["threshold"]).item()), "/", float(np.asarray(saved_diag["temperature"]).item()))

    if not MINECLIP_CKPT.exists(): raise FileNotFoundError(MINECLIP_CKPT)
    tester=MineCLIPTester(MINECLIP_CKPT)
    tester.text_feats(TARGET_PROMPTS)
    if RUN_COMPETITIVE or RUN_NACLIP_COMPETITIVE:
        tester.text_feats(COMPETITIVE_TARGET_PROMPTS); tester.text_feats(COMPETITIVE_NEGATIVE_PROMPTS)

    unet=None
    if RUN_ORIGINAL_UNET:
        if UNET_CKPT.exists():
            try:
                unet=OriginalUNet(UNET_CKPT,tester); print("[OK] Original U-Net loaded")
            except Exception as e:
                print(f"[WARN] U-Net load failed; continue without it: {type(e).__name__}: {e}")
        else:
            print("[WARN] U-Net checkpoint does not exist; skip:",UNET_CKPT)

    ref_rows=[]; current_rows=[]; recorded_rows=[]; stat_rows=[]; temp_rows=[]; previous={}; curves=defaultdict(lambda:{"frame":[],"gaussian_score":[],"p95_minus_p50":[]})
    nvis=0; t0=time.perf_counter()

    for pi,fi in enumerate(idxs):
        frame=frames[fi]; maps=OrderedDict(); debug={}; fs=time.perf_counter()
        meta=trajectory_meta(saved_diag,fi)
        with torch.inference_mode():
            x,last,gh,gw=tester.encode_to_last(frame)
            if unet is not None: maps["unet_original"]=unet.generate(frame,TARGET_PROMPTS)
            need_raw=RUN_CURRENT_TEMPORAL_L1 or RUN_NO_TEMPORAL or RUN_COMPETITIVE
            raw=tester.raw_value_patch(x,last) if need_raw else None
            if RUN_CURRENT_TEMPORAL_L1: maps["current_temporal_l1"]=tester.normal_map(tester.temporal_l1(raw),TARGET_PROMPTS,gh,gw)
            if RUN_NO_TEMPORAL: maps["no_temporal"]=tester.normal_map(raw,TARGET_PROMPTS,gh,gw)
            if RUN_COMPETITIVE:
                maps["competitive"],d=tester.competitive_map(raw,COMPETITIVE_TARGET_PROMPTS,COMPETITIVE_NEGATIVE_PROMPTS,gh,gw)
                debug.update({"competitive_"+k:v for k,v in d.items()})
            nac=None
            if RUN_NACLIP or RUN_NACLIP_COMPETITIVE: nac=tester.naclip_patch(x,last,gh,gw)
            if RUN_NACLIP: maps["naclip"]=tester.normal_map(nac,TARGET_PROMPTS,gh,gw)
            if RUN_NACLIP_COMPETITIVE:
                maps["naclip_competitive"],d=tester.competitive_map(nac,COMPETITIVE_TARGET_PROMPTS,COMPETITIVE_NEGATIVE_PROMPTS,gh,gw)
                debug.update({"naclip_competitive_"+k:v for k,v in d.items()})

        frame_metrics={}
        for name,m in maps.items():
            s=get_map_stats(m); stat_rows.append({"frame_index":fi,"processed_index":pi,"method":name,**meta,**s})
            curves[name]["frame"].append(fi); curves[name]["gaussian_score"].append(s["gaussian_score"]); curves[name]["p95_minus_p50"].append(s["p95_minus_p50"])
            if name in previous:
                temp_rows.append({"frame_index":fi,"processed_index":pi,"method":name,**meta,
                                  "temporal_pearson":safe_pearson(previous[name],m),
                                  "temporal_weighted_center_disp_norm":norm_dist(weighted_center(previous[name]),weighted_center(m))})
            previous[name]=m.copy()

        if "unet_original" in maps:
            ref=maps["unet_original"]
            for name,m in maps.items():
                if name=="unet_original": continue
                q=compare_ref(m,ref); frame_metrics[name]=q; ref_rows.append({"frame_index":fi,"processed_index":pi,"method":name,**meta,**q})

        # Direct ablation comparison against the user's current Temporal(L=1) map.
        if "current_temporal_l1" in maps:
            current_ref = maps["current_temporal_l1"]
            for name,m in maps.items():
                if name in ("unet_original", "current_temporal_l1"):
                    continue
                qcur = compare_current(m, current_ref)
                current_rows.append({"frame_index":fi,"processed_index":pi,"method":name,**meta,**qcur})

        # Check whether the test harness reproduces the relevance map actually
        # saved by real_approach_test.py at collection time.
        recorded_map=recorded_relevance_full(saved_diag,fi)
        if recorded_map is not None and "current_temporal_l1" in maps:
            qrec=compare_ref(maps["current_temporal_l1"],recorded_map)
            qrec={k.replace("_vs_unet","_vs_recorded"):v for k,v in qrec.items()}
            recorded_rows.append({"frame_index":fi,"processed_index":pi,"method":"current_temporal_l1",**meta,**qrec})
            debug["recorded_current_relevance"]=recorded_map

        if SAVE_RAW_MAPS:
            payload={name:m.astype(np.float32) for name,m in maps.items()}; payload.update({name:m.astype(np.float32) for name,m in debug.items()})
            np.savez_compressed(rawdir/f"frame_{fi:06d}.npz",**payload)

        if SAVE_VISUALIZATIONS and pi%max(1,VIS_EVERY_N_PROCESSED_FRAMES)==0 and nvis<MAX_VISUALIZATIONS:
            save_panel(visdir/f"frame_{fi:06d}.png",frame,maps,fi,frame_metrics); nvis+=1

        if pi==0 or (pi+1)%10==0 or pi+1==len(idxs):
            print(f"[{pi+1:4d}/{len(idxs):4d}] frame={fi:6d} time={time.perf_counter()-fs:.3f}s")
            if frame_metrics:
                for name,q in frame_metrics.items(): print(f"    {name:22s} Spearman={q['spearman_vs_unet']:.3f} Top10IoU={q['top10_iou_vs_unet']:.3f}")

    summary=build_summary(ref_rows,stat_rows,temp_rows)
    current_summary=build_vs_current_summary(current_rows)
    save_csv(outdir/"per_frame_metrics.csv",ref_rows)
    save_csv(outdir/"per_frame_vs_current.csv",current_rows)
    save_csv(outdir/"recorded_current_consistency.csv",recorded_rows)
    save_csv(outdir/"map_stats.csv",stat_rows)
    save_csv(outdir/"temporal_metrics.csv",temp_rows)
    save_csv(outdir/"summary_metrics.csv",summary)
    save_csv(outdir/"summary_vs_current.csv",current_summary)
    summary_plots(outdir,summary); curve_plots(outdir,curves)

    print("\n"+"="*80+"\nSUMMARY")
    for r in summary:
        print("\n["+DISPLAY_NAMES.get(r["method"],r["method"])+"]")
        if r["method"]!="unet_original":
            print(f"  Spearman vs U-Net : {r.get('spearman_vs_unet_mean',np.nan):.4f}")
            print(f"  Top-10% IoU       : {r.get('top10_iou_vs_unet_mean',np.nan):.4f}")
            print(f"  Center distance   : {r.get('weighted_center_dist_norm_vs_unet_mean',np.nan):.4f}")
            print(f"  Gaussian diff     : {r.get('gaussian_score_abs_diff_vs_unet_mean',np.nan):.4f}")
        print(f"  P95-P50           : {r.get('p95_minus_p50_mean',np.nan):.4f}")
        print(f"  Temporal Pearson  : {r.get('temporal_pearson_mean',np.nan):.4f}")

    if recorded_rows:
        sp=np.asarray([r["spearman_vs_recorded"] for r in recorded_rows],dtype=np.float64)
        mae=np.asarray([r["mae_vs_recorded"] for r in recorded_rows],dtype=np.float64)
        print("\n[Recorded-current consistency]")
        print(f"  Spearman mean : {np.nanmean(sp):.6f}")
        print(f"  MAE mean      : {np.nanmean(mae):.8f}")
        print("  CSV           : recorded_current_consistency.csv")

    print(f"\nElapsed: {time.perf_counter()-t0:.2f}s")
    print("Output:",outdir)
    print("First inspect: metrics_explained_zh.txt, summary_metrics.csv, visualizations/")

if __name__ == "__main__":
    main()
