#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze LS-Imagine train_eps/eval_eps .npz datasets.

Main outputs
------------
summary.json
    Global episode/zoom/jump/matching statistics and correlations.
episodes.csv
    Per-episode success/zoom/jump statistics.
zoom_jump_events.csv
    Every accepted zoom/jump with before/after score, relevance statistics,
    yaw/pitch, future returns, and ScoreStorage matching information.
initial_direction_bins.csv
    First-N-step observations grouped by relative viewing direction; useful for
    studying "scan surroundings then choose the highest-value direction".
event_images/*.png
    Current RGB + relevance, accepted zoom RGB + relevance, and, when available,
    the future real frame matched by ScoreStorage.
npz_schema.json
    Keys/dtypes/shapes found in saved episodes.

Important: old .npz files usually contain is_zoomed only AFTER zoom validation.
They cannot recover rejected Natural-Zoom candidates, zoom_prob, adaptive threshold,
gate_reason, or raw MineCLIP cosine statistics unless those were logged at training time.
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None

ACTION_NAMES = [
    "noop", "attack", "turn_up", "turn_down", "turn_left", "turn_right",
    "forward", "back", "left", "right", "jump", "use",
]
TURN_UP, TURN_DOWN, TURN_LEFT, TURN_RIGHT = 2, 3, 4, 5

OPTIONAL_DIAG_KEYS = [
    "zoom_candidate", "is_check", "zoom_prob", "adaptive_threshold",
    "check_threshold", "raw_cosine_p50", "raw_cosine_p95",
    "raw_cosine_p95_p50", "patch_cc05_10x16", "raw_zoom_factor",
    "actual_zoom_factor", "gaussian_score", "zoomed_gaussian",
    "gaussian_gain_ok", "zoom_p95_gain", "zoom_contrast_gain",
    "zoom_cc_retention", "post_zoom_semantic_gain_ok",
]

def optional_diag_at(ep, t):
    out = {}
    for k in OPTIONAL_DIAG_KEYS:
        if k not in ep:
            continue
        a = np.asarray(ep[k])
        if a.ndim == 0:
            out[k] = safe_float(a)
        elif t < a.shape[0]:
            v = a[t]
            # string/object diagnostics (e.g. gate_reason) are intentionally skipped
            try:
                out[k] = safe_float(v)
            except Exception:
                pass
    return out


def scalar_array(x, length, default=0.0, dtype=np.float64):
    if x is None:
        return np.full(length, default, dtype=dtype)
    a = np.asarray(x)
    if a.ndim == 0:
        return np.full(length, a.item(), dtype=dtype)
    if a.shape[0] == 0:
        return np.full(length, default, dtype=dtype)
    a = a.reshape(a.shape[0], -1)[:, 0]
    if len(a) < length:
        a = np.concatenate([a, np.full(length-len(a), default)])
    return a[:length].astype(dtype, copy=False)


def bool_array(x, length, default=False):
    return scalar_array(x, length, int(default)) > 0.5


def safe_float(x, default=np.nan):
    try:
        a = np.asarray(x)
        return float(a.reshape(-1)[0]) if a.size else float(default)
    except Exception:
        return float(default)


def image_uint8(img):
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    if np.issubdtype(a.dtype, np.floating) and a.size and np.nanmax(a) <= 1.5:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def relevance01(hm):
    a = np.squeeze(np.asarray(hm, dtype=np.float32))
    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1])
    if a.size and np.nanmax(a) > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)


def resize_10x16(x):
    if cv2 is not None:
        return cv2.resize(np.asarray(x, np.float32), (16, 10), interpolation=cv2.INTER_AREA)
    ys = np.linspace(0, x.shape[0]-1, 10).astype(int)
    xs = np.linspace(0, x.shape[1]-1, 16).astype(int)
    return x[np.ix_(ys, xs)]


def largest_cc_fraction(binary):
    b = np.asarray(binary, np.uint8)
    if b.size == 0 or b.max() == 0:
        return 0.0
    if cv2 is not None:
        n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if n <= 1:
            return 0.0
        return float(np.max(stats[1:, cv2.CC_STAT_AREA]) / b.size)
    H, W = b.shape
    seen = np.zeros_like(b, bool)
    best = 0
    for y in range(H):
        for x in range(W):
            if not b[y, x] or seen[y, x]:
                continue
            stack, area = [(y, x)], 0
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop(); area += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0: continue
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and b[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
            best = max(best, area)
    return float(best / b.size)


def relevance_metrics(hm):
    x = relevance01(hm)
    flat = x.reshape(-1)
    if flat.size == 0:
        return {k: np.nan for k in [
            "rel_mean","rel_std","rel_max","rel_p50","rel_p95","rel_p95_p50",
            "rel_area05","rel_cc05_approx_10x16","rel_center_x","rel_center_y","rel_center_dist"
        ]}
    p50, p95 = np.percentile(flat, [50, 95])
    low = resize_10x16(x)
    cc = largest_cc_fraction((low > 0.5).astype(np.uint8))
    H, W = x.shape; total = float(x.sum())
    if total > 1e-8:
        yy, xx = np.mgrid[0:H, 0:W]
        cx = float((xx*x).sum()/total); cy = float((yy*x).sum()/total)
        nx = (cx-(W-1)/2)/max((W-1)/2, 1); ny = (cy-(H-1)/2)/max((H-1)/2, 1)
        cd = float(np.sqrt(nx*nx + ny*ny))
    else:
        cx = cy = cd = np.nan
    return {
        "rel_mean": float(flat.mean()), "rel_std": float(flat.std()), "rel_max": float(flat.max()),
        "rel_p50": float(p50), "rel_p95": float(p95), "rel_p95_p50": float(p95-p50),
        "rel_area05": float(np.mean(flat > 0.5)), "rel_cc05_approx_10x16": cc,
        "rel_center_x": cx, "rel_center_y": cy, "rel_center_dist": cd,
    }


def decode_actions(action, T):
    if action is None: return np.full(T, -1, np.int32)
    a = np.asarray(action)
    if a.ndim == 1:
        out = np.rint(a[:T]).astype(np.int32)
        if len(out) < T: out = np.pad(out, (0, T-len(out)), constant_values=-1)
        return out
    flat = a.reshape(a.shape[0], -1)[:T]
    idx = np.argmax(flat, -1).astype(np.int32)
    idx[np.all(np.isclose(flat, 0), axis=-1)] = -1
    if len(idx) < T: idx = np.pad(idx, (0, T-len(idx)), constant_values=-1)
    return idx


def reconstruct_orientation(action_idx, turn_deg=10.0):
    yaw = np.zeros(len(action_idx), np.float32); pitch = np.zeros(len(action_idx), np.float32)
    y = p = 0.0
    for t, a in enumerate(action_idx):
        if a == TURN_LEFT: y -= turn_deg
        elif a == TURN_RIGHT: y += turn_deg
        elif a == TURN_UP: p -= turn_deg
        elif a == TURN_DOWN: p += turn_deg
        y = ((y + 180) % 360) - 180; p = float(np.clip(p, -70, 70))
        yaw[t], pitch[t] = y, p
    return yaw, pitch


def heading_bin(yaw, bin_deg):
    return int((round((yaw % 360) / bin_deg) * bin_deg) % 360)


def discounted_future_sum(x, horizon, gamma):
    x = np.asarray(x, np.float64); T = len(x); out = np.zeros(T)
    weights = gamma ** np.arange(horizon)
    for t in range(T):
        seg = x[t+1:min(T, t+1+horizon)]
        if len(seg): out[t] = np.sum(seg * weights[:len(seg)])
    return out


def future_max(x, horizon):
    x = np.asarray(x, np.float64); T = len(x)
    vals = np.full(T, np.nan); idx = np.full(T, -1, np.int32)
    for t in range(T):
        lo, hi = t+1, min(T, t+1+horizon)
        if lo < hi:
            j = int(np.nanargmax(x[lo:hi])); vals[t] = x[lo+j]; idx[t] = lo+j
    return vals, idx


def future_any(x, horizon):
    x = np.asarray(x, bool); T = len(x); out = np.zeros(T, bool)
    for t in range(T): out[t] = bool(np.any(x[t+1:min(T, t+1+horizon)]))
    return out


def steps_to_next_true(x):
    x = np.asarray(x, bool); out = np.full(len(x), -1, np.int32); nxt = -1
    for t in range(len(x)-1, -1, -1):
        if x[t]: nxt = t
        if nxt >= 0: out[t] = nxt-t
    return out


def safe_corr(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12: return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def load_episode(path):
    with np.load(path, allow_pickle=False) as f: return {k: f[k] for k in f.files}


def event_panel(out_path, ep_name, t, ep, m, future_match=None):
    image = image_uint8(ep["image"][t]); hm = relevance01(ep["heatmap"][t])
    zi = zh = None
    if "zoomed_image" in ep and t < len(ep["zoomed_image"]) and np.any(ep["zoomed_image"][t]):
        zi = image_uint8(ep["zoomed_image"][t])
    if "heatmap_on_zoomed" in ep and t < len(ep["heatmap_on_zoomed"]) and np.any(ep["heatmap_on_zoomed"][t]):
        zh = relevance01(ep["heatmap_on_zoomed"][t])
    mi = mh = None
    if future_match is not None and 0 <= future_match < len(ep["image"]):
        mi = image_uint8(ep["image"][future_match]); mh = relevance01(ep["heatmap"][future_match])

    cols = 3 if mi is not None else 2
    fig, ax = plt.subplots(3, cols, figsize=(5.0*cols, 11))
    ax = np.asarray(ax).reshape(3, cols)
    score = safe_float(ep["score"][t]) if "score" in ep else np.nan
    scorez = safe_float(ep["score_on_zoomed"][t]) if "score_on_zoomed" in ep else np.nan

    ax[0,0].imshow(image); ax[0,0].set_title(f"Current RGB | step={t}")
    ax[1,0].imshow(hm, vmin=0, vmax=1); ax[1,0].set_title(
        f"Current relevance\nmean={m['rel_mean']:.3f} gap={m['rel_p95_p50']:.3f} CC≈{m['rel_cc05_approx_10x16']:.3f}")
    ax[2,0].imshow(image); ax[2,0].imshow(hm, alpha=.35, vmin=0, vmax=1); ax[2,0].set_title(f"Current overlay | score={score:.5f}")

    if zi is not None: ax[0,1].imshow(zi); ax[0,1].set_title("Accepted zoom RGB")
    else: ax[0,1].text(.5,.5,"No saved zoom image",ha="center",va="center")
    if zh is not None:
        zm = relevance_metrics(zh); ax[1,1].imshow(zh, vmin=0, vmax=1); ax[1,1].set_title(
            f"Zoomed relevance\nmean={zm['rel_mean']:.3f} gap={zm['rel_p95_p50']:.3f} CC≈{zm['rel_cc05_approx_10x16']:.3f}")
    else: ax[1,1].text(.5,.5,"No saved zoom relevance",ha="center",va="center")
    if zi is not None and zh is not None: ax[2,1].imshow(zi); ax[2,1].imshow(zh, alpha=.35, vmin=0, vmax=1)
    else: ax[2,1].text(.5,.5,"No zoom overlay",ha="center",va="center")
    ax[2,1].set_title(f"Zoom overlay | score_z={scorez:.5f} gain={scorez-score:+.5f}")

    if cols == 3:
        ax[0,2].imshow(mi); ax[0,2].set_title(f"Matched future RGB | step={future_match}")
        fm = relevance_metrics(mh); ax[1,2].imshow(mh, vmin=0, vmax=1); ax[1,2].set_title(
            f"Matched relevance\nmean={fm['rel_mean']:.3f} gap={fm['rel_p95_p50']:.3f} CC≈{fm['rel_cc05_approx_10x16']:.3f}")
        ax[2,2].imshow(mi); ax[2,2].imshow(mh, alpha=.35, vmin=0, vmax=1)
        ms = safe_float(ep["score"][future_match]) if "score" in ep else np.nan
        ax[2,2].set_title(f"Matched overlay | real score={ms:.5f}")

    for a in ax.flat: a.axis("off")
    iz = int(bool(safe_float(ep["is_zoomed"][t]))) if "is_zoomed" in ep else 0
    j = int(bool(safe_float(ep["jump"][t]))) if "jump" in ep else 0
    c = int(bool(safe_float(ep["is_calculated"][t]))) if "is_calculated" in ep else 0
    fig.suptitle(f"{ep_name} | step={t} | zoom={iz} jump={j} calculated={c}")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--out-dir", default="npz_analysis")
    p.add_argument("--last-episodes", type=int, default=0)
    p.add_argument("--future-horizons", type=int, nargs="+", default=[25,50,100])
    p.add_argument("--gamma", type=float, default=.997)
    p.add_argument("--initial-window", type=int, default=80)
    p.add_argument("--heading-bin-deg", type=float, default=45.)
    p.add_argument("--turn-deg", type=float, default=10.)
    p.add_argument("--max-event-images", type=int, default=300)
    p.add_argument("--save-all-step-csv", action="store_true")
    args = p.parse_args()

    train_dir = Path(args.train_dir).expanduser(); out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir/"event_images").mkdir(exist_ok=True)
    files = sorted(train_dir.glob("*.npz"))
    if args.last_episodes > 0: files = files[-args.last_episodes:]
    if not files: raise FileNotFoundError(f"No .npz in {train_dir}")

    schema = {}
    for path in files[:5]:
        try:
            with np.load(path, allow_pickle=False) as f:
                for k in f.files:
                    schema.setdefault(k, {"dtype":str(f[k].dtype), "example_shape":list(f[k].shape)})
        except Exception: pass
    (out_dir/"npz_schema.json").write_text(json.dumps(schema,indent=2,ensure_ascii=False),encoding="utf-8")

    episode_rows, event_rows, step_rows, initial_rows = [], [], [], []
    corr = {k:[] for k in ["score","rel_mean","rel_p95_p50","rel_area05","rel_cc05","rel_center_dist"]}
    for h in args.future_horizons:
        for k in [f"future_max_score_gain_{h}",f"future_return_{h}",f"future_success_{h}"]: corr[k]=[]
    total_steps = image_count = 0
    print(f"Found {len(files)} episodes")

    for ei, path in enumerate(files):
        try: ep = load_episode(path)
        except Exception as e: print(f"[WARN] {path.name}: {e}"); continue
        T = len(ep["reward"]) if "reward" in ep else len(ep.get("image", []))
        if T <= 0: continue
        total_steps += max(T-1,0)
        reward = scalar_array(ep.get("reward"),T); intrinsic=scalar_array(ep.get("intrinsic"),T)
        score=scalar_array(ep.get("score"),T); scorez=scalar_array(ep.get("score_on_zoomed"),T)
        success=bool_array(ep.get("success"),T); iz=bool_array(ep.get("is_zoomed"),T)
        jump=bool_array(ep.get("jump"),T); calc=bool_array(ep.get("is_calculated"),T)
        js=scalar_array(ep.get("jumping_steps"),T,np.nan); acc=scalar_array(ep.get("accumulated_reward"),T,np.nan)
        ai=decode_actions(ep.get("action"),T); yaw,pitch=reconstruct_orientation(ai,args.turn_deg)
        sts=steps_to_next_true(success); succ_steps=np.flatnonzero(success); first_succ=int(succ_steps[0]) if len(succ_steps) else -1
        future={}
        for h in args.future_horizons:
            fm,bi=future_max(score,h)
            future[h]={"return":discounted_future_sum(reward,h,args.gamma),"intr":discounted_future_sum(intrinsic,h,args.gamma),
                       "max_score":fm,"best_step":bi,"success":future_any(success,h)}
        mets=[relevance_metrics(ep["heatmap"][t]) if "heatmap" in ep else relevance_metrics(np.zeros((64,64))) for t in range(T)]
        zidx=np.flatnonzero(iz); jidx=np.flatnonzero(jump); cidx=np.flatnonzero(iz & calc)
        er={"episode":path.stem,"file":str(path),"length":T-1,"success":int(success.any()),"first_success_step":first_succ,
            "zoom_count":len(zidx),"jump_count":len(jidx),"calculated_zoom_count":len(cidx),"has_zoom":int(len(zidx)>0),
            "has_jump":int(len(jidx)>0),"has_calculated_zoom":int(len(cidx)>0),"zoom_rate_per_step":len(zidx)/max(T-1,1),
            "jump_rate_per_step":len(jidx)/max(T-1,1),"reward_sum":float(np.nansum(reward)),"intrinsic_sum":float(np.nansum(intrinsic)),
            "score_mean":float(np.nanmean(score)),"score_max":float(np.nanmax(score))}
        if len(zidx):
            g=scorez[zidx]-score[zidx]; er.update({"zoom_score_mean":float(np.nanmean(score[zidx])),"zoom_score_on_zoomed_mean":float(np.nanmean(scorez[zidx])),
                "zoom_score_gain_mean":float(np.nanmean(g)),"zoom_score_gain_median":float(np.nanmedian(g)),"zoom_score_gain_positive_fraction":float(np.mean(g>0))})
        else: er.update({k:np.nan for k in ["zoom_score_mean","zoom_score_on_zoomed_mean","zoom_score_gain_mean","zoom_score_gain_median","zoom_score_gain_positive_fraction"]})
        if len(cidx): er.update({"jumping_steps_mean":float(np.nanmean(js[cidx])),"jumping_steps_median":float(np.nanmedian(js[cidx])),"jumping_steps_p95":float(np.nanpercentile(js[cidx],95))})
        else: er.update({"jumping_steps_mean":np.nan,"jumping_steps_median":np.nan,"jumping_steps_p95":np.nan})
        episode_rows.append(er)

        for t in np.flatnonzero(iz | jump):
            m=mets[t]; r={"episode":path.stem,"step":int(t),"is_zoomed":int(iz[t]),"jump":int(jump[t]),"is_calculated":int(calc[t]),
                "success_episode":int(success.any()),"score":float(score[t]),"score_on_zoomed":float(scorez[t]),"score_gain":float(scorez[t]-score[t]),
                "jumping_steps":float(js[t]),"accumulated_reward":float(acc[t]),"yaw_deg":float(yaw[t]),"pitch_deg":float(pitch[t]),
                "heading_bin":heading_bin(float(yaw[t]),args.heading_bin_deg),"action_idx":int(ai[t]),
                "action_name":ACTION_NAMES[ai[t]] if 0<=ai[t]<len(ACTION_NAMES) else "none","steps_to_success":int(sts[t]),**m}
            r.update(optional_diag_at(ep, t))
            if "heatmap_on_zoomed" in ep and t<len(ep["heatmap_on_zoomed"]) and np.any(ep["heatmap_on_zoomed"][t]):
                zm=relevance_metrics(ep["heatmap_on_zoomed"][t]); r.update({"zoom_"+k:v for k,v in zm.items()});
                r["zoom_rel_mean_gain"]=zm["rel_mean"]-m["rel_mean"]; r["zoom_rel_p95_p50_gain"]=zm["rel_p95_p50"]-m["rel_p95_p50"]
                r["zoom_rel_cc05_gain"]=zm["rel_cc05_approx_10x16"]-m["rel_cc05_approx_10x16"]
            for h in args.future_horizons:
                f=future[h]; r[f"future_return_{h}"]=float(f["return"][t]); r[f"future_intrinsic_{h}"]=float(f["intr"][t]);
                r[f"future_max_score_{h}"]=float(f["max_score"][t]); r[f"future_max_score_gain_{h}"]=float(f["max_score"][t]-score[t]);
                r[f"future_best_score_step_{h}"]=int(f["best_step"][t]); r[f"future_success_{h}"]=int(f["success"][t])
            match=None
            if iz[t] and calc[t] and np.isfinite(js[t]):
                cand=t+int(round(js[t]))
                if 0<=cand<T: match=cand; r["matched_future_step"]=cand; r["matched_future_score"]=float(score[cand]); r["matched_score_margin_vs_zoom"]=float(score[cand]-scorez[t])
            event_rows.append(r)
            if image_count<args.max_event_images and "image" in ep and "heatmap" in ep:
                try:
                    name=f"{path.stem}__step_{t:05d}__z{int(iz[t])}_j{int(jump[t])}_c{int(calc[t])}.png"
                    event_panel(out_dir/"event_images"/name,path.stem,int(t),ep,m,match); image_count+=1
                except Exception as e: print(f"[WARN] image {path.stem} step {t}: {e}")

        for t in range(T):
            m=mets[t]; r={"episode":path.stem,"step":t,"success_episode":int(success.any()),"success_at_step":int(success[t]),"steps_to_success":int(sts[t]),
                "reward":float(reward[t]),"intrinsic":float(intrinsic[t]),"score":float(score[t]),"is_zoomed":int(iz[t]),"jump":int(jump[t]),"is_calculated":int(calc[t]),
                "yaw_deg":float(yaw[t]),"pitch_deg":float(pitch[t]),"heading_bin":heading_bin(float(yaw[t]),args.heading_bin_deg),"action_idx":int(ai[t]),
                "action_name":ACTION_NAMES[ai[t]] if 0<=ai[t]<len(ACTION_NAMES) else "none",**m}
            r.update(optional_diag_at(ep, t))
            for h in args.future_horizons:
                f=future[h]; r[f"future_return_{h}"]=float(f["return"][t]); r[f"future_intrinsic_{h}"]=float(f["intr"][t]);
                r[f"future_max_score_{h}"]=float(f["max_score"][t]); r[f"future_max_score_gain_{h}"]=float(f["max_score"][t]-score[t]); r[f"future_success_{h}"]=int(f["success"][t])
            corr["score"].append(r["score"]); corr["rel_mean"].append(r["rel_mean"]); corr["rel_p95_p50"].append(r["rel_p95_p50"])
            corr["rel_area05"].append(r["rel_area05"]); corr["rel_cc05"].append(r["rel_cc05_approx_10x16"]); corr["rel_center_dist"].append(r["rel_center_dist"])
            for h in args.future_horizons:
                corr[f"future_max_score_gain_{h}"].append(r[f"future_max_score_gain_{h}"]); corr[f"future_return_{h}"].append(r[f"future_return_{h}"]); corr[f"future_success_{h}"].append(r[f"future_success_{h}"])
            if args.save_all_step_csv: step_rows.append(r)
            if t<args.initial_window: initial_rows.append(r)
        if (ei+1)%20==0 or ei+1==len(files): print(f"[{ei+1}/{len(files)}] steps={total_steps} accepted_zoom={sum(x['zoom_count'] for x in episode_rows)}")

    groups={}
    for r in initial_rows: groups.setdefault((r["episode"],r["heading_bin"]),[]).append(r)
    dir_rows=[]
    for (epname,hbin),rows in groups.items():
        d={"episode":epname,"heading_bin":hbin,"num_observations":len(rows),"first_step":min(x["step"] for x in rows),"last_step":max(x["step"] for x in rows),
           "mean_score":float(np.mean([x["score"] for x in rows])),"max_score":float(np.max([x["score"] for x in rows])),"mean_rel_mean":float(np.mean([x["rel_mean"] for x in rows])),
           "max_rel_p95_p50":float(np.max([x["rel_p95_p50"] for x in rows])),"max_rel_cc05":float(np.max([x["rel_cc05_approx_10x16"] for x in rows])),
           "min_rel_center_dist":float(np.nanmin([x["rel_center_dist"] for x in rows]))}
        for h in args.future_horizons:
            d[f"max_future_return_{h}"]=float(np.max([x[f"future_return_{h}"] for x in rows])); d[f"max_future_max_score_gain_{h}"]=float(np.max([x[f"future_max_score_gain_{h}"] for x in rows]));
            d[f"any_future_success_{h}"]=int(any(x[f"future_success_{h}"] for x in rows))
        dir_rows.append(d)

    correlations={}
    feats=["score","rel_mean","rel_p95_p50","rel_area05","rel_cc05","rel_center_dist"]
    for h in args.future_horizons:
        correlations[str(h)]={}
        for feat in feats:
            correlations[str(h)][feat]={
                "corr_future_max_score_gain":safe_corr(corr[feat],corr[f"future_max_score_gain_{h}"]),
                "corr_future_return":safe_corr(corr[feat],corr[f"future_return_{h}"]),
                "corr_future_success":safe_corr(corr[feat],corr[f"future_success_{h}"]),
            }

    neps=len(episode_rows); nz=sum(x["zoom_count"] for x in episode_rows); nj=sum(x["jump_count"] for x in episode_rows); nc=sum(x["calculated_zoom_count"] for x in episode_rows); ns=sum(x["success"] for x in episode_rows)
    zrows=[x for x in event_rows if x["is_zoomed"]]; gains=np.asarray([x["score_gain"] for x in zrows],np.float64)
    summary={
        "train_dir":str(train_dir),"episodes":neps,"environment_steps_approx":total_steps,"episodes_with_zoom":sum(x["has_zoom"] for x in episode_rows),
        "episodes_with_jump":sum(x["has_jump"] for x in episode_rows),"episodes_with_calculated_zoom":sum(x["has_calculated_zoom"] for x in episode_rows),
        "successful_episodes":ns,"success_rate":ns/neps if neps else None,"accepted_zoom_steps":nz,"jump_steps":nj,"calculated_zoom_steps":nc,
        "accepted_zoom_rate_per_env_step":nz/total_steps if total_steps else None,"jump_rate_per_env_step":nj/total_steps if total_steps else None,
        "calculated_fraction_of_accepted_zoom":nc/nz if nz else None,"zoom_score_gain_mean":float(np.nanmean(gains)) if len(gains) else None,
        "zoom_score_gain_median":float(np.nanmedian(gains)) if len(gains) else None,"zoom_score_gain_positive_fraction":float(np.mean(gains>0)) if len(gains) else None,
        "future_horizons":args.future_horizons,"initial_window":args.initial_window,"heading_bin_deg":args.heading_bin_deg,"correlations":correlations,
        "important_limitations":{
            "natural_zoom_candidate_count":"NOT RECOVERABLE unless candidate/is_check was saved in transition.",
            "zoom_prob_and_adaptive_threshold":"NOT RECOVERABLE unless explicitly stored.",
            "raw_mineclip_cosine_p50_p95":"NOT RECOVERABLE from interpolated saved relevance map.",
            "cc_note":"rel_cc05_approx_10x16 is an approximation obtained by downsampling saved relevance to 10x16.",
            "direction_note":"yaw/pitch are relative directions reconstructed from camera actions, not absolute world headings."
        },
        "recommended_future_logging_fields":["zoom_candidate","zoom_prob","adaptive_threshold","zoom_gate_reason","raw_cosine_p50","raw_cosine_p95","raw_cosine_p95_p50","patch_cc05_10x16","raw_zoom_factor","actual_zoom_factor","gaussian_score","zoomed_gaussian","gaussian_gain_ok","zoom_p95_gain","zoom_contrast_gain","zoom_cc_retention","post_zoom_semantic_gain_ok"],
        "optional_diagnostics_detected_in_npz": [k for k in OPTIONAL_DIAG_KEYS if k in schema]
    }
    write_csv(out_dir/"episodes.csv",episode_rows); write_csv(out_dir/"zoom_jump_events.csv",event_rows); write_csv(out_dir/"initial_direction_bins.csv",dir_rows)
    if args.save_all_step_csv: write_csv(out_dir/"steps.csv",step_rows)
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    # compact plots
    if episode_rows:
        fig,ax=plt.subplots(figsize=(8,4.5)); labels=["episodes","zoom eps","jump eps","calc eps","success eps"];
        vals=[neps,sum(x["has_zoom"] for x in episode_rows),sum(x["has_jump"] for x in episode_rows),sum(x["has_calculated_zoom"] for x in episode_rows),ns]
        ax.bar(labels,vals); ax.set_ylabel("count"); ax.set_title("Episode-level counts"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(out_dir/"episode_counts.png",dpi=150); plt.close(fig)
    if zrows:
        x=np.asarray([r["score"] for r in zrows]); y=np.asarray([r["score_on_zoomed"] for r in zrows]);
        fig,ax=plt.subplots(figsize=(5.5,5)); ax.scatter(x,y,s=12,alpha=.5); lo=float(np.nanmin(np.r_[x,y])); hi=float(np.nanmax(np.r_[x,y])); ax.plot([lo,hi],[lo,hi]); ax.set_xlabel("score"); ax.set_ylabel("score_on_zoomed"); ax.set_title("Accepted zoom: before vs after"); fig.tight_layout(); fig.savefig(out_dir/"zoom_score_before_after.png",dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4.5)); ax.hist(y-x,bins=40); ax.set_xlabel("score_on_zoomed - score"); ax.set_ylabel("count"); ax.set_title("Accepted zoom score gain"); fig.tight_layout(); fig.savefig(out_dir/"zoom_score_gain_hist.png",dpi=150); plt.close(fig)

    print("\n================ SUMMARY ================")
    print(f"Episodes:                    {neps}")
    print(f"Approx env steps:            {total_steps}")
    print(f"Episodes with accepted zoom: {sum(x['has_zoom'] for x in episode_rows)}")
    print(f"Episodes with jump:          {sum(x['has_jump'] for x in episode_rows)}")
    print(f"Episodes with calc zoom:     {sum(x['has_calculated_zoom'] for x in episode_rows)}")
    print(f"Successful episodes:         {ns}")
    print(f"Accepted zoom steps:         {nz}")
    print(f"Jump steps:                  {nj}")
    print(f"Calculated zoom steps:       {nc}")
    if nz:
        print(f"Calculated / accepted zoom:  {nc/nz:.3f}")
        print(f"Mean zoom score gain:        {np.nanmean(gains):+.6f}")
        print(f"Positive score gain frac:    {np.mean(gains>0):.3f}")
    print(f"\nOutput directory: {out_dir}")
    print("NOTE: current npz counts accepted zoom, not rejected pre-acceptance candidates.")

if __name__ == "__main__":
    main()
