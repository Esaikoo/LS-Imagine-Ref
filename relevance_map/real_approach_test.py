#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual real-distance MineCLIP relevance diagnostic for LS-Imagine.

Put at: LS-Imagine-Ref/relevance_map/real_approach_test.py
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from matplotlib import colormaps

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TASK_SPECS = PROJECT_ROOT / "envs" / "tasks" / "task_specs.yaml"
if not TASK_SPECS.exists():
    raise FileNotFoundError(f"Cannot find project root: {TASK_SPECS}")
os.chdir(PROJECT_ROOT)

import minedojo
from minedojo.tasks import _meta_task_make
from minedojo.sim import InventoryItem
from envs.tasks.minedojo.wrappers import MinedojoClipReward
from envs.tasks.base.ls_imagine_wrapper import BASIC_ACTIONS, NOOP_ACTION

# ---------------------------------------------------------------------
# Five tasks
# ---------------------------------------------------------------------
TASKS = {
    "harvest_log_in_plains": dict(
        task_id="harvest", fast_reset=5, prompt="Cut a tree",
        threshold=0.25, temperature=0.05, target_item="log",
        lidar_kind="block", lidar_aliases=["log"],
        sim_specs=dict(
            target_names="log", target_quantities=1,
            specified_biome="plains", break_speed_multiplier=100,
        ),
    ),
    "harvest_water_with_bucket": dict(
        task_id="harvest", fast_reset=5, prompt="Obtain water",
        threshold=0.25, temperature=0.05, target_item="water_bucket",
        lidar_kind="block", lidar_aliases=["water"],
        sim_specs=dict(
            target_names="water_bucket", target_quantities=1,
            specified_biome="plains", break_speed_multiplier=100,
            initial_inventory=[dict(slot="mainhand", name="bucket", quantity=1)],
        ),
    ),
    "harvest_sand": dict(
        task_id="harvest", fast_reset=5, prompt="Obtain sand",
        threshold=0.25, temperature=0.05, target_item="sand",
        lidar_kind="block", lidar_aliases=["sand"],
        sim_specs=dict(
            target_names="sand", target_quantities=1,
            break_speed_multiplier=100,
        ),
    ),
    "mine_iron_ore": dict(
        task_id="harvest", fast_reset=10, prompt="Mine iron ore",
        threshold=0.25, temperature=0.05, target_item="iron_ore",
        lidar_kind="block", lidar_aliases=["iron_ore", "iron ore"],
        sim_specs=dict(
            target_names="iron_ore", target_quantities=1,
            break_speed_multiplier=100,
            initial_inventory=[dict(slot="mainhand", name="stone_pickaxe", quantity=1)],
        ),
    ),
    "shear_sheep": dict(
        task_id="harvest", fast_reset=5, prompt="Shear sheep",
        threshold=0.25, temperature=0.05, target_item="wool",
        lidar_kind="entity", lidar_aliases=["sheep"],
        sim_specs=dict(
            target_names="wool", target_quantities=1,
            initial_mobs="sheep",
            initial_mob_spawn_range_low=(-15, 1, -15),
            initial_mob_spawn_range_high=(15, 1, 15),
            specified_biome="plains", break_speed_multiplier=100,
            initial_inventory=[dict(slot="mainhand", name="shears", quantity=1)],
        ),
    ),
}

# ---------------------------------------------------------------------
# Manual actions
#
# IMPORTANT:
# Do not manually reproduce the raw MineDojo action dictionary here.
# Import BASIC_ACTIONS and NOOP_ACTION directly from LS-Imagine, so this
# diagnostic always uses exactly the same primitive low-level actions
# as the real LS-Imagine environment.
# ---------------------------------------------------------------------

COMMAND_TO_LS_ACTION = {
    ".": "noop",
    "noop": "noop",

    "w": "forward",
    "forward": "forward",

    "s": "back",
    "back": "back",

    "a": "left",
    "left": "left",

    "d": "right",
    "right": "right",

    "j": "jump",
    "jump": "jump",

    "x": "attack",
    "attack": "attack",

    "e": "use",
    "use": "use",

    "cu": "turn_up",
    "up": "turn_up",

    "cd": "turn_down",
    "down": "turn_down",

    "cl": "turn_left",
    "turn_left": "turn_left",

    "cr": "turn_right",
    "turn_right": "turn_right",
}


def manual_action(command):
    """
    Build exactly the same raw action dictionary as LSImagineWrapper.

    BASIC_ACTIONS only stores the non-default part.
    NOOP_ACTION stores every required MineDojo low-level key.
    """
    name = COMMAND_TO_LS_ACTION.get(command)
    if name is None:
        return None

    action = copy.deepcopy(NOOP_ACTION)

    delta = copy.deepcopy(
        BASIC_ACTIONS[name]
    )

    action.update(delta)

    return action


def action_nonzero_summary(action):
    """
    Compact human-readable action printout for debugging.
    """
    parts = []

    for key in (
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "use",
        "attack",
    ):
        value = np.asarray(action[key]).reshape(-1)
        if len(value) and int(value[0]) != 0:
            parts.append(f"{key}={int(value[0])}")

    camera = np.asarray(action["camera"]).reshape(-1)
    if len(camera) >= 2 and (
        abs(float(camera[0])) > 1e-8
        or abs(float(camera[1])) > 1e-8
    ):
        parts.append(
            f"camera=({float(camera[0]):.1f},"
            f"{float(camera[1]):.1f})"
        )

    if not parts:
        return "noop"

    return ", ".join(parts)


def parse_command_with_repeat(text):
    """
    Examples:
        w
        w 10
        cr
        cr 3

    If a repeat is explicitly supplied, it overrides the defaults below.
    """
    parts = text.strip().lower().split()

    if not parts:
        return None, None

    command = parts[0]

    if command not in COMMAND_TO_LS_ACTION:
        return None, None

    if len(parts) >= 2:
        try:
            repeat = int(parts[1])
        except ValueError:
            return None, None

        if repeat <= 0:
            return None, None

        return command, repeat

    # Holding movement for several consecutive *real simulator steps* makes
    # manual control usable. Every sub-step is still independently recorded.
    if command in ("w", "forward", "s", "back", "a", "left", "d", "right"):
        repeat = MOVE_REPEAT

    elif command in ("cu", "up", "cd", "down", "cl", "turn_left", "cr", "turn_right"):
        repeat = CAMERA_REPEAT

    elif command in ("j", "jump"):
        repeat = JUMP_REPEAT

    elif command in ("x", "attack", "e", "use"):
        repeat = INTERACTION_REPEAT

    else:
        repeat = 1

    return command, repeat


HELP = """
Manual control
--------------
Environment actions:
  w            forward for MOVE_REPEAT real simulator steps
  s            back
  a            move left
  d            move right

  w 1          exactly one forward simulator step
  w 10         ten consecutive forward simulator steps
  cr 3         turn right three x 10 degrees

  j            jump + forward
  x            attack
  e            use

  cu           camera up    (-10 deg)
  cd           camera down  (+10 deg)
  cl           camera left  (-10 deg)
  cr           camera right (+10 deg)
  .            no-op

Annotations (do NOT step environment):
  mark far
  mark mid
  mark near
  mark contact
  mark <custom>

  note <text>
  save
  h / help
  q / quit

Every repeated sub-step is a REAL Minecraft env.step() and is recorded into:
frames / actions / positions / cosine / relevance / lidar.

Recommended:
  mark far -> approach -> mark mid -> mark near -> mark contact
  -> attack/use if desired.
"""

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
def parse_seed(v):
    try: return int(v)
    except ValueError: return str(v)

def init_inventory(items):
    if not items: return None
    return [InventoryItem(slot=x["slot"], name=x["name"], variant=None,
                          quantity=x.get("quantity", 1)) for x in items]

def lidar_rays():
    # Diagnostic only: 49 front-facing rays, <=64m.
    ang = np.arange(-45, 46, 15)
    return [(np.deg2rad(p), np.deg2rad(y), 64.0) for p in ang for y in ang]

def make_env(cfg, world_seed, task_seed, use_lidar):
    """
    Create the raw MineDojo meta-task in the same way LS-Imagine does.

    IMPORTANT:
      Do NOT call minedojo.make(..., event_level_control=False).

    minedojo.make() always appends ARNNWrapper, and that wrapper requires
    event_level_control=True. LS-Imagine intentionally bypasses that top-level
    wrapper and calls _meta_task_make() directly, because LS-Imagine uses
    keyboard/mouse-level action dictionaries with event_level_control=False.
    """
    specs = copy.deepcopy(cfg["sim_specs"])
    inv = init_inventory(specs.pop("initial_inventory", None))

    meta_task = cfg["task_id"]  # all five requested tasks use "harvest"

    kwargs = dict(
        image_size=(160, 256),

        # Same underlying simulator settings as LS-Imagine.
        fast_reset=False,
        event_level_control=False,
        use_voxel=False,

        seed=task_seed,
        world_seed=world_seed,

        **specs,
    )

    if inv is not None:
        kwargs["initial_inventory"] = inv

    if use_lidar:
        kwargs["use_lidar"] = True
        kwargs["lidar_rays"] = lidar_rays()
    else:
        kwargs["use_lidar"] = False

    print("\nMineDojo environment (raw meta task, no ARNNWrapper):")
    print(f"  meta_task: {meta_task}")
    for k, v in kwargs.items():
        print(
            f"  {k}: {len(v)} rays"
            if k == "lidar_rays"
            else f"  {k}: {v}"
        )

    env = _meta_task_make(
        meta_task,
        **kwargs,
    )

    print("  action_space:", env.action_space)
    return env

# ---------------------------------------------------------------------
# Observation / relevance
# ---------------------------------------------------------------------
def rgb_frame(obs):
    x = np.asarray(obs["rgb"])
    if x.shape[0] == 3: x = x.transpose(1, 2, 0)
    return np.ascontiguousarray(x, dtype=np.uint8)

def relevance_from_cosine(cosine, grid, tau, temp):
    gh, gw = grid
    sim = cosine.reshape(gh, gw)
    z = np.clip((sim - tau) / temp, -30, 30)
    small = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    big = F.interpolate(torch.from_numpy(small)[None, None], size=(160, 256),
                        mode="bilinear", align_corners=False)[0, 0].numpy()
    return small, big.astype(np.float32)

def norm_name(x): return str(x).strip().lower().replace(" ", "_")

def inventory_qty(obs, item):
    inv = obs.get("inventory")
    if inv is None: return np.nan
    names, qtys = np.asarray(inv["name"]), np.asarray(inv["quantity"])
    target = norm_name(item)
    return float(sum(int(q) for n, q in zip(names, qtys) if norm_name(n) == target))

def location(obs):
    out = dict(pos_x=np.nan, pos_y=np.nan, pos_z=np.nan, yaw=np.nan, pitch=np.nan)
    loc = obs.get("location_stats")
    if loc is None: return out
    if "pos" in loc:
        p = np.asarray(loc["pos"]).reshape(-1)
        if len(p) >= 3: out.update(pos_x=float(p[0]), pos_y=float(p[1]), pos_z=float(p[2]))
    for k in ("yaw", "pitch"):
        if k in loc: out[k] = float(np.asarray(loc[k]).reshape(-1)[0])
    return out

def lidar_metrics(obs, cfg):
    out = dict(
        target_lidar_visible=0, target_lidar_ray_count=0,
        target_lidar_min_distance=np.nan, target_lidar_median_distance=np.nan,
        target_lidar_ray_yaw_deg=np.nan, target_lidar_ray_pitch_deg=np.nan,
        center_block_name="", center_block_distance=np.nan,
        center_entity_name="", center_entity_distance=np.nan,
    )
    rays = obs.get("rays")
    if rays is None: return out
    yaw = np.asarray(rays.get("ray_yaw", []), dtype=np.float32)
    pitch = np.asarray(rays.get("ray_pitch", []), dtype=np.float32)
    if len(yaw):
        ci = int(np.argmin(np.abs(yaw) + np.abs(pitch)))
        bn, bd = np.asarray(rays.get("block_name", [])), np.asarray(rays.get("block_distance", []))
        en, ed = np.asarray(rays.get("entity_name", [])), np.asarray(rays.get("entity_distance", []))
        if len(bn) > ci: out["center_block_name"] = str(bn[ci])
        if len(bd) > ci: out["center_block_distance"] = float(bd[ci])
        if len(en) > ci: out["center_entity_name"] = str(en[ci])
        if len(ed) > ci: out["center_entity_distance"] = float(ed[ci])
    if cfg["lidar_kind"] == "block":
        names = np.asarray(rays.get("block_name", [])); dist = np.asarray(rays.get("block_distance", []), np.float32)
    else:
        names = np.asarray(rays.get("entity_name", [])); dist = np.asarray(rays.get("entity_distance", []), np.float32)
    aliases = [norm_name(x) for x in cfg["lidar_aliases"]]
    idx = [i for i, n in enumerate(names) if any(a in norm_name(n) for a in aliases)]
    if not idx: return out
    idx = np.asarray(idx, np.int64); d = dist[idx]
    valid = np.isfinite(d) & (d >= 0); idx, d = idx[valid], d[valid]
    if not len(d): return out
    j = int(np.argmin(d)); ri = int(idx[j])
    out.update(
        target_lidar_visible=1, target_lidar_ray_count=int(len(d)),
        target_lidar_min_distance=float(d.min()), target_lidar_median_distance=float(np.median(d)),
    )
    if len(yaw) > ri: out["target_lidar_ray_yaw_deg"] = float(np.rad2deg(yaw[ri]))
    if len(pitch) > ri: out["target_lidar_ray_pitch_deg"] = float(np.rad2deg(pitch[ri]))
    return out

def cc_metrics(rel):
    binary = (rel > 0.5).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return dict(cc_area=0.0, cc_x=-1, cc_y=-1, cc_w=0, cc_h=0)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    H, W = rel.shape
    return dict(
        cc_area=float(stats[k, cv2.CC_STAT_AREA] / (H * W)),
        cc_x=int(stats[k, cv2.CC_STAT_LEFT]), cc_y=int(stats[k, cv2.CC_STAT_TOP]),
        cc_w=int(stats[k, cv2.CC_STAT_WIDTH]), cc_h=int(stats[k, cv2.CC_STAT_HEIGHT]),
    )

def entropy01(small):
    x = small.reshape(-1).astype(np.float64); total = x.sum()
    if total <= 1e-12: return 0.0
    p = x / total
    return float((-np.sum(p * np.log(p + 1e-12))) / np.log(len(p)))

def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def heat_rgb(rel):
    return (colormaps["turbo"](np.clip(rel, 0, 1))[..., :3] * 255).astype(np.uint8)

def panel(frame, rel, m, prompt, stage="", note=""):
    H, W, _ = frame.shape; top = 88
    heat = heat_rgb(rel)
    over = np.clip(0.5 * frame.astype(np.float32) + 0.5 * heat.astype(np.float32), 0, 255).astype(np.uint8)
    can = Image.new("RGB", (W * 3, H + top), "black")
    can.paste(Image.fromarray(frame), (0, top)); can.paste(Image.fromarray(over), (W, top)); can.paste(Image.fromarray(heat), (2*W, top))
    d = ImageDraw.Draw(can)
    d.text((5, 5), f"RGB | {prompt}", fill="white"); d.text((W+5, 5), "overlay", fill="white"); d.text((2*W+5, 5), "relevance", fill="white")
    ld = "NA" if not np.isfinite(m["target_lidar_min_distance"]) else f'{m["target_lidar_min_distance"]:.2f}m'
    d.text((5, 27), f'step={m["step"]} stage={stage or "-"} lidar={ld} pos=({m["pos_x"]:.1f},{m["pos_y"]:.1f},{m["pos_z"]:.1f})', fill="white")
    d.text((5, 47), f'cos mean={m["cos_mean"]:.4f} max={m["cos_max"]:.4f} P95={m["cos_p95"]:.4f} lift={m["cos_p95_p50"]:.4f}', fill="white")
    d.text((5, 67), f'rel mean={m["rel_mean"]:.3f} >.5={m["rel_area_05"]:.3f} cc={m["cc_area"]:.3f} sharp={m["sharpness"]:.0f} note={note[:28]}', fill="white")
    if m["cc_w"] > 0:
        d.rectangle((W+m["cc_x"], top+m["cc_y"], W+m["cc_x"]+m["cc_w"], top+m["cc_y"]+m["cc_h"]), outline="white", width=1)
    return can

# ---------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------
class Recorder:
    def __init__(self, out, task, cfg, world_seed, task_seed, tau, temp, show, checkpoint_every):
        self.out = out; out.mkdir(parents=True, exist_ok=False)
        self.views = out / "views"; self.views.mkdir()
        self.task, self.cfg, self.world_seed, self.task_seed = task, cfg, world_seed, task_seed
        self.tau, self.temp, self.show, self.checkpoint_every = tau, temp, show, checkpoint_every
        self.frames=[]; self.cos=[]; self.rel=[]; self.rows=[]; self.labels=[]; self.notes=[]; self.actions=[]; self.commands=[]; self.infos=[]
        self.start_pos=None; self.prev_pos=None; self.path=0.0

    def observe(self, obs, clip, reward=0.0, done=False, info=None):
        frame = rgb_frame(obs)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.perf_counter(); before=getattr(clip,"vision_forward_count",None)
        bundle=clip.get_frame_bundle(obs); sim=clip.get_patch_similarity(bundle,[self.cfg["prompt"]])
        after=getattr(clip,"vision_forward_count",None)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms=(time.perf_counter()-t0)*1000
        if before is not None and after is not None and after-before != 1:
            raise RuntimeError(f"Vision forward delta should be 1, got {after-before}")
        c=sim[0,0].detach().cpu().numpy().astype(np.float32); small,big=relevance_from_cosine(c,bundle["grid_size"],self.tau,self.temp)
        loc=location(obs); p=np.array([loc["pos_x"],loc["pos_y"],loc["pos_z"]],np.float64)
        if np.all(np.isfinite(p)):
            if self.start_pos is None: self.start_pos=p.copy()
            if self.prev_pos is not None: self.path += float(np.linalg.norm(p-self.prev_pos))
            self.prev_pos=p.copy(); dist_start=float(np.linalg.norm(p-self.start_pos))
        else: dist_start=np.nan
        flat=c.reshape(-1); cc=cc_metrics(big); lidar=lidar_metrics(obs,self.cfg)
        m=dict(
            step=len(self.frames), reward=float(reward), done=int(done),
            cos_min=float(flat.min()), cos_mean=float(flat.mean()), cos_std=float(flat.std()), cos_max=float(flat.max()),
            cos_p50=float(np.percentile(flat,50)), cos_p75=float(np.percentile(flat,75)), cos_p90=float(np.percentile(flat,90)), cos_p95=float(np.percentile(flat,95)), cos_p99=float(np.percentile(flat,99)),
            cos_range=float(flat.max()-flat.min()), cos_p95_p50=float(np.percentile(flat,95)-np.percentile(flat,50)),
            cos_top5_mean=float(np.sort(flat)[-5:].mean()), cos_top10_mean=float(np.sort(flat)[-10:].mean()),
            rel_mean=float(big.mean()), rel_std=float(big.std()), rel_max=float(big.max()), rel_area_03=float((big>.3).mean()), rel_area_05=float((big>.5).mean()), rel_area_07=float((big>.7).mean()), rel_area_09=float((big>.9).mean()), rel_entropy=entropy01(small),
            argmax_patch_row=int(np.argmax(flat)//16), argmax_patch_col=int(np.argmax(flat)%16),
            sharpness=sharpness(frame), distance_from_start=dist_start, cumulative_path=self.path,
            target_inventory_quantity=inventory_qty(obs,self.cfg["target_item"]), mineclip_ms=float(ms),
            **loc, **lidar, **cc,
        )
        self.frames.append(frame); self.cos.append(c.reshape(10,16)); self.rel.append(small); self.rows.append(m); self.labels.append(""); self.notes.append("")
        self.infos.append({k:(v.item() if isinstance(v,(np.integer,np.floating)) else v) for k,v in (info or {}).items() if isinstance(v,(str,bool,int,float,np.integer,np.floating))})
        self.render_current()
        self.print_row(m)
        if self.checkpoint_every and len(self.frames)%self.checkpoint_every==0: self.checkpoint()
        return m

    def print_row(self,m):
        ld="NA" if not np.isfinite(m["target_lidar_min_distance"]) else f'{m["target_lidar_min_distance"]:.2f}m'
        print(f'\nstep={m["step"]} pos=({m["pos_x"]:.2f},{m["pos_y"]:.2f},{m["pos_z"]:.2f}) lidar={ld} rays={m["target_lidar_ray_count"]}')
        print(f'cos mean={m["cos_mean"]:.4f} max={m["cos_max"]:.4f} P90={m["cos_p90"]:.4f} P95={m["cos_p95"]:.4f} P95-P50={m["cos_p95_p50"]:.4f}')
        print(f'rel mean={m["rel_mean"]:.4f} >0.5={m["rel_area_05"]:.3f} CC={m["cc_area"]:.3f} sharp={m["sharpness"]:.1f} inventory={m["target_inventory_quantity"]}')
        print("view:", self.out/"current_view.png")

    def render_current(self):
        i=len(self.frames)-1
        big=F.interpolate(torch.from_numpy(self.rel[i])[None,None],size=(160,256),mode="bilinear",align_corners=False)[0,0].numpy()
        im=panel(self.frames[i],big,self.rows[i],self.cfg["prompt"],self.labels[i],self.notes[i])
        im.save(self.views/f"step_{i:04d}.png"); im.save(self.out/"current_view.png")
        if self.show:
            cv2.imshow("Real Approach MineCLIP Relevance",cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)); cv2.waitKey(1)

    def mark(self,label=None,note=None):
        i=len(self.frames)-1
        if label is not None:self.labels[i]=label
        if note is not None:self.notes[i]=note
        self.render_current(); self.write_csv()

    def add_action(self,a,cmd):
        self.actions.append(np.array([
            int(a["forward"]),int(a["back"]),int(a["left"]),int(a["right"]),int(a["jump"]),int(a["use"]),int(a["attack"]),float(a["camera"][0]),float(a["camera"][1])
        ],np.float32)); self.commands.append(cmd)

    def write_csv(self):
        if not self.rows:return
        fields=list(self.rows[0].keys())+["stage_label","note"]
        with (self.out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i,r in enumerate(self.rows):
                x=dict(r);x["stage_label"]=self.labels[i];x["note"]=self.notes[i];w.writerow(x)

    def save_npz(self,name):
        actions=np.stack(self.actions) if self.actions else np.zeros((0,9),np.float32)
        np.savez_compressed(self.out/name,
            frames=np.stack(self.frames), cosines=np.stack(self.cos).astype(np.float32), relevance_patch=np.stack(self.rel).astype(np.float32),
            actions=actions, action_commands=np.asarray(self.commands,str), stage_labels=np.asarray(self.labels,str), notes=np.asarray(self.notes,str),
            positions=np.asarray([[r["pos_x"],r["pos_y"],r["pos_z"]] for r in self.rows],np.float32),
            target_lidar_distance=np.asarray([r["target_lidar_min_distance"] for r in self.rows],np.float32),
            target_lidar_visible=np.asarray([r["target_lidar_visible"] for r in self.rows],np.uint8),
            reward=np.asarray([r["reward"] for r in self.rows],np.float32), target_inventory_quantity=np.asarray([r["target_inventory_quantity"] for r in self.rows],np.float32),
            task_name=np.asarray(self.task), prompt=np.asarray(self.cfg["prompt"]), world_seed=np.asarray(str(self.world_seed)), task_seed=np.asarray(self.task_seed), threshold=np.asarray(self.tau), temperature=np.asarray(self.temp))

    def checkpoint(self):
        self.write_csv(); self.save_npz("trajectory_checkpoint.npz"); print("[checkpoint saved]")

    def finalize(self):
        self.write_csv(); self.save_npz("trajectory.npz")
        with (self.out/"info_scalar.jsonl").open("w",encoding="utf-8") as f:
            for i,x in enumerate(self.infos):f.write(json.dumps({"step":i,**x},ensure_ascii=False)+"\n")
        with (self.out/"session_config.json").open("w",encoding="utf-8") as f:
            json.dump(dict(task=self.task,prompt=self.cfg["prompt"],world_seed=self.world_seed,task_seed=self.task_seed,threshold=self.tau,temperature=self.temp,yaml_fast_reset=self.cfg["fast_reset"],sim_specs=self.cfg["sim_specs"]),f,indent=2,ensure_ascii=False,default=str)
        raw=[Image.fromarray(x) for x in self.frames]
        raw[0].save(self.out/"trajectory_rgb.gif",save_all=True,append_images=raw[1:],duration=180,loop=0)
        diag=[]
        for i in range(len(self.frames)):
            big=F.interpolate(torch.from_numpy(self.rel[i])[None,None],size=(160,256),mode="bilinear",align_corners=False)[0,0].numpy()
            diag.append(panel(self.frames[i],big,self.rows[i],self.cfg["prompt"],self.labels[i],self.notes[i]))
        diag[0].save(self.out/"trajectory_relevance.gif",save_all=True,append_images=diag[1:],duration=220,loop=0)
        print("\nSaved:",self.out);print("  trajectory.npz\n  metrics.csv\n  trajectory_rgb.gif\n  trajectory_relevance.gif\n  views/")

# ---------------------------------------------------------------------
# PyCharm run configuration
#
# Edit these values and simply click Run in PyCharm.
# No bash arguments are required.
# ---------------------------------------------------------------------

TASK_NAME = "harvest_log_in_plains"
# TASK_NAME = "mine_iron_ore"

# Minecraft map seed.
WORLD_SEED = 43

# MineDojo task/spawn RNG seed.
TASK_SEED = 0

# None -> use the task's configured threshold / temperature.
# For the current tree experiment you can keep 0.30 / 0.03.
RELEVANCE_THRESHOLD = 0.288
RELEVANCE_TEMPERATURE = 0.016

MAX_ENV_STEPS = 500

OUTPUT_ROOT = (
    SCRIPT_DIR
    / "real_approach_runs"
)

# Diagnostic-only Lidar. It does NOT affect MineCLIP relevance or actions.
USE_LIDAR = True

# Set True when your PyCharm session has access to a display.
# A live window will update after every real observation.
SHOW_WINDOW = False

# Manual input mode:
#
#   "console":
#       Type commands in PyCharm Run console, e.g. w / cr / x / q.
#       The program prints the exact received string as [INPUT DEBUG].
#
#   "window":
#       Click the OpenCV current-view window and press keys directly.
#       This completely avoids stdin/input() and is often easier in PyCharm.
#
# If your machine/server cannot display OpenCV windows, use "console".
CONTROL_MODE = "script"

CHECKPOINT_EVERY = 10


# ---------------------------------------------------------------------
# Manual-control responsiveness
#
# One low-level MineDojo forward step can cause only a small visual/position
# change. These defaults make one typed "w" behave like holding W briefly.
#
# IMPORTANT:
# Every repeated step is still individually saved, so the trajectory remains
# real environment data rather than frame skipping.
# ---------------------------------------------------------------------

MOVE_REPEAT = 4
CAMERA_REPEAT = 1
JUMP_REPEAT = 1
INTERACTION_REPEAT = 1


# ---------------------------------------------------------------------
# PRESET ACTION PLAN
#
# Edit this list in PyCharm, then click Run.
#
# Each tuple is:
#   ("command", repeat)
#
# Commands:
#   w/s/a/d       move
#   cu/cd/cl/cr   camera
#   j             jump + forward
#   x             attack
#   e             use
#   .             noop
#
# Annotation / utility:
#   ("mark", "far")
#   ("mark", "mid")
#   ("mark", "near")
#   ("mark", "contact")
#   ("note", "free text")
#   ("save",)
#
# IMPORTANT:
# repeat means REAL MineDojo env.step() calls.
# Every sub-step is independently recorded in NPZ/CSV/GIF.
#
# Recommended:
#   First run SCAN_ONLY=True to rotate 360 degrees and inspect all directions.
#   Then keep the SAME WORLD_SEED/TASK_SEED, set SCAN_ONLY=False, edit
#   ACTION_PLAN, and run the real approach trajectory.
# ---------------------------------------------------------------------

SCAN_ONLY = True

# 36 x 10° = approximately one full turn.
SCAN_PLAN = [
    ("cr", 36),
]

# Used when SCAN_ONLY=False.
ACTION_PLAN = [
    # ("mark", "far"),

    # Replace this after inspecting the scan run.
    ("cl", 9),   # 0-repeat entries are skipped.
    ("cd", 1),
    ("w", 60),
    ("cl", 2.8),
    ("w", 43),
    ("cd", 1),

    # ("mark", "mid"),


    # ("mark", "near"),
    # ("cl", 1),
    # ("w", 32),

    # ("mark", "contact"),

    # Tree/sand/iron -> attack.
    # Water/sheep usually -> use, i.e. ("e", N).
    ("e", 1),
    (".", 15),
]


def get_position_from_obs(obs):
    loc = obs.get("location_stats", {})

    if "pos" not in loc:
        return None

    pos = np.asarray(loc["pos"], dtype=np.float64).reshape(-1)

    if len(pos) < 3:
        return None

    return pos[:3].copy()


def get_yaw_pitch_from_obs(obs):
    loc = obs.get("location_stats", {})

    def scalar(key):
        if key not in loc:
            return np.nan
        arr = np.asarray(loc[key]).reshape(-1)
        return float(arr[0]) if len(arr) else np.nan

    return scalar("yaw"), scalar("pitch")


def validate_action(env, action):
    """
    Explicitly verify that the action belongs to the raw MineDojo action space.
    """
    contains = getattr(
        env.action_space,
        "contains",
        None,
    )

    if contains is None:
        return

    try:
        valid = bool(
            contains(action)
        )
    except Exception as exc:
        print(
            "[ACTION CHECK] action_space.contains() "
            f"raised: {exc}"
        )
        return

    print(
        "[ACTION CHECK] valid:",
        valid,
        "|",
        action_nonzero_summary(action),
    )

    if not valid:
        print(
            "[WARNING] The action is not accepted by env.action_space.\n"
            "action =", action, "\n"
            "space  =", env.action_space
        )


def step_manual_command(
    env,
    recorder,
    clip,
    previous_obs,
    command,
    repeat,
):
    """
    Execute a human command for `repeat` consecutive REAL env steps.

    Returns:
        last_obs, done
    """
    action = manual_action(
        command
    )

    if action is None:
        raise ValueError(
            f"Unknown command: {command}"
        )

    print(
        "\n"
        "============================================================"
    )
    print(
        f"[COMMAND] {command!r} "
        f"repeat={repeat} | "
        f"{action_nonzero_summary(action)}"
    )

    validate_action(
        env,
        action,
    )

    start_frame = rgb_frame(
        previous_obs
    )

    start_pos = get_position_from_obs(
        previous_obs
    )

    start_yaw, start_pitch = (
        get_yaw_pitch_from_obs(
            previous_obs
        )
    )

    obs = previous_obs
    done = False

    last_info = {}

    for substep in range(repeat):
        # Use a fresh deepcopy because MineDojo / wrappers may retain action refs.
        act = copy.deepcopy(
            action
        )

        recorder.add_action(
            act,
            command,
        )

        obs, reward, done, info = (
            env.step(
                act
            )
        )

        last_info = info or {}

        print(
            f"[ENV STEP] "
            f"{substep + 1}/{repeat} "
            f"reward={reward} "
            f"done={done}"
        )

        if "error" in last_info:
            print(
                "[ENV INFO ERROR]",
                last_info["error"],
            )

        recorder.observe(
            obs,
            clip,
            reward,
            done,
            info,
        )

        if done:
            break

    # --------------------------------------------------------
    # Action-effect diagnostics
    # --------------------------------------------------------

    end_frame = rgb_frame(
        obs
    )

    end_pos = get_position_from_obs(
        obs
    )

    end_yaw, end_pitch = (
        get_yaw_pitch_from_obs(
            obs
        )
    )

    frame_abs = np.abs(
        end_frame.astype(np.float32)
        - start_frame.astype(np.float32)
    )

    frame_mae = float(
        np.mean(frame_abs)
    )

    pixel_changed = float(
        np.mean(
            np.max(
                frame_abs,
                axis=-1,
            )
            > 1.0
        )
    )

    if (
        start_pos is not None
        and end_pos is not None
    ):
        pos_delta = float(
            np.linalg.norm(
                end_pos
                - start_pos
            )
        )

        pos_text = (
            f"{pos_delta:.4f} blocks"
        )
    else:
        pos_delta = np.nan
        pos_text = "NA"

    yaw_delta = (
        end_yaw - start_yaw
        if np.isfinite(start_yaw)
        and np.isfinite(end_yaw)
        else np.nan
    )

    pitch_delta = (
        end_pitch - start_pitch
        if np.isfinite(start_pitch)
        and np.isfinite(end_pitch)
        else np.nan
    )

    print(
        "\n[COMMAND RESULT]"
    )
    print(
        "  position delta:",
        pos_text,
    )
    print(
        "  yaw delta:",
        yaw_delta,
    )
    print(
        "  pitch delta:",
        pitch_delta,
    )
    print(
        "  frame MAE:",
        f"{frame_mae:.4f}",
    )
    print(
        "  changed pixel ratio:",
        f"{pixel_changed:.4f}",
    )

    # If all diagnostics are exactly unchanged, make it obvious.
    no_pose_change = (
        (
            not np.isfinite(pos_delta)
            or pos_delta < 1e-6
        )
        and (
            not np.isfinite(yaw_delta)
            or abs(yaw_delta) < 1e-6
        )
        and (
            not np.isfinite(pitch_delta)
            or abs(pitch_delta) < 1e-6
        )
    )

    if (
        no_pose_change
        and frame_mae < 1e-3
    ):
        print(
            "\n"
            "[WARNING] This command produced essentially NO state change.\n"
            "Please copy this whole COMMAND RESULT plus the printed "
            "action_space and send it back for diagnosis."
        )

    print(
        "============================================================"
    )

    return obs, done



# ---------------------------------------------------------------------
# Scripted action-plan execution
# ---------------------------------------------------------------------

def normalize_plan_entry(entry):
    if not isinstance(entry, (tuple, list)) or len(entry) == 0:
        raise ValueError(f"Invalid plan entry: {entry!r}")

    kind = str(entry[0]).strip().lower()

    if kind in ("mark", "note"):
        if len(entry) < 2:
            raise ValueError(f"{kind} requires text: {entry!r}")
        return kind, str(entry[1])

    if kind == "save":
        return "save", None

    if kind not in COMMAND_TO_LS_ACTION:
        raise ValueError(
            f"Unknown action command in plan: {kind!r}"
        )

    repeat = int(entry[1]) if len(entry) >= 2 else 1
    return kind, repeat


def execute_action_plan(
    env,
    recorder,
    clip,
    obs,
    plan,
):
    done = False

    print(
        "\n"
        "============================================================"
    )
    print("EXECUTING PRESET ACTION PLAN")
    print(
        "============================================================"
    )
    print("entries:", len(plan))

    for plan_index, entry in enumerate(plan):
        kind, value = normalize_plan_entry(entry)

        print(
            f"\n[PLAN {plan_index + 1}/{len(plan)}] "
            f"{entry!r}"
        )

        if kind == "mark":
            recorder.mark(label=value)
            print(f"[PLAN] marked current frame: {value}")
            continue

        if kind == "note":
            recorder.mark(note=value)
            print(f"[PLAN] note on current frame: {value}")
            continue

        if kind == "save":
            recorder.checkpoint()
            print("[PLAN] checkpoint saved")
            continue

        repeat = int(value)

        if repeat <= 0:
            print(
                f"[PLAN] skipped {kind!r} because repeat={repeat}"
            )
            continue

        obs, done = step_manual_command(
            env=env,
            recorder=recorder,
            clip=clip,
            previous_obs=obs,
            command=kind,
            repeat=repeat,
        )

        current = recorder.rows[-1]

        if (
            np.isfinite(current["target_inventory_quantity"])
            and current["target_inventory_quantity"] >= 1
        ):
            print(
                "\n[TARGET INVENTORY DETECTED] "
                f"{recorder.cfg['target_item']}="
                f"{current['target_inventory_quantity']}"
            )

        if done:
            print(
                "\n[PLAN] Environment terminated; "
                "remaining actions will not run."
            )
            break

    return obs, done


def main():
    if TASK_NAME not in TASKS:
        raise KeyError(
            f"Unknown TASK_NAME: {TASK_NAME}. "
            f"Available: {sorted(TASKS)}"
        )

    cfg = copy.deepcopy(
        TASKS[TASK_NAME]
    )

    world_seed = parse_seed(
        str(WORLD_SEED)
    )

    tau = (
        cfg["threshold"]
        if RELEVANCE_THRESHOLD is None
        else float(
            RELEVANCE_THRESHOLD
        )
    )

    temperature = (
        cfg["temperature"]
        if RELEVANCE_TEMPERATURE is None
        else float(
            RELEVANCE_TEMPERATURE
        )
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    seed_name = (
        str(world_seed)
        .replace("/", "_")
        .replace(" ", "_")
    )

    output_dir = (
        Path(OUTPUT_ROOT)
        .expanduser()
        .resolve()
        / TASK_NAME
        / (
            f"world_{seed_name}_"
            f"taskseed_{TASK_SEED}_"
            f"{stamp}"
        )
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "REAL APPROACH MINECLIP TEST"
    )
    print(
        "============================================================"
    )
    print(
        "TASK_NAME:",
        TASK_NAME,
    )
    print(
        "prompt:",
        cfg["prompt"],
    )
    print(
        "WORLD_SEED:",
        world_seed,
    )
    print(
        "TASK_SEED:",
        TASK_SEED,
    )
    print(
        "tau / T:",
        tau,
        "/",
        temperature,
    )
    print(
        "MOVE_REPEAT:",
        MOVE_REPEAT,
    )
    print(
        "SHOW_WINDOW:",
        SHOW_WINDOW,
    )
    print(
        "CONTROL_MODE:",
        CONTROL_MODE,
    )
    print(
        "SCAN_ONLY:",
        SCAN_ONLY,
    )
    print(
        "output:",
        output_dir,
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # MineCLIP
    # --------------------------------------------------------

    print(
        "\nLoading MineCLIP..."
    )

    clip = (
        MinedojoClipReward()
    )

    # Cache prompt once.
    clip.get_text_feats_cached(
        [
            cfg["prompt"]
        ]
    )

    if hasattr(
        clip,
        "vision_forward_count",
    ):
        clip.vision_forward_count = 0

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = make_env(
        cfg,
        world_seed,
        TASK_SEED,
        USE_LIDAR,
    )

    print(
        "\nRAW MineDojo action_space:"
    )
    print(
        env.action_space
    )

    recorder = Recorder(
        output_dir,
        TASK_NAME,
        cfg,
        world_seed,
        TASK_SEED,
        tau,
        temperature,
        SHOW_WINDOW,
        CHECKPOINT_EVERY,
    )

    print(
        HELP
    )

    try:
        obs = env.reset()

        print(
            "\nEnvironment reset complete."
        )

        recorder.observe(
            obs,
            clip,
        )

        # --------------------------------------------------------
        # Fully scripted control: no stdin and no window key capture.
        # --------------------------------------------------------

        plan = SCAN_PLAN if SCAN_ONLY else ACTION_PLAN

        obs, done = execute_action_plan(
            env=env,
            recorder=recorder,
            clip=clip,
            obs=obs,
            plan=plan,
        )

        recorder.finalize()

        if hasattr(
            clip,
            "vision_forward_count",
        ):
            print(
                "MineCLIP Vision forwards:",
                clip.vision_forward_count,
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted; saving..."
        )

        if recorder.frames:
            recorder.finalize()

        raise

    finally:
        try:
            env.close()
        except Exception:
            pass

        if SHOW_WINDOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
