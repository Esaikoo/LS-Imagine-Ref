"""Coordinate-free online 2D search-behaviour memory.

This is a rolling estimated behaviour map, not a measured occupancy map. Without
depth, observed cells describe short-range view directions; target cells describe
the viewpoint from which a cue was seen, never an inferred true target position.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .odometry import PoseNet, normalized_rgb, wrap_angle

SPATIAL_VERSION = 2
MAP_CHANNELS = ("visited", "observed", "target_vantage", "revisit", "uncertainty", "frontier")
POSE_DIM, SUMMARY_DIM, DIRECTION_BINS = 4, 12, 12
DEFAULTS = dict(vnext_map_size=65, vnext_map_resolution=1.5, vnext_map_patch=21,
    vnext_pose_hidden=128, vnext_pose_min_samples=256, vnext_pose_ready_error=.25,
    vnext_pose_max_sigma=1.0, vnext_memory_dim=128, vnext_map_view_radius=2,
    vnext_map_target_decay=.999, vnext_pose_loss_weight=1.,
    vnext_pose_min_calibration=32, vnext_pose_min_moving=8,
    vnext_pose_calibration_ratio=.8, vnext_pose_cache_size=4096,
    vnext_map_ledger_capacity=2048)


def cfg(config, key):
    return getattr(config, key, DEFAULTS[key])


def centered_cell(value, resolution=1.):
    """Nearest cell with translation-invariant ties, shared with GT labels."""
    if torch.is_tensor(value):
        return torch.floor(value/resolution+.5).long()
    return np.floor(np.asarray(value)/resolution+.5).astype(np.int64)


def heading_sector(value, bins=DIRECTION_BINS):
    if torch.is_tensor(value):
        return torch.remainder(torch.floor(value/(2*math.pi)*bins+.5).long(), bins)
    return np.floor(np.asarray(value)/(2*math.pi)*bins+.5).astype(np.int64) % bins


def encode_grid(grid):
    """Immutable bounded replay representation, BCHW float -> BHWC uint8."""
    grid = grid.float()
    values = grid.clone()
    for index in (0, 1, 3):
        values[:, index] = torch.log1p(values[:, index].clamp(0, 255))/math.log(256.)
    values[:, 4] = values[:, 4]/(1.+values[:, 4])
    return (values.clamp(0, 1).permute(0, 2, 3, 1)*255).round().to(torch.uint8)


class MapEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.map = nn.Sequential(nn.Conv2d(6, 24, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten())
        self.output = nn.Sequential(nn.Linear(32*9+5, dim), nn.LayerNorm(dim), nn.SiLU())

    def forward(self, patch, pose):
        patch = torch.as_tensor(patch)
        lead = patch.shape[:-3]
        if patch.shape[-1] != 6:
            raise ValueError("Spatial map patch must have six channels")
        values = patch.float()/255. if patch.dtype == torch.uint8 else patch.float()
        pose = torch.as_tensor(pose, device=patch.device).float()
        if pose.shape != (*lead, POSE_DIM):
            raise ValueError("Spatial pose must have x,z,yaw-radians,uncertainty")
        p = torch.stack((torch.tanh(pose[..., 0]/100.), torch.tanh(pose[..., 1]/100.),
            pose[..., 2].sin(), pose[..., 2].cos(), pose[..., 3]/(1.+pose[..., 3])), -1)
        encoded = self.map(values.reshape(-1, *patch.shape[-3:]).permute(0, 3, 1, 2))
        return self.output(torch.cat((encoded, p.reshape(-1, 5)), -1)).reshape(*lead, -1)


class ViewHistoryEncoder(nn.Module):
    """Read a bounded episode-long visual history without assuming positions.

    The RSSM feature only queries history; it is not concatenated to the output.
    Therefore a missing history cannot masquerade as learned persistent memory.
    Frames are actual pooled RGB/heatmap observations, not stale learned latents.
    """
    def __init__(self, feature_dim, dim):
        super().__init__()
        self.frames = nn.Sequential(nn.Linear(64+9, dim), nn.LayerNorm(dim), nn.SiLU(),
                                    nn.Linear(dim, dim), nn.SiLU())
        self.query = nn.Linear(feature_dim, dim)
        self.key = nn.Linear(dim, dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU())

    def forward(self, feature, frames, meta, valid, tick):
        lead, slots = frames.shape[:-4], frames.shape[-4]
        frame = frames.float()/255. if frames.dtype == torch.uint8 else frames.float()
        frame = frame.nan_to_num(nan=0., posinf=0., neginf=0.).clamp(0,1)
        meta = meta.float().nan_to_num(nan=0., posinf=0., neginf=0.)
        age = ((tick[..., None]-meta[..., 0]).clamp_min(0)/1000.).clamp(max=10.)
        recent = ((tick[..., None]-meta[..., 1]).clamp_min(0)/1000.).clamp(max=10.)
        metadata = torch.stack((age, recent, torch.log1p(meta[..., 2].clamp(0,1.e6))/8.,
            meta[..., 3].clamp(0,1), meta[..., 4].clamp(0, 1), torch.log1p(meta[..., 5].clamp(0,1.e6))/8.,
            meta[..., 6].sin(), meta[..., 6].cos(), (meta[..., 7]/70.).clamp(-1,1)), -1)
        encoded = self.frames(torch.cat((frame.reshape(*lead, slots, 64), metadata), -1))
        query = F.normalize(self.query(feature.detach().float()), dim=-1)
        keys = F.normalize(self.key(encoded), dim=-1)
        weights = (keys*query[..., None, :]).sum(-1)*8.
        valid = valid.bool()
        weights = weights.masked_fill(~valid, -1.e4).softmax(-1)*valid.float()
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1.e-6)
        context = self.output((encoded*weights[..., None]).sum(-2))
        return context*valid.any(-1, keepdim=True).float()


class SpatialMapMemory(nn.Module):
    version = SPATIAL_VERSION

    def __init__(self, feature_dim, config):
        super().__init__()
        self.config = config
        self.fix_version = int(getattr(config, "vnext_spatial_fix_version", 1))
        self.dim = int(cfg(config, "vnext_memory_dim"))
        self.goal_search = bool(getattr(config, "goal_search_enabled", False))
        self.dynamic_goal = getattr(config, "target_item", "") == "wool"
        if self.goal_search:
            self.goal_memory_encoder = nn.Linear(6, self.dim, bias=False)
        self.size = int(cfg(config, "vnext_map_size"))
        self.resolution = float(cfg(config, "vnext_map_resolution"))
        self.patch_size = int(cfg(config, "vnext_map_patch"))
        if self.size < self.patch_size+4 or self.size % 2 != 1 or self.patch_size % 2 != 1 or self.resolution <= 0:
            raise ValueError("Map/patch sizes must be odd, map at least patch+4, positive resolution")
        max_steps = int(getattr(config, "episode_max_steps_override", 0) or
                        getattr(config, "episode_max_steps", 1000))
        requested_capacity = int(cfg(config, "vnext_map_ledger_capacity"))
        self.ledger_capacity = max(2048, requested_capacity, max_steps+1)
        if requested_capacity < 0 or self.ledger_capacity > 8192:
            raise ValueError("Spatial visit ledger must cover episode_max_steps+1 and fit 8192 entries")
        self.pose = PoseNet(int(cfg(config, "vnext_pose_hidden")),
            int(cfg(config, "vnext_pose_min_samples")), float(cfg(config, "vnext_pose_ready_error")),
            min_calibration=int(cfg(config, "vnext_pose_min_calibration")),
            min_moving=int(cfg(config, "vnext_pose_min_moving")),
            calibration_ratio=float(cfg(config, "vnext_pose_calibration_ratio")),
            cache_size=int(cfg(config, "vnext_pose_cache_size")), fix_version=min(self.fix_version, 1))
        self.encoder = MapEncoder(self.dim)
        self.event_head = nn.Linear(self.dim, 3)  # current evidence, new view, real progress
        if self.fix_version >= 3:
            self.history_slots = int(getattr(config, "vnext_slots", 16))
            self.segment_resume_pairs = int(getattr(config, "vnext_segment_resume_pairs", 3))
            if self.history_slots < 2 or self.segment_resume_pairs < 1:
                raise ValueError("Persistent history needs >=2 slots and positive segment confirmation")
            if int(getattr(config, "vnext_memory_save_every", 8)) < 1:
                raise ValueError("Persistent history save interval must be positive")
            self.history_encoder = ViewHistoryEncoder(feature_dim, self.dim)
            self.history_head = nn.Linear(self.dim, 3)
            self.fusion = nn.Sequential(nn.Linear(2*self.dim+3, self.dim),
                                        nn.LayerNorm(self.dim), nn.SiLU())
        # Legacy accounting calls remain harmless for the spatial implementation.
        self.register_buffer("place_positive", torch.zeros(()))
        self.register_buffer("place_negative", torch.zeros(()))

    def initial(self, batch, device):
        state = dict(grid=torch.zeros(batch, 6, self.size, self.size, device=device),
            directions=torch.zeros(batch, self.size, self.size, DIRECTION_BINS, device=device),
            pose=torch.zeros(batch, POSE_DIM, device=device),
            origin=torch.full((batch, 2), -(self.size//2)*self.resolution, device=device),
            previous_image=torch.zeros(batch, 64, 64, 3, device=device),
            has_previous=torch.zeros(batch, dtype=torch.bool, device=device),
            tracking_ready=torch.full((batch,), self.pose.ready, dtype=torch.bool, device=device),
            ledger_cells=torch.zeros(batch, self.ledger_capacity, 2, dtype=torch.long, device=device),
            ledger_values=torch.zeros(batch, self.ledger_capacity, 6, device=device),
            ledger_directions=torch.zeros(batch, self.ledger_capacity, DIRECTION_BINS, device=device),
            ledger_used=torch.zeros(batch, dtype=torch.long, device=device),
            previous_cell=torch.full((batch, 2), -100000, device=device, dtype=torch.long),
            dwell=torch.zeros(batch, device=device), motion_ema=torch.zeros(batch, device=device),
            tick=torch.zeros(batch, device=device, dtype=torch.long),
            camera_pitch=torch.zeros(batch, device=device),
            pitch_known=torch.zeros(batch, device=device, dtype=torch.bool),
            trajectory_started=torch.zeros(batch, device=device, dtype=torch.bool),
            pose_gap_count=torch.zeros(batch, device=device, dtype=torch.long),
            first_gap_step=torch.full((batch,), -1, device=device, dtype=torch.long))
        if self.fix_version >= 3:
            state.update(segment=torch.zeros(batch, device=device, dtype=torch.long),
                ledger_segment=torch.zeros(batch, self.ledger_capacity, device=device, dtype=torch.long),
                trusted_run=torch.zeros(batch, device=device, dtype=torch.long),
                global_continuity=state["tracking_ready"].clone(),
                history_frames=torch.zeros(batch, self.history_slots, 4, 4, 4, device=device, dtype=torch.uint8),
                history_meta=torch.zeros(batch, self.history_slots, 8, device=device),
                history_valid=torch.zeros(batch, self.history_slots, device=device, dtype=torch.bool),
                history_yaw=torch.zeros(batch, device=device))
        if self.goal_search:
            state["history_goal"] = torch.zeros(batch, self.history_slots, 5, device=device)
            state["ledger_goal"] = torch.zeros(batch, self.ledger_capacity, 5, device=device)
        return state

    @torch.no_grad()
    def update_history(self, state, image, heatmap, control, reward, first, goal=None, peak_rejected=None):
        """Maintain true observed viewpoints across map segments and uncertainty.

        Metadata: created tick, last tick, visits, max ground cue, actual progress
        count, own interaction attempts, own yaw radians, own camera pitch degrees.
        A visual match is only a similar view, never proof of spatial revisiting.
        """
        batch = len(image)
        rgb = normalized_rgb(image)
        heat = torch.as_tensor(heatmap, device=rgb.device)
        heat = heat.float()/255. if heat.dtype == torch.uint8 else heat.float()
        if heat.ndim == 3:
            heat = heat[..., None]
        if heat.shape != (batch, 64, 64, 1):
            raise ValueError("View history needs [batch,64,64,1] heatmap observations")
        frame = torch.cat((rgb, heat), -1).nan_to_num().clamp(0, 1)
        pooled = (F.adaptive_avg_pool2d(frame.permute(0, 3, 1, 2), (4, 4))
                  .permute(0, 2, 3, 1)*255.).round().to(torch.uint8)
        state["history_yaw"] = wrap_angle(state["history_yaw"]+
            torch.where(first, torch.zeros_like(control[:, 8]), control[:, 8])*math.pi/180.)
        _, cue, _ = self.ground_evidence(rgb, heat, state["camera_pitch"], state["pitch_known"])
        if goal is not None:
            cue = torch.where(goal[:, 7] > .5, goal[:, 0]*(goal[:, 6] == 2), cue)
        attempted = ((control[:, 0]>0)|(control[:, 6]>0)) & ~first
        progress = torch.zeros(batch, device=rgb.device, dtype=torch.bool)
        if reward is not None:
            progress = (torch.as_tensor(reward, device=rgb.device).reshape(batch, -1).sum(-1)>0) & ~first
        repeated = torch.zeros(batch, device=rgb.device, dtype=torch.bool)
        for b in range(batch):
            valid = state["history_valid"][b]
            distance = (state["history_frames"][b].float()-pooled[b].float()).abs().mean((-1,-2,-3))/255.
            yaw_distance = wrap_angle(state["history_meta"][b, :, 6]-state["history_yaw"][b]).abs()
            pitch_distance = (state["history_meta"][b, :, 7]-state["camera_pitch"][b]).abs()
            matches = valid & (distance<.04) & (yaw_distance<math.pi/12) & (pitch_distance<15.)
            repeated[b] = matches.any()
            if bool(matches.any()):
                slot = int(distance.masked_fill(~matches, 1.e4).argmin())
            else:
                # Save novelty/key outcomes immediately, plus bounded periodic
                # observations. Protect the first viewpoint for the episode.
                novel = not bool(valid.any()) or float(distance.masked_fill(~valid, 1.e4).min())>.08
                periodic = int(state["tick"][b]) % int(getattr(self.config, "vnext_memory_save_every", 8)) == 0
                goal_event = goal is not None and bool((goal[b, 7] > .5) & (
                    (goal[b, 6] > 0) | peak_rejected[b]))
                if not (novel or periodic or goal_event or bool(attempted[b]|progress[b])):
                    continue
                if not bool(valid.all()):
                    slot = int((~valid).long().argmax())
                else:
                    meta = state["history_meta"][b]
                    retention = 2*meta[:, 4].clamp(0,1)+meta[:, 3]-.001*(state["tick"][b]-meta[:, 1])
                    if goal is not None:
                        evidence = state["history_goal"][b]
                        decay = self.goal_evidence_decay(state["tick"][b] - evidence[:, 3])
                        retention += evidence[:, 1] * decay
                    retention[0] += 10000.
                    slot = int(retention.argmin())
                if goal is not None:
                    state["history_goal"][b, slot].zero_()
                state["history_meta"][b, slot].zero_()
                state["history_meta"][b, slot, 0] = state["tick"][b]
            meta = state["history_meta"][b, slot]
            meta[1] = state["tick"][b]
            meta[2] += 1
            meta[3] = torch.maximum(meta[3], cue[b])
            meta[4] += progress[b].float()
            meta[5] += attempted[b].float()
            meta[6] = state["history_yaw"][b]
            meta[7] = state["camera_pitch"][b]
            state["history_frames"][b, slot] = pooled[b]
            state["history_valid"][b, slot] = True
            if goal is not None:
                self.write_goal_evidence(state["history_goal"][b, slot], goal[b], state["tick"][b], peak_rejected[b])
        return repeated

    def goal_evidence_decay(self, age):
        # Evidence belongs to an observed viewpoint. Even static categories
        # can disappear; a mobile entity ages much faster.
        return torch.exp(-age.clamp_min(0).float()/(24. if self.dynamic_goal else 160.))

    def write_goal_evidence(self, slot, goal, tick, peak_rejected):
        if goal[7] <= .5:
            return
        candidate = goal[0] * (goal[6] == 1)
        verified = goal[0] * (goal[6] == 2)
        rejected = goal[5].clamp_min(.5) * peak_rejected
        update = torch.stack((candidate, verified, rejected))
        if bool(update.max() > 0):
            decay = self.goal_evidence_decay(tick - slot[3])
            slot[:3] = torch.maximum(slot[:3] * decay, update)
            slot[3] = tick
            slot[4] = slot[4] * decay + (verified > 0).float()

    def goal_summary(self, state, absolute_cell, map_write):
        tick = state["tick"][:, None]
        history = state["history_goal"]
        hist = self.goal_evidence_decay(tick-history[..., 3]) * state["history_valid"]
        present = (history[..., :3] > 0).any(-1) & state["history_valid"]
        hs = (history[..., :3] * hist[..., None]).amax(1)
        ledger = state["ledger_goal"]
        nearby = (state["ledger_cells"] - absolute_cell[:, None]).abs().sum(-1) <= 4
        used = torch.arange(self.ledger_capacity, device=ledger.device)[None] < state["ledger_used"][:, None]
        active = used & nearby & map_write[:, None]
        active &= state["ledger_segment"] == state["segment"][:, None]
        local = self.goal_evidence_decay(tick-ledger[..., 3]) * active
        ls = (ledger[..., :3]*local[..., None]).amax(1)
        confirmation = ((history[..., 4]*hist).sum(-1)/8.).clamp(0, 1)
        freshness = (hist*present).amax(-1)
        return torch.cat((torch.maximum(hs, ls), confirmation[:, None], freshness[:, None],
            ls[:, 1:2]), -1)

    @staticmethod
    def goal_direction_novelty(state, directions, map_write):
        # Option-relative directions in the currently trusted local segment.
        offsets = directions.new_tensor([0., -30., 30., 0., -15., 15., 30., 0., 0., 0., 180., -30.])
        sector = heading_sector(state["pose"][:, 2:3] + offsets[None]*math.pi/180.)
        sector_novelty = 1.-directions.gather(-1, sector.long())
        sector_novelty[:, 0] = 0.
        sector_novelty[:, 7:10] = 0.
        return sector_novelty * map_write[:, None]

    def fuse(self, map_context, history_context, local_valid, trajectory_valid, uncertainty, available):
        quality = torch.stack((local_valid.float(), trajectory_valid.float(),
            uncertainty.float()/(1.+uncertainty.float())), -1)
        return self.fusion(torch.cat((map_context, history_context, quality), -1))*available[..., None]

    @torch.no_grad()
    def start_local_segment(self, state, restart):
        """Start at the current view, without assigning unknown inter-segment pose."""
        for b in restart.nonzero().flatten().tolist():
            state["segment"][b] += 1
            state["grid"][b].zero_()
            state["directions"][b].zero_()
            state["pose"][b, :2] = 0
            state["pose"][b, 3] = 0
            state["origin"][b] = -(self.size//2)*self.resolution
            state["previous_cell"][b] = -100000
            state["dwell"][b] = 0
            state["motion_ema"][b] = 0
            state["tracking_ready"][b] = True

    @staticmethod
    def ground_evidence(image, heatmap, pitch, pitch_known):
        """Return qualified ground-view/cue proxies, never target truth.

        A horizontal/descending textured view may support a short directional
        observation proxy. A cue additionally needs a strong, spatially
        contrasting lower-view heatmap; a uniform or weak sky heatmap cannot
        paint target viewpoints. No rays, depth or simulator pose are read.
        """
        pixels = normalized_rgb(image)
        batch = len(pixels)
        heat = torch.as_tensor(heatmap, device=pixels.device)
        heat = heat.float()/255. if heat.dtype == torch.uint8 else heat.float()
        if heat.shape == (batch, 64, 64, 1):
            heat = heat[..., 0]
        if heat.shape != (batch, 64, 64):
            raise ValueError("Ground-view evidence requires [batch,64,64] heatmaps")
        finite = torch.isfinite(pixels).flatten(1).all(-1) & torch.isfinite(heat).flatten(1).all(-1)
        pixels = pixels.nan_to_num().clamp(0, 1)
        heat = heat.nan_to_num().clamp(0, 1)
        lower = pixels[:, 32:]
        texture = lower.reshape(batch, -1, 3).std(1, unbiased=False).mean(-1) > .025
        ground = pitch_known & finite & texture & (pitch >= -5.) & (pitch <= 45.)
        lower_heat = heat[:, 32:].reshape(batch, -1)
        peak = lower_heat.topk(32, dim=-1).values.mean(-1)
        contrast = peak-heat.reshape(batch, -1).median(-1).values
        cue_valid = ground & (peak >= .65) & (contrast >= .15)
        return ground, torch.where(cue_valid, peak, torch.zeros_like(peak)), cue_valid

    @staticmethod
    def integrate(pose, delta, sigma):
        result = pose.clone()
        angle = pose[:, 2]
        result[:, 0] += angle.cos()*delta[:, 0]-angle.sin()*delta[:, 1]
        result[:, 1] += angle.sin()*delta[:, 0]+angle.cos()*delta[:, 1]
        result[:, 2] = wrap_angle(angle+delta[:, 2])
        result[:, 3] = (pose[:, 3].square()+sigma[:, :2].square().sum(-1)).sqrt().clamp(max=1000.)
        return result

    def recenter(self, state):
        """Roll when approaching the boundary, discarding exposed strips, no wrap."""
        cells = centered_cell(state["pose"][:, :2]-state["origin"], self.resolution)
        margin = self.patch_size//2+1
        for b in range(len(cells)):
            if bool(((cells[b] < margin) | (cells[b] >= self.size-margin)).any()):
                shift = cells[b]-self.size//2
                sx, sz = int(shift[0]), int(shift[1])
                if abs(sx) >= self.size or abs(sz) >= self.size:
                    state["grid"][b].zero_(); state["directions"][b].zero_()
                else:
                    state["grid"][b] = torch.roll(state["grid"][b], (-sz, -sx), dims=(-2, -1))
                    state["directions"][b] = torch.roll(state["directions"][b], (-sz, -sx), dims=(0, 1))
                    if sx > 0:
                        state["grid"][b, :, :, -sx:] = 0; state["directions"][b, :, -sx:] = 0
                    elif sx < 0:
                        state["grid"][b, :, :, :-sx] = 0; state["directions"][b, :, :-sx] = 0
                    if sz > 0:
                        state["grid"][b, :, -sz:, :] = 0; state["directions"][b, -sz:] = 0
                    elif sz < 0:
                        state["grid"][b, :, :-sz, :] = 0; state["directions"][b, :-sz] = 0
                state["origin"][b] += shift.float()*self.resolution
                self.restore_ledger(state, b)
        return centered_cell(state["pose"][:, :2]-state["origin"], self.resolution)

    def restore_ledger(self, state, batch_index):
        """Restore only genuinely visited vantage cells visible in the window."""
        b = batch_index
        used = int(state["ledger_used"][b])
        if not used:
            return
        # Both origin and ledger coordinates use centered integer cells.
        cells = state["ledger_cells"][b, :used]-centered_cell(state["origin"][b], self.resolution)
        inside = ((cells >= 0) & (cells < self.size)).all(-1)
        if self.fix_version >= 3:
            inside &= state["ledger_segment"][b, :used] == state["segment"][b]
        cells = cells[inside]
        if not len(cells):
            return
        x, z = cells[:, 0], cells[:, 1]
        values = state["ledger_values"][b, :used][inside]
        for channel in (0, 3, 4):
            state["grid"][b, channel, z, x] = values[:, channel]
        for channel in (1, 2):
            state["grid"][b, channel, z, x] = torch.maximum(state["grid"][b, channel, z, x], values[:, channel])
        state["directions"][b, z, x] = state["ledger_directions"][b, :used][inside]

    def ledger_entry(self, state, batch_index, absolute_cell):
        """Find/add one estimated visited cell; never treat projected FOV as visited."""
        b = batch_index
        used = int(state["ledger_used"][b])
        if used:
            match = (state["ledger_cells"][b, :used] == absolute_cell).all(-1)
            if self.fix_version >= 3:
                match &= state["ledger_segment"][b, :used] == state["segment"][b]
            if bool(match.any()):
                return int(match.long().argmax())
        if used >= self.ledger_capacity:
            # Capacity >= episode budget+1 makes this a protocol error rather
            # than a reason to quietly erase the oldest visit history.
            raise RuntimeError("Spatial episode visit ledger exhausted; check episode step budget/reset")
        state["ledger_cells"][b, used] = absolute_cell
        if self.fix_version >= 3:
            state["ledger_segment"][b, used] = state["segment"][b]
        state["ledger_used"][b] += 1
        return used

    def patch(self, state, cells):
        encoded = encode_grid(state["grid"])
        radius = self.patch_size//2
        patches = [encoded[b, int(cells[b, 1])-radius:int(cells[b, 1])+radius+1,
                           int(cells[b, 0])-radius:int(cells[b, 0])+radius+1]
                   for b in range(len(cells))]
        return torch.stack(patches).clone()

    @torch.no_grad()
    def online(self, feat, image, heatmap, state=None, is_first=None, previous_control=None, reward=None,
               camera_pitch=None, goal_context=None, goal_peak_rejected=None):
        device, batch = feat.device, len(feat)
        fresh = self.initial(batch, device)
        missing_state = state is None
        state = fresh if missing_state else {k: state.get(k, v).detach().clone() for k, v in fresh.items()}
        first = torch.zeros(batch, device=device, dtype=torch.bool) if is_first is None else is_first.reshape(batch).bool()
        for key in state:
            state[key] = torch.where(first.reshape(batch, *([1]*(state[key].ndim-1))), fresh[key], state[key])
        if missing_state and is_first is not None:
            # A learned policy can begin halfway through random prefill. Its
            # unknown prefix is not an episode-origin pose and cannot be filled
            # in with zeros simply because the network has since qualified.
            state["tracking_ready"] &= first
        if self.fix_version >= 2:
            if missing_state:
                # An omitted reset flag cannot certify the episode origin.
                state["tracking_ready"] &= first
            state["trajectory_started"] |= first & state["tracking_ready"]
        if self.fix_version >= 3:
            # A segment can recover later, but global episode-origin continuity
            # is never silently regained after an unknown prefix or motion gap.
            state["global_continuity"] &= ~first | state["tracking_ready"]
            if missing_state:
                state["global_continuity"] &= first
        was_tracking = state["tracking_ready"].clone()
        # Qualification may revoke tracking, but can only enable it at reset.
        state["tracking_ready"] &= self.pose.ready
        qualification_lost = was_tracking & ~state["tracking_ready"]
        current = normalized_rgb(torch.as_tensor(image, device=device))
        goal = peak_rejected = None
        if self.goal_search:
            from apm.outcome import goal_context_tensor
            goal = goal_context_tensor(goal_context, feat)
            if goal_peak_rejected is None:
                peak_rejected = (goal[:,5] > .5) & (goal[:,6] == 0)
            else:
                peak_rejected = torch.as_tensor(goal_peak_rejected, device=device).reshape(batch).bool()
            peak_rejected = peak_rejected & (goal[:,7]>.5)
        control = torch.zeros(batch, 9, device=device) if previous_control is None else previous_control.to(device).float()
        if control.shape != (batch, 9) or not bool(torch.isfinite(control).all()):
            raise ValueError("Spatial memory needs finite [batch,9] own controls")
        valid = state["has_previous"] & ~first
        if self.fix_version >= 2:
            if camera_pitch is not None:
                pitch = torch.as_tensor(camera_pitch, device=device).float()
                if pitch.shape not in ((batch,), (batch, 1)):
                    raise ValueError("Spatial camera pitch must match the batch")
                pitch = pitch.reshape(batch)
                state["pitch_known"] = torch.isfinite(pitch) & (pitch.abs() <= 90.)
                state["camera_pitch"] = pitch.nan_to_num().clamp(-90., 90.)
            else:
                state["pitch_known"] |= first
                if previous_control is None:
                    state["pitch_known"] &= ~valid
                state["camera_pitch"] = torch.where(first, torch.zeros_like(state["camera_pitch"]),
                    state["camera_pitch"]+control[:, 7]).clamp(-90., 90.)
        predicted = self.pose(state["previous_image"], current, control)
        delta, sigma = predicted["delta"].clone(), predicted["sigma"].clone()
        if not bool(torch.isfinite(delta).all() & torch.isfinite(sigma).all()):
            raise FloatingPointError("PoseNet returned nonfinite online predictions")
        reliable = (sigma[:, :2].mean(-1) <= float(cfg(self.config, "vnext_pose_max_sigma"))) & state["tracking_ready"]
        # Unknown translation stays unknown; commands do not imply movement.
        raw_motion = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        accepted = reliable & valid
        trusted_pair = accepted.clone()
        motion_unknown = torch.zeros_like(valid)
        rejected_motion = torch.zeros_like(raw_motion)
        if self.fix_version:
            supported, _, motion_radius = self.pose.translation_support(predicted, state["previous_image"], current)
            accepted &= supported
            # A near-zero mean is not a confident static observation unless its
            # interval is also small. Rejected means remain uncertainty, never
            # effective map displacement or control progress.
            motion_probability = predicted.get("motion_probability", torch.ones_like(raw_motion))
            trusted_pair &= supported | ((motion_probability <= .2) & (sigma[:, :2].mean(-1) <= .1))
            if self.fix_version >= 2:
                known_control = previous_control is not None
                confident_static = (motion_probability <= .2) & (sigma[:, :2].mean(-1) <= .1) & (raw_motion <= .03)
                pair_motion_known = (supported | confident_static) & known_control
                motion_unknown = valid & ~pair_motion_known
                # Count loss of a previously continuous segment once, not every
                # unknown warm-up pair before localization ever began.
                lost = (motion_unknown & was_tracking) | qualification_lost
                state["pose_gap_count"] += lost.long()
                state["tracking_ready"] &= ~motion_unknown
                newly_lost = was_tracking & ~state["tracking_ready"]
                state["first_gap_step"] = torch.where(newly_lost & (state["first_gap_step"] < 0),
                    state["tick"], state["first_gap_step"])
                if self.fix_version >= 3:
                    state["global_continuity"] &= ~lost
                    confirmed = valid & pair_motion_known & bool(self.pose.ready)
                    state["trusted_run"] = torch.where(confirmed, state["trusted_run"]+1,
                                                       torch.zeros_like(state["trusted_run"]))
                    restart = (~state["tracking_ready"]) & confirmed & (state["trusted_run"]>=self.segment_resume_pairs)
                    self.start_local_segment(state, restart)
                    accepted = valid & supported & was_tracking & state["tracking_ready"] & ~restart
                    trusted_pair = valid & pair_motion_known & state["tracking_ready"] & ~restart
                else:
                    accepted &= state["tracking_ready"]
                    trusted_pair &= state["tracking_ready"] & pair_motion_known
            rejected_motion = torch.where(valid & ~accepted, raw_motion, torch.zeros_like(raw_motion))
            sigma[:, :2] = (sigma[:, :2].square()+
                torch.where(accepted[:, None], torch.zeros_like(delta[:, :2]), delta[:, :2]).square()).sqrt()
            delta[:, :2] = torch.where(accepted[:, None], delta[:, :2], torch.zeros_like(delta[:, :2]))
        else:
            delta[:, :2] *= reliable[:, None]
        # Known camera commands keep directional coverage usable during warmup.
        delta[:, 2] = torch.where(reliable, delta[:, 2], control[:, 8]*math.pi/180.)
        delta *= valid[:, None]
        sigma = torch.where(valid[:, None], sigma, torch.zeros_like(sigma))
        if self.fix_version >= 3:
            # The current view defines a recovered segment's new zero origin.
            # Do not add the last bootstrap pair or its uncertainty twice.
            sigma = torch.where(restart[:, None], torch.zeros_like(sigma), sigma)
        state["pose"] = self.integrate(state["pose"], delta, sigma)
        map_write = state["tracking_ready"] if self.fix_version >= 2 else torch.ones_like(valid)
        cells = self.recenter(state)
        absolute_cell = centered_cell(state["pose"][:, :2], self.resolution)
        same = (absolute_cell == state["previous_cell"]).all(-1) & valid
        if self.fix_version >= 2:
            same &= map_write
        state["dwell"] = torch.where(same, state["dwell"]+1, torch.zeros_like(state["dwell"]))
        displacement = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        state["motion_ema"].lerp_(displacement, .1)
        h = torch.as_tensor(heatmap, device=device)
        h = h.float()/255. if h.dtype == torch.uint8 else h.float()
        cue = h.reshape(batch, -1).topk(min(32, h[0].numel()), dim=-1).values.mean(-1).clamp(0, 1)
        raw_cue = cue.nan_to_num().clone()
        ground_view, cue_valid = torch.ones_like(valid), torch.ones_like(valid)
        if self.fix_version >= 2:
            ground_view, cue, cue_valid = self.ground_evidence(current, heatmap,
                state["camera_pitch"], state["pitch_known"])
            ground_view &= map_write
            cue_valid &= map_write
            cue = torch.where(cue_valid, cue, torch.zeros_like(cue))
        if goal is not None:
            cue = torch.where(goal[:, 7] > .5, goal[:, 0]*(goal[:, 6] == 2)*map_write, cue)
            cue_valid &= (goal[:, 7] <= .5) | (goal[:, 6] == 2)
        state["grid"][:, 2] *= float(cfg(self.config, "vnext_map_target_decay"))
        state["ledger_values"][..., 2] *= float(cfg(self.config, "vnext_map_target_decay"))
        visited_before, repeated = [], []
        radius = int(cfg(self.config, "vnext_map_view_radius"))
        for b in range(batch):
            x, z = int(cells[b, 0]), int(cells[b, 1])
            if not bool(map_write[b]):
                visited_before.append(torch.zeros((), device=device))
                repeated.append(torch.zeros((), device=device, dtype=torch.bool))
                continue
            entry = self.ledger_entry(state, b, absolute_cell[b])
            historic = state["ledger_values"][b, entry]
            state["grid"][b, 0, z, x] = historic[0]
            state["grid"][b, 3, z, x] = historic[3]
            state["grid"][b, 2, z, x] = torch.maximum(state["grid"][b, 2, z, x], historic[2])
            state["directions"][b, z, x] = state["ledger_directions"][b, entry]
            angle = state["pose"][b, 2]
            direction = int(heading_sector(angle))
            old_view = (state["directions"][b, z, x, direction] > 0) & ground_view[b]
            visited_before.append(state["grid"][b, 0, z, x].clone())
            repeated.append(old_view)
            if not bool(same[b]):
                state["grid"][b, 0, z, x] += 1
            if bool(old_view):
                state["grid"][b, 3, z, x] += 1
            state["directions"][b, z, x, direction] += float(ground_view[b])
            state["grid"][b, 1, z, x] += float(bool(ground_view[b]) and not bool(old_view))
            # Short directional view proxy only; not a raycast through obstacles.
            if bool(ground_view[b]) and not bool(old_view):
                for offset in (-math.pi/8, 0., math.pi/8):
                    for distance in range(1, radius+1):
                        xx = x+int(round(-math.sin(float(angle)+offset)*distance))
                        zz = z+int(round(math.cos(float(angle)+offset)*distance))
                        if 0 <= xx < self.size and 0 <= zz < self.size:
                            state["grid"][b, 1, zz, xx] += 1./(1.+distance)
            state["grid"][b, 2, z, x] = torch.maximum(state["grid"][b, 2, z, x], cue[b])
            state["grid"][b, 4, z, x] = state["pose"][b, 3]
            # Persist the visited viewpoint only. Nearby view-proxy cells are
            # not inserted as physical visits or claimed reconstructed terrain.
            state["ledger_values"][b, entry, 0] = state["grid"][b, 0, z, x]
            state["ledger_values"][b, entry, 1] = (state["directions"][b, z, x] > 0).float().sum()
            state["ledger_values"][b, entry, 2] = state["grid"][b, 2, z, x]
            state["ledger_values"][b, entry, 3] = state["grid"][b, 3, z, x]
            state["ledger_values"][b, entry, 4] = state["pose"][b, 3]
            state["ledger_directions"][b, entry] = state["directions"][b, z, x]
            if goal is not None:
                self.write_goal_evidence(state["ledger_goal"][b, entry], goal[b], state["tick"][b], peak_rejected[b])
        known = (state["grid"][:, 1:2] > 0).float()
        adjacent = F.max_pool2d(known, 3, 1, 1)
        state["grid"][:, 5:6] = adjacent*(1.-known)
        patch = self.patch(state, cells)
        context = self.encoder(patch, state["pose"])
        if self.fix_version >= 2:
            context *= map_write[:, None]
        history_repeated = torch.zeros_like(valid)
        if self.fix_version >= 3:
            history_repeated = self.update_history(state, current, heatmap, control, reward, first, goal, peak_rejected)
            history_context = self.history_encoder(feat, state["history_frames"], state["history_meta"],
                state["history_valid"], state["tick"].float())
            context = self.fuse(context, history_context, map_write, state["global_continuity"],
                state["pose"][:, 3], state["history_valid"].any(-1).float())
        rows = torch.arange(batch, device=device)
        directions = (state["directions"][rows, cells[:, 1], cells[:, 0]] > 0).float()
        visits = state["grid"][rows, 0, cells[:, 1], cells[:, 0]]
        visited = state["ledger_used"].float()
        if self.fix_version >= 3:
            used = torch.arange(self.ledger_capacity, device=device)[None] < state["ledger_used"][:, None]
            visited = (used & (state["ledger_segment"]==state["segment"][:, None])).sum(-1).float()
        observed = known.sum((-1, -2, -3)).float()
        repeated = torch.stack(repeated).float()
        confidence = sigma[:, :2].mean(-1).neg().exp()*state["tracking_ready"].float()
        if self.fix_version:
            confidence *= trusted_pair.float()
        u = state["pose"][:, 3]
        summary = torch.stack((state["tracking_ready"].float(),
            (state["motion_ema"]/.5).clamp(0, 1), (state["dwell"]/50.).clamp(0, 1),
            repeated, (visited/(self.size**2)).clamp(max=1.), observed/(self.size**2), u/(1.+u),
            (displacement/.5).clamp(0, 1), visits/(1.+visits), cue,
            directions.mean(-1), confidence), -1)
        if goal is not None:
            goal_memory = self.goal_summary(state, absolute_cell, map_write)
            context += self.goal_memory_encoder(goal_memory)
        state["previous_image"] = current.clone()
        state["has_previous"][:] = True
        state["previous_cell"] = absolute_cell.clone()
        state["tick"] += 1
        info = dict(log_spatial_patch=patch, log_spatial_pose=state["pose"].clone(),
            log_spatial_version=torch.full_like(u, SPATIAL_VERSION), log_spatial_summary=summary,
            log_spatial_directions=directions, log_spatial_novelty=(1.-repeated)*ground_view.float(),
            log_spatial_revisit=repeated, log_spatial_known_cells=observed,
            log_spatial_visited_cells=visited)
        if self.fix_version:
            info.update(log_spatial_raw_motion=raw_motion*valid.float(),
                log_spatial_rejected_motion=rejected_motion,
                log_spatial_motion_accepted=accepted.float(),
                log_spatial_pair_valid=valid.float(),
                log_spatial_motion_probability=motion_probability*valid.float())
        if self.fix_version >= 2:
            trajectory_valid = state["global_continuity"] if self.fix_version >= 3 else state["tracking_ready"]
            info.update(log_spatial_trajectory_valid=trajectory_valid.float(),
                log_spatial_map_write=map_write.float(), log_spatial_ground_view=ground_view.float(),
                log_spatial_cue_valid=cue_valid.float(), log_spatial_raw_cue=raw_cue,
                log_spatial_motion_unknown=motion_unknown.float(),
                log_spatial_pose_gap_count=state["pose_gap_count"].float(),
                log_spatial_first_gap_step=state["first_gap_step"].float(),
                log_spatial_camera_pitch=state["camera_pitch"].clone(),
                log_spatial_pitch_known=state["pitch_known"].float())
        if self.fix_version >= 3:
            info.update(log_spatial_history_frames=state["history_frames"].clone(),
                log_spatial_history_meta=state["history_meta"].clone(),
                log_spatial_history_valid=state["history_valid"].float(),
                log_spatial_history_tick=(state["tick"]-1).float(),
                log_spatial_segment=state["segment"].float(),
                log_spatial_local_valid=state["tracking_ready"].float(),
                log_spatial_memory_available=state["history_valid"].any(-1).float(),
                log_spatial_visual_revisit=history_repeated.float())
        if goal is not None:
            info.update(log_goal_memory=goal_memory, log_goal_direction_novelty=
                        self.goal_direction_novelty(state, directions, map_write))
        return context, state, info

    def batch(self, features, data):
        lead, device = features.shape[:-1], features.device
        patch = torch.as_tensor(data.get("vnext_spatial_patch", np.zeros((*lead, self.patch_size, self.patch_size, 6), np.uint8)), device=device)
        pose = torch.as_tensor(data.get("vnext_spatial_pose", np.zeros((*lead, 4), np.float32)), device=device).float()
        valid = torch.as_tensor(data.get("vnext_spatial_valid", np.zeros(lead)), device=device).bool()
        snapshot_valid = valid.clone()
        if self.fix_version >= 3:
            local_valid = torch.as_tensor(data.get("vnext_spatial_local_valid", np.zeros(lead)), device=device).bool() & snapshot_valid
            trajectory_valid = torch.as_tensor(data.get("vnext_spatial_trajectory_valid", np.zeros(lead)), device=device).bool() & snapshot_valid
            shape = (*lead, self.history_slots)
            frames = torch.as_tensor(data.get("vnext_spatial_history_frames", np.zeros((*shape,4,4,4), np.uint8)), device=device)
            meta = torch.as_tensor(data.get("vnext_spatial_history_meta", np.zeros((*shape,8), np.float32)), device=device).float()
            history_valid = torch.as_tensor(data.get("vnext_spatial_history_valid", np.zeros(shape)), device=device).bool()
            tick = torch.as_tensor(data.get("vnext_spatial_history_tick", np.zeros(lead)), device=device).float()
            present = torch.as_tensor(data.get("vnext_spatial_history_snapshot_valid", np.zeros(lead)), device=device).bool()
            if frames.shape != (*shape,4,4,4) or meta.shape != (*shape,8) or history_valid.shape != shape or tick.shape != lead:
                raise ValueError("Persistent view history Replay shapes differ from configured slots")
            present &= snapshot_valid & history_valid.any(-1)
            finite = torch.isfinite(meta).flatten(-2).all(-1) & torch.isfinite(tick)
            finite &= torch.isfinite(frames).flatten(-4).all(-1)
            causal = (~history_valid) | ((meta[...,0]>=0) &
                (meta[...,1]>=meta[...,0]) & (meta[...,1]<=tick[...,None]))
            present &= finite & causal.all(-1)
            history_valid &= present[...,None]
            history_context = self.history_encoder(features, frames, meta.nan_to_num(), history_valid, tick.nan_to_num())
            map_finite = torch.isfinite(pose).all(-1) & torch.isfinite(patch).flatten(-3).all(-1)
            local_valid &= map_finite & (pose[...,3]>=0)
            pose = pose.nan_to_num(nan=0., posinf=0., neginf=0.).clone()
            pose[...,3].clamp_min_(0.)
            patch = patch.nan_to_num(nan=0., posinf=0., neginf=0.)
            map_context = self.encoder(patch, pose)*local_valid[...,None]
            context = self.fuse(map_context, history_context, local_valid, trajectory_valid,
                pose[...,3], present.float())
            if self.goal_search:
                goal_memory = torch.as_tensor(data.get("vnext_goal_memory", np.zeros((*lead,6))),
                    device=device).float()
                if goal_memory.shape != (*lead,6):
                    raise ValueError("Goal memory Replay needs [batch,time,6]")
                goal_memory = goal_memory.nan_to_num().clamp(0, 1)
                context = context + self.goal_memory_encoder(goal_memory)*present[...,None]
            return context, dict(valid=present, snapshot_valid=snapshot_valid, patch=patch, pose=pose,
                map_valid=local_valid, map_context=map_context, history_context=history_context)
        if self.fix_version >= 2:
            valid &= torch.as_tensor(data.get("vnext_spatial_trajectory_valid", np.zeros(lead)), device=device).bool()
        context = self.encoder(patch, pose)*valid[..., None]
        return context, dict(valid=valid, snapshot_valid=snapshot_valid, patch=patch, pose=pose)

    def loss(self, features, data, context, details, train_mask):
        device = features.device
        lead = features.shape[:-1]
        labels, known = [], []
        version = torch.as_tensor(data.get("search_label_version", np.zeros(lead)), device=device) >= 3
        for name in ("visible", "new", "progress"):
            labels.append(torch.as_tensor(data.get("search_"+name, np.zeros(lead)), device=device).float())
            known.append(torch.as_tensor(data.get("search_"+name+"_valid", np.zeros(lead)), device=device).bool() & version)
        labels, known = torch.stack(labels, -1), torch.stack(known, -1)
        map_valid = details.get("map_valid", details["valid"])
        mask = known & train_mask[..., None] & map_valid[..., None]
        map_context = details.get("map_context", context)
        loss = F.binary_cross_entropy_with_logits(self.event_head(map_context), labels.nan_to_num().clamp(0, 1), reduction="none")
        map_loss = (loss*mask).sum()/mask.sum().clamp_min(1)
        history_loss = map_loss*0
        if self.fix_version >= 3:
            history_mask = known & train_mask[...,None] & details["valid"][...,None]
            error = F.binary_cross_entropy_with_logits(self.history_head(context),
                labels.nan_to_num().clamp(0,1), reduction="none")
            history_loss = (error*history_mask).sum()/history_mask.sum().clamp_min(1)
        pose_known = torch.as_tensor(data.get("search_pose_valid", np.zeros(lead)), device=device).bool()
        pose_known &= torch.as_tensor(data.get("vnext_pose_prev_valid", np.zeros(lead)), device=device).bool()
        window = torch.as_tensor(data.get("vnext_window_valid", np.ones(lead)), device=device).bool()
        calibration = torch.as_tensor(data.get("vnext_calibration", np.zeros(lead)), device=device).bool()
        pose_mask = pose_known & train_mask.bool() & ~calibration & window
        pose_calibration = pose_known & calibration & window
        current = torch.as_tensor(data["image"], device=device)
        previous = torch.as_tensor(data.get("vnext_pose_prev_image", data["image"]), device=device)
        control = torch.as_tensor(data.get("vnext_pose_control", np.zeros((*lead, 9))), device=device).float()
        target = torch.as_tensor(data.get("search_pose_delta", np.zeros((*lead, 3))), device=device).float()
        pose_loss, metrics, _ = self.pose.supervised_loss(previous, current, control, target, pose_mask,
            calibration_mask=pose_calibration, frame_ids=data.get("vnext_frame_id"))
        total = map_loss+history_loss+float(cfg(self.config, "vnext_pose_loss_weight"))*pose_loss
        metrics.update({"B/map_loss": map_loss.detach(), "B/total_loss": total.detach(),
                        "B/map_valid": map_valid.float().mean().detach()})
        if self.fix_version >= 3:
            metrics.update({"B/history_loss":history_loss.detach(),
                "B/history_valid":details["valid"].float().mean().detach(),
                "B/memory_valid":details["valid"].float().mean().detach()})
        return total, metrics, features.new_zeros(2)

    @torch.no_grad()
    def visualization_state(self, state):
        if state is None:
            return None
        used = int(state["ledger_used"][0])
        ledger = {"cells": state["ledger_cells"][0, :used].detach().cpu().numpy().copy(),
                  "values": state["ledger_values"][0, :used].detach().cpu().numpy().copy(),
                  "directions": state["ledger_directions"][0, :used].detach().cpu().numpy().copy()}
        result = dict(grid=state["grid"][0].permute(1, 2, 0).detach().cpu().numpy().copy(),
            pose=state["pose"][0].detach().cpu().numpy().copy(),
            origin=state["origin"][0].detach().cpu().numpy().copy(), resolution=self.resolution,
            channels=MAP_CHANNELS, predicted=True, observation_type="directional-view proxy",
            target_type="cue-observation viewpoint", tracking_ready=bool(state["tracking_ready"][0]),
            localization=("estimated" if bool(state["tracking_ready"][0]) else "unknown: pose not qualified at episode start"),
            ledger=ledger, ledger_capacity=self.ledger_capacity)
        if self.fix_version >= 2:
            started = bool(state["trajectory_started"][0])
            result.update(trajectory_valid=bool(state["tracking_ready"][0]),
                trajectory_started=started, pose_gap_count=int(state["pose_gap_count"][0]),
                first_gap_step=int(state["first_gap_step"][0]),
                camera_pitch=float(state["camera_pitch"][0]), pitch_known=bool(state["pitch_known"][0]),
                map_write_policy="qualified continuous origin pose only; ground-view proxies require known pitch",
                localization=("estimated continuous origin pose" if bool(state["tracking_ready"][0]) else
                              "unknown after motion/qualification gap; historical prefix retained" if started else
                              "unknown: no qualified episode-origin pose"))
        if self.fix_version >= 3:
            ledger["segments"] = state["ledger_segment"][0, :used].detach().cpu().numpy().copy()
            result.update(trajectory_valid=bool(state["global_continuity"][0]),
                local_tracking_ready=bool(state["tracking_ready"][0]),
                active_segment=int(state["segment"][0]), memory_available=bool(state["history_valid"][0].any()),
                history_frames=state["history_frames"][0].detach().cpu().numpy().copy(),
                history_meta=state["history_meta"][0].detach().cpu().numpy().copy(),
                history_valid=state["history_valid"][0].detach().cpu().numpy().copy(),
                map_write_policy="qualified local segments; unknown inter-segment transforms are never filled",
                localization=("estimated continuous origin pose" if bool(state["global_continuity"][0]) else
                    "estimated local segment; global relation unknown" if bool(state["tracking_ready"][0]) else
                    "view history active; local pose currently unknown"))
        if self.goal_search:
            ledger["goal_evidence"] = state["ledger_goal"][0, :used].detach().cpu().numpy().copy()
            result["history_goal"] = state["history_goal"][0].detach().cpu().numpy().copy()
            result["goal_evidence_tick"] = int(state["tick"][0])
        return result
