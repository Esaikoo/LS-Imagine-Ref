"""Frame-local policy fusion and causal, short-horizon target tracking."""
import torch
from torch.nn import functional as F

from .goal_reference import GRID, VERSION, _heatmap


NO_TRUSTED_TARGET = 0
UNCERTAIN_CANDIDATE = 1
TRUSTED_VISIBLE = 2
HEATMAP, RGB, HARD = range(3)


def proposals(heat, saliency, probability):
    """Three heat peaks, three independent RGB peaks, two difficult regions."""
    hot = F.adaptive_max_pool2d(heat, (GRID, GRID)).squeeze(1)
    n = len(hot)
    axis = torch.arange(GRID, device=hot.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    available = torch.ones_like(hot, dtype=torch.bool)
    ids, boxes, sources = [], [], []
    difficulty = torch.maximum(hot*(1.-probability),
                               (1.-2.*(probability-.5).abs())*.6)
    for source, scores, count, radius in ((HEATMAP, hot, 3, 2), (RGB, saliency, 3, 2),
                                           (HARD, difficulty, 2, 1)):
        for _ in range(count):
            index = scores.masked_fill(~available, -torch.inf).flatten(1).argmax(1)
            ids.append(index)
            y = index // GRID
            x = index % GRID
            box = torch.stack((x-radius, y-radius, x+radius+1, y+radius+1), -1)
            boxes.append(box.float().clamp(0, GRID)/GRID)
            sources.append(torch.full_like(index, source))
            near = ((xx-x[:, None, None]).abs() <= radius) & ((yy-y[:, None, None]).abs() <= radius)
            available &= ~near
    return torch.stack(ids, 1), torch.stack(boxes, 1), torch.stack(sources, 1)


def fuse_heatmap(raw, region, gate, epsilon=.05, overlap=.5):
    """Blend class evidence with the original U-Net image-space guidance."""
    mixed = epsilon*raw + (1.-epsilon)*(overlap*raw*region + (1.-overlap)*region)
    return ((1.-gate)*raw + gate*mixed).clamp(0., 1.)


def _track(center, embed, candidate, qualified, first, previous, actual):
    if previous is None:
        count = torch.zeros_like(candidate, dtype=torch.long)
    else:
        old_center, old_embed, count = previous
        motion = (center-old_center).abs().amax(-1)
        bound = .22
        if actual is not None:
            control = torch.as_tensor(actual, device=motion.device).reshape(-1, 9)
            # Rotation widens the allowed shift without using world yaw/pose.
            bound = bound + control[:, -2:].abs().sum(-1).clamp_max(30.)*.01
        similarity = (old_embed*embed).sum(-1)
        same = (similarity > .7) & (motion < bound) & ~first
        count = torch.where(same, count, torch.zeros_like(count))
    count = torch.where(candidate, (count+1).clamp_max(8), torch.zeros_like(count))
    trusted = (count >= 2) & qualified
    state = torch.where(trusted, TRUSTED_VISIBLE,
                        torch.where(candidate, UNCERTAIN_CANDIDATE, NO_TRUSTED_TARGET))
    return state, (center.detach(), embed.detach(), count.detach())


@torch.no_grad()
def build_policy_observation(raw_obs, verifier, state=None, previous_control=None):
    """Return legal guidance for online and replay, with no labels or mutation.

    The world-model heatmap is frame-local so that recomputation on the exact
    same frame and parameters is independent of the sampled replay boundary.
    Only the separate decision state uses previous causal observations.
    """
    source = verifier.observation_images(raw_obs)
    prediction = verifier(source, raw_obs["heatmap"])
    leading = prediction["leading"]
    p = prediction["probability"]
    raw = prediction["heatmap"]
    reliability = verifier.reliability[0]
    visible_reliability = verifier.reliability[1]
    candidates, boxes, sources = proposals(raw, .5*(prediction["saliency"]+p), p)
    scores = p.flatten(1).gather(1, candidates)
    # A local region score, qualified against the region calibration split,
    # supplies evidence even when no frame-wide absence labels exist.
    index = scores.argmax(-1)
    best_pixel = candidates.gather(1, index[:, None]).squeeze(1)
    peak = scores.gather(1, index[:, None]).squeeze(1)
    embeddings = prediction["embeddings"].flatten(2)
    embedding = embeddings.gather(2, best_pixel[:, None, None].expand(-1, embeddings.shape[1], 1)).squeeze(-1)
    center = (torch.stack((best_pixel % GRID, best_pixel // GRID), -1).float() + .5)/GRID
    region_noise = prediction["region_uncertainty"].flatten(1).gather(1, best_pixel[:, None]).squeeze(1)
    region = F.interpolate(p[:, None], size=raw.shape[-2:], mode="bilinear", align_corners=False)
    local_conflict = raw*(1.-region)
    conflict = local_conflict.flatten(1).amax(-1)
    max_heat = raw.flatten(1).amax(-1)
    conflict = torch.where(max_heat >= .4, conflict, 0.)
    strength = torch.maximum((2.*peak-1.).clamp_min(0.), conflict)
    hot_grid = F.adaptive_max_pool2d(raw, (GRID, GRID)).flatten(1)
    peak_id = hot_grid.argmax(-1, keepdim=True)
    raw_peak_p = p.flatten(1).gather(1, peak_id).squeeze(1)
    raw_peak_noise = prediction["region_uncertainty"].flatten(1).gather(1, peak_id).squeeze(1)
    broad = (raw.flatten(1) >= .45).float().mean(-1) >= .08
    peak_rejected = ((reliability >= .15) & broad & (max_heat >= .55)
                     & (raw_peak_p <= .3) & (raw_peak_noise < .15))
    enabled = verifier.fusion_enabled
    valid = (reliability > 0.) & (strength > .12) & (region_noise < .25)
    gate = verifier.max_gate * reliability * strength * (1.-region_noise) * valid.float()
    if not enabled:
        gate = torch.zeros_like(gate)
    fused = fuse_heatmap(raw, region, gate[:, None, None, None],
                         verifier.fusion_epsilon, verifier.fusion_overlap)
    original = torch.as_tensor(raw_obs["heatmap"], device=verifier.device)
    heatmap_policy = fused.reshape_as(original)
    if original.dtype == torch.uint8:
        heatmap_policy = (heatmap_policy*255.).round().byte()
    else:
        heatmap_policy = heatmap_policy.to(original.dtype)
    # State qualification is separate from a visibility head's confidence.
    # Low-risk control can use sparse region evidence, whereas the controller
    # still owns all centered and actual-interaction safety gates.
    candidate = valid & (peak >= .58)
    qualified = valid & (peak >= .72) & (reliability >= .15) & (region_noise < .12)
    first = torch.as_tensor(raw_obs.get("is_first", torch.zeros(leading)),
                            device=verifier.device).reshape(-1).bool()
    if len(leading) == 2:
        # A training window can start mid-episode. Fusion stays exact; any
        # reconstructed state begins with zero previous evidence.
        batch, time = leading
        states = []
        controls = (None if previous_control is None else
                    torch.as_tensor(previous_control, device=verifier.device).reshape(batch, time, 9))
        for offset in range(time):
            sl = slice(offset, None, time)
            current, state = _track(center[sl], embedding[sl], candidate[sl], qualified[sl],
                                    first[sl], state, None if controls is None else controls[:, offset])
            states.append(current)
        goal_state = torch.stack(states, 1).reshape(-1)
    else:
        goal_state, state = _track(center, embedding, candidate, qualified, first, state, previous_control)
    visibility = torch.lerp(peak, prediction["visible_probability"], visible_reliability*.5)
    confidence = reliability*(2.*(peak-.5).abs())*(1.-region_noise)
    uncertainty = (1.-confidence).clamp(0., 1.)
    context = torch.stack((visibility, confidence, uncertainty,
                           center[:, 0]*2.-1., center[:, 1]*2.-1.,
                           conflict * reliability, goal_state.float(), valid.float()), -1)
    # Spatial inputs and logs keep their existing scale and dimensions.
    obs = dict(raw_obs)
    obs["heatmap"] = heatmap_policy
    evidence = {
        "context": context.reshape(*leading, 8),
        "state": goal_state.reshape(leading),
        "region": region.permute(0, 2, 3, 1).reshape(*leading, *region.shape[-2:], 1),
        "gate": gate.reshape(leading),
        "boxes": boxes.reshape(*leading, 8, 4),
        "candidate_scores": scores.reshape(*leading, 8),
        "candidate_sources": sources.reshape(*leading, 8),
        "conflict": (conflict*reliability).reshape(leading),
        "peak_rejected": peak_rejected.reshape(leading),
        "target_class_visible": visibility.reshape(leading),
        "match_uncertainty": uncertainty.reshape(leading),
        "evidence_valid": valid.reshape(leading),
        "interaction_probability": prediction["interaction_probability"].reshape(leading),
        "interaction_valid": (verifier.reliability[2] > 0).expand(leading),
        "heatmap_policy": heatmap_policy,
        "version": torch.full(leading, VERSION, device=verifier.device, dtype=torch.uint8),
    }
    return obs, evidence, state


@torch.no_grad()
def fuse_zoomed_observation(raw_obs, verifier):
    """Use the same frame-local fusion for existing Long Imagination inputs."""
    zoomed = raw_obs.get("zoomed_image")
    heatmap = raw_obs.get("heatmap_on_zoomed")
    if zoomed is None or heatmap is None or not verifier.fusion_enabled:
        return raw_obs
    valid = torch.as_tensor(raw_obs["is_zoomed"], device=verifier.device).bool()
    if not bool(valid.any()):
        return raw_obs
    rgb = torch.as_tensor(zoomed, device=verifier.device)
    heat = torch.as_tensor(heatmap, device=verifier.device)
    valid = valid.reshape(heat.shape[:-2])
    filtered = {"image": rgb[valid], "heatmap": heat[valid]}
    fused, _, _ = build_policy_observation(filtered, verifier)
    out = dict(raw_obs)
    out["heatmap_on_zoomed"] = heat.clone()
    out["heatmap_on_zoomed"][valid] = fused["heatmap"]
    return out
