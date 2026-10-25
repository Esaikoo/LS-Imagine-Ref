"""B v2: recurrent search belief, durable visual keyframes and imagined memory.

Only RGB, heatmap and executed actions are inference inputs. Place IDs and ray
events are supervision, never retrieval keys or policy inputs. Replay rebuilds
the bank from the entire past and burns in the recurrent belief on recent history.
"""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GRID = 8
DESC_DIM = GRID * GRID * 4
META_DIM = 7  # created, last_seen, visits, candidate, negative, ambiguity, novelty
B_VERSION = 2
MATCH_DISTANCE = 0.012


def describe_torch(image, heatmap):
    lead = image.shape[:-3]
    x = image.float().reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2)
    h = heatmap.float()
    if h.ndim == image.ndim - 1:
        h = h.unsqueeze(-1)
    h = h.reshape(-1, *h.shape[-3:]).permute(0, 3, 1, 2)
    out = torch.cat([F.adaptive_avg_pool2d(x, (GRID, GRID)),
                     F.adaptive_avg_pool2d(h, (GRID, GRID))], 1)
    return out.flatten(1).reshape(*lead, DESC_DIM)


def describe_numpy(image, heatmap):
    x, h = np.asarray(image, np.float32), np.asarray(heatmap, np.float32)
    lead = x.shape[:-3]
    if x.shape[-3:] != (64, 64, 3):
        raise ValueError("B replay expects the configured 64x64 RGB observations")
    x = x.reshape(-1, 8, 8, 8, 8, 3).mean((2, 4)).transpose(0, 3, 1, 2)
    h = h.reshape(-1, 8, 8, 8, 8).mean((2, 4))
    return np.concatenate([x.reshape(-1, 192), h.reshape(-1, 64)], -1).reshape(*lead, DESC_DIM) / 255.


def _numpy_bank_step(bank, valid, meta, indices, tick, scene, save_every):
    """Deterministic bank policy, mirrored by SearchMemory.add and regression tested."""
    distances = ((bank[:, :192] - scene[:192]) ** 2).mean(-1)
    distances[~valid] = 1.e6
    nearest = int(distances.argmin())
    match = bool(valid.any() and distances[nearest] < MATCH_DISTANCE)
    cue = float(scene[192:].max())
    novelty = min(float(distances.min()), .2) / .2 if valid.any() else 1.
    if match:
        slot = nearest
        meta[slot, 1] = tick
        meta[slot, 2] += 1
        meta[slot, 3] = cue
        meta[slot, 4] += cue < .15
        meta[slot, 5] = 4 * cue * (1 - cue)
    elif not valid.any() or tick % save_every == 0 or cue > meta[:, 3].max() + .1:
        if not valid.all():
            slot = int(np.flatnonzero(~valid)[0])
        else:
            retention = 2 * meta[:, 3] + meta[:, 6] - .0005 * (tick - meta[:, 1])
            # One episode anchor survives; the remaining slots compete on evidence/novelty.
            if len(bank) > 1:
                retention[0] += 1.e6
            slot = int(retention.argmin())
            if 2 * cue + novelty < retention[slot]:
                return
        valid[slot] = True
        meta[slot] = (tick, tick, 1, cue, float(cue < .15), 4 * cue * (1 - cue), novelty)
    else:
        return
    bank[slot] = scene
    indices[slot] = tick


def _prefix_state(descriptors, index, slots, save_every):
    bank = np.zeros((slots, DESC_DIM), np.float32)
    valid = np.zeros(slots, bool)
    meta = np.zeros((slots, META_DIM), np.float32)
    indices = np.full(slots, -1, np.int64)
    for t in range(index):
        _numpy_bank_step(bank, valid, meta, indices, t, descriptors[t], save_every)
    return bank, valid, meta, indices


def make_prefix(episode, index, slots=16, lookback=256, save_every=8):
    """Full-past keyframe bank, not uniform sampling limited to the recent lookback."""
    descriptors = describe_numpy(np.asarray(episode["image"])[:index],
                                 np.asarray(episode["heatmap"])[:index]) if index else np.zeros((0, DESC_DIM))
    bank, valid, _, _ = _prefix_state(descriptors, index, slots, save_every)
    return bank, valid.astype(np.float32)


def sample_episodes_with_history(episodes, length, slots=16, lookback=128, seed=0, save_every=8):
    """Normal stitched RSSM windows + one past-only bank and bounded belief burn-in.

    History tensors are batch metadata, not per-transition NPZ fields. At any
    stitched real episode boundary the bank and recurrent belief are both reset.
    """
    import tools
    from search_evidence import complete_episode_labels
    rng = np.random.RandomState(seed)
    cache = OrderedDict()
    while True:
        candidates = [ep for ep in episodes.values() if len(ep["reward"]) >= 2]
        if not candidates:
            raise ValueError("Replay has no episode with transitions")
        probs = np.asarray([len(ep["reward"]) for ep in candidates], np.float64)
        probs /= probs.sum()
        ret, history, size = None, None, 0
        while size < length:
            original = candidates[rng.choice(len(candidates), p=probs)]
            episode = tools.replace_none_with_zeros(tools.selective_deepcopy(complete_episode_labels(original)))
            total = len(episode["reward"])
            index = int(rng.randint(0, total - 1)) if ret is None else 0
            n = min(length - size, total - index)
            part = {k: np.asarray(v[index:index+n]).copy() for k, v in episode.items()
                    if not k.startswith(("log_", "search_b_"))}
            part["search_b_real_first"] = np.asarray(episode["is_first"][index:index+n], np.float32).copy()
            if ret is None:
                # Cache only raw descriptors, never encoder outputs that become stale.
                key = (id(original), total)
                if key not in cache or cache[key][0] is not original:
                    cache[key] = (original, describe_numpy(episode["image"], episode["heatmap"]))
                    while len(cache) > 16:
                        cache.popitem(last=False)
                cache.move_to_end(key)
                descriptors = cache[key][1]
                start = max(0, index - lookback)
                bank, valid, meta, ids = _prefix_state(descriptors, start, slots, save_every)
                place = np.asarray(episode["search_place_id"], np.float32)
                place_valid = np.asarray(episode["search_place_valid"], bool)
                place = np.where(place_valid, place, 0)
                prefix_place = np.where(valid, place[np.maximum(ids, 0)], 0)
                hscene = np.zeros((lookback, DESC_DIM), np.float32)
                haction = np.zeros((lookback, 12), np.float32)
                hvalid = np.zeros(lookback, np.float32)
                hplace = np.zeros(lookback, np.float32)
                count = index - start
                if count:
                    hscene[-count:] = descriptors[start:index]
                    haction[-count:] = np.asarray(episode["action"])[start:index, :12]
                    hvalid[-count:] = 1
                    hplace[-count:] = place[start:index]
                history = dict(search_b_prefix=bank, search_b_prefix_valid=valid.astype(np.float32),
                               search_b_prefix_meta=meta, search_b_prefix_indices=ids,
                               search_b_prefix_place=prefix_place, search_b_prefix_tick=np.float32(start),
                               search_b_history=hscene, search_b_history_action=haction,
                               search_b_history_valid=hvalid, search_b_history_place=hplace)
                ret = part
            else:
                ret = {k: np.concatenate([ret[k], part[k]], 0) for k in ret}
            ret["is_first"][size] = True
            size += n
        ret.update(history)
        yield ret


class SearchMemory(nn.Module):
    def __init__(self, slots=16, dim=128, hidden=128, save_every=8, feature_dim=None,
                 dynamic_target=False):
        super().__init__()
        self.slots, self.dim, self.save_every = int(slots), int(dim), int(save_every)
        if min(self.slots, self.dim, self.save_every) < 1:
            raise ValueError("Invalid B memory dimensions")
        self.encoder = nn.Sequential(nn.Linear(DESC_DIM, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.combine = nn.Sequential(nn.Linear(2 * dim + 7, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.belief = nn.GRUCell(dim + 12, dim)
        self.revisit_head = nn.Linear(dim, 1)
        self.evidence_head = nn.Linear(dim, 2)  # currently visible / new scene, supervised only
        # Learnable evidence half-life; dynamic targets start with faster decay.
        self.decay_log_rate = nn.Parameter(torch.tensor(-4.2 if dynamic_target else -6.2))
        self.feature_dim = feature_dim
        if feature_dim is not None:
            self.prediction_input = nn.Sequential(nn.Linear(feature_dim + 13 + 1, hidden),
                                                  nn.SiLU(), nn.Linear(hidden, dim))
            self.prediction = nn.GRUCell(dim, dim)

    def initial(self, batch, device):
        return (torch.zeros(batch, self.slots, DESC_DIM, device=device),
                torch.zeros(batch, self.slots, dtype=torch.bool, device=device),
                torch.zeros(batch, self.slots, META_DIM, device=device),
                torch.full((batch, self.slots), -1, dtype=torch.long, device=device),
                torch.zeros(batch, dtype=torch.long, device=device),
                torch.zeros(batch, self.dim, device=device))

    def reset(self, state, reset):
        fresh = self.initial(len(reset), reset.device)
        return tuple(torch.where(reset.reshape(-1, *([1] * (x.ndim-1))), y, x)
                     for x, y in zip(state, fresh))

    def retrieve(self, scene, bank, valid, meta, tick):
        age = (tick[:, None] - meta[..., 1]).float().clamp_min(0)
        decay = torch.exp(-F.softplus(self.decay_log_rate) * age)
        # Place addresses depend on RGB, not a target appearing/disappearing.
        # Evidence values age BEFORE encoding so raw heatmaps cannot bypass decay.
        place_input = torch.cat((scene[:, None], bank), 1).float()
        place_input = torch.cat((place_input[..., :192],
                                 torch.zeros_like(place_input[..., 192:])), -1)
        aged_bank = torch.cat((bank[..., :192].float(),
                               bank[..., 192:].float() * decay[..., None]), -1)
        value_input = torch.cat((scene[:, None].float(), aged_bank), 1)
        encoded = F.normalize(self.encoder(torch.cat((place_input, value_input), 1)), dim=-1)
        addresses, values = encoded.split(bank.shape[1] + 1, dim=1)
        cosine = torch.einsum("bd,bkd->bk", addresses[:, 0], addresses[:, 1:])
        weights = (cosine * 5).masked_fill(~valid, -1.e4).softmax(-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.e-6)
        pooled = torch.einsum("bk,bkd->bd", weights, values[:, 1:])
        candidate = meta[..., 3] * decay
        negative = meta[..., 4] * decay
        ambiguity = meta[..., 5] * decay
        # Visit counts and static scene appearance remain; target evidence expires.
        stats = torch.stack((torch.log1p(age)/8, torch.log1p(meta[..., 2])/5,
                             candidate, torch.log1p(negative)/5, ambiguity), -1)
        stats = (stats * weights[..., None]).sum(1)
        max_sim = cosine.masked_fill(~valid, -1).max(-1).values
        max_sim = torch.where(valid.any(-1), max_sim, torch.zeros_like(max_sim))
        combined = torch.cat((values[:, 0], pooled, stats, max_sim[:, None],
                              valid.float().mean(-1, keepdim=True)), -1)
        return self.combine(combined), cosine

    @torch.no_grad()
    def add(self, bank, valid, meta, indices, tick, scene):
        bank, valid, meta, indices = (x.detach().clone() for x in (bank, valid, meta, indices))
        dist = (bank[..., :192] - scene[:, None, :192]).square().mean(-1).masked_fill(~valid, 1.e6)
        minimum, nearest = dist.min(-1)
        has = valid.any(-1)
        match = has & (minimum < MATCH_DISTANCE)
        cue = scene[:, 192:].max(-1).values
        novelty = torch.where(has, minimum.clamp(max=.2)/.2, torch.ones_like(minimum))
        retention = 2 * meta[..., 3] + meta[..., 6] - .0005 * (tick[:, None] - meta[..., 1])
        if self.slots > 1:
            retention[:, 0] += 1.e6
        free = (~valid).float().argmax(-1)
        replace = retention.argmin(-1)
        slot = torch.where(match, nearest, torch.where(valid.all(-1), replace, free))
        allowed = ~has | (tick % self.save_every == 0) | (cue > meta[..., 3].max(-1).values + .1)
        incoming = 2 * cue + novelty
        allowed &= ~valid.all(-1) | (incoming >= retention.gather(1, slot[:, None]).squeeze(1))
        write = match | allowed
        rows = write.nonzero().squeeze(-1)
        if rows.numel():
            si = slot[rows]
            old = meta[rows, si]
            is_match = match[rows]
            new = torch.stack((tick[rows].float(), tick[rows].float(), torch.ones_like(cue[rows]),
                               cue[rows], (cue[rows] < .15).float(),
                               4 * cue[rows] * (1-cue[rows]), novelty[rows]), -1)
            new[:, 0] = torch.where(is_match, old[:, 0], new[:, 0])
            new[:, 2] += torch.where(is_match, old[:, 2], torch.zeros_like(old[:, 2]))
            new[:, 4] += torch.where(is_match, old[:, 4], torch.zeros_like(old[:, 4]))
            new[:, 6] = torch.where(is_match, old[:, 6], new[:, 6])
            bank[rows, si], valid[rows, si], meta[rows, si] = scene[rows], True, new
            indices[rows, si] = tick[rows]
        return bank, valid, meta, indices, tick + 1

    def step(self, scene, action, state):
        bank, valid, meta, indices, tick, belief = state
        retrieved, cosine = self.retrieve(scene, bank, valid, meta, tick)
        belief = self.belief(torch.cat((retrieved, action[..., :12].float()), -1), belief.float())
        next_bank = self.add(bank, valid, meta, indices, tick, scene)
        return belief, (*next_bank, belief), cosine

    @torch.no_grad()
    def online_step(self, scene, state=None, is_first=None, previous_action=None):
        if scene.ndim != 2:
            raise ValueError("B online_step expects [batch, descriptor]")
        state = self.initial(len(scene), scene.device) if state is None else tuple(x.detach() for x in state)
        reset = torch.zeros(len(scene), dtype=torch.bool, device=scene.device) if is_first is None else is_first.reshape(-1).bool()
        state = self.reset(state, reset)
        action = torch.zeros(len(scene), 12, device=scene.device) if previous_action is None else previous_action[..., :12]
        action = action * (~reset).float()[:, None]
        used = state[1].any(-1).float()
        context, state, _ = self.step(scene, action, state)
        return context.detach(), tuple(x.detach() for x in state), {
            "log_b_revisit_prob": self.revisit_head(context).sigmoid().squeeze(-1),
            "log_b_memory_used": used,
            "log_b_memory_slots": state[1].sum(-1).float(),
            "log_b_keyframe_indices": state[3].clone(),
            "log_b_observation_index": state[4] - 1,
        }

    def imagine_step(self, memory, next_feature, action, elapsed=1):
        if self.feature_dim is None:
            raise ValueError("B imagined memory needs a configured RSSM feature dimension")
        if action.shape[-1] == 12:
            action = F.pad(action, (0, 1))
        dt = torch.as_tensor(elapsed, device=memory.device, dtype=torch.float32)
        dt = torch.broadcast_to(dt, memory.shape[:-1] + (1,)) if dt.ndim else dt.expand(*memory.shape[:-1], 1)
        inp = torch.cat((F.layer_norm(next_feature.float(), (next_feature.shape[-1],)),
                         action.float(), torch.log1p(dt.clamp(0, 10000))/8), -1)
        lead = memory.shape[:-1]
        return self.prediction(self.prediction_input(inp).reshape(-1, self.dim),
                               memory.float().reshape(-1, self.dim)).reshape(*lead, self.dim)

    def _burn_in(self, scene, batch):
        b = len(scene)
        state = self.initial(b, scene.device)
        places = torch.zeros(b, self.slots, device=scene.device)
        # RSSM's synthetic is_first cannot prove a true episode boundary.
        if "search_b_real_first" not in batch:
            raise ValueError("B replay missing historical context and real episode boundaries")
        if "search_b_prefix" not in batch:
            if not batch["search_b_real_first"][:, 0].bool().all():
                raise ValueError("B replay missing historical context")
            return state, places
        bank = batch["search_b_prefix"].float()
        if bank.shape != (b, self.slots, DESC_DIM):
            raise ValueError("B v2 prefix must match [batch, configured slots, descriptor]")
        state = (bank, batch["search_b_prefix_valid"].bool(), batch["search_b_prefix_meta"].float(),
                 batch["search_b_prefix_indices"].long(), batch["search_b_prefix_tick"].long(), state[-1])
        places = batch["search_b_prefix_place"].float()
        with torch.no_grad():
            for i in range(batch["search_b_history"].shape[1]):
                active = batch["search_b_history_valid"][:, i].bool()
                if not active.any():
                    continue
                old = state
                _, new, _ = self.step(batch["search_b_history"][:, i], batch["search_b_history_action"][:, i], state)
                changed = new[3] != old[3]
                places = torch.where(changed & active[:, None],
                                     batch["search_b_history_place"][:, i, None], places)
                state = tuple(torch.where(active.reshape(-1, *([1] * (x.ndim-1))), x, y)
                              for x, y in zip(new, old))
        return tuple(x.detach() for x in state), places

    def train_sequence(self, scene, batch):
        b, t, _ = scene.shape
        state, bank_places = self._burn_in(scene, batch)
        first = batch["search_b_real_first"].reshape(b, t).bool()
        contexts, place_errors, place_masks = [], [], []
        actions = batch.get("action", torch.zeros(b, t, 12, device=scene.device))
        for i in range(t):
            reset = first[:, i]
            state = self.reset(state, reset)
            bank_places = torch.where(reset[:, None], torch.zeros_like(bank_places), bank_places)
            old_ids, valid_bank = state[3], state[1]
            action = actions[:, i, :12] * (~reset).float()[:, None]
            context, state, cosine = self.step(scene[:, i], action, state)
            contexts.append(context)
            place = batch.get("search_place_id", torch.zeros(b, t, device=scene.device))[:, i]
            known = batch.get("search_place_valid", torch.zeros(b, t, device=scene.device))[:, i].bool()
            mask = valid_bank & (bank_places > 0) & known[:, None] & (place[:, None] > 0)
            labels = (bank_places == place[:, None]).float()
            place_errors.append(F.binary_cross_entropy_with_logits((cosine.float()-.5)*8, labels, reduction="none"))
            place_masks.append(mask)
            bank_places = torch.where(state[3] != old_ids, torch.where(known, place, torch.zeros_like(place))[:, None], bank_places)
        contexts = torch.stack(contexts, 1)
        logits = self.revisit_head(contexts).squeeze(-1).float()
        labels = batch["search_revisit"].reshape(b, t).float()
        mask = batch["search_revisit_valid"].reshape(b, t).bool() & (batch["search_label_version"].reshape(b,t) == 2)
        mask &= torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
        errors = F.binary_cross_entropy_with_logits(logits, labels.nan_to_num().clamp(0,1), reduction="none")
        revisit = (errors * mask).sum()/mask.sum().clamp_min(1)
        evidence_logits = self.evidence_head(contexts).float()
        evidence = torch.zeros((), device=scene.device)
        for j, key in enumerate(("visible", "new")):
            y = batch.get("search_"+key, torch.zeros_like(labels)).reshape(b,t).float()
            known = batch.get("search_"+key+"_valid", torch.zeros_like(labels)).reshape(b,t).bool()
            known &= (batch["search_label_version"].reshape(b,t) == 2) & torch.isfinite(y) & (y>=0) & (y<=1)
            e = F.binary_cross_entropy_with_logits(evidence_logits[...,j], y.nan_to_num().clamp(0,1), reduction="none")
            evidence = evidence + (e*known).sum()/known.sum().clamp_min(1)/2
        place_mask = torch.stack(place_masks,1)
        place_loss = (torch.stack(place_errors,1)*place_mask).sum()/place_mask.sum().clamp_min(1)
        loss = revisit + .25*evidence + .1*place_loss
        return contexts, loss, {
            "B/revisit_loss":revisit.detach(), "B/revisit_valid":mask.sum().detach(),
            "B/revisit_positive_rate":(labels.nan_to_num().clamp(0,1)*mask).sum().detach()/mask.sum().clamp_min(1),
            "B/revisit_brier":((logits.sigmoid()-labels.nan_to_num().clamp(0,1)).square()*mask).sum().detach()/mask.sum().clamp_min(1),
            "B/evidence_loss":evidence.detach(), "B/place_loss":place_loss.detach(),
            "B/place_valid":place_mask.sum().detach(),
            "B/evidence_decay_rate":F.softplus(self.decay_log_rate).detach(),
        }

    def transition_loss(self, memory, features, batch, horizon=6):
        """Learn ordinary and elapsed-time jump updates from executed trajectories."""
        from search_a import valid_windows
        total = memory.shape[1]
        losses, counts = [], []
        if total < 2:
            return memory.sum()*0, torch.zeros((),device=memory.device)
        horizons = {1, min(horizon, total-1), total-1}
        # Train multiple elapsed times, not just a single fixed jump length.
        h = 2
        while h < total:
            horizons.add(h)
            h *= 2
        for h in sorted(horizons):
            valid = valid_windows(batch, h, total)
            actions = batch["action"][:, h:].clone()
            if actions.shape[-1] == 12:
                actions = F.pad(actions, (0,1))
            if h > 1:
                actions = torch.zeros_like(actions)
                actions[..., -1] = 1
            predicted = self.imagine_step(memory[:, :-h].detach(), features[:, h:], actions, elapsed=h)
            error = F.smooth_l1_loss(predicted, memory[:, h:].detach(), reduction="none").mean(-1)
            losses.append((error*valid).sum()/valid.sum().clamp_min(1))
            counts.append(valid.sum())
        return torch.stack(losses).mean(), torch.stack(counts).sum()


def check_b_checkpoint(checkpoint, enabled):
    has_b = any(".search_b." in k for k in checkpoint["agent_state_dict"])
    if enabled and (not has_b or checkpoint.get("search_b_version") != B_VERSION):
        raise ValueError("B v2 needs a B v2 checkpoint; A-only and package B v1 are not compatible")
    if not enabled and has_b:
        raise ValueError("B checkpoint requires apm_ab; use search_b_control=False for fixed-checkpoint ablation")
