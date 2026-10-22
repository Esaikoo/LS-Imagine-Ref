"""A v2: future RSSM feature alignment and verified exploration-event prediction.

Only posterior features and real 12-way actions are inputs. Privileged evidence
is a masked training target, never a policy observation. No B/C dependency.
"""
import torch
from torch import nn
from torch.nn import functional as F

ACTION_DIM = 12
MODEL_VERSION = 2
EVENTS = ("new", "discovery", "revisit")
MACRO_NAMES = ("turn_left", "turn_right", "forward", "forward_left", "forward_right", "actor")
ACTOR_ID = 5


def macro_actions(horizon, device):
    if horizon < 2:
        raise ValueError("A horizon must be >= 2")
    idx = torch.full((5, horizon), 6, dtype=torch.long, device=device)
    idx[0], idx[1] = 4, 5
    idx[3, horizon // 2:] = 4
    idx[4, horizon // 2:] = 5
    return F.one_hot(idx, ACTION_DIM).float()


def heatmap_evidence(heatmaps):
    """Current visual proxy is only a gate; it is NOT a discovery label."""
    leading = heatmaps.shape[:-3]
    flat = heatmaps.reshape(-1, *heatmaps.shape[-3:])
    ground = flat[:, flat.shape[1] // 3:, :, :].flatten(1)
    top_k = max(1, int(ground.shape[-1] * 0.02))
    return ground.topk(top_k, dim=-1).values.mean(-1).reshape(*leading).clamp(0, 1)


def normalize_feature(feat):
    # Match a detached, per-feature-vector normalized posterior target.
    return F.layer_norm(feat.float(), (feat.shape[-1],))


def valid_windows(batch, horizon, total_time):
    first = batch["is_first"].reshape(batch["action"].shape[0], total_time).bool()
    count = total_time - horizon
    invalid = torch.stack([first[:, t+1:t+horizon+1].any(-1) for t in range(count)], 1)
    actions = batch["action"]
    if actions.shape[-1] > ACTION_DIM:
        jumps = actions[..., ACTION_DIM].bool()
        invalid |= torch.stack([jumps[:, t+1:t+horizon+1].any(-1) for t in range(count)], 1)
    for key in ("is_terminal", "is_last"):
        if key in batch:
            end = batch[key].reshape_as(first).bool()
            invalid |= torch.stack([end[:, t:t+horizon].any(-1) for t in range(count)], 1)
    return ~invalid


class SearchOutcomeHead(nn.Module):
    def __init__(self, feature_dim, horizon=6, hidden=256, future_weight=1.0,
                 event_weight=1.0):
        super().__init__()
        self.horizon = int(horizon)
        if self.horizon < 2 or min(future_weight, event_weight) < 0:
            raise ValueError("Invalid A horizon/loss weights")
        self.future_weight, self.event_weight = future_weight, event_weight
        self.input = nn.Sequential(nn.Linear(feature_dim + horizon * ACTION_DIM, hidden), nn.SiLU())
        self.residual = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.future = nn.Linear(hidden, feature_dim)
        self.events = nn.Linear(hidden, len(EVENTS))
        self.register_buffer("label_counts", torch.zeros(len(EVENTS)))
        self.register_buffer("positive_counts", torch.zeros(len(EVENTS)))

    def forward(self, feat, actions):
        if actions.shape[-2:] != (self.horizon, ACTION_DIM) or feat.shape[:-1] != actions.shape[:-2]:
            raise ValueError("A expects matching feature batches and [..., H, 12] actions")
        inp = torch.cat([normalize_feature(feat), actions.flatten(-2)], -1)
        hidden = self.input(inp)
        hidden = hidden + self.residual(hidden)
        return {"future": self.future(hidden), "event_logits": self.events(hidden)}

    def training_loss(self, posterior_features, batch):
        batch_size, total_time = posterior_features.shape[:2]
        h = self.horizon
        if total_time <= h:
            raise ValueError("batch_length must exceed search_a_horizon")
        actions = batch["action"]
        windows = torch.stack([actions[:, t+1:t+h+1, :ACTION_DIM] for t in range(total_time-h)], 1)
        valid = valid_windows(batch, h, total_time)
        pred = self(posterior_features[:, :-h], windows)
        # Stop-gradient future targets prevent minimizing loss by moving targets.
        target = normalize_feature(posterior_features[:, h:].detach())
        feature_error = F.smooth_l1_loss(pred["future"].float(), target, reduction="none").mean(-1)
        future_loss = (feature_error * valid).sum() / valid.sum().clamp_min(1)
        loss = self.future_weight * future_loss
        metrics = {"A/future_loss": future_loss.detach(), "A/valid_windows": valid.sum().detach()}
        counts, positives = [], []
        for i, event in enumerate(EVENTS):
            key = "search_" + event
            labels = batch.get(key, torch.zeros_like(batch["is_first"])).reshape(batch_size, total_time)
            known = batch.get(key + "_valid", torch.zeros_like(labels)).reshape_as(labels).bool()
            version = batch.get("search_label_version", torch.zeros_like(labels)).reshape_as(labels)
            known &= (version == MODEL_VERSION) & torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
            labels = labels.nan_to_num().clamp(0, 1)
            if event == "revisit":
                target_event, event_known = labels[:, h:], known[:, h:]
            else:
                ys = torch.stack([labels[:, t+1:t+h+1] for t in range(total_time-h)], 1)
                ks = torch.stack([known[:, t+1:t+h+1] for t in range(total_time-h)], 1)
                positive = ((ys > 0.5) & ks).any(-1)
                # A known positive establishes "any"; negative requires every step known.
                target_event = positive.float()
                event_known = positive | ks.all(-1)
            mask = valid & event_known
            count = mask.sum()
            logits = pred["event_logits"][..., i].float()
            error = F.binary_cross_entropy_with_logits(logits, target_event.float(), reduction="none")
            event_loss = (error * mask).sum() / count.clamp_min(1)
            loss = loss + self.event_weight * event_loss / len(EVENTS)
            counts.append(count)
            positives.append((target_event * mask).sum())
            metrics.update({f"A/{event}_loss": event_loss.detach(),
                            f"A/{event}_valid": count.detach(),
                            f"A/{event}_positive_rate": ((target_event * mask).sum() / count.clamp_min(1)).detach(),
                            f"A/{event}_brier": (((logits.sigmoid()-target_event)**2 * mask).sum()/count.clamp_min(1)).detach()})
        # These buffers summarize labels used in completed optimization steps.
        metrics["A/prediction_loss"] = loss.detach()
        return loss, metrics, torch.stack(counts).detach(), torch.stack(positives).detach()

    @torch.no_grad()
    def record_supervision(self, counts, positives):
        if self.event_weight > 0:
            self.label_counts.add_(counts)
            self.positive_counts.add_(positives)

    def ready_events(self, minimum=128, positives=1):
        return ((self.label_counts >= minimum) & (self.positive_counts >= positives)
                & ((self.label_counts - self.positive_counts) >= 1))


@torch.no_grad()
def actor_action_sequence(dynamics, actor, latent, first_action, horizon):
    """A candidate for continuing the original actor, using ordinary RSSM steps."""
    state = {k: v.clone() for k, v in latent.items()}
    sequence = []
    action = first_action
    for index in range(horizon):
        if index:
            action = actor(dynamics.get_feat(state)).mode()
        sequence.append(action)
        state = dynamics.img_step(state, F.pad(action, (0, 1)), sample=False)
    return torch.stack(sequence, 1)


class SearchAController:
    def __init__(self, horizon=6, override_prob=0.25, low_evidence=0.15,
                 new_weight=0.5, discovery_weight=1.0, revisit_weight=0.5,
                 cost_weight=0.05, start_step=10000, epsilon=0.15,
                 min_labels=128, min_positives=1, margin=0.0):
        self.horizon = int(horizon)
        self.override_prob, self.low_evidence = float(override_prob), float(low_evidence)
        self.weights = (float(new_weight), float(discovery_weight), -float(revisit_weight))
        self.cost_weight, self.margin = float(cost_weight), float(margin)
        self.start_step, self.epsilon = int(start_step), float(epsilon)
        self.min_labels, self.min_positives = min_labels, min_positives
        if self.horizon < 2 or not 0 <= self.override_prob <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("Invalid A control configuration")

    def initial(self, batch, device):
        return (torch.full((batch,), -1, device=device, dtype=torch.long),
                torch.zeros(batch, device=device, dtype=torch.long), torch.zeros(batch, device=device))

    @torch.no_grad()
    def choose(self, head, feat, obs, actor_actions, state, is_first, step, training,
               actor_sequence):
        batch, device = feat.shape[0], feat.device
        ids, pos, credit = self.initial(batch, device) if state is None else tuple(v.clone() for v in state)
        first = is_first.reshape(batch).bool()
        ids[first], pos[first], credit[first] = -1, 0, 0
        output = actor_actions.clone()
        info = {
            "log_search_override": torch.zeros(batch, device=device),
            "log_search_decision": torch.zeros(batch, device=device),
            "log_search_macro_id": torch.full((batch,), -1, device=device, dtype=torch.long),
            "log_search_scores": torch.zeros(batch, len(MACRO_NAMES), device=device),
            "log_search_predictions": torch.zeros(batch, len(MACRO_NAMES), len(EVENTS), device=device),
        }
        candidates = macro_actions(self.horizon, device)
        ready = head.ready_events(self.min_labels, self.min_positives)
        if step >= self.start_step:
            eligible = (ids < 0) & (heatmap_evidence(obs["heatmap"]) < self.low_evidence)
            if training:
                eligible &= torch.rand(batch, device=device) < self.override_prob
            else:
                credit[eligible] += self.override_prob
                eligible &= credit >= 1
                credit[eligible] -= 1
            # Training exploration obtains evidence; eval never uses untrained heads.
            if not ready.any() and (not training or self.epsilon == 0):
                eligible[:] = False
            if eligible.any():
                starts = eligible.nonzero().squeeze(-1)
                seq = candidates[None].expand(len(starts), -1, -1, -1)
                continuation = actor_sequence(starts)
                seq = torch.cat([seq, continuation[:, None]], 1)
                fs = feat[starts, None].expand(-1, len(MACRO_NAMES), -1)
                probabilities = head(fs, seq)["event_logits"].sigmoid()
                weights = torch.tensor(self.weights, device=device) * ready
                cost = (1 - seq[..., 0]).mean(-1)  # fraction of non-noop actions
                scores = (probabilities * weights).sum(-1) - self.cost_weight * cost
                best_macro = scores[:, :ACTOR_ID].argmax(-1)
                best_score = scores[:, :ACTOR_ID].max(-1).values
                choices = torch.where(best_score > scores[:, ACTOR_ID] + self.margin,
                                      best_macro, torch.full_like(best_macro, ACTOR_ID))
                if not ready.any():
                    choices[:] = ACTOR_ID
                if training and self.epsilon:
                    random = torch.rand(len(starts), device=device) < self.epsilon
                    choices = torch.where(random, torch.randint(ACTOR_ID, (len(starts),), device=device), choices)
                info["log_search_decision"][starts] = 1
                info["log_search_macro_id"][starts] = choices
                info["log_search_scores"][starts] = scores
                info["log_search_predictions"][starts] = probabilities
                selected = starts[choices < ACTOR_ID]
                ids[selected], pos[selected] = choices[choices < ACTOR_ID], 0
        executing = ids >= 0
        if executing.any():
            subset = executing.nonzero().squeeze(-1)
            output[subset] = candidates[ids[subset], pos[subset]]
            info["log_search_macro_id"][subset] = ids[subset]
            pos[subset] += 1
            finished = subset[pos[subset] >= self.horizon]
            ids[finished], pos[finished] = -1, 0
        info["log_search_override"] = executing.float()
        return output, (ids, pos, credit), info


def check_checkpoint(checkpoint, enabled):
    keys = checkpoint["agent_state_dict"]
    has_search = any(".search_a." in key for key in keys)
    if enabled and (not has_search or checkpoint.get("search_a_version") != MODEL_VERSION):
        raise ValueError("A v2 requires an A v2 checkpoint; old RGB-proxy A checkpoints cannot be silently loaded")
    if not enabled and has_search:
        raise ValueError("To ablate an A checkpoint use search_a_control=False; enabled=False is for baseline architecture")
