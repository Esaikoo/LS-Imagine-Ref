"""A v2-v4 core and v5 action-prefix interfaces (full head in search_feedback).

Only posterior features and real 12-way actions are inputs. Privileged evidence
is a masked training target, never a policy observation. Memory is optional.
"""
import torch
from torch import nn
from torch.nn import functional as F

ACTION_DIM = 12
MODEL_VERSION = 5
LABEL_VERSION = 2  # Label semantics remain compatible with A v2 replay.
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
                 event_weight=1.0, memory_dim=0, version=3, visible_weight=0.5):
        super().__init__()
        self.horizon = int(horizon)
        self.model_version = int(version)
        self.visible_weight = float(visible_weight)
        if self.model_version not in (2, 3, 4, 5) or self.visible_weight < 0:
            raise ValueError("Unsupported A model version or visibility weight")
        if self.horizon < 2 or min(future_weight, event_weight) < 0:
            raise ValueError("Invalid A horizon/loss weights")
        self.future_weight, self.event_weight = future_weight, event_weight
        self.input = nn.Sequential(nn.Linear(
            feature_dim + horizon * ACTION_DIM + (horizon if version >= 4 else 0), hidden), nn.SiLU())
        self.residual = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.future = nn.Linear(hidden, feature_dim)
        self.events = nn.Linear(hidden, len(EVENTS))
        self.memory_adapter = nn.Linear(memory_dim, hidden, bias=False) if memory_dim else None
        if self.memory_adapter is not None:
            nn.init.zeros_(self.memory_adapter.weight)
        self.register_buffer("label_counts", torch.zeros(len(EVENTS)))
        self.register_buffer("positive_counts", torch.zeros(len(EVENTS)))
        if version >= 4:
            self.register_buffer("prefix_counts", torch.zeros(horizon, 12 if version >= 5 else 5))
            self.register_buffer("prefix_positives", torch.zeros(horizon, 12 if version >= 5 else 5))
        if self.model_version >= 3:
            self.current_visible = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.future_visible = nn.Linear(hidden, 1)
            self.register_buffer("visible_counts", torch.zeros(2))
            self.register_buffer("visible_positives", torch.zeros(2))

    def forward(self, feat, actions, memory=None):
        length = actions.shape[-2]
        if (actions.shape[-1] != ACTION_DIM or feat.shape[:-1] != actions.shape[:-2]
                or length < 1 or length > self.horizon
                or (self.model_version < 4 and length != self.horizon)):
            raise ValueError("A expects aligned real action prefixes, at most H steps")
        parts = [normalize_feature(feat)]
        if self.model_version >= 4:
            # Padding is not a noop, and unexecuted suffix actions are not inputs.
            padded = F.pad(actions, (0, 0, 0, self.horizon - length))
            mask = actions.new_zeros(*actions.shape[:-2], self.horizon)
            mask[..., :length] = 1
            parts.extend((padded.flatten(-2), mask))
        else:
            parts.append(actions.flatten(-2))
        inp = torch.cat(parts, -1)
        hidden = self.input(inp)
        if memory is not None:
            if self.memory_adapter is None:
                raise ValueError("Memory supplied to an A-only head")
            if memory.shape[:-1] != feat.shape[:-1]:
                raise ValueError("B memory context shape mismatch")
            hidden = hidden + self.memory_adapter(memory)
        hidden = hidden + self.residual(hidden)
        result = {"future": self.future(hidden), "event_logits": self.events(hidden)}
        if self.model_version >= 5:
            result["_hidden"] = hidden
        if self.model_version >= 3:
            result["visible_logits"] = self.future_visible(hidden).squeeze(-1)
        return result

    def visibility(self, feat):
        if self.model_version < 3:
            return None
        return self.current_visible(normalize_feature(feat)).squeeze(-1).sigmoid()

    def ready_visibility(self, minimum=128, positives=1):
        if self.model_version < 3 or self.visible_weight <= 0:
            return torch.zeros(2, dtype=torch.bool, device=self.label_counts.device)
        return ((self.visible_counts >= minimum) & (self.visible_positives >= positives)
                & ((self.visible_counts - self.visible_positives) >= 1))

    def training_loss(self, posterior_features, batch, memory=None):
        if self.model_version < 4:
            return self._loss_at_horizon(posterior_features, batch, memory, self.horizon)
        horizons = (list(range(1, self.horizon + 1)) if self.model_version >= 5 else
                    sorted(set(k for k in (1, 2, 3, 4, self.horizon) if k <= self.horizon)))
        outputs = [self._loss_at_horizon(posterior_features, batch, memory, k)
                   for k in horizons]
        metrics = {}
        for key in outputs[0][1]:
            metrics[key] = torch.stack([out[1][key].float() for out in outputs]).mean()
        for k, (_, values, _, _) in zip(horizons, outputs):
            metrics.update({f"A/H{k}/" + name[2:]: value for name, value in values.items()})
        counts = posterior_features.new_zeros(self.horizon, self.prefix_counts.shape[-1])
        positives = torch.zeros_like(counts)
        for k, (_, _, n, pos) in zip(horizons, outputs):
            counts[k-1], positives[k-1] = n, pos
        return torch.stack([out[0] for out in outputs]).mean(), metrics, counts, positives

    def _loss_at_horizon(self, posterior_features, batch, memory, h):
        batch_size, total_time = posterior_features.shape[:2]
        if total_time <= h:
            raise ValueError("batch_length must exceed search_a_horizon")
        actions = batch["action"]
        windows = torch.stack([actions[:, t+1:t+h+1, :ACTION_DIM] for t in range(total_time-h)], 1)
        valid = valid_windows(batch, h, total_time)
        pred = (self(posterior_features[:, :-h], windows) if memory is None else
                self(posterior_features[:, :-h], windows, memory=memory[:, :-h]))
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
            known &= (version == LABEL_VERSION) & torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
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
            probability = logits.sigmoid()
            positive_mask = mask & (target_event > .5)
            negative_mask = mask & ~positive_mask
            positive_count = positive_mask.sum()
            metrics.update({
                f"A/{event}_positive_count": positive_count.detach(),
                f"A/{event}_positive_prediction": ((probability * positive_mask).sum()
                    / positive_count.clamp_min(1)).detach(),
                f"A/{event}_positive_recall": (((probability >= .5) & positive_mask).sum()
                    / positive_count.clamp_min(1)).detach(),
                f"A/{event}_negative_prediction": ((probability * negative_mask).sum()
                    / negative_mask.sum().clamp_min(1)).detach(),
            })
            metrics.update({f"A/{event}_loss": event_loss.detach(),
                            f"A/{event}_valid": count.detach(),
                            f"A/{event}_positive_rate": ((target_event * mask).sum() / count.clamp_min(1)).detach(),
                            f"A/{event}_brier": (((logits.sigmoid()-target_event)**2 * mask).sum()/count.clamp_min(1)).detach()})
        metrics["A/base_prediction_loss"] = loss.detach()
        if self.model_version >= 3:
            labels = batch.get("search_visible", torch.zeros_like(batch["is_first"])).reshape(batch_size, total_time)
            known = batch.get("search_visible_valid", torch.zeros_like(labels)).reshape_as(labels).bool()
            version = batch.get("search_label_version", torch.zeros_like(labels)).reshape_as(labels)
            known &= (version == LABEL_VERSION) & torch.isfinite(labels) & (labels >= 0) & (labels <= 1)
            labels = labels.nan_to_num().clamp(0, 1)
            current = self.current_visible(normalize_feature(posterior_features)).squeeze(-1)
            for name, logits, truth, mask in (
                    ("current_visible", current, labels, known),
                    ("future_visible", pred["visible_logits"], labels[:, h:], known[:, h:] & valid)):
                logits = logits.float()
                count = mask.sum()
                error = F.binary_cross_entropy_with_logits(logits, truth.float(), reduction="none")
                visible_loss = (error * mask).sum() / count.clamp_min(1)
                loss = loss + self.visible_weight * visible_loss / 2
                counts.append(count)
                positives.append((truth * mask).sum())
                metrics.update({
                    f"A/{name}_loss": visible_loss.detach(),
                    f"A/{name}_valid": count.detach(),
                    f"A/{name}_positive_rate": ((truth * mask).sum() / count.clamp_min(1)).detach(),
                    f"A/{name}_brier": (((logits.sigmoid() - truth)**2 * mask).sum() / count.clamp_min(1)).detach()})
        if self.model_version >= 5:
            extra, extra_metrics, extra_counts, extra_positive = self.additional_loss(
                pred, posterior_features, batch, windows, valid, feature_error, h)
            loss = loss + extra
            metrics.update(extra_metrics)
            counts.extend(extra_counts.unbind())
            positives.extend(extra_positive.unbind())
        # These buffers summarize labels used in completed optimization steps.
        metrics["A/prediction_loss"] = loss.detach()
        return loss, metrics, torch.stack(counts).detach(), torch.stack(positives).detach()

    @torch.no_grad()
    def record_supervision(self, counts, positives):
        if self.model_version >= 4:
            active = counts.new_tensor([self.event_weight > 0] * 3 +
                                      [self.visible_weight > 0] * 2 + ([True] * 7 if self.model_version >= 5 else []))
            self.prefix_counts.add_(counts * active)
            self.prefix_positives.add_(positives * active)
            counts, positives = counts.sum(0), positives.sum(0)
        if self.event_weight > 0:
            self.label_counts.add_(counts[:len(EVENTS)])
            self.positive_counts.add_(positives[:len(EVENTS)])
        if self.model_version >= 3 and self.visible_weight > 0 and len(counts) > len(EVENTS):
            self.visible_counts.add_(counts[3:5])
            self.visible_positives.add_(positives[3:5])

    def ready_events(self, minimum=128, positives=1):
        return ((self.label_counts >= minimum) & (self.positive_counts >= positives)
                & ((self.label_counts - self.positive_counts) >= 1))


    def ready_prefix(self, length, minimum=128, positives=1):
        if self.model_version < 4:
            raise ValueError("Prefix control requires a trained v4 prediction head")
        n, p = self.prefix_counts[length-1, :5], self.prefix_positives[length-1, :5]
        return (n >= minimum) & (p >= positives) & ((n-p) >= 1)


@torch.no_grad()
def actor_action_sequence(dynamics, actor, latent, first_action, horizon,
                          memory=None, memory_transition=None, prefix_actions=None):
    """A candidate for continuing the original actor, using ordinary RSSM steps."""
    state = {k: v.clone() for k, v in latent.items()}
    sequence = []
    action = first_action
    for index in range(horizon):
        if prefix_actions is not None and index < prefix_actions.shape[1]:
            action = prefix_actions[:, index]
        elif index:
            feat = dynamics.get_feat(state)
            inp = torch.cat((feat, memory), -1) if memory is not None else feat
            action = actor(inp).mode()
        sequence.append(action)
        state = dynamics.img_step(state, F.pad(action, (0, 1)), sample=False)
        if memory is not None and memory_transition is not None:
            memory = memory_transition(memory, dynamics.get_feat(state), action)
    return torch.stack(sequence, 1)


class LegacySearchAController:
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
               actor_sequence, memory=None):
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
                memory_for_candidates = (memory[starts, None].expand(-1, len(MACRO_NAMES), -1)
                                         if memory is not None else None)
                result = head(fs, seq) if memory_for_candidates is None else head(
                    fs, seq, memory=memory_for_candidates)
                probabilities = result["event_logits"].sigmoid()
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



REPAIRED_MACRO_NAMES = MACRO_NAMES + ("look_down_forward", "look_up_forward")
REPAIRED_CONTROL_DEFAULTS = {
    "search_a_control": True, "search_a_control_version": 3,
    "search_a_revisit_weight": 0.15, "search_a_margin": 0.02,
    "search_a_commit_steps": 2, "search_a_budget_window": 100,
    "search_a_budget_fraction": 0.2, "search_a_cooldown": 2,
    "search_a_revisit_gain_cap": 0.02, "search_a_min_information_gain": 0.005,
    "search_a_repeat_cost": 0.03, "search_a_visible_score_weight": 1.0,
    "search_a_visible_high": 0.6, "search_a_visible_low": 0.2,
    "search_a_conflict_steps": 8, "search_a_protect_interaction": True,
    "search_a_min_labels": 512, "search_a_min_positives": 8,
}


def repaired_macro_actions(horizon, device):
    # Keep historical IDs 0..5, including actor=5, stable in saved diagnostics.
    seq = F.one_hot(torch.full((8, horizon), 6, device=device, dtype=torch.long), ACTION_DIM).float()
    seq[:5] = macro_actions(horizon, device)
    # Repaired two-step prefixes must distinguish forward from forward+turn.
    seq[3, 1] = F.one_hot(torch.tensor(4, device=device), ACTION_DIM).float()
    seq[4, 1] = F.one_hot(torch.tensor(5, device=device), ACTION_DIM).float()
    seq[ACTOR_ID] = F.one_hot(torch.zeros(horizon, device=device, dtype=torch.long), ACTION_DIM).float()
    for index, action in ((6, 3), (7, 2)):
        # The first two actions scan vertically; the remainder approaches.
        seq[index, :min(2, horizon)] = F.one_hot(torch.tensor(action, device=device), ACTION_DIM).float()
    return seq


class SearchAController(LegacySearchAController):
    """V3 closed-loop control, with an exact legacy path for old checkpoints."""
    def __init__(self, *args, control_version=2, commit_steps=2, budget_window=100,
                 budget_fraction=0.2, cooldown=2, revisit_gain_cap=0.02,
                 min_information_gain=0.005, repeat_cost=0.03, visible_score_weight=1.,
                 visible_high=0.6, visible_low=0.2, conflict_steps=8,
                 protect_interaction=True, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_version = int(control_version)
        self.commit_steps = min(int(commit_steps), self.horizon)
        self.budget_window, self.budget_fraction = int(budget_window), float(budget_fraction)
        self.cooldown_steps = int(cooldown)
        self.revisit_gain_cap, self.min_information_gain = float(revisit_gain_cap), float(min_information_gain)
        self.repeat_cost, self.visible_score_weight = float(repeat_cost), float(visible_score_weight)
        self.visible_high, self.visible_low = float(visible_high), float(visible_low)
        self.conflict_steps, self.protect_interaction = int(conflict_steps), bool(protect_interaction)
        self.seed = int(seed) + 1729
        self._rng = None
        self._pending_rng_state = None
        if (self.control_version not in (2, 3) or self.commit_steps < 1 or self.budget_window < 1
                or not 0 <= self.budget_fraction <= 1 or self.cooldown_steps < 0
                or not 0 <= self.visible_low < self.visible_high <= 1 or self.conflict_steps < 1):
            raise ValueError("Invalid repaired A control settings")

    @property
    def macro_names(self):
        return REPAIRED_MACRO_NAMES if self.control_version >= 3 else MACRO_NAMES

    def rng_state(self):
        return self._rng.get_state().cpu() if self._rng is not None else self._pending_rng_state

    def load_rng_state(self, state):
        self._pending_rng_state = state.cpu() if state is not None else None
        self._rng = None

    def _generator(self, device):
        if self._rng is None:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self.seed)
            if self._pending_rng_state is not None:
                self._rng.set_state(self._pending_rng_state)
                self._pending_rng_state = None
        return self._rng

    def initial(self, batch, device):
        base = super().initial(batch, device)
        if self.control_version == 2:
            return base
        zero = torch.zeros(batch, device=device, dtype=torch.long)
        return (*base, torch.zeros(batch, self.budget_window, device=device),
                zero.clone(), zero.clone(), zero.clone(),
                torch.full_like(zero, -1), zero.clone(), zero.clone())

    @torch.no_grad()
    def choose(self, head, feat, obs, actor_actions, state, is_first, step, training,
               actor_sequence, memory=None, candidate_sequence=None):
        if self.control_version == 2:
            return super().choose(head, feat, obs, actor_actions, state, is_first,
                                  step, training, actor_sequence, memory)
        batch, device = feat.shape[0], feat.device
        values = self.initial(batch, device) if state is None else tuple(v.clone() for v in state)
        if len(values) != 10:
            raise ValueError("Repaired controller requires its own recurrent state")
        first = is_first.reshape(batch).bool()
        initial = self.initial(batch, device)
        for value, reset in zip(values, initial):
            value[first] = reset[first]
        ids, pos, credit, history, cursor, age, cooldown, last_id, conflict, interaction_run = values
        rows = torch.arange(batch, device=device)
        history[rows, cursor] = 0  # Remove the oldest action before this step.
        used = history.sum(-1)
        full_limit = int(self.budget_window * self.budget_fraction)
        growing_limit = torch.floor((age + 1).clamp(max=self.budget_window) * self.budget_fraction)
        limit = growing_limit.clamp(min=min(self.commit_steps, full_limit), max=full_limit)
        budget_ok = used < limit
        cooling = cooldown > 0
        cooldown = (cooldown - 1).clamp_min(0)

        evidence = heatmap_evidence(obs["heatmap"])
        visibility_ready = head.ready_visibility(self.min_labels, self.min_positives)
        visible = head.visibility(feat)
        if visible is None:
            visible = torch.full((batch,), -1., device=device)
        conflict_now = (evidence >= self.low_evidence) & (visible < self.visible_low) & visibility_ready[0]
        conflict = torch.where(conflict_now, conflict + 1, torch.zeros_like(conflict))
        high_evidence = evidence >= self.low_evidence
        if visibility_ready[0]:
            high_evidence = (visible >= self.visible_high) | (high_evidence & (conflict < self.conflict_steps))
        proposes_interaction = (actor_actions.argmax(-1) == 1) | (actor_actions.argmax(-1) == 11)
        interaction_run = torch.where(proposes_interaction, interaction_run + 1, torch.zeros_like(interaction_run))
        interaction_run.clamp_(max=self.commit_steps + 1)
        # Give a short chance to interact; ineffective repeated attack/use must not lock out search.
        interaction = proposes_interaction & (interaction_run <= self.commit_steps) & self.protect_interaction

        info = {
            "log_search_override": torch.zeros(batch, device=device),
            "log_search_decision": torch.zeros(batch, device=device),
            "log_search_macro_id": torch.full((batch,), -1, dtype=torch.long, device=device),
            "log_search_scores": torch.zeros(batch, 8, device=device),
            "log_search_predictions": torch.zeros(batch, 8, len(EVENTS), device=device),
            "log_search_visible_predictions": torch.zeros(batch, 8, device=device),
            "log_search_plan_actions": torch.full((batch, self.horizon), -1, dtype=torch.long, device=device),
            "log_search_actor_action": actor_actions.argmax(-1),
            "log_search_current_visible": visible,
            "log_search_visibility_ready": torch.full((batch,), float(visibility_ready[0]), device=device),
            "log_search_future_visibility_ready": torch.full((batch,), float(visibility_ready[1]), device=device),
            "log_search_interrupt": torch.zeros(batch, device=device),
            "log_search_completed_length": torch.zeros(batch, device=device),
        }
        active = ids >= 0
        interrupted = active & (high_evidence | interaction | ~budget_ok)
        info["log_search_interrupt"][interrupted & ~budget_ok] = 3
        info["log_search_interrupt"][interrupted & interaction] = 2
        info["log_search_interrupt"][interrupted & high_evidence] = 1
        info["log_search_completed_length"][interrupted] = pos[interrupted].float()
        ids[interrupted], pos[interrupted] = -1, 0
        cooldown[interrupted] = self.cooldown_steps
        output = actor_actions.clone()
        candidates = repaired_macro_actions(self.horizon, device)
        ready = head.ready_events(self.min_labels, self.min_positives)
        ready_any = bool(ready.any() or visibility_ready[1])
        eligible = ((ids < 0) & ~interrupted & ~cooling & ~high_evidence & ~interaction & budget_ok)
        eligible &= step >= self.start_step
        if training:
            eligible &= torch.rand(batch, device=device, generator=self._generator(device)) < self.override_prob
        else:
            credit[eligible] += self.override_prob
            eligible &= credit >= 1
            credit[eligible] -= 1
        if not ready_any and (not training or self.epsilon == 0):
            eligible[:] = False
        if eligible.any():
            starts = eligible.nonzero().squeeze(-1)
            seq = candidates[None].expand(len(starts), -1, -1, -1).clone()
            if candidate_sequence is not None:
                seq = candidate_sequence(starts, candidates)
                if seq.shape != (len(starts), 8, self.horizon, ACTION_DIM):
                    raise ValueError("Invalid conditional search candidate shape")
            seq[:, ACTOR_ID] = actor_sequence(starts)
            fs = feat[starts, None].expand(-1, 8, -1)
            mem = memory[starts, None].expand(-1, 8, -1) if memory is not None else None
            pred = head(fs, seq) if mem is None else head(fs, seq, memory=mem)
            probabilities = pred["event_logits"].sigmoid()
            target_visible = pred.get("visible_logits", torch.zeros_like(probabilities[..., 0])).sigmoid()
            target_visible = target_visible * visibility_ready[1]
            # Compare benefit to Actor. Reduced revisiting alone is not enough.
            information = probabilities[..., 0] * self.weights[0] * ready[0]
            information += probabilities[..., 1] * self.weights[1] * ready[1]
            information += self.visible_score_weight * target_visible
            revisit_gain = self.weights[2] * ready[2] * (
                probabilities[..., 2] - probabilities[:, ACTOR_ID:ACTOR_ID+1, 2])
            revisit_gain = revisit_gain.clamp(-self.revisit_gain_cap, self.revisit_gain_cap)
            camera = seq[..., 2:6].sum(-1).mean(-1)
            cost = 0.2 * (1 - seq[..., 0]).mean(-1) + 0.8 * camera
            repeat = (torch.arange(8, device=device)[None] == last_id[starts, None]).float()
            repeat[:, ACTOR_ID] = 0
            scores = information + revisit_gain - self.cost_weight * cost - self.repeat_cost * repeat
            gain = information - information[:, ACTOR_ID:ACTOR_ID+1]
            choices_scores = scores.clone()
            choices_scores[gain < self.min_information_gain] = -torch.inf
            choices_scores[:, ACTOR_ID] = -torch.inf
            best_score, best = choices_scores.max(-1)
            choices = torch.where(best_score > scores[:, ACTOR_ID] + self.margin,
                                  best, torch.full_like(best, ACTOR_ID))
            if not ready_any:
                choices[:] = ACTOR_ID
            if training and self.epsilon:
                random = torch.rand(len(starts), device=device, generator=self._generator(device)) < self.epsilon
                explore_ids = torch.tensor([0, 1, 2, 3, 4, 6, 7], device=device)
                random_ids = torch.randint(len(explore_ids), (len(starts),), device=device, generator=self._generator(device))
                choices = torch.where(random, explore_ids[random_ids], choices)
            info["log_search_decision"][starts] = 1
            info["log_search_macro_id"][starts] = choices
            info["log_search_scores"][starts] = scores
            info["log_search_predictions"][starts] = probabilities
            info["log_search_visible_predictions"][starts] = target_visible
            info["log_search_plan_actions"][starts] = seq[torch.arange(len(starts), device=device), choices].argmax(-1)
            selected = starts[choices != ACTOR_ID]
            ids[selected], pos[selected] = choices[choices != ACTOR_ID], 0

        executing = ids >= 0
        if executing.any():
            selected = executing.nonzero().squeeze(-1)
            output[selected] = candidates[ids[selected], pos[selected]]
            info["log_search_macro_id"][selected] = ids[selected]
            last_id[selected] = ids[selected]
            pos[selected] += 1
            finished = selected[pos[selected] >= self.commit_steps]
            info["log_search_completed_length"][finished] = pos[finished].float()
            ids[finished], pos[finished] = -1, 0
            cooldown[finished] = self.cooldown_steps
        history[rows, cursor] = executing.float()
        info["log_search_override"] = executing.float()
        info["log_search_budget_used"] = history.sum(-1)
        state = (ids, pos, credit, history, (cursor + 1) % self.budget_window,
                 (age + 1).clamp(max=self.budget_window), cooldown, last_id, conflict, interaction_run)
        return output, state, info


def check_checkpoint(checkpoint, enabled, expected_version=None):
    keys = checkpoint["agent_state_dict"]
    has_search = any(".search_a." in key for key in keys)
    version = checkpoint.get("search_a_version")
    if enabled and (not has_search or version not in (2, 3, 4, 5)):
        raise ValueError("A requires an explicitly versioned v2/v3/v4/v5 checkpoint")
    if enabled and expected_version is not None and version != expected_version:
        raise ValueError("A checkpoint/model version mismatch; start a fresh run for the new architecture")
    if not enabled and has_search:
        raise ValueError("To ablate an A checkpoint use search_a_control=False; enabled=False is for baseline architecture")
    return version if has_search else None


def restore_a_config(config, checkpoint):
    """Restore old architecture/control exactly before explicit runtime overrides."""
    has_a = any(".search_a." in key for key in checkpoint["agent_state_dict"])
    check_checkpoint(checkpoint, has_a)
    for key, value in checkpoint.get("config", {}).items():
        if key.startswith("search_a_"):
            config[key] = value
    config["search_a_enabled"] = has_a
    if has_a:
        config["search_a_model_version"] = checkpoint["search_a_version"]
        if checkpoint["search_a_version"] == 2:
            config["search_a_separate_optimizer"] = False
            config["search_a_control_version"] = checkpoint.get("config", {}).get("search_a_control_version", 2)


def apply_repaired_control(config):
    if not config.get("search_a_enabled"):
        raise ValueError("Repaired A control requires trained A weights")
    if config.get("search_a_model_version", 2) not in (2, 3):
        raise ValueError("Legacy repaired control is only for A v2/v3; keep v4 prefix control")
    config.update(REPAIRED_CONTROL_DEFAULTS)


PREFIX_MACRO_NAMES = ("scan_left", "scan_right", "forward", "left_then_forward",
                      "right_then_forward", "actor", "down_then_forward", "up_then_forward")


def prefix_macro_actions(length, device):
    if length < 2:
        raise ValueError("Search execution needs at least two real actions")
    ids = torch.full((8, length), 6, device=device, dtype=torch.long)
    ids[0], ids[1] = 4, 5
    for index, first in ((3, 4), (4, 5), (6, 3), (7, 2)):
        ids[index, 0] = first
    ids[ACTOR_ID] = 0  # replaced with an imagined Actor continuation
    return F.one_hot(ids, ACTION_DIM).float()


def prefix_candidate_predictions(head, feat, sequence, previous_action,
                                 memory=None, minimum=512, positives=8):
    """Policy-only predictions. No simulator labels or coordinates are inputs."""
    batch, candidates, length, _ = sequence.shape
    probabilities = feat.new_zeros(batch, candidates, 3)
    future_visible = feat.new_zeros(batch, candidates)
    ready = torch.zeros(5, device=feat.device, dtype=torch.bool)
    current_visible = feat.new_zeros(batch)
    if head is not None:
        if head.model_version < 4:
            raise ValueError("Prefix controller requires A v4; old heads need explicit migration")
        expanded = feat[:, None].expand(-1, candidates, -1)
        mem = memory[:, None].expand(-1, candidates, -1) if memory is not None else None
        output = head(expanded, sequence, memory=mem)
        probabilities = output["event_logits"].sigmoid()
        future_visible = output["visible_logits"].sigmoid()
        current_visible = head.visibility(feat)
        ready = head.ready_prefix(length, minimum, positives)
    first = sequence[:, :, 0].argmax(-1)
    opposite = torch.tensor([0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 10, 11],
                            device=feat.device)
    reverses = (first == opposite[previous_action.long()][:, None]) & (
        (previous_action[:, None] >= 2) & (previous_action[:, None] <= 5))
    # Broadcasting previous_action needs a candidate axis.
    cost = 0.2 * (1 - sequence[..., 0]).mean(-1) + 0.8 * sequence[..., 2:6].sum(-1).mean(-1)
    flags = ready[[0, 1, 2, 4]].float()[None, None].expand(batch, candidates, -1)
    descriptor = torch.cat((probabilities, future_visible[..., None], flags,
                            cost[..., None], reverses.float()[..., None],
                            torch.full_like(cost[..., None], length / 6.)), -1)
    if head is not None and head.model_version >= 5:
        support = head.return_support(length, minimum)
        descriptor = torch.cat((descriptor, head.project_returns(output, length),
                                output["reliability_logits"].sigmoid(),
                                support.expand(batch, candidates, -1).float(),
                                output["first_logits"].sigmoid()[..., None]), -1)
    return probabilities, future_visible, current_visible, ready, descriptor


class PrefixSearchController(SearchAController):
    """V4 prefix execution; optional C chooses the options through the same executor."""
    def __init__(self, *args, control_version=4, **kwargs):
        if control_version != 4:
            raise ValueError("Prefix controller is version 4")
        super().__init__(*args, control_version=3, **kwargs)
        self.control_version = 4
        if (not 2 <= self.commit_steps <= self.horizon
                or self.commit_steps not in (2, 3, 4, self.horizon)):
            raise ValueError("Invalid prefix execution duration")

    @property
    def macro_names(self):
        return PREFIX_MACRO_NAMES

    def initial(self, batch, device):
        base = super().initial(batch, device)
        zero = torch.zeros(batch, device=device, dtype=torch.long)
        return (*base, zero.clone(), zero.clone(), zero.clone())

    @torch.no_grad()
    def choose(self, head, feat, obs, actor_actions, state, is_first, step, training,
               actor_sequence, memory=None, candidate_sequence=None, selector=None,
               policy_memory=None):
        from search_c import CONTEXT_DIM
        batch, device = feat.shape[0], feat.device
        fresh = self.initial(batch, device)
        values = fresh if state is None else tuple(v.clone() for v in state)
        if len(values) != 13:
            raise ValueError("Prefix controller state version mismatch")
        first = is_first.reshape(batch).bool()
        for value, reset in zip(values, fresh):
            value[first] = reset[first]
        (ids, pos, credit, history, cursor, age, cooldown, last_id, conflict,
         interaction_run, previous_action, decision_id, previous_high) = values
        rows = torch.arange(batch, device=device)
        history[rows, cursor] = 0
        used = history.sum(-1)
        full_limit = int(self.budget_window * self.budget_fraction)
        limit = torch.floor((age+1).clamp(max=self.budget_window) * self.budget_fraction)
        limit = limit.clamp(min=min(self.commit_steps, full_limit), max=full_limit)
        budget_ok = used < limit
        can_start = (limit - used) >= self.commit_steps
        cooling = cooldown > 0
        cooldown = (cooldown-1).clamp_min(0)
        evidence = heatmap_evidence(obs["heatmap"])
        visible = head.visibility(feat) if head is not None else feat.new_zeros(batch)
        ready = (head.ready_prefix(self.commit_steps, self.min_labels, self.min_positives)
                 if head is not None else torch.zeros(5, device=device, dtype=torch.bool))
        high = evidence >= self.low_evidence
        conflicting = high & (visible < self.visible_low) & ready[3]
        conflict = torch.where(conflicting, conflict+1, torch.zeros_like(conflict))
        if ready[3]:
            high = (visible >= self.visible_high) | (high & (conflict < self.conflict_steps))
        proposed = actor_actions.argmax(-1)
        interact = (proposed == 1) | (proposed == 11)
        interaction_run = torch.where(interact, interaction_run+1, torch.zeros_like(interaction_run))
        interaction_run.clamp_(max=self.commit_steps+1)
        protected = interact & (interaction_run <= 2) & self.protect_interaction
        is_search = (ids >= 0) & (ids != ACTOR_ID)
        # Learned C can choose despite an already-known weak/strong heatmap.
        # Newly acquired evidence can still interrupt an ongoing search option.
        evidence_interrupt = high if selector is None else high & ~previous_high.bool()
        interrupted = is_search & (evidence_interrupt | protected | ~budget_ok)
        reason = torch.zeros(batch, device=device)
        reason[interrupted & ~budget_ok] = 3
        reason[interrupted & protected] = 2
        reason[interrupted & evidence_interrupt] = 1
        completed = torch.where(interrupted, pos.float(), torch.zeros_like(pos).float())
        ids[interrupted], pos[interrupted] = -1, 0
        cooldown[interrupted] = self.cooldown_steps
        info = {
            "log_search_override": feat.new_zeros(batch),
            "log_search_decision": feat.new_zeros(batch),
            "log_search_macro_id": torch.full((batch,), -1, device=device, dtype=torch.long),
            "log_search_scores": feat.new_zeros(batch, 8),
            "log_search_predictions": feat.new_zeros(batch, 8, 3),
            "log_search_visible_predictions": feat.new_zeros(batch, 8),
            "log_search_plan_actions": torch.full((batch, self.horizon), -1, device=device, dtype=torch.long),
            "log_search_actor_action": proposed,
            "log_search_current_visible": visible,
            "log_search_visibility_ready": torch.full_like(visible, float(ready[3])),
            "log_search_future_visibility_ready": torch.full_like(visible, float(ready[4])),
            "log_search_interrupt": reason,
            "log_search_completed_length": completed,
            "log_search_prediction_horizon": torch.full_like(visible, self.commit_steps),
            "log_search_candidate_version": torch.full_like(visible, 4),
            "log_search_valid": torch.ones_like(visible),
        }
        if head is not None and head.model_version >= 5:
            info["log_search_return_predictions"] = feat.new_zeros(batch, 8, 6)
            info["log_search_reliability"] = feat.new_zeros(batch, 8, 6)
            info["log_search_return_support"] = feat.new_zeros(batch, 8, 6)
            info["log_search_first_predictions"] = feat.new_zeros(batch, 8)
        if selector is not None:
            info.update(search_c_version=torch.full_like(visible, selector.model_version),
                        search_c_decision=torch.zeros_like(visible),
                        search_c_choice=torch.zeros_like(proposed),
                        search_c_behavior_prob=torch.zeros_like(visible),
                        search_c_context=feat.new_zeros(batch, CONTEXT_DIM),
                        search_c_mask=feat.new_zeros(batch, 8))
        eligible = ids < 0
        allowed = ~interrupted & ~cooling & ~protected & can_start & (step >= self.start_step)
        if selector is None:
            eligible &= allowed & ~high
            if training:
                eligible &= torch.rand(batch, device=device, generator=self._generator(device)) < self.override_prob
            else:
                credit[eligible] += self.override_prob
                eligible &= credit >= 1
                credit[eligible] -= 1
            if not bool(ready.any()) and (not training or self.epsilon == 0):
                eligible[:] = False
        output = actor_actions.clone()
        templates = prefix_macro_actions(self.commit_steps, device)
        if eligible.any():
            starts = eligible.nonzero().squeeze(-1)
            seq = templates[None].expand(len(starts), -1, -1, -1).clone()
            seq[:, ACTOR_ID] = actor_sequence(starts)[:, :self.commit_steps]
            probs, future, _, _, descriptor = prefix_candidate_predictions(
                head, feat[starts], seq, previous_action[starts],
                memory[starts] if memory is not None else None, self.min_labels, self.min_positives)
            information = probs[..., 0] * self.weights[0] * ready[0]
            information += probs[..., 1] * self.weights[1] * ready[1]
            information += self.visible_score_weight * future * ready[4]
            revisit_gain = (self.weights[2] * ready[2] *
                            (probs[..., 2] - probs[:, ACTOR_ID:ACTOR_ID+1, 2]))
            repeat = (torch.arange(8, device=device)[None] == last_id[starts, None]).float()
            repeat[:, ACTOR_ID] = 0
            scores = information + revisit_gain.clamp(-self.revisit_gain_cap, self.revisit_gain_cap)
            scores -= self.cost_weight * descriptor[..., 8]
            scores -= self.repeat_cost * (repeat + descriptor[..., 9])
            if selector is None:
                compare = scores.clone()
                compare[information-information[:, ACTOR_ID:ACTOR_ID+1] < self.min_information_gain] = -torch.inf
                compare[:, ACTOR_ID] = -torch.inf
                score, choices = compare.max(-1)
                choices = torch.where(score > scores[:, ACTOR_ID]+self.margin,
                                      choices, torch.full_like(choices, ACTOR_ID))
                if not bool(ready.any()):
                    choices[:] = ACTOR_ID
                if training and self.epsilon:
                    random = torch.rand(len(starts), device=device, generator=self._generator(device)) < self.epsilon
                    options = torch.tensor([0, 1, 2, 3, 4, 6, 7], device=device)
                    draws = torch.randint(7, (len(starts),), device=device, generator=self._generator(device))
                    choices = torch.where(random, options[draws], choices)
            else:
                context = feat.new_zeros(len(starts), CONTEXT_DIM)
                context[:, :8] = torch.stack((
                    evidence[starts], visible[starts], torch.full_like(visible[starts], float(ready[3])),
                    used[starts]/max(1, full_limit), cooldown[starts].float()/max(1, self.cooldown_steps),
                    (conflict[starts].float()/self.conflict_steps).clamp(max=1),
                    (last_id[starts] >= 0).float(),
                    torch.full_like(visible[starts], float(policy_memory is not None))), -1)
                context[:, 8:] = F.one_hot(previous_action[starts], 12).float()
                mask = allowed[starts, None].expand(-1, 8).clone()
                mask[:, ACTOR_ID] = True
                state_features = feat[starts]
                if policy_memory is not None:
                    state_features = torch.cat((state_features, policy_memory[starts]), -1)
                choices, probability = selector.choose(state_features, descriptor, context, mask, training)
                info["search_c_decision"][starts] = 1
                info["search_c_choice"][starts] = choices
                info["search_c_behavior_prob"][starts] = probability
                info["search_c_context"][starts] = context
                info["search_c_mask"][starts] = mask.float()
            decision_id[starts] += 1
            info["log_search_decision"][starts] = 1
            info["log_search_macro_id"][starts] = choices
            info["log_search_scores"][starts] = scores
            info["log_search_predictions"][starts] = probs
            info["log_search_visible_predictions"][starts] = future
            if descriptor.shape[-1] == 30:
                info["log_search_return_predictions"][starts] = descriptor[..., 11:17]
                info["log_search_reliability"][starts] = descriptor[..., 17:23]
                info["log_search_return_support"][starts] = descriptor[..., 23:29]
                info["log_search_first_predictions"][starts] = descriptor[..., 29]
            info["log_search_plan_actions"][starts, :self.commit_steps] = seq[
                torch.arange(len(starts), device=device), choices].argmax(-1)
            selected = starts if selector is not None else starts[choices != ACTOR_ID]
            ids[selected] = choices if selector is not None else choices[choices != ACTOR_ID]
            pos[selected] = 0
        active = ids >= 0
        searching = active & (ids != ACTOR_ID)
        if active.any():
            chosen = searching.nonzero().squeeze(-1)
            output[chosen] = templates[ids[chosen], pos[chosen]]
            info["log_search_macro_id"][active] = ids[active]
            last_id[chosen] = ids[chosen]
            pos[active] += 1
            finished = active & (pos >= self.commit_steps)
            info["log_search_completed_length"][finished & searching] = pos[finished & searching].float()
            cooldown[finished & searching] = self.cooldown_steps
            ids[finished], pos[finished] = -1, 0
        history[rows, cursor] = searching.float()
        info["log_search_override"] = searching.float()
        info["log_search_budget_used"] = history.sum(-1)
        info["log_search_decision_id"] = decision_id.clone()
        state = (ids, pos, credit, history, (cursor+1) % self.budget_window,
                 (age+1).clamp(max=self.budget_window), cooldown, last_id, conflict,
                 interaction_run, output.argmax(-1), decision_id, high.long())
        return output, state, info


def migrate_v3_head(source, destination):
    """Explicit warm start only; no optimizer/counter or scientific-run continuation."""
    if source.model_version != 3 or destination.model_version != 4:
        raise ValueError("Only explicit v3-to-v4 A head migration is supported")
    old, new = source.state_dict(), destination.state_dict()
    copied, initialized = [], []
    for name, tensor in new.items():
        if "counts" in name or "positives" in name:
            tensor.zero_()
            initialized.append(name)
        elif name == "input.0.weight":
            if tensor.shape[0] != old[name].shape[0] or tensor.shape[1] != old[name].shape[1]+destination.horizon:
                raise ValueError("Incompatible feature dimensions for A migration")
            tensor.zero_()
            tensor[:, :old[name].shape[1]].copy_(old[name])
            copied.append(name + " (old columns)")
            initialized.append(name + " (prefix mask columns)")
        elif name in old and tensor.shape == old[name].shape:
            tensor.copy_(old[name])
            copied.append(name)
        else:
            initialized.append(name)
    destination.load_state_dict(new, strict=True)
    return {"copied": copied, "initialized": initialized, "warm_start_only": True}
