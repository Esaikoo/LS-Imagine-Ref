"""Learning-based macro selection: C1 factual feedback; C2 calibrated Dyna updates.

A/B predictions are policy inputs, never reward labels. Semi-Markov discrete
SAC uses executed duration, real task/evidence rewards and terminal masking.
All privileged fields are confined to build_transitions(), not policy inputs.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

C_VERSION = 2
C_CHOICES = 8
CONTEXT_DIM = 20
PREDICTION_DIM = 11

C_FIELDS = {
    "search_c_version": (),
    "search_c_decision": (),
    "search_c_choice": (),
    "search_c_behavior_prob": (),
    "search_c_context": (CONTEXT_DIM,),
    "search_c_mask": (C_CHOICES,),
    "search_data_source": (),
    "search_branch_group": (),
    "search_branch_choice": (),
    "search_branch_verified": (),
    "search_branch_start": (),
}


def complete_c_fields(episode):
    result = episode.copy()
    length = len(result["reward"])
    for name, shape in C_FIELDS.items():
        if name not in result:
            result[name] = np.zeros((length, *shape), np.float32)
    return result


def check_c_checkpoint(checkpoint, enabled):
    has = any(k.startswith("_search_c.") for k in checkpoint["agent_state_dict"])
    if bool(enabled) != has or (has and checkpoint.get("search_c_version") not in (1, 2)):
        raise ValueError("C checkpoint/configuration mismatch; no partial silent loading")
    return has


def restore_c_config(config, checkpoint):
    has = any(k.startswith("_search_c.") for k in checkpoint["agent_state_dict"])
    for name, value in checkpoint.get("config", {}).items():
        if name.startswith("search_c_"):
            config[name] = value
    config["search_c_enabled"] = has
    config["search_c_model_version"] = checkpoint.get("search_c_version") or 1
    check_c_checkpoint(checkpoint, has)


def apply_c_control_mode(config, mode):
    """Runtime ablation keeps trained shapes and weights strictly loadable."""
    if mode == "checkpoint":
        return
    if not config.get("search_c_enabled"):
        raise ValueError("C runtime ablation requires a trained C checkpoint")
    if mode == "off":
        config["search_c_control"] = False
    elif mode == "no_predictions":
        config["search_c_use_predictions"] = False
    else:
        raise ValueError("C control mode must be checkpoint, off, or no_predictions")


def build_transitions(data, config, max_duration):
    """Action-row i describes a decision at observation i-1.

    A later decision at j closes the prior option at observation j-1. A true
    terminal/truncation closes it at its own observation. Unknown feedback
    terms contribute no evidence reward; factual task reward/cost still apply.
    """
    def arr(key, default=0):
        value = data.get(key)
        if value is None:
            return np.full(shape, default, np.float32)
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    shape = np.shape(data["reward"])
    first = arr("is_first").astype(bool)
    last = arr("is_last").astype(bool) | arr("is_terminal").astype(bool)
    starts = (arr("search_c_decision") > 0) & np.isin(arr("search_c_version"), [1, 2])
    terminal = arr("is_terminal").astype(bool)
    choice = arr("search_c_choice").astype(int)
    context = arr("search_c_context")
    masks = arr("search_c_mask")
    reward = arr("reward")
    event_version = arr("search_label_version") == 2
    def evidence(name):
        value = arr("search_"+name)
        known = (arr("search_"+name+"_valid") > 0) & event_version
        known &= np.isfinite(value) & (value >= 0) & (value <= 1)
        return np.where(known, value, 0), known
    visible, visible_known = evidence("visible")
    new, new_known = evidence("new")
    revisit, revisit_known = evidence("revisit")
    discovery, discovery_known = evidence("first_discovery")
    actions = arr("action").argmax(-1)
    result = []
    gamma = float(config.search_c_discount)
    for b in range(shape[0]):
        for row in np.flatnonzero(starts[b]):
            if row == 0 or first[b, row] or last[b, row-1]:
                continue
            a = int(choice[b, row])
            if not 0 <= a < C_CHOICES or masks.ndim != 3 or not masks[b, row, a]:
                continue
            end = None
            for e in range(row, min(shape[1], row + max_duration)):
                if first[b, e]:
                    break
                if last[b, e]:
                    end = e
                    break
                if e+1 < shape[1] and starts[b, e+1] and not first[b, e+1]:
                    end = e
                    break
            if end is None:
                continue  # right-censored option; never invent its outcome
            duration = end - row + 1
            discounts = gamma ** np.arange(duration)
            ys = slice(row, end+1)
            term = bool(terminal[b, end])
            actual = actions[b, ys]
            previous = actions[b, row-1:end]
            reverse = ((actual == 2) & (previous == 3)) | ((actual == 3) & (previous == 2))
            reverse |= ((actual == 4) & (previous == 5)) | ((actual == 5) & (previous == 4))
            components = {
                "task": config.search_c_task_weight * reward[b, ys],
                "discovery": config.search_c_discovery_weight * discovery[b, ys] * discovery_known[b, ys],
                "new": config.search_c_new_weight * new[b, ys] * new_known[b, ys],
                "revisit": -config.search_c_revisit_weight * revisit[b, ys] * revisit_known[b, ys],
                "cost": -config.search_c_step_cost * np.ones(duration)
                        - config.search_c_reverse_cost * reverse,
            }
            totals = {k: float(np.dot(discounts, v)) for k, v in components.items()}
            # Bounded potential shaping; cycling away/back cannot mint a
            # positive discovery reward. Terminal potential is defined as zero.
            potential = 0.
            if visible_known[b, row-1] and (term or visible_known[b, end]):
                end_phi = 0. if term else float(visible[b, end])
                potential = config.search_c_visibility_potential * (
                    gamma**duration * end_phi - float(visible[b, row-1]))
            totals["potential"] = potential
            next_mask = np.zeros(C_CHOICES, np.float32)
            next_mask[5] = 1
            next_context = context[b, row].copy()
            next_context[8:] = np.eye(12, dtype=np.float32)[actions[b, end]]
            next_context[4] = 1. if a != 5 else 0.
            if not last[b, end]:
                next_mask = masks[b, end+1]
                next_context = context[b, end+1]
                if not next_mask.any():
                    continue
            result.append(dict(batch=b, start=row-1, end=end, action=a,
                               duration=duration, reward=sum(totals.values()),
                               discount=0. if term else gamma**duration,
                               context=context[b, row], mask=masks[b, row],
                               next_context=next_context, next_mask=next_mask,
                               **{"reward_"+k: v for k, v in totals.items()}))
    return result


class SearchBehavior(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        import tools
        self.config = config
        self.seed = int(config.seed) + 5849
        self._rng = None
        self._pending_rng = None
        self.model_version = int(getattr(config, "search_c_model_version", 1))
        if self.model_version not in (1, 2):
            raise ValueError("Unsupported C architecture")
        self.prediction_dim = 30 if self.model_version >= 2 else PREDICTION_DIM
        size = feature_dim + C_CHOICES * self.prediction_dim + CONTEXT_DIM
        hidden = int(config.search_c_hidden)
        def network():
            return nn.Sequential(nn.Linear(size, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, C_CHOICES))
        self.actor, self.q1, self.q2 = network(), network(), network()
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[5] = 3.0  # conservative initial Actor option
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self._q_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self._value_opt = tools.Optimizer("search_c_value", self._q_parameters,
                                         config.search_c_lr, clip=100., wd=0., use_amp=False)
        self._actor_opt = tools.Optimizer("search_c_actor", self.actor.parameters(),
                                         config.search_c_lr, clip=100., wd=0., use_amp=False)

    def _input(self, state, prediction, context):
        if prediction.shape[-1] < self.prediction_dim:
            prediction = F.pad(prediction, (0, self.prediction_dim - prediction.shape[-1]))
        prediction = prediction[..., :self.prediction_dim]
        if not self.config.search_c_use_predictions:
            prediction = prediction.clone()
            prediction[..., 11:] = 0
            prediction[..., :8] = 0  # keep action cost/reversal/duration metadata
            context = context.clone()
            context[..., [1, 2, 5]] = 0  # A visibility, readiness and conflict evidence
        return torch.cat((F.layer_norm(state.float(), (state.shape[-1],)),
                          prediction.float().flatten(-2), context.float()), -1).detach()

    @staticmethod
    def distribution(logits, mask):
        if not mask.bool().any(-1).all():
            raise ValueError("C needs at least one executable option")
        probs = logits.masked_fill(~mask.bool(), -1.e9).softmax(-1)
        return probs, probs.clamp_min(1.e-8).log()

    def generator(self, device):
        if self._rng is None:
            self._rng = torch.Generator(device=device).manual_seed(self.seed)
            if self._pending_rng is not None:
                self._rng.set_state(self._pending_rng)
                self._pending_rng = None
        return self._rng

    def rng_state(self):
        return self._rng.get_state().cpu() if self._rng is not None else self._pending_rng

    def load_rng_state(self, value):
        self._pending_rng = value.cpu() if value is not None else None
        self._rng = None

    @torch.no_grad()
    def choose(self, state, predictions, context, mask, training):
        probs, _ = self.distribution(self.actor(self._input(state, predictions, context)), mask)
        selected = (torch.multinomial(probs, 1, generator=self.generator(probs.device)).squeeze(-1)
                    if training else probs.argmax(-1))
        probability = (probs.gather(-1, selected[:, None]).squeeze(-1)
                       if training else torch.ones_like(selected, dtype=probs.dtype))
        return selected, probability

    def train_transitions(self, state, predictions, next_state, next_predictions, records, imagined=None):
        if not records:
            return {}
        import tools
        device = state.device
        real_count = len(records)
        weights = torch.ones(real_count, device=device)
        if imagined is not None and len(imagined["records"]):
            virtual_count = len(imagined["records"])
            weight = imagined["weights"] * self.config.search_c_imagined_weight * real_count / virtual_count
            weights = torch.cat((weights, weight))
            state = torch.cat((state, imagined["state"]))
            next_state = torch.cat((next_state, imagined["next_state"]))
            predictions = torch.cat((predictions, imagined["predictions"]))
            next_predictions = torch.cat((next_predictions, imagined["next_predictions"]))
            records = records + imagined["records"]
        t = lambda key: torch.as_tensor(np.stack([r[key] for r in records]), device=device)
        x = self._input(state, predictions, t("context"))
        nx = self._input(next_state, next_predictions, t("next_context"))
        mask, nmask = t("mask"), t("next_mask")
        action = t("action").long()
        entropy_scale = float(self.config.search_c_entropy)
        with tools.RequiresGrad(self):
            with torch.no_grad():
                np_, nl = self.distribution(self.actor(nx), nmask)
                nq = torch.minimum(self.target_q1(nx), self.target_q2(nx))
                next_value = (np_ * (nq - entropy_scale * nl)).sum(-1)
                target = t("reward").float() + t("discount").float() * next_value
            q1 = self.q1(x).gather(-1, action[:, None]).squeeze(-1)
            q2 = self.q2(x).gather(-1, action[:, None]).squeeze(-1)
            qloss = ((F.smooth_l1_loss(q1, target, reduction="none") +
                      F.smooth_l1_loss(q2, target, reduction="none")) * weights).sum() / real_count
            metrics = self._value_opt(qloss, self._q_parameters)
            probs, logprobs = self.distribution(self.actor(x), mask)
            with torch.no_grad():
                q = torch.minimum(self.q1(x), self.q2(x))
            actor_loss = ((probs * (entropy_scale * logprobs - q)).sum(-1) * weights).sum() / real_count
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
        if metrics["search_c_value_update_applied"]:
            tau = float(self.config.search_c_target_tau)
            with torch.no_grad():
                for source, target_net in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                    for src, dst in zip(source.parameters(), target_net.parameters()):
                        dst.lerp_(src, tau)
                self.updates.add_(1)
        metrics.update({"C/transitions": real_count,
                        "C/imagined_transitions": len(records) - real_count,
                        "C/imagined_relative_weight": float(weights[real_count:].sum() / real_count),
                        "C/imagined_search_return": (float(t("reward")[real_count:].float().mean())
                            if len(records) > real_count else 0.),
                        "C/mean_duration": float(t("duration").float().mean()),
                        "C/real_search_return": float(t("reward")[:real_count].float().mean()),
                        "C/entropy": float(-(probs * logprobs).sum(-1).mean().detach()),
                        "C/updates": float(self.updates)})
        for key in records[0]:
            if key.startswith("reward_"):
                metrics["C/" + key] = float(t(key)[:real_count].float().mean())
        return metrics

