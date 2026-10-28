"""Factual feedback, calibrated imagination and task-Actor search objectives.

No simulator labels enter forward/decision methods. Unknown supervision is
masked per component; imagined transitions are never written as real Replay.
"""
import torch
from torch import nn
from torch.nn import functional as F
from search_a import SearchOutcomeHead

COMPONENTS = ("task", "discovery", "new", "revisit", "cost", "potential")
FULL_PREDICTION_DIM = 30  # old 11 + returns 6 + reliability 6 + support 6 + first 1


def window_targets(batch, horizon, config):
    """Discounted factual outcomes for action rows t+1..t+H, never beyond reset."""
    reward = batch["reward"].float()
    b, total = reward.shape
    n = total - horizon
    version = batch.get("search_label_version", torch.zeros_like(reward)) == 2
    def ev(name):
        y = batch.get("search_" + name, torch.zeros_like(reward)).float()
        known = batch.get("search_" + name + "_valid", torch.zeros_like(reward)).bool()
        known = known & version & torch.isfinite(y) & (y >= 0) & (y <= 1)
        return torch.where(known, y, torch.zeros_like(y)), known
    visible, vk = ev("visible")
    discovery, dk = ev("first_discovery")
    new, nk = ev("new")
    revisit, rk = ev("revisit")
    actions = batch["action"][..., :12].argmax(-1)
    previous = torch.cat((actions[:, :1], actions[:, :-1]), 1)
    reverse = (((actions == 2) & (previous == 3)) | ((actions == 3) & (previous == 2))
               | ((actions == 4) & (previous == 5)) | ((actions == 5) & (previous == 4)))
    finite_reward = torch.isfinite(reward)
    values = torch.stack((reward.nan_to_num() * config.search_c_task_weight,
                          discovery * config.search_c_discovery_weight,
                          new * config.search_c_new_weight,
                          -revisit * config.search_c_revisit_weight,
                          -torch.ones_like(reward) * config.search_c_step_cost
                          -reverse.float() * config.search_c_reverse_cost), -1)
    known = torch.stack((finite_reward, dk, nk, rk, torch.ones_like(vk)), -1)
    powers = reward.new_tensor(config.search_c_discount) ** torch.arange(horizon, device=reward.device)
    targets = sum(values[:, j+1:j+1+n] * powers[j] for j in range(horizon))
    mask = torch.stack([known[:, j+1:j+1+n] for j in range(horizon)]).all(0)
    terminal = batch.get("is_terminal", torch.zeros_like(reward)).bool()[:, horizon:]
    # Time-limit truncations bootstrap; real terminals have zero potential.
    end_phi = torch.where(terminal, torch.zeros_like(visible[:, horizon:]), visible[:, horizon:])
    potential = config.search_c_visibility_potential * (
        config.search_c_discount ** horizon * end_phi - visible[:, :n])
    potential_known = vk[:, :n] & (vk[:, horizon:] | terminal)
    targets = torch.cat((targets, potential[..., None]), -1)
    mask = torch.cat((mask, potential_known[..., None]), -1)
    first = torch.stack([discovery[:, j+1:j+1+n] for j in range(horizon)]).max(0).values
    first_known = (first > .5) | torch.stack([dk[:, j+1:j+1+n] for j in range(horizon)]).all(0)
    return targets.detach(), mask, first.detach(), first_known


class FullSearchOutcomeHead(SearchOutcomeHead):
    """A v5: all-prefix prediction, factual returns and outcome reliability."""
    def __init__(self, *args, feedback_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_config = feedback_config
        hidden = self.future.in_features
        self.return_head = nn.Linear(hidden, 6)
        self.first_head = nn.Linear(hidden, 1)
        self.reliability = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                        nn.Linear(hidden // 2, 6))
        nn.init.zeros_(self.return_head.weight)
        nn.init.zeros_(self.return_head.bias)
        nn.init.constant_(self.reliability[-1].bias, -2.)
    def forward(self, feat, actions, memory=None):
        result = super().forward(feat, actions, memory)
        hidden = result.pop("_hidden")
        result.update(returns=self.return_head(hidden),
                      first_logits=self.first_head(hidden).squeeze(-1),
                      reliability_logits=self.reliability(hidden.detach()))
        return result

    def additional_loss(self, pred, features, batch, windows, valid, feature_error, horizon):
        target, known, first, fk = window_targets(batch, horizon, self.feedback_config)
        mask = known & valid[..., None]
        count = mask.sum((0, 1))
        error = F.smooth_l1_loss(pred["returns"].float(), target, reduction="none")
        return_loss = ((error * mask).sum((0, 1)) / count.clamp_min(1)).mean()
        # Detached factual errors prevent the predictor gaming its own confidence.
        calibration_target = torch.exp(-(pred["returns"].float().detach() - target).abs()
                                       -feature_error.detach()[..., None]).clamp(0, 1)
        cal_error = F.binary_cross_entropy_with_logits(
            pred["reliability_logits"].float(), calibration_target, reduction="none")
        cal_loss = ((cal_error * mask).sum((0, 1)) / count.clamp_min(1)).mean()
        fm = fk & valid
        fe = F.binary_cross_entropy_with_logits(pred["first_logits"].float(), first, reduction="none")
        first_loss = (fe * fm).sum() / fm.sum().clamp_min(1)
        loss = return_loss + .25 * cal_loss + .25 * first_loss
        metrics = {"A/return_loss": return_loss.detach(), "A/reliability_loss": cal_loss.detach(),
                   "A/first_discovery_loss": first_loss.detach(),
                   "A/first_discovery_positive_count": (first * fm).sum().detach()}
        for i, name in enumerate(COMPONENTS):
            metrics["A/return_" + name + "_mae"] = (((pred["returns"][..., i] - target[..., i]).abs()
                * mask[..., i]).sum() / count[i].clamp_min(1)).detach()
            metrics["A/reliability_" + name] = ((pred["reliability_logits"][..., i].sigmoid()
                * mask[..., i]).sum() / count[i].clamp_min(1)).detach()
            metrics["A/reliability_" + name + "_brier"] = (((pred["reliability_logits"][..., i].sigmoid()
                - calibration_target[..., i]).square() * mask[..., i]).sum() / count[i].clamp_min(1)).detach()
        counts = torch.cat((count, fm.sum()[None]))
        positive = torch.cat((((target.abs() > 1.e-6) & mask).sum((0, 1)),
                              (first * fm).sum()[None]))
        return loss, metrics, counts.detach(), positive.detach()

    def return_support(self, length, minimum=None):
        minimum = minimum or self.feedback_config.search_a_min_labels
        count = self.prefix_counts[length-1, 5:11]
        supported = count >= minimum
        # Never fabricate a discovery bonus before observing any positive example.
        supported = supported.clone()
        supported[1] &= self.prefix_positives[length-1, 11] >= self.feedback_config.search_a_min_positives
        return supported

    def project_returns(self, output, length):
        """Respect known reward semantics even before regression converges."""
        cfg = self.feedback_config
        total = sum(cfg.search_c_discount ** j for j in range(length))
        value = output["returns"]
        lower = value.new_tensor([-cfg.search_c_return_clip, 0., 0.,
            -cfg.search_c_revisit_weight * total,
            -(cfg.search_c_step_cost + cfg.search_c_reverse_cost) * total,
            -cfg.search_c_visibility_potential])
        upper = value.new_tensor([cfg.search_c_return_clip, cfg.search_c_discovery_weight,
            cfg.search_c_new_weight * total, 0., -cfg.search_c_step_cost * total,
            cfg.search_c_visibility_potential])
        return torch.minimum(torch.maximum(value, lower), upper)

    def trusted_return(self, output, length, search_only=False):
        support = self.return_support(length).float()
        reliability = output["reliability_logits"].sigmoid()
        weights = reliability * support
        if search_only:
            weights = weights.clone()
            weights[..., 0] = 0  # original task reward is already in ImagBehavior
        prediction = self.project_returns(output, length)
        return (prediction * weights).sum(-1), weights


def actor_search_bonus(head, features, actions, memory, jump_record, config):
    """Align outgoing primitive action t with reward at t+1; no invented jump actions."""
    bonus = features.new_zeros(*features.shape[:-1], 1)
    if head is None or head.model_version < 5 or config.search_c_actor_bonus <= 0:
        return bonus, {}
    with torch.no_grad():
        out = head(features[:-1], actions[:-1, ..., None, :],
                   memory=memory[:-1] if memory is not None else None)
        reward, weights = head.trusted_return(out, 1, search_only=True)
        # First-discovery is a one-time event within each imagined rollout.
        survival = torch.cumprod(torch.cat((torch.ones_like(out["first_logits"][:1]),
                    1-out["first_logits"].sigmoid()[:-1]), 0), 0)
        raw = head.project_returns(out, 1)
        reward -= (raw[..., 1] * weights[..., 1]) * (1-survival)
        active = ~jump_record[:-1].bool()
        bonus[1:, ..., 0] = (reward * active).clamp(-.25, .25) * config.search_c_actor_bonus
    return bonus, {"C/task_actor_search_bonus": bonus.mean().detach(),
                   "C/task_actor_reliability": weights.mean().detach()}


@torch.no_grad()
def imagined_search_transitions(agent, snapshot, state, predictions, records):
    """One short Dyna branch per executable option from real replay starts.

    No recursive synthetic replay. Weight is bounded against the real batch;
    task feedback and event validity still come exclusively from executed data.
    """
    import numpy as np
    from search_a import prefix_macro_actions, actor_action_sequence, prefix_candidate_predictions
    cfg, head = agent._config, agent._wm.search_a
    if (agent._search_c.model_version < 2 or head is None or head.model_version < 5
            or cfg.search_c_imagined_weight <= 0 or not cfg.search_c_use_predictions):
        return None
    k = cfg.search_a_commit_steps
    if not bool(head.return_support(k).any()):
        return None
    latent, memory, sequence = snapshot
    count = min(len(records), cfg.search_c_imagined_starts)
    pairs = [(i, a) for i in range(count) for a in range(8) if records[i]["mask"][a] > 0]
    if not pairs:
        return None
    device = state.device
    rows = torch.tensor([i for i, a in pairs], device=device)
    choices = torch.tensor([a for i, a in pairs], device=device)
    current = {key: value[rows].detach() for key, value in latent.items()}
    mem = memory[rows].detach() if memory is not None else None
    seq = sequence[rows, choices]
    feature = agent._wm.dynamics.get_feat(current)
    output = head(feature, seq, memory=mem)
    supported = head.return_support(k).float()
    confidence = output["reliability_logits"].sigmoid()
    weights = (confidence * supported).sum(-1) / supported.sum().clamp_min(1)
    components = head.project_returns(output, k) * supported
    reward = components.sum(-1)
    discount = feature.new_full((len(rows),), cfg.search_c_discount ** k)
    for index in range(k):
        action = seq[:, index]
        current = agent._wm.dynamics.img_step(current, F.pad(action, (0, 1)), sample=False)
        feat = agent._wm.dynamics.get_feat(current)
        # Predicted termination is not a ground-truth terminal label.
        discount *= (1 - agent._wm.heads["end"](feat).mean.squeeze(-1)).clamp(0, 1)
        if mem is not None and cfg.search_b_control:
            mem = agent._wm.search_b.imagine_step(mem, feat, action)
    ns = torch.cat((feat, mem), -1) if mem is not None else feat
    next_seq = prefix_macro_actions(k, device)[None].expand(len(rows), -1, -1, -1).clone()
    first = agent._task_behavior.actor(ns).mode()
    next_seq[:, 5] = actor_action_sequence(agent._wm.dynamics, agent._task_behavior.actor,
        current, first, k, memory=mem,
        memory_transition=agent._wm.search_b.imagine_step if mem is not None and cfg.search_b_control else None)
    result = prefix_candidate_predictions(head, feat, next_seq, seq[:, -1].argmax(-1),
                                          mem, cfg.search_a_min_labels, cfg.search_a_min_positives)
    virtual = []
    for j, (i, a) in enumerate(pairs):
        context = records[i]["context"].copy()
        nc = context.copy()
        nc[1], nc[2] = float(result[2][j]), float(result[3][3])
        nc[8:] = seq[j, -1].cpu().numpy()
        nm = records[i]["mask"].copy()
        if a != 5:
            nc[3] = min(1., nc[3] + k / max(1., cfg.search_a_budget_fraction * cfg.search_a_budget_window))
            nc[4] = 1.
            nc[6] = 1.
            nm[:] = 0
            nm[5] = 1  # executor cooldown, never invent a fresh budget
        virtual.append(dict(action=a, reward=float(reward[j]), discount=float(discount[j]),
            duration=k, context=context, mask=records[i]["mask"], next_context=nc, next_mask=nm,
            **{"reward_" + key: float(components[j, c]) for c, key in enumerate(COMPONENTS)}))
    return dict(state=state[rows], predictions=predictions[rows], next_state=ns,
                next_predictions=result[4], records=virtual, weights=weights)
