"""Bounded, inference-independent exports of real search decisions.

The collector never changes observations, rewards, RNGs, model state or actions.
Only a few RGB/heatmap/patch snapshots are retained. Selected episode suites
render with Agg, and optional map-only exports run in a bounded CPU worker.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock

import numpy as np


PHASES = ("SEARCH", "APPROACH", "INTERACT", "RECOVERY")
SOURCES = {-1: "unspecified", 0: "actor", 1: "learned", 2: "rule interaction",
           3: "rule recovery", 4: "settle", 5: "training probe",
           6: "bounded recovery probe", 7: "alignment probe",
           8: "interaction probe", 9: "bounded search probe"}


def _get(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _array(value, default=None):
    if value is None:
        return np.asarray(default) if default is not None else None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(mapping, key, default=0.):
    value = _array(mapping.get(key), default).reshape(-1)
    return float(value[0]) if len(value) and np.isfinite(value[0]) else float(default)


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rgb(value):
    arr = _array(value)
    if arr is None or arr.ndim != 3:
        return None
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255. if arr.max(initial=0) <= 1.01 else 1.), 0, 255).astype(np.uint8)
    return arr.copy()


def _goal_heatmap(value):
    """Normalize actual policy heatmaps to the common display scale [0, 1]."""
    heat = _array(value)
    if heat is None or heat.squeeze().ndim != 2 or not heat.size:
        return None
    scale = 255. if heat.dtype == np.uint8 or heat.max() > 1.01 else 1.
    return heat.squeeze().astype(np.float32) / scale


def _goal_heat_delta(original, effective):
    original, effective = _goal_heatmap(original), _goal_heatmap(effective)
    if original is None or effective is None or original.shape != effective.shape:
        return None
    delta = np.abs(effective - original)
    return float(delta.mean()), float(delta.max())


def _map_rgb(grid):
    """Search-behavior colors; never imply obstacle geometry or true targets."""
    grid = np.asarray(grid, dtype=float)
    out = np.full((*grid.shape[:2], 3), [236, 237, 239], dtype=np.uint8)
    out[grid[..., 1] > 0] = [192, 211, 221]
    visited = grid[..., 0] > 0
    out[visited] = [72, 105, 135]
    out[grid[..., 3] > 0] = [32, 62, 90]
    out[grid[..., 2] > .2] = [60, 173, 105]
    return out


@dataclass
class _Episode:
    rows: list = field(default_factory=list)
    a: list = field(default_factory=list)
    c: list = field(default_factory=list)
    last: object = None
    groups: dict = field(default_factory=dict)
    pending: object = None
    terminal: object = None
    goal: object = None
    reward: float = 0.
    success: bool = False
    omitted_rows: int = 0
    pose_steps: int = 0
    localized_steps: int = 0
    local_tracking_steps: int = 0


class SearchVisualizer:
    """Collect one environment at a time; methods also accept numpy policy logs.

    observe runs BEFORE env.step and its RGB therefore matches the outgoing
    decision. feedback runs AFTER the corresponding real step. Episode metadata
    may contain privileged audit labels, consumed only after the episode ends.
    """
    def __init__(self, root, config, *, scheduled_eval_maps=True):
        self.config = config
        self.checkpoint_id = None
        self.bank_info = {}
        self.enabled = bool(_get(config, "vnext_visual_enabled", True))
        self.interval = int(_get(config, "vnext_visual_every", 10000))
        if self.interval <= 0:
            raise ValueError("vnext_visual_every must be positive")
        self.max_rows = int(_get(config, "vnext_visual_max_steps", 4096))
        self.max_active = int(_get(config, "vnext_visual_max_active", 8))
        if self.max_rows < 1 or self.max_active < 1:
            raise ValueError("Visualization buffers must have positive bounds")
        task = str(_get(config, "task", "task")).replace("/", "_").replace("\\", "_")
        self.root = Path(root) / "visualizations" / task
        self.strict = bool(_get(config, "vnext_visual_strict", True))
        self._all_eval_maps = (self.enabled and scheduled_eval_maps and
                               bool(_get(config, "vnext_visual_map_all_eval_episodes", False)))
        self.map_min_span = float(_get(config, "vnext_visual_map_min_span", 24.))
        if not math.isfinite(self.map_min_span) or self.map_min_span <= 0.:
            raise ValueError("vnext_visual_map_min_span must be positive and finite")
        self._map_executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="B-map")
                              if self._all_eval_maps else None)
        self._map_jobs = deque()
        self._render_lock = Lock()
        self._active = {}
        self._partial_prefixes = set()
        self._latest_train = None
        self._last_exported_train_id = None
        self._next_train = self.interval
        self._next_eval = 0
        self._last_eval_step = None
        self._eval_selected = False
        self.closed = False
        self.stats = dict(exports=0, errors=0, skipped_no_episode=0, skipped_duplicate=0,
                          incomplete_resets=0, skipped_partial_prefix=0, map_exports=0, map_errors=0)
        self.last_error = None

    def set_goal_bank(self, bank):
        manifest = bank["manifest"]
        import hashlib
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        self.bank_info = dict(task=manifest.get("task_id"), version=manifest.get("version"),
            positives=len(manifest["entries"]), negatives=len(manifest.get("negative_entries", [])),
            fingerprint=hashlib.sha256(encoded).hexdigest(),
            fusion_enabled=_get(self.config, "goal_fusion_enabled", True),
            max_gate=_get(self.config, "goal_fusion_max_gate", .8),
            epsilon=_get(self.config, "goal_fusion_epsilon", .05),
            overlap=_get(self.config, "goal_fusion_overlap_weight", .5))
        positives = list(zip(manifest["entries"], _array(bank["pixels"])))
        negatives = list(zip(manifest.get("negative_entries", []), _array(bank["negative_pixels"])))
        examples = positives[:4] if not negatives else positives[:2] + negatives[:2]
        self.goal_references = []
        for entry, frame in examples:
            name = Path(entry.get("path", "reference")).stem.split("_", 1)[-1].rsplit("_", 1)[0].replace("_", " ")
            roi = entry.get("source_roi_xyxy")
            size = "" if roi is None else f" · {roi[2]-roi[0]}×{roi[3]-roi[1]} native"
            role = "Negative" if entry.get("role") == "negative" else "Positive"
            self.goal_references.append((f"{role} · {name}{size}", _rgb(frame)))

    def set_step(self, step):
        """Start/resume at a real training step; never fabricate past exports."""
        step = max(0, int(step))
        self._next_train = (step // self.interval + 1) * self.interval
        self._next_eval = ((step + self.interval - 1) // self.interval) * self.interval
        self._last_eval_step = None
        self._eval_selected = False

    def _scheduled_eval(self, step):
        # Keep the same decision across the whole round. If eval and visualization
        # intervals differ, select the first real evaluation at or after a threshold.
        if step != self._last_eval_step:
            self._last_eval_step = step
            self._eval_selected = step >= self._next_eval
            if self._eval_selected:
                self._next_eval = (step // self.interval + 1) * self.interval
        return self._eval_selected

    def _open(self):
        if self.closed:
            raise RuntimeError("SearchVisualizer is already closed")

    def observe(self, mode, env_id, obs, policy_output, training_step):
        self._open()
        if not self.enabled:
            return
        if mode not in ("train", "eval"):
            raise ValueError("Visualization mode must be train or eval")
        if mode == "train":
            self.maybe_export_train(training_step)
        # Random prefill / legacy runs have no predicted spatial state.
        if _scalar(policy_output, "log_spatial_version") < 2:
            return
        key = (mode, str(env_id))
        first = bool(_scalar(obs, "is_first"))
        # Learning may start part-way through a random-prefill episode. Its
        # partial predicted map/history is not a complete visualization episode.
        # Track the skipped prefix once, and wait for a real environment reset.
        if key not in self._active and not first:
            if key not in self._partial_prefixes:
                self._partial_prefixes.add(key)
                self.stats["skipped_partial_prefix"] += 1
            return
        if first:
            self._partial_prefixes.discard(key)
        if first and key in self._active:
            self.stats["incomplete_resets"] += 1
            del self._active[key]
        if key not in self._active:
            if len(self._active) >= self.max_active:
                raise RuntimeError("Visualization active-environment bound exceeded")
            self._active[key] = _Episode()
        ep = self._active[key]
        index = len(ep.rows) + ep.omitted_rows
        context = _array(policy_output.get("log_vnext_context"), np.zeros(24)).reshape(-1)
        visible = float(context[11]) if len(context) > 11 else 0.
        option = int(_scalar(policy_output, "log_vnext_choice"))
        q = _array(policy_output.get("log_vnext_q"), []).reshape(-1)
        pose = _array(policy_output.get("log_spatial_pose"), np.zeros(4)).reshape(-1)
        pose = np.pad(pose[:4], (0, max(0, 4-len(pose))))
        spatial_summary = _array(policy_output.get("log_spatial_summary"), []).reshape(-1)
        pose_ready = bool(len(spatial_summary) and np.isfinite(spatial_summary[0]) and spatial_summary[0] > .5)
        if "log_spatial_trajectory_valid" in policy_output:
            pose_ready = bool(_scalar(policy_output,"log_spatial_trajectory_valid") > .5)
        local_ready = bool(_scalar(policy_output, "log_spatial_local_valid", float(pose_ready)))
        ep.pose_steps += 1
        ep.localized_steps += int(pose_ready)
        ep.local_tracking_steps += int(local_ready)
        phase = int(_scalar(policy_output, "log_vnext_phase"))
        row = dict(step=index, phase=phase, option=option,
                   actor=int(_scalar(policy_output, "log_vnext_actor_action")),
                   source=int(_scalar(policy_output, "log_vnext_source", -1)),
                   override=bool(_scalar(policy_output, "log_vnext_override")),
                   visible=visible, pose=pose.tolist(), pose_ready=pose_ready, reward=0., success=False,
                   pose_gap_count=int(_scalar(policy_output,"log_spatial_pose_gap_count")),
                   segment=int(_scalar(policy_output,"log_spatial_segment")),
                   local_ready=local_ready,
                   pose_model_ready=(bool(_scalar(policy_output, "log_spatial_pose_model_ready"))
                                     if "log_spatial_pose_model_ready" in policy_output else None),
                   map_write=(bool(_scalar(policy_output,"log_spatial_map_write"))
                              if "log_spatial_map_write" in policy_output else None),
                   memory_available=bool(_scalar(policy_output,"log_spatial_memory_available")),
                   goal_state=int(_scalar(policy_output, "log_goal_state", -1)),
                   goal_gate=_scalar(policy_output, "log_goal_gate"),
                   goal_model_update=int(_scalar(policy_output, "log_goal_model_update", -1)),
                   goal_conflict=_scalar(policy_output, "log_goal_conflict"),
                   goal_verify=bool(_scalar(policy_output, "log_goal_verify_used")),
                   goal_reject=bool(_scalar(policy_output, "log_goal_reject_used")),
                   value=None)
        if len(ep.rows) >= self.max_rows:
            ep.omitted_rows += 1
            return
        previous = ep.rows[-1] if ep.rows else None
        decision = bool(_scalar(policy_output, "log_vnext_decision"))
        row["control_kind"] = row["source"]
        if row["source"] == 4 and (_scalar(policy_output,"log_vnext_recovery_probe") > .5 or
                (not decision and previous and previous["control_kind"] == 10)):
            row["control_kind"] = 10  # recovery probe whose chosen option is settle
        if row["source"] == 7 and row["goal_verify"]:
            row["control_kind"] = 11  # temporal verification uses the alignment source
        ep.rows.append(row)
        goal_priority = (3 * row["goal_conflict"] + 2 * row["goal_reject"]
                         + 2 * (row["goal_state"] == 2 and
                                (previous is None or previous["goal_state"] != 2)))
        goal_rank, goal_reason, goal_delta = None, None, None
        if 0 <= row["goal_state"] <= 2:
            goal_original = _array(obs.get("goal_original_heatmap", obs.get("heatmap")))
            goal_effective = _array(policy_output.get("log_goal_heatmap_policy"))
            if row["goal_reject"]:
                goal_rank, goal_reason = (4, goal_priority), "rejected candidate"
            elif row["goal_state"] == 2 and (previous is None or previous["goal_state"] != 2):
                goal_rank, goal_reason = (3, goal_priority), "trusted target onset"
            elif row["goal_verify"]:
                goal_rank, goal_reason = (2, goal_priority), "verification action"
            else:
                if row["goal_gate"] > 0:
                    goal_delta = _goal_heat_delta(goal_original, goal_effective)
                if goal_delta and goal_delta[0] > 0.:
                    goal_rank, goal_reason = (1, goal_delta[0]), "largest applied heatmap change"
                else:
                    peak = float(goal_original.max(initial=0)) if goal_original is not None else 0.
                    goal_rank, goal_reason = (0, peak), "strongest original heat (fallback)"
        goal_snapshot = goal_rank is not None and (
            ep.goal is None or goal_rank > ep.goal["goal_rank"])
        b_group = ("qualified continuous track" if pose_ready else
                   "qualified local track" if local_ready else "visual history without local pose")
        if not local_ready and not row["memory_available"]:
            b_group = None
        if previous and previous["segment"] != row["segment"]:
            b_group = "local frame reset"
        goal_group = None
        if goal_rank is not None:
            goal_group = ("peak conflict" if row["goal_reject"] else "trusted regional evidence"
                          if row["goal_state"] == 2 else "fusion inactive" if row["goal_gate"] == 0
                          else "applied fusion")
        if goal_group and goal_original is not None:
            heat = np.asarray(goal_original).squeeze()
            if heat.ndim == 2 and heat[:max(1,len(heat)//3)].max() >= (.7 if heat.dtype.kind=="f" and heat.max()<=1 else 178) and row["goal_state"] == 0:
                goal_group = "upper-view distractor suspected"
            elif row["goal_conflict"] >= .25:
                goal_group = "regional disagreement"
        # A calibrated fusion case has an actual quantized policy-input change.
        # Keep its strongest example independently of rejected/trusted/fallback rankings.
        qualified_delta = None
        if goal_rank is not None and row["goal_gate"] > 0 and row["goal_model_update"] > 0:
            reliability = _scalar(policy_output, "log_goal_region_cal_reliability", np.nan)
            if np.isfinite(reliability) and reliability >= .15:
                qualified_delta = goal_delta or _goal_heat_delta(goal_original, goal_effective)
        previous_fusion = ep.groups.get("calibrated active fusion")
        keep_fusion = (qualified_delta is not None and qualified_delta[0] > 0 and
                       (previous_fusion is None or qualified_delta[0] > previous_fusion["goal_heat_delta"][0]))
        keep_fallback = (goal_rank is not None and row["goal_gate"] == 0 and
                         "zero-gate fallback" not in ep.groups and
                         _goal_heat_delta(goal_original, goal_effective) == (0., 0.))
        goal_example = goal_group and (goal_group not in ep.groups or goal_snapshot)
        ep.pending = None
        if not (decision or goal_snapshot or goal_example or keep_fusion or keep_fallback or index == 0 or (b_group and b_group not in ep.groups)):
            return
        prediction = _array(policy_output.get("log_vnext_prediction"))
        mask = _array(policy_output.get("log_vnext_mask"), []).reshape(-1).astype(bool)
        has_prediction = (prediction is not None and prediction.ndim == 2
                          and prediction.shape[1] >= 7 and len(mask) == len(prediction)
                          and np.isfinite(prediction).all() and bool(np.any(prediction)))
        rank = int(_scalar(policy_output, "log_visual_rank_choice", -1))
        if "log_visual_rank_choice" in policy_output:
            has_prediction = has_prediction and rank >= 0
        q_computed = has_prediction or (rank >= 0 if "log_visual_rank_choice" in policy_output else bool(np.any(q)))
        row["value"] = float(q[option]) if q_computed and 0 <= option < len(q) else None
        legal_values = q[mask] if len(q) == len(mask) else np.array([])
        margin = float(np.diff(np.sort(legal_values))[-1]) if len(legal_values) > 1 else float("inf")
        phase_change = previous is not None and previous["phase"] != phase
        priority = 4*int(row["override"]) + 3*int(phase_change) + int(visible >= .7)
        priority += max(0., .1-margin) if math.isfinite(margin) else 0.
        snap = dict(row=row, environment_step=int(training_step),
                    weight_update=int(_scalar(policy_output, "log_visual_weight_update", -1)),
                    rank_choice=rank, rank_mask=_array(policy_output.get("log_visual_rank_mask")),
                    template_commands=_array(policy_output.get("log_visual_template_commands")) if rank >= 0 else None,
                    template_lengths=_array(policy_output.get("log_visual_template_lengths")) if rank >= 0 else None,
                    memory_input=(bool(_scalar(policy_output, "log_visual_memory_input"))
                                  if "log_visual_memory_input" in policy_output else None),
                    event_support=_array(policy_output.get("log_visual_event_support")),
                    requested=int(_scalar(policy_output, "log_vnext_requested", -1)),
                    horizon=int(_scalar(policy_output, "log_vnext_horizon", 0)), rgb=_rgb(obs.get("image", obs.get("rgb"))),
                    heatmap=_array(obs.get("heatmap")),
                    patch=_array(policy_output.get("log_spatial_patch")),
                    prediction=None if prediction is None else prediction.copy(),
                    mask=mask.copy(), q=q.copy(),
                    a_score=_array(policy_output.get("log_vnext_a_score")),
                    supported=_array(policy_output.get("log_vnext_score_supported")),
                    priority=priority, has_prediction=has_prediction,
                    q_computed=q_computed,
                    reasons=[r for r, yes in (("override", row["override"]),
                             ("phase_change", phase_change), ("close_values", margin < .1),
                             ("target_evidence", visible >= .7)) if yes])
        if decision and not local_ready and row["memory_available"]:
            frames = _array(policy_output.get("log_spatial_history_frames"))
            valid = _array(policy_output.get("log_spatial_history_valid"))
            if frames is not None and valid is not None and valid.any():
                snap["history_preview"] = np.concatenate(list(frames[valid.astype(bool)][:3, ..., :3]), axis=1)
        # Copy retained arrays only at actual decision points, not every frame.
        for field_name in ("heatmap", "patch", "a_score", "supported"):
            if snap[field_name] is not None:
                snap[field_name] = snap[field_name].copy()
        if row["goal_state"] >= 0:
            for name in ("ref_map", "heatmap_policy", "boxes", "candidate_scores",
                         "candidate_sources", "forecast", "new_area_forecast", "context"):
                value = _array(policy_output.get("log_goal_" + name))
                snap["goal_" + name] = None if value is None else value.copy()
            for key in ("task_q", "search_q", "applied"):
                value = _array(policy_output.get("log_vnext_" + key))
                snap[key] = None if value is None else value.copy()
            snap["goal_priority"] = goal_priority
            snap["priority"] += goal_priority
        if goal_rank is not None:
            native = None
            if (_get(self.config, "goal_ref_use_raw_image", True)
                    and _scalar(obs, "goal_image_valid") > .5):
                native = _rgb(obs.get("goal_image"))
            if native is not None:
                snap["goal_rgb"], snap["goal_rgb_source"] = native, "native D input"
            else:
                snap["goal_rgb"] = _rgb(obs.get("goal_policy_rgb", obs.get("image")))
                snap["goal_rgb_source"] = "policy frame"
            snap["goal_original_heatmap"] = None if goal_original is None else goal_original.copy()
            snap["goal_heat_delta"] = goal_delta or _goal_heat_delta(goal_original, goal_effective)
            snap["goal_region_cal_reliability"] = _scalar(
                policy_output, "log_goal_region_cal_reliability", np.nan)
            snap["goal_rank"], snap["goal_selection_reason"] = goal_rank, goal_reason
            if goal_snapshot:
                ep.goal = snap
            if keep_fallback:
                ep.groups["zero-gate fallback"] = snap
            if keep_fusion:
                ep.groups["calibrated active fusion"] = snap
            if goal_example:
                ep.groups[goal_group] = snap
        ep.pending = snap
        if index == 0:
            ep.groups["reset observation"] = snap
        if b_group:
            ep.groups.setdefault(b_group, snap)
        if decision:
            criteria = []
            if row["source"] == 1 and row["override"] and snap["requested"] != row["actor"] and snap["requested"] >= 0:
                criteria.append("learned change to Actor command")
            if rank >= 0 and row["source"] >= 2 and rank != option:
                criteria.append("rule superseded ranking")
            if has_prediction and mask.sum() >= 2:
                span = np.ptp(prediction[mask, :7], axis=0).max()
                if span >= .20:
                    criteria.append("divergent legal forecasts")
                supported = snap.get("supported")
                if supported is not None and np.sum(mask & supported.astype(bool)) >= 2:
                    criteria.append("multiple supported forecasts")
            elif not mask[1:].any():
                criteria.append("no eligible alternative")
            for reason in criteria:
                snap["reasons"].append(reason)
                ep.groups.setdefault(reason, snap)
            if not snap["reasons"]:
                snap["reasons"].append("retained controller decision")
            if has_prediction and mask.sum() >= 2:
                ep.a = self._retain(ep.a, snap, 3)
            ep.c = self._retain(ep.c, snap, 2)
            ep.last = snap

    @staticmethod
    def _retain(bucket, snapshot, limit):
        values = [s for s in bucket if s["row"]["step"] != snapshot["row"]["step"]]
        values.append(snapshot)
        return sorted(values, key=lambda s: (-s["priority"], -s["row"]["step"]))[:limit]

    def feedback(self, mode, env_id, reward, info, *, arrival=None, done=False):
        self._open()
        ep = self._active.get((mode, str(env_id)))
        if ep is None:
            return
        r = float(np.asarray(reward).sum())
        success = bool(np.asarray(info.get("success", False)).any())
        ep.reward += r
        ep.success |= success
        if ep.rows and not ep.omitted_rows:
            ep.rows[-1]["reward"] = r
            ep.rows[-1]["success"] = success

        # Audit only after env.step. Never feed the arrival into the decision.
        event = ep.pending
        if arrival is not None and not ep.omitted_rows:
            if event is not None:
                event["feedback"] = dict(reward=r, success=success, arrival_index=event["row"]["step"]+1)
                if (_scalar(arrival, "log_actuator_valid") == 1 and not _scalar(arrival, "log_env_error")
                        and not _scalar(arrival, "is_first")):
                    actual = _array(arrival.get("log_applied_control"))
                    if actual is not None and actual.size == 9 and np.isfinite(actual).all():
                        event["env_applied"] = actual.reshape(9).copy()
                        option = event["row"]["option"]
                        if (event["row"]["option"] == 0 or event["row"]["override"]):
                            event["feedback"]["control_matches_controller"] = (bool(np.allclose(actual, event["applied"])) if event.get("applied") is not None else None)
                        if (event["horizon"] == 1 and event["has_prediction"] and
                                0 <= option < len(event["mask"]) and event["mask"][option]
                                and _scalar(arrival, "search_progress_valid") == 1 and
                                event.get("applied") is not None and np.allclose(actual, event["applied"])):
                            label = _scalar(arrival, "search_progress")
                            if abs(float(event["prediction"][option, 5]) - label) >= .5:
                                event["feedback"]["audited_progress"] = label
                                event["reasons"].append("executed H1 progress mismatch")
                                ep.groups.setdefault("executed H1 progress mismatch", event)
                event["arrival_rgb"] = _rgb(arrival.get("image"))
            if done:
                ep.terminal = dict(rgb=_rgb(arrival.get("image")), step=len(ep.rows), success=success)
        ep.pending = None

    def episode_end(self, mode, env_id, episode, training_step, eval_index=None,
                    eval_total=None, map_state=None):
        self._open()
        key = (mode, str(env_id))
        self._partial_prefixes.discard(key)
        ep = self._active.pop(key, None)
        complete_trace = ep is not None and bool(ep.rows)
        if not self.enabled:
            return None
        if mode == "train":
            if not complete_trace:
                return None
            snap = self._complete(ep, episode, env_id, training_step, map_state)
            # A threshold strictly before this completion uses the PREVIOUS
            # complete episode; a completion exactly at the threshold may use it.
            self.maybe_export_train(int(training_step)-1)
            self._latest_train = snap
            return self.maybe_export_train(training_step)
        if eval_index is None or eval_total is None:
            raise ValueError("Evaluation visualization needs 1-based index and total")
        if self._all_eval_maps and not self._scheduled_eval(int(training_step)):
            return None
        last = int(eval_index) == int(eval_total)
        if not self._all_eval_maps and not (last and complete_trace):
            return None
        path = self.root / "eval" / ("step_%09d" % training_step) / ("episode_%02d" % eval_index)
        if last and complete_trace:
            snap = self._complete(ep, episode, env_id, training_step, map_state)
            return self._export(snap, path, training_step, "eval")
        # A selected evaluation with no complete trace still gets a status map.
        # Do not use a partial trace or its terminal position as map geometry.
        snap = (self._complete(ep, episode, env_id, training_step, map_state) if complete_trace else
                self._complete(_Episode(), episode, env_id, training_step, None))
        self._queue_episode_map(snap, path / "B_map.png")
        return None

    def _queue_episode_map(self, snap, path):
        # Keep at most four outstanding CPU snapshots. Normal rollouts take
        # far longer than a render. If the writer falls behind, preserve every
        # episode with backpressure instead of growing memory or dropping maps.
        while self._map_jobs and (self._map_jobs[0].done() or len(self._map_jobs) >= 4):
            self._map_jobs.popleft().result()
        map_snapshot = {key: snap[key] for key in ("summary", "rows", "map_state", "terminal_pose")}
        self._map_jobs.append(self._map_executor.submit(self._export_episode_map, map_snapshot, path))

    def _export_episode_map(self, snap, path):
        if path.is_file():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".map-", dir=path.parent) as tmp:
                pending = Path(tmp) / path.name
                with self._render_lock:
                    self._render_map(snap, pending)
                os.replace(pending, path)
            self.stats["map_exports"] += 1
        except Exception as error:
            self.stats["map_errors"] += 1
            print(f"[search visualization] episode map failed: {path} ({error})", file=sys.stderr, flush=True)

    def maybe_export_train(self, training_step):
        self._open()
        if not self.enabled:
            return None
        result = None
        while self._next_train <= int(training_step):
            point = self._next_train
            self._next_train += self.interval
            snap = self._latest_train
            if snap is None or snap["summary"]["episode_completed_at_training_step"] > point:
                self.stats["skipped_no_episode"] += 1
                continue
            path = self.root / "train" / ("step_%09d" % point) / "episode_last"
            identity = (snap["summary"]["episode_id"], snap["summary"]["episode_completed_at_training_step"])
            repeated = identity == self._last_exported_train_id
            result = self._export(snap, path, point, "train", repeated=repeated)
            if result is not None:
                self._last_exported_train_id = identity
        return result

    def _complete(self, ep, episode, env_id, training_step, map_state):
        reward = _array(episode.get("reward"), [])
        success = _array(episode.get("success"), [])
        visible = _array(episode.get("search_visible"), []).reshape(-1)
        valid = _array(episode.get("search_visible_valid"), []).reshape(-1).astype(bool)
        known = len(visible) == len(valid) and valid.any()
        hits = np.flatnonzero(valid & (visible > .5)) if known else np.array([], int)
        predicted = [r["step"] for r in ep.rows if r["visible"] >= .7]
        revision = int(_get(self.config, "vnext_spatial_fix_version", 0))
        state_fields = map_state if map_state is not None else {}
        positions = np.asarray([r["pose"][:2] for r in ep.rows], dtype=float).reshape(-1, 2)
        last_row = ep.rows[-1] if ep.rows else {}
        terminal_pose = _array(state_fields.get("pose"))
        if terminal_pose is not None:
            terminal_pose = terminal_pose.reshape(-1)
            if len(terminal_pose) < 4 or not np.isfinite(terminal_pose[:4]).all():
                raise ValueError("Terminal visualization pose must be finite [x,z,yaw,uncertainty]")
            terminal_pose = terminal_pose[:4].copy()
        include_terminal = terminal_pose is not None and ep.omitted_rows == 0
        local_terminal = bool(_scalar(state_fields, "local_tracking_ready",
                                      _scalar(state_fields, "tracking_ready", last_row.get("local_ready", False))))
        continuity_key = "trajectory_valid" if revision >= 3 else "tracking_ready"
        terminal_tracking_ready = bool(_scalar(state_fields, continuity_key, last_row.get("pose_ready", False)))
        if revision >= 3:
            terminal_tracking_ready &= local_terminal
        localized_fraction = ep.localized_steps/max(1, ep.pose_steps)
        local_tracking_fraction = ep.local_tracking_steps/max(1, ep.pose_steps)
        available = np.asarray([row["pose_ready"] for row in ep.rows], bool)
        local_available = np.asarray([row["local_ready"] for row in ep.rows], bool)
        segments = np.asarray([row["segment"] for row in ep.rows], int)
        active_segment = int(_scalar(state_fields, "active_segment", last_row.get("segment", 0)))
        if include_terminal:
            positions = np.concatenate((positions, terminal_pose[None, :2]), axis=0)
            available = np.r_[available, terminal_tracking_ready]
            local_available = np.r_[local_available, local_terminal]
            segments = np.r_[segments, active_segment]
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        rendered_segment = None
        rendered_segment_steps = 0
        rendered_segment_extent = None
        if revision >= 3:
            # Select a qualified local estimate without joining unrelated origins.
            local_available &= np.isfinite(positions).all(-1)
            moves = local_available[:-1] & local_available[1:] & (segments[:-1] == segments[1:])
            moving_segments = np.unique(segments[1:][moves & (lengths > 1.e-6)])
            available_segments = segments[local_available]
            if len(moving_segments):
                # Prefer the widest qualified route over a nearly stationary
                # terminal fragment. Keep separate coordinate frames intact.
                rendered_segment = max(map(int, moving_segments), key=lambda segment: (
                    np.linalg.norm(np.ptp(positions[local_available & (segments == segment)], axis=0)),
                    segment == active_segment, segment))
            elif active_segment in available_segments:
                rendered_segment = active_segment
            elif len(available_segments):
                rendered_segment = int(available_segments[-1])
            if rendered_segment is not None:
                selected = local_available & (segments == rendered_segment)
                rendered_segment_steps = int(selected.sum())
                rendered_segment_extent = np.ptp(positions[selected], axis=0).tolist()
        if revision >= 2:
            same_path = available[:-1] & available[1:]
            if revision >= 3:
                same_path &= segments[:-1] == segments[1:]
            lengths = lengths[same_path]
        trajectory = float(lengths.sum())
        fully_localized = ep.localized_steps == ep.pose_steps and terminal_tracking_ready and ep.omitted_rows == 0
        terminal_gap = int(_scalar(state_fields, "pose_gap_count", _scalar(state_fields, "gap_count")))
        terminal_first_gap = int(_scalar(state_fields, "first_gap_step", -1.))
        first_gap = next((row["step"] for row in ep.rows if row["pose_gap_count"]>0), None)
        if terminal_first_gap >= 0:
            first_gap = terminal_first_gap if first_gap is None else min(first_gap, terminal_first_gap)
        grid = None if map_state is None else _array(map_state.get("grid"))
        while grid is not None and grid.ndim > 3 and grid.shape[0] == 1:
            grid = grid[0]
        if grid is not None and (grid.ndim != 3 or grid.shape[-1] < 6):
            raise ValueError("Spatial visualization grid must have shape [H,W,>=6]")
        state = None if map_state is None else {k: _json(v) for k, v in map_state.items() if k != "grid"}
        if grid is not None:
            state["grid"] = grid.copy()
        has_ledger = map_state is not None and "ledger" in map_state
        ledger_values = np.empty((0, 6), dtype=float)
        if has_ledger:
            ledger_cells = _array(map_state["ledger"].get("cells", [])).reshape(-1, 2)
            ledger_values = _array(map_state["ledger"].get("values", [])).reshape(-1, 6)
            if len(ledger_cells) != len(ledger_values):
                raise ValueError("Spatial ledger cells and values have different lengths")
            if int(_get(self.config,"vnext_spatial_fix_version",0)) >= 3:
                tags = _array(map_state["ledger"].get("segments",[])).reshape(-1)
                active = int(map_state.get("active_segment",0))
                if len(tags) != len(ledger_values):
                    raise ValueError("Segmented ledger requires one tag per historical cell")
                ledger_values = ledger_values[tags==active]
        visited_cells = (int((ledger_values[:, 0] > 0).sum()) if has_ledger else
                         int((grid[..., 0] > 0).sum()) if grid is not None else None)
        revisits = (float(ledger_values[:, 3].sum()) if has_ledger else
                    float(grid[..., 3].sum()) if grid is not None else None)
        summary = dict(episode_id=str(env_id), episode_completed_at_training_step=int(training_step),
            spatial_fix_version=revision,
            active_segment=active_segment,
            rendered_segment=rendered_segment,
            rendered_segment_steps=rendered_segment_steps,
            rendered_segment_extent_m=rendered_segment_extent,
            local_tracking_ready=local_terminal,
            pose_qualification=state_fields.get("pose_qualification"),
            persistent_memory_available=bool(map_state.get("memory_available",False)) if map_state else False,
            history_slots_retained=(int(np.asarray(map_state.get("history_valid",[])).sum()) if map_state else 0),
            memory_available_fraction=(sum(r.get("memory_available", False) for r in ep.rows) /
                                       max(1, len(ep.rows))),
            map_scope=("single qualified local segment; disconnected history retained separately"
                       if revision>=3 else "episode-origin map"),
            pose_gap_count=max(max((row["pose_gap_count"] for row in ep.rows), default=0), terminal_gap),
            first_pose_gap_step=first_gap,
            success=bool(success.any()) if success.size else ep.success,
            total_steps=max(0, len(reward)-1) if reward.size else len(ep.rows)+ep.omitted_rows,
            total_reward=float(reward.sum()) if reward.size else ep.reward,
            estimated_trajectory_length=trajectory if fully_localized else None,
            integrated_path_length=trajectory,
            trajectory_length_valid=fully_localized,
            localized_fraction=localized_fraction,
            unlocalized_steps=ep.pose_steps-ep.localized_steps,
            local_tracking_fraction=local_tracking_fraction,
            local_tracking_steps=ep.local_tracking_steps,
            pose_observation_steps=ep.pose_steps,
            terminal_tracking_ready=terminal_tracking_ready,
            localization_note=("Predicted pose is qualified" if fully_localized else
                               "Unqualified pose disables translation integration; origin clustering does not imply actual stillness"),
            estimated_terminal_pose=None if terminal_pose is None else terminal_pose.tolist(),
            trajectory_timing="decision-time estimates plus separately predicted terminal observation pose",
            first_target_step=int(hits[0]) if len(hits) else None,
            target_step_source="ray audit; sparse proxy, not full visibility truth" if known else "unavailable",
            first_predicted_target_step=predicted[0] if predicted else None,
            estimated_explored_cells=(int((ledger_values[:, 1] > 0).sum()) if has_ledger else
                                      int((grid[..., 1] > 0).sum()) if grid is not None else None),
            estimated_visited_cells=visited_cells,
            retained_observed_proxy_cells=int((grid[..., 1] > 0).sum()) if grid is not None else None,
            number_of_revisits=revisits,
            full_episode_ledger_available=has_ledger,
            ledger_cells=len(ledger_values) if has_ledger else None,
            ledger_capacity=None if map_state is None else map_state.get("ledger_capacity"),
            coverage_scope=("episode visited-cell ledger plus rolling directional-view proxy; not true terrain"
                            if has_ledger else "retained rolling search map; not whole-world ground truth"),
            map_source="predicted visual odometry, no simulator coordinates",
            map_snapshot_available=grid is not None,
            target_map_semantics="target observation viewpoints, not localized target blocks",
            omitted_timeline_steps=ep.omitted_rows,
            candidate_prediction_count=len(ep.a),
            candidate_prediction_note=("computed action-conditioned predictions" if ep.a else
                                       "No eligible computed candidate prediction in this episode"))
        # Prefer distinct recorded mechanisms over near-identical top-ranked frames.
        preferred = [ep.groups.get(k) for k in ("learned change to Actor command",
                     "executed H1 progress mismatch", "rule superseded ranking",
                     "multiple supported forecasts", "divergent legal forecasts")] + ep.a
        distinct = {}
        for event in preferred:
            if event is not None and event["has_prediction"] and event["mask"].sum() >= 2:
                distinct.setdefault(event["row"]["step"], event)
        ep.a = list(distinct.values())[:3]
        # Ensure the final real decision is represented in the C explanations.
        if ep.last is not None and all(s["row"]["step"] != ep.last["row"]["step"] for s in ep.c):
            ep.c = ([ep.c[0]] if ep.c else []) + [ep.last]
        if ep.goal is not None:
            summary.update(goal_ref_version=int(_get(self.config,"goal_ref_version",2)),
                first_trusted_target_step=next((r["step"] for r in ep.rows if r["goal_state"] == 2), None),
                goal_verify_steps=sum(r["goal_verify"] for r in ep.rows),
                goal_reject_steps=sum(r["goal_reject"] for r in ep.rows),
                goal_conflict_mean=float(np.mean([r["goal_conflict"] for r in ep.rows])),
                goal_keyframe=self._decision_json(ep.goal))
        return dict(summary=summary, rows=ep.rows, a=ep.a, c=ep.c, map_state=state,
                    groups=ep.groups, terminal=ep.terminal, goal=ep.goal, terminal_pose=terminal_pose)

    def _export(self, snap, path, training_step, mode, repeated=False):
        if (path / "summary.json").exists():
            self.stats["skipped_duplicate"] += 1
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".render-", dir=path.parent) as tmp:
                tmp = Path(tmp)
                with self._render_lock:
                    self._render(snap, tmp)
                from training_state import CODE_VERSION
                from .paper_archive import write_keyframes
                write_keyframes(tmp, snap, getattr(self, "goal_references", []))
                summary = dict(snap["summary"], task=str(_get(self.config,"task","unknown")),
                               code_version=CODE_VERSION, bank=self.bank_info,
                               frame_alignment="Decision t uses observation t. Arrival RGB, reward and environment control belong to transition t to t+1.",
                               support_semantics="Predictive support is measured on held-out labels. Missing support, ranking or actuator telemetry is unknown.",
                               audit_semantics="H1 mismatch uses an actual matching control and a valid post-action training label; it cannot enter a policy input.",
                               checkpoint_id=self.checkpoint_id, run_id=str(_get(self.config,"logdir","")),
                               weight_source="frozen checkpoint" if self.checkpoint_id else "live training weights",
                               calibrated_fusion_mean_delta=(snap["groups"]["calibrated active fusion"]["goal_heat_delta"][0]
                                      if "calibrated active fusion" in snap["groups"] else None),
                               case_groups={k: v["row"]["step"] for k,v in snap["groups"].items()},
                               mode=mode, export_training_step=int(training_step),
                               repeated_complete_episode=bool(repeated),
                               episode_age_at_export_steps=int(training_step)-snap["summary"]["episode_completed_at_training_step"],
                               selection_policy="latest completed episode at threshold; reuse explicitly recorded",
                               selected_keyframes_for_A=[self._decision_json(s) for s in snap["a"]],
                               selected_keyframes_for_C=[self._decision_json(s) for s in snap["c"]])
                (tmp / "summary.json").write_text(json.dumps(_json(summary), indent=2, allow_nan=False)+"\n", encoding="utf-8")
                rows = snap["rows"]
                map_state = snap["map_state"] or {}
                ledger = map_state.get("ledger", {})
                np.savez_compressed(tmp / "episode_trace.npz",
                    control_kind=np.asarray([r["control_kind"] for r in rows], np.int16),
                    actor=np.asarray([r["actor"] for r in rows], np.int16),
                    override=np.asarray([r["override"] for r in rows], bool),
                    pose=np.asarray([r["pose"] for r in rows], np.float32),
                    pose_ready=np.asarray([r["pose_ready"] for r in rows], bool),
                    pose_gap_count=np.asarray([r["pose_gap_count"] for r in rows],np.int16),
                    segment=np.asarray([r["segment"] for r in rows],np.int32),
                    local_valid=np.asarray([r["local_ready"] for r in rows],bool),
                    memory_available=np.asarray([r["memory_available"] for r in rows],bool),
                    phase=np.asarray([r["phase"] for r in rows], np.int16),
                    option=np.asarray([r["option"] for r in rows], np.int16),
                    source=np.asarray([r["source"] for r in rows], np.int16),
                    visible=np.asarray([r["visible"] for r in rows], np.float32),
                    goal_state=np.asarray([r["goal_state"] for r in rows], np.int8),
                    goal_verify=np.asarray([r["goal_verify"] for r in rows], bool),
                    goal_reject=np.asarray([r["goal_reject"] for r in rows], bool),
                    goal_gate=np.asarray([r["goal_gate"] for r in rows], np.float32),
                    goal_model_update=np.asarray([r["goal_model_update"] for r in rows], np.int32),
                    goal_conflict=np.asarray([r["goal_conflict"] for r in rows], np.float32),
                    ledger_goal_evidence=np.asarray(ledger.get("goal_evidence", []), np.float32).reshape(-1, 5),
                    history_goal=np.asarray(map_state.get("history_goal", []), np.float32).reshape(-1, 5),
                    reward=np.asarray([r["reward"] for r in rows], np.float32),
                    success=np.asarray([r["success"] for r in rows], bool),
                    terminal_pose=(np.empty((0,), np.float32) if snap["terminal_pose"] is None
                                   else snap["terminal_pose"].astype(np.float32)),
                    map_grid=np.asarray(map_state.get("grid", np.empty((0, 0, 6))), np.float32),
                    map_origin=np.asarray(map_state.get("origin", []), np.float32),
                    map_resolution=np.asarray(map_state.get("resolution", 0.), np.float32),
                    ledger_cells=np.asarray(ledger.get("cells", []), np.int64).reshape(-1, 2),
                    ledger_values=np.asarray(ledger.get("values", []), np.float32).reshape(-1, 6),
                    ledger_directions=np.asarray(ledger.get("directions", []), np.float32).reshape(-1, 12),
                    ledger_segments=np.asarray(ledger.get("segments", []),np.int32),
                    history_frames=np.asarray(map_state.get("history_frames", []),np.uint8),
                    history_meta=np.asarray(map_state.get("history_meta", []),np.float32),
                    history_query_weights=np.asarray(map_state.get("history_query_weights", []), np.float32),
                    history_query_tick=np.asarray(map_state.get("history_query_tick", []), np.int32),
                    history_valid=np.asarray(map_state.get("history_valid", []),bool))
                if path.exists():
                    raise FileExistsError("Refusing to replace an incomplete visualization directory: %s" % path)
                tmp.rename(path)
            self.stats["exports"] += 1
            return path
        except Exception as error:
            self.stats["errors"] += 1
            self.last_error = "%s: %s" % (type(error).__name__, error)
            print("[search visualization] export failed: %s (%s)" % (path, self.last_error), file=sys.stderr, flush=True)
            if self.strict:
                raise
            return None

    @staticmethod
    def _decision_json(snap):
        return _json(dict(step=snap["row"]["step"], actor=snap["row"]["actor"],
            selected_option=snap["row"]["option"], source=SOURCES.get(snap["row"]["source"], "unspecified"),
            phase=snap["row"]["phase"], reasons=snap["reasons"],
            predictions=snap["prediction"] if snap["has_prediction"] else None,
            legal_mask=snap["mask"], a_score=snap["a_score"],
            c_value=snap["q"] if snap["q_computed"] else None,
            goal_state=snap["row"].get("goal_state", -1), goal_context=snap.get("goal_context"),
            goal_forecast=snap.get("goal_forecast"), goal_candidates=snap.get("goal_candidate_scores"),
            goal_new_area_forecast=snap.get("goal_new_area_forecast"),
            goal_gate=snap["row"].get("goal_gate"), goal_conflict=snap["row"].get("goal_conflict"),
            goal_model_update=snap["row"].get("goal_model_update"),
            goal_selection_reason=snap.get("goal_selection_reason"),
            goal_rgb_source=snap.get("goal_rgb_source"),
            goal_region_cal_reliability=snap.get("goal_region_cal_reliability"),
            goal_heat_delta_mean_max=snap.get("goal_heat_delta"),
            task_q=snap.get("task_q") if snap["q_computed"] else None,
            search_q=snap.get("search_q") if snap["q_computed"] else None,
            controller_applied=snap.get("applied"), actual_control=snap.get("env_applied"),
            unexecuted_candidates="predictions only; no real future frames are claimed"))

    def _figure(self, size):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=size, dpi=120)
        FigureCanvasAgg(fig)
        protocol = _get(self.config, "goal_ref_version", None)
        tag = f"D v{protocol}" if protocol in (1, 2) else "D protocol unrecorded"
        fig.text(.985, .995, tag, ha="right", va="top", fontsize=8, color="#52677a")
        return fig

    @staticmethod
    def _save(fig, path, *, tight=True):
        fig.savefig(path, dpi=120, bbox_inches="tight" if tight else None, facecolor="white")
        fig.clear()

    def _render(self, snap, folder):
        from .paper_figures import decision, history, storyboard, timeline
        history(self, snap, folder / "B_history.png")
        storyboard(self, snap, folder / "episode_events.png")
        eligible = list(snap.get("groups", {}).values()) + snap["a"] + snap["c"]
        learned = next((e for e in eligible if e["row"]["source"] == 1 and e["row"]["override"]
                        and e["has_prediction"] and e["mask"].sum() >= 2), None)
        overridden = snap.get("groups", {}).get("rule superseded ranking")
        if overridden is not None and overridden["has_prediction"]:
            decision(self, overridden, folder / "AC_rule_override.png", "AC")
        if learned is not None:
            decision(self, learned, folder / "AC_decision.png", "AC")
        focus = learned or (snap["c"][0] if snap["c"] else None)
        if focus is not None:
            timeline(self, snap, folder / "C1_window.png", focus["row"]["step"])
        self._render_map(snap, folder / "B_map.png")
        self._render_timeline(snap, folder / "C1_timeline.png")
        fusion = snap.get("groups", {}).get("calibrated active fusion")
        if fusion is not None:
            self._render_goal(fusion, folder / "D_calibrated_fusion.png")
        fallback = snap.get("groups", {}).get("zero-gate fallback")
        if fallback is not None and fallback is not snap["goal"]:
            self._render_goal(fallback, folder / "D_zero_gate_fallback.png")
        if snap["goal"] is not None:
            self._render_goal(snap["goal"], folder / "G_goal_verification.png")
        if not snap["a"]:
            self._render_unavailable_a(snap["summary"], folder / "A_status.png")
        for i, event in enumerate(snap["a"][:3], 1):
            self._render_decision(event, folder / ("A1_%02d.png" % i), "A · Candidate outcome predictions")
        for i, event in enumerate(snap["a"][:2], 1):
            self._render_outcomes(event, folder / ("A2_%02d.png" % i))
        for i, event in enumerate(snap["c"][:2], 1):
            self._render_decision(event, folder / ("C2_%02d.png" % i), "C · Actual high-level decision")

    def _render_map(self, snap, path):
        from matplotlib.lines import Line2D
        from matplotlib import patheffects
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        fig = self._figure((10, 7))
        # Fix the physical panel to 4.55 × 4.55 inches. Both data axes
        # share the same span so a cell and a metre have consistent scale.
        ax = fig.add_axes([.12, .18, .455, .65])
        sidebar = fig.add_axes([.64, .18, .33, .65])
        sidebar.set_axis_off()
        ax.set_facecolor("#ecedef")
        state = snap["map_state"]
        summary = snap["summary"]
        segmented = summary["spatial_fix_version"] >= 3
        segment = summary.get("rendered_segment") if segmented else None
        prior_segment = segmented and segment is not None and segment != summary["active_segment"]
        poses = np.asarray([r["pose"] for r in snap["rows"]], float).reshape(-1, 4)
        if segmented:
            available = np.asarray([r["local_ready"] and r["segment"] == segment for r in snap["rows"]], bool)
            terminal_valid = summary["local_tracking_ready"] and segment == summary["active_segment"]
        else:
            available = np.asarray([r["pose_ready"] for r in snap["rows"]], bool)
            terminal_valid = summary["terminal_tracking_ready"]
        if snap["terminal_pose"] is not None and not summary["omitted_timeline_steps"]:
            poses = np.concatenate((poses, snap["terminal_pose"][None]), axis=0)
            available = np.r_[available, terminal_valid]
        available &= np.isfinite(poses).all(-1)
        has_pose = bool(available.any())
        if not has_pose:
            ax.text(.5, .60, "UNLOCALIZED", ha="center", va="center",
                    fontsize=15, weight="bold", transform=ax.transAxes)
            message = ("No qualified positions in this episode." if len(poses) else
                       "No complete B trace for this episode.")
            ax.text(.5, .51, message + "\nVisual history is shown when available.",
                    ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.set_xlim(-self.map_min_span/2, self.map_min_span/2)
            ax.set_ylim(-self.map_min_span/2, self.map_min_span/2)
            ax.set_xticks([]); ax.set_yticks([])
            diagnostic = summary.get("pose_qualification")
            if diagnostic is not None:
                moving, static = diagnostic["moving_ratio"], diagnostic["static_p95"]
                move_text = "waiting for held-out pairs" if moving is None else (
                    f"{100*moving:.1f}% of zero-motion error (need < {100*diagnostic['moving_ratio_limit']:.0f}%)")
                static_text = "waiting for held-out pairs" if static is None else (
                    f"{static:.3f} m p95 (limit {diagnostic['static_p95_limit']:.2f} m)")
                status = "Ready, no supported segment" if diagnostic["ready"] else "Not yet qualified"
                sidebar.text(0, .91, "Pose calibration", fontsize=11, weight="bold")
                sidebar.text(0, .86, f"{status}\nMoving: {move_text}\nStatic: {static_text}",
                             fontsize=8, va="top", linespacing=1.6, wrap=True)
        else:
            if state is not None and "grid" in state and not prior_segment:
                grid = np.asarray(state["grid"]); resolution = float(state.get("resolution", 1.))
                origin = np.asarray(state.get("origin", [0., 0.]), float).reshape(-1)[:2]
                h, w = grid.shape[:2]
                ax.imshow(_map_rgb(grid), origin="lower", interpolation="nearest",
                          extent=[origin[0]-.5*resolution, origin[0]+(w-.5)*resolution,
                                  origin[1]-.5*resolution, origin[1]+(h-.5)*resolution])
            # Only cells from the selected coordinate frame share its origin.
            ledger = None if state is None else state.get("ledger")
            bounds = [poses[available,:2]]
            resolution = float((state or {}).get("resolution", _get(self.config, "vnext_map_resolution", 1.5)))
            if ledger is not None:
                goal_evidence = np.asarray(ledger.get("goal_evidence", []), dtype=float).reshape(-1, 5)
                cells = np.asarray(ledger.get("cells", []), dtype=float).reshape(-1, 2)
                values = np.asarray(ledger.get("values", []), dtype=float).reshape(-1, 6)
                if segmented:
                    segments = np.asarray(ledger.get("segments", [])).reshape(-1)
                    if len(segments) != len(cells):
                        raise ValueError("Cannot draw untagged disconnected map segments")
                    selected = segments == segment
                    if len(goal_evidence) == len(cells):
                        goal_evidence = goal_evidence[selected]
                    cells, values = cells[selected], values[selected]
                if len(cells):
                    xy = cells * resolution
                    bounds.append(xy)
                    patches = [Rectangle((x-.5*resolution, z-.5*resolution), resolution, resolution)
                               for x, z in xy]
                    colors = _map_rgb(values[:, None, :]).reshape(-1, 3)/255.
                    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none"), autolim=True)
                    if len(goal_evidence) == len(cells):
                        for index, color, marker, name in (
                                (0, "#db8d2d", ".", "candidate viewpoint"),
                                (1, "#198544", "o", "model-trusted viewpoint"),
                                (2, "#ab3b42", "x", "rejected viewpoint")):
                            show = goal_evidence[:, index] > .1
                            if show.any():
                                ax.scatter(*xy[show].T, c=color, marker=marker, s=16, zorder=5, label=name)
            # NaN at unavailable steps also prevents a line across any gap.
            drawn = poses.copy()
            drawn[~available,:2] = np.nan
            first, last = poses[available][[0, -1]]
            x, z, yaw = last[:3]

            xy = np.concatenate(bounds, axis=0)
            lower, upper = xy.min(axis=0), xy.max(axis=0)
            center = (lower + upper) / 2.
            data_span = float((upper - lower).max())
            margin = max(resolution, data_span*.08)
            span = max(self.map_min_span, data_span + 2.*margin)
            ax.set_xlim(center[0]-span/2., center[0]+span/2.)
            ax.set_ylim(center[1]-span/2., center[1]+span/2.)
            ax.set_aspect("equal", adjustable="box")
            ax.plot(drawn[:, 0], drawn[:, 1], color="#284d7a", linewidth=1.5, zorder=3,
                    path_effects=[patheffects.withStroke(linewidth=2.8, foreground="#edf4fa")])
            ax.scatter(first[0], first[1], marker="*", s=46, color="#bc5149",
                       edgecolors="white", linewidths=.4, zorder=4)
            arrow = (-math.sin(yaw)*13., math.cos(yaw)*13.)
            ax.annotate("", xy=(x, z), xytext=(-arrow[0], -arrow[1]), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-|>", color="#1765b0", lw=1.5,
                                        shrinkA=0, shrinkB=0), zorder=5)
            sidebar.text(0., .25, f"View width: {span:.1f} m  |  Cell: {resolution:.2g} m",
                         fontsize=8, color="#536477")
            label_prefix = "Local-segment" if segmented else "Episode-relative"
            ax.set_xlabel(label_prefix+" x (estimated metres)")
            ax.set_ylabel(label_prefix+" z (estimated metres)")
            handles = [Line2D([], [], marker="s", linestyle="", color=c, label=t) for c, t in
                       (("#ecedef", "unknown"), ("#c0d3dd", "directional observation proxy"),
                        ("#486987", "visited"), ("#203e5a", "repeat visits"),
                        ("#3cad69", "ground-view heatmap cue" if summary["spatial_fix_version"]>=2 else "target-evidence viewpoint"))]
            goal_handles, _ = ax.get_legend_handles_labels()
            route_handles = [Line2D([], [], color="#284d7a", label="Estimated route"),
                             Line2D([], [], marker="*", linestyle="", color="#bc5149", label="Route start"),
                             Line2D([], [], marker=">", linestyle="", color="#1765b0", label="Last heading")]
            sidebar.legend(handles=route_handles + handles + goal_handles, loc="upper left",
                           fontsize=8, frameon=False, borderaxespad=0, labelspacing=.30)
        if not has_pose:
            tracking_status = "no qualified map pose"
        elif segmented:
            tracking_status = "segment %d (%s)" % (segment, "earlier qualified" if prior_segment else "current")
        else:
            tracking_status = "qualified estimated positions"
        if segmented:
            tracking_status += " | persistent visual history: " + (
                "active" if summary["persistent_memory_available"] else "unavailable")
        extent = summary.get("rendered_segment_extent_m")
        detail_text = "" if extent is None else (
            " | %.2f × %.2f m estimated span" % tuple(extent) if max(extent) > 1.e-6
            else " | no estimated translation in this segment")
        fig.text(.10, .945, "B · Qualified Local Spatial Memory", fontsize=16,
                 fontweight="bold", color="#17324d")
        fig.text(.10, .895, "%s | %d steps | %s%s" %
                 ("Success" if summary["success"] else "Failure", summary["total_steps"],
                  tracking_status, detail_text), fontsize=10, color="#536477")
        ax.set_title(tracking_status.split(" | ")[0], fontsize=10, color="#33475b")
        if summary["spatial_fix_version"]>=3 and state is not None:
            frames = np.asarray(state.get("history_frames",[]))
            valid = np.asarray(state.get("history_valid",[]),bool).reshape(-1)
            if frames.ndim==4 and len(frames)==len(valid) and valid.any():
                selected = frames[np.flatnonzero(valid)[:4],...,:3]
                if selected.dtype!=np.uint8:
                    factor = 255. if selected.max(initial=0)<=1.01 else 1.
                    selected = np.clip(selected*factor,0,255).astype(np.uint8)
                ribbon = np.concatenate(list(selected),axis=1)
                preview = sidebar.inset_axes([0., .02, 1., .16])
                preview.imshow(ribbon, interpolation="nearest")
                preview.set_xticks([])
                preview.set_yticks([])
                preview.set_title("Observed RGB tokens (%d slots)" % summary["history_slots_retained"],
                                  fontsize=8)
        if segmented:
            footer = (f"Segment coverage: {summary['rendered_segment_steps']} decisions | pose breaks: {summary['pose_gap_count']}\n"
                      "Local tracking: %.1f%% | continuous episode-origin tracking: %.1f%%\n"
                      "Qualified local segments have separate origins.\n"
                      "Visual history stays active independently of pose. Target marks denote observation viewpoints." %
                      (100*summary["local_tracking_fraction"], 100*summary["localized_fraction"]))
        else:
            footer = ("Localized observations: %.1f%% (%d unlocalized steps).\n"
                      "Unlocalized movement remains unknown. Target marks denote observation viewpoints." %
                      (100*summary["localized_fraction"], summary["unlocalized_steps"]))
        fig.text(.08, .02, footer, fontsize=8)
        self._save(fig, path, tight=False)

    def _render_timeline(self, snap, path):
        from .paper_figures import timeline
        timeline(self, snap, path)

    def _names(self, count, phase=None):
        try:
            from apm.control import option_names
            names = tuple(option_names(True))
        except ImportError:
            names = ("actor", "left", "right", "forward", "left_forward", "right_forward",
                     "scan", "clear_obstacle", "interact", "settle", "back", "recovery_turn")
        names=list(names)
        if int(_get(self.config,"vnext_spatial_fix_version",0))>=2 and phase==1:
            names[6:8]=["align_up","align_down"]
        return [names[i] if i < len(names) else "option_%d" % i for i in range(count)]

    def _render_goal(self, event, path):
        from .paper_figures import goal
        goal(self, event, path)

    def _render_unavailable_a(self, summary, path):
        fig = self._figure((10, 2.6))
        fig.text(.06, .73, "A · Candidate forecast unavailable", fontsize=16, fontweight="bold", color="#17324d")
        fig.text(.06, .47, summary["candidate_prediction_note"] + ".", fontsize=11)
        fig.text(.06, .25, "This episode had no logged decision with multiple legal, computed forecasts. "
                 "See C for the actions that actually ran.", fontsize=9, color="#536477")
        self._save(fig, path)

    def _render_decision(self, event, path, title):
        from .paper_figures import decision
        decision(self, event, path, "C" if title.startswith("C") else "A")

    def _render_outcomes(self, event, path):
        legal = np.flatnonzero(event["mask"])
        selected = event["row"]["option"]
        from .paper_figures import shortlist
        order = shortlist(event)[:3]
        fig = self._figure((12 if len(order) == 2 else 14, 6))
        grid = fig.add_gridspec(1, 2+len(order), width_ratios=[1.2, .62] + [1]*len(order),
                               left=.05, right=.96, top=.76, bottom=.18, wspace=.22)
        observation = fig.add_subplot(grid[0])
        if event["rgb"] is not None:
            observation.imshow(event["rgb"], interpolation="nearest")
        observation.axis("off")
        observation.set_title("Current observation", color="#33475b")
        names = self._names(len(event["mask"]), event["row"]["phase"])
        labels = ["New surface", "First discovery", "Discovery", "Surface revisit", "Interactable", "Progress", "Blocked"]
        forecast = event.get("goal_forecast")
        if forecast is not None:
            labels += ["Verified discovery", "Reobserve"]
        new_area = event.get("goal_new_area_forecast")
        if new_area is not None:
            labels.append("New map area")
        label_ax = fig.add_subplot(grid[1])
        label_ax.set(xlim=(0, 1), ylim=(len(labels)-.5, -.5))
        label_ax.axis("off")
        for y, label in enumerate(labels):
            label_ax.text(.98, y, label, ha="right", va="center", fontsize=9, color="#33475b")
        for i, option in enumerate(order, start=2):
            ax = fig.add_subplot(grid[i], sharey=label_ax)
            p = event["prediction"][option, :7]
            if forecast is not None:
                p = np.r_[p, forecast[option]]
            if new_area is not None:
                p = np.r_[p, new_area[option]]
            bars = ax.barh(np.arange(len(labels)), p, color="#408b63" if option == selected else "#779bb7", height=.65)
            support_mask = event.get("event_support")
            if support_mask is not None:
                for j, bar in enumerate(bars):
                    if j < support_mask.shape[-1] and not support_mask[option,j]:
                        bar.set_alpha(.42)
                        bar.set_hatch("//")
            ax.set(xlim=(0, 1), xticks=(0, .5, 1))
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=.25)
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.set_title(names[option].replace("_", " ") + ("\nExecuted" if option == selected else "\nAlternative"),
                         fontsize=10, color="#33475b")
            support = event.get("supported")
            ax.set_xlabel("Score support: " + ("not recorded" if support is None else "yes" if support[option] else "no"), fontsize=8)
        fig.text(.05, .92, "A · Predicted consequences", fontsize=17, fontweight="bold", color="#17324d")
        fig.text(.05, .86, f"Decision {event['row']['step']}   ·   Predicted event probabilities on a 0–1 scale",
                 fontsize=11, color="#536477")
        reasons = ", ".join(event.get("reasons", [])[:3]) or "retained decision"
        fig.text(.05, .09, f"Selection: {reasons}", fontsize=9, color="#536477")
        fig.text(.05, .045, "Independent events (probabilities need not sum to one). Hatching: event lacks held-out support (when recorded). Goal-specific rows follow Blocked.",
                 fontsize=9, color="#536477")
        self._save(fig, path)

    def close(self):
        # Flush maps for completed episodes. An unfinished episode has no export.
        if self._map_executor is not None:
            self._map_executor.shutdown(wait=True)
            self._map_jobs.clear()
        self._active.clear()
        self._partial_prefixes.clear()
        self._latest_train = None
        self.closed = True
