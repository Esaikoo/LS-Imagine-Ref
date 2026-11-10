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
                   memory_available=bool(_scalar(policy_output,"log_spatial_memory_available")),
                   goal_state=int(_scalar(policy_output, "log_goal_state", -1)),
                   goal_gate=_scalar(policy_output, "log_goal_gate"),
                   goal_model_update=int(_scalar(policy_output, "log_goal_model_update", -1)),
                   goal_conflict=_scalar(policy_output, "log_goal_conflict"),
                   goal_verify=bool(_scalar(policy_output, "log_goal_verify_used")),
                   goal_reject=bool(_scalar(policy_output, "log_goal_reject_used")),
                   goal_absence=bool(_scalar(policy_output, "log_goal_absence_confirmed")),
                   value=None)
        if len(ep.rows) >= self.max_rows:
            ep.omitted_rows += 1
            return
        previous = ep.rows[-1] if ep.rows else None
        ep.rows.append(row)
        decision = bool(_scalar(policy_output, "log_vnext_decision"))
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
        if not decision and not goal_snapshot:
            return
        prediction = _array(policy_output.get("log_vnext_prediction"))
        mask = _array(policy_output.get("log_vnext_mask"), []).reshape(-1).astype(bool)
        has_prediction = (prediction is not None and prediction.ndim == 2
                          and prediction.shape[1] >= 7 and len(mask) == len(prediction)
                          and np.isfinite(prediction).all() and bool(np.any(prediction)))
        q_computed = has_prediction or bool(np.any(q))
        row["value"] = float(q[option]) if q_computed and 0 <= option < len(q) else None
        legal_values = q[mask] if len(q) == len(mask) else np.array([])
        margin = float(np.diff(np.sort(legal_values))[-1]) if len(legal_values) > 1 else float("inf")
        phase_change = previous is not None and previous["phase"] != phase
        priority = 4*int(row["override"]) + 3*int(phase_change) + int(visible >= .7)
        priority += max(0., .1-margin) if math.isfinite(margin) else 0.
        snap = dict(row=row.copy(), rgb=_rgb(obs.get("image", obs.get("rgb"))),
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
        if goal_snapshot:
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
            ep.goal = snap
        if decision:
            if has_prediction and mask.sum() >= 2:
                ep.a = self._retain(ep.a, snap, 3)
            ep.c = self._retain(ep.c, snap, 2)
            ep.last = snap

    @staticmethod
    def _retain(bucket, snapshot, limit):
        values = [s for s in bucket if s["row"]["step"] != snapshot["row"]["step"]]
        values.append(snapshot)
        return sorted(values, key=lambda s: (-s["priority"], -s["row"]["step"]))[:limit]

    def feedback(self, mode, env_id, reward, info):
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
        # Ensure the final real decision is represented in the C explanations.
        if ep.last is not None and all(s["row"]["step"] != ep.last["row"]["step"] for s in ep.c):
            ep.c = ([ep.c[0]] if ep.c else []) + [ep.last]
        if ep.goal is not None:
            summary.update(goal_ref_version=1,
                first_trusted_target_step=next((r["step"] for r in ep.rows if r["goal_state"] == 2), None),
                goal_verify_steps=sum(r["goal_verify"] for r in ep.rows),
                goal_reject_steps=sum(r["goal_reject"] for r in ep.rows),
                goal_confirmed_absence_steps=sum(r["goal_absence"] for r in ep.rows),
                goal_conflict_mean=float(np.mean([r["goal_conflict"] for r in ep.rows])),
                goal_keyframe=self._decision_json(ep.goal))
        return dict(summary=summary, rows=ep.rows, a=ep.a, c=ep.c, map_state=state,
                    goal=ep.goal, terminal_pose=terminal_pose)

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
                summary = dict(snap["summary"], mode=mode, export_training_step=int(training_step),
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
                    goal_absence_confirmed=np.asarray([r["goal_absence"] for r in rows], bool),
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
            goal_absence_confirmed=snap["row"].get("goal_absence"),
            goal_gate=snap["row"].get("goal_gate"), goal_conflict=snap["row"].get("goal_conflict"),
            goal_model_update=snap["row"].get("goal_model_update"),
            goal_selection_reason=snap.get("goal_selection_reason"),
            goal_rgb_source=snap.get("goal_rgb_source"),
            goal_region_cal_reliability=snap.get("goal_region_cal_reliability"),
            goal_heat_delta_mean_max=snap.get("goal_heat_delta"),
            task_q=snap.get("task_q") if snap["q_computed"] else None,
            search_q=snap.get("search_q") if snap["q_computed"] else None,
            actual_control=snap.get("applied"),
            unexecuted_candidates="predictions only; no real future frames are claimed"))

    @staticmethod
    def _figure(size):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=size, dpi=120)
        FigureCanvasAgg(fig)
        return fig

    @staticmethod
    def _save(fig, path, *, tight=True):
        fig.savefig(path, dpi=120, bbox_inches="tight" if tight else None, facecolor="white")
        fig.clear()

    def _render(self, snap, folder):
        self._render_map(snap, folder / "B_map.png")
        self._render_timeline(snap, folder / "C1_timeline.png")
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
        fig.text(.10, .945, "B · Estimated 2D search memory", fontsize=16,
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
            footer = ("Local tracking: %.1f%% | continuous episode-origin tracking: %.1f%%\n"
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
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        import textwrap
        rows = snap["rows"]; x = np.asarray([r["step"] for r in rows])
        fig = self._figure((14, 7.5))
        axes = fig.subplots(5, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 2, 1.4]})
        fig.subplots_adjust(left=.10, right=.73, top=.88, bottom=.13, hspace=.24)
        fig.suptitle("C · Episode control timeline", x=.10, ha="left", fontsize=16, fontweight="bold")
        phase_colors = ["#729fc1", "#e2b45a", "#67a578", "#c47b75"]
        axes[0].imshow(np.asarray([r["phase"] for r in rows])[None], aspect="auto", interpolation="nearest",
                       cmap=ListedColormap(phase_colors), vmin=0, vmax=3,
                       extent=[-.5, len(rows)-.5, 0, 1])
        axes[0].set_yticks([]); axes[0].set_ylabel("Phase")
        axes[0].legend(handles=[Patch(color=c, label=t) for c, t in zip(phase_colors, PHASES)],
                       loc="center left", bbox_to_anchor=(1.02, .5), ncol=2, fontsize=8, frameon=False)
        options = np.asarray([r["option"] for r in rows])
        option_colors = ["#355c7d", "#729fc1", "#91b6ca", "#a5c78b", "#73986a", "#c2b271",
                         "#a48cb4", "#bc8e86", "#5f9f77", "#8a999d", "#bd9e61", "#bb716e"]
        axes[1].imshow(options[None], aspect="auto", interpolation="nearest", cmap=ListedColormap(option_colors),
                       vmin=0, vmax=11, extent=[-.5, len(rows)-.5, 0, 1])
        axes[1].set_yticks([]); axes[1].set_ylabel("Option ID")
        starts = np.r_[0, np.flatnonzero(np.diff(options)) + 1]
        ends = np.r_[starts[1:], len(options)]
        for a, b in zip(starts, ends):
            if b-a >= max(1, len(rows)//60):
                axes[1].text((a+b-1)/2, .45, str(options[a]), ha="center", va="center", fontsize=6)
        axes[2].imshow(np.asarray([r["source"] for r in rows])[None], aspect="auto", interpolation="nearest",
                       cmap=ListedColormap(["#999999", "#355c7d", "#e2a657", "#7da67d", "#b36a5e", "#8e7ba9", "#6ba5b4", "#c69c68", "#b58bba", "#b87558", "#4aa69d"]),
                       vmin=-1, vmax=9, extent=[-.5, len(rows)-.5, 0, 1])
        axes[2].set_yticks([]); axes[2].set_ylabel("Source")
        source_colors = ["#999999", "#355c7d", "#e2a657", "#7da67d", "#b36a5e", "#8e7ba9", "#6ba5b4", "#c69c68", "#b58bba", "#b87558", "#4aa69d"]
        present = {r["source"] for r in rows}
        axes[2].legend(handles=[Patch(color=c, label=SOURCES[i]) for i, c in zip(range(-1, 10), source_colors)
                                if i in present],
                       loc="center left", bbox_to_anchor=(1.02, .5), ncol=2, fontsize=7, frameon=False)
        axes[3].plot(x, [r["visible"] for r in rows], label="A visibility estimate", color="#46836e")
        axes[3].set_ylabel("Probability")
        axes[3].set_ylim(-.16, 1.10)
        axes[3].set_yticks([0, .5, 1.])
        if rows[0]["goal_state"] >= 0:
            for state, color, name in ((0, "#7d8894", "untrusted"), (1, "#ce902c", "verify"),
                                        (2, "#198544", "trusted")):
                selected = np.asarray([r["goal_state"] == state for r in rows])
                if selected.any():
                    axes[3].scatter(x[selected], np.full(selected.sum(), -.06), s=5, c=color, label=name)
            for key, color, label in (("goal_verify", "#ce902c", "verify action"),
                                       ("goal_reject", "#ab3b42", "reject action")):
                used = np.asarray([r[key] for r in rows])
                if used.any():
                    axes[3].scatter(x[used], np.full(used.sum(), -.11), s=12, c=color, marker="x", label=label)
        values = [np.nan if r["value"] is None else r["value"] for r in rows]
        axes[4].plot(x, values, color="#536891", marker=".", markersize=2)
        axes[4].set_ylabel("Selected C value")
        finite_values = np.asarray(values)[np.isfinite(values)]
        if len(finite_values):
            from matplotlib.ticker import FormatStrFormatter
            lo, hi = finite_values.min(), finite_values.max()
            padding = max(.01, (hi-lo)*.1)
            axes[4].set_ylim(lo-padding, hi+padding)
            axes[4].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        else:
            axes[4].text(.5, .5, "No C value computed for this episode", transform=axes[4].transAxes,
                         ha="center", va="center", color="#536477", fontsize=9)
        hits = [r["step"] for r in rows if r["option"] == 8]
        if hits:
            axes[3].axvline(hits[0], color="#a37932", linestyle="--", label="first interaction option")
        if snap["summary"]["first_predicted_target_step"] is not None:
            axes[3].axvline(snap["summary"]["first_predicted_target_step"], color="#589d70", linestyle=":", label="first predicted target")
        if snap["summary"]["first_target_step"] is not None:
            axes[3].axvline(snap["summary"]["first_target_step"], color="#4f8a91", linestyle="--", label="first ray-audited target")
        if rows[0]["goal_state"] >= 0:
            reward_steps = np.asarray([r["reward"] > 0 for r in rows])
            if reward_steps.any():
                axes[3].scatter(x[reward_steps], np.full(reward_steps.sum(), 1.02),
                                s=12, marker="^", c="#478858", label="actual positive reward")
        axes[3].axvline(x[-1], color="#3b895b" if snap["summary"]["success"] else "#ad5e56",
                       linewidth=1.5, label="success" if snap["summary"]["success"] else "failure")
        axes[4].set_xlabel("Decision index (0-based; reward follows the selected action)")
        axes[3].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
        names = self._names(12)
        legend = "Option IDs: " + "; ".join("%d %s" % (i, name) for i, name in enumerate(names))
        if int(_get(self.config,"vnext_spatial_fix_version",0))>=2:
            legend += "; APPROACH phase: 6=align_up, 7=align_down (not scan/clear)"
        fig.text(.04, .015, textwrap.fill(legend, width=160), fontsize=7)
        for ax in axes[3:]:
            ax.grid(alpha=.2)
        self._save(fig, path)

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

    @staticmethod
    def _decision_images(fig, panel, event):
        patch = event.get("patch")
        if patch is not None and patch.ndim == 3 and patch.shape[-1] >= 6:
            patch = _map_rgb(patch.astype(float)/255. if patch.dtype == np.uint8 else patch)[::-1]
        else:
            patch = None
        row = event["row"]
        b_title = "B · Local search patch"
        b_note = None
        if not row["local_ready"]:
            patch = event.get("history_preview")
            b_title = "B · Retained visual history"
            b_note = ("Pose calibration pending" if row.get("pose_model_ready") is False
                      else "Local pose unavailable")
        tiles = [("Current observation", event.get("rgb"), None),
                 ("Original U-Net", _goal_heatmap(event.get("heatmap")), "turbo"),
                 (b_title, patch, None)]
        if event["row"]["goal_state"] >= 0:
            tiles += [("D · Region evidence", _goal_heatmap(event.get("goal_ref_map")), "turbo"),
                      ("Effective heatmap", _goal_heatmap(event.get("goal_heatmap_policy")), "turbo")]
        grid = panel.subgridspec(1, len(tiles), wspace=.22)
        for i, (title, image, cmap) in enumerate(tiles):
            ax = fig.add_subplot(grid[i])
            ax.axis("off")
            ax.set_title(title, fontsize=10, color="#33475b", pad=8)
            if image is None:
                ax.text(.5, .5, b_note if title == b_title and b_note else "Not recorded",
                        transform=ax.transAxes, ha="center", va="center", color="#657487", fontsize=9)
            else:
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                if title == b_title and b_note:
                    ax.text(.5, -.14, b_note + " · RGB tokens", transform=ax.transAxes,
                            ha="center", va="top", color="#657487", fontsize=8)

    def _render_goal(self, event, path):
        from matplotlib.patches import Rectangle
        fig = self._figure((13, 6))
        refs = getattr(self, "goal_references", ())
        axes = fig.subplots(2, 4)
        fig.subplots_adjust(top=.83, bottom=.17, left=.035, right=.93, hspace=.33, wspace=.22)
        for ax in axes.flat:
            ax.axis("off")
        for i, ax in enumerate(axes[0]):
            label, rgb = refs[i] if i < len(refs) else ("Bank example", None)
            ax.set_title(label, fontsize=9)
            if rgb is not None:
                ax.imshow(rgb, interpolation="nearest")
            else:
                ax.text(.5, .5, "Not available", transform=ax.transAxes,
                        ha="center", va="center", color="#657487")

        rgb = event.get("goal_rgb")
        context = event.get("goal_context")
        context = context.reshape(-1) if context is not None else np.empty(0)
        ax = axes[1,0]
        candidate_notes = []
        if rgb is not None:
            ax.imshow(rgb, extent=(0, 1, 1, 0), interpolation="nearest")
            ax.set(xlim=(0, 1), ylim=(1, 0))
            boxes = event.get("goal_boxes")
            scores = event.get("goal_candidate_scores")
            sources = event.get("goal_candidate_sources")
            if boxes is not None and scores is not None and sources is not None:
                for index, (box, score, source) in enumerate(zip(boxes, scores, sources), 1):
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        continue
                    color = ("#e9a435", "#40bad4", "#d670bc")[int(source) % 3]
                    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                           ec=color, linewidth=1.2))
                    ax.text((x0+x1)/2, (y0+y1)/2, str(index), color=color, fontsize=7,
                            ha="center", va="center", clip_on=True,
                            bbox=dict(facecolor="black", edgecolor="none", alpha=.65, pad=.30))
                    candidate_notes.append(f"{index}  {score:.2f}")
            if len(context) >= 8 and np.isfinite(context[3:5]).all():
                x, y = (context[3:5] + 1.) / 2.
                ax.plot(x, y, marker="+", color="white", markersize=7)
        size = "" if rgb is None else " %dx%d" % (rgb.shape[1], rgb.shape[0])
        ax.set_title(event.get("goal_rgb_source", "D frame") + size + " / region scores")
        heat_artist = None
        for ax, key, title in zip(axes[1,1:], ("goal_original_heatmap", "goal_ref_map", "goal_heatmap_policy"),
                                  ("Original U-Net", "D region scores", "Effective heatmap")):
            image = _goal_heatmap(event.get(key))
            if image is not None:
                heat_artist = ax.imshow(image, cmap="turbo", vmin=0, vmax=1,
                                        interpolation="nearest")
            else:
                ax.text(.5, .5, "Not recorded", transform=ax.transAxes,
                        ha="center", va="center", color="#657487")
            ax.set_title(title)
        if heat_artist is not None:
            color_ax = fig.add_axes([.94, .18, .012, .25])
            fig.colorbar(heat_artist, cax=color_ax, ticks=(0., .5, 1.))
        row = event["row"]
        evidence = ("valid" if context[7] > .5 else "unqualified") if len(context) >= 8 else "unknown"
        cal = event.get("goal_region_cal_reliability", np.nan)
        cal_text = f"{cal:.3f}" if np.isfinite(cal) else "n/a"
        title = (f"D: frame {row['step']} | {('untrusted', 'uncertain', 'trusted')[row['goal_state']]}"
                 f" | evidence {evidence} | cal {cal_text} | gate {row['goal_gate']:.3f}"
                 f" | conflict {row['goal_conflict']:.3f}")
        if row.get("goal_absence"):
            title += " | absence qualified"
        delta = event.get("goal_heat_delta")
        change = "n/a" if delta is None else f"{delta[0]:.4f} / {delta[1]:.4f}"
        reason = event.get("goal_selection_reason", "unspecified")
        fig.suptitle(title + f"\nSelected: {reason} | heat change mean/max {change}", fontsize=12)
        update = row["goal_model_update"]
        update = str(update) if update >= 0 else "n/a"
        fig.text(.035, .066, "Amber U-Net, cyan RGB, magenta difficult regions. White + marks the selected "
                 f"region center. Model update {update}.", fontsize=9, color="#536477")
        if candidate_notes:
            fig.text(.035, .036, "Proposal scores   " + "   ·   ".join(candidate_notes),
                     fontsize=9, color="#536477")
        self._save(fig, path)

    def _render_unavailable_a(self, summary, path):
        fig = self._figure((10, 2.6))
        fig.text(.06, .73, "A · Candidate forecast unavailable", fontsize=16, fontweight="bold", color="#17324d")
        fig.text(.06, .47, summary["candidate_prediction_note"] + ".", fontsize=11)
        fig.text(.06, .25, "This episode had no logged decision with multiple legal, computed forecasts. "
                 "See C for the actions that actually ran.", fontsize=9, color="#536477")
        self._save(fig, path)

    def _render_decision(self, event, path, title):
        fig = self._figure((15, 8.8))
        panel = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.8), left=.05, right=.95,
                                 bottom=.12, top=.82, hspace=.20)
        self._decision_images(fig, panel[0], event)
        ax = fig.add_subplot(panel[1])
        ax.axis("off")
        count = max(len(event["mask"]), len(event["q"]))
        names = self._names(count, event["row"]["phase"])
        selected = event["row"]["option"]
        row = event["row"]
        goal = row["goal_state"] >= 0
        columns = ["Option", "Legal", "New surface", "Discovery", "Revisit", "Progress", "A score", "C value"]
        if goal:
            columns += ["Task Q", "Search Q", "Verified\ndiscovery", "Reobserve"]

        def display(value):
            return "—" if value is None or not np.isfinite(value) else f"{value:.3f}"

        data = []
        for i, name in enumerate(names):
            legal = i < len(event["mask"]) and event["mask"][i]
            predicted = legal and event["has_prediction"] and i < len(event["prediction"])
            evaluated = legal and event["q_computed"] and i < len(event["q"])
            probs = event["prediction"][i, [0, 2, 3, 5]] if predicted else [None]*4
            score, support = event["a_score"], event["supported"]
            a_score = display(score[i] if predicted and score is not None and i < len(score) else None)
            if a_score != "—" and support is not None and i < len(support) and support[i]:
                a_score += " *"
            values = [("> " if i == selected else "") + name.replace("_", " "),
                      "Yes" if legal else "No", *map(display, probs), a_score,
                      display(event["q"][i] if evaluated else None)]
            if goal:
                for key in ("task_q", "search_q"):
                    q = event.get(key)
                    values.append(display(q[i] if evaluated and q is not None and i < len(q) else None))
                q = event.get("goal_forecast")
                values.extend(map(display, q[i] if predicted and q is not None and i < len(q) else (None, None)))
            data.append(values)
        if data:
            table = ax.table(cellText=data, colLabels=columns, cellLoc="center", bbox=(0, 0, 1, 1),
                             colWidths=[.18, .06] + [.76/(len(columns)-2)]*(len(columns)-2))
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor("#dce3ea")
                cell.set_linewidth(.6)
                cell.set_facecolor("#e6f3ea" if r == selected+1 else "#f6f8fa" if r % 2 == 0 else "white")
                if c == 0:
                    cell.set_text_props(ha="left")
                if r == 0:
                    cell.set_facecolor("#233f59")
                    cell.set_height(cell.get_height()*1.45)
                    cell.set_text_props(color="white", weight="bold")
        else:
            ax.text(.5, .5, "Candidate table not recorded", ha="center", va="center", fontsize=11)
        phase = PHASES[min(3, max(0, row["phase"]))]
        source = SOURCES.get(row["source"], "unspecified")
        fig.text(.05, .94, title, fontsize=17, color="#17324d", fontweight="bold")
        fig.text(.05, .89, f"Decision {row['step']}   ·   {phase}   ·   Actor command {row['actor']}   ·   {source}",
                 fontsize=11, color="#536477")
        notes = ["Green highlights the executed option. * marks a supported A score. "
                 "A dash means unavailable or illegal. Heatmaps use 0–1."]
        if goal:
            notes.append("D state %d  ·  gate %.3f  ·  conflict %.3f  ·  verify %d  ·  reject %d" %
                         (row["goal_state"], row["goal_gate"], row["goal_conflict"], row["goal_verify"], row["goal_reject"]))
            if event.get("applied") is not None:
                controls = ", ".join(f"{float(value):g}" for value in event["applied"])
                notes.append("Applied [attack, forward, back, left, right, jump, use, pitch, yaw] = " + controls)
        fig.text(.05, .035, "\n".join(notes), fontsize=9, color="#536477", linespacing=1.5)
        self._save(fig, path)

    def _render_outcomes(self, event, path):
        legal = np.flatnonzero(event["mask"])
        selected = event["row"]["option"]
        order = ([selected] if selected in legal else []) + [i for i in legal if i != selected]
        order = order[:3]
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
            ax.barh(np.arange(len(labels)), p, color="#408b63" if option == selected else "#779bb7", height=.65)
            ax.set(xlim=(0, 1), xticks=(0, .5, 1))
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=.25)
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.set_title(names[option].replace("_", " ") + ("\nExecuted" if option == selected else "\nAlternative"),
                         fontsize=10, color="#33475b")
        fig.text(.05, .92, "A · Predicted consequences", fontsize=17, fontweight="bold", color="#17324d")
        fig.text(.05, .86, f"Decision {event['row']['step']}   ·   Predicted event probabilities on a 0–1 scale",
                 fontsize=11, color="#536477")
        fig.text(.05, .055, "Only the executed action has an observed outcome. All bars show predictions.",
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
