"""Bounded, inference-independent exports of real search decisions.

The collector never changes observations, rewards, RNGs, model state or actions.
Only a few RGB/heatmap/patch snapshots are retained. Rendering is synchronous at
selected completed episodes, using Agg figures without changing pyplot state.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np


PHASES = ("SEARCH", "APPROACH", "INTERACT", "RECOVERY")
SOURCES = {-1: "unspecified", 0: "actor", 1: "learned", 2: "rule interaction",
           3: "rule recovery", 4: "settle", 5: "training probe"}


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
    reward: float = 0.
    success: bool = False
    omitted_rows: int = 0
    pose_steps: int = 0
    localized_steps: int = 0


class SearchVisualizer:
    """Collect one environment at a time; methods also accept numpy policy logs.

    observe runs BEFORE env.step and its RGB therefore matches the outgoing
    decision. feedback runs AFTER the corresponding real step. Episode metadata
    may contain privileged audit labels, consumed only after the episode ends.
    """
    def __init__(self, root, config):
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
        self._active = {}
        self._partial_prefixes = set()
        self._latest_train = None
        self._last_exported_train_id = None
        self._next_train = self.interval
        self.closed = False
        self.stats = dict(exports=0, errors=0, skipped_no_episode=0, skipped_duplicate=0,
                          incomplete_resets=0, skipped_partial_prefix=0)
        self.last_error = None

    def set_step(self, step):
        """Start/resume at a real training step; never fabricate past exports."""
        self._next_train = (max(0, int(step)) // self.interval + 1) * self.interval

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
        ep.pose_steps += 1
        ep.localized_steps += int(pose_ready)
        phase = int(_scalar(policy_output, "log_vnext_phase"))
        row = dict(step=index, phase=phase, option=option,
                   actor=int(_scalar(policy_output, "log_vnext_actor_action")),
                   source=int(_scalar(policy_output, "log_vnext_source", -1)),
                   override=bool(_scalar(policy_output, "log_vnext_override")),
                   visible=visible, pose=pose.tolist(), pose_ready=pose_ready, reward=0., success=False,
                   value=None)
        if len(ep.rows) >= self.max_rows:
            ep.omitted_rows += 1
            return
        previous = ep.rows[-1] if ep.rows else None
        ep.rows.append(row)
        decision = bool(_scalar(policy_output, "log_vnext_decision"))
        if not decision:
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
        # Copy retained arrays only at actual decision points, not every frame.
        for field_name in ("heatmap", "patch", "a_score", "supported"):
            if snap[field_name] is not None:
                snap[field_name] = snap[field_name].copy()
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
        if not self.enabled or ep is None or not ep.rows:
            return None
        if mode == "eval" and (eval_index is None or eval_total is None):
            raise ValueError("Evaluation visualization needs 1-based index and total")
        if mode == "eval" and int(eval_index) != int(eval_total):
            return None
        snap = self._complete(ep, episode, env_id, training_step, map_state)
        if mode == "train":
            # A threshold strictly before this completion uses the PREVIOUS
            # complete episode; a completion exactly at the threshold may use it.
            self.maybe_export_train(int(training_step)-1)
            self._latest_train = snap
            return self.maybe_export_train(training_step)
        path = self.root / "eval" / ("step_%09d" % training_step) / ("episode_%02d" % eval_index)
        return self._export(snap, path, training_step, "eval")

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
        known = len(visible) == len(valid) and len(visible) > 0
        hits = np.flatnonzero(valid & (visible > .5)) if known else np.array([], int)
        predicted = [r["step"] for r in ep.rows if r["visible"] >= .7]
        positions = np.asarray([r["pose"][:2] for r in ep.rows], dtype=float)
        terminal_pose = None if map_state is None else _array(map_state.get("pose"))
        if terminal_pose is not None:
            terminal_pose = terminal_pose.reshape(-1)
            if len(terminal_pose) < 4 or not np.isfinite(terminal_pose[:4]).all():
                raise ValueError("Terminal visualization pose must be finite [x,z,yaw,uncertainty]")
            terminal_pose = terminal_pose[:4].copy()
            positions = np.concatenate((positions, terminal_pose[None, :2]), axis=0)
        trajectory = float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum())
        localized_fraction = ep.localized_steps/max(1, ep.pose_steps)
        terminal_tracking_ready = (bool(_scalar(map_state, "tracking_ready")) if map_state is not None
                                   and "tracking_ready" in map_state else bool(ep.rows[-1]["pose_ready"]))
        fully_localized = ep.localized_steps == ep.pose_steps and terminal_tracking_ready
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
        visited_cells = (int((ledger_values[:, 0] > 0).sum()) if has_ledger else
                         int((grid[..., 0] > 0).sum()) if grid is not None else None)
        revisits = (float(ledger_values[:, 3].sum()) if has_ledger else
                    float(grid[..., 3].sum()) if grid is not None else None)
        summary = dict(episode_id=str(env_id), episode_completed_at_training_step=int(training_step),
            success=bool(success.any()) if success.size else ep.success,
            total_steps=max(0, len(reward)-1) if reward.size else len(ep.rows)+ep.omitted_rows,
            total_reward=float(reward.sum()) if reward.size else ep.reward,
            estimated_trajectory_length=trajectory if fully_localized else None,
            integrated_path_length=trajectory,
            trajectory_length_valid=fully_localized,
            localized_fraction=localized_fraction,
            unlocalized_steps=ep.pose_steps-ep.localized_steps,
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
        return dict(summary=summary, rows=ep.rows, a=ep.a, c=ep.c, map_state=state,
                    terminal_pose=terminal_pose)

    def _export(self, snap, path, training_step, mode, repeated=False):
        if (path / "summary.json").exists():
            self.stats["skipped_duplicate"] += 1
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".render-", dir=path.parent) as tmp:
                tmp = Path(tmp)
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
                    phase=np.asarray([r["phase"] for r in rows], np.int16),
                    option=np.asarray([r["option"] for r in rows], np.int16),
                    source=np.asarray([r["source"] for r in rows], np.int16),
                    visible=np.asarray([r["visible"] for r in rows], np.float32),
                    reward=np.asarray([r["reward"] for r in rows], np.float32),
                    success=np.asarray([r["success"] for r in rows], bool),
                    terminal_pose=(np.empty((0,), np.float32) if snap["terminal_pose"] is None
                                   else snap["terminal_pose"].astype(np.float32)),
                    map_grid=np.asarray(map_state.get("grid", np.empty((0, 0, 6))), np.float32),
                    map_origin=np.asarray(map_state.get("origin", []), np.float32),
                    map_resolution=np.asarray(map_state.get("resolution", 0.), np.float32),
                    ledger_cells=np.asarray(ledger.get("cells", []), np.int64).reshape(-1, 2),
                    ledger_values=np.asarray(ledger.get("values", []), np.float32).reshape(-1, 6),
                    ledger_directions=np.asarray(ledger.get("directions", []), np.float32).reshape(-1, 12))
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
            unexecuted_candidates="predictions only; no real future frames are claimed"))

    @staticmethod
    def _figure(size):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=size, dpi=120)
        FigureCanvasAgg(fig)
        return fig

    @staticmethod
    def _save(fig, path):
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        fig.clear()

    def _render(self, snap, folder):
        self._render_map(snap, folder / "B_map.png")
        self._render_timeline(snap, folder / "C1_timeline.png")
        for i, event in enumerate(snap["a"][:3], 1):
            self._render_decision(event, folder / ("A1_%02d.png" % i), "A: candidate outcome predictions")
        for i, event in enumerate(snap["a"][:2], 1):
            self._render_outcomes(event, folder / ("A2_%02d.png" % i))
        for i, event in enumerate(snap["c"][:2], 1):
            self._render_decision(event, folder / ("C2_%02d.png" % i), "C: actual high-level decision")

    def _render_map(self, snap, path):
        from matplotlib.lines import Line2D
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        fig = self._figure((9, 7)); ax = fig.add_subplot(111)
        ax.set_facecolor("#ecedef")
        state = snap["map_state"]
        if state is not None and "grid" in state:
            grid = np.asarray(state["grid"]); resolution = float(state.get("resolution", 1.))
            origin = np.asarray(state.get("origin", [0., 0.]), float).reshape(-1)[:2]
            h, w = grid.shape[:2]
            ax.imshow(_map_rgb(grid), origin="lower", interpolation="nearest",
                      extent=[origin[0]-.5*resolution, origin[0]+(w-.5)*resolution,
                              origin[1]-.5*resolution, origin[1]+(h-.5)*resolution])
        else:
            ax.text(.03, .96, "Full map unavailable: estimated trajectory only", transform=ax.transAxes, va="top")
        # A sparse ledger retains visited cells outside the rolling window.
        # Rectangles in metric coordinates avoid allocating a giant dense raster
        # when a poor odometry estimate spans a large range.
        ledger = None if state is None else state.get("ledger")
        if ledger is not None:
            cells = np.asarray(ledger.get("cells", []), dtype=float).reshape(-1, 2)
            values = np.asarray(ledger.get("values", []), dtype=float).reshape(-1, 6)
            if len(cells):
                resolution = float(state.get("resolution", 1.))
                patches = [Rectangle((c[0]*resolution-.5*resolution, c[1]*resolution-.5*resolution),
                                     resolution, resolution) for c in cells]
                colors = _map_rgb(values[:, None, :]).reshape(-1, 3)/255.
                ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none"), autolim=True)
                ax.autoscale_view()
        poses = np.asarray([r["pose"] for r in snap["rows"]], float)
        if snap["terminal_pose"] is not None:
            poses = np.concatenate((poses, snap["terminal_pose"][None]), axis=0)
        ax.plot(poses[:, 0], poses[:, 1], color="#284d7a", linewidth=1.4)
        ax.scatter([0.], [0.], marker="*", s=100, color="#bc5149", zorder=4)
        scale = max(.5, float(np.ptp(poses[:, :2], axis=0).max())*.04)
        x, z, yaw = poses[-1, :3]
        ax.arrow(x, z, -math.sin(yaw)*scale, math.cos(yaw)*scale,
                 width=scale*.12, head_width=scale*.6, color="#1765b0", zorder=5)
        summary = snap["summary"]
        tracking_status = "pose qualified" if summary["terminal_tracking_ready"] else "UNLOCALIZED: translation disabled"
        ax.set_title("B: estimated 2D search memory | %s | %d steps\n%s" %
                     ("success" if summary["success"] else "failure", summary["total_steps"], tracking_status))
        ax.set_xlabel("Episode-relative x (estimated metres)"); ax.set_ylabel("Episode-relative z (estimated metres)")
        ax.set_aspect("equal", adjustable="datalim")
        handles = [Line2D([], [], marker="s", linestyle="", color=c, label=t) for c, t in
                   (("#ecedef", "unknown"), ("#c0d3dd", "directional observation proxy"),
                    ("#486987", "visited"), ("#203e5a", "repeat visits"),
                    ("#3cad69", "target-evidence viewpoint"))]
        ax.legend(handles=handles, loc="upper right", fontsize=7)
        fig.subplots_adjust(bottom=.21)
        localization_note = ("Unlocalized translation is NOT integrated: a point at the origin does NOT mean the agent stayed still."
                             if summary["unlocalized_steps"] or not summary["terminal_tracking_ready"] else
                             "Qualified predicted pose; trajectory remains an estimate.")
        scope_note = ("Episode visited-cell ledger plus rolling directional-view proxy; no true terrain is shown."
                      if summary["full_episode_ledger_available"] else
                      "Retained rolling map; path outside it does not imply retained coverage.")
        footer = (localization_note + "\n" +
                  "Localized observations: %.1f%% (%d unlocalized steps). Target marks are viewpoints, not block positions.\n" %
                  (100*summary["localized_fraction"], summary["unlocalized_steps"]) + scope_note)
        fig.text(.08, .02, footer, fontsize=8)
        self._save(fig, path)

    def _render_timeline(self, snap, path):
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        import textwrap
        rows = snap["rows"]; x = np.asarray([r["step"] for r in rows])
        fig = self._figure((13, 6)); axes = fig.subplots(4, 1, sharex=True,
                                                      gridspec_kw={"height_ratios": [1, 1, 1, 2]})
        phase_colors = ["#729fc1", "#e2b45a", "#67a578", "#c47b75"]
        axes[0].imshow(np.asarray([r["phase"] for r in rows])[None], aspect="auto", interpolation="nearest",
                       cmap=ListedColormap(phase_colors), vmin=0, vmax=3,
                       extent=[-.5, len(rows)-.5, 0, 1])
        axes[0].set_yticks([]); axes[0].set_ylabel("Phase")
        axes[0].set_title("C: real episode phases, options and control sources")
        axes[0].legend(handles=[Patch(color=c, label=t) for c, t in zip(phase_colors, PHASES)],
                       loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=4, fontsize=6)
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
                       cmap=ListedColormap(["#999999", "#355c7d", "#e2a657", "#7da67d", "#b36a5e", "#8e7ba9", "#6ba5b4"]),
                       vmin=-1, vmax=5, extent=[-.5, len(rows)-.5, 0, 1])
        axes[2].set_yticks([]); axes[2].set_ylabel("Source")
        source_colors = ["#999999", "#355c7d", "#e2a657", "#7da67d", "#b36a5e", "#8e7ba9", "#6ba5b4"]
        axes[2].legend(handles=[Patch(color=c, label=SOURCES[i]) for i, c in zip(range(-1, 6), source_colors)],
                       loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=7, fontsize=6)
        axes[3].plot(x, [r["visible"] for r in rows], label="current target probability", color="#46836e")
        values = [np.nan if r["value"] is None else r["value"] for r in rows]
        axes[3].plot(x, values, label="selected C value (not measured success rate)", color="#536891", alpha=.8)
        hits = [r["step"] for r in rows if r["option"] == 8]
        if hits:
            axes[3].axvline(hits[0], color="#a37932", linestyle="--", label="first interaction option")
        if snap["summary"]["first_predicted_target_step"] is not None:
            axes[3].axvline(snap["summary"]["first_predicted_target_step"], color="#589d70", linestyle=":", label="first predicted target")
        if snap["summary"]["first_target_step"] is not None:
            axes[3].axvline(snap["summary"]["first_target_step"], color="#4f8a91", linestyle="--", label="first ray-audited target")
        axes[3].axvline(x[-1], color="#3b895b" if snap["summary"]["success"] else "#ad5e56",
                       linewidth=1.5, label="success" if snap["summary"]["success"] else "failure")
        axes[3].set_xlabel("Decision index (0-based; reward/outcome follows its action)")
        axes[3].legend(fontsize=6, loc="upper left", ncol=2)
        names = self._names(12)
        legend = "Option IDs: " + "; ".join("%d %s" % (i, name) for i, name in enumerate(names))
        fig.text(.04, .015, textwrap.fill(legend, width=160), fontsize=7)
        axes[3].grid(alpha=.2); fig.tight_layout(rect=[0, .08, 1, 1])
        self._save(fig, path)

    def _names(self, count):
        try:
            from apm.control import option_names
            names = tuple(option_names(True))
        except ImportError:
            names = ("actor", "left", "right", "forward", "left_forward", "right_forward",
                     "scan", "clear_obstacle", "interact", "settle", "back", "recovery_turn")
        return [names[i] if i < len(names) else "option_%d" % i for i in range(count)]

    def _images(self, fig, event):
        axes = [fig.add_subplot(2, 3, i) for i in (1, 2, 3)]
        if event["rgb"] is not None: axes[0].imshow(event["rgb"], interpolation="nearest")
        axes[0].set_title("Actual current RGB")
        h = event["heatmap"]
        if h is not None:
            h = h.squeeze().astype(float)
            if h.max(initial=0) > 1.01: h = h/255.
            axes[1].imshow(h, cmap="turbo", vmin=0, vmax=1)
        axes[1].set_title("Original U-Net heatmap")
        patch = event["patch"]
        if patch is not None and patch.ndim == 3 and patch.shape[-1] >= 6:
            axes[2].imshow(_map_rgb(patch.astype(float)/255. if patch.dtype == np.uint8 else patch), origin="lower", interpolation="nearest")
        axes[2].set_title("B: predicted local search patch")
        for ax in axes: ax.axis("off")

    def _render_decision(self, event, path, title):
        fig = self._figure((13, 8)); self._images(fig, event)
        ax = fig.add_subplot(2, 1, 2); ax.axis("off")
        count = max(len(event["mask"]), len(event["q"]))
        names = self._names(count); selected = event["row"]["option"]
        data = []
        for i, name in enumerate(names):
            predicted = event["has_prediction"] and i < len(event["prediction"])
            probs = event["prediction"][i, [0, 2, 3, 5]] if predicted else [None]*4
            score = event["a_score"]
            a_score = None if score is None or i >= np.asarray(score).size else np.asarray(score).reshape(-1)[i]
            value = event["q"][i] if event["q_computed"] and i < len(event["q"]) else None
            fmt = lambda v: "--" if v is None else "%.3f" % v
            data.append([("> " if i == selected else "")+name,
                         "yes" if i < len(event["mask"]) and event["mask"][i] else "no",
                         *[fmt(v) for v in probs], fmt(a_score), fmt(value)])
        table = ax.table(cellText=data, colLabels=["Option", "Legal", "p_new", "p_target", "p_revisit", "p_progress", "A score", "C value"],
                         cellLoc="center", colWidths=[.23, .07, .10, .10, .10, .10, .10, .10], loc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
        if 0 <= selected < len(data):
            for col in range(8): table[(selected+1, col)].set_facecolor("#dbeadd")
        row = event["row"]
        fig.suptitle("%s | step %d | %s | Actor command %d | source: %s" %
                     (title, row["step"], PHASES[min(3, max(0, row["phase"]))], row["actor"], SOURCES.get(row["source"], "unspecified")), fontsize=11)
        fig.text(.07, .01, "p_target = target-discovery event. A score and C value are separate. -- means not computed/logged.", fontsize=8)
        self._save(fig, path)

    def _render_outcomes(self, event, path):
        fig = self._figure((13, 4)); axes = fig.subplots(1, 4)
        if event["rgb"] is not None: axes[0].imshow(event["rgb"], interpolation="nearest")
        axes[0].axis("off"); axes[0].set_title("Actual current RGB")
        legal = np.flatnonzero(event["mask"])
        selected = event["row"]["option"]
        order = ([selected] if selected in legal else []) + [i for i in legal if i != selected]
        names = self._names(len(event["mask"]))
        for ax, option in zip(axes[1:], order[:3]):
            p = event["prediction"][option, [0, 2, 3, 5]]
            ax.barh(["new", "target", "revisit", "progress"], p,
                    color="#6a9a76" if option == selected else "#8ca8bf")
            ax.set_xlim(0, 1); ax.invert_yaxis()
            ax.set_title(names[option]+("\nselected" if option == selected else "\nunexecuted prediction"), fontsize=9)
        for ax in axes[1+min(3, len(order)):]: ax.axis("off")
        fig.suptitle("A: structured predicted consequences | step %d" % event["row"]["step"])
        fig.text(.1, .015, "These are event predictions, not generated RGB frames or observed alternative futures.", fontsize=9)
        fig.tight_layout(rect=[0, .06, 1, .9]); self._save(fig, path)

    def close(self):
        # Synchronous rendering has no pending jobs. Do not export an incomplete
        # episode or synthesize a final training threshold during shutdown.
        self._active.clear()
        self._partial_prefixes.clear()
        self._latest_train = None
        self.closed = True
