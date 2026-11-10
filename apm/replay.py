"""Causal spatial replay windows without reconstructing long RSSM histories."""
import hashlib
import numpy as np

from .evidence import SPATIAL_LABEL_SHAPES


def identity(value):
    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "little") % (2**31-1)


def _actual_control_valid(episode, index):
    """Validate an incoming real transition before missing fields are filled.

    Older collected episodes have actual controls but no explicit validity
    marker. Zero controls are legitimate no-ops; missing/None/nonfinite controls
    are unknown. A present marker is authoritative, including unknown entries.
    This must inspect the source episode, never a schema-padded replay row.
    """
    controls = episode.get("log_applied_control")
    if index <= 0 or controls is None or controls[index] is None:
        return False
    value = np.asarray(controls[index])
    if value.shape != (9,):
        raise ValueError("Spatial replay actual control must have nine components")
    if not np.isfinite(value).all():
        return False
    for name in ("log_actuator_before", "log_actuator_after"):
        states = episode.get(name)
        if states is not None:
            state = states[index]
            if state is None or np.shape(state) != (3,) or not np.isfinite(state).all():
                return False
    if "log_actuator_valid" in episode:
        marker = episode["log_actuator_valid"][index]
        if marker is None or not np.isfinite(marker) or not float(marker) > .5:
            return False
    first = episode.get("is_first")
    if first is None or first[index] is None or bool(first[index]):
        return False
    for name, row in (("log_env_error", index), ("log_env_error", index-1),
                      ("is_last", index-1), ("is_terminal", index-1)):
        values = episode.get(name)
        if values is not None:
            marker = values[row]
            if marker is None or not np.isfinite(marker) or bool(marker):
                return False
    return True


def sample_spatial_episodes(episodes, length, seed=0, patch_size=21, history_slots=16, goal_enabled=False, goal_use_raw_image=False):
    """Preserve original LS episode/window sampling; align outgoing map logs.

    Cache row i contains observation i and logs of the decision at i-1. Thus
    observation i reads map logs from row i+1, which were computed BEFORE that
    action was executed. A terminal observation has no such decision/map.
    Pose image pairs use their real predecessor even at synthetic window starts.
    """
    from .control import FIELDS, GOAL_FIELDS
    rng = np.random.RandomState(seed)
    templates = {}
    shapes = dict(FIELDS, **SPATIAL_LABEL_SHAPES)
    shapes.update(log_spatial_patch=(patch_size, patch_size, 6), log_spatial_pose=(4,),
        log_spatial_summary=(12,), log_spatial_directions=(12,), log_spatial_version=(),
        log_spatial_novelty=(), log_spatial_revisit=(), log_spatial_known_cells=(),
        log_spatial_visited_cells=(), log_spatial_motion_accepted=(), log_spatial_raw_motion=(),
        log_spatial_rejected_motion=(), log_spatial_pair_valid=(), log_spatial_motion_probability=(),
        log_spatial_trajectory_valid=(), log_spatial_map_write=(), log_spatial_ground_view=(),
        log_spatial_cue_valid=(), log_spatial_raw_cue=(), log_spatial_motion_unknown=(),
        log_spatial_pose_gap_count=(), log_spatial_first_gap_step=(),
        log_spatial_camera_pitch=(), log_spatial_pitch_known=(),
        log_spatial_history_frames=(history_slots,4,4,4),
        log_spatial_history_meta=(history_slots,8), log_spatial_history_valid=(history_slots,),
        log_spatial_history_tick=(), log_spatial_segment=(),
        log_spatial_local_valid=(), log_spatial_memory_available=(), log_spatial_visual_revisit=(),
        log_actuator_before=(3,), log_actuator_after=(3,),
        log_applied_control=(9,), log_env_error=(), log_actuator_valid=(),
        log_search_override=(), log_search_decision=(), log_vnext_world_group=())
    goal_dtypes = {}
    discovery_cache = {}
    if goal_enabled:
        from .evidence import GOAL_LABEL_SHAPES, GOAL_LABEL_DTYPES
        from .outcome import goal_discovery_labels
        shapes.update(GOAL_LABEL_SHAPES)
        goal_dtypes.update(GOAL_LABEL_DTYPES)
        shapes.update(
            log_goal_version=(), log_goal_model_update=(), log_goal_context=(8,), log_goal_state=(), log_goal_gate=(),
            log_goal_conflict=(), log_goal_peak_rejected=(), log_goal_region_cal_reliability=(),
            log_goal_interaction_probability=(),
            log_goal_interaction_valid=(), log_goal_boxes=(8,4), log_goal_candidate_scores=(8,),
            log_goal_candidate_sources=(8,), log_goal_ref_map=(64,64),
            log_goal_heatmap_policy=(64,64), log_goal_memory=(6,))
        shapes.update(GOAL_FIELDS)
        goal_dtypes.update(log_goal_ref_map=np.uint8, log_goal_heatmap_policy=np.uint8)
        if goal_use_raw_image:
            shapes.update(goal_image=(96,96,3), goal_image_valid=())
            goal_dtypes.update(goal_image=np.uint8, goal_image_valid=np.uint8)
    while True:
        candidates = [(str(key), ep) for key, ep in episodes.items() if len(ep["reward"]) >= 2]
        if not candidates:
            raise ValueError("No spatial replay transitions")
        probabilities = np.asarray([len(ep["reward"]) for _, ep in candidates], float)
        probabilities /= probabilities.sum()
        segments, size = [], 0
        while size < length:
            key, ep = candidates[rng.choice(len(candidates), p=probabilities)]
            start = int(rng.randint(0, len(ep["reward"])-1)) if not segments else 0
            count = min(length-size, len(ep["reward"])-start)
            segments.append((key, ep, start, count)); size += count
        names = set(shapes).union(*(set(ep) for _, ep in candidates))
        names = {name for name in names if not name.startswith("vnext_memory_")}
        result = {}
        for name in names:
            if name not in templates:
                example = None
                for _, ep in candidates:
                    if name in ep:
                        example = next((v for v in ep[name] if v is not None), None)
                        if example is not None:
                            break
                if example is None:
                    shape = ((64,64,3) if name == "zoomed_image" else
                             (64,64) if name == "heatmap_on_zoomed" else shapes.get(name, ()))
                    dtype = goal_dtypes.get(name, np.uint8 if name in (
                        "zoomed_image", "heatmap_on_zoomed", "log_spatial_patch",
                        "log_spatial_history_frames") else np.float32)
                    example = np.zeros(shape, dtype)
                templates[name] = np.zeros_like(example)
            fill = templates[name]
            result[name] = np.asarray([fill if value is None else value
                for _, ep, start, count in segments
                for value in (ep[name][start:start+count] if name in ep else [None]*count)]).copy()
        patches = np.zeros((length,patch_size,patch_size,6), np.uint8)
        poses = np.zeros((length,4), np.float32)
        summaries = np.zeros((length,12), np.float32)
        spatial_valid = np.zeros(length, np.float32)
        trajectory_valid = np.zeros(length, np.float32)
        history_frames = np.zeros((length,history_slots,4,4,4),np.uint8)
        history_meta = np.zeros((length,history_slots,8),np.float32)
        history_valid = np.zeros((length,history_slots),np.float32)
        history_tick = np.zeros(length,np.float32)
        history_snapshot_valid = np.zeros(length,np.float32)
        local_valid = np.zeros(length,np.float32)
        previous = np.zeros((length,64,64,3), np.uint8)
        controls = np.zeros((length,9), np.float32)
        actuator_valid = np.zeros(length, np.float32)
        pair_valid = np.zeros(length, np.float32)
        calibration = np.zeros(length, np.float32)
        frameids = np.zeros(length, np.int64)
        realfirst = np.zeros(length, bool)
        observations = np.zeros(length, np.int64)
        goal_context = np.zeros((length,8), np.float32)
        goal_boxes = np.zeros((length,8,4), np.float32)
        goal_memory = np.zeros((length,6), np.float32)
        goal_valid = np.zeros(length, np.float32)
        goal_discovery = np.zeros(length, np.float32)
        goal_discovery_valid = np.zeros(length, np.float32)
        if len(discovery_cache) > len(episodes) + 32:
            discovery_cache = {key: value for key, value in discovery_cache.items() if key in episodes}
        row = 0
        for key, ep, start, count in segments:
            total = len(ep["reward"])
            if goal_enabled:
                cached = discovery_cache.get(key)
                if cached is None or cached[0] != total:
                    cached = (total, *goal_discovery_labels(ep))
                    discovery_cache[key] = cached
                goal_discovery[row:row+count] = cached[1][start:start+count]
                goal_discovery_valid[row:row+count] = cached[2][start:start+count]
            group = ep.get("log_vnext_world_group")
            groupkey = str(np.asarray(group).reshape(-1)[0]) if group is not None else key
            holdout = identity(groupkey) % 5 == 0
            result["is_first"][row] = True
            for i in range(start,start+count):
                realfirst[row] = bool(ep["is_first"][i])
                calibration[row] = holdout
                frameids[row] = identity(key+":"+str(i))
                observations[row] = i
                actuator_valid[row] = _actual_control_valid(ep, i)
                if goal_enabled and i+1 < total and "log_goal_context" in ep:
                    version = ep.get("log_goal_version", np.zeros(total))[i+1]
                    context = np.asarray(ep["log_goal_context"][i+1])
                    if version == 1 and context.shape == (8,) and np.isfinite(context).all():
                        goal_context[row] = context
                        goal_valid[row] = 1.
                        if "log_goal_boxes" in ep:
                            boxes = np.asarray(ep["log_goal_boxes"][i+1])
                            if boxes.shape == (8,4) and np.isfinite(boxes).all():
                                goal_boxes[row] = boxes
                        if "log_goal_memory" in ep:
                            memory = np.asarray(ep["log_goal_memory"][i+1])
                            if memory.shape == (6,) and np.isfinite(memory).all():
                                goal_memory[row] = memory
                if i+1 < total and "log_spatial_patch" in ep:
                    version = ep.get("log_spatial_version", np.zeros(total))
                    if int(np.asarray(version[i+1])) == 2:
                        patch = np.asarray(ep["log_spatial_patch"][i+1], np.uint8)
                        if patch.shape != (patch_size,patch_size,6):
                            raise ValueError("Spatial replay patch/configuration mismatch")
                        patches[row] = patch
                        poses[row] = ep["log_spatial_pose"][i+1]
                        if "log_spatial_summary" in ep:
                            summaries[row] = ep["log_spatial_summary"][i+1]
                        spatial_valid[row] = 1
                        if "log_spatial_trajectory_valid" in ep:
                            marker = ep["log_spatial_trajectory_valid"][i+1]
                            trajectory_valid[row] = bool(marker is not None and np.isfinite(marker) and float(marker)>.5)
                        # Continuous global pose and a usable local segment are
                        # distinct. Historical data cannot fabricate either flag.
                        local_marker = ep.get("log_spatial_local_valid",np.zeros(total))[i+1]
                        local_valid[row] = bool(local_marker is not None and np.isfinite(local_marker) and float(local_marker)>.5)
                        bank_names = ("log_spatial_history_frames", "log_spatial_history_meta",
                                      "log_spatial_history_valid", "log_spatial_history_tick")
                        if all(name in ep for name in bank_names):
                            bank, meta, mask, tick = (np.asarray(ep[name][i+1]) for name in bank_names)
                            if bank.shape != (history_slots,4,4,4) or meta.shape != (history_slots,8) or mask.shape != (history_slots,):
                                raise ValueError("Spatial history snapshot/configuration mismatch")
                            finite = np.isfinite(bank).all() and np.isfinite(meta).all() and np.isfinite(mask).all() and np.isfinite(tick).all()
                            if finite and np.asarray(tick).size == 1:
                                history_frames[row] = bank
                                history_meta[row] = meta
                                history_valid[row] = mask > .5
                                history_tick[row] = float(tick)
                                available = ep.get("log_spatial_memory_available",np.zeros(total))[i+1]
                                # Snapshot availability is independent of pose;
                                # bank entries must nevertheless be causal.
                                selected_meta = meta[mask>.5]
                                causal = bool((selected_meta[:,0]>=0).all() and
                                    (selected_meta[:,1]>=selected_meta[:,0]).all() and
                                    (selected_meta[:,1]<=float(tick)).all())
                                marker_valid = available is not None and np.isfinite(available) and float(available)>.5
                                values_valid = bool((bank>=0).all() and (bank<=255).all() and
                                    (mask>=0).all() and (mask<=1).all() and float(tick)>=0)
                                history_snapshot_valid[row] = bool(marker_valid and values_valid and (mask>.5).any() and causal)
                if i > 0 and not realfirst[row]:
                    error = bool(ep.get("log_env_error", np.zeros(total))[i])
                    if not error and bool(actuator_valid[row]):
                        previous[row] = ep["image"][i-1]
                        if "log_applied_control" in ep:
                            controls[row] = ep["log_applied_control"][i]
                            pair_valid[row] = 1
                row += 1
        result.update(vnext_spatial_patch=patches, vnext_spatial_pose=poses,
            vnext_spatial_summary=summaries, vnext_spatial_valid=spatial_valid,
            vnext_spatial_trajectory_valid=trajectory_valid,
            vnext_spatial_local_valid=local_valid,
            vnext_spatial_history_frames=history_frames, vnext_spatial_history_meta=history_meta,
            vnext_spatial_history_valid=history_valid, vnext_spatial_history_tick=history_tick,
            vnext_spatial_history_snapshot_valid=history_snapshot_valid,
            vnext_pose_prev_image=previous, vnext_pose_control=controls,
            vnext_pose_prev_valid=pair_valid, vnext_calibration=calibration,
            vnext_window_valid=np.ones(length,np.float32),vnext_real_first=realfirst,
            vnext_frame_id=frameids,vnext_observation_index=observations)
        # Override schema padding using source provenance. In particular, legacy
        # episodes without a validity field must not silently disable all A
        # future losses merely because the fixed mixed-replay schema adds it.
        result["log_actuator_valid"] = actuator_valid
        for name in ("log_applied_control", "log_actuator_before", "log_actuator_after"):
            result[name] = np.nan_to_num(result[name], nan=0., posinf=0., neginf=0.)
        if goal_enabled:
            result.update(vnext_goal_context=goal_context, vnext_goal_context_valid=goal_valid,
                          vnext_goal_boxes=goal_boxes,
                          vnext_goal_memory=goal_memory,
                          vnext_goal_discovery_gt=goal_discovery,
                          vnext_goal_discovery_valid=goal_discovery_valid)
        yield result
