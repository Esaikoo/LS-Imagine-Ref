"""Causal spatial replay windows without reconstructing long RSSM histories."""
import hashlib
import numpy as np

from .evidence import SPATIAL_LABEL_SHAPES


def identity(value):
    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "little") % (2**31-1)


def sample_spatial_episodes(episodes, length, seed=0, patch_size=21):
    """Preserve original LS episode/window sampling; align outgoing map logs.

    Cache row i contains observation i and logs of the decision at i-1. Thus
    observation i reads map logs from row i+1, which were computed BEFORE that
    action was executed. A terminal observation has no such decision/map.
    Pose image pairs use their real predecessor even at synthetic window starts.
    """
    from .control import FIELDS
    rng = np.random.RandomState(seed)
    templates = {}
    shapes = dict(FIELDS, **SPATIAL_LABEL_SHAPES)
    shapes.update(log_spatial_patch=(patch_size, patch_size, 6), log_spatial_pose=(4,),
        log_spatial_summary=(12,), log_spatial_directions=(12,), log_spatial_version=(),
        log_spatial_novelty=(), log_spatial_revisit=(), log_spatial_known_cells=(),
        log_spatial_visited_cells=(), log_actuator_before=(3,), log_actuator_after=(3,),
        log_applied_control=(9,), log_env_error=(), log_actuator_valid=(),
        log_search_override=(), log_search_decision=(), log_vnext_world_group=())
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
                    example = np.zeros(shape, np.uint8 if name in ("zoomed_image", "heatmap_on_zoomed", "log_spatial_patch") else np.float32)
                templates[name] = np.zeros_like(example)
            fill = templates[name]
            result[name] = np.asarray([fill if value is None else value
                for _, ep, start, count in segments
                for value in (ep[name][start:start+count] if name in ep else [None]*count)]).copy()
        patches = np.zeros((length,patch_size,patch_size,6), np.uint8)
        poses = np.zeros((length,4), np.float32)
        summaries = np.zeros((length,12), np.float32)
        spatial_valid = np.zeros(length, np.float32)
        previous = np.zeros((length,64,64,3), np.uint8)
        controls = np.zeros((length,9), np.float32)
        pair_valid = np.zeros(length, np.float32)
        calibration = np.zeros(length, np.float32)
        frameids = np.zeros(length, np.int64)
        realfirst = np.zeros(length, bool)
        observations = np.zeros(length, np.int64)
        row = 0
        for key, ep, start, count in segments:
            total = len(ep["reward"])
            group = ep.get("log_vnext_world_group")
            groupkey = str(np.asarray(group).reshape(-1)[0]) if group is not None else key
            holdout = identity(groupkey) % 5 == 0
            result["is_first"][row] = True
            for i in range(start,start+count):
                realfirst[row] = bool(ep["is_first"][i])
                calibration[row] = holdout
                frameids[row] = identity(key+":"+str(i))
                observations[row] = i
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
                if i > 0 and not realfirst[row]:
                    error = bool(ep.get("log_env_error", np.zeros(total))[i])
                    if not error and not bool(ep.get("is_last", np.zeros(total))[i-1]):
                        previous[row] = ep["image"][i-1]
                        if "log_applied_control" in ep:
                            controls[row] = ep["log_applied_control"][i]
                            pair_valid[row] = 1
                row += 1
        result.update(vnext_spatial_patch=patches, vnext_spatial_pose=poses,
            vnext_spatial_summary=summaries, vnext_spatial_valid=spatial_valid,
            vnext_pose_prev_image=previous, vnext_pose_control=controls,
            vnext_pose_prev_valid=pair_valid, vnext_calibration=calibration,
            vnext_window_valid=np.ones(length,np.float32),vnext_real_first=realfirst,
            vnext_frame_id=frameids,vnext_observation_index=observations)
        yield result
