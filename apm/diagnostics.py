"""Read-only diagnostics for estimated spatial memory and offline pose audits.

All statistics use the originally collected map, not a map rebuilt with the
current weights. Ground truth is read only here, after rollout; no value from
this module is supplied to a policy, loss, pose correction or map update.
"""
import numpy as np


def _rows(episode, name, length, width=None):
    """Accept cache lists with an unknown initial row as well as saved arrays."""
    value = episode.get(name)
    if value is None or len(value) != length:
        return None
    expected = (length,) if width is None else (length, width)
    try:
        result = np.asarray(value, dtype=np.float64)
        if result.shape == expected:
            return result
    except (TypeError, ValueError):
        pass
    result = np.full(expected, np.nan, np.float64)
    for index, item in enumerate(value):
        if item is None:
            continue
        try:
            item = np.asarray(item, dtype=np.float64)
            if width is None and item.size == 1:
                result[index] = float(item.reshape(-1)[0])
            elif width is not None and item.shape == (width,):
                result[index] = item
        except (TypeError, ValueError):
            continue
    return result


def spatial_episode_diagnostics(episode):
    """Return available scalar measurements; unknown errors are never zero.

    Row i+1 contains the decision/map at observation i, while audit pose row i
    belongs directly to observation i. The final observation has no decision
    map, so it is deliberately excluded. Estimated path length is available
    only when the episode was tracked continuously from its true reset. Map
    visited counts refer to estimated episode ledger cells; observed counts refer to the rolling view proxy, never global physical area.
    """
    length = len(episode.get("reward", episode.get("log_spatial_version", [])))
    if length < 2:
        return {}
    versions = _rows(episode, "log_spatial_version", length)
    if versions is None:
        return {}
    selected = np.isfinite(versions[1:]) & (versions[1:] == 2)
    if not selected.any():
        return {}
    result = dict(spatial_decision_count=float(selected.sum()),
                  spatial_localization_rate=0., spatial_trajectory_available=0.,
                  spatial_map_proxy_available=0., spatial_audit_available=0.,
                  spatial_audit_count=0., spatial_audit_fraction=0.)
    poses = _rows(episode, "log_spatial_pose", length, 4)
    summaries = _rows(episode, "log_spatial_summary", length, 12)
    if poses is None or summaries is None:
        return result
    poses = poses[1:]
    summaries = summaries[1:]
    tracked = selected & np.isfinite(poses).all(-1) & np.isfinite(summaries).all(-1)
    tracked &= summaries[:, 0] > .5  # episode-origin tracking, not network ready
    errors = _rows(episode, "log_env_error", length)
    if errors is not None:
        # A transport failure invalidates its resulting observation.
        tracked &= ~(np.nan_to_num(errors[:-1], nan=1.) > .5)
    result["spatial_localization_rate"] = float(tracked.sum())/(length-1)

    # These are explicitly predicted search-behaviour proxies on valid states.
    if tracked.any():
        result["spatial_map_proxy_available"] = 1.
        for field, metric in (("log_spatial_visited_cells", "spatial_visited_cells"),
                              ("log_spatial_known_cells", "spatial_local_observed_cells")):
            values = _rows(episode, field, length)
            if values is not None:
                valid = tracked & np.isfinite(values[1:]) & (values[1:] >= 0)
                if valid.any():
                    result[metric] = float(values[1:][np.flatnonzero(valid)[-1]])
        revisit = _rows(episode, "log_spatial_revisit", length)
        if revisit is not None:
            valid = tracked & np.isfinite(revisit[1:]) & (revisit[1:] >= 0) & (revisit[1:] <= 1)
            if valid.any():
                result["spatial_view_revisit_rate"] = float(revisit[1:][valid].mean())

    first = _rows(episode, "is_first", length)
    from_reset = (first is not None and np.isfinite(first[0]) and first[0] > .5
                  and not np.any(np.nan_to_num(first[1:], nan=1.) > .5))
    full_tracking = bool(from_reset and tracked.all())
    if not full_tracking:
        return result
    result["spatial_trajectory_available"] = 1.
    result["spatial_path_observation_count"] = float(len(poses))
    if len(poses) >= 2:
        result["spatial_estimated_path_length"] = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=-1).sum())

    truth = _rows(episode, "log_spatial_gt_pose", length, 3)
    truth_valid = _rows(episode, "log_spatial_gt_valid", length)
    if truth is None or truth_valid is None:
        return result
    truth, truth_valid = truth[:-1], truth_valid[:-1]
    pairs = np.isfinite(truth).all(-1) & np.isfinite(truth_valid) & (truth_valid > .5)
    result["spatial_audit_count"] = float(pairs.sum())
    result["spatial_audit_fraction"] = float(pairs.mean())
    # A late valid GT point cannot supply the unknown episode-origin pose.
    if not pairs[0] or pairs.sum() < 2:
        return result
    origin = truth[0]
    offset = truth[:, :2]-origin[:2]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    relative = np.stack((c*offset[:, 0]+s*offset[:, 1],
                         -s*offset[:, 0]+c*offset[:, 1]), -1)
    position_error = np.linalg.norm(poses[:, :2]-relative, axis=-1)
    heading_error = poses[:, 2]-(truth[:, 2]-origin[2])
    heading_error = np.abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
    result.update(spatial_audit_available=1.,
        spatial_ate_rmse=float(np.sqrt(np.mean(position_error[pairs]**2))),
        spatial_heading_error_deg=float(np.rad2deg(heading_error[pairs]).mean()))
    return result
