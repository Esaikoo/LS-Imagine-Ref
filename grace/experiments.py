"""Training architecture identity, separate from frozen-policy overrides and wire versions."""

ENGINEERING_VERSION = "GRACE-1.0.0"
BASELINE_VERSION = "VN4R3D2P3R1"
BASELINE_COMMIT = "f6ee70d9"

# A, B, C, D and policy heatmap fusion, in that order.
MODULES = {
    "grace_full": (True, True, True, True, True),
    "grace_d_only": (False, False, False, True, True),
    "grace_dac": (True, False, True, True, True),
    "grace_no_fusion": (True, True, True, True, False),
}


def module_manifest(mode):
    return dict(zip(("A", "B", "C", "D", "heatmap_fusion"), MODULES[mode]))


def validate_training_mode(values):
    mode = values.get("experiment_mode", "")
    if not mode:
        return
    if mode not in MODULES:
        raise ValueError(f"Unknown GRACE experiment_mode: {mode}")
    a, b, c, d, fusion = MODULES[mode]
    expected = dict(search_vnext_enabled=True, vnext_spatial_enabled=True,
                    goal_ref_enabled=d, goal_search_enabled=c, vnext_control=c,
                    vnext_memory_enabled=b, goal_fusion_enabled=fusion,
                    vnext_spatial_fix_version=3, goal_ref_version=2,
                    search_a_enabled=False, search_b_enabled=False,
                    search_c_enabled=False, search_branch_enabled=False)
    for key, wanted in expected.items():
        if values.get(key) != wanted:
            raise ValueError(f"{mode} requires {key}={wanted}, got {values.get(key)}")


def checkpoint_mode(checkpoint, requested="", *, evaluation=False):
    """Never infer a named ablation from missing parameters or runtime switches."""
    saved = checkpoint.get("config", {})
    mode = saved.get("experiment_mode", "")
    if not mode:
        if any(key in checkpoint for key in ("experiment_mode", "experiment_modules", "experiment_parameter_counts")):
            raise ValueError("Inconsistent GRACE checkpoint mode metadata")
        if requested:
            if requested != "grace_full":
                raise ValueError("A legacy checkpoint cannot initialize a GRACE training ablation")
            validate_training_mode(dict(saved, experiment_mode=requested))
            if (saved.get("goal_guidance_revision"), saved.get("vnext_pose_loss_version"),
                    saved.get("goal_ref_border_padding")) != (2, 4, "replicate"):
                raise ValueError("Request the historical preset for this pre-R1 checkpoint")
        return ""
    validate_training_mode(saved)
    if checkpoint.get("experiment_mode") != mode or checkpoint.get("experiment_modules") != module_manifest(mode):
        raise ValueError("Inconsistent GRACE checkpoint mode metadata")
    if requested != mode and (requested or not evaluation):
        raise ValueError(f"Checkpoint mode {mode} differs from requested mode {requested!r}")
    state = checkpoint["agent_state_dict"]
    for prefix, enabled in zip(("_vnext.a.", "_vnext.b.", "_vnext_control.", "_goal_reference."), MODULES[mode]):
        if any(key.startswith(prefix) for key in state) != enabled:
            raise ValueError(f"Checkpoint parameters {prefix} disagree with {mode}")
    if "optims_state_dict" in checkpoint:
        optimizers = checkpoint["optims_state_dict"]
        keys = ("_vnext._a_opt", "_vnext._b_opt", "_vnext_control._optimizer", "_goal_reference.optimizer")
        for key, enabled in zip(keys, MODULES[mode]):
            if (key in optimizers) != enabled:
                raise ValueError(f"Checkpoint optimizer {key} disagrees with {mode}")
    return mode


def parameter_counts(agent):
    """Stored scalar counts include target copies; shared Parameters count once."""
    system = agent._vnext
    modules = {"A": system.a if system is not None else None,
               "B": system.b if system is not None else None,
               "C": agent._vnext_control, "D": agent._goal_reference}
    counts = {name: sum(p.numel() for p in mod.parameters()) if mod is not None else 0
              for name, mod in modules.items()}
    counts["total"] = sum(p.numel() for p in agent.parameters())
    return counts
