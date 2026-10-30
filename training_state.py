"""Read-only resume planning and atomic training checkpoints.

Model updates and consumed environment interactions are separate after a crash.
A resumed experiment always writes into a new directory and reads ancestor
Replay directories without modifying them. Environment and sampler cursors are
not reconstructed from old checkpoints.
"""
import collections
import copy
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import torch

SCHEMA_VERSION = 1
_RUNTIME_KEYS = {
    "resume_from", "logdir", "traindir", "evaldir", "offline_traindir",
    "offline_evaldir", "device", "steps", "log_every", "log_all_metrics",
    "use_wandb", "wandb_key", "results_dir", "name", "debug",
}
_CLOCKS = ("_should_train", "_should_log", "_should_reset")
_SCALERS = ("_wm._model_opt", "_wm._a_opt", "_task_behavior._actor_opt",
            "_task_behavior._value_opt", "_search_c._actor_opt", "_search_c._value_opt",
            "_vnext._a_opt", "_vnext._b_opt", "_vnext_control._q_opt")


def _path(value):
    return Path(value).expanduser().resolve()


def _unique_paths(values):
    result = []
    for value in values:
        if not value:
            continue
        path = _path(value)
        if path not in result:
            result.append(path)
    return result


def _under(path, parent):
    return path == parent or parent in path.parents


def _check_destination(directory, source_runs):
    destination = _path(directory)
    for source in source_runs:
        if _under(destination, _path(source)):
            raise ValueError("Resume output must be outside the read-only source run: " + str(source))
    return destination


def _config_dict(config):
    values = dict(vars(config)) if hasattr(config, "__dict__") else dict(config)
    # Credentials must not enter a checkpoint or resume report.
    return {key: (None if key == "wandb_key" or any(
        marker in key.lower() for marker in ("password", "private_key", "api_token", "api_key")) else value)
        for key, value in values.items()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _replay_from_sources(sources, limit):
    records = {}
    for directory in sources:
        if not directory.is_dir():
            raise FileNotFoundError("Missing ancestor Replay directory: " + str(directory))
        for filename in sorted(directory.glob("*.npz")):
            try:
                with np.load(filename, allow_pickle=False) as archive:
                    episode = {key: archive[key] for key in archive.files}
                length = len(episode["reward"])
                if length < 1 or any(np.ndim(value) == 0 or len(value) != length
                                     for value in episode.values()):
                    raise ValueError("Replay arrays must have equal time lengths")
            except Exception as error:
                raise ValueError("Unreadable Replay file: " + str(filename)) from error
            match = re.match(r"^(.*)-(\d+)$", filename.stem)
            identity = match.group(1) if match else filename.stem
            previous = records.get(identity)
            priority = (length, filename.stat().st_mtime_ns)
            if previous is None or priority >= previous[0]:
                records[identity] = (priority, episode)
    if not records or not any(len(item[1]["reward"]) >= 2 for item in records.values()):
        raise ValueError("Resume requires saved real transitions; random prefill is not repeated")
    replay_steps = sum(len(item[1]["reward"]) - 1 for item in records.values())
    replay = collections.OrderedDict((key, records[key][1]) for key in sorted(records))
    cached_steps = replay_steps
    if limit:
        while len(replay) > 1 and cached_steps > int(limit):
            _, episode = replay.popitem(last=False)
            cached_steps -= len(episode["reward"]) - 1
    return replay, int(replay_steps)


def _logged_step(source_run, repeat):
    result = 0
    filename = source_run / "metrics.jsonl"
    if filename.exists():
        for line in filename.read_text().splitlines():
            try:
                value = json.loads(line).get("step", 0)
                if np.isfinite(value):
                    result = max(result, int(value) // repeat)
            except (ValueError, TypeError):
                continue
    return result


def _completed_branches(source_runs):
    groups = set()
    for source in source_runs:
        for filename in source.glob("branches/*/feedback.json"):
            try:
                record = json.loads(filename.read_text())
                if len(record.get("options", [])) >= 2:
                    groups.add(int(record.get("group", filename.parent.name)))
            except (ValueError, TypeError, OSError):
                continue
    return groups


def _validate_search_versions(checkpoint, values):
    from search_vnext import check_vnext_checkpoint
    enabled = values.get("search_vnext_enabled", False) if isinstance(values, dict) else getattr(values, "search_vnext_enabled", False)
    check_vnext_checkpoint(checkpoint, enabled)
    from search_a import check_checkpoint
    from search_memory import check_b_checkpoint
    from search_c import check_c_checkpoint
    get = values.get if isinstance(values, dict) else lambda key, default=None: getattr(values, key, default)
    enabled_a, enabled_b, enabled_c = (bool(get("search_" + item + "_enabled", False)) for item in ("a", "b", "c"))
    check_checkpoint(checkpoint, enabled_a, get("search_a_model_version", 2))
    check_b_checkpoint(checkpoint, enabled_b)
    check_c_checkpoint(checkpoint, enabled_c)
    if enabled_b and checkpoint.get("search_b_version") != get("search_b_model_version"):
        raise ValueError("B checkpoint/model version mismatch")
    if enabled_c and checkpoint.get("search_c_version") != get("search_c_model_version"):
        raise ValueError("C checkpoint/model version mismatch")


def prepare_resume(config):
    """Restore saved training configuration and inspect source data, without writes."""
    requested = getattr(config, "resume_from", "")
    if not requested:
        return None
    requested = _path(requested)
    if requested.is_dir():
        candidates = [requested / name for name in ("interrupted.pt", "latest.pt")]
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            raise FileNotFoundError("No interrupted.pt or latest.pt in " + str(requested))
        checkpoint_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    else:
        checkpoint_path = requested
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("audit_only"):
        raise ValueError("An audit-only checkpoint cannot resume formal training")
    if not all(key in checkpoint for key in ("agent_state_dict", "optims_state_dict", "step", "config")):
        raise ValueError("Resume requires a full training checkpoint and saved configuration")
    saved = dict(checkpoint["config"])
    incoming = dict(vars(config))
    if bool(incoming.get("search_vnext_enabled", False)) != bool(saved.get("search_vnext_enabled", False)):
        raise ValueError("Cannot resume across legacy/vNext algorithms; start a fresh experiment")
    for key in ("task", "target_item"):
        if key in saved and incoming.get(key) is not None and incoming[key] != saved[key]:
            raise ValueError("Resume " + key + " mismatch: " + str(incoming[key]) + " != " + str(saved[key]))
    repeat = int(saved.get("action_repeat", 1))
    if repeat < 1:
        raise ValueError("Invalid saved action_repeat")
    restored = copy.deepcopy(saved)
    # Saved intervals are decision-step units; main normalizes them once again.
    for key in ("steps", "eval_every", "log_every", "time_limit"):
        if key in restored:
            restored[key] *= repeat
    for key, value in incoming.items():
        if key in _RUNTIME_KEYS or key.startswith(("minedojo_reset_", "env_reset_", "env_recovery_")):
            restored[key] = value
        elif key not in restored:
            restored[key] = value
    for key, metadata in (("search_a_model_version", "search_a_version"),
                          ("search_b_model_version", "search_b_version"),
                          ("search_c_model_version", "search_c_version"),
                          ("search_a_control_version", "search_a_control_version")):
        if key not in saved and checkpoint.get(metadata) is not None:
            restored[key] = checkpoint[metadata]
    _validate_search_versions(checkpoint, restored)
    restored["resume_from"] = str(checkpoint_path)
    restored["traindir"] = incoming.get("traindir")
    restored["evaldir"] = incoming.get("evaldir")
    source_run = checkpoint_path.parent
    source_runs = _unique_paths(list(checkpoint.get("resume_source_runs", [])) + [source_run])
    _check_destination(restored["logdir"], source_runs)
    for key in ("traindir", "evaldir"):
        if restored.get(key):
            _check_destination(restored[key], source_runs)
    declared = list(checkpoint.get("replay_sources", []))
    if not declared:
        prior_train = saved.get("traindir")
        if prior_train and _path(prior_train).is_dir():
            declared.append(prior_train)
        elif (source_run / "train_eps").is_dir():
            declared.append(source_run / "train_eps")
    if (source_run / "train_eps").is_dir():
        declared.append(source_run / "train_eps")
    sources = _unique_paths(declared)
    if not sources:
        raise ValueError("No saved Replay source accompanies this checkpoint")
    for key in ("logdir", "traindir", "evaldir"):
        if restored.get(key):
            _check_destination(restored[key], sources)
    replay, replay_steps = _replay_from_sources(sources, restored.get("dataset_size"))
    model_step = int(checkpoint["step"])
    old_offset = int(checkpoint.get("replay_budget_offset", 0))
    if model_step < 0 or old_offset < 0:
        raise ValueError("Checkpoint progress and consumed Replay offset must be nonnegative")
    logged_step = _logged_step(source_run, repeat)
    environment_step = max(model_step, replay_steps + old_offset, logged_step)
    budget_offset = environment_step - replay_steps
    target = int(restored["steps"]) // repeat
    if environment_step >= target:
        raise ValueError("Consumed interaction budget already reaches the requested total target: "
                         + str(environment_step) + " >= " + str(target))
    groups = _completed_branches(source_runs)
    every = int(restored.get("search_branch_every", 100000))
    if every < 1:
        raise ValueError("search_branch_every must be positive")
    runtime = checkpoint.get("runtime_state", {})
    next_branch = runtime.get("next_branch_step", checkpoint.get("next_branch_step"))
    notes = []
    if next_branch is None:
        if groups:
            next_branch = (max(groups) + 1) * every
        else:
            next_branch = (environment_step // every + 1) * every
            notes.append("legacy_branch_schedule_inferred_from_consumed_budget")
    next_branch = int(next_branch)
    while next_branch // every in groups:
        next_branch += every
    if "rng_state" not in checkpoint:
        notes.append("legacy_checkpoint_has_no_global_rng_state")
    notes.append("environment_and_replay_sampling_cursor_are_recreated")
    vars(config).clear()
    vars(config).update(restored)
    return dict(checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                source_run=source_run, source_runs=source_runs,
                model_step=model_step, environment_step=environment_step,
                replay_sources=sources, replay=replay, replay_steps=replay_steps,
                budget_offset=budget_offset, next_branch_step=next_branch,
                completed_branch_groups=sorted(groups), runtime_notes=notes)


def _resolve(obj, path):
    for key in path.split("."):
        obj = getattr(obj, key)
    return obj


def _behavior(agent):
    obj = getattr(agent, "_task_behavior", None)
    return getattr(obj, "_orig_mod", obj)


def _optimizer_update_estimate(checkpoint):
    counts = []
    for path, state in checkpoint["optims_state_dict"].items():
        if "_search_c" in path:
            continue
        for item in state.get("state", {}).values():
            step = item.get("step")
            if step is not None:
                counts.append(int(step.item() if torch.is_tensor(step) else step))
    return max(counts, default=0)


def _restore_rng(state, notes):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i, value in enumerate(state["cuda"][:count]):
                torch.cuda.set_rng_state(value.cpu(), device=i)
            if count != len(state["cuda"]):
                notes.append("cuda_rng_device_count_differs")
        else:
            notes.append("cuda_rng_unavailable_on_current_device")


def restore_training_state(agent, plan, logger):
    """Strictly restore model/optimizer state, but count all consumed interactions."""
    import tools
    checkpoint, config = plan["checkpoint"], agent._config
    _validate_search_versions(checkpoint, config)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    if getattr(agent, "_search_a", None) is not None:
        agent._search_a.load_rng_state(checkpoint.get("search_a_rng_state"))
    if getattr(agent, "_search_c", None) is not None:
        agent._search_c.load_rng_state(checkpoint.get("search_c_rng_state"))
    if getattr(agent, "_vnext_control", None) is not None:
        agent._vnext_control.load_rng_state(checkpoint.get("vnext_rng_state"))
    scalers = checkpoint.get("amp_scalers", {})
    for path, state in scalers.items():
        optimizer = _resolve(agent, path)
        if optimizer is not None:
            optimizer._scaler.load_state_dict(state)
    if not scalers and "model_amp_scaler" in checkpoint:
        agent._wm._model_opt._scaler.load_state_dict(checkpoint["model_amp_scaler"])
    runtime = checkpoint.get("runtime_state", {})
    estimate = _optimizer_update_estimate(checkpoint)
    agent._update_count = int(runtime.get("update_count", estimate))
    behavior = _behavior(agent)
    if behavior is not None:
        behavior._updates = int(runtime.get("behavior_updates", estimate))
        behavior.jump_prob = float(runtime.get("jump_prob", config.jump_prob))
    if "update_count" not in runtime or "behavior_updates" not in runtime:
        plan["runtime_notes"].append("legacy_update_counters_inferred_from_applied_optimizer_steps")
    agent._step = int(plan["environment_step"])
    agent._should_pretrain._once = False
    # Old model may trail already-consumed data by thousands of interactions.
    # Re-anchor clocks instead of replaying a burst of missed optimization calls.
    for name in _CLOCKS:
        clock = getattr(agent, name, None)
        if clock is not None and hasattr(clock, "_last"):
            clock._last = agent._step if name != "_should_log" else None
    logger.step = int(config.action_repeat) * agent._step
    agent._replay_sources = [str(path) for path in plan["replay_sources"]]
    agent._replay_budget_offset = int(plan["budget_offset"])
    agent._resume_source_runs = [str(path) for path in plan["source_runs"]]
    agent._resume_readonly_replay_sources = [str(path) for path in plan["replay_sources"]]
    _restore_rng(checkpoint.get("rng_state"), plan["runtime_notes"])
    plan["runtime_restore"] = dict(update_count=agent._update_count,
        behavior_updates=getattr(behavior, "_updates", None),
        jump_prob=getattr(behavior, "jump_prob", None),
        global_rng_available="rng_state" in checkpoint,
        replay_sampling_cursor_restored=False)
    return plan["runtime_restore"]


def write_resume_manifest(plan, config):
    destination = _check_destination(config.logdir, plan["source_runs"])
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "resume.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite a resume manifest: " + str(path))
    record = dict(schema_version=SCHEMA_VERSION,
        source_checkpoint=str(plan["checkpoint_path"]),
        source_run=str(plan["source_run"]),
        model_checkpoint_step=plan["model_step"],
        consumed_interaction_step=plan["environment_step"],
        target_interaction_step=int(config.steps),
        replay_sources=[str(p) for p in plan["replay_sources"]],
        replay_steps=plan["replay_steps"], replay_budget_offset=plan["budget_offset"],
        completed_branch_groups=plan["completed_branch_groups"],
        next_branch_step=plan["next_branch_step"],
        skip_random_prefill=True, skip_initial_pretraining=True,
        fresh_environment=True, exact_trajectory_resume=False,
        runtime_restore=plan.get("runtime_restore", {}),
        notes=list(dict.fromkeys(plan["runtime_notes"])))
    path.write_text(json.dumps(_jsonable(record), indent=2) + "\n")
    return path


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def save_training_checkpoint(agent, config, next_branch_step, name="latest.pt"):
    """Atomically publish a checkpoint; an interrupted save cannot replace latest."""
    import tools
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Checkpoint name must be a local .pt filename")
    source_runs = getattr(agent, "_resume_source_runs", [])
    directory = _check_destination(config.logdir, source_runs +
        list(getattr(agent, "_resume_readonly_replay_sources", [])))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    pending = target.with_suffix(target.suffix + ".tmp")
    sources = _unique_paths(list(getattr(agent, "_replay_sources", [])) +
                            [getattr(config, "traindir", None) or directory / "train_eps"])
    behavior = _behavior(agent)
    scalers = {}
    for path in _SCALERS:
        try:
            optimizer = _resolve(agent, path)
        except AttributeError:
            continue
        if optimizer is not None and hasattr(optimizer, "_scaler"):
            scalers[path] = optimizer._scaler.state_dict()
    runtime = dict(update_count=int(getattr(agent, "_update_count", 0)),
        behavior_updates=int(getattr(behavior, "_updates", 0)),
        jump_prob=float(getattr(behavior, "jump_prob", config.jump_prob)),
        next_branch_step=int(next_branch_step),
        initial_pretraining_pending=bool(agent._should_pretrain._once),
        clocks={name: getattr(getattr(agent, name, None), "_last", None) for name in _CLOCKS})
    wm_a, wm_b = getattr(agent._wm, "search_a", None), getattr(agent._wm, "search_b", None)
    controller, c = getattr(agent, "_search_a", None), getattr(agent, "_search_c", None)
    payload = dict(agent_state_dict=agent.state_dict(),
        optims_state_dict=tools.recursively_collect_optim_state_dict(agent),
        step=int(agent._step), config=_config_dict(config),
        search_vnext_version=1 if getattr(config, "search_vnext_enabled", False) else None,
        actuator_version=1 if getattr(config, "search_vnext_enabled", False) else None,
        vnext_memory_version=1 if getattr(config, "search_vnext_enabled", False) else None,
        vnext_rng_state=(agent._vnext_control.rng_state()
                         if getattr(agent, "_vnext_control", None) is not None else None),
        search_a_version=config.search_a_model_version if config.search_a_enabled else None,
        search_a_control_version=config.search_a_control_version if config.search_a_enabled else None,
        search_a_label_version=(3 if getattr(config, "search_vnext_enabled", False) else
                                2 if config.search_a_enabled else None),
        search_a_rng_state=controller.rng_state() if controller is not None else None,
        search_a_macro_names=list(controller.macro_names) if controller is not None else [],
        model_amp_scaler=agent._wm._model_opt._scaler.state_dict(),
        amp_scalers=scalers,
        search_b_version=getattr(wm_b, "model_version", None) if config.search_b_enabled else None,
        search_c_version=getattr(c, "model_version", None) if config.search_c_enabled else None,
        search_c_rng_state=c.rng_state() if c is not None else None,
        eval_protocol_version=(7 if getattr(config, "search_vnext_enabled", False) else
                               6 if getattr(config, "search_eval_evidence", False) else 5),
        search_candidate_version=(1 if getattr(config, "search_vnext_enabled", False)
                                  else config.search_a_control_version),
        runtime_state=runtime, rng_state=_rng_state(), training_state_version=SCHEMA_VERSION,
        next_branch_step=int(next_branch_step),
        replay_sources=[str(path) for path in sources],
        replay_budget_offset=int(getattr(agent, "_replay_budget_offset", 0)),
        resume_source_runs=[str(path) for path in _unique_paths(source_runs)],
        interrupted=name == "interrupted.pt")
    try:
        with pending.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
    except BaseException:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target
