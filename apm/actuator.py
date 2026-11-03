"""Shared command-to-control conversion; preserves the original LS wrapper semantics.

The state is derived solely from the agent's own commands and public settings.
Camera angles are degrees. Vector controls never contain simulator coordinates.
"""
import hashlib
import numpy as np
import torch

ACTUATOR_VERSION = 1
CONTROL_KEYS = ("attack", "forward", "back", "left", "right", "jump", "use", "pitch", "yaw")
CONTROL_DIM = len(CONTROL_KEYS)
STATE_KEYS = ("sticky_attack_remaining", "sticky_jump_remaining", "pitch")
STATE_DIM = len(STATE_KEYS)
# Keep IDs exactly aligned with BASIC_ACTIONS, including jump+forward.
_COMMANDS = np.zeros((12, CONTROL_DIM), np.float32)
_COMMANDS[1, 0] = 1
_COMMANDS[2, 7] = -10
_COMMANDS[3, 7] = 10
_COMMANDS[4, 8] = -10
_COMMANDS[5, 8] = 10
_COMMANDS[6, 1] = 1
_COMMANDS[7, 2] = 1
_COMMANDS[8, 3] = 1
_COMMANDS[9, 4] = 1
_COMMANDS[10, [1, 5]] = 1
_COMMANDS[11, 6] = 1


def initial_state(batch_shape=()):
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    return np.zeros(tuple(batch_shape) + (STATE_DIM,), np.float32)


def initial_state_torch(batch_shape=(), device=None):
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    return torch.zeros(tuple(batch_shape) + (STATE_DIM,), device=device)


def apply_command(command, state, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70)):
    command = np.asarray(command)
    if not np.issubdtype(command.dtype, np.integer):
        if not np.all(np.isfinite(command)) or np.any(command != np.floor(command)):
            raise ValueError("Command IDs must be integers")
        command = command.astype(np.int64)
    if np.any((command < 0) | (command >= 12)):
        raise ValueError("Command ID outside original 12-action space")
    state = np.asarray(state, np.float32)
    if state.shape[-1:] != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("Invalid actuator state")
    shape = np.broadcast_shapes(command.shape, state.shape[:-1])
    control = _COMMANDS[np.broadcast_to(command, shape)].copy()
    nxt = np.broadcast_to(state, shape + (STATE_DIM,)).copy()
    if sticky_attack:
        nxt[..., 0] = np.where(control[..., 0] > 0, sticky_attack, nxt[..., 0])
        active = nxt[..., 0] > 0
        control[..., 0] = np.where(active, 1, control[..., 0])
        control[..., 5] = np.where(active, 0, control[..., 5])
        nxt[..., 0] = np.maximum(nxt[..., 0] - 1, 0)
    if sticky_jump:
        nxt[..., 1] = np.where(control[..., 5] > 0, sticky_jump, nxt[..., 1])
        active = nxt[..., 1] > 0
        control[..., 5] = np.where(active, 1, control[..., 5])
        control[..., 1] = np.where(active, 1, control[..., 1])
        nxt[..., 1] = np.maximum(nxt[..., 1] - 1, 0)
    if pitch_limit:
        lo, hi = pitch_limit
        allowed = (nxt[..., 2] + control[..., 7] >= lo) & (nxt[..., 2] + control[..., 7] <= hi)
        control[..., 7] = np.where(allowed, control[..., 7], 0)
        nxt[..., 2] += control[..., 7]
    return control, nxt


def apply_command_torch(command, state, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70)):
    command = torch.as_tensor(command, device=state.device)
    if command.is_floating_point():
        if not bool(torch.isfinite(command).all()) or bool((command != command.floor()).any()):
            raise ValueError("Command IDs must be integers")
    command = command.long()
    if bool(((command < 0) | (command >= 12)).any()):
        raise ValueError("Command ID outside original 12-action space")
    if state.shape[-1:] != (STATE_DIM,) or not bool(torch.isfinite(state).all()):
        raise ValueError("Invalid actuator state")
    shape = torch.broadcast_shapes(command.shape, state.shape[:-1])
    control = torch.as_tensor(_COMMANDS, device=state.device)[command.expand(shape)].clone()
    nxt = state.float().expand(*shape, STATE_DIM).clone()
    if sticky_attack:
        nxt[..., 0] = torch.where(control[..., 0] > 0, float(sticky_attack), nxt[..., 0])
        active = nxt[..., 0] > 0
        control[..., 0] = torch.where(active, 1., control[..., 0])
        control[..., 5] = torch.where(active, 0., control[..., 5])
        nxt[..., 0] = (nxt[..., 0] - 1).clamp_min(0)
    if sticky_jump:
        nxt[..., 1] = torch.where(control[..., 5] > 0, float(sticky_jump), nxt[..., 1])
        active = nxt[..., 1] > 0
        control[..., 5] = torch.where(active, 1., control[..., 5])
        control[..., 1] = torch.where(active, 1., control[..., 1])
        nxt[..., 1] = (nxt[..., 1] - 1).clamp_min(0)
    if pitch_limit:
        lo, hi = pitch_limit
        candidate = nxt[..., 2] + control[..., 7]
        control[..., 7] = torch.where((candidate >= lo) & (candidate <= hi), control[..., 7], 0.)
        nxt[..., 2] += control[..., 7]
    return control, nxt


def rollout_commands(commands, state, **kwargs):
    commands = np.asarray(commands)
    if commands.ndim < 1 or commands.shape[-1] < 1:
        raise ValueError("A nonempty command sequence is required")
    current = np.asarray(state, np.float32)
    controls = []
    for i in range(commands.shape[-1]):
        control, current = apply_command(commands[..., i], current, **kwargs)
        controls.append(control)
    return np.stack(controls, -2), current


def rollout_commands_torch(commands, state, **kwargs):
    if commands.ndim < 1 or commands.shape[-1] < 1:
        raise ValueError("A nonempty command sequence is required")
    controls, current = [], state
    for i in range(commands.shape[-1]):
        control, current = apply_command_torch(commands[..., i], current, **kwargs)
        controls.append(control)
    return torch.stack(controls, -2), current


def candidate_mask(commands, state, durations=None, **kwargs):
    """Mask no-op pitch candidates and exact duplicates; first candidate wins.

    Commands have shape [options, max_duration]. Different final sticky state or
    true duration must remain distinct even when current controls happen to match.
    """
    commands = np.asarray(commands)
    if commands.ndim != 2:
        raise ValueError("Expected [options, max_duration] command IDs")
    durations = np.full(len(commands), commands.shape[-1], int) if durations is None else np.asarray(durations, int)
    if durations.shape != (len(commands),) or np.any(durations < 1) or np.any(durations > commands.shape[-1]):
        raise ValueError("Invalid option durations")
    mask, seen = np.ones(len(commands), bool), set()
    for i, (sequence, duration) in enumerate(zip(commands, durations)):
        applied, end = rollout_commands(sequence[:duration], state, **kwargs)
        # The noop option remains legal; only requested camera-only dead actions are removed.
        requested_camera = np.isin(sequence[:duration], [2, 3, 4, 5]).any()
        if requested_camera and not applied.any():
            mask[i] = False
        key = (duration, applied.tobytes(), end.tobytes())
        if key in seen:
            mask[i] = False
        else:
            seen.add(key)
    return mask


def world_group_id(session, generation):
    """Audit-only reset-group ID, exactly representable even after float32 cast."""
    digest = hashlib.blake2s((str(session) + ":" + str(int(generation))).encode()).digest()
    return 1 + int.from_bytes(digest[:3], "little") % ((1 << 24) - 1)
