"""Bounded recovery for MineDojo transport failures during episode reset.

No action is retried and no observation/transition is fabricated. The helper
replaces only the failing environment's bridge, preserving its RNG stream and
the existing simulator/task wrappers.
"""
import copy
import errno
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import gym


_TRANSPORT_ERRNOS = frozenset(getattr(errno, name) for name in (
    "ETIMEDOUT", "ECONNRESET", "ECONNABORTED", "ECONNREFUSED", "EPIPE",
    "ENOTCONN", "ENETDOWN", "ENETRESET", "ENETUNREACH", "EHOSTUNREACH",
    "ESHUTDOWN") if hasattr(errno, name))


def is_transport_error(error):
    """Recognize connection failures without masking file/model/label errors."""
    return isinstance(error, (socket.timeout, TimeoutError, ConnectionError)) or (
        isinstance(error, OSError) and error.errno in _TRANSPORT_ERRNOS)


def _environment_chain(environment):
    # Read own attributes: gym.Wrapper.__getattr__ otherwise delegates to children.
    seen = set()
    current = environment
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = vars(current).get("env")


def _owned_simulator(environment):
    for current in _environment_chain(environment):
        if "_bridge_env" in vars(current):
            return current
    raise RuntimeError("Cannot locate this environment's MineDojo bridge")


def clear_stale_reset_state(environment):
    """Force a full mission on a fresh bridge, never /kill against an empty one."""
    for current in _environment_chain(environment):
        attrs = vars(current)
        if "_server_start" in attrs:
            current._server_start = False
        if "_bridge_env" in attrs:
            for key in ("_prev_obs", "_prev_info", "_prev_action"):
                if key in attrs:
                    setattr(current, key, None)


def restart_minedojo_bridge(environment, bridge_factory=None):
    """Replace only the owned bridge; keep simulator/mission configuration."""
    sim = _owned_simulator(environment)
    old = sim._bridge_env
    rng_state = copy.deepcopy(old._rng.bit_generator.state)
    if bridge_factory is None:
        from minedojo.sim.bridge.bridge_env import BridgeEnv
        bridge_factory = BridgeEnv
    # Do not use InstanceManager.shutdown or kill processes by name/PID.
    old.close()
    new = bridge_factory(agent_count=getattr(old, "_agent_count", 1),
                         is_fault_tolerant=getattr(old, "_is_fault_tolerant", True),
                         seed=0)
    new._rng.bit_generator.state = rng_state
    sim._bridge_env = new
    clear_stale_reset_state(environment)


class ResetRecoveryError(RuntimeError):
    """A reset transport failure exhausted the bounded recovery budget."""


class ResetRecovery(gym.Wrapper):
    """Retry reset after a transport failure, at most additional retries times.

    A custom restart(env) may replace default bridge recovery, e.g. to preserve
    a paired experiment's explicit Java/mission seed. Successful reset outputs
    are returned unchanged; model/label/configuration errors propagate directly.
    """
    def __init__(self, env, retries=2, restart=None, log_path=None, label="minedojo"):
        super().__init__(env)
        if isinstance(retries, bool) or int(retries) != retries or retries < 0:
            raise ValueError("reset retries must be a nonnegative integer")
        self._reset_retries = int(retries)
        self._restart = restart or restart_minedojo_bridge
        self._recovery_log_path = Path(log_path) if log_path else None
        self._recovery_label = str(label)
        self.reset_recoveries = 0

    def _record(self, event, attempt, error=None):
        record = dict(time=datetime.now(timezone.utc).isoformat(),
                      environment=self._recovery_label, event=event,
                      attempt=attempt, max_attempts=self._reset_retries + 1)
        if error is not None:
            record.update(error_type=type(error).__name__, error=str(error))
        if self._recovery_log_path is not None:
            self._recovery_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._recovery_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if event in ("retry", "exhausted", "restart_failed"):
            detail = str(error) if error is not None else ""
            print(f"[env recovery] {self._recovery_label}: {event} "
                  f"attempt={attempt}/{self._reset_retries + 1} {detail}", flush=True)

    def reset(self, **kwargs):
        recovered = False
        for index in range(self._reset_retries + 1):
            try:
                observation = self.env.reset(**kwargs)
            except Exception as error:
                if not is_transport_error(error):
                    raise
                attempt = index + 1
                if index == self._reset_retries:
                    self._record("exhausted", attempt, error)
                    raise ResetRecoveryError(
                        f"{self._recovery_label}: reset transport failed after "
                        f"{attempt} attempts; last error: {type(error).__name__}: {error}"
                    ) from error
                self._record("retry", attempt, error)
                try:
                    self._restart(self.env)
                    clear_stale_reset_state(self.env)
                except Exception as restart_error:
                    self._record("restart_failed", attempt, restart_error)
                    # Invalid callbacks/factories are programming errors, not
                    # another reset to silently try on a half-rebuilt bridge.
                    if not is_transport_error(restart_error):
                        raise
                    raise ResetRecoveryError(
                        f"{self._recovery_label}: rebuilding the owned bridge failed "
                        f"after reset attempt {attempt}"
                    ) from restart_error
                self.reset_recoveries += 1
                recovered = True
                continue
            if recovered:
                self._record("recovered", index + 1)
            return observation
