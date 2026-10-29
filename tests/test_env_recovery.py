"""Reset recovery regressions; no Minecraft client or real training required."""
import copy
import errno
import json
import pathlib
import socket
import sys
import tempfile
import unittest
from unittest import mock

import gym
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from envs.recovery import (ResetRecovery, ResetRecoveryError, clear_stale_reset_state,
                           is_transport_error, restart_minedojo_bridge)


class FakeEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Dict({"value": gym.spaces.Discrete(10)})

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.reset_calls = 0
        self.step_calls = 0
        self.received_kwargs = []

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.received_kwargs.append(kwargs)
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def step(self, action):
        self.step_calls += 1
        raise socket.timeout("step result uncertain")


class FakeBridge:
    def __init__(self, seed=7, agent_count=1, is_fault_tolerant=True):
        self._rng = np.random.default_rng(seed)
        self._agent_count = agent_count
        self._is_fault_tolerant = is_fault_tolerant
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeSim(FakeEnv):
    def __init__(self, bridge):
        super().__init__([])
        self._bridge_env = bridge
        self._prev_obs = object()
        self._prev_info = {"old": True}
        self._prev_action = {"forward": 1}
        self.task_config = {"world_seed": "42", "target": "sand"}


class FakeFastReset(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._server_start = True
        self.reset_count = 4


class RecoveryTests(unittest.TestCase):
    def test_success_returns_same_observation_without_restart(self):
        observation = {"value": np.array([3])}
        env = FakeEnv([observation])
        callback = mock.Mock()
        wrapper = ResetRecovery(env, restart=callback)
        self.assertIs(wrapper.reset(seed=19), observation)
        self.assertEqual(env.received_kwargs, [{"seed": 19}])
        self.assertEqual(wrapper.reset_recoveries, 0)
        callback.assert_not_called()

    def test_timeout_rebuilds_then_returns_real_reset_only(self):
        observation = {"value": np.array([5])}
        env = FakeEnv([socket.timeout("mission sent; first frame unavailable"), observation])
        callback = mock.Mock()
        wrapper = ResetRecovery(env, restart=callback)
        self.assertIs(wrapper.reset(), observation)
        callback.assert_called_once_with(env)
        self.assertEqual((env.reset_calls, env.step_calls), (2, 0))
        self.assertEqual(wrapper.reset_recoveries, 1)

    def test_bounded_retries_log_exhaustion_and_preserve_cause(self):
        errors = [socket.timeout("wait1"), ConnectionResetError("wait2"), socket.timeout("wait3")]
        env = FakeEnv(errors)
        callback = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "reset.jsonl"
            wrapper = ResetRecovery(env, restart=callback, log_path=path, label="train-0")
            with self.assertRaisesRegex(ResetRecoveryError, "after 3 attempts") as caught:
                wrapper.reset()
            self.assertIs(caught.exception.__cause__, errors[-1])
            self.assertEqual(callback.call_count, 2)
            self.assertEqual(env.reset_calls, 3)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["event"] for r in events], ["retry", "retry", "exhausted"])
            self.assertTrue(all(r["environment"] == "train-0" for r in events))

    def test_zero_retries_does_not_restart(self):
        callback = mock.Mock()
        env = FakeEnv([socket.timeout("failed")])
        with self.assertRaises(ResetRecoveryError):
            ResetRecovery(env, retries=0, restart=callback).reset()
        callback.assert_not_called()
        self.assertEqual(env.reset_calls, 1)

    def test_logic_label_and_filesystem_errors_are_not_retried(self):
        errors = [ValueError("bad target label"), RuntimeError("invalid voxel shape"),
                  TypeError("bad config"), OSError(errno.ENOSPC, "disk full"),
                  FileNotFoundError("missing model")]
        for error in errors:
            with self.subTest(error=error):
                env = FakeEnv([error])
                callback = mock.Mock()
                with self.assertRaises(type(error)) as caught:
                    ResetRecovery(env, restart=callback).reset()
                self.assertIs(caught.exception, error)
                callback.assert_not_called()
                self.assertEqual(env.reset_calls, 1)

    def test_transport_errno_classification(self):
        for error in (TimeoutError(), BrokenPipeError(), ConnectionAbortedError(),
                      OSError(errno.ENOTCONN, "socket disconnected"),
                      OSError(errno.EHOSTUNREACH, "no route")):
            self.assertTrue(is_transport_error(error))
        self.assertFalse(is_transport_error(OSError(errno.EINVAL, "bad arguments")))

    def test_default_replacement_preserves_rng_config_and_only_owned_bridge(self):
        old = FakeBridge(seed=193)
        old._rng.integers(0, 10, size=17)
        original_rng = copy.deepcopy(old._rng.bit_generator.state)
        expected = np.random.default_rng()
        expected.bit_generator.state = copy.deepcopy(original_rng)
        sim = FakeSim(old)
        fast = FakeFastReset(sim)
        other = FakeBridge(seed=99)
        task_config = sim.task_config
        created = []
        def factory(**kwargs):
            result = FakeBridge(**kwargs)
            created.append(result)
            return result
        restart_minedojo_bridge(fast, bridge_factory=factory)
        self.assertEqual(old.closed, 1)
        self.assertEqual(other.closed, 0)
        self.assertIs(sim.task_config, task_config)
        self.assertIs(sim._bridge_env, created[0])
        np.testing.assert_array_equal(sim._bridge_env._rng.integers(0, 1000, size=10),
                                      expected.integers(0, 1000, size=10))
        self.assertFalse(fast._server_start)
        self.assertEqual(fast.reset_count, 4)
        self.assertIsNone(sim._prev_obs)
        self.assertIsNone(sim._prev_info)
        self.assertIsNone(sim._prev_action)

    def test_custom_restart_retains_explicit_seed_contract_and_clears_stale_state(self):
        observation = {"value": 7}
        old = FakeBridge()
        sim = FakeSim(old)
        fast = FakeFastReset(sim)
        sim.outcomes = [socket.timeout("bad reset"), observation]
        callback = mock.Mock(side_effect=lambda env: setattr(sim, "_bridge_env", FakeBridge(seed=1234)))
        wrapper = ResetRecovery(fast, restart=callback)
        self.assertIs(wrapper.reset(), observation)
        callback.assert_called_once_with(fast)
        self.assertEqual(old.closed, 0)
        self.assertFalse(fast._server_start)
        self.assertIsNone(sim._prev_obs)
        self.assertIsNone(sim._prev_info)

    def test_restart_failure_is_clear_and_bounded(self):
        env = FakeEnv([socket.timeout("reset failed")])
        error = ConnectionResetError("bridge construction unavailable")
        callback = mock.Mock(side_effect=error)
        with self.assertRaisesRegex(ResetRecoveryError, "rebuilding") as caught:
            ResetRecovery(env, restart=callback).reset()
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual((env.reset_calls, callback.call_count), (1, 1))

    def test_callback_logic_error_propagates(self):
        env = FakeEnv([socket.timeout("reset failed")])
        error = ValueError("invalid restart callback configuration")
        with self.assertRaises(ValueError) as caught:
            ResetRecovery(env, restart=mock.Mock(side_effect=error)).reset()
        self.assertIs(caught.exception, error)

    def test_step_timeout_is_never_replayed_by_reset_wrapper(self):
        env = FakeEnv([])
        callback = mock.Mock()
        wrapper = ResetRecovery(env, restart=callback)
        with self.assertRaises(socket.timeout):
            wrapper.step(1)
        self.assertEqual(env.step_calls, 1)
        callback.assert_not_called()

    def test_invalid_recovery_budget_is_rejected(self):
        for retries in (-1, 1.5, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                ResetRecovery(FakeEnv([]), retries=retries)


def native_recovery_audit(argv):
    """Inject one first-observation timeout into a live local MineDojo reset."""
    import argparse
    import gc
    import os
    from types import SimpleNamespace
    import torch
    import ruamel.yaml as yaml
    os.environ["MINEDOJO_HEADLESS"] = "1"
    from expr import make_env
    import tools
    from envs.recovery import _environment_chain, _owned_simulator
    from search_evidence import SEARCH_LABEL_KEYS

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh native recovery output directory")
    args.output.mkdir(parents=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    presets = yaml.YAML(typ="safe").load((root / "configs.yaml").read_text())
    values = {}
    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and key in dst:
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)
    for name in ("defaults", "minedojo", "apm_a", "apm_full", "apm_a_smoke"):
        merge(values, presets[name])
    values.update(task="minedojo_harvest_sand", target_item="sand",
                  logdir=str(args.output), results_dir=str(args.output),
                  name="reset_recovery_native", env_seed=0, seed=0,
                  device="cuda:0", minedojo_reset_retries=2,
                  episode_max_steps_override=32, compile=False, use_wandb=False)
    config = SimpleNamespace(**values)
    tools.set_seed_everywhere(config.seed)
    env = None
    before = recovered = stepped = None
    try:
        print("Native recovery: create ordinary train environment", flush=True)
        env = make_env(config, "train", 0)
        before = env.reset()
        assert np.asarray(before["is_first"]).all()
        sim = _owned_simulator(env)
        old = sim._bridge_env
        old_instances = list(old._instances)
        assert len(old_instances) == 1 and old_instances[0].running
        fast = next(item for item in _environment_chain(env)
                    if item.__class__.__name__ == "MinedojoSemifastResetWrapper")
        fast.reset_count = fast.reset_freq
        print("Native recovery: force ordinary hard reset with one injected Peek timeout", flush=True)
        with mock.patch.object(old, "_query_first_obs",
                side_effect=socket.timeout("injected native first-observation timeout")) as injected:
            recovered = env.reset()
        injected.assert_called_once()
        assert sim._bridge_env is not old
        assert old._already_closed
        assert all(not instance.running for instance in old_instances)
        assert len(sim._bridge_env._instances) == 1
        assert sim._bridge_env._instances[0].running
        assert np.asarray(recovered["is_first"]).all()
        assert not np.asarray(recovered["is_last"]).any()
        print("Native recovery: new owned client is live; execute one forward action", flush=True)
        action = np.eye(12, dtype=np.float32)[6]
        stepped, reward, done, info = env.step({"action": action})
        assert not done
        assert not np.asarray(stepped["is_first"]).any()
        assert set(before) == set(recovered) == set(stepped)
        for obs in (before, recovered, stepped):
            assert np.asarray(obs["image"]).shape == (64, 64, 3)
            assert np.asarray(obs["heatmap"]).shape in ((64, 64), (64, 64, 1))
            for key in SEARCH_LABEL_KEYS:
                assert key in obs and np.isfinite(np.asarray(obs[key])).all()
            assert float(obs["search_label_version"]) == 2.
        cache = {}
        initial = dict(recovered, reward=np.float32(0), discount=np.float32(1))
        tools.add_to_cache(cache, "native", initial)
        transition = dict(stepped, action=action, reward=np.float32(reward),
                          discount=np.float32(0 if done else 1))
        tools.add_to_cache(cache, "native", transition)
        assert {len(value) for value in cache["native"].values()} == {2}
        rows = [json.loads(line) for line in
                (args.output / "env_recovery.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["retry", "recovered"], rows
        assert env.reset_recoveries == 1
        report = dict(passed=True, injected_error="socket.timeout at _query_first_obs",
                      old_bridge_closed=True, new_bridge_live=True,
                      reset_recoveries=env.reset_recoveries,
                      successful_initial_resets=1, successful_recovered_resets=1,
                      real_agent_actions=1, replay_rows=2, reset_is_first=True,
                      step_is_first=False, image_shape=list(np.asarray(stepped["image"]).shape),
                      labels_present=list(SEARCH_LABEL_KEYS), events=rows)
        (args.output / "native_recovery.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.output / "observations.npz",
                            before=before["image"], recovered=recovered["image"],
                            after_step=stepped["image"])
        print("PASS: ordinary hard-reset timeout rebuilt only its client; real next action and Replay valid",
              args.output, flush=True)
    finally:
        if env is not None:
            env.close()
        env = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--minedojo":
        native_recovery_audit(sys.argv[2:])
    else:
        unittest.main(verbosity=2)
