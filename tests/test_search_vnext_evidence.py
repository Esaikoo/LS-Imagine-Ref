"""vNext actuator/evidence regressions; no game or privileged policy input."""
import ast
import uuid
import importlib.util
import pathlib
import sys
import unittest
import copy
import numpy as np
import torch
import gym

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_actuator import (initial_state, apply_command, apply_command_torch,
    rollout_commands, candidate_mask, CONTROL_KEYS, world_group_id)
from search_evidence import SearchEvidence, SearchEvidenceWrapper, empty_labels, VNEXT_LABEL_KEYS


def load_wrapper():
    spec = importlib.util.spec_from_file_location("ls_actuator_test_wrapper",
        ROOT / "envs/tasks/base/ls_imagine_wrapper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LSImagineWrapper


def observation(block="sand", distance=2., cell=0, center=True):
    n = 45
    blocks = np.full(n, block, dtype="<U24")
    distances = np.full(n, distance, np.float32)
    if not center:
        blocks[22] = "air"
        distances[22] = 32
    return {
        "rgb": np.zeros((3,64,64), np.uint8),
        "location_stats": {"pos": np.array([0.,64.,0.]), "yaw": 0., "pitch": 0.},
        "rays": {"block_name": blocks, "block_distance": distances,
                 "entity_name": np.full(n, "null"), "entity_distance": np.full(n, 32.),
                 "traced_block_x": np.full(n, cell*2), "traced_block_y": np.zeros(n),
                 "traced_block_z": np.zeros(n)}
    }


class FakeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})
        self.applied = []
        self.current = observation()
        self.reward = 0.
    def reset(self, **kwargs):
        return copy.deepcopy(self.current)
    def step(self, action):
        self.applied.append(copy.deepcopy(action))
        return copy.deepcopy(self.current), self.reward, False, {"real_done": False}


class ActuatorTests(unittest.TestCase):
    def test_full_wrapper_matches_original_controls(self):
        Wrapper = load_wrapper()
        old_env, new_env = FakeEnv(), FakeEnv()
        old, new = Wrapper(old_env), Wrapper(new_env, vnext=True)
        old.reset()
        initial = new.reset()
        np.testing.assert_array_equal(initial["log_actuator_after"], [0,0,0])
        self.assertEqual(initial["search_label_version"], 3)
        self.assertTrue(new.observation_space["search_label_version"].contains(initial["search_label_version"]))
        for command in np.random.RandomState(23).randint(12, size=250):
            old.step(int(command))
            obs, _, _, _ = new.step(int(command))
            a, b = old_env.applied[-1], new_env.applied[-1]
            for name in ("attack", "forward", "back", "left", "right", "jump", "use", "camera"):
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            self.assertEqual(obs["log_requested_action"], command)
            expected = np.array([a[k] for k in CONTROL_KEYS[:7]] + list(a["camera"]), np.float32)
            np.testing.assert_array_equal(obs["log_applied_control"], expected)
        self.assertNotIn("log_applied_control", old.observation_space.spaces)
        self.assertIn("search_center_interactable", new.observation_space.spaces)

    def test_numpy_torch_batched_parity(self):
        state = initial_state(7)
        tensor = torch.as_tensor(state)
        for command in np.random.RandomState(5).randint(12, size=(70,7)):
            applied, state = apply_command(command, state)
            actual, tensor = apply_command_torch(torch.as_tensor(command), tensor)
            np.testing.assert_array_equal(actual.numpy(), applied)
            np.testing.assert_array_equal(tensor.numpy(), state)

    def test_sticky_attack_does_not_cancel_existing_jump(self):
        _, state = apply_command(10, initial_state())
        applied, state = apply_command(1, state)
        np.testing.assert_array_equal(applied[[0,1,5]], [1,1,1])
        self.assertEqual(state[1], 8)

    def test_sticky_attack_override_parity(self):
        state = initial_state()
        tensor = torch.as_tensor(state)
        for command in (10,1,0,10,0,1,0):
            a,state = apply_command(command,state,sticky_attack=3)
            b,tensor = apply_command_torch(command,tensor,sticky_attack=3)
            np.testing.assert_array_equal(a,b.numpy())
            np.testing.assert_array_equal(state,tensor.numpy())

    def test_pitch_and_dedup_keep_different_end_state(self):
        state = np.array([0,9,0], np.float32)
        controls, _ = rollout_commands(np.array([0]), state)
        jump, _ = rollout_commands(np.array([10]), state)
        np.testing.assert_array_equal(controls,jump)
        self.assertTrue(candidate_mask(np.array([[0],[10]]),state).all())
        self.assertFalse(candidate_mask(np.array([[2]]),np.array([0,0,-70],np.float32))[0])

    def test_invalid_ids_do_not_wrap(self):
        for command in (-1,12,1.5):
            with self.assertRaises(ValueError):
                apply_command(command, initial_state())

    def test_error_transition_is_not_invented_failure_truth(self):
        env=FakeEnv()
        def fail(action):
            return observation(),0.,False,{"error":"transport"}
        env.step=fail
        wrapped=SearchEvidenceWrapper(env,target="sand",enabled=True,version=3)
        final=load_wrapper()(wrapped,vnext=True)
        final.reset()
        obs,_,done,_=final.step(6)
        self.assertTrue(done)
        self.assertFalse(obs["is_terminal"])
        self.assertEqual(obs["log_env_error"],1)
        self.assertFalse(any(v for k,v in obs.items() if k.startswith("search_") and k.endswith("_valid")))

    def test_repeat_not_silently_approximated(self):
        with self.assertRaises(ValueError):
            load_wrapper()(FakeEnv(), vnext=True, repeat=2)


class WorldGroupTests(unittest.TestCase):
    def load_reset_class(self):
        # Test the production class body against an exact small FastReset model,
        # avoiding Java/importing all visual backends for a counter regression.
        class Fast(gym.Wrapper):
            def __init__(self, env, **kwargs):
                super().__init__(env)
                self._server_start=False
            def reset(self):
                if not self._server_start:
                    self._server_start=True
                    return self.env.reset()
                return observation()
        tree=ast.parse((ROOT/"envs/tasks/minedojo/wrappers.py").read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MinedojoSemifastResetWrapper")
        module=ast.Module(body=[node],type_ignores=[])
        scope={"FastResetWrapper":Fast,"uuid":uuid,"world_group_id":world_group_id}
        exec(compile(module,"reset_class_regression","exec"),scope)
        return scope["MinedojoSemifastResetWrapper"]

    def test_fast_world_group_shared_hard_and_recovery_change_it(self):
        cls=self.load_reset_class()
        wrapper=cls(FakeEnv(),reset_freq=2)
        wrapper.reset(); a=wrapper._vnext_world_group
        wrapper.reset(); self.assertEqual(wrapper._vnext_world_group,a)
        wrapper.reset(); b=wrapper._vnext_world_group
        self.assertNotEqual(a,b)
        wrapper.reset(); self.assertEqual(wrapper._vnext_world_group,b)
        wrapper._server_start=False
        wrapper.reset()
        self.assertNotEqual(wrapper._vnext_world_group,b)
        self.assertEqual(wrapper._vnext_world_generation,3)

    def test_failed_reset_does_not_publish_group(self):
        wrapper=self.load_reset_class()(FakeEnv(),reset_freq=2)
        def fail():
            raise TimeoutError("fault injection")
        wrapper.env.reset=fail
        with self.assertRaises(TimeoutError):
            wrapper.reset()
        self.assertEqual(wrapper._vnext_world_generation,0)
        self.assertEqual(wrapper._vnext_world_group,0)

    def test_fixed_seed_groups_survive_wrapper_reset_and_resume(self):
        cls=load_wrapper()
        a=cls(FakeEnv(),vnext=True,vnext_world_seed=72)
        b=cls(FakeEnv(),vnext=True,vnext_world_seed="72")
        c=cls(FakeEnv(),vnext=True,vnext_world_seed=73)
        group=a.reset()["log_vnext_world_group"]
        self.assertEqual(b.reset()["log_vnext_world_group"],group)
        self.assertNotEqual(c.reset()["log_vnext_world_group"],group)
        for i in range(3):
            a.env._vnext_world_group=900+i
            self.assertEqual(a.reset()["log_vnext_world_group"],group)
        restored=cls(FakeEnv(),vnext=True,vnext_world_seed=72)
        self.assertEqual(restored.reset()["log_vnext_world_group"],group)

    def test_world_group_survives_float32_and_wrapper_logging(self):
        for generation in range(200):
            value=world_group_id("test-session",generation)
            self.assertEqual(int(np.float32(value)),value)
        lower=FakeEnv()
        lower._vnext_world_group=32401
        upper=load_wrapper()(lower,vnext=True)
        first=upper.reset()
        self.assertEqual(first["log_vnext_world_group"],32401)
        for _ in range(3):
            current,_,_,_=upper.step(0)
            self.assertEqual(current["log_vnext_world_group"],32401)
        lower._vnext_world_group=14522
        self.assertEqual(upper.reset()["log_vnext_world_group"],14522)


class EvidenceTests(unittest.TestCase):
    def test_center_same_target_joint(self):
        evidence = SearchEvidence("sand", version=3)
        far = evidence.observe(observation(distance=20), initial=True)
        self.assertEqual(far["search_visible"], 1)
        self.assertEqual(far["search_center_target"], 1)
        self.assertEqual(far["search_reachable"], 0)
        self.assertEqual(far["search_center_interactable"], 0)
        close = evidence.observe(observation(distance=2))
        self.assertEqual(close["search_center_interactable"],1)
        other = evidence.observe(observation(distance=2, center=False))
        self.assertEqual(other["search_visible"],1)
        self.assertEqual(other["search_center_target"],0)
        self.assertEqual(other["search_center_interactable"],0)

    def test_tool_unknown_not_positive(self):
        evidence = SearchEvidence("iron_ore", version=3)
        raw = observation(block="iron_ore")
        unknown = evidence.observe(raw, initial=True)
        self.assertEqual(unknown["search_reachable"],1)
        self.assertEqual(unknown["search_center_interactable_valid"],0)
        raw["inventory"]={"name":np.array(["iron_pickaxe"]),"quantity":np.array([1])}
        backpack_only=evidence.observe(raw)
        self.assertEqual(backpack_only["search_center_interactable_valid"],0)
        raw["equipment"]={"name":np.array(["wooden_pickaxe"]),"quantity":np.array([1])}
        illegal = evidence.observe(raw)
        self.assertEqual(illegal["search_center_interactable_valid"],1)
        self.assertEqual(illegal["search_center_interactable"],0)
        raw["equipment"]["name"][0]="iron_pickaxe"
        legal = evidence.observe(raw)
        self.assertEqual(legal["search_center_interactable"],1)

    def test_continuous_dwell_is_not_return(self):
        new = SearchEvidence("sand", version=3)
        old = SearchEvidence("sand")
        returns, dwells, legacy = [], [], []
        for t in range(20):
            a = new.observe(observation(),initial=t==0)
            b = old.observe(observation(),initial=t==0)
            returns.append(a["search_return"])
            dwells.append(a["search_dwell"])
            legacy.append(b["search_revisit"])
        self.assertEqual(sum(returns),0)
        self.assertGreater(sum(dwells),0)
        self.assertGreater(sum(legacy),0)

    def test_return_is_one_event(self):
        evidence = SearchEvidence("sand",version=3)
        evidence.observe(observation(cell=0),initial=True)
        for t in range(7):
            evidence.observe(observation(cell=1))
        result = evidence.observe(observation(cell=0))
        self.assertEqual(result["search_return_valid"],1)
        self.assertEqual(result["search_return"],1)
        self.assertEqual(result["search_revisit"],1)
        again = evidence.observe(observation(cell=0))
        self.assertEqual(again["search_return"],0)

    def test_unknown_gap_not_known_return_or_negative(self):
        evidence = SearchEvidence("sand",version=3)
        evidence.observe(observation(),initial=True)
        for _ in range(8):
            evidence.observe({})
        result = evidence.observe(observation())
        self.assertEqual(result["search_return_valid"],0)
        self.assertEqual(result["search_new_valid"],0)
        self.assertEqual(result["search_first_discovery"],0)
        # A fresh contiguous observation can prove continued dwelling next step.
        self.assertEqual(evidence.observe(observation())["search_return_valid"],1)

    def test_wrapper_progress_from_real_reward_and_blocked_mask(self):
        env = FakeEnv()
        wrapper = SearchEvidenceWrapper(env,target="sand",enabled=True,version=3)
        obs = wrapper.reset()
        self.assertEqual(obs["search_progress_valid"],0)
        env.reward=1.
        obs,_,_,_=wrapper.step({"attack":np.array(1)})
        self.assertEqual(obs["search_progress"],1)
        self.assertEqual(obs["search_progress_valid"],1)
        self.assertEqual(obs["search_blocked_valid"],0)
        env.reward=0.
        obs,_,_,_=wrapper.step({"forward":np.array(1)})
        self.assertEqual(obs["search_blocked_valid"],1)
        self.assertEqual(obs["search_blocked"],1)

    def test_disabled_audit_has_unknown_v3_labels(self):
        wrapper=SearchEvidenceWrapper(FakeEnv(),enabled=False,version=3)
        obs=wrapper.reset()
        self.assertTrue(set(VNEXT_LABEL_KEYS).issubset(obs))
        self.assertEqual(obs["search_label_version"],3)
        self.assertFalse(any(v for k,v in obs.items() if k.startswith("search_") and k.endswith("_valid")))

    def test_entity_hit_is_not_occluded_by_no_block_sentinel(self):
        raw=observation(block="air")
        raw["rays"]["block_distance"][:]=-1
        raw["rays"]["entity_name"]=np.full(45,"sheep")
        raw["rays"]["entity_distance"][:]=2
        raw["equipment"]={"name":np.array(["shears"]),"quantity":np.array([1])}
        result=SearchEvidence("wool",version=3).observe(raw,True)
        self.assertEqual(result["search_visible"],1)
        self.assertEqual(result["search_center_interactable"],1)
        raw["rays"]["block_name"]=np.full(45,"stone")
        raw["rays"]["block_distance"][:]=1
        blocked=SearchEvidence("wool",version=3).observe(raw,True)
        self.assertEqual(blocked["search_visible"],0)
        self.assertEqual(blocked["search_center_interactable"],0)

    def test_v2_shape_and_semantics_retained(self):
        old=empty_labels(2)
        self.assertNotIn("search_center_target",old)
        self.assertEqual(SearchEvidence("sand").observe(observation(),True)["search_label_version"],2)


if __name__ == "__main__":
    unittest.main()
