"""vNext actuator/evidence regressions; no game or privileged policy input."""
import ast
import uuid
import importlib.util
import pathlib
import sys
import unittest
import copy
import json
from types import SimpleNamespace
import tempfile
from unittest import mock
import numpy as np
import torch
import gym

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apm.actuator import (initial_state, apply_command, apply_command_torch,
    rollout_commands, candidate_mask, CONTROL_KEYS, world_group_id)
from apm.evidence import (SearchEvidence, SearchEvidenceWrapper, empty_labels,
                          VNEXT_LABEL_KEYS, GOAL_LABEL_SHAPES, lidar_rays, goal_ray_cells)


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


def goal_observation(block="sand"):
    obs = observation(block="air", distance=32.)
    obs["rgb"] = np.zeros((3, 160, 256), np.uint8)
    obs["rgb"][0] = 53
    rays = obs["rays"]
    offsets = np.asarray(lidar_rays())
    rays["ray_pitch"], rays["ray_yaw"] = offsets[:, 0], offsets[:, 1]
    rays["block_name"][22] = block
    rays["block_distance"][22] = 2.
    rays["traced_block_y"][22] = 65
    rays["traced_block_z"][22] = 2
    rays["is_solid"] = np.ones(45, bool)
    obs["inventory"] = dict(name=np.array(["sand"]), quantity=np.array([0]))
    return obs


class GoalEvidenceTests(unittest.TestCase):
    def test_only_projected_rays_supervise_sparse_regions(self):
        obs = goal_observation()
        target = SearchEvidence("sand", version=3, goal_supervision=True)
        labels = target.observe(obs, initial=True)
        self.assertEqual(int(labels["goal_region_valid"].sum()), 1)
        self.assertEqual(int(labels["goal_region_gt"][8,8,0]), 1)
        self.assertEqual(int(labels["goal_visible_gt"]), 1)
        self.assertEqual(int(labels["goal_visible_valid"]), 1)
        self.assertEqual(int(labels["goal_instance_group"][8,8,0]), 1)
        # The same block can be viewed from different positions.
        obs["location_stats"]["pos"][0] += .25
        self.assertEqual(int(target.observe(obs)["goal_instance_group"][8,8,0]), 1)
        obs["rays"]["traced_block_x"][22] = 1
        self.assertEqual(int(target.observe(obs)["goal_instance_group"][8,8,0]), 2)

    def test_miss_does_not_teach_absence_and_opaque_first_hit_teaches_region_only(self):
        target = SearchEvidence("sand", version=3, goal_supervision=True)
        for block in ("air", "glass"):
            labels = target.observe(goal_observation(block))
            self.assertFalse(labels["goal_region_valid"].any())
            self.assertFalse(labels["goal_visible_valid"])
        stone = target.observe(goal_observation("stone"))
        self.assertTrue(stone["goal_region_valid"][8,8,0])
        self.assertFalse(stone["goal_region_gt"].any())
        self.assertFalse(stone["goal_visible_valid"])
        fabricated = goal_observation()
        fabricated["rays"]["traced_block_z"][22] = 32
        fabricated["rays"]["block_distance"][22] = 31.
        self.assertFalse(target.observe(fabricated)["goal_region_valid"].any())

    def test_projection_uses_native_aspect_and_yaw_sign(self):
        obs = goal_observation()
        cells = goal_ray_cells(obs,45)
        np.testing.assert_array_equal(cells[22], [8,8])
        self.assertLess(cells[23,1], 8)
        self.assertGreater(cells[21,1], 8)
        self.assertTrue((goal_ray_cells({"rgb": obs["rgb"]}, 45) == -1).all())

    def test_extra_train_rays_expand_projected_labels_without_changing_a(self):
        original = goal_observation("stone")
        obs = copy.deepcopy(original)
        angles = np.asarray(lidar_rays(goal=True))
        rays = obs["rays"]
        count = len(angles)
        self.assertEqual(count, 77)
        self.assertEqual(lidar_rays(goal=True)[:45], lidar_rays())
        for key, values in rays.items():
            rays[key] = np.pad(values, (0, count-45), mode="edge")
        rays["ray_pitch"], rays["ray_yaw"] = angles[:, 0], angles[:, 1]
        cells = goal_ray_cells(obs, count)
        self.assertGreaterEqual(len({tuple(cell) for cell in cells if cell[0] >= 0}), 12)
        index = next(i for i in range(45, count) if cells[i, 0] >= 0)
        rays["block_name"][index] = "sand"
        rays["block_distance"][index] = 2.
        rays["traced_block_y"][index] = 65
        rays["traced_block_z"][index] = 2
        target = SearchEvidence("sand", version=3, goal_supervision=True)
        baseline = target.observe(original, initial=True)
        result = target.observe(obs, initial=True)
        for key in VNEXT_LABEL_KEYS:
            np.testing.assert_equal(result[key], baseline[key], err_msg=key)
        self.assertTrue(result["goal_visible_gt"])
        self.assertTrue(result["goal_region_gt"][tuple(cells[index])][0])

    def test_visibility_remains_known_when_projected_cell_varies_in_fov(self):
        obs = goal_observation("air")
        index = 13
        self.assertTrue((goal_ray_cells(obs, 45)[index] < 0).all())
        rays = obs["rays"]
        rays["block_name"][index] = "sand"
        rays["block_distance"][index] = 2.
        rays["traced_block_y"][index] = 65
        rays["traced_block_z"][index] = 2
        result = SearchEvidence("sand", version=3, goal_supervision=True).observe(obs)
        self.assertFalse(result["goal_region_valid"].any())
        self.assertTrue(result["goal_visible_valid"])
        self.assertTrue(result["goal_visible_gt"])

    def test_target_categories_and_sheep_do_not_fabricate_instance_identity(self):
        for target, block in (("sand","red_sand"),("log","birch_log"),
                              ("water_bucket","water"),("iron_ore","iron_ore")):
            labels = SearchEvidence(target, version=3, goal_supervision=True).observe(goal_observation(block))
            self.assertTrue(labels["goal_visible_gt"], target)
        obs = goal_observation("stone")
        obs["rays"]["entity_name"] = obs["rays"]["entity_name"].astype("<U24")
        obs["rays"]["entity_name"][22] = "EntitySheep"
        obs["rays"]["entity_distance"][22] = 1.
        labels = SearchEvidence("wool", version=3, goal_supervision=True).observe(obs)
        self.assertTrue(labels["goal_visible_gt"])
        self.assertFalse(labels["goal_instance_group"].any())

    def test_training_only_wrapper_image_fold_and_actual_outcomes(self):
        from apm.replay import identity
        env = FakeEnv()
        env.current = goal_observation()
        wrapper = load_wrapper()(SearchEvidenceWrapper(env, target="sand", enabled=True,
            version=3, goal_supervision=True), vnext=True, goal_reference=True, vnext_world_seed=15)
        initial = wrapper.reset()
        self.assertEqual(initial["goal_image"].shape, (96,96,3))
        self.assertTrue(initial["goal_image_valid"])
        self.assertTrue((initial["goal_image"][...,0] == 53).all())
        self.assertEqual(initial["goal_holdout"], identity(str(wrapper._world_group))%5 == 0)
        self.assertFalse(initial["goal_interaction_valid"])
        missed, _, _, _ = wrapper.step(1)
        self.assertTrue(missed["goal_interaction_valid"])
        self.assertFalse(missed["goal_interaction_outcome"])
        env.current["inventory"]["quantity"][0] = 1
        picked_up, _, _, _ = wrapper.step(6)
        self.assertTrue(picked_up["goal_interaction_valid"])
        self.assertTrue(picked_up["goal_interaction_outcome"])
        self.assertEqual(picked_up["goal_holdout"], initial["goal_holdout"])
        env.current["goal_visible_gt"] = 1
        eval_wrapper = load_wrapper()(SearchEvidenceWrapper(env, target="sand", enabled=False,
            version=3), vnext=True, goal_reference=True)
        evaluation = eval_wrapper.reset()
        for key in GOAL_LABEL_SHAPES:
            self.assertFalse(evaluation[key].any(), key)
        self.assertNotIn("rays", evaluation)
        self.assertNotIn("location_stats", evaluation)

    def test_reference_capture_and_build_preserve_training_provenance(self):
        from scripts import build_goal_references as refs

        class CaptureEnv:
            def __init__(self, config):
                self.env = FakeEnv()
                self.env.current = goal_observation()
                self.env.current["rgb"][0] = 40 if config.env_seed == 50101 else 140
                self.wrapper = load_wrapper()(SearchEvidenceWrapper(self.env, target="sand", enabled=True,
                    version=3, goal_supervision=True), vnext=True, goal_reference=True,
                    vnext_world_seed=config.env_seed)

            def reset(self):
                return self.wrapper.reset()

            def step(self, action):
                return self.wrapper.step(int(np.argmax(action["action"])))

            def close(self):
                pass

        def make_env(config, mode, index):
            self.assertEqual(mode, "train")
            self.assertEqual(index, 0)
            self.assertTrue(config.goal_ref_enabled)
            return CaptureEnv(config)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            capture = root/"capture"
            with mock.patch("expr.make_env", side_effect=make_env):
                refs.collect(SimpleNamespace(task="harvest_sand", collect_world_seeds=[50101,50102],
                    output=capture, capture_commands=[6], capture_steps=2, minedojo_gradle_offline=True))
            paths = list((capture/"train_eps").glob("*.npz"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data["goal_image"].shape, (3,96,96,3))
                    self.assertEqual(len({len(data[key]) for key in data.files}), 1)
                    self.assertEqual(data["log_actuator_valid"].tolist(), [0,1,1])
            bank = root/"bank"
            refs.build(SimpleNamespace(task="harvest_sand", train_replay=[capture/"train_eps"],
                verified_train_archive=[], annotations=None, world_seeds=None, count=2, min_worlds=2, output=bank))
            entries = json.loads((bank/"manifest.json").read_text())["entries"]
            self.assertEqual({entry["world_seed"] for entry in entries}, {50101,50102})
            self.assertTrue(all(entry["split"] == "train" for entry in entries))
            self.assertTrue(all(refs.file_sha(bank/entry["path"]) == entry["sha256"] for entry in entries))
            (root/"eval_eps").mkdir()
            invalid = root/"eval_eps"/"bad.npz"
            invalid.touch()
            with self.assertRaisesRegex(ValueError, "Evaluation"):
                refs.sources_from_args([], [invalid], "harvest_sand")


class ActuatorTests(unittest.TestCase):
    def test_full_wrapper_matches_original_controls(self):
        Wrapper = load_wrapper()
        old_env, new_env = FakeEnv(), FakeEnv()
        old, new = Wrapper(old_env), Wrapper(new_env, vnext=True)
        old.reset()
        initial = new.reset()
        self.assertEqual(float(initial["log_actuator_valid"]), 0.)
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
            self.assertEqual(float(obs["log_actuator_valid"]), 1.)
        self.assertNotIn("log_applied_control", old.observation_space.spaces)
        self.assertIn("search_center_interactable", new.observation_space.spaces)

    def test_control_validity_marks_noop_error_and_reset_separately(self):
        env = FakeEnv()
        wrapped = load_wrapper()(env, vnext=True)
        self.assertEqual(float(wrapped.reset()["log_actuator_valid"]), 0.)
        obs, _, _, _ = wrapped.step(0)
        self.assertFalse(obs["log_applied_control"].any())
        self.assertEqual(float(obs["log_actuator_valid"]), 1.)
        def fail(action):
            return observation(), 0., False, {"error": "transport"}
        env.step = fail
        obs, _, done, _ = wrapped.step(6)
        self.assertTrue(done)
        self.assertEqual(float(obs["log_actuator_valid"]), 0.)
        self.assertEqual(float(obs["log_env_error"]), 1.)
        self.assertEqual(float(wrapped.reset()["log_actuator_valid"]), 0.)

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
