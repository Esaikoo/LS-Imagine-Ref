"""Spatial prediction/control replay, leakage and checkpoint integration."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tests"))
from test_search_a import make_agent, replay_batch
from test_search_vnext_integration import config as base_config
from apm.evidence import VNEXT_LABEL_KEYS, SPATIAL_LABEL_SHAPES
from apm.outcome import _hash_int, check_vnext_checkpoint
from apm.replay import sample_spatial_episodes
from expr import make_dataset
from training_state import save_training_checkpoint, prepare_resume, restore_training_state
import tools

torch.set_num_threads(2)


def config(**updates):
    values = dict(vnext_spatial_enabled=True, vnext_map_size=35,
        vnext_map_patch=21, vnext_pose_hidden=32, vnext_pose_min_samples=1,
        vnext_visual_enabled=False, batch_size=2, batch_length=12,
        episode_max_steps=64, episode_max_steps_override=64)
    values.update(updates)
    return base_config(**values)


def collected_episode(agent, length=40):
    raw = replay_batch(length+1)
    shape = (1, length+1)
    for key in VNEXT_LABEL_KEYS:
        raw.setdefault(key, np.zeros(shape, np.float32))
    for key, tail in SPATIAL_LABEL_SHAPES.items():
        raw[key] = np.zeros((*shape, *tail), np.float32)
    raw["search_label_version"][:] = 3
    for name in ("visible", "center_target", "reachable", "center_interactable", "progress",
                 "new", "discovery", "first_discovery", "revisit", "blocked"):
        raw["search_"+name+"_valid"] = np.ones(shape, np.float32)
    raw["search_pose_valid"][:, 1:] = 1
    raw["is_last"][:, -1] = True
    raw["success"] = np.zeros(shape, np.float32)
    raw["first_success_step"] = np.full(shape, -1, np.float32)
    raw["log_actuator_valid"] = np.zeros(shape, np.float32)
    raw["log_applied_control"] = np.zeros((*shape, 9), np.float32)
    raw["log_actuator_before"] = np.zeros((*shape, 3), np.float32)
    raw["log_actuator_after"] = np.zeros((*shape, 3), np.float32)
    cache = {}
    observation = lambda t: {k: v[:, t].copy() for k, v in raw.items() if k != "action"}
    initial = {k: v[0].copy() for k, v in observation(0).items()}
    initial["action"] = np.zeros(12, np.float32)
    tools.add_to_cache(cache, "spatial", initial)
    state = None
    for t in range(length):
        out, state = agent._policy(observation(t), state, training=True)
        transition = {k: v[0].copy() for k, v in observation(t+1).items()}
        transition.update({k: v[0].detach().cpu().numpy() for k, v in out.items()})
        transition["log_applied_control"] = transition["log_vnext_applied"].copy()
        transition["log_actuator_before"] = transition["log_vnext_actuator_before"].copy()
        transition["log_actuator_after"] = transition["log_vnext_actuator_after"].copy()
        transition["log_actuator_valid"] = np.float32(1)
        tools.add_to_cache(cache, "spatial", transition)
    episode = {k: np.asarray(v) for k, v in cache["spatial"].items()}
    group = next(i for i in range(100) if _hash_int(str(i)) % 5 != 0)
    episode["log_vnext_world_group"] = np.full(length+1, group, np.int64)
    return episode


class SpatialIntegrationTests(unittest.TestCase):
    def test_clearance_endpoint_obstruction_uses_real_movement_label(self):
        cfg = config(batch_size=1,batch_length=4)
        agent = make_agent(cfg)
        ep = {key:value[:4].copy() for key,value in collected_episode(agent,12).items()}
        ep["log_vnext_version"][:] = 2
        ep["log_vnext_decision"][:] = 0; ep["log_vnext_decision"][1] = 1
        ep["log_vnext_horizon"][1] = 3
        ep["log_vnext_choice"][:] = 7; ep["log_vnext_phase"][:] = 3
        ep["log_vnext_option_id"][:] = 1
        ep["log_vnext_plan_commands"][1] = [1,1,6]
        ep["log_requested_action"] = np.array([0,1,1,6])
        ep["log_applied_control"][:] = 0
        ep["log_applied_control"][1:3,0] = 1
        ep["log_applied_control"][3,1] = 1
        ep["log_actuator_after"][:] = 0
        ep["search_blocked"][:] = [0,1,0,0]
        ep["search_blocked_valid"][:] = [0,0,0,1]
        ep["is_first"][:] = [True,False,False,False]
        ep["is_last"][:] = False; ep["is_terminal"][:] = False
        data = {key:value[None] for key,value in ep.items()}
        data.update(vnext_spatial_patch=np.zeros((1,4,21,21,6),np.uint8),
                    vnext_spatial_pose=np.zeros((1,4,4),np.float32),
                    vnext_spatial_valid=np.ones((1,4),np.float32))
        feature_dim = cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        _,_,records,_,_ = agent._vnext.training_loss(torch.zeros(1,4,feature_dim),data)
        record = next(row for row in records if row[0] == 3)
        self.assertEqual(float(record[2][0,0,6]),0.)
        self.assertTrue(bool(record[3][0,0,6]))

    def test_online_cache_replay_and_complete_training_update(self):
        cfg = config()
        agent = make_agent(cfg)
        ep = collected_episode(agent)
        self.assertEqual({len(v) for v in ep.values()}, {41})
        self.assertEqual(ep["log_spatial_patch"].dtype, np.uint8)
        data = next(make_dataset({"spatial": ep}, cfg))
        self.assertEqual(data["vnext_spatial_patch"].shape, (2, 12, 21, 21, 6))
        before = agent._vnext.b.pose.fusion[-1].bias.detach().clone()
        encoder = next(agent._vnext.b.encoder.parameters()).detach().clone()
        agent._train(data)
        self.assertTrue(agent._metrics["B/update_applied"][-1])
        self.assertGreater(agent._metrics["B/pose_samples"][-1], 0)
        self.assertFalse(torch.equal(before, agent._vnext.b.pose.fusion[-1].bias))
        self.assertFalse(torch.equal(encoder, next(agent._vnext.b.encoder.parameters())))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in agent._metrics.values()))

    def test_pose_predecessor_and_outgoing_map_row_are_real_and_aligned(self):
        agent = make_agent(config())
        ep = collected_episode(agent, 22)
        for i in range(len(ep["image"])):
            ep["image"][i].fill(i)
            ep["log_spatial_patch"][i].fill(i)
            ep["log_spatial_pose"][i] = [i, 0, 0, 0]
            ep["log_applied_control"][i].fill(i)
        generator = sample_spatial_episodes({"spatial": ep}, 32, seed=7)
        nonzero_window_starts = 0
        for _ in range(4):
            data = next(generator)
            for row, index in enumerate(data["vnext_observation_index"]):
                if index > 0:
                    np.testing.assert_array_equal(data["vnext_pose_prev_image"][row], ep["image"][index-1])
                    np.testing.assert_array_equal(data["vnext_pose_control"][row], ep["log_applied_control"][index])
                    self.assertEqual(data["vnext_pose_prev_valid"][row], 1)
                    if row == 0:
                        self.assertTrue(data["is_first"][row])
                        self.assertFalse(data["vnext_real_first"][row])
                        nonzero_window_starts += 1
                else:
                    self.assertEqual(data["vnext_pose_prev_valid"][row], 0)
                if index+1 < len(ep["reward"]):
                    np.testing.assert_array_equal(data["vnext_spatial_patch"][row], ep["log_spatial_patch"][index+1])
                    self.assertEqual(float(data["vnext_spatial_pose"][row, 0]), index+1)
                else:
                    self.assertEqual(data["vnext_spatial_valid"][row], 0)
        self.assertGreater(nonzero_window_starts, 0)

    def test_coordinates_and_training_targets_cannot_change_policy(self):
        cfg = config()
        agent = make_agent(cfg)
        raw = replay_batch(12)
        obs = {k: v[:, 0] for k, v in raw.items() if k != "action"}
        noise = dict(obs)
        noise.update(search_pose_delta=np.full((1, 3), 1.e9, np.float32),
                     search_pose_valid=np.ones(1), log_spatial_gt_pose=np.full((1, 3), 1.e9),
                     location_stats={"pos": np.full((1, 3), 1.e9)},
                     search_area_revisit=np.ones(1))
        rng = torch.get_rng_state()
        left, left_state = agent._policy(obs, None, training=False)
        torch.set_rng_state(rng)
        right, right_state = agent._policy(noise, None, training=False)
        for key in ("action", "log_spatial_patch", "log_spatial_pose", "log_spatial_summary"):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        torch.testing.assert_close(left_state[2]["pose"], right_state[2]["pose"], rtol=0, atol=0)

    def test_mixed_prefill_learned_formal_batch_keeps_pose_training(self):
        cfg = config(batch_size=16, batch_length=32)
        agent = make_agent(cfg)
        learned = collected_episode(agent, 45)
        prefill = {k: v.copy() for k, v in learned.items()
                   if not k.startswith(("log_vnext_", "log_spatial_", "log_search_"))}
        replay = {"prefill": prefill}
        stream = make_dataset(replay, cfg)
        first = next(stream)
        self.assertEqual(float(first["vnext_spatial_valid"].sum()), 0.)
        self.assertGreater(float(first["vnext_pose_prev_valid"].sum()), 0.)
        agent._train(first)
        self.assertGreater(agent._metrics["B/pose_samples"][-1], 0)
        self.assertEqual(float(agent._metrics["A/current_loss"][-1]), 0.)
        replay["learned"] = learned
        data = next(stream)
        self.assertEqual(set(first), set(data))
        self.assertEqual(data["vnext_spatial_patch"].shape, (16, 32, 21, 21, 6))
        agent._train(data)
        self.assertGreater(agent._metrics["B/pose_samples"][-1], 0)
        self.assertTrue(np.isfinite(float(agent._metrics["A/prediction_loss"][-1])))

    def test_version_two_checkpoint_resume_and_optimizer_roundtrip(self):
        cfg = config()
        agent = make_agent(cfg)
        ep = collected_episode(agent, 20)
        with tempfile.TemporaryDirectory() as directory:
            cfg.logdir = Path(directory)/"source"
            cfg.traindir = cfg.logdir/"train_eps"
            cfg.traindir.mkdir(parents=True)
            tools.save_episodes(cfg.traindir, {"spatial": ep})
            agent._train(next(make_dataset({"spatial": ep}, cfg)))
            path = save_training_checkpoint(agent, cfg, 100000)
            checkpoint = torch.load(path, map_location="cpu")
            check_vnext_checkpoint(checkpoint, True)
            self.assertEqual(checkpoint["search_vnext_version"], 2)
            self.assertEqual(checkpoint["vnext_memory_version"], 2)
            self.assertEqual(checkpoint["pose_version"], 1)
            self.assertEqual(checkpoint["spatial_map_version"], 2)
            resume_cfg = copy.deepcopy(cfg)
            resume_cfg.resume_from = str(path)
            resume_cfg.logdir = str(Path(directory)/"resume")
            resume_cfg.traindir = resume_cfg.evaldir = None
            resume_cfg.steps = 1000
            plan = prepare_resume(resume_cfg)
            other = make_agent(resume_cfg, seed=71)
            restore_training_state(other, plan, SimpleNamespace(step=0))
            for key, value in agent.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(len(agent._vnext._b_opt.state), len(other._vnext._b_opt.state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
