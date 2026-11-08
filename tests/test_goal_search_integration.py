"""vNext4 normal-entry regression with deterministic synthetic test labels."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_search_a import replay_batch
from test_spatial_integration import config, collected_episode
from apm.evidence import GOAL_LABEL_SHAPES, GOAL_LABEL_DTYPES
from apm.replay import sample_spatial_episodes
from expr import LS_Imagine, make_dataset
from training_state import prepare_resume, restore_training_state, save_training_checkpoint
import tools


def fixture_bank():
    pixels = np.full((2,64,64,3), 153, np.uint8)
    pixels[1,:,:,1] = 120
    entries = [dict(task_id="harvest_sand", split="train", world_seed=91, role="positive",
                    sha256=hashlib.sha256(frame.tobytes()).hexdigest()) for frame in pixels]
    manifest = dict(version=1, task_id="harvest_sand", target_class="sand", entries=entries,
                    pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    return dict(manifest=manifest, pixels=pixels)


def agent_for(cfg, bank=None):
    spaces = gym.spaces.Dict({
        "image": gym.spaces.Box(0,255,(64,64,3),np.uint8),
        "heatmap": gym.spaces.Box(0,255,(64,64,1),np.uint8)})
    agent = LS_Imagine(spaces, gym.spaces.Discrete(12), cfg, SimpleNamespace(step=48),
                       None, goal_bank=fixture_bank() if bank is None else bank)
    agent.requires_grad_(False)
    return agent


def goal_config(**updates):
    values = dict(task="minedojo_harvest_sand", target_item="sand", goal_ref_enabled=True,
                  goal_search_enabled=True, goal_ref_dim=32, goal_ref_batch_frames=16,
                  goal_ref_use_raw_image=False, vnext_spatial_fix_version=3,
                  goal_ref_train_every=1, check_obs_enabled=False)
    values.update(updates)
    return config(**values)


class GoalSearchIntegrationTests(unittest.TestCase):
    def test_online_encoder_and_training_batch_use_identical_fusion_without_oracles(self):
        agent = agent_for(goal_config())
        # Deterministic calibrated-output fixture. No assertion of learned quality.
        with torch.no_grad():
            agent._goal_reference.reliability[0] = .9
            agent._goal_reference.match[-1].weight.zero_()
            agent._goal_reference.match[-1].bias.fill_(-2.)
        batch = replay_batch()
        batch["heatmap"].fill(210)
        obs = {key: value[:,0] for key,value in batch.items() if key != "action"}
        extra = dict(obs, goal_region_gt=np.ones((1,16,16,1)),
                     goal_region_valid=np.ones((1,16,16,1)), goal_visible_gt=np.ones(1),
                     search_pose_delta=np.full((1,3),1.e8),
                     location_stats={"pos": np.full((1,3),1.e8)})
        inputs = []
        hook = agent._wm.encoder.register_forward_pre_hook(
            lambda _, args: inputs.append(args[0]["heatmap"].clone()))
        rng = torch.get_rng_state()
        left, _ = agent._policy(obs,None,False)
        torch.set_rng_state(rng)
        right, _ = agent._policy(extra,None,False)
        hook.remove()
        for key in ("action", "log_goal_context", "log_goal_heatmap_policy", "log_spatial_pose"):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        model_batch = agent._goal_model_batch(batch)
        self.assertTrue(np.all(batch["heatmap"] == 210))
        self.assertTrue((left["log_goal_heatmap_policy"] < 210).all())
        torch.testing.assert_close(left["log_goal_heatmap_policy"], model_batch["heatmap"][:,0])
        torch.testing.assert_close(inputs[0], model_batch["heatmap"][:,0,:,:,None].float()/255.)
        self.assertFalse(inputs[0].requires_grad)
        self.assertEqual(int(agent._goal_reference.updates), 0)

    def test_causal_replay_old_unknown_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = goal_config(logdir=root/"source", traindir=root/"source"/"train_eps",
                              video_pred_log=False)
            cfg.traindir.mkdir(parents=True)
            agent = agent_for(cfg)
            episode = collected_episode(agent, 24)
            shape = (len(episode["reward"]),)
            for key, tail in GOAL_LABEL_SHAPES.items():
                episode[key] = np.zeros((*shape,*tail), GOAL_LABEL_DTYPES[key])
            episode["goal_label_version"][:] = 1
            episode["goal_region_gt"][:,8,8,0] = 1
            episode["goal_region_valid"][:,8:10,8,0] = 1
            episode["goal_visible_gt"][:] = 1
            episode["goal_visible_valid"][:] = 1
            old = {key: value for key,value in episode.items() if not key.startswith(("goal_", "log_goal_"))}
            unknown = next(sample_spatial_episodes({"old": old},12,history_slots=cfg.vnext_slots,goal_enabled=True))
            self.assertFalse(unknown["goal_region_valid"].any())
            self.assertFalse(unknown["vnext_goal_context_valid"].any())
            row = next(sample_spatial_episodes({"new": episode},12,history_slots=cfg.vnext_slots,goal_enabled=True))
            for index, source in enumerate(row["vnext_observation_index"]):
                if source + 1 < shape[0]:
                    np.testing.assert_equal(row["vnext_goal_context"][index], episode["log_goal_context"][source+1])
            tools.save_episodes(cfg.traindir, {"fixture":episode})
            agent._train(next(make_dataset({"fixture":episode}, cfg)))
            self.assertGreater(int(agent._goal_reference.updates), 0)
            checkpoint_path = save_training_checkpoint(agent,cfg,100000)
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            incoming = goal_config(logdir=root/"new", resume_from=str(checkpoint_path), steps=128,
                                   video_pred_log=True)
            plan = prepare_resume(incoming)
            self.assertTrue(incoming.video_pred_log)
            restored = agent_for(incoming, plan["checkpoint"]["goal_reference_bank"])
            restore_training_state(restored,plan,SimpleNamespace(step=0))
            self.assertEqual(restored._goal_reference.bank_fingerprint, agent._goal_reference.bank_fingerprint)
            self.assertEqual(int(restored._goal_reference.updates), int(agent._goal_reference.updates))
            self.assertTrue(restored._goal_reference.optimizer.state)
            self.assertEqual(checkpoint_hash, hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                prepare_resume(goal_config(logdir=root/"old_attempt", resume_from=str(checkpoint_path),
                                           goal_ref_enabled=False, goal_search_enabled=False))


if __name__ == "__main__":
    unittest.main()
