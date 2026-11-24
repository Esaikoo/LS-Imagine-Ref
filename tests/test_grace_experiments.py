"""GRACE training-time architecture, data isolation and checkpoint contracts."""
from pathlib import Path
import importlib
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import ruamel.yaml as yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grace.experiments import MODULES, module_manifest, validate_training_mode
from grace.goal_guidance import build_policy_observation
from grace.outcome import restore_prediction_config
from test_goal_search_integration import agent_for, goal_config
from test_goal_guidance import MatchedRegions, obs as goal_obs
from test_search_a import replay_batch
from test_spatial_integration import collected_episode
from expr import make_dataset
from training_state import prepare_resume, save_training_checkpoint, validate_goal_checkpoint
import tools

PRESETS = yaml.YAML(typ="safe").load((ROOT / "configs.yaml").read_text())


def config(mode, **kwargs):
    return goal_config(**dict(PRESETS[mode], **kwargs))


class GraceExperimentTests(unittest.TestCase):
    def test_presets_and_shared_import_objects(self):
        historical = PRESETS["apm_vnext4"]
        self.assertEqual(dict(historical, experiment_mode="grace_full"), PRESETS["grace_full"])
        for mode in MODULES:
            validate_training_mode(vars(config(mode)))
        for path in (ROOT / "grace").rglob("*.py"):
            if path.stem in ("__init__", "experiments"):
                continue
            name = ".".join(path.relative_to(ROOT / "grace").with_suffix("").parts)
            self.assertIs(importlib.import_module("apm." + name), importlib.import_module("grace." + name))
        from apm.outcome import SearchVNext
        from grace.outcome import PredictiveMemorySystem
        self.assertIs(SearchVNext, PredictiveMemorySystem)

    def test_launcher_and_conflicting_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "python"
            capture.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            capture.chmod(0o755)
            env = dict(os.environ, PATH=tmp+":"+os.environ["PATH"], LOG_ROOT=tmp, NTFY_TOPIC="")
            env.pop("A_MODE", None)
            env.pop("EXPERIMENT_MODE", None)
            def launch(**extra):
                return subprocess.run(["sh", "scripts/train.sh", "harvest_sand"], cwd=ROOT,
                                      env=dict(env, **extra), capture_output=True, text=True)
            for mode in MODULES:
                run = launch(EXPERIMENT_MODE=mode)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("\n"+mode+"\n", run.stdout)
            self.assertIn("\napm_vnext\n", launch().stdout)
            self.assertIn("\napm_a\n", launch(A_MODE="full").stdout)
            self.assertNotEqual(launch(A_MODE="full", EXPERIMENT_MODE="grace_full").returncode, 0)

    def test_checkpoint_mode_and_optimizer_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode, (a, b, c, d, _) in MODULES.items():
                with self.subTest(mode=mode):
                    cfg = config(mode, logdir=Path(tmp)/mode)
                    agent = agent_for(cfg)
                    self.assertEqual(agent._vnext is not None, a)
                    self.assertEqual(agent._vnext is not None and agent._vnext.b is not None, b)
                    self.assertEqual(agent._vnext_control is not None, c)
                    ck = torch.load(save_training_checkpoint(agent, cfg, 100000), map_location="cpu")
                    self.assertEqual(ck["experiment_modules"], module_manifest(mode))
                    self.assertEqual(len(ck["optims_state_dict"]), 3+sum((a,b,c,d)))
                    fresh = {}
                    restore_prediction_config(fresh, ck)
                    validate_goal_checkpoint(ck, fresh, evaluation=True)
                    self.assertEqual(fresh["experiment_mode"], mode)
                    restored = agent_for(cfg, ck["goal_reference_bank"])
                    restored.load_state_dict(ck["agent_state_dict"], strict=True)
                    tools.recursively_load_optim_state_dict(restored, ck["optims_state_dict"])
                    for other in MODULES.keys() - {mode}:
                        with self.assertRaises(ValueError):
                            validate_goal_checkpoint(ck, vars(config(other)))
                        with self.assertRaises(ValueError):
                            restore_prediction_config({"experiment_mode":other}, ck)
                    with self.assertRaises(ValueError):
                        validate_goal_checkpoint(dict(ck,experiment_modules={}), vars(cfg))
                    missing = dict(ck["optims_state_dict"])
                    missing.pop("_goal_reference.optimizer")
                    with self.assertRaises(ValueError):
                        validate_goal_checkpoint(dict(ck,optims_state_dict=missing), vars(cfg))

    def test_output_failure_isolated_to_named_experiment_sink(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tools, "SummaryWriter") as writer:
            root = Path(tmp)
            log = tools.Logger(config("grace_full", goal_ref_enabled=False), root, 3)
            backend = writer.return_value
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                (root/"metrics.jsonl").mkdir()
                log.scalar("first_sample", 1.)
                log.write()
                backend.add_scalar.assert_called_with("scalars/first_sample", 1., 3)
                (root/"metrics.jsonl").rmdir()
                log.step = 4
                backend.add_scalar.side_effect = OSError("event writer unavailable")
                log.scalar("second_sample", 2.)
                log.write()
                log.finish()
            self.assertIn("metrics.jsonl export failed", errors.getvalue())
            self.assertIn("event writer unavailable", errors.getvalue())
            self.assertEqual(json.loads((root/"metrics.jsonl").read_text()), {"step":4, "second_sample":2.})
            self.assertFalse(log._scalars or log._videos or log._images)

    def test_resume_from_historical_preset_preserves_requested_full_identity(self):
        from test_training_state import replay
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = config("grace_full", logdir=root/"old", traindir=root/"old"/"train_eps")
            old.experiment_mode = ""
            replay(old.traindir, "episode", 8)
            path = save_training_checkpoint(agent_for(old), old, 100000)
            requested = config("grace_full", logdir=root/"new", traindir=None, evaldir=None,
                               resume_from=str(path), steps=100)
            plan = prepare_resume(requested)
            self.assertEqual(requested.experiment_mode, "grace_full")
            self.assertEqual(plan["replay_steps"], 8)

    def test_d_only_uses_original_actor_and_legal_fused_observation(self):
        agent = agent_for(config("grace_d_only"))
        raw = {k:v[:,0] for k,v in replay_batch(12).items()}
        agent._goal_reference.reliability.fill_(1.)
        torch.manual_seed(19)
        clean, evidence, _ = build_policy_observation(raw, agent._goal_reference)
        self.assertTrue(torch.equal(clean["heatmap"], evidence["heatmap_policy"]))
        inputs = []
        h = agent._wm.encoder.register_forward_pre_hook(lambda _, args: inputs.append(args[0]["heatmap"].clone()))
        proposed = []
        actor = agent._task_behavior.actor.register_forward_hook(lambda _, args, y: proposed.append(y.mode().clone()))
        try:
            out, state = agent._policy(raw, None, False)
        finally:
            h.remove()
            actor.remove()
        torch.testing.assert_close(inputs[0], evidence["heatmap_policy"][..., None].float()/255.)
        self.assertTrue(torch.equal(out["action"], proposed[0]))
        self.assertTrue((out["log_vnext_source"] == 0).all())
        self.assertIsNone(state[2])
        self.assertNotIn("log_vnext_decision", out)

    def test_dac_factual_training_without_maps_or_history(self):
        cfg = config("grace_dac", vnext_start_step=0, vnext_base_episode_fraction=0.)
        agent = agent_for(cfg)
        ep = collected_episode(agent, 40)
        self.assertFalse(any(k.startswith("log_spatial_history") for k in ep))
        self.assertIn("log_grace_dac", ep)
        data = next(make_dataset({"dac":ep}, cfg))
        self.assertFalse(any(k.startswith(("vnext_spatial_", "vnext_pose_")) for k in data))
        feature = torch.randn((*data["reward"].shape, agent._vnext_control.feature_dim))
        memory = agent._vnext.memory_batch(feature, data)
        self.assertEqual(memory.shape[-1], cfg.vnext_memory_dim)
        self.assertEqual(torch.count_nonzero(memory), 0)
        metrics = agent._vnext.train_step(feature, data)
        self.assertGreater(metrics["A/future_valid_labels"], 0)
        self.assertGreater(metrics["A/future_grad_norm"], 0)
        self.assertIsNone(agent._vnext._b_opt)
        c_metrics = agent._vnext_control.train_batch(feature, memory, agent._vnext, data)
        self.assertGreater(c_metrics["C/transitions"], 0)
        self.assertTrue(agent._vnext_control._optimizer.state)
        foreign = dict(data, log_grace_dac=np.zeros_like(data["log_grace_dac"]))
        self.assertEqual(agent._vnext_control.train_batch(feature, memory, agent._vnext, foreign)["C/transitions"], 0)

    def test_no_fusion_preserves_qualified_non_heatmap_evidence(self):
        verifier = MatchedRegions()
        verifier.region[:,7:10,7:10] = .98
        raw = goal_obs(190)
        _, full, _ = build_policy_observation(raw, verifier)
        verifier.fusion_enabled = False
        clean, no_mix, _ = build_policy_observation(raw, verifier)
        self.assertGreater(full["gate"].item(), 0)
        self.assertEqual(no_mix["gate"].item(), 0)
        self.assertTrue(torch.equal(clean["heatmap"],raw["heatmap"]))
        for key in ("context","state","boxes","region","conflict","peak_rejected"):
            torch.testing.assert_close(full[key],no_mix[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
