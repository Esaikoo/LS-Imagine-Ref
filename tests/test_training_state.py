"""Checkpoint resume regressions; no simulator or server interaction."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools
from training_state import (prepare_resume, restore_training_state,
                            write_resume_manifest, save_training_checkpoint)


class Scaler:
    def __init__(self, scale=17):
        self.scale = scale
    def state_dict(self):
        return {"scale": self.scale}
    def load_state_dict(self, state):
        self.scale = state["scale"]


class Opt:
    def __init__(self, params):
        self._opt = torch.optim.Adam(list(params), lr=.01)
        self._scaler = Scaler()


class Head(nn.Linear):
    def __init__(self, version):
        super().__init__(2, 2)
        self.model_version = version


class Search(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.model_version = 2
        self.register_buffer("updates", torch.tensor(5))
        self._rng = torch.Generator().manual_seed(seed)
        self.macro_names = ["scan_left", "actor"]
    def rng_state(self):
        return self._rng.get_state()
    def load_rng_state(self, value):
        if value is not None:
            self._rng.set_state(value)


class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.search_a = Head(5)
        self.search_b = Head(3)
        self._model_opt = Opt(self.parameters())


class Behavior(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.4, .7]))
        self._actor_opt = Opt(self.parameters())
        self._value_opt = Opt(self.parameters())
        self._updates = 12
        self.jump_prob = .7


class Agent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._wm = World()
        self._task_behavior = Behavior()
        self._search_c = Search(29)
        self._search_a = Search(13)
        self._step = 7
        self._update_count = 12
        self._should_train = tools.Every(32)
        self._should_log = tools.Every(1000)
        self._should_reset = tools.Every(0)
        self._should_pretrain = tools.Once()
        self._should_pretrain._once = False
    def update(self):
        opt = self._wm._model_opt._opt
        opt.zero_grad()
        loss = sum(parameter.square().sum() for parameter in self._wm.parameters())
        loss.backward()
        opt.step()


def config(root, **changes):
    values = dict(resume_from="", logdir=Path(root), traindir=Path(root)/"train_eps",
        evaldir=Path(root)/"eval_eps", steps=1000000, device="cpu",
        task="minedojo_harvest_sand", target_item="sand", seed=7, action_repeat=1,
        eval_every=10000, log_every=1000, time_limit=1000, dataset_size=1000000,
        search_a_enabled=True, search_a_model_version=5, search_a_control_version=4,
        search_b_enabled=True, search_b_model_version=3,
        search_c_enabled=True, search_c_model_version=2, search_branch_every=100000,
        search_eval_evidence=True, jump_prob=.7, wandb_key="must-not-persist")
    values.update(changes)
    return SimpleNamespace(**values)


def replay(path, identity, steps):
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path/(identity+"-"+str(steps+1)+".npz"),
                        reward=np.zeros(steps+1, np.float32),
                        action=np.zeros((steps+1,12), np.float32))


class TrainingStateTests(unittest.TestCase):
    def fixture(self, directory, steps=7):
        cfg = config(directory)
        cfg.traindir.mkdir(parents=True)
        replay(cfg.traindir, "episode", steps)
        agent = Agent(cfg)
        agent.update()
        checkpoint = save_training_checkpoint(agent, cfg, 100000)
        return cfg, agent, checkpoint

    def test_strict_restore_preserves_source_optimizer_rng_amp_and_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"
            cfg, old, checkpoint = self.fixture(source)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            random.seed(33); np.random.seed(33); torch.manual_seed(33)
            checkpoint = save_training_checkpoint(old,cfg,100000)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            expected = (random.random(),np.random.rand(),torch.rand(4))
            incoming = config(Path(tmp)/"new",resume_from=str(checkpoint),traindir=None,evaldir=None,
                              seed=0,device="cpu",minedojo_reset_retries=4)
            plan = prepare_resume(incoming)
            self.assertEqual(incoming.seed,7)
            self.assertEqual(incoming.minedojo_reset_retries,4)
            new = Agent(incoming)
            new._wm._model_opt._scaler.scale=1
            logger = SimpleNamespace(step=0)
            restore_training_state(new,plan,logger)
            for key,value in old.state_dict().items():
                torch.testing.assert_close(new.state_dict()[key],value)
            source_opt=old._wm._model_opt._opt.state_dict()
            target_opt=new._wm._model_opt._opt.state_dict()
            self.assertEqual(source_opt["param_groups"],target_opt["param_groups"])
            for key in source_opt["state"]:
                for field,value in source_opt["state"][key].items():
                    torch.testing.assert_close(target_opt["state"][key][field],value)
            self.assertEqual(new._wm._model_opt._scaler.scale,17)
            torch.testing.assert_close(new._search_a.rng_state(),old._search_a.rng_state())
            torch.testing.assert_close(new._search_c.rng_state(),old._search_c.rng_state())
            self.assertEqual(new._update_count,12)
            self.assertEqual(new._task_behavior._updates,12)
            self.assertFalse(new._should_pretrain())
            self.assertEqual(new._should_train(new._step),0)
            self.assertEqual(logger.step,7)
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(4),expected[2])
            manifest=write_resume_manifest(plan,incoming)
            data=json.loads(manifest.read_text())
            self.assertTrue(data["skip_random_prefill"])
            self.assertFalse(data["exact_trajectory_resume"])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            saved=torch.load(checkpoint,map_location="cpu")
            self.assertIsNone(saved["config"]["wandb_key"])

    def test_models_rollback_but_budget_includes_saved_real_data_and_completed_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, agent, checkpoint=self.fixture(Path(tmp)/"source",steps=111409)
            agent._step=102500
            checkpoint=save_training_checkpoint(agent,cfg,100000)
            branch=cfg.logdir/"branches/000001"
            branch.mkdir(parents=True)
            (branch/"feedback.json").write_text(json.dumps(
                dict(group=1,real_steps=33,options=[{}, {}, {}])))
            (cfg.logdir/"metrics.jsonl").write_text(json.dumps(dict(step=111409))+"\n")
            incoming=config(Path(tmp)/"new",resume_from=str(checkpoint),traindir=None,evaldir=None)
            plan=prepare_resume(incoming)
            self.assertEqual(plan["model_step"],102500)
            self.assertEqual(plan["environment_step"],111409)
            self.assertEqual(plan["budget_offset"],0)
            self.assertEqual(plan["next_branch_step"],200000)
            new=Agent(incoming)
            restore_training_state(new,plan,SimpleNamespace(step=0))
            self.assertEqual(new._step,111409)
            self.assertEqual(new._should_train(new._step),0)

    def test_chained_sources_deduplicate_longest_episode_and_keep_budget_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,checkpoint=self.fixture(Path(tmp)/"first",steps=3)
            agent._step=3
            checkpoint=save_training_checkpoint(agent,cfg,100000)
            incoming=config(Path(tmp)/"second",resume_from=str(checkpoint),traindir=None,evaldir=None)
            plan=prepare_resume(incoming)
            second=Agent(incoming)
            restore_training_state(second,plan,SimpleNamespace(step=0))
            incoming.traindir=Path(tmp)/"second/train_eps"
            replay(incoming.traindir,"episode",5)
            replay(incoming.traindir,"another",2)
            cp2=save_training_checkpoint(second,incoming,100000)
            (incoming.logdir/"metrics.jsonl").write_text('{"step": 9}\n')
            third_cfg=config(Path(tmp)/"third",resume_from=str(cp2),traindir=None,evaldir=None)
            plan2=prepare_resume(third_cfg)
            self.assertEqual(len(plan2["replay_sources"]),2)
            self.assertEqual(plan2["replay_steps"],7)
            self.assertEqual(plan2["environment_step"],9)
            self.assertEqual(plan2["budget_offset"],2)
            third=Agent(third_cfg)
            restore_training_state(third,plan2,SimpleNamespace(step=0))
            third_cfg.traindir=third_cfg.logdir/"train_eps"
            replay(third_cfg.traindir,"third",1)
            cp3=save_training_checkpoint(third,third_cfg,100000)
            fourth_cfg=config(Path(tmp)/"fourth",resume_from=str(cp3),traindir=None,evaldir=None)
            plan3=prepare_resume(fourth_cfg)
            self.assertEqual(plan3["replay_steps"],8)
            self.assertEqual(plan3["environment_step"],10)
            self.assertEqual(plan3["budget_offset"],2)
            self.assertEqual(len(plan3["replay_sources"]),3)

    def test_action_repeat_intervals_normalize_once_and_target_is_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,_=self.fixture(Path(tmp)/"source")
            cfg.action_repeat=2
            cfg.eval_every=5000;cfg.log_every=500;cfg.time_limit=500
            cp=save_training_checkpoint(agent,cfg,100000)
            incoming=config(Path(tmp)/"new",resume_from=str(cp),traindir=None,evaldir=None,
                            action_repeat=1,steps=2000000,log_every=2000)
            prepare_resume(incoming)
            self.assertEqual(incoming.action_repeat,2)
            self.assertEqual(incoming.eval_every,10000)
            self.assertEqual(incoming.time_limit,1000)
            self.assertEqual(incoming.log_every,2000)
            self.assertEqual(incoming.steps,2000000)

    def test_old_checkpoint_counters_are_marked_inferred_not_recomputed_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,cp=self.fixture(Path(tmp)/"source")
            checkpoint=torch.load(cp,map_location="cpu")
            checkpoint.pop("runtime_state")
            checkpoint.pop("rng_state")
            torch.save(checkpoint,cp)
            incoming=config(Path(tmp)/"new",resume_from=str(cp),traindir=None,evaldir=None)
            plan=prepare_resume(incoming)
            new=Agent(incoming)
            restore_training_state(new,plan,SimpleNamespace(step=0))
            self.assertEqual(new._update_count,1)
            self.assertEqual(new._task_behavior.jump_prob,.7)
            self.assertIn("legacy_update_counters_inferred_from_applied_optimizer_steps",plan["runtime_notes"])
            self.assertIn("legacy_checkpoint_has_no_global_rng_state",plan["runtime_notes"])

    def test_incompatible_task_and_exhausted_budget_and_source_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,cp=self.fixture(Path(tmp)/"source")
            for changes in (dict(task="minedojo_harvest_log_in_plains"),
                            dict(target_item="log"),dict(steps=7),dict(logdir=cfg.logdir)):
                values=dict(resume_from=str(cp),traindir=None,evaldir=None)
                values.update(changes)
                with self.assertRaises(ValueError):
                    prepare_resume(config(Path(tmp)/"new",**values))
            incoming=config(Path(tmp)/"new",resume_from=str(cp),traindir=None,evaldir=None)
            plan=prepare_resume(incoming)
            new=Agent(incoming)
            del plan["checkpoint"]["agent_state_dict"]["_task_behavior.weight"]
            with self.assertRaises(RuntimeError):
                restore_training_state(new,plan,SimpleNamespace(step=0))

    def test_prepare_rejects_search_version_mismatch_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,cp=self.fixture(Path(tmp)/"source")
            original=torch.load(cp,map_location="cpu")
            for field,value in (("search_a_version",4),("search_b_version",2),("search_c_version",1)):
                altered=copy.deepcopy(original)
                altered[field]=value
                wrong=cfg.logdir/("wrong_"+field+".pt")
                torch.save(altered,wrong)
                incoming=config(Path(tmp)/"new",resume_from=str(wrong),traindir=None,evaldir=None)
                with self.assertRaises(ValueError):
                    prepare_resume(incoming)

    def test_atomic_interrupted_save_leaves_latest_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,agent,latest=self.fixture(Path(tmp)/"source")
            before=latest.read_bytes()
            interrupted=save_training_checkpoint(agent,cfg,100000,name="interrupted.pt")
            self.assertTrue(torch.load(interrupted,map_location="cpu")["interrupted"])
            self.assertEqual(latest.read_bytes(),before)
            def failure(payload,stream):
                stream.write(b"partial")
                raise OSError("simulated write failure")
            with mock.patch("training_state.torch.save",side_effect=failure):
                with self.assertRaises(OSError):
                    save_training_checkpoint(agent,cfg,100000)
            self.assertEqual(latest.read_bytes(),before)
            self.assertFalse((cfg.logdir/"latest.pt.tmp").exists())
            with self.assertRaises(ValueError):
                save_training_checkpoint(agent,cfg,100000,name="../latest.pt")


if __name__=="__main__":
    unittest.main(verbosity=2)
