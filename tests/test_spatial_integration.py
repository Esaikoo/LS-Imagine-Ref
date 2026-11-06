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
from apm.outcome import _hash_int, check_vnext_checkpoint, restore_vnext_config, SearchVNext
from apm.replay import sample_spatial_episodes
from expr import make_dataset
from training_state import save_training_checkpoint, prepare_resume, restore_training_state
import tools

torch.set_num_threads(2)


def config(**updates):
    values = dict(vnext_spatial_enabled=True, vnext_map_size=35,
        vnext_spatial_fix_version=1,
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
    def test_fix3_persistent_policy_never_reads_oracle_pose_or_labels(self):
        cfg=config(vnext_spatial_fix_version=3)
        agent=make_agent(cfg)
        raw=replay_batch(12)
        obs={k:v[:,0] for k,v in raw.items() if k!="action"}
        noisy=dict(obs,search_pose_delta=np.full((1,3),1.e8),
            search_pose_valid=np.ones(1),log_spatial_gt_pose=np.full((1,3),1.e8),
            location_stats={"pos":np.full((1,3),1.e8)},
            search_center_interactable=np.ones(1))
        rng=torch.get_rng_state()
        left,ls=agent._policy(obs,None,False)
        torch.set_rng_state(rng)
        right,rs=agent._policy(noisy,None,False)
        for k in ["action","log_spatial_history_frames","log_spatial_history_meta",
                  "log_spatial_summary","log_spatial_pose"]:
            torch.testing.assert_close(left[k],right[k],rtol=0,atol=0)
        torch.testing.assert_close(ls[2]["history_valid"],rs[2]["history_valid"])

    def test_training_driver_delivers_terminal_feedback_before_reset(self):
        from unittest import mock
        import collections
        cfg=config(vnext_spatial_fix_version=3)
        owner=make_agent(cfg)
        class Env:
            id="terminal_fixture"
            def __init__(self):self.t=0
            def observation(self,done=False):
                return dict(image=np.zeros((4,4,3),np.uint8),heatmap=np.zeros((4,4),np.uint8),
                    is_first=self.t==0,is_last=done,is_terminal=done,is_zoomed=False,
                    score=0.,score_on_zoomed=0.,jumping_steps=0,accumulated_reward=0.,
                    is_calculated=False,intrinsic=0.,log_actuator_valid=float(self.t>0),
                    log_env_error=0.)
            def reset(self):
                self.t=0
                return lambda:self.observation()
            def step(self,action):
                self.t+=1;done=self.t==2
                return lambda:(self.observation(done),2. if done else 0.,done,
                    dict(real_done=True,success=done,first_success_step=2))
        class Policy:
            def __call__(self,obs,reset,state):
                if state is None:
                    control=owner._vnext_control.initial(1,torch.device("cpu"))
                    control["trial_option"][:]=3;control["last_option"][:]=3
                    control["trial_phase"][:]=0
                    state=(None,None,None,control)
                return {"action":torch.eye(12)[[6]]},state
            def observe_vnext_results(self,state,results,training=False):
                owner.observe_vnext_results(state,results,training)
        logger=SimpleNamespace(step=0,scalar=lambda *a:None,video=lambda *a:None,write=lambda **kw:None)
        storage=mock.Mock();storage.count_data_pairs.return_value=0
        storage.get_and_remove_less_than.return_value=[]
        with mock.patch.object(tools,"save_episodes"):
            tools.simulate(Policy(),[Env()],collections.OrderedDict(),".",logger,storage,
                           2,.99,steps=2,is_training=True)
        self.assertEqual(int(owner._vnext_control.phase_completed[0,3]),1)
        self.assertEqual(int(owner._vnext_control.phase_positive[0,3]),1)

    def test_fix3_history_stays_available_without_global_pose_and_replay_is_causal(self):
        cfg=config(vnext_spatial_fix_version=3,batch_size=2,batch_length=12)
        agent=make_agent(cfg)
        ep=collected_episode(agent,36)
        self.assertFalse(ep["log_spatial_trajectory_valid"].any())
        self.assertTrue((ep["log_spatial_memory_available"][1:]>0).all())
        data=next(make_dataset({"history":ep},cfg))
        self.assertGreater(float(data["vnext_spatial_history_snapshot_valid"].sum()),0)
        features=torch.randn(2,12,cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete,requires_grad=True)
        memory=agent._vnext.memory_batch(features,data)
        self.assertTrue(memory.any())
        before=agent._vnext.b.history_head.weight.detach().clone()
        metrics=agent._vnext.train_step(features,data)
        self.assertGreater(float(metrics["B/history_valid"]),0)
        self.assertGreater(float(metrics["B/history_loss"]),0)
        self.assertGreater(float(metrics["A/future_grad_norm"]),0)
        self.assertFalse(torch.equal(before,agent._vnext.b.history_head.weight))
        self.assertIsNone(features.grad)
        sample=next(sample_spatial_episodes({"history":ep},24,seed=3,history_slots=cfg.vnext_slots))
        for row,index in enumerate(sample["vnext_observation_index"]):
            if index+1<len(ep["reward"]):
                np.testing.assert_array_equal(sample["vnext_spatial_history_frames"][row],
                    ep["log_spatial_history_frames"][index+1])
            else:
                self.assertEqual(sample["vnext_spatial_history_snapshot_valid"][row],0)

    def test_fix3_legacy_and_new_history_keep_one_schema_without_fabrication(self):
        cfg=config(vnext_spatial_fix_version=3,batch_size=2,batch_length=12)
        agent=make_agent(cfg);ep=collected_episode(agent,28)
        old={k:v.copy() for k,v in ep.items() if not k.startswith("log_spatial_history_")
             and k!="log_spatial_memory_available"}
        episodes={"old":old};stream=make_dataset(episodes,cfg)
        first=next(stream)
        self.assertFalse(first["vnext_spatial_history_snapshot_valid"].any())
        episodes["new"]=ep
        data=next(stream)
        self.assertEqual(set(first),set(data))
        self.assertEqual(data["vnext_spatial_history_frames"].dtype,np.uint8)
        self.assertEqual(data["vnext_spatial_history_frames"].shape,(2,12,cfg.vnext_slots,4,4,4))
        self.assertTrue(np.isfinite(data["vnext_spatial_history_meta"]).all())

    def test_fix3_future_bank_metadata_and_unknown_nan_pose_are_not_certified(self):
        cfg=config(vnext_spatial_fix_version=3,batch_size=1,batch_length=12)
        agent=make_agent(cfg);ep=collected_episode(agent,24)
        for row in range(1,len(ep["reward"])):
            ep["log_spatial_history_meta"][row,:,1]=ep["log_spatial_history_tick"][row]+10
        data=next(make_dataset({"future":ep},cfg))
        self.assertFalse(data["vnext_spatial_history_snapshot_valid"].any())
        data["vnext_spatial_pose"][:]=np.nan
        data["vnext_spatial_local_valid"][:]=0
        dim=cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        context,details=agent._vnext.b.batch(torch.randn(1,12,dim),data)
        self.assertTrue(torch.isfinite(context).all())
        self.assertFalse(context.any())
        self.assertFalse(details["map_valid"].any())

    def test_fix3_checkpoint_roundtrip_preserves_history_and_phase_experience(self):
        cfg=config(vnext_spatial_fix_version=3,batch_size=1,batch_length=12)
        agent=make_agent(cfg);ep=collected_episode(agent,24)
        agent._train(next(make_dataset({"new":ep},cfg)))
        agent._vnext_control.phase_coverage[0,6]=17
        with tempfile.TemporaryDirectory() as directory:
            cfg.logdir=Path(directory)/"source";cfg.traindir=cfg.logdir/"train_eps"
            cfg.traindir.mkdir(parents=True)
            tools.save_episodes(cfg.traindir,{"new":ep})
            path=save_training_checkpoint(agent,cfg,100000)
            other=make_agent(config(vnext_spatial_fix_version=3,batch_size=1,batch_length=12))
            other.load_state_dict(torch.load(path,map_location="cpu")["agent_state_dict"],strict=True)
            for k,v in agent.state_dict().items():
                torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(int(other._vnext_control.phase_coverage[0,6]),17)

    def test_explicit_fix2_upgrade_keeps_learned_weights_and_named_optimizer_moments(self):
        import hashlib
        source_cfg=config(vnext_spatial_fix_version=2,batch_size=1,batch_length=12)
        source=make_agent(source_cfg);ep=collected_episode(source,24)
        source._train(next(make_dataset({"old":ep},source_cfg)))
        source._step=50
        source._vnext.a.current_counts.fill_(123)
        source._vnext.a.cal_counts.fill_(5)
        with tempfile.TemporaryDirectory() as directory:
            source_cfg.logdir=Path(directory)/"source";source_cfg.traindir=source_cfg.logdir/"train_eps"
            source_cfg.traindir.mkdir(parents=True)
            tools.save_episodes(source_cfg.traindir,{"old":ep})
            path=save_training_checkpoint(source,source_cfg,100000)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            cfg=config(vnext_spatial_fix_version=3,batch_size=1,batch_length=12,
                resume_upgrade_spatial=True,resume_from=str(path),logdir=str(Path(directory)/"upgraded"),
                traindir=None,evaldir=None,steps=1000,vnext_search_fraction=.15,vnext_total_fraction=.30)
            plan=prepare_resume(cfg)
            self.assertEqual(cfg.vnext_spatial_fix_version,3)
            upgraded=make_agent(cfg,seed=47)
            restore_training_state(upgraded,plan,SimpleNamespace(step=0))
            self.assertEqual(upgraded._step,50)
            self.assertFalse(upgraded._should_pretrain._once)
            self.assertFalse(upgraded._vnext.a.current_counts.any())
            self.assertFalse(upgraded._vnext.a.cal_counts.any())
            self.assertFalse(upgraded._vnext_control.phase_coverage.any())
            torch.testing.assert_close(upgraded._vnext.a.events[0].weight,source._vnext.a.events[0].weight)
            for name,param in source._vnext.b.named_parameters():
                old_state=source._vnext._b_opt.state.get(param)
                if old_state:
                    new_param=dict(upgraded._vnext.b.named_parameters())[name]
                    new_state=upgraded._vnext._b_opt.state[new_param]
                    for key,value in old_state.items():
                        torch.testing.assert_close(value,new_state[key])
            for name,param in upgraded._vnext.b.named_parameters():
                if name.startswith(("history_","fusion")):
                    self.assertNotIn(param,upgraded._vnext._b_opt.state)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
            self.assertEqual(torch.load(path,map_location="cpu")["spatial_fix_version"],2)
            # Production creates the new run's replay directory before saving.
            cfg.traindir=Path(cfg.logdir)/"train_eps"
            cfg.traindir.mkdir(parents=True,exist_ok=True)
            saved=save_training_checkpoint(upgraded,cfg,100000)
            result=torch.load(saved,map_location="cpu")
            self.assertEqual(result["spatial_fix_version"],3)
            self.assertEqual(result["spatial_upgrade_lineage"][0]["source_step"],50)
            self.assertTrue(plan["spatial_upgrade"]["initialized_tensors"])
            normal=config(vnext_spatial_fix_version=3,batch_size=1,batch_length=12,
                resume_from=str(saved),resume_upgrade_spatial=False,
                logdir=str(Path(directory)/"normal_fix3"),traindir=None,evaldir=None,steps=2000)
            nextplan=prepare_resume(normal)
            continued=make_agent(normal,seed=91)
            restore_training_state(continued,nextplan,SimpleNamespace(step=0))
            self.assertEqual(normal.vnext_spatial_fix_version,3)
            self.assertEqual(len(continued._spatial_upgrade_lineage),1)
            self.assertIsNone(nextplan["spatial_upgrade"])
            for key,value in upgraded.state_dict().items():
                torch.testing.assert_close(value,continued.state_dict()[key],rtol=0,atol=0)

    def test_ordinary_fix2_resume_does_not_silently_enable_fix3(self):
        cfg=config(vnext_spatial_fix_version=2,batch_size=1,batch_length=12)
        agent=make_agent(cfg);ep=collected_episode(agent,16)
        agent._step=40
        with tempfile.TemporaryDirectory() as directory:
            cfg.logdir=Path(directory)/"source";cfg.traindir=cfg.logdir/"train_eps"
            cfg.traindir.mkdir(parents=True);tools.save_episodes(cfg.traindir,{"old":ep})
            path=save_training_checkpoint(agent,cfg,100000)
            target=config(vnext_spatial_fix_version=3,resume_from=str(path),resume_upgrade_spatial=False,
                logdir=str(Path(directory)/"resume"),traindir=None,evaldir=None,steps=1000)
            plan=prepare_resume(target)
            self.assertEqual(target.vnext_spatial_fix_version,2)
            self.assertIsNone(plan["spatial_upgrade"])

    def test_revision_two_unlocalized_map_does_not_disable_A_or_pose_learning(self):
        cfg=config(vnext_spatial_fix_version=2,batch_size=2,batch_length=12)
        agent=make_agent(cfg)
        ep=collected_episode(agent,28)
        self.assertFalse(ep["log_spatial_trajectory_valid"].any())
        data=next(make_dataset({"spatial":ep},cfg))
        self.assertGreater(float(data["vnext_spatial_valid"].sum()),0)
        self.assertEqual(float(data["vnext_spatial_trajectory_valid"].sum()),0)
        feature_dim=cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        features=torch.randn(2,12,feature_dim,requires_grad=True)
        with torch.no_grad():
            memory=agent._vnext.memory_batch(features,data)
            self.assertFalse(memory.any())
        before=agent._vnext.a.events[0].weight.detach().clone()
        metrics=agent._vnext.train_step(features,data)
        self.assertGreater(float(metrics["A/current_valid_labels"]),0)
        self.assertGreater(float(metrics["A/future_valid_labels"]),0)
        self.assertGreater(float(metrics["A/future_grad_norm"]),0)
        self.assertGreater(float(metrics["B/pose_samples"]),0)
        self.assertFalse(torch.equal(before,agent._vnext.a.events[0].weight))
        self.assertIsNone(features.grad)
        # Maps are outgoing decisions: validity must follow row i+1, not i.
        ep["log_spatial_trajectory_valid"][:]=np.arange(len(ep["reward"]))%2
        sample=next(sample_spatial_episodes({"spatial":ep},20,seed=3))
        for row,index in enumerate(sample["vnext_observation_index"]):
            expected=ep["log_spatial_trajectory_valid"][index+1] if index+1<len(ep["reward"]) else 0
            self.assertEqual(sample["vnext_spatial_trajectory_valid"][row],expected)

    def test_production_sampler_without_legacy_validity_trains_future_heads(self):
        cfg = config(batch_size=1,batch_length=32,vnext_aux_every=1)
        agent = make_agent(cfg)
        ep = collected_episode(agent,40)
        # This is the schema that the actual 400k run collected: controls are
        # present, but no explicit validity field was written by the wrapper.
        del ep["log_actuator_valid"]
        before_control = ep["log_applied_control"].copy()
        data = next(make_dataset({"spatial":ep},cfg))
        self.assertGreater(float(data["log_actuator_valid"].sum()),0)
        a = agent._vnext.a
        before_events = a.events[0].weight.detach().clone()
        before_future = a.future[0].weight.detach().clone()
        feature_dim = cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        features = torch.randn(1,32,feature_dim,requires_grad=True)
        metrics = agent._vnext.train_step(features,data)
        for horizon in (1,2,3,8):
            self.assertGreater(float(metrics["A/H%d/train_windows"%horizon]),0)
            self.assertGreater(float(metrics["A/H%d/label_count"%horizon]),0)
            self.assertGreater(float(metrics["A/H%d/prediction_loss"%horizon]),0)
        self.assertGreater(float(metrics["A/future_grad_norm"]),0)
        self.assertGreater(float(metrics["A/future_prediction_loss"]),0)
        self.assertFalse(torch.equal(before_events,a.events[0].weight))
        self.assertFalse(torch.equal(before_future,a.future[0].weight))
        self.assertGreater(float(a.support_counts.sum()),0)
        self.assertIsNone(features.grad)
        self.assertNotIn("log_actuator_valid",ep)
        np.testing.assert_array_equal(ep["log_applied_control"],before_control)

    def test_actual_control_provenance_excludes_unknown_error_and_boundary_rows(self):
        agent = make_agent(config())
        ep = collected_episode(agent,12)
        del ep["log_actuator_valid"]
        ep["log_applied_control"] = list(ep["log_applied_control"])
        ep["log_applied_control"][2] = None
        ep["log_applied_control"][3] = np.full(9,np.nan,np.float32)
        ep["log_applied_control"][4] = np.zeros(9,np.float32) # actual no-op
        ep["log_env_error"] = np.zeros(13,np.float32)
        ep["log_env_error"][5] = 1
        ep["is_first"][7] = True
        ep["is_last"][8] = True
        ep["is_terminal"][10] = True
        data = next(sample_spatial_episodes({"spatial":ep},64,seed=8))
        expected = np.array([0,1,0,0,1,0,0,0,1,0,1,0,1],np.float32)
        np.testing.assert_array_equal(data["log_actuator_valid"],expected[data["vnext_observation_index"]])
        self.assertTrue(np.isfinite(data["log_applied_control"]).all())
        for row,index in enumerate(data["vnext_observation_index"]):
            self.assertEqual(data["vnext_pose_prev_valid"][row],expected[index])

    def test_explicit_control_invalid_and_missing_actual_control_are_never_inferred(self):
        agent = make_agent(config())
        ep = collected_episode(agent,12)
        explicit = copy.deepcopy(ep)
        explicit["log_actuator_valid"] = [np.float32(1)]*13
        explicit["log_actuator_valid"][3] = None
        explicit["log_actuator_valid"][4] = np.nan
        explicit["log_actuator_valid"][5] = 0
        stream = sample_spatial_episodes({"explicit":explicit},64,seed=2)
        data = next(stream)
        indices = data["vnext_observation_index"]
        self.assertFalse(data["log_actuator_valid"][np.isin(indices,[0,3,4,5])].any())
        self.assertTrue(data["log_actuator_valid"][~np.isin(indices,[0,3,4,5])].all())
        del ep["log_applied_control"]
        del ep["log_actuator_valid"]
        # Policy intent is not an actual environment control observation.
        self.assertIn("log_vnext_applied",ep)
        data = next(sample_spatial_episodes({"missing":ep},32,seed=0))
        self.assertFalse(data["log_actuator_valid"].any())
        self.assertFalse(data["vnext_pose_prev_valid"].any())

    def test_nonfinite_batched_controls_do_not_poison_masked_future_loss(self):
        cfg = config(batch_size=1,batch_length=12)
        agent = make_agent(cfg)
        data = next(make_dataset({"spatial":collected_episode(agent,20)},cfg))
        data["log_applied_control"][:] = np.nan
        data["log_actuator_valid"][:] = np.nan
        data["log_env_error"] = np.full((1,12),np.nan,np.float32)
        feature_dim = cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        metrics = agent._vnext.train_step(torch.randn(1,12,feature_dim),data)
        self.assertEqual(float(metrics["A/future_train_windows"]),0)
        self.assertEqual(float(metrics["A/future_grad_norm"]),0)
        self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))

    def test_missing_map_holdout_cannot_certify_map_conditioned_current_head(self):
        cfg = config(batch_size=1,batch_length=12)
        agent = make_agent(cfg)
        data = next(make_dataset({"spatial":collected_episode(agent,20)},cfg))
        data["vnext_calibration"][:] = 1
        data["vnext_spatial_valid"][:] = 0
        feature_dim = cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        agent._vnext.train_step(torch.randn(1,12,feature_dim),data)
        self.assertFalse(agent._vnext.a.current_cal_counts.any())
        self.assertFalse(agent._vnext.a.cal_counts.any())
        self.assertFalse(agent._vnext.a.current_counts.any())

    def test_spatial_fix_revision_is_explicit_and_legacy_restore_cannot_inherit_new_default(self):
        agent = make_agent(config(vnext_spatial_fix_version=0))
        checkpoint = dict(agent_state_dict=agent.state_dict(),search_vnext_version=2,
            vnext_memory_version=2,pose_version=1,spatial_map_version=2,
            actuator_version=1,search_a_label_version=3,config={})
        check_vnext_checkpoint(checkpoint,True)
        restored = restore_vnext_config({"vnext_spatial_fix_version":1},checkpoint)
        self.assertEqual(restored["vnext_spatial_fix_version"],0)
        checkpoint["spatial_fix_version"] = 1
        with self.assertRaisesRegex(ValueError,"revision"):
            check_vnext_checkpoint(checkpoint,True)
        checkpoint["config"]["vnext_spatial_fix_version"] = 1
        check_vnext_checkpoint(checkpoint,True)
        self.assertEqual(restore_vnext_config({},checkpoint)["vnext_spatial_fix_version"],1)
        checkpoint["spatial_fix_version"] = 2
        checkpoint["config"]["vnext_spatial_fix_version"] = 2
        check_vnext_checkpoint(checkpoint,True)
        self.assertEqual(restore_vnext_config({},checkpoint)["vnext_spatial_fix_version"],2)
        checkpoint["spatial_fix_version"] = 3
        checkpoint["config"]["vnext_spatial_fix_version"] = 3
        check_vnext_checkpoint(checkpoint,True)
        self.assertEqual(restore_vnext_config({},checkpoint)["vnext_spatial_fix_version"],3)
        checkpoint["spatial_fix_version"] = 4
        checkpoint["config"]["vnext_spatial_fix_version"] = 4
        with self.assertRaisesRegex(ValueError,"revision"):
            check_vnext_checkpoint(checkpoint,True)
        with self.assertRaisesRegex(ValueError,"revision"):
            SearchVNext(24,config(vnext_spatial_fix_version=9))

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
            self.assertEqual(checkpoint["spatial_fix_version"],1)
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
