"""B regressions: real dataflow, long history, isolation, joint optimization and checkpoints."""
import collections
import pathlib
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from test_search_a import config_for_test, make_agent, replay_batch, fake_rays
from search_memory import (SearchMemory, B_VERSION, DESC_DIM, describe_numpy, describe_torch,
                           _prefix_state, sample_episodes_with_history, check_b_checkpoint)
from search_a import SearchOutcomeHead, actor_action_sequence
from search_evidence import SearchEvidence, complete_episode_labels
import tools
torch.set_num_threads(2)

def cfg(**kw):
    values = dict(logdir=str(pathlib.Path(__file__).resolve().parents[1]/"logs"/"test_apm_b"/"unit"),
                  search_b_enabled=True, search_b_slots=4, search_b_dim=16,
                  search_b_hidden=32, search_b_lookback=12, search_b_save_every=2)
    values.update(kw)
    return config_for_test(**values)

def batch(seed=6, length=12, slots=4, lookback=12):
    raw = replay_batch(30)
    raw["search_place_id"][:] = 1
    raw["search_place_valid"][:] = 1
    raw["search_visible_valid"][:] = 1
    raw["search_visible"][:, 7] = 1
    ep = {k:v[0] for k,v in raw.items()}
    gen = sample_episodes_with_history(collections.OrderedDict(one=ep), length,
                                      slots=slots, lookback=lookback, save_every=2, seed=seed)
    return next(tools.from_generator(gen, 1))

def tensors(data):
    return {k:torch.as_tensor(v) for k,v in data.items()}

class MemoryTests(unittest.TestCase):
    def test_descriptor_parity(self):
        data = replay_batch()
        a = describe_numpy(data["image"], data["heatmap"])
        b = describe_torch(torch.tensor(data["image"])/255., torch.tensor(data["heatmap"])/255.)
        np.testing.assert_allclose(a, b.numpy(), atol=1e-6)

    def test_full_history_and_numpy_online_parity(self):
        rng = np.random.RandomState(32)
        # Distinct frames, competing priorities and revisits across a long episode.
        scene = rng.uniform(0,1,(360,DESC_DIM)).astype(np.float32)
        scene[100:180] = scene[20]
        scene[300:] = scene[50]
        model = SearchMemory(slots=4, dim=16, hidden=32, save_every=2)
        state = model.initial(1, "cpu")
        with torch.no_grad():
            for row in scene:
                state = (*model.add(*state[:5], torch.from_numpy(row[None])), state[-1])
        expected = _prefix_state(scene, len(scene), 4, 2)
        for a,b in zip(expected,state[:4]):
            np.testing.assert_allclose(a,b[0].numpy(),atol=2e-5)
        self.assertEqual(state[3][0,0].item(),0)  # anchor survives beyond recent burn-in
        self.assertTrue((state[3] < 360).all())

    def test_retrieve_before_write_reset_and_executed_action(self):
        model=SearchMemory(slots=4,dim=16,hidden=32)
        scene=torch.rand(2,DESC_DIM)
        first,state,info=model.online_step(scene)
        self.assertEqual(info["log_b_memory_used"].tolist(),[0,0])
        _,state,info=model.online_step(scene,state,torch.tensor([1,0]))
        self.assertEqual(info["log_b_memory_used"].tolist(),[0,1])
        self.assertEqual(state[4].tolist(),[1,2])
        a,_,_=model.online_step(scene,previous_action=torch.zeros(2,12))
        b,_,_=model.online_step(scene,previous_action=F.one_hot(torch.ones(2,dtype=torch.long),12))
        self.assertFalse(torch.equal(a,b))

    def test_sampler_metadata_and_online_burn_in_agree(self):
        data=tensors(batch())
        self.assertEqual(data["search_b_prefix"].shape,(1,4,DESC_DIM))
        self.assertEqual(data["search_b_history"].shape,(1,12,DESC_DIM))
        model=SearchMemory(slots=4,dim=16,hidden=32,save_every=2)
        scene=describe_torch(data["image"]/255.,data["heatmap"]/255.)
        initial,_=model._burn_in(scene,data)
        contexts,_,_=model.train_sequence(scene,data)
        state=initial
        for t in range(scene.shape[1]):
            out,state,_=model.online_step(scene[:,t],state,data["search_b_real_first"][:,t],data["action"][:,t])
            torch.testing.assert_close(out,contexts[:,t])
        self.assertTrue((data["search_b_prefix_indices"] < data["search_b_prefix_tick"][:,None]).all())

    def test_stitched_episode_reset_and_unknown_supervision(self):
        data=tensors(batch(length=48))
        self.assertGreater(data["search_b_real_first"].sum().item(),0)
        model=SearchMemory(slots=4,dim=16,hidden=32,save_every=2)
        scene=describe_torch(data["image"]/255.,data["heatmap"]/255.)
        context,_,_=model.train_sequence(scene,data)
        reset=int(torch.where(data["search_b_real_first"][0].bool())[0][-1])
        changed=scene.clone();changed[:,:reset]=0
        other,_,_=model.train_sequence(changed,data)
        torch.testing.assert_close(context[:,reset:],other[:,reset:])
        for k in data:
            if k.endswith("_valid") and not k.startswith("search_b_"):
                data[k]=torch.zeros_like(data[k])
        _,loss,metrics=model.train_sequence(scene,data)
        self.assertEqual(loss.item(),0.)
        self.assertEqual(metrics["B/revisit_valid"].item(),0.)

    def test_a_gradient_trains_memory_and_time_predictor(self):
        data=tensors(batch())
        model=SearchMemory(slots=4,dim=16,hidden=32,save_every=2,feature_dim=32)
        scene=describe_torch(data["image"]/255.,data["heatmap"]/255.)
        memory,_,_=model.train_sequence(scene,data)
        head=SearchOutcomeHead(32,hidden=32,memory_dim=16)
        # Zero initialization deliberately preserves A outputs at the first update.
        torch.nn.init.normal_(head.memory_adapter.weight,std=.1)
        feat=torch.randn(1,12,32,requires_grad=True)
        loss,_,_,_=head.training_loss(feat,data,memory)
        loss.backward(retain_graph=True)
        self.assertGreater(model.belief.weight_hh.grad.abs().sum().item(),0)
        loss,count=model.transition_loss(memory,feat,data)
        loss.backward()
        self.assertGreater(count.item(),0)
        self.assertGreater(model.prediction.weight_ih.grad.abs().sum().item(),0)
        one=model.imagine_step(memory,feat,data["action"],elapsed=1)
        jump=model.imagine_step(memory,feat,data["action"],elapsed=6)
        self.assertEqual(one.shape,memory.shape)
        self.assertFalse(torch.equal(one,jump))
        self.assertTrue(torch.isfinite(feat.grad).all())

    def test_prefix_transition_durations_are_not_repeated(self):
        data = tensors(batch())
        model = SearchMemory(slots=4, dim=16, hidden=32, feature_dim=32)
        memory = torch.randn(1, 12, 16)
        features = torch.randn(1, 12, 32)
        with mock.patch.object(model, "imagine_step", wraps=model.imagine_step) as calls:
            loss, count = model.transition_loss(memory, features, data, (1, 3, 6))
        durations = [call.kwargs["elapsed"] for call in calls.call_args_list]
        self.assertEqual(durations, [1, 2, 3, 4, 6, 8, 11])
        self.assertEqual(len(durations), len(set(durations)))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(count), 0)

    def test_evidence_decay_leaves_raw_keys_intact(self):
        static=SearchMemory(slots=4,dim=16,hidden=32)
        dynamic=SearchMemory(slots=4,dim=16,hidden=32,dynamic_target=True)
        self.assertGreater(dynamic.decay_log_rate.item(),static.decay_log_rate.item())
        _,state,_=static.online_step(torch.ones(1,DESC_DIM))
        bank=state[0].clone()
        recent,_=static.retrieve(torch.ones(1,DESC_DIM),*state[:3],state[4])
        old,_=static.retrieve(torch.ones(1,DESC_DIM),*state[:3],state[4]+1000)
        self.assertFalse(torch.equal(recent,old))
        torch.testing.assert_close(state[0],bank)

    def test_expired_target_evidence_cannot_bypass_decay(self):
        torch.manual_seed(19)
        model=SearchMemory(slots=2,dim=16,hidden=32)
        scene=torch.full((1,DESC_DIM),.3)
        low=model.initial(1,"cpu")
        bank=low[0].clone();bank[:,:,:192]=.3
        valid=torch.ones(1,2,dtype=torch.bool)
        meta=low[2].clone();meta[:,:,2]=1
        with_cue=bank.clone();with_cue[:,:,192:]=1
        cue_meta=meta.clone();cue_meta[:,:,3]=1
        recent0,similarity0=model.retrieve(scene,bank,valid,meta,torch.tensor([1]))
        recent1,similarity1=model.retrieve(scene,with_cue,valid,cue_meta,torch.tensor([1]))
        self.assertGreater((recent0-recent1).abs().max().item(),1.e-5)
        # Same static location must match despite target entering/leaving the view.
        torch.testing.assert_close(similarity0,similarity1)
        old0,_=model.retrieve(scene,bank,valid,meta,torch.tensor([100000]))
        old1,_=model.retrieve(scene,with_cue,valid,cue_meta,torch.tensor([100000]))
        torch.testing.assert_close(old0,old1,atol=1.e-6,rtol=0)
        self.assertEqual(with_cue[:,:,192:].min().item(),1.)  # raw replay is untouched

    def test_missing_context_must_not_silently_reset_history(self):
        data=tensors(replay_batch())
        scene=describe_torch(data["image"]/255.,data["heatmap"]/255.)
        model=SearchMemory(slots=4,dim=16,hidden=32)
        # Original RSSM sampler sets is_first[0] even for a mid-episode crop.
        with self.assertRaisesRegex(ValueError,"historical context"):
            model.train_sequence(scene,data)
        data["search_b_real_first"]=torch.zeros_like(data["is_first"])
        with self.assertRaisesRegex(ValueError,"historical context"):
            model.train_sequence(scene,data)
        data["search_b_real_first"][:,0]=True
        model.train_sequence(scene,data)

    def test_prefix_is_future_blind_and_matches_entire_past(self):
        from search_memory import make_prefix
        raw=replay_batch(180)
        ep={k:v[0] for k,v in raw.items()}
        # A stable old landmark followed by a clearly different scene.
        ep["image"][:60]=20;ep["image"][60:]=200
        ep["heatmap"][:]=0
        bank,valid=make_prefix(ep,150,slots=4,lookback=12,save_every=2)
        changed={k:v.copy() for k,v in ep.items()}
        changed["image"][150:]=255;changed["heatmap"][150:]=255
        later,later_valid=make_prefix(changed,150,slots=4,lookback=12,save_every=2)
        np.testing.assert_array_equal(bank,later)
        np.testing.assert_array_equal(valid,later_valid)
        self.assertTrue(np.any(np.isclose(bank[valid.astype(bool),:192].mean(-1),20/255.)))

    def test_place_ids_are_optional_training_labels(self):
        collector=SearchEvidence("log")
        obs=fake_rays()
        obs["location_stats"]=dict(pos=np.array([0,64,0]),yaw=0.,pitch=0.)
        first=collector.observe(obs,initial=True)
        second=collector.observe(obs)
        self.assertEqual(first["search_place_id"],second["search_place_id"])
        self.assertEqual(first["search_place_valid"],1)
        obs["location_stats"]["pos"]=np.array([10,64,0])
        third=collector.observe(obs)
        self.assertNotEqual(first["search_place_id"],third["search_place_id"])
        unknown=complete_episode_labels(dict(reward=np.zeros(4)))
        self.assertFalse(np.asarray(unknown["search_place_valid"]).any())

class JointTests(unittest.TestCase):
    def test_real_joint_update_zoom_actor_critic_checkpoint(self):
        agent=make_agent(cfg())
        data=batch()
        data["is_zoomed"][:,2]=1
        data["is_calculated"][:,2]=1
        data["jumping_steps"][:,2]=2
        modules=[agent._wm.search_b, agent._task_behavior.actor, agent._task_behavior.value]
        before=[{k:p.detach().clone() for k,p in m.named_parameters()} for m in modules]
        agent._train(data)
        for name,old,module in zip(("memory","actor","value"),before,modules):
            self.assertTrue(any(not torch.equal(old[k],p) for k,p in module.named_parameters()),name)
        for key in ("model_loss","actor_loss","value_loss","A/prediction_loss",
                    "B/revisit_loss","B/consistency_loss","B/place_loss"):
            self.assertIn(key,agent._metrics)
        self.assertTrue(all(np.isfinite(v).all() for vals in agent._metrics.values() for v in vals))
        cp=dict(agent_state_dict=agent.state_dict(),search_a_version=2,search_b_version=B_VERSION,
                optims_state_dict=tools.recursively_collect_optim_state_dict(agent))
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/"latest.pt";torch.save(cp,path);cp=torch.load(path,map_location="cpu")
        check_b_checkpoint(cp,True)
        other=make_agent(cfg(search_b_control=False))
        other.load_state_dict(cp["agent_state_dict"],strict=True)
        tools.recursively_load_optim_state_dict(other,cp["optims_state_dict"])
        with self.assertRaises(ValueError):check_b_checkpoint(cp,False)
        cp["search_b_version"]=1
        with self.assertRaises(ValueError):check_b_checkpoint(cp,True)

    def test_policy_actual_action_no_oracle_and_continuation(self):
        agent=make_agent(cfg(search_a_control=False))
        raw=replay_batch()
        obs={k:v[:,0] for k,v in raw.items() if k not in ("action","discount")}
        torch.manual_seed(7);out,state=agent._policy(obs,None,False)
        changed={k:(v+1000 if k.startswith("search_") else v) for k,v in obs.items()}
        torch.manual_seed(7);other,state2=agent._policy(changed,None,False)
        torch.testing.assert_close(out["action"],other["action"])
        torch.testing.assert_close(state[-1][-1],state2[-1][-1])
        obs["is_first"][:]=False
        with mock.patch.object(agent._wm.search_b,"online_step",wraps=agent._wm.search_b.online_step) as call:
            agent._policy(obs,state,False)
        torch.testing.assert_close(call.call_args.kwargs["previous_action"],out["action"])
        seq=actor_action_sequence(agent._wm.dynamics,agent._task_behavior.actor,state[0],out["action"],6,
                                  memory=state[-1][-1],memory_transition=agent._wm.search_b.imagine_step)
        self.assertEqual(seq.shape,(1,6,12))

    def test_normal_and_jump_imagination_carry_memory(self):
        agent=make_agent(cfg())
        state=agent._wm.dynamics.initial(2)
        state["search_memory"]=torch.randn(2,16)
        action=F.one_hot(torch.zeros(2,dtype=torch.long),13).float()
        successor=agent._wm.dynamics.img_step(state,action)
        for jump in (False,True):
            next_state=agent._task_behavior._advance_memory(state,dict(successor),action,jump=jump)
            self.assertEqual(next_state["search_memory"].shape,(2,16))
            self.assertFalse(torch.equal(next_state["search_memory"],state["search_memory"]))
        feat=agent._task_behavior.policy_features(state)
        self.assertEqual(feat.shape[-1],agent._task_behavior._base_feat_size+16)
        agent._config.search_b_control=False
        self.assertEqual(agent._task_behavior.policy_features(state)[:,-16:].abs().sum().item(),0)

    def test_forced_jump_joint_backward(self):
        agent=make_agent(cfg())
        for all_jump in (False, True):
            def select(probability, _):
                mask=torch.zeros_like(probability,dtype=torch.bool)
                mask[::1 if all_jump else 2]=True
                return mask
            with mock.patch("models.probability_to_bool",side_effect=select):
                agent._train(batch())
        self.assertTrue(all(np.isfinite(v).all() for vals in agent._metrics.values() for v in vals))

    def test_default_memory_size_joint_update(self):
        agent=make_agent(cfg(search_b_slots=16,search_b_dim=128,search_b_hidden=128,search_b_lookback=128))
        agent._train(batch(slots=16,lookback=128))
        self.assertIn("B/total_loss",agent._metrics)
        self.assertTrue(all(np.isfinite(v).all() for vals in agent._metrics.values() for v in vals))

    def test_b_only_and_disabled_architecture(self):
        bonly=make_agent(cfg(search_a_enabled=False))
        bonly._train(batch())
        self.assertIsNone(bonly._wm.search_a)
        self.assertIn("B/revisit_loss",bonly._metrics)
        aonly=make_agent()
        self.assertIsNone(aonly._wm.search_b)
        self.assertFalse(any("memory_adapter" in k or ".search_b." in k for k in aonly.state_dict()))
        with self.assertRaises(ValueError):
            check_b_checkpoint(dict(agent_state_dict=aonly.state_dict()),True)

    def test_two_environment_batch_with_different_history_lengths(self):
        parts=[batch(seed=seed) for seed in (3,6)]
        data={k:np.concatenate([p[k] for p in parts],0) for k in parts[0]}
        agent=make_agent(cfg())
        agent._train(data)
        self.assertTrue(all(np.isfinite(v).all() for vals in agent._metrics.values() for v in vals))
        # Independent rows must give the same result as separate rollout.
        tensor=tensors(data)
        scene=describe_torch(tensor["image"]/255.,tensor["heatmap"]/255.)
        together,_,_=agent._wm.search_b.train_sequence(scene,tensor)
        for i in range(2):
            single,_,_=agent._wm.search_b.train_sequence(scene[i:i+1],{k:v[i:i+1] for k,v in tensor.items()})
            torch.testing.assert_close(together[i:i+1],single)

    def test_b_logs_survive_prefill_npz(self):
        cache={}
        for t in range(10):
            row=dict(reward=np.float32(0), image=np.zeros((64,64,3),np.uint8))
            if t>=5:
                row.update(log_b_keyframe_indices=np.array([0,t-5,-1,-1],np.int64),
                           log_b_observation_index=np.int64(t-5),
                           log_b_revisit_prob=np.float32(.5))
            tools.add_to_cache(cache,"ep",row)
        with tempfile.TemporaryDirectory() as folder:
            tools.save_episodes(folder,cache)
            for path in pathlib.Path(folder).glob("*.npz"):
                with np.load(path,allow_pickle=False) as ep:
                    self.assertEqual(ep["log_b_keyframe_indices"].shape,(10,4))
                    np.testing.assert_array_equal(ep["log_b_keyframe_indices"][9],[4,8,-1,-1])
                    self.assertTrue((ep["log_b_keyframe_indices"][:5]==-1).all())
                    self.assertEqual({len(ep[k]) for k in ep.files},{10})

if __name__=="__main__":
    unittest.main(verbosity=2)
