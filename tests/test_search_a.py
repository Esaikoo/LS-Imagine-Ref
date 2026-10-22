"""CPU regressions and real-model integration. Reuses the supplied test entry point."""
import collections
import copy
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from search_a import (SearchOutcomeHead, SearchAController, macro_actions, actor_action_sequence,
                      check_checkpoint, normalize_feature, MODEL_VERSION)
from search_evidence import (SearchEvidence, SearchEvidenceWrapper, SEARCH_LABEL_KEYS,
                             complete_episode_labels, lidar_rays)

torch.set_num_threads(2)


def config_for_test(**overrides):
    import ruamel.yaml as yaml
    raw = yaml.YAML(typ="safe").load((pathlib.Path(__file__).resolve().parents[1]/"configs.yaml").read_text())
    config = {}
    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and key in dst:
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)
    for name in ("defaults", "minedojo", "apm_a", "apm_a_smoke"):
        merge(config, raw[name])
    config.update(device="cpu", num_actions=12, episode_max_steps=24, search_a_start_step=0,
                  search_a_override_prob=1.0, search_a_low_evidence=1.1, search_a_min_labels=1,
                  search_a_min_positives=0)
    config.update(overrides)
    return SimpleNamespace(**config)


def replay_batch(time=12):
    rng = np.random.RandomState(4)
    data = dict(image=rng.randint(256,size=(1,time,64,64,3),dtype=np.uint8),
                heatmap=rng.randint(256,size=(1,time,64,64),dtype=np.uint8),
                action=np.eye(12,dtype=np.float32)[(np.arange(time)%12)[None]],
                is_first=np.zeros((1,time),bool), discount=np.ones((1,time),np.float32))
    data["is_first"][:,0] = True
    for key in ("is_terminal","is_last","is_zoomed","jump","is_calculated","reward","intrinsic",
                "reward_on_zoomed","intrinsic_on_zoomed","jumping_steps","accumulated_reward"):
        data[key] = np.zeros((1,time),np.float32)
    data["zoomed_image"] = data["image"].copy()
    data["heatmap_on_zoomed"] = data["heatmap"].copy()
    for key in SEARCH_LABEL_KEYS:
        data[key] = np.zeros((1,time),np.float32)
    data["search_label_version"][:] = MODEL_VERSION
    for event in ("new","discovery","revisit"):
        data["search_"+event+"_valid"][:] = 1
        data["search_"+event][:, 7] = 1
    return data


def make_agent(config=None, seed=5):
    import gym
    from expr import LS_Imagine
    config = config or config_for_test()
    spaces = {
        "image":gym.spaces.Box(0,255,(64,64,3),np.uint8),
        "heatmap":gym.spaces.Box(0,255,(64,64,1),np.uint8),
        # Deliberately use a vector and a broad encoder regex in the leak test.
        "search_discovery":gym.spaces.Box(0,1,(1,),np.float32)}
    torch.manual_seed(seed)
    agent = LS_Imagine(gym.spaces.Dict(spaces),gym.spaces.Discrete(12),config,
                       SimpleNamespace(step=24),None)
    agent.requires_grad_(False)
    return agent


def fake_rays(block="stone", entity="null", block_distance=5., entity_distance=5., x=0):
    count = 45
    return {"rays":dict(block_name=np.full(count,block), entity_name=np.full(count,entity),
                       block_distance=np.full(count,block_distance),
                       entity_distance=np.full(count,entity_distance),
                       traced_block_x=np.arange(count)+x,
                       traced_block_y=np.zeros(count),traced_block_z=np.zeros(count))}


class EvidenceTests(unittest.TestCase):
    def test_target_aliases_and_occlusion(self):
        for target, block, entity in (("log","oak wood","null"),("sand","sand","null"),
                                     ("water_bucket","water","null"),("iron_ore","iron ore","null"),
                                     ("wool","stone","sheep")):
            collector=SearchEvidence(target)
            collector.observe(fake_rays(), initial=True)
            obs=fake_rays(block,entity,entity_distance=3.)
            labels=collector.observe(obs)
            self.assertEqual(labels["search_discovery"],1)
            self.assertEqual(labels["search_discovery_valid"],1)
        collector=SearchEvidence("log")
        labels=collector.observe(fake_rays("oak wood","cow",entity_distance=2.),initial=True)
        self.assertEqual(labels["search_visible"],0)
        labels=SearchEvidence("wool").observe(fake_rays(entity="sheep",entity_distance=8.),initial=True)
        self.assertEqual(labels["search_visible"],0)  # sheep behind a nearer block

    def test_unknown_labels_and_positive_transition(self):
        c=SearchEvidence("log")
        self.assertEqual(c.observe({},initial=True)["search_discovery_valid"],0)
        r=c.observe(fake_rays("oak wood"))
        self.assertEqual(r["search_visible"],1)
        self.assertEqual(r["search_discovery_valid"],0)  # previous observation unknown
        self.assertEqual(c.observe(fake_rays("oak wood"))["search_discovery"],0)
        c.observe(fake_rays())
        self.assertEqual(c.observe(fake_rays("oak wood"))["search_discovery"],1)
        broken=fake_rays();broken["rays"]["block_distance"][0]=np.nan
        self.assertEqual(c.observe(broken)["search_visible_valid"],0)

    def test_surface_new_revisit_and_episode_reset(self):
        c=SearchEvidence("log",revisit_gap=2)
        self.assertEqual(c.observe(fake_rays(),initial=True)["search_new"],1)
        self.assertEqual(c.observe(fake_rays())["search_revisit"],0)
        self.assertEqual(c.observe(fake_rays())["search_revisit"],1)
        self.assertEqual(c.observe(fake_rays(x=100))["search_new"],1)
        self.assertEqual(c.observe(fake_rays(),initial=True)["search_revisit"],0)
        sky=fake_rays(block="air")
        self.assertEqual(c.observe(sky)["search_new_valid"],0)
        self.assertEqual(len(lidar_rays()),45)

    def test_eval_wrapper_never_uses_privileged_data(self):
        import gym
        env=gym.Env()
        env.reset=lambda:fake_rays("oak wood")
        env.step=lambda action:(fake_rays("oak wood"),0.,False,{})
        wrapper=SearchEvidenceWrapper(env,enabled=False)
        obs=wrapper.reset()
        self.assertEqual(obs["search_discovery_valid"],0)
        self.assertEqual(obs["search_visible_valid"],0)

    def test_minedojo_sensor_enabled_only_for_label_collection(self):
        from envs.tasks.minedojo import make_minedojo
        task=dict(search_evidence_specs=dict(enabled=True,target="log",distance=32.))
        with mock.patch("envs.tasks.minedojo._meta_task_make",return_value=object()) as make:
            with mock.patch("envs.tasks.minedojo._add_wrappers",side_effect=lambda env,*a,**k:env):
                make_minedojo("harvest",task,{})
                self.assertTrue(make.call_args.kwargs["use_lidar"])
                self.assertEqual(len(make.call_args.kwargs["lidar_rays"]),45)
                task["search_evidence_specs"]["enabled"]=False
                make_minedojo("harvest",task,{})
                self.assertFalse(make.call_args.kwargs["use_lidar"])


class PredictionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model=SearchOutcomeHead(32,horizon=6,hidden=32)
        self.feat=torch.randn(1,18,32,requires_grad=True)
        self.batch={
            "action":F.pad(F.one_hot(torch.arange(18)[None]%12,12).float(),(0,1)),
            "is_first":torch.zeros(1,18),
            "is_zoomed":torch.ones(1,18),
            "search_label_version":torch.full((1,18),MODEL_VERSION)}
        self.batch["is_first"][:,0]=1

    def test_shapes_alignment_and_stop_gradient_targets(self):
        with mock.patch.object(self.model,"forward",wraps=self.model.forward) as call:
            loss,metrics,_,_=self.model.training_loss(self.feat,self.batch)
        inputs,windows=call.call_args.args
        torch.testing.assert_close(inputs,self.feat[:,:-6])
        torch.testing.assert_close(windows[:,0],self.batch["action"][:,1:7,:12])
        self.assertEqual(metrics["A/valid_windows"].item(),12)
        loss.backward()
        self.assertGreater(self.feat.grad[:,:-6].abs().sum().item(),0)
        self.assertEqual(self.feat.grad[:,-6:].abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_boundaries_and_auxiliary_zoom_does_not_remove_real_frames(self):
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),12)
        self.batch["action"][:,8,12]=1
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),6)
        self.batch["action"][:,:,12]=0
        self.batch["is_terminal"]=torch.zeros(1,18)
        self.batch["is_terminal"][:,8]=1
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),6)
        self.batch["is_first"][:,1:]=1
        loss,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(loss.item(),0)

    def test_unknown_is_not_negative_and_any_positive_is_known(self):
        self.batch["search_discovery"]=torch.full((1,18),float("nan"))
        self.batch["search_discovery_valid"]=torch.zeros(1,18)
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/discovery_valid"].item(),0)
        self.batch["search_discovery"][:,6]=1
        self.batch["search_discovery_valid"][:,6]=1
        _,m,counts,positive=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/discovery_valid"].item(),6)
        self.assertEqual(m["A/discovery_positive_rate"].item(),1)
        self.assertFalse(self.model.ready_events(1,0).any())
        self.model.record_supervision(counts,positive)
        self.assertFalse(self.model.ready_events(1,0)[1])  # no negative example yet
        self.batch["search_discovery"][:]=0
        self.batch["search_discovery_valid"][:]=1
        _,_,counts,positive=self.model.training_loss(self.feat,self.batch)
        self.model.record_supervision(counts,positive)
        self.assertTrue(self.model.ready_events(1,1)[1])

    def test_revisit_is_endpoint_target_not_any_in_window(self):
        self.batch["search_revisit"]=torch.zeros(1,18)
        self.batch["search_revisit"][:,1]=1
        self.batch["search_revisit_valid"]=torch.ones(1,18)
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/revisit_positive_rate"].item(),0)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.head=SearchOutcomeHead(32,hidden=32)
        self.head.label_counts[:]=10
        self.head.positive_counts[:]=5
        self.obs={"heatmap":torch.zeros(2,64,64,1)}
        self.feat=torch.zeros(2,32)
        self.actor=F.one_hot(torch.tensor([0,0]),12).float()

    def choose(self, ctrl, state=None, first=None, best=0, training=False):
        def predict(feat,seq):
            logits=torch.full((*feat.shape[:-1],3),-8.)
            logits[:,best,1]=8.
            return {"future":torch.zeros_like(feat),"event_logits":logits}
        continuation=lambda indices:self.actor[indices,None].expand(-1,6,-1)
        with mock.patch.object(self.head,"forward",side_effect=predict):
            return ctrl.choose(self.head,self.feat,self.obs,self.actor,state,
                               torch.zeros(2) if first is None else first,100,training,continuation)

    def controller(self, **kw):
        defaults=dict(start_step=0,override_prob=1.,epsilon=0.,min_labels=1,min_positives=1)
        defaults.update(kw)
        return SearchAController(**defaults)

    def test_model_selects_macro_or_actor(self):
        out,state,info=self.choose(self.controller(),best=0)
        self.assertEqual(out.argmax(-1).tolist(),[4,4])
        self.assertEqual(info["log_search_override"].tolist(),[1,1])
        out,state,info=self.choose(self.controller(),best=5)
        self.assertEqual(out.argmax(-1).tolist(),[0,0])
        self.assertEqual(info["log_search_override"].sum().item(),0)
        self.assertEqual(info["log_search_decision"].sum().item(),2)

    def test_reset_finish_and_deterministic_schedule(self):
        ctrl=self.controller(override_prob=.25)
        state=None
        rng=torch.random.get_rng_state()
        for t in range(9):
            out,state,info=self.choose(ctrl,state=state)
            self.assertEqual(info["log_search_override"][0].item(),float(t>=3))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(state[0].tolist(),[-1,-1])
        ctrl=self.controller()
        _,state,_=self.choose(ctrl)
        ctrl.override_prob=0
        _,state,info=self.choose(ctrl,state,first=torch.tensor([1,0]))
        self.assertEqual(info["log_search_override"].tolist(),[0,1])

    def test_unknown_event_heads_do_not_control_eval(self):
        self.head.label_counts.zero_()
        _,_,info=self.choose(self.controller())
        self.assertEqual(info["log_search_decision"].sum().item(),0)

    def test_actions_stay_twelve_dimensional(self):
        actions=macro_actions(6,"cpu")
        self.assertEqual(tuple(actions.shape),(5,6,12))
        self.assertTrue((actions.sum(-1)==1).all())
        self.assertEqual(actions[:,0].argmax(-1).tolist(),[4,5,6,6,6])


class ReplayAndIntegrationTests(unittest.TestCase):
    def test_mixed_old_new_replay_and_npz(self):
        import tools
        old=replay_batch(18)
        old={k:list(v[0]) for k,v in old.items() if not k.startswith("search_")}
        new=copy.deepcopy(old)
        for key in SEARCH_LABEL_KEYS:
            new[key]=[np.float32(MODEL_VERSION if key=="search_label_version" else 1)]*18
        cache=collections.OrderedDict(old=old,new=new)
        with tempfile.TemporaryDirectory() as directory:
            tools.save_episodes(directory,cache)
            loaded=tools.load_episodes(directory)
            generator=tools.sample_episodes(loaded,24)
            batch=next(tools.from_generator(generator,4))
        self.assertEqual(batch["action"].shape,(4,24,12))
        self.assertTrue((batch["search_discovery_valid"]==0).any())
        self.assertTrue((batch["search_discovery_valid"]==1).any())
        self.assertEqual({len(v) for v in complete_episode_labels(old).values()},{18})

    def test_metadata_backfills_entire_prefill(self):
        import tools
        cache={}
        for t in range(8):
            row=dict(reward=0.,image=np.zeros((4,4,3),np.uint8))
            if t>=5:
                row.update(log_search_override=np.float32(1),
                           log_search_scores=np.zeros(6,np.float32))
            tools.add_to_cache(cache,"ep",row)
        self.assertEqual({len(v) for v in cache["ep"].values()},{8})
        self.assertEqual(np.asarray(cache["ep"]["log_search_scores"]).shape,(8,6))

    def test_real_joint_update_actual_action_and_checkpoint(self):
        import tools
        agent=make_agent(config_for_test(search_a_epsilon=1.))
        data=replay_batch()
        data["is_zoomed"][:,2]=1
        data["is_calculated"][:,2]=1
        data["jumping_steps"][:,2]=2
        before=next(agent._wm.search_a.parameters()).detach().clone()
        agent._train(data)
        self.assertFalse(torch.equal(before,next(agent._wm.search_a.parameters())))
        self.assertTrue(all(np.isfinite(v).all() for values in agent._metrics.values() for v in values))
        self.assertGreater(agent._wm.search_a.label_counts.sum().item(),0)
        obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
        out,state=agent._policy(obs,None,training=True)
        self.assertEqual(out["log_search_override"].sum().item(),1)
        obs["is_first"][:]=False
        with mock.patch.object(agent._wm.dynamics,"obs_step",wraps=agent._wm.dynamics.obs_step) as call:
            agent._policy(obs,state,training=True)
        torch.testing.assert_close(call.call_args.args[1],out["action"])
        with mock.patch.object(agent._wm.dynamics,"img_step",wraps=agent._wm.dynamics.img_step) as call:
            seq=actor_action_sequence(agent._wm.dynamics,agent._task_behavior.actor,state[0],out["action"],6)
        self.assertEqual(seq.shape,(1,6,12))
        self.assertTrue(all((args.args[1][:,-1]==0).all() for args in call.call_args_list))
        checkpoint=dict(agent_state_dict=agent.state_dict(),step=agent._step,search_a_version=MODEL_VERSION,
                        optims_state_dict=tools.recursively_collect_optim_state_dict(agent))
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/"latest.pt";torch.save(checkpoint,p);checkpoint=torch.load(p,map_location="cpu")
        check_checkpoint(checkpoint,True)
        other=make_agent(config_for_test(search_a_control=False))
        other.load_state_dict(checkpoint["agent_state_dict"],strict=True)
        tools.recursively_load_optim_state_dict(other,checkpoint["optims_state_dict"])
        self.assertIsNone(other._search_a)
        torch.testing.assert_close(other._wm.search_a.label_counts,agent._wm.search_a.label_counts)
        self.assertNotIn("log_search_override",other._policy(obs,None,False)[0])

    def test_privileged_labels_never_enter_encoder_or_policy(self):
        cfg=config_for_test(search_a_control=False)
        cfg.encoder["mlp_keys"]=".*"
        agent=make_agent(cfg)
        self.assertNotIn("search_discovery",agent._wm.encoder.mlp_shapes)
        data=replay_batch()
        obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
        with mock.patch.object(agent._wm.encoder,"forward",wraps=agent._wm.encoder.forward) as call:
            torch.manual_seed(7);out1,state1=agent._policy(obs,None,False)
        self.assertFalse(any(k.startswith("search_") for k in call.call_args.args[0]))
        changed={k:(v+1000 if k.startswith("search_") else v) for k,v in obs.items()}
        torch.manual_seed(7);out2,state2=agent._policy(changed,None,False)
        torch.testing.assert_close(out1["action"],out2["action"])
        torch.testing.assert_close(state1[0]["deter"],state2[0]["deter"])

    def test_master_off_and_frozen_loss(self):
        off=make_agent(config_for_test(search_a_enabled=False))
        on=make_agent()
        self.assertIsNone(off._wm.search_a)
        self.assertIsNone(off._search_a)
        for (name,p),(other,q) in zip(off._task_behavior.actor.named_parameters(),on._task_behavior.actor.named_parameters()):
            self.assertEqual(name,other);torch.testing.assert_close(p,q)
        self.assertFalse(any(".search_a." in k for k in off.state_dict()))
        frozen=make_agent(config_for_test(search_a_train=False,search_a_control=False))
        before={k:v.clone() for k,v in frozen._wm.search_a.state_dict().items()}
        frozen._train(replay_batch())
        for k,v in frozen._wm.search_a.state_dict().items():
            torch.testing.assert_close(v,before[k])
        for overrides in (dict(search_a_train=False), dict(search_a_loss_weight=0.),
                          dict(search_a_event_weight=0.)):
            inactive=make_agent(config_for_test(search_a_epsilon=1., **overrides))
            self.assertEqual(inactive._search_a.epsilon,0.)
            data=replay_batch()
            obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
            out,_=inactive._policy(obs,None,True)
            self.assertEqual(out["log_search_override"].item(),0.)

    def test_reject_v1_checkpoint_and_wrong_off_mode(self):
        a=make_agent()
        cp=dict(agent_state_dict=a.state_dict())
        with self.assertRaises(ValueError):check_checkpoint(cp,True)
        cp["search_a_version"]=MODEL_VERSION
        with self.assertRaises(ValueError):check_checkpoint(cp,False)


if __name__=="__main__":
    unittest.main(verbosity=2)
