"""Integration checks for the task-preserving vNext path; no simulator required."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))
from test_search_a import config_for_test, make_agent, replay_batch
from search_vnext import _hash_int, check_vnext_checkpoint
from search_evidence import VNEXT_LABEL_KEYS
from expr import make_dataset
import tools
from training_state import save_training_checkpoint, prepare_resume, restore_training_state

torch.set_num_threads(2)

def config(**kw):
    values=dict(search_vnext_enabled=True,search_a_enabled=False,search_b_enabled=False,
        search_c_enabled=False,search_branch_enabled=False,compile=False,
        vnext_hidden=32,vnext_memory_dim=16,vnext_slots=4,
        vnext_future_dim=16,vnext_bootstrap_heads=3,vnext_start_step=0,
        vnext_warmup=8,vnext_full_step=16,vnext_base_episode_fraction=0.,
        batch_size=2,batch_length=12,episode_max_steps=48,
        episode_max_steps_override=48,precision=32)
    values.update(kw)
    return config_for_test(**values)

def episode(agent,length=40):
    raw=replay_batch(length+1)
    for key in VNEXT_LABEL_KEYS:
        raw.setdefault(key,np.zeros((1,length+1),np.float32))
    raw["search_label_version"][:]=3
    for name in ("visible","center_target","reachable","center_interactable","progress","blocked","return","dwell"):
        raw["search_"+name+"_valid"][:]=1
    raw["search_visible"][:,4:18]=1
    raw["search_center_target"][:,7:15]=1
    raw["search_reachable"][:,7:15]=1
    raw["search_center_interactable"][:,7:15]=1
    raw["search_progress"][:,12]=1
    for name in ("new","discovery","revisit"):
        raw["search_"+name+"_valid"][:]=1
    raw["reward"][:,12]=2.
    raw["is_last"][:,-1]=True
    raw["success"]=np.zeros((1,length+1),np.float32)
    raw["success"][:,12]=1.
    raw["first_success_step"]=np.full((1,length+1),12,np.int32)
    cache={}
    def obs(t):
        return {k:v[:,t] for k,v in raw.items() if k!="action"}
    initial={k:v[0].copy() for k,v in obs(0).items()}
    initial["action"]=np.zeros(12,np.float32)
    tools.add_to_cache(cache,"fixture",initial)
    state=None
    for t in range(length):
        with torch.no_grad():
            out,state=agent._policy(obs(t),state,training=True)
        transition={k:v[0].copy() for k,v in obs(t+1).items()}
        transition.update({k:v[0].detach().cpu().numpy() for k,v in out.items()})
        tools.add_to_cache(cache,"fixture",transition)
    result={k:np.asarray(v) for k,v in cache["fixture"].items()}
    group=next(i for i in range(100) if _hash_int(str(i))%5!=0)
    result["log_vnext_world_group"]=np.full(length+1,group,np.int64)
    return result

class IntegrationTests(unittest.TestCase):
    def test_base_initialization_and_task_shapes_unchanged(self):
        base=make_agent(config(search_vnext_enabled=False),seed=19)
        new=make_agent(config(),seed=19)
        for prefix in ("_wm","_task_behavior"):
            left=getattr(base,prefix).state_dict()
            right=getattr(new,prefix).state_dict()
            self.assertEqual(set(left),set(right))
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        self.assertIsNone(new._wm.search_a)
        self.assertIsNone(new._wm.search_b)
        self.assertIsNone(new._search_c)

    def test_joint_update_real_replay_memory_and_c_optimizers(self):
        cfg=config()
        agent=make_agent(cfg)
        replay={"fixture":episode(agent)}
        self.assertEqual({len(v) for v in replay["fixture"].values()},{41})
        batch=next(make_dataset(replay,cfg))
        before=agent._vnext.a.current_heads[0].weight.detach().clone()
        agent._train(batch)
        self.assertTrue(agent._metrics["A/update_applied"][-1])
        self.assertGreater(agent._metrics["C/transitions"][-1],0)
        self.assertTrue(agent._metrics["C/update_applied"][-1])
        self.assertFalse(torch.equal(before,agent._vnext.a.current_heads[0].weight))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in agent._metrics.values()))
        states=tools.recursively_collect_optim_state_dict(agent)
        self.assertTrue(any("_vnext." in k for k in states))
        self.assertTrue(any("_vnext_control." in k for k in states))

    def test_auxiliary_update_does_not_change_task_or_backprop_feature(self):
        cfg=config()
        agent=make_agent(cfg)
        data=next(make_dataset({"fixture":episode(agent)},cfg))
        fd=cfg.dyn_deter+cfg.dyn_stoch*cfg.dyn_discrete
        features=torch.randn(cfg.batch_size,cfg.batch_length,fd,requires_grad=True)
        original={k:v.detach().clone() for k,v in agent._wm.state_dict().items()}
        agent._vnext.train_step(features,data,24)
        self.assertIsNone(features.grad)
        for key,value in agent._wm.state_dict().items():
            torch.testing.assert_close(value,original[key],rtol=0,atol=0)

    def test_control_off_exact_base_action_and_no_oracle_input(self):
        cfg=config(vnext_control=False)
        base=make_agent(config(search_vnext_enabled=False),seed=17)
        new=make_agent(cfg,seed=17)
        raw=replay_batch(12)
        obs={k:v[:,0] for k,v in raw.items() if k!="action"}
        state=torch.get_rng_state()
        expected,_=base._policy(obs,None,False)
        torch.set_rng_state(state)
        actual,_=new._policy(obs,None,False)
        torch.testing.assert_close(expected["action"],actual["action"],rtol=0,atol=0)
        noisy=dict(obs)
        noisy["search_center_interactable"]=np.array([np.nan])
        noisy["search_oracle_vector"]=np.full((1,800),1e10)
        noisy["log_applied_control"]=np.full((1,9),1e10)
        torch.set_rng_state(state)
        again,_=new._policy(noisy,None,False)
        torch.testing.assert_close(actual["action"],again["action"],rtol=0,atol=0)

    def test_mixed_prefill_and_learned_episodes_have_fixed_batch_schema(self):
        from search_vnext_control import build_transitions
        cfg=config(batch_size=16,batch_length=32)
        agent=make_agent(cfg)
        learned=episode(agent,48)
        prefill={k:v.copy() for k,v in learned.items()
                 if not k.startswith(("log_vnext_","log_search_"))}
        data=next(make_dataset({"prefill":prefill,"learned":learned},cfg))
        self.assertEqual(data["log_vnext_mask"].shape,(16,32,12))
        self.assertEqual(data["log_vnext_memory_refs"].shape,(16,32,4))
        self.assertGreater(len(build_transitions(data)),0)
        # A pure-prefill batch already exposes zero-valid C fields, so keys
        # remain stable when learned episodes are added later.
        only={"prefill":prefill}
        stream=make_dataset(only,cfg)
        initial=next(stream)
        only["learned"]=learned
        later=next(stream)
        self.assertEqual(set(initial),set(later))
        self.assertEqual(initial["log_vnext_version"].sum(),0)

    def test_base_window_sampling_matches_original_distribution(self):
        from search_vnext import sample_episodes_vnext
        one={k:v[0] for k,v in replay_batch(13).items()}
        two={k:v[0] for k,v in replay_batch(19).items()}
        episodes={"one":one,"two":two}
        old=tools.sample_episodes(episodes,12,seed=37)
        new=sample_episodes_vnext(episodes,12,slots=4,seed=37)
        for _ in range(12):
            a,b=next(old),next(new)
            for key in ("image","heatmap","action","is_first","is_last","reward"):
                np.testing.assert_array_equal(a[key],b[key])

    def test_prefill_mid_episode_actuator_and_memory_indices(self):
        cfg=config(vnext_control=False)
        agent=make_agent(cfg)
        raw=replay_batch(12)
        obs={k:v[:,0] for k,v in raw.items() if k!="action"}
        obs["is_first"][:]=False
        obs["vnext_bootstrap_actuator"]=np.array([[0,7,-30]],np.float32)
        obs["vnext_bootstrap_control"]=np.array([[0,1,0,0,0,1,0,0,0]],np.float32)
        obs["vnext_episode_step"]=np.array([7])
        out,state=agent._policy(obs,None,False)
        torch.testing.assert_close(out["log_vnext_actuator_before"],
                                   torch.tensor([[0.,7.,-30.]]))
        self.assertEqual(int(out["log_vnext_remaining"][0]),41)
        cache={"episode":{"reward":[0.]*8}}
        transition={k:v[0].detach().cpu().numpy() for k,v in out.items()}
        transition["reward"]=0.
        tools.add_to_cache(cache,"episode",transition)
        refs=np.asarray(cache["episode"]["log_vnext_memory_refs"][-1])
        self.assertTrue(np.all(refs[refs>=0]==7))
        meta=np.asarray(cache["episode"]["log_vnext_memory_meta"][-1])
        self.assertTrue(np.all(meta[refs>=0,1]==7))

    def test_formal_auxiliary_dimensions_and_batch(self):
        from search_vnext import SearchVNext
        cfg=config(vnext_slots=16,vnext_memory_dim=128,vnext_hidden=256,
                   vnext_future_dim=128,batch_size=16,batch_length=32)
        generator_agent=make_agent(cfg)
        data=next(make_dataset({"fixture":episode(generator_agent,48)},cfg))
        module=SearchVNext(5120,cfg)
        feature=torch.randn(16,32,5120,requires_grad=True)
        metrics=module.train_step(feature,data,2500)
        self.assertIsNone(feature.grad)
        self.assertTrue(np.isfinite(float(metrics["A/prediction_loss"])))
        self.assertEqual(module.memory_batch(feature,data).shape,(16,32,128))

    def test_active_replay_prefix_is_atomic_and_not_a_false_terminal(self):
        from expr import save_active_replay
        from unittest import mock
        cfg=config()
        agent=make_agent(cfg)
        ep=episode(agent,16)
        prefix={k:v[:9].copy() for k,v in ep.items()}
        prefix["is_last"][-1]=False
        with tempfile.TemporaryDirectory() as directory:
            save_active_replay([SimpleNamespace(id="active")],{"active":prefix},directory)
            path=Path(directory)/"active-9.npz"
            with np.load(path) as archive:
                self.assertEqual(len(archive["reward"]),9)
                self.assertFalse(archive["is_last"][-1])
            before=path.read_bytes()
            with mock.patch.object(tools.os,"replace",side_effect=OSError("injected disk failure")):
                with self.assertRaises(OSError):
                    tools.save_episodes(directory,{"active":prefix})
            self.assertEqual(before,path.read_bytes())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_checkpoint_roundtrip_and_legacy_resume_rejected(self):
        cfg=config()
        agent=make_agent(cfg)
        with tempfile.TemporaryDirectory() as directory:
            cfg.logdir=Path(directory)/"source";cfg.traindir=cfg.logdir/"train_eps"
            cfg.logdir.mkdir();cfg.traindir.mkdir()
            data=episode(agent)
            tools.save_episodes(cfg.traindir,{"fixture":data})
            agent._train(next(make_dataset({"fixture":data},cfg)))
            saved=save_training_checkpoint(agent,cfg,100000)
            ckpt=torch.load(saved,map_location="cpu")
            check_vnext_checkpoint(ckpt,True)
            self.assertEqual(ckpt["search_a_label_version"],3)
            with self.assertRaises(ValueError):
                check_vnext_checkpoint(ckpt,False)
            nextcfg=copy.deepcopy(cfg);nextcfg.resume_from=str(saved)
            nextcfg.logdir=str(Path(directory)/"resume");nextcfg.traindir=None
            nextcfg.evaldir=None;nextcfg.steps=1000
            plan=prepare_resume(nextcfg)
            other=make_agent(nextcfg,seed=29)
            restore_training_state(other,plan,SimpleNamespace(step=0))
            for key,value in agent.state_dict().items():
                torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(agent._vnext_control.rng_state()[0],other._vnext_control.rng_state()[0])
            with self.assertRaisesRegex(ValueError,"legacy/vNext"):
                bad=config(search_vnext_enabled=False,resume_from=str(saved),logdir=str(Path(directory)/"bad"))
                prepare_resume(bad)

if __name__=="__main__":
    unittest.main(verbosity=2)

