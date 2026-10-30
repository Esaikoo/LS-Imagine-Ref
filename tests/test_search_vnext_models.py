"""Meaningful CPU regressions for independent vNext A/B and causal replay."""
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from search_vnext import (SearchVNext,SlotMemory,sample_episodes_vnext,CURRENT_EVENTS,
                         FUTURE_EVENTS,check_vnext_checkpoint,restore_vnext_config)


def config():
    return SimpleNamespace(vnext_slots=3,vnext_memory_dim=16,vnext_hidden=32,
        vnext_future_dim=8,vnext_bootstrap_heads=3,vnext_aux_horizon=8,
        vnext_aux_every=4,vnext_lr=.001,vnext_memory_enabled=True)


def episode(n=12,slots=3,offset=0):
    rng=np.random.RandomState(2026+offset)
    ep=dict(image=rng.randint(0,255,(n,64,64,3),dtype=np.uint8),
        heatmap=rng.randint(0,255,(n,64,64),dtype=np.uint8),
        is_first=np.arange(n)==0,is_last=np.arange(n)==n-1,
        is_terminal=np.arange(n)==n-1,reward=np.zeros(n,np.float32),
        action=np.eye(12,dtype=np.float32)[np.arange(n)%12],
        search_label_version=np.full(n,3,np.float32),
        log_applied_control=np.zeros((n,9),np.float32),
        log_actuator_after=np.zeros((n,3),np.float32),
        log_vnext_choice=np.zeros(n,np.int32),log_vnext_phase=np.zeros(n,np.int32),
        search_place_id=np.arange(n,dtype=np.float32)+1,
        search_place_valid=np.ones(n,np.float32),
        log_vnext_memory_refs=np.full((n,slots),-1,np.int32),
        log_vnext_memory_meta=np.zeros((n,slots,8),np.float32))
    for i in range(1,n):
        ep["log_vnext_memory_refs"][i,0]=i-1
    for j,name in enumerate(set(CURRENT_EVENTS+FUTURE_EVENTS)):
        ep["search_"+name]=(np.arange(n)%(j+2)==0).astype(np.float32)
        ep["search_"+name+"_valid"]=np.ones(n,np.float32)
    ep["reward"][-1]=1
    return ep


def batch(n=12,slots=3,b=2):
    ep=episode(n,slots)
    generator=sample_episodes_vnext({"episode0":ep},n,slots,seed=11)
    rows=[next(generator) for _ in range(b)]
    return {k:np.stack([x[k] for x in rows]) for k in rows[0]}


class VNextModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stitched_success_is_real_not_padding(self):
        a=episode(3,offset=1);b=episode(4,offset=2)
        a["image"][:]=11;b["image"][:]=22
        g=sample_episodes_vnext({"a":a,"b":b},16,3,seed=5)
        data=next(g)
        self.assertEqual(len(data["reward"]),16)
        self.assertTrue(np.all(np.isin(data["image"][:,0,0,0],[11,22])))
        self.assertEqual(data["vnext_window_valid"].sum(),16)
        self.assertGreater(data["vnext_real_first"].sum(),1)
        self.assertGreater(data["reward"].sum(),1)

    def test_memory_row_shift_and_scope(self):
        e=episode(10)
        g=sample_episodes_vnext({"e":e},10,3,seed=2)
        data=next(g)
        for t in range(10):
            ids=data["vnext_memory_indices"][t]
            for idx in ids[ids>=0]:
                keyframe=data["vnext_memory_frames"][idx]
                # A bank may only point at an existing actual source frame.
                self.assertTrue(any(np.array_equal(keyframe[...,:3],im) for im in e["image"]))
        self.assertLessEqual((data["vnext_memory_indices"]>=0).sum(),10)

    def test_future_reference_rejected(self):
        ep=episode(4)
        ep["log_vnext_memory_refs"][1:,0]=3
        with self.assertRaisesRegex(ValueError,"future"):
            next(sample_episodes_vnext({"bad":ep},12,3))

    def test_no_memory_for_prefill(self):
        ep=episode(4)
        del ep["log_vnext_memory_refs"];del ep["log_vnext_memory_meta"]
        d=next(sample_episodes_vnext({"prefill":ep},8,3))
        self.assertFalse(d["vnext_memory_valid"].any())

    def test_train_detaches_main_features_and_updates_modules(self):
        model=SearchVNext(24,config())
        d=batch()
        d["vnext_calibration"][:]=0
        feat=torch.randn(2,12,24,requires_grad=True)
        a=[p.detach().clone() for p in model.a.parameters()]
        b=[p.detach().clone() for p in model.b.parameters()]
        metrics=model.train_step(feat,d)
        self.assertIsNone(feat.grad)
        self.assertEqual(metrics["A/update_applied"],1)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a,model.a.parameters())))
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(b,model.b.parameters())))
        memory=model.memory_batch(feat,d)
        self.assertEqual(memory.shape,(2,12,16))
        current=model.current(feat[:,0],memory[:,0])
        self.assertEqual(current["probs"].shape,(2,5))
        out=model.predict(feat[:,0],memory[:,0],torch.zeros(2,12,3,9),torch.zeros(2,3))
        self.assertEqual(out["descriptor"].shape,(2,12,12))
        self.assertFalse(out["support"].any())

    def test_current_and_future_unknown_not_negative(self):
        model=SearchVNext(24,config())
        d=batch()
        d["vnext_calibration"][:]=0
        d["search_label_version"][:]=2
        metrics=model.train_step(torch.randn(2,12,24),d)
        self.assertEqual(float(model.a.current_counts.sum()),0)
        self.assertEqual(float(model.a.support_counts.sum()),0)
        self.assertTrue(torch.isfinite(metrics["A/prediction_loss"]))

    def test_empty_and_permutation_invariant(self):
        m=SlotMemory(24,config())
        f=torch.randn(2,24,requires_grad=True)
        keys=torch.randn(2,3,16);meta=torch.randn(2,3,8)
        valid=torch.ones(2,3,dtype=torch.bool)
        a,_=m.read(f,keys,meta,valid)
        order=torch.tensor([2,0,1])
        b,_=m.read(f,keys[:,order],meta[:,order],valid[:,order])
        torch.testing.assert_close(a,b)
        empty,_=m.read(f,keys,meta,torch.zeros_like(valid))
        self.assertFalse(empty.any())
        a.sum().backward()
        self.assertIsNone(f.grad)

    def test_online_bank_is_past_and_reset(self):
        model=SearchVNext(24,config())
        feat=torch.randn(2,24)
        image=torch.zeros(2,64,64,3,dtype=torch.uint8)
        heat=torch.zeros(2,64,64,dtype=torch.uint8)
        current,memory,state,info=model.online(feat,image,heat,is_first=torch.ones(2))
        self.assertFalse(memory.any())
        self.assertEqual(info["log_vnext_memory_refs"][:,0].tolist(),[0,0])
        _,memory,state,_=model.online(feat,image,heat,state,is_first=torch.zeros(2))
        self.assertTrue(memory.any())
        _,memory,_,_=model.online(feat,image,heat,state,is_first=torch.ones(2))
        self.assertFalse(memory.any())

    def test_calibration_never_trains_or_recounts_duplicates(self):
        model=SearchVNext(24,config())
        d=batch()
        d["vnext_calibration"][:]=1
        d["vnext_memory_calibration"][:]=1
        weights=[x.detach().clone() for x in model.parameters()]
        features=torch.randn(2,12,24)
        first=model.train_step(features,d)
        count=model.a.current_cal_counts.clone()
        self.assertGreater(float(count.sum()),0)
        second=model.train_step(features,d)
        torch.testing.assert_close(model.a.current_cal_counts,count)
        self.assertEqual(second["A/update_applied"],0)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(weights,model.parameters())))
        old=model.a.current_cal_errors.clone()
        with torch.no_grad():
            for head in model.a.current_heads:head.bias.add_(2)
        model.train_step(features,d)
        torch.testing.assert_close(model.a.current_cal_counts,count)
        self.assertFalse(torch.equal(old,model.a.current_cal_errors))

    def test_environment_error_invalidates_factual_windows(self):
        model=SearchVNext(24,config());d=batch()
        d["vnext_calibration"][:]=0
        d["log_env_error"]=np.ones((2,12),np.float32)
        model.train_step(torch.randn(2,12,24),d)
        self.assertEqual(float(model.a.support_counts.sum()),0)

    def test_raw_replay_and_online_pixels_encode_identically(self):
        m=SlotMemory(24,config())
        raw=torch.randint(0,256,(5,64,64,4),dtype=torch.uint8)
        torch.testing.assert_close(m.encoder(raw),m.encoder(raw.float()/255.))

    def test_formal_batch_and_window(self):
        m=SearchVNext(128,SimpleNamespace(vnext_slots=16,vnext_memory_dim=128,
            vnext_hidden=64,vnext_future_dim=32,vnext_bootstrap_heads=3,
            vnext_aux_horizon=8,vnext_aux_every=4))
        d=batch(n=32,slots=16,b=16);d["vnext_calibration"][:]=0
        features=torch.randn(16,32,128,requires_grad=True)
        metrics=m.train_step(features,d)
        self.assertIsNone(features.grad)
        self.assertTrue(torch.isfinite(metrics["A/prediction_loss"]))
        self.assertEqual(m.memory_batch(features,d).shape,(16,32,128))

    def test_calibration_horizons_and_collision_owners_are_independent(self):
        model=SearchVNext(24,config())
        def record(h,pvalue,option,phase,offset=0):
            prob=torch.full((1,2,7),pvalue)
            target=torch.zeros_like(prob);known=torch.ones_like(prob,dtype=torch.bool)
            train=torch.zeros(1,2,dtype=torch.bool);cal=~train
            identity=torch.tensor([[10+offset,20+offset]])
            return (h,prob,target,known,train,cal,identity,
                torch.full((1,2),option),torch.full((1,2),phase))
        model._record([record(1,.2,1,0),record(2,.7,2,1),record(8,.9,3,2)],torch.zeros(2))
        self.assertEqual(model.a.cal_identity.shape[0],9)
        torch.testing.assert_close(model.a.cal_counts[1,1,0],torch.full((7,),2.))
        torch.testing.assert_close(model.a.cal_counts[2,2,1],torch.full((7,),2.))
        torch.testing.assert_close(model.a.cal_counts[8,3,2],torch.full((7,),2.))
        old=model.a.cal_errors[1].clone()
        model._record([record(2,.1,2,1)],torch.zeros(2))
        torch.testing.assert_close(model.a.cal_errors[1],old)
        torch.testing.assert_close(model.a.cal_counts[2,2,1],torch.full((7,),2.))
        # 65536 offset maps to the same slots but belongs to different real frames.
        model._record([record(2,.4,4,3,65536)],torch.zeros(2))
        self.assertFalse(model.a.cal_counts[2,2,1].any())
        torch.testing.assert_close(model.a.cal_counts[2,4,3],torch.full((7,),2.))
        self.assertTrue(bool((model.a.cal_counts>=0).all()))

    def test_option_support_requires_entire_executed_plan(self):
        from search_actuator import apply_command_torch
        d={"reward":np.zeros((1,7)), "log_vnext_decision":np.zeros((1,7)),
           "log_vnext_plan_commands":np.zeros((1,7,3)),
           "log_vnext_version":np.ones((1,7)),
           "log_vnext_horizon":np.zeros((1,7)),
           "log_vnext_option_id":np.zeros((1,7)),
           "log_vnext_choice":np.zeros((1,7)),
           "log_vnext_phase":np.zeros((1,7)),
           "log_requested_action":np.zeros((1,7))}
        d["log_vnext_decision"][0,1]=1
        d["log_vnext_horizon"][0,1]=3
        d["log_vnext_option_id"][0,1:4]=7
        d["log_vnext_choice"][0,1:4]=4
        d["log_vnext_plan_commands"][0,1]=[4,6,6]
        d["log_requested_action"][0,1:4]=[4,6,6]
        actuator=torch.zeros(1,7,3);controls=torch.zeros(1,7,9)
        for i in range(1,7):
            applied,nxt=apply_command_torch(torch.tensor([int(d["log_requested_action"][0,i])]),actuator[:,i-1])
            controls[:,i]=applied;actuator[:,i]=nxt
        verified=SearchVNext.matched_option_windows(d,actuator,controls,3)
        self.assertTrue(verified[0,0]);self.assertEqual(int(verified.sum()),1)
        d["log_vnext_option_id"][0,2]=8
        self.assertFalse(SearchVNext.matched_option_windows(d,actuator,controls,3).any())
        d["log_vnext_option_id"][0,2]=7
        d["log_requested_action"][0,3]=1
        self.assertFalse(SearchVNext.matched_option_windows(d,actuator,controls,3).any())

    def test_online_and_replay_memory_use_identical_age_metadata(self):
        memory=SlotMemory(24,config());features=torch.randn(1,24)
        old=torch.randint(0,256,(64,64,4),dtype=torch.uint8)
        state=memory.initial(1,"cpu")
        state["frames"][0,0]=old.float()/255.
        state["refs"][0,0]=0;state["tick"][:]=100
        state["meta"][0,0]=torch.tensor([0.,1.,2.,.8,1.,.2,3.,1.])
        data=dict(vnext_memory_frames=torch.stack((old,old*0,old*0))[None],
            vnext_memory_indices=torch.tensor([[[0,-1,-1]]]),
            vnext_memory_valid=torch.tensor([[[1,0,0]]]),
            vnext_memory_meta=state["meta"][:,None].clone(),
            vnext_observation_index=torch.tensor([[100]]))
        image=torch.zeros(1,64,64,3,dtype=torch.uint8);heat=torch.zeros(1,64,64,dtype=torch.uint8)
        online,_,_=memory.online(features,image,heat,state,torch.zeros(1),torch.zeros(1,9))
        replay,_=memory.batch(features[:,None],data)
        torch.testing.assert_close(online,replay[:,0],rtol=1e-5,atol=1e-6)
        # Prefill may shift all frame clocks; age-based reading stays invariant.
        shifted=data["vnext_memory_meta"].clone();shifted[...,:2]+=500
        data["vnext_memory_meta"]=shifted;data["vnext_observation_index"]+=500
        after,_=memory.batch(features[:,None],data)
        torch.testing.assert_close(replay,after)

    def test_history_attempt_supervision_is_not_current_new_label(self):
        memory=SlotMemory(24,config());d=batch()
        d["vnext_memory_meta"][...,6]=3
        features=torch.randn(2,12,24)
        context,details=memory.batch(features,d)
        mask=torch.ones(2,12,dtype=torch.bool)
        with torch.no_grad():
            memory.history_head.weight.zero_();memory.history_head.bias.zero_()
            memory.history_head.bias[2]=-4
        _,first,_=memory.loss(features,d,context,details,mask)
        with torch.no_grad():memory.history_head.bias[2]=4
        _,second,_=memory.loss(features,d,context,details,mask)
        self.assertLess(float(second["B/history_loss"]),float(first["B/history_loss"]))

    def test_remaining_budget_uses_logged_control_field(self):
        model=SearchVNext(24,config());d=batch()
        d["log_vnext_remaining"]=np.full((2,12),42,np.float32)
        values=[]
        hook=model.a.register_forward_pre_hook(lambda module,args:values.append(args[-1].detach().clone()))
        model.training_loss(torch.randn(2,12,24),d);hook.remove()
        self.assertTrue(values)
        self.assertTrue(all(bool((value==42).all()) for value in values))

    def test_checkpoint_strict_and_config(self):
        m=SearchVNext(24,config())
        checkpoint=dict(agent_state_dict={"_vnext."+k:v for k,v in m.state_dict().items()},
            search_vnext_version=1,vnext_memory_version=1,search_a_label_version=3,actuator_version=1,
            config={"vnext_slots":3})
        check_vnext_checkpoint(checkpoint,True)
        with self.assertRaises(ValueError):
            check_vnext_checkpoint(checkpoint,False)
        checkpoint["search_a_label_version"]=2
        with self.assertRaises(ValueError):
            check_vnext_checkpoint(checkpoint,True)
        c={}
        restore_vnext_config(c,{"agent_state_dict":{}})
        self.assertFalse(c["search_vnext_enabled"])


if __name__=="__main__":
    unittest.main()

