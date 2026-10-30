import pathlib
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from search_vnext_control import (VNextController,FIELDS,N_OPTIONS,ACTOR,SEARCH,ALIGN,
    INTERACT,RECOVERY,CONTEXT_DIM,PREDICTION_DIM,build_transitions,complete_fields)


class FakeAB:
    def __init__(self,supported=True):
        self.calls=0;self.supported=supported
    def predict_candidates(self,feat,memory,controls,actuator,**kw):
        self.calls+=1
        return dict(descriptor=feat.new_zeros(len(feat),N_OPTIONS,PREDICTION_DIM),
                    support=torch.full((len(feat),N_OPTIONS),self.supported,device=feat.device))


def config(**kw):
    result=dict(seed=0,episode_max_steps=1000,episode_max_steps_override=0,
        target_item="sand",vnext_memory_dim=4,vnext_c_hidden=16,
        vnext_start_step=0,vnext_base_episode_fraction=0.,
        vnext_probe_probability=1.,vnext_stall_steps=2000)
    result.update(kw);return SimpleNamespace(**result)


def current(batch=1,visible=0.,joint=0.,ready=True):
    return dict(visible=torch.full((batch,),visible),center_interactable=torch.full((batch,),joint),
        ready=dict(visible=torch.full((batch,),ready),center_interactable=torch.full((batch,),ready)))


def replay(batch=1,time=10):
    d=dict(reward=np.zeros((batch,time),np.float32),is_first=np.zeros((batch,time),bool),
        is_last=np.zeros((batch,time),bool),is_terminal=np.zeros((batch,time),bool),
        success=np.zeros((batch,time),np.float32))
    for name,dims in FIELDS.items():d[name]=np.zeros((batch,time,*dims),np.float32)
    d["is_first"][:,0]=True
    d["log_vnext_version"][:]=1
    d["log_vnext_mask"][:,:,ACTOR]=1
    d["log_vnext_remaining_after"][:]=500
    d["log_vnext_context"][:,:,0]=.5
    d["log_vnext_context"][:,:,1]=1
    for r in range(1,time,3):
        d["log_vnext_decision"][:,r]=1
        d["log_vnext_option_end"][:,min(r+2,time-1)]=1
    return d


class VNextControlTests(unittest.TestCase):
    def run_step(self,c,ab,state=None,proposal=10,cur=None,first=False,training=True,obs=None,step=100000,batch=1):
        feat=torch.ones(batch,8);mem=torch.zeros(batch,4)
        return c.choose(ab,feat,mem,cur or current(batch),obs or {},
            F.one_hot(torch.full((batch,),proposal),12).float(),state,
            torch.full((batch,),first),step,training)

    def test_constructor_preserves_random_stream(self):
        torch.manual_seed(73);expected=torch.rand(4)
        torch.manual_seed(73);VNextController(8,config());actual=torch.rand(4)
        self.assertTrue(torch.equal(expected,actual))

    def test_only_actor_skips_all_prediction(self):
        c=VNextController(8,config(vnext_control=False));ab=FakeAB()
        action,state,info=self.run_step(c,ab,proposal=10,first=True)
        self.assertEqual(ab.calls,0);self.assertEqual(action.argmax(-1).item(),10)
        self.assertEqual(state["actuator"][0,1].item(),9)
        self.assertEqual(state["previous_control"][0,5].item(),1)

    def test_exact_logging_fields(self):
        c=VNextController(8,config());ab=FakeAB()
        _,_,info=self.run_step(c,ab,first=True)
        self.assertEqual(set(info),set(FIELDS))
        for key,dims in FIELDS.items():self.assertEqual(tuple(info[key].shape),(1,*dims))

    def test_attack_protected_beyond_legacy_two(self):
        c=VNextController(8,config());ab=FakeAB();state=None
        for i in range(12):
            action,state,info=self.run_step(c,ab,state,proposal=1,first=i==0)
            self.assertEqual(action.argmax(-1).item(),1)
            self.assertEqual(info["log_vnext_override"].item(),0)

    def test_visibility_is_not_interaction_permission(self):
        c=VNextController(8,config());ab=FakeAB();state=None
        for i in range(6):
            _,state,info=self.run_step(c,ab,state,proposal=0,cur=current(visible=.99,joint=0),first=i==0)
            self.assertEqual(info["log_vnext_mask"][0,8].item(),0)
        self.assertEqual(state["phase"].item(),ALIGN)

    def test_trusted_joint_can_test_attack(self):
        c=VNextController(8,config(vnext_probe_probability=0.));ab=FakeAB();state=None
        for i in range(2):
            action,state,info=self.run_step(c,ab,state,proposal=10,cur=current(visible=.99,joint=.99),first=i==0,training=False)
        self.assertEqual(action.argmax(-1).item(),0) # settle while prior jump persists
        self.assertEqual(info["log_vnext_choice"].item(),9)
        self.assertEqual(state["phase"].item(),INTERACT)
        # A stable actuator allows actual attack independently of a failed Actor.
        state["actuator"][:,1]=0;state["active"][:]=-1
        action,state,info=self.run_step(c,ab,state,proposal=0,cur=current(visible=.99,joint=.99),training=False)
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(info["log_vnext_choice"].item(),8)

    def test_unknown_joint_never_forces_attack(self):
        c=VNextController(8,config(vnext_probe_probability=0.));ab=FakeAB();state=None
        for i in range(6):
            action,state,info=self.run_step(c,ab,state,proposal=0,cur=current(visible=.99,joint=.99,ready=False),first=i==0,training=False)
            self.assertNotEqual(info["log_vnext_choice"].item(),8)

    def test_no_jump_retrigger_during_settle(self):
        c=VNextController(8,config(vnext_probe_probability=0.));ab=FakeAB()
        state=c.initial(1,"cpu");state["age"][:]=100;state["actuator"][:,1]=9
        state["phase"][:]=INTERACT;state["actionable_run"][:]=3;state["visible_run"][:]=3
        for i in range(3):
            _,state,info=self.run_step(c,ab,state,proposal=10,cur=current(visible=.99,joint=.99),training=False)
            self.assertEqual(state["actuator"][0,1].item(),8-i)
        self.assertEqual(info["log_vnext_option_duration"].item(),3)

    def test_episode_base_only_persists_and_eval_not_sampled(self):
        c=VNextController(8,config(vnext_base_episode_fraction=1.));ab=FakeAB();state=None
        for i in range(10):
            _,state,info=self.run_step(c,ab,state,first=i==0)
            self.assertEqual(info["log_vnext_base_episode"].item(),1)
            self.assertEqual(info["log_vnext_override"].item(),0)
        _,state,info=self.run_step(c,ab,first=True,training=False)
        self.assertEqual(info["log_vnext_base_episode"].item(),0)

    def test_rolling_search_and_total_budget(self):
        c=VNextController(8,config());ab=FakeAB();state=None;records=[]
        for i in range(340):
            _,state,info=self.run_step(c,ab,state,proposal=0,first=i==0)
            records.append(int(info["log_vnext_override"].item()))
            self.assertLessEqual(info["log_vnext_search_used"].item(),10)
            self.assertLessEqual(info["log_vnext_extra_used"].item(),25)
        self.assertGreater(sum(records),0)
        self.assertTrue(all(sum(records[max(0,i-99):i+1])<=10 for i in range(len(records))))

    def test_partial_batch_reset_untouched_rows(self):
        c=VNextController(8,config(vnext_control=False));ab=FakeAB()
        _,state,_=self.run_step(c,ab,batch=2,first=True)
        state["age"][:]=50;state["actuator"][:,1]=4
        action,state,info=c.choose(ab,torch.ones(2,8),torch.zeros(2,4),current(2),{},
            F.one_hot(torch.tensor([0,0]),12).float(),state,torch.tensor([True,False]),100000,True)
        self.assertEqual(state["age"].tolist(),[1,51])
        self.assertEqual(state["actuator"][:,1].tolist(),[0,3])

    def test_rng_restore_replays_probe_choices(self):
        c=VNextController(8,config());ab=FakeAB()
        saved=c.rng_state();weights=copy_state(c.state_dict())
        _,_,x=self.run_step(c,ab,first=True)
        c.load_rng_state(saved);c.load_state_dict(weights)
        _,_,y=self.run_step(c,ab,first=True)
        self.assertTrue(torch.equal(x["log_vnext_choice"],y["log_vnext_choice"]))

    def test_variable_length_actuator_end(self):
        c=VNextController(8,config())
        state=torch.tensor([[0.,9.,0.]])
        commands,lengths=c.templates(state,torch.tensor([100]))
        lengths[:,6]=1
        controls,end=c.controls(commands,state,lengths)
        self.assertEqual(end[0,6,1].item(),8)
        self.assertEqual(controls[0,6,1:].abs().sum().item(),0)

    def test_window_tail_is_censored(self):
        d=replay(time=7);d["log_vnext_option_end"][:,6]=0
        records=build_transitions(d)
        self.assertEqual(len(records),1);self.assertFalse(records[0]["terminal"])

    def test_timeout_terminates_success_probability(self):
        d=replay(time=7);d["log_vnext_remaining_after"][:,6]=0
        records=build_transitions(d)
        self.assertTrue(records[-1]["terminal"]);self.assertEqual(records[-1]["success"],0)

    def test_success_only_real_row_not_positive_progress(self):
        d=replay();d["search_progress"]=np.ones((1,10));d["search_progress_valid"]=np.ones((1,10))
        self.assertTrue(all(r["success"]==0 for r in build_transitions(d)))
        d["success"][0,3]=1;d["is_last"][0,3]=True
        records=build_transitions(d);self.assertEqual(records[0]["success"],1)

    def test_reset_boundary_not_a_transition(self):
        d=replay();d["is_first"][0,3]=True
        self.assertTrue(all(r["start"]!=0 for r in build_transitions(d)))

    def test_twin_q_update_is_real_and_isolated(self):
        c=VNextController(8,config());ab=FakeAB();d=replay(batch=2)
        d["is_last"][:,-1]=True
        feat=torch.randn(2,10,8,requires_grad=True)
        memory=torch.randn(2,10,4,requires_grad=True)
        before=copy_state(c.state_dict());metrics=c.train_batch(feat,memory,ab,d)
        self.assertEqual(metrics["C/update_applied"],1)
        self.assertIsNone(feat.grad);self.assertIsNone(memory.grad)
        self.assertGreater(metrics["C/value_grad_norm"],0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in c.state_dict().items() if k.startswith(("q1.","q2."))))
        values=c.values(torch.zeros(2,8),None,torch.zeros(2,12,12),torch.zeros(2,24))
        self.assertTrue(bool(((values>=0)&(values<=1)).all()))

    def test_factual_success_target_one(self):
        c=VNextController(8,config());ab=FakeAB();d=replay(time=4)
        d["success"][0,3]=1;d["is_last"][0,3]=True
        metrics=c.train_batch(torch.zeros(1,4,8),None,ab,d)
        self.assertEqual(metrics["C/target_success_probability"],1.)

    def test_stuck_sky_recovers_actual_pitch_without_a_support(self):
        c=VNextController(8,config(vnext_probe_probability=0.,vnext_stall_steps=20))
        ab=FakeAB(supported=False);state=c.initial(1,"cpu")
        state["age"][:]=100;state["actuator"][:,2]=-70
        state["stalled"][:]=21;state["phase"][:]=RECOVERY;state["has_visual"][:]=True
        # Only the current K3 lease survives the image change caused by step 1.
        for i in range(3):
            image=torch.full((1,64,64,3),i/2.)
            action,state,info=self.run_step(c,ab,state,proposal=2,training=False,obs={"image":image})
            self.assertEqual(action.argmax(-1).item(),3)
            self.assertEqual(info["log_vnext_choice"].item(),11)
            self.assertEqual(info["log_vnext_recovery_exception"].item(),1)
            self.assertEqual(state["actuator"][0,2].item(),-60+10*i)
        self.assertEqual(info["log_vnext_option_duration"].item(),3)
        self.assertEqual(ab.calls,0)
        self.assertEqual(info["log_vnext_extra_used"].item(),3)

    def test_limit_run_detects_continuous_camera_noops(self):
        c=VNextController(8,config(vnext_probe_probability=0.))
        ab=FakeAB(supported=False);state=c.initial(1,"cpu")
        state["age"][:]=100;state["actuator"][:,2]=70
        for i in range(3):
            action,state,info=self.run_step(c,ab,state,proposal=3,training=False)
        self.assertEqual(action.argmax(-1).item(),2)
        self.assertEqual(state["actuator"][0,2].item(),60)
        self.assertEqual(info["log_vnext_recovery_exception"].item(),1)

    def test_recovery_still_respects_budget_and_off_switch(self):
        for enabled in (True,False):
            c=VNextController(8,config(vnext_probe_probability=0.,vnext_control=enabled,vnext_stall_steps=20))
            ab=FakeAB(supported=False);state=c.initial(1,"cpu")
            state["age"][:]=100;state["actuator"][:,2]=-70
            state["stalled"][:]=30;state["phase"][:]=RECOVERY
            state["history"][0,:25]=2
            action,state,info=self.run_step(c,ab,state,proposal=2,training=False)
            self.assertEqual(action.argmax(-1).item(),2)
            self.assertEqual(info["log_vnext_override"].item(),0)
            self.assertEqual(info["log_vnext_recovery_exception"].item(),0)

    def test_new_credible_target_interrupts_camera_recovery(self):
        c=VNextController(8,config(vnext_probe_probability=0.,vnext_stall_steps=20));ab=FakeAB(supported=False)
        state=c.initial(1,"cpu");state["age"][:]=100
        state["actuator"][:,2]=-70;state["stalled"][:]=30
        _,state,info=self.run_step(c,ab,state,proposal=2,training=False)
        self.assertEqual(info["log_vnext_choice"].item(),11)
        action,state,info=self.run_step(c,ab,state,proposal=1,training=False,
            cur=current(visible=.99,joint=.99))
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(info["log_vnext_override"].item(),0)
        self.assertEqual(info["log_vnext_recovery_exception"].item(),0)

    def test_environment_error_is_censored_not_failure(self):
        d=replay(time=4);d["is_last"][0,3]=True
        d["log_env_error"]=np.zeros((1,4));d["log_env_error"][0,3]=1
        self.assertEqual(build_transitions(d),[])

    def test_all_censored_no_update(self):
        c=VNextController(8,config());d=replay(time=3)
        d["log_vnext_option_end"][:]=0
        self.assertEqual(c.train_batch(torch.zeros(1,3,8),None,FakeAB(),d),{"C/transitions":0.})

def copy_state(state):
    return {k:v.detach().clone() for k,v in state.items()}

if __name__=="__main__":unittest.main()

