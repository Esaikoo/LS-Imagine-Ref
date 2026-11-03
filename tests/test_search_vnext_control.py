import pathlib
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from apm.control import (VNextController,FIELDS,N_OPTIONS,ACTOR,SEARCH,ALIGN,
    INTERACT,RECOVERY,CONTEXT_DIM,PREDICTION_DIM,build_transitions,complete_fields,
    factual_search_rewards,option_names)


class FakeAB:
    def __init__(self,supported=True):
        self.calls=0;self.supported=supported
    def predict_candidates(self,feat,memory,controls,actuator,**kw):
        self.calls+=1
        return dict(descriptor=feat.new_zeros(len(feat),N_OPTIONS,PREDICTION_DIM),
                    support=torch.full((len(feat),N_OPTIONS),self.supported,device=feat.device))


class SpatialAB(FakeAB):
    def __init__(self,preferred=0,score_supported=True):
        super().__init__();self.preferred=preferred;self.score_supported=score_supported
    def predict_candidates(self,*args,**kw):
        result=super().predict_candidates(*args,**kw)
        ref=result["descriptor"]
        scores=ref.new_zeros(ref.shape[:2]);scores[:,self.preferred]=1
        result.update(score=scores,score_supported=torch.full_like(scores,self.score_supported,dtype=torch.bool),
                      supported=torch.ones(*ref.shape[:2],7,device=ref.device,dtype=torch.bool))
        return result


class UnsupportedSpatialAB(SpatialAB):
    """Even an inconsistent optimistic score must not certify its own support."""
    def predict_candidates(self,*args,**kw):
        result=super().predict_candidates(*args,**kw)
        result["support"][:]=False
        result["supported"][:]=False
        result["score"]*=100
        return result


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


class SpatialControlTests(unittest.TestCase):
    run_step = VNextControlTests.run_step
    def setUp(self):
        self.cfg=config(vnext_spatial_enabled=True,vnext_probe_probability=0.)

    @staticmethod
    def summary(motion=0.,ready=True):
        value=torch.zeros(1,12)
        value[:,0]=float(ready);value[:,3]=1;value[:,10]=1
        value[:,7]=motion/.5;value[:,11]=float(ready)
        return value

    def test_spatial_context_and_option_names_are_explicit(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu")
        state["spatial_summary"]=torch.arange(12).float()[None]/12
        ctx=c.context(state,current(),torch.tensor([500]),torch.zeros(1),torch.zeros(1),torch.zeros(1))
        expected=state["spatial_summary"][:,[0,1,2,3,6,7,8,10,11]]
        torch.testing.assert_close(ctx[:,15:],expected)
        self.assertEqual(option_names(True)[6:8],("scan","clear_obstacle"))
        self.assertEqual(option_names(False)[6:8],("align_up","align_down"))

    def test_turning_cannot_reset_stall_or_failed_interaction(self):
        c=VNextController(8,config(vnext_spatial_enabled=True,vnext_control=False));ab=SpatialAB()
        state=c.initial(1,"cpu");state["has_visual"][:]=True;state["interact_failures"][:]=1
        for i in range(8):
            state["previous_control"][:,8]=10
            _,state,info=self.run_step(c,ab,state,proposal=4,training=False,
                obs={"image":torch.full((1,64,64,3),float(i%2)),"vnext_spatial_summary":self.summary()})
        self.assertEqual(state["stalled"].item(),8)
        self.assertEqual(state["interact_failures"].item(),1)
        self.assertEqual(info["log_vnext_spatial_progress"].item(),0)

    def test_verified_translation_resets_failure_history(self):
        c=VNextController(8,config(vnext_spatial_enabled=True,vnext_control=False));state=c.initial(1,"cpu")
        state["stalled"][:]=50;state["interact_failures"][:]=2;state["clear_attempts"][:]=2
        _,state,_=self.run_step(c,SpatialAB(),state,proposal=6,training=False,
            obs={"vnext_spatial_summary":self.summary(motion=.1)})
        self.assertEqual(state["stalled"].item(),0)
        self.assertEqual(state["interact_failures"].item(),0)
        self.assertEqual(state["clear_attempts"].item(),0)

    def test_supported_a_score_really_changes_selected_action(self):
        for supported,expected in ((True,1),(False,ACTOR)):
            c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
            _,_,info=self.run_step(c,SpatialAB(1,supported),state,proposal=0,training=False)
            self.assertEqual(info["log_vnext_choice"].item(),expected)
            self.assertEqual(info["log_vnext_version"].item(),2)
            self.assertEqual(info["log_vnext_a_score"][0,1].item(),float(supported))

    def test_empirical_map_q_can_act_without_a_forecast_support(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        c.updates.fill_(128);c.coverage[1:3]=16
        # A optimistically advertises left, while trained C prefers right.
        with torch.no_grad():c.q1[-1].bias[2]=.2;c.q2[-1].bias[2]=.2
        coverage=c.coverage.clone()
        action,_,info=self.run_step(c,UnsupportedSpatialAB(1),state,proposal=0,training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(info["log_vnext_choice"].item(),2)
        self.assertEqual(action.argmax(-1).item(),5)
        self.assertEqual(info["log_vnext_source"].item(),1)
        self.assertEqual(info["log_vnext_empirical_support"][0,2].item(),1)
        self.assertEqual(info["log_vnext_a_score"].abs().sum().item(),0)
        self.assertEqual(info["log_vnext_score_supported"].sum().item(),0)
        torch.testing.assert_close(c.coverage,coverage)  # eval never invents trials

    def test_empirical_support_requires_updates_trials_and_localization(self):
        cases=((127,16,True,1.),(128,15,True,1.),(128,16,False,0.),(128,16,True,.2))
        for updates,trials,ready,confidence in cases:
            c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
            c.updates.fill_(updates);c.coverage[1]=trials
            with torch.no_grad():c.q1[-1].bias[1]=8;c.q2[-1].bias[1]=8
            summary=self.summary(ready=ready);summary[:,11]=confidence
            _,_,info=self.run_step(c,UnsupportedSpatialAB(1),state,proposal=0,training=False,
                obs={"vnext_spatial_summary":summary})
            self.assertEqual(info["log_vnext_choice"].item(),ACTOR,(updates,trials,ready,confidence))
            self.assertEqual(info["log_vnext_empirical_support"][0,1].item(),0)

    def test_empirical_support_does_not_bypass_clearance_legality(self):
        for recovery,jumping in ((False,False),(True,True)):
            c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
            c.updates.fill_(128);c.coverage[7]=16
            with torch.no_grad():c.q1[-1].bias[7]=8;c.q2[-1].bias[7]=8
            if recovery:
                state["phase"][:]=RECOVERY;state["stalled"][:]=2001;state["forward_stall"][:]=4
            if jumping:state["actuator"][:,1]=4
            _,_,info=self.run_step(c,UnsupportedSpatialAB(7),state,proposal=0,training=False,
                obs={"vnext_spatial_summary":self.summary()})
            self.assertEqual(info["log_vnext_empirical_support"][0,7].item(),1)
            self.assertEqual(info["log_vnext_mask"][0,7].item(),0)
            self.assertNotEqual(info["log_vnext_choice"].item(),7)

    def test_coverage_counts_online_option_starts_not_replay_updates(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        for _ in range(3):
            _,state,_=self.run_step(c,SpatialAB(1),state,proposal=0,training=True,
                obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(c.coverage[1].item(),1)  # one actual K3 decision
        before=c.coverage.clone();d=replay(time=4)
        d["log_vnext_version"][:]=2;d["is_last"][0,-1]=True
        for _ in range(2):
            c.train_batch(torch.zeros(1,4,8),None,UnsupportedSpatialAB(1),d)
        torch.testing.assert_close(c.coverage,before)

    def test_empirical_thresholds_can_be_configured(self):
        c=VNextController(8,config(vnext_spatial_enabled=True,vnext_c_min_option_trials=3,vnext_c_min_updates=5))
        state=c.initial(1,"cpu");state["spatial_summary"]=self.summary()
        c.updates.fill_(5);c.coverage[3]=3
        mask=c.empirical_support(state)
        self.assertTrue(bool(mask[0,3]));self.assertFalse(bool(mask[0,ACTOR]))

    def test_recovery_settle_expires_real_jump_then_allows_clearance(self):
        c=VNextController(8,self.cfg);ab=UnsupportedSpatialAB()
        state=c.initial(1,"cpu");state["age"][:]=100;state["phase"][:]=RECOVERY
        state["stalled"][:]=2001;state["forward_stall"][:]=3;state["actuator"][:,1]=9
        c.updates.fill_(128);c.coverage[7]=16
        with torch.no_grad():c.q1[-1].bias[7]=8;c.q2[-1].bias[7]=8
        for index in range(9):
            # Rendering changes during a jump must not give the Actor a chance
            # to re-trigger it before this bounded physical wait has finished.
            action,state,info=self.run_step(c,ab,state,proposal=10,training=False,
                obs={"image":torch.full((1,64,64,3),float(index%2)),
                     "vnext_spatial_summary":self.summary()})
            self.assertEqual(action.argmax(-1).item(),0)
            self.assertEqual(info["log_vnext_choice"].item(),9)
            self.assertEqual(info["log_vnext_source"].item(),4)
            self.assertEqual(state["actuator"][0,1].item(),8-index)
            self.assertEqual(state["phase"].item(),RECOVERY)
        self.assertEqual(info["log_vnext_settle_used"].item(),9)
        action,state,info=self.run_step(c,ab,state,proposal=10,training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(info["log_vnext_choice"].item(),7)
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(state["actuator"][0,1].item(),0)

    def test_recovery_settle_respects_settle_total_budget_and_off_switch(self):
        for reason in ("settle_cap","total_cap","off"):
            c=VNextController(8,config(vnext_spatial_enabled=True,vnext_probe_probability=0.,
                vnext_control=reason!="off"))
            state=c.initial(1,"cpu");state["age"][:]=100;state["phase"][:]=RECOVERY
            state["stalled"][:]=2001;state["actuator"][:,1]=9
            if reason=="settle_cap":state["history"][0,:13]=3
            if reason=="total_cap":state["history"][0,:26]=2
            _,_,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=10,training=False,
                obs={"vnext_spatial_summary":self.summary()})
            self.assertEqual(info["log_vnext_mask"][0,9].item(),0,reason)
            self.assertNotEqual(info["log_vnext_choice"].item(),9,reason)

    def test_real_target_or_progress_can_interrupt_recovery_settle(self):
        for cause in ("target","progress"):
            c=VNextController(8,self.cfg);state=c.initial(1,"cpu")
            state["age"][:]=100;state["phase"][:]=RECOVERY;state["stalled"][:]=2001
            state["actuator"][:,1]=9
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=10,training=False,
                obs={"vnext_spatial_summary":self.summary()})
            self.assertEqual(info["log_vnext_choice"].item(),9)
            kwargs={"obs":{"vnext_spatial_summary":self.summary()}}
            if cause=="target":kwargs["cur"]=current(visible=.99,joint=.99)
            else:kwargs["obs"]["obs_reward"]=torch.tensor([1.])
            proposal=1 if cause=="target" else 6
            action,_,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=proposal,training=False,**kwargs)
            self.assertEqual(action.argmax(-1).item(),proposal,cause)
            self.assertEqual(info["log_vnext_override"].item(),0,cause)

    def test_clearance_is_bounded_learned_and_requires_blocked_motion(self):
        c=VNextController(8,self.cfg);ab=SpatialAB()
        state=c.initial(1,"cpu");state["age"][:]=100;state["stalled"][:]=2001
        state["forward_stall"][:]=3
        with torch.no_grad():
            c.q1[-1].bias[7]=8;c.q2[-1].bias[7]=8
        for expected in (1,1,6):
            action,state,info=self.run_step(c,ab,state,proposal=0,training=False,
                obs={"vnext_spatial_summary":self.summary()})
            self.assertEqual(action.argmax(-1).item(),expected)
            self.assertEqual(info["log_vnext_choice"].item(),7)
            self.assertEqual(info["log_vnext_source"].item(),1)
        self.assertEqual(state["clear_attempts"].item(),1)
        _,state,info=self.run_step(c,ab,state,proposal=0,training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(info["log_vnext_mask"][0,7].item(),0)
        # Mere absence of odometry is not evidence of a blocked path.
        state=c.initial(1,"cpu");state["age"][:]=100;state["stalled"][:]=2001
        _,_,info=self.run_step(c,ab,state,proposal=0,training=False,
            obs={"vnext_spatial_summary":self.summary(ready=False)})
        self.assertEqual(info["log_vnext_mask"][0,7].item(),0)

    def test_search_reward_uses_valid_factual_labels_not_predictions(self):
        d=replay(time=4);d["search_new_area"]=np.array([[0,1,0,0]],np.float32)
        d["search_new_area_valid"]=np.array([[0,1,1,1]],np.float32)
        d["search_first_discovery"]=np.array([[0,0,1,1]],np.float32)
        d["search_first_discovery_valid"]=np.array([[0,1,1,0]],np.float32)
        r=factual_search_rewards(d,self.cfg)
        self.assertAlmostEqual(r[0,1],.05-.002,places=6)
        self.assertAlmostEqual(r[0,2],.5-.002,places=6)
        self.assertAlmostEqual(r[0,3],-.002,places=6)
        # Unknown labels and optimistic model forecasts cannot create rewards.
        d["log_vnext_prediction"][:]=1
        np.testing.assert_array_equal(r,factual_search_rewards(d,self.cfg))

    def test_walking_inside_cell_is_not_penalized_as_area_revisit(self):
        d=replay(time=10)
        d["log_applied_control"]=np.zeros((1,10,9),np.float32)
        d["log_applied_control"][:,:,1]=1
        for name in ("area_revisit","view_revisit","new_area","first_discovery"):
            d["search_"+name]=np.zeros((1,10),np.float32)
            d["search_"+name+"_valid"]=np.ones((1,10),np.float32)
        d["search_view_revisit"][:]=1  # same direction during ordinary walking
        d["search_new_area"][0,1]=1
        d["search_first_discovery"][0,7]=1
        expected=np.full((1,10),-.002,np.float32)
        expected[0,1]+=.05;expected[0,7]+=.5
        reward=factual_search_rewards(d,self.cfg)
        np.testing.assert_allclose(reward,expected,atol=1e-7)
        self.assertEqual(int((reward>.1).sum()),1)  # first target only once

    def test_area_and_view_revisit_apply_to_distinct_actions(self):
        d=replay(time=7);controls=np.zeros((1,7,9),np.float32)
        controls[0,0,1]=1                 # translating back into an old area
        controls[0,1,8]=10                # stationary repeated camera direction
        controls[0,2,[1,8]]=[1,10]        # moving and turning inside the same area
        controls[0,3,[5,8]]=[1,10]        # jumping is not a stationary sweep
        controls[0,4,[0,8]]=[1,10]        # do not penalize interaction dwell
        controls[0,6,1]=1                 # unknown area label
        d["log_applied_control"]=controls
        d["search_area_revisit"]=np.array([[1,0,0,0,0,0,1]],np.float32)
        d["search_area_revisit_valid"]=np.array([[1,1,1,1,1,1,0]],np.float32)
        d["search_view_revisit"]=np.ones((1,7),np.float32)
        d["search_view_revisit_valid"]=np.ones((1,7),np.float32)
        expected=np.array([[-.027,-.027,-.002,-.002,-.002,-.002,-.002]],np.float32)
        np.testing.assert_allclose(factual_search_rewards(d,self.cfg),expected,atol=1e-7)

    def test_first_sweep_has_information_reward_and_second_sweep_costs(self):
        from apm.evidence import SpatialEvidenceWrapper, SPATIAL_LABEL_SHAPES
        from test_spatial_evidence import FakeWorld, observation
        world=FakeWorld();wrapped=SpatialEvidenceWrapper(world,supervision=True)
        rows=[wrapped.reset()]
        for step in range(1,73):
            world.observation=observation(yaw=(step*10+180)%360-180)
            rows.append(wrapped.step({"camera":[0,10]})[0])
        d={"reward":np.zeros((1,len(rows)),np.float32),
           "log_applied_control":np.zeros((1,len(rows),9),np.float32)}
        d["log_applied_control"][0,1:,8]=10
        for key in SPATIAL_LABEL_SHAPES:
            d[key]=np.asarray([row[key] for row in rows])[None]
        reward=factual_search_rewards(d,self.cfg)[0]
        self.assertEqual(d["search_new_view"][0,1:37].sum(),11)
        self.assertEqual(d["search_view_revisit"][0,1:37].sum(),1)
        self.assertAlmostEqual(float(reward[1:37].sum()),11*.01-.025-36*.002,places=6)
        self.assertGreater(float(reward[1:37].sum()),0.)
        self.assertEqual(d["search_new_view"][0,37:].sum(),0)
        self.assertEqual(d["search_view_revisit"][0,37:].sum(),12)
        self.assertLess(float(reward[37:].sum()),0.)

    def test_real_same_cell_forward_receives_only_step_cost(self):
        from apm.evidence import SpatialEvidenceWrapper, SPATIAL_LABEL_SHAPES
        from test_spatial_evidence import FakeWorld, observation
        world=FakeWorld();wrapped=SpatialEvidenceWrapper(world,supervision=True)
        rows=[wrapped.reset()]
        for step in range(1,7):
            world.observation=observation(z=step*.05)
            rows.append(wrapped.step({"forward":1})[0])
        d={"reward":np.zeros((1,len(rows)),np.float32),
           "log_applied_control":np.zeros((1,len(rows),9),np.float32)}
        d["log_applied_control"][0,1:,1]=1
        for key in SPATIAL_LABEL_SHAPES:
            d[key]=np.asarray([row[key] for row in rows])[None]
        np.testing.assert_allclose(factual_search_rewards(d,self.cfg)[0,1:],-.002,atol=1e-7)

    def test_spatial_signed_smdp_return_and_gradient_isolation(self):
        c=VNextController(8,self.cfg);d=replay(time=4)
        d["log_vnext_version"][:]=2;d["is_last"][0,3]=True
        d["search_new_area"]=np.array([[0,1,0,0]],np.float32)
        d["search_new_area_valid"]=np.ones((1,4),np.float32)
        d["search_first_discovery"]=np.array([[0,0,0,1]],np.float32)
        d["search_first_discovery_valid"]=np.ones((1,4),np.float32)
        feature=torch.randn(1,4,8,requires_grad=True);memory=torch.randn(1,4,4,requires_grad=True)
        before=copy_state(c.search_q1.state_dict())
        m=c.train_batch(feature,memory,SpatialAB(),d)
        expected=.048+.97*(-.002)+.97**2*(.498)
        self.assertAlmostEqual(m["C/search_target_return"],expected,places=6)
        self.assertEqual(m["C/target_success_probability"],0)
        self.assertGreater(m["C/search_value_loss"],0)
        self.assertIsNone(feature.grad);self.assertIsNone(memory.grad)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in c.search_q1.state_dict().items()))

    def test_spatial_ignores_legacy_transition_protocol(self):
        c=VNextController(8,self.cfg)
        self.assertEqual(c.train_batch(torch.zeros(1,4,8),None,SpatialAB(),replay(time=4)),{"C/transitions":0.})

    def test_false_high_visibility_cannot_permanently_block_recovery(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu")
        state["age"][:]=100;state["stalled"][:]=2001
        _,state,_=self.run_step(c,SpatialAB(),state,proposal=0,
            cur=current(visible=.99,joint=0.),training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(state["phase"].item(),RECOVERY)
        # A credible interaction remains protected even in the same geometry.
        state["actionable_run"][:]=3
        action,state,info=self.run_step(c,SpatialAB(),state,proposal=1,
            cur=current(visible=.99,joint=.99),training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(state["phase"].item(),INTERACT)
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(info["log_vnext_override"].item(),0)

    def test_clearance_stops_after_failed_attempt_limit(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu")
        state["age"][:]=100;state["stalled"][:]=2001;state["forward_stall"][:]=20
        state["clear_attempts"][:]=2
        with torch.no_grad():c.q1[-1].bias[7]=8;c.q2[-1].bias[7]=8
        _,_,info=self.run_step(c,SpatialAB(),state,proposal=0,training=False,
            obs={"vnext_spatial_summary":self.summary()})
        self.assertEqual(info["log_vnext_mask"][0,7].item(),0)

    def test_scan_counts_toward_real_search_budget(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        chosen=[]
        for _ in range(40):
            _,state,info=self.run_step(c,SpatialAB(6),state,proposal=0,training=False,
                obs={"vnext_spatial_summary":self.summary()})
            chosen.append(int(info["log_vnext_choice"].item()==6))
            self.assertLessEqual(info["log_vnext_search_used"].item(),10)
            self.assertLessEqual(info["log_vnext_extra_used"].item(),25)
        self.assertEqual(sum(chosen),9)  # three K3 sweeps, no unbudgeted scan


if __name__=="__main__":unittest.main()

