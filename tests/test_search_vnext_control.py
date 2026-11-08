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
        vnext_probe_probability=1.,vnext_stall_steps=2000,
        vnext_spatial_fix_version=0)
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

    def test_goal_search_rejects_sustained_sky_attack_with_local_evidence(self):
        controller = VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            goal_search_enabled=True,vnext_probe_probability=0.))
        state = controller.initial(1,torch.device("cpu"))
        state["actuator"][:,2] = -40
        state["age"][:] = 100
        cur = current(visible=.8,joint=.0)
        # A's uncertain/global visibility is not a license to follow a locally
        # rejected bright region. The second corroborated frame is required.
        cur["goal_context"] = torch.tensor([[.45,.8,.2,0.,-.6,.4,0.,0.]])
        action,state,info = self.run_step(controller,UnsupportedSpatialAB(),state,
            proposal=1,cur=cur,training=False,obs={"goal_peak_rejected":torch.tensor([True])})
        self.assertEqual(action.argmax(-1).item(),1)
        action,state,info = self.run_step(controller,UnsupportedSpatialAB(),state,
            proposal=1,cur=cur,training=False,obs={"goal_peak_rejected":torch.tensor([True])})
        self.assertEqual(action.argmax(-1).item(),3)
        self.assertEqual(info["log_goal_reject_used"].item(),1)
        self.assertEqual(info["log_vnext_source"].item(),3)
        self.assertEqual(info["log_vnext_version"].item(),3)
        self.assertEqual(info["log_vnext_applied"][0,0].item(),0)

    def test_goal_verify_is_bounded_and_d_visibility_does_not_authorize_attack(self):
        controller = VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            goal_search_enabled=True,goal_verify_max_trials=1,vnext_probe_probability=0.))
        cur = current(visible=.8,joint=.99)
        cur["goal_context"] = torch.tensor([[.6,.7,.2,.65,0.,0.,1.,1.]])
        action,state,info = self.run_step(controller,UnsupportedSpatialAB(),
            proposal=0,cur=cur,training=False,first=True)
        self.assertEqual(action.argmax(-1).item(),5)
        self.assertEqual(info["log_goal_verify_used"].item(),1)
        self.assertEqual(info["log_vnext_mask"][0,8].item(),0)
        self.assertEqual(info["log_vnext_phase"].item(),ALIGN)
        _,state,info = self.run_step(controller,UnsupportedSpatialAB(),state,
            proposal=0,cur=cur,training=False)
        self.assertEqual(info["log_goal_verify_used"].item(),0)
        self.assertEqual(state["goal_verify_trials"].item(),1)

    def test_goal_budget_counts_only_calibrated_absence_and_resets_on_unknown(self):
        controller = VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            goal_search_enabled=True,vnext_probe_probability=0.))
        state = controller.initial(1,torch.device("cpu"))
        state["age"][:] = 100
        state["search_probe_cooldown"][:] = 100
        cur = current(ready=False)
        ab = UnsupportedSpatialAB()
        for _ in range(26):
            _,state,info = self.run_step(controller,ab,state,cur=cur,training=False)
        self.assertEqual(state["goal_absent_run"].item(),0)
        self.assertEqual(info["log_goal_absence_confirmed"].item(),0)
        for _ in range(26):
            _,state,info = self.run_step(controller,ab,state,cur=cur,training=False,
                obs={"goal_absence_confirmed":torch.tensor([True])})
        self.assertEqual(state["goal_absent_run"].item(),26)
        self.assertEqual(info["log_goal_absence_confirmed"].item(),1)
        _,state,_ = self.run_step(controller,ab,state,cur=cur,training=False)
        self.assertEqual(state["goal_absent_run"].item(),0)

    def test_goal_uncertain_evidence_does_not_erase_independent_a_visibility(self):
        controller = VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            goal_search_enabled=True,vnext_probe_probability=0.))
        cur = current(visible=.9,joint=.99)
        cur["goal_context"] = torch.tensor([[.52,.25,.75,0.,0.,.15,0.,1.]])
        state = controller.initial(1,torch.device("cpu"))
        state["search_probe_cooldown"][:] = 100
        for _ in range(4):
            _,state,info = self.run_step(controller,UnsupportedSpatialAB(),state,cur=cur,training=False)
        self.assertEqual(info["log_vnext_phase"].item(),ALIGN)
        self.assertEqual(info["log_vnext_mask"][0,8].item(),0)

    def test_goal_search_probe_uses_local_novelty_without_forecast_qualification(self):
        controller = VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            goal_search_enabled=True,vnext_probe_probability=0.))
        novelty = F.one_hot(torch.tensor([2]),N_OPTIONS).float()
        action,_,info = self.run_step(controller,UnsupportedSpatialAB(),
            proposal=0,training=False,first=True,obs={"goal_direction_novelty":novelty})
        self.assertEqual(action.argmax(-1).item(),5)
        self.assertEqual(info["log_goal_search_probe_used"].item(),1)
        self.assertEqual(info["log_goal_search_learned_used"].item(),0)
        self.assertEqual(info["log_goal_search_eligible"].item(),1)

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


class SpatialRecoveryFixTests(unittest.TestCase):
    run_step = VNextControlTests.run_step
    summary = staticmethod(SpatialControlTests.summary)

    def setUp(self):
        self.cfg=config(vnext_spatial_enabled=True,vnext_spatial_fix_version=1,
            vnext_probe_probability=0.,vnext_stall_steps=8)

    @staticmethod
    def frame(shift=0):
        y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing="ij")
        p=((x//8+y//8+shift)%2).float()*.6+.1
        return torch.stack((p,p*.7,p*.3),-1)[None]

    def test_false_b_motion_cannot_reset_independent_stall(self):
        cfg=config(vnext_spatial_enabled=True,vnext_spatial_fix_version=1,vnext_control=False,
                   vnext_stall_steps=8)
        c=VNextController(8,cfg);state=None;ab=UnsupportedSpatialAB()
        for i in range(16):
            _,state,info=self.run_step(c,ab,state,proposal=6,first=i==0,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(motion=.12)})
        self.assertGreaterEqual(state["stalled"].item(),15)
        self.assertGreaterEqual(state["forward_stall"].item(),12)
        self.assertEqual(info["log_vnext_motion_rejected"].item(),1)
        self.assertEqual(info["log_vnext_spatial_progress"].item(),0)
        self.assertEqual(state["phase"].item(),RECOVERY)

    def test_visual_translation_is_independent_of_pose_qualification(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=6,training=False,
            obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
        state["stalled"][:]=50;state["forward_stall"][:]=20
        state["interact_failures"][:]=2;state["clear_attempts"][:]=2
        _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=6,training=False,
            obs={"image":self.frame(1),"vnext_spatial_summary":self.summary(ready=False)})
        self.assertEqual(info["log_vnext_visual_progress"].item(),1)
        self.assertEqual(state["stalled"].item(),0)
        self.assertEqual(state["forward_stall"].item(),0)
        self.assertEqual(state["interact_failures"].item(),0)

    def test_rotating_image_cannot_erase_blocked_history(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=4,training=False,
            obs={"image":self.frame(),"vnext_spatial_summary":self.summary()})
        state["stalled"][:]=5;state["forward_stall"][:]=5;state["clear_attempts"][:]=1
        _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=4,training=False,
            obs={"image":self.frame(1),"vnext_spatial_summary":self.summary(motion=.2)})
        self.assertEqual(info["log_vnext_visual_progress"].item(),0)
        self.assertEqual(state["stalled"].item(),6)
        self.assertEqual(state["forward_stall"].item(),5)
        self.assertEqual(state["clear_attempts"].item(),1)

    def test_repeating_jump_animation_is_not_endless_visual_progress(self):
        c=VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=1,
            vnext_control=False));state=None;progress=[]
        for i in range(40):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=10,first=i==0,training=False,
                obs={"image":self.frame(i%2),"vnext_spatial_summary":self.summary(motion=.2)})
            progress.append(info["log_vnext_visual_progress"].item())
        self.assertLessEqual(sum(progress),1)
        self.assertEqual(sum(progress[4:]),0)
        self.assertGreaterEqual(state["forward_stall"].item(),35)

    def test_missing_image_is_unknown_not_motion_or_blocked_evidence(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        state["has_visual"][:]=True;state["previous_control"][:,1]=1
        state["stalled"][:]=5;state["forward_stall"][:]=2
        _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=6,training=False,
            obs={"vnext_spatial_summary":self.summary(motion=.2)})
        self.assertEqual(state["stalled"].item(),6)
        self.assertEqual(state["forward_stall"].item(),2)
        self.assertEqual(info["log_vnext_motion_rejected"].item(),1)

    def test_cold_clearance_trial_is_real_bounded_and_not_learned(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        state["stalled"][:]=30;state["forward_stall"][:]=6
        for expected in [1,1,6]:
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
            self.assertEqual(action.argmax(-1).item(),expected)
            self.assertEqual(info["log_vnext_choice"].item(),7)
            self.assertEqual(info["log_vnext_source"].item(),6)
            self.assertEqual(info["log_vnext_score_supported"].sum().item(),0)
        self.assertEqual(state["clear_attempts"].item(),1)
        self.assertEqual(state["recovery_trials"].item(),1)
        self.assertEqual(info["log_vnext_extra_used"].item(),3)
        self.assertEqual(c.coverage.sum().item(),0)  # eval never adds experience

    def test_cold_recovery_waits_for_sticky_jump_then_clears(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        state["stalled"][:]=30;state["forward_stall"][:]=6;state["actuator"][:,1]=9
        for i in range(9):
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=10,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
            self.assertEqual(action.argmax(-1).item(),0)
            self.assertEqual(state["actuator"][0,1].item(),8-i)
        action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=10,training=False,
            obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(info["log_vnext_choice"].item(),7)

    def test_full_repeated_forward_jump_false_pose_chain_reaches_actual_clearance(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        events=[];ab=UnsupportedSpatialAB()
        for _ in range(65):
            action,state,info=self.run_step(c,ab,state,proposal=10,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(motion=.2)})
            events.append((int(action.argmax(-1)),int(info["log_vnext_choice"]),
                           int(state["actuator"][0,1]),int(info["log_vnext_source"])))
            self.assertLessEqual(info["log_vnext_extra_used"].item(),25)
            self.assertLessEqual(state["recovery_trials"].item(),4)
        self.assertTrue(any(option==9 for command,option,jump,source in events))
        attacks=[i for i,(command,option,jump,source) in enumerate(events)
                 if command==1 and option==7 and source==6]
        self.assertTrue(attacks,events)
        self.assertTrue(all(events[i][2]==0 for i in attacks))
        self.assertEqual(c.coverage.sum().item(),0)
        self.assertGreaterEqual(state["forward_stall"].item(),6)

    def test_episode_reset_clears_recovery_and_visual_memory_only_reset_rows(self):
        c=VNextController(8,self.cfg);state=c.initial(2,"cpu")
        state["recovery_trials"][:]=4;state["forward_stall"][:]=30
        state["visual_valid"][:]=True;state["visual_yaw"][:]=80
        _,state,_=c.choose(UnsupportedSpatialAB(),torch.ones(2,8),torch.zeros(2,4),current(2),{},
            F.one_hot(torch.tensor([0,0]),12).float(),state,torch.tensor([True,False]),100000,False)
        self.assertEqual(state["recovery_trials"].tolist(),[0,4])
        self.assertEqual(state["forward_stall"].tolist(),[0,30])
        self.assertFalse(state["visual_valid"][0].any())
        self.assertTrue(state["visual_valid"][1].all())

    def test_recovery_probes_exhaust_and_return_to_search(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        state["stalled"][:]=30
        starts=0
        for i in range(70):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
            starts+=int(info["log_vnext_recovery_probe"].item())
            self.assertLessEqual(info["log_vnext_extra_used"].item(),25)
        self.assertEqual(starts,4)
        self.assertEqual(state["recovery_trials"].item(),4)
        self.assertEqual(state["phase"].item(),SEARCH)
        self.assertEqual(info["log_vnext_override"].item(),0)

    def test_camera_recovery_cannot_repeat_forever(self):
        c=VNextController(8,self.cfg);state=c.initial(1,"cpu");state["age"][:]=100
        state["stalled"][:]=30;state["actuator"][:,2]=-70
        for _ in range(100):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=2,training=False,
                obs={"image":torch.full((1,64,64,3),.7),"vnext_spatial_summary":self.summary(ready=False)})
        self.assertEqual(state["recovery_trials"].item(),4)
        self.assertEqual(state["phase"].item(),SEARCH)

    def test_recovery_probe_still_respects_protection_budget_and_off(self):
        for reason in ["attack","budget","off","unknown_obstruction"]:
            c=VNextController(8,config(vnext_spatial_enabled=True,vnext_spatial_fix_version=1,
                vnext_probe_probability=0.,vnext_control=reason!="off",vnext_stall_steps=8))
            state=c.initial(1,"cpu");state["age"][:]=100;state["stalled"][:]=30
            state["forward_stall"][:]=0 if reason=="unknown_obstruction" else 6
            if reason=="budget":state["history"][0,:26]=2
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,
                proposal=1 if reason=="attack" else 0,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":self.summary(ready=False)})
            self.assertNotEqual(info["log_vnext_choice"].item(),7,reason)
            if reason!="unknown_obstruction":self.assertEqual(info["log_vnext_override"].item(),0,reason)


class SpatialInteractionV2Tests(unittest.TestCase):
    run_step = VNextControlTests.run_step
    frame = staticmethod(SpatialRecoveryFixTests.frame)

    def cfg(self,**values):
        kwargs=dict(vnext_spatial_enabled=True,vnext_spatial_fix_version=2,
            vnext_probe_probability=0.,vnext_stall_steps=2000,
            vnext_interact_result_wait=4,vnext_retry_cooldown=5)
        kwargs.update(values)
        return config(**kwargs)

    def credible(self,x=0.,y=0.,joint=.99,direction_ready=True):
        result=current(visible=.99,joint=joint)
        result.update(direction=torch.tensor([[x,y]]),direction_ready=torch.tensor([direction_ready]))
        return result

    def test_persistent_target_retries_after_two_inconclusive_attacks_but_is_bounded(self):
        c=VNextController(8,self.cfg(vnext_interact_max_trials=4));state=c.initial(1,"cpu")
        state["age"][:]=100;events=[]
        for _ in range(100):
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False,obs={"image":self.frame()})
            events.append((int(action.argmax(-1)),int(info["log_vnext_choice"]),int(info["log_vnext_decision"])))
            self.assertLessEqual(state["interaction_trials"].item(),4)
            self.assertLessEqual(info["log_vnext_extra_used"].item(),25)
        self.assertEqual(state["interaction_trials"].item(),4)
        self.assertEqual(sum(cmd==1 and option==8 for cmd,option,start in events),8)
        self.assertEqual(sum(cmd==6 and option==8 for cmd,option,start in events),4)
        self.assertEqual(state["interact_failures"].item(),4)
        self.assertFalse(state["pending_interact"].item())
        self.assertEqual(state["phase"].item(),ALIGN)
        self.assertEqual(info["log_vnext_interaction_exhausted"].item(),1)

    def test_last_permitted_interaction_completes_its_pickup_lease(self):
        c=VNextController(8,self.cfg(vnext_interact_max_trials=1));state=c.initial(1,"cpu")
        state["age"][:]=100;state["actionable_run"][:]=3;state["visible_run"][:]=3
        commands=[]
        for _ in range(3):
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False)
            commands.append(int(action.argmax(-1)))
            self.assertEqual(info["log_vnext_choice"].item(),8)
        self.assertEqual(commands,[1,1,6])
        self.assertTrue(state["pending_interact"].item())
        self.assertEqual(state["interaction_wait"].item(),4)

    def test_delayed_task_reward_clears_pending_result_without_false_failure(self):
        c=VNextController(8,self.cfg());state=c.initial(1,"cpu")
        state["age"][:]=100;state["actionable_run"][:]=3;state["visible_run"][:]=3
        for _ in range(3):
            _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False)
        for _ in range(2):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False)
            self.assertEqual(state["interact_failures"].item(),0)
            self.assertTrue(state["pending_interact"].item())
            self.assertNotEqual(info["log_vnext_choice"].item(),8)
        _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
            cur=current(visible=.99,joint=.99),training=False,obs={"reward":torch.ones(1)})
        self.assertEqual(state["interact_failures"].item(),0)
        self.assertFalse(state["pending_interact"].item())
        self.assertEqual(state["interaction_wait"].item(),0)
        self.assertEqual(state["interaction_trials"].item(),0)

    def test_historical_two_failures_do_not_permanently_mask_credible_target(self):
        c=VNextController(8,self.cfg());state=c.initial(1,"cpu")
        state["age"][:]=100;state["phase"][:]=INTERACT
        state["actionable_run"][:]=3;state["visible_run"][:]=3
        state["interact_failures"][:]=2;state["interaction_trials"][:]=2
        action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
            cur=current(visible=.99,joint=.99),training=False)
        self.assertEqual(action.argmax(-1).item(),1)
        self.assertEqual(info["log_vnext_choice"].item(),8)
        self.assertEqual(state["interaction_trials"].item(),3)

    def test_result_wait_expiry_is_inconclusive_then_cooldown_not_immediate_failure(self):
        c=VNextController(8,self.cfg(vnext_retry_cooldown=20));state=c.initial(1,"cpu")
        state["age"][:]=100;state["actionable_run"][:]=3;state["visible_run"][:]=3
        for _ in range(3):
            _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False)
        for tick in range(4):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=current(visible=.99,joint=.99),training=False)
            self.assertEqual(state["interact_failures"].item(),int(tick==3))
            self.assertNotEqual(info["log_vnext_choice"].item(),8)
        self.assertEqual(state["interaction_trials"].item(),1)

    def test_alignment_phase_templates_do_not_attack_or_count_as_clearance(self):
        c=VNextController(8,self.cfg())
        commands,lengths=c.templates(torch.zeros(3,3),torch.tensor([100,100,100]),
            torch.tensor([ALIGN,SEARCH,RECOVERY]))
        self.assertEqual(commands[0,6].tolist(),[2,0,0])
        self.assertEqual(commands[0,7].tolist(),[3,0,0])
        self.assertEqual(lengths[0,[1,2,6,7]].tolist(),[1,1,1,1])
        self.assertEqual(commands[1,7].tolist(),[1,1,6])
        self.assertEqual(commands[2,6].tolist(),[4,5,5])
        self.assertEqual(option_names(True,ALIGN)[6:8],("align_up","align_down"))
        state=c.initial(1,"cpu");state["age"][:]=100;state["phase"][:]=ALIGN;state["visible_run"][:]=3
        _,state,info=self.run_step(c,SpatialAB(7),state,proposal=0,
            cur=current(visible=.99,joint=0.),training=False)
        self.assertEqual(info["log_vnext_choice"].item(),7)
        self.assertEqual(state["clear_attempts"].item(),0)
        self.assertEqual(state["clear_cooldown"].item(),0)
        self.assertEqual(state["recovery_trials"].item(),0)

    def test_qualified_visual_direction_can_try_each_camera_axis_without_A_or_Q_support(self):
        for x,y,command,option in [(-.6,0.,4,1),(.6,0.,5,2),(0.,-.6,2,6),(0.,.6,3,7)]:
            c=VNextController(8,self.cfg(vnext_probe_probability=1.));state=c.initial(1,"cpu")
            state["age"][:]=100;state["phase"][:]=ALIGN;state["visible_run"][:]=3
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=self.credible(x,y,joint=0.),training=True)
            self.assertEqual(action.argmax(-1).item(),command)
            self.assertEqual(info["log_vnext_choice"].item(),option)
            self.assertEqual(info["log_vnext_source"].item(),7)
            self.assertEqual(info["log_vnext_behavior_probability"].item(),1)
            self.assertEqual(state["alignment_trials"].item(),1)
            self.assertEqual(info["log_vnext_option_duration"].item(),1)

    def test_unknown_direction_cannot_force_alignment_or_bypass_pitch_clamp(self):
        for ready,pitch in [(False,0.),(True,70.)]:
            c=VNextController(8,self.cfg());state=c.initial(1,"cpu")
            state["age"][:]=100;state["phase"][:]=ALIGN;state["visible_run"][:]=3
            state["actuator"][:,2]=pitch
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=self.credible(0.,.6,joint=0.,direction_ready=ready),training=False)
            self.assertEqual(action.argmax(-1).item(),0)
            self.assertEqual(info["log_vnext_alignment_probe"].item(),0)

    def test_alignment_probe_is_bounded_and_respects_off_protection_and_budget(self):
        for reason in ["off","attack","budget","exhausted"]:
            c=VNextController(8,self.cfg(vnext_control=reason!="off"));state=c.initial(1,"cpu")
            state["age"][:]=100;state["phase"][:]=ALIGN;state["visible_run"][:]=3
            if reason=="budget":state["history"][0,:26]=2
            if reason=="exhausted":state["alignment_trials"][:]=4
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,
                proposal=1 if reason=="attack" else 0,cur=self.credible(.6,0.,joint=0.),training=False)
            self.assertEqual(info["log_vnext_alignment_probe"].item(),0,reason)
            self.assertEqual(info["log_vnext_override"].item(),0,reason)

    def test_water_and_wool_keep_legal_use_interactions(self):
        for target in ["water_bucket","wool"]:
            c=VNextController(8,self.cfg(target_item=target))
            commands,lengths=c.templates(torch.zeros(1,3),torch.tensor([100]),torch.tensor([INTERACT]))
            self.assertEqual(commands[0,8].tolist(),[11,11,11])
            self.assertEqual(lengths[0,8].item(),2)

    def test_partial_reset_clears_only_new_interaction_and_alignment_state(self):
        c=VNextController(8,self.cfg());state=c.initial(2,"cpu")
        state["interaction_trials"][:]=5;state["interaction_wait"][:]=3
        state["pending_interact"][:]=True;state["alignment_trials"][:]=4
        _,state,info=c.choose(UnsupportedSpatialAB(),torch.ones(2,8),torch.zeros(2,4),current(2),{},
            F.one_hot(torch.tensor([0,0]),12).float(),state,torch.tensor([True,False]),100000,False)
        self.assertEqual(state["interaction_trials"].tolist(),[0,5])
        self.assertEqual(state["interaction_wait"].tolist(),[0,2])
        self.assertEqual(state["alignment_trials"].tolist(),[0,4])
        self.assertEqual(state["pending_interact"].tolist(),[False,True])

    def test_replay_reconstructs_alignment_camera_candidates_using_logged_phase(self):
        class RecordingAB(UnsupportedSpatialAB):
            def __init__(self):super().__init__();self.saved=[]
            def predict_candidates(self,feat,memory,controls,actuator,**kw):
                self.saved.append((controls.detach().clone(),kw["phase"].detach().clone()))
                return super().predict_candidates(feat,memory,controls,actuator,**kw)
        c=VNextController(8,self.cfg());ab=RecordingAB();d=replay(time=7)
        d["log_vnext_version"][:]=2
        d["log_vnext_context"][:,:,1:5]=0;d["log_vnext_context"][:,:,2]=1
        d["log_vnext_mask"][:,:,[1,2,3,6,7]]=1
        m=c.train_batch(torch.randn(1,7,8),torch.zeros(1,7,4),ab,d)
        self.assertGreater(m["C/transitions"],0)
        self.assertTrue(ab.saved)
        for controls,phase in ab.saved:
            self.assertTrue((phase==ALIGN).all())
            self.assertTrue((controls[:,7,:,0]==0).all())
            self.assertTrue((controls[:,7,0,7]==10).all())
            self.assertTrue((controls[:,7,1:]==0).all())


class GuidanceV3Tests(unittest.TestCase):
    run_step = VNextControlTests.run_step
    frame = staticmethod(SpatialRecoveryFixTests.frame)

    def cfg(self,**values):
        kwargs=dict(vnext_spatial_enabled=True,vnext_spatial_fix_version=3,
            vnext_probe_probability=0.,vnext_stall_steps=2000,
            vnext_interact_result_wait=4,vnext_retry_cooldown=5,
            vnext_search_fraction=.15,vnext_total_fraction=.30)
        kwargs.update(values)
        return config(**kwargs)

    def state(self,c):
        state=c.initial(1,"cpu");state["age"][:]=100
        return state

    def probe_current(self,ready=True):
        cur=current(visible=.99,joint=.95)
        cur["ready"]["center_interactable"][:]=False
        cur["interaction_probe_ready"]=torch.tensor([ready])
        cur["interaction_probe_quality"]=torch.tensor([.7])
        return cur

    def certify(self,c,phase,option):
        c.updates.fill_(128)
        c.phase_coverage[phase,option]=16
        c.phase_completed[phase,option]=16
        c.phase_positive[phase,option]=4

    def test_phase_coverage_cannot_reuse_recovery_as_search(self):
        c=VNextController(8,self.cfg(vnext_probe_probability=1.))
        c.coverage[6]=100000;c.phase_coverage[RECOVERY,6]=100000
        state=self.state(c)
        _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            obs={"image":self.frame()})
        self.assertEqual(info["log_vnext_choice"].item(),6)
        self.assertEqual(info["log_vnext_source"].item(),9)
        self.assertEqual(c.phase_coverage[SEARCH,6].item(),1)
        self.assertEqual(c.phase_coverage[RECOVERY,6].item(),100000)

    def test_empirical_qualified_search_can_guide_without_localized_B(self):
        c=VNextController(8,self.cfg());self.certify(c,SEARCH,6)
        with torch.no_grad():
            c.q1[-1].bias[6]=3;c.q2[-1].bias[6]=3
        state=self.state(c)
        _,_,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=False,
            obs={"image":self.frame(),"vnext_spatial_summary":torch.zeros(1,12)})
        self.assertEqual(info["log_vnext_choice"].item(),6)
        self.assertEqual(info["log_vnext_source"].item(),1)
        self.assertEqual(info["log_vnext_empirical_support"][0,6].item(),1)

    def test_counts_without_factual_outcomes_do_not_certify_Q(self):
        c=VNextController(8,self.cfg());c.updates.fill_(500)
        c.phase_coverage[:]=500;c.coverage[:]=500
        support=c.empirical_support(self.state(c))
        self.assertFalse(support.any())
        _,_,info=self.run_step(c,UnsupportedSpatialAB(),self.state(c),proposal=0,training=False)
        self.assertEqual(info["log_vnext_source"].item(),9)

    def test_phase_outcomes_and_twin_disagreement_are_independent_checks(self):
        c=VNextController(8,self.cfg());self.certify(c,RECOVERY,6)
        self.assertFalse(c.empirical_support(self.state(c))[0,6])
        self.certify(c,SEARCH,6)
        with torch.no_grad():
            c.q1[-1].bias[6]=4;c.q2[-1].bias[6]=-4
        _,_,info=self.run_step(c,UnsupportedSpatialAB(),self.state(c),proposal=0,training=False)
        self.assertEqual(info["log_vnext_empirical_support"][0,6].item(),0)
        self.assertNotEqual(info["log_vnext_source"].item(),1)

    def test_online_completed_trials_are_unique_and_replay_does_not_inflate_them(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        for i in range(4):
            _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
                obs={"image":torch.roll(self.frame(),i*4,2)})
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)
        before=[v.clone() for v in (c.phase_coverage,c.phase_completed,c.phase_positive)]
        d=replay(time=7);d["log_vnext_version"][:]=2
        for _ in range(3):
            metrics=c.train_batch(torch.ones(1,7,8),torch.zeros(1,7,4),UnsupportedSpatialAB(),d)
            self.assertGreater(metrics["C/transitions"],0)
        for a,b in zip(before,(c.phase_coverage,c.phase_completed,c.phase_positive)):
            self.assertTrue(torch.equal(a,b))

    def test_static_repeated_view_is_not_positive_trial_outcome(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        for _ in range(4):
            _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
                obs={"image":self.frame()})
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),0)

    def test_low_cost_interaction_probe_is_bounded_waits_and_keeps_pickup(self):
        c=VNextController(8,self.cfg());state=self.state(c);events=[]
        for _ in range(120):
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=self.probe_current(),training=False,obs={"image":self.frame()})
            events.append((int(action.argmax(-1)),int(info["log_vnext_choice"]),int(info["log_vnext_source"]),
                           int(info["log_vnext_decision"])))
            self.assertLessEqual(info["log_vnext_extra_used"].item(),30)
        self.assertEqual(sum(o==8 and start for a,o,source,start in events),3)
        self.assertEqual(sum(a==1 and o==8 for a,o,source,start in events),6)
        self.assertEqual(sum(a==6 and o==8 for a,o,source,start in events),3)
        self.assertTrue(all(source==8 for a,o,source,start in events if o==8))
        self.assertEqual(state["interaction_probe_trials"].item(),3)
        self.assertEqual(state["interaction_trials"].item(),3)
        self.assertFalse(state["pending_interact"].item())

    def test_raw_joint_probability_without_visual_probe_permission_cannot_force_attack(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        for _ in range(30):
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,
                cur=self.probe_current(False),training=False)
            self.assertNotEqual(action.argmax(-1).item(),1)
            self.assertNotEqual(info["log_vnext_source"].item(),8)
        self.assertEqual(state["interaction_trials"].item(),0)

    def test_new_probes_respect_actor_attack_off_and_budget(self):
        for reason in ["attack","off","budget"]:
            c=VNextController(8,self.cfg(vnext_control=reason!="off"))
            state=self.state(c)
            if reason=="budget":state["history"][0,:31]=2
            action,state,info=self.run_step(c,UnsupportedSpatialAB(),state,
                proposal=1 if reason=="attack" else 0,cur=self.probe_current(),training=False)
            self.assertEqual(info["log_vnext_override"].item(),0,reason)
            self.assertEqual(state["interaction_probe_trials"].item(),0)

    def test_long_unlocalized_search_has_real_scanning_and_bounded_diverse_trials(self):
        c=VNextController(8,self.cfg());state=self.state(c);choices=set();probes=0
        for _ in range(300):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=False,
                obs={"image":self.frame(),"vnext_spatial_summary":torch.zeros(1,12)})
            if info["log_vnext_source"].item()==9:
                choices.add(int(info["log_vnext_choice"]))
                probes+=int(info["log_vnext_decision"])
            self.assertLessEqual(info["log_vnext_extra_used"].item(),30)
            self.assertLessEqual(info["log_vnext_search_used"].item(),15)
        self.assertGreaterEqual(probes,8)
        self.assertIn(6,choices)
        self.assertGreaterEqual(len(choices),4)
        self.assertEqual(c.phase_coverage.sum().item(),0) # eval is not training
        self.assertEqual(c.phase_completed.sum().item(),0)

    def test_stuck_sideways_actor_gets_recovery_after_previous_allowance_exhausted(self):
        c=VNextController(8,self.cfg(vnext_stall_steps=8));state=self.state(c);later=[];passage=[]
        for i in range(300):
            _,state,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=8,training=False,
                obs={"image":self.frame()})
            if i>120 and info["log_vnext_recovery_probe"].item():later.append(i)
            if info["log_vnext_recovery_probe"].item():passage.append(int(info["log_vnext_choice"]))
            self.assertLessEqual(info["log_vnext_extra_used"].item(),30)
        self.assertTrue(later)
        self.assertTrue(any(o in (3,4,5) for o in passage),passage)
        self.assertGreater(state["recovery_round"].item(),0)

    def test_new_episode_state_reset_does_not_reset_learned_phase_experience(self):
        c=VNextController(8,self.cfg(vnext_control=False));state=c.initial(2,"cpu")
        c.phase_coverage[SEARCH,6]=23
        state["episode_phase_coverage"][:]=4
        state["interaction_probe_trials"][:]=2;state["recovery_round"][:]=3
        _,state,_=c.choose(UnsupportedSpatialAB(),torch.ones(2,8),torch.zeros(2,4),current(2),{},
            F.one_hot(torch.tensor([0,0]),12).float(),state,torch.tensor([True,False]),100000,False)
        self.assertEqual(state["interaction_probe_trials"].tolist(),[0,2])
        self.assertEqual(state["recovery_round"].tolist(),[0,3])
        self.assertEqual(state["episode_phase_coverage"][0].sum().item(),1)
        self.assertGreaterEqual(state["episode_phase_coverage"][1].sum().item(),4*4*12)
        self.assertEqual(c.phase_coverage[SEARCH,6].item(),23)

    def test_recovery_probe_does_not_claim_exact_actor_equivalent_plan(self):
        c=VNextController(8,self.cfg(vnext_stall_steps=8));state=self.state(c)
        state["stalled"][:]=30
        _,_,info=self.run_step(c,UnsupportedSpatialAB(),state,proposal=7,training=False,
            obs={"image":self.frame()})
        self.assertNotEqual(info["log_vnext_choice"].item(),10)
        self.assertEqual(info["log_vnext_source"].item(),6)

    def test_terminal_success_feedback_is_recorded_before_immediate_reset_once(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            obs={"image":self.frame()})
        self.assertEqual(state["trial_option"].item(),6)
        c.observe_result(state,torch.tensor([2.]),torch.tensor([True]),training=True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)
        self.assertEqual(state["trial_option"].item(),-1)
        c.observe_result(state,torch.tensor([2.]),torch.tensor([True]),training=True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        # The next policy call sees reset RGB, not the preceding terminal reward.
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            first=True,obs={"image":self.frame(),"reward":torch.zeros(1)})
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)

    def test_terminal_truncation_closes_actual_partial_trial_without_fake_success(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            obs={"image":self.frame()})
        c.observe_result(state,0.,True,training=True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),0)

    def test_nonterminal_hook_defers_feedback_to_choose_without_double_count(self):
        c=VNextController(8,self.cfg());state=self.state(c)
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            obs={"image":self.frame()})
        c.observe_result(state,2.,False,training=True)
        self.assertEqual(state["trial_feedback"].item(),0)
        self.assertEqual(c.phase_completed.sum().item(),0)
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
            obs={"image":self.frame(),"reward":torch.tensor([2.])})
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)
        c.observe_result(state,0.,True,training=True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)

    def test_invalid_terminal_result_does_not_certify_prior_tentative_trial(self):
        for reward,valid in [(2.,False),(float("nan"),True)]:
            c=VNextController(8,self.cfg());state=self.state(c)
            _,state,_=self.run_step(c,UnsupportedSpatialAB(),state,proposal=0,training=True,
                obs={"image":self.frame()})
            state["trial_positive"][:]=True;state["trial_feedback"][:]=1
            c.observe_result(state,reward,True,training=True,valid=valid)
            self.assertEqual(c.phase_completed.sum().item(),0)
            self.assertEqual(c.phase_positive.sum().item(),0)
            self.assertEqual(state["trial_option"].item(),-1)

    def test_terminal_hook_skips_fresh_reset_actor_and_evaluation_learning(self):
        c=VNextController(8,self.cfg())
        state=c.initial(1,"cpu")
        self.assertIs(c.observe_result(state,2.,True,training=True),state)
        self.assertEqual(c.phase_completed.sum().item(),0)
        # A stale/reset token without a corresponding executed last option is
        # not a factual trial, even if an old pending flag was populated.
        state["trial_option"][:]=6;state["trial_feedback"][:]=1;state["trial_positive"][:]=True
        c.observe_result(state,2.,True,training=True)
        self.assertEqual(c.phase_completed.sum().item(),0)
        self.assertEqual(c.phase_positive.sum().item(),0)
        # Actor trials do not qualify unexecuted search options.
        c_off=VNextController(8,self.cfg(vnext_control=False));state=self.state(c_off)
        _,state,_=self.run_step(c_off,UnsupportedSpatialAB(),state,proposal=0,training=True)
        c_off.observe_result(state,2.,True,training=True)
        self.assertEqual(c_off.phase_completed.sum().item(),0)
        # Evaluation can close runtime state but never trains qualification.
        _,state,_=self.run_step(c,UnsupportedSpatialAB(),self.state(c),proposal=0,training=False)
        c.observe_result(state,2.,True,training=False)
        self.assertEqual(c.phase_completed.sum().item(),0)
        self.assertEqual(c.phase_positive.sum().item(),0)
        self.assertEqual(state["trial_option"].item(),-1)

    def test_terminal_hook_partial_batch_preserves_other_live_trial(self):
        c=VNextController(8,self.cfg());state=c.initial(2,"cpu");state["age"][:]=100
        _,state,_=c.choose(UnsupportedSpatialAB(),torch.ones(2,8),torch.zeros(2,4),current(2),
            {"image":self.frame().expand(2,-1,-1,-1)},
            F.one_hot(torch.tensor([0,0]),12).float(),state,torch.tensor([False,False]),100000,True)
        self.assertEqual(state["trial_option"].tolist(),[6,6])
        c.observe_result(state,torch.tensor([2.,0.]),torch.tensor([True,False]),True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),1)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)
        self.assertEqual(state["trial_option"].tolist(),[-1,6])
        c.observe_result(state,torch.tensor([0.,0.]),torch.tensor([False,True]),True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),2)
        self.assertEqual(c.phase_positive[SEARCH,6].item(),1)
        c.observe_result(state,torch.tensor([2.,2.]),torch.tensor([True,True]),True)
        self.assertEqual(c.phase_completed[SEARCH,6].item(),2)

    def test_historical_state_dict_shapes_remain_strictly_loadable(self):
        old=VNextController(8,self.cfg(vnext_spatial_fix_version=2))
        restored=VNextController(8,self.cfg(vnext_spatial_fix_version=2))
        restored.load_state_dict(old.state_dict(),strict=True)
        self.assertNotIn("phase_coverage",old.state_dict())
        new=VNextController(8,self.cfg())
        self.assertEqual(tuple(new.state_dict()["phase_coverage"].shape),(4,12))
        self.assertEqual(new.q1[0].weight.shape,old.q1[0].weight.shape)


if __name__=="__main__":unittest.main()

