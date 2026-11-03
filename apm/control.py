"""vNext factual success-value controller, independent of the task Actor."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1
SPATIAL_VERSION = 2
ACTOR = 0
SEARCH, ALIGN, INTERACT, RECOVERY = range(4)
OPTION_NAMES = ("actor","left","right","forward","left_forward","right_forward",
                "align_up","align_down","interact","settle","back","recovery_turn")
N_OPTIONS, PREDICTION_DIM, CONTEXT_DIM = 12, 12, 24
SOURCE_NAMES = ("actor", "learned", "rule_interaction", "rule_camera_recovery", "settle", "training_probe")


def option_names(spatial=False):
    """Protocol-specific names; IDs are stable within each checkpoint version."""
    if not spatial:
        return OPTION_NAMES
    return (*OPTION_NAMES[:6], "scan", "clear_obstacle", *OPTION_NAMES[8:])
DEFAULTS = dict(vnext_control=True,vnext_interaction=True,vnext_memory_dim=128,
    vnext_c_hidden=128,vnext_c_lr=1e-4,vnext_c_batch=128,vnext_c_tau=.01,
    vnext_start_step=2500,vnext_warmup=25000,vnext_full_step=100000,
    vnext_search_initial=.05,vnext_search_fraction=.10,vnext_search_max=.15,
    vnext_total_fraction=.25,vnext_settle_limit=12,vnext_budget_window=100,
    vnext_base_episode_fraction=.20,vnext_probe_probability=.15,
    vnext_visible_high=.7,vnext_visible_low=.25,vnext_interactable_high=.85,
    vnext_phase_hysteresis=2,vnext_stall_steps=20,vnext_interaction_protect=32,
    vnext_interact_steps=2,vnext_interact_retries=2,vnext_retry_cooldown=20,
    vnext_commit_steps=3,vnext_c_margin=.01,vnext_c_tie_margin=.02,
    vnext_spatial_enabled=False,vnext_c_discount=.97,vnext_search_value_clip=3.,
    vnext_search_value_weight=.25,vnext_a_score_weight=.10,
    vnext_reward_new=.05,vnext_reward_view=.01,vnext_reward_target=.5,vnext_reward_revisit=.025,
    vnext_reward_cost=.002,vnext_reward_progress=.25,
    vnext_motion_confidence=.4,vnext_motion_threshold=.03,
    vnext_clearance_enabled=True,vnext_clearance_attempts=2,
    vnext_c_min_option_trials=16,vnext_c_min_updates=128)
FIELDS = {k:() for k in (
    "version","decision","choice","option_id","option_end","option_duration",
    "behavior_probability","phase","phase_reason","remaining","remaining_after",
    "override","actor_action","base_episode","search_used","extra_used","settle_used",
    "interact_failures","requested","horizon","recovery_exception",
    "source","stalled","clear_attempts","spatial_progress")}
FIELDS.update(context=(CONTEXT_DIM,),mask=(N_OPTIONS,),prediction=(N_OPTIONS,PREDICTION_DIM),
              q=(N_OPTIONS,),task_q=(N_OPTIONS,),search_q=(N_OPTIONS,),a_score=(N_OPTIONS,),
              score_supported=(N_OPTIONS,),empirical_support=(N_OPTIONS,),
              actuator_before=(3,),actuator_after=(3,),applied=(9,),plan_commands=(3,),actor_plan=(3,))
FIELDS = {"log_vnext_"+k:v for k,v in FIELDS.items()}


def cfg(config,key):
    return getattr(config,key,DEFAULTS[key])


def array(data,key,shape,default=0):
    val=data.get(key)
    return np.full(shape,default,np.float32) if val is None else (
        val.detach().cpu().numpy() if torch.is_tensor(val) else np.asarray(val))


def complete_fields(episode):
    result=episode.copy()
    for key,shape in FIELDS.items():
        result.setdefault(key,np.zeros((len(result["reward"]),*shape),np.float32))
    return result


def factual_search_rewards(data, config):
    """Bounded auxiliary rewards from real, valid labels, never A's predictions.

    Missing sensor labels are unknown, not fabricated discoveries. Interaction
    dwell does not count as wasteful spatial revisiting. The task Actor never
    receives this reward; only the separate high-level search critic does.
    """
    shape = np.shape(data["reward"])
    def label(*names):
        value = np.zeros(shape, np.float32)
        used = np.zeros(shape, bool)
        for name in names:
            raw = array(data, "search_"+name, shape)
            valid = array(data, "search_"+name+"_valid", shape) > .5
            valid &= np.isfinite(raw) & (raw >= 0) & (raw <= 1) & ~used
            value[valid] = raw[valid]
            used |= valid
        return value
    new = label("new_area", "new")
    new_view = label("new_view")
    target = label("first_discovery")
    area_revisit = label("area_revisit")
    view_revisit = label("view_revisit")
    progress = label("progress")
    controls = array(data,"log_applied_control",(*shape,9))
    translation = np.abs(controls[...,1:5]).sum(-1) > 0
    rotation = np.abs(controls[...,7:9]).sum(-1) > 0
    pure_rotation = rotation & ~translation & (controls[...,5] == 0)
    interacting = (controls[...,0]>0) | (controls[...,6]>0)
    # Remaining inside a cell while walking is ordinary locomotion, not a
    # return to that area. View repetition is relevant only to stationary
    # camera sweeps; it must not penalize every forward step in a 1.5m cell.
    revisit = (area_revisit*translation + view_revisit*pure_rotation) * ~interacting
    reward = (float(cfg(config,"vnext_reward_new"))*new
              +float(cfg(config,"vnext_reward_view"))*new_view
              +float(cfg(config,"vnext_reward_target"))*target
              +float(cfg(config,"vnext_reward_progress"))*progress
              -float(cfg(config,"vnext_reward_revisit"))*revisit
              -float(cfg(config,"vnext_reward_cost")))
    return np.clip(reward,-1.,1.).astype(np.float32)


def build_transitions(data,max_duration=3):
    """Outgoing log row r belongs to observation r-1; window tails are censored."""
    shape=np.shape(data["reward"])
    if len(shape)!=2:
        raise ValueError("vNext C needs [batch,time] Replay")
    first=array(data,"is_first",shape).astype(bool)
    last=array(data,"is_last",shape).astype(bool)|array(data,"is_terminal",shape).astype(bool)
    starts=array(data,"log_vnext_decision",shape)>.5
    version=array(data,"log_vnext_version",shape)
    choice=array(data,"log_vnext_choice",shape).astype(int)
    ends=array(data,"log_vnext_option_end",shape)>.5
    contexts=array(data,"log_vnext_context",(*shape,CONTEXT_DIM))
    masks=array(data,"log_vnext_mask",(*shape,N_OPTIONS))
    remaining=array(data,"log_vnext_remaining_after",shape,-1)
    success=array(data,"success",shape)>.5
    env_error=array(data,"log_env_error",shape)>.5
    result=[]
    for b in range(shape[0]):
        for r in np.flatnonzero(starts[b]&np.isin(version[b],(VERSION,SPATIAL_VERSION))):
            if r==0 or first[b,r] or last[b,r-1]:
                continue
            a=int(choice[b,r])
            if not 0<=a<N_OPTIONS or masks[b,r,a]<=0:
                continue
            end=None
            for e in range(r,min(shape[1],r+max_duration)):
                if first[b,e]:
                    break
                if ends[b,e] or last[b,e] or remaining[b,e]==0:
                    end=e
                    break
                if e+1<shape[1] and starts[b,e+1] and not first[b,e+1]:
                    end=e
                    break
            if end is None or env_error[b,r:end+1].any():
                continue
            terminal=bool(last[b,end] or remaining[b,end]==0)
            if not terminal:
                if end+1>=shape[1] or first[b,end+1] or version[b,end+1]!=version[b,r]:
                    continue
                nm,nc=masks[b,end+1].copy(),contexts[b,end+1].copy()
                if not nm[ACTOR] or not np.isfinite(nm).all():
                    continue
            else:
                nm=np.eye(N_OPTIONS,dtype=np.float32)[ACTOR]
                nc=contexts[b,r].copy()
                nc[0]=0
            result.append(dict(batch=b,start=r-1,end=end,action=a,duration=end-r+1,
                version=int(version[b,r]),
                success=float(success[b,r:end+1].any()),terminal=terminal,
                context=contexts[b,r].copy(),mask=masks[b,r].copy(),
                next_context=nc,next_mask=nm))
    return result


class VNextController(nn.Module):
    version=VERSION
    def __init__(self,feature_dim,config,memory_dim=None):
        super().__init__()
        self.config,self.feature_dim=config,int(feature_dim)
        self.spatial = bool(cfg(config,"vnext_spatial_enabled"))
        self.version = SPATIAL_VERSION if self.spatial else VERSION
        if self.spatial:
            if not 0 < float(cfg(config,"vnext_c_discount")) <= 1:
                raise ValueError("High-level search discount must be in (0,1]")
            if float(cfg(config,"vnext_search_value_clip")) <= 0:
                raise ValueError("High-level search value clip must be positive")
            if min(int(cfg(config,"vnext_c_min_option_trials")),int(cfg(config,"vnext_c_min_updates"))) < 1:
                raise ValueError("Empirical control support requires positive online trials and updates")
        self.memory_dim=int(cfg(config,"vnext_memory_dim") if memory_dim is None else memory_dim)
        if int(cfg(config,"vnext_commit_steps"))!=3:
            raise ValueError("vNext protocol commits at most K3")
        size=self.feature_dim+self.memory_dim+CONTEXT_DIM+N_OPTIONS*PREDICTION_DIM
        def network():
            hidden=int(cfg(config,"vnext_c_hidden"))
            n=nn.Sequential(nn.Linear(size,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,N_OPTIONS))
            nn.init.zeros_(n[-1].weight);nn.init.zeros_(n[-1].bias)
            return n
        with torch.random.fork_rng(devices=[]):
            self.q1,self.q2=network(),network()
            self.target_q1,self.target_q2=copy.deepcopy(self.q1),copy.deepcopy(self.q2)
            if self.spatial:
                self.search_q1,self.search_q2=network(),network()
                self.target_search_q1=copy.deepcopy(self.search_q1)
                self.target_search_q2=copy.deepcopy(self.search_q2)
                self.target_search_q1.requires_grad_(False)
                self.target_search_q2.requires_grad_(False)
        self.target_q1.requires_grad_(False);self.target_q2.requires_grad_(False)
        self._parameters_q=list(self.q1.parameters())+list(self.q2.parameters())
        if self.spatial:
            self._parameters_q+=list(self.search_q1.parameters())+list(self.search_q2.parameters())
        self._optimizer=torch.optim.Adam(self._parameters_q,lr=float(cfg(config,"vnext_c_lr")))
        self.register_buffer("updates",torch.zeros((),dtype=torch.long))
        self.register_buffer("coverage",torch.zeros(N_OPTIONS,dtype=torch.long))
        self._rng=np.random.RandomState(int(getattr(config,"seed",0))+19741)

    def rng_state(self):
        return self._rng.get_state()

    def load_rng_state(self,value):
        if value is not None:self._rng.set_state(value)

    def optimizer_state_dict(self):
        return self._optimizer.state_dict()

    def load_optimizer_state_dict(self,value):
        self._optimizer.load_state_dict(value)

    def initial(self,batch,device):
        from apm.actuator import initial_state
        z=lambda *dims:torch.zeros(batch,*dims,device=device)
        return dict(actuator=torch.as_tensor(initial_state((batch,)),device=device),
            previous_control=z(9),phase=z().long(),age=z().long(),
            history=z(int(cfg(self.config,"vnext_budget_window"))).long(),
            cursor=z().long(),active=torch.full((batch,),-1,device=device,dtype=torch.long),
            position=z().long(),length=torch.ones(batch,device=device,dtype=torch.long),
            commands=z(3).long(),decision_id=z().long(),base_only=z().bool(),
            visible_run=z().long(),invisible_run=z().long(),actionable_run=z().long(),not_actionable_run=z().long(),
            stalled=z().long(),previous_visual=z(192),has_visual=z().bool(),
            interaction_run=z().long(),interact_failures=z().long(),cooldown=z().long(),
            pending_interact=z().bool(),phase_age=z().long(),limit_run=z().long(),
            spatial_summary=z(12),clear_attempts=z().long(),clear_cooldown=z().long(),
            source=z().long(),forward_stall=z().long())

    def previous_control(self,state,is_first=None,device=None):
        if state is None:return None
        val=state["previous_control"]
        return val if is_first is None else val*(~is_first.reshape(-1).bool()).float()[:,None]

    def get_actuator(self,state,batch=None,device=None):
        return state["actuator"] if state is not None else self.initial(batch,device)["actuator"]

    @staticmethod
    def current_value(current,name,reference):
        val=current.get(name,0.)
        return torch.as_tensor(val,device=reference.device).float().expand(len(reference))

    @staticmethod
    def ready(current,name,reference):
        r=current.get("ready",{})
        names=("visible","center_target","reachable","center_interactable","progress")
        val=r.get(name,False) if isinstance(r,dict) else r[...,names.index(name)]
        return torch.as_tensor(val,device=reference.device).bool().expand(len(reference))

    def input(self,feat,memory,prediction,context):
        if memory is None:memory=feat.new_zeros(len(feat),self.memory_dim)
        if prediction.shape!=(len(feat),N_OPTIONS,PREDICTION_DIM):
            raise ValueError("vNext candidate descriptor shape mismatch")
        return torch.cat((F.layer_norm(feat.detach().float(),(self.feature_dim,)),
            memory.detach().float(),prediction.detach().float().flatten(1),context.detach().float()),-1)

    def values(self,feat,memory,prediction,context,target=False):
        x=self.input(feat,memory,prediction,context)
        a,b=(self.target_q1,self.target_q2) if target else (self.q1,self.q2)
        task=torch.minimum(a(x).sigmoid(),b(x).sigmoid())
        if not self.spatial:return task
        a,b=(self.target_search_q1,self.target_search_q2) if target else (self.search_q1,self.search_q2)
        clip=float(cfg(self.config,"vnext_search_value_clip"))
        search=torch.minimum(a(x),b(x)).clamp(-clip,clip)/clip
        return task+float(cfg(self.config,"vnext_search_value_weight"))*search

    @staticmethod
    def prediction_score(result, reference):
        shape=reference.shape[:2]
        score=torch.as_tensor(result.get("score",reference.new_zeros(shape)),device=reference.device).float()
        supported=torch.as_tensor(result.get("score_supported",reference.new_zeros(shape)),device=reference.device).bool().clone()
        if score.shape!=shape or supported.shape!=shape:
            raise ValueError("Candidate exploration score/support shape differs")
        finite=torch.isfinite(score)
        supported &= finite
        # Empirical C eligibility does not certify A. Unsupported predictions
        # cannot acquire a direct score merely because the Q head has learned.
        forecast_support=torch.as_tensor(result.get("support",reference.new_zeros(shape)),device=reference.device).bool()
        if forecast_support.ndim==3:forecast_support=forecast_support.any(-1)
        if forecast_support.shape!=shape:
            raise ValueError("Candidate forecast support shape differs")
        supported &= forecast_support
        return score.nan_to_num().clamp(-1,1)*supported, supported

    def empirical_support(self, state):
        """Independent Q eligibility from real online trials and legal odometry.

        Coverage is incremented by online decisions, never by replay training.
        It is qualification only: phase, budget, clearance and interaction
        masks still determine whether any qualified option may be executed.
        """
        summary=state["spatial_summary"]
        confident=(summary[:,0]>.5)&(summary[:,11]>=float(cfg(self.config,"vnext_motion_confidence")))
        enough_updates=self.updates>=int(cfg(self.config,"vnext_c_min_updates"))
        enough_trials=self.coverage>=int(cfg(self.config,"vnext_c_min_option_trials"))
        support=confident[:,None]&enough_updates&enough_trials[None]
        support[:,ACTOR]=False
        return support

    def templates(self,actuator,remaining):
        device=actuator.device
        ids=torch.tensor([[0,0,0],[4,4,4],[5,5,5],[6,6,6],[4,6,6],[5,6,6],
            [2,0,0],[3,0,0],[1,1,0],[0,0,0],[7,7,7],[4,4,6]],device=device)
        commands=ids[None].expand(len(actuator),-1,-1).clone()
        if self.spatial:
            commands[:,6]=torch.tensor([4,5,5],device=device)  # bounded view sweep
            commands[:,7]=torch.tensor([1,1,6],device=device)  # clear then test passage
        # Bounded recovery may correct a saturated own camera estimate. It is
        # neither a target bearing estimate nor an oracle-derived direction.
        boundary=actuator[:,2].abs()>=60
        correction=torch.where(actuator[:,2]<0,torch.full_like(actuator[:,2],3,dtype=torch.long),
                               torch.full_like(actuator[:,2],2,dtype=torch.long))
        commands[:,11]=torch.where(boundary[:,None],correction[:,None],commands[:,11])
        if getattr(self.config,"target_item","sand") in ("water_bucket","wool"):
            commands[:,8]=11
        lengths=torch.full((len(actuator),N_OPTIONS),3,device=device,dtype=torch.long)
        lengths[:,6:8]=1
        if self.spatial:lengths[:,6:8]=3
        lengths[:,8]=min(3,max(1,int(cfg(self.config,"vnext_interact_steps"))))
        lengths[:,9]=actuator[:,1].long().clamp(min=1,max=3)
        return commands,torch.minimum(lengths,remaining.long().clamp(min=1)[:,None])

    @staticmethod
    def controls(commands,actuator,lengths=None):
        from apm.actuator import apply_command_torch
        state=actuator[:,None].expand(-1,N_OPTIONS,-1).clone()
        outputs=[]
        for t in range(commands.shape[-1]):
            applied,nxt=apply_command_torch(commands[...,t],state)
            valid=torch.ones_like(commands[...,t],dtype=torch.bool) if lengths is None else t<lengths
            outputs.append(applied*valid[...,None])
            state=torch.where(valid[...,None],nxt,state)
        return torch.stack(outputs,-2),state

    def context(self,state,current,remaining,search_used,extra_used,settle_used):
        ref=state["previous_control"];out=ref.new_zeros(len(ref),CONTEXT_DIM)
        out[:,0]=remaining.float()/self.max_steps()
        out[:,1:5]=F.one_hot(state["phase"],4).float()
        out[:,5]=search_used/state["history"].shape[1]
        out[:,6]=extra_used/state["history"].shape[1]
        out[:,7]=settle_used/max(1,int(cfg(self.config,"vnext_settle_limit")))
        out[:,8]=state["actuator"][:,1]/10.
        out[:,9]=state["actuator"][:,0]/10.
        out[:,10]=state["actuator"][:,2]/70.
        out[:,11]=self.current_value(current,"visible",ref)
        out[:,12]=self.current_value(current,"center_interactable",ref)
        out[:,13]=state["stalled"].float().clamp(max=100)/100
        out[:,14]=state["interact_failures"].float()/max(1,int(cfg(self.config,"vnext_interact_retries")))
        if self.spatial:
            out[:,15:]=state["spatial_summary"][:,[0,1,2,3,6,7,8,10,11]]
        else:out[:,15:]=ref
        return out

    def max_steps(self):
        override=int(getattr(self.config,"episode_max_steps_override",0))
        return max(1,override or int(getattr(self.config,"episode_max_steps",1000)))


    @torch.no_grad()
    def choose(self,ab,feat,memory,current,obs,actor_action,state,is_first,step,
               training,actor_sequence=None):
        from apm.actuator import apply_command_torch
        batch,device=len(feat),feat.device
        fresh=self.initial(batch,device)
        state=fresh if state is None else {k:state.get(k,v).detach().clone() for k,v in fresh.items()}
        first=is_first.reshape(batch).bool()
        for key in state:
            reset=first.reshape(-1,*([1]*(state[key].ndim-1)))
            state[key]=torch.where(reset,fresh[key],state[key])
        if training and first.any():
            base=self._rng.rand(int(first.sum()))<float(cfg(self.config,"vnext_base_episode_fraction"))
            state["base_only"][first]=torch.as_tensor(base,device=device)
        proposed=actor_action.argmax(-1)
        spatial_progress=torch.zeros(batch,device=device,dtype=torch.bool)
        moving_progress=torch.zeros_like(spatial_progress)
        if self.spatial:
            summary=torch.as_tensor(obs.get("vnext_spatial_summary",feat.new_zeros(batch,12)),device=device).float()
            if summary.shape!=(batch,12) or not torch.isfinite(summary).all():
                raise ValueError("Spatial control needs finite [batch,12] summary")
            state["spatial_summary"]=summary.detach().clone()
            confident=(summary[:,0]>.5)&(summary[:,11]>=float(cfg(self.config,"vnext_motion_confidence")))
            moving_progress=confident&(summary[:,7]*.5>=float(cfg(self.config,"vnext_motion_threshold")))
            # Looking in a genuinely new direction is useful observation, but
            # a second sweep of the same direction cannot erase stagnation.
            new_view=confident&(summary[:,3]<.5)&(summary[:,10]<.99)
            spatial_progress=moving_progress|new_view
        visual_changed=torch.zeros(batch,device=device,dtype=torch.bool)
        image=obs.get("image")
        if image is not None:
            raw_dtype=image.dtype
            image=image.float()
            if raw_dtype==torch.uint8:image=image/255.
            visual=F.adaptive_avg_pool2d(image.permute(0,3,1,2),(8,8)).flatten(1)
            delta=(visual-state["previous_visual"]).abs().mean(-1)
            visual_changed=state["has_visual"]&(delta>.015)
            static=state["has_visual"]&(delta<.004)
            if self.spatial:
                control=state["previous_control"]
                # Before odometry is calibrated, only non-rotating/non-jumping
                # translational visual change is tentative movement evidence.
                translation=control[:,1:5].abs().sum(-1)>0
                clean_motion=translation&(control[:,5]==0)&(control[:,7:9].abs().sum(-1)==0)
                fallback=(state["spatial_summary"][:,0]<.5)&clean_motion&visual_changed
                spatial_progress|=fallback
            else:
                state["stalled"]=torch.where(static,state["stalled"]+1,torch.zeros_like(state["stalled"]))
            state["previous_visual"],state["has_visual"]=visual,torch.ones_like(first)
        if self.spatial:
            state["stalled"]=torch.where(spatial_progress,torch.zeros_like(state["stalled"]),state["stalled"]+1)
            prev=state["previous_control"]
            forward=(prev[:,1]>0)&(prev[:,7:9].abs().sum(-1)==0)
            confident=(state["spatial_summary"][:,0]>.5)&(state["spatial_summary"][:,11]>=float(cfg(self.config,"vnext_motion_confidence")))
            state["forward_stall"]=torch.where(forward&confident&~moving_progress,
                state["forward_stall"]+1,torch.where(moving_progress,torch.zeros_like(state["forward_stall"]),state["forward_stall"]))
            state["clear_cooldown"]=(state["clear_cooldown"]-1).clamp_min(0)
        reward=torch.as_tensor(obs.get("reward",obs.get("obs_reward",feat.new_zeros(batch))),
                               device=device).reshape(batch,-1).sum(-1)
        progress=reward>0
        proposed_control,_=apply_command_torch(proposed,state["actuator"])
        clipped=((proposed==2)|(proposed==3))&(proposed_control[:,7]==0)&(state["actuator"][:,2].abs()>=60)
        state["limit_run"]=torch.where(clipped&~progress,state["limit_run"]+1,torch.zeros_like(state["limit_run"]))
        state["interact_failures"]+=(state["pending_interact"]&~progress).long()
        state["pending_interact"][:]=False
        state["cooldown"]=(state["cooldown"]-1).clamp_min(0)
        new_evidence=(moving_progress if self.spatial else visual_changed)&(state["cooldown"]==0)
        state["interact_failures"][new_evidence|progress]=0
        if self.spatial:
            state["clear_attempts"][moving_progress|progress]=0
            state["forward_stall"][moving_progress|progress]=0
        state["stalled"][progress]=0
        failed=state["interact_failures"]>=int(cfg(self.config,"vnext_interact_retries"))
        visible=self.current_value(current,"visible",feat)
        actionable=self.current_value(current,"center_interactable",feat)
        high=(visible>=float(cfg(self.config,"vnext_visible_high")))&self.ready(current,"visible",feat)
        low=visible<float(cfg(self.config,"vnext_visible_low"))
        act=(actionable>=float(cfg(self.config,"vnext_interactable_high")))&self.ready(current,"center_interactable",feat)
        for key,condition in (("visible_run",high),("invisible_run",low),("actionable_run",act),("not_actionable_run",~act)):
            state[key]=torch.where(condition,state[key]+1,torch.zeros_like(state[key]))
        hysteresis=int(cfg(self.config,"vnext_phase_hysteresis"))
        phase=state["phase"].clone()
        recovery_lease=(state["active"]==11)&(state["position"]<state["length"])&(
            (state["commands"][:,0]==2)|(state["commands"][:,0]==3))
        recovery_settle=torch.zeros_like(first)
        if self.spatial:
            recovery_lease=(state["position"]<state["length"])&torch.isin(state["active"],torch.tensor([7,10,11],device=device))
            recovery_settle=(state["active"]==9)&(state["phase"]==RECOVERY)&(state["position"]<state["length"])
            recovery_lease|=recovery_settle
        phase[state["invisible_run"]>=hysteresis+1]=SEARCH
        phase[(state["visible_run"]>=hysteresis)&((phase==SEARCH)|(phase==RECOVERY))]=ALIGN
        phase[(phase==INTERACT)&(state["not_actionable_run"]>=hysteresis+1)]=ALIGN
        phase[state["actionable_run"]>=hysteresis]=INTERACT
        stalled=state["stalled"]>=int(cfg(self.config,"vnext_stall_steps"))
        bounded_dead=(state["limit_run"]>=3)|stalled
        phase[bounded_dead&~act&(~high if not self.spatial else ~progress)]=RECOVERY
        # Keep only the current committed short recovery when its first camera
        # correction changes the image. A new credible target/real progress may
        # interrupt immediately; no long lease or unbounded scripted scanning.
        phase[recovery_lease&~act&~progress&(~high if not self.spatial else ~moving_progress)]=RECOVERY
        phase[recovery_lease&high&(~bounded_dead if self.spatial else torch.ones_like(high))]=ALIGN
        if self.spatial:
            newly_credible=high&(state["visible_run"]<=hysteresis)
            phase[recovery_settle&newly_credible]=ALIGN
        phase[recovery_lease&act]=INTERACT
        emergency=(phase==RECOVERY)&(state["actuator"][:,2].abs()>=60)&bounded_dead&~act&~progress
        if not self.spatial:emergency&=~high
        already_correcting=((state["actuator"][:,2]<0)&(proposed==3))|((state["actuator"][:,2]>0)&(proposed==2))
        emergency&=~already_correcting
        changed=phase!=state["phase"]
        state["phase_age"]=torch.where(changed,torch.zeros_like(state["phase_age"]),state["phase_age"]+1)
        state["phase"]=phase
        interact=(proposed==1)|(proposed==11)
        state["interaction_run"]=torch.where(interact,state["interaction_run"]+1,torch.zeros_like(state["interaction_run"]))
        protect=interact&((state["interaction_run"]<=int(cfg(self.config,"vnext_interaction_protect")))|progress)

        history=state["history"];rows=torch.arange(batch,device=device)
        history[rows,state["cursor"]]=0
        search_used=(history==1).sum(-1).float()
        extra_used=(history>0).sum(-1).float()
        settle_used=(history==3).sum(-1).float()
        window=history.shape[1]
        fraction=float(cfg(self.config,"vnext_search_initial")) if step<int(cfg(self.config,"vnext_full_step")) else float(cfg(self.config,"vnext_search_fraction"))
        fraction=min(fraction,float(cfg(self.config,"vnext_search_max")))
        span=(state["age"]+1).clamp(max=window).float()
        search_limit=torch.floor(span*fraction).clamp_min(3)
        extra_limit=torch.floor(span*float(cfg(self.config,"vnext_total_fraction"))).clamp_min(3)
        remaining=(self.max_steps()-state["age"]).clamp_min(0)
        commands,lengths=self.templates(state["actuator"],remaining)
        context=self.context(state,current,remaining,search_used,extra_used,settle_used)
        mask=torch.zeros(batch,N_OPTIONS,device=device,dtype=torch.bool);mask[:,ACTOR]=True
        allowed=(~state["base_only"]&~protect&~progress&(step>=int(cfg(self.config,"vnext_start_step"))))
        if not bool(cfg(self.config,"vnext_control")):allowed[:]=False
        total_ok=extra_used[:,None]+lengths<=extra_limit[:,None]
        search_ok=search_used[:,None]+lengths<=search_limit[:,None]
        for o in (1,2,3,4,5):
            mask[:,o]=allowed&(phase==SEARCH)&total_ok[:,o]&search_ok[:,o]
        mask[:,3]|=allowed&(phase==ALIGN)&total_ok[:,3]
        direction=current.get("direction")
        if direction is not None:
            direction=torch.as_tensor(direction,device=device)
            mask[:,6]=allowed&(phase==ALIGN)&(direction[:,1]<-.1)&total_ok[:,6]
            mask[:,7]=allowed&(phase==ALIGN)&(direction[:,1]>.1)&total_ok[:,7]
        if self.spatial:
            mask[:,6]=allowed&((phase==SEARCH)|(phase==RECOVERY))&total_ok[:,6]
            mask[:,6]&=(phase!=SEARCH)|search_ok[:,6]
            mask[:,7]=allowed&(phase==RECOVERY)&total_ok[:,7]&~act
            mask[:,7]&=(state["forward_stall"]>=3)&(state["actuator"][:,1]==0)
            mask[:,7]&=(state["actuator"][:,2].abs()<=45)&(state["clear_cooldown"]==0)
            mask[:,7]&=state["clear_attempts"]<int(cfg(self.config,"vnext_clearance_attempts"))
            if not bool(cfg(self.config,"vnext_clearance_enabled")):mask[:,7]=False
        if bool(cfg(self.config,"vnext_interaction")):
            mask[:,8]=allowed&(phase==INTERACT)&act&(state["actuator"][:,1]==0)&(state["cooldown"]==0)&~failed&total_ok[:,8]
            mask[:,9]=allowed&(phase==INTERACT)&(state["actuator"][:,1]>0)&(settle_used+lengths[:,9]<=int(cfg(self.config,"vnext_settle_limit")))&total_ok[:,9]
        if self.spatial:
            # A blocked jump-forward Actor must be allowed to let the original
            # sticky jump expire before a learned clearance attempt. No timer
            # is cleared here; neutral commands execute normal simulator ticks.
            mask[:,9]|=allowed&(phase==RECOVERY)&(state["actuator"][:,1]>0)&total_ok[:,9]
            mask[:,9]&=settle_used+lengths[:,9]<=int(cfg(self.config,"vnext_settle_limit"))
        for o in (10,11):
            mask[:,o]=allowed&(phase==RECOVERY)&total_ok[:,o]
        controls,end_actuator=self.controls(commands,state["actuator"],lengths)
        for o in ((1,2,6) if self.spatial else (1,2,6,7)):
            mask[:,o]&=controls[:,o,...,-2:].abs().sum((-1,-2))>0
        for o in range(1,N_OPTIONS):
            for previous in range(1,o):
                same=(controls[:,o]-controls[:,previous]).abs().sum((-1,-2))==0
                same&=(end_actuator[:,o]-end_actuator[:,previous]).abs().sum(-1)==0
                same&=lengths[:,o]==lengths[:,previous]
                mask[:,o]&=~(mask[:,previous]&same)

        active=state["active"]
        # A commitment reserves its remaining budget at start. Do not demand
        # another full K3 allowance on the second step of an active option.
        compatible=torch.zeros_like(first)
        compatible|=((active>=1)&(active<=5)&(phase==SEARCH))
        compatible|=(active==3)&(phase==ALIGN)
        compatible|=((active==6)|(active==7))&(phase==ALIGN)
        if self.spatial:
            compatible &= ~((active==6)|(active==7))
            compatible|=(active==6)&((phase==SEARCH)|(phase==RECOVERY))
            compatible|=(active==7)&(phase==RECOVERY)
            compatible|=(active==9)&(phase==RECOVERY)
        compatible|=((active==8)|(active==9))&(phase==INTERACT)
        compatible|=((active==10)|(active==11))&(phase==RECOVERY)
        budget_one=extra_used<extra_limit
        search_active=(active>=1)&(active<=(6 if self.spatial else 5))&(phase==SEARCH)
        budget_one&=~search_active|(search_used<search_limit)
        budget_one&=(active!=9)|(settle_used<int(cfg(self.config,"vnext_settle_limit")))
        abort=((active>0)&(~compatible|~allowed|~budget_one|changed))|((active==ACTOR)&changed)
        state["active"][abort]=-1;state["position"][abort]=0
        new=state["active"]<0
        info={key:feat.new_zeros(batch,*dims) for key,dims in FIELDS.items()}
        info["log_vnext_version"].fill_(self.version)
        info.update(log_vnext_actor_action=proposed,log_vnext_phase=phase,
            log_vnext_phase_reason=changed.float()+2*stalled.float()+4*(state["limit_run"]>=3).float(),
            log_vnext_remaining=remaining,log_vnext_remaining_after=(remaining-1).clamp_min(0),
            log_vnext_base_episode=state["base_only"].float(),
            log_vnext_interact_failures=state["interact_failures"],log_vnext_stalled=state["stalled"],
            log_vnext_clear_attempts=state["clear_attempts"],log_vnext_spatial_progress=spatial_progress.float())
        descriptor=feat.new_zeros(batch,N_OPTIONS,PREDICTION_DIM)
        q=feat.new_zeros(batch,N_OPTIONS)
        a_score=feat.new_zeros(batch,N_OPTIONS)
        score_supported=torch.zeros(batch,N_OPTIONS,device=device,dtype=torch.bool)
        empirical_support=torch.zeros(batch,N_OPTIONS,device=device,dtype=torch.bool)
        task_q=feat.new_zeros(batch,N_OPTIONS)
        search_q=feat.new_zeros(batch,N_OPTIONS)
        probability=feat.new_ones(batch)
        selected=torch.zeros(batch,device=device,dtype=torch.long)
        actor_plan=proposed[:,None].expand(-1,3).clone()
        if new.any():
            starts=new.nonzero().squeeze(-1)
            state["source"][starts]=0
            # Explicit actuator recovery is a small, logged safety experiment,
            # not evidence of learned exploration. It needs no uncalibrated A
            # prediction and never bypasses protection, phase or budget limits.
            recover=starts[emergency[starts]&mask[starts,11]]
            if len(recover):
                selected[recover]=11
                state["source"][recover]=3
                mask[recover]=False
                mask[recover,ACTOR]=True
                mask[recover,11]=True
                if training:self.coverage[11]+=len(recover)
            forced_recovery=emergency&mask[:,11]
            calculate=starts[mask[starts,1:].any(-1)&~forced_recovery[starts]]
            if len(calculate):
                commands[calculate,ACTOR]=actor_plan[calculate]
                if actor_sequence is not None:
                    seq=actor_sequence(calculate,k=3)
                    if seq.ndim==3:seq=seq.argmax(-1)
                    commands[calculate,ACTOR]=seq
                    actor_plan[calculate]=seq
                controls,end_actuator=self.controls(commands,state["actuator"],lengths)
                result=ab.predict_candidates(feat[calculate].detach(),
                    memory[calculate] if memory is not None else None,controls[calculate],
                    state["actuator"][calculate],lengths=lengths[calculate],
                    phase=phase[calculate],remaining=remaining[calculate],
                    option_ids=torch.arange(N_OPTIONS,device=device))
                descriptor[calculate]=result["descriptor"].detach()
                if self.spatial:
                    a_score[calculate],score_supported[calculate]=self.prediction_score(result,descriptor[calculate])
                    empirical_support[calculate]=self.empirical_support(state)[calculate]
                support=result.get("support",torch.zeros(len(calculate),N_OPTIONS,device=device,dtype=torch.bool))
                if support.ndim==3:support=support.any(-1)
                local_mask=mask[calculate].clone()
                # Actor is closed-loop at execution. Merge only exact planned
                # equivalence, retaining its final actuator state and duration.
                for o in range(1,N_OPTIONS):
                    same=(controls[calculate,o]-controls[calculate,ACTOR]).abs().sum((-1,-2))==0
                    same&=(end_actuator[calculate,o]-end_actuator[calculate,ACTOR]).abs().sum(-1)==0
                    same&=lengths[calculate,o]==lengths[calculate,ACTOR]
                    local_mask[:,o]&=~same
                if not training:
                    gate=(support.bool()|empirical_support[calculate]) if self.spatial else support.bool()
                    local_mask[:,1:]&=gate[:,1:]
                    if self.spatial:
                        # Clearance requires a trained blocked-outcome head or
                        # a supported composite score in addition to odometry.
                        event_support=result.get("supported")
                        blocked=(torch.as_tensor(event_support,device=device)[:,7,6].bool()
                                 if event_support is not None else score_supported[calculate,7])
                        local_mask[:,7]=mask[calculate,7]&(blocked|empirical_support[calculate,7])
                    local_mask[:,8]=mask[calculate,8]
                    local_mask[:,9]=mask[calculate,9]
                q[calculate]=self.values(feat[calculate],
                    memory[calculate] if memory is not None else None,
                    descriptor[calculate],context[calculate])
                x=self.input(feat[calculate],memory[calculate] if memory is not None else None,
                             descriptor[calculate],context[calculate])
                task_q[calculate]=torch.minimum(self.q1(x).sigmoid(),self.q2(x).sigmoid())
                if self.spatial:
                    search_q[calculate]=torch.minimum(self.search_q1(x),self.search_q2(x))
                    q[calculate]+=float(cfg(self.config,"vnext_a_score_weight"))*a_score[calculate]
                values=q[calculate].masked_fill(~local_mask,-torch.inf)
                greedy=values.argmax(-1)
                gain=values.gather(-1,greedy[:,None]).squeeze(-1)-values[:,ACTOR]
                greedy=torch.where(gain>float(cfg(self.config,"vnext_c_margin")),greedy,torch.zeros_like(greedy))
                # Calibrated visual interaction is a bounded real experiment,
                # never a fabricated reward. It can repair an Actor that does
                # not propose attack despite a credible joint actionable cue.
                test=local_mask[:,8]&~interact[calculate]
                settle=local_mask[:,9]&(proposed[calculate]==10)
                greedy=torch.where(test,torch.full_like(greedy,8),greedy)
                greedy=torch.where(settle,torch.full_like(greedy,9),greedy)
                best=greedy.clone()
                source=torch.where(greedy>0,torch.ones_like(greedy),torch.zeros_like(greedy))
                source=torch.where(test,torch.full_like(source,2),source)
                source=torch.where(settle,torch.full_like(source,4),source)
                if training:
                    eps=float(cfg(self.config,"vnext_probe_probability"))
                    for j,row in enumerate(calculate.tolist()):
                        legal=local_mask[j].nonzero().squeeze(-1).cpu().numpy()
                        legal=legal[legal!=ACTOR]
                        least=np.array([],dtype=int)
                        if len(legal):
                            counts=self.coverage[torch.as_tensor(legal,device=device)].cpu().numpy()
                            least=legal[counts==counts.min()]
                            if self._rng.rand()<eps:
                                best[j]=int(self._rng.choice(least))
                                source[j]=5
                        p=(1-eps if len(least) else 1.)*float(best[j]==greedy[j])
                        if len(least) and int(best[j]) in least:p+=eps/len(least)
                        probability[row]=p
                        self.coverage[best[j]]+=1
                selected[calculate]=best;mask[calculate]=local_mask
                state["source"][calculate]=source
            state["active"][starts]=selected[starts];state["position"][starts]=0
            state["length"][starts]=lengths[starts,selected[starts]]
            state["commands"][starts]=commands[starts,selected[starts]]
            state["decision_id"][starts]+=1
            info["log_vnext_decision"][starts]=1
            info["log_vnext_horizon"][starts]=state["length"][starts].to(info["log_vnext_horizon"])
            info["log_vnext_plan_commands"][starts]=state["commands"][starts].to(info["log_vnext_plan_commands"])
        option=state["active"].clone()
        command=proposed.clone();extra=option>0
        command[extra]=state["commands"][rows[extra],state["position"][extra]]
        before=state["actuator"].clone()
        applied,next_actuator=apply_command_torch(command,before)
        state["actuator"],state["previous_control"]=next_actuator,applied
        state["position"]+=1
        finished=state["position"]>=state["length"]
        info["log_vnext_option_end"]=finished.float()
        info["log_vnext_option_duration"]=torch.where(finished,state["position"],torch.zeros_like(state["position"]))
        state["pending_interact"]=(option==8)&finished
        if self.spatial:
            cleared=(option==7)&finished
            state["clear_attempts"]+=cleared.long()
            state["clear_cooldown"][cleared]=int(cfg(self.config,"vnext_retry_cooldown"))
        state["cooldown"][state["pending_interact"]]=int(cfg(self.config,"vnext_retry_cooldown"))
        state["active"][finished]=-1
        category=torch.where(extra,torch.full_like(option,2),torch.zeros_like(option))
        category=torch.where(extra&(phase==SEARCH)&(option>=1)&(option<=(6 if self.spatial else 5)),torch.ones_like(category),category)
        category=torch.where(option==9,torch.full_like(category,3),category)
        history[rows,state["cursor"]]=category
        info.update(log_vnext_choice=option,log_vnext_option_id=state["decision_id"].clone(),
            log_vnext_context=context,log_vnext_mask=mask.float(),log_vnext_prediction=descriptor,
            log_vnext_q=q,log_vnext_task_q=task_q,log_vnext_search_q=search_q,
            log_vnext_a_score=a_score,log_vnext_score_supported=score_supported.float(),
            log_vnext_empirical_support=empirical_support.float(),
            log_vnext_source=state["source"].clone(),
            log_vnext_override=extra.float(),log_vnext_behavior_probability=probability,
            log_vnext_actuator_before=before,log_vnext_actuator_after=next_actuator,
            log_vnext_applied=applied,log_vnext_requested=command,
            log_vnext_search_used=(history==1).sum(-1).float(),
            log_vnext_extra_used=(history>0).sum(-1).float(),
            log_vnext_settle_used=(history==3).sum(-1).float(),
            log_vnext_actor_plan=actor_plan,
            log_vnext_recovery_exception=((option==11)&(emergency|recovery_lease)).float())
        state["cursor"]=(state["cursor"]+1)%window;state["age"]+=1
        return F.one_hot(command,12).to(actor_action.dtype),state,info


    def train_batch(self,feat,memory,ab,data,actor_sequence=None):
        records=[r for r in build_transitions(data,3) if r["version"]==self.version]
        if not records:return {"C/transitions":0.}
        if len(records)>int(cfg(self.config,"vnext_c_batch")):
            indices=self._rng.choice(len(records),int(cfg(self.config,"vnext_c_batch")),replace=False)
            records=[records[i] for i in indices]
        device=feat.device
        def tensor(key):
            return torch.as_tensor(np.stack([r[key] for r in records]),device=device)
        batches=tensor("batch").long();starts=tensor("start").long();ends=tensor("end").long()
        sf,nf=feat[batches,starts].detach(),feat[batches,ends].detach()
        sm=memory[batches,starts].detach() if memory is not None else None
        nm=memory[batches,ends].detach() if memory is not None else None
        context,ncontext=tensor("context").float(),tensor("next_context").float()
        def predict(features,mem,ctx,mask,obs_indices):
            output=features.new_zeros(len(features),N_OPTIONS,PREDICTION_DIM)
            score=features.new_zeros(len(features),N_OPTIONS)
            use=mask[:,1:].bool().any(-1)
            if not use.any():return output,score
            rows=use.nonzero().squeeze(-1)
            a=ctx[rows].new_zeros(len(rows),3)
            a[:,0]=ctx[rows,9]*10;a[:,1]=ctx[rows,8]*10;a[:,2]=ctx[rows,10]*70
            remaining=(ctx[rows,0]*self.max_steps()).round().long()
            commands,lengths=self.templates(a,remaining)
            if actor_sequence is not None:
                with torch.no_grad():
                    seq=actor_sequence(batches[rows],obs_indices[rows],k=3)
                    commands[:,ACTOR]=seq.argmax(-1) if seq.ndim==3 else seq
            elif "log_vnext_actor_plan" in data:
                plans=torch.as_tensor(data["log_vnext_actor_plan"],device=device)
                indices=(obs_indices[rows]+1).clamp(max=plans.shape[1]-1)
                commands[:,ACTOR]=plans[batches[rows],indices].long()
            controls,_=self.controls(commands,a,lengths)
            with torch.no_grad():
                pred=ab.predict_candidates(features[rows],mem[rows] if mem is not None else None,
                    controls,a,lengths=lengths,phase=ctx[rows,1:5].argmax(-1),
                    remaining=remaining,option_ids=torch.arange(N_OPTIONS,device=device))
            output[rows]=pred["descriptor"].detach()
            if self.spatial:score[rows]=self.prediction_score(pred,output[rows])[0]
            return output,score
        mask,nmask=tensor("mask").bool(),tensor("next_mask").bool()
        desc,_=predict(sf,sm,context,mask,starts)
        ndesc,next_score=predict(nf,nm,ncontext,nmask,ends)
        x=self.input(sf,sm,desc,context);nx=self.input(nf,nm,ndesc,ncontext)
        search_target=None
        with torch.no_grad():
            online=torch.minimum(self.q1(nx).sigmoid(),self.q2(nx).sigmoid())
            if self.spatial:
                clip=float(cfg(self.config,"vnext_search_value_clip"))
                online+=float(cfg(self.config,"vnext_search_value_weight"))*torch.minimum(
                    self.search_q1(nx),self.search_q2(nx)).clamp(-clip,clip)/clip
                online+=float(cfg(self.config,"vnext_a_score_weight"))*next_score
            na=online.masked_fill(~nmask,-torch.inf).argmax(-1)
            tv=torch.minimum(self.target_q1(nx).sigmoid(),self.target_q2(nx).sigmoid()).gather(-1,na[:,None]).squeeze(-1)
            target=torch.where(tensor("terminal").bool(),torch.zeros_like(tv),tv)
            target=torch.maximum(target,tensor("success").float()).clamp(0,1)
            if self.spatial:
                gamma=float(cfg(self.config,"vnext_c_discount"))
                rewards=factual_search_rewards(data,self.config)
                returns=[]
                for record in records:
                    # Replay row start+1 is the outgoing action's real result.
                    raw=rewards[record["batch"],record["start"]+1:record["end"]+1]
                    returns.append(sum((gamma**i)*float(v) for i,v in enumerate(raw)))
                realized=torch.as_tensor(returns,device=device)
                nt=torch.minimum(self.target_search_q1(nx),self.target_search_q2(nx))
                nt=nt.gather(-1,na[:,None]).squeeze(-1)
                duration=tensor("duration").float()
                continuation=torch.where(tensor("terminal").bool(),torch.zeros_like(nt),nt)
                search_target=(realized+(gamma**duration)*continuation).clamp(-clip,clip)
        action=tensor("action").long()
        with torch.enable_grad():
            for p in self._parameters_q:p.requires_grad_(True)
            q1=self.q1(x).gather(-1,action[:,None]).squeeze(-1)
            q2=self.q2(x).gather(-1,action[:,None]).squeeze(-1)
            loss=F.binary_cross_entropy_with_logits(q1,target)+F.binary_cross_entropy_with_logits(q2,target)
            task_loss=loss.detach()
            search_loss=loss*0
            if self.spatial:
                s1=self.search_q1(x).gather(-1,action[:,None]).squeeze(-1)
                s2=self.search_q2(x).gather(-1,action[:,None]).squeeze(-1)
                search_loss=F.smooth_l1_loss(s1,search_target)+F.smooth_l1_loss(s2,search_target)
                loss=loss+search_loss
            self._optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self._parameters_q,100.)
            finite=bool(torch.isfinite(loss)&torch.isfinite(norm))
            if finite:
                self._optimizer.step();self.updates.add_(1)
                tau=float(cfg(self.config,"vnext_c_tau"))
                with torch.no_grad():
                    pairs=[(self.q1,self.target_q1),(self.q2,self.target_q2)]
                    if self.spatial:pairs.extend([(self.search_q1,self.target_search_q1),(self.search_q2,self.target_search_q2)])
                    for net,tnet in pairs:
                        for p,tp in zip(net.parameters(),tnet.parameters()):tp.lerp_(p,tau)
            self._optimizer.zero_grad(set_to_none=True)
            for p in self._parameters_q:p.requires_grad_(False)
        metrics={"C/transitions":len(records),"C/value_loss":float(loss.detach()),
            "C/value_grad_norm":float(norm),"C/update_applied":float(finite),
            "C/updates":float(self.updates),"C/target_success_probability":float(target.mean()),
            "C/real_success_fraction":float(tensor("success").float().mean()),
            "C/real_duration":float(tensor("duration").float().mean())}
        if self.spatial:
            metrics.update({"C/task_value_loss":float(task_loss),"C/search_value_loss":float(search_loss.detach()),
                            "C/search_target_return":float(search_target.mean()),
                            "C/search_reward":float(realized.mean())})
        return metrics

