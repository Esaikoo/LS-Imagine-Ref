"""Task-isolated A/B: factual outcome heads and causal raw-keyframe memory.

No module in this file updates the task encoder, RSSM, actor or critic. Labels
and place IDs are consumed only by losses; online methods accept legal visuals,
detached features, and the agent's own control history.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VNEXT_VERSION = 1
PREDICTION_DIM = 12
CURRENT_EVENTS = ("visible", "center_target", "reachable", "center_interactable", "progress")
FUTURE_EVENTS = ("new", "first_discovery", "discovery", "revisit", "center_interactable", "progress", "blocked")
META_DIM = 8
DEFAULTS = dict(
    vnext_memory_enabled=True, vnext_memory_dim=128, vnext_slots=16,
    vnext_hidden=256, vnext_future_dim=128, vnext_bootstrap_heads=3,
    vnext_lr=0.0001, vnext_grad_clip=100., vnext_memory_save_every=8,
    vnext_snapshot_every=64, vnext_aux_horizon=8, vnext_aux_every=4,
    vnext_min_labels=128, vnext_min_positives=8, vnext_min_calibration=32,
    vnext_max_disagreement=0.15, vnext_match_threshold=0.94,
    vnext_dynamic_target=False,
)


def cfg(config, key):
    return getattr(config, key, DEFAULTS[key])


def _hash_int(value):
    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "little") % (2**31-1)


def _frames(image, heatmap):
    """Accept uint8 replay or normalized policy tensors, retain spatial layout."""
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(heatmap):
        heatmap = torch.as_tensor(heatmap, device=image.device)
    if heatmap.ndim == image.ndim - 1:
        heatmap = heatmap.unsqueeze(-1)
    if image.shape[-3:] != (64, 64, 3) or heatmap.shape[-3:] != (64, 64, 1):
        raise ValueError("vNext memory requires RGB64 and heatmap64")
    if image.dtype == torch.uint8:
        x = image.float() / 255.
    else:
        x = image.float()
    if heatmap.dtype == torch.uint8:
        h = heatmap.float() / 255.
    else:
        h = heatmap.float()
    return torch.cat((x, h), -1)


class TinyFrameEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,16,3,2,1), nn.SiLU(),
            nn.Conv2d(16,32,3,2,1), nn.SiLU(),
            nn.Conv2d(32,64,3,2,1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(64*4*4,dim), nn.LayerNorm(dim))
    def forward(self, frame):
        lead = frame.shape[:-3]
        x = frame.float().reshape(-1,64,64,4).permute(0,3,1,2)
        if frame.dtype == torch.uint8:x=x/255.
        return self.net(x).reshape(*lead,-1)


class SlotMemory(nn.Module):
    """Explicit finite memory; immutable collected references define replay."""
    def __init__(self, feature_dim, config):
        super().__init__()
        self.config = config
        self.slots = int(cfg(config,"vnext_slots"))
        self.dim = int(cfg(config,"vnext_memory_dim"))
        self.encoder = TinyFrameEncoder(self.dim)
        self.query = nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,self.dim))
        self.key = nn.Linear(self.dim+META_DIM,self.dim)
        self.value = nn.Sequential(nn.Linear(self.dim+META_DIM,self.dim),nn.SiLU())
        self.combine = nn.Sequential(nn.Linear(self.dim*2,self.dim),nn.LayerNorm(self.dim),nn.SiLU())
        self.event_head = nn.Linear(self.dim,3)  # visible/new/actual progress
        self.history_head = nn.Linear(self.dim,3)  # retained cue/progress/attempt
        self.return_head = nn.Linear(self.dim,1)
        self.register_buffer("place_positive",torch.zeros(()))
        self.register_buffer("place_negative",torch.zeros(()))

    def initial(self,batch,device):
        return dict(frames=torch.zeros(batch,self.slots,64,64,4,device=device),
            refs=torch.full((batch,self.slots),-1,dtype=torch.long,device=device),
            meta=torch.zeros(batch,self.slots,META_DIM,device=device),
            tick=torch.zeros(batch,dtype=torch.long,device=device))

    def read(self,feature,encoded,meta,valid):
        q = self.query(feature.detach().float())
        raw = torch.cat((encoded.float(),meta.float()),-1)
        key = self.key(raw)
        score = (F.normalize(q,dim=-1).unsqueeze(-2)*F.normalize(key,dim=-1)).sum(-1)*8
        w = score.masked_fill(~valid.bool(),-1.e4).softmax(-1)*valid.float()
        w = w/w.sum(-1,keepdim=True).clamp_min(1.e-6)
        pooled = (w.unsqueeze(-1)*self.value(raw)).sum(-2)
        result = self.combine(torch.cat((q,pooled),-1))
        return result*valid.any(-1,keepdim=True).float(), w

    @torch.no_grad()
    def online(self,feat,image,heatmap,state=None,is_first=None,previous_control=None,reward=None):
        batch,device = len(feat),feat.device
        fresh = self.initial(batch,device)
        state = fresh if state is None else {k:v.detach().clone() for k,v in state.items()}
        first = torch.zeros(batch,dtype=torch.bool,device=device) if is_first is None else is_first.reshape(batch).bool()
        for k,v in state.items():
            state[k] = torch.where(first.reshape(batch,*([1]*(v.ndim-1))),fresh[k],v)
        current = _frames(image,heatmap).to(device)
        valid = state["refs"] >= 0
        encoded = self.encoder(state["frames"])
        meta = state["meta"].clone()
        age = (state["tick"][:,None]-meta[...,1]).clamp_min(0)
        context,_ = self.read(feat,encoded,self.read_meta(meta,state["tick"]),valid)
        current_key = self.encoder(current)
        similarity = (F.normalize(current_key,dim=-1)[:,None]*F.normalize(encoded,dim=-1)).sum(-1)
        # Spatial RGB agreement plus motion support prevents untrained semantic
        # cosine from collapsing every pale scene into one keyframe.
        layout = F.adaptive_avg_pool2d(state["frames"][...,:3].reshape(-1,64,64,3).permute(0,3,1,2),(8,8)).reshape(batch,self.slots,-1)
        current_layout = F.adaptive_avg_pool2d(current[...,:3].permute(0,3,1,2),(8,8)).flatten(1)
        distance = (layout-current_layout[:,None]).abs().mean(-1)
        score = similarity.masked_fill(~valid,-2)
        nearest = score.argmax(-1)
        rows = torch.arange(batch,device=device)
        stationary = torch.ones(batch,dtype=torch.bool,device=device)
        if previous_control is not None:
            c = previous_control.float()
            stationary = (c[...,1:6].abs().sum(-1)==0)&(c[...,7:9].abs().sum(-1)==0)
        learned = bool((self.place_positive >= 32)&(self.place_negative >= 32))
        match = valid.any(-1)&(score[rows,nearest]>cfg(self.config,"vnext_match_threshold"))
        match &= distance[rows,nearest] < (.09 if learned else .025)
        match &= stationary | (learned & (distance[rows,nearest] < .04))
        cue = current[...,3].flatten(1).topk(32,dim=-1).values.mean(-1)
        novelty = 1-similarity.masked_fill(~valid,-1).max(-1).values
        free = (~valid).float().argmax(-1)
        retention = 2*state["meta"][...,3]+state["meta"][...,5]-.001*age
        if self.slots>1:
            retention[:,0] += 10000
        replace = retention.argmin(-1)
        slot = torch.where(match,nearest,torch.where(valid.all(-1),replace,free))
        allowed = (~valid.any(-1)) | (state["tick"] % int(cfg(self.config,"vnext_memory_save_every"))==0)
        allowed |= cue > state["meta"][...,3].max(-1).values+.1
        attempted = torch.zeros(batch,device=device)
        if previous_control is not None:
            attempted = ((previous_control[...,0]>0)|(previous_control[...,6]>0)).float()
        progress = torch.zeros(batch,device=device) if reward is None else (torch.as_tensor(reward,device=device).reshape(batch,-1).sum(-1)>0).float()
        progress *= (~first).float()
        attempted *= (~first).float()
        allowed |= (progress>0)|(attempted>0)
        write = match | allowed
        written = torch.full((batch,),-1,device=device,dtype=torch.long)
        for b in range(batch):
            if not bool(write[b]):
                continue
            s = int(slot[b])
            old = state["meta"][b,s].clone()
            # Metadata is history only: created,last,visits,cue,actual_progress,
            # uncertainty,attempts,view_motion. No privileged labels.
            state["frames"][b,s] = current[b]
            state["refs"][b,s] = state["tick"][b]
            state["meta"][b,s] = torch.stack((
                torch.where(match[b],old[0],state["tick"][b].float()),
                state["tick"][b].float(),
                torch.where(match[b],old[2]+1,old[2]*0+1),
                cue[b],torch.maximum(old[4] if match[b] else old[4]*0,progress[b]),
                novelty[b].clamp(0,2),
                (old[6] if match[b] else old[6]*0)+attempted[b],
                stationary[b].float()))
            written[b] = s
        snap = (state["tick"] % int(cfg(self.config,"vnext_snapshot_every")) == 0).float()
        state["tick"] += 1
        return context,state,dict(log_vnext_memory_refs=state["refs"].clone(),
            log_vnext_memory_meta=state["meta"].clone(),
            log_vnext_memory_write=written,log_vnext_memory_snapshot=snap,
            log_vnext_memory_used=valid.any(-1).float(),
            log_vnext_memory_slots=(state["refs"]>=0).sum(-1).float(),
            log_vnext_memory_tick=state["tick"].clone()-1)

    def read_meta(self,meta,tick):
        """Canonical causal read metadata; raw logged values remain immutable."""
        result=meta.float().clone()
        age=(tick[...,None]-result[...,1]).clamp_min(0)
        created=(tick[...,None]-result[...,0]).clamp_min(0)
        rate=.01 if (cfg(self.config,"vnext_dynamic_target") or getattr(self.config,"target_item","")=="wool") else .001
        result[...,3:5]*=torch.exp(-rate*age)[...,None]
        result[...,0]=torch.log1p(created)/8
        result[...,1]=torch.log1p(age)/8
        result[...,2]=torch.log1p(result[...,2].clamp_min(0))/5
        result[...,6]=torch.log1p(result[...,6].clamp_min(0))/5
        return result

    def batch(self,features,data):
        device = features.device
        frames = torch.as_tensor(data["vnext_memory_frames"],device=device)
        encoded = self.encoder(frames)
        indices = torch.as_tensor(data["vnext_memory_indices"],device=device).long()
        valid = torch.as_tensor(data["vnext_memory_valid"],device=device).bool()
        b = torch.arange(len(features),device=device)[:,None,None]
        selected = encoded[b,indices.clamp_min(0)]
        meta = torch.as_tensor(data["vnext_memory_meta"],device=device).float()
        tick = torch.as_tensor(data.get("vnext_observation_index",np.zeros(features.shape[:2])),device=device).float()
        context,weights = self.read(features,selected,self.read_meta(meta,tick),valid)
        return context,dict(encoded=encoded,selected=selected,valid=valid,weights=weights)

    def loss(self,features,data,context,details,train_mask):
        device = features.device
        zero = context.sum()*0
        labels,known = _labels(data,("visible","new","progress"),device)
        mask = known & train_mask[...,None]
        event_logits = self.event_head(context)
        loss = _masked_bce(event_logits,labels,mask)
        events = torch.as_tensor(data["vnext_memory_targets"],device=device).float()
        valid_events = torch.as_tensor(data["vnext_memory_target_valid"],device=device).bool()
        key_logits = self.event_head(details["encoded"])
        key_cal = torch.as_tensor(data["vnext_memory_calibration"],device=device).bool()
        keymask = valid_events & (~key_cal)[...,None]
        key_loss = _masked_bce(key_logits,events,keymask)
        frame_place = torch.as_tensor(data["vnext_memory_place"],device=device).float()
        indices = torch.as_tensor(data["vnext_memory_indices"],device=device).long()
        b = torch.arange(len(features),device=device)[:,None,None]
        slot_place = frame_place[b,indices.clamp_min(0)]
        current_place = torch.as_tensor(data.get("search_place_id",np.zeros(features.shape[:2])),device=device).float()
        place_valid = torch.as_tensor(data.get("search_place_valid",np.zeros(features.shape[:2])),device=device).bool()
        current_frames = _frames(torch.as_tensor(data["image"],device=device),torch.as_tensor(data["heatmap"],device=device))
        current_encoded = self.encoder(current_frames)
        cosine = (F.normalize(current_encoded,dim=-1)[...,None,:]*F.normalize(details["selected"],dim=-1)).sum(-1)
        pk = details["valid"] & (slot_place>0) & place_valid[...,None] & train_mask[...,None]
        py = (current_place[...,None]==slot_place).float()
        place_loss = _masked_bce((cosine-.5)*8,py,pk)
        # Retaining history target evidence is supervised separately from
        # recognizing the current scene.
        # Gather retained raw-frame labels, never future/current observations.
        slot_y = events[b,indices.clamp_min(0)]
        slot_k = valid_events[b,indices.clamp_min(0)] & details["valid"][...,None]
        # Explicitly retain cue, actual progress, and prior interaction attempts.
        # Attempt history comes from own recorded controls, never ray truth.
        retained=(slot_y*slot_k).amax(-2)
        hy=torch.stack((retained[...,0],retained[...,2],
            ((torch.as_tensor(data["vnext_memory_meta"],device=device)[...,6]>0)&details["valid"]).any(-1).float()),-1)
        known_history=slot_k.any(-2)
        hk=torch.stack((known_history[...,0],known_history[...,2],details["valid"].any(-1)),-1)&train_mask[...,None]
        history_loss = _masked_bce(self.history_head(context),hy,hk)
        ry,rk = _labels(data,("revisit",),device)
        return_loss = _masked_bce(self.return_head(context),ry,rk&train_mask[...,None])
        total = loss+key_loss+.2*place_loss+.25*history_loss+.25*return_loss
        counts = torch.stack(((py*pk).sum(),((1-py)*pk).sum())).detach()
        return total,dict(**{"B/current_evidence_loss":loss.detach(),"B/keyframe_evidence_loss":key_loss.detach(),
            "B/place_loss":place_loss.detach(),"B/history_loss":history_loss.detach(),"B/revisit_loss":return_loss.detach(),
            "B/total_loss":total.detach()}),counts


def _labels(data,names,device):
    shape = np.shape(data["is_first"])
    values,valids = [],[]
    version = torch.as_tensor(data.get("search_label_version",np.zeros(shape)),device=device)==3
    for name in names:
        value = torch.as_tensor(data.get("search_"+name,np.zeros(shape)),device=device).float()
        known = torch.as_tensor(data.get("search_"+name+"_valid",np.zeros(shape)),device=device).bool()
        known &= version & torch.isfinite(value) & (value>=0) & (value<=1)
        values.append(value.nan_to_num().clamp(0,1))
        valids.append(known)
    return torch.stack(values,-1),torch.stack(valids,-1)


def _masked_bce(logits,target,mask):
    error = F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction="none")
    return (error*mask).sum()/mask.sum().clamp_min(1)


class OutcomeHeads(nn.Module):
    def __init__(self,feature_dim,config):
        super().__init__()
        self.config = config
        dim = int(cfg(config,"vnext_memory_dim"))
        hidden = int(cfg(config,"vnext_hidden"))
        self.heads = int(cfg(config,"vnext_bootstrap_heads"))
        self.current_trunk = nn.Sequential(nn.LayerNorm(feature_dim+dim),nn.Linear(feature_dim+dim,hidden),nn.SiLU())
        self.current_heads = nn.ModuleList([nn.Linear(hidden,len(CURRENT_EVENTS)) for _ in range(self.heads)])
        self.trunk = nn.Sequential(nn.Linear(hidden+8*9+3+4+2,hidden),nn.LayerNorm(hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.SiLU())
        self.events = nn.ModuleList([nn.Linear(hidden,len(FUTURE_EVENTS)) for _ in range(self.heads)])
        future = int(cfg(config,"vnext_future_dim"))
        self.future = nn.ModuleList([nn.Linear(hidden,future) for _ in range(self.heads)])
        generator = torch.Generator().manual_seed(9187+feature_dim)
        projection = torch.randn(feature_dim,future,generator=generator)
        self.register_buffer("projection",F.normalize(projection,dim=0))
        self.register_buffer("current_counts",torch.zeros(5))
        self.register_buffer("current_positives",torch.zeros(5))
        self.register_buffer("current_cal_counts",torch.zeros(5))
        self.register_buffer("current_cal_positives",torch.zeros(5))
        self.register_buffer("current_cal_errors",torch.zeros(5))
        self.register_buffer("support_counts",torch.zeros(9,12,4,7))
        self.register_buffer("support_positives",torch.zeros(9,12,4,7))
        self.register_buffer("cal_counts",torch.zeros(9,12,4,7))
        self.register_buffer("cal_positives",torch.zeros(9,12,4,7))
        self.register_buffer("cal_errors",torch.zeros(9,12,4,7))
        self.register_buffer("cal_identity",torch.full((9,65536),-1,dtype=torch.long))
        self.register_buffer("cal_valid",torch.zeros(9,65536,7,dtype=torch.bool))
        self.register_buffer("cal_label",torch.zeros(9,65536,7))
        self.register_buffer("cal_error",torch.zeros(9,65536,7))
        self.register_buffer("cal_option",torch.full((9,65536),-1,dtype=torch.long))
        self.register_buffer("cal_phase",torch.full((9,65536),-1,dtype=torch.long))
        self.register_buffer("updates",torch.zeros((),dtype=torch.long))

    def current_logits(self,feat,memory):
        x = self.current_trunk(torch.cat((feat.detach().float(),memory.float()),-1))
        return torch.stack([head(x) for head in self.current_heads],-2),x

    def ready_current(self):
        n = self.current_cal_counts.clamp_min(1)
        rate = self.current_cal_positives/n
        baseline = (rate*(1-rate)).clamp_min(.001)
        return ((self.current_counts>=cfg(self.config,"vnext_min_labels"))&
            (self.current_positives>=cfg(self.config,"vnext_min_positives"))&
            ((self.current_counts-self.current_positives)>=cfg(self.config,"vnext_min_positives"))&
            (self.current_cal_counts>=cfg(self.config,"vnext_min_calibration"))&
            (self.current_cal_positives>=min(4,cfg(self.config,"vnext_min_positives")))&
            ((self.current_cal_counts-self.current_cal_positives)>=min(4,cfg(self.config,"vnext_min_positives")))&
            (self.current_cal_errors/n <= baseline))

    def forward(self,feat,memory,controls,actuator,lengths,phase,remaining):
        logits,x = self.current_logits(feat,memory)
        lead = controls.shape[:-2]
        k = controls.shape[-2]
        if not 1<=k<=8 or controls.shape[-1]!=9:
            raise ValueError("Expected one to eight actual nine-component controls")
        seq = controls.float().clone()
        seq[...,7:9] /= 10.
        seq = F.pad(seq,(0,0,0,8-k)).flatten(-2)
        act = actuator.float()/actuator.new_tensor([10.,10.,90.])
        phase_vec = F.one_hot(phase.long().clamp(0,3),4).float()
        meta = torch.stack((lengths.float()/8,remaining.float()/1000),-1)
        hidden = self.trunk(torch.cat((x,seq,act,phase_vec,meta),-1))
        return dict(logits=torch.stack([head(hidden) for head in self.events],-2),
            future=torch.stack([head(hidden) for head in self.future],-2),current_logits=logits)


class SearchVNext(nn.Module):
    model_version = VNEXT_VERSION
    def __init__(self,feature_dim,config):
        super().__init__()
        self.config = config
        self.memory_dim = int(cfg(config,"vnext_memory_dim"))
        self.b = SlotMemory(feature_dim,config)
        self.a = OutcomeHeads(feature_dim,config)
        self._a_opt = torch.optim.Adam(self.a.parameters(),lr=cfg(config,"vnext_lr"))
        self._b_opt = torch.optim.Adam(self.b.parameters(),lr=cfg(config,"vnext_lr"))

    def zero_memory(self,feat):
        return feat.new_zeros(*feat.shape[:-1],self.memory_dim)

    def memory_batch(self,features,data):
        if not cfg(self.config,"vnext_memory_enabled"):
            return self.zero_memory(features)
        return self.b.batch(features.detach(),data)[0]

    @torch.no_grad()
    def online_memory(self,feat,image,heatmap,state=None,is_first=None,previous_control=None,reward=None):
        context,state,info = self.b.online(feat,image,heatmap,state,is_first,previous_control,reward)
        if not cfg(self.config,"vnext_memory_enabled"):
            context.zero_()
        return context,state,info

    @torch.no_grad()
    def online(self,feat,image,heatmap,memory_state=None,is_first=None,previous_control=None,reward=None):
        memory,state,info = self.online_memory(feat,image,heatmap,memory_state,is_first,previous_control,reward)
        return self.current(feat,memory),memory,state,info

    @torch.no_grad()
    def current(self,feat,memory=None):
        memory = self.zero_memory(feat) if memory is None else memory
        logits,_ = self.a.current_logits(feat,memory)
        probs = logits.sigmoid()
        mean = probs.mean(-2)
        ready = self.a.ready_current().expand(*mean.shape[:-1],5)
        result = {name:mean[...,i] for i,name in enumerate(CURRENT_EVENTS)}
        result.update(probs=mean,disagreement=probs.std(-2,unbiased=False),
            ready={name:ready[...,i] for i,name in enumerate(CURRENT_EVENTS)},current_ready=ready)
        return result

    @torch.no_grad()
    def predict(self,feat,memory,controls,start_actuator,lengths=None,phase=None,remaining=None,option_ids=None):
        batch,num,k,_ = controls.shape
        device = feat.device
        memory = self.zero_memory(feat) if memory is None else memory
        def expand(value,tail=()):
            if value.shape == (batch,*tail):
                value = value[:,None].expand(batch,num,*tail)
            return value
        lengths = torch.full((batch,num),k,device=device,dtype=torch.long) if lengths is None else expand(lengths)
        phase = torch.zeros(batch,num,device=device,dtype=torch.long) if phase is None else expand(phase)
        remaining = torch.full((batch,num),1000,device=device) if remaining is None else expand(remaining)
        actuator = expand(start_actuator,(3,))
        features = feat[:,None].expand(-1,num,-1)
        memories = memory[:,None].expand(-1,num,-1)
        output = self.a(features,memories,controls,actuator,lengths,phase,remaining)
        draws = output["logits"].sigmoid()
        probs = draws.mean(-2)
        uncertainty = draws.std(-2,unbiased=False)
        ids = torch.arange(num,device=device)[None].expand(batch,-1) if option_ids is None else torch.as_tensor(option_ids,device=device)
        if ids.ndim==1:
            ids=ids[None].expand(batch,-1)
        h=lengths.long().clamp(1,8); ids=ids.long().clamp(0,11); ph=phase.long().clamp(0,3)
        n=self.a.support_counts[h,ids,ph]; pos=self.a.support_positives[h,ids,ph]
        cn=self.a.cal_counts[h,ids,ph]; cp=self.a.cal_positives[h,ids,ph]
        rate=cp/cn.clamp_min(1)
        supported=(n>=cfg(self.config,"vnext_min_labels"))&(pos>=cfg(self.config,"vnext_min_positives"))
        supported &= (n-pos)>=cfg(self.config,"vnext_min_positives")
        supported &= (cp>=min(4,cfg(self.config,"vnext_min_positives")))&((cn-cp)>=min(4,cfg(self.config,"vnext_min_positives")))
        supported &= (cn>=cfg(self.config,"vnext_min_calibration"))&(self.a.cal_errors[h,ids,ph]/cn.clamp_min(1)<=(rate*(1-rate)).clamp_min(.001))
        supported &= uncertainty<=cfg(self.config,"vnext_max_disagreement")
        support=supported[...,[0,4,5]].any(-1)
        cv=output["current_logits"].sigmoid().mean(-2)[...,0]
        descriptor=torch.cat((probs,uncertainty.mean(-1,keepdim=True),supported.float().mean(-1,keepdim=True),
            lengths.float()[...,None]/8,cv[...,None],uncertainty.max(-1,keepdim=True).values),-1)
        return dict(probs=probs,events=probs,uncertainty=uncertainty,disagreement=uncertainty,
            supported=supported,support=support,descriptor=descriptor,future=output["future"].mean(-2))

    predict_candidates = predict

    @staticmethod
    def matched_option_windows(data,act,controls,h):
        """Eligibility evidence requires an actually executed saved plan.

        Factual A losses still learn all real sequences. A window that changes
        macro/phase or deviates from an Actor imagined plan cannot certify that
        candidate's counterfactual support.
        """
        b,t,_=act.shape;n=t-h;device=act.device
        good=torch.zeros(b,n,device=device,dtype=torch.bool)
        if h>3 or "log_vnext_decision" not in data or "log_vnext_plan_commands" not in data:
            return good
        def value(key,tail=(),default=0):
            return torch.as_tensor(data.get(key,np.full((b,t,*tail),default)),device=device)
        start=value("log_vnext_decision")[:,1:n+1]>.5
        version=value("log_vnext_version")
        horizon=value("log_vnext_horizon")[:,1:n+1]
        ids=value("log_vnext_option_id")
        choices=value("log_vnext_choice")
        phases=value("log_vnext_phase")
        plan=value("log_vnext_plan_commands",(3,))[:,1:n+1,:h].long()
        if "log_vnext_actor_plan" in data:
            actor_plan=value("log_vnext_actor_plan",(3,))[:,1:n+1,:h].long()
            plan=torch.where((choices[:,1:n+1]==0)[...,None],actor_plan,plan)
        requested=value("log_requested_action" if "log_requested_action" in data else "log_vnext_requested").long()
        good=start&(horizon>=h)
        for j in range(h):
            r=slice(j+1,j+n+1)
            good&=(version[:,r]==1)&(ids[:,r]==ids[:,1:n+1])
            good&=(choices[:,r]==choices[:,1:n+1])&(phases[:,r]==phases[:,1:n+1])
            good&=requested[:,r]==plan[...,j]
        from search_actuator import rollout_commands_torch
        expected,end=rollout_commands_torch(plan,act[:,:n])
        factual=torch.stack([controls[:,j+1:j+n+1] for j in range(h)],-2)
        good&=(expected-factual).abs().amax((-1,-2))<1.e-5
        good&=(end-act[:,h:]).abs().amax(-1)<1.e-5
        return good

    def training_loss(self,features,data):
        features=features.detach().float(); device=features.device
        b,t,_=features.shape
        calibration=torch.as_tensor(data.get("vnext_calibration",np.zeros((b,t))),device=device).bool()
        window_valid=torch.as_tensor(data.get("vnext_window_valid",np.ones((b,t))),device=device).bool()
        calibration &= window_valid
        train=(~calibration)&window_valid
        memory,details=self.b.batch(features,data)
        if not cfg(self.config,"vnext_memory_enabled"):
            memory=memory*0
        current_y,current_k=_labels(data,CURRENT_EVENTS,device)
        current_logits,_=self.a.current_logits(features,memory)
        # Independent heads use deterministic frame-identity bootstrap masks;
        # this preserves natural sampling rates without positive oversampling.
        frameid=torch.as_tensor(data.get("vnext_frame_id",np.zeros((b,t))),device=device).long()
        draws=[]; loss=features.sum()*0
        for head in range(self.a.heads):
            boot=((frameid*5+frameid//(head+1)*3+head*71)%7)!=0
            loss=loss+_masked_bce(current_logits[...,head,:],current_y,current_k&train[...,None]&boot[...,None])/self.a.heads
        metrics={"A/current_loss":loss.detach()}
        records=[("current",current_logits.sigmoid().mean(-2).detach(),current_y,current_k,train,calibration,frameid)]
        horizons=[h for h in (1,2,3) if h<t]
        if int(self.a.updates)%int(cfg(self.config,"vnext_aux_every"))==0 and t>int(cfg(self.config,"vnext_aux_horizon")):
            horizons.append(int(cfg(self.config,"vnext_aux_horizon")))
        controls=torch.as_tensor(data.get("log_applied_control",np.zeros((b,t,9))),device=device).float()
        act=torch.as_tensor(data.get("log_actuator_after",np.zeros((b,t,3))),device=device).float()
        phase=torch.as_tensor(data.get("log_vnext_phase",np.zeros((b,t))),device=device).long()
        option=torch.as_tensor(data.get("log_vnext_choice",np.zeros((b,t))),device=device).long()
        first=torch.as_tensor(data["is_first"],device=device).bool()
        last=torch.as_tensor(data.get("is_last",np.zeros((b,t))),device=device).bool()
        terminal=torch.as_tensor(data.get("is_terminal",np.zeros((b,t))),device=device).bool()
        fy,fk=_labels(data,FUTURE_EVENTS,device)
        for h in horizons:
            n=t-h
            seq=torch.stack([controls[:,j+1:j+n+1] for j in range(h)],-2)
            valid=~torch.stack([first[:,j+1:j+n+1]|terminal[:,j:j+n]|last[:,j:j+n] for j in range(h)]).any(0)
            control_known=torch.as_tensor(data.get("log_actuator_valid",np.ones((b,t)) if "log_applied_control" in data else np.zeros((b,t))),device=device).bool()
            valid &= torch.stack([control_known[:,j+1:j+n+1] for j in range(h)]).all(0)
            env_error=torch.as_tensor(data.get("log_env_error",np.zeros((b,t))),device=device).bool()
            valid &= ~torch.stack([env_error[:,j+1:j+n+1] for j in range(h)]).any(0)
            windows=torch.stack([fy[:,j+1:j+n+1] for j in range(h)],-2)
            known=torch.stack([fk[:,j+1:j+n+1] for j in range(h)],-2)
            target=windows.max(-2).values
            target[...,4]=fy[:,h:,4]  # endpoint interaction affordance
            known_window=known.all(-2)|((windows>.5)&known).any(-2)
            known_window[...,4]=fk[:,h:,4]
            mask=known_window&valid[...,None]
            lengths=torch.full((b,n),h,device=device)
            remain=torch.as_tensor(data.get("log_vnext_remaining",np.full((b,t),1000)),device=device).float()[:,1:n+1]
            pred=self.a(features[:,:n],memory[:,:n],seq,act[:,:n],lengths,phase[:,1:n+1],remain)
            hloss=features.sum()*0
            for head in range(self.a.heads):
                boot=((frameid[:,:n]*5+frameid[:,:n]//(head+1)*3+head*71)%7)!=0
                mk=mask&train[:,:n,None]&boot[...,None]
                hloss+=_masked_bce(pred["logits"][...,head,:],target,mk)/self.a.heads
                future_target=F.layer_norm(features[:,h:],(features.shape[-1],))@self.a.projection
                ferr=F.smooth_l1_loss(pred["future"][...,head,:],future_target,reduction="none").mean(-1)
                vm=valid&train[:,:n]&boot
                hloss+=(ferr*vm).sum()/vm.sum().clamp_min(1)/self.a.heads
            loss=loss+hloss/len(horizons)
            metrics["A/H%d/prediction_loss"%h]=hloss.detach()
            verified=valid&self.matched_option_windows(data,act,controls,h)
            records.append((h,pred["logits"].sigmoid().mean(-2).detach(),target,mask&verified[...,None],
                            train[:,:n],calibration[:,:n],frameid[:,:n],option[:,1:n+1],phase[:,1:n+1]))
        b_loss,bmetrics,bcounts=self.b.loss(features,data,memory,details,train)
        if not cfg(self.config,"vnext_memory_enabled"):
            b_loss=b_loss*0
            bcounts=bcounts*0
        metrics.update(bmetrics);metrics["A/prediction_loss"]=loss.detach()
        return loss+b_loss,metrics,records,bcounts,memory

    @torch.no_grad()
    def _record(self,records,bcounts):
        self.b.place_positive+=bcounts[0];self.b.place_negative+=bcounts[1]
        for record in records:
            kind,p,y,k,train,cal,ids,*rest=record
            binid=0 if kind=="current" else int(kind)
            n=p.shape[-1]
            if kind=="current":
                self.a.current_counts+=(k&train[...,None]).sum((0,1))
                self.a.current_positives+=(y*(k&train[...,None])).sum((0,1))
            else:
                option,phase=rest
                bucket=(option.long().clamp(0,11)*4+phase.long().clamp(0,3)).flatten()
                weights=(k&train[...,None]).float().reshape(-1,n)
                self.a.support_counts[kind].view(48,n).index_add_(0,bucket,weights)
                self.a.support_positives[kind].view(48,n).index_add_(0,bucket,y.reshape(-1,n)*weights)
            # Independent reservoir for each true horizon. Collision replacement
            # removes the previous owner's statistics before adding the current
            # real held-out identity; sampled identities refresh their errors.
            hashid=(ids*31+binid*971)%self.a.cal_identity.shape[-1]
            flat=hashid.flatten()
            active=(cal&k.any(-1)).flatten().nonzero().squeeze(-1)
            if not len(active):continue
            positions=active[torch.argsort(flat[active],stable=True)]
            keep=torch.ones(len(positions),dtype=torch.bool,device=p.device)
            keep[1:]=flat[positions[1:]]!=flat[positions[:-1]]
            positions=positions[keep]
            slots=flat[positions];identity=ids.flatten()[positions]
            newk=k.reshape(-1,n)[positions]
            newy=y.reshape(-1,n)[positions]
            newerror=(p.reshape(-1,n)[positions]-newy).square()
            oldk=self.a.cal_valid[binid,slots,:n].clone()
            oldy=self.a.cal_label[binid,slots,:n].clone()
            olderror=self.a.cal_error[binid,slots,:n].clone()
            if kind=="current":
                self.a.current_cal_counts+=(newk.float()-oldk.float()).sum(0)
                self.a.current_cal_positives+=(newy*newk-oldy*oldk).sum(0)
                self.a.current_cal_errors+=(newerror*newk-olderror*oldk).sum(0)
            else:
                oldo=self.a.cal_option[binid,slots].clamp(0,11)
                oldp=self.a.cal_phase[binid,slots].clamp(0,3)
                old_bucket=oldo*4+oldp
                newo=option.flatten()[positions].long().clamp(0,11)
                newp=phase.flatten()[positions].long().clamp(0,3)
                new_bucket=newo*4+newp
                for target,old,new in (
                    (self.a.cal_counts[kind],oldk.float(),newk.float()),
                    (self.a.cal_positives[kind],oldy*oldk,newy*newk),
                    (self.a.cal_errors[kind],olderror*oldk,newerror*newk)):
                    target.view(48,n).index_add_(0,old_bucket,-old)
                    target.view(48,n).index_add_(0,new_bucket,new)
                self.a.cal_option[binid,slots]=newo
                self.a.cal_phase[binid,slots]=newp
            self.a.cal_identity[binid,slots]=identity
            self.a.cal_valid[binid,slots,:n]=newk
            self.a.cal_label[binid,slots,:n]=newy
            self.a.cal_error[binid,slots,:n]=newerror

    def train_step(self,features,data,step=None):
        prior=[p.requires_grad for p in self.parameters()]
        try:
            self.requires_grad_(True)
            self._a_opt.zero_grad(set_to_none=True);self._b_opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss,metrics,records,bcounts,_=self.training_loss(features,data)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite vNext A/B loss")
                calibration=np.asarray(data.get("vnext_calibration",np.zeros(features.shape[:2]))).astype(bool)
                window=np.asarray(data.get("vnext_window_valid",np.ones(features.shape[:2]))).astype(bool)
                has_train=bool(((~calibration)&window).any())
                if not has_train:
                    self._record(records,bcounts*0)
                    metrics.update({"A/grad_norm":0.,"B/grad_norm":0.,
                        "A/update_applied":0.,"B/update_applied":0.})
                    return metrics
                loss.backward()
            a_norm=torch.nn.utils.clip_grad_norm_(self.a.parameters(),cfg(self.config,"vnext_grad_clip"))
            b_norm=torch.nn.utils.clip_grad_norm_(self.b.parameters(),cfg(self.config,"vnext_grad_clip"))
            if not bool(torch.isfinite(a_norm)&torch.isfinite(b_norm)):
                raise FloatingPointError("Nonfinite vNext A/B gradient")
            self._a_opt.step()
            if cfg(self.config,"vnext_memory_enabled"):self._b_opt.step()
            self._record(records,bcounts);self.a.updates+=1
            metrics.update({"A/grad_norm":a_norm.detach(),"B/grad_norm":b_norm.detach(),
                "A/update_applied":1.,"B/update_applied":float(cfg(self.config,"vnext_memory_enabled")),})
            return metrics
        finally:
            for p,value in zip(self.parameters(),prior):
                p.requires_grad_(value)


def sample_episodes_vnext(episodes,length,slots=16,seed=0,**unused):
    """Full real stitched windows, immutable causal slots, deduplicated pixels.

    Row t's collected memory is the state written at observation t-1. Every
    reference is local to that real episode; synthetic RSSM window resets never
    reset the historical bank. No padded fake frames enter the base model.
    """
    rng=np.random.RandomState(seed)
    schema_templates={}
    while True:
        candidates=[(key,ep) for key,ep in episodes.items() if len(ep["reward"])>=2]
        if not candidates:
            raise ValueError("No vNext replay transitions")
        # Match the original LS sampler's episode-length distribution.
        probs=np.array([len(ep["reward"]) for _,ep in candidates],float);probs/=probs.sum()
        segments=[];size=0
        while size<length:
            key,ep=candidates[rng.choice(len(candidates),p=probs)]
            start=int(rng.randint(0,len(ep["reward"])-1)) if not segments else 0
            n=min(length-size,len(ep["reward"])-start)
            segments.append((str(key),ep,start,n))
            size+=n
        out={}
        from search_vnext_control import FIELDS
        # Prefill and learned episodes coexist. A fixed schema is essential:
        # batching keys from only the first sampled window drops C feedback or
        # raises KeyError when another window comes from random prefill.
        trace_shapes=dict(FIELDS)
        trace_shapes.update(log_vnext_memory_refs=(slots,),log_vnext_memory_meta=(slots,META_DIM),
            log_vnext_memory_tick=(),log_vnext_memory_write=(),log_vnext_memory_snapshot=(),
            log_vnext_memory_used=(),log_vnext_memory_slots=(),
            log_vnext_world_group=(),log_search_override=(),log_search_decision=())
        names=set().union(*(set(ep) for _,ep in candidates))|set(trace_shapes)
        negative={"log_vnext_memory_refs","log_vnext_memory_tick","log_vnext_memory_write"}
        for name in names:
            if name.startswith("vnext_memory_"):
                continue
            example=schema_templates.get(name)
            if example is None:
                if name in trace_shapes:
                    example=np.zeros(trace_shapes[name],np.float32)
                else:
                    for _,ep in candidates:
                        if name in ep:
                            example=next((v for v in ep[name] if v is not None),None)
                            if example is not None:break
                if example is None:
                    if name=="zoomed_image":example=np.zeros((64,64,3),np.uint8)
                    elif name=="heatmap_on_zoomed":example=np.zeros((64,64),np.uint8)
                    else:example=np.float32(0)
                schema_templates[name]=np.zeros_like(example)
            fill=np.full_like(example,-1) if name in negative else np.zeros_like(example)
            pieces=[]
            for _,ep,start,n in segments:
                vals=ep[name][start:start+n] if name in ep else [None]*n
                pieces.extend(fill if v is None else v for v in vals)
            out[name]=np.asarray(pieces).copy()
        refs_out=np.full((length,slots),-1,np.int64)
        meta=np.zeros((length,slots,META_DIM),np.float32)
        realfirst=np.zeros(length,bool);first=np.zeros(length,bool)
        calibration=np.zeros(length,np.float32);frameids=np.zeros(length,np.int64);observation_indices=np.zeros(length,np.int64)
        pool={};sources=[];row=0
        for key,ep,start,n in segments:
            total=len(ep["reward"])
            group=ep.get("log_vnext_world_group")
            groupkey=str(np.asarray(group).reshape(-1)[0]) if group is not None else key
            holdout=_hash_int(groupkey)%5==0
            refs=ep.get("log_vnext_memory_refs");metas=ep.get("log_vnext_memory_meta")
            first[row]=True
            for i in range(start,start+n):
                realfirst[row]=bool(ep["is_first"][i])
                frameids[row]=_hash_int(key+":"+str(i));calibration[row]=holdout;observation_indices[row]=i
                if not realfirst[row] and refs is not None and metas is not None:
                    rr=np.asarray(refs[i],np.int64)
                    if rr.shape!=(slots,):
                        raise ValueError("vNext slot count differs from replay")
                    if np.any(rr[rr>=0]>=i):
                        raise ValueError("vNext replay has a current/future keyframe")
                    meta[row]=np.asarray(metas[i],np.float32)
                    for si,ref in enumerate(rr):
                        if ref<0:continue
                        identity=(key,int(ref))
                        if identity not in pool:
                            pool[identity]=len(sources);sources.append((ep,int(ref),holdout))
                        refs_out[row,si]=pool[identity]
                row+=1
        if len(sources)>slots+length:
            raise ValueError("Invalid memory write rate or cross-episode references")
        capacity=slots+length
        frames=np.zeros((capacity,64,64,4),np.uint8)
        targets=np.zeros((capacity,3),np.float32);target_valid=np.zeros_like(targets)
        places=np.zeros(capacity,np.float32)
        frame_calibration=np.ones(capacity,np.float32)
        for j,(ep,ref,holdout) in enumerate(sources):
            frame_calibration[j]=holdout
            total=len(ep["reward"])
            frames[j]=np.concatenate((np.asarray(ep["image"][ref],np.uint8),
                np.asarray(ep["heatmap"][ref],np.uint8).reshape(64,64,1)),-1)
            version=np.asarray(ep.get("search_label_version",np.zeros(total)))[ref]
            for q,name in enumerate(("visible","new","progress")):
                targets[j,q]=np.asarray(ep.get("search_"+name,np.zeros(total)))[ref]
                target_valid[j,q]=bool(np.asarray(ep.get("search_"+name+"_valid",np.zeros(total)))[ref]) and version==3
            if bool(np.asarray(ep.get("search_place_valid",np.zeros(total)))[ref]):
                places[j]=np.asarray(ep["search_place_id"])[ref]
        # Original base-model semantics: each stitched segment starts at a real
        # episode boundary except the first synthetic training-window reset.
        out["is_first"]=np.asarray(out["is_first"],bool)|first
        out.update(vnext_memory_frames=frames,vnext_memory_indices=refs_out,
            vnext_memory_valid=(refs_out>=0).astype(np.float32),vnext_memory_meta=meta,
            vnext_memory_targets=targets,vnext_memory_target_valid=target_valid,
            vnext_memory_place=places,vnext_memory_calibration=frame_calibration,vnext_real_first=realfirst,
            vnext_window_valid=np.ones(length,np.float32),vnext_frame_id=frameids,vnext_observation_index=observation_indices,
            vnext_calibration=calibration)
        yield out


def check_vnext_checkpoint(checkpoint,enabled):
    state=checkpoint.get("agent_state_dict",{})
    has=any(key.startswith("_vnext.") for key in state)
    if bool(enabled)!=has:
        raise ValueError("vNext architecture differs; old ABC weights cannot initialize vNext")
    if not has:return
    expected={"search_vnext_version":1,"vnext_memory_version":1,
              "search_a_label_version":3,"actuator_version":1}
    for key,value in expected.items():
        if checkpoint.get(key)!=value:
            raise ValueError("Unsupported vNext checkpoint "+key)
    if any(key.startswith(("_search_c.","_wm.search_a.","_wm.search_b.")) for key in state):
        raise ValueError("Mixed legacy/vNext checkpoint")


def restore_vnext_config(config,checkpoint):
    has=any(key.startswith("_vnext.") for key in checkpoint.get("agent_state_dict",{}))
    config["search_vnext_enabled"]=has
    if not has:return config
    saved=checkpoint.get("config",{})
    for key,value in saved.items():
        if key.startswith("vnext_"):
            config[key]=value
    config["search_a_enabled"]=False;config["search_b_enabled"]=False;config["search_c_enabled"]=False
    config["compile"]=False;config["search_branch_enabled"]=False
    check_vnext_checkpoint(checkpoint,True)
    return config


