"""Independent source350 repair with train-only frozen distribution anchors."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import goal_bc as bc
import goal_continuation_repair as continuation
import goal_information_probe as probe
import goal_library as gl
import goal_residual_worker as residual
import goal_within_horizon_repair as previous
import long_horizon as lh

FORMAT = "ls_imagine_retention_repair_worker_v1"
PROTOCOL = "source350_factual256_train_only_teacher_original_reference32_retained84_12_v1"
SEMANTICS = "independent_fixed50_factual_and_frozen350_KL_not_behavior_acceptance"
SOURCE_VERSION, PLANNED_UPDATES = 350, 50
LEARNING_RATE, RETENTION_NLL_LIMIT = 1e-5, .02
DEPENDENCIES = (*previous.DEPENDENCIES[:-1], "conflict_diagnosis_report", "conflict_diagnosis_manifest",
                "retention_repair_check_report")
CHECKS = (*previous.CHECKS, "teacher_anchor_contract", "teacher_kl_contract", "teacher_frozen_contract")


def sampling_plan(batch_size):
    plan = previous.sampling_plan(batch_size)
    plan.update(protocol=PROTOCOL, loss="paired factual mixed-onehot NLL plus frozen350 full-distribution KL",
        original_teacher_rows_per_batch=192, reference_teacher_train_rows=32,
        retained84_teacher_manual_train_rows=12, anchor_goal_sources=["own_real_endpoint", "fixed_old_targets"],
        cached_anchor_contexts_per_branch=88, original_kl_weight=1.0, reference_kl_weight=4.0, retained84_kl_weight=4.0,
        teacher_source_version=350, teacher_is_expert=False, teacher_trainable=False,
        development_rows_as_anchors=False, worker_facts_as_anchors=False,
        retained84_factual_labels_used_for_added_NLL=False, retained84_train_distributions_used_for_retention=True, training_seed=7,
        added_learning_rate=LEARNING_RATE, fixed_additional_updates=50)
    return plan


def validate_anchor_rows(data, refs, old):
    gl.require(np.array_equal(refs, np.flatnonzero(data.reference_arrays["train"])) and len(refs) == 32 and
        np.array_equal(old, np.flatnonzero(data.continuation_arrays["train"] & data.continuation_arrays["manual"])) and
        len(old) == 12, "分布锚点必须为全部32参考训练及12旧续段手工训练，禁止留出/worker/筛选")
    for arrays, rows, endpoint in ((data.reference_arrays, refs, 80), (data.continuation_arrays, old, 84)):
        gl.require(np.all(arrays["train"][rows]) and
            np.all(arrays["frame"][rows] + arrays["remaining"][rows] == endpoint),
            "锚点自身真实状态/remaining/80或84帧终点不能移植")


class RepairData(previous.RepairData):
    def __init__(self, accepted, lineage):
        self.__dict__.update(accepted.__dict__)
        self.retention_lineage = copy.deepcopy(lineage)
        self.anchor_reference_rows = np.flatnonzero(self.reference_arrays["train"])
        self.anchor_retained_rows = np.flatnonzero(self.continuation_arrays["train"] & self.continuation_arrays["manual"])
        validate_anchor_rows(self, self.anchor_reference_rows, self.anchor_retained_rows)
        arrays, contexts = {k: [] for k in ("features", "goals", "remaining")}, []
        for pool, source, rows, runs in (("reference", self.reference_arrays, self.anchor_reference_rows, self.runs),
            ("retained84", self.continuation_arrays, self.anchor_retained_rows, self.continuation_runs)):
            for goal_source in ("own_real_endpoint", "fixed_old_targets"):
                for i in rows:
                    run = runs[int(source["run"][i])]
                    goal = source["goals"][i] if goal_source == "own_real_endpoint" else self.fixed_goals[source["target"][i]]
                    arrays["features"].append(source["features"][i]); arrays["goals"].append(goal)
                    arrays["remaining"].append(source["remaining"][i])
                    contexts.append(dict(pool=pool, row=int(i), run=int(source["run"][i]),
                        trajectory_path=run["trajectory_path"], trajectory_sha256=run["trajectory_sha256"],
                        frame=int(source["frame"][i]), remaining=int(source["remaining"][i]),
                        endpoint_frame=80 if pool == "reference" else 84, target=int(source["target"][i]),
                        split="train", goal_source=goal_source,
                        state_sha256=hashlib.sha256(source["features"][i].tobytes()).hexdigest(),
                        input_goal_sha256=hashlib.sha256(goal.tobytes()).hexdigest(),
                        own_endpoint_sha256=hashlib.sha256(source["goals"][i].tobytes()).hexdigest(),
                        expert_label=False, factual_action_added_to_NLL=False))
        self.anchor_arrays = {k: np.asarray(v, dtype=np.int64 if k == "remaining" else np.float32) for k,v in arrays.items()}
        self.anchor_contexts, self.anchor_id = contexts, probe.array_digest(self.anchor_arrays)
        gl.require(len(contexts) == 88, "保留锚点上下文遗漏")

    def identity(self):
        return dict(super().identity(), retention_protocol=PROTOCOL, retention_lineage=self.retention_lineage,
            teacher_anchor_id=self.anchor_id, teacher_anchor_contexts=self.anchor_contexts,
            retained_continuations_role="all40 evaluated; train12 manual frozen350 distributions anchored; no added factual NLL",
            teacher_is_expert=False)


def kl_rows(teacher, student):
    """KL(teacher || deployed student), including the original single unimix."""
    p, q = teacher.double(), student.double()
    gl.require(p.shape == q.shape and p.ndim == 2 and p.shape[1] == 12 and
        bool(torch.isfinite(p).all() and torch.isfinite(q).all()) and bool((p > 0).all() and (q > 0).all()),
        "教师KL概率形状/数值异常")
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    result = (p * (p.log() - q.log())).sum(-1)
    gl.require(bool(torch.isfinite(result).all()) and bool((result >= -1e-12).all()), "教师KL非有限或负值")
    return result.clamp_min(0)


class Trainer(previous.Trainer):
    def __init__(self, data, base, refs, options, source, device):
        super().__init__(data, base, refs, options, source, device)
        self.teacher = continuation.load_policy(refs["source350_checkpoint"]["source_path"], data.cache, device)
        self.teacher_hash = gl.tensor_digest(self.teacher.state_dict(), {})
        self.teacher_versions = {n:p._version for n,p in self.teacher.named_parameters()}
        self.anchor_inputs = tuple(torch.as_tensor(data.anchor_arrays[k].copy(), device=device)
                                   for k in ("features", "goals", "remaining"))
        with torch.no_grad(), gl.encoder_precision(device):
            self.anchor_probabilities = {n:self.teacher.action_dist(*self.anchor_inputs, n).probs.detach().clone()
                                         for n in residual.NAMES}
        self.teacher_cache_id = probe.array_digest({n:p.cpu().numpy() for n,p in self.anchor_probabilities.items()})
        self.teacher_forward_queries = 176  # Runtime cost; distinct from restored production counters.
        self.check_teacher(content=True)

    def sampling_plan(self):
        return sampling_plan(self.options["batch_size"])

    def check_teacher(self, content=False):
        validate_anchor_rows(self.data, self.data.anchor_reference_rows, self.data.anchor_retained_rows)
        gl.require(all(not p.requires_grad and p.grad is None for p in self.teacher.parameters()) and
            self.teacher_versions == {n:p._version for n,p in self.teacher.named_parameters()}, "冻结350教师产生梯度或改变")
        gl.require(not {id(p) for p in self.teacher.parameters()} &
            {id(p) for opt in self.optimizers.values() for group in opt.param_groups for p in group["params"]},
            "教师不能进入优化器")
        gl.require(probe.array_digest(self.data.anchor_arrays) == self.data.anchor_id and
            probe.array_digest({k:t.cpu().numpy() for k,t in zip(("features","goals","remaining"),self.anchor_inputs)}) == self.data.anchor_id and
            self.teacher_cache_id == probe.array_digest({n:p.cpu().numpy() for n,p in self.anchor_probabilities.items()}),
            "真实锚点状态或教师缓存被替换")
        self.teacher.check_frozen(content=content)
        if content:
            gl.require(gl.tensor_digest(self.teacher.state_dict(), {}) == self.teacher_hash, "教师内容改变")

    def teacher_binding(self):
        return dict(source_version=350, checkpoint_sha256=self.reference["source350_checkpoint"]["sha256"],
            model_id=self.teacher.model_id, state_digest=self.teacher_hash, anchor_id=self.data.anchor_id,
            anchor_probability_id=self.teacher_cache_id, train_only=True, expert_labels=False)

    def distillation_counts(self):
        return dict(step=self.step, original_teacher_queries_per_branch=192*self.step,
            cached_reference_context_uses_per_branch=64*self.step, cached_retained84_context_uses_per_branch=24*self.step,
            cached_teacher_initial_queries=176, cached_teacher_queries_are_not_independent_evidence=True)

    def update(self, verification_only=False):
        gl.require(self.step == 50 if verification_only else self.step < 50, "固定50步不能延长")
        original, refs, added, order = selection = self.draw()
        x,g,r,labels = self.batch(selection)
        ox,og,orr,_ = self.data.cache.worker_batch(original, self.device)
        self.check_teacher()
        with torch.no_grad(), gl.encoder_precision(self.device):
            teacher_original = {n:self.teacher.action_dist(ox,og,orr,n).probs.detach() for n in residual.NAMES}
        self.teacher_forward_queries += 384
        result = dict(step=self.step+1, shared_batch_id=probe.array_digest(dict(original=original,reference=refs,feedback=added,order=order)),
            verification_only=verification_only, anchor_id=self.data.anchor_id)
        self.model.train(True)
        with gl.encoder_precision(self.device):
            base = self.model.base_preferences(x,g,r)
            for name in residual.NAMES:
                opt = self.optimizers[name]; opt.zero_grad(set_to_none=True)
                factual = F.nll_loss(self.model.distribution(base+self.model.correction(name,x,g,r)).logits,labels)
                original_kl = kl_rows(teacher_original[name],self.model.action_dist(ox,og,orr,name).probs).mean()
                anchors = kl_rows(self.anchor_probabilities[name],self.model.action_dist(*self.anchor_inputs,name).probs)
                reference_kl, retained_kl = anchors[:64].mean(), anchors[64:].mean()
                loss = factual + original_kl + 4*reference_kl + 4*retained_kl
                gl.require(bool(torch.isfinite(loss)), "保留修复损失非有限")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.heads[name].parameters(),self.options["grad_clip"])
                gl.require(bool(torch.isfinite(norm)), "保留修复梯度非有限"); opt.step()
                gl.require(all(bool(torch.isfinite(p).all()) for p in self.model.heads[name].parameters()), "权重非有限")
                result.update({name+"_nll":float(factual.detach()),name+"_loss":float(loss.detach()),
                    name+"_original_kl":float(original_kl.detach()),name+"_reference_kl":float(reference_kl.detach()),
                    name+"_retained84_kl":float(retained_kl.detach()),name+"_grad_norm":float(norm)})
        for s,pool,rows,key in ((0,self.data.cache.tables,original,"action_id"),
            (1,self.data.reference_arrays,refs,"labels"),(2,self.data.feedback_arrays,added,"labels")):
            self.counts["source_remaining"][s] += np.bincount(pool["remaining"][rows]-1,minlength=16)
            self.counts["source_action"][s] += np.bincount(pool[key][rows],minlength=12)
        self.counts["reference_target"] += np.bincount(self.data.reference_arrays["target"][refs],minlength=2)
        self.counts["feedback_row"] += np.bincount(added,minlength=80)
        self.step += 1; self.check_ownership(); self.check_teacher()
        return result

    def payload(self, verification_artifact=False):
        payload = residual.Trainer.payload(self,verification_artifact)
        payload.update(checkpoint_format=FORMAT,semantics=SEMANTICS,sampling_protocol=PROTOCOL,
            repair_sampling=self.sampling_plan(),sampling_counts=copy.deepcopy(self.counts),source_worker_version=350,
            planned_additional_updates=50,teacher_binding=self.teacher_binding(),distillation_counts=self.distillation_counts())
        validate_payload(payload,self.identity,self.options,verification_artifact)
        return payload

    def restore(self,payload,allow_verification=False):
        validate_payload(payload,self.identity,self.options,allow_verification)
        gl.require(payload["teacher_binding"] == self.teacher_binding(), "恢复教师/checkpoint/锚点缓存身份不同")
        residual.Trainer.restore(self,dict(payload,checkpoint_format=residual.FORMAT,
            semantics=residual.SEMANTICS,sampling_protocol=residual.SAMPLING),allow_verification)
        self.counts = {k:v.copy() for k,v in payload["sampling_counts"].items()}
        self.check_teacher(content=True)


def validate_payload(payload,identity,options,allow_verification=False):
    gl.require(payload.get("checkpoint_format") == FORMAT and payload.get("semantics") == SEMANTICS and
        payload.get("sampling_protocol") == PROTOCOL and payload.get("source_worker_version") == 350 and
        payload.get("planned_additional_updates") == 50 and options["learning_rate"] == LEARNING_RATE and options["seed"] == 7 and
        payload.get("repair_sampling") == sampling_plan(options["batch_size"]), "保留修复格式/源350/固定50/损失或采样协议不同")
    residual.validate_payload(dict(payload,checkpoint_format=residual.FORMAT,semantics=residual.SEMANTICS,
        sampling_protocol=residual.SAMPLING),identity,options,allow_verification)
    # Reuse exact three-pool count/split checks with a metadata-only compatibility view.
    previous.validate_payload(dict(payload,checkpoint_format=previous.FORMAT,semantics=previous.SEMANTICS,
        sampling_protocol=previous.PROTOCOL,repair_sampling=previous.sampling_plan(256),
        options=dict(options,learning_rate=previous.LEARNING_RATE)),identity,dict(options,learning_rate=previous.LEARNING_RATE),allow_verification)
    gl.require(identity.get("retention_protocol") == PROTOCOL and identity.get("teacher_is_expert") is False and
        identity["retention_lineage"].get("experiment_family") == PROTOCOL and
        identity["retention_lineage"].get("source_version") == 350 and
        identity["retention_lineage"].get("old400_weights_used") is False and
        identity["retention_lineage"].get("train_anchor_rows") == dict(reference=32,retained84_manual=12),
        "独立保留协议/训练锚点或源350身份不同")
    contexts = identity["teacher_anchor_contexts"]
    expected = [(pool,source) for pool,size in (("reference",32),("retained84",12))
                for source in ("own_real_endpoint","fixed_old_targets") for _ in range(size)]
    gl.require(len(contexts) == 88 and [(c["pool"],c["goal_source"]) for c in contexts] == expected and
        all(c["split"] == "train" and c["expert_label"] is False and c["factual_action_added_to_NLL"] is False and
            c["frame"]+c["remaining"] == c["endpoint_frame"] == (80 if c["pool"] == "reference" else 84)
            for c in contexts), "锚点上下文/真实80或84帧预算/训练划分异常")
    step,b = payload["counters"]["step"],payload["teacher_binding"]
    for name,optimizer in payload["optimizers"].items():
        groups=optimizer["param_groups"]
        gl.require(len(groups) == 1 and groups[0]["lr"] == LEARNING_RATE and
            len(groups[0]["params"]) == len(payload["heads"][name]) and
            (not optimizer["state"] if step == 0 else
             set(optimizer["state"]) == set(groups[0]["params"]) and
             all(float(v["step"]) == step for v in optimizer["state"].values())),
            "新Adam学习率/参数所有权或生产更新步不同")
    gl.require(b.get("source_version") == 350 and b.get("train_only") is True and b.get("expert_labels") is False and
        b.get("checkpoint_sha256") == identity["within_horizon_lineage"]["source350_checkpoint_sha256"] and
        b.get("model_id") == identity["within_horizon_lineage"]["source350_model_id"] and b.get("anchor_id") == identity["teacher_anchor_id"] and
        all(isinstance(b.get(k),str) and len(b[k]) == 64 for k in ("state_digest","anchor_probability_id")) and
        payload["distillation_counts"] == dict(step=step,original_teacher_queries_per_branch=192*step,
            cached_reference_context_uses_per_branch=64*step,cached_retained84_context_uses_per_branch=24*step,
            cached_teacher_initial_queries=176,cached_teacher_queries_are_not_independent_evidence=True),
        "教师来源/锚点或KL查询计数异常")


def save_checkpoint(trainer,path,check_hash,verification_artifact=False):
    return residual.save_checkpoint(trainer,path,check_hash,verification_artifact,overwrite=Path(path).exists())


def read_checkpoint(path,trainer=None,check_hash=None):
    path = Path(path).resolve(); payload = bc._torch_load(path)
    gl.require(payload.get("checkpoint_format") == FORMAT and set(payload.get("dependencies",{})) == set(DEPENDENCIES),
        "需要独立保留修复worker，不能加载旧400/best/其他格式")
    validate_payload(payload,payload["input_identity"],payload["options"],bool(trainer is not None))
    refs = {}
    for n,item in payload["dependencies"].items():
        gl.require(isinstance(item.get("path"),str) and item["path"] and not Path(item["path"]).is_absolute(), "共享依赖必须为相对路径")
        p = (path.parent/item["path"]).resolve()
        gl.require(p.is_file() and bc.file_hash(p) == item["sha256"],f"保留依赖SHA不同：{n}"); refs[n] = p
    if trainer is not None:
        gl.require(all(refs[n] == trainer.reference[n]["source_path"] and payload["dependencies"][n]["sha256"] ==
            trainer.reference[n]["sha256"] for n in DEPENDENCIES), "恢复共享依赖身份不同")
    check_path = refs["retention_repair_check_report"]
    checked = json.loads(check_path.read_text(encoding="utf-8"))
    gl.require(checked.get("command") == "check" and checked.get("status") in ("passed","passed_with_warnings") and
        checked.get("repair_format") == FORMAT and checked.get("repair_identity") == payload["input_identity"] and
        checked.get("options") == payload["options"] and checked.get("sampling_plan") == sampling_plan(256) and
        checked.get("teacher_binding") == payload["teacher_binding"] and checked.get("planned_repair_updates") == 50 and
        set(CHECKS) <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"} and
        payload.get("accepted_check_sha256") == bc.file_hash(check_path) and
        (check_hash is None or check_hash == payload["accepted_check_sha256"]), "生产修复必须绑定完整独立check")
    root = Path(__file__).resolve().parent
    for name,digest in checked["repair_code"].items():
        code = (root/name).resolve()
        gl.require(root in code.parents and bc.file_hash(code) == digest,f"修复代码改变：{name}")
    for name,digest in checked["check_artifacts"].items():
        p = (check_path.parent/name).resolve()
        gl.require(p.parent == check_path.parent and bc.file_hash(p) == digest,f"check产物改变：{name}")
    lineage = payload["input_identity"]["within_horizon_lineage"]
    for n in previous.DEPENDENCIES[3:-1]:
        gl.require(payload["dependencies"][n]["sha256"] == lineage[n+"_sha256"], "旧来源SHA不同")
    for n in ("conflict_diagnosis_report","conflict_diagnosis_manifest"):
        gl.require(payload["dependencies"][n]["sha256"] == payload["input_identity"]["retention_lineage"][n+"_sha256"], "诊断来源SHA不同")
    source,_ = continuation.read_checkpoint(refs["source350_checkpoint"])
    verified = json.loads(refs["source350_verify_report"].read_text(encoding="utf-8"))
    gl.require(refs["source350_checkpoint"].name == "latest.pt" and not source.get("verification_artifact") and
        source["counters"]["step"] == 50 and source["input_identity"] == lineage["source350_identity"] and
        verified.get("command") == "verify" and verified.get("status") in ("passed","passed_with_warnings") and
        verified.get("worker_version") == 350 and verified.get("model_id") == payload["teacher_binding"]["model_id"] and
        verified.get("repair_identity") == source["input_identity"] and verified.get("counters") == source["counters"] and
        verified.get("backend") == source["backend"] == payload["backend"] and
        verified.get("source_inputs_before") == verified.get("source_inputs_after") and
        Path(verified["arguments"]["checkpoint"]).resolve() == refs["source350_checkpoint"], "教师必须为已验收生产350")
    return payload,refs


def load_policy(path,cache,device):
    payload,refs = read_checkpoint(path)
    gl.require(Path(path).name == "latest.pt" and payload["counters"]["step"] == 50 and
        cache.cache_id == payload["input_identity"]["cache_id"] and cache.metadata["bundle_id"] == payload["input_identity"]["bundle_id"] and
        refs["frozen_bundle"] == (cache.directory/"frozen_bundle.pt").resolve() and payload["backend"] == bc.backend_info(device),
        "独立推理只接受固定50步生产latest，同后端；禁止best/验收副本")
    base,_ = residual.load_base(refs["base_checkpoint"],refs["base_verify_report"],cache)
    base = {k:v for k,v in base.items() if k not in ("optimizers","candidate")}; base["frozen_bundle"] = cache.bundle
    rng = bc.capture_rng(device)
    try:
        model = residual.ResidualPolicy(cache,base,payload["input_identity"],payload["options"],device)
        for n in residual.NAMES:
            lh.require_compatible(model.heads[n].state_dict(),payload["heads"][n]); model.heads[n].load_state_dict(payload["heads"][n],strict=True)
    finally:
        bc.restore_rng(rng,device)
    model.source_worker_version,model.retention_updates,model.worker_version = 350,50,400
    model.experiment_family = PROTOCOL
    model.model_id = gl.tensor_digest(model.heads.state_dict(),dict(format=FORMAT,input_identity=model.identity,
        options=payload["options"],worker_version=400,retention_updates=50))
    return model.requires_grad_(False).eval()


@torch.no_grad()
def evaluate(trainer):
    result,retained,rows = previous.evaluate(trainer)
    trainer.check_teacher(content=True)
    anchor_kl = {}
    with gl.encoder_precision(trainer.device):
        for n in residual.NAMES:
            values = kl_rows(trainer.anchor_probabilities[n],trainer.model.eval().action_dist(*trainer.anchor_inputs,n).probs)
            anchor_kl[n] = dict(reference_train=float(values[:64].mean()),retained84_manual_train=float(values[64:].mean()))
    for source in ("own_real_endpoint","fixed_old_targets"):
        for condition in ("visual_correct_hold","worker_continuation"):
            result["continuation"][source][condition]["role"] = "all40 evaluation; manual train12 distribution anchors only"
    result.update(retention_repair_format=FORMAT,retention_protocol=PROTOCOL,teacher_binding=trainer.teacher_binding(),
        teacher_anchor_kl=anchor_kl,distillation_counts=trainer.distillation_counts(),retained84_labels_used_for_training=False,
        retained84_train_distributions_used_for_retention=True)
    result["within_horizon"].update(format=FORMAT,comparison_protocol=PROTOCOL)
    return result,retained,rows


retention_review = previous.retention_review
validate_feedback = previous.validate_feedback
