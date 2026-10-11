"""Independent fixed50 source350 repair with train-only distribution retention."""
import argparse
import copy
from datetime import datetime
import gc
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t04_goal_bc as t04
import t05_goal_control as legacy
import t05_within_horizon_repair as accepted_repair
import t05_within_horizon_conflict_diagnose as conflict

CODE = ("goal_retention_repair.py", "scripts/t05_retention_repair.py",
        "goal_retention_control.py", "scripts/t05_retention_control.py", "scripts/t05_retention_pipeline.py")


class Report(baseline.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        saved = self.save()
        failed = not saved or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_RETENTION_REPAIR_{self.data['command'].upper()}; "
              f"report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def code_identity():
    import goal_bc as bc
    return {**accepted_repair.code_identity(), **{n:bc.file_hash(ROOT/n) for n in (*conflict.CODE,*CODE)}}


def retention_plan():
    return dict(accepted_repair.retention_plan(),
        original84frame_continuations="all40 evaluated; train12 manual frozen350 distributions anchored, not added NLL",
        teacher_anchors_train_only=True, teacher_is_expert=False, source_version=350,
        independent_repair_family="retention_constrained", gate_changed=False)


def confirmation_plan(source):
    import goal_retention_control as acceptance
    import goal_random_control as pilot
    return dict(accepted_repair.confirmation_plan(source), randomization_seed=acceptance.RANDOMIZATION_SEED,
        schedule=pilot.schedule(0,5,acceptance.RANDOMIZATION_SEED), implementation_pending=False,
        experiment_family="retention_constrained", requires_predeclared_retention_pass=True,
        checkpoint_binding="independent retention format production350+50 latest plus verify; no best",
        t06_approved=False)


def historical_inputs(args):
    import goal_bc as bc
    import goal_continuation_control as acceptance
    import goal_library as gl
    import goal_within_horizon_conflict_diagnose as diagnosis
    directory = baseline.project_path(args.diagnosis_dir)
    record = legacy.read_json(directory/"report.json"); legacy.accepted(record,"analyze")
    manifest = legacy.read_json(directory/"diagnosis_manifest.json")
    source,protected,files = conflict.historical_inputs(SimpleNamespace(
        repair_verify_dir=record["arguments"]["repair_verify_dir"],device=args.device))
    gl.require(record.get("diagnostic_format") == diagnosis.FORMAT and
        record.get("comparison_protocol") == diagnosis.PROTOCOL and record.get("query_rows") == 184 and
        record.get("episodes") == 34 and record.get("diagnostic_distribution_rows") == 2208 and
        record.get("historical_roundtrip_verified") is True and record.get("comparison_queries_complete") is True and
        record.get("source_inputs_before") == record.get("source_inputs_after") and
        record["arguments"]["device"] == args.device and
        all(record.get(k) == 0 for k in ("optimizer_updates","new_env_steps","worker_control_actions","intervention_actions")) and
        record.get("t06_approved") is False and record.get("behavior_accepted") is False and
        record.get("accepted_source_code") == accepted_repair.code_identity() and
        record.get("diagnostic_code") == {n:bc.file_hash(ROOT/n) for n in conflict.CODE} and
        manifest.get("format") == diagnosis.FORMAT and manifest.get("comparison_protocol") == diagnosis.PROTOCOL and
        manifest.get("diagnosis_id") == record["diagnosis_id"] == gl.tensor_digest({},
            {k:v for k,v in manifest.items() if k != "diagnosis_id"}) and
        manifest.get("input_identity") == record["input_identity"] == source["repair_payload"]["input_identity"] and
        isinstance(record.get("source_artifacts_sha256"),dict) and
        manifest.get("source_artifacts_sha256") == record["source_artifacts_sha256"] and
        manifest.get("source_verify_report_sha256") == bc.file_hash(source["repair_verify_dir"]/"report.json") and
        manifest.get("source_checkpoint_sha256") == bc.file_hash(source["repair_checkpoint"]) and
        manifest.get("source350_checkpoint_sha256") == bc.file_hash(source["context"]["checkpoint"]) and
        acceptance.passed(record,("historical_identity","strict_frozen_versions","diagnostic_contracts",
            "historical_goal_context_roundtrip","real_query_alignment","recorded_forward_roundtrip",
            "same_state_goal_and_version_comparison","retention_review_preserved","no_training","source_inputs_unchanged")),
        "需要完整通过184事实/350与旧400冲突诊断及全部历史SHA；旧保留与行为FAIL保持")
    for filename,digest in record["source_artifacts_sha256"].items():
        p=Path(filename).resolve()
        gl.require(p in files and bc.file_hash(p) == digest,"原诊断绑定的真实输入SHA不同："+filename)
    gl.require({"queries.json","diagnostics.json","regressions.json","retention.json","frame_metrics.csv"} <=
        set(manifest["artifacts"]),"原诊断产物缺失")
    for name,digest in manifest["artifacts"].items():
        p=(directory/name).resolve()
        gl.require(not Path(name).is_absolute() and p.parent == directory and bc.file_hash(p) == digest,
            "原诊断产物SHA不同："+name); files.append(p)
    retained=legacy.read_json(directory/"retention.json")
    gl.require(retained["original_ten_checks"] == source["retention"] and
        retained["within_predeclared_retention_limit"] == source["retention"]["within_predeclared_retention_limit"] and
        retained.get("gate_changed") is False and retained.get("autonomous_confirmation_executed") is False,
        "诊断不能清除旧保留失败或改门槛")
    source.update(retention_diagnosis_dir=directory,retention_diagnosis_record=record,retention_diagnosis_manifest=manifest)
    protected.append(directory)
    files += [directory/"report.json",directory/"diagnosis_manifest.json",*[ROOT/n for n in CODE]]
    return source,list(dict.fromkeys(protected)),list(dict.fromkeys(p.resolve() for p in files))


def load_inputs(args, report, source):
    import numpy as np
    import goal_bc as bc
    import goal_information_probe as probe
    import goal_library as gl
    import goal_retention_repair as repair
    class Audit:
        directory=report.directory
        def __init__(self): self.data={}
        def check(self,name,level,detail): gl.require(level != "FAIL",detail)
    # Strict historical reconstruction only; never alter an accepted module's globals.
    accepted,_,base,refs,saved = accepted_repair.load_inputs(SimpleNamespace(command="check",device=args.device),Audit(),source)
    paths=dict(conflict_diagnosis_report=source["retention_diagnosis_dir"]/"report.json",
        conflict_diagnosis_manifest=source["retention_diagnosis_dir"]/"diagnosis_manifest.json")
    lineage=dict(**{k+"_sha256":bc.file_hash(p) for k,p in paths.items()},
        diagnosis_id=source["retention_diagnosis_manifest"]["diagnosis_id"],
        old400_checkpoint_sha256=bc.file_hash(source["repair_checkpoint"]),
        old400_retention=copy.deepcopy(source["retention"]), experiment_family=repair.PROTOCOL,
        source_version=350, old400_weights_used=False, train_anchor_rows=dict(reference=32,retained84_manual=12))
    data=repair.RepairData(accepted,lineage)
    options=dict(saved["options"],learning_rate=repair.LEARNING_RATE,seed=7)
    # Original accepted query lists remain fixed even though the new training sampler has seed7.
    refs.pop("within_horizon_repair_check_report")
    refs.update({k:dict(source_path=p.resolve(),sha256=bc.file_hash(p)) for k,p in paths.items()})
    check_path=report.directory/"report.json" if args.command == "check" else baseline.project_path(args.check_dir)/"report.json"
    refs["retention_repair_check_report"]=dict(source_path=check_path,sha256=None)
    future,retention=confirmation_plan(source),retention_plan()
    if args.command != "check":
        checked=legacy.read_json(check_path); legacy.accepted(checked,"check")
        gl.require(checked["repair_format"] == repair.FORMAT and checked["repair_code"] == code_identity() and
            checked["repair_identity"] == data.identity() and checked["options"] == options and
            checked["backend"] == bc.backend_info(args.device) and checked["sampling_plan"] == repair.sampling_plan(256) and
            checked["retention_plan"] == retention and checked["confirmation_plan"] == future and
            checked["planned_repair_updates"] == 50 and checked["source_inputs_before"] == checked["source_inputs_after"] and
            set(repair.CHECKS) <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"},
            "独立保留check身份/教师协议/固定预算/代码/计划或后端不同")
        for name,digest in checked["check_artifacts"].items():
            p=(check_path.parent/name).resolve()
            gl.require(p.parent == check_path.parent and bc.file_hash(p) == digest,"check产物SHA不同："+name)
        expected={**{"reference__"+k:v for k,v in data.reference_arrays.items()},
            **{"feedback__"+k:v for k,v in data.feedback_arrays.items()},
            **{"retained84__"+k:v for k,v in data.continuation_arrays.items()},
            **{"anchor__"+k:v for k,v in data.anchor_arrays.items()}}
        with np.load(check_path.parent/"supervision.npz",allow_pickle=False) as stored:
            gl.require(set(stored.files) == set(expected) and all(np.array_equal(stored[k],v) for k,v in expected.items()),
                "check事实/锚点真实状态或自身目标被替换")
        gl.require(legacy.read_json(check_path.parent/"teacher_anchors.json")["contexts"] == data.anchor_contexts,
            "锚点上下文/来源不同")
        refs["retention_repair_check_report"]["sha256"]=bc.file_hash(check_path)
    report.data.update(repair_format=repair.FORMAT,semantics=repair.SEMANTICS,repair_code=code_identity(),
        repair_identity=data.identity(),input_identity=data.identity(),options=options,backend=bc.backend_info(args.device),
        planned_repair_updates=50,source_worker_version=350,sampling_plan=repair.sampling_plan(256),
        retention_plan=retention,confirmation_plan=future,original_numerical_gate=source["repair_checked"]["original_numerical_gate"],
        factual_pools=dict(manual_train=24,manual_development_holdout=16,worker_facts_only=40),
        teacher_train_pools=dict(reference=32,retained84_manual=12),teacher_is_expert=False,
        retained84_labels_used_for_training=False,retained84_train_distributions_used_for_retention=True,
        old400_weights_used=False,old400_retention_failures_preserved=source["retention"],dual_goal_correction_train_coverage=False)
    report.check("real_within_horizon_supervision","PASS",
        "完整80事实/自身80帧/真实incoming与原划分；NLL只训24手工，冻结350锚点仅32参考及12旧手工训练；不使用留出或worker锚点")
    return data,options,base,refs,saved


def make_trainer(inputs, device):
    import numpy as np
    import torch
    import goal_retention_repair as repair
    data, options, base, refs, saved = inputs
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    return repair.Trainer(data, base, refs, options, saved, device)


def evaluation_outputs(trainer, report, initial=None):
    import goal_retention_repair as repair
    metrics, retained, rows = repair.evaluate(trainer)
    if initial is not None:
        metrics["retention_review"] = repair.retention_review(initial, metrics)
        baseline.write_json(report.directory / "retention.json", metrics["retention_review"])
    t04.append_json(report.directory / "evaluation.jsonl", metrics)
    baseline.write_json(report.directory / "metrics.json", metrics)
    baseline.write_json(report.directory / "queries.json", rows)
    baseline.write_json(report.directory / "retained_frame_metrics.json", retained)
    flat = [dict({k: v for k, v in row.items() if k not in ("metrics", "probabilities", "raw_preferences")}, **row["metrics"]) for row in rows]
    baseline.write_csv(report.directory / "frame_metrics.csv", flat)
    hold = metrics["within_horizon"]["by_pool"]["manual_development_holdout"]
    print(f"[RETENTION REPAIR] step={trainer.step}/50 dev_actual_nll={hold['metrics']['actual_endpoint_factual_nll']:.6f} "
          f"dev_fixed_nll={hold['metrics']['fixed_goal_factual_nll']:.6f} dev_actual_mode={hold['mode_matches']['actual_endpoint']}/16 "
          f"source350+{trainer.step}", flush=True)
    return metrics, rows


def teacher_contracts(trainer,report):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_retention_repair as repair
    data=trainer.data
    trainer.check_teacher(content=True)
    report.data["teacher_binding"]=trainer.teacher_binding()
    for name,refs,old in (("teacher_holdout_guard",np.flatnonzero(~data.reference_arrays["train"]),data.anchor_retained_rows),
        ("teacher_worker_guard",data.anchor_reference_rows,np.flatnonzero(data.continuation_arrays["train"] & ~data.continuation_arrays["manual"])),
        ("teacher_subset_guard",data.anchor_reference_rows[:-1],data.anchor_retained_rows)):
        t02.rejection(report,name,lambda refs=refs,old=old:repair.validate_anchor_rows(data,refs,old),"禁止留出/worker/筛选")
    with torch.no_grad(),gl.encoder_precision(trainer.device):
        for name in ("goal","no_goal"):
            p=trainer.anchor_probabilities[name]
            current=trainer.model.action_dist(*trainer.anchor_inputs,name).probs
            gl.require(torch.allclose(p,current,atol=3e-5,rtol=3e-5) and
                float(repair.kl_rows(p,current).max()) <= 1e-10,"step0教师和学生部署概率/KL不同")
    # Tiny distribution-only derivative check: no model parameters or optimizer are touched.
    p=torch.tensor([[.5]+[.5/11]*11],device=trainer.device,dtype=torch.float64)
    logits=torch.zeros_like(p,requires_grad=True); q=logits.softmax(-1)
    loss=repair.kl_rows(p,q).mean(); gradient,=torch.autograd.grad(loss,logits)
    gl.require(float(loss)>0 and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum())>0 and
        float(repair.kl_rows(p,p).max()) == 0,"KL自身为零/分布变化/学生梯度守卫失败")
    t02.rejection(report,"teacher_probability_guard",lambda:repair.kl_rows(p,torch.full_like(p,float("nan"))),"概率形状/数值")
    original=trainer.anchor_probabilities["goal"]
    try:
        trainer.anchor_probabilities["goal"]=original.clone()
        trainer.anchor_probabilities["goal"][0,0] += .01
        t02.rejection(report,"teacher_cache_guard",trainer.check_teacher,"教师缓存")
    finally:
        trainer.anchor_probabilities["goal"]=original
    original_inputs=trainer.anchor_inputs
    try:
        trainer.anchor_inputs=(original_inputs[0].clone(),*original_inputs[1:])
        trainer.anchor_inputs[0][0,0] += .01
        t02.rejection(report,"teacher_state_guard",trainer.check_teacher,"真实锚点状态")
    finally:
        trainer.anchor_inputs=original_inputs
    trainer.check_teacher(content=True)
    baseline.write_json(report.directory/"teacher_anchors.json",dict(binding=trainer.teacher_binding(),
        contexts=data.anchor_contexts,probabilities={k:v.cpu().tolist() for k,v in trainer.anchor_probabilities.items()},
        plan=trainer.sampling_plan(),expert_labels=False,train_only=True))
    report.check("teacher_anchor_contract","PASS","全部32参考训练/12旧续段手工训练×自身/固定目标=88上下文；不使用开发、worker或移植动作")
    report.check("teacher_kl_contract","PASS","部署概率KL(350||student)含原一次unimix；自身为零、变化非零、学生梯度有效；原192/参考64/旧24权重1/4/4")
    report.check("teacher_frozen_contract","PASS","350教师无优化器/梯度、参数版本及内容未变；缓存与原真实锚点SHA绑定")


def check(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_retention_repair as repair
    import goal_within_horizon_supervision as supervision
    trainer = make_trainer(inputs, args.device)
    teacher_contracts(trainer,report)
    data, _, _, _, source = inputs
    gl.require(all(t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        not trainer.optimizers[k].state for k in ("goal", "no_goal")), "只能继承350权重，不能恢复旧优化器")
    report.check("warm_start", "PASS", "生产350双头逐值继承，新Adam步0；固定50步/1e-5，无RSSM/候选模型")
    rng = trainer.sampler.get_state()
    first = trainer.draw()
    trainer.sampler.set_state(rng)
    gl.require(t01.same(first, trainer.draw()), "混合采样RNG恢复后不同")
    trainer.sampler.set_state(rng)
    report.check("sampler_contract", "PASS", "每批192原训练+48原参考（每目标24）+16全部手工训练；双头同批，均匀有放回，无加权/纠偏过采样")
    t02.rejection(report, "sampling_budget_guard", lambda: repair.sampling_plan(255), "固定256")
    a, actual_pool = data.feedback_arrays, data.manual_train_rows.copy()
    try:
        for name, bad in (("sample_holdout_guard", np.flatnonzero(a["manual"] & ~a["train"])),
            ("sample_worker_guard", np.flatnonzero(~a["manual"] & a["train"])), ("sample_subset_guard", actual_pool[:-1])):
            data.manual_train_rows = bad
            t02.rejection(report, name, trainer.draw, "全部24")
    finally:
        data.manual_train_rows = actual_pool
        trainer.sampler.set_state(rng)
    def validate(changed):
        return repair.validate_feedback(changed, data.feedback_runs, data.identity())
    for name, key, value, message in (("training_split_guard", "train", not bool(a["train"][0]), "整局划分"),
        ("remaining_guard", "remaining", 1, "remaining"), ("action_guard", "labels", (int(a["labels"][0])+1)%12, "真实下一动作"),
        ("manual_source_guard", "manual", not bool(a["manual"][0]), "纠偏来源")):
        bad = {k: v.copy() for k, v in a.items()}
        bad[key][0] = value
        t02.rejection(report, name, lambda bad=bad: validate(bad), message)
    for name, key in (("own_endpoint_guard", "goals"), ("real_state_guard", "features")):
        bad = {k: v.copy() for k, v in a.items()}
        bad[key][:4] = -bad[key][:4]
        t02.rejection(report, name, lambda bad=bad: validate(bad), "真实80帧终点")
    payload = trainer.payload()
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "protocol_guard", lambda: trainer.restore(dict(payload, sampling_protocol="wrong")), "采样协议")
    x, g, r, _ = data.feedback_batch(np.arange(80), args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        raw = trainer.model.base_preferences(x, g, r) + trainer.model.correction("goal", x, g, r)
        gl.require(torch.equal(trainer.model.distribution(raw).probs, trainer.model.action_dist(x, g, r).probs), "部署unimix不同")
        for mode in ("no_goal", "zero_goal", "base"):
            gl.require(torch.equal(trainer.model.action_dist(x, g, r, mode).probs,
                trainer.model.action_dist(x, torch.flip(g, [0]), r, mode).probs), "无目标/置零/底座依赖目标")
    report.check("distribution_contract", "PASS", "raw偏好相加及原一次unimix不变；no_goal/zero/base严格目标不变")
    initial, rows = evaluation_outputs(trainer, report)
    maximum_error = 0.0
    for actual, original in zip(rows, data.feedback_queries):
        for c in supervision.CONDITIONS:
            error = float(np.max(np.abs(np.asarray(actual["probabilities"][c]) - original["probabilities"][c])))
            maximum_error = max(maximum_error, error)
            gl.require(np.allclose(actual["probabilities"][c], original["probabilities"][c], atol=3e-5, rtol=3e-5) and
                np.allclose(actual["raw_preferences"][c], original["raw_preferences"][c], atol=3e-5, rtol=3e-5) and
                actual["metrics"][c + "_mode"] == original["metrics"][c + "_mode"], "step0不能重现原350六条件概率/模式")
    report.data.update(source350_roundtrip_queries=80, source350_distribution_roundtrip_rows=480, maximum_probability_error=maximum_error)
    report.check("source350_roundtrip", "PASS", f"未更新双头重现原80状态/480分布与mode，最大误差={maximum_error:.6g}；实际/固定目标分开")
    summary = initial["within_horizon"]
    gl.require(summary["missing_manual_branches"] == ["target1_train_correction"] and
        summary["by_target_split_branch"]["target1_train_correction"]["metrics"] is None,
        "目标1训练纠偏缺失必须保留null，不重新划分或填零")
    baseline.write_json(report.directory / "coverage.json", dict(by_pool={p: dict(rows=v["rows"], independent_episodes=v["independent_episodes"],
        action_counts=v["action_counts"]) for p, v in summary["by_pool"].items()},
        source_branch_coverage=summary["by_target_split_branch"], missing_manual_branches=summary["missing_manual_branches"],
        repeated_rows_are_not_independent_evidence=True, expert_labels_generated=False, t06_approved=False))
    report.check("diagnostic_contracts", "PASS", "两目标/整局划分/保持与纠偏分别报告；旧84帧全部评价，训练12只作分布锚点；空组null，偏航两局不筛除")
    report.check("correction_coverage", "WARN", "训练23noop/1上转/0下转；目标1预算内训练纠偏缺失，不能声称双目标纠偏泛化")
    np.savez_compressed(report.directory / "supervision.npz", **{"reference__"+k:v for k,v in data.reference_arrays.items()},
        **{"feedback__"+k:v for k,v in a.items()}, **{"retained84__"+k:v for k,v in data.continuation_arrays.items()},
        **{"anchor__"+k:v for k,v in data.anchor_arrays.items()})
    for name, content in (("initial_metrics.json", initial), ("retention_plan.json", report.data["retention_plan"]),
                           ("confirmation_plan.json", report.data["confirmation_plan"])):
        baseline.write_json(report.directory / name, content)
    names = ("supervision.npz", "coverage.json", "initial_metrics.json", "retention_plan.json", "confirmation_plan.json", "queries.json", "teacher_anchors.json")
    report.data["check_artifacts"] = {n: bc.file_hash(report.directory / n) for n in names}
    report.check("initial_metrics", "PASS", "step0原26局/两目标两来源双分支10项保留限度0.02已声明；80帧事实与旧84帧保持/纠偏另报，不选best")
    trainer.model.check_frozen(content=True)
    gl.require(trainer.step == 0 and all(not trainer.optimizers[k].state and
        t01.same(trainer.model.heads[k].state_dict(), source["heads"][k]) and
        all(p.grad is None for p in trainer.model.heads[k].parameters()) for k in ("goal", "no_goal")), "check不能更新权重/优化器/梯度")
    report.data.update(counters=trainer.counters(), optimizer_updates=0, worker_version=350,
        teacher_forward_queries=trainer.teacher_forward_queries,teacher_optimizer_updates=0)
    report.check("no_updates", "PASS", "check仅初始化/采样/前向，双头与新优化器步0；不启动训练或环境")


def train(args, report, inputs):
    import numpy as np
    import goal_bc as bc
    import goal_library as gl
    import goal_retention_repair as repair
    trainer = make_trainer(inputs, args.device)
    report.data["teacher_binding"] = trainer.teacher_binding()
    digest = trainer.reference["retention_repair_check_report"]["sha256"]
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    baseline.write_json(report.directory / "initial_metrics.json", initial)
    current, _ = evaluation_outputs(trainer, report, initial)
    gl.require(t01.same({k:v for k,v in current.items() if k != "retention_review"}, initial), "训练step0与独立check不同")
    estimate = 3 * bc.tensor_storage_bytes(trainer.model.heads.state_dict()) + 1024**2
    bc.require_disk_space(report.directory, 3 * estimate)
    report.check("storage_preflight", "PASS", "双头/优化器快照空间已预检；原模型/缓存/轨迹只共享引用")
    report.check("paired_training", "PASS", "350权重warm start，双头同批192/48/16真实NLL；固定追加50步，开发/worker不训练；旧84帧训练12只作教师分布锚点")
    while trainer.step < 50:
        report.require_writable()
        values = trainer.update()
        report.data.update(counters=trainer.counters(), optimizer_updates=trainer.counters()["optimizer_updates"],
                           worker_version=350+trainer.step, additional_updates=trainer.step)
        t04.append_json(report.directory / "training.jsonl", values)
        if trainer.step % 10 == 0:
            print(f"[TRAIN RETENTION REPAIR] step={trainer.step}/50 goal_nll={values['goal_nll']:.4f} no_goal_nll={values['no_goal_nll']:.4f}", flush=True)
            current, _ = evaluation_outputs(trainer, report, initial)
            repair.save_checkpoint(trainer, report.directory / "latest.pt", digest)
            counts, correction = trainer.counts["feedback_row"], trainer.data.feedback_arrays["correction"]
            baseline.write_json(report.directory / "sampling.json", dict(plan=trainer.sampling_plan(), end_step=trainer.step,
                cumulative_counts={k:v.tolist() for k,v in trainer.counts.items()}, sampled_manual_labels=int(counts.sum()),
                sampled_correction_labels=int(counts[correction].sum()), unique_manual_rows_seen=int(np.count_nonzero(counts)),
                unique_correction_rows_seen=int(np.count_nonzero(counts[correction])), labels_per_branch=trainer.counters()["labels_seen_per_branch"],
                resampled_labels_are_not_independent_evidence=True, retained84_labels_used_for_training=False))
            report.save()
    trainer.model.check_frozen(content=True)
    trainer.check_teacher(content=True)
    report.data["distillation_counts"] = trainer.distillation_counts()
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "冻结bundle改变")
    report.data.update(final_metrics=current, worker_version=400, additional_updates=50,
                       teacher_forward_queries=trainer.teacher_forward_queries,teacher_optimizer_updates=0,
                       checkpoint_sha256=bc.file_hash(report.directory / "latest.pt"))
    baseline.write_json(report.directory / "diagnostics.json", dict(initial=initial, final=current, worker_version=400,
        source_worker_version=350, additional_updates=50, fixed_budget_latest_primary=True, checkpoint_selected_by_validation=False,
        dual_goal_correction_train_coverage=False, behavior_accepted=False, t06_approved=False,
        retention_plan=report.data["retention_plan"], confirmation_plan=report.data["confirmation_plan"]))
    within = current["retention_review"]["within_predeclared_retention_limit"]
    report.check("retention_review", "PASS" if within else "WARN", "相对350 step0原26局及两个目标参考双分支10项NLL增加≤0.02；超限保留，不延预算/选best")
    report.check("training", "PASS", "双头各固定50次更新，源350+50=400；冻结底座/WM/目标更新0、新环境步0")
    report.check("frozen_dependencies", "PASS", "独立格式latest/新优化器/三池计数/RNG及来源SHA共享；无best选择或旧文件回写")


def verify(args, report, inputs):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_library as gl
    import goal_continuation_repair as continuation
    import goal_retention_repair as repair
    trainer = make_trainer(inputs, args.device)
    report.data["teacher_binding"] = trainer.teacher_binding()
    path = baseline.project_path(args.checkpoint)
    digest = trainer.reference["retention_repair_check_report"]["sha256"]
    payload, _ = repair.read_checkpoint(path, trainer, digest)
    trainer.restore(payload)
    trained = legacy.read_json(path.parent / "report.json")
    legacy.accepted(trained, "train")
    gl.require(path.name == "latest.pt" and trainer.step == 50 and trained.get("repair_identity") == trainer.identity and
        trained.get("counters") == payload["counters"] and trained.get("worker_version") == 400 and
        trained.get("source_worker_version") == 350 and trained.get("additional_updates") == 50 and
        trained.get("repair_code") == report.data["repair_code"] and trained.get("options") == trainer.options and
        trained.get("backend") == bc.backend_info(args.device) and
        trained.get("retention_plan") == report.data["retention_plan"] and
        trained.get("confirmation_plan") == report.data["confirmation_plan"] and
        trained.get("source_inputs_before") == trained.get("source_inputs_after") and
        trained.get("new_env_steps") == 0 and trained.get("expert_labels_generated") is False and
        {"paired_training", "training", "frozen_dependencies", "source_inputs_unchanged"} <=
            {r["name"] for r in trained["checks"] if r["level"] == "PASS"} and
        trained.get("checkpoint_sha256") == bc.file_hash(path), "只验收本轮固定50步生产latest，不接受best/中途或验收副本")
    report.check("strict_load", "PASS", "生产350+50=400双头/优化器/三池计数/采样RNG严格恢复")
    t02.rejection(report, "legacy_format_guard", lambda: continuation.read_checkpoint(path), "独立续段修复")
    t02.rejection(report, "artifact_guard", lambda: trainer.restore(dict(payload, verification_artifact=True)), "合成验收")
    t02.rejection(report, "source_version_guard", lambda: trainer.restore(dict(payload, source_worker_version=300)), "源350")
    t02.rejection(report,"teacher_source_guard",lambda:trainer.restore(dict(payload,
        teacher_binding=dict(payload["teacher_binding"],checkpoint_sha256="wrong"))),"教师来源")
    t02.rejection(report,"teacher_counter_guard",lambda:trainer.restore(dict(payload,
        distillation_counts=dict(payload["distillation_counts"],cached_teacher_initial_queries=177))),"KL查询计数")
    changed_optimizers=copy.deepcopy(payload["optimizers"])
    changed_optimizers["goal"]["param_groups"][0]["lr"] *= 2
    t02.rejection(report,"learning_rate_guard",lambda:trainer.restore(dict(payload,
        optimizers=changed_optimizers)),"新Adam学习率")
    t02.rejection(report, "counter_guard", lambda: trainer.restore(dict(payload, counters=dict(payload["counters"], base_updates=1))), "计数")
    t02.rejection(report, "identity_guard", lambda: trainer.restore(dict(payload,
        input_identity=dict(payload["input_identity"], within_horizon_id="wrong"))), "身份")
    counts = copy.deepcopy(payload["sampling_counts"])
    counts["feedback_row"][0] += 1
    t02.rejection(report, "sample_count_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "计数")
    counts = copy.deepcopy(payload["sampling_counts"])
    donor = next(int(i) for i in trainer.data.manual_train_rows if counts["feedback_row"][i] > 0)
    counts["feedback_row"][donor] -= 1
    counts["feedback_row"][0] += 1  # trial0 is a worker episode; counts remain total-preserving.
    t02.rejection(report, "sample_split_guard", lambda: trainer.restore(dict(payload, sampling_counts=counts)), "混入留出/worker")
    heads = copy.deepcopy(payload["heads"])
    heads["goal"]["state_projection.weight"] = heads["goal"]["state_projection.weight"][:1]
    t02.rejection(report, "shape_guard", lambda: trainer.restore(dict(payload, heads=heads)), "权重不兼容")
    artifact_path = report.directory / "roundtrip_verification.pt"
    artifact = repair.save_checkpoint(trainer, artifact_path, digest, verification_artifact=True)
    other = make_trainer(inputs, args.device)
    again, _ = repair.read_checkpoint(artifact_path, other, digest)
    other.restore(again, allow_verification=True)
    restored = other.payload(True)
    for key in ("heads", "optimizers", "counters", "sampler_state", "rng_state", "sampling_counts", "input_identity", "options", "teacher_binding", "distillation_counts"):
        gl.require(t01.same(artifact[key], restored[key]), f"预算内恢复{key}不同")
    inference = repair.load_policy(path, trainer.data.cache, args.device)
    x, g, r, _ = trainer.data.feedback_batch(np.arange(80), args.device)
    with torch.no_grad(), gl.encoder_precision(args.device):
        for mode in ("goal", "no_goal", "zero_goal", "base"):
            gl.require(torch.equal(inference.action_dist(x,g,r,mode).probs, trainer.model.action_dist(x,g,r,mode).probs) and
                torch.equal(trainer.model.action_dist(x,g,r,mode).probs, other.model.action_dist(x,g,r,mode).probs), "恢复/独立推理分布不同")
        swapped = torch.as_tensor(trainer.data.fixed_goals[1-trainer.data.feedback_arrays["target"]], device=args.device)
        gl.require(torch.equal(inference.action_dist(x,swapped,r).probs, trainer.model.action_dist(x,swapped,r).probs), "交换目标分布不同")
    t02.rejection(report, "inference_artifact_guard", lambda: repair.load_policy(artifact_path, trainer.data.cache, args.device), "合成验收")
    report.check("roundtrip", "PASS", "双头/优化器/计数/三池RNG往返相同；独立推理无优化器，五分布同值")
    initial = legacy.read_json(baseline.project_path(args.check_dir) / "initial_metrics.json")
    metrics, _ = evaluation_outputs(trainer, report, initial)
    gl.require(t01.same(metrics, trained["final_metrics"]), "独立verify与生产latest最终指标不同")
    rng = bc.capture_rng(args.device)
    left = trainer.update(verification_only=True)
    bc.restore_rng(rng, args.device)
    right = other.update(verification_only=True)
    gl.require(t01.same(left,right) and trainer.teacher_binding() == other.teacher_binding() and
        trainer.distillation_counts() == other.distillation_counts() and trainer.counters() == other.counters() and
        t01.same(trainer.counts, other.counts) and t01.same(trainer.sampler.get_state(), other.sampler.get_state()) and
        all(t01.same(trainer.model.heads[k].state_dict(), other.model.heads[k].state_dict()) and
            t01.same(trainer.optimizers[k].state_dict(), other.optimizers[k].state_dict()) for k in ("goal", "no_goal")),
        "恢复后下一真实混合batch/优化器更新不同")
    trainer.model.check_frozen(content=True)
    other.model.check_frozen(content=True)
    trainer.check_teacher(content=True); other.check_teacher(content=True)
    gl.require(bc.bundle_id(trainer.data.cache.bundle) == trainer.data.cache.metadata["bundle_id"], "验收改变冻结bundle")
    report.data.update(counters=payload["counters"], optimizer_updates=4, verification_only=True,
        teacher_forward_queries=trainer.teacher_forward_queries+other.teacher_forward_queries,teacher_optimizer_updates=0,
        verification_updates_per_copy=1, verification_updates_not_saved_to_training=True, worker_version=400,
        additional_updates=50, model_id=inference.model_id, metrics=metrics, checkpoint_sha256=bc.file_hash(path))
    report.check("next_update_equivalence", "PASS", "两个内存副本各一次真实混合更新完全相同；源latest不回写，不计入生产50步")
    report.check("frozen_dependencies", "PASS", "底座/WM/目标未变，验收快照禁止训练/控制，生产计数仍350+50=400")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--diagnosis-dir", required=True)
    training = commands.add_parser("train")
    training.add_argument("--check-dir", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--check-dir", required=True)
    verification.add_argument("--checkpoint", required=True)
    for command in (checking, training, verification):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir")
        command.add_argument("--feedback-dir")
        command.add_argument("--no-feedback",action="store_true",help="仅由总流程统一打包，不改变实验")
    args = parser.parse_args()
    try:
        if args.command != "check":
            args.check_dir = str(baseline.project_path(args.check_dir))
            checked = legacy.read_json(Path(args.check_dir) / "report.json")
            legacy.accepted(checked, "check")
            args.diagnosis_dir = checked["arguments"]["diagnosis_dir"]
        args.diagnosis_dir = str(baseline.project_path(args.diagnosis_dir))
        source, protected, files = historical_inputs(args)
        if args.command != "check":
            protected.append(Path(args.check_dir))
            files += [Path(args.check_dir) / "report.json", *[Path(args.check_dir) / n for n in checked["check_artifacts"]]]
        if args.command == "verify":
            checkpoint = baseline.project_path(args.checkpoint)
            args.checkpoint = str(checkpoint)
            protected.append(checkpoint.parent)
            files += [checkpoint, checkpoint.parent / "report.json"]
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / (
            f"retention_repair_{args.command}_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        gl_paths = [p.resolve() for p in protected]
        if any(directory == p or p in directory.parents or directory in p.parents for p in gl_paths):
            parser.error("输出须独立于原模型/缓存/轨迹/目标/check/验收及快照，禁止覆盖")
        directory.mkdir(parents=True, exist_ok=False)
        args.output_dir = str(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    report.data.update(optimizer_updates=0, new_env_steps=0, worker_control_actions=0, intervention_actions=0,
        labels_generated=False, expert_labels_generated=False, behavior_accepted=False, t06_approved=False,
        dual_goal_correction_train_coverage=False,
        training_scope="independent fixed-budget factual development repair; no expert or behavior approval")
    print(f"OUTPUT_DIR={directory}", flush=True)
    before, rng = None, None
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(), "CUDA不可用")
            torch.cuda.set_device(device)
        files = list(dict.fromkeys(p.resolve() for p in files))
        before = {str(p): baseline.file_signature(p) for p in files}
        report.data["source_inputs_before"] = before
        bc.require_disk_space(directory, 128 * 1024**2)
        baseline.write_json(directory / "output_probe.json", dict(atomic_output_probe_only=True))
        gl.require(legacy.read_json(directory / "output_probe.json")["atomic_output_probe_only"] is True, "原子JSON预检失败")
        report.check("output_preflight", "PASS", "独立新目录/原子JSON/空间预检通过，旧目录拒绝覆盖")
        report.check("historical_identity", "PASS", "完整冲突诊断/80事实/生产350及旧400/check/train/verify与51旧代码SHA绑定，旧FAIL不改写")
        rng = bc.capture_rng(args.device)
        inputs = load_inputs(args, report, source)
        {"check":check, "train":train, "verify":verify}[args.command](args, report, inputs)
        gl.require(not any(n in ("minedojo", "mineclip") or n.startswith(("minedojo.", "mineclip.")) for n in sys.modules),
                   "离线修复不能导入MineDojo/MineCLIP")
        report.check("scope", "WARN", "独立350保留约束固定短修复；旧FAIL保留，目标1新训练纠偏缺失；只有10项保留通过才能自主确认，T06仍需真实门槛及人工复核")
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        if rng is not None:
            bc.restore_rng(rng, args.device)
        if before is not None:
            try:
                after = {name:baseline.file_signature(Path(name)) for name in before}
                report.data["source_inputs_after"] = after
                report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL",
                    "原模型/缓存/轨迹/目标/监督/诊断/验收及51个旧代码大小和修改时间不变；只写新目录")
            except OSError as error:
                report.check("source_inputs_unchanged", "FAIL", str(error))
        gc.collect()
    result = report.finish()
    if args.no_feedback:
        return result
    try:
        import t05_result_bundle as feedback
        sources = [("run",directory),("conflict_diagnosis",source["retention_diagnosis_dir"])]
        if args.command != "check":
            sources.append(("repair_check",Path(args.check_dir)))
        if args.command == "verify":
            sources.append(("repair_train",baseline.project_path(args.checkpoint).parent))
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT / "relevance_map/t05_feedback" / (
            directory.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in [*protected,directory]):
            raise ValueError("反馈包输出须独立于全部源目录与新结果目录")
        index = feedback.build_bundle(sources, export, directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export / 'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"[FAIL] feedback_export: {type(error).__name__}: {error}；原报告已保留", file=sys.stderr, flush=True)
        result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())
