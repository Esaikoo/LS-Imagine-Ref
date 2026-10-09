"""Diagnose frozen production350/400 on all184 accepted real factual states."""
import argparse
import copy
from datetime import datetime
import gc
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import t01_checkpoint_check as t01
import t02_goal_library as t02
import t05_goal_control as legacy
import t05_within_horizon_repair as source_repair
import t05_within_horizon_supervision as source_supervision

CODE = ("goal_within_horizon_conflict_diagnose.py", "scripts/t05_within_horizon_conflict_diagnose.py")


class Report(baseline.Report):
    def finish(self):
        levels = {r["level"] for r in self.data["checks"]}
        self.data["status"] = "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter()-self.started, 3)
        saved = self.save()
        failed = not saved or self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_WITHIN_HORIZON_CONFLICT_DIAGNOSE; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def production_binding(checkpoint, checked, trained, verified, payload, device):
    import goal_bc as bc
    import goal_library as gl
    import goal_within_horizon_repair as repair
    repair.validate_payload(payload, payload["input_identity"], payload["options"])
    legacy.accepted(checked, "check")
    legacy.accepted(trained, "train")
    legacy.accepted(verified, "verify")
    expected_counters = dict(step=50, goal_updates=50, no_goal_updates=50, labels_seen_per_branch=12800,
        optimizer_updates=100, base_updates=0, wm_updates=0, new_env_steps=0)
    gl.require(checkpoint.name == "latest.pt" and not payload.get("verification_artifact") and
        payload["counters"] == expected_counters and trained.get("counters") == verified.get("counters") == expected_counters and
        trained.get("worker_version") == verified.get("worker_version") == 400 and
        trained.get("additional_updates") == verified.get("additional_updates") == 50 and
        trained.get("optimizer_updates") == 100 and verified.get("optimizer_updates") == 4 and
        verified.get("verification_only") is True and verified.get("verification_updates_per_copy") == 1 and
        verified.get("verification_updates_not_saved_to_training") is True and
        baseline.project_path(verified["arguments"]["checkpoint"]) == checkpoint and
        trained.get("checkpoint_sha256") == verified.get("checkpoint_sha256") == bc.file_hash(checkpoint),
        "只接受固定350+50生产latest400与独立verify；best/验收副本或计数不同")
    check_dir = baseline.project_path(verified["arguments"]["check_dir"])
    for record in (checked, trained, verified):
        gl.require(record.get("repair_format") == repair.FORMAT and record.get("repair_code") == source_repair.code_identity() and
            record.get("repair_identity") == payload["input_identity"] and record.get("options") == payload["options"] and
            record.get("source_worker_version") == 350 and record.get("planned_repair_updates") == 50 and
            record.get("backend") == bc.backend_info(device) and record.get("source_inputs_before") is not None and
            record.get("source_inputs_before") == record.get("source_inputs_after") and
            record.get("new_env_steps") == record.get("worker_control_actions") == record.get("intervention_actions") == 0 and
            record.get("expert_labels_generated") is False and record.get("behavior_accepted") is False and
            record.get("t06_approved") is False and record.get("dual_goal_correction_train_coverage") is False and
            record.get("retained84_labels_used_for_training") is False and
            record.get("sampling_plan") == repair.sampling_plan(256) and
            record.get("retention_plan") == source_repair.retention_plan() == checked["retention_plan"] and
            record.get("confirmation_plan") == checked["confirmation_plan"],
            "400生产验收身份/预算/后端/计划或旧记录不同")
    gl.require(baseline.project_path(trained["arguments"]["check_dir"]) == check_dir and
        verified["arguments"]["supervision_dir"] == trained["arguments"]["supervision_dir"] == checked["arguments"]["supervision_dir"] and
        checked.get("counters", {}).get("step") == 0 and checked.get("worker_version") == 350 and
        checked["confirmation_plan"].get("requires_predeclared_retention_pass") is True and
        checked["confirmation_plan"].get("implementation_pending") is True and
        set(repair.CHECKS) <= {r["name"] for r in checked["checks"] if r["level"] == "PASS"} and
        {"paired_training", "training", "frozen_dependencies", "source_inputs_unchanged"} <=
            {r["name"] for r in trained["checks"] if r["level"] == "PASS"} and
        {"strict_load", "roundtrip", "next_update_equivalence", "frozen_dependencies", "source_inputs_unchanged"} <=
            {r["name"] for r in verified["checks"] if r["level"] == "PASS"},
        "需要完整400生产check/train/verify；只读诊断不能变成自主确认")


def historical_inputs(args):
    import goal_bc as bc
    import goal_library as gl
    import goal_within_horizon_repair as repair
    directory = baseline.project_path(args.repair_verify_dir)
    verified = legacy.read_json(directory / "report.json")
    legacy.accepted(verified, "verify")
    checkpoint = baseline.project_path(verified["arguments"]["checkpoint"])
    checked_dir = baseline.project_path(verified["arguments"]["check_dir"])
    checked, trained = (legacy.read_json(p / "report.json") for p in (checked_dir, checkpoint.parent))
    payload, dependencies = repair.read_checkpoint(checkpoint)
    production_binding(checkpoint, checked, trained, verified, payload, args.device)
    gl.require(dependencies["within_horizon_repair_check_report"] == checked_dir / "report.json",
               "400verify的check路径与生产依赖不同")
    source, protected, files = source_repair.historical_inputs(SimpleNamespace(
        supervision_dir=checked["arguments"]["supervision_dir"], device=args.device))
    gl.require(dependencies["source350_checkpoint"] == source["context"]["checkpoint"] and
        dependencies["source350_verify_report"] == source["context"]["verify_dir"] / "report.json" and
        dependencies["supervision_manifest"] == source["supervision_dir"] / "supervision_manifest.json",
        "400依赖必须绑定本次真实监督及原生产350路径")
    initial = legacy.read_json(checked_dir / "initial_metrics.json")
    metrics = legacy.read_json(directory / "metrics.json")
    retention = repair.retention_review(initial, metrics)
    gl.require(initial["step"] == 0 and metrics["step"] == 50 and
        initial == legacy.read_json(checked_dir / "metrics.json") and
        metrics == trained["final_metrics"] == verified["metrics"] == legacy.read_json(checkpoint.parent / "metrics.json") and
        initial == legacy.read_json(checkpoint.parent / "initial_metrics.json") and
        metrics["retention_review"] == retention == legacy.read_json(directory / "retention.json") ==
            legacy.read_json(checkpoint.parent / "retention.json") and
        checked["retention_plan"] == legacy.read_json(checked_dir / "retention_plan.json") and
        checked["confirmation_plan"] == source_repair.confirmation_plan(source) == legacy.read_json(checked_dir / "confirmation_plan.json") and
        metrics["behavior_accepted"] is False and metrics["t06_approved"] is False,
        "固定400最终指标/逐项保留失败/预声明计划或旧行为结论不同")
    for name, digest in checked["check_artifacts"].items():
        path = (checked_dir / name).resolve()
        gl.require(path.parent == checked_dir and not Path(name).is_absolute() and bc.file_hash(path) == digest,
                   f"原400check产物SHA不同：{name}")
        files.append(path)
    initial_queries = legacy.read_json(checked_dir / "queries.json")
    final_queries = legacy.read_json(directory / "queries.json")
    gl.require(final_queries == legacy.read_json(checkpoint.parent / "queries.json") and
        len(initial_queries) == len(final_queries) == 80 and
        legacy.read_json(directory / "retained_frame_metrics.json") == legacy.read_json(checkpoint.parent / "retained_frame_metrics.json"),
        "400生产/verify全部事实指标不同或遗漏")
    selected = {checked_dir: ("report.json", "initial_metrics.json", "metrics.json", "queries.json", "retained_frame_metrics.json", "retention_plan.json", "confirmation_plan.json"),
        checkpoint.parent: ("report.json", "initial_metrics.json", "metrics.json", "queries.json", "retained_frame_metrics.json", "retention.json", "sampling.json"),
        directory: ("report.json", "metrics.json", "queries.json", "retained_frame_metrics.json", "retention.json")}
    for folder, names in selected.items():
        protected.append(folder)
        files.extend(folder / n for n in names)
    files.extend([checkpoint, *dependencies.values(), *[ROOT / n for n in CODE]])
    source.update(repair_verify_dir=directory, repair_check_dir=checked_dir, repair_checkpoint=checkpoint,
        repair_payload=payload, repair_verified=verified, repair_checked=checked, repair_trained=trained,
        initial_metrics=initial, final_metrics=metrics, retention=retention,
        initial_queries=initial_queries, final_queries=final_queries,
        retained_metrics={"350":legacy.read_json(checked_dir / "retained_frame_metrics.json"),
                          "400":legacy.read_json(directory / "retained_frame_metrics.json")})
    return source, list(dict.fromkeys(p.resolve() for p in protected)), list(dict.fromkeys(p.resolve() for p in files))


def contracts(report, plan, source, device):
    import goal_within_horizon_conflict_diagnose as diagnosis
    first = plan[0]
    changes = (("incoming_guard", dict(incoming_action_frame=first["state_frame"])),
        ("remaining_guard", dict(remaining=0)), ("factual_action_guard", dict(actual_action=(first["actual_action"]+1)%12)),
        ("own_endpoint_guard", dict(endpoint_frame=84)), ("state_guard", dict(state_sha256="0"*64)),
        ("borrowed_goal_guard", dict(endpoint_goal_sha256="0"*64)),
        ("episode_guard", dict(episode_key=plan[16]["episode_key"])),
        ("split_guard", dict(episode_split="development_holdout")),
        ("pool_guard", dict(pool="within80")), ("training_approval_guard", dict(approved_for_training=True)),
        ("expert_guard", dict(expert_labels=True)), ("version_guard", dict(compared_versions=[350,401])),
        ("hindsight_deployment_guard", dict(actual_endpoint_is_deployment_input=True)),
        ("after_endpoint_query_guard", dict(state_frame=80, incoming_action_frame=81, remaining=0)))
    for name, delta in changes:
        t02.rejection(report, name, lambda d=delta:diagnosis.validate_query(dict(first, **d), first), "查询真实状态")
    for name, rows in (("missing_query_guard", plan[:-1]), ("duplicate_query_guard", [plan[0], *plan[:-1]]),
                       ("query_order_guard", [plan[1], plan[0], *plan[2:]])):
        t02.rejection(report, name, lambda r=rows:diagnosis.validate_queries(r, plan), "184查询")
    altered = dict(source["repair_payload"], verification_artifact=True)
    t02.rejection(report, "inference_artifact_guard", lambda:production_binding(source["repair_checkpoint"],
        source["repair_checked"], source["repair_trained"], source["repair_verified"], altered, device), "合成验收")
    t02.rejection(report, "best_checkpoint_guard", lambda:production_binding(source["repair_checkpoint"].with_name("best.pt"),
        source["repair_checked"], source["repair_trained"], source["repair_verified"], source["repair_payload"], device), "生产latest400")
    verified = copy.deepcopy(source["repair_verified"])
    verified["counters"]["step"] = 51
    t02.rejection(report, "verification_update_guard", lambda:production_binding(source["repair_checkpoint"],
        source["repair_checked"], source["repair_trained"], verified, source["repair_payload"], device), "生产latest400")
    report.check("diagnostic_contracts", "PASS", "184事实自身状态/终点/incoming/预算/来源/划分/版本及负守卫；内存副本不写成轨迹")


def forward(policy, state, own, fixed, swapped, remaining):
    import numpy as np
    import goal_library as gl
    probabilities, raw = {}, {}
    for condition, goal, mode in (("actual_endpoint",own,"goal"), ("fixed_goal",fixed,"goal"),
        ("swapped_goal",swapped,"goal"), ("no_goal",fixed,"no_goal"),
        ("zero_goal",fixed,"zero_goal"), ("base",fixed,"base")):
        probabilities[condition], raw[condition] = source_supervision.distribution(policy,state,goal,remaining,mode)
    for mode in ("no_goal", "zero_goal", "base"):
        for goal in (own, swapped):
            p, r = source_supervision.distribution(policy,state,goal,remaining,mode)
            gl.require(np.array_equal(p,probabilities[mode]) and np.array_equal(r,raw[mode]),
                       "no_goal/zero/base分布或raw依赖目标")
    return probabilities, raw


def within_roundtrip(row, original, probabilities, raw, metrics, version):
    import numpy as np
    import goal_library as gl
    import goal_within_horizon_conflict_diagnose as diagnosis
    expected = dict(trial_index=row["run"], state_frame=row["state_frame"], incoming_action_frame=row["incoming_action_frame"],
        target=row["target"], episode_split=row["episode_split"], actual_action=row["actual_action"], remaining=row["remaining"],
        endpoint_frame=80, branch=row["branch"], decision_source=row["decision_source"],
        correction_applied=row["correction_applied"], state_sha256=row["state_sha256"],
        endpoint_goal_sha256=row["endpoint_goal_sha256"], fixed_goal_sha256=row["fixed_goal_sha256"],
        worker_version=350, evaluated_worker_version=version, repair_step=version-350)
    gl.require(all(original.get(k) == v for k,v in expected.items()), "原350/400真实80帧查询身份不同")
    maximum = 0.
    for c in diagnosis.CONDITIONS:
        p, r = np.asarray(original["probabilities"][c]), np.asarray(original["raw_preferences"][c])
        gl.require(p.shape == r.shape == (12,) and np.allclose(p,probabilities[c],atol=3e-5,rtol=3e-5) and
            np.allclose(r,raw[c],atol=3e-5,rtol=3e-5) and original["metrics"][c+"_mode"] == metrics[c+"_mode"],
            "原350/400完整概率/raw/mode重现不同")
        maximum = max(maximum, float(np.abs(p-probabilities[c]).max()))
    gl.require(set(original["metrics"]) == set(metrics) and all(
        (type(v) is bool and v == original["metrics"][k]) or
        (type(v) is not bool and np.isclose(v,original["metrics"][k],atol=3e-5,rtol=3e-5)) for k,v in metrics.items()),
        "原350/400六种事实指标重现不同")
    return maximum


def retained_roundtrip(rows, source):
    import numpy as np
    import goal_library as gl
    maximum = 0.
    for version in ("350", "400"):
        saved = source["retained_metrics"][version]
        gl.require(len(saved) == 208, "原参考/84帧保留评价缺失")
        for pool, saved_pool in (("reference","reference"), ("retained84","retained84_evaluation")):
            selected = [r for r in rows if r["pool"] == pool]
            for goal_source, goal_condition in (("own_real_endpoint","actual_endpoint"), ("fixed_old_targets","fixed_goal")):
                original = [r for r in saved if r["pool"] == saved_pool and r["goal_source"] == goal_source]
                gl.require(len(original) == len(selected), "原保留事实重复或遗漏")
                for row, previous in zip(selected,original):
                    gl.require(previous["run" if pool == "reference" else "trial_index"] == row["run"] and
                        previous["frame"] == row["state_frame"] and previous["next_action_frame"] == row["incoming_action_frame"] and
                        previous["remaining"] == row["remaining"] and previous["label"] == row["actual_action"] and
                        previous["split"] == row["episode_split"] and
                        (pool != "reference" or previous["target"] == row["target"]), "原保留状态/动作/划分或预算不同")
                    m = row["versions"][version]["metrics"]
                    for name, condition in (("goal",goal_condition), ("no_goal","no_goal"), ("zero_goal","zero_goal"),
                                            ("base","base"), ("swapped_goal","swapped_goal")):
                        p, nll = m[condition+"_factual_probability"], m[condition+"_factual_nll"]
                        gl.require(np.isclose(p,previous[name+"_probability"],atol=3e-5,rtol=3e-5) and
                            np.isclose(nll,previous[name+"_nll"],atol=3e-5,rtol=3e-5) and
                            m[condition+"_mode"] == previous[name+"_mode"] and
                            m[condition+"_mode_matches_factual"] == previous[name+"_match"],
                            "原350/400参考或84帧事实概率/NLL/mode重现不同")
                        maximum = max(maximum, abs(p-previous[name+"_probability"]))
    return maximum


def retention_roundtrip(rows, source):
    import numpy as np
    import goal_library as gl
    maximum = 0.
    for version, metrics in (("350",source["initial_metrics"]), ("400",source["final_metrics"])):
        for target in (0,1):
            selected = [r for r in rows if r["pool"] == "reference" and r["episode_split"] == "validation" and r["target"] == target]
            gl.require(len(selected) == 16, "原参考每目标完整16留出事实缺失")
            for goal_source, condition in (("own_real_endpoint","actual_endpoint"), ("fixed_old_targets","fixed_goal")):
                for name in ("goal","no_goal"):
                    actual = float(np.mean([r["versions"][version]["metrics"][(condition if name == "goal" else name)+"_factual_nll"] for r in selected]))
                    expected = metrics["reference"][goal_source]["by_split"]["validation"]["by_target"][str(target)][name+"_nll"]
                    gl.require(np.isclose(actual,expected,atol=3e-5,rtol=3e-5), "原参考逐状态复算保留NLL不同")
                    maximum = max(maximum,abs(actual-expected))
    return dict(original_ten_checks=source["retention"], recomputed_reference_comparisons=16,
        maximum_reference_mean_nll_error=maximum,
        original_full26_role="two accepted aggregate checks reproduced from original metrics; full26 states not queried here",
        within_predeclared_retention_limit=source["retention"]["within_predeclared_retention_limit"],
        failed_comparisons=[k for k,v in source["retention"]["comparisons"].items() if not v["within_limit"]],
        confirmation_requires_retention_pass=True, autonomous_confirmation_executed=False,
        diagnostic_pass_does_not_clear_retention_failures=True, gate_changed=False, t06_approved=False)


def analyze(args, report, source):
    import numpy as np
    import torch
    import goal_bc as bc
    import goal_continuation_repair as continuation
    import goal_library as gl
    import goal_within_horizon_conflict_diagnose as diagnosis
    import goal_within_horizon_repair as repair
    import goal_within_horizon_supervision as supervision
    class Audit:
        directory = report.directory
        def __init__(self):
            self.data = {}
        def check(self,name,level,detail):
            gl.require(level != "FAIL", f"原已验收只读加载失败：{name}: {detail}")
    data, _, _, _, source350 = source_repair.load_inputs(SimpleNamespace(command="verify",
        check_dir=str(source["repair_check_dir"]),device=args.device), Audit(),source)
    policies = {350:continuation.load_policy(source["context"]["checkpoint"],data.cache,args.device),
                400:repair.load_policy(source["repair_checkpoint"],data.cache,args.device)}
    gl.require(policies[350].worker_version == 350 and policies[400].worker_version == 400 and
        policies[350].identity == source350["input_identity"] and policies[400].identity == data.identity() and
        policies[350].model_id == source["supervision_record"]["model_id"] and
        policies[400].model_id == source["repair_verified"]["model_id"] and
        policies[350].base_hash == policies[400].base_hash and
        all(policies[350].identity[k] == policies[400].identity[k] for k in
            ("cache_id","bundle_id","fixed_goals_id","feature_dim","goal_dim","action_dim","horizon")),
        "需要同冻结底座/WM/两目标与已验收身份的生产350/400独立推理")
    plan = diagnosis.query_plan(data)
    names = source["feedback_manifest"]["design"]["action_names"]
    report.data.update(diagnostic_format=diagnosis.FORMAT,comparison_protocol=diagnosis.PROTOCOL,
        model_ids={str(v):p.model_id for v,p in policies.items()},
        diagnostic_code={n:bc.file_hash(ROOT/n) for n in CODE},
        accepted_source_code=source_repair.code_identity(),backend=bc.backend_info(args.device),
        input_identity=data.identity(),original_numerical_gate=source["repair_checked"]["original_numerical_gate"],
        original_retention_plan=source["repair_checked"]["retention_plan"],original_confirmation_plan=source["repair_checked"]["confirmation_plan"])
    report.check("strict_frozen_versions","PASS","生产350与固定350+50=400同底座/WM/目标独立推理；无优化器/RSSM或候选模型")
    contracts(report,plan,source,args.device)
    modules = list(policies.values())
    versions = [{n:p._version for n,p in m.named_parameters()} for m in modules]
    digests = [gl.tensor_digest(m.state_dict(),{}) for m in modules]
    bundle = bc.bundle_id(data.cache.bundle)
    rng = bc.capture_rng(args.device)
    fixed_before = data.fixed_goals.copy()
    rows, maximum = [], 0.
    by_pool = {p:a for p,a,_ in diagnosis.pools(data)}
    with torch.no_grad():
        for identity in plan:
            row = dict(identity)
            array, index = by_pool[row["pool"]], row["pool_row"]
            state = torch.as_tensor(array["features"][index:index+1].copy(),device=args.device)
            own, fixed, swapped = array["goals"][index], data.fixed_goals[row["target"]], data.fixed_goals[1-row["target"]]
            values = {}
            for version,policy in policies.items():
                p,r = forward(policy,state,own,fixed,swapped,row["remaining"])
                m = supervision.factual_metrics(p,r,row["actual_action"])
                if row["pool"] == "within80":
                    original = source["initial_queries" if version == 350 else "final_queries"][index]
                    maximum = max(maximum,within_roundtrip(row,original,p,r,m,version))
                    if version == 350:
                        # Source supervision predates repair metadata; its actual model is350.
                        original = dict(data.feedback_queries[index],evaluated_worker_version=350,repair_step=0)
                        maximum = max(maximum,within_roundtrip(row,original,p,r,m,350))
                values[str(version)] = dict(probabilities={c:v.tolist() for c,v in p.items()},
                    raw_preferences={c:v.tolist() for c,v in r.items()},metrics=m)
            row.update(versions=values,comparison=diagnosis.compare(values,names),
                actual_endpoint_vs_fixed_cosine_distance=float(1-np.dot(own,fixed)))
            rows.append(row)
            if len(rows)%16 == 0 or len(rows) == 184:
                print(f"[CONFLICT DIAGNOSE {len(rows)}/184] pool={row['pool']} frame={row['state_frame']} remaining={row['remaining']}",flush=True)
    maximum = max(maximum,retained_roundtrip(rows,source))
    altered = copy.deepcopy(rows[0]["versions"])
    altered["400"]["raw_preferences"]["base"][0] += 1
    t02.rejection(report,"base_version_guard",lambda:diagnosis.compare(altered,names),"同状态350/400冻结底座")
    retention = retention_roundtrip(rows,source)
    summary = diagnosis.summarize(rows,plan,names)
    summary.update(retention=retention, original_numerical_gate=source["repair_checked"]["original_numerical_gate"],
        source_model_ids=report.data["model_ids"], probability_roundtrip_tolerance=dict(atol=3e-5,rtol=3e-5),
        maximum_probability_error=maximum, old84_facts_used_for_fixed400_training=False,
        raw_preference_interpretation="offset-invariant pair margins and residual changes; no parameter-level causal claim")
    for module,digest,version in zip(modules,digests,versions):
        module.check_frozen(content=True)
        gl.require(gl.tensor_digest(module.state_dict(),{}) == digest and
            {n:p._version for n,p in module.named_parameters()} == version and
            all(not p.requires_grad and p.grad is None for p in module.parameters()), "只读诊断改变参数/版本/梯度")
    gl.require(bc.bundle_id(data.cache.bundle) == bundle and np.array_equal(fixed_before,data.fixed_goals) and
        t01.same(rng,bc.capture_rng(args.device)), "只读诊断改变冻结WM/目标或查询RNG")
    diagnosis.validate_queries(rows,plan)
    baseline.write_json(report.directory/"queries.json",rows)
    baseline.write_csv(report.directory/"frame_metrics.csv",diagnosis.flat_rows(rows,names))
    baseline.write_json(report.directory/"diagnostics.json",summary)
    baseline.write_json(report.directory/"regressions.json",diagnosis.regressions(rows))
    baseline.write_json(report.directory/"retention.json",retention)
    manifest = dict(format=diagnosis.FORMAT,comparison_protocol=diagnosis.PROTOCOL,query_plan=plan,
        source_verify_report_sha256=bc.file_hash(source["repair_verify_dir"]/"report.json"),
        source_checkpoint_sha256=bc.file_hash(source["repair_checkpoint"]),
        source350_checkpoint_sha256=bc.file_hash(source["context"]["checkpoint"]),
        model_ids=report.data["model_ids"],input_identity=data.identity(),diagnostic_code=report.data["diagnostic_code"],
        source_artifacts_sha256=report.data["source_artifacts_sha256"],
        artifacts={p.name:bc.file_hash(p) for p in report.directory.iterdir() if p.is_file() and p.name != "report.json"},
        query_rows=184,episodes=34,diagnostic_distributions_saved=2208,
        optimizer_updates=0,new_env_steps=0,worker_control_actions=0,intervention_actions=0,
        approved_for_training=False,behavior_accepted=False,t06_approved=False)
    manifest["diagnosis_id"] = gl.tensor_digest({},manifest)
    baseline.write_json(report.directory/"diagnosis_manifest.json",manifest)
    report.data.update(diagnosis_id=manifest["diagnosis_id"],query_rows=184,episodes=34,
        diagnostic_distribution_rows=2208,main_comparison_forward_calls=2208,target_invariance_forward_calls=2208,
        maximum_probability_error=maximum,source_reference_scalar_distribution_checks=1280,
        source_retained84_scalar_distribution_checks=800,source_within80_full_distribution_checks=960,
        source_supervision_full_distribution_checks=480,
        within_predeclared_retention_limit=retention["within_predeclared_retention_limit"],
        retention_failed_comparisons=retention["failed_comparisons"],
        same_state_within_query=True,same_state_across_trials=False)
    report.check("real_query_alignment","PASS","原参考64/预算内80/旧续段40完整184真实状态；自身incoming/预算/终点与整局划分不变")
    report.check("recorded_forward_roundtrip","PASS",f"350/400已保存完整80事实六分布及参考/84帧标量概率/NLL/mode重现；最大概率误差={maximum:.6g}")
    report.check("same_state_goal_and_version_comparison","PASS","184状态×两版本×六条件完整概率/raw/事实排名与动作间偏好差；底座逐值同，后三条件严格目标不变")
    report.check("retention_review_preserved","PASS","预声明10项按原指标复算，8参考项另由完整同状态留出复核；超限不清除，原26局仅复算已验收汇总")
    report.check("retention_for_confirmation","PASS" if retention["within_predeclared_retention_limit"] else "WARN",
        f"原400保留超限项={len(retention['failed_comparisons'])}；诊断不启动自主确认、不改变门槛")
    report.check("correction_coverage","WARN","原目标1预算内训练纠偏缺失仍保留；事实拟合与noop/上转/jump/下转概率变化不等于专家准确率")
    report.check("no_training","PASS","参数/版本/梯度/WM/目标/查询RNG未变；无优化器/RSSM/环境，更新/新环境步/动作执行0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-verify-dir",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--feedback-dir")
    args = parser.parse_args()
    args.command = "analyze"
    args.repair_verify_dir = str(baseline.project_path(args.repair_verify_dir))
    try:
        source,protected,files = historical_inputs(args)
        directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT/"relevance_map/t05_outputs"/(
            "within_horizon_conflict_diagnose_"+datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(directory == p or p in directory.parents or directory in p.parents for p in protected):
            parser.error("输出必须独立于全部模型/缓存/历史/目标/验收目录，禁止覆盖")
        directory.mkdir(parents=True,exist_ok=False)
        args.output_dir = str(directory)
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f"[FAIL] input_or_output: {type(error).__name__}: {error}",file=sys.stderr,flush=True)
        return 2
    report = Report(directory,args)
    report.data.update(optimizer_updates=0,new_env_steps=0,worker_control_actions=0,intervention_actions=0,
        labels_generated=False,expert_labels_generated=False,approved_for_training=False,
        autonomous_confirmation_executed=False,behavior_accepted=False,t06_approved=False)
    print(f"OUTPUT_DIR={directory}",flush=True)
    before,rng,hashes = None,None,None
    try:
        import torch
        import goal_bc as bc
        import goal_library as gl
        device = torch.device(args.device)
        if device.type == "cuda":
            gl.require(torch.cuda.is_available(),"CUDA不可用")
            torch.cuda.set_device(device)
        before = {str(p):baseline.file_signature(p) for p in files}
        report.data["source_inputs_before"] = before
        hashes = {str(p):bc.file_hash(p) for p in files}
        report.data["source_artifacts_sha256"] = hashes
        bc.require_disk_space(directory,256*1024**2)
        report.require_writable()
        baseline.write_json(directory/"output_probe.json",dict(atomic_output_probe_only=True))
        gl.require(legacy.read_json(directory/"output_probe.json")["atomic_output_probe_only"] is True,"原子JSON预检失败")
        report.check("output_preflight","PASS","独立新目录/原子JSON与空间预检通过，旧目录拒绝覆盖")
        report.check("historical_identity","PASS","生产350/固定400/check/train/verify/真实184事实及49旧代码SHA绑定；旧行为与保留失败保持")
        report.require_writable()
        rng = bc.capture_rng(args.device)
        analyze(args,report,source)
        gl.require(not any(n in ("minedojo","mineclip") or n.startswith(("minedojo.","mineclip.")) for n in sys.modules),
                   "只读冲突诊断不能导入MineDojo/MineCLIP")
        report.check("scope","WARN","同状态事实排名与raw变化仅作离线描述；不识别参数因果收益、不选择新监督或best、不训练/执行400确认/批准T06")
    except (Exception,KeyboardInterrupt) as error:
        baseline.record_exception(report,error)
    finally:
        if rng is not None:
            bc.restore_rng(rng,args.device)
        if before is not None:
            try:
                after = {name:baseline.file_signature(Path(name)) for name in before}
                after_hashes = {name:bc.file_hash(Path(name)) for name in hashes} if hashes is not None else None
                report.data["source_inputs_after"] = after
                report.data["source_sha256_unchanged"] = hashes is not None and hashes == after_hashes
                report.check("source_inputs_unchanged","PASS" if before == after and hashes is not None and hashes == after_hashes else "FAIL",
                    "全部源模型/缓存/真实历史/目标/验收及49旧代码大小/修改时间/SHA不变；只写独立新目录")
            except OSError as error:
                report.check("source_inputs_unchanged","FAIL",str(error))
        gc.collect()
    result = report.finish()
    try:
        import t05_result_bundle as feedback
        sources = [("run",directory),("repair_verify",source["repair_verify_dir"]),
            ("repair_train",source["repair_checkpoint"].parent),("repair_check",source["repair_check_dir"]),
            ("supervision",source["supervision_dir"]),("feedback_evaluate",source["feedback_dir"]),
            ("feedback_check",source["feedback_checked_dir"]),("diagnosis",source["diagnosis_directory"]),
            ("source_evaluate",source["eval_dir"]),("source350_verify",source["context"]["verify_dir"])]
        export = baseline.project_path(args.feedback_dir) if args.feedback_dir else ROOT/"relevance_map/t05_feedback"/(
            directory.name+"_"+datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        if any(export == p or p in export.parents or export in p.parents for p in [*protected,directory]):
            raise ValueError("反馈包输出须独立于全部源目录与新结果目录")
        index = feedback.build_bundle(sources,export,directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export/'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}",flush=True)
    except (Exception,KeyboardInterrupt) as error:
        print(f"[FAIL] feedback_export: {type(error).__name__}: {error}；原报告已保留",file=sys.stderr,flush=True)
        result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())
