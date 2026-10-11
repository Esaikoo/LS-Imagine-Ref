"""One fixed retention repair and, if eligible, preregistered autonomous confirmation.

Uses subprocesses to release inference/training memory between stages. Never
retries a stage, resumes a partial run, selects a checkpoint, or changes a gate.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "ls_imagine_retention_pipeline_v1"
STAGES = ("repair_check", "repair_train", "repair_verify", "confirmation_check", "confirmation_evaluate")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b""):
        result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def run_stage(name, script, arguments, directory, state):
    output = directory/name
    require(not output.exists(), "阶段结果已经存在，禁止续跑或覆盖："+name)
    command = [sys.executable, str(ROOT/"scripts"/script), *arguments, "--output-dir", str(output)]
    entry = dict(name=name, command=command, output_dir=str(output), started_at=datetime.now().isoformat(),
                 status="running")
    state["stages"].append(entry); write(directory/"report.json", state)
    print("[PIPELINE] " + name, flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    if name == "confirmation_evaluate":
        environment["MINEDOJO_HEADLESS"] = "1"
    tail = []
    log_path = directory/(name+".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
                tail.append(line.rstrip("\n")); tail = tail[-40:]
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            entry.update(status="interrupted", log_path=str(log_path), log_tail=tail)
            write(directory/"report.json", state)
            raise
    entry.update(returncode=code, log_path=str(log_path), log_sha256=digest(log_path), log_tail=tail,
                 finished_at=datetime.now().isoformat())
    report_path = output/"report.json"
    if report_path.is_file():
        record = read(report_path)
        entry.update(report_path=str(report_path), report_sha256=digest(report_path), status=record.get("status"))
    else:
        entry["status"] = "failed"
        record = {}
    write(directory/"report.json", state)
    require(code == 0 and record.get("status") in ("passed", "passed_with_warnings"),
            "阶段未通过，全部后续阶段停止，无重试："+name)
    return record


def pipeline(args, directory, state):
    common = ["--device",args.device]
    check = run_stage("repair_check", "t05_retention_repair.py", ["check", "--diagnosis-dir",args.diagnosis_dir,
        *common,"--no-feedback"], directory, state)
    train = run_stage("repair_train", "t05_retention_repair.py", ["train", "--check-dir",str(directory/"repair_check"),
        *common,"--no-feedback"], directory, state)
    verify = run_stage("repair_verify", "t05_retention_repair.py", ["verify", "--check-dir",str(directory/"repair_check"),
        "--checkpoint",str(directory/"repair_train/latest.pt"),*common,"--no-feedback"], directory, state)
    before = read(directory/"repair_check/initial_metrics.json")
    final = read(directory/"repair_train/metrics.json")
    reviewed = read(directory/"repair_verify/retention.json")
    comparisons = reviewed["comparisons"]
    require(len(comparisons) == 10 and reviewed.get("predeclared_max_absolute_nll_increase") == .02 and
        reviewed == final["retention_review"] == read(directory/"repair_train/retention.json") and
        final == train["final_metrics"] == verify["metrics"] == read(directory/"repair_verify/metrics.json") and
        before == read(directory/"repair_train/initial_metrics.json") and
        check["confirmation_plan"] == train["confirmation_plan"] == verify["confirmation_plan"] and
        check["confirmation_plan"]["requires_predeclared_retention_pass"] is True and
        all(type(item.get("within_limit")) is bool and item["within_limit"] == (item["increase"] <= .02)
            for item in comparisons.values()) and
        reviewed["within_predeclared_retention_limit"] == all(item["within_limit"] for item in comparisons.values()),
        "独立验收的10项保留结果/预声明门槛或最终模型不同")
    state.update(retention=reviewed, retention_passed=reviewed["within_predeclared_retention_limit"],
        checkpoint_path=str(directory/"repair_train/latest.pt"), checkpoint_sha256=digest(directory/"repair_train/latest.pt"),
        model_id=verify["model_id"], experiment_family="retention_constrained_source350_fixed50",
        additional_updates=50, failed_retention_checks=[k for k,v in comparisons.items() if not v["within_limit"]])
    write(directory/"retention.json",reviewed)
    if not state["retention_passed"]:
        state.update(status="retention_not_passed", engineering_status="passed_with_warnings",
            failed_rules=state["failed_retention_checks"], next_required_action="review retention failures; no autonomous confirmation executed")
        return 3
    run_stage("confirmation_check", "t05_retention_control.py", ["check", "--repair-verify-dir",
        str(directory/"repair_verify"), *common], directory, state)
    state["autonomous_confirmation_requested"] = True
    evaluated = run_stage("confirmation_evaluate", "t05_retention_control.py", ["evaluate", "--repair-verify-dir",
        str(directory/"repair_verify"), "--check-dir",str(directory/"confirmation_check"), *common], directory, state)
    gate = read(directory/"confirmation_evaluate/gate.json")
    require(gate == evaluated["numerical_gate"] and type(gate.get("passed")) is bool and
        gate.get("human_review_pending") is True and gate.get("t06_approved") is False and
        evaluated.get("planned_trials") == 30 and evaluated.get("optimizer_updates") == 0 and
        evaluated.get("intervention_actions") == 0 and evaluated.get("worker_version") == 400,
        "真实自主确认/gate身份或冻结计数不同")
    state.update(autonomous_confirmation_executed=True,numerical_behavior_gate=gate,
        numerical_behavior_passed=gate["passed"], engineering_status="passed_with_warnings",
        ready_for_human_t06_review=gate["passed"], failed_rules=gate["failed_rules"],
        status="human_review_required" if gate["passed"] else "behavior_not_passed",
        next_required_action="review all30 videos, starts, position and orientation against the original T06 gate"
            if gate["passed"] else "review the failed per-target behavioral rules; no automatic retraining")
    write(directory/"gate.json",gate)
    return 0 if gate["passed"] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--feedback-dir")
    args = parser.parse_args()
    args.diagnosis_dir = str(resolve(args.diagnosis_dir))
    source = Path(args.diagnosis_dir)
    directory = resolve(args.output_dir) if args.output_dir else ROOT/"relevance_map/t05_outputs"/(
        "retention_pipeline_"+datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    try:
        require(source.is_dir(), "诊断目录不存在")
        require(directory != source and source not in directory.parents and directory not in source.parents,
                "输出必须独立于原诊断")
        require(not directory.exists(), "总流程结果目录已经存在，禁止覆盖或续跑")
        directory.mkdir(parents=True,exist_ok=False)
    except (OSError,ValueError) as error:
        print(f"[FAIL] input_or_output: {error}",file=sys.stderr,flush=True); return 2
    state = dict(format=FORMAT,command="run",arguments=vars(args),status="running",started_at=datetime.now().isoformat(),
        stages=[],planned_stages=list(STAGES),fixed_additional_updates=50,maximum_new_env_steps=2400,
        retention_passed=False,numerical_behavior_passed=False,autonomous_confirmation_executed=False,
        ready_for_human_t06_review=False,human_review_pending=True,behavior_accepted=False,t06_approved=False,
        no_retries_or_budget_extension=True,checkpoint_selection="fixed final production latest only",
        source_code_sha256={n:digest(ROOT/n) for n in ("goal_retention_repair.py","scripts/t05_retention_repair.py",
            "goal_retention_control.py","scripts/t05_retention_control.py","scripts/t05_retention_pipeline.py")})
    print(f"OUTPUT_DIR={directory}",flush=True)
    started=time.perf_counter(); write(directory/"report.json",state)
    try:
        result=pipeline(args,directory,state)
    except (Exception,KeyboardInterrupt) as error:
        state.update(status="failed",engineering_status="failed",error=f"{type(error).__name__}: {error}",
            next_required_action="inspect the failed stage; source artifacts and failed output are preserved")
        (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(f"[FAIL] pipeline: {state['error']}",file=sys.stderr,flush=True); result=2
    state.update(elapsed_seconds=round(time.perf_counter()-started,3),finished_at=datetime.now().isoformat(),
        unexecuted_stages=[n for n in STAGES if n not in {s["name"] for s in state["stages"]}])
    stage_records={n:read(directory/n/"report.json") for n in STAGES if (directory/n/"report.json").is_file()}
    state["reported_new_env_steps"]=sum(r.get("new_env_steps",0) for r in stage_records.values())
    state["autonomous_confirmation_executed"]=state["reported_new_env_steps"] > 0
    state["reported_costs_by_stage"]={n:{k:r.get(k) for k in ("optimizer_updates","new_env_steps",
        "teacher_forward_queries","teacher_optimizer_updates","worker_action_queries","worker_control_actions")}
        for n,r in stage_records.items()}
    write(directory/"report.json",state)
    try:
        import t05_result_bundle as feedback
        sources=[("run",directory),*[(n,directory/n) for n in STAGES if (directory/n/"report.json").is_file()],("conflict_diagnosis",source)]
        export=resolve(args.feedback_dir) if args.feedback_dir else ROOT/"relevance_map/t05_feedback"/(
            directory.name+"_"+datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
        index=feedback.build_bundle(sources,export,directory.name)
        print(f"FEEDBACK_FILE={index['json_path']}\nFEEDBACK_INDEX={export/'bundle_index.json'}\nFEEDBACK_ZIP={index['zip_path']}",flush=True)
    except (Exception,KeyboardInterrupt) as error:
        state["feedback_export_error"]=f"{type(error).__name__}: {error}"; write(directory/"report.json",state)
        print(f"[FAIL] feedback_export: {error}; all stage artifacts preserved",file=sys.stderr,flush=True); result=2
    print(f"[PIPELINE RESULT] status={state['status']}; retention_passed={int(state['retention_passed'])}; "
          f"numerical_behavior_passed={int(state['numerical_behavior_passed'])}; human_review_pending=1; t06_approved=0",flush=True)
    if state.get("failed_rules"):
        print("FAILED_RULES="+json.dumps(state["failed_rules"],ensure_ascii=False),flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
