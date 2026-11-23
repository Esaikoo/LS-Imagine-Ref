"""Bounded, pickle-free decision archives. All frame indices refer to pre-action observations."""
import json
from pathlib import Path
import numpy as np
from .visualization import SOURCES, _json


def write_keyframes(folder, snap, references):
    arrays, records = {}, {}
    def pack(event):
        step=str(event["row"]["step"])
        if step not in records:
            record={}
            for name,value in event.items():
                if isinstance(value,np.ndarray):
                    key=f"s{step}_{name}"
                    arrays[key]=value
                    record[name]={"array":key}
                else:
                    record[name]=_json(value)
            records[step]=record
        return step
    meta=dict(schema=1,a=[pack(e) for e in snap["a"]],c=[pack(e) for e in snap["c"]],
              goal=pack(snap["goal"]) if snap["goal"] else None,
              groups={reason:pack(e) for reason,e in snap["groups"].items()}, records=records,
              references=[],terminal=None)
    for i,(name,image) in enumerate(references):
        key=f"reference_{i}"
        arrays[key]=image
        meta["references"].append([name,key])
    terminal=snap.get("terminal")
    if terminal:
        meta["terminal"]={k:v for k,v in terminal.items() if k!="rgb"}
        if terminal.get("rgb") is not None:
            arrays["terminal_rgb"]=terminal["rgb"]
    arrays["schema_json"]=np.array(json.dumps(meta,allow_nan=False,ensure_ascii=True))
    np.savez_compressed(folder/"decision_frames.npz",**arrays)


def load_episode(folder,replay=None):
    folder=Path(folder)
    s=json.loads((folder/"summary.json").read_text())
    with np.load(folder/"episode_trace.npz",allow_pickle=False) as archive:
        trace=dict(archive)
    rows=[]
    for i in range(len(trace["phase"])):
        r={k:v[i].tolist() for k,v in trace.items() if k in (
            "pose","pose_ready","local_valid","segment","memory_available","phase","option","source",
            "actor","control_kind","override","goal_state","goal_gate","goal_conflict","goal_model_update","reward","success")}
        r["local_ready"]=r.pop("local_valid",r["pose_ready"])
        r.update(step=i,actor=r.get("actor",-1),override=r.get("override"))
        rows.append(r)
    state=dict(grid=trace["map_grid"],origin=trace["map_origin"],resolution=trace["map_resolution"],
               history_frames=trace.get("history_frames",[]),history_valid=trace.get("history_valid",[]),
               active_segment=s.get("active_segment",0))
    for key in ("history_meta", "history_query_weights", "history_query_tick"):
        if key in trace and trace[key].size:
            state[key] = trace[key]
    state["ledger"]={key:trace["ledger_"+key] for key in ("cells","values","directions","segments","goal_evidence")
                     if "ledger_"+key in trace}
    snap=dict(summary=s,rows=rows,map_state=state,terminal_pose=trace.get("terminal_pose"),
              a=[],c=[],goal=None,groups={},terminal=None)
    if snap["terminal_pose"] is not None and not snap["terminal_pose"].size:
        snap["terminal_pose"]=None
    if not s.get("map_snapshot_available", True):
        state.pop("grid",None)
    references=[]
    if (folder/"decision_frames.npz").is_file():
        with np.load(folder/"decision_frames.npz",allow_pickle=False) as archive:
            meta=json.loads(str(archive["schema_json"]))
            if meta["schema"]!=1:
                raise ValueError("Unsupported decision archive schema")
            events={k:{n:(archive[v["array"]] if isinstance(v,dict) and "array" in v else v)
                        for n,v in rec.items()} for k,rec in meta["records"].items()}
            for key in ("a","c"):
                snap[key]=[events[k] for k in meta[key]]
            snap["goal"]=events.get(meta["goal"])
            snap["groups"]={k:events[v] for k,v in meta["groups"].items()}
            references=[(name,archive[k]) for name,k in meta["references"]]
            if meta["terminal"]:
                snap["terminal"]=dict(meta["terminal"],rgb=archive.get("terminal_rgb"))
    else:
        # Old archives have selected predictions and factual rewards but no image tensors.
        source_names={name:i for i,name in SOURCES.items()}
        raw={}
        if replay:
            replay=Path(replay)
            if replay.stem.rsplit("-",1)[0]!=s["episode_id"]:
                raise ValueError("Replay episode identity differs from summary")
            with np.load(replay,allow_pickle=False) as f:
                if len(f["reward"])!=len(rows)+1 or not np.allclose(f["reward"][1:],trace["reward"]):
                    raise ValueError("Replay/trace arrival reward or length mismatch")
                keys=("image","heatmap","goal_image","goal_image_valid","log_goal_ref_map","log_goal_heatmap_policy",
                      "log_goal_boxes","log_goal_candidate_sources","log_vnext_requested","log_vnext_score_supported",
                      "log_actuator_valid","log_applied_control")
                raw={k:f[k] for k in keys if k in f}
        def legacy(record):
            i=record["step"];r=dict(rows[i],actor=record["actor"],source=source_names[record["source"]])
            if r.get("override") is None and "override" in record.get("reasons", []):
                r["override"] = True
            def field(k):
                return None if record.get(k) is None else np.asarray(record[k])
            e=dict(row=r,rgb=None,requested=-1,rank_choice=-1,mask=field("legal_mask").astype(bool),
                   prediction=field("predictions"),has_prediction=field("predictions") is not None,
                   q=field("c_value"),q_computed=field("c_value") is not None,a_score=field("a_score"),
                   supported=None,reasons=record.get("reasons",[]),goal_context=field("goal_context"),
                   goal_forecast=field("goal_forecast"),goal_new_area_forecast=field("goal_new_area_forecast"),
                   goal_selection_reason=record.get("goal_selection_reason") or "archived decision",
                   goal_region_cal_reliability=record.get("goal_region_cal_reliability"),
                   feedback=dict(reward=r["reward"],success=r["success"],arrival_index=i+1),
                   applied=field("controller_applied") if "controller_applied" in record else field("actual_control"))
            if "controller_applied" in record:
                e["env_applied"] = field("actual_control")
            if raw:
                native="goal_image" in raw and bool(raw["goal_image_valid"][i])
                e.update(rgb=raw["image"][i],arrival_rgb=raw["image"][i+1],
                    goal_rgb=raw["goal_image" if native else "image"][i],goal_original_heatmap=raw["heatmap"][i])
                for key in ("ref_map","heatmap_policy","boxes","candidate_sources"):
                    if "log_goal_"+key in raw:
                        e["goal_"+key]=raw["log_goal_"+key][i+1]
                if "log_vnext_requested" in raw:
                    e["requested"]=int(raw["log_vnext_requested"][i+1])
                if "log_vnext_score_supported" in raw:
                    e["supported"]=raw["log_vnext_score_supported"][i+1]
                if raw.get("log_actuator_valid",np.zeros(len(rows)+1))[i+1]:
                    e["env_applied"]=raw["log_applied_control"][i+1]
            return e
        snap["a"]=[legacy(r) for r in s["selected_keyframes_for_A"]]
        snap["c"]=[legacy(r) for r in s["selected_keyframes_for_C"]]
        if s.get("goal_keyframe"):
            snap["goal"]=legacy(s["goal_keyframe"])
    return snap,references
