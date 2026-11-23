"""Publication panels drawn from a single completed episode or its archived records."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from .actuator import CONTROL_KEYS
from .visualization import PHASES, SOURCES, _goal_heatmap, _rgb

COMMANDS = ("wait", "attack", "look up", "look down", "turn left", "turn right",
            "forward", "back", "left", "right", "jump forward", "use")
INK, BLUE, GOLD, GREEN = "#173552", "#43759b", "#cc8951", "#318269"
DISPLAY_SOURCES = {**SOURCES, 10:"recovery probe (settle)", 11:"verification (align)"}
SOURCE_COLORS = ["#969da2", "#304f6c", GOLD, "#70a577", "#aa6255", "#9783b0",
                 "#609eaa", "#b3916d", "#986aa4", "#a94e36", "#208f88", "#88663e", "#a56899"]


def label(names, index):
    return names[index].replace("_", " ") if 0 <= index < len(names) else "unrecorded"


def panel(ax, value, title, heat=False):
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")
    if value is None:
        ax.text(.5, .5, "Not recorded", ha="center", va="center", transform=ax.transAxes, color="#777777")
        return None
    return ax.imshow(value, interpolation="nearest", **(dict(cmap="viridis", vmin=0, vmax=1) if heat else {}))


def shortlist(event):
    legal = np.flatnonzero(event["mask"])
    selected = event["row"]["option"]
    score = event.get("a_score")
    order = legal if score is None else legal[np.argsort(-np.asarray(score)[legal], kind="stable")]
    return list(dict.fromkeys(int(i) for i in (selected, 0, event.get("rank_choice",-1), *order)
                              if i in legal))[:4]


def command_text(value):
    if value is None:
        return "not recorded"
    return ", ".join((f"{name} {x:+g}°" if name in ("pitch","yaw") else name)
                     for name, x in zip(CONTROL_KEYS, np.ravel(value)) if abs(x) > .001) or "no input"


def decision(viz, event, path, kind):
    row = event["row"]
    names = viz._names(len(event["mask"]), row["phase"])
    ids = shortlist(event)
    has_table = kind != "C" and bool(ids) and event["has_prediction"]
    fig = viz._figure((10.2, 5.2 if has_table else 3.7))
    grid = fig.add_gridspec(2 if has_table else 1, 2, left=.035, right=.98, top=.82,
                            bottom=.13, width_ratios=[1,2.65], wspace=.19, hspace=.22)
    ax = fig.add_subplot(grid[:,0])
    panel(ax, event.get("rgb"), "Current observation")
    title = {"A":"A  |  Forecasts for legal options", "C":"C  |  From proposal to executed control",
             "AC":"A + C  |  Forecast, selection and feedback"}[kind]
    fig.suptitle(title, x=.035, y=.96, ha="left", fontsize=15, weight="bold", color=INK)
    fig.text(.036,.855,f"Decision {row['step']}   ·   {PHASES[row['phase']]}   ·   "
             f"Actor proposal: {label(COMMANDS, row['actor'])}", fontsize=10)
    rank_mask = event.get("rank_mask")
    templates, lengths = event.get("template_commands"), event.get("template_lengths")
    if has_table:
        body = fig.add_subplot(grid[0,1])
        body.axis("off")
        headers = ["Option", "Planned commands¹", "Discover", "Revisit", "A score",
                   "Support" if kind == "A" else "Rank score" if rank_mask is not None else "Logged C value"]
        data = []
        def number(value):
            return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"
        for i in ids:
            support = event.get("supported")
            supported = "not recorded" if support is None else "yes" if support[i] else "no"
            score = (number(event["q"][i]) if event["q_computed"] else "—")
            if rank_mask is not None and not rank_mask[i]:
                score = "ineligible"
            template = "not recorded"
            if templates is not None and lengths is not None:
                template = " → ".join(label(COMMANDS, cmd) for cmd in templates[i,:int(lengths[i])])
            data.append([label(names,i)+(" *" if i==row["option"] else ""), template,
                         number(event["prediction"][i,2]), number(event["prediction"][i,3]),
                         number(event["a_score"][i] if event.get("a_score") is not None else None),
                         supported if kind=="A" else score])
        tab = body.table(cellText=data, colLabels=headers, cellLoc="center", bbox=[0,0,1,1],
                         colWidths=[.16,.35,.11,.09,.10,.19])
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#dee9ed" if r==0 else "#f1f5f5")
    body = fig.add_subplot(grid[-1,1])
    body.axis("off")
    source = DISPLAY_SOURCES.get(row.get("control_kind", row["source"]), "unrecorded")
    feedback = event.get("feedback")
    result = "unrecorded" if not feedback else f"reward {feedback['reward']:g}, success {int(feedback['success'])}"
    override = row.get("override")
    takeover = "unrecorded" if override is None else "yes" if override else "no"
    entries = [
        ("Before rules", label(names, event.get("rank_choice",-1))),
        ("Final option", f"{label(names,row['option'])}  ·  {source}  ·  takeover: {takeover}"),
        ("Issued command", label(COMMANDS, event.get("requested",-1))),
        ("Controller control", command_text(event.get("applied"))),
        ("Environment receipt", command_text(event.get("env_applied"))),
        ("Observed outcome", result)]
    if kind == "A":
        entries = entries[1:3]
    for i,(key,value) in enumerate(entries):
        y = 1 - (i+.5)/len(entries)
        body.text(0,y,key,fontsize=9,color=BLUE,va="center",transform=body.transAxes)
        body.text(.31,y,value,fontsize=10,color=INK,va="center",transform=body.transAxes)
    reasons = ", ".join(event.get("reasons",[])[:3]) or "retained decision"
    fig.text(.035,.065,f"Selection: {reasons}.",fontsize=8)
    if has_table:
        note = "¹ Candidate plans. * Selected option. Unexecuted alternatives have forecasts only."
        if event.get("supported") is None:
            note += " Support was not recorded in this archive."
        fig.text(.035,.029,note,fontsize=8)
    viz._save(fig,path)


def timeline(viz, snap, path, focus=None, radius=24):
    rows=snap["rows"]
    if focus is not None:
        rows=[r for r in rows if abs(r["step"]-focus)<=radius]
    if not rows:
        return
    x=np.array([r["step"] for r in rows])
    fig=viz._figure((10.7,5.1))
    axes=fig.subplots(4,1,sharex=True,gridspec_kw={"height_ratios":[1,1,1,1.5]})
    fig.subplots_adjust(left=.14,right=.65,top=.86,bottom=.13,hspace=.55)
    title="Episode control timeline" if focus is None else f"Decision window  |  Steps {x[0]}–{x[-1]}"
    fig.suptitle(title,ha="left",x=.12,fontsize=14,color=INK,weight="bold")
    tracks=[("Phase","phase",dict(enumerate(PHASES)),["#729fc1","#dfb160","#64a880","#c07870"]),
            ("Control source","control_kind",DISPLAY_SOURCES,SOURCE_COLORS),
            ("D evidence","goal_state",{-1:"unrecorded",0:"untrusted",1:"candidate",2:"trusted"},
             ["#cfd4d8","#e3e9ec",GOLD,GREEN])]
    for ax,(name,key,lookup,colors) in zip(axes,tracks):
        if key == "control_kind":
            vals = np.array([r.get(key, r.get("source", -1)) for r in rows])
        else:
            vals = np.array([r.get(key, -1) for r in rows])
        ax.imshow(vals[None],aspect="auto",interpolation="nearest",cmap=ListedColormap(colors),
                  vmin=min(lookup)-.5,vmax=max(lookup)+.5,extent=(x[0]-.5,x[-1]+.5,0,1))
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center", labelpad=9)
        ax.legend(handles=[Patch(color=colors[k-min(lookup)],label=lookup[k]) for k in sorted(set(vals)) if k in lookup],
                  loc="center left",bbox_to_anchor=(1.015,.5),ncol=2,frameon=False,fontsize=7, columnspacing=.9, handlelength=1.0)
    axes[-1].plot(x,[r["reward"] for r in rows],color=BLUE,linewidth=1)
    hits=[r["step"] for r in rows if r["success"]]
    if hits:
        axes[-1].axvline(hits[0],color=GREEN,ls="--",label="observed success")
        axes[-1].legend(loc="center left",bbox_to_anchor=(1.015,.5),frameon=False,fontsize=8)
    axes[-1].set_ylabel("Reward");axes[-1].set_xlabel("Decision step (reward measured after the action)")
    viz._save(fig,path)


def goal(viz,event,path):
    fig=viz._figure((10.5,4.2))
    grid=fig.add_gridspec(2,5,height_ratios=[2,1],left=.025,right=.95,bottom=.13,top=.83,hspace=.42)
    rgb=event.get("goal_rgb")
    axes=[fig.add_subplot(grid[0,i]) for i in range(5)]
    panel(axes[0],rgb,"Observed RGB")
    artist=panel(axes[1],_goal_heatmap(event.get("goal_original_heatmap")),"Original affordance",True)
    panel(axes[2],rgb,"Candidate regions")
    boxes=event.get("goal_boxes")
    if rgb is not None and boxes is not None:
        h,w=rgb.shape[:2]
        for i,(x,y,u,v) in enumerate(boxes):
            src=event.get("goal_candidate_sources")
            color=("#ffac45","#55ceef","#ea77ce")[int(src[i])] if src is not None else "white"
            axes[2].add_patch(Rectangle((x*w-.5,y*h-.5),(u-x)*w,(v-y)*h,fc="none",ec=color,lw=1))
    panel(axes[3],_goal_heatmap(event.get("goal_ref_map")),"Regional evidence",True)
    panel(axes[4],_goal_heatmap(event.get("goal_heatmap_policy")),"Fused policy input",True)
    if artist is not None:
        fig.colorbar(artist,cax=fig.add_axes([.965,.49,.009,.29]),ticks=[0,.5,1])
    for i,(name,frame) in enumerate(getattr(viz,"goal_references",[])[:4]):
        panel(fig.add_subplot(grid[1,i]),frame,name.split(" · ")[0]+" reference")
    row=event["row"];cal=event.get("goal_region_cal_reliability")
    note=fig.add_subplot(grid[1,4]);note.axis("off")
    gate = row.get("goal_gate")
    gate_text = f"{gate:.2f}" if gate is not None and np.isfinite(gate) else "unknown"
    cal_text = f"{cal:.2f}" if cal is not None and np.isfinite(cal) else "unknown"
    note.text(0, .65, f"Fusion gate  {gate_text}\nCalibration  {cal_text}", fontsize=9)
    fig.suptitle(f"D  |  Regional evidence at decision {row['step']}",ha="left",x=.025,color=INK,weight="bold")
    fig.text(.025,.04,f"Same-frame inference  ·  {event.get('goal_selection_reason','archived decision')}  ·  proposal outlines: heat / RGB / hard (amber / cyan / pink).",fontsize=8)
    viz._save(fig,path)


def history(viz,snap,path):
    s=snap["summary"]; state=snap.get("map_state") or {}
    frames = np.asarray(state.get("history_frames", []))
    valid = np.asarray(state.get("history_valid", []), bool)
    slots = np.flatnonzero(valid) if frames.ndim == 4 and len(frames) == len(valid) else []
    meta = np.asarray(state.get("history_meta", []))
    weights = np.asarray(state.get("history_query_weights", []))
    tick = state.get("history_query_tick")
    cols = min(max(1, len(slots)), 8)
    bands = max(1, (len(slots) + cols - 1) // cols)
    fig = viz._figure((10, 2.65 + 1.8*bands))
    grid = fig.add_gridspec(bands + 1, cols, left=.14, right=.98, top=.80,
                            bottom=.15, hspace=.95, wspace=.28, height_ratios=[1]*bands+[.9])
    for index, slot in enumerate(slots):
        ax = fig.add_subplot(grid[index//cols, index%cols])
        panel(ax, _rgb(frames[slot,...,:3]), f"Slot {slot}")
        times = (f"{int(meta[slot,0])} → {int(meta[slot,1])}" if
                 meta.ndim == 2 and meta.shape[0] == len(valid) else "time: n/a")
        weight = f"w {weights[slot]:.3f}" if weights.shape == valid.shape else "w: n/a"
        ax.text(.5,-.08,times+"\n"+weight,ha="center",va="top",fontsize=8,transform=ax.transAxes)
    if not len(slots):
        panel(fig.add_subplot(grid[0,:]), None, "No retained slot images available")
    track = fig.add_subplot(grid[-1,:])
    for offset,key,color in ((0,"memory_available",GREEN),(1,"local_ready",BLUE),(2,"pose_ready",GOLD)):
        ids = [r["step"] for r in snap["rows"] if r[key]]
        track.scatter(ids,[offset]*len(ids),s=4,color=color,marker="|")
    track.set(yticks=[0,1,2],yticklabels=["Visual history","Local tracking","Continuous pose"],
              xlim=(-.5,max(.5,s["total_steps"]-.5)),ylim=(-.6,2.6),xlabel="Decision step")
    fig.suptitle("B  |  Retained visual memory slots", x=.035, y=.96, ha="left", color=INK, weight="bold")
    detail = f"Query at memory tick {int(tick)}" if tick is not None else "Query weights not recorded"
    fig.text(.035,.87,f"{len(slots)} occupied slots   ·   {detail}   ·   Times: creation → last visit (memory ticks)",fontsize=9)
    route = [e["memory_input"] for e in snap.get("groups",{}).values() if e.get("has_prediction") and e.get("memory_input") is not None]
    routed = "observed" if any(route) else "off on sampled decisions" if route else "not recorded"
    fig.text(.035,.025,f"Tiles show compressed RGB tokens (4 × 4 in fix3). w = retrieval weight at the recorded query. A/C memory input: {routed}.",fontsize=8)
    viz._save(fig,path)


def storyboard(viz,snap,path):
    groups=snap.get("groups",{})
    candidates=[groups.get(k) for k in ("reset observation","visual history without local pose",
                "qualified local track","trusted regional evidence","learned change to Actor command")]
    candidates.extend(snap.get("c",[]))
    unique={e["row"]["step"]:e for e in candidates if e is not None}
    events=sorted(unique.values(),key=lambda e:e["row"]["step"])[:5]
    terminal=snap.get("terminal")
    if not events and not terminal:
        return
    fig=viz._figure((max(5,1.8*(len(events)+bool(terminal))),2.9))
    axes=fig.subplots(1,len(events)+bool(terminal),squeeze=False)[0]
    fig.subplots_adjust(left=.02,right=.98,top=.74,bottom=.23,wspace=.12)
    for ax,e in zip(axes,events):
        r=e["row"];panel(ax,e["rgb"],f"Step {r['step']}")
        source = DISPLAY_SOURCES.get(r.get("control_kind", r["source"]), "unrecorded")
        ax.text(.5,-.08,f"{PHASES[r['phase']]}\n{source}",
                ha="center",va="top",transform=ax.transAxes,fontsize=8)
    if terminal:
        panel(axes[-1],terminal["rgb"],f"Arrival {terminal['step']}")
        axes[-1].text(.5,-.08,"Success" if terminal["success"] else "Episode ended",ha="center",transform=axes[-1].transAxes)
    fig.suptitle("Observed events in one episode",x=.02,ha="left",color=INK,weight="bold")
    viz._save(fig,path)
