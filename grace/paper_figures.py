"""Publication panels drawn from a single completed episode or its archived records."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from .figure_style import (INK, TEXT, MUTED, PANEL, LINE, MISSING, ACTOR as BLUE,
                           LEARNED as GREEN, SEARCH, ALIGN, RECOVERY, SELECTED,
                           surface, box, txt, heading, lines)
from .actuator import CONTROL_KEYS
from .visualization import PHASES, SOURCES, _goal_heatmap, _rgb

COMMANDS = ("wait", "attack", "look up", "look down", "turn left", "turn right",
            "forward", "back", "left", "right", "jump forward", "use")
DISPLAY_SOURCES = {**SOURCES, 10: "recovery probe (settle)", 11: "verification (align)"}
SOURCE_PALETTE = {-1: MISSING, 0: BLUE, 1: GREEN, 2: "#477F86", 3: RECOVERY, 4: "#8994A2",
                  5: "#827B4F", 6: "#C48079", 7: ALIGN, 8: "#AC835E", 9: SEARCH,
                  10: "#9B6562", 11: "#A387AE"}
SOURCE_COLORS = [SOURCE_PALETTE[i] for i in sorted(DISPLAY_SOURCES)]


def label(names, index):
    return names[index].replace("_", " ") if 0 <= index < len(names) else "Not recorded"


def panel(ax, value, title, heat=False):
    ax.set_title(title, fontsize=8.2, color=INK, pad=3)
    ax.axis("off")
    if value is None:
        ax.text(.5, .5, "Not recorded", ha="center", va="center", transform=ax.transAxes, color="#777777")
        return None
    ax.set_xlim(-.5, value.shape[1]-.5)
    ax.set_ylim(value.shape[0]-.5,-.5)
    return ax.imshow(value, interpolation="nearest", **(dict(cmap="viridis", vmin=0, vmax=1) if heat else {}))


def shortlist(event):
    legal = np.flatnonzero(event["mask"])
    selected = event["row"]["option"]
    score = event.get("a_score")
    order = legal if score is None else legal[np.argsort(-np.asarray(score)[legal], kind="stable")]
    return list(dict.fromkeys(int(i) for i in (selected, 0, event.get("rank_choice",-1), *order)
                              if i in legal))[:4]


def command_text(value):
    control = np.ravel(value) if value is not None else np.empty(0)
    if control.size != len(CONTROL_KEYS) or not np.isfinite(control).all():
        return "Not recorded"
    return ", ".join((f"{name} {x:+g}°" if name in ("pitch","yaw") else name)
                     for name, x in zip(CONTROL_KEYS, control) if abs(x) > .001) or "no input"


def _control_flow(event, names, kind):
    row = event["row"]
    source_id = row.get("control_kind", row["source"])
    source = DISPLAY_SOURCES.get(source_id, "Not recorded")
    stage = "Pre-rule: " + label(names, event.get("rank_choice", -1))
    final = label(names, row["option"])
    takeover = row.get("override")
    detail = source + ("" if takeover is None else " · takeover" if takeover else " · retained")
    receipt = command_text(event.get("env_applied"))
    feedback = event.get("feedback")
    outcome = (f"Arrival {feedback['arrival_index']} · reward {feedback['reward']:g} · "
               + ("success" if feedback["success"] else "no success")) if feedback else "Feedback: Not recorded"
    color = SOURCE_PALETTE.get(source_id, MISSING)
    return [
        ("Actor proposal", label(COMMANDS, row["actor"]), ([] if kind=="C" else ["Original command"]), BLUE),
        ("Controller choice", final, [stage, detail], color),
        ("Issued command", label(COMMANDS, event.get("requested", -1)),
         ["Controller: " + command_text(event.get("applied"))], color),
        ("Observed feedback" if kind == "AC" else "Environment receipt",
         "Env: "+receipt if kind == "AC" else receipt, [outcome], INK),
    ]


def format_score(value):
    if not np.isfinite(value):
        return "—"
    if value == 0:
        return "0.000"
    if 0 < abs(value) < .0005:
        return f"{value:+.1e}"
    return f"{value:+.3f}"


def _candidate_panel(ax, events, event, names, kind, width):
    row = event["row"]
    n = len(events)
    gap = .39
    rank_mask = event.get("rank_mask") if event.get("rank_choice",-1) >= 0 else None
    headers = ("Legal option", "Discover", "Revisit", "A score", "Support" if kind == "A" else
               "Pre-rule C" if rank_mask is not None else "Logged C")
    xs = [0, .315, .51, .70, .84]
    if width < 4:
        last = "Supp." if kind == "A" else "C rank" if rank_mask is not None else "C log"
        headers = ("Option", "Discover", "Revisit", "A", last)
    has_detail = kind != "A" or event.get("supported") is not None
    if not has_detail:
        xs = [0,.31,.57,.84]
    box(ax, 0, 0, width, .25, PANEL)
    for x, name in zip(xs, headers):
        txt(ax, x*width+.055, .125, name, 7.6, INK, True)
    prediction = event.get("prediction") if event.get("has_prediction") else None
    supported = event.get("supported")
    c = event.get("q") if event.get("q_computed") else None
    a = event.get("a_score") if prediction is not None else None
    for j, i in enumerate(events):
        y = .25+j*gap
        chosen = i == row["option"]
        if chosen:
            box(ax, 0, y, width, gap, SELECTED)
            box(ax, 0, y, .035, gap, GREEN)
        ax.plot([0,width], [y+gap,y+gap], linewidth=.5, color=LINE)
        txt(ax,.065,y+.115,label(names,i),8,BLUE if i==0 else TEXT,chosen)
        tags = []
        if chosen:
            tags.append("selected")
        if i == 0:
            tags.append("Actor option")
        if tags:
            txt(ax,.065,y+.275," · ".join(tags),7,MUTED)
        for x, col in zip(xs[1:3], (2,3)):
            left=x*width+.055
            p = prediction[i,col] if prediction is not None else np.nan
            if np.isfinite(p) and 0 <= p <= 1:
                length=(.086 if has_detail else .133)*width
                box(ax,left,y+.165,length,.055,LINE)
                box(ax,left,y+.165,length*p,.055,GREEN if chosen else BLUE)
                txt(ax,left+length+.035,y+.19,f"{p:.2f}",7.4)
            else:
                txt(ax,left,y+.19,"—",8,MUTED)
        av = a[i] if a is not None else np.nan
        txt(ax,xs[3]*width+.055,y+.19,format_score(av),7.8)
        if not has_detail:
            continue
        if kind == "A":
            value = ("Yes" if supported[i] else "No") if supported is not None else "—"
        elif rank_mask is not None and not rank_mask[i]:
            value = "Ineligible"
        elif c is not None and np.isfinite(c[i]):
            value = f"{c[i]:.3f}"
            if rank_mask is not None:
                eligible = np.asarray(rank_mask,dtype=bool) & np.isfinite(c)
                rank = 1 + np.count_nonzero(eligible & (np.asarray(c) > c[i]))
                value = f"{rank} / {value}"
        else:
            value = "—"
        txt(ax,xs[4]*width+.055,y+.19,value,7.5)
    if not n:
        txt(ax,.06,.45,"No legal options recorded",8,MUTED)


def decision(viz, event, path, kind):
    row = event["row"]
    names = viz._names(len(event["mask"]), row["phase"])
    ids = shortlist(event)
    narrow = getattr(viz, "single_column", False)
    w = 3.40 if narrow else 7.25
    has_table = kind != "C"
    cards = _control_flow(event, names, kind)
    prepared = [(key, lines(main, 22), [line for value in details for line in lines(value, 25)], color)
                for key, main, details, color in cards]
    flow_h = .42 + max(.15*len(main)+.135*len(details) for _,main,details,_ in prepared)
    table_h = .25+.39*max(1,len(ids))
    observation_h = max(1.27, table_h) if has_table else 0
    table_top = 1.27 if narrow else .55
    if narrow and has_table:
        observation_h = table_top - .55 + table_h
    notes = []
    support = event.get("supported")
    if has_table:
        if support is None:
            notes.append("Not recorded: forecast support.")
        elif kind == "AC":
            active = [label(names,i) for i in ids if support[i]]
            notes.append("Supported scores: "+(", ".join(active) if active else "none among shown options")+".")
        if kind == "AC" and event.get("rank_mask") is None:
            notes.append("Logged C values lack a pre-rule eligibility record.")
        notes.append("Bars: event probabilities (0–1). " + ("A and C scores use model units." if kind == "AC" else "A score uses model units."))
        if kind == "AC" and event.get("rank_mask") is not None:
            notes.append("Pre-rule C: eligible rank / score.")
    templates = []
    if kind == "AC":
        plan, length = event.get("template_commands"), event.get("template_lengths")
        if plan is None or length is None:
            templates.append("Candidate plans: Not recorded.")
        else:
            for i in ids:
                sequence = " → ".join(label(COMMANDS,k) for k in plan[i,:int(length[i])])
                templates.append(f"{label(names,i)}: {sequence or 'empty plan'}")
    plan_lines = lines("    /    ".join(templates),(53 if narrow else 105)) if templates else []
    reason = ", ".join(str(s).replace("_"," ") for s in event.get("reasons",[])[:3])
    if reason:
        notes.append("Frame: "+reason+".")
    footer = lines("  ".join(notes),(57 if narrow else 112)) if notes else []
    flow_y = .60+observation_h+.15*len(plan_lines)
    if kind=="C":
        flow_h=max(flow_h,1.18)
    flow_cols = 2 if narrow else 4
    flow_rows = 4 // flow_cols
    foot_y = flow_y + (flow_rows*(flow_h+.075) if kind != "A" else 0)
    h = foot_y+.155*len(footer)+.13
    fig=viz._figure((w,h))
    heading(fig, {"A":"A   Candidate outcomes", "AC":"A + C   From forecast to action",
                  "C":"C   Action and environment receipt"}[kind],
            f"Decision {row['step']}   /   {PHASES[row['phase']]}   /   {DISPLAY_SOURCES.get(row.get('control_kind',row['source']), 'Not recorded')}")
    if has_table:
        side=.59 if narrow else min(observation_h-.34,1.22)
        ax=surface(fig,.13,.54,side,side)
        panel(ax,event.get("rgb"),"")
        info=surface(fig,.83,.64,w-1.0,.36) if narrow else surface(fig,.13,.56+side,1.39,.31)
        txt(info,0,.08,"Actor: "+label(COMMANDS,row["actor"]),7.6,BLUE)
        txt(info,0,.24,"Final: "+label(names,row["option"]),7.6,INK,True)
        table_x = .13 if narrow else 1.55
        table_width = w-table_x-.13
        table=surface(fig,table_x,table_top,table_width,table_h)
        _candidate_panel(table,ids,event,names,kind,table_width)
    for idx,line in enumerate(plan_lines):
        fig.text(.13/w,1-(.59+observation_h+.15*idx)/h,line,va="top",fontsize=7.3,color=MUTED)
    if kind != "A":
        gap=.12
        cw=(w-.26-(flow_cols-1)*gap)/flow_cols
        for j,(key,main,details,color) in enumerate(prepared):
            cx=.13+(j%flow_cols)*(cw+gap)
            cy=flow_y+(j//flow_cols)*(flow_h+.075)
            ax=surface(fig,cx,cy,cw,flow_h)
            box(ax,0,0,cw,flow_h,PANEL)
            box(ax,0,0,cw,.033,color)
            txt(ax,.065,.16,key,8.1,color,True)
            y=.37
            for line in main:
                txt(ax,.065,y,line,8.4,INK,True)
                y+=.15
            for line in details:
                txt(ax,.065,y,line,7.35,MUTED)
                y+=.135
            if kind=="C" and j==0:
                thumbnail=surface(fig,.20,flow_y+.51,.57,.57)
                panel(thumbnail,event.get("rgb"),"")
            if j%flow_cols < flow_cols-1:
                txt(ax,cw+gap/2,flow_h/2,"›",11,MUTED,ha="center",clip_on=False)
    for j,line in enumerate(footer):
        fig.text(.13/w,1-(foot_y+.15*j)/h,line,va="top",fontsize=7.25,color=MUTED)
    viz._save(fig,path)


def timeline(viz, snap, path, focus=None, radius=24):
    rows = snap["rows"]
    if focus is not None:
        rows = [r for r in rows if abs(r["step"]-focus) <= radius]
    if not rows:
        return
    x = np.asarray([r["step"] for r in rows])
    marker = focus
    if marker is None:
        events = list(snap.get("a", []))+list(snap.get("c", []))
        selected = next((e for e in events if e["row"].get("override") and e["row"].get("source")==1), None)
        if selected is not None:
            marker = selected["row"]["step"]
    tracks = [
        ("Phase","phase",dict(enumerate(PHASES)),[BLUE,SEARCH,GREEN,RECOVERY]),
        ("Source","control_kind",DISPLAY_SOURCES,SOURCE_COLORS),
        ("D evidence","goal_state",{-1:"Not recorded",0:"untrusted",1:"candidate",2:"trusted"},
         [MISSING,"#DAE1E6",SEARCH,GREEN])]
    fig = viz._figure((7.25,2.82))
    heading(fig, "C   Control timeline" if focus is None else "C   Decision window",
            f"Steps {x[0]}–{x[-1]}" + (f"   /   Dotted line: decision {marker}" if marker is not None else ""))
    axes = fig.subplots(4,1,sharex=True,gridspec_kw={"height_ratios":[1,1,1,1.25]})
    fig.subplots_adjust(left=.115,right=.975,top=.79,bottom=.47,hspace=.18)
    legends = []
    for ax,(name,key,lookup,colors) in zip(axes,tracks):
        vals = np.asarray([r.get(key, r.get("source",-1) if key=="control_kind" else -1) for r in rows])
        ax.imshow(vals[None], aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
                  vmin=min(lookup)-.5,vmax=max(lookup)+.5,extent=(x[0]-.5,x[-1]+.5,0,1))
        legends.append([(lookup[k],colors[k-min(lookup)]) for k in sorted(set(vals)) if k in lookup])
    reward_ax=axes[-1]
    rewards=np.asarray([r["reward"] for r in rows])
    idx=np.flatnonzero(np.isfinite(rewards)&(rewards!=0))
    reward_ax.axhline(0,color=LINE,lw=.7)
    if len(idx):
        reward_ax.vlines(x[idx],0,rewards[idx],color=BLUE,lw=1)
        reward_ax.scatter(x[idx],rewards[idx],s=12,color=BLUE,zorder=3)
        low,high=min(0,min(rewards[idx])),max(0,max(rewards[idx]))
        pad=.45*(high-low)
        reward_ax.set_ylim(low-pad,high+pad)
    hits=[r["step"] for r in rows if r["success"]]
    if hits:
        reward_ax.scatter(hits[0],.65,transform=reward_ax.get_xaxis_transform(),marker="*",s=38,
                          color=GREEN,zorder=5,clip_on=False)
    for ax,name in zip(axes,("Phase","Source","D evidence","Reward")):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x",length=0,labelsize=7,colors=MUTED,pad=2)
        ax.set_ylabel(name,rotation=0,ha="right",va="center",fontsize=8,color=INK,labelpad=6)
        if marker is not None and x[0]<=marker<=x[-1]:
            ax.axvline(marker,color=INK,lw=.85,ls=":")
    reward_ax.set_xlim(x[0]-.5,x[-1]+.5)
    # Use observed categories only; fold the source legend by measured text width.
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    legend_y=1.78
    for name,entries in zip(("Phase","Source","Evidence"),legends):
        pos=.85
        fig.text(.13/7.25,1-legend_y/2.82,name,va="center",fontsize=7.3,color=MUTED,weight="bold")
        for text,color in entries:
            advance=TextPath((0,0),text,prop=FontProperties(size=7.5)).get_extents().width/72+.24
            if pos+advance>7.16:
                pos=.85
                legend_y+=.18
            fig.add_artist(Rectangle((pos/7.25,1-(legend_y+.037)/2.82),.082/7.25,.074/2.82,
                                     transform=fig.transFigure,facecolor=color,edgecolor="none"))
            fig.text((pos+.115)/7.25,1-legend_y/2.82,text,va="center",fontsize=7.5,color=TEXT)
            pos+=advance
        legend_y+=.2
    note = ("Rewards: "+", ".join(f"{v:g}" for v in np.unique(rewards[idx])[:6]) if len(idx) else
            "No nonzero reward in view")
    if hits:
        note+=f"   /   Star: first observed success at {hits[0]}"
    fig.text(.115,1-legend_y/2.82,note,fontsize=7.5,color=MUTED,va="center")
    viz._save(fig,path)


def goal(viz,event,path):
    fig=viz._figure((7.25,2.94))
    row=event["row"]
    heading(fig,"D   Regional verification",f"Decision {row['step']}   /   {event.get('goal_selection_reason') or 'archived observation'}")
    rgb=event.get("goal_rgb")
    boxes=event.get("goal_boxes")
    frames=[rgb,_goal_heatmap(event.get("goal_original_heatmap")),rgb if boxes is not None else None,
            _goal_heatmap(event.get("goal_ref_map")),_goal_heatmap(event.get("goal_heatmap_policy"))]
    names=[event.get("display_rgb_label","Observed RGB"),"Original affordance","Region proposals","Region evidence","Policy heatmap"]
    axes=[]
    for i,(frame,title) in enumerate(zip(frames,names)):
        ax=surface(fig,.13+i*1.40,.78,1.30,1.15)
        artist=panel(ax,frame,title,heat=i in (1,3,4))
        axes.append(ax)
        if i==1 and artist is not None:
            cb=fig.colorbar(artist,cax=fig.add_axes((.973,.355,.009,.36)),ticks=[0,.5,1])
            cb.ax.tick_params(labelsize=6.8,length=1,pad=1)
            cb.outline.set_linewidth(.4)
    if rgb is not None and boxes is not None:
        h,w=rgb.shape[:2]
        for j,(x,y,u,v) in enumerate(boxes):
            src=event.get("goal_candidate_sources")
            color=("#ffac45","#55ceef","#ea77ce")[int(src[j])] if src is not None else "white"
            axes[2].add_patch(Rectangle((x*w-.5,y*h-.5),(u-x)*w,(v-y)*h,fc="none",ec=color,lw=.85))
    refs=getattr(viz,"goal_references",[])[:4]
    for i,(name,frame) in enumerate(refs):
        ax=surface(fig,.16+i*.67,2.07,.53,.43)
        panel(ax,frame,"")
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,edgecolor=LINE,lw=1))
        ax.text(.5,-.11,name.split(" · ")[0],va="top",ha="center",fontsize=6.9,transform=ax.transAxes,color=MUTED)
    status=surface(fig,2.96 if refs else .13,2.04,4.05 if refs else 6.89,.56)
    sw=status.get_xlim()[1]
    box(status,0,0,sw,.56,PANEL)
    gate=row.get("goal_gate")
    for i,(name,value) in enumerate((("Fusion gate",gate),("Calibration",event.get("goal_region_cal_reliability")))):
        x=.10+i*(sw/2)
        txt(status,x,.14,name,7.3,MUTED)
        txt(status,x,.38,f"{value:.3f}" if value is not None and np.isfinite(value) else "Not recorded",9,INK,True)
    note = "Shared map scale: 0–1."
    if gate==0:
        note+="   Gate closed."
    if boxes is not None:
        note+="   Proposals: heat (amber), RGB (cyan), hard negative (pink)."
    else:
        note+="   Proposal geometry not recorded."
    fig.text(.018,.044,note,fontsize=7.4,color=MUTED)
    viz._save(fig,path)


def history(viz,snap,path):
    state=snap.get("map_state") or {}
    frames=np.asarray(state.get("history_frames",[]))
    valid=np.asarray(state.get("history_valid",[]),bool)
    slots=np.flatnonzero(valid) if frames.ndim==4 and len(frames)==len(valid) else []
    meta=np.asarray(state.get("history_meta",[]))
    weights=np.asarray(state.get("history_query_weights",[]))
    has_weights=bool(len(slots)) and weights.shape==valid.shape
    tick=state.get("history_query_tick")
    cols=max(1,min(4 if getattr(viz,"single_column",False) else 8,len(slots)))
    bands=max(1,(len(slots)+cols-1)//cols)
    w=3.40 if getattr(viz,"single_column",False) else max(3.62,.87*cols+.26)
    cell=(w-.26)/cols
    row_h=1.02 if has_weights else .85
    track_top=.63+bands*row_h
    track_h=.65
    h=track_top+track_h+.41
    fig=viz._figure((w,h))
    heading(fig,"B   Retained visual history",f"{len(slots)} occupied slots   /   "+
            (f"Query weights at tick {int(tick)}" if has_weights and tick is not None else
             "Query weights not recorded" if not has_weights else "Query time not recorded"))
    slot_values=weights[slots] if has_weights else []
    for j,i in enumerate(slots):
        x=.13+j%cols*cell
        y=.55+j//cols*row_h
        card=surface(fig,x,y,cell-.05,row_h-.04)
        largest=has_weights and np.isfinite(weights[i]) and weights[i]>0 and weights[i]==max(slot_values)
        box(card,0,0,cell-.05,row_h-.04,SELECTED if largest else PANEL,GREEN if largest else LINE)
        txt(card,.058,.105,f"Slot {i}",7.6,INK,True)
        ax=surface(fig,x+.06,y+.205,cell-.17,.38)
        panel(ax,_rgb(frames[i,...,:3]),"")
        has_time=(meta.ndim==2 and meta.shape[0]==len(valid) and meta.shape[1]>=2
                  and np.isfinite(meta[i,:2]).all())
        txt(card,.055,.69,f"{int(meta[i,0])} → {int(meta[i,1])}" if has_time else "Time n/a",7.2,MUTED)
        if has_weights:
            p=weights[i]
            if np.isfinite(p) and 0<=p<=1:
                box(card,.055,.856,.25,.038,LINE)
                box(card,.055,.856,.25*p,.038,GREEN)
                txt(card,.35,.865,f"{p:.3f}",7.3,INK)
    if not len(slots):
        fig.text(.13/w,1-.96/h,"No recorded slot images",color=MUTED,fontsize=9)
    rows=snap["rows"]
    total=max(1,snap["summary"]["total_steps"])
    track=surface(fig,.13,track_top,w-.26,track_h)
    right=w-.34
    left=1.05
    for j,(key,name,color) in enumerate((("memory_available","Visual history",GREEN),
                                          ("local_ready","Local tracking",BLUE),("pose_ready","Continuous pose",SEARCH))):
        y=.095+j*.17
        txt(track,.01,y,name,7.2,INK)
        flags=np.full(total,-1,dtype=np.int8)
        for r in rows:
            if r.get(key) is not None and 0<=r["step"]<total:
                flags[r["step"]]=int(bool(r[key]))
        if (flags==1).any():
            # Represent missing observations as gaps, even when other steps are valid.
            track.imshow(flags[None,:],extent=(left,right,y+.035,y-.035),aspect="auto",
                         interpolation="nearest",cmap=ListedColormap(["white",LINE,color]),vmin=-1.5,vmax=1.5)
        else:
            text = ("Inactive throughout" if (flags==0).all() else
                    "Inactive where recorded" if (flags==0).any() else "Not recorded")
            txt(track,left,y,text,7.1,MUTED)
    track.set_xlim(0,w-.26)
    track.set_ylim(track_h,0)
    txt(track,left,.575,"Step 0",6.9,MUTED)
    txt(track,right,.575,str(total-1),6.9,MUTED,ha="right")
    route=[e["memory_input"] for e in snap.get("groups",{}).values()
           if e.get("has_prediction") and e.get("memory_input") is not None]
    routed="observed" if any(route) else "off in samples" if route else "not recorded"
    note=f"Compressed RGB tokens. Ticks: created → visited. A/C input: {routed}."
    for j,line in enumerate(lines(note,int(16*w))):
        fig.text(.13/w,1-(track_top+track_h+.03+.14*j)/h,line,fontsize=7.3,va="top",color=MUTED)
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
