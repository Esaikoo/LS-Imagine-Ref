"""Redraw an exported MineDojo episode without loading a checkpoint or launching the game."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grace.paper_archive import load_episode
from grace.visualization import EpisodeVisualizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--replay",type=Path,help="Matching old episode NPZ, if available")
    parser.add_argument("--format",choices=("png","pdf","svg"),default="png")
    parser.add_argument("--single-column",action="store_true",
                        help="Use compact 85 mm panels for A1, A+C, C2 and B history")
    parser.add_argument("--case",help="Render one recorded case group (names in summary.json case_groups)")
    parser.add_argument("--style", choices=("paper","original","both"), help="Imagination cache style, default paper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--focus", type=int, help="Center the control timeline on this decision")
    group.add_argument("--best-fusion", action="store_true", help="Scan an entire run for the strongest saved calibrated change")
    parser.add_argument("--radius", type=int, default=24, help="Number of decisions either side of --focus")
    args=parser.parse_args()
    if args.radius < 1 or (args.case and (args.focus is not None or args.best_fusion)):
        parser.error("Use a positive window radius and one episode selector")
    if args.best_fusion:
        candidates = []
        for path in sorted(args.episode.rglob("summary.json")):
            summary = json.loads(path.read_text())
            delta = summary.get("calibrated_fusion_mean_delta")
            if delta is not None and delta > 0 and "calibrated active fusion" in summary.get("case_groups", {}):
                candidates.append((delta, path.parent))
        if not candidates:
            print("No recorded calibrated active fusion in this export.")
            return
        _, args.episode = max(candidates, key=lambda entry: entry[0])
        args.case = "calibrated active fusion"
        print("Selected calibrated fusion from", args.episode)
    if (args.episode/"sequence.npz").is_file():
        import numpy as np
        from grace.imagination_figures import render
        meta=json.loads((args.episode/"manifest.json").read_text())
        if meta["status"]!="saved" or args.case or args.replay or args.focus is not None:
            parser.error("Use a saved imagination sequence without episode options")
        args.output.mkdir(parents=True,exist_ok=False)
        with np.load(args.episode/"sequence.npz",allow_pickle=False) as data:
            styles = ("original", "paper") if args.style == "both" else (args.style or "paper",)
            for style in styles:
                stem = "imagination_" + style if args.style else "imagination"
                render(data["rgb"], data["heat"], data["jumps"], meta,
                       args.output / (stem+"."+args.format), dpi=260, style=style)
        return
    if args.style:
        parser.error("--style requires an imagination sequence")
    snap,refs=load_episode(args.episode,args.replay)
    mode = snap["summary"].get("experiment_mode", "")
    if mode == "grace_d_only" and args.focus is not None:
        parser.error("D-only episodes have no C control timeline")
    if args.focus is not None and not any(r["step"] == args.focus for r in snap["rows"]):
        parser.error("--focus is outside the recorded timeline")
    class Offline(EpisodeVisualizer):
        def _save(self,fig,path,*,tight=True):
            from grace.figure_style import save
            save(fig,path.with_suffix("."+args.format),260,tight=tight)
            fig.clear()
    if args.case and args.case not in snap["groups"]:
        parser.error("Unavailable group. Recorded: " + ", ".join(snap["groups"]))
    args.output.mkdir(parents=True,exist_ok=False)
    viz=Offline(args.output,dict(experiment_mode=mode, vnext_spatial_fix_version=snap["summary"].get("spatial_fix_version",0),
                                   goal_ref_version=snap["summary"].get("goal_ref_version")),
                scheduled_eval_maps=False)
    viz.single_column=args.single_column
    viz.goal_references=refs
    if args.focus is not None:
        from grace.paper_figures import timeline
        timeline(viz, snap, args.output/"C1_window.png", args.focus, args.radius)
    elif args.case:
        from grace.paper_figures import decision, goal
        event=snap["groups"][args.case]
        if event.get("goal_rgb") is not None:
            goal(viz,event,args.output/"regional_evidence.png")
        if event.get("has_prediction"):
            decision(viz,event,args.output/"candidate_comparison.png","AC")
        if mode != "grace_d_only":
            decision(viz,event,args.output/"control.png","C")
    else:
        viz._render(snap,args.output)
    viz.close()


if __name__=="__main__":
    main()
