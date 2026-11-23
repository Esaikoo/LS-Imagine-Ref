"""Read-only decoding of one already-computed LS-Imagine trajectory at a training threshold."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch


def transition_types(flags, start, sequence_length):
    """Column t labels the outgoing edge t -> t+1; the terminal padding has no edge."""
    flags = np.asarray(flags)
    if flags.ndim != 1 or not len(flags) or start < 0 or start + len(flags) > sequence_length:
        raise ValueError("Invalid imagination sequence span")
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Non-binary transition flags")
    return ["terminal" if start + i == sequence_length - 1 else "long" if flag else "short"
            for i, flag in enumerate(flags)]


def render(rgb, heat, transitions, meta, path, dpi=140, style="paper"):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle, Patch

    if style not in ("original", "paper"):
        raise ValueError(f"Unknown imagination style: {style}")
    n = len(rgb)
    if n != len(heat) or n != len(transitions):
        raise ValueError("Different numbers of RGB, heatmap and transition columns")
    edges = transition_types(transitions, meta["start"], meta["sequence_length"])
    paper = style == "paper"
    colors = {"short": "#3c71ae" if paper else "blue", "long": "#bc4231" if paper else "red",
              "terminal": "#858585"}
    # Limit the physical file dimensions, while keeping one column per source state.
    fig = Figure(figsize=(min(24, max(6.5, .92*n + 1.2)), 4.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, n, squeeze=False)
    fig.subplots_adjust(left=.08, right=.99, top=.826, bottom=.107, wspace=.04, hspace=.25)
    for i, edge in enumerate(edges):
        for row in range(3):
            ax = axes[row, i]
            if row != 1:
                ax.imshow(rgb[i], interpolation="nearest")
            if row != 0:
                ax.imshow(heat[i], interpolation="nearest", vmin=0, vmax=1,
                          cmap="viridis" if paper else "jet", alpha=1 if row == 1 else .3)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,
                                   edgecolor=colors[edge],lw=3,clip_on=False))
            if i == 0:
                ax.set_ylabel(("RGB", "Heatmap", "Overlay")[row], fontsize=9)
            if row == 0:
                ax.set_title(str(meta["start"] + i), fontsize=8, pad=3)
    version = meta.get("d_version")
    provenance = meta.get("source", "Archived model decode")
    display_version = (f"D v{version}" if version else
                       "D not used" if meta.get("fixture") else "D version unrecorded")
    fig.suptitle(f"LS-Imagine  |  {style.capitalize()} style  ·  {provenance}  ·  {display_version}",
                 fontsize=12, y=.98)
    step_kind = meta.get("step_kind", "Environment step")
    fig.text(.08, 1-.48/fig.get_figheight(),
             f"{step_kind} {meta['environment_step']:,}   ·   Imagination step across columns   ·   "
             f"Trajectory {meta['column']} (zero-based) of {meta['batch_size']}   ·   "
             f"Showing {n} of {meta['sequence_length']} states", fontsize=9)
    fig.legend(handles=[Patch(color=colors[key], label=title) for key, title in
                        (("short", "Short outgoing transition"), ("long", "Long outgoing transition"),
                         ("terminal", "Final state: no outgoing transition"))],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clear()


class ImaginationSampler:
    def __init__(self,config,initial_step):
        self.interval=int(getattr(config,"vnext_visual_every",10000))
        if self.interval<=0:
            raise ValueError("Imagination sampling interval must be positive")
        self.root=Path(config.logdir)/"check_obs"
        self.last=int(initial_step)//self.interval
        self.d_version = getattr(config, "goal_ref_version", None) if getattr(config, "goal_ref_enabled", False) else None

    def capture(self,step,features,jumps,decoder):
        bucket=int(step)//self.interval
        if bucket<=self.last:
            return
        self.last=bucket
        folder=self.root/f"step_{bucket*self.interval:09d}"
        meta=dict(environment_step=int(step),threshold=bucket*self.interval,status="pending",
                  selection="first trajectory with a long outgoing transition, in batch order; otherwise column zero",
                  d_version=self.d_version, source="Model decode",
                  transition_index="outgoing from the displayed state")
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            try:
                folder.mkdir()
            except FileExistsError:
                return
            self._write_meta(folder,meta)
            if features is None or jumps is None or not features.numel() or not jumps.numel():
                meta.update(status="skipped",reason="no imagination states")
            else:
                py_state=random.getstate()
                np_state=np.random.get_state()
                try:
                    with torch.no_grad(),torch.random.fork_rng(devices=[features.device] if features.is_cuda else []):
                        if jumps.shape!=(*features.shape[:2],1):
                            raise ValueError("Feature/transition dimensions differ")
                        raw_flags = jumps[..., 0].detach()
                        if not ((raw_flags == 0) | (raw_flags == 1)).all() or raw_flags[-1].any():
                            raise ValueError("Invalid jump_record or nonzero terminal padding")
                        flags = raw_flags.bool()
                        cols = flags[:-1].any(dim=0).nonzero().flatten()
                        col = int(cols[0]) if len(cols) else 0
                        full = flags[:, col].cpu().numpy()
                        hits = np.flatnonzero(full[:-1])
                        pred = decoder(features[:, col].detach().unsqueeze(0))
                        rgb=pred["image"].mode()[0].float().clamp(0,1).mul(255).round().byte().cpu().numpy()
                        heat=pred["heatmap"].mode()[0,...,0].float().clamp(0,1).cpu().numpy()
                        meta.update(column=col,start=0,sequence_length=len(full),batch_size=features.shape[1],
                                    trajectories_with_long_transition=len(cols),
                                    selection_reason="first eligible trajectory" if len(cols) else "no long transition in batch",
                                    long_transition_indices=hits.tolist())
                        np.savez_compressed(folder/"sequence.npz",rgb=rgb,heat=heat,jumps=full)
                        render(rgb,heat,full,meta,folder/"imagination.png")
                        meta["status"]="saved"
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
            self._write_meta(folder,meta)
        except Exception as error:
            print(f"[imagination] {folder}: {error}",file=sys.stderr,flush=True)
            if folder.is_dir():
                meta.update(status="error",reason=str(error))
                try:
                    self._write_meta(folder,meta)
                except OSError as log_error:
                    print(f"[imagination] Could not persist failure status: {log_error}", file=sys.stderr,flush=True)

    @staticmethod
    def _write_meta(folder,meta):
        pending=folder/"manifest.tmp"
        pending.write_text(json.dumps(meta,indent=2)+"\n")
        pending.replace(folder/"manifest.json")
