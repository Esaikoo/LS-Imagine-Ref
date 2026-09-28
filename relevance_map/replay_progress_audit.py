#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze LS-Imagine train_eps replays for NACLIP-P85 ScoreStorage behavior.

Edit REPLAY_DIR only, then run in PyCharm.

Outputs:
  00_summary.txt
  01_phase_summary.csv
  02_zoom_events.csv
  03_episode_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

REPLAY_DIR = Path(
    r"/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/train_eps"
)
OUTPUT_DIR = PROJECT_ROOT / "relevance_map" / "replay_progress_audit_outputs"

MAX_STEPS = 1000
ACTION_REPEAT = 1

PHASES = [
    (0, 250_000, "0-250k"),
    (250_000, 500_000, "250k-500k"),
    (500_000, 750_000, "500k-750k"),
    (750_000, 1_000_001, "750k-1M"),
]

# Official ScoreStorage uses strict: score_on_zoomed < score
CROSSING_EPS = 0.0


def flat_bool(x):
    return np.asarray(x).reshape(-1).astype(bool)


def flat_float(x):
    return np.asarray(x).reshape(-1).astype(np.float64)


def phase_of(step):
    for lo, hi, name in PHASES:
        if lo <= step < hi:
            return name
    return "outside_0_1M"


def replay_one(score, source, zoomed_score):
    """
    Reproduce ScoreStorage:
      resolve if current score > zoomed score
      OR age >= MAX_STEPS
    """
    rel = np.flatnonzero(
        score[source:] > (zoomed_score + CROSSING_EPS)
    )

    if len(rel):
        j_cross = int(rel[0])
        if j_cross <= MAX_STEPS:
            k = source + j_cross
            return {
                "reason": "crossing",
                "replay_j": j_cross,
                "crossing_step": k,
                "crossing_score": float(score[k]),
                "crossing_minus_zoomed_score":
                    float(score[k] - zoomed_score),
            }

    if (len(score) - 1 - source) >= MAX_STEPS:
        k = source + MAX_STEPS
        return {
            "reason": "timeout",
            "replay_j": MAX_STEPS,
            "crossing_step": k,
            "crossing_score": float(score[k]),
            "crossing_minus_zoomed_score": np.nan,
        }

    return {
        "reason": "pending",
        "replay_j": np.nan,
        "crossing_step": np.nan,
        "crossing_score": np.nan,
        "crossing_minus_zoomed_score": np.nan,
    }


def numeric_stats(series, prefix):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_p25": float(x.quantile(0.25)),
        f"{prefix}_p75": float(x.quantile(0.75)),
        f"{prefix}_p90": float(x.quantile(0.90)),
        f"{prefix}_max": float(x.max()),
    }


def summarize(df, name):
    eligible = df[df["scorestorage_eligible"] == 1]
    calc = eligible[eligible["is_calculated"] == 1]
    crossing = eligible[eligible["replay_reason"] == "crossing"]
    timeout = eligible[eligible["replay_reason"] == "timeout"]
    pending = eligible[eligible["replay_reason"] == "pending"]

    n = len(eligible)
    nc = len(calc)

    row = {
        "phase": name,
        "all_is_zoomed": int(len(df)),
        "accepted_zoom": int(n),
        "is_calculated": int(nc),
        "resolve_rate": nc / n if n else np.nan,
        "crossing_count": int(len(crossing)),
        "crossing_rate": len(crossing) / n if n else np.nan,
        "timeout_count": int(len(timeout)),
        "timeout_rate": len(timeout) / n if n else np.nan,
        "pending_count": int(len(pending)),
        "pending_rate": len(pending) / n if n else np.nan,
    }

    j = pd.to_numeric(
        calc["saved_jumping_steps"], errors="coerce"
    ).dropna()

    row.update(numeric_stats(j, "J"))
    row["J_le_5_count"] = int((j <= 5).sum())
    row["J_le_5_rate"] = float((j <= 5).mean()) if len(j) else np.nan
    row["J_le_20_count"] = int((j <= 20).sum())
    row["J_le_20_rate"] = float((j <= 20).mean()) if len(j) else np.nan

    row.update(
        numeric_stats(
            eligible["zoomed_minus_source_score"],
            "zoomed_minus_source_score",
        )
    )
    row.update(
        numeric_stats(
            crossing["crossing_minus_zoomed_score"],
            "crossing_minus_zoomed_score",
        )
    )

    checkable = calc[
        calc["saved_j_matches_replay"].isin([0, 1])
    ]
    row["saved_J_replay_match_rate"] = (
        float(checkable["saved_j_matches_replay"].mean())
        if len(checkable)
        else np.nan
    )

    return row


def main():
    replay_dir = REPLAY_DIR.expanduser().resolve()
    if not replay_dir.exists():
        raise FileNotFoundError(
            f"REPLAY_DIR does not exist:\n{replay_dir}\n"
            "Edit REPLAY_DIR at the top of the script."
        )

    files = sorted(replay_dir.glob("*.npz"), key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No npz files found in {replay_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_rows = []
    episode_rows = []
    global_step = 0

    required = {
        "reward", "is_zoomed", "is_calculated",
        "score", "score_on_zoomed", "jumping_steps",
    }

    print(f"Found {len(files)} episodes")

    for fi, path in enumerate(files):
        with np.load(path, allow_pickle=False) as ep:
            missing = required - set(ep.files)
            if missing:
                print(f"[SKIP] {path.name}: missing {sorted(missing)}")
                continue

            reward = flat_float(ep["reward"])
            score = flat_float(ep["score"])
            score_z = flat_float(ep["score_on_zoomed"])
            zoomed = flat_bool(ep["is_zoomed"])
            calc = flat_bool(ep["is_calculated"])
            saved_j = flat_float(ep["jumping_steps"])

            n = len(reward)
            if "is_last" in ep:
                is_last = flat_bool(ep["is_last"])
            else:
                is_last = np.zeros(n, dtype=bool)
                is_last[-1] = True

            ep_start = global_step
            ep_steps = max(0, n - 1) * ACTION_REPEAT

            zoom_indices = np.flatnonzero(zoomed)

            for s in zoom_indices:
                s = int(s)

                # tools.simulate only enqueues zoom when not done.
                eligible = not bool(is_last[s])

                replay = (
                    replay_one(score, s, float(score_z[s]))
                    if eligible
                    else {
                        "reason": "terminal_not_enqueued",
                        "replay_j": np.nan,
                        "crossing_step": np.nan,
                        "crossing_score": np.nan,
                        "crossing_minus_zoomed_score": np.nan,
                    }
                )

                sj = float(saved_j[s])
                match = np.nan
                if (
                    eligible
                    and bool(calc[s])
                    and np.isfinite(replay["replay_j"])
                ):
                    match = int(
                        abs(sj - replay["replay_j"]) < 0.5
                    )

                gstep = ep_start + s * ACTION_REPEAT

                event_rows.append({
                    "episode_file": path.name,
                    "episode_index": fi,
                    "local_source_step": s,
                    "global_source_step": gstep,
                    "phase": phase_of(gstep),
                    "scorestorage_eligible": int(eligible),
                    "is_calculated": int(calc[s]),
                    "source_current_score": float(score[s]),
                    "score_on_zoomed": float(score_z[s]),
                    "zoomed_minus_source_score":
                        float(score_z[s] - score[s]),
                    "saved_jumping_steps": sj,
                    "replay_reason": replay["reason"],
                    "replay_j": replay["replay_j"],
                    "replay_crossing_local_step":
                        replay["crossing_step"],
                    "replay_crossing_score":
                        replay["crossing_score"],
                    "crossing_minus_zoomed_score":
                        replay["crossing_minus_zoomed_score"],
                    "saved_j_matches_replay": match,
                })

            episode_rows.append({
                "episode_file": path.name,
                "episode_index": fi,
                "global_start_step": ep_start,
                "global_end_step": ep_start + ep_steps,
                "frames": n,
                "action_steps": n - 1,
                "is_zoomed_count": int(zoomed.sum()),
                "is_calculated_count": int(calc.sum()),
            })

            global_step += ep_steps

        if (fi + 1) % 100 == 0 or fi == len(files) - 1:
            print(
                f"[{fi+1}/{len(files)}] "
                f"reconstructed steps={global_step:,}, "
                f"zoom events={len(event_rows):,}"
            )

    events = pd.DataFrame(event_rows)
    episodes = pd.DataFrame(episode_rows)

    if events.empty:
        raise RuntimeError("No is_zoomed events found.")

    summary_rows = [summarize(events, "ALL")]
    for _, _, phase_name in PHASES:
        summary_rows.append(
            summarize(
                events[events["phase"] == phase_name],
                phase_name,
            )
        )

    summary = pd.DataFrame(summary_rows)

    events.to_csv(
        OUTPUT_DIR / "02_zoom_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    episodes.to_csv(
        OUTPUT_DIR / "03_episode_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "01_phase_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "LS-Imagine Replay Progress Audit",
        "=" * 72,
        f"Replay dir: {replay_dir}",
        f"NPZ files: {len(files)}",
        f"Reconstructed steps: {global_step:,}",
        "",
    ]

    for row in summary_rows:
        lines += [
            f"[{row['phase']}]",
            f"accepted zoom       = {row['accepted_zoom']}",
            f"is_calculated       = {row['is_calculated']}",
            f"resolve rate        = {row['resolve_rate']:.4f}",
            f"timeout rate        = {row['timeout_rate']:.4f}",
            f"J mean              = {row['J_mean']:.2f}",
            f"J median            = {row['J_median']:.2f}",
            f"J P25/P75/P90       = "
            f"{row['J_p25']:.2f} / {row['J_p75']:.2f} / {row['J_p90']:.2f}",
            f"J max               = {row['J_max']:.2f}",
            f"J <= 5              = {row['J_le_5_rate']:.4f}",
            f"J <= 20             = {row['J_le_20_rate']:.4f}",
            f"zoomed-source gap   = "
            f"{row['zoomed_minus_source_score_mean']:.6f} mean",
            f"crossing overshoot  = "
            f"{row['crossing_minus_zoomed_score_mean']:.6f} mean",
            f"saved-J replay match= "
            f"{row['saved_J_replay_match_rate']:.4f}",
            "",
        ]

    if global_step < 900_000:
        lines.append(
            "[WARN] Reconstructed steps < 900k. "
            "This may not be the full 1M train_eps directory."
        )

    report = "\n".join(lines)
    (OUTPUT_DIR / "00_summary.txt").write_text(
        report, encoding="utf-8"
    )

    print()
    print(report)
    print()
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
