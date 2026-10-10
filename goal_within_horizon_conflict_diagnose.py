"""Read-only, same-state production350/400 factual conflict measurements."""
import hashlib

import numpy as np

import goal_action_choice_diagnose as choices
import goal_library as gl
import goal_within_horizon_supervision as supervision

FORMAT = "ls_imagine_within_horizon_conflict_diagnosis_v1"
PROTOCOL = "production350_400_reference64_within80_retained84_40_readonly_v1"
CONDITIONS = supervision.CONDITIONS
VERSIONS = (350, 400)
POOL_SIZES = {"reference": 64, "within80": 80, "retained84": 40}


def pools(data):
    return (("reference", data.reference_arrays, data.runs),
            ("within80", data.feedback_arrays, data.feedback_runs),
            ("retained84", data.continuation_arrays, data.continuation_runs))


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def query_plan(data):
    """Keep source order, episode splits, actual endpoints and incoming actions."""
    rows = []
    for pool, arrays, runs in pools(data):
        gl.require(len(arrays["labels"]) == POOL_SIZES[pool], "完整184事实池大小不同")
        for index in range(POOL_SIZES[pool]):
            run = runs[int(arrays["run"][index])]
            reference = pool == "reference"
            manual = not reference and bool(arrays["manual"][index])
            correction = not reference and bool(arrays["correction"][index])
            train = bool(arrays["train"][index])
            split = ("train" if train else "validation") if reference else run["episode_split"]
            branch = "script_fact" if reference else "worker_fact" if not manual else "correction" if correction else "holding"
            if pool == "within80":
                original = data.feedback_queries[index]
                branch = original["branch"]
                source = original["decision_source"]
            elif reference:
                source = "accepted_real_reference_script"
            else:
                source = run["decision_sources"][index % 4]
            factual_pool = ("reference_" + split if reference else "manual_" + split if manual else "worker_facts_only")
            frame, remaining = int(arrays["frame"][index]), int(arrays["remaining"][index])
            row = dict(query_index=len(rows), pool=pool, pool_row=index, run=int(arrays["run"][index]),
                repeat=run["repeat"], target=int(arrays["target"][index]), episode_split=split,
                factual_pool=factual_pool, branch=branch, decision_source=source, manual=manual,
                correction_applied=correction, state_frame=frame, incoming_action_frame=frame + 1,
                remaining=remaining, endpoint_frame=run["endpoint_frame"], actual_action=int(arrays["labels"][index]),
                trajectory_path=run["trajectory_path"], trajectory_sha256=run["trajectory_sha256"],
                episode_key=run["trajectory_path"] + "#" + run["trajectory_sha256"],
                state_sha256=sha(arrays["features"][index]), endpoint_goal_sha256=sha(arrays["goals"][index]),
                fixed_goal_sha256=sha(data.fixed_goals[int(arrays["target"][index])]),
                original_fact_source="reference_script" if reference else "worker350_within_horizon" if pool == "within80" else "worker300_continuation",
                eligible_source_row_for_fixed400_added_training=bool(train and (reference or pool == "within80" and manual)),
                actual_endpoint_is_hindsight=True, actual_endpoint_is_deployment_input=False,
                expert_labels=False, approved_for_training=False, compared_versions=list(VERSIONS))
            gl.require(frame >= run["start_frame"] and frame < run["endpoint_frame"] and
                remaining == run["endpoint_frame"] - frame and row["actual_action"] == run["action_ids"][frame-run["start_frame"]],
                "事实状态/真实incoming/本局终点与预算不同")
            rows.append(row)
    gl.require(len(rows) == 184 and len({(r["episode_key"], r["state_frame"]) for r in rows}) == 184,
               "完整184查询重复、缺失或借用另一局状态")
    return rows


def validate_query(row, expected):
    gl.require(all(row.get(k) == v for k, v in expected.items()) and
        type(row.get("remaining")) is int and 1 <= row["remaining"] <= 16 and
        row["incoming_action_frame"] == row["state_frame"] + 1 and
        row["state_frame"] + row["remaining"] == row["endpoint_frame"] and
        row["expert_labels"] is False and row["approved_for_training"] is False and
        row["actual_endpoint_is_deployment_input"] is False,
        "查询真实状态/下一动作/预算/自身终点/划分/来源或版本不同；禁止移植事实")


def validate_queries(rows, expected):
    gl.require(len(rows) == len(expected) == 184 and
        [(r.get("query_index"), r.get("pool"), r.get("pool_row")) for r in rows] ==
        [(r["query_index"], r["pool"], r["pool_row"]) for r in expected],
        "184查询遗漏、重复、计划外或顺序不同")
    for row, identity in zip(rows, expected):
        validate_query(row, identity)


def historical_goal_bindings(data):
    """Map original evaluation contexts without changing the six main conditions."""
    result = []
    for pool, arrays, runs in pools(data):
        if pool == "within80":
            continue
        for source in ("own_real_endpoint", "fixed_old_targets"):
            for index in range(POOL_SIZES[pool]):
                run = runs[int(arrays["run"][index])]
                target = int(arrays["target"][index])
                selected = arrays["goals"][index] if source == "own_real_endpoint" else data.fixed_goals[target]
                alternate, donor = data.fixed_goals[1-target], None
                if pool == "reference" and source == "own_real_endpoint":
                    candidates = [i for i,item in enumerate(runs)
                                  if item["repeat"] == run["repeat"] and item["target"] != target]
                    gl.require(len(candidates) == 1, "旧参考实际交换目标需要同repeat唯一另一目标局")
                    donor = runs[candidates[0]]
                    indices = np.flatnonzero(arrays["run"] == candidates[0])
                    gl.require(len(indices) == 16 and donor["endpoint_frame"] == run["endpoint_frame"] == 80 and
                        bool(arrays["train"][indices[0]]) == bool(arrays["train"][index]),
                        "旧参考实际交换目标必须同repeat/同划分，仅作评价")
                    alternate = arrays["goals"][indices[0]]
                result.append(dict(pool=pool, pool_row=index, run=int(arrays["run"][index]),
                    state_frame=int(arrays["frame"][index]), remaining=int(arrays["remaining"][index]),
                    goal_source=source, goal_sha256=sha(selected), historical_swapped_goal_sha256=sha(alternate),
                    historical_swapped_goal_source="opposite_episode_actual_endpoint_same_repeat" if donor else "opposite_fixed_visual_target",
                    donor_trajectory_path=donor["trajectory_path"] if donor else None,
                    donor_trajectory_sha256=donor["trajectory_sha256"] if donor else None,
                    donor_run=candidates[0] if donor else None,
                    main_swapped_goal_source="opposite_fixed_visual_target",
                    main_swapped_goal_sha256=sha(data.fixed_goals[1-target]),
                    historical_swap_has_distinct_context=donor is not None,
                    evaluation_only=True, approved_for_training=False, endpoint_label_replaced=False))
    gl.require(len(result) == 208, "旧保留目标来源/交换上下文有遗漏")
    return result


def validate_historical_binding(binding, expected):
    gl.require(binding == expected and binding.get("evaluation_only") is True and
        binding.get("approved_for_training") is False and binding.get("endpoint_label_replaced") is False,
        "旧保留目标来源/交换上下文不同；不能将另一局评价目标替换本局标签")


def compare(versions, action_names):
    """Raw margins cancel additive common offsets; changes are descriptive."""
    gl.require(set(versions) == {"350", "400"} and len(action_names) == 12 and
        len(set(action_names)) == 12 and all(n in action_names for n in ("noop", "jump", "turn_up", "turn_down")),
        "两生产版本或原宏动作名称不同")
    result = {}
    selected = {n: action_names.index(n) for n in ("noop", "turn_up", "jump", "turn_down")}
    pairs = (("turn_up", "jump"), ("turn_down", "noop"), ("jump", "noop"), ("turn_up", "noop"))
    for condition in CONDITIONS:
        old, new = (versions[str(v)] for v in VERSIONS)
        p0, p1 = (choices.probability(v["probabilities"][condition]) for v in (old, new))
        raw0, raw1 = (np.asarray(v["raw_preferences"][condition], np.float64) for v in (old, new))
        base0, base1 = (np.asarray(v["raw_preferences"]["base"], np.float64) for v in (old, new))
        gl.require(np.array_equal(base0, base1) and np.array_equal(old["probabilities"]["base"], new["probabilities"]["base"]),
                   "同状态350/400冻结底座概率或raw不同")
        m0, m1 = old["metrics"], new["metrics"]
        result[condition] = dict(
            factual_nll_delta=m1[condition+"_factual_nll"]-m0[condition+"_factual_nll"],
            factual_probability_delta=m1[condition+"_factual_probability"]-m0[condition+"_factual_probability"],
            total_variation=float(np.abs(p1-p0).sum()/2), mode_changed=bool(p0.argmax() != p1.argmax()),
            lost_factual_mode=bool(m0[condition+"_mode_matches_factual"] and not m1[condition+"_mode_matches_factual"]),
            gained_factual_mode=bool(not m0[condition+"_mode_matches_factual"] and m1[condition+"_mode_matches_factual"]),
            probability_delta=(p1-p0).tolist(), residual_raw_delta=((raw1-base1)-(raw0-base0)).tolist(),
            selected_actions={name: dict(probability350=float(p0[i]), probability400=float(p1[i]),
                probability_delta=float(p1[i]-p0[i]), residual_raw350=float(raw0[i]-base0[i]),
                residual_raw400=float(raw1[i]-base1[i])) for name, i in selected.items()},
            raw_pair_margins={a+"_minus_"+b: dict(before=float(raw0[selected[a]]-raw0[selected[b]]),
                after=float(raw1[selected[a]]-raw1[selected[b]]),
                change=float((raw1[selected[a]]-raw1[selected[b]])-(raw0[selected[a]]-raw0[selected[b]]))) for a, b in pairs})
    return result


def aggregate(rows):
    if not rows:
        return dict(rows=0, independent_episodes=0, metrics=None, comparisons=None, action_counts=None)
    names = [k for k, v in rows[0]["versions"]["350"]["metrics"].items()
             if type(v) in (int, float, bool) and not k.endswith("_mode")]
    return dict(rows=len(rows), independent_episodes=len({r["episode_key"] for r in rows}),
        action_counts=np.bincount([r["actual_action"] for r in rows], minlength=12).tolist(),
        metrics={str(v): {k: float(np.mean([r["versions"][str(v)]["metrics"][k] for r in rows])) for k in names} for v in VERSIONS},
        comparisons={c: dict(mean_factual_nll_delta=float(np.mean([r["comparison"][c]["factual_nll_delta"] for r in rows])),
            mean_total_variation=float(np.mean([r["comparison"][c]["total_variation"] for r in rows])),
            lost_factual_modes=sum(r["comparison"][c]["lost_factual_mode"] for r in rows),
            gained_factual_modes=sum(r["comparison"][c]["gained_factual_mode"] for r in rows),
            mode_changes=sum(r["comparison"][c]["mode_changed"] for r in rows),
            matches350=sum(r["versions"]["350"]["metrics"][c+"_mode_matches_factual"] for r in rows),
            matches400=sum(r["versions"]["400"]["metrics"][c+"_mode_matches_factual"] for r in rows),
            selected_action_mean_probability_delta={n:float(np.mean([r["comparison"][c]["selected_actions"][n]["probability_delta"] for r in rows]))
                for n in ("noop", "turn_up", "jump", "turn_down")},
            mean_raw_pair_margin_change={n:float(np.mean([r["comparison"][c]["raw_pair_margins"][n]["change"] for r in rows]))
                for n in rows[0]["comparison"][c]["raw_pair_margins"]}) for c in CONDITIONS})


def summarize(rows, expected, action_names):
    validate_queries(rows, expected)
    groups = {}
    for pool in POOL_SIZES:
        splits = ("train", "validation") if pool == "reference" else ("train", "development_holdout")
        branches = ("script_fact",) if pool == "reference" else ("holding", "correction", "manual_noop_without_requested_preference", "worker_fact")
        for target in (0, 1):
            for split in splits:
                for branch in branches:
                    selected = [r for r in rows if (r["pool"], r["target"], r["episode_split"], r["branch"]) == (pool, target, split, branch)]
                    groups[f"{pool}_target{target}_{split}_{branch}"] = aggregate(selected)
    return dict(format=FORMAT, comparison_protocol=PROTOCOL, query_rows=184, episodes=34,
        distributions_saved=184*2*6, action_names=action_names, by_pool={p:aggregate([r for r in rows if r["pool"] == p]) for p in POOL_SIZES},
        by_target_split_branch=groups,
        by_pool_remaining={p:{str(b):aggregate([r for r in rows if r["pool"] == p and r["remaining"] == b])
            for b in range(1, 17 if p == "reference" else 5)} for p in POOL_SIZES},
        by_pool_action={p:{n:aggregate([r for r in rows if r["pool"] == p and r["actual_action"] == i])
            for i, n in enumerate(action_names)} for p in POOL_SIZES},
        missing_within80_train_correction_targets=[t for t in (0,1) if not any(r["pool"] == "within80" and
            r["target"] == t and r["episode_split"] == "train" and r["manual"] and r["correction_applied"] for r in rows)],
        cross_episode_states_are_not_matched=True, factual_mode_match_is_not_expert_accuracy=True,
        parameter_causality_established=False, automatic_repair_or_confirmation=False,
        approved_for_training=False, behavior_accepted=False, t06_approved=False)


def regressions(rows):
    """Index all improvements and regressions, retaining the full query table."""
    def pointer(row, condition):
        return dict(query_index=row["query_index"], pool=row["pool"], run=row["run"], target=row["target"],
            episode_split=row["episode_split"], state_frame=row["state_frame"], remaining=row["remaining"],
            actual_action=row["actual_action"], branch=row["branch"], condition=condition,
            before=row["versions"]["350"]["metrics"][condition+"_mode"],
            after=row["versions"]["400"]["metrics"][condition+"_mode"],
            **{k:row["comparison"][condition][k] for k in ("factual_nll_delta", "factual_probability_delta")})
    return dict(lost_factual_modes=[pointer(r,c) for r in rows for c in CONDITIONS if r["comparison"][c]["lost_factual_mode"]],
        gained_factual_modes=[pointer(r,c) for r in rows for c in CONDITIONS if r["comparison"][c]["gained_factual_mode"]],
        all_actual_correction_queries=[r["query_index"] for r in rows if r["correction_applied"]],
        full_query_table_retained=True, training_subset_selected=False, expert_accuracy=False)


def flat_rows(rows, action_names):
    result = []
    for row in rows:
        scalar = {k:v for k,v in row.items() if type(v) in (str,int,float,bool) or v is None}
        scalar["actual_action_name"] = action_names[row["actual_action"]]
        for version in VERSIONS:
            scalar.update({f"v{version}_{k}":v for k,v in row["versions"][str(version)]["metrics"].items()})
        for condition in CONDITIONS:
            values = row["comparison"][condition]
            scalar.update({f"{condition}_{k}":v for k,v in values.items() if type(v) in (int,float,bool)})
            scalar.update({f"{condition}_{name}_probability_delta":v["probability_delta"] for name,v in values["selected_actions"].items()})
            scalar.update({f"{condition}_{name}_raw_margin_change":v["change"] for name,v in values["raw_pair_margins"].items()})
        result.append(scalar)
    return result
