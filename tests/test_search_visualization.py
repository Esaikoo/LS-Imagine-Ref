"""Real plotting plus bounded scheduling/causal-frame visualization regressions."""
import copy
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.visualization import SearchVisualizer
from evaluate_checkpoints import (create_search_visualizer, record_search_visual_decision,
                                  finish_search_visual_episode, worker_runtime_args,
                                  reference_specification, validate_reference)


def fixture(index=0, predicted=True):
    image = np.full((64, 64, 3), index, np.uint8)
    obs = dict(image=image, heatmap=np.full((64, 64), 80, np.uint8), is_first=index == 0)
    prediction = np.zeros((12, 12), np.float32)
    if predicted:
        prediction[:, :7] = np.linspace(.05, .65, 7)
        prediction[:, 9] = 3/8
    context = np.zeros(24, np.float32); context[11] = .75 if index > 0 else .2
    output = dict(log_spatial_version=np.float32(2),
                  log_spatial_summary=np.r_[1., np.zeros(11)].astype(np.float32),
                  log_spatial_patch=np.zeros((21, 21, 6), np.uint8),
                  log_spatial_pose=np.array([index*.2, index*.1, index*.05, .1]),
                  log_vnext_context=context, log_vnext_decision=1.,
                  log_vnext_choice=index % 4, log_vnext_actor_action=6.,
                  log_vnext_source=1., log_vnext_override=float(index % 4 != 0),
                  log_vnext_phase=index % 4,
                  log_vnext_mask=np.ones(12), log_vnext_prediction=prediction,
                  log_vnext_q=np.linspace(.1, .8, 12) if predicted else np.zeros(12),
                  log_vnext_a_score=np.linspace(-.2, .4, 12))
    return obs, output


def completed(viz, mode, env_id, end, length=5, predicted=True, **kwargs):
    for i in range(length):
        obs, output = fixture(i, predicted)
        viz.observe(mode, env_id, obs, output, end-length+i)
        viz.feedback(mode, env_id, float(i == length-1), dict(success=i == length-1))
    episode = dict(reward=np.array([0.]*(length)+[1.]), success=np.array([False]*length+[True]),
                   search_visible=np.array([0.]*(length-1)+[1., 1.]),
                   search_visible_valid=np.ones(length+1))
    grid = np.zeros((9, 9, 6), np.float32)
    grid[4, 4, :2] = [1, 1]; grid[4, 5, 2] = .8
    state = dict(grid=grid, pose=[length*.2, length*.1, length*.05, .1], origin=[-4., -4.],
                 resolution=1., predicted=True, tracking_ready=True)
    return viz.episode_end(mode, env_id, episode, end, map_state=state, **kwargs)


class VisualizationTests(unittest.TestCase):
    def test_fix3_segment_map_and_history_are_distinct_from_global_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            viz=SearchVisualizer(directory,dict(task="sand",vnext_spatial_fix_version=3))
            for i in range(4):
                obs,out=fixture(i)
                out.update(log_spatial_trajectory_valid=0.,log_spatial_pose_gap_count=1.,
                    log_spatial_segment=float(i>=2),log_spatial_local_valid=1.,
                    log_spatial_memory_available=1.,log_vnext_source=8. if i<2 else 9.)
                viz.observe("eval","segmented",obs,out,100)
                viz.feedback("eval","segmented",0.,{})
            state=dict(grid=np.zeros((9,9,6)),pose=[.5,.1,0.,.2],origin=[-4.,-4.],
                resolution=1.,tracking_ready=False,local_tracking_ready=True,active_segment=1,
                memory_available=True,gap_count=1,
                history_frames=np.full((2,4,4,4),80,np.uint8),
                history_meta=np.zeros((2,8)),history_valid=np.ones(2),
                ledger=dict(cells=np.array([[100,100],[0,0]]),
                    values=np.array([[8,8,0,0,0,0],[1,1,0,0,0,0]]),
                    segments=np.array([0,1])))
            path=viz.episode_end("eval","segmented",dict(reward=np.zeros(5)),100,
                                eval_index=3,eval_total=3,map_state=state)
            summary=json.loads((path/"summary.json").read_text())
            self.assertIsNone(summary["estimated_trajectory_length"])
            self.assertFalse(summary["trajectory_length_valid"])
            self.assertTrue(summary["persistent_memory_available"])
            self.assertEqual(summary["memory_available_fraction"],1.)
            self.assertEqual(summary["history_slots_retained"],2)
            self.assertEqual(summary["estimated_visited_cells"],1)
            self.assertEqual(summary["active_segment"],1)
            self.assertEqual(summary["rendered_segment"],1)
            self.assertEqual(summary["local_tracking_fraction"],1.)
            self.assertEqual(summary["localized_fraction"],0.)
            sources={x["source"] for x in summary["selected_keyframes_for_C"]}
            self.assertIn("bounded search probe",sources)
            self.assertTrue((path/"B_map.png").exists())
            with np.load(path/"episode_trace.npz",allow_pickle=False) as data:
                np.testing.assert_array_equal(data["segment"],[0,0,1,1])
                np.testing.assert_array_equal(data["local_valid"],[1,1,1,1])
                np.testing.assert_array_equal(data["memory_available"],[1,1,1,1])
                np.testing.assert_array_equal(data["ledger_segments"],[0,1])
                np.testing.assert_array_equal(data["history_frames"],state["history_frames"])
                self.assertEqual(data["history_frames"].dtype,np.uint8)

    def test_fix3_prefers_wider_qualified_segment_over_terminal_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_spatial_fix_version=3))
            for i in range(5):
                obs, output = fixture(i)
                if i >= 2:
                    output["log_spatial_pose"][:2] = [(.01 if i == 3 else 0.), 0.]
                output.update(log_spatial_trajectory_valid=float(i < 2),
                              log_spatial_segment=i//2, log_spatial_local_valid=1.)
                viz.observe("eval", "prior_segment", obs, output, 100)
            state = dict(grid=np.zeros((9,9,6)), pose=[0.,0.,0.,.1], resolution=1.,
                         tracking_ready=True, local_tracking_ready=True, trajectory_valid=False,
                         active_segment=2, pose_gap_count=2, first_gap_step=2)
            state["pose"] = [.01, .01, 0., .1]
            with mock.patch.object(viz, "_render") as render:
                path = viz.episode_end("eval", "prior_segment", dict(reward=np.zeros(6)), 100,
                                       eval_index=1, eval_total=1, map_state=state)
            summary = json.loads((path/"summary.json").read_text())
            self.assertEqual(summary["rendered_segment"],0)
            self.assertEqual(summary["rendered_segment_steps"],2)
            np.testing.assert_allclose(summary["rendered_segment_extent_m"], [.2, .1])
            self.assertEqual(summary["pose_gap_count"],2)
            self.assertEqual(summary["first_pose_gap_step"],2)
            self.assertAlmostEqual(summary["localized_fraction"],2/5)
            self.assertEqual(summary["local_tracking_fraction"],1.)
            self.assertIsNone(summary["estimated_trajectory_length"])
            with mock.patch.object(viz, "_save") as save:
                viz._render_map(render.call_args.args[0], path/"B_map.png")
            ax = save.call_args.args[0].axes[0]
            self.assertIn("earlier qualified",ax.get_title())
            np.testing.assert_allclose(ax.lines[0].get_xdata()[:2],[0.,.2])
            self.assertTrue(np.isnan(ax.lines[0].get_xdata()[2:]).all())
            self.assertFalse(save.call_args.args[0].axes[1].child_axes)
            self.assertAlmostEqual(np.ptp(ax.get_xlim()), 24.)
            self.assertAlmostEqual(np.ptp(ax.get_ylim()), 24.)

    def test_pose_gap_does_not_create_complete_path_and_align_names_match_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            viz=SearchVisualizer(directory,dict(task="sand",vnext_spatial_fix_version=2))
            for i in range(4):
                obs,out=fixture(i)
                out["log_vnext_phase"]=1
                out["log_vnext_choice"]=7
                out["log_vnext_source"]=7
                out["log_spatial_trajectory_valid"]=float(i<2)
                out["log_spatial_pose_gap_count"]=float(i>=2)
                if i>=2:out["log_spatial_pose"][:2]=100.
                viz.observe("eval","gap",obs,out,80)
                viz.feedback("eval","gap",0.,{})
            state=dict(grid=np.zeros((9,9,6)),pose=[100.,100.,0.,10.],
                       origin=[-4.,-4.],resolution=1.,tracking_ready=False,gap_count=1)
            path=viz.episode_end("eval","gap",dict(reward=np.zeros(5)),80,
                                  eval_index=3,eval_total=3,map_state=state)
            summary=json.loads((path/"summary.json").read_text())
            self.assertIsNone(summary["estimated_trajectory_length"])
            self.assertFalse(summary["trajectory_length_valid"])
            self.assertEqual(summary["pose_gap_count"],1)
            self.assertEqual(summary["first_pose_gap_step"],2)
            self.assertAlmostEqual(summary["integrated_path_length"],np.sqrt(.05),places=6)
            self.assertEqual(viz._names(12,1)[6:8],["align_up","align_down"])
            self.assertEqual(viz._names(12,0)[6:8],["scan","clear_obstacle"])
            self.assertTrue(any(r["source"]=="alignment probe" for r in summary["selected_keyframes_for_C"]))
            with np.load(path/"episode_trace.npz") as data:
                np.testing.assert_array_equal(data["pose_ready"],[1,1,0,0])
                np.testing.assert_array_equal(data["pose_gap_count"],[0,0,1,1])

    def test_bounded_recovery_probe_is_distinct_from_learned_control(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_spatial_fix_version=1))
            for i in range(2):
                obs, output = fixture(i)
                output["log_vnext_source"] = 6.
                output["log_vnext_phase"] = 3
                output["log_vnext_choice"] = 7
                viz.observe("eval", "probe", obs, output, 50)
                viz.feedback("eval", "probe", 0., {})
            path = viz.episode_end("eval", "probe", dict(reward=np.zeros(3)), 50,
                                   eval_index=3, eval_total=3)
            summary = json.loads((path/"summary.json").read_text())
            self.assertEqual(summary["spatial_fix_version"], 1)
            self.assertTrue(any(row["source"] == "bounded recovery probe"
                                for row in summary["selected_keyframes_for_C"]))
            with np.load(path/"episode_trace.npz") as trace:
                np.testing.assert_array_equal(trace["source"], [6, 6])

    def test_reference_runtime_offline_flag_does_not_relax_seed_or_world_checks(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = dict(seed=1, runs=1, episodes=3, fps=20)
            (root/"manifest.json").write_text(json.dumps(protocol))
            folder = root/"harvest_sand"/"run_00"; folder.mkdir(parents=True)
            config = dict(seed=1, minedojo_gradle_offline=False, search_eval_evidence=False)
            task = dict(LS_Imagine_specs={})
            sim = dict(world_seed="42", start_time=6000, allow_time_passage=False)
            (folder/"settings.json").write_text(json.dumps(dict(config=config, task_specs=task, sim_specs=sim)))
            args = SimpleNamespace(**protocol, reference_output=root, worker_episode=None,
                                   heatmap_mode="unet", a_control="checkpoint", c_control="checkpoint")
            changed = dict(config, minedojo_gradle_offline=True, search_eval_evidence=True)
            result = validate_reference(args, "harvest_sand", 0, changed,
                                        dict(task, minedojo_gradle_offline=True), sim)
            self.assertTrue(result["common_config_matches"])
            self.assertTrue(result["audit_only_differences"]["candidate"]["task"]["minedojo_gradle_offline"])
            audit = result["audit_only_differences"]
            self.assertFalse(audit["reference"]["config"]["search_eval_evidence"])
            self.assertTrue(audit["candidate"]["config"]["search_eval_evidence"])
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_reference(args, "harvest_sand", 0, dict(changed, seed=2), task, sim)
            with self.assertRaisesRegex(ValueError, "sim_specs"):
                validate_reference(args, "harvest_sand", 0, changed, task, dict(sim, world_seed="43"))

    def test_evaluation_reaps_remaining_minecraft_processes_when_one_exits(self):
        import psutil
        from evaluate_checkpoints import reap_evaluation_process_tree

        parent, live, exited = mock.Mock(), mock.Mock(), mock.Mock()
        parent.children.return_value = [live, exited]
        exited.terminate.side_effect = psutil.NoSuchProcess(42)
        with mock.patch.object(psutil, "wait_procs", side_effect=[([parent, exited], [live]), ([live], [])]):
            reap_evaluation_process_tree(parent)
        parent.terminate.assert_called_once_with()
        live.terminate.assert_called_once_with()
        live.kill.assert_called_once_with()

    def test_standalone_eval_offline_launch_is_runtime_only(self):
        from types import SimpleNamespace
        self.assertEqual(worker_runtime_args(SimpleNamespace()), [])
        self.assertEqual(worker_runtime_args(SimpleNamespace(minedojo_gradle_offline=True)),
                         ["--minedojo-gradle-offline"])
        task = dict(LS_Imagine_specs={}, minedojo_gradle_offline=True)
        before = copy.deepcopy(task)
        base = reference_specification(dict(LS_Imagine_specs={}), {})
        offline = reference_specification(task, {})
        self.assertEqual(base[:2], offline[:2])
        self.assertTrue(offline[2]["task"]["minedojo_gradle_offline"])
        self.assertEqual(task, before)

    def test_standalone_eval_factory_respects_saved_flags_and_root(self):
        flags = dict(search_vnext_enabled=True, vnext_spatial_enabled=True,
                     vnext_visual_enabled=True, vnext_visual_root="",
                     vnext_visual_map_all_eval_episodes=True)
        with mock.patch("apm.visualization.SearchVisualizer") as constructor:
            self.assertIsNone(create_search_visualizer({}, "unused"))
            self.assertIsNone(create_search_visualizer(dict(flags, vnext_spatial_enabled=False), "unused"))
            self.assertIsNone(create_search_visualizer(dict(flags, vnext_visual_enabled=False), "unused"))
            constructor.assert_not_called()
            self.assertIs(create_search_visualizer(flags, Path("run")), constructor.return_value)
            self.assertEqual(constructor.call_args.args[0], Path("run"))
            self.assertFalse(constructor.call_args.kwargs["scheduled_eval_maps"])
            self.assertTrue(constructor.call_args.args[1]["vnext_visual_map_all_eval_episodes"])
            custom = dict(flags, vnext_visual_root="custom-visual-root")
            create_search_visualizer(custom, "ignored")
            self.assertEqual(constructor.call_args.args[0], Path("custom-visual-root"))

    def test_standalone_eval_taps_pre_action_native_frame_without_input_mutation(self):
        visualizer = mock.Mock()
        obs = dict(image=np.zeros((64, 64, 3), np.uint8), heatmap=np.zeros((64, 64), np.uint8), is_first=True)
        old = copy.deepcopy(obs)
        rgb = np.full((160, 256, 3), 55, np.uint8)
        heatmap = np.ones((160, 256, 1), np.float32)
        result = dict(action=np.eye(12)[[0]], log_spatial_pose=np.array([[0., 0., .1, .2]]),
                      log_goal_version=np.array([1.]))
        record_search_visual_decision(visualizer, "ep", obs, result, 100, rgb, heatmap)
        args = visualizer.observe.call_args.args
        self.assertEqual(args[:2], ("eval", "ep"))
        self.assertIs(args[2]["image"], rgb)
        self.assertIs(args[2]["heatmap"], heatmap)
        self.assertIs(args[2]["goal_original_heatmap"], obs["heatmap"])
        self.assertIs(args[2]["goal_policy_rgb"], obs["image"])
        np.testing.assert_array_equal(args[3]["log_spatial_pose"], result["log_spatial_pose"][0])
        for key in obs: np.testing.assert_array_equal(obs[key], old[key])
        record_search_visual_decision(None, "no-op", None, None, 100)

    def test_standalone_eval_terminal_map_only_for_last_rollout(self):
        visualizer, agent = mock.Mock(), mock.Mock()
        state, terminal, episode = object(), dict(image="terminal"), dict(reward=[0, 1])
        for index in (1, 2):
            finish_search_visual_episode(visualizer, agent, state, terminal,
                                           "ep", episode, 100, index, 3)
            self.assertIsNone(visualizer.episode_end.call_args.kwargs["map_state"])
        agent.visualization_map.assert_not_called()
        finish_search_visual_episode(visualizer, agent, state, terminal, "ep", episode, 100, 3, 3)
        agent.visualization_map.assert_called_once_with(state, 0, terminal)
        self.assertIs(visualizer.episode_end.call_args.kwargs["map_state"], agent.visualization_map.return_value)
        visualizer.episode_end.side_effect = OSError("plot error")
        with self.assertRaisesRegex(OSError, "plot error"):
            finish_search_visual_episode(visualizer, agent, state, terminal, "ep", episode, 100, 3, 3)

    def test_maps_for_selected_eval_round_and_latest_complete_train_episode_only(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10,
                                                     vnext_visual_map_all_eval_episodes=True))
            try:
                with mock.patch.object(viz, "_render_timeline"), mock.patch.object(viz, "_render_decision"), mock.patch.object(viz, "_render_outcomes"):
                    self.assertIsNone(viz.episode_end("train", "prefill", dict(reward=[0, 0]), 0))
                    self.assertIsNone(completed(viz, "train", "learned", 8))
                    completed(viz, "eval", "ep_1", 10, eval_index=1, eval_total=3)
                    viz.episode_end("eval", "no_trace", dict(reward=[0, 0]), 10,
                                    eval_index=2, eval_total=3)
                    completed(viz, "eval", "ep_3", 10, eval_index=3, eval_total=3)
                    viz.maybe_export_train(10)
                    self.assertIsNone(completed(viz, "train", "between_intervals", 16))
                    for index in (1, 2, 3):
                        self.assertIsNone(completed(viz, "eval", f"skipped_{index}", 15,
                                                    eval_index=index, eval_total=3))
                    viz.close()
                eval_round = viz.root/"eval/step_000000010"
                for index in (1, 2):
                    self.assertEqual({p.name for p in (eval_round/f"episode_{index:02d}").iterdir()},
                                     {"B_map.png"})
                self.assertTrue((eval_round/"episode_03/summary.json").is_file())
                self.assertFalse((viz.root/"eval/step_000000015").exists())
                self.assertEqual(len(list((viz.root/"eval").glob("**/B_map.png"))), 3)
                train = viz.root/"train/step_000000010/episode_last"
                self.assertEqual(list((viz.root/"train").glob("**/B_map.png")), [train/"B_map.png"])
                self.assertEqual(json.loads((train/"summary.json").read_text())["episode_id"], "learned")
                self.assertFalse((viz.root/"B_episodes").exists())
                self.assertEqual(viz.stats["map_exports"], 2)
                self.assertEqual(viz.stats["map_errors"], 0)
                with Image.open(eval_round/"episode_02/B_map.png") as blank, Image.open(train/"B_map.png") as route:
                    self.assertEqual(blank.size, (1200, 840))
                    self.assertEqual(blank.size, route.size)
            finally:
                viz.close()

    def test_b_map_uses_fixed_square_panel_and_a_minimum_world_span(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_map_min_span=24.))
            for i in range(2):
                viz.observe("eval", "scale", *fixture(i), 100)
            with mock.patch.object(viz, "_render") as render:
                path = viz.episode_end("eval", "scale", dict(reward=[0, 0, 0]), 100,
                                       eval_index=1, eval_total=1)
            snap = render.call_args.args[0]
            bounds = []
            for span in (0.2, 80.):
                snap["rows"][-1]["pose"][:2] = [span, 0.]
                with mock.patch.object(viz, "_save") as save:
                    viz._render_map(snap, path/"B_map.png")
                ax = save.call_args.args[0].axes[0]
                width, height = np.diff(ax.get_xlim()).item(), np.diff(ax.get_ylim()).item()
                self.assertAlmostEqual(width, height)
                self.assertGreaterEqual(width, max(24., span))
                bounds.append(ax.get_position().bounds)
            np.testing.assert_allclose(*bounds)

    def test_eval_only_last_and_real_complete_png_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="minedojo_harvest_sand", vnext_visual_every=10,
                                                     vnext_visual_strict=True))
            for index in (1, 2):
                self.assertIsNone(completed(viz, "eval", "eval_%d" % index, 100,
                                            eval_index=index, eval_total=3))
            self.assertFalse(viz.root.exists())
            path = completed(viz, "eval", "eval_3", 100, eval_index=3, eval_total=3)
            files = {p.name for p in path.iterdir()}
            self.assertTrue({"B_map.png", "C1_timeline.png", "summary.json", "episode_trace.npz"} <= files)
            self.assertLessEqual(len(list(path.glob("A1_*.png"))), 3)
            self.assertLessEqual(len(list(path.glob("A2_*.png"))), 2)
            self.assertLessEqual(len(list(path.glob("C2_*.png"))), 2)
            for png in path.glob("*.png"):
                with Image.open(png) as im:
                    im.verify()
            summary = json.loads((path/"summary.json").read_text())
            self.assertEqual(summary["episode_id"], "eval_3")
            self.assertTrue(summary["success"])
            np.testing.assert_allclose(summary["estimated_terminal_pose"], [1., .5, .25, .1])
            self.assertEqual(summary["first_target_step"], 4)
            self.assertEqual(summary["localized_fraction"], 1.)
            self.assertEqual(summary["unlocalized_steps"], 0)
            self.assertTrue(summary["trajectory_length_valid"])
            self.assertIn("viewpoints", summary["target_map_semantics"])
            self.assertTrue(all("predictions only" in s["unexecuted_candidates"]
                                for s in summary["selected_keyframes_for_A"]))
            viz.close()

    def test_unlocalized_decision_keeps_only_real_visual_history(self):
        with tempfile.TemporaryDirectory() as root:
            viz = SearchVisualizer(root, dict(task="sand", vnext_spatial_fix_version=3))
            obs, out = fixture()
            frames = np.zeros((3, 4, 4, 4), dtype=np.uint8)
            frames[0, :, :, :3], frames[2, :, :, :3] = 40, 80
            out.update(log_spatial_local_valid=0., log_spatial_pose_model_ready=0.,
                       log_spatial_memory_available=1., log_spatial_history_frames=frames,
                       log_spatial_history_valid=np.array([1., 0., 1.]))
            viz.observe("eval", "unlocalized", obs, out, 0)
            event = viz._active[("eval", "unlocalized")].c[0]
            self.assertFalse(event["row"]["pose_model_ready"])
            self.assertFalse(event["row"]["local_ready"])
            self.assertEqual(event["history_preview"].shape, (4, 8, 3))
            self.assertTrue((event["history_preview"][:, :4] == 40).all())
            self.assertTrue((event["history_preview"][:, 4:] == 80).all())

    def test_goal_keyframe_prefers_actual_fusion_over_raw_heat_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", goal_ref_use_raw_image=True))
            for i, (raw, effective, gate) in enumerate(((40, 40, 0), (200, 200, 0),
                                                       (60, 64, .04), (230, 230, .2))):
                obs, out = fixture(i)
                obs.update(heatmap=np.full((64, 64), raw, np.uint8),
                           goal_image=np.full((96, 96, 3), 80 + i, np.uint8),
                           goal_image_valid=True)
                out.update(log_goal_state=0, log_goal_gate=gate, log_goal_conflict=0.,
                           log_goal_model_update=12, log_goal_region_cal_reliability=.125,
                           log_goal_context=np.array([.5, .1, .9, 0., 0., 0., 0., 1.]),
                           log_goal_heatmap_policy=np.full((64, 64), effective, np.uint8),
                           log_goal_ref_map=np.full((64, 64), 100, np.uint8),
                           log_goal_boxes=np.array([[.25, .25, .75, .75]]),
                           log_goal_candidate_scores=np.array([.5]),
                           log_goal_candidate_sources=np.array([1]),
                           log_goal_new_area_forecast=np.full(12, .25))
                viz.observe("eval", "goal", obs, out, 100)
                viz.feedback("eval", "goal", 0., {})
                if i == 1:
                    keyframe = viz._active[("eval", "goal")].goal
                    self.assertEqual(keyframe["row"]["step"], 1)
                    self.assertEqual(keyframe["goal_selection_reason"],
                                     "strongest original heat (fallback)")
            keyframe = viz._active[("eval", "goal")].goal
            self.assertEqual(keyframe["goal_rgb"].shape, (96, 96, 3))
            self.assertTrue((keyframe["goal_rgb"] == 82).all())
            path = viz.episode_end("eval", "goal", dict(reward=np.zeros(5)), 100,
                                   eval_index=3, eval_total=3)
            with Image.open(path/"G_goal_verification.png") as image:
                image.verify()
            summary = json.loads((path/"summary.json").read_text())
            saved = summary["goal_keyframe"]
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["goal_rgb_source"], "native D input")
            self.assertEqual(saved["goal_selection_reason"], "largest applied heatmap change")
            self.assertAlmostEqual(saved["goal_heat_delta_mean_max"][0], 4/255., places=6)
            self.assertEqual(saved["goal_region_cal_reliability"], .125)
            self.assertEqual(saved["goal_new_area_forecast"], [.25] * 12)

    def test_train_threshold_uses_latest_completed_before_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10))
            with mock.patch.object(viz, "_render"):
                completed(viz, "train", "earlier", 5, length=2)
                completed(viz, "train", "last_before_ten", 9, length=2)
                # Its observe at step 10 crosses the threshold while still open.
                completed(viz, "train", "future_completion", 15, length=6)
            paths = list(viz.root.glob("train/step_000000010/episode_last/summary.json"))
            self.assertEqual(len(paths), 1)
            saved = json.loads(paths[0].read_text())
            self.assertEqual(saved["episode_id"], "last_before_ten")
            self.assertEqual(saved["episode_completed_at_training_step"], 9)
            self.assertEqual(saved["export_training_step"], 10)

    def test_exact_threshold_completion_and_resume_no_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10))
            viz.set_step(30)
            with mock.patch.object(viz, "_render"):
                completed(viz, "train", "at_forty", 40, length=2)
            self.assertEqual(viz.stats["exports"], 1)
            self.assertEqual(len(list(viz.root.glob("train/*/episode_last/summary.json"))), 1)
            self.assertEqual(viz.stats["skipped_no_episode"], 0)

    def test_reused_complete_episode_at_later_threshold_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10))
            with mock.patch.object(viz, "_render"):
                completed(viz, "train", "last_complete", 9, length=2)
                first = viz.maybe_export_train(10)
                second = viz.maybe_export_train(20)
            a = json.loads((first/"summary.json").read_text())
            b = json.loads((second/"summary.json").read_text())
            self.assertFalse(a["repeated_complete_episode"])
            self.assertTrue(b["repeated_complete_episode"])
            self.assertEqual(b["episode_age_at_export_steps"], 11)
            self.assertEqual(a["episode_id"], b["episode_id"])

    def test_prefill_and_uncomputed_candidates_never_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10))
            obs, output = fixture()
            output.pop("log_spatial_version")
            viz.observe("train", "random", obs, output, 0)
            self.assertFalse(viz._active)
            with mock.patch.object(viz, "_render_outcomes") as outcomes:
                path = completed(viz, "eval", "no_prediction", 10, length=2,
                                 predicted=False, eval_index=1, eval_total=1)
                outcomes.assert_not_called()
            self.assertEqual([item.name for item in path.glob("A*.png")], ["A_status.png"])
            saved = json.loads((path/"summary.json").read_text())
            self.assertEqual(saved["selected_keyframes_for_A"], [])
            self.assertIn("No eligible", saved["candidate_prediction_note"])
            self.assertIsNone(saved["selected_keyframes_for_C"][0]["c_value"])

    def test_mid_episode_prefill_prefix_waits_for_real_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_every=10))
            for i in range(5):
                obs, output = fixture(i+1)  # Learner first sees an ongoing episode.
                viz.observe("train", "prefill_tail", obs, output, i+5)
                viz.feedback("train", "prefill_tail", 0., {})
            self.assertFalse(viz._active)
            self.assertEqual(viz.stats["skipped_partial_prefix"], 1)
            self.assertEqual(len(viz._partial_prefixes), 1)
            result = viz.episode_end("train", "prefill_tail", dict(reward=np.zeros(6)), 10)
            self.assertIsNone(result)
            self.assertFalse(viz._partial_prefixes)
            self.assertIsNone(viz._latest_train)
            self.assertFalse(list(viz.root.glob("**/summary.json")))
            # A subsequent real reset may start a complete episode, even if an
            # environment happens to reuse its identifier.
            with mock.patch.object(viz, "_render"):
                result = completed(viz, "train", "prefill_tail", 20, length=3)
            self.assertIsNotNone(result)
            summary = json.loads((result/"summary.json").read_text())
            self.assertEqual(summary["total_steps"], 3)
            self.assertEqual(summary["episode_completed_at_training_step"], 20)
            self.assertEqual(viz.stats["skipped_partial_prefix"], 1)

    def test_buffers_are_bounded_and_inputs_not_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_max_steps=8))
            for i in range(100):
                obs, output = fixture(i)
                before = copy.deepcopy((obs, output))
                viz.observe("eval", "long", obs, output, 100)
                for d, expected in zip((obs, output), before):
                    for key in d:
                        np.testing.assert_array_equal(d[key], expected[key])
                viz.feedback("eval", "long", 0., {})
            ep = viz._active[("eval", "long")]
            self.assertEqual(len(ep.rows), 8)
            self.assertEqual(ep.omitted_rows, 92)
            self.assertLessEqual(len(ep.a), 3); self.assertLessEqual(len(ep.c), 2)
            self.assertNotIn("image", ep.rows[0])
            viz.close()
            self.assertFalse(viz._active)
            with self.assertRaises(RuntimeError): viz.observe("eval", "x", *fixture(), 100)

    def test_unlocalized_map_never_claims_actual_static_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_max_steps=3))
            for i in range(5):
                obs, output = fixture(i)
                output["log_spatial_summary"][0] = 0.
                output["log_spatial_pose"][:2] = 0.
                viz.observe("eval", "unlocalized", obs, output, 100)
                viz.feedback("eval", "unlocalized", 0., {})
            state = dict(grid=np.zeros((9, 9, 6)), pose=[0., 0., .5, 10.],
                         origin=[-4., -4.], resolution=1., tracking_ready=False)
            with mock.patch.object(viz, "_render") as render:
                path = viz.episode_end("eval", "unlocalized",
                                       dict(reward=np.zeros(6), search_visible=np.zeros(6),
                                            search_visible_valid=np.zeros(6)), 100,
                                       eval_index=1, eval_total=1, map_state=state)
            with mock.patch.object(viz, "_save") as save:
                viz._render_map(render.call_args.args[0], path/"B_map.png")
            ax = save.call_args.args[0].axes[0]
            self.assertIn("UNLOCALIZED", [t.get_text() for t in ax.texts])
            self.assertFalse(ax.get_xlabel())
            self.assertEqual(len(ax.collections), 0)
            summary = json.loads((path/"summary.json").read_text())
            self.assertEqual(summary["localized_fraction"], 0.)
            self.assertEqual(summary["unlocalized_steps"], 5)
            self.assertFalse(summary["terminal_tracking_ready"])
            self.assertFalse(summary["trajectory_length_valid"])
            self.assertIsNone(summary["estimated_trajectory_length"])
            self.assertIsNone(summary["first_target_step"])
            self.assertEqual(summary["target_step_source"], "unavailable")
            self.assertEqual(summary["integrated_path_length"], 0.)
            self.assertIn("does not imply actual stillness", summary["localization_note"])
            with np.load(path/"episode_trace.npz") as data:
                self.assertFalse(data["pose_ready"].any())

    def test_full_episode_ledger_outside_rolling_window_is_retained_and_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="synthetic_sparse_map_test"))
            for i in range(2):
                obs, output = fixture(i)
                viz.observe("eval", "sparse", obs, output, 100)
            values = np.zeros((2, 6)); values[:, 0] = [2, 3]; values[:, 1] = [1, 12]; values[:, 3] = [1, 2]
            state = dict(grid=np.zeros((9, 9, 6)), pose=[.4, .2, .1, .1], origin=[-4, -4], resolution=1.,
                         tracking_ready=True, ledger=dict(cells=np.array([[0, 0], [100000, 0]]),
                         values=values, directions=np.ones((2, 12))), ledger_capacity=1024)
            with mock.patch.object(viz, "_render_timeline"), mock.patch.object(viz, "_render_decision"), mock.patch.object(viz, "_render_outcomes"):
                path = viz.episode_end("eval", "sparse", dict(reward=[0, 0, 0]), 100,
                                       eval_index=1, eval_total=1, map_state=state)
            summary = json.loads((path/"summary.json").read_text())
            self.assertTrue(summary["full_episode_ledger_available"])
            self.assertEqual(summary["estimated_visited_cells"], 2)
            self.assertEqual(summary["number_of_revisits"], 3.)
            with Image.open(path/"B_map.png") as im:
                self.assertLess(im.width, 3000)
            with np.load(path/"episode_trace.npz") as data:
                self.assertEqual(data["ledger_cells"].shape, (2, 2))
                self.assertEqual(data["map_grid"].shape, (9, 9, 6))

    def test_export_error_explicit_and_no_partial_success_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_strict=False))
            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output), mock.patch.object(viz, "_render", side_effect=OSError("disk test")):
                path = completed(viz, "eval", "failure", 100, eval_index=1, eval_total=1)
            self.assertIsNone(path); self.assertEqual(viz.stats["errors"], 1)
            self.assertIn("disk test", viz.last_error)
            self.assertIn("export failed", error_output.getvalue())
            self.assertFalse(list(viz.root.glob("**/summary.json")))
            self.assertFalse(list(viz.root.glob("**/.render-*")))
            viz.close(); viz.close()

    def test_rendering_preserves_python_numpy_and_torch_rng(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            viz = SearchVisualizer(directory, dict(task="sand", vnext_visual_strict=True))
            random.seed(119); np.random.seed(119); torch.manual_seed(119)
            before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            completed(viz, "eval", "rng", 100, length=2, eval_index=1, eval_total=1)
            self.assertEqual(random.getstate(), before[0])
            after = np.random.get_state()
            self.assertEqual(after[0], before[1][0]); np.testing.assert_array_equal(after[1], before[1][1])
            self.assertEqual(after[2:], before[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))


if __name__ == "__main__":
    unittest.main()
