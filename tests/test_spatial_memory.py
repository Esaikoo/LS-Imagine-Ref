"""Planar motion, causal search mapping and real gradient regressions; no game."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.odometry import PoseNet, local_pose_delta, wrap_angle
from apm.spatial_memory import (SpatialMapMemory, SPATIAL_VERSION, SUMMARY_DIM,
                                centered_cell, heading_sector)


def config(**updates):
    values = dict(vnext_memory_dim=16, vnext_map_size=35, vnext_map_patch=21,
                  vnext_pose_hidden=32, vnext_pose_min_samples=1)
    values.update(updates)
    return SimpleNamespace(**values)


def inputs(batch=1):
    return (torch.zeros(batch, 24), torch.zeros(batch, 64, 64, 3),
            torch.zeros(batch, 64, 64), torch.zeros(batch, 9))


class SpatialMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_goal_history_survives_unknown_pose_without_fabricating_map_points(self):
        model = SpatialMapMemory(24, config(vnext_spatial_fix_version=3,goal_search_enabled=True,
            target_item="wool"))
        feat,image,heat,control = inputs()
        first = torch.tensor([True])
        goal = torch.tensor([[.9,.8,.1,0.,0.,0.,2.,1.]])
        context,state,logs = model.online(feat,image,heat,is_first=first,goal_context=goal)
        self.assertEqual(context.shape,(1,16))
        self.assertFalse(bool(state["tracking_ready"][0]))
        self.assertEqual(state["ledger_used"].item(),0)
        self.assertGreater(logs["log_goal_memory"][0,1].item(),.8)
        self.assertFalse(logs["log_goal_direction_novelty"].any())
        self.assertGreater(state["history_goal"][0,0,1].item(),.8)
        _,next_state,after = model.online(feat,image,heat,state=state,is_first=~first,
            previous_control=control,goal_context=torch.zeros_like(goal))
        self.assertLess(after["log_goal_memory"][0,1].item(),logs["log_goal_memory"][0,1].item())
        self.assertGreater(after["log_goal_memory"][0,1].item(),.7)
        self.assertEqual(next_state["ledger_used"].item(),0)

    def test_supervision_local_rotation_and_angle_seam(self):
        result = local_pose_delta([2, 60, 3], [1, 99, 3], 90, 95)
        np.testing.assert_allclose(result, [0, 1, math.radians(5)], atol=1.e-6)
        result = local_pose_delta([0, 0, 0], [0, 10, 0], 179, -179)
        self.assertAlmostEqual(float(result[2]), math.radians(2), places=5)
        # Singleton batch/time dimensions must not collapse in the angle term.
        value = local_pose_delta(np.zeros((1, 1, 3)), np.ones((1, 1, 3)),
                                 np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
        self.assertEqual(value.shape, (1, 1, 3))
        self.assertAlmostEqual(float(wrap_angle(2*math.pi+.2)), .2)

    def test_integration_local_forward_respects_estimated_yaw(self):
        pose = torch.tensor([[0., 0., math.pi/2, 0.]])
        delta = torch.tensor([[0., 1., math.pi, 0.]])[:, :3]
        out = SpatialMapMemory.integrate(pose, delta, torch.zeros(1, 3))
        torch.testing.assert_close(out[:, :2], torch.tensor([[-1., 0.]]), atol=1.e-6, rtol=0)
        self.assertAlmostEqual(float(out[0, 2]), -math.pi/2, places=5)
        torch.testing.assert_close(pose, torch.tensor([[0., 0., math.pi/2, 0.]]))

    def test_negative_cell_and_sector_boundaries_match_numpy_and_torch(self):
        values = np.array([-1.5001, -1.5, -.7501, -.75, -.7499, -.1, 0., .7499, .75, .7501])
        expected = np.array([-1, -1, -1, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(centered_cell(values, 1.5), expected)
        np.testing.assert_array_equal(centered_cell(torch.tensor(values), 1.5).numpy(), expected)
        np.testing.assert_array_equal(centered_cell(values+4.5, 1.5), expected+3)
        angle = np.deg2rad(np.array([-15.001, -15., -.1, 0., 14.999, 15., 30., 359., 360.]))
        expected = np.array([11, 0, 0, 0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(heading_sector(angle), expected)
        np.testing.assert_array_equal(heading_sector(torch.tensor(angle)).numpy(), expected)

    def test_zero_predictor_cannot_qualify_on_small_real_moving_steps(self):
        model = PoseNet(hidden=16, min_samples=4, min_calibration=4, min_moving=2, cache_size=64)
        image = torch.zeros(8, 64, 64, 3)
        target = torch.zeros(8, 3)
        target[[0, 1, 4, 5], 0] = .2
        train = torch.arange(8) < 4
        calibration = ~train
        ids = torch.arange(8)
        _, _, pred = model.supervised_loss(image, image, torch.zeros(8, 9), target, train,
            calibration_mask=calibration, frame_ids=ids)
        self.assertLess(float(model.position_error_ema), .25)
        self.assertEqual(int(model.calibration_moving_samples), 2)
        self.assertFalse(model.ready)
        self.assertAlmostEqual(float(model.calibration_moving_error), .2, places=5)
        # Qualification must reflect correct held-out predictions, not an easy
        # overall MAE. Refreshing the same identities cannot create new support.
        model.record_predictions(target, target, train, calibration, ids)
        self.assertTrue(model.ready)
        self.assertEqual(int(model.supervised_samples), 4)
        self.assertEqual(int(model.calibration_samples), 4)
        self.assertEqual(float(model.calibration_moving_error), 0.)

    def test_stationary_resampling_never_creates_moving_or_unique_support(self):
        model = PoseNet(hidden=16, min_samples=4, min_calibration=4, min_moving=2, cache_size=64)
        image = torch.zeros(8, 64, 64, 3)
        train = torch.arange(8) < 4
        for _ in range(5):
            model.supervised_loss(image, image, torch.zeros(8, 9), torch.zeros(8, 3), train,
                calibration_mask=~train, frame_ids=torch.arange(8))
        self.assertEqual(int(model.supervised_samples), 4)
        self.assertEqual(int(model.calibration_samples), 4)
        self.assertEqual(int(model.moving_samples), 0)
        self.assertFalse(model.ready)

    def test_command_only_walker_cannot_qualify_when_blocked(self):
        model = PoseNet(hidden=16,min_samples=4,min_calibration=4,min_moving=2,cache_size=64)
        target = torch.zeros(8,3); target[[0,1,4,5],0] = .2
        estimate = target.clone(); estimate[[2,3,6,7],0] = .2
        train = torch.arange(8)<4
        model.record_predictions(estimate,target,train,~train,torch.arange(8))
        self.assertEqual(float(model.calibration_moving_error),0.)
        self.assertLess(float(model.calibration_static_error),.25)
        self.assertFalse(model.ready)

    def test_tracking_can_enable_only_at_real_episode_start(self):
        model = SpatialMapMemory(24, config(vnext_pose_min_calibration=4,
            vnext_pose_min_moving=2, vnext_pose_cache_size=64))
        f, image, heat, control = inputs()
        _, state, before = model.online(f, image, heat, is_first=torch.ones(1).bool())
        self.assertEqual(float(before["log_spatial_summary"][0, 0]), 0.)
        target = torch.zeros(8, 3); target[[0, 1, 4, 5], 0] = .2
        train = torch.arange(8) < 4
        model.pose.record_predictions(target, target, train, ~train, torch.arange(8))
        self.assertTrue(model.pose.ready)
        _, state, middle = model.online(f, image, heat, state, previous_control=control)
        self.assertEqual(float(middle["log_spatial_summary"][0, 0]), 0.)
        snapshot = model.visualization_state(state)
        self.assertFalse(snapshot["tracking_ready"])
        self.assertIn("unknown", snapshot["localization"])
        _, _, reset = model.online(f, image, heat, state, is_first=torch.ones(1).bool())
        self.assertEqual(float(reset["log_spatial_summary"][0, 0]), 1.)
        # Missing history in mid-prefill must remain unlocalized too.
        _, _, prefix = model.online(f, image, heat, None, is_first=torch.zeros(1).bool())
        self.assertEqual(float(prefix["log_spatial_summary"][0, 0]), 0.)

    def test_calibration_targets_do_not_backpropagate(self):
        model = PoseNet(hidden=16, min_samples=4, min_calibration=4, min_moving=2, cache_size=64)
        image = torch.zeros(4, 64, 64, 3)
        target = torch.zeros(4, 3); target[:2, 0] = .2
        loss, _, _ = model.supervised_loss(image, image, torch.zeros(4, 9), target,
            torch.zeros(4).bool(), calibration_mask=torch.ones(4).bool(), frame_ids=torch.arange(4))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(model.supervised_samples), 0)
        self.assertEqual(int(model.calibration_samples), 4)
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum()) == 0. for p in model.parameters()))

    def test_untrained_forward_command_does_not_invent_translation(self):
        model = SpatialMapMemory(24, config())
        feature, image, heatmap, control = inputs()
        control[:, 1] = 1
        _, state, _ = model.online(feature, image, heatmap, is_first=torch.ones(1).bool())
        for _ in range(8):
            _, state, info = model.online(feature, image, heatmap, state, previous_control=control)
        self.assertEqual(float(state["pose"][0, :2].abs().sum()), 0.)
        self.assertEqual(float(info["log_spatial_summary"][0, 0]), 0.)
        self.assertEqual(float(info["log_spatial_visited_cells"][0]), 1.)
        self.assertGreater(float(info["log_spatial_pose"][0, 3]), 0.)

    def test_same_place_new_direction_then_repeated_direction(self):
        model = SpatialMapMemory(24, config())
        f, image, h, control = inputs()
        _, state, initial = model.online(f, image, h)
        self.assertEqual(float(initial["log_spatial_novelty"][0]), 1.)
        first_patch = initial["log_spatial_patch"].clone()
        control[:, 8] = 30.
        for index in range(12):
            _, state, info = model.online(f, image, h, state, previous_control=control)
            if index < 11:
                self.assertEqual(float(info["log_spatial_novelty"][0]), 1.)
        self.assertEqual(float(info["log_spatial_novelty"][0]), 0.)
        self.assertEqual(float(info["log_spatial_directions"].sum()), 12.)
        self.assertEqual(float(info["log_spatial_visited_cells"][0]), 1.)
        self.assertEqual(info["log_spatial_summary"].shape, (1, SUMMARY_DIM))
        torch.testing.assert_close(initial["log_spatial_patch"], first_patch)

    def test_rolling_map_preserves_world_locations_and_clears_exposed_cells(self):
        model = SpatialMapMemory(24, config())
        state = model.initial(1, "cpu")
        mid = model.size//2
        state["grid"][0, 0, mid, mid] = 7.
        state["grid"][0, 0, mid, 0] = 99.
        state["pose"][0, 0] = 12.
        cells = model.recenter(state)
        self.assertEqual(int(cells[0, 0]), mid)
        zero = torch.round(-state["origin"][0]/model.resolution).long()
        self.assertEqual(float(state["grid"][0, 0, zero[1], zero[0]]), 7.)
        self.assertFalse(bool((state["grid"][0, 0] == 99).any()))
        self.assertEqual(float(state["grid"][0, :, :, -8:].abs().sum()), 0.)
        state["pose"][0, 0] = -1000.
        model.recenter(state)
        self.assertEqual(float(state["grid"].abs().sum()), 0.)
        self.assertEqual(float(state["directions"].abs().sum()), 0.)

    def test_episode_ledger_recognizes_return_after_local_window_was_discarded(self):
        # Exercise ledger mechanics with exact mocked odometry, independently of
        # the new visual-pair/motion classification guard tested below.
        model = SpatialMapMemory(24, config(vnext_map_resolution=1., vnext_spatial_fix_version=0,
            vnext_pose_min_calibration=4, vnext_pose_min_moving=2, vnext_pose_cache_size=64))
        target = torch.zeros(8, 3); target[[0, 1, 4, 5], 0] = .2
        train = torch.arange(8) < 4
        model.pose.record_predictions(target, target, train, ~train, torch.arange(8))
        f, image, heat, control = inputs()
        heat[:] = 1.
        _, state, _ = model.online(f, image, heat, is_first=torch.ones(1).bool())
        heat.zero_()
        estimate = torch.zeros(1, 3)
        def odometry(*_):
            return dict(delta=estimate.clone(), sigma=torch.zeros(1, 3))
        with mock.patch.object(model.pose, "forward", side_effect=odometry):
            estimate[:, 0] = 1.
            control[:, 3] = 1.
            for _ in range(40):
                _, state, info = model.online(f, image, heat, state, previous_control=control)
            self.assertEqual(int(state["ledger_used"][0]), 41)
            # The true origin viewpoint has left the local rolling window.
            local_origin_cell = centered_cell(-state["origin"][0], model.resolution)
            self.assertTrue(bool(((local_origin_cell < 0) | (local_origin_cell >= model.size)).any()))
            estimate[:, 0] = -1.
            for _ in range(40):
                _, state, info = model.online(f, image, heat, state, previous_control=control)
            self.assertEqual(float(info["log_spatial_revisit"][0]), 1.)
            self.assertEqual(int(state["ledger_used"][0]), 41)
            origin = (state["ledger_cells"][0, :41] == 0).all(-1).nonzero().item()
            self.assertEqual(float(state["ledger_values"][0, origin, 0]), 2.)
            self.assertGreater(float(state["ledger_values"][0, origin, 2]), .9)
            # Same place, a genuinely new heading is a new view, not a new cell.
            estimate[:] = torch.tensor([0., 0., math.pi/2])
            _, state, new_view = model.online(f, image, heat, state, previous_control=control)
            self.assertEqual(float(new_view["log_spatial_novelty"][0]), 1.)
            self.assertEqual(float(new_view["log_spatial_visited_cells"][0]), 41.)
            estimate[:, 2] = -math.pi/2
            _, state, old_view = model.online(f, image, heat, state, previous_control=control)
            self.assertEqual(float(old_view["log_spatial_revisit"][0]), 1.)
            self.assertEqual(float(old_view["log_spatial_directions"][0].sum()), 2.)
        snapshot = model.visualization_state(state)
        self.assertEqual(snapshot["ledger"]["cells"].shape, (41, 2))
        self.assertEqual(snapshot["ledger"]["values"].shape, (41, 6))
        self.assertEqual(snapshot["ledger"]["directions"].shape, (41, 12))

    def test_ledger_capacity_covers_episode_and_reset_never_leaks_prior_cells(self):
        model = SpatialMapMemory(24, config(episode_max_steps=3000))
        self.assertEqual(model.ledger_capacity, 3001)
        f, image, heat, control = inputs(2)
        _, state, _ = model.online(f, image, heat)
        state["ledger_used"][:] = 3
        state["ledger_cells"][:, 1:3] = 123
        _, reset, _ = model.online(f, image, heat, state,
                                  is_first=torch.tensor([True, False]), previous_control=control)
        self.assertEqual(int(reset["ledger_used"][0]), 1)
        self.assertEqual(int(reset["ledger_used"][1]), 3)
        self.assertFalse(bool((reset["ledger_cells"][0] == 123).any()))
        self.assertTrue(all(torch.is_tensor(value) for value in reset.values()))
        with self.assertRaisesRegex(ValueError, "8192"):
            SpatialMapMemory(24, config(episode_max_steps=8192))
        full = model.initial(1, "cpu")
        full["ledger_used"][0] = model.ledger_capacity
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            model.ledger_entry(full, 0, torch.tensor([9999, 9999]))

    def test_episode_reset_is_per_environment_and_snapshot_is_labelled(self):
        model = SpatialMapMemory(24, config())
        f, image, h, control = inputs(2)
        _, state, _ = model.online(f, image, h)
        control[:, 8] = 30.
        _, state, _ = model.online(f, image, h, state, previous_control=control)
        old = state["pose"].clone()
        _, reset, info = model.online(f, image, h, state, is_first=torch.tensor([True, False]))
        self.assertEqual(float(reset["pose"][0].abs().sum()), 0.)
        self.assertAlmostEqual(float(reset["pose"][1, 2]), math.pi/6, places=5)
        torch.testing.assert_close(state["pose"], old)
        snapshot = model.visualization_state(reset)
        self.assertTrue(snapshot["predicted"])
        self.assertEqual(snapshot["grid"].shape, (model.size, model.size, 6))
        self.assertEqual(snapshot["target_type"], "cue-observation viewpoint")
        self.assertEqual(float(info["log_spatial_version"][0]), SPATIAL_VERSION)

    def test_replay_pose_and_map_encoder_receive_finite_gradients(self):
        model = SpatialMapMemory(24, config())
        b, t = 2, 4
        shape = (b, t)
        image = torch.randint(0, 255, (*shape, 64, 64, 3), dtype=torch.uint8)
        patch = torch.randint(0, 255, (*shape, 21, 21, 6), dtype=torch.uint8)
        data = dict(image=image, vnext_spatial_patch=patch,
            vnext_spatial_pose=torch.zeros(*shape, 4), vnext_spatial_valid=torch.ones(shape),
            vnext_pose_prev_image=torch.zeros_like(image), vnext_pose_control=torch.zeros(*shape, 9),
            vnext_pose_prev_valid=torch.ones(shape), search_pose_delta=torch.full((*shape, 3), .05),
            search_pose_valid=torch.ones(shape), search_label_version=torch.full(shape, 3))
        for name in ("visible", "new", "progress"):
            data["search_"+name] = torch.ones(shape)
            data["search_"+name+"_valid"] = torch.ones(shape)
        feat = torch.randn(*shape, 24, requires_grad=True)
        context, details = model.batch(feat, data)
        self.assertEqual(context.shape, (b, t, 16))
        loss, metrics, _ = model.loss(feat, data, context, details, torch.ones(shape).bool())
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNone(feat.grad)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.pose.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters()))
        self.assertEqual(float(metrics["B/pose_samples"]), b*t)

    def test_unknown_and_nonfinite_labels_do_not_create_supervision(self):
        model = PoseNet(hidden=16)
        image = torch.zeros(2, 64, 64, 3)
        target = torch.tensor([[float("nan"), 0, 0], [float("inf"), 0, 0]])
        loss, metrics, _ = model.supervised_loss(image, image, torch.zeros(2, 9), target, torch.ones(2).bool())
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(metrics["B/pose_samples"]), 0.)
        self.assertEqual(int(model.supervised_samples), 0)

    def test_online_does_not_accept_supervision_or_use_feature_as_position(self):
        model = SpatialMapMemory(24, config())
        f, image, heat, _ = inputs()
        first, s1, _ = model.online(f, image, heat)
        second, s2, _ = model.online(f+1.e6, image, heat)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(s1["pose"], s2["pose"])
        with self.assertRaises(TypeError):
            model.online(f, image, heat, search_pose_delta=torch.ones(1, 3))

    def qualified_memory(self, **updates):
        values=dict(vnext_pose_min_calibration=4,vnext_pose_min_moving=2,vnext_pose_cache_size=64)
        values.update(updates)
        model=SpatialMapMemory(24,config(**values))
        target=torch.zeros(8,3);target[[0,1,4,5],0]=.2
        train=torch.arange(8)<4
        model.pose.record_predictions(target,target,train,~train,torch.arange(8))
        self.assertTrue(model.pose.ready)
        return model

    def test_same_frame_command_shortcut_cannot_accumulate_fake_visited_cells(self):
        model=self.qualified_memory()
        f,image,heat,control=inputs();control[:,1]=1.
        _,state,_=model.online(f,image,heat,is_first=torch.ones(1).bool())
        def fake(*_):
            return dict(delta=torch.tensor([[0.,.12,0.]]),sigma=torch.full((1,3),.01),
                        motion_probability=torch.tensor([.99]))
        with mock.patch.object(model.pose,'forward',side_effect=fake):
            for _ in range(40):
                _,state,info=model.online(f,image,heat,state,previous_control=control)
        self.assertEqual(float(state['pose'][0,:2].abs().sum()),0.)
        self.assertEqual(int(state['ledger_used'][0]),1)
        self.assertEqual(float(info['log_spatial_summary'][0,7]),0.)
        self.assertEqual(float(info['log_spatial_summary'][0,11]),0.)
        self.assertEqual(float(info['log_spatial_motion_accepted'][0]),0.)
        self.assertGreater(float(info['log_spatial_rejected_motion'][0]),.1)
        self.assertGreater(float(state['pose'][0,3]),0.)

    def test_uncertain_or_static_classifier_never_certifies_translation(self):
        model=self.qualified_memory()
        before=torch.zeros(3,64,64,3);after=torch.ones_like(before)
        prediction=dict(delta=torch.tensor([[0.,.15,0.],[0.,.15,0.],[0.,.02,0.]]),
            sigma=torch.tensor([[.01,.01,.01],[.2,.2,.01],[.01,.01,.01]]),
            motion_probability=torch.tensor([.1,.99,.99]))
        accepted,_,_=model.pose.translation_support(prediction,before,after)
        self.assertFalse(bool(accepted.any()))
        model.pose.cal_identity[:8]=torch.arange(8)
        model.pose.cal_baseline[:8]=0
        model.pose.cal_error[:8]=.18
        prediction['motion_probability'][:]=.99
        prediction['sigma'][:]=.01
        accepted,_,bound=model.pose.translation_support(prediction,before,after)
        self.assertEqual(accepted.tolist(),[True,True,False])
        self.assertAlmostEqual(float(model.pose.static_error_bound()),.18,places=5)
        torch.testing.assert_close(bound,torch.full_like(bound,.03))
        # Gated calibration is only a qualification statistic; it cannot tune
        # this same frame pair's online gate up/down on every resampling.
        model.pose.cal_error.zero_()
        again,_,_=model.pose.translation_support(prediction,before,after)
        torch.testing.assert_close(accepted,again)

    def test_static_tail_rejects_rare_large_false_steps_hidden_by_small_mean(self):
        model=PoseNet(hidden=16,min_samples=4,min_calibration=4,min_moving=2,cache_size=128)
        target=torch.zeros(40,3);target[[0,1,2,3,20,21,22,23],0]=.2
        estimate=target.clone();estimate[[24,25],0]=.1
        train=torch.arange(40)<20
        model.record_predictions(estimate,target,train,~train,torch.arange(40))
        self.assertLess(float(model.calibration_static_error),.03)
        self.assertGreater(float(model.static_error_bound()),.05)
        self.assertEqual(float(model.calibration_moving_error),0.)
        self.assertFalse(model.ready)

    def test_supported_translation_can_move_map_while_camera_turns(self):
        model=self.qualified_memory()
        f,image,heat,control=inputs()
        _,state,_=model.online(f,image,heat,is_first=torch.ones(1).bool())
        control[:,1]=1.;control[:,8]=10.
        def measured(*_):
            return dict(delta=torch.tensor([[0.,.2,math.radians(10)]]),sigma=torch.full((1,3),.01),
                        motion_probability=torch.tensor([.99]))
        with mock.patch.object(model.pose,'forward',side_effect=measured):
            _,state,info=model.online(f,image+1.,heat,state,previous_control=control)
        self.assertAlmostEqual(float(state['pose'][0,1]),.2,places=5)
        self.assertEqual(float(info['log_spatial_motion_accepted'][0]),1.)
        self.assertGreater(float(info['log_spatial_summary'][0,11]),.9)
        self.assertAlmostEqual(float(state['pose'][0,2]),math.radians(10),places=5)

    def test_always_static_classifier_cannot_qualify_from_unused_good_raw_means(self):
        model=PoseNet(hidden=16,min_samples=4,min_calibration=4,min_moving=2,cache_size=64)
        target=torch.zeros(8,3);target[[0,1,4,5],0]=.2
        train=torch.arange(8)<4
        prediction=dict(delta=target.clone(),log_std=torch.full_like(target,-5.),
                        sigma=torch.full_like(target,.007),motion_logit=torch.full((8,),-10.),
                        motion_probability=torch.full((8,),.0001))
        before=torch.zeros(8,64,64,3);after=torch.ones_like(before)
        with mock.patch.object(model,'forward',return_value=prediction):
            _,metrics,_=model.supervised_loss(before,after,torch.zeros(8,9),target,train,
                calibration_mask=~train,frame_ids=torch.arange(8))
        self.assertEqual(float(metrics['B/pose_position_error']),0.)
        self.assertAlmostEqual(float(model.calibration_moving_error),.2,places=5)
        self.assertFalse(model.ready)

    def test_motion_head_and_raw_translation_train_even_before_online_gate_opens(self):
        torch.manual_seed(314)
        model=PoseNet(hidden=16)
        image=torch.zeros(4,64,64,3);after=image.clone();after[2:]=1.
        target=torch.zeros(4,3);target[2:,1]=.2
        control=torch.zeros(4,9);control[:,1]=1.
        loss,metrics,prediction=model.supervised_loss(image,after,control,target,torch.ones(4).bool(),record=False)
        self.assertFalse(bool(model.translation_support(prediction,image,after)[0].any()))
        loss.backward()
        grad=model.fusion[-1].weight.grad
        self.assertGreater(float(grad[:2].abs().sum()),0.)
        self.assertGreater(float(grad[2].abs().sum()),0.)
        self.assertEqual(float(metrics['B/pose_static_samples']),2.)
        self.assertEqual(float(metrics['B/pose_motion_train_samples']),2.)

    def test_static_regression_and_classification_improve_with_real_optimizer_steps(self):
        torch.manual_seed(77)
        model=PoseNet(hidden=16)
        with torch.no_grad():
            model.fusion[-1].bias[1]=math.atanh(.12/model.max_translation)
            model.fusion[-1].bias[2]=2.
        image=torch.zeros(4,64,64,3);control=torch.zeros(4,9);control[:,1]=1.
        target=torch.zeros(4,3);valid=torch.ones(4).bool()
        optimizer=torch.optim.Adam(model.parameters(),lr=.005)
        before=model(image,image,control)
        for _ in range(60):
            loss,_,_=model.supervised_loss(image,image,control,target,valid,record=False)
            optimizer.zero_grad();loss.backward();optimizer.step()
        after=model(image,image,control)
        self.assertLess(float(after['delta'][:,:2].norm(dim=-1).mean()),.03)
        self.assertLess(float(after['motion_probability'].mean()),.2)
        self.assertLess(float(after['delta'][:,:2].norm(dim=-1).mean()),float(before['delta'][:,:2].norm(dim=-1).mean()))

    def test_pose_residual_separates_own_camera_turn_from_translation(self):
        model = PoseNet(hidden=16, loss_version=2)
        x = torch.linspace(0, 1, 64).view(1,1,64,1)
        previous = x.expand(1,64,64,3)
        control = torch.zeros(1,9)
        control[:,8] = 10.
        pixel_shift = math.tan(math.radians(10)) / math.tan(math.radians(35)) / (256./160.) * 32
        current = (previous + pixel_shift / 63).clamp_max(1)
        residual = model._camera_residual(previous,current,control)
        self.assertLess(float(residual.abs().mean()),1e-6)
        self.assertGreater(float((current-previous).abs().mean()),.05)
        with torch.no_grad():
            prediction = model(previous,current,control)
        torch.testing.assert_close(prediction["visual_difference"],(current-previous).abs().mean((1,2,3)))
        self.assertNotIn("camera_grid",model.state_dict())

    def test_gated_pose_loss_trains_blocked_steps_and_inertial_motion_only_from_training(self):
        model = PoseNet(hidden=16, loss_version=2)
        image = torch.zeros(10,64,64,3)
        after = torch.ones_like(image)
        control = torch.zeros(10,9)
        control[[4,5,6],1] = 1.
        target = torch.zeros(10,3)
        target[5:8,1] = .2
        target[9,1] = .2
        delta = target.clone()
        delta[4,1] = .13  # rare false motion on a blocked walk
        delta[8,1] = 1.0  # worst static case belongs to calibration
        delta.requires_grad_()
        logits = torch.tensor([-2.,-2.,-2.,-2.,3.,3.,3.,0.,3.,3.], requires_grad=True)
        prediction = dict(delta=delta, log_std=torch.full_like(delta,-4.),
                          sigma=torch.full_like(delta,.02), motion_logit=logits,
                          motion_probability=logits.sigmoid())
        train = torch.arange(10)<8
        with mock.patch.object(model,'forward',return_value=prediction):
            loss,metrics,_ = model.supervised_loss(image,after,control,target,train,
                calibration_mask=~train,frame_ids=torch.arange(10))
        self.assertGreater(float(metrics['B/pose_gated_static_tail_loss']),.06)
        self.assertLess(float(metrics['B/pose_gated_static_tail_loss']),.09)
        loss.backward()
        self.assertGreater(float(logits.grad[4]),0.)
        self.assertGreater(float(delta.grad[4,1]),0.)
        self.assertLess(float(logits.grad[7]),0.)  # true movement without a new walk command
        torch.testing.assert_close(logits.grad[8:], torch.zeros(2))
        torch.testing.assert_close(delta.grad[8:], torch.zeros(2,3))
        self.assertFalse(model.ready)

    def test_pose_revision_three_learns_through_a_closed_gate_without_holdout_gradients(self):
        model = PoseNet(hidden=16, loss_version=3)
        target = torch.tensor([[0., .2, 0.], [0., .2, 0.], [0., .2, 0.], [0., 0., 0.]])
        delta = torch.tensor([[0., .0, 0.], [0., .17, 0.], [0., .17, 0.], [0., .14, 0.]], requires_grad=True)
        logit = torch.tensor([-8., -8., -8., 2.], requires_grad=True)
        sigma = torch.full((4, 3), .2, requires_grad=True)
        motion = dict(delta=delta, motion_logit=logit, sigma=sigma, visual_difference=torch.ones(4))
        moving = torch.tensor([True, True, False, False])  # row 2 is held out
        static = torch.tensor([False, False, False, True])
        direct, margin, stopped = model._usable_motion_loss(motion, delta-target, target, moving, static)
        (direct + margin + stopped).backward()
        self.assertLess(float(delta.grad[0, 1]), 0.)
        self.assertEqual(float(logit.grad[0]), 0.)
        self.assertLess(float(logit.grad[1]), 0.)
        self.assertGreater(float(sigma.grad[1, :2].sum()), 0.)
        torch.testing.assert_close(delta.grad[2:], torch.zeros(2, 3))
        self.assertEqual(float(logit.grad[2]), 0.)
        self.assertGreater(float(logit.grad[3]), 0.)
        self.assertFalse(model.qualification_diagnostics()["ready"])

    def test_revision_zero_keeps_checkpoint_shape_and_original_motion_semantics(self):
        old=PoseNet(hidden=16,fix_version=0);new=PoseNet(hidden=16,fix_version=1)
        self.assertEqual({k:tuple(v.shape) for k,v in old.state_dict().items()},
                         {k:tuple(v.shape) for k,v in new.state_dict().items()})
        image=torch.zeros(1,64,64,3);control=torch.zeros(1,9)
        self.assertNotIn('motion_probability',old(image,image,control))
        self.assertIn('motion_probability',new(image,image,control))
        self.assertAlmostEqual(float(new(image,image,control)['delta'][0,2]),0.,places=7)
        model=self.qualified_memory(vnext_spatial_fix_version=0)
        f,image,heat,control=inputs()
        _,state,_=model.online(f,image,heat,is_first=torch.ones(1).bool())
        with mock.patch.object(model.pose,'forward',return_value=dict(delta=torch.tensor([[0.,.12,0.]]),sigma=torch.full((1,3),.01))):
            _,state,info=model.online(f,image,heat,state,previous_control=control)
        self.assertAlmostEqual(float(state['pose'][0,1]),.12,places=5)
        self.assertNotIn('log_spatial_motion_probability',info)

    @staticmethod
    def ground_frame():
        y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        tile = ((x//8+y//8)%2).float()*.6+.1
        return torch.stack((tile, tile*.8, tile*.4), -1)[None]

    @staticmethod
    def pose_prediction(probability=.01, movement=0.):
        return dict(delta=torch.tensor([[0., movement, 0.]]), sigma=torch.full((1,3), .01),
                    motion_probability=torch.tensor([probability]))

    def test_revision_two_unknown_motion_breaks_origin_trajectory_until_reset(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs();image=self.ground_frame();control[:,1]=1
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            _,state,start=model.online(f,image,heat,is_first=torch.ones(1).bool())
        ledger=state['ledger_values'].clone();grid=state['grid'].clone()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.5,.12)):
            context,state,info=model.online(f,image+0.1,heat,state,previous_control=control)
        self.assertFalse(bool(state['tracking_ready'][0]))
        self.assertEqual(float(info['log_spatial_trajectory_valid'][0]),0.)
        self.assertEqual(float(info['log_spatial_motion_unknown'][0]),1.)
        self.assertEqual(float(info['log_spatial_map_write'][0]),0.)
        self.assertEqual(float(info['log_spatial_novelty'][0]),0.)
        self.assertEqual(float(info['log_spatial_revisit'][0]),0.)
        self.assertEqual(int(state['pose_gap_count'][0]),1)
        self.assertEqual(int(state['first_gap_step'][0]),1)
        self.assertFalse(bool(context.any()))
        torch.testing.assert_close(state['ledger_values'],ledger)
        torch.testing.assert_close(state['grid'],grid)
        # A later individually good pair cannot supply the missed displacement.
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.99,.2)):
            context,state,again=model.online(f,image,heat,state,previous_control=control)
        self.assertEqual(float(state['pose'][0,:2].abs().sum()),0.)
        self.assertFalse(bool(state['tracking_ready'][0]))
        self.assertEqual(float(again['log_spatial_map_write'][0]),0.)
        self.assertFalse(bool(context.any()))
        snapshot=model.visualization_state(state)
        self.assertFalse(snapshot['trajectory_valid'])
        self.assertIn('gap',snapshot['localization'])

    def test_revision_two_confident_static_retains_continuous_location(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs();image=self.ground_frame()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.01,.01)):
            _,state,_=model.online(f,image,heat,is_first=torch.ones(1).bool())
            for _ in range(4):
                _,state,info=model.online(f,image,heat,state,previous_control=control)
        self.assertTrue(bool(state['tracking_ready'][0]))
        self.assertEqual(int(state['pose_gap_count'][0]),0)
        self.assertEqual(int(state['ledger_used'][0]),1)
        self.assertEqual(float(info['log_spatial_motion_unknown'][0]),0.)
        self.assertEqual(float(info['log_spatial_summary'][0,0]),1.)
        self.assertEqual(float(state['pose'][0,:2].abs().sum()),0.)

    def test_revision_two_no_unknown_origin_or_camera_prefix_is_invented(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs();image=self.ground_frame()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            context,state,info=model.online(f,image,heat,is_first=torch.zeros(1).bool())
            self.assertFalse(bool(context.any()))
            self.assertFalse(bool(state['pitch_known'][0]))
            self.assertEqual(int(state['ledger_used'][0]),0)
            _,state,info=model.online(f,image,heat,state,previous_control=control,camera_pitch=torch.tensor([20.]))
        self.assertTrue(bool(state['pitch_known'][0]))
        self.assertFalse(bool(state['tracking_ready'][0]))
        self.assertEqual(float(info['log_spatial_map_write'][0]),0.)

    def test_revision_two_sky_cannot_consume_ground_direction_or_write_target_cue(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs();image=self.ground_frame()
        heat[:,40:48,20:28]=.95
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            _,state,sky=model.online(f,image,heat,is_first=torch.ones(1).bool(),camera_pitch=torch.tensor([-70.]))
            self.assertFalse(bool(state['directions'].any()))
            self.assertFalse(bool(state['grid'][:,1:3].any()))
            self.assertEqual(float(sky['log_spatial_ground_view'][0]),0.)
            self.assertEqual(float(sky['log_spatial_novelty'][0]),0.)
            _,state,ground=model.online(f,image,heat,state,previous_control=control,camera_pitch=torch.tensor([10.]))
        self.assertEqual(float(ground['log_spatial_ground_view'][0]),1.)
        self.assertEqual(float(ground['log_spatial_novelty'][0]),1.)
        self.assertEqual(float(ground['log_spatial_cue_valid'][0]),1.)
        self.assertGreater(float(state['grid'][:,2].max()),.9)
        self.assertEqual(float(ground['log_spatial_directions'].sum()),1.)

    def test_revision_two_weak_uniform_or_upper_heatmap_is_not_target_evidence(self):
        image=self.ground_frame();pitch=torch.zeros(1);known=torch.ones(1).bool()
        for kind in ['weak','uniform','upper','nonfinite']:
            heat=torch.zeros(1,64,64)
            if kind=='weak':heat[:,40:48,20:28]=.33
            if kind=='uniform':heat[:]=.99
            if kind=='upper':heat[:,:16]=.99
            if kind=='nonfinite':heat[:,40:48]=float('nan')
            ground,cue,valid=SpatialMapMemory.ground_evidence(image,heat,pitch,known)
            self.assertFalse(bool(valid.any()),kind)
            self.assertFalse(bool(cue.any()),kind)
        heat=torch.zeros(1,64,64);heat[:,40:48,20:28]=.95
        ground,cue,valid=SpatialMapMemory.ground_evidence(image,heat,pitch,known)
        self.assertTrue(bool(ground.all()) and bool(valid.all()))
        sky=torch.zeros(1,64,64,3);sky[...,2]=1
        self.assertFalse(bool(SpatialMapMemory.ground_evidence(sky,heat,pitch,known)[0].any()))

    def test_revision_two_pitch_follows_own_controls_and_resets_per_row(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs(2);image=self.ground_frame().expand(2,-1,-1,-1)
        pred=dict(delta=torch.zeros(2,3),sigma=torch.full((2,3),.01),motion_probability=torch.full((2,),.01))
        with mock.patch.object(model.pose,'forward',return_value=pred):
            _,state,_=model.online(f,image,heat,is_first=torch.ones(2).bool())
            control[:,7]=torch.tensor([-10.,20.])
            _,state,info=model.online(f,image,heat,state,previous_control=control)
            self.assertEqual(state['camera_pitch'].tolist(),[-10.,20.])
            state['pose_gap_count'][0]=4;state['tracking_ready'][0]=False
            control.zero_()
            _,state,info=model.online(f,image,heat,state,is_first=torch.tensor([True,False]),previous_control=control)
        self.assertEqual(state['camera_pitch'].tolist(),[0.,20.])
        self.assertEqual(state['pose_gap_count'].tolist(),[0,0])
        self.assertTrue(bool(state['tracking_ready'].all()))

    def test_revision_two_revoked_qualification_does_not_reanchor_mid_episode(self):
        model=self.qualified_memory(vnext_spatial_fix_version=2)
        f,_,heat,control=inputs();image=self.ground_frame()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            _,state,_=model.online(f,image,heat,is_first=torch.ones(1).bool())
            with mock.patch.object(PoseNet,'ready',new_callable=mock.PropertyMock,return_value=False):
                _,state,info=model.online(f,image,heat,state,previous_control=control)
            self.assertEqual(float(info['log_spatial_trajectory_valid'][0]),0.)
            self.assertEqual(int(state['pose_gap_count'][0]),1)
            _,state,info=model.online(f,image,heat,state,previous_control=control)
        self.assertFalse(bool(state['tracking_ready'][0]))
        self.assertEqual(float(info['log_spatial_map_write'][0]),0.)

    def test_revision_two_invalid_trajectory_context_does_not_block_pose_training(self):
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=2))
        shape=(1,4);image=torch.randint(255,(*shape,64,64,3),dtype=torch.uint8)
        data=dict(image=image,vnext_spatial_patch=torch.zeros(*shape,21,21,6,dtype=torch.uint8),
            vnext_spatial_pose=torch.zeros(*shape,4),vnext_spatial_valid=torch.ones(shape),
            vnext_spatial_trajectory_valid=torch.zeros(shape),
            vnext_pose_prev_image=torch.zeros_like(image),vnext_pose_control=torch.zeros(*shape,9),
            vnext_pose_prev_valid=torch.ones(shape),search_pose_delta=torch.full((*shape,3),.05),
            search_pose_valid=torch.ones(shape),search_label_version=torch.full(shape,3))
        for name in ('visible','new','progress'):
            data['search_'+name]=torch.ones(shape);data['search_'+name+'_valid']=torch.ones(shape)
        features=torch.randn(*shape,24,requires_grad=True)
        context,details=model.batch(features,data)
        self.assertFalse(bool(context.any()))
        self.assertTrue(bool(details['snapshot_valid'].all()))
        self.assertFalse(bool(details['valid'].any()))
        loss,metrics,_=model.loss(features,data,context,details,torch.ones(shape).bool())
        loss.backward()
        self.assertEqual(float(metrics['B/map_loss']),0.)
        self.assertEqual(float(metrics['B/pose_samples']),4.)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.pose.parameters()))
        self.assertIsNone(features.grad)

    def test_revision_one_keeps_historical_map_and_cue_semantics(self):
        model=self.qualified_memory(vnext_spatial_fix_version=1)
        f,image,heat,control=inputs();heat[:]=1.
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            _,state,info=model.online(f,image,heat,is_first=torch.ones(1).bool(),camera_pitch=torch.tensor([-70.]))
        self.assertGreater(float(state['grid'][:,2].max()),.9)
        self.assertGreater(float(state['grid'][:,1].sum()),0.)
        self.assertNotIn('log_spatial_trajectory_valid',info)

    def test_revision_three_unqualified_pose_keeps_episode_visual_history_active(self):
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=3,vnext_slots=4))
        f,_,h,control=inputs();image=self.ground_frame();state=None
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.5,.12)):
            for t in range(160):
                current=image.clone();current[...,t//40%3]*=.2
                context,state,info=model.online(f,current,h,state,is_first=torch.tensor([t==0]),previous_control=control)
                self.assertTrue(torch.isfinite(context).all())
                self.assertTrue(bool(context.any()))
                self.assertEqual(float(info['log_spatial_memory_available'][0]),1.)
                self.assertEqual(float(info['log_spatial_map_write'][0]),0.)
                self.assertEqual(float(info['log_spatial_trajectory_valid'][0]),0.)
        self.assertEqual(int(state['tick'][0]),160)
        self.assertEqual(int(state['ledger_used'][0]),0)
        self.assertGreater(int(state['history_valid'].sum()),1)
        self.assertEqual(float(state['history_meta'][0,0,0]),0.)

    def test_revision_three_gap_recovers_local_segment_without_rejoining_old_coordinates(self):
        model=self.qualified_memory(vnext_spatial_fix_version=3,vnext_slots=4)
        f,_,h,control=inputs();image=self.ground_frame()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            _,state,_=model.online(f,image,h,is_first=torch.ones(1).bool(),previous_control=control)
        state['pose'][0,0]=9.
        cells=model.recenter(state)
        entry=model.ledger_entry(state,0,centered_cell(state['pose'][0,:2],model.resolution))
        state['ledger_values'][0,entry,0]=8.
        history=state['history_frames'].clone()
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.5,.12)):
            context,state,gap=model.online(f,image,h,state,previous_control=control)
        self.assertTrue(bool(context.any()))
        self.assertFalse(bool(state['tracking_ready'][0]))
        self.assertEqual(float(gap['log_spatial_trajectory_valid'][0]),0.)
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction()):
            for i in range(3):
                _,state,info=model.online(f,image,h,state,previous_control=control)
                self.assertEqual(float(info['log_spatial_map_write'][0]),float(i==2))
        self.assertEqual(int(state['segment'][0]),1)
        self.assertEqual(float(state['pose'][0,:2].abs().sum()),0.)
        self.assertTrue(bool(state['tracking_ready'][0]))
        self.assertFalse(bool(state['global_continuity'][0]))
        self.assertEqual(int(state['pose_gap_count'][0]),1)
        self.assertEqual(float(info['log_spatial_trajectory_valid'][0]),0.)
        self.assertEqual(float(info['log_spatial_local_valid'][0]),1.)
        self.assertGreaterEqual(int(state['ledger_used'][0]),3)
        self.assertIn(0,state['ledger_segment'][0,:state['ledger_used'][0]].tolist())
        self.assertIn(1,state['ledger_segment'][0,:state['ledger_used'][0]].tolist())
        # The old remote 9m cell must not reappear in the new segment's map.
        self.assertEqual(float(state['grid'][:,0].max()),1.)
        torch.testing.assert_close(state['history_frames'],history)
        snapshot=model.visualization_state(state)
        self.assertFalse(snapshot['trajectory_valid'])
        self.assertTrue(snapshot['local_tracking_ready'])
        self.assertEqual(snapshot['active_segment'],1)

    def test_revision_three_history_changes_context_for_identical_current_view(self):
        torch.manual_seed(81)
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=3,vnext_slots=4))
        f,_,h,control=inputs();common=self.ground_frame()
        states=[]
        with mock.patch.object(model.pose,'forward',return_value=self.pose_prediction(.5,.12)):
            for channel in [0,2]:
                old=torch.zeros_like(common);old[...,channel]=1
                _,state,_=model.online(f,old,h,is_first=torch.ones(1).bool(),previous_control=control)
                context,state,_=model.online(f,common,h,state,previous_control=control)
                states.append(context)
        self.assertGreater(float((states[0]-states[1]).abs().max()),1.e-4)

    @staticmethod
    def history_replay(info):
        data={}
        for tail in ['patch','pose','history_frames','history_meta','history_valid','history_tick','local_valid','trajectory_valid']:
            data['vnext_spatial_'+tail]=info['log_spatial_'+tail][:,None].clone()
        data['vnext_spatial_valid']=torch.ones(len(info['log_spatial_pose']),1)
        data['vnext_spatial_history_snapshot_valid']=torch.ones_like(data['vnext_spatial_valid'])
        return data

    def test_revision_three_online_replay_agree_and_memory_learns_without_pose(self):
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=3,vnext_slots=4))
        f,image,h,control=inputs();image=self.ground_frame();f.normal_()
        online,state,info=model.online(f,image,h,is_first=torch.ones(1).bool(),previous_control=control)
        data=self.history_replay(info);data['image']=image[:,None]
        data['search_label_version']=torch.full((1,1),3.)
        for name in ['visible','new','progress']:
            data['search_'+name]=torch.ones(1,1);data['search_'+name+'_valid']=torch.ones(1,1)
        features=f[:,None].clone().requires_grad_()
        context,details=model.batch(features,data)
        torch.testing.assert_close(context[:,0],online,atol=1.e-6,rtol=1.e-5)
        self.assertTrue(bool(details['valid'].all()))
        self.assertFalse(bool(details['map_valid'].any()))
        before=model.history_encoder.frames[0].weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        loss,metrics,_=model.loss(features,data,context,details,torch.ones(1,1).bool())
        optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertEqual(float(metrics['B/map_loss']),0.)
        self.assertGreater(float(metrics['B/history_loss']),0.)
        self.assertEqual(float(metrics['B/history_valid']),1.)
        self.assertFalse(torch.equal(before,model.history_encoder.frames[0].weight))
        self.assertIsNone(features.grad)

    def test_revision_three_missing_history_never_fabricates_available_memory(self):
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=3,vnext_slots=4))
        features=torch.randn(1,2,24)
        context,details=model.batch(features,{'vnext_spatial_valid':torch.ones(1,2)})
        self.assertFalse(bool(context.any()))
        self.assertFalse(bool(details['valid'].any()))

    def test_revision_three_multi_environment_reset_preserves_other_history(self):
        model=SpatialMapMemory(24,config(vnext_spatial_fix_version=3,vnext_slots=4))
        f,image,h,control=inputs(2);image=self.ground_frame().expand(2,-1,-1,-1).clone()
        _,state,_=model.online(f,image,h,is_first=torch.ones(2).bool(),previous_control=control)
        _,state,_=model.online(f,image,h,state,previous_control=control)
        second=state['history_meta'][1].clone()
        image[0].zero_()
        _,state,info=model.online(f,image,h,state,is_first=torch.tensor([True,False]),previous_control=control)
        self.assertEqual(int(state['history_valid'][0].sum()),1)
        self.assertEqual(float(state['history_meta'][0,:,2].sum()),1.)
        self.assertGreater(float(state['history_meta'][1,:,2].sum()),float(second[:,2].sum()))
        self.assertEqual(state['tick'].tolist(),[1,3])


if __name__ == "__main__":
    unittest.main()
