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
        model = SpatialMapMemory(24, config(vnext_map_resolution=1.,
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


if __name__ == "__main__":
    unittest.main()
