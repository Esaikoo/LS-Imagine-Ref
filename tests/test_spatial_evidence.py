"""Spatial loss-target boundaries and read-only collected-map diagnostics."""
import copy
from pathlib import Path
import sys
import unittest

import gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.evidence import SpatialEvidenceWrapper, SPATIAL_LABEL_SHAPES
from apm.diagnostics import spatial_episode_diagnostics


def observation(x=0., z=0., yaw=0., y=64.):
    return dict(location_stats=dict(pos=np.array([x, y, z]), yaw=np.array([yaw])))


class FakeWorld(gym.Env):
    action_space = gym.spaces.Dict({})
    observation_space = gym.spaces.Dict({})

    def __init__(self):
        self.observation = observation()
        self.error = False

    def reset(self, **kwargs):
        return copy.deepcopy(self.observation)

    def step(self, action):
        return copy.deepcopy(self.observation), 0., False, ({"error": "transport"} if self.error else {})


class SpatialEvidenceTests(unittest.TestCase):
    def wrapper(self, supervision=True, audit=True):
        env = FakeWorld()
        return env, SpatialEvidenceWrapper(env, supervision=supervision, audit=audit)

    def test_body_frame_and_yaw_wrap(self):
        env, wrapper = self.wrapper()
        env.observation = observation(100, 200, 179)
        initial = wrapper.reset()
        self.assertEqual(initial["search_pose_valid"], 0)
        angle = np.deg2rad(179)
        local = np.array([.1, .2])
        world = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ local
        env.observation = observation(100+world[0], 200+world[1], -179)
        obs, _, _, _ = wrapper.step({})
        self.assertEqual(obs["search_pose_valid"], 1)
        np.testing.assert_allclose(obs["search_pose_delta"], [.1, .2, np.deg2rad(2)], atol=1e-6)

    def test_cell_boundaries_use_floor_plus_half(self):
        for x, expected_new in ((.749, 0), (.75, 1), (-.75, 0), (-.751, 1)):
            env, wrapper = self.wrapper()
            wrapper.reset()
            env.observation = observation(x=x)
            obs, _, _, _ = wrapper.step({"forward": 1})
            self.assertEqual(obs["search_new_area_valid"], 1)
            self.assertEqual(obs["search_new_area"], expected_new, x)

    def test_same_cell_forward_and_actual_return_are_distinct(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        for x in (.1, .2, .4, .6):
            env.observation = observation(x=x)
            obs, _, _, _ = wrapper.step({"forward": 1})
            self.assertEqual(obs["search_new_area"], 0)
            self.assertEqual(obs["search_area_revisit"], 0)
        env.observation = observation(x=1.)
        obs, _, _, _ = wrapper.step({"forward": 1})
        self.assertEqual(obs["search_new_area"], 1)
        env.observation = observation(x=.4)
        obs, _, _, _ = wrapper.step({"back": 1})
        self.assertEqual(obs["search_area_revisit"], 1)
        env.observation = observation(x=.3)
        obs, _, _, _ = wrapper.step({"back": 1})
        self.assertEqual(obs["search_area_revisit"], 0)

    def test_new_direction_and_turning_back(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        env.observation = observation(yaw=30)
        obs, _, _, _ = wrapper.step({"camera": [0, 30]})
        self.assertEqual(obs["search_new_view"], 1)
        self.assertEqual(obs["search_new_area"], 0)
        env.observation = observation(yaw=0)
        obs, _, _, _ = wrapper.step({"camera": [0, -30]})
        self.assertEqual(obs["search_new_view"], 0)
        self.assertEqual(obs["search_view_revisit"], 1)
        self.assertEqual(obs["search_area_revisit"], 0)

    def test_continuous_camera_motion_inside_one_sector_is_not_a_return(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        for yaw in (0,10,20,30,40):
            env.observation = observation(yaw=yaw)
            obs,_,_,_ = wrapper.step({"camera":[0,10]})
            self.assertEqual(obs["search_view_revisit"],0)
        env.observation = observation(yaw=10)
        obs,_,_,_ = wrapper.step({"camera":[0,-30]})
        self.assertEqual(obs["search_view_revisit"],1)

    def test_eval_targets_unknown_and_audit_separate(self):
        env, wrapper = self.wrapper(supervision=False, audit=True)
        wrapper.reset()
        env.observation = observation(15, -4, 30)
        obs, _, _, _ = wrapper.step({})
        for key in SPATIAL_LABEL_SHAPES:
            if key.startswith("search_"):
                self.assertFalse(np.asarray(obs[key]).any(), key)
        self.assertEqual(obs["log_spatial_gt_valid"], 1)
        np.testing.assert_allclose(obs["log_spatial_gt_pose"], [15, -4, np.deg2rad(30)], atol=1e-6)
        _, private = self.wrapper(supervision=False, audit=False)
        self.assertEqual(private.reset()["log_spatial_gt_valid"], 0)

    def test_error_masks_transition_and_next_pose_pair(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        env.error = True
        env.observation = observation(x=.2)
        obs, _, _, _ = wrapper.step({})
        for key in SPATIAL_LABEL_SHAPES:
            if key.endswith("valid"):
                self.assertEqual(obs[key], 0, key)
        env.error = False
        env.observation = observation(x=.3)
        obs, _, _, _ = wrapper.step({})
        self.assertEqual(obs["search_pose_valid"], 0)
        env.observation = observation(x=.4)
        obs, _, _, _ = wrapper.step({})
        self.assertEqual(obs["search_pose_valid"], 1)

    def test_reset_discards_old_origin_and_visits(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        env.observation = observation(x=1.)
        wrapper.step({})
        env.observation = observation(1000, 1000, 80)
        obs = wrapper.reset()
        self.assertEqual(obs["search_pose_valid"], 0)
        self.assertEqual(obs["search_area_coverage"], 1)
        self.assertEqual(len(wrapper.views), 1)
        env.observation = observation(1000, 1000, 80, y=80)
        obs, _, _, _ = wrapper.step({})
        np.testing.assert_array_equal(obs["search_pose_delta"], [0, 0, 0])

    def test_teleports_and_missing_pose_do_not_become_motion_truth(self):
        env, wrapper = self.wrapper()
        wrapper.reset()
        env.observation = observation(x=100.)
        obs, _, _, _ = wrapper.step({})
        self.assertEqual(obs["search_pose_valid"], 0)
        self.assertEqual(obs["search_new_area_valid"], 0)
        env.observation = dict(location_stats={})
        obs, _, _, _ = wrapper.step({})
        self.assertEqual(obs["search_pose_valid"], 0)
        self.assertEqual(obs["log_spatial_gt_valid"], 0)


def episode(points=3):
    length = points+1
    data = dict(reward=np.zeros(length), is_first=np.zeros(length, bool),
        log_spatial_version=np.r_[0., np.full(points, 2.)],
        log_spatial_pose=np.zeros((length, 4)), log_spatial_summary=np.zeros((length, 12)),
        log_spatial_gt_pose=np.zeros((length, 3)), log_spatial_gt_valid=np.ones(length),
        log_spatial_visited_cells=np.arange(length), log_spatial_known_cells=np.arange(length)*2,
        log_spatial_revisit=np.zeros(length), log_env_error=np.zeros(length))
    data["is_first"][0] = True
    data["log_spatial_summary"][1:, 0] = 1
    data["log_spatial_pose"][1:, 1] = np.arange(points)
    data["log_spatial_gt_pose"][:, 1] = np.arange(length)
    return data


class SpatialDiagnosticsTests(unittest.TestCase):
    def test_outgoing_map_alignment_and_terminal_exclusion(self):
        data = episode()
        data["log_spatial_pose"][0] = 1000  # unknown cache row cannot be used
        data["log_spatial_gt_pose"][-1] = 99999  # no terminal map was collected
        data["log_spatial_revisit"][2] = 1
        result = spatial_episode_diagnostics(data)
        self.assertEqual(result["spatial_audit_available"], 1)
        self.assertAlmostEqual(result["spatial_ate_rmse"], 0.)
        self.assertAlmostEqual(result["spatial_estimated_path_length"], 2.)
        self.assertEqual(result["spatial_visited_cells"], 3)
        self.assertAlmostEqual(result["spatial_view_revisit_rate"], 1./3)

    def test_origin_heading_alignment_preserves_drift(self):
        data = episode()
        data["log_spatial_gt_pose"][:, 0] = 100-np.arange(4)
        data["log_spatial_gt_pose"][:, 1] = 200
        data["log_spatial_gt_pose"][:, 2] = np.pi/2
        data["log_spatial_pose"][1:, 1] = [0, 1.2, 2.4]
        result = spatial_episode_diagnostics(data)
        self.assertAlmostEqual(result["spatial_ate_rmse"], np.sqrt((.2**2+.4**2)/3), places=6)
        self.assertEqual(result["spatial_heading_error_deg"], 0.)

    def test_heading_error_uses_circular_distance(self):
        data = episode()
        data["log_spatial_gt_pose"][:, :2] = 0
        data["log_spatial_gt_pose"][:3, 2] = np.deg2rad([179, -179, 177])
        data["log_spatial_pose"][1:, :2] = 0
        data["log_spatial_pose"][1:, 2] = np.deg2rad([0, 2, -2])
        self.assertAlmostEqual(spatial_episode_diagnostics(data)["spatial_heading_error_deg"], 0., places=5)

    def test_unqualified_zero_map_is_unknown_not_perfect_accuracy(self):
        data = episode()
        data["log_spatial_pose"][:] = 0
        data["log_spatial_summary"][:, 0] = 0
        result = spatial_episode_diagnostics(data)
        self.assertEqual(result["spatial_localization_rate"], 0)
        self.assertEqual(result["spatial_trajectory_available"], 0)
        self.assertEqual(result["spatial_audit_available"], 0)
        self.assertNotIn("spatial_estimated_path_length", result)
        self.assertNotIn("spatial_ate_rmse", result)

    def test_half_episode_and_revoked_tracking_cannot_claim_full_path(self):
        for kind in ("late_start", "revoked", "not_reset"):
            data = episode()
            if kind == "late_start": data["log_spatial_version"][1] = 0
            if kind == "revoked": data["log_spatial_summary"][2:, 0] = 0
            if kind == "not_reset": data["is_first"][0] = False
            result = spatial_episode_diagnostics(data)
            self.assertEqual(result["spatial_trajectory_available"], 0, kind)
            self.assertEqual(result["spatial_audit_available"], 0, kind)
            self.assertNotIn("spatial_estimated_path_length", result)

    def test_unknown_audit_origin_is_not_aligned_to_a_later_point(self):
        data = episode();data["log_spatial_gt_valid"][0] = 0
        result = spatial_episode_diagnostics(data)
        self.assertEqual(result["spatial_trajectory_available"], 1)
        self.assertEqual(result["spatial_audit_available"], 0)
        self.assertEqual(result["spatial_audit_count"], 2)
        self.assertNotIn("spatial_ate_rmse", result)

    def test_diagnostics_are_read_only_and_accept_unknown_initial_cache_row(self):
        data = episode()
        data["log_spatial_pose"] = [None, *data["log_spatial_pose"][1:]]
        data["log_spatial_summary"] = [None, *data["log_spatial_summary"][1:]]
        before = copy.deepcopy(data)
        result = spatial_episode_diagnostics(data)
        self.assertEqual(result["spatial_audit_available"], 1)
        self.assertIsNone(data["log_spatial_pose"][0])
        for key in data:
            if isinstance(data[key], list):
                for left, right in zip(data[key][1:], before[key][1:]): np.testing.assert_array_equal(left, right)
            else: np.testing.assert_array_equal(data[key], before[key])

    def test_no_spatial_trace_produces_no_conditional_metrics(self):
        self.assertEqual(spatial_episode_diagnostics(dict(reward=np.zeros(4))), {})


if __name__ == "__main__":
    unittest.main()
