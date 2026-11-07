"""Causal target confirmation and bounded, frame-local policy guidance."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.goal_guidance import build_policy_observation, proposals


class MatchedRegions:
    """Controlled model probabilities for exercising downstream decisions."""

    device = torch.device("cpu")
    fusion_enabled = True
    max_gate = .8
    fusion_epsilon = .05
    fusion_overlap = .5

    def __init__(self):
        self.reliability = torch.tensor([1., 0., 0.])
        self.region = torch.full((1,16,16), .04)

    def observation_images(self, data):
        return data["image"]

    def __call__(self, image, heatmap):
        return dict(leading=(1,), probability=self.region, saliency=self.region,
                    heatmap=heatmap[:,None].float()/255.,
                    region_uncertainty=torch.zeros_like(self.region),
                    visible_probability=torch.tensor([.5]),
                    interaction_probability=torch.tensor([.5]),
                    embeddings=torch.nn.functional.normalize(torch.ones((1,8,16,16)), dim=1))


def obs(heat, first=False):
    return {"image": torch.zeros((1,64,64,3), dtype=torch.uint8),
            "heatmap": torch.full((1,64,64), heat, dtype=torch.uint8),
            "is_first": torch.tensor([first])}


class GoalGuidanceTests(unittest.TestCase):
    def test_rgb_proposal_outside_unet_hotspot(self):
        heat = torch.zeros((1,1,64,64))
        heat[:,:,:12,:12] = 1
        saliency = torch.zeros((1,16,16))
        saliency[:,12,11] = 1
        cells, boxes, sources = proposals(heat, saliency, saliency)
        self.assertEqual(cells[0,3].item(), 12*16+11)
        self.assertEqual(sources.tolist(), [[0,0,0,1,1,1,2,2]])
        self.assertEqual(len(torch.unique(cells)), 8)
        self.assertTrue(bool((boxes>=0).all() & (boxes<=1).all()))

    def test_verified_high_heat_mismatch_is_bounded_and_causal(self):
        model = MatchedRegions()
        raw = obs(255, True)
        fused, evidence, _ = build_policy_observation(raw, model)
        self.assertEqual(evidence["state"].item(), 0)
        self.assertTrue(evidence["peak_rejected"].item())
        self.assertTrue(evidence["evidence_valid"].item())
        self.assertGreater(evidence["conflict"].item(), .5)
        self.assertLessEqual(evidence["gate"].item(), .800001)
        self.assertLess(fused["heatmap"].float().mean().item(), 150.)
        self.assertGreater(fused["heatmap"].float().mean().item(), 0.)
        self.assertTrue(bool((raw["heatmap"] == 255).all()))

    def test_rgb_only_match_can_supply_guidance_and_track(self):
        model = MatchedRegions()
        model.region[:,12,11] = .93
        first, evidence, state = build_policy_observation(obs(0, True), model)
        self.assertGreater(first["heatmap"].max().item(), 0)
        self.assertEqual(evidence["state"].item(), 1)
        second, evidence, state = build_policy_observation(obs(0), model, state,
                                                             torch.zeros(1,9))
        self.assertEqual(evidence["state"].item(), 2)
        torch.testing.assert_close(first["heatmap"], second["heatmap"])
        _, evidence, state = build_policy_observation(obs(0, True), model, state)
        self.assertEqual(evidence["state"].item(), 1, "Reset must drop earlier visual confirmation")
        model.region.fill_(.1)
        _, evidence, _ = build_policy_observation(obs(0), model, state)
        self.assertEqual(evidence["state"].item(), 0, "Loss of visibility drops target tracking")

    def test_unqualified_or_disabled_fusion_preserves_raw(self):
        model = MatchedRegions()
        raw = obs(220, True)
        model.reliability.zero_()
        fused, evidence, _ = build_policy_observation(raw, model)
        self.assertEqual(evidence["gate"].item(), 0.)
        self.assertFalse(evidence["peak_rejected"].item())
        self.assertFalse(evidence["evidence_valid"].item())
        torch.testing.assert_close(fused["heatmap"], raw["heatmap"])
        model.reliability[0] = 1
        model.fusion_enabled = False
        fused, evidence, _ = build_policy_observation(raw, model)
        self.assertTrue(evidence["peak_rejected"].item())
        torch.testing.assert_close(fused["heatmap"], raw["heatmap"])


if __name__ == "__main__":
    unittest.main()
