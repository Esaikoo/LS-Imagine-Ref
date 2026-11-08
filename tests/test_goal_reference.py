"""D model regressions. Generated fixtures only exercise tensor and label contracts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.goal_reference import GoalReference, TASKS, load_reference_bank
from apm.goal_guidance import build_policy_observation, fuse_zoomed_observation


def config(task="harvest_sand", **changes):
    args = dict(task=f"minedojo_{task}", goal_ref_dim=32, goal_ref_use_raw_image=False,
                goal_ref_batch_frames=8)
    args.update(changes)
    return SimpleNamespace(**args)


def bank(task="harvest_sand"):
    pixels = np.zeros((2, 64, 64, 3), np.uint8)
    pixels[0, :, :, 0] = 180
    pixels[1, :, :, 1] = 210
    entries = [dict(task_id=task, split="train", world_seed=17+i) for i in range(2)]
    manifest = dict(version=1, task_id=task, target_class=TASKS[task], entries=entries,
                    pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    return {"manifest": manifest, "pixels": pixels}


def observed():
    rng = np.random.default_rng(3)
    return dict(image=rng.integers(256, size=(2, 4, 64, 64, 3), dtype=np.uint8),
                heatmap=np.full((2, 4, 64, 64), 180, np.uint8),
                is_first=np.array([[True, False, False, False]]*2))


def labeled():
    data = observed()
    data.update(goal_region_gt=np.zeros((2, 4, 16, 16, 1), np.uint8),
                goal_region_valid=np.zeros((2, 4, 16, 16, 1), np.uint8),
                goal_instance_group=np.zeros((2, 4, 16, 16, 1), np.int32),
                goal_visible_gt=np.zeros((2, 4), bool), goal_visible_valid=np.zeros((2, 4), bool),
                log_applied_control=np.zeros((2, 4, 9), np.float32),
                log_actuator_valid=~data["is_first"], vnext_frame_id=np.arange(8).reshape(2, 4))
    data["goal_region_valid"][:, :, 8, 8:10, 0] = 1
    data["goal_region_gt"][0, :, 8, 8, 0] = 1
    data["goal_visible_gt"][0] = True
    data["goal_visible_valid"][0] = True
    data["goal_instance_group"][0, :, 8, 8, 0] = 42
    return data


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_category_and_portable_bank(self):
        for task in TASKS:
            model = GoalReference(config(task), bank(task))
            self.assertEqual(model.bank_manifest["target_class"], TASKS[task])
            clone = GoalReference(config(task), model.export_bank())
            clone.load_state_dict(model.state_dict())
            self.assertEqual(clone.bank_fingerprint, model.bank_fingerprint)
        with self.assertRaises(ValueError):
            GoalReference(config("shear_sheep"), bank("harvest_sand"))

    def test_manifest_verifies_bytes_and_separates_distractors(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"train.png"
            Image.fromarray(bank()["pixels"][0]).save(path)
            entry = dict(task_id="harvest_sand", path=path.name, split="train", world_seed=19,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (path.parent/"manifest.json").write_text(json.dumps(dict(version=1, entries=[
                entry, dict(entry, role="negative", distractor_type="sky")])))
            loaded = load_reference_bank(root, "harvest_sand", 32)
            model = GoalReference(config(), loaded)
            self.assertEqual(len(model.bank_pixels), 1)
            self.assertEqual(len(model.negative_pixels), 1)
            self.assertEqual(len(model._cached_reference()), 1)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_reference_bank(root, "harvest_sand", 32)

    def test_unknown_replay_does_not_train_or_calibrate(self):
        model = GoalReference(config(), bank())
        data = observed()
        data["goal_region_gt"] = np.ones((2, 4, 16, 16, 1), np.uint8)
        data["goal_visible_valid"] = np.ones((2, 4), np.uint8)
        before = [p.clone() for p in model.parameters()]
        report = model.train_batch(data)
        self.assertEqual(report["D/update"], 0.)
        self.assertEqual(len(model.optimizer.state), 0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,model.parameters())))
        self.assertEqual(model.cal_counts.sum().item(), 0)

    def test_sparse_supervision_train_actual_pairing_and_holdout(self):
        model = GoalReference(config(), bank())
        data = labeled()
        data["goal_holdout"] = np.array([[False]*4,[True]*4])
        before = model._cached_reference().clone()
        report = model.train_batch(data)
        self.assertEqual(report["D/region_labels"], 8.)
        self.assertEqual(report["D/positive_regions"], 4.)
        self.assertEqual(report["D/visible_labels"], 4.)
        self.assertEqual(report["D/interaction_labels"], 0.)
        self.assertEqual(report["D/crossview_pairs"], 3.)
        self.assertGreater(report["D/hard_negatives"], 0.)
        self.assertGreater(report["D/grad_norm"], 0.)
        self.assertFalse(torch.equal(before, model._cached_reference()))
        self.assertEqual(model.cal_counts[0, 1].item(), 4)
        model.train_batch(data)
        self.assertEqual(model.cal_counts[0, 1].item(), 4, "Replay repeats are not new calibration samples")

    def test_reference_crop_learning_excludes_its_source_world(self):
        assets = bank()
        assets["manifest"]["entries"][1]["world_seed"] = 17
        same_world = GoalReference(config(), assets)
        self.assertFalse(same_world.cross_world_references.any())
        self.assertEqual(same_world.train_batch(labeled())["D/reference_assets"], 0.)
        other_world = GoalReference(config(), bank())
        self.assertEqual(other_world.cross_world_references.tolist(), [[False, True], [True, False]])
        report = other_world.train_batch(labeled())
        self.assertEqual(report["D/reference_assets"], 2.)
        self.assertGreater(report["D/reference_asset_loss"], 0.)

    def test_interaction_feedback_aligns_to_actual_source(self):
        model = GoalReference(config(), bank())
        data = labeled()
        data["goal_interaction_outcome"] = np.zeros((2,4), np.uint8)
        data["goal_interaction_valid"] = np.zeros((2,4), np.uint8)
        data["goal_interaction_valid"][0,1:3] = 1
        data["goal_interaction_outcome"][0,1:3] = 1
        data["log_applied_control"][0,2,0] = 1
        _, _, labels, _ = model._supervision(data)
        self.assertEqual(labels[:,1,1].nonzero().flatten().tolist(), [1])
        self.assertEqual(labels[:,1,0].nonzero().flatten().tolist(), [1])

    def test_reference_world_cannot_certify_even_with_bad_split_flag(self):
        model = GoalReference(config(), bank())
        data = labeled()
        data["goal_holdout"] = np.ones((2,4), np.uint8)
        data["log_vnext_world_group"] = np.full((2,4), int(model.bank_world_groups[0]), np.int32)
        model.train_batch(data)
        self.assertEqual(model.cal_counts.sum().item(), 0)

    def test_nonfinite_truth_is_unknown(self):
        model = GoalReference(config(), bank())
        data = observed()
        data["goal_visible_gt"] = np.full((2,4), np.nan, np.float32)
        data["goal_visible_valid"] = np.ones((2,4), np.float32)
        self.assertEqual(model.train_batch(data)["D/update"], 0.)

    def test_online_and_replay_fusion_use_same_frame_and_ignore_truth(self):
        model = GoalReference(config(goal_ref_use_raw_image=True), bank())
        model.reliability[0] = .8
        data = labeled()
        data["goal_image"] = np.full((2,4,96,96,3), 56, np.uint8)
        data["goal_image_valid"] = np.array([[1,0,1,0],[0,1,0,1]], np.uint8)
        replay, facts, _ = build_policy_observation(data, model)
        self.assertEqual(tuple(facts["context"].shape), (2,4,8))
        self.assertEqual(tuple(facts["boxes"].shape), (2,4,8,4))
        self.assertEqual(tuple(facts["region"].shape), (2,4,64,64,1))
        self.assertTrue(np.all(data["heatmap"] == 180))
        state = None
        for i in range(4):
            online = {k: v[:, i] for k,v in data.items() if not k.startswith("goal_")
                      or k in ("goal_image", "goal_image_valid")}
            online["search_pose_delta"] = np.full((2,3), 123.4, np.float32)
            online["goal_visible_gt"] = np.full(2, 9., np.float32)
            policy, _, state = build_policy_observation(online, model, state)
            torch.testing.assert_close(policy["heatmap"], replay["heatmap"][:, i])

    def test_zoom_fusion_uses_actual_valid_zoomed_rows(self):
        model = GoalReference(config(), bank())
        model.reliability[0] = .9
        data = observed()
        data["is_zoomed"] = np.array([[0,0,1,0],[1,0,0,0]], bool)
        data["heatmap_on_zoomed"] = data["heatmap"].copy()
        data["zoomed_image"] = data["image"].copy()
        fused = fuse_zoomed_observation(data, model)
        policy, _, _ = build_policy_observation(data, model)
        valid = torch.as_tensor(data["is_zoomed"])
        torch.testing.assert_close(fused["heatmap_on_zoomed"][valid], policy["heatmap"][valid])
        self.assertTrue(np.all(data["heatmap_on_zoomed"] == 180))


if __name__ == "__main__":
    unittest.main()
