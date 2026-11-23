"""Tests for read-only publication archives and threshold-limited imagination exports."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from apm.imagination_figures import ImaginationSampler, render, transition_types
from apm.paper_archive import load_episode
from apm.paper_figures import shortlist, timeline
from apm.visualization import SearchVisualizer
from test_search_visualization import fixture


def imagination_fixture(directory, mixed):
    """Known RGB progression and explicitly supplied edge flags; no learned model."""
    features = torch.arange(15, dtype=torch.float32)[:, None, None].expand(15, 3, 1)/14
    jumps = torch.zeros(15, 3, 1)
    if mixed:
        jumps[[1, 7, 13], 1] = 1
        jumps[4, 2] = 1
    decodes = []
    def decoder(features):
        decodes.append(features.shape)
        y, x = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
        time = features[..., 0, None, None]
        heat = (1-(x-time).abs()*3).clamp(0,1)
        rgb = torch.stack(torch.broadcast_tensors(time, x, y), dim=-1)
        return dict(image=SimpleNamespace(mode=lambda:rgb), heatmap=SimpleNamespace(mode=lambda:heat[...,None]))
    cfg = SimpleNamespace(logdir=directory, vnext_visual_every=10000)
    sampler = ImaginationSampler(cfg, 0)
    sampler.capture(10000, features, jumps, decoder)
    sampler.capture(10001, features, jumps, decoder)
    return Path(directory)/"check_obs/step_000010000", decodes


class PaperTests(unittest.TestCase):
    def test_threshold_rng_no_backfill_and_outgoing_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=SimpleNamespace(logdir=tmp,vnext_visual_every=10)
            sampler=ImaginationSampler(cfg,18)
            feat=torch.randn(7,3,8,requires_grad=True)
            jumps=torch.zeros(7,3,1)
            jumps[[1,4],2]=1
            originals=feat.detach().clone(),jumps.clone()
            class Dist:
                def __init__(self,v): self.v=v
                def mode(self): return self.v
            def decode(f):
                self.assertFalse(torch.is_grad_enabled())
                # Deliberately exercise all three RNGs inside the callback.
                random.random();np.random.rand();torch.rand(2)
                image=f[...,0,None,None,None].expand(-1,-1,11,15,3)
                return dict(image=Dist(image),heatmap=Dist(image[...,:1]))
            before=random.getstate(),np.random.get_state(),torch.get_rng_state()
            for step in (19,20,20,29):
                sampler.capture(step,feat,jumps,decode)
            root=Path(tmp)/"check_obs"
            path=root/"step_000000020"
            self.assertEqual(len(list(root.glob("*/*.png"))),1)
            meta=json.loads((path/"manifest.json").read_text())
            self.assertEqual((meta["column"],meta["long_transition_indices"]),(2,[1,4]))
            self.assertEqual(random.getstate(),before[0])
            after=np.random.get_state()
            np.testing.assert_array_equal(before[1][1],after[1])
            self.assertEqual(before[1][2:],after[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
            self.assertTrue(torch.equal(feat,originals[0]) and torch.equal(jumps,originals[1]))
            self.assertIsNone(feat.grad)
            resumed=ImaginationSampler(cfg,25)
            resumed.capture(26,feat,jumps,decode)
            resumed.capture(30,feat,torch.zeros_like(jumps),decode)
            resumed.capture(40,None,None,decode)
            self.assertEqual(len(list(root.glob("*/*.png"))),2)
            self.assertEqual(json.loads((root/"step_000000040/manifest.json").read_text())["status"],"skipped")
            ImaginationSampler(cfg,0).capture(20,None,None,decode)
            self.assertEqual(json.loads((path/"manifest.json").read_text()),meta)

    def check_fifteen_columns(self, mixed):
        from matplotlib.colors import to_rgba
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            folder, decodes = imagination_fixture(tmp, mixed)
            self.assertEqual(decodes, [(1, 15, 1)])
            meta = json.loads((folder/"manifest.json").read_text())
            self.assertEqual(meta["column"], 1 if mixed else 0)
            self.assertEqual(meta["long_transition_indices"], [1,7,13] if mixed else [])
            with np.load(folder/"sequence.npz") as data:
                rgb, heat, flags = (data[k] for k in ("rgb","heat","jumps"))
            self.assertEqual(rgb.shape[0],15)
            np.testing.assert_array_equal(rgb[:,0,0,0], (torch.arange(15)/14.).mul(255).round().numpy())
            expected = ["short"]*14 + ["terminal"]
            for i in meta["long_transition_indices"]:
                expected[i] = "long"
            self.assertEqual(transition_types(flags,0,15),expected)
            for style,colors in [("paper",["#3c71ae","#bc4231","#858585"]),("original",["blue","red","#858585"])]:
                palette=dict(zip(("short","long","terminal"),colors))
                def inspect(figure, *args, **kwargs):
                    self.assertEqual(len(figure.axes),45)
                    for axis,edge in zip(figure.axes,expected*3):
                        self.assertEqual(axis.patches[0].get_edgecolor(),to_rgba(palette[edge]))
                with patch("matplotlib.figure.Figure.savefig",inspect):
                    render(rgb,heat,flags,meta,folder/"offline.png",style=style)

    def test_fifteen_steps_mixed_outgoing_flags(self):
        self.check_fifteen_columns(True)

    def test_fifteen_steps_without_long_transitions(self):
        self.check_fifteen_columns(False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_decoder_preserves_default_generators_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            feat = torch.randn(15, 2, 3, device="cuda", requires_grad=True)
            feat.sum().backward()
            grad = feat.grad.clone()
            flags = torch.zeros(15, 2, 1, device=feat.device)
            flags[2,1] = 1
            cpu_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state(feat.device)
            def decoder(f):
                torch.rand(1)
                torch.rand(1, device=f.device)
                image = f[:, :, None, None, :].expand(-1,-1,8,8,3)
                return dict(image=SimpleNamespace(mode=lambda:image),
                            heatmap=SimpleNamespace(mode=lambda:image[...,:1]))
            ImaginationSampler(SimpleNamespace(logdir=tmp,vnext_visual_every=10),0).capture(10,feat,flags,decoder)
            self.assertTrue((Path(tmp)/"check_obs/step_000000010/imagination.png").is_file())
            self.assertTrue(torch.equal(cpu_before,torch.get_rng_state()))
            self.assertTrue(torch.equal(cuda_before,torch.cuda.get_rng_state(feat.device)))
            self.assertTrue(torch.equal(grad,feat.grad))

    def test_history_weights_are_the_actual_masked_read(self):
        from apm.spatial_memory import ViewHistoryEncoder
        encoder = ViewHistoryEncoder(8,16)
        args=(torch.randn(2,8),torch.randint(0,255,(2,4,4,4,4),dtype=torch.uint8),
              torch.zeros(2,4,8),torch.tensor([[1,0,1,0],[0,0,0,0]],dtype=torch.bool),torch.tensor([3,5]))
        normal=encoder(*args)
        logged,weights=encoder(*args,return_weights=True)
        self.assertTrue(torch.equal(normal,logged))
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(weights[0].sum().item(),1.,places=6)
        self.assertTrue(torch.all(weights[~args[3]]==0))

    def test_calibrated_fusion_requires_observed_change_and_model_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            viz=SearchVisualizer(tmp,dict(goal_ref_version=2))
            for i,(gate,cal,update,effective) in enumerate([
                    (0,.6,10,50),(.3,.1,10,50),(.3,.6,0,50),(.3,.6,10,0),
                    (.3,.6,10,20),(.3,.6,10,60),(.3,.6,10,40)]):
                obs,output=fixture(i)
                obs["heatmap"]=np.zeros((64,64),np.uint8)
                output.update(log_goal_state=0,log_goal_gate=gate,log_goal_model_update=update,
                              log_goal_region_cal_reliability=cal,
                              log_goal_heatmap_policy=np.full((64,64),effective,np.uint8))
                viz.observe("eval","fusion",obs,output,100)
                groups=viz._active[("eval","fusion")].groups
                if i<4:
                    self.assertNotIn("calibrated active fusion",groups)
            self.assertEqual(groups["calibrated active fusion"]["row"]["step"],5)
            # A zero gate only proves fallback if the two recorded maps actually match.
            obs,output=fixture(7)
            output.update(log_goal_state=0,log_goal_gate=0,log_goal_heatmap_policy=obs["heatmap"])
            viz.observe("eval","fusion",obs,output,100)
            self.assertIn("zero-gate fallback",groups)
            folder=viz.episode_end("eval","fusion",dict(reward=[0]*9,success=[0]*9),100,1,1)
            snap,_=load_episode(folder)
            self.assertTrue((folder/"D_calibrated_fusion.png").exists())
            self.assertAlmostEqual(snap["summary"]["calibrated_fusion_mean_delta"],60/255,places=5)
            self.assertEqual(snap["groups"]["calibrated active fusion"]["row"]["step"],5)
            viz.close()

    def test_archived_timeline_keeps_missing_goal_state_unknown(self):
        rows = [dict(step=0, phase=0, source=1, success=False, reward=0.0),
                dict(step=1, phase=0, source=2, success=False, reward=0.0, goal_state=0)]
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            viz = SearchVisualizer(tmp, {}, scheduled_eval_maps=False)
            def inspect_figure(figure, path):
                for axis in figure.axes[:3]:
                    seen.append(axis.images[0].get_array().ravel().tolist())
                figure.clear()
            viz._save = inspect_figure
            timeline(viz, {"rows": rows}, Path(tmp) / "timeline.png")
            viz.close()
        self.assertEqual(seen, [[0, 0], [1, 2], [-1, 0]])

    def test_archive_aligns_real_feedback_and_preserves_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            viz=SearchVisualizer(tmp,dict(task="fixture",vnext_spatial_fix_version=3))
            obs,log=fixture(0)
            mask=np.zeros(12);mask[[0,3]]=1
            log.update(log_vnext_mask=mask,log_vnext_choice=3,log_vnext_source=1,
                       log_vnext_override=1,log_vnext_requested=6,log_vnext_actor_action=4,
                       log_visual_rank_choice=3,log_vnext_applied=np.zeros(9),
                       log_visual_rank_mask=mask, log_vnext_score_supported=mask,
                       log_visual_template_commands=np.tile([4,5,5],(12,1)),
                       log_visual_template_lengths=np.full(12,3),
                       log_visual_event_support=np.ones((12,10),bool))
            viz.observe("eval","sample",obs,log,20)
            viz.feedback("eval","sample",2.,{"success":True},arrival=dict(image=obs["image"]+9,
                         log_actuator_valid=1,log_applied_control=np.zeros(9)),done=True)
            folder=viz.episode_end("eval","sample",dict(reward=[0,2],success=[0,1]),20,1,1)
            snap,_=load_episode(folder)
            e=snap["a"][0]
            self.assertEqual(shortlist(e),[3,0])
            np.testing.assert_array_equal(e["template_commands"][3],[4,5,5])
            self.assertEqual(e["template_lengths"][3],3)
            np.testing.assert_array_equal(e["supported"],mask)
            np.testing.assert_array_equal(e["rank_mask"],mask)
            self.assertTrue(e["event_support"].all())
            self.assertEqual(e["feedback"],dict(reward=2.,success=True,arrival_index=1,control_matches_controller=None))
            self.assertTrue((e["rgb"]==0).all() and (e["arrival_rgb"]==9).all())
            self.assertEqual(snap["terminal"]["step"],1)
            self.assertTrue((folder/"AC_decision.png").is_file())
            self.assertIn("learned change to Actor command",snap["groups"])
            viz.close()


if __name__=="__main__":
    unittest.main()
