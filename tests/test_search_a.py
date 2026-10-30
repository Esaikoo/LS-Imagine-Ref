"""CPU regressions and real-model integration. Reuses the supplied test entry point."""
import collections
import copy
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from search_a import (SearchOutcomeHead, SearchAController, macro_actions, actor_action_sequence,
                      check_checkpoint, normalize_feature, MODEL_VERSION, LABEL_VERSION)
from search_evidence import (SearchEvidence, SearchEvidenceWrapper, SEARCH_LABEL_KEYS,
                             complete_episode_labels, lidar_rays)

torch.set_num_threads(2)


def config_for_test(**overrides):
    import ruamel.yaml as yaml
    raw = yaml.YAML(typ="safe").load((pathlib.Path(__file__).resolve().parents[1]/"configs.yaml").read_text())
    config = {}
    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and key in dst:
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)
    for name in ("defaults", "minedojo", "apm_a", "apm_a_smoke"):
        merge(config, raw[name])
    config.update(device="cpu", num_actions=12, episode_max_steps=24, search_a_start_step=0,
                  logdir=str(pathlib.Path(__file__).resolve().parents[1]/"logs"/"test_apm_a_v2"/"unit"),
                  search_a_override_prob=1.0, search_a_low_evidence=1.1, search_a_min_labels=1,
                  search_a_min_positives=0, search_a_protect_interaction=False)
    config.update(search_a_model_version=3, search_a_control_version=3, search_a_commit_steps=2)
    config.update(overrides)
    return SimpleNamespace(**config)


def replay_batch(time=12):
    rng = np.random.RandomState(4)
    data = dict(image=rng.randint(256,size=(1,time,64,64,3),dtype=np.uint8),
                heatmap=rng.randint(256,size=(1,time,64,64),dtype=np.uint8),
                action=np.eye(12,dtype=np.float32)[(np.arange(time)%12)[None]],
                is_first=np.zeros((1,time),bool), discount=np.ones((1,time),np.float32))
    data["is_first"][:,0] = True
    for key in ("is_terminal","is_last","is_zoomed","jump","is_calculated","reward","intrinsic",
                "reward_on_zoomed","intrinsic_on_zoomed","jumping_steps","accumulated_reward"):
        data[key] = np.zeros((1,time),np.float32)
    data["zoomed_image"] = data["image"].copy()
    data["heatmap_on_zoomed"] = data["heatmap"].copy()
    for key in SEARCH_LABEL_KEYS:
        data[key] = np.zeros((1,time),np.float32)
    data["search_label_version"][:] = LABEL_VERSION
    for event in ("new","discovery","revisit"):
        data["search_"+event+"_valid"][:] = 1
        data["search_"+event][:, 7] = 1
    return data


def make_agent(config=None, seed=5):
    import gym
    from expr import LS_Imagine
    config = config or config_for_test()
    spaces = {
        "image":gym.spaces.Box(0,255,(64,64,3),np.uint8),
        "heatmap":gym.spaces.Box(0,255,(64,64,1),np.uint8),
        # Deliberately use a vector and a broad encoder regex in the leak test.
        "search_discovery":gym.spaces.Box(0,1,(1,),np.float32)}
    torch.manual_seed(seed)
    agent = LS_Imagine(gym.spaces.Dict(spaces),gym.spaces.Discrete(12),config,
                       SimpleNamespace(step=24),None)
    agent.requires_grad_(False)
    return agent


def fake_rays(block="stone", entity="null", block_distance=5., entity_distance=5., x=0):
    count = 45
    return {"rays":dict(block_name=np.full(count,block), entity_name=np.full(count,entity),
                       block_distance=np.full(count,block_distance),
                       entity_distance=np.full(count,entity_distance),
                       traced_block_x=np.arange(count)+x,
                       traced_block_y=np.zeros(count),traced_block_z=np.zeros(count))}


class EvidenceTests(unittest.TestCase):
    def test_target_aliases_and_occlusion(self):
        for target, block, entity in (("log","oak wood","null"),("sand","sand","null"),
                                     ("water_bucket","water","null"),("iron_ore","iron ore","null"),
                                     ("wool","stone","sheep")):
            collector=SearchEvidence(target)
            collector.observe(fake_rays(), initial=True)
            obs=fake_rays(block,entity,entity_distance=3.)
            labels=collector.observe(obs)
            self.assertEqual(labels["search_discovery"],1)
            self.assertEqual(labels["search_discovery_valid"],1)
        collector=SearchEvidence("log")
        labels=collector.observe(fake_rays("oak wood","cow",entity_distance=2.),initial=True)
        self.assertEqual(labels["search_visible"],0)
        labels=SearchEvidence("wool").observe(fake_rays(entity="sheep",entity_distance=8.),initial=True)
        self.assertEqual(labels["search_visible"],0)  # sheep behind a nearer block

    def test_unknown_labels_and_positive_transition(self):
        c=SearchEvidence("log")
        self.assertEqual(c.observe({},initial=True)["search_discovery_valid"],0)
        r=c.observe(fake_rays("oak wood"))
        self.assertEqual(r["search_visible"],1)
        self.assertEqual(r["search_discovery_valid"],0)  # previous observation unknown
        self.assertEqual(c.observe(fake_rays("oak wood"))["search_discovery"],0)
        c.observe(fake_rays())
        self.assertEqual(c.observe(fake_rays("oak wood"))["search_discovery"],1)
        broken=fake_rays();broken["rays"]["block_distance"][0]=np.nan
        self.assertEqual(c.observe(broken)["search_visible_valid"],0)

    def test_surface_new_revisit_and_episode_reset(self):
        c=SearchEvidence("log",revisit_gap=2)
        self.assertEqual(c.observe(fake_rays(),initial=True)["search_new"],1)
        self.assertEqual(c.observe(fake_rays())["search_revisit"],0)
        self.assertEqual(c.observe(fake_rays())["search_revisit"],1)
        self.assertEqual(c.observe(fake_rays(x=100))["search_new"],1)
        self.assertEqual(c.observe(fake_rays(),initial=True)["search_revisit"],0)
        sky=fake_rays(block="air")
        self.assertEqual(c.observe(sky)["search_new_valid"],0)
        self.assertEqual(len(lidar_rays()),45)

    def test_eval_wrapper_never_uses_privileged_data(self):
        import gym
        env=gym.Env()
        env.reset=lambda:fake_rays("oak wood")
        env.step=lambda action:(fake_rays("oak wood"),0.,False,{})
        wrapper=SearchEvidenceWrapper(env,enabled=False)
        obs=wrapper.reset()
        self.assertEqual(obs["search_discovery_valid"],0)
        self.assertEqual(obs["search_visible_valid"],0)

    def test_minedojo_sensor_enabled_only_for_label_collection(self):
        from envs.tasks.minedojo import make_minedojo
        task=dict(search_evidence_specs=dict(enabled=True,target="log",distance=32.))
        with mock.patch("envs.tasks.minedojo._meta_task_make",return_value=object()) as make:
            with mock.patch("envs.tasks.minedojo._add_wrappers",side_effect=lambda env,*a,**k:env):
                make_minedojo("harvest",task,{})
                self.assertTrue(make.call_args.kwargs["use_lidar"])
                self.assertEqual(len(make.call_args.kwargs["lidar_rays"]),45)
                task["search_evidence_specs"]["enabled"]=False
                make_minedojo("harvest",task,{})
                self.assertFalse(make.call_args.kwargs["use_lidar"])


class PredictionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model=SearchOutcomeHead(32,horizon=6,hidden=32)
        self.feat=torch.randn(1,18,32,requires_grad=True)
        self.batch={
            "action":F.pad(F.one_hot(torch.arange(18)[None]%12,12).float(),(0,1)),
            "is_first":torch.zeros(1,18),
            "is_zoomed":torch.ones(1,18),
            "search_label_version":torch.full((1,18),LABEL_VERSION)}
        self.batch["is_first"][:,0]=1

    def test_shapes_alignment_and_stop_gradient_targets(self):
        with mock.patch.object(self.model,"forward",wraps=self.model.forward) as call:
            loss,metrics,_,_=self.model.training_loss(self.feat,self.batch)
        inputs,windows=call.call_args.args
        torch.testing.assert_close(inputs,self.feat[:,:-6])
        torch.testing.assert_close(windows[:,0],self.batch["action"][:,1:7,:12])
        self.assertEqual(metrics["A/valid_windows"].item(),12)
        loss.backward()
        self.assertGreater(self.feat.grad[:,:-6].abs().sum().item(),0)
        self.assertEqual(self.feat.grad[:,-6:].abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_boundaries_and_auxiliary_zoom_does_not_remove_real_frames(self):
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),12)
        self.batch["action"][:,8,12]=1
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),6)
        self.batch["action"][:,:,12]=0
        self.batch["is_terminal"]=torch.zeros(1,18)
        self.batch["is_terminal"][:,8]=1
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/valid_windows"].item(),6)
        self.batch["is_first"][:,1:]=1
        loss,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(loss.item(),0)

    def test_unknown_is_not_negative_and_any_positive_is_known(self):
        self.batch["search_discovery"]=torch.full((1,18),float("nan"))
        self.batch["search_discovery_valid"]=torch.zeros(1,18)
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/discovery_valid"].item(),0)
        self.batch["search_discovery"][:,6]=1
        self.batch["search_discovery_valid"][:,6]=1
        _,m,counts,positive=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/discovery_valid"].item(),6)
        self.assertEqual(m["A/discovery_positive_rate"].item(),1)
        self.assertFalse(self.model.ready_events(1,0).any())
        self.model.record_supervision(counts,positive)
        self.assertFalse(self.model.ready_events(1,0)[1])  # no negative example yet
        self.batch["search_discovery"][:]=0
        self.batch["search_discovery_valid"][:]=1
        _,_,counts,positive=self.model.training_loss(self.feat,self.batch)
        self.model.record_supervision(counts,positive)
        self.assertTrue(self.model.ready_events(1,1)[1])

    def test_revisit_is_endpoint_target_not_any_in_window(self):
        self.batch["search_revisit"]=torch.zeros(1,18)
        self.batch["search_revisit"][:,1]=1
        self.batch["search_revisit_valid"]=torch.ones(1,18)
        _,m,_,_=self.model.training_loss(self.feat,self.batch)
        self.assertEqual(m["A/revisit_positive_rate"].item(),0)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.head=SearchOutcomeHead(32,hidden=32)
        self.head.label_counts[:]=10
        self.head.positive_counts[:]=5
        self.obs={"heatmap":torch.zeros(2,64,64,1)}
        self.feat=torch.zeros(2,32)
        self.actor=F.one_hot(torch.tensor([0,0]),12).float()

    def choose(self, ctrl, state=None, first=None, best=0, training=False):
        def predict(feat,seq):
            logits=torch.full((*feat.shape[:-1],3),-8.)
            logits[:,best,1]=8.
            return {"future":torch.zeros_like(feat),"event_logits":logits}
        continuation=lambda indices:self.actor[indices,None].expand(-1,6,-1)
        with mock.patch.object(self.head,"forward",side_effect=predict):
            return ctrl.choose(self.head,self.feat,self.obs,self.actor,state,
                               torch.zeros(2) if first is None else first,100,training,continuation)

    def controller(self, **kw):
        defaults=dict(start_step=0,override_prob=1.,epsilon=0.,min_labels=1,min_positives=1)
        defaults.update(kw)
        return SearchAController(**defaults)

    def test_model_selects_macro_or_actor(self):
        out,state,info=self.choose(self.controller(),best=0)
        self.assertEqual(out.argmax(-1).tolist(),[4,4])
        self.assertEqual(info["log_search_override"].tolist(),[1,1])
        out,state,info=self.choose(self.controller(),best=5)
        self.assertEqual(out.argmax(-1).tolist(),[0,0])
        self.assertEqual(info["log_search_override"].sum().item(),0)
        self.assertEqual(info["log_search_decision"].sum().item(),2)

    def test_reset_finish_and_deterministic_schedule(self):
        ctrl=self.controller(override_prob=.25)
        state=None
        rng=torch.random.get_rng_state()
        for t in range(9):
            out,state,info=self.choose(ctrl,state=state)
            self.assertEqual(info["log_search_override"][0].item(),float(t>=3))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(state[0].tolist(),[-1,-1])
        ctrl=self.controller()
        _,state,_=self.choose(ctrl)
        ctrl.override_prob=0
        _,state,info=self.choose(ctrl,state,first=torch.tensor([1,0]))
        self.assertEqual(info["log_search_override"].tolist(),[0,1])

    def test_unknown_event_heads_do_not_control_eval(self):
        self.head.label_counts.zero_()
        _,_,info=self.choose(self.controller())
        self.assertEqual(info["log_search_decision"].sum().item(),0)

    def test_actions_stay_twelve_dimensional(self):
        actions=macro_actions(6,"cpu")
        self.assertEqual(tuple(actions.shape),(5,6,12))
        self.assertTrue((actions.sum(-1)==1).all())
        self.assertEqual(actions[:,0].argmax(-1).tolist(),[4,5,6,6,6])


class ReplayAndIntegrationTests(unittest.TestCase):
    def test_mixed_old_new_replay_and_npz(self):
        import tools
        old=replay_batch(18)
        old={k:list(v[0]) for k,v in old.items() if not k.startswith("search_")}
        new=copy.deepcopy(old)
        for key in SEARCH_LABEL_KEYS:
            new[key]=[np.float32(MODEL_VERSION if key=="search_label_version" else 1)]*18
        cache=collections.OrderedDict(old=old,new=new)
        with tempfile.TemporaryDirectory() as directory:
            tools.save_episodes(directory,cache)
            loaded=tools.load_episodes(directory)
            generator=tools.sample_episodes(loaded,24)
            batch=next(tools.from_generator(generator,4))
        self.assertEqual(batch["action"].shape,(4,24,12))
        self.assertTrue((batch["search_discovery_valid"]==0).any())
        self.assertTrue((batch["search_discovery_valid"]==1).any())
        self.assertEqual({len(v) for v in complete_episode_labels(old).values()},{18})

    def test_metadata_backfills_entire_prefill(self):
        import tools
        cache={}
        for t in range(8):
            row=dict(reward=0.,image=np.zeros((4,4,3),np.uint8))
            if t>=5:
                row.update(log_search_override=np.float32(1),
                           log_search_scores=np.zeros(6,np.float32))
            tools.add_to_cache(cache,"ep",row)
        self.assertEqual({len(v) for v in cache["ep"].values()},{8})
        self.assertEqual(np.asarray(cache["ep"]["log_search_scores"]).shape,(8,6))

    def test_real_joint_update_actual_action_and_checkpoint(self):
        import tools
        agent=make_agent(config_for_test(search_a_epsilon=1.))
        data=replay_batch()
        data["is_zoomed"][:,2]=1
        data["is_calculated"][:,2]=1
        data["jumping_steps"][:,2]=2
        before=next(agent._wm.search_a.parameters()).detach().clone()
        agent._train(data)
        self.assertFalse(torch.equal(before,next(agent._wm.search_a.parameters())))
        self.assertTrue(all(np.isfinite(v).all() for values in agent._metrics.values() for v in values))
        self.assertGreater(agent._wm.search_a.label_counts.sum().item(),0)
        obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
        out,state=agent._policy(obs,None,training=True)
        self.assertEqual(out["log_search_override"].sum().item(),1)
        obs["is_first"][:]=False
        with mock.patch.object(agent._wm.dynamics,"obs_step",wraps=agent._wm.dynamics.obs_step) as call:
            agent._policy(obs,state,training=True)
        torch.testing.assert_close(call.call_args.args[1],out["action"])
        with mock.patch.object(agent._wm.dynamics,"img_step",wraps=agent._wm.dynamics.img_step) as call:
            seq=actor_action_sequence(agent._wm.dynamics,agent._task_behavior.actor,state[0],out["action"],6)
        self.assertEqual(seq.shape,(1,6,12))
        self.assertTrue(all((args.args[1][:,-1]==0).all() for args in call.call_args_list))
        checkpoint=dict(agent_state_dict=agent.state_dict(),step=agent._step,search_a_version=MODEL_VERSION,
                        optims_state_dict=tools.recursively_collect_optim_state_dict(agent))
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/"latest.pt";torch.save(checkpoint,p);checkpoint=torch.load(p,map_location="cpu")
        check_checkpoint(checkpoint,True)
        other=make_agent(config_for_test(search_a_control=False))
        other.load_state_dict(checkpoint["agent_state_dict"],strict=True)
        tools.recursively_load_optim_state_dict(other,checkpoint["optims_state_dict"])
        self.assertIsNone(other._search_a)
        torch.testing.assert_close(other._wm.search_a.label_counts,agent._wm.search_a.label_counts)
        self.assertNotIn("log_search_override",other._policy(obs,None,False)[0])

    def test_privileged_labels_never_enter_encoder_or_policy(self):
        cfg=config_for_test(search_a_control=False)
        cfg.encoder["mlp_keys"]=".*"
        agent=make_agent(cfg)
        self.assertNotIn("search_discovery",agent._wm.encoder.mlp_shapes)
        data=replay_batch()
        obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
        with mock.patch.object(agent._wm.encoder,"forward",wraps=agent._wm.encoder.forward) as call:
            torch.manual_seed(7);out1,state1=agent._policy(obs,None,False)
        self.assertFalse(any(k.startswith("search_") for k in call.call_args.args[0]))
        changed={k:(v+1000 if k.startswith("search_") else v) for k,v in obs.items()}
        torch.manual_seed(7);out2,state2=agent._policy(changed,None,False)
        torch.testing.assert_close(out1["action"],out2["action"])
        torch.testing.assert_close(state1[0]["deter"],state2[0]["deter"])

    def test_master_off_and_frozen_loss(self):
        off=make_agent(config_for_test(search_a_enabled=False))
        on=make_agent()
        self.assertIsNone(off._wm.search_a)
        self.assertIsNone(off._search_a)
        for (name,p),(other,q) in zip(off._task_behavior.actor.named_parameters(),on._task_behavior.actor.named_parameters()):
            self.assertEqual(name,other);torch.testing.assert_close(p,q)
        self.assertFalse(any(".search_a." in k for k in off.state_dict()))
        frozen=make_agent(config_for_test(search_a_train=False,search_a_control=False))
        before={k:v.clone() for k,v in frozen._wm.search_a.state_dict().items()}
        frozen._train(replay_batch())
        for k,v in frozen._wm.search_a.state_dict().items():
            torch.testing.assert_close(v,before[k])
        for overrides in (dict(search_a_train=False), dict(search_a_loss_weight=0.),
                          dict(search_a_event_weight=0.,search_a_visible_weight=0.)):
            inactive=make_agent(config_for_test(search_a_epsilon=1., **overrides))
            self.assertEqual(inactive._search_a.epsilon,0.)
            data=replay_batch()
            obs={k:v[:,0] for k,v in data.items() if k not in ("action","discount")}
            out,_=inactive._policy(obs,None,True)
            self.assertEqual(out["log_search_override"].item(),0.)

    def test_reject_v1_checkpoint_and_wrong_off_mode(self):
        a=make_agent()
        cp=dict(agent_state_dict=a.state_dict())
        with self.assertRaises(ValueError):check_checkpoint(cp,True)
        cp["search_a_version"]=MODEL_VERSION
        with self.assertRaises(ValueError):check_checkpoint(cp,False)


class LoggerPersistenceTests(unittest.TestCase):
    def test_console_setting_never_filters_json_or_tensorboard(self):
        import contextlib
        import io
        import json
        import tools
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        values = {
            "eval_success": 0.125, "eval_first_success_step": 123.5,
            "eval_search_rate": 0.375, "A/prediction_loss": 0.003217,
            "model_loss": 1234., "actor_loss": -0.25, "value_loss": 0.5,
            "image_loss": 0.75, "zoomed_image_loss": 0.625,
            "kl": 3., "model_grad_norm": 2., "train_return": 1.,
            "eval_return": 2., "A/new_loss": 0.25, "A/revisit_loss": 0.125,
            "A/discovery_valid": 6., "future_metric_not_in_any_allowlist": 7.,
            "B/revisit_loss": 0.25, "B/consistency_loss": 0.125, "B/place_valid": 3.,
            "eval_b_revisit_rate": 0.75,
        }
        for verbose in (False, True):
            with self.subTest(verbose=verbose), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                logger = tools.Logger(SimpleNamespace(use_wandb=False, log_all_metrics=verbose),
                                      root, 160000)
                try:
                    for name, value in values.items():
                        logger.scalar(name, value)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        logger.write(fps=True)
                    row = json.loads((root / "metrics.jsonl").read_text())
                    self.assertEqual(set(row), set(values) | {"step", "fps"})
                    self.assertEqual(row["step"], 160000)
                    for name, value in values.items():
                        self.assertEqual(row[name], value)
                    self.assertIn("eval_success 12.5%", output.getvalue())
                    self.assertIn("A/prediction_loss 0.003217", output.getvalue())
                    self.assertEqual("model_loss" in output.getvalue(), verbose)
                    events = EventAccumulator(str(root)).Reload()
                    expected = {tools.Logger._scalar_tag(k) for k in set(values) | {"fps"}}
                    self.assertEqual(set(events.Tags()["scalars"]), expected)
                    self.assertEqual(events.Scalars("scalars/model_loss")[0].value, 1234.)
                    self.assertEqual(events.Scalars("scalars/actor_loss")[0].value, -0.25)
                    self.assertEqual(events.Scalars("A/new_loss")[0].value, 0.25)
                finally:
                    logger._writer.close()

    def test_finish_saves_pending_partial_interval_and_video(self):
        import json
        import tools
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tools, "SummaryWriter") as writer:
            logger = tools.Logger(SimpleNamespace(use_wandb=False), pathlib.Path(directory), 1000000)
            agent = make_agent()
            agent._logger = logger
            agent._metrics = {"actor_loss": [2., 4.], "C/updates": [7., 9.]}
            agent.flush_metrics()
            logger.video("eval_openl", np.zeros((1, 2, 4, 4, 3), np.uint8))
            logger.finish()
            row = json.loads((pathlib.Path(directory)/"metrics.jsonl").read_text())
            self.assertEqual(row["actor_loss"], 3.)
            self.assertEqual(row["C/updates"], 8.)
            self.assertEqual(row["step"], 1000000)
            self.assertFalse(any(agent._metrics.values()))
            writer.return_value.add_video.assert_called_once()
            writer.return_value.flush.assert_called_once()
            writer.return_value.close.assert_called_once()

    def test_non_console_and_offline_metrics_are_still_saved(self):
        import contextlib
        import io
        import json
        import tools
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            # Older config objects do not contain log_all_metrics.
            logger = tools.Logger(SimpleNamespace(use_wandb=False), root, 160000)
            try:
                logger.scalar("model_grad_norm", 42.)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    logger.write()
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(json.loads((root / "metrics.jsonl").read_text()),
                                 {"step": 160000, "model_grad_norm": 42.})
                logger.offline_scalar("original_offline_metric", 5., 160001)
                logger._writer.flush()
                events = EventAccumulator(str(root)).Reload()
                self.assertEqual(events.Scalars("scalars/model_grad_norm")[0].value, 42.)
                self.assertEqual(events.Scalars("scalars/original_offline_metric")[0].value, 5.)
            finally:
                logger._writer.close()

    def test_wandb_keeps_all_scalars_without_console_verbosity(self):
        import contextlib
        import io
        import tools
        values = {"model_loss": 2., "actor_loss": -0.5, "value_loss": 1.,
                  "A/prediction_loss": 0.25, "A/revisit_loss": 0.125,
                  "A/discovery_valid": 6., "new_unlisted_metric": 3.}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tools, "wandb") as backend:
            logger = tools.Logger(SimpleNamespace(use_wandb=True, wandb_key="test",
                                                   log_all_metrics=False),
                                  pathlib.Path(directory), 160000)
            for name, value in values.items():
                logger.scalar(name, value)
            with contextlib.redirect_stdout(io.StringIO()):
                logger.write(fps=True)
            actual = {name: value for call in backend.log.call_args_list
                      for name, value in call.args[0].items()}
            expected = {("diagnostics/" + name if name in tools.Logger.DIAGNOSTIC_METRICS else name): value
                        for name, value in values.items()}
            self.assertEqual(actual, dict(expected, fps=0))
            logger.offline_scalar("original_offline_metric", 9., 160001)
            self.assertEqual(backend.log.call_args.args[0], {"original_offline_metric": 9.})


class EvaluationAControlTests(unittest.TestCase):
    def test_disable_only_control_retains_model_and_strict_checkpoint(self):
        from evaluate_checkpoints import apply_a_control_mode
        cfg = config_for_test()
        before = copy.deepcopy(vars(cfg))
        apply_a_control_mode(vars(cfg), 'off')
        changed = {k for k in before if before[k] != vars(cfg)[k]}
        self.assertEqual(changed, {'search_a_control'})
        enabled = make_agent(config_for_test())
        disabled = make_agent(cfg)
        disabled.load_state_dict(enabled.state_dict(), strict=True)
        self.assertIsNotNone(disabled._wm.search_a)
        self.assertIsNone(disabled._search_a)
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))

    def test_checkpoint_mode_unchanged_and_non_a_rejected(self):
        from evaluate_checkpoints import apply_a_control_mode
        config = dict(search_a_enabled=False, search_a_control=True)
        before = dict(config)
        apply_a_control_mode(config, 'checkpoint')
        self.assertEqual(config, before)
        with self.assertRaises(ValueError):
            apply_a_control_mode(config, 'off')

    def test_evaluation_restore_preserves_requested_B_and_C_off(self):
        from test import configure_evaluation
        saved = dict(vars(config_for_test()), search_b_enabled=True, search_b_control=True,
                     search_c_enabled=True, search_c_control=True,
                     search_a_model_version=4, search_a_control_version=4)
        checkpoint = dict(config=saved, step=10000, search_a_version=4,
                          search_b_version=2, search_c_version=1,
                          agent_state_dict={"_wm.search_a.x": torch.zeros(1),
                                            "_wm.search_b.x": torch.zeros(1),
                                            "_search_c.x": torch.zeros(1)})
        cfg = config_for_test(search_b_control=False, search_c_control=False)
        configure_evaluation(cfg, checkpoint)
        self.assertTrue(cfg.search_b_enabled)
        self.assertTrue(cfg.search_c_enabled)
        self.assertFalse(cfg.search_b_control)
        self.assertFalse(cfg.search_c_control)
        self.assertEqual(cfg.search_a_model_version, 4)

    def test_v4_rejects_old_repaired_control_and_C_requires_step(self):
        from evaluate_checkpoints import apply_a_control_mode, configure_search_checkpoint
        with self.assertRaises(ValueError):
            apply_a_control_mode(dict(search_a_enabled=True, search_a_model_version=4), "repaired")
        with self.assertRaises(ValueError):
            configure_search_checkpoint(vars(config_for_test()), dict(
                config={"search_c_enabled": True}, search_c_version=1,
                agent_state_dict={"_search_c.x": torch.zeros(1)}))

    def test_AC_and_BC_presets_can_be_selected_directly(self):
        import yaml
        raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1]/"configs.yaml").read_text())
        for preset in ("apm_ac", "apm_bc"):
            values = {}
            for name in ("defaults", "minedojo", preset):
                values.update(raw[name])
            self.assertFalse(values["compile"])
            self.assertEqual(values["search_a_control_version"], 4)
            self.assertTrue(values["search_c_enabled"])
            if preset == "apm_ac":
                self.assertTrue(values["search_a_enabled"])
                self.assertEqual(values["search_a_model_version"], 4)

    def test_scene_retry_never_retries_policy_failures_or_changed_spawn(self):
        import json
        from evaluate_checkpoints import retryable_initial_scene
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            (folder/'episodes').mkdir()
            (folder/'initials').mkdir()
            record = dict(status='error',steps=0,success=False,termination='error',
                          error='Initial scene mismatch')
            checks = dict(match=False,checks=dict(terrain_sha256=False,pos=True,life_stats=True))
            rp=folder/'episodes/episode_006.json'
            cp=folder/'initials/episode_006_comparison.json'
            rp.write_text(json.dumps(record));cp.write_text(json.dumps(checks))
            self.assertTrue(retryable_initial_scene(folder,6))
            rp.write_text(json.dumps(dict(record,steps=1)))
            self.assertFalse(retryable_initial_scene(folder,6))
            rp.write_text(json.dumps(dict(record,status='completed',termination='time_limit')))
            self.assertFalse(retryable_initial_scene(folder,6))
            rp.write_text(json.dumps(record))
            checks['checks']['pos']=False;cp.write_text(json.dumps(checks))
            self.assertFalse(retryable_initial_scene(folder,6))

    def test_reference_requires_same_weights_and_search_settings(self):
        import json
        from evaluate_checkpoints import validate_reference
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            reference = root/'reference'
            folder = reference/'harvest_sand'/'run_00'
            folder.mkdir(parents=True)
            checkpoint_root = root/'ckpt'
            args = SimpleNamespace(reference_output=reference,worker_episode=18,
                checkpoint_root=checkpoint_root,a_control='off',
                seed=2026102600,runs=1,episodes=20,fps=20,heatmap_mode='unet')
            (reference/'manifest.json').write_text(json.dumps(dict(seed=args.seed,runs=1,episodes=20,fps=20)))
            config = dict(search_a_enabled=True,search_a_control=True,search_a_horizon=6,
                          search_b_enabled=False,seed=args.seed,device='cpu')
            specs = dict(fast_reset=None)
            sim = dict(seed=2026102618,world_seed='2026102618')
            settings = dict(config=config,task_specs=specs,sim_specs=sim,
                            checkpoint=str(checkpoint_root/'harvest_sand'/'latest.pt'))
            path = folder/'settings_episode_018.json'
            path.write_text(json.dumps(settings))
            off = dict(config,search_a_control=False)
            self.assertTrue(validate_reference(args,'harvest_sand',0,off,specs,sim)['common_config_matches'])
            with self.assertRaises(ValueError):
                validate_reference(args,'harvest_sand',0,dict(off,search_a_horizon=9),specs,sim)
            settings['checkpoint'] = str(root/'different.pt')
            path.write_text(json.dumps(settings))
            with self.assertRaises(ValueError):
                validate_reference(args,'harvest_sand',0,off,specs,sim)



class RepairedControlTests(unittest.TestCase):
    setUp = ControlTests.setUp
    choose = ControlTests.choose

    def controller(self, **kwargs):
        settings = dict(control_version=3,start_step=0,override_prob=1.,epsilon=0.,
                        min_labels=1,min_positives=1)
        settings.update(kwargs)
        return SearchAController(**settings)

    def test_new_evidence_interrupts_on_same_step_without_restart(self):
        ctrl = self.controller()
        _, state, info = self.choose(ctrl)
        self.assertEqual(info["log_search_override"].sum().item(),2)
        self.obs["heatmap"].fill_(1.)
        out, state, info = self.choose(ctrl,state)
        self.assertEqual(out.argmax(-1).tolist(),[0,0])
        self.assertEqual(info["log_search_interrupt"].tolist(),[1,1])
        self.assertEqual(info["log_search_completed_length"].tolist(),[1,1])
        self.assertEqual(info["log_search_decision"].sum().item(),0)
        self.obs["heatmap"].zero_()
        _, _, info = self.choose(ctrl,state)
        self.assertEqual(info["log_search_override"].sum().item(),0)

    def test_actor_interaction_preempts_macro(self):
        ctrl = self.controller()
        _, state, _ = self.choose(ctrl)
        self.actor = F.one_hot(torch.tensor([1,11]),12).float()
        out, _, info = self.choose(ctrl,state)
        torch.testing.assert_close(out,self.actor)
        self.assertEqual(info["log_search_interrupt"].tolist(),[2,2])

    def test_repeated_ineffective_interaction_does_not_lock_out_search(self):
        ctrl = self.controller()
        self.actor = F.one_hot(torch.tensor([1,11]),12).float()
        state = None
        for _ in range(2):
            _, state, info = self.choose(ctrl,state)
            self.assertEqual(info["log_search_override"].sum().item(),0)
        _, _, info = self.choose(ctrl,state)
        self.assertEqual(info["log_search_override"].sum().item(),2)

    def test_rolling_budget_commit_limit_and_episode_reset(self):
        ctrl = self.controller(cooldown=0)
        state, actions = None, []
        for _ in range(240):
            _,state,info = self.choose(ctrl,state)
            actions.append(int(info["log_search_override"][0]))
            self.assertLessEqual(info["log_search_completed_length"].max().item(),2)
        for start in range(len(actions)-99):
            self.assertLessEqual(sum(actions[start:start+100]),20)
        self.assertGreater(sum(actions),0)
        _,state,info = self.choose(ctrl,state,first=torch.ones(2))
        self.assertEqual(info["log_search_override"].sum().item(),2)
        self.assertEqual(state[3].sum(-1).tolist(),[1.,1.])

    def test_vertical_candidates_keep_actor_id_and_twelve_actions(self):
        for candidate, action in ((6,3),(7,2)):
            out,_,info = self.choose(self.controller(),best=candidate)
            self.assertEqual(out.argmax(-1).tolist(),[action,action])
            self.assertEqual(tuple(out.shape),(2,12))
            self.assertEqual(tuple(info["log_search_scores"].shape),(2,8))
            self.assertEqual(info["log_search_macro_id"].tolist(),[candidate,candidate])
        out,_,info = self.choose(self.controller(),best=5)
        torch.testing.assert_close(out,self.actor)
        self.assertEqual(info["log_search_override"].sum().item(),0)

    def test_predictor_compares_conditional_actor_followup(self):
        ctrl=self.controller()
        def tails(indices,templates):
            result=templates[None].expand(len(indices),-1,-1,-1).clone()
            result[:,:,2:]=self.actor[indices,None,None]
            return result
        def prediction(feat,seq):
            self.assertTrue((seq[:,:5,2:].argmax(-1)==0).all())
            logits=torch.full((*feat.shape[:-1],3),-8.)
            logits[:,0,1]=8.
            return dict(future=torch.zeros_like(feat),event_logits=logits)
        with mock.patch.object(self.head,"forward",side_effect=prediction):
            out,_,_=ctrl.choose(self.head,self.feat,self.obs,self.actor,None,torch.zeros(2),100,False,
                               lambda i:self.actor[i,None].expand(-1,6,-1),candidate_sequence=tails)
        self.assertEqual(out.argmax(-1).tolist(),[4,4])

    def test_revisit_improvement_alone_does_not_justify_override(self):
        def predict(feat,actions):
            logits = torch.full((*feat.shape[:-1],3),-8.)
            logits[:,5,2] = 8.
            return dict(future=torch.zeros_like(feat),event_logits=logits)
        ctrl = self.controller()
        with mock.patch.object(self.head,"forward",side_effect=predict):
            out,_,info=ctrl.choose(self.head,self.feat,self.obs,self.actor,None,torch.zeros(2),100,False,
                                  lambda i:self.actor[i,None].expand(-1,6,-1))
        torch.testing.assert_close(out,self.actor)
        self.assertEqual(info["log_search_override"].sum().item(),0)

    def test_training_exploration_has_private_restorable_rng(self):
        ctrl = self.controller(epsilon=1.,cooldown=0)
        self.head.label_counts.zero_()
        global_rng = torch.random.get_rng_state()
        _,state,_ = self.choose(ctrl,training=True)
        saved_rng = ctrl.rng_state()
        restored = self.controller(epsilon=1.,cooldown=0)
        restored.load_rng_state(saved_rng)
        one = self.choose(ctrl,state,training=True)
        two = self.choose(restored,state,training=True)
        torch.testing.assert_close(one[0],two[0])
        self.assertTrue(torch.equal(global_rng,torch.random.get_rng_state()))
        history, state = [], None
        for _ in range(120):
            _,state,info = self.choose(ctrl,state,training=True)
            history.append(int(info["log_search_override"][0]))
        self.assertTrue(all(sum(history[i:i+100])<=20 for i in range(21)))

    def test_visibility_conflict_uses_only_trained_prediction(self):
        ctrl = self.controller(conflict_steps=3)
        self.obs["heatmap"].fill_(1.)
        state = None
        with mock.patch.object(self.head,"visibility",return_value=torch.zeros(2)):
            # Unready prediction cannot overrule the old heatmap gate.
            _,state,info = self.choose(ctrl,state)
            self.assertEqual(info["log_search_override"].sum().item(),0)
            self.head.visible_counts[:] = 10
            self.head.visible_positives[:] = 5
            for _ in range(2):
                _,state,info = self.choose(ctrl,state)
                self.assertEqual(info["log_search_override"].sum().item(),0)
            _,_,info = self.choose(ctrl,state)
            self.assertEqual(info["log_search_override"].sum().item(),2)


class RepairTrainingTests(unittest.TestCase):
    def test_visibility_targets_use_label_v2_and_mask_unknowns(self):
        head=SearchOutcomeHead(32,hidden=32)
        feat=torch.randn(1,12,32,requires_grad=True)
        data={k:torch.as_tensor(v) for k,v in replay_batch().items()}
        data["search_visible_valid"][:]=1
        data["search_visible"][:,6:]=1
        loss,metrics,counts,positive=head.training_loss(feat,data)
        self.assertEqual(metrics["A/current_visible_valid"].item(),12)
        self.assertEqual(metrics["A/future_visible_valid"].item(),6)
        self.assertEqual(metrics["A/future_visible_positive_rate"].item(),1)
        loss.backward()
        self.assertGreater(head.current_visible[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(head.future_visible.weight.grad.abs().sum().item(),0)
        head.record_supervision(counts,positive)
        self.assertTrue(head.ready_visibility(1,1)[0])
        self.assertFalse(head.ready_visibility(1,1)[1])  # All terminal labels positive.
        data["search_visible_valid"].zero_()
        _,metrics,_,_=head.training_loss(feat,data)
        self.assertEqual(metrics["A/current_visible_loss"].item(),0)
        self.assertEqual(metrics["A/future_visible_loss"].item(),0)

    def test_scaled_input_changes_only_shared_gradient(self):
        head=SearchOutcomeHead(32,hidden=32)
        source=torch.randn(1,12,32)
        data={k:torch.as_tensor(v) for k,v in replay_batch().items()}
        grads, losses = [], []
        for alpha in (0.,0.1,1.):
            feat=source.clone().requires_grad_(True)
            head.zero_grad()
            scaled=feat.detach()+alpha*(feat-feat.detach())
            loss,_,_,_=head.training_loss(scaled,data)
            loss.backward()
            losses.append(loss.item())
            grads.append(feat.grad.clone())
            self.assertGreater(head.input[0].weight.grad.abs().sum().item(),0)
        self.assertEqual(losses[0],losses[2])
        self.assertEqual(grads[0].abs().sum().item(),0)
        torch.testing.assert_close(grads[1],grads[2]*0.1)

    def test_detached_A_does_not_change_base_model_update(self):
        baseline=make_agent(config_for_test(search_a_enabled=False))
        repaired=make_agent(config_for_test(search_a_shared_grad_scale=0.))
        before=next(repaired._wm.search_a.parameters()).detach().clone()
        data=replay_batch()
        rng=torch.random.get_rng_state()
        baseline._wm._train(copy.deepcopy(data),step=20000)
        torch.random.set_rng_state(rng)
        _,_,_,metrics=repaired._wm._train(copy.deepcopy(data),step=20000)
        for name,param in baseline._wm.named_parameters():
            torch.testing.assert_close(param,dict(repaired._wm.named_parameters())[name],
                                       atol=1e-7,rtol=1e-5,msg=name)
        self.assertFalse(torch.equal(before,next(repaired._wm.search_a.parameters())))
        self.assertEqual(metrics["A/shared_grad_scale"],0)
        self.assertEqual(metrics["search_a_update_applied"],1)

    def test_shared_gradient_schedule_uses_restored_environment_step(self):
        agent=make_agent(config_for_test(search_a_grad_warmup=10,search_a_grad_ramp=20,
                                        search_a_shared_grad_scale=0.1))
        for step, expected in ((5,0.),(20,0.05),(30,0.1)):
            _,_,_,metrics=agent._wm._train(replay_batch(),step=step)
            self.assertAlmostEqual(float(metrics["A/shared_grad_scale"]),expected,places=6)
            self.assertEqual(metrics["model_update_applied"],1)
            self.assertEqual(metrics["search_a_update_applied"],1)

    def test_independent_clipping_and_atomic_nonfinite_skip(self):
        import tools
        base=torch.nn.Parameter(torch.tensor(1.))
        extra=torch.nn.Parameter(torch.tensor(1.))
        primary=tools.Optimizer("base",[base],.1,clip=1.,wd=0.,opt="sgd")
        auxiliary=tools.Optimizer("extra",[extra],.1,clip=1.,wd=0.,opt="sgd")
        primary(base.square()+extra.square()*1e6,[base],auxiliary=auxiliary)
        self.assertAlmostEqual(base.item(),.9,places=5)
        saved=base.detach().clone(),extra.detach().clone()
        result=primary(base.square()+extra*float("inf"),[base],auxiliary=auxiliary)
        torch.testing.assert_close(base,saved[0])
        torch.testing.assert_close(extra,saved[1])
        self.assertEqual(result["extra_update_applied"],0)

    def test_old_checkpoint_remains_strictly_loadable(self):
        from search_a import restore_a_config,apply_repaired_control
        old_cfg=config_for_test(search_a_model_version=2,search_a_control_version=2,
                                search_a_separate_optimizer=False)
        old=make_agent(old_cfg)
        saved={k:v for k,v in vars(old_cfg).items() if k.startswith("search_a_")
               and k not in ("search_a_model_version","search_a_control_version")}
        checkpoint=dict(agent_state_dict=old.state_dict(),search_a_version=2,config=saved,step=1000000)
        new_cfg=vars(config_for_test())
        restore_a_config(new_cfg,checkpoint)
        self.assertEqual(new_cfg["search_a_model_version"],2)
        self.assertEqual(new_cfg["search_a_control_version"],2)
        new=make_agent(SimpleNamespace(**new_cfg))
        new.load_state_dict(checkpoint["agent_state_dict"],strict=True)
        apply_repaired_control(new_cfg)
        repaired=make_agent(SimpleNamespace(**new_cfg))
        repaired.load_state_dict(checkpoint["agent_state_dict"],strict=True)
        self.assertEqual(repaired._search_a.control_version,3)
        self.assertEqual(repaired._wm.search_a.model_version,2)

    def test_control_diagnostics_count_actual_actions(self):
        import tools
        episode=dict(action=np.eye(12)[[0,4,4,1]],
                     log_search_decision=[0,1,0,0],log_search_macro_id=[-1,0,0,-1],
                     log_search_completed_length=[0,0,0,2],log_search_interrupt=[0,0,0,1])
        result=tools.search_episode_diagnostics(episode)
        self.assertAlmostEqual(result["action_turn_rate"],2/3)
        self.assertAlmostEqual(result["action_interaction_rate"],1/3)
        self.assertEqual(result["a_interrupt_rate"],1)
        self.assertEqual(result["a_mean_execution_length"],2)


class TrainNotificationTests(unittest.TestCase):
    def test_shell_notifications_preserve_result_arguments_and_signals(self):
        import http.server
        import json
        import os
        import signal
        import subprocess
        import threading
        import time

        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body=self.rfile.read(int(self.headers.get("Content-Length","0"))).decode("utf-8")
                requests.append(dict(path=self.path,body=body,title=self.headers.get("Title"),
                                     priority=self.headers.get("Priority")))
                self.send_response(403 if self.path=="/reject" else 200)
                self.end_headers()
                self.wfile.write(b'{"event":"message"}')
            def log_message(self,*args):
                pass

        server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        script=pathlib.Path(__file__).resolve().parents[1]/"scripts"/"train.sh"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=pathlib.Path(tmp)
                binary=root/"bin"
                binary.mkdir()
                fake=binary/"python"
                fake.write_text(
                    "#!/usr/bin/python3\nimport json,os,pathlib,sys,time\n"
                    "pathlib.Path(os.environ['CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
                    "time.sleep(30) if os.environ.get('FAKE_WAIT') else None\n"
                    "sys.exit(int(os.environ.get('FAKE_TRAIN_STATUS','0')))\n")
                fake.chmod(0o755)
                base=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ["PATH"],
                          LOG_ROOT=str(root/"logs"),A_MODE="full",
                          NTFY_SERVER="http://127.0.0.1:%d"%server.server_port,
                          TRAIN_LOG_FILE="logs/train/unit/example.log")
                for index,(status,topic,rejected) in enumerate(
                        ((0,"unit-training",False),(7,"unit-training",False),
                         (7,"reject",True),(0,"reject",True),(9,"",False))):
                    with self.subTest(status=status,topic=topic):
                        capture=root/("args_%d.json"%index)
                        env=dict(base,NTFY_TOPIC=topic,CAPTURE_ARGS=str(capture),FAKE_TRAIN_STATUS=str(status))
                        previous=len(requests)
                        result=subprocess.run(["sh",str(script),"harvest_sand","--seed","7"],env=env,
                                              capture_output=True,text=True,timeout=10)
                        self.assertEqual(result.returncode,status,result.stderr)
                        args=json.loads(capture.read_text())
                        self.assertEqual(args[-2:],["--seed","7"])
                        self.assertIn("apm_a",args)
                        if not topic:
                            self.assertEqual(len(requests),previous)
                        else:
                            self.assertEqual(len(requests),previous+1)
                            record=requests[-1]
                            self.assertIn("退出码：%d"%status,record["body"])
                            self.assertIn("日志：logs/train/unit/example.log",record["body"])
                            self.assertEqual(record["priority"],"high")
                            self.assertTrue(record["title"].startswith(
                                "Training completed:" if status==0 else "Training stopped:"))
                            if rejected:
                                self.assertIn("notification request failed",result.stderr)
                for sig,expected in ((signal.SIGTERM,143),(signal.SIGINT,130)):
                    capture=root/("signal_args_%d.json"%expected)
                    env=dict(base,NTFY_TOPIC="unit-signal",CAPTURE_ARGS=str(capture),FAKE_WAIT="1")
                    previous=len(requests)
                    proc=subprocess.Popen(["sh",str(script),"harvest_sand"],env=env,
                                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    try:
                        deadline=time.monotonic()+5
                        while not capture.exists() and time.monotonic()<deadline:
                            time.sleep(.02)
                        self.assertTrue(capture.exists())
                        proc.send_signal(sig)
                        out,err=proc.communicate(timeout=5)
                        self.assertEqual(proc.returncode,expected,out+err)
                        self.assertEqual(len(requests),previous+1,out+err)
                        self.assertIn("退出码：%d"%expected,requests[-1]["body"])
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.communicate()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)



class PrefixAndFeedbackTests(unittest.TestCase):
    def test_prefix_targets_and_loss_scale(self):
        head = SearchOutcomeHead(32, hidden=32, version=4)
        features = torch.randn(1, 12, 32, requires_grad=True)
        batch = {k: torch.as_tensor(v) for k, v in replay_batch().items()}
        batch["search_visible_valid"][:] = 1
        batch["search_visible"][:, 5:8] = 1
        with mock.patch.object(head, "forward", wraps=head.forward) as calls:
            loss, metrics, counts, positives = head.training_loss(features, batch)
        lengths = [c.args[1].shape[-2] for c in calls.call_args_list]
        self.assertEqual(lengths, [1, 2, 3, 4, 6])
        for call, k in zip(calls.call_args_list, lengths):
            torch.testing.assert_close(call.args[1][:, 0], batch["action"][:, 1:k+1])
            torch.testing.assert_close(call.args[0], features[:, :-k])
        self.assertAlmostEqual(float(loss), np.mean([
            float(metrics[f"A/H{k}/prediction_loss"]) for k in lengths]), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        head.record_supervision(counts, positives)
        self.assertEqual(float(head.prefix_counts[4].sum()), 0)
        self.assertTrue(head.ready_prefix(3, 1, 0)[3])
        self.assertEqual(float(metrics["A/H3/valid_windows"]), 9)

    def test_prefix_has_no_future_action_input_and_masks_padding(self):
        head = SearchOutcomeHead(16, hidden=16, version=4)
        feat = torch.randn(2, 16)
        actions = F.one_hot(torch.randint(12, (2, 6)), 12).float()
        with mock.patch.object(head.input[0], "forward", wraps=head.input[0].forward) as call:
            predicted = head(feat, actions[:, :3])
        x = call.call_args.args[0]
        torch.testing.assert_close(x[:, -6:], torch.tensor([[1, 1, 1, 0, 0, 0.]]).expand(2, -1))
        self.assertEqual(float(x[:, 16+36:16+72].abs().sum()), 0)
        actions[:, 3:] = torch.flip(actions[:, 3:], (-1,))
        torch.testing.assert_close(predicted["event_logits"], head(feat, actions[:, :3])["event_logits"])

    def test_explicit_migration_does_not_claim_new_prefixes_are_trained(self):
        from search_a import migrate_v3_head
        old, new = SearchOutcomeHead(16, hidden=16, version=3), SearchOutcomeHead(16, hidden=16, version=4)
        old.label_counts[:] = 10000
        report = migrate_v3_head(old, new)
        self.assertTrue(report["warm_start_only"])
        torch.testing.assert_close(new.current_visible[0].weight, old.current_visible[0].weight)
        self.assertFalse(new.ready_prefix(3, 1, 0).any())
        self.assertEqual(float(new.label_counts.sum()), 0)

    def test_untrained_duration_is_rejected(self):
        from search_a import PrefixSearchController
        with self.assertRaises(ValueError):
            PrefixSearchController(horizon=6, commit_steps=5)

    def test_first_discovery_is_not_repeated_reacquisition(self):
        evidence = SearchEvidence("sand")
        evidence.observe(fake_rays(), initial=True)
        first = evidence.observe(fake_rays("sand"))
        evidence.observe(fake_rays("air"))
        second = evidence.observe(fake_rays("sand"))
        self.assertEqual(first["search_first_discovery"], 1)
        self.assertEqual(second["search_first_discovery"], 0)
        self.assertEqual(second["search_discovery"], 1)
        evidence.observe({}, initial=True)
        unknown = evidence.observe(fake_rays("sand"))
        self.assertEqual(unknown["search_first_discovery_valid"], 0)

    def test_default_prefix_really_moves_and_has_bounded_budget(self):
        from search_a import PrefixSearchController, prefix_macro_actions
        seq = prefix_macro_actions(3, "cpu").argmax(-1)
        self.assertEqual(seq[6].tolist(), [3, 6, 6])
        self.assertEqual(seq[7].tolist(), [2, 6, 6])
        c = PrefixSearchController(commit_steps=3, budget_window=20, budget_fraction=.2,
                                   start_step=0, override_prob=1., epsilon=1.,
                                   low_evidence=1.1, protect_interaction=False)
        head = SearchOutcomeHead(16, hidden=16, version=4)
        feat = torch.randn(1, 16)
        actor = F.one_hot(torch.tensor([6]), 12).float()
        state, taken = None, []
        for i in range(80):
            _, state, info = c.choose(head, feat, {"heatmap": torch.zeros(1, 64, 64, 1)},
                actor, state, torch.tensor([i == 0]), 50, True,
                lambda idx: actor[:, None].expand(len(idx), 6, -1))
            taken.append(int(info["log_search_override"]))
            self.assertLessEqual(sum(taken[-20:]), 4)
            self.assertLessEqual(float(info["log_search_completed_length"].max()), 3)

    def test_evaluation_resets_on_timeout_without_real_done(self):
        import tools
        class Env:
            id = "unused"
            def __init__(self, name="episode", duration=2, kind="timeout"):
                self.name, self.duration, self.kind = name, duration, kind
                self.resets = self.steps = self.t = 0
            def observation(self, done=False):
                return dict(image=np.zeros((4, 4, 3), np.uint8),
                            heatmap=np.zeros((4, 4), np.uint8),
                            is_first=self.t == 0, is_last=done, is_terminal=done and self.kind == "terminal",
                            is_zoomed=False, score=0., score_on_zoomed=0.,
                            jumping_steps=0, accumulated_reward=0., is_calculated=False,
                            intrinsic=0.)
            def reset(self):
                self.resets += 1
                self.id = "%s_%d" % (self.name, self.resets)
                self.t = 0
                return lambda: self.observation()
            def step(self, action):
                self.steps += 1
                self.t += 1
                done = self.t >= self.duration
                return lambda: (self.observation(done), 0., done,
                                {"real_done": False, "first_success_step": self.duration,
                                 "success": done and self.kind == "success"})
        env = Env()
        storage = mock.Mock()
        storage.count_data_pairs.return_value = 1  # expose deferred-target bug
        storage.get_and_remove_less_than.return_value = []
        values, saved = {}, []
        logger = SimpleNamespace(step=0, scalar=lambda k, v: values.update({k: v}),
                                 video=lambda *a: None, write=lambda **k: None)
        def policy(obs, done, state):
            return {"action": torch.eye(12)[[6]*len(done)]}, None
        cases = [([env], 3), ([Env("a", 2, "success"), Env("b", 3, "terminal")], 3),
                 ([Env("a"), Env("b"), Env("c")], 1)]
        for envs, count in cases:
            saved.clear()
            with mock.patch.object(tools, "save_episodes",
                                   side_effect=lambda d, eps: saved.extend(eps.keys())):
                tools.simulate(policy, envs, collections.OrderedDict(), ".", logger,
                               storage, 3, .99, is_eval=True, episodes=count)
            self.assertEqual(sum(e.resets for e in envs), count)
            self.assertEqual(len(saved), count)
            self.assertEqual(len(set(saved)), count)
            self.assertEqual(values["eval_episodes"], count)
            self.assertEqual(values["eval_protocol_version"], 5)
        self.assertEqual(env.steps, 6)


class SearchFeedbackLearningTests(unittest.TestCase):
    def config(self, **changes):
        return config_for_test(search_a_model_version=4, search_a_control_version=4,
                               search_a_commit_steps=3, search_c_enabled=True, **changes)

    def dataset(self):
        from search_c import complete_c_fields
        data = replay_batch(time=12)
        episode = complete_c_fields({k: v[0] for k, v in data.items()})
        data = {k: v[None] for k, v in episode.items()}
        data["search_c_version"][:] = 1
        for r in (1, 4, 7, 10):
            data["search_c_decision"][0, r] = 1
            data["search_c_choice"][0, r] = 2
            data["search_c_mask"][0, r] = 1
            data["search_c_context"][0, r, 8+6] = 1
        data["search_visible_valid"][:] = 1
        data["search_first_discovery_valid"][:] = 1
        data["search_first_discovery"][0, 3] = 1
        return data

    def test_semimarkov_discount_true_feedback_and_censoring(self):
        from search_c import build_transitions
        cfg, data = self.config(), self.dataset()
        data["is_last"][0, 8] = 1
        data["is_first"][0, 9] = 1
        records = build_transitions(data, cfg, 3)
        self.assertEqual([(r["start"], r["end"], r["duration"]) for r in records],
                         [(0, 3, 3), (3, 6, 3), (6, 8, 2)])
        self.assertAlmostEqual(records[0]["discount"], cfg.search_c_discount**3)
        self.assertAlmostEqual(records[-1]["discount"], cfg.search_c_discount ** records[-1]["duration"])
        self.assertGreater(records[0]["reward_discovery"], 0)
        self.assertEqual(records[1]["reward_discovery"], 0)
        data["search_first_discovery_valid"][:] = 0
        records = build_transitions(data, cfg, 3)
        self.assertTrue(all(r["reward_discovery"] == 0 for r in records))

    def test_real_rollout_records_replay_train_and_strict_reload(self):
        import tools
        from search_c import check_c_checkpoint
        from evaluate_checkpoints import configure_search_checkpoint
        for enabled_b in (False, True):
            cfg = self.config(search_b_enabled=enabled_b, search_b_lookback=4, search_b_slots=2)
            agent = make_agent(cfg)
            source = replay_batch(time=12)
            state, episode = None, {}
            for t in range(12):
                obs = {k: v[:, t] for k, v in source.items() if k != "action"}
                action, state = agent._policy(obs, state, training=True)
                for name, val in action.items():
                    episode.setdefault(name, []).append(val.detach().cpu().numpy()[0])
            self.assertGreater(sum(episode["search_c_decision"]), 0)
            data = self.dataset()
            # Observation row t+1 stores the policy decision made at observation t.
            for name, values in episode.items():
                if name == "action" or name.startswith("search_c_"):
                    array = np.asarray(values)
                    data[name] = np.concatenate([np.zeros_like(array[:1]), array[:-1]], 0)[None]
            if enabled_b:
                data["search_b_real_first"] = data["is_first"].copy()
            before = agent._search_c.actor[-1].weight.detach().clone()
            post, _, context, _ = agent._wm._train(data, step=24000)
            metrics = agent._train_search_c(post, context.get("search_memory"), data)
            self.assertGreater(metrics["C/transitions"], 0)
            self.assertFalse(torch.equal(before, agent._search_c.actor[-1].weight))
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in agent._search_c.parameters()))
            checkpoint = dict(agent_state_dict=agent.state_dict(),
                              optims_state_dict=tools.recursively_collect_optim_state_dict(agent),
                              config=vars(cfg), step=24000, search_a_version=4,
                              search_b_version=2 if enabled_b else None, search_c_version=1)
            check_c_checkpoint(checkpoint, True)
            restored_cfg = vars(config_for_test())
            configure_search_checkpoint(restored_cfg, checkpoint)
            restored = make_agent(SimpleNamespace(**restored_cfg))
            restored.load_state_dict(checkpoint["agent_state_dict"], strict=True)
            tools.recursively_load_optim_state_dict(restored, checkpoint["optims_state_dict"])
            self.assertEqual(int(restored._search_c.updates), int(agent._search_c.updates))

    def test_offline_and_random_prefill_missing_C_is_unknown(self):
        import tools
        data = replay_batch()
        ep = {k: v[0] for k, v in data.items()}
        ep = {k: v for k, v in ep.items() if not k.startswith("search_c_")}
        source = collections.OrderedDict(old=ep, new=self.dataset_episode())
        batch = next(tools.from_generator(tools.sample_episodes(source, 20), 2))
        self.assertEqual(batch["search_c_context"].shape, (2, 20, 20))
        self.assertTrue(np.isfinite(batch["search_c_context"]).all())

    def dataset_episode(self):
        return {k: v[0] for k, v in self.dataset().items()}

    def test_C_ablation_masks_A_visibility_and_has_private_rng(self):
        from search_c import SearchBehavior
        cfg = self.config(search_c_use_predictions=False)
        c = SearchBehavior(16, cfg)
        state = torch.randn(2, 16)
        a, b = torch.rand(2, 8, 11), torch.rand(2, 8, 11)
        b[..., 8:] = a[..., 8:]  # candidate metadata is kept in this ablation
        context = torch.randn(2, 20)
        other = context.clone()
        other[:, [1, 2, 5]] += 10
        torch.testing.assert_close(c._input(state, a, context), c._input(state, b, other))
        rng = torch.random.get_rng_state()
        c.choose(state, a, context, torch.ones(2, 8), True)
        torch.testing.assert_close(rng, torch.random.get_rng_state())
        _, probability = c.choose(state, a, context, torch.ones(2, 8), False)
        torch.testing.assert_close(probability, torch.ones(2))

    def test_C_runtime_ablation_keeps_strict_weights_and_is_explicit(self):
        from search_c import apply_c_control_mode
        from evaluate_checkpoints import apply_a_control_mode
        agent = make_agent(self.config())
        saved = agent.state_dict()
        config = vars(self.config()).copy()
        with self.assertRaises(ValueError):
            apply_a_control_mode(config, "off")
        apply_c_control_mode(config, "off")
        apply_a_control_mode(config, "off")
        restored = make_agent(SimpleNamespace(**config))
        restored.load_state_dict(saved, strict=True)
        self.assertIsNone(restored._search_a)
        config = vars(self.config()).copy()
        apply_c_control_mode(config, "no_predictions")
        restored = make_agent(SimpleNamespace(**config))
        restored.load_state_dict(saved, strict=True)
        self.assertFalse(restored._search_c.config.search_c_use_predictions)
        with self.assertRaises(ValueError):
            apply_c_control_mode({"search_c_enabled": False}, "off")

    def test_C_without_A_and_B_works(self):
        for use_b in (False, True):
            agent = make_agent(self.config(search_a_enabled=False, search_b_enabled=use_b,
                                           search_b_lookback=4, search_b_slots=2))
            data = self.dataset()
            obs = {k: v[:, 0] for k, v in data.items() if k != "action"}
            action, _ = agent._policy(obs, None, training=True)
            self.assertEqual(action["action"].shape, (1, 12))
            self.assertEqual(float(action["search_c_decision"]), 1)
            self.assertIsNone(agent._wm.search_a)
            if use_b:
                data["search_b_real_first"] = data["is_first"].copy()
            post, _, context, _ = agent._wm._train(data, step=24000)
            metrics = agent._train_search_c(post, context.get("search_memory"), data)
            self.assertGreater(metrics["C/transitions"], 0)
            self.assertTrue(np.isfinite(metrics["search_c_value_loss"]))


def audit_replay(argv):
    """Bounded development audit; freezes the actual RSSM, never starts a game.

    Historical RSSM may have seen every episode. Held-out refers only to this
    new head fitting, not whole-system generalization or counterfactual truth.
    """
    import argparse
    import gc
    import hashlib
    import json
    import time
    import gym
    import networks
    from search_a import migrate_v3_head
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--evaluate-head", type=pathlib.Path,
                        help="Score an existing development head; no parameter updates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1028)
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 3000:
        parser.error("The bounded audit allows 1..3000 updates; not a training launcher")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    # mmap avoids holding the original optimizer tensors in RAM.
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", mmap=True)
    config = SimpleNamespace(**checkpoint["config"])
    if checkpoint.get("search_a_version") != 3 or config.search_b_enabled:
        raise ValueError("This migration audit expects an A-only v3 reference")
    config.device = args.device
    config.num_actions = 12
    state = checkpoint["agent_state_dict"]
    digest = hashlib.sha256()
    for prefix in ("_wm.encoder.", "_wm.dynamics.", "_wm.search_a."):
        for key in sorted(k for k in state if k.startswith(prefix)):
            digest.update(key.encode())
            digest.update(state[key].contiguous().numpy().tobytes())
    identity = digest.hexdigest()
    shapes = {"image": (64, 64, 3), "heatmap": (64, 64, 1), "obs_reward": (1,)}
    encoder = networks.MultiEncoder(shapes, **config.encoder)
    dynamics = networks.RSSM(config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_rec_depth, config.dyn_discrete, config.act, config.norm,
        config.dyn_mean_act, config.dyn_std_act, config.dyn_min_std,
        config.unimix_ratio, config.initial, config.num_actions, encoder.outdim, config.device)
    for name, module in (("encoder", encoder), ("dynamics", dynamics)):
        prefix = "_wm."+name+"."
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)
        module.to(args.device).eval().requires_grad_(False)
    feature_dim = config.dyn_deter + config.dyn_stoch * config.dyn_discrete
    head_args = dict(feature_dim=feature_dim, horizon=config.search_a_horizon,
                    hidden=config.search_a_hidden, future_weight=config.search_a_future_weight,
                    event_weight=config.search_a_event_weight,
                    visible_weight=config.search_a_visible_weight)
    old = SearchOutcomeHead(**head_args, version=3)
    prefix = "_wm.search_a."
    old.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    old.to(args.device).eval().requires_grad_(False)
    model = SearchOutcomeHead(**head_args, version=4).to(args.device)
    migration = migrate_v3_head(old, model)

    class StateOnlyHead(SearchOutcomeHead):
        def forward(self, feat, actions, memory=None):
            return super().forward(feat, torch.zeros_like(actions), memory)
    state_only = StateOnlyHead(**head_args, version=4).to(args.device)
    state_only.load_state_dict(model.state_dict(), strict=True)
    step = int(checkpoint.get("step", 0))
    del state, checkpoint
    gc.collect()
    episodes, carry = [], None
    for index, path in enumerate(sorted(args.replay.glob("*.npz"))):
        replay_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_path = args.output / (path.stem + ".features.npz")
        metadata = dict(reference_modules_sha256=identity, replay_sha256=replay_digest,
                        posterior_sampling_seed=args.seed+index, format_version=1)
        meta_path = cache_path.with_suffix(".json")
        cached = cache_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == metadata
        with np.load(path) as z:
            episode = {k: z[k] for k in z.files if not k.startswith("log_")}
        if not cached:
            torch.manual_seed(args.seed+index)
            features, carry = [], None
            with torch.no_grad():
                for start in range(0, len(episode["reward"]), 32):
                    end = start+32
                    obs = {k: torch.as_tensor(v[start:end][None], device=args.device).float()
                           for k, v in episode.items() if k in ("image", "heatmap", "obs_reward", "is_first")}
                    obs["image"] = obs["image"]/255.
                    obs["heatmap"] = obs["heatmap"].unsqueeze(-1)/255.
                    action = F.pad(torch.as_tensor(episode["action"][start:end][None],
                                                   device=args.device).float(), (0, 1))
                    post, _ = dynamics.observe(encoder(obs), action, obs["is_first"], state=carry)
                    features.append(dynamics.get_feat(post).cpu().numpy()[0].astype(np.float16))
                    carry = {k: v[:, -1] for k, v in post.items()}
            np.savez(cache_path, features=np.concatenate(features))
            meta_path.write_text(json.dumps(metadata, indent=2)+"\n")
        with np.load(cache_path) as z:
            features = z["features"]
        slim = {k: v for k, v in episode.items()
                if k in ("action", "is_first", "is_last", "is_terminal") or k.startswith("search_")}
        episodes.append(dict(name=path.name, features=features, data=slim,
                             success=bool(np.asarray(episode.get("success", [])).any())))
        print("Cached", index+1, path.name, features.shape, flush=True)
    del encoder, dynamics, episode, carry
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if len(episodes) < 8 and args.evaluate_head is None:
        raise ValueError("At least 8 complete episodes required for a development split")
    train, heldout = [], []
    for success in (False, True):
        group = [e for e in episodes if e["success"] == success]
        rng.shuffle(group)
        split = max(1, len(group)//3) if len(group) >= 3 else 0
        heldout.extend(group[:split])
        train.extend(group[split:])
    if not heldout and args.evaluate_head is None:
        raise ValueError("No episode-level held-out set")

    def tensors(ep, start, end):
        feat = torch.as_tensor(ep["features"][None, start:end], device=args.device).float()
        data = {k: torch.as_tensor(v[None, start:end], device=args.device)
                for k, v in ep["data"].items()}
        return feat, data

    def evaluate(head, selection, k, shuffled=False):
        # Accumulate numerators and denominators, never average rare labels
        # with unknown windows. A permutation is a diagnostic, not branch truth.
        accum, weights = {}, {}
        gen = torch.Generator(device=args.device).manual_seed(args.seed+100)
        original_forward = head.forward
        def permuted(feat, actions, memory=None):
            flat = actions.reshape(-1, *actions.shape[-2:])
            idx = torch.randperm(len(flat), device=flat.device, generator=gen)
            return original_forward(feat, flat[idx].reshape_as(actions), memory)
        if shuffled:
            head.forward = permuted
        try:
            with torch.no_grad():
                for ep in selection:
                    for start in range(0, len(ep["features"])-k, 128-k):
                        feat, data = tensors(ep, start, start+128)
                        _, values, _, _ = head._loss_at_horizon(feat, data, None, k)
                        for name, value in values.items():
                            metric = name[2:]
                            if metric in ("future_loss",):
                                count = float(values["A/valid_windows"])
                            else:
                                count = 0
                                for suffix in ("positive_rate", "positive_prediction", "positive_recall",
                                               "negative_prediction", "loss", "brier"):
                                    if metric.endswith("_"+suffix):
                                        event = metric[:-len(suffix)-1]
                                        count = float(values.get("A/"+event+"_valid", 0))
                                        positives = float(values.get("A/"+event+"_positive_count", 0))
                                        if suffix in ("positive_prediction", "positive_recall"):
                                            count = positives
                                        elif suffix == "negative_prediction":
                                            count -= positives
                                        break
                            if count:
                                accum[metric] = accum.get(metric, 0.) + float(value)*count
                                weights[metric] = weights.get(metric, 0.)+count
        finally:
            head.forward = original_forward
        return {k: dict(value=accum[k]/weights[k], count=weights[k]) for k in accum}

    if args.evaluate_head is not None:
        saved = torch.load(str(args.evaluate_head), map_location=args.device)
        if saved["reference_modules_sha256"] != identity:
            raise ValueError("Development head and frozen RSSM identity differ")
        model.load_state_dict(saved["head"], strict=True)
        report = dict(mode="challenge_evaluation_only", updates=0,
            episodes=[e["name"] for e in episodes], reference_modules_sha256=identity,
            after={str(k): evaluate(model, episodes, k) for k in (1, 2, 3, 4, 6)},
            shuffled_actions={str(k): evaluate(model, episodes, k, shuffled=True) for k in (1, 2, 3, 4, 6)},
            old_v3_h6=evaluate(old, episodes, old.horizon),
            caveats=["Selected development challenge cases, not natural-frequency validation.",
                     "No refitting or test-driven parameter selection performed."])
        (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
        print("Challenge evaluation saved:", args.output/"audit.json", flush=True)
        return

    before = {str(k): evaluate(model, heldout, k) for k in (1, 2, 3, 4, 6)}
    old_metrics = evaluate(old, heldout, old.horizon)
    opt = torch.optim.Adam(model.parameters(), lr=1.e-4)
    noact_opt = torch.optim.Adam(state_only.parameters(), lr=1.e-4)
    trainable = [ep for ep in train if len(ep["features"]) >= 32]
    start_time = time.time()
    for update in range(args.updates):
        # Identical sampled windows and update count for both heads.
        picks = [trainable[rng.randint(len(trainable))] for _ in range(4)]
        windows = [tensors(ep, offset, offset+32)
                   for ep in picks for offset in [rng.randint(len(ep["features"])-31)]]
        feat = torch.cat([w[0] for w in windows], 0)
        data = {k: torch.cat([w[1][k] for w in windows], 0) for k in windows[0][1]}
        for head, optimizer in ((model, opt), (state_only, noact_opt)):
            optimizer.zero_grad(set_to_none=True)
            loss, _, counts, positives = head.training_loss(feat, data)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite offline audit loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 100.)
            optimizer.step()
            head.record_supervision(counts, positives)
        if (update+1) % 100 == 0:
            print("Offline updates", update+1, "seconds", round(time.time()-start_time), flush=True)
    after, shuffled, no_action, constant = {}, {}, {}, {}
    for k in (1, 2, 3, 4, 6):
        key = str(k)
        after[key] = evaluate(model, heldout, k)
        shuffled[key] = evaluate(model, heldout, k, shuffled=True)
        no_action[key] = evaluate(state_only, heldout, k)
        rates = evaluate(model, train, k)
        constant[key] = {}
        for event in ("new", "discovery", "revisit", "future_visible"):
            name = event+"_positive_rate"
            if name in rates and name in after[key]:
                q, p = rates[name]["value"], after[key][name]["value"]
                constant[key][event+"_brier"] = (1-p)*q*q+p*(1-q)**2
    report = dict(reference_modules_sha256=identity, reference_checkpoint_step=step,
        training_episodes=[e["name"] for e in train], heldout_episodes=[e["name"] for e in heldout],
        updates_per_head=args.updates, seed=args.seed, migration=migration,
        before=before, after=after, shuffled_actions=shuffled, no_action_head=no_action,
        train_frequency_baseline=constant, old_v3_h6=old_metrics,
        caveats=["Development split only: historical RSSM/old head may have seen these episodes.",
                 "Episodes were deliberately selected to include successes and failures, not a success-rate estimate.",
                 "Unexecuted candidate branches have no truth in this replay.",
                 "Constant and state-only comparisons diagnose dependence, not causal proof.",
                 "No environment interaction or original Actor/world-model update."])
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    torch.save(dict(head=model.state_dict(), model_version=4, config=head_args,
                    reference_modules_sha256=identity, audit_only=True),
               args.output/"a_v4_development_head.pt")
    print("Development audit saved:", args.output/"audit.json", flush=True)

class FullPlanTests(unittest.TestCase):
    def cfg(self, **kw):
        values = dict(search_a_model_version=5, search_a_control_version=4, search_a_horizon=8,
            search_a_commit_steps=3, search_b_enabled=True, search_b_model_version=3,
            search_b_dim=16, search_b_hidden=32, search_b_slots=2, search_b_lookback=4,
            search_c_enabled=True, search_c_model_version=2, search_c_imagined_weight=.25,
            search_c_actor_bonus=.25, search_c_imagined_starts=2,
            search_a_min_labels=1, search_a_min_positives=0)
        values.update(kw)
        return config_for_test(**values)

    def dataset(self, cfg):
        from expr import make_dataset
        data = SearchFeedbackLearningTests().dataset()
        data["search_c_version"][:] = 2
        return next(make_dataset(__import__("collections").OrderedDict(one={k:v[0] for k,v in data.items()}), cfg))

    def test_full_joint_training_imagination_memory_and_checkpoint(self):
        from evaluate_checkpoints import configure_search_checkpoint
        cfg = self.cfg()
        agent = make_agent(cfg)
        data = self.dataset(cfg)
        before = agent._wm.search_b.latent_adapter.weight.detach().clone()
        agent._train(data)
        agent._train(data)
        self.assertTrue(any(value > 0 for value in agent._metrics["C/imagined_transitions"]))
        self.assertTrue(np.isfinite(agent._metrics["C/task_actor_search_bonus"]).all())
        self.assertGreater(float((agent._wm.search_b.latent_adapter.weight-before).abs().sum()), 0)
        self.assertTrue(torch.isfinite(agent._wm.search_a.prefix_counts).all())
        self.assertEqual(tuple(agent._wm.search_a.prefix_counts.shape), (8,12))
        ckpt = dict(agent_state_dict=agent.state_dict(), config=vars(cfg).copy(), step=10,
                    search_a_version=5, search_b_version=3, search_c_version=2)
        restored = vars(config_for_test()).copy()
        configure_search_checkpoint(restored, ckpt)
        clone = make_agent(SimpleNamespace(**restored))
        clone.load_state_dict(ckpt["agent_state_dict"], strict=True)
        batch = replay_batch()
        obs = {k:v[:,0] for k,v in batch.items() if k != "action"}
        with torch.no_grad():
            output, state = clone._policy(obs, None, False)
        self.assertEqual(output["action"].shape, (1,12))
        self.assertEqual(output["search_c_version"].item(), 2)

    def test_rssm_mixed_boolean_resets_match_numeric_without_mutation(self):
        agent=make_agent(self.cfg())
        dynamics=agent._wm.dynamics
        b=4
        previous=dynamics.initial(b)
        previous={key:value.detach().clone()+.1 for key,value in previous.items()}
        action=F.pad(F.one_hot(torch.tensor([2,3,4,5]),12).float(),(0,1))
        original_action=action.clone()
        original_state={key:value.clone() for key,value in previous.items()}
        embed=torch.randn(b,agent._wm.embed_size)
        first=torch.tensor([True,False,True,False])
        torch.manual_seed(42)
        numeric=dynamics.obs_step({k:v.clone() for k,v in previous.items()},action.clone(),embed,first.float(),sample=False)
        torch.manual_seed(42)
        boolean=dynamics.obs_step(previous,action,embed,first,sample=False)
        for expected,actual in zip(numeric,boolean):
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key])
        torch.testing.assert_close(action,original_action)
        for key in previous:
            torch.testing.assert_close(previous[key],original_state[key])

    def test_formal_batch_and_history_with_mixed_lengths_train_all_modules(self):
        cfg=self.cfg(batch_size=16,batch_length=32,search_b_lookback=128)
        agent=make_agent(cfg)
        data=self.dataset(cfg)
        valid=torch.as_tensor(data["search_b_history_valid"]).bool()
        prior=torch.cat((torch.zeros_like(valid[:,:1]),valid[:,:-1]),1)
        first=~valid | ~prior
        count=first.sum(0)
        self.assertTrue(((count>0)&(count<16)).any())  # exact server failure condition
        before=agent._wm.search_b.latent_adapter.weight.detach().clone()
        agent._train(data)
        self.assertTrue(all(np.isfinite(v).all() for values in agent._metrics.values()
                            for v in (values if isinstance(values,list) else [values])))
        self.assertGreater(agent._metrics["B/transition_valid"][-1],0)
        self.assertGreater(agent._metrics["C/transitions"][-1],0)
        self.assertGreater(float((before-agent._wm.search_b.latent_adapter.weight).abs().sum()),0)
        empty=copy.deepcopy(data)
        empty["search_b_history_valid"][:]=0
        features=agent._wm.memory_history_features(empty)
        self.assertEqual(features.shape[:2],(16,128))
        self.assertTrue(torch.isfinite(features).all())

    def test_memory_off_is_preserved_in_full_imagined_search(self):
        from unittest import mock
        cfg=self.cfg(search_b_control=False)
        agent=make_agent(cfg)
        data=self.dataset(cfg)
        post,_,context,_=agent._wm._train(data)
        with mock.patch.object(agent._wm.search_b,"imagine_step",
                side_effect=AssertionError("B-off imagination must keep zero memory")):
            metrics=agent._train_search_c(post,context["search_memory"],data)
        self.assertGreater(metrics["C/imagined_transitions"],0)

    def test_return_targets_unknown_masks_terminal_and_reliability_gradient(self):
        from search_feedback import window_targets, FullSearchOutcomeHead
        cfg = self.cfg(search_b_enabled=False)
        raw = SearchFeedbackLearningTests().dataset()
        data = {k:torch.tensor(v) for k,v in raw.items()}
        target, known, first, fk = window_targets(data, 3, cfg)
        self.assertAlmostEqual(target[0,0,1].item(), cfg.search_c_discount**2, places=5)
        data["search_first_discovery_valid"][:] = 0
        _, known, _, fk = window_targets(data, 3, cfg)
        self.assertFalse(known[...,1].any())
        self.assertFalse(fk.any())
        head = FullSearchOutcomeHead(16, 8, 32, version=5, feedback_config=cfg)
        features = torch.randn(1,12,16)
        loss, metrics, counts, positives = head.training_loss(features, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(head.return_head.weight.grad.abs().sum()), 0)
        head.zero_grad()
        result = head(features[:,:1], torch.nn.functional.one_hot(torch.zeros(1,1,3,dtype=torch.long),12).float())
        result["reliability_logits"].sum().backward()
        self.assertIsNone(head.input[0].weight.grad)
        self.assertIsNotNone(head.reliability[0].weight.grad)
        head.record_supervision(counts, positives)
        self.assertFalse(head.return_support(3)[1])
        projected = head.project_returns({"returns": torch.full((2,6), 100.)}, 3)
        self.assertTrue((projected[:,4] < 0).all())  # moving cannot mint a cost bonus
        self.assertTrue((projected[:,3] <= 0).all())
        self.assertTrue((projected[:,1] <= cfg.search_c_discovery_weight).all())

    def test_terminal_differs_from_truncation(self):
        from search_c import build_transitions
        data = SearchFeedbackLearningTests().dataset()
        data["is_last"][0,8] = 1
        cfg = self.cfg()
        rec = build_transitions(data,cfg,3)
        self.assertGreater(rec[-1]["discount"],0)
        data["is_terminal"][0,8] = 1
        rec = build_transitions(data,cfg,3)
        self.assertEqual(rec[-1]["discount"],0)

    def test_actor_bonus_alignment_masks_long_jumps_and_excludes_task_reward(self):
        from search_feedback import actor_search_bonus, FullSearchOutcomeHead
        cfg = self.cfg(search_b_enabled=False, search_c_step_cost=0., search_c_reverse_cost=0.)
        head = FullSearchOutcomeHead(16,8,32,version=5,feedback_config=cfg)
        head.prefix_counts.fill_(100)
        head.prefix_positives.fill_(10)
        with torch.no_grad():
            head.return_head.weight.zero_()
            head.return_head.bias.zero_()
            head.return_head.bias[0] = 5
        features=torch.randn(4,2,16)
        actions=torch.nn.functional.one_hot(torch.zeros(4,2,dtype=torch.long),12).float()
        jumps=torch.zeros(4,2)
        bonus,_=actor_search_bonus(head,features,actions,None,jumps,cfg)
        self.assertEqual(float(bonus.abs().sum()),0)
        with torch.no_grad():
            head.return_head.bias[2]=.1
        jumps[1]=1
        bonus,_=actor_search_bonus(head,features,actions,None,jumps,cfg)
        self.assertEqual(float(bonus[0].sum()),0)
        self.assertEqual(float(bonus[2].sum()),0)
        self.assertGreater(float(bonus[1].sum()),0)

    def test_branch_comparison_rejects_same_seed_different_scene(self):
        from search_branches import compare_branch_scenes
        base=dict(world_seed=1,mission_seed=1,java_launch_seed=1,terrain_sha256="x",rays_sha256="r",
            inventory={},life_stats={},policy_observation_keys=["image"],
            image_sha256="i",heatmap_sha256="h",
            location_stats=dict(pos=[0,1,0],pitch=0,yaw=0,biome_id=1,is_raining=False,sea_level=64))
        self.assertTrue(compare_branch_scenes(base,copy.deepcopy(base))["match"])
        changed=copy.deepcopy(base);changed["image_sha256"]="other"
        self.assertFalse(compare_branch_scenes(base,changed)["match"])
        changed=copy.deepcopy(base);changed["location_stats"]["pos"][0]=2
        self.assertFalse(compare_branch_scenes(base,changed)["match"])
        prefix={"image":np.zeros((3,64,64,3),np.uint8),"heatmap":np.zeros((3,64,64),np.uint8),
                "action":np.zeros((3,12),np.float32)}
        noise={k:v.copy() for k,v in prefix.items()}
        noise["image"][:]=1
        self.assertTrue(compare_branch_scenes(base,base,prefix,noise)["match"])
        noise["image"][:]=20
        self.assertFalse(compare_branch_scenes(base,base,prefix,noise)["match"])
        different=copy.deepcopy(base);different["rays_sha256"]="different"
        self.assertFalse(compare_branch_scenes(base,different,prefix,prefix)["match"])

    def test_real_branch_budget_replay_and_training_with_native_heatmap_shape(self):
        from search_branches import collect_real_branches
        from expr import make_dataset
        from unittest import mock
        import tempfile
        from collections import OrderedDict
        cfg=self.cfg()
        agent=make_agent(cfg)
        initial_step=agent._step
        raw=replay_batch(12)
        class Env:
            def reset(self):
                self.t=0
                return {k:v[0,0].copy() for k,v in raw.items() if k!="action"}
            def step(self,action):
                self.t+=1
                return ({k:v[0,self.t].copy() for k,v in raw.items() if k!="action"},
                        0.,False,{"real_done":False,"success":False})
            def close(self): pass
        scene=dict(world_seed=1,mission_seed=1,java_launch_seed=1,terrain_sha256="x",rays_sha256="r",
            inventory={},life_stats={},policy_observation_keys=["image"],
            image_sha256="i",heatmap_sha256="h",
            location_stats=dict(pos=[0,1,0],pitch=0,yaw=0,biome_id=1,is_raining=False,sea_level=64))
        with tempfile.TemporaryDirectory() as tmp:
            cfg.logdir=tmp;cfg.traindir=str(pathlib.Path(tmp)/"train_eps")
            cfg.steps=initial_step+10;cfg.search_branch_count=2;cfg.search_branch_prefix=2
            replay=OrderedDict()
            logger=SimpleNamespace(step=initial_step,scalar=lambda *a:None,write=lambda:None)
            with mock.patch("search_branches.make_branch_env",side_effect=lambda *a:(Env(),None)), mock.patch(
                    "search_branches.scene_record",return_value=scene):
                used=collect_real_branches(agent,cfg,replay,logger,1)
            self.assertEqual(used,10)
            self.assertEqual(agent._step,cfg.steps)
            self.assertEqual(len(replay),2)
            for episode in replay.values():
                self.assertEqual(set(len(v) for v in episode.values()),{6})
                self.assertTrue(episode["search_branch_verified"].all())
                self.assertEqual(episode["search_c_decision"].sum(),1)
                self.assertFalse(episode["is_terminal"][-1])
                self.assertTrue(episode["is_last"][-1])
            agent._train(next(make_dataset(replay,cfg)))
            self.assertGreater(agent._metrics["C/transitions"][-1],0)
            self.assertTrue((pathlib.Path(tmp)/"branches/000001/feedback.json").exists())

    def test_full_presets_and_default_launcher(self):
        import ruamel.yaml as yaml
        root=pathlib.Path(__file__).resolve().parents[1]
        config=yaml.YAML(typ="safe").load((root/"configs.yaml").read_text())["apm_full"]
        self.assertEqual([config["search_"+x+"_model_version"] for x in ("a","b","c")],[5,3,2])
        self.assertTrue(config["search_branch_enabled"])
        text=(root/"scripts/train.sh").read_text()
        self.assertIn('MODE="${A_MODE:-vnext}"',text)
        self.assertIn('vnext) set -- --configs minedojo apm_vnext',text)
        self.assertIn('abc) set -- --configs minedojo apm_a apm_full',text)

class TrainingEntryRecoveryTests(unittest.TestCase):
    def config(self, root, **kw):
        values=dict(logdir=str(root/"out"),task="minedojo_harvest_sand",target_item="sand",
                    episode_max_steps_override=24,eval_episode_num=0,video_pred_log=False,
                    search_branch_enabled=False,steps=30)
        values.update(kw)
        return FullPlanTests().cfg(**values)

    def environment(self):
        import gym
        return SimpleNamespace(
            observation_space=gym.spaces.Dict({
                "image":gym.spaces.Box(0,255,(64,64,3),np.uint8),
                "heatmap":gym.spaces.Box(0,255,(64,64,1),np.uint8)}),
            action_space=gym.spaces.Discrete(12),close=mock.Mock())

    def test_main_resume_skips_random_prefill_and_restores_before_sampling(self):
        import expr,tools
        from training_state import save_training_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);source=root/"source";(source/"train_eps").mkdir(parents=True)
            cfg=self.config(root,logdir=str(source))
            cfg.traindir=source/"train_eps";cfg.evaldir=source/"eval_eps"
            original=make_agent(cfg)
            original._step=24
            episode={key:value[0] for key,value in SearchFeedbackLearningTests().dataset().items()}
            tools.save_episodes(cfg.traindir,{"fixture":episode})
            save_training_checkpoint(original,cfg,100000)
            (source/"metrics.jsonl").write_text('{"step":27}\n')
            saved=(source/"latest.pt").read_bytes()
            cfg=self.config(root,resume_from=str(source))
            env=self.environment()
            def advance(agent,envs,cache,directory,logger,*args,**kwargs):
                self.assertTrue(kwargs["is_training"])
                self.assertFalse(agent._should_pretrain._once)
                self.assertEqual(agent._step,27)
                self.assertEqual(kwargs["steps"],3)
                self.assertGreater(len(cache),0)
                torch.testing.assert_close(agent._wm.search_b.latent_adapter.weight,
                                           original._wm.search_b.latent_adapter.weight)
                agent._step+=3;logger.step=agent._step
                return None
            with mock.patch.object(expr,"make_env",return_value=env),mock.patch.object(
                    tools,"simulate",side_effect=advance) as calls:
                expr.main(cfg)
            self.assertEqual(calls.call_count,1)
            self.assertEqual(saved,(source/"latest.pt").read_bytes())
            resumed=torch.load(pathlib.Path(cfg.logdir)/"latest.pt",map_location="cpu")
            self.assertEqual(resumed["step"],30)
            self.assertTrue((pathlib.Path(cfg.logdir)/"resume.json").exists())
            env.close.assert_called_once()

    def test_fatal_reset_saves_current_state_and_preserves_original_exception(self):
        import expr,tools
        from envs.recovery import ResetRecoveryError
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);replay=root/"replay";replay.mkdir()
            episode={key:value[0] for key,value in SearchFeedbackLearningTests().dataset().items()}
            tools.save_episodes(replay,{"fixture":episode})
            cfg=self.config(root,offline_traindir=str(replay),steps=10)
            env=self.environment()
            def fail(agent,*args,**kwargs):
                agent._step=3
                raise ResetRecoveryError("injected reset timeout")
            with mock.patch.object(expr,"make_env",return_value=env),mock.patch.object(
                    tools,"simulate",side_effect=fail):
                with self.assertRaisesRegex(ResetRecoveryError,"injected reset timeout"):
                    expr.main(cfg)
            emergency=pathlib.Path(cfg.logdir)/"interrupted.pt"
            self.assertTrue(emergency.exists())
            saved=torch.load(emergency,map_location="cpu")
            self.assertEqual(saved["step"],3)
            self.assertIn(str(replay.resolve()),saved["replay_sources"])
            self.assertFalse((pathlib.Path(cfg.logdir)/"latest.pt").exists())
            env.close.assert_called_once()


def audit_startup_replay(argv):
    """Reproduce server startup locally with real Replay, batch=16 and history=128."""
    import argparse, gc, json, re, gym
    import ruamel.yaml as yaml
    import tools
    from expr import LS_Imagine, make_dataset
    from models import WorldModel
    parser=argparse.ArgumentParser()
    parser.add_argument("--replay",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--updates",type=int,default=3)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a fresh startup audit output directory")
    if args.updates < 1:
        raise ValueError("Startup audit needs at least one optimization step")
    args.output.mkdir(parents=True)
    presets=yaml.YAML(typ="safe").load((pathlib.Path(__file__).resolve().parents[1]/"configs.yaml").read_text())
    settings={}
    def merge(dst,src):
        for key,value in src.items():
            if isinstance(value,dict) and key in dst:
                merge(dst[key],value)
            else:
                dst[key]=copy.deepcopy(value)
    for name in ("defaults","minedojo","apm_a","apm_full"):
        merge(settings,presets[name])
    settings.update(device=args.device,num_actions=12,episode_max_steps=1000,compile=False,
                    logdir=str(args.output),use_wandb=False,log_all_metrics=False)
    formal=SimpleNamespace(**settings)
    replay=tools.load_episodes(args.replay)
    if not replay:
        raise ValueError("Replay directory has no readable episodes")
    dataset=make_dataset(replay,formal)
    data=next(dataset)
    valid=np.asarray(data["search_b_history_valid"],bool)
    first=~valid | ~np.concatenate((np.zeros_like(valid[:,:1]),valid[:,:-1]),axis=1)
    mixed=((first.sum(0)>0)&(first.sum(0)<formal.batch_size))
    assert mixed.any(),"Fixture did not cover the server's mixed reset condition"
    sample=next(iter(replay.values()))
    spaces={"image":gym.spaces.Box(0,255,(64,64,3),np.uint8),
            "heatmap":gym.spaces.Box(0,255,(64,64,1),np.uint8)}
    for key,value in sample.items():
        if re.match(formal.encoder["mlp_keys"],key) and not key.startswith("search_"):
            spaces[key]=gym.spaces.Box(-np.inf,np.inf,np.shape(value)[1:],np.float32)
    obs_space=gym.spaces.Dict(spaces)
    act_space=gym.spaces.Discrete(12)
    tools.set_seed_everywhere(formal.seed)
    print("Checking formal RSSM/encoder history reconstruction:",formal.batch_size,formal.search_b_lookback,
          "mixed reset timesteps",int(mixed.sum()),flush=True)
    wm=WorldModel(obs_space,act_space,2500,formal).to(args.device)
    wm.requires_grad_(False)
    features=wm.memory_history_features(data)
    assert torch.isfinite(features).all()
    assert features.shape==(16,128,5120),features.shape
    report=dict(source_episodes=len(replay),batch_size=16,batch_length=32,history=128,
                mixed_reset_timesteps=int(mixed.sum()),formal_history_shape=list(features.shape),
                device=args.device,formal_history_pass=True,real_environment_steps=0)
    print("Formal 5120-D RSSM history PASS",flush=True)
    del features,wm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Preserve formal batch/sequence/history and B dimensions; reduce core width
    # for bounded backpropagation on the local 6 GB GPU.
    merge(settings,presets["apm_a_smoke"])
    settings.update(batch_size=16,batch_length=32,search_b_lookback=128,imag_horizon=15,
                    max_imag_sequences_num=1024,device=args.device)
    cfg=SimpleNamespace(**settings)
    logger=tools.Logger(cfg,args.output,2500)
    agent=LS_Imagine(obs_space,act_space,cfg,logger,None).to(args.device)
    agent.requires_grad_(False)
    before=agent._wm.search_b.latent_adapter.weight.detach().clone()
    for update in range(args.updates):
        current=data if update==0 else next(dataset)
        agent._train(current)
        print("Startup GPU optimization",update+1,"PASS",flush=True)
    assert all(float(v)==1. for v in agent._metrics["model_update_applied"])
    assert all(float(v)==1. for v in agent._metrics["search_a_update_applied"])
    assert all(float(v)==1. for v in agent._metrics["actor_update_applied"])
    assert all(float(v)==1. for v in agent._metrics["value_update_applied"])
    assert float((before-agent._wm.search_b.latent_adapter.weight).abs().sum())>0
    obs={key:np.asarray(value)[:,0] for key,value in data.items()
         if not key.startswith(("search_b_","log_")) and key!="action"}
    with torch.no_grad():
        output,_=agent._policy(obs,None,False)
    assert output["action"].shape==(16,12)
    assert torch.isfinite(output["action"]).all()
    for values in agent._metrics.values():
        assert all(np.isfinite(v).all() for v in (values if isinstance(values,list) else [values]))
    report.update(training_updates=args.updates,small_model_joint_training_pass=True,
                  actor_batch_shape=list(output["action"].shape))
    agent.flush_metrics()
    torch.save(dict(agent_state_dict=agent.state_dict(),step=2500,config=vars(cfg),
                    search_a_version=5,search_b_version=3,search_c_version=2,
                    audit_only=True),args.output/"startup.pt")
    logger.finish()
    (args.output/"startup_audit.json").write_text(json.dumps(report,indent=2))
    print("PASS: real server Replay, formal history width, mixed batch resets, joint GPU updates and batch policy",
          args.output,flush=True)


def audit_real_branches(argv):
    """Bounded real branch + replay/GPU training check using an existing smoke checkpoint."""
    import argparse, json, re, gym
    import tools
    from collections import OrderedDict
    from datetime import datetime
    from expr import LS_Imagine, make_dataset
    from search_branches import collect_real_branches
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    parser.add_argument("--group",type=int,default=1)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new audit output directory")
    args.output.mkdir(parents=True)
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    cfg=SimpleNamespace(**ckpt["config"])
    cfg.logdir=str(args.output);cfg.traindir=str(args.output/"train_eps")
    cfg.results_dir=str(args.output);cfg.use_wandb=False;cfg.log_all_metrics=False
    cfg.search_branch_count=2;cfg.search_branch_prefix=2
    cfg.steps=int(ckpt["step"])+2*(cfg.search_branch_prefix+cfg.search_a_commit_steps)
    sample=next((args.checkpoint.parent/"train_eps").glob("*.npz"))
    raw=np.load(sample)
    spaces={"image":gym.spaces.Box(0,255,(64,64,3),np.uint8),
            "heatmap":gym.spaces.Box(0,255,(64,64,1),np.uint8)}
    for key in raw.files:
        if re.match(cfg.encoder["mlp_keys"],key) and not key.startswith("search_"):
            spaces[key]=gym.spaces.Box(-np.inf,np.inf,raw[key].shape[1:],np.float32)
    logger=tools.Logger(cfg,args.output,ckpt["step"])
    agent=LS_Imagine(gym.spaces.Dict(spaces),gym.spaces.Discrete(12),cfg,logger,None).to(cfg.device)
    agent.load_state_dict(ckpt["agent_state_dict"],strict=True)
    tools.recursively_load_optim_state_dict(agent,ckpt["optims_state_dict"])
    agent.requires_grad_(False)
    replay=OrderedDict()
    consumed=collect_real_branches(agent,cfg,replay,logger,args.group)
    assert 0 < consumed <= 2*(cfg.search_branch_prefix+cfg.search_a_commit_steps)
    assert agent._step == int(ckpt["step"])+consumed
    assert len(replay)==2
    report=json.loads((args.output/"branches"/("%06d"%args.group)/"feedback.json").read_text())
    for ep in replay.values():
        assert len({len(v) for v in ep.values()})==1
        assert ep["search_branch_verified"].all(), report
    before=agent._wm.search_b.latent_adapter.weight.detach().clone()
    for _ in range(2):
        agent._train(next(make_dataset(replay,cfg)))
    assert any(x>0 for x in agent._metrics["C/transitions"])
    assert any(x>0 for x in agent._metrics["C/imagined_transitions"])
    assert float((before-agent._wm.search_b.latent_adapter.weight).abs().sum())>0
    agent.flush_metrics()
    assert all(torch.isfinite(v).all() for v in agent.state_dict().values())
    ckpt.update(agent_state_dict=agent.state_dict(),step=agent._step,config=vars(cfg),
                optims_state_dict=tools.recursively_collect_optim_state_dict(agent))
    torch.save(ckpt,args.output/"latest.pt")
    logger.finish()
    print("PASS: verified same-scene real branches, exact budget, shared Replay, GPU ABC updates and checkpoint",args.output,flush=True)


if __name__=="__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--startup-replay":
        audit_startup_replay(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--branch-checkpoint":
        audit_real_branches(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--audit-replay":
        audit_replay(sys.argv[2:])
    else:
        unittest.main(verbosity=2)
