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

if __name__=="__main__":
    unittest.main(verbosity=2)
