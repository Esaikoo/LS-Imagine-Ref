"""T01 offline migration/round-trip acceptance; never creates an environment.

Uses the accepted T00 config, observation spaces and a few stored observations.
Synthetic optimizer probes are marked as verification artifacts, not training.
"""

import argparse
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import t00_baseline as baseline


class Report(baseline.Report):
    def finish(self):
        failed = any(item["level"] == "FAIL" for item in self.data["checks"])
        self.data["status"] = "failed" if failed else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        print(f"[{'FAIL' if failed else 'PASS'}] T01_CHECKS; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(left, right):
    import numpy as np
    import torch

    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def same_forward(left, right):
    import torch

    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.shape == right.shape and torch.allclose(left, right, atol=1e-5, rtol=1e-5)
    return set(left) == set(right) and all(same_forward(left[key], right[key]) for key in left)


def expect_rejection(report, name, operation, expected, contains):
    try:
        operation()
    except expected as error:
        require(contains in str(error), f"{name} 出错但没有命中预期保护: {error}")
        report.check(name, "PASS", str(error))
    else:
        raise AssertionError(f"{name} 未拒绝不兼容输入")


def check_weights(agent, saved, *, base_only=False):
    import long_horizon as lh

    current = lh.normalize(agent.state_dict())
    expected = lh.normalize(saved)
    if base_only:
        current = {key: value for key, value in current.items() if not key.startswith("_experiment_modules.")}
    lh.require_compatible(current, expected)
    seen = set()
    for key, tensor in current.items():
        signature = (tensor.data_ptr(), expected[key].data_ptr(), tuple(tensor.shape), tensor.dtype)
        if signature in seen:
            continue
        seen.add(signature)
        require(same(tensor, expected[key]), f"权重数值不一致: {key}")
    return len(current)


def load_baseline(args, report):
    import gym
    import numpy as np

    source = baseline.project_path(args.baseline_dir)
    accepted = json.loads((source / "report.json").read_text(encoding="utf-8"))
    require(accepted.get("command") == "evaluate" and accepted.get("status") in ("passed", "passed_with_warnings"),
            "--baseline-dir 必须指向通过验收的 T00 evaluate 输出目录")
    baseline_checkpoint = accepted.get("baseline_summary", {}).get("checkpoint")
    require(baseline_checkpoint is not None and baseline.project_path(baseline_checkpoint) == baseline.project_path(args.checkpoint),
            "T00 报告的 checkpoint 路径与本次输入不同")
    accepted_signature = accepted.get("checkpoint_file_after", accepted.get("checkpoint_file_before"))
    require(accepted_signature == baseline.file_signature(baseline.project_path(args.checkpoint)),
            "checkpoint 的大小/修改时间与 T00 验收记录不同，请重新确认基线来源")
    shapes = accepted.get("observation_shapes", {})
    require({"image", "heatmap", "obs_reward", "is_first", "is_terminal"} <= set(shapes), "T00 报告缺少必要观测形状")
    saved = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    sections = baseline.yaml_read(ROOT / "configs.yaml")
    config = copy.deepcopy(sections["defaults"])
    baseline.merge(config, sections["minedojo"])
    baseline.merge(config, saved)
    require(config.get("num_actions", 0) > 1, "T00 配置没有有效动作维度")
    config.update(device=args.device, compile=False, envs=1, parallel=False, use_wandb=False,
                  video_pred_log=False, logdir=str(report.directory), init_checkpoint=args.checkpoint,
                  checkpoint_load="initialize")
    # Only shapes are used by WorldModel. Match T00 shapes exactly, including
    # the obs_reward MLP input; don't invent a smaller observation interface.
    spaces = {}
    for key, shape in shapes.items():
        visual = key in ("image", "heatmap", "zoomed_image", "heatmap_on_zoomed")
        spaces[key] = gym.spaces.Box(0 if visual else -np.inf, 255 if visual else np.inf,
                                     shape=tuple(shape), dtype=np.uint8 if visual else np.float32)
    obs_space = gym.spaces.Dict(spaces)
    act_space = gym.spaces.Box(0, 1, shape=(config["num_actions"],), dtype=np.float32)
    act_space.discrete = True
    episodes = sorted((source / "eval_eps").glob("*.npz"))
    require(bool(episodes), "T00 输出缺少 eval_eps/*.npz，无法检查真实回放输入")
    observations = []
    with np.load(episodes[0], allow_pickle=False) as episode:
        require(set(shapes) <= set(episode.files), "T00 replay 的观测字段不完整")
        count = min(3, len(episode["image"]))
        require(count >= 2, "T00 replay 至少需要两个观测")
        for index in range(count):
            # Replay flags are scalar even where Gym declares shape=(1,).
            # Preserve actual recorded shapes, just as tools.simulate does.
            obs = {key: np.array(episode[key][index:index + 1], copy=True) for key in shapes}
            for key in ("heatmap", "heatmap_on_zoomed"):
                if key in obs and obs[key].ndim == 4 and obs[key].shape[-1] == 1:
                    obs[key] = obs[key][..., 0]
            for key in ("is_first", "is_last", "is_terminal", "is_zoomed", "is_calculated", "jump"):
                if key in obs and obs[key].shape == (1, 1):
                    obs[key] = obs[key][:, 0]
            observations.append(obs)
    report.data["baseline"] = {"directory": str(source), "replay": str(episodes[0]), "observations": count,
                               "observation_shapes": shapes, "historical_training_config_verified": False}
    report.check("baseline_input", "PASS", "读取已通过 T00 的配置、观测形状及 3 帧以内回放；不启动 MineDojo/MineCLIP")
    return config, obs_space, act_space, observations


def rng_draw(device):
    import numpy as np
    import torch

    return {"python": random.random(), "numpy": np.random.rand(3), "torch": torch.rand(3),
            "cuda": torch.rand(3, device=device) if torch.device(device).type == "cuda" else None}


def policy_snapshot(agent, observations, seed):
    import numpy as np
    import torch
    import tools

    tools.set_seed_everywhere(seed)
    previous = None
    result = []
    before = (agent._step, agent._update_count, agent._logger.step)
    with torch.no_grad():
        for observation in observations:
            output, previous = agent(observation, np.zeros(1, dtype=bool), previous, training=False)
            action = output["action"]
            require(action.shape == (1, agent._config.num_actions), "原策略动作形状错误")
            require(bool(torch.isfinite(output["logprob"]).all()), "原策略 logprob 非有限")
            reference = torch.nn.functional.one_hot(action.argmax(-1), agent._config.num_actions).to(action.dtype)
            require(torch.allclose(action, reference, atol=1e-6, rtol=1e-5), "原策略动作不是 onehot")
            result.append({"output": {key: value.detach().cpu().clone() for key, value in output.items()},
                           "state": {key: value.detach().cpu().clone() for key, value in previous[0].items()}})
    require(before == (agent._step, agent._update_count, agent._logger.step), "原策略评估更新了训练计数")
    return result


def actor_inputs(agent, observation):
    import torch

    with torch.no_grad():
        obs = agent._wm.preprocess(observation)
        embed = agent._wm.encoder(obs)
        state, _ = agent._wm.dynamics.obs_step(None, None, embed, obs["is_first"])
        features = agent._wm.dynamics.get_feat(state)
    # Use different goals AND remaining budgets. Initial extra weights are
    # zero, so every one of these should reproduce the original actor.
    features = features.repeat(4, 1)
    goals = torch.randn(4, agent._config.goal_dim, device=features.device)
    remaining = torch.linspace(1, agent._config.macro_max_steps, 4, device=features.device)
    return features, goals, remaining


def synthetic_optimizer_probe(agent, inputs):
    import torch
    import torch.nn.functional as F

    base_versions = {name: parameter._version for name, parameter in agent.named_parameters()
                     if not name.startswith("_experiment_modules.")}
    features, goals, remaining = inputs
    worker = agent._experiment_modules["worker"]
    optimizer = agent._experiment_optimizers["worker"]
    worker.requires_grad_(True)
    try:
        optimizer.zero_grad(set_to_none=True)
        dist = worker(features, goals, remaining)
        targets = F.one_hot((dist.probs.detach().argmax(-1) + 1) % agent._config.num_actions,
                            agent._config.num_actions).float()
        loss = -dist.log_prob(targets).mean()
        require(bool(torch.isfinite(loss)), "合成 worker loss 非有限")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        require(bool(optimizer.state), "新 worker 优化器没有产生状态")
    finally:
        worker.requires_grad_(False)
    if "manager" in agent._experiment_modules:
        manager = agent._experiment_modules["manager"]
        optimizer = agent._experiment_optimizers["manager"]
        manager.requires_grad_(True)
        try:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(manager(features), torch.zeros(len(features), dtype=torch.long, device=features.device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            require(bool(optimizer.state), "新 manager 优化器没有产生状态")
        finally:
            manager.requires_grad_(False)
    require(base_versions == {name: parameter._version for name, parameter in agent.named_parameters() if name in base_versions},
            "合成验收意外更新了原 WM/actor/value")
    require(not any(parameter.requires_grad for parameter in agent._wm.parameters()), "冻结 WM 被意外解冻")


def module_snapshot(agent, inputs):
    import torch

    with torch.no_grad():
        result = {"worker_probs": agent._experiment_modules["worker"](*inputs).probs.detach().cpu().clone()}
        if "manager" in agent._experiment_modules:
            result["manager_logits"] = agent._experiment_modules["manager"](inputs[0]).detach().cpu().clone()
    return result


def check_mode(args, report, mode, config_dict, spaces, observations, source):
    import torch
    import expr
    import long_horizon as lh
    import tools

    directory = report.directory / mode
    directory.mkdir()
    config = argparse.Namespace(**copy.deepcopy(config_dict))
    config.experiment_mode = mode
    config.freeze_wm = mode != "flat_ls"
    config.logdir = str(directory)
    baseline.write_json(directory / "resolved_config.json", vars(config))
    logger = argparse.Namespace(step=0)
    tools.set_seed_everywhere(config.seed)
    agent = expr.LS_Imagine(*spaces, config, logger, dataset=None).to(config.device)
    agent.requires_grad_(False)
    agent.eval()
    migration = lh.initialize_from_checkpoint(agent, source, args.checkpoint)
    migration["mode"] = mode
    migration["new_keys"] = sorted(key for key in lh.normalize(agent.state_dict()) if key.startswith("_experiment_modules."))
    migration["compiled_names_normalized"] = sum(name != lh.canonical(name) for name in source["agent_state_dict"])
    require(all(not optimizer.state for _, optimizer in lh.base_optimizers(agent).values()), "旧优化器状态被意外导入")
    require(all(not optimizer.state for optimizer in agent._experiment_optimizers.values()), "新优化器没有从空状态开始")
    weight_count = check_weights(agent, source["agent_state_dict"], base_only=True)
    report.check(f"{mode}/strict_migration", "PASS", f"{weight_count} 个原权重项形状/类型/数值全部一致；没有导入旧优化器")
    inputs = None
    if mode == "flat_ls":
        require(not migration["new_keys"] and not agent._experiment_optimizers, "默认 flat_ls 意外增加可训练参数")
        expected_behavior = policy_snapshot(agent, observations, config.seed + 100)
        report.check(f"{mode}/policy", "PASS", "真实回放的原策略前向产生有效 onehot 动作；计数未变化")
    else:
        worker = agent._experiment_modules["worker"]
        old_first = agent._task_behavior.actor.layers.Actor_linear0.weight
        new_first = worker.actor.layers.Actor_linear0.weight
        require(torch.equal(new_first[:, :worker.feature_dim], old_first), "worker 第一层没有精确复制原输入权重")
        require(int(torch.count_nonzero(new_first[:, worker.feature_dim:])) == 0, "新增目标/时间输入权重不为零")
        inputs = actor_inputs(agent, observations[0])
        with torch.no_grad():
            old_probs = agent._task_behavior.actor(inputs[0]).probs
            new_probs = worker(*inputs).probs
        error = float((old_probs - new_probs).abs().max())
        require(torch.allclose(old_probs, new_probs, atol=1e-5, rtol=1e-5), f"worker 初始动作分布改变，max_error={error}")
        migration["worker_first_layer"] = {"source": list(old_first.shape), "target": list(new_first.shape),
                                            "zero_extra_columns": config.goal_dim + 1, "max_probability_error": error}
        base_ids = {id(parameter) for name, parameter in agent.named_parameters() if not name.startswith("_experiment_modules.")}
        optimizer_ids = []
        for optimizer in agent._experiment_optimizers.values():
            ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            require(not ids & base_ids and not any(ids & previous for previous in optimizer_ids), "新优化器与原模型/其他新模块共享参数")
            optimizer_ids.append(ids)
        require(set.union(*optimizer_ids) == {id(parameter) for parameter in agent._experiment_modules.parameters()},
                "新优化器遗漏模块参数，或持有设备迁移前的旧参数对象")
        require(not any(parameter.requires_grad for parameter in agent._wm.parameters()), "原型 WM 未冻结")
        report.check(f"{mode}/worker_initialization", "PASS", f"目标/剩余步数输入置零；4 组初始分布与原 actor 一致，最大误差 {error:.3g}")
        report.check(f"{mode}/optimizers", "PASS", f"新增模块优化器独立: {sorted(agent._experiment_optimizers)}；WM 冻结")
        expect_rejection(report, f"{mode}/runtime_guard", lambda: agent({}, [], training=False), NotImplementedError, "T01")
        synthetic_optimizer_probe(agent, inputs)
        # This fixture only tests serialization of a goal library and its ID.
        # Real goal construction and its encoder are implemented in T02.
        agent._goal_encoder_id = "t01_synthetic_encoder"
        agent._goal_library = {"kind": "serialization_fixture", "goal_count": config.goal_count,
                               "features": torch.arange(config.goal_count * config.goal_dim, dtype=torch.float32)
                                                .reshape(config.goal_count, config.goal_dim)}
        agent._worker_version = 1
        expected_behavior = module_snapshot(agent, inputs)
        report.check(f"{mode}/optimizer_probe", "PASS", "仅对新模块做一次合成更新，产生非空优化器状态；原 WM/actor/value 未更新")
    baseline.write_json(directory / "migration.json", migration)

    # Known counter values let the check detect accidental resets on resume.
    agent._step, agent._update_count, agent._logger.step = 17, 5, 17 * config.action_repeat
    for name in ("_should_train", "_should_log", "_should_reset"):
        getattr(agent, name)._last = 17
    agent._should_pretrain._once = False
    if config.critic["slow_target"]:
        agent._task_behavior._updates = 3
    expected_counters = lh.counters(agent)
    path = directory / "roundtrip_test.pt"
    lh.save_checkpoint(agent, path, verification_artifact=True,
                       metadata={"purpose": "T01 synthetic acceptance only; NOT a trained goal policy"})
    expected_rng = rng_draw(config.device)
    checkpoint = lh.read_checkpoint(path)
    expect_rejection(report, f"{mode}/artifact_guard", lambda: lh.restore_checkpoint(agent, checkpoint), ValueError, "合成验收")
    bad_mode = dict(checkpoint, experiment=dict(checkpoint["experiment"], mode="hierarchical" if mode == "flat_ls" else "flat_ls"))
    expect_rejection(report, f"{mode}/mode_guard", lambda: lh.restore_checkpoint(agent, bad_mode, allow_verification=True), ValueError, "模式")
    bad_state = dict(checkpoint["agent_state_dict"])
    first = next(name for name, tensor in bad_state.items() if tensor.ndim > 1)
    bad_state[first] = bad_state[first][:1]
    bad_shape = dict(checkpoint, agent_state_dict=bad_state)
    expect_rejection(report, f"{mode}/shape_guard", lambda: lh.restore_checkpoint(agent, bad_shape, allow_verification=True), ValueError, "权重不兼容")
    bad_steps = dict(checkpoint, training_state=dict(checkpoint["training_state"], update_count=-1))
    expect_rejection(report, f"{mode}/counter_guard", lambda: lh.restore_checkpoint(agent, bad_steps, allow_verification=True), ValueError, "计数器")
    if mode != "flat_ls":
        agent._goal_encoder_id = "wrong_encoder"
        expect_rejection(report, f"{mode}/encoder_guard", lambda: lh.restore_checkpoint(agent, checkpoint, allow_verification=True), ValueError, "编码器")
        agent._goal_encoder_id = None
    # Destroy selected state before restoring. Reading the serialized file is
    # essential: state_dict() alone aliases the in-memory parameter tensors.
    with torch.no_grad():
        next(agent.parameters()).add_(1.0)
        for module in agent._experiment_modules.values():
            next(module.parameters()).add_(1.0)
    for _, optimizer in lh.base_optimizers(agent).values():
        optimizer.state.clear()
    for optimizer in agent._experiment_optimizers.values():
        optimizer.state.clear()
    agent._step = agent._update_count = agent._logger.step = 0
    agent._goal_library = agent._goal_encoder_id = None
    agent._worker_version = 0
    rng_draw(config.device)
    lh.restore_checkpoint(agent, checkpoint, allow_verification=True)
    require(same(expected_rng, rng_draw(config.device)), "Python/NumPy/Torch/CUDA 随机状态恢复失败")
    check_weights(agent, checkpoint["agent_state_dict"])
    require(same(expected_counters, lh.counters(agent)), "训练/日志/调度/value 计数恢复失败")
    saved_base = lh.normalize(checkpoint["optims_state_dict"])
    for name, (_, optimizer) in lh.base_optimizers(agent).items():
        require(same(optimizer.state_dict(), saved_base[name]), f"原优化器恢复失败: {name}")
    for name, optimizer in agent._experiment_optimizers.items():
        require(same(optimizer.state_dict(), checkpoint["new_optims_state_dict"][name]), f"新优化器恢复失败: {name}")
    for name, scaler in lh.amp_scalers(agent).items():
        require(same(scaler.state_dict(), checkpoint["amp_scalers_state_dict"][name]), f"AMP scaler 恢复失败: {name}")
    require(same(agent._goal_library, checkpoint["experiment"]["goal_library"]), "目标库字段恢复失败")
    require(agent._goal_encoder_id == checkpoint["experiment"]["goal_encoder_id"] and
            agent._worker_version == checkpoint["experiment"]["worker_version"], "目标编码器/低层版本恢复失败")
    actual_behavior = policy_snapshot(agent, observations, config.seed + 100) if mode == "flat_ls" else module_snapshot(agent, inputs)
    if mode == "flat_ls":
        require(len(expected_behavior) == len(actual_behavior) and
                all(same_forward(before, after) for before, after in zip(expected_behavior, actual_behavior)),
                "保存/恢复后原策略前向结果不一致")
        require(all(torch.equal(before["output"]["action"].argmax(-1), after["output"]["action"].argmax(-1))
                    for before, after in zip(expected_behavior, actual_behavior)), "保存/恢复后原策略选择动作改变")
    else:
        require(same_forward(expected_behavior, actual_behavior), "保存/恢复后新模块前向结果不一致")
    report.data.setdefault("modes", {})[mode] = {"checkpoint": str(path), "verification_artifact": True,
                                               "base_weight_items": weight_count, "new_weight_items": len(migration["new_keys"]),
                                               "counters": expected_counters, "new_optimizers": sorted(agent._experiment_optimizers)}
    report.check(f"{mode}/roundtrip", "PASS", "权重、优化器、全部计数、目标元数据及 RNG 恢复一致；恢复后前向一致")
    # Release the large CPU/GPU models before checking the next mode.
    del agent, checkpoint, bad_mode, bad_shape, bad_steps, bad_state, inputs
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-dir", required=True, help="已通过 T00 的 evaluate 输出目录")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", nargs="+", choices=("flat_ls", "goal_worker", "hierarchical"),
                        default=["flat_ls", "goal_worker", "hierarchical"])
    parser.add_argument("--output-root", default="relevance_map/t01_outputs")
    args = parser.parse_args()
    args.command = "t01"
    require(len(args.modes) == len(set(args.modes)), "--modes 不能重复")
    source = baseline.project_path(args.checkpoint)
    baseline_dir = baseline.project_path(args.baseline_dir)
    output_root = baseline.project_path(args.output_root)
    for protected in (source.parent, baseline_dir):
        if output_root == protected or protected in output_root.parents:
            raise SystemExit("输出目录必须独立于原 checkpoint 和 T00 输出目录")
    directory = output_root / ("check_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    os.chdir(ROOT)
    try:
        import torch
        import long_horizon as lh

        # Existing networks create some unregistered tensors on CUDA during
        # construction. Don't claim CPU-only support for this model check.
        if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("T01 完整模型验收需要可用 CUDA，请使用 --device cuda:0")
        baseline.runtime_info(report)
        report.data["runtime"]["torch"] = torch.__version__
        report.data["checkpoint_file_before"] = baseline.file_signature(source)
        config, obs_space, act_space, observations = load_baseline(args, report)
        checkpoint = lh.read_checkpoint(source)
        require(checkpoint.get("checkpoint_format") is None or checkpoint.get("experiment", {}).get("mode") == "flat_ls",
                "T01 迁移检查需要原 flat LS checkpoint")
        report.data["legacy_optimizer_present"] = bool(checkpoint.get("optims_state_dict"))
        # The legacy optimizer is deliberately never imported. Release its
        # large moment tensors now, rather than retaining them for every mode.
        checkpoint.pop("optims_state_dict", None)
        report.check("scope", "PASS", "离线初始化/序列化验收；不启动环境、不做真实训练；合成产物禁止用于真实训练")
        for current_mode in args.modes:
            check_mode(args, report, current_mode, config, (obs_space, act_space), observations, checkpoint)
        # Legacy checkpoints do not have trustworthy steps/RNG. Verify that
        # callers can't accidentally reinterpret initialization as resume.
        if checkpoint.get("checkpoint_format") is None:
            expect_rejection(report, "legacy_resume_guard", lambda: lh.restore_checkpoint(None, checkpoint), ValueError, "旧 latest.pt")
    except (Exception, KeyboardInterrupt) as error:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.check("execution", "FAIL", f"{type(error).__name__}: {error}；详情见 error.txt")
    finally:
        before = report.data.get("checkpoint_file_before")
        if before:
            try:
                after = baseline.file_signature(source)
                report.data["checkpoint_file_after"] = after
                report.check("source_checkpoint_unchanged", "PASS" if before == after else "FAIL", "核对原文件大小与修改时间（非内容哈希）；没有写入原运行目录")
            except OSError as error:
                report.check("source_checkpoint_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
