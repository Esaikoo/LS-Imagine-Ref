"""T00: inspect a saved LS-Imagine agent, or evaluate it without training.

Run from any working directory; relative CLI paths refer to the project root.
The inspect command deliberately does not import expr, MineDojo or MineCLIP.
"""

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_signature(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def yaml_read(path):
    # Match expr.py's YAML numeric parsing (notably 1e-4 learning rates).
    from ruamel.yaml import YAML

    with path.open(encoding="utf-8") as stream:
        value = YAML(typ="safe").load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是字典: {path}")
    return value


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


class Report:
    def __init__(self, directory, args):
        self.directory = directory
        self.data = {
            "command": args.command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "output_dir": str(directory),
            "checks": [],
        }
        self.started = time.perf_counter()

    def check(self, name, level, detail):
        self.data["checks"].append({"name": name, "level": level, "detail": detail})
        print(f"[{level}] {name}: {detail}", flush=True)
        self.save()

    def save(self):
        write_json(self.directory / "report.json", self.data)

    def finish(self):
        levels = {item["level"] for item in self.data["checks"]}
        self.data["status"] = (
            "failed" if "FAIL" in levels else "passed_with_warnings" if "WARN" in levels else "passed"
        )
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        label = "T00_CHECKPOINT_INSPECT" if self.data["command"] == "inspect" else "T00_BASELINE_EVAL"
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] {label}; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def runtime_info(report):
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    report.data["runtime"] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(ROOT),
        "current_git_commit": git("rev-parse", "HEAD"),
        "current_git_status": git("status", "--short"),
        "checkpoint_training_git_commit": None,
    }


def resolve_config(args, report):
    from ruamel.yaml import YAML

    config_file = project_path(args.config_file)
    sections = yaml_read(config_file)
    config = copy.deepcopy(sections["defaults"])
    for section in args.configs:
        if section not in sections:
            raise ValueError(f"未知配置段: {section}")
        merge(config, sections[section])
    sources = [str(config_file)]
    if args.run_config:
        path = project_path(args.run_config)
        saved = yaml_read(path)
        if "defaults" in saved:
            raise ValueError("--run-config 需要展开后的运行配置；分组配置请使用 --config-file")
        merge(config, saved)
        sources.append(str(path))
    for expression in args.set:
        key, separator, value = expression.partition("=")
        if not separator or not key:
            raise ValueError(f"--set 必须为 key=value: {expression}")
        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(current.get(part), dict):
                raise ValueError(f"未知配置路径: {key}")
            current = current[part]
        if parts[-1] not in current:
            raise ValueError(f"未知配置项: {key}")
        current[parts[-1]] = YAML(typ="safe").load(value)
    config["task"] = args.task
    if args.command == "evaluate":
        config.update(
            device=args.device, seed=args.seed, compile=False, envs=1, parallel=False,
            use_wandb=False, video_pred_log=False, eval_episode_num=args.episodes,
            logdir=str(report.directory), traindir=str(report.directory / "unused_train_eps"),
            evaldir=str(report.directory / "eval_eps"), results_dir=str(report.directory / "environment"),
            name="t00_baseline",
        )
    specs_file = ROOT / "envs/tasks/task_specs.yaml"
    suite, separator, task = args.task.partition("_")
    if suite != "minedojo" or not separator:
        raise ValueError("T00 当前只支持 minedojo_<task> 任务")
    specs = copy.deepcopy(yaml_read(specs_file)[task])
    config["episode_max_steps"] = int(specs["terminal_specs"]["max_steps"])
    if config["episode_max_steps"] <= 0:
        raise ValueError("任务 max_steps 必须大于 0")
    report.data["configuration"] = {
        "sources": sources,
        "training_config_provided": bool(args.run_config),
        "task_specs_path": str(specs_file),
        "task_specs_sha256": hashlib.sha256(specs_file.read_bytes()).hexdigest(),
        "training_config_identity_verified": False,
    }
    write_json(report.directory / "resolved_config.json", config)
    write_json(report.directory / "task_specs.json", specs)
    report.check("config_source", "WARN", "旧 checkpoint 不包含完整运行配置；本次配置已保存，仍需核对原训练设置")
    return config, specs


def source_action_names():
    tree = ast.parse((ROOT / "envs/tasks/base/ls_imagine_wrapper.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASIC_ACTIONS" for target in node.targets
        ):
            return [ast.literal_eval(key) for key in node.value.keys]
    raise ValueError("未在 LSImagineWrapper 中找到 BASIC_ACTIONS")


def mineclip_path():
    tree = ast.parse((ROOT / "envs/tasks/base/clip_reward.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ClipReward":
            constructor = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
            names = [arg.arg for arg in constructor.args.args]
            defaults = dict(zip(names[-len(constructor.args.defaults):], constructor.args.defaults))
            return project_path(ast.literal_eval(defaults["ckpt"]))
    raise ValueError("未找到 ClipReward 的默认外部权重路径")


def load_checkpoint(args, config, report):
    import torch

    path = project_path(args.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}；请确认文件名为 latest.pt")
    report.data["checkpoint_file_before"] = file_signature(path)
    started = time.perf_counter()
    # Explicit for PyTorch >=2.6 as well as the project's older PyTorch.
    # This command loads the user's own LS-Imagine training checkpoint.
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch.load(path, **kwargs)
    report.data["checkpoint_load_seconds"] = round(time.perf_counter() - started, 3)
    report.data["runtime"]["torch"] = torch.__version__
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("agent_state_dict"), dict):
        raise ValueError("需要 expr.py 保存的字典，且包含 agent_state_dict；不是单独的 MineCLIP 权重")
    report.data["checkpoint_keys"] = list(checkpoint)
    report.check("checkpoint_format", "PASS", f"找到 agent_state_dict；文件 {path.stat().st_size / 1024**2:.1f} MiB")
    has_optim = isinstance(checkpoint.get("optims_state_dict"), dict)
    report.data["optimizer_state_present"] = has_optim
    report.check("optimizer_state", "PASS" if has_optim else "WARN", "存在（评估不恢复优化器）" if has_optim else "未找到；策略评估不依赖优化器")
    state = {}
    renamed = 0
    for key, value in checkpoint["agent_state_dict"].items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f"agent_state_dict 项必须为名称和 Tensor: {key}")
        normalized = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if normalized in state:
            raise ValueError(f"去除 torch.compile 前缀后名称冲突: {normalized}")
        state[normalized] = value
        renamed += int(normalized != key)
    del checkpoint
    if not state:
        raise ValueError("agent_state_dict 为空")
    shapes = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()}
    write_json(report.directory / "checkpoint_shapes.json", shapes)
    report.data["checkpoint_structure"] = {
        "state_dict_entries": len(state),
        "compile_prefixes_normalized": renamed,
        "training_step": None,
        "note": "仅去掉路径段 _orig_mod；不删除网络层，完整兼容性由 evaluate 的 strict=True 验证",
    }
    for prefix in ("_wm.", "_task_behavior.actor.", "_task_behavior.value."):
        count = sum(key.startswith(prefix) for key in state)
        report.check(prefix.rstrip("."), "PASS" if count else "FAIL", f"{count} 个权重项")
    names = source_action_names()
    report.data["action_names"] = names
    stoch_dim = config["dyn_stoch"] * config["dyn_discrete"] if config["dyn_discrete"] else config["dyn_stoch"]
    expected = {
        "_task_behavior.actor.mean_layer.weight": (len(names), config["units"]),
        "_task_behavior.actor.layers.Actor_linear0.weight": (config["units"], stoch_dim + config["dyn_deter"]),
        "_wm.dynamics._img_in_layers.0.weight": (config["dyn_hidden"], stoch_dim + len(names) + 1),
        "_wm.dynamics._cell.layers.GRU_linear.weight": (3 * config["dyn_deter"], config["dyn_hidden"] + config["dyn_deter"]),
    }
    channels = sum(ch for name, ch in (("image", 3), ("heatmap", 1)) if re.match(config["encoder"]["cnn_keys"], name))
    if channels:
        kernel = config["encoder"]["kernel_size"]
        expected["_wm.encoder._cnn.layers.0.weight"] = (config["encoder"]["cnn_depth"], channels, kernel, kernel)
    for key, shape in expected.items():
        actual = tuple(state[key].shape) if key in state else None
        report.check(key, "PASS" if actual == shape else "FAIL", f"保存形状 {actual}，配置期望 {shape}")
    report.data["encoder_input_weights"] = {
        key: shapes[key] for key in state if key.startswith("_wm.encoder.") and key.endswith("0.weight")
    }
    report.check("compile_names", "PASS", f"规范化 {renamed} 个编译前缀；评估使用 compile=False")
    report.check("training_step", "WARN", "当前保存格式没有确切训练步数；不能仅凭 latest.pt 判断已训练 1M")
    return state


def inspect_replay(args, report):
    import numpy as np

    source = project_path(args.checkpoint).parent
    candidates = [project_path(path) for path in args.replay_dir] if args.replay_dir else [source / "train_eps", source / "eval_eps"]
    summaries = []
    train_fields = {"image", "heatmap", "action", "is_first", "is_last", "is_terminal", "reward"}
    for directory in candidates:
        files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
        steps = 0
        unknown_lengths = 0
        for path in files:
            match = re.search(r"-(\d+)\.npz$", path.name)
            if match:
                steps += max(0, int(match.group(1)) - 1)
            else:
                unknown_lengths += 1
        samples = []
        for path in files[:args.replay_samples]:
            sample = {"path": str(path)}
            try:
                with np.load(path, allow_pickle=False) as episode:
                    sample["keys"] = list(episode.files)
                    sample["missing_fields"] = sorted(train_fields - set(episode.files))
                    sample["shapes"] = {key: list(episode[key].shape) for key in train_fields if key in episode.files}
                    sample["steps"] = max(0, len(episode["reward"]) - 1) if "reward" in episode.files else None
                    sample["success"] = bool(np.any(episode["success"])) if "success" in episode.files else None
                    issues = []
                    if "reward" in episode.files:
                        length = len(episode["reward"])
                        if length < 2:
                            issues.append("没有完整的观测-动作转移")
                        if not np.isfinite(episode["reward"]).all():
                            issues.append("reward 含非有限值")
                        for key in train_fields & set(episode.files):
                            if not episode[key].shape or episode[key].shape[0] != length:
                                issues.append(f"{key} 的时间长度与 reward 不一致")
                    if "action" in episode.files:
                        action = episode["action"]
                        if action.ndim != 2 or action.shape[-1] != len(report.data["action_names"]):
                            issues.append("action 形状不符合当前离散动作表")
                    sample["issues"] = issues
                sample["readable"] = True
            except Exception as error:
                sample.update(readable=False, error=f"{type(error).__name__}: {error}")
            samples.append(sample)
        summaries.append({
            "directory": str(directory), "exists": directory.is_dir(), "episode_files": len(files),
            "steps_hint_from_filenames": steps, "unrecognized_filename_lengths": unknown_lengths,
            "bytes": sum(path.stat().st_size for path in files), "samples": samples,
        })
        if files:
            broken = [item for item in samples if not item["readable"] or item.get("missing_fields") or item.get("issues")]
            report.check("replay", "WARN" if broken else "PASS", f"{directory}: {len(files)} 局；抽查 {len(samples)} 个文件，异常 {len(broken)} 个")
    write_json(report.directory / "replay_summary.json", summaries)
    if not any(item["episode_files"] for item in summaries):
        report.check("replay", "WARN", "未找到旧 replay；evaluate 会自动保存新 eval_eps。T02 仍可能需要增加在线采集量")
    report.data["replay_total_files"] = sum(item["episode_files"] for item in summaries)
    report.data["replay_steps_are_training_step"] = False


def check_assets(report):
    path = mineclip_path()
    report.data["external_mineclip"] = {"path": str(path), "exists": path.is_file()}
    report.check("external_mineclip", "PASS" if path.is_file() else "FAIL", f"当前共享 MineCLIP 工厂的默认权重: {path}")


def failed_checks(report):
    return any(item["level"] == "FAIL" for item in report.data["checks"])


def evaluate(args, config_dict, state, report):
    import collections
    import functools
    import numpy as np
    import torch
    import imageio.v2 as imageio
    import expr
    import tools
    from envs.tasks import get_specs
    from parallel import Damy

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查当前训练环境及可见 GPU")
    config = argparse.Namespace(**config_dict)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    # Match expr.main's task preparation, including nested specs used by make_env.
    _, task_specs, sim_specs = get_specs(config.task.split("_", 1)[1], target_item=config.target_item)
    task_specs["concentration_specs"].update(
        max_steps=config.episode_max_steps,
        gaussian_reward_weight=config.gaussian_reward_weight,
        gaussian_sigma_weight=config.gaussian_sigma_weight,
    )
    condition = task_specs["success_specs"]
    task_specs["clip_specs"]["target_object"] = condition["all"]["item"]["type"] if "all" in condition else condition["any"]["item"]["type"]
    report.data["seed"] = {
        "process_seed": args.seed,
        "scope": "Python/NumPy/PyTorch；复用原环境 reset/fast_reset 流程",
        "configured_world_seed": sim_specs.get("world_seed"),
        "exact_world_reproducibility_claimed": False,
    }
    write_json(report.directory / "effective_task_specs.json", task_specs)

    class CheckedEnv:
        def __init__(self, env):
            self.env = env
            self.steps = 0
            self.completed = []

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self):
            self.steps = 0
            self.started = time.perf_counter()
            return self.env.reset()

        def step(self, action):
            observation, reward, done, info = self.env.step(action)
            self.steps += 1
            if "error" in info:
                raise RuntimeError(f"环境报告错误: {info['error']}")
            if "success" not in info:
                raise RuntimeError("环境未提供 success，不能以正奖励代替真实任务成功")
            if not np.isfinite(float(reward)):
                raise RuntimeError("环境 reward 非有限值")
            if self.steps > config.episode_max_steps:
                raise RuntimeError("评估超过任务 max_steps，终止条件异常")
            if done:
                self.completed.append({"length": self.steps, "seconds": round(time.perf_counter() - self.started, 3)})
                # Evaluation discards pending ScoreStorage items at done. The
                # original collector's real_done filter can otherwise keep
                # stepping a timed-out episode before its next reset. Only
                # control the collector here; leave observation terminal flags
                # and all policy/environment behavior unchanged.
                info = dict(info, real_done=True)
            elif self.steps == config.episode_max_steps:
                raise RuntimeError("达到任务 max_steps 但环境没有结束")
            return observation, reward, done, info

    class EvalLogger:
        step = 0

        def __init__(self):
            self.rows = []
            self.scalars = {}
            self.saved_files = set()

        def scalar(self, name, value):
            self.scalars[name] = float(value)

        def video(self, name, value):
            # tools.simulate saves exactly one npz immediately before this call.
            new_files = set(episode_dir.glob("*.npz")) - self.saved_files
            if len(new_files) != 1:
                raise RuntimeError(f"每局评估应产生一个新 replay，实际 {len(new_files)} 个")
            path = new_files.pop()
            self.saved_files.add(path)
            index = len(self.rows) + 1
            video = report.directory / "videos" / f"episode_{index:03d}.mp4"
            with imageio.get_writer(str(video), fps=16, codec="libx264") as writer:
                for frame in np.asarray(value)[0]:
                    writer.append_data(frame.astype(np.uint8))
            with np.load(path, allow_pickle=False) as episode:
                success = bool(np.any(episode["success"][:config.episode_max_steps + 1]))
                first = min(float(episode["first_success_step"][-1]), config.episode_max_steps) if success else None
                row = {
                    "episode": index, "process_seed": args.seed, "success": int(success),
                    "return": float(episode["reward"][:config.episode_max_steps + 1].sum()),
                    "length": len(episode["reward"]) - 1, "first_success_step": first,
                    "zoom_frames": int(np.count_nonzero(episode["is_zoomed"])),
                    "seconds": checked.completed[-1]["seconds"], "replay": str(path), "video": str(video),
                }
            self.rows.append(row)
            with (report.directory / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            print(f"[EPISODE {index}/{args.episodes}] success={row['success']} length={row['length']} return={row['return']:.3f} video={video.name}", flush=True)

        def write(self, **kwargs):
            write_json(report.directory / "eval_metrics.json", self.scalars)

    episode_dir = report.directory / "eval_eps"
    episode_dir.mkdir()
    (report.directory / "videos").mkdir()
    logger = EvalLogger()
    env = None
    try:
        started = time.perf_counter()
        env = expr.make_env(config, "eval", 0)
        report.data["environment_initialization_seconds"] = round(time.perf_counter() - started, 3)
        config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
        report.data["observation_shapes"] = {key: list(value.shape) for key, value in env.observation_space.spaces.items()}
        write_json(report.directory / "resolved_config.json", vars(config))
        started = time.perf_counter()
        agent = expr.LS_Imagine(env.observation_space, env.action_space, config, logger, dataset=None).to(device)
        report.data["agent_initialization_seconds"] = round(time.perf_counter() - started, 3)
        expected = agent.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = {
            key: {"checkpoint": list(state[key].shape), "model": list(expected[key].shape)}
            for key in set(expected) & set(state) if expected[key].shape != state[key].shape
        }
        compatibility = {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatched}
        write_json(report.directory / "load_compatibility.json", compatibility)
        if missing or unexpected or mismatched:
            raise RuntimeError("checkpoint 与当前完整模型不匹配；见 load_compatibility.json，请核对原训练配置及代码版本")
        agent.load_state_dict(state, strict=True)
        state.clear()
        del expected
        agent.requires_grad_(False)
        agent.eval()
        versions = {name: parameter._version for name, parameter in agent.named_parameters()}
        checked = CheckedEnv(env)
        report.check("strict_load", "PASS", f"完整权重严格加载；动作维度 {config.num_actions}；没有跳过层")
        started = time.perf_counter()
        with torch.no_grad():
            tools.simulate(
                functools.partial(agent, training=False), [Damy(checked)], collections.OrderedDict(),
                episode_dir, logger, tools.ScoreStorage(config.episode_max_steps),
                config.episode_max_steps, config.discount, is_eval=True, episodes=args.episodes,
                is_training=False,
            )
        changed = [name for name, parameter in agent.named_parameters() if parameter._version != versions[name]]
        if changed or agent._update_count != 0 or agent._step != 0:
            raise RuntimeError(f"评估意外更新了模型或训练计数器: {changed[:5]}")
        if len(logger.rows) != args.episodes:
            raise RuntimeError(f"要求 {args.episodes} 局，实际保存 {len(logger.rows)} 局")
        success_count = sum(row["success"] for row in logger.rows)
        successful_times = [row["first_success_step"] for row in logger.rows if row["success"]]
        summary = {
            "episodes": len(logger.rows), "success_count": success_count,
            "success_rate": success_count / len(logger.rows),
            "mean_return": float(np.mean([row["return"] for row in logger.rows])),
            "mean_length": float(np.mean([row["length"] for row in logger.rows])),
            "mean_first_success_step_on_success": float(np.mean(successful_times)) if successful_times else None,
            "optimizer_updates": agent._update_count, "agent_training_step": agent._step,
            "evaluation_seconds": round(time.perf_counter() - started, 3),
            "checkpoint": str(project_path(args.checkpoint)), "task": args.task, "process_seed": args.seed,
            "policy": "original LS_Imagine, training=False, actor.mode(), no heatmap ablation",
            "collector_reset_on_done": True,
            "small_sample_note": "用于加载和行为验收；少量局数不能确认成功率复现",
        }
        write_json(report.directory / "summary.json", summary)
        report.data["baseline_summary"] = summary
        report.check("no_training", "PASS", "optimizer_updates=0；agent_training_step=0；参数版本未变化")
        report.check("evaluation", "PASS", f"完成 {len(logger.rows)} 局，成功 {success_count} 局，success_rate={summary['success_rate']:.4f}")
    finally:
        if env is not None:
            env.close()


def parser():
    command = argparse.ArgumentParser(description="T00 checkpoint 检查 / 原策略基线评估（不训练）")
    sub = command.add_subparsers(dest="command", required=True)
    for name in ("inspect", "evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--checkpoint", required=True)
        item.add_argument("--task", default="minedojo_harvest_log_in_plains")
        item.add_argument("--config-file", default="configs.yaml")
        item.add_argument("--configs", nargs="+", default=["minedojo"])
        item.add_argument("--run-config", help="原训练的展开配置（如有）；不从 checkpoint 猜测")
        item.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        item.add_argument("--output-root", default="relevance_map/t00_outputs")
        item.add_argument("--replay-dir", action="append", default=[], help="可重复指定；默认检查 checkpoint 同级 train_eps/eval_eps")
        item.add_argument("--replay-samples", type=int, default=3)
        if name == "evaluate":
            item.add_argument("--device", default="cuda:0")
            item.add_argument("--seed", type=int, default=0)
            item.add_argument("--episodes", type=int, default=3)
    return command


def main():
    args = parser().parse_args()
    if args.replay_samples < 0 or (args.command == "evaluate" and args.episodes < 1):
        raise SystemExit("--replay-samples 必须非负，--episodes 必须为正数")
    output_root = project_path(args.output_root)
    source_run = project_path(args.checkpoint).parent
    if output_root == source_run or source_run in output_root.parents:
        raise SystemExit("输出目录不能位于原 checkpoint 的运行目录内；请指定独立的 --output-root")
    output = output_root / (args.command + "_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    report = Report(output, args)
    print(f"OUTPUT_DIR={output}", flush=True)
    # Existing environment modules use project-relative paths internally.
    os.chdir(ROOT)
    try:
        runtime_info(report)
        config, _ = resolve_config(args, report)
        state = load_checkpoint(args, config, report)
        inspect_replay(args, report)
        check_assets(report)
        if args.command == "evaluate" and not failed_checks(report):
            evaluate(args, config, state, report)
        if args.command == "inspect":
            report.check("scope", "PASS", "完成 CPU 文件/结构检查；没有启动 MineDojo，没有执行策略。完整加载需运行 evaluate")
    except (Exception, KeyboardInterrupt) as error:
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.check("execution", "FAIL", f"{type(error).__name__}: {error}；详情见 error.txt")
    finally:
        before = report.data.get("checkpoint_file_before")
        if before:
            try:
                after = file_signature(project_path(args.checkpoint))
                report.data["checkpoint_file_after"] = after
                report.check("source_checkpoint_unchanged", "PASS" if before == after else "FAIL", "检查文件大小与修改时间（非内容哈希）；工具未写入原运行目录")
            except OSError as error:
                report.check("source_checkpoint_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
