# T00：checkpoint 检查与原策略基线验收

本地实现日期：2026-10-06。**尚未运行脚本、测试、训练或服务器评估，服务器验收待用户执行。**

T00 回答三个问题：旧 checkpoint 能否读取、与当前模型结构是否兼容、原策略能否在真实环境里执行并保存结果。这里不训练新模型，也不判断新方案是否提高成功率。

## 1. 运行位置和输出保护

在原训练使用的 Python 环境中运行；服务器项目根目录：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T00_CHECKPOINT=/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt
```

工具不会写回 `latest.pt`，不会恢复优化器训练，也不会向原运行目录写日志。每次运行会创建新的独立目录，并打印 `OUTPUT_DIR=...`。默认输出根目录：

```text
/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t00_outputs/
```

目录位于现有 Git 忽略范围。相对参数路径以项目根目录为基准。可以指定 `--output-root /其他独立目录`；工具会拒绝把结果放在原 checkpoint 的运行目录内。

## 2. 先检查文件，不启动 Minecraft

```bash
python scripts/t00_baseline.py inspect \
  --checkpoint "$T00_CHECKPOINT" \
  --task minedojo_harvest_log_in_plains
```

这一步只在 CPU 上读取 checkpoint 和少量已有 replay，不导入 MineDojo/MineCLIP，不启动环境，不执行策略。它仍需要原环境中的 PyTorch、NumPy 和 ruamel.yaml；checkpoint 中的优化器数据会一并读取，再释放，不会用于更新参数。

检查内容：

- checkpoint 文件存在，并包含 `agent_state_dict`；另外报告优化器状态是否存在。
- 找到 WM、actor 和 value 的保存项，输出完整名称、形状和类型。
- 核对 actor 输入/输出、RSSM 动作输入、GRU 和视觉输入卷积的关键形状。当前动作表有 **12 个动作**；RSSM 额外有一个 jump 标记，因此其输入多出的那一维不是第 13 个环境动作。
- 将 `torch.compile` 保存名称中的 `_orig_mod` 路径段规范化，名称冲突会报错；不会跳过网络层。
- 查找当前共享 MineCLIP 工厂实际使用的默认外部权重 `weights/mineclip_attn.pth`。文件存在只代表路径可用，实际权重加载在评估阶段验证。
- 默认发现 checkpoint 同级的 `train_eps/` 和 `eval_eps/`，每个目录抽查 3 个 npz，记录字段、形状、成功标记和文件数量。步数来自文件名估算，不代表 checkpoint 确切训练步数，也不代表已经检查了全部回放。
- 保存本次配置、任务配置、当前 Git 版本、运行环境和 checkpoint 读取耗时。

当前 `configs.yaml` 的 `defaults + minedojo` 配置下，关键形状预期为：

| 保存项 | 预期形状 |
| --- | --- |
| actor 输出权重 | `[12, 1024]` |
| actor 第一层输入权重 | `[1024, 5120]` |
| RSSM 动作输入权重 | `[1024, 1037]` |
| RSSM GRU 权重 | `[12288, 5120]` |
| 图像与 heatmap 的第一层卷积权重 | `[96, 4, 4, 4]` |

这些形状随原训练配置变化；若不一致，先核对配置，不能通过宽松加载绕过。

**正常验收应看到：**

```text
OUTPUT_DIR=.../inspect_<时间戳>
[PASS] checkpoint_format: ...
[PASS] _wm: ...
[PASS] _task_behavior.actor: ...
[PASS] _task_behavior.value: ...
... 关键形状匹配 ...
[PASS] external_mineclip: ...
[PASS] source_checkpoint_unchanged: ...
[PASS] T00_CHECKPOINT_INSPECT; report=.../report.json
```

正常退出码为 `0`。以上为输出示意，不是已经运行得到的结果。

**以下 WARN 可以出现：**

- `config_source`：旧保存格式没有完整运行配置。默认使用当前项目配置，不声称自动恢复了当时的所有设置。
- `training_step`：旧保存格式没有确切训练步数，不能证明 `latest.pt` 已训练满 1M，也无法恢复其原训练代码版本。
- `replay`：没有旧 replay，或抽查发现回放异常。没有 replay 不妨碍策略评估；后续数据量和质量还要单独验收。
- `optimizer_state`：缺少优化器状态不妨碍只评估策略。

因此 `report.json` 正常可能为 `"status": "passed_with_warnings"`，终端仍显示最终 PASS。**inspect 的 PASS 不等于完整模型严格加载已通过**，还要执行下一步。

检查产物：

| 文件 | 用途 |
| --- | --- |
| `report.json` | 状态、检查结果、路径、版本及读取耗时 |
| `checkpoint_shapes.json` | 全部保存项名称、形状和类型 |
| `resolved_config.json` | 本次展开的模型/运行配置 |
| `task_specs.json` | 本次任务、heatmap、progress、zoom 和终止配置 |
| `replay_summary.json` | 旧回放目录、文件数、步数估计及抽查结果 |
| `error.txt` | 出现读取/配置等异常时的详细错误；普通形状 FAIL 不一定产生此文件 |

## 3. 再评估原策略 3 局

inspect 没有 FAIL 后执行：

```bash
python scripts/t00_baseline.py evaluate \
  --checkpoint "$T00_CHECKPOINT" \
  --task minedojo_harvest_log_in_plains \
  --device cuda:0 \
  --seed 0 \
  --episodes 3
```

这里会启动 Minecraft/MineDojo，加载外部 MineCLIP，构建当前原始 `LS_Imagine`，检查**完整保存名称和形状**，再 `strict=True` 加载。原训练使用 `torch.compile` 时只规范化保存前缀；本次评估关闭编译以避免首次编译成本。

策略使用已有评估逻辑 `training=False` 和 `actor.mode()`；保留当前 heatmap、NACLIP progress、Natural Zoom、WM 和环境包装流程。不会执行预训练、随机预填充、梯度更新或 imagination 训练，也不会恢复优化器状态。模型构造时打印的 `Optimizer ... has ... variables` 是原类的初始化信息，不意味着发生训练。

收集器复用 `tools.simulate(is_eval=True)` 和原 ScoreStorage 逻辑。评估每局结束时会清空待处理项；T00 的外层保护确保随即重置，避免原 `real_done` 过滤在超时时继续执行已结束的一局。不会改动环境产生的观测终止标记或训练路径。

`--seed` 控制 Python、NumPy 和 PyTorch 的随机性，并保存到结果中；原环境工厂未显式传递 Minecraft 世界种子，本工具不承诺相同数字能逐像素重现同一个世界。继续复用原 fast_reset 流程，不把 3 局当成 3 个独立世界种子。

**正常验收应看到：**

```text
[PASS] strict_load: 完整权重严格加载；动作维度 12；没有跳过层
[EPISODE 1/3] success=... length=... return=... video=episode_001.mp4
[EPISODE 2/3] success=... length=... return=... video=episode_002.mp4
[EPISODE 3/3] success=... length=... return=... video=episode_003.mp4
[PASS] no_training: optimizer_updates=0；agent_training_step=0；参数版本未变化
[PASS] evaluation: 完成 3 局，成功 ... 局，success_rate=...
[PASS] source_checkpoint_unchanged: ...
[PASS] T00_BASELINE_EVAL; report=.../report.json
```

同时检查以下结果：

- `load_compatibility.json` 的 `missing_keys` 和 `unexpected_keys` 为 `[]`，`shape_mismatches` 为 `{}`。
- `summary.json` 中 `episodes=3`、`optimizer_updates=0`、`agent_training_step=0`。这里的 `0` 是本次评估未增加训练计数，不是说旧模型没训练过。
- `episodes.csv` 有 3 行结果，每行包含真实 `success`、回报、局长、首次成功时间（失败局为空）、zoom 帧数、视频路径和 replay 路径。
- `videos/` 有 3 个能播放的 MP4，`eval_eps/` 有 3 个对应的 npz。视频使用原策略输入的 64×64 RGB 帧、16 fps；能观察动作、场景变化与各局重置。
- 本任务每局最多 1000 次外层 `env.step`；若更早结束会记录实际长度。没有环境 error、非法动作或持续执行已终止 episode 的情况。
- 原 checkpoint 的大小、修改时间未变化；输出保存在新的评估目录中。

成功必须来自环境的 `info["success"]`，不会把 MineCLIP 高分或正 intrinsic reward 当作完成任务。MP4 编码失败、环境报告错误、缺少 success、超出终止步数或完整权重不兼容都会返回 FAIL，并保留错误信息及已经产生的独立结果。

评估另外保存 `effective_task_specs.json`、`eval_metrics.json`、`summary.json`、`load_compatibility.json`；`report.json` 记录环境/agent 初始化时间、评估耗时和本次配置。

**3 局成功率只可能为 0、1/3、2/3 或 1，不能要求它等于图中的约 80%。** 训练图的 `train_success` 与这里的原策略评估口径也不同。T00 的功能验收看加载、执行、重置和数据记录是否正确；0/3 不会被工具伪装成程序错误，但需结合视频排查旧策略/当前配置是否匹配，再决定是否继续做 T01。

需要更可靠的初步基线时，另外运行 `--episodes 30`（仍不训练，会创建新目录）。T00 没有要求先跑新的 1M 训练。

## 4. 配置或数据位置不同时

旧训练如果改过网络超参数，应向 inspect 和 evaluate 传入**相同**的补充参数，例如：

```bash
--run-config /绝对路径/原训练展开配置.yaml
```

也可按项目现有分组方式指定 `--configs minedojo debug`，或逐项覆盖：

```bash
--set dyn_deter=4096 --set actor.layers=5
```

`--run-config` 接受展开后的 YAML/JSON 字典；分组配置文件使用 `--config-file` 和 `--configs`。未知覆盖键会报错。评估使用独立日志、关闭编译和训练，并以 CLI 的任务、设备、种子及局数为准。

当前任务设置读取 `envs/tasks/task_specs.yaml`，并执行原 `expr.main` 的任务准备。`--run-config` 不能自动找回历史 MineCLIP 特征实现、历史 task_specs 或历史动作语义。结构兼容不证明这几项与原实验相同；要核对任务快照和代码版本。

replay 在其他目录时重复传入：

```bash
--replay-dir /绝对路径/train_eps --replay-dir /绝对路径/eval_eps
```

没有旧 replay 时先完成 3 局评估，新回放会自动生成。这几局不保证足够训练后续目标库或低层策略；数据采集量在 T02 再确定，不需要手工标注 trajectory。

## 5. 可选的低成本错误验收

无需修改真实权重或配置文件。以下命令预期退出码为 `2`，最终显示 FAIL，并写入各自新的报告目录：

```bash
# 文件缺失：应提示 checkpoint 不存在，不会启动环境。
python scripts/t00_baseline.py inspect \
  --checkpoint "${T00_CHECKPOINT}.does_not_exist"

# 人为制造配置不匹配：应打印关键形状 FAIL，不会启动环境。
python scripts/t00_baseline.py inspect \
  --checkpoint "$T00_CHECKPOINT" --set dyn_deter=1
```

验收缺少 replay 的提示：

```bash
python scripts/t00_baseline.py inspect \
  --checkpoint "$T00_CHECKPOINT" \
  --replay-dir relevance_map/t00_outputs/nonexistent_replay
```

若该目录确实不存在，应出现 replay WARN、提示 evaluate 自动生成新回放；在其他检查通过的前提下，最终仍为 PASS。此命令不会创建指定的 replay 目录。

## 6. 反馈内容与提交备注

反馈正常检查目录中的 `report.json`、`replay_summary.json`，以及评估目录中的 `report.json`、`summary.json`、`episodes.csv`。如失败，补充 `error.txt`（如有）、`load_compatibility.json`（如有）和终端末尾输出。不需要传输整个 checkpoint。

中文 Git 提交备注（仅文本，助手未执行提交或推送）：

```text
实现T00：新增checkpoint检查与原策略基线评估工具及服务器验收说明
```
