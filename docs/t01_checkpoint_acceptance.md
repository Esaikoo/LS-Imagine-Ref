# T01：模式、初始化与保存/恢复验收

状态：本地代码已实现；服务器待用户运行验收。助手只做了静态阅读与差异核对，没有执行脚本、测试或训练。

T00 已在服务器通过程序验收：`evaluate_20261006T085822_184404` 完成 3 局，成功 2 局，全部视频保存，完整权重严格加载，没有训练更新，原 checkpoint 的大小与修改时间未变化。这个小样本用于程序验收，不代表成功率已经复现。

## 本次实现什么

| 模式 | T01 的模块 | T01 可以做的事情 |
| --- | --- | --- |
| `flat_ls`（默认） | 原 WM、actor/value；没有新增可训练参数 | 原训练/评估路径；显式初始化；新版 checkpoint 保存/恢复 |
| `goal_worker` | 原模型 + 独立目标条件 worker | 原模型严格迁移、worker 前向、独立优化器及保存/恢复检查 |
| `hierarchical` | 原模型 + worker + 高层目标 logits 头 | 同上，另检查高层模块的优化器和保存/恢复 |

新增入口在 `long_horizon.py`，模式和配置在 `configs.yaml`，验收脚本是 `scripts/t01_checkpoint_check.py`，原训练入口的模式检查和 checkpoint 接口接在 `expr.py`。

目标库构建属于 T02，真实低层训练属于 T04，宏模型和高层优化属于 T07/T08。T01 的新模式尚不能进行真实目标控制或完整训练；通过 `expr.py` 或 agent 真实执行入口调用它们，会在执行前报告尚未实现的步骤。验收通过仅说明工程接口可以继续开发。

### worker 怎样复用原 actor

worker 输入为 `[当前 RSSM 特征, 目标特征, 剩余步数 / 最大步数]`。它复制原 actor 的所有层，仅扩展第一层输入。

当前配置下，第一层由 `(1024, 5120)` 变成 `(1024, 5249)`：原来的 5120 列原样复制，新增的 128 个目标维度和 1 个剩余步数维度初始权重都为零。这样，worker 在初始化时输出的动作分布应与原 actor 接近一致；后续才通过 T04 的真实训练学会利用目标。

worker 使用独立 Adam 优化器，hierarchical 的高层头也有独立优化器，均不包含原 WM、actor/value 参数。新模式默认 `freeze_wm=True`；目标原型也允许显式配置这一项，但 T01 验收使用冻结版本。`flat_ls` 保持原学习路径，不启用冻结 WM。

### 初始化与恢复的区别

- `checkpoint_load=initialize`：严格读取原模型的全部权重，规范化编译前缀，核对名称、形状、类型和共享参数一致性。新增模块按明确规则初始化；不导入旧优化器状态，也不猜旧模型的训练步数。原训练入口已收集的预填充步数保留为新运行计数。
- `checkpoint_load=resume`：只支持本次新增的 checkpoint 格式，恢复权重、原/新优化器、AMP scaler、交互与更新计数、调度器、value 更新次数、跳跃概率和 Python/NumPy/Torch/CUDA 随机状态。模式、任务、学习配置及目标编码器标识不兼容时拒绝加载。

旧 `latest.pt` 缺少可信的计数和 RNG 元数据，所以只能作为初始化来源，不能声称精确恢复。新版仍保留 `agent_state_dict` 和 `optims_state_dict` 两个原有键，并增加配置、模式、目标库、目标编码器标识、worker 版本及训练状态。文件通过临时文件写完后替换，避免把未写完的内容当作有效 checkpoint。

恢复模型和 RNG 不等于恢复 MineDojo 的世界状态。环境、正在进行的 episode、replay 采样器与待处理 `ScoreStorage` 不在 checkpoint 中。当前 flat 训练入口使用 `resume` 时还要求显式指定已有 replay 的 `offline_traindir`，避免重新预填充；这是已有数据上的继续运行入口，不能承诺中断前后逐帧一致。T01 核心验收只检查模块级恢复。

## 服务器运行命令

在服务器拉取这次代码后，使用原 `ls` 环境运行：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T00_CHECKPOINT=/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt
T00_BASELINE=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t00_outputs/evaluate_20261006T085822_184404

python scripts/t01_checkpoint_check.py \
  --checkpoint "$T00_CHECKPOINT" \
  --baseline-dir "$T00_BASELINE" \
  --device cuda:0
```

这条命令不启动 MineDojo，也不加载外部 MineCLIP，因而无需 `MINEDOJO_HEADLESS=1`。它读取 T00 已保存的配置、观测空间形状和最多 3 帧 replay，依次构建三种模式，并核对原 checkpoint 的路径、大小与修改时间是否仍符合 T00 记录。当前网络构造中有直接创建 CUDA 张量的旧逻辑，因此完整模型验收需要可用 CUDA。

脚本会加载一次旧 checkpoint，然后释放未使用的旧优化器状态，逐模式检查并释放模型。每个模式会保存一份完整权重的验收文件，总磁盘占用预计约 3 GiB，实际以输出为准；仍需给原 checkpoint 加载和保存/读取往返留足内存。无需运行 1M 训练，也不需要重新采集轨迹。

若资源有限，可以分别运行三个模式；三个都通过才算完成 T01 的核心验收：

```bash
python scripts/t01_checkpoint_check.py \
  --checkpoint "$T00_CHECKPOINT" \
  --baseline-dir "$T00_BASELINE" \
  --device cuda:0 --modes goal_worker
```

将 `--modes goal_worker` 分别换为 `flat_ls`、`hierarchical` 即可。每次运行都会创建新的输出目录。

## 验收时应看到什么

开头打印：

```text
OUTPUT_DIR=.../relevance_map/t01_outputs/check_<时间戳>
[PASS] baseline_input: ...
[PASS] scope: ...
```

三种模式应分别出现：

```text
[PASS] flat_ls/strict_migration: ...
[PASS] flat_ls/policy: ...
[PASS] flat_ls/roundtrip: ...
[PASS] goal_worker/strict_migration: ...
[PASS] goal_worker/worker_initialization: ...
[PASS] goal_worker/optimizers: ...
[PASS] goal_worker/optimizer_probe: ...
[PASS] goal_worker/roundtrip: ...
[PASS] hierarchical/strict_migration: ...
[PASS] hierarchical/worker_initialization: ...
[PASS] hierarchical/optimizers: ...
[PASS] hierarchical/optimizer_probe: ...
[PASS] hierarchical/roundtrip: ...
...
[PASS] source_checkpoint_unchanged: ...
[PASS] T01_CHECKS; report=.../report.json
```

`*_guard` 检查也应输出 `[PASS]`。这些行会附带预期拒绝原因，例如“模式不兼容”或“合成验收 checkpoint 不能恢复到真实训练”；意思是保护规则正确拦截了刻意构造的错误输入，并非验收失败。

具体检查：

1. 原权重按名称、形状、类型和数值完整迁移，所有原权重项一致，没有跳过层，也没有导入旧优化器。
2. `flat_ls` 没有新增可训练参数；真实回放前向给出合法 onehot 动作，不推进训练计数。
3. worker 第一层原列完全一致，新增列全零；4 组目标/剩余步数组合的初始动作分布与原 actor 在 `atol=rtol=1e-5` 下相符。
4. 新模式 WM 冻结，优化器参数互不重叠。验收仅用合成数据更新新模块一次，以产生非空 Adam 状态；原 WM、actor/value 参数版本不变。这不是实际 worker 训练，也不是学习效果验证。
5. 保存后刻意改动权重、清空优化器与目标元数据、重置部分计数，再从磁盘恢复；权重和优化器逐项一致，交互/更新/日志/调度/value 计数、目标库字段、编码器 ID、worker 版本及随机序列一致，恢复后前向结果一致。
6. 错误模式、错误权重形状、负计数、错误编码器、旧格式精确恢复，以及合成产物用于真实训练均被拒绝。
7. 原 checkpoint 文件大小与修改时间不变，输出位于独立目录。这个检查不是内容哈希；工具不写入原运行目录。

T01 的原优化器在初始化后为空，往返检查包括其参数组和空状态；非空 Adam moment 的往返由新增 worker/高层优化器验证。验收没有进行原模型训练。

## 输出文件和需要反馈的内容

```text
relevance_map/t01_outputs/check_<时间戳>/
  report.json
  flat_ls/
    resolved_config.json
    migration.json
    roundtrip_test.pt
  goal_worker/
    resolved_config.json
    migration.json
    roundtrip_test.pt
  hierarchical/
    resolved_config.json
    migration.json
    roundtrip_test.pt
  error.txt                 # 发生错误才生成
```

所有 `roundtrip_test.pt` 都有 `verification_artifact=True`；其中的训练计数和目标库是验收用的合成数据。它们不是正式初始化产物或已训练策略，请保留原 `latest.pt` 作为后续初始化输入；代码会阻止验收文件进入真实训练。

反馈完整终端日志和 `report.json` 即可；不需要下载或发送大型 `.pt` 文件。若失败，再提供 `error.txt` 和失败模式的 `migration.json`（已生成时）。助手根据反馈修复本地代码。

## 可选：再验证一局真实原策略

核心验收已经使用真实回放检查了原策略前向。如果还想确认这次代码合入后原策略能继续与环境交互，可以运行下面这一局；这会启动 MineDojo，必须带 headless 前缀：

```bash
MINEDOJO_HEADLESS=1 python scripts/t00_baseline.py evaluate \
  --checkpoint "$T00_CHECKPOINT" \
  --run-config "$T00_BASELINE/resolved_config.json" \
  --device cuda:0 --seed 0 --episodes 1
```

应有完整加载、1 局结果、视频保存、无训练更新和最终 `[PASS] T00_BASELINE_EVAL`。这一局成功或失败都不单独决定 T01 验收；它主要检查原环境执行路径是否正常。

## 中文 Git 提交备注

```text
实现T01：新增实验模式、严格权重迁移和checkpoint保存恢复验收

- 保持flat_ls默认策略，增加goal_worker与hierarchical模块入口
- 复用原actor权重并将新增目标和剩余步数输入置零
- 为新增模块建立独立优化器，支持冻结WM的原型配置
- 保存配置、目标元数据、优化器、AMP状态、计数器及随机状态
- 新增基于T00回放的离线服务器验收，拦截不兼容加载和合成产物误用
- 记录T00服务器验收通过并补充T01运行命令和验收说明
```
