# T04：目标条件低层 BC 与候选类别预测

状态：本地实现完成，服务器待验收。助手只做静态阅读、修改和差异核对；未运行脚本、测试、训练，未连接服务器。

T03 已通过用户服务器验收：`build_20261006T115019`、`verify_20261006T115438_289349`；128 局、8192 段、69632 个动作标签，102/26 局分开。未来扰动/完整前缀最大误差 0，过去动作扰动最大差异 1.86406；当前状态维度 5120。附件统计确认每种长度有 512 段，16 类均有训练与留出样本。这支持继续做离线训练，不代表目标已经可控。

## 这一步训练什么

低层输入 `当前 RSSM 状态 + 真实终点目标向量 + 剩余步数`，预测原 12 维 onehot 动作。比如片段 20→24，四条监督依次为：

| 当前状态 | 同一目标 | 剩余步数 | 下一动作标签 |
| --- | --- | --- | --- |
| state[20] | GoalEncoder(obs[24]) | 4 | action[21] |
| state[21] | GoalEncoder(obs[24]) | 3 | action[22] |
| state[22] | GoalEncoder(obs[24]) | 2 | action[23] |
| state[23] | GoalEncoder(obs[24]) | 1 | action[24] |

低层复制原 actor，扩展第一层的目标/剩余步数权重初始化为零，初始动作分布应与原 actor 一致；训练时更新整个新低层 actor。原 actor、encoder/RSSM 和目标编码器保持冻结。低层用动作负对数似然 BC 损失，先学习真实回放中已有的行为。

候选模型输入只有片段起点的当前状态，监督是终点目标类别，用交叉熵训练。它不读取指定目标、剩余步数、未来图像或未来奖励。输出表示“这批历史行为中，之后出现哪些类别”，混合了采样的 1–16 步长度；不是到达成功率，也不是高层策略。`GoalBCModel.propose` 返回前 4 类及目标库中的真实代表向量，供后续步骤使用。

低层的目标来自真实终点连续向量，候选的输出指向目标库代表向量。报告中的 `prototype_goal_nll` 用于检查这种目标替换的误差；它不能代替 T05 的实际控制测试。

## 先准备缓存，再做短跑

在服务器拉取代码后，沿用原 `ls` 环境。这三个命令均不启动 MineDojo，不需要 headless 前缀。以后启动 MineDojo 的命令仍须显式加 `MINEDOJO_HEADLESS=1`。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T03_DATA="$PWD/relevance_map/t03_outputs/build_20261006T115019"
T03_VERIFY="$PWD/relevance_map/t03_outputs/verify_20261006T115438_289349"
T04_CACHE="$PWD/relevance_map/t04_outputs/cache_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py prepare \
  --dataset-dir "$T03_DATA" \
  --t03-verify-dir "$T03_VERIFY" \
  --output-dir "$T04_CACHE" \
  --device cuda:0
```

`prepare` 自动读取 T03 清单中的原 `latest.pt`、T00 配置、T02 目标库及源 replay；无需重新收集或手工标注轨迹。仅这一步读取原 2.9 GiB checkpoint，提取所需冻结 encoder/RSSM 和原 actor；不导入旧优化器。之后训练读取缓存和冻结 bundle，同时检查原文件大小/修改时间。

每局从真实 reset 逐步恢复到最晚所需当前帧，再保存被片段使用的状态。状态采用 T03 的 FP32 posterior mode，单帧计算方式保持一致。重复使用同一帧的片段共享一行状态，目标和标签分开保存；不会把未来终点放入该帧的状态输入。缓存只需构建一次。每个状态约 `5120 × 4 = 20 KiB`；实际大小由唯一当前帧数量决定，程序打印 `size_mib`。需保留原 checkpoint、目标库和源 replay，以便来源检查与独立重编码。

成功应看到：

```text
[PASS] t03_input
[CACHE] unique_states=... feature_dim=5120 size_mib=...
[STATES] episodes=1/128 ...
...
[STATES] episodes=128/128 ...
[PASS] causal_state_cache
[PASS] action_supervision: 69632 个 ... 标签 ...
[PASS] frozen_no_training
[WARN] scope: ... 真实行为需 T05 验收
[PASS] source_inputs_unchanged
[PASS] T04_CACHE_PREPARE
```

产物：`states.npy`、`tables.npz`、`frozen_bundle.pt`、`cache_manifest.json`、`diagnostics.json`、`report.json`。本次样本预期为训练/验证片段 **6528/1664**，动作标签 **55488/14144**；没有新增环境步或优化器更新。状态数因片段重叠而小于动作标签总数，不应固定要求某个数值。

接着只做 100 次离线更新：

```bash
T04_SMOKE="$PWD/relevance_map/t04_outputs/train_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" \
  --output-dir "$T04_SMOKE" \
  --device cuda:0 \
  --steps 100 --eval-every 50 --log-every 20 --save-every 100

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" \
  --checkpoint "$T04_SMOKE/latest.pt" \
  --device cuda:0
```

训练默认 batch 256，低层学习率 `3e-5`，候选模型学习率 `3e-4`，候选隐藏维度 512，梯度裁剪 100。低层均匀抽取展开的动作标签，较长片段因此贡献更多条目；候选模型均匀抽取片段起点。两个优化器分别采样，仅使用训练 episode。重叠片段产生的条目数量不是独立轨迹数或新环境步数。

短跑末尾应出现 `[PASS] T04_BC_TRAIN`，报告计数应为：

```text
step=100
worker_updates=100
candidate_updates=100
worker_version=100
worker_labels_seen=25600
candidate_labels_seen=25600
new_env_steps=0
```

训练输出为 `latest.pt`、`training.jsonl`、`evaluation.jsonl`、`metrics.json`、`report.json`。从训练开始前就记录验证指标，结束后 `metrics.json` 保存最后一次评估，`evaluation.jsonl` 保留各次评估。训练集动作指标固定抽查最多 2048 条，留出集动作指标使用全量；候选指标两侧均使用全量片段。

独立 `verify` 应通过 `content_identity`、`real_supervision`、`causal_cache`、`strict_load`、`cache_guard`、`counter_guard`、`artifact_guard`、`shape_guard`、`roundtrip`、`next_update_equivalence`、`frozen_ownership`、`source_inputs_unchanged`，最后显示 `[PASS] T04_BC_VERIFY`。它默认重算 2 局完整因果历史，并对保存恢复的两份模型各做一次内存更新，检查下一次更新一致性。验收输出的 `roundtrip_verification.pt` 带验收标记，禁止用于真实 resume；原训练文件不会被覆盖。

## 如何看指标

| 指标 | 含义与用途 |
| --- | --- |
| `worker_nll`、`worker_accuracy` | 给定真实终点目标，下一动作拟合效果；同时看留出集 |
| `original_nll`、`original_accuracy` | 原 actor 在相同确定性状态上的预测对照，不是原环境成功率 |
| `zero_goal_nll`、`shuffled_goal_nll` | 同一训练低层移除或打乱目标后的误差；目标输入消融诊断 |
| `goal_sensitivity_l1` | 正确与打乱目标的动作分布差异；有差异不等于控制正确 |
| `prototype_goal_nll` | 把连续真实终点换成对应类别代表向量后的误差 |
| `worker_entropy`、动作标签/预测直方图 | 是否只预测少数常见动作或分布异常 |
| `candidate_ce`、`candidate_accuracy` | 未来类别交叉熵及 top1 准确率 |
| `candidate_top4_recall` | 前 4 类覆盖实际未来类别的比例，对应后续 4 个候选 |
| `candidate_macro_recall`、逐类召回 | 防止多数类别掩盖少数类别失败 |
| `candidate_majority_baseline` | 始终选训练集最多类别的留出表现；判断是否只学了类别频率 |
| `candidate_frequency_top4_baseline`、`candidate_frequency_ce` | 固定选择训练集最常见 4 类的覆盖率，以及训练频率分布的交叉熵；与模型 top4/CE 直接比较 |

100 次更新只检查运行、有限损失、冻结约束与恢复，不硬性要求误差下降多少。工程通过后再看较长训练：留出 NLL 改善、有目标依赖、候选覆盖优于简单频率对照，才支持继续测试。若训练误差下降但留出误差变差，或正确目标/打乱目标几乎无差别，应反馈结果分析；不要立即加大真实训练预算。静态指标即便改善，也需 T05 检查真实行为。

`zero_goal_nll` 是同一模型的推理消融，不能替代“独立训练无目标模型”的公平对照。后者可另建输出目录运行 `train --conditioning no_goal --steps 2000`，其余参数相同；它保留当前状态和剩余步数，仅移除目标。初轮工程短跑不要求同时做这组。

## 工程通过后继续离线训练

复用同一缓存与短跑 checkpoint，恢复到累计 2000 次更新，新增 1900 次更新：

```bash
T04_TRAIN="$PWD/relevance_map/t04_outputs/train_full_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" \
  --resume "$T04_SMOKE/latest.pt" \
  --output-dir "$T04_TRAIN" \
  --device cuda:0 \
  --steps 2000 --eval-every 100 --log-every 20 --save-every 100

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" \
  --checkpoint "$T04_TRAIN/latest.pt" \
  --device cuda:0
```

`--steps` 是累计更新目标；恢复时未指定的训练参数自动继承 checkpoint。显式改变 batch、学习率、条件模式、隐藏维度、种子等会拒绝精确恢复。恢复要求同一缓存 ID、冻结模型/目标库和相同计算环境信息（PyTorch、设备类型，GPU 时还包括 CUDA/cuDNN、型号与计算能力，CPU 时包括线程数）；日志间隔可以变化。这仍是离线 BC，没有新增真实交互，不是 2000 环境步，也不是 1M 训练。

T04 `latest.pt` 保存新 worker/candidate、独立优化器、计数、采样器/RNG，以及冻结 encoder/RSSM、原 actor 和完整目标库。它使用独立 T04 格式，不含原 WM decoder/value 或旧优化器，不能传给原训练入口/T00 当作完整 agent。真实目标执行循环、高层与宏模型由 T05 及后续接入；原默认 flat LS 路径保持现有行为。

## 反馈和中文提交备注

先反馈 `prepare/train/verify` 日志、缓存 `diagnostics.json`、训练 `metrics.json`、`evaluation.jsonl`，及三个目录的 `report.json`；有报错附 `error.txt`。不需要下载缓存状态、原 checkpoint 或轨迹。

```text
feat: 实现T04目标条件低层BC与候选类别预测训练

- 复用T03真实片段及整局划分，每局恢复一次冻结因果状态并共享缓存
- 从原actor初始化目标条件低层，独立训练当前状态到未来类别的候选模型
- 新增prepare/train/verify与累计更新恢复，保存新优化器、计数、采样器及RNG
- 验收真实动作对齐、状态重编码、冻结约束与恢复后下一次更新一致性
- 输出原actor、目标置零/打乱及代表目标对照指标，保留无目标BC入口
- 更新T03验收记录、T04服务器命令及阶段边界，不改默认flat训练路径
```

以上为提交备注文本，助手未执行 Git 提交或推送。
