# T03：真实目标片段与因果状态验收

状态：本地实现完成，用户服务器验收通过。助手只做静态文件阅读和差异核对，没有执行脚本、测试、训练或连接服务器。

已通过的运行：`build_20261006T115019`、`verify_20261006T115438_289349`；128 局、8192 段、69632 个下一动作标签，102/26 局分开；每种长度 512 段，16 类均有训练/留出样本。未来扰动与完整前缀对照最大误差 0，过去动作扰动最大差异 1.86406；严格加载的当前状态维度 5120，无优化器更新，源文件未变。旧 replay 无环境种子，仍只做 episode 留出；目标控制效果由后续 T05 验收。

T02 已通过用户服务器验收：构建 `build_20261006T110428`，独立检查 `verify_20261006T110657_826654`，最大特征误差约 `1.19e-07`；16 类均有训练/留出样本。图集有树木远近与不同局部视角，也有天空、地形、暗场景类别；可用于此阶段数据原型，可执行性仍待 T05。

## T03 做什么

沿用 T02 选中的 128 局及 102/26 局划分；每局按长度分层抽取最多 64 个不同片段，默认长度 1–16 个已记录的 `env.step`。目标是片段真实终点 RGB/heatmap 的冻结连续编码。目标类别另外保存，供候选模型做未来类别监督，不表示目标到达成功或任务完成。

举例，片段从第 20 帧到第 24 帧：

```text
状态帧       当前状态用到的真实历史       下一动作标签       剩余步数
obs[20]      obs/action[0..20]          action[21]         4
obs[21]      obs/action[0..21]          action[22]         3
obs[22]      obs/action[0..22]          action[23]         2
obs[23]      obs/action[0..23]          action[24]         1

四步使用同一个目标：GoalEncoder(obs[24].image, obs[24].heatmap)
```

回放是在执行动作后写入新的观测，所以 `action[t]` 是产生 `obs[t]` 的历史动作；要在 `obs[t]` 学下一动作，标签必须取 `action[t+1]`。reset 的 `action[0]` 是零占位，不能当作动作标签。当前状态需要原 encoder 的 `image/heatmap/obs_reward` 和历史动作；终点目标编码仍只看 RGB/heatmap。

未来目标作为指定条件是事后监督所必需的。未来图像、未来奖励和待预测动作不会送入当前 RSSM 状态。保存的片段不包含未来 RSSM 状态，也不把类别当作“已到达”的监督。

## 当前状态与因果前缀

`SegmentDataset[row]` 分开返回以下信息：

- `history`：真实 reset 到 `end-1` 的 RGB、heatmap、obs_reward、is_first 和 incoming action；没有终点图和 `action[end]`。
- `current_indices`、`loss_mask`：片段之前的帧只用于预热，片段范围 `[start,end)` 才计算动作损失。
- `next_actions`：`action[start+1:end+1]`。
- `goals`：逐步重复同一个终点目标向量。
- `remaining`：`end-start, ..., 1`。
- `goal_id`、`endpoint_index`：监督元数据，与当前历史输入分开。

`FrozenStateEncoder` 严格加载原 WM 的 encoder/RSSM，不构造 decoder、actor 或优化器。它从第 0 帧逐步 `obs_step`，每步读取当前帧和 incoming action，并把虚拟动作通道置零。对片段训练应先恢复 `history`，再取 `current_indices` 对应的状态；不能把片段起点强行当成 reset。

原型固定使用确定性 posterior mode 和 FP32；这使同一真实历史的恢复可重复，不等同于原 LS 策略采样潜状态的逐位复现。T04/T05 的训练与实际执行必须遵循同一状态约定。原 RSSM 的先验内部仍会抽样，但本工具恢复调用方 CPU/CUDA RNG，不引入优化器更新。后续可在 T04 对同一 episode 因果恢复一次并缓存，避免每个片段都重新预热。

## 数据边界

- 每个片段只来自一个完整真实 episode，目标严格在起点之后且最长 16 步。
- 复用 T02 的首次 `is_last/is_terminal` 或最大步数截断。可以把真实终点作为目标，不能从终止帧继续预测动作。
- 原始 `image/heatmap` 在 zoom 触发帧仍保留；不使用辅助 zoom 图、虚拟动作通道或训练采样器的 padding。
- 校验 onehot 真实动作、零 reset 动作、标志与输入长度；不合规回放直接失败，不补造标签。
- 保持 T02 的整局划分，不重新随机拆帧，避免同一局跨训练/检查集。
- 原回放没有每局环境种子，所以这一步只做 episode 留出，不声称种子泛化；报告预期 `[WARN] seed_holdout`。

片段会重叠，同一真实动作可能出现在多个片段中。`action_labels` 是监督条目数量，不是额外环境交互步数，也不是独立样本数量。

## 服务器命令

拉取代码，在原 `ls` 环境运行：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T00_CHECKPOINT=/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt
T00_BASELINE=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t00_outputs/evaluate_20261006T085822_184404
T02_LIBRARY=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t02_outputs/build_20261006T110428/goal_library.pt
T02_VERIFY=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t02_outputs/verify_20261006T110657_826654
T03_BUILD_DIR="$PWD/relevance_map/t03_outputs/build_$(date +%Y%m%dT%H%M%S)"

python scripts/t03_goal_segments.py build \
  --checkpoint "$T00_CHECKPOINT" \
  --baseline-dir "$T00_BASELINE" \
  --library "$T02_LIBRARY" \
  --t02-verify-dir "$T02_VERIFY" \
  --output-dir "$T03_BUILD_DIR" \
  --device cuda:0 \
  --horizon 16 --segments-per-episode 64 --seed 0
```

构建通过后，在同一终端独立验收：

```bash
python scripts/t03_goal_segments.py verify \
  --dataset-dir "$T03_BUILD_DIR" \
  --device cuda:0 --samples 64 --prefix-steps 24
```

若换终端，将 `T03_BUILD_DIR` 设成构建打印的实际 `OUTPUT_DIR`。验收默认从清单读取目标库、原 checkpoint 和源回放路径，无需重新指定。

两条命令都不启动 MineDojo，不执行策略，不加载 MineCLIP，不需要 `MINEDOJO_HEADLESS=1`，也不做 1M 训练或 BC 更新。未来启动 MineDojo 的命令仍显式带该前缀。

构建只加载目标库，读取原 checkpoint 文件记录，不加载 2.9 GiB 权重。验收会 CPU 加载一次原 checkpoint，严格提取 encoder/RSSM 后释放其余权重及优化器数据，再在 GPU 上恢复一段最多 25 帧的真实历史做因果探针。仅缓存少量解压 episode，不缓存整份原始 RGB，不保存全量 RSSM 状态。

## 预期结果

构建应有：

```text
[PASS] t02_input: ...
[PASS] baseline_input: ...
[SEGMENTS] episodes=.../128 segments=...
[PASS] real_segments: ...
[PASS] hindsight_targets: ...
[PASS] serialization: ...
[WARN] seed_holdout: ...
[PASS] source_inputs_unchanged: ...
[PASS] T03_SEGMENTS_BUILD; report=...
```

默认最多 8192 段。短 episode 的合法起终点对不足时会少于这个数，最终以诊断为准。

验收应有：

```text
[PASS] content_identity: ...
[PASS] reset_guard / terminal_guard / virtual_guard / endpoint_guard / horizon_guard: ...
[PASS] alignment_markers: ...
[PASS] probe_scope: ...
[PASS] episode_split_and_boundaries: ...
[PASS] endpoint_goal_<row>: ...
[PASS] real_action_alignment: ...
[PASS] baseline_input: ...
Encoder CNN shapes: ...
Encoder MLP shapes: ...
[PASS] future_invariance: ...
[PASS] prefix_equivalence: ...
[PASS] past_action_sensitivity: ...
[PASS] causal_state: ...
[PASS] no_training: ...
[PASS] source_inputs_unchanged: ...
[PASS] T03_SEGMENTS_VERIFY; report=...
```

关键检查是：

1. 用可区分的动作 marker 确认 off-by-one 对齐，并检查跨 reset、终止、虚拟图和越界会被拒绝。合成数据只在内存里用于验收。
2. 全量检查 128 局源动作和边界，按原种子重新生成采样索引，确认与保存片段一致。
3. 抽查 64 段，比较真实终点重编码、实际动作标签、统一目标和递减步数。
4. 在真实原 encoder/RSSM 上，只扰动检查点之后的图像、heatmap、obs_reward 和动作，当前及此前状态不变；把输入截为完整真实前缀后结果一致。
5. 改变过去动作，RSSM 状态有响应；冻结参数和调用方 RNG 没有变化。

这里验证数据和因果状态接口，没有训练 worker，没有验证目标可执行性。

## 输出与反馈

构建目录：`segments.npz`、`segments_manifest.json`、`alignment_examples.json`、`coverage.csv`、`diagnostics.json`、`report.json`。源 replay 保留在原目录，片段文件只保存索引和目标，不复制 RGB；以后使用这份数据仍需要原回放。

独立验收输出到另一目录，包含 `report.json`、`numerical_checks.json`，异常时保留 `error.txt`。内容 ID 校验片段数组和元数据；源 checkpoint/replay 使用大小与修改时间核对，不宣称做了完整文件内容哈希。

请反馈终端日志、构建 `diagnostics.json` / `alignment_examples.json` / `coverage.csv` 和验收 `report.json`。出错时附 `error.txt`；数值检查失败时再附 `numerical_checks.json`。无需发送 `.pt`、原回放或大型文件。

## 中文 Git 提交备注

```text
实现T03：构建因果目标片段并增加动作对齐与历史状态验收

- 沿用T02整局划分，采样1至16步真实片段及终点连续目标
- 明确当前观测对应下一动作标签，统一目标并递减剩余步数
- 保留从reset开始的真实前缀，分开状态输入与未来监督
- 严格复用原encoder/RSSM，以确定性状态恢复进行因果检查
- 增加边界拒绝、采样重现、未来扰动和过去动作响应验收
- 记录T02服务器通过及图集判断，更新TODO和服务器运行说明
```
