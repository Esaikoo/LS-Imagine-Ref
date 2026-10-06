# T05：真实目标控制评估与第一道行为验收

状态：本地实现完成，服务器待验收。助手只修改代码并静态核对，没有运行脚本、测试或训练，没有连接服务器。原 flat LS 训练入口和 T01 新模式运行保护保持原样；T05 使用独立诊断入口。

T04 的缓存、100/2000 次训练及恢复工程验收已通过。它还没有证明低层会追随目标：留出 worker NLL 在 400 次约为 1.7509，2000 次回升到 2.7347；候选模型也出现过拟合。因此先做少量真实配对测试，不继续增加离线更新或启动 1M。

2026-10-07 用户反馈早期 400 次模型及新快照恢复验收通过；T05 prepare 在同位置/同动作但不同画面时失败。已修复 `world_seed="0"` 的随机世界问题，见 [T05 种子修复与重跑说明](t05_seed_recovery.md)。只需重跑 check/prepare，不重训 T04。新 case 实验 seed=s，实际世界种子为 s+1，两个编号会分别保存及打印。

## 本次实现

- `goal_segments.py` 新增单帧真实状态更新：沿用 T03 的 FP32 posterior mode；输入是当前真实 RGB/heatmap/reward 和产生这帧的 incoming action。reset 清空历史，每步继续推进 RSSM。
- `goal_control.py` 和 `scripts/t05_goal_control.py` 提供 `check / prepare / evaluate`。目标保持固定，剩余步数逐步递减，最多执行 16 个真实动作，环境结束立即停止。
- 评估包含正确目标、目标置零、交换两目标、原 actor、参考动作重放；支持另行训练的 `no_goal` BC 对照。候选模型不参与目标选择。
- 保留真实视频、图像/热力图/状态/动作、位置/朝向、物品增减、任务成功、每步动作概率和目标距离，输出配对差异与终点图集。
- T04 训练新增 `best_worker.pt / best_candidate.pt / best_checkpoints.json`。分别按留出 worker NLL 和候选 CE 选择，各文件是当时的完整 T04 快照，权重、优化器与计数来自同一次更新。`latest.pt` 仍用于继续训练。

最佳保存只覆盖本次命令的评估窗口，包括恢复起点。它不会找回旧 2000 次训练中被覆盖的 400 次权重。`best_candidate.pt` 与 `best_worker.pt` 也不是已经拼接成一个模型；本次只使用最佳 worker，候选模型以后再处理。

## 评估目标和控制条件

默认 `prepare` 用原 actor 从固定起点预热 32 步，然后从同一起点用两组固定随机数采样原 actor 的 16 步动作，获得两个真实终点。目标是冻结 GoalEncoder 对终点真实 RGB/heatmap 的连续编码。不是任意挑选的聚类中心，也不是虚拟 zoom 图。

为复现起点，每个分支/对照新建任务与模拟器，设置相同世界种子和模拟器种子，使用显式初始位置/朝向、晴天、冻结时间，关闭 fast reset。原任务的观测、动作、奖励/成功条件、MineCLIP 和 Natural Zoom 包装继续使用。这是受控短程诊断，会改变重置、天气和时间条件，不能与 T00 的完整任务成功率直接比较。

实验 seed=0 会明确使用非零世界种子 `"1"`，case 1 使用世界种子 `"2"`，其余按同一固定映射。MineDojo 的 Java 生成器会把数字世界种子 0 当作随机分支，因此禁止直接传 `world_seed="0"`。case 的 `scenario.json` 和 benchmark 都保存映射；旧协议参考数据需要重新 prepare。

程序核对从 reset 到控制起点的整段历史：RGB、heatmap、reward、incoming action、结束标志、RSSM 状态、位置/朝向、物品和生命值。配对阈值预先写在 `PAIR_LIMITS` 和 `benchmark.json`：逐帧 RGB MAE≤1、99 分位绝对误差≤8；heatmap MAE≤0.25；状态相对 L2≤1e-4；位置≤0.05 格；朝向≤0.25°；reward≤1e-6；动作/物品/生命/标志一致。它们用于核对复现，未经服务器测量，不是已经校准好的通用阈值。

每个目标还会重放参考动作作正对照，核对整段实际结果。起点不匹配的 trial 不计入效果比较；正对照不匹配时，排除对应目标的整组比较并标记工程失败。这样可以分辨环境复现问题和低层控制问题。失败数据和已完成的视频仍保留，不自动筛选“容易的种子”凑通过结果。

同一个起点的两次目标测试使用相同动作随机数；默认采用分布最大概率动作 `mode`。改为 `sample` 时，使用独立的固定均匀随机数序列，不受 MineCLIP 或环境消耗随机数影响。所有模式都使用 T03 的确定性状态约定；`original` 是**相同状态约定下的原 actor 对照**，不是 T00 原 agent 的采样潜状态执行路径。

官方接口核对参考：[MineDojoSim 初始位置、世界种子及重置接口](https://github.com/MineDojo/MineDojo/blob/main/minedojo/sim/sim.py)、[HarvestMeta 支持的参数](https://github.com/MineDojo/MineDojo/blob/main/minedojo/tasks/meta/harvest.py)。没有把只属于模拟器的 `start_time/regenerate_world_after_reset` 等参数直接塞给 HarvestMeta；世界通过每个分支的新实例重建，实际匹配仍由观测验证。

## 第一步：补跑早期 worker 并独立验收

如果补跑保存时报 `No space left on device`，先按 [T04 磁盘恢复说明](t04_disk_recovery.md) 释放空间并检查存储修复，再补跑。新最佳快照共享 T04 缓存的冻结模型文件，必须保留该依赖。仅更换同一盘上的目录不能解决磁盘写满。

拉取本地改动后，在原 `ls` 环境执行。下面恢复 100 次 checkpoint，再做 **300 次离线更新** 到累计 400 次。无需重建 T02/T03/T04 缓存，不启动 MineDojo。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T04_100="$PWD/relevance_map/t04_outputs/train_20261006T125314/latest.pt"
T04_EARLY="$PWD/relevance_map/t04_outputs/train_early_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --resume "$T04_100" \
  --output-dir "$T04_EARLY" --device cuda:0 \
  --steps 400 --eval-every 100 --log-every 100 --save-every 100

T05_WORKER="$T04_EARLY/best_worker.pt"
T04_EARLY_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --output-dir "$T04_EARLY_VERIFY" --device cuda:0
```

应看到原来的 T04 训练/恢复验收通过，以及 `[PASS] best_checkpoints`。看 `best_checkpoints.json` 确认 worker 的实际更新步数：预计接近 400，不能强制认为一定是 400。必须 verify **实际要用于 T05 的文件**；对 `latest.pt` 的验收不能替代对 `best_worker.pt` 的验收。

如果暂时只想验证评估入口，也可以直接使用已经通过 verify 的 100 次模型及 `verify_20261006T125335_825066`；这不代表它已经学会控制。

## 第二步：离线检查在线接口

```bash
T05_CHECK="$PWD/relevance_map/t05_outputs/check_early_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_EARLY_VERIFY" \
  --output-dir "$T05_CHECK" --device cuda:0 --prefix-steps 32
```

这一步不启动 MineDojo/MineCLIP。应看到：

```text
[PASS] strict_inference_load
[PASS] incremental_episode_0
[PASS] incremental_episode_127
[PASS] incremental_causal_state
[PASS] execution_contract
[WARN] scope
[PASS] no_training: ... optimizer_updates=0 ... 新增真实动作步=0
[PASS] source_inputs_unchanged
[PASS] T05_OFFLINE_CHECK
```

它比较真实回放上的逐帧状态与 T03 完整前缀状态，检查 reset 清空、未来不影响当前状态、RNG、动作和剩余步数，以及拒绝不同历史的配对。边界探针只在内存，不能成为训练或评估目标。

## 第三步：只建立一个起点、两个真实目标

```bash
T05_BENCHMARK="$PWD/relevance_map/t05_outputs/prepare_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py prepare \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --output-dir "$T05_BENCHMARK" \
  --device cuda:0 --seeds 0 --prefix-steps 32 --horizon 16 \
  --reference-mode original_sample
```

应看到视频预检、`[CASE] ... matched_prefix=1`、`paired_references / real_targets / no_training / source_inputs_unchanged` 通过，最后 `[PASS] T05_BENCHMARK_PREPARE`。

查看 `targets.png`：每一行依次是控制起点、一个真实目标、对应灰度 heatmap。类别编号只是描述，两个目标即便属于同一聚类类别也可以有效；程序使用连续目标向量。

若两目标过近或起点已接近目标，程序会失败并保留 `goal_diagnostics.json` 和参考视频。默认距离下限 0.01 是诊断区分度门槛，不是“到达成功阈值”。不要为了通过而不断降低门槛。可以先看图和动作，判断原 actor 是否几乎不变、目标表示是否缺乏区分度。若只需要排查转向控制接口，可以**另建独立诊断**：

```bash
T05_TURN_BENCHMARK="$PWD/relevance_map/t05_outputs/prepare_turn_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py prepare \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --output-dir "$T05_TURN_BENCHMARK" \
  --device cuda:0 --seeds 0 --prefix-steps 0 --horizon 16 \
  --reference-mode turn_pair
```

`turn_pair` 的参考分支分别左转/右转前 8 步，再 noop 8 步。它是视角切换诊断，不能替代寻找/接近/砍树的能力验收，不能与默认原 actor 参考混合汇总。若预热提前任务结束，原数据保留，可根据错误原因减少预热长度并重建独立 benchmark。

若 `prefix_comparison.json` 显示位置、画面或状态不匹配，先把报告发回来修复重置/复现协议；不要据此评价 worker 控制效果。

## 第四步：评估早期 worker

```bash
T05_EVAL="$PWD/relevance_map/t05_outputs/evaluate_early_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --benchmark-dir "$T05_BENCHMARK" \
  --output-dir "$T05_EVAL" --device cuda:0 \
  --execution-policy mode --action-seed 0
```

一个起点 × 两目标 × 五条件，共 **10 个短 trial**：正确目标、目标置零、交换两目标、原 actor、参考动作重放。每个 trial 先重放 32 步预热，再执行最多 16 步。参考建立需 96 个显式真实动作，评估最多 480 个，合计最多 **576 个显式动作调用**，不做优化器更新。MineDojo 初始化/内部重置耗时和内部模拟器动作不计入这个计数；应另外看总运行时间，不能把 576 当作整个系统所有底层 tick 的开销。

每个条件需要新任务实例/世界初始化，所以实际等待时间可能主要来自启动和重置。先跑一个起点确认工程可用，再决定多起点预算。

工程完成应看到：

```text
[PASS] benchmark_identity
[PASS] video_preflight
[WARN] no_goal_control
[TRIAL] ... target=0 mode=... delta=... margin=... steps=...
...
[PASS] pairing: 匹配 10/10
[PASS] reference_replay
[PASS] real_execution
[PASS] artifacts
[WARN] behavior_acceptance
[PASS] no_training
[PASS] source_inputs_unchanged
[PASS] T05_CONTROL_EVAL
```

`[WARN] no_goal_control` 表示目前只是同一 worker 目标置零消融，还没有独立无目标 BC。即使工程 PASS，也**不会自动通过 T05 的行为验收**。

产物：

- `report.json`：工程检查、来源身份、真实动作计数、是否使用候选模型。
- `diagnostics.json`：各模式目标进展、任务成功、逐目标配对差异、两目标的初始概率 L1/动作差异、正对照结果。
- `trials.jsonl / trials.csv`：每个 seed/target/mode 的结果。
- `outcomes.png`：起点、真实目标与各条件实际终点并排。
- 每个 trial 的 `video.mp4 / trajectory.npz / events.json / control_trace.json / metrics.json / prefix_comparison.json`。视频为原生分辨率 RGB；NPZ 保留 WM 所读的 64×64 RGB/heatmap。事件包括位置/朝向/生命/物品与实际原生动作，不能把发出 attack 当成成功破坏方块。

`distance_improvement > 0` 表示结束画面更接近指定目标；`target_preference_margin > 0` 表示终点更接近自己的目标而不是另一个目标。距离为 `1-cos`；都只是目标表示下的诊断。`reference_pose_match` 另外检查距参考终点≤1 格、yaw/pitch 误差≤15°，也是诊断，不能代替语义任务完成。`task_success` 单独来自环境成功条件；短转向测试中它为 0 很正常。

## 第五步：同一 benchmark 比较 2000 次模型

不重建参考目标，只换 worker；它们共享相同冻结 WM/目标库/原 actor。先为 2000 次文件执行 offline check，再评估。这样可以判断过拟合是否已经影响真实控制。

```bash
T04_LATE="$PWD/relevance_map/t04_outputs/train_full_20261006T130644/latest.pt"
T04_LATE_VERIFY="$PWD/relevance_map/t04_outputs/verify_20261006T131507_375579"
T05_LATE_CHECK="$PWD/relevance_map/t05_outputs/check_late_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_LATE" \
  --t04-verify-dir "$T04_LATE_VERIFY" \
  --output-dir "$T05_LATE_CHECK" --device cuda:0 --prefix-steps 32

T05_LATE_EVAL="$PWD/relevance_map/t05_outputs/evaluate_late_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_LATE" \
  --check-dir "$T05_LATE_CHECK" --benchmark-dir "$T05_BENCHMARK" \
  --output-dir "$T05_LATE_EVAL" --device cuda:0 \
  --execution-policy mode --action-seed 0
```

如果改用 `sample`，早期和后期都另跑 `sample` 并使用相同 `action-seed`；不要拿早期 mode 与后期 sample 直接归因于训练步数。种子 0 的单次短测只用来筛查，不能支持稳定提升结论。

## 独立 no_goal BC 对照：有控制迹象后再补

目标置零/打乱属于推理消融。要排除“只是在原 actor 基础上继续 BC”造成改善，需独立无目标 BC；它不是重复跑环境训练。更新预算和训练超参数需与实际测试 worker 相同。以下只适用于早期 worker 确实为 400 次；若最佳为 300 次，把 no_goal 的累计预算改为 300，或改测已经独立 verify 的早期 `latest.pt`（400 次）。

```bash
T04_NO_GOAL="$PWD/relevance_map/t04_outputs/train_no_goal_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --output-dir "$T04_NO_GOAL" \
  --device cuda:0 --conditioning no_goal --steps 400 \
  --eval-every 100 --log-every 100 --save-every 100

T04_NO_GOAL_VERIFY="$PWD/relevance_map/t04_outputs/verify_no_goal_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_NO_GOAL/latest.pt" \
  --output-dir "$T04_NO_GOAL_VERIFY" --device cuda:0

T05_WITH_NO_GOAL="$PWD/relevance_map/t05_outputs/evaluate_no_goal_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --benchmark-dir "$T05_BENCHMARK" \
  --no-goal-checkpoint "$T04_NO_GOAL/latest.pt" \
  --no-goal-verify-dir "$T04_NO_GOAL_VERIFY" \
  --output-dir "$T05_WITH_NO_GOAL" --device cuda:0 \
  --execution-policy mode --action-seed 0
```

这次增加两个 trial，合计 12 个。程序检查缓存、冻结依赖、更新数与超参数，防止把不同数据/预算的 no_goal 当成公平对照。即使预算相同，按验证集选择模型与固定训练末尾的选择规则也可能不同，正式实验还应统一模型选择规则。

## 验收后如何决定下一步

先反馈早期模型的一起点试运行：`check/prepare/evaluate` 的终端日志和 `report.json`、`diagnostics.json`、`outcomes.png`、`targets.png`；有明显异常再看对应 `video.mp4 / control_trace.json / events.json`。无需下载 checkpoint、完整缓存或源 replay。

先确认配对/正对照全部通过，然后检查：

1. 同一起点换目标，行为是否有稳定的对应变化；仅动作概率不同还不够。
2. 两个终点是否各自接近被指定的目标，并有正的目标偏好；优于置零/交换目标对照。
3. 视频与位置、朝向、物品变化是否支持距离指标，还是仅纹理/视角相似。
4. 早期与 2000 次模型在同一 benchmark 的效果是否一致；有迹象再补独立 no_goal 和多个预先指定种子。

如果工程通过而 worker 几乎忽略目标，或所有目标都引向同样行为，返回 T02/T03/T04 查目标区分度、相同状态下监督的歧义和行为覆盖。若更接近目标却没有物理/任务意义，先查表示与目标设计。若重置/参考动作重放不通过，先修复 T05 复现协议。这些结果都不要求先跑 1M。

## 中文 Git 提交备注

```text
实现T05真实目标控制配对评估并补充T04最佳模型保存

- 新增逐帧因果RSSM状态更新及离线对齐验收
- 用同一起点的真实参考终点建立固定目标，隔离候选模型误差
- 加入正确目标、目标置零、目标交换、原actor和参考动作重放对照
- 支持同预算独立无目标BC对照，记录视频、物理事件和目标进展
- 核对重置历史与正对照，排除不能复现的组并保留失败诊断
- 分别保存完整最佳worker和candidate快照，更新TODO及服务器验收命令
```

助手不代为 commit/push；由用户提交、推送、服务器拉取后运行验收。
