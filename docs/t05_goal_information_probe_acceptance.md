# T05：小型目标信息探针

2026-10-07。本地代码已实现，尚未执行 Python、训练或测试；服务器验收待用户运行。仍属于 T05，T06 未开始，旧 30 trial 继续暂缓。

## 这一步回答什么

前两轮里，有目标 BC 与独立无目标 BC 的留出表现几乎相同。这一步先回答：换成容易使用目标的小网络后，真实未来目标能否帮助预测下一动作？它能为后续修改 worker 提供依据，但不能证明给定目标可以被执行到达。

新增 `goal_information_probe.py` 和 `scripts/t05_goal_information_probe.py`，独立于可执行策略。原 LS-Imagine、T04 worker、目标库、候选模型及训练默认值均不修改。探针快照使用独立格式，T05 真实 worker 加载器会拒绝它。

复用已经验收的 T04 缓存，不重建数据，不需要新 trajectory。加载时会读取冻结 bundle 的 CPU 权重并核对内容身份，不实例化或运行原 encoder/RSSM/actor；不启动 MineCLIP 或 MineDojo。约 819 MiB 的 states.npy 用只读内存映射读取，原回放只用于输入身份与动作对齐核对。

## 两组怎么比较

每组都是一个从头初始化的小动作预测头：

1. 5120 维真实当前因果状态，投影到 128 维并做独立 LayerNorm。
2. 128 维真实未来终点目标，单独 LayerNorm；无目标组始终输入零向量。LayerNorm 没有可训练偏置，零目标不会变成有信息的条件。
3. 拼上 `remaining / 16`，经过 256 维隐藏层输出 12 类下一动作概率。

默认每头 724620 个参数。两头复制同一初始权重，各自拥有优化器；每次更新只抽一个真实训练 batch，同时供两组训练。无目标组也保留 remaining，避免把步数信息混入目标收益。相同原始输入下两头初始输出相同；实际比较时一组接收真实目标、一组接收零目标，因此不要求两组初始概率相同。

这是普通分类 softmax，没有原 actor 的 unimix 或策略优化目标。主要解释同一探针实验内有/无目标的差值；小头绝对 NLL 比旧 actor 更低，不能直接解释为策略更好。

固定 `uniform_rows`，每头先更新 400 次、batch=256、Adam 学习率 `3e-4`，两组均抽取 102400 个动作标签。采样分布沿用原条目均匀方式；没有候选模型的批次抽取，因此不声称与旧 T04 的每一个 batch 完全相同。新增真实环境步数为 0，重叠/重复标签不计作新数据。

训练只有真实 `obs[t] → action[t+1]` 的交叉熵。更换目标、零目标是评价条件，不给它们造动作标签，也不要求更换目标后动作必须不同。

## 服务器运行

拉取用户提交的本地修改后，在原 `ls` 环境运行。以下三步均离线，无需 `MINEDOJO_HEADLESS=1`。这些变量不依赖之前终端中的 T05 变量。

### 1. 输入与接口检查

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T04_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_INFO_CHECK="$PWD/relevance_map/t05_outputs/information_check_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_information_probe.py check \
  --cache-dir "$T04_CACHE" --t04-verify-dir "$T04_VERIFY" \
  --output-dir "$T05_INFO_CHECK" --device cuda:0 \
  --seed 0 --batch-size 256 --state-dim 128 --hidden 256 \
  --learning-rate 3e-4 --grad-clip 10
```

应看到：

- `accepted_real_cache / source_action_alignment / query_contracts`：输入身份、真实下一动作和整局划分一致；当前缓存应为训练 816、留出 832 个查询。
- `paired_initialization / normalized_input_paths`：初始权重一致，每头 724620 参数，两路输入 RMS 接近 1。
- `goal_free_invariance`：无目标组的输出不随目标变化。
- `remaining_guard / shape_guard / worker_use_guard`：错误输入与误用探针被预期拒绝，所以显示 PASS。
- `no_training / source_inputs_unchanged`，最后 `[PASS] T05_INFORMATION_CHECK`。

check 只初始化和检查新头，优化器更新数为 0，不保存可训练的旧模型副本。WARN 的 scope 是用途说明，不表示执行失败。

### 2. 固定短预算训练两组

第一步通过后执行：

```bash
T05_INFO_TRAIN="$PWD/relevance_map/t05_outputs/information_train_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_information_probe.py train \
  --cache-dir "$T04_CACHE" --check-dir "$T05_INFO_CHECK" \
  --output-dir "$T05_INFO_TRAIN" --device cuda:0 \
  --steps 400 --eval-every 100 --log-every 100 --save-every 100
```

训练配置从已通过的 check 读取，两组在同一个命令中运行，不需要分别训练。每次输出：

```text
[TRAIN PROBE] step=... goal_nll=... no_goal_nll=...
[INFORMATION] step=... queries=832 goal_nll=... no_goal_nll=... gap=... swap_modes=...
[LONG BUDGET] step=... rows=... gap=...
```

`gap = 无目标头 NLL − 有目标头 NLL`，正值表示真实目标帮助预测。NLL 越小越好；动作切换比例只是敏感度，不是到达率。

最终应看到 `[PASS] T05_INFORMATION_TRAIN`，报告 counters 为两头各 400 次更新、每头 102400 个抽取标签、优化器更新总数 800、`new_env_steps=0`。`sampling.json` 的直方图总和为 102400；同一个 batch 供两个头训练，直方图不重复累计为 204800。

快照仅包含两个新头、两套 Adam 状态、配置、输入身份、计数和 RNG。默认成熟快照约 17 MiB，预检查按每份约 18 MiB 估算，检查 latest、best_pair 与一次临时写入的空间，加 64 MiB 余量；输出通常远低于 0.2 GiB，不重复保存冻结 bundle 或 states.npy。磁盘实际可用量仍在运行时检查。

主要产物：

| 文件 | 用途 |
| --- | --- |
| `latest.pt` | 固定 400 次更新后的两个头，用于主比较与恢复验收 |
| `best_pair.pt / best_pair.json` | 两头完整留出平均 NLL 最低的同一步快照，作为次要诊断参考 |
| `evaluation.jsonl` | 第 0/100/200/300/400 次评价，观察收益和过拟合走势 |
| `metrics.json / diagnostics.json` | 最后一步的完整、查询、按局及按 remaining 结果 |
| `sampling.json / training.jsonl` | 实际抽样计数、共享 batch 身份及每次损失/梯度 |
| `query_metrics.csv / query_probabilities.npz` | 每个固定查询的真实标签、替换来源与动作概率 |

两头共同保存 best_pair，避免拿不同训练步数的两个“各自最佳”模型做主比较。第一轮只分析固定预算 latest，不因为中间某一点评价较好而自动延长更新。

### 3. 独立保存恢复验收

第二步通过后执行：

```bash
T05_INFO_VERIFY="$PWD/relevance_map/t05_outputs/information_verify_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_information_probe.py verify \
  --cache-dir "$T04_CACHE" --check-dir "$T05_INFO_CHECK" \
  --checkpoint "$T05_INFO_TRAIN/latest.pt" \
  --output-dir "$T05_INFO_VERIFY" --device cuda:0
```

应看到 `strict_load / roundtrip / next_update_equivalence / frozen_dependencies / source_inputs_unchanged`，最后 `[PASS] T05_INFORMATION_VERIFY`。会验证缓存、计数、采样协议、形状及误用快照的拒绝规则，并独立导出相同第 400 步评价结果。

verify 的两份内存模型各做一次真实 batch 更新，检查下一次抽样、权重、优化器与计数一致；不写回训练快照。`roundtrip_verification.pt` 标记为验收产物，不能 resume 真实训练，也不能用作控制策略。

## 结果如何决定下一步

主要读取 `metrics.json` 中：

- `validation_queries.row_weighted`：与先前诊断相同的 832 个去重查询。
- `validation_queries.per_episode / episode_comparison / episode_macro`：收益是否广泛出现，还是被少数局主导。
- `validation_queries.per_remaining / remaining_13_plus`：长预算是否有额外目标信息；13–16 汇总按实际查询条目加权，同时保留每个 remaining 的值。
- `validation_full.row_weighted / remaining_macro`：完整 14144 个留出条目及预算宏平均，单独解释其统计口径。
- `no_goal_minus_goal_nll / zero_goal_minus_goal_nll / swapped_goal_minus_goal_nll`：独立无目标对照、同一有目标头置零/换目标的差值。后两者会改变有目标头的输入分布，不能替代独立无目标对照。

若真实目标相对独立无目标有较清楚的留出收益，并且更换目标使这项收益减弱，可以优先尝试 worker 的独立目标残差模块；仍需后续重复种子和真实执行验证。只在训练集改善或 mode 切换很多，不能据此进入 T06。

若收益仍接近零，本次容量与预算下没有检出额外信息，下一步检查表示/行为覆盖。不能由一次小模型的负结果证明信息不存在，也不直接将原因锁定为数据不足。本探针改变了容量、初始化和融合方式，正结果支持“目标可被预测器利用”，不单独证明哪个因素是原 worker 的唯一问题。

工程验收不要求 gap 超过任意阈值。负结果也可能是有用的诊断；通过三步程序后先反馈分析，不跑 1M、不接高层或宏模型。

请反馈三步完整终端日志，以及训练目录中的 `diagnostics.json / metrics.json / evaluation.jsonl / sampling.json` 和 verify 的 `report.json`。不需要上传快照、冻结权重或原始轨迹。

## 中文 Git 提交备注

```text
feat: 增加T05小型目标信息探针与双组恢复验收

- 复用真实因果状态缓存，分别归一化状态和目标后预测下一动作
- 有目标与无目标头使用相同初始化、真实batch及固定更新预算
- 输出完整留出、去重查询、逐局和长预算目标收益及替换诊断
- 独立保存新模块和优化器，校验完整恢复与下一次更新一致
- 更新服务器运行说明和TODO，目标控制待验证、T06仍未开始
```
