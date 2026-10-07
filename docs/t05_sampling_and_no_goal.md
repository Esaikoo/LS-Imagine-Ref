# T05：独立无目标对照与剩余步数均衡采样

2026-10-07 本地实现；用户反馈 A1–A4 和 B 均已通过服务器工程验收，但均衡采样未改善目标学习。助手只静态修改、阅读和差异核对，不执行 Python、训练或测试，不提交/推送。最新结论见 [均衡采样结果分析](t05_balanced_result_analysis.md)；下一步已实现的小型目标信息探针见 [服务器验收说明](t05_goal_information_probe_acceptance.md)。以下A/B命令保留复现，当前不用重跑。

## 当前依据与顺序

用户的 `goal_learning_20261006T181732` 已通过离线诊断工程验收。留出 832 个查询，替换目标改变 mode 动作约 2.88%，正确目标比零目标 NLL 低约 0.00503。第一层目标/状态贡献 RMS 比约 0.00612，是归一化/非线性之前的局部值，不能理解为整个网络只使用 0.61% 的目标。目标路径有作用，但目标控制仍未验收。

原训练池共有 55488 个动作标签：remaining=1 有 6528 条（11.76%），remaining=16 只有 408 条（0.735%）。均匀抽展开条目会使长预算的起始动作训练较少。新均衡采样让每种 remaining 的期望比例为 1/16（6.25%），16 步条目的期望权重为原来的 8.5 倍。它增加重复抽取，不增加真实数据、分支覆盖或环境步数；仍可能过拟合。

执行顺序分两轮：

1. 现在先做下文 A：CPU 采样检查 → 独立无目标 BC 400 次 → verify → 与已有 400 次 goal worker 在同一批查询上比较。保持原 `uniform_rows` 采样，判断改善有多少来自继续模仿学习。
2. 反馈第一轮结果后，再做 B：从原 actor 分别初始化均衡采样的 goal/no_goal，各 400 次，沿用相同数据、种子、超参数和固定更新预算。不能直接把新均衡 goal 与旧均匀 no_goal 当作目标条件收益。

## A 轮已验收结果：2026-10-07

用户反馈 `sampling_check_20261007T012105_672507`、`train_no_goal_uniform_20261007T012127`、`verify_no_goal_uniform_20261007T012218`、`goal_learning_no_goal_uniform_20261007T012311` 全部通过工程验收。原采样批次/RNG、共享冻结依赖、恢复后的下一次更新与冻结模块检查通过。独立 no_goal 累计更新400次，采样102400条；remaining=1/16实际采样12101/750条，约11.82%/0.732%，符合原采样规则。

附件 diagnostics.json 和 goal_sensitivity.json 的敏感度统计完全一致。832个固定留出查询中：

| 条件 | NLL，越低越好 |
| --- | ---: |
| 原 actor | 4.283060 |
| 正确目标 BC | 1.723798 |
| 同一目标 BC 推理时目标置零 | 1.728827 |
| 同一目标 BC 推理时替换目标 | 1.730948 |
| 独立无目标 BC | 1.724742 |

正确目标比独立no_goal仅低0.000944，约0.055%的NLL变化；26局中15局较好、11局较差。13–16步查询的平均差仅0.00000355；16步单独是独立无目标稍好约0.000363。没有稳定额外目标收益证据。置零/替换目标是在已经用目标训练的同一个模型上移除条件，不能代替独立训练的无目标对照。

完整14144条留出上的no_goal NLL=1.752972；此前goal为约1.7509。它与832个诊断查询的NLL不同，因为抽样和权重不同，不是对齐异常。原actor到BC的动作拟合改善基本能由继续模仿学习解释，不能换算成环境任务成功率。

下一步执行B：用已实现均衡采样做一次goal/no_goal各400次的短对照。A不用重跑，也不需要新代码。采样器CPU检查通过只证明比例/边界正确，不证明均衡训练有效。若B在相同留出查询、按局与长预算上仍没有目标收益，则停止继续调采样或延长BC，再评估目标注入与真实行为覆盖；T06及旧环境30trial继续暂缓。

仍处于 T05；旧 30 trial 环境评估暂缓，T06 未开始。T02/T03/T04 原缓存不用重建。以下均不启动 MineDojo，无需 `MINEDOJO_HEADLESS=1`。

## 实现与兼容性

- `goal_sampling.py` 只依赖 NumPy；`uniform_rows` 保留旧采样调用及 RNG 消耗顺序。`uniform_remaining` 先均匀选 remaining，再在对应训练池里均匀抽真实条目，采用有放回采样。
- 某种 remaining 缺少训练样本时，均衡模式明确失败，不用留出数据补齐。候选模型仍均匀抽片段起点；验证仍遍历原留出条目。
- `scripts/t04_goal_bc.py train --worker-sampling ...` 记录模式、协议版本与采样 RNG。旧快照缺省为 `uniform_rows`；精确 resume 禁止切换采样方式。
- `sampling.json` 保存训练池数量、期望/实际比例和相对权重。实际计数只覆盖本次 invocation 的更新，resume 之前的计数不伪造。`training.jsonl` 逐更新保存 remaining 直方图。
- `metrics.json` 和 `evaluation.jsonl` 新增 `validation.per_remaining` 和 `validation.worker_nll_remaining_macro` 等，分别表示每个预算的误差和各预算均值。原 `worker_nll` 仍按留出条目加权；`best_worker.pt` 仍沿用这一选择规则。本轮固定比较 400 次的 `latest.pt`，避免两组选择不同更新数。
- 两种采样会消耗不同数量的随机数，因此跨采样方式候选批次也会变化；同采样方式的 goal/no_goal 使用相同种子则动作/候选抽样序列一致。后续候选预测不作为本轮目标控制策略或效果依据。
- 共享冻结依赖和磁盘预检查保留，不重复保存冻结 WM；旧模型在推理时的内容身份不因选项归一化被修改。

## A：已通过，保留复现命令

拉取更新后，从项目目录依次运行。任何一步出现 FAIL，先反馈本次 error/report，不继续依赖它的下一步。

### A1：CPU 采样检查

只读取三个小文件：`cache_manifest.json / tables.npz / report.json`，不加载 `states.npy`、冻结模型、MineCLIP 或环境。此工具验收抽样分布与兼容性；完整缓存内容校验仍由 train/verify 执行。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_WORKER="$PWD/relevance_map/t04_outputs/train_early_fixed_20261006T155121/best_worker.pt"
T04_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_SAMPLING_CHECK="$PWD/relevance_map/t05_outputs/sampling_check_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_sampling_check.py \
  --cache-dir "$T04_CACHE" --output-dir "$T05_SAMPLING_CHECK" \
  --samples 65536 --seed 0
```

预期最后 `[PASS] T05_SAMPLING_CHECK`。`legacy_rng_equivalence`、`sampler_resume_equivalence` 和所有 guard 通过；guard 的 PASS 表示成功拒绝坏输入。uniform_remaining 中每种 remaining 的抽样比例接近 6.25%，不要求计数完全相等。采样检查结果只写小型 JSON，不产生训练 checkpoint。

### A2：独立无目标 BC，原采样方式

这是新训练，从缓存中的原 actor 初始化，**不传 `--resume`**。无目标模型仍保留 remaining 输入，和目标版唯一的条件差别是目标向量置零。

```bash
T04_NO_GOAL="$PWD/relevance_map/t04_outputs/train_no_goal_uniform_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --output-dir "$T04_NO_GOAL" \
  --device cuda:0 --conditioning no_goal --worker-sampling uniform_rows \
  --steps 400 --seed 0 --batch-size 256 --candidate-hidden 512 \
  --learning-rate 3e-5 --candidate-learning-rate 3e-4 --grad-clip 100 \
  --eval-every 100 --log-every 100 --save-every 100
```

预期 `[PASS] T04_BC_TRAIN`，step=400，new_env_steps=0；本次采样 102400 个标签。`sampling.json` 模式为 uniform_rows，比例接近原训练池。三份紧凑快照约 423 MiB，写入时还需一次临时替换空间；verify 另保存约一份快照，沿用自动空间预检查。训练/诊断没有最低 NLL 或敏感度的自动通过线。

### A3：独立恢复验收

```bash
T04_NO_GOAL_VERIFY="$PWD/relevance_map/t04_outputs/verify_no_goal_uniform_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_NO_GOAL/latest.pt" \
  --output-dir "$T04_NO_GOAL_VERIFY" --device cuda:0
```

预期 `[PASS] T04_BC_VERIFY`，step=400；新增 `sampling_mode_guard / sampling_protocol_guard / sampling_contract` 及已有 roundtrip、next_update_equivalence、frozen_ownership 全部通过。内存验收更新不写回真实训练快照。

### A4：同查询比较

```bash
T05_DIAG_NO_GOAL="$PWD/relevance_map/t05_outputs/goal_learning_no_goal_uniform_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_learning_diagnose.py analyze \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_VERIFY" \
  --no-goal-checkpoint "$T04_NO_GOAL/latest.pt" \
  --no-goal-verify-dir "$T04_NO_GOAL_VERIFY" \
  --output-dir "$T05_DIAG_NO_GOAL" --device cuda:0 --seed 0
```

预期 `[PASS] independent_no_goal`、`[NO GOAL BC]` 的 train/validation 输出以及最终 `[PASS] T05_GOAL_LEARNING_DIAGNOSE`。诊断只额外加载独立 worker，复用冻结模型。

请反馈 A1–A4 的终端日志，以及 A4 的 `diagnostics.json / goal_sensitivity.json / report.json`、A2 的 `sampling.json / metrics.json`。第一轮不需要再生成同一批分支图来证明分支充分。

重点看留出 `no_goal_bc_minus_correct_nll`：正值表示正确目标版的真实动作似然较好，负值表示独立无目标较好。结合按局、remaining=13–16 的结果、正确/置零/替换目标差距一起看。均值接近零或只由少数局贡献时，不能将原 actor 到新 BC 的改善归因于目标学习；即使有差距，也仍需真实行为评估。

## B：已通过工程验收，保留复现命令

入口已经实现，两组都从原 actor 初始化，不 resume 原 400 次模型，不改变网络或学习率。先按A1的变量设置定义项目目录与T04_CACHE，再依次执行下列命令；任一步FAIL先反馈，不继续依赖它的下一步。

用户反馈B的两次训练、两次verify以及 `goal_learning_balanced_20261007T013929` 均通过。两份实际采样记录一致，remaining=16抽取6316次，但同832个留出查询的goal/no_goal NLL为1.768369/1.768776，额外收益仅0.000407，26局13局稍好、13局稍差。16步goal NLL由原约1.677169升到1.841027。均衡模式暂不作为默认，下一步先检查目标信息可利用性，不延长训练。

```bash
T04_BALANCED_GOAL="$PWD/relevance_map/t04_outputs/train_goal_balanced_$(date +%Y%m%dT%H%M%S)"
T04_BALANCED_NO_GOAL="$PWD/relevance_map/t04_outputs/train_no_goal_balanced_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --output-dir "$T04_BALANCED_GOAL" \
  --device cuda:0 --conditioning goal --worker-sampling uniform_remaining \
  --steps 400 --seed 0 --batch-size 256 --candidate-hidden 512 \
  --learning-rate 3e-5 --candidate-learning-rate 3e-4 --grad-clip 100 \
  --eval-every 100 --log-every 100 --save-every 100

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --output-dir "$T04_BALANCED_NO_GOAL" \
  --device cuda:0 --conditioning no_goal --worker-sampling uniform_remaining \
  --steps 400 --seed 0 --batch-size 256 --candidate-hidden 512 \
  --learning-rate 3e-5 --candidate-learning-rate 3e-4 --grad-clip 100 \
  --eval-every 100 --log-every 100 --save-every 100

T04_BALANCED_VERIFY="$PWD/relevance_map/t04_outputs/verify_goal_balanced_$(date +%Y%m%dT%H%M%S)"
T04_BALANCED_NO_GOAL_VERIFY="$PWD/relevance_map/t04_outputs/verify_no_goal_balanced_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_BALANCED_GOAL/latest.pt" \
  --output-dir "$T04_BALANCED_VERIFY" --device cuda:0

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_BALANCED_NO_GOAL/latest.pt" \
  --output-dir "$T04_BALANCED_NO_GOAL_VERIFY" --device cuda:0

T05_DIAG_BALANCED="$PWD/relevance_map/t05_outputs/goal_learning_balanced_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_learning_diagnose.py analyze \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_BALANCED_GOAL/latest.pt" \
  --t04-verify-dir "$T04_BALANCED_VERIFY" \
  --no-goal-checkpoint "$T04_BALANCED_NO_GOAL/latest.pt" \
  --no-goal-verify-dir "$T04_BALANCED_NO_GOAL_VERIFY" \
  --output-dir "$T05_DIAG_BALANCED" --device cuda:0 --seed 0
```

两组 `sampling.json` 的计数应完全相同；每种 remaining 约 6400 个标签。用相同诊断种子比较 A/B 的原查询，并重点看长预算 NLL 和独立 no_goal 差距。整体旧权重 NLL 与均衡 macro NLL 分开解释；动作切换比例升高单独不算成功。若均衡后仍忽略目标，下一步再考虑目标注入或更有分歧的真实行为数据，不继续机械延长训练。

请反馈B的两份训练/verify日志、最后analyze日志及diagnostics.json / goal_sensitivity.json，两组各自的sampling.json / metrics.json。正确目标对独立no_goal的额外收益和长预算表现是本轮重点；不设置任意mode变化率作为自动效果验收线。

## 中文 Git 提交备注

```text
feat: 增加T05剩余步数均衡采样与独立无目标对照验收

- 保留旧条目均匀采样，新增按remaining均衡的真实训练采样
- 记录实际采样比例、逐预算验证误差和均值指标
- 兼容旧快照并校验采样协议，禁止resume切换采样方式
- 增加CPU采样检查和恢复检查，完善独立无目标公平比较
- 更新T05诊断结果与分轮服务器命令，T06仍未开始
```
