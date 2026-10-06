# T05：低层目标学习的离线诊断

2026-10-07 本地实现；服务器待验收。助手仅静态阅读、修改和差异核对，没有本地执行 Python、测试、训练或环境，也未提交/推送。

## 当前进度

用户反馈 `repeated_check_20261006T174205` 和 `repeated_adopt_20261006T174237` 均通过工程验收。目标探针比较了同一世界两条真实参考中的 32 个缓存状态：更换目标平均概率 L1=0.011754，只有 1/32 个状态改变 mode 动作；目标 0 与零目标的 mode 动作 32/32 相同，目标 1 则 31/32 相同。两个真实目标的表示距离为 0.44455。工程接口可以使用，但目标控制能力没有通过。

此前 T04 在 400 次更新时，留出正确目标 NLL=1.7509，零目标=1.7572，打乱目标=1.7567。改成新目标后动作变化弱，值得检查数据监督和条件输入，尚不能只凭这一个世界确认原因。

**上次 T05 的后续 30 trial `evaluate` 暂缓，不需要现在继续。已完成的 check/adopt 和参考产物保留，不重做。当前仍是 T05 的验证与改进，T06 宏转移采集尚未开始。**

本次新增只读诊断，依次查看目标敏感度、数据分歧和目标输入路径。暂不改变 actor 结构或训练目标，不重建 T02/T03/T04 缓存，不延长 BC 或运行 1M。

## 本地新增入口

- `goal_learning_diagnostics.py`：去重、按持续时间采样、同 split/同 remaining/跨 episode 的目标替换、起点近邻候选和按局描述统计。
- `scripts/t05_goal_learning_diagnose.py analyze`：严格读取已通过 T04 verify 的实际 worker；不构造优化器、不加载 MineCLIP、不启动环境。读取已有 T03/T04 缓存和真实 replay，只保存小型报告、概率表及图集，不复制 checkpoint 或状态缓存。
- 可选独立 `no_goal` BC：核对相同数据、冻结依赖、训练配置和累计更新数及其独立 verify，只额外加载低层权重，复用冻结依赖。

### 扩大目标探针

每个训练 episode 最多抽 8 个查询、每个留出 episode 最多 32 个，去重键为 `(真实当前状态, remaining)`；默认最多 816/832 个查询。局数沿用 102/26 的原划分，不把重叠条目当作独立 episode 或世界种子。

各查询保持相同当前状态和剩余步数，比较真实未来目标、零目标、替换的真实目标及原 actor。替换目标来自另一局、同一 split、相同 remaining；预先随机抽最多 8 个候选，选表示距离最远的目标。选择不依赖模型动作或效果。没有合法候选则记录不支持并从替换统计中排除，原/零目标仍保留。

替换目标对应的原动作标签只用于检查似然变化，不作为新的训练监督，也不表示这个替换目标从当前起点可达。输出概率 L1、mode 变化、NLL 差值、目标距离、每局/每个 remaining 统计与逐条记录。按条目均值和按局均值分别报告；不提供 p 值或控制效果验收线。

### 数据中的行为分歧

检查全量缓存动作是否与 `obs[t] -> action[t+1]` 对齐，真实未来终点与剩余步数是否匹配。统计同一个记录历史状态出现多少 remaining 值及重叠标签，记录 remaining/动作的真实训练标签直方图，检查 16 步预算监督是否稀少。每个精确记录状态只有一条观察到的延续是旧数据的结构属性，本身不证明目标 BC 不可能成功。

每局最多选 16 个真实片段，按持续时间覆盖。编码片段起点的 RGB/heatmap，并读取缓存中的起点 RSSM 状态。只在同一 split、相同持续时间、不同 episode 之间，按两种起点 cosine 距离的均值寻找最多 2 个近邻。匹配完成后才统计未来目标距离、第一处分歧动作、动作序列差异和原 RGB MAE。

固定距离分组只是描述性检查，不是旧 T05 起点阈值。图集按 1–4/5–8/9–12/13–16 步轮流展示各组起点距离最近的候选，不按未来效果筛选，避免短预算例子占满图集。相似画面/历史不等于同一物理状态，没有位置、种子等完整遥测不能作相同起点声明；有限采样没有找到分支也不能证明全量数据不存在分支。

### 输入路径与梯度

从留出查询中均匀覆盖抽取最多 64 条，拆分 actor 第一层的状态、目标、remaining 输入贡献，并核对分解与真实前向一致。记录每元素 RMS、贡献比例、当前 BC loss 对输入的梯度，以及由第一层输出梯度解析得到的权重块梯度。

模型参数保持冻结，只有新建局部输入参与求导，不调用优化器，不写入参数 `.grad`，不更新权重。梯度只描述当前模型的当前留出损失，不能还原 400 次训练中的梯度过程；贡献较小也不自动证明应该放大目标，因为后续 LayerNorm/非线性会影响行为。

## 现在只运行这一条诊断

在服务器拉取本次代码后执行。它不启动 MineDojo，因此无需 `MINEDOJO_HEADLESS=1`。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_WORKER="$PWD/relevance_map/t04_outputs/train_early_fixed_20261006T155121/best_worker.pt"
T04_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_DIAG="$PWD/relevance_map/t05_outputs/goal_learning_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_learning_diagnose.py analyze \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_VERIFY" --output-dir "$T05_DIAG" \
  --device cuda:0 --seed 0 \
  --train-per-episode 8 --validation-per-episode 32 \
  --segments-per-episode 16 --gradient-rows 64
```

预期看到：

```text
[PASS] strict_inference_load
[PASS] diagnostic_contracts
[PASS] real_supervision_mapping
[WARN] independent_no_goal
[GOAL SENSITIVITY] split=train ...
[GOAL SENSITIVITY] split=validation ...
[PASS] heldout_goal_queries
[PASS] frozen_input_gradient_probe
[INPUT PATH] goal/state_contribution_ratio=... goal_input_gradient_rms=...
[BRANCH CANDIDATES] episodes=16/128 ...
...
[PASS] real_action_labels
[PASS] branch_candidates
[PASS] no_training
[PASS] source_inputs_unchanged
[PASS] T05_GOAL_LEARNING_DIAGNOSE
```

`independent_no_goal` WARN 是预期情况：本轮先读已有数据，还没有运行新的独立无目标 BC。`scope` WARN 也是正常的解释边界。PASS 验收程序和数据边界，不要求动作变化率达到某个值，目标忽略本身不会被当成程序异常。

本轮没有新的 checkpoint，没有新的环境步，没有参数更新。会顺序读取源 replay、计算少量冻结 CNN 特征及 actor 前向，不重算完整 WM 历史；实际耗时取决于磁盘和 GPU，不按 1M 训练估计。

请反馈终端日志、`diagnostics.json` 和 `branch_examples_validation.png`。细节在 `goal_sensitivity.json / input_path.json / branch_coverage.json / supervision_structure.json`，逐条数据在 `probe_rows.csv / probe_probabilities.npz / branch_pairs.csv / branch_segments.json`。若失败，反馈本次 `error.txt / report.json`，不要先重训或重建缓存。

## 根据报告决定下一步

- 若更多留出局仍对目标弱敏感，且起点近邻几乎没有不同目标对应不同真实动作的候选，优先检查/补充行为覆盖。近邻图集需要人工检查，不能只凭计数确认数据不足。
- 若有可检查的分支候选，目标贡献、梯度与敏感度仍很弱，再设计目标注入或训练约束的最小改动，保持现有初始化与对照可比较。
- 若离线目标使用证据仍与继续 BC 难以区分，补独立 no_goal 对照。它不能用同一个 goal worker 置零替代。
- 有更多目标学习线索后再决定是否恢复现有随机重复环境评估。真实行为尚未验收前，T06 继续保持未开始。

## 独立 no_goal 对照的备用命令：先反馈本轮报告再决定

以下适用于当前 400 次、seed=0、batch=256 等已知配置；命令从原 actor 初始化，**不传 `--resume`**。训练是离线 BC，不启动 MineDojo。累计更新预算和数据保持一致，不做 2000 次或 1M。

```bash
T04_NO_GOAL="$PWD/relevance_map/t04_outputs/train_no_goal_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --output-dir "$T04_NO_GOAL" \
  --device cuda:0 --conditioning no_goal --steps 400 \
  --seed 0 --batch-size 256 --candidate-hidden 512 \
  --learning-rate 3e-5 --candidate-learning-rate 3e-4 --grad-clip 100 \
  --eval-every 100 --log-every 100 --save-every 100

T04_NO_GOAL_VERIFY="$PWD/relevance_map/t04_outputs/verify_no_goal_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T04_NO_GOAL/latest.pt" \
  --output-dir "$T04_NO_GOAL_VERIFY" --device cuda:0

T05_DIAG_NO_GOAL="$PWD/relevance_map/t05_outputs/goal_learning_no_goal_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_learning_diagnose.py analyze \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_VERIFY" \
  --no-goal-checkpoint "$T04_NO_GOAL/latest.pt" \
  --no-goal-verify-dir "$T04_NO_GOAL_VERIFY" \
  --output-dir "$T05_DIAG_NO_GOAL" --device cuda:0 --seed 0
```

这里比较两个模型在固定 400 次更新后的相同诊断查询，不把最佳选择窗口自动视为相同。输出保存各 checkpoint 选择信息；正式实验还应统一模型选择规则。新增无目标训练将生成约三份紧凑快照及 verify 产物，沿用现有磁盘预检查，不重复保存冻结 WM。

## 中文 Git 提交备注

```text
feat: 增加T05低层目标学习离线诊断

- 扩展整局留出上的目标替换、置零及独立无目标BC对照
- 检查真实监督对齐和相似起点的行为分歧候选
- 记录冻结actor的目标输入贡献及当前损失梯度
- 更新T05验收进度，暂缓真实重复评估并保留T06未开始状态
```
