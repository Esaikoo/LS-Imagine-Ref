# A v2 实现、数据流与消融实验报告

后续 B 阶段的改动、数据流、完整指标、消融入口与测试结论见 [B 阶段实现与验收报告](README_B_CN.md)。以下保留 A 阶段及此前日志修复的历史说明。


更新日期：2026-10-23。项目：`/home/user1/dl/projects/LS-Imagine-Ref`。环境：WSL Ubuntu-20.04 / conda `ls_imagine`。

## 1. 这次完成了什么

**A 已从“未来 RGB 缩略图 + 热力图代理评分”，补强为“未来 RSSM 特征对齐 + 探索事件预测 + 候选动作比较”的独立模块。** 它参加世界模型训练，也在训练交互和评测中改变真实动作。支持完全关闭、只加预测损失、完整开启三种消融模式。

目的：让智能体在当前目标证据较弱时，先比较“向左看、向右看、前进、继续原策略”等行动的预期观察后果，再决定是否花几步探索。是否提高采集成功率，仍需正式对照实验确认。

没有提交 Git，没有执行 1M 训练，也没有加入 B 的可学习持久记忆或 C 的可学习搜索切换策略。下面的 `full` 指本次 A v2 的全部功能，不代表整套 A+B+C 技术方案已经完成。

| 项目 | 原压缩包 A | 当前 A v2 |
| --- | --- | --- |
| 未来观察预测 | 下部 RGB 缩略图 | H=6 步后的 RSSM 后验特征，归一化后做 stop-gradient 对齐 |
| 探索事件 | 未来热力图代理证据 | 新表面区域、射线目标发现、重复观察三个概率 |
| 标签来源 | 图像/热力图代理 | 实际轨迹中的训练端射线命中、表面区域记录；缺失标签有独立掩码 |
| 行动比较 | 五个固定片段 | 五个片段 + 原 Actor 的六步续行动作候选 |
| 动作执行 | A 可以接管 | 保留接管，真实动作反馈 RSSM、存入 Replay |
| 控制可靠性 | 预测头可直接打分 | 预热、低证据门控、有效正负监督门槛、平分保留 Actor |
| 消融 | 开关不完整 | full、off、loss_only，另支持同一检查点关闭动作干预 |

## 2. 训练流和行动选择

### 新增训练支路

```text
真实图像/热力图 → 原 Encoder/RSSM → 当前后验特征 z[t]
实际动作 a[t+1:t+6] ────────────→ A 预测头
                                   ├→ 未来特征预测
                                   └→ new / discovery / revisit 概率
实际未来观察 → RSSM → detach(normalize(z[t+6]))：特征监督
训练端射线证据 → 事件标签 + 有效掩码 ──────────：事件监督
原 WorldModel 损失 + λA × A 损失 → 同一个世界模型优化器
```

默认：

`L_A = SmoothL1(预测特征, detach(归一化未来后验特征)) + (BCE_new + BCE_discovery + BCE_revisit) / 3`

特征项、事件项、总 A 项均有独立权重，默认 1。各事件按自身有效样本平均，不把未知标签当作负例。梯度影响当前 RSSM 表征；未来目标端停止梯度。原有重建、奖励、KL、长程想象和 Actor-Critic 目标保留，没有新增 C 的探索奖励。

只用**真实执行过**的动作及其未来结果训练，不伪造未执行候选的真值。窗口排除跨回合、跨终止边界及第 13 维虚拟跳跃动作；带 `is_zoomed` 辅助裁剪的真实观察仍可用。终止帧可以作为预测终点。

### 新增行动选择支路

```text
当前 RSSM 特征 → 五个固定六步片段 + Actor 的六步续行动作
→ A 预测事件概率 → 比较评分 → 保留 Actor 或执行探索片段
→ 真实环境 → Replay → 再训练
```

五个片段：左转、右转、前进、先前进后左转、先前进后右转。Actor 候选通过普通 RSSM 单步推进生成；Actor 获胜只执行当前一步，下次重新决策。固定片段获胜连续执行，回合重置清空未完成片段。

默认评分：`0.5p(new) + p(discovery) − 0.5p(revisit) − 0.05×动作成本`。成本是非 noop 动作比例，并非真实能耗/碰撞代价。未来特征预测辅助共享表征学习，排序直接使用事件概率。

默认 10000 个训练步后允许接管，当前热力图弱证据阈值 0.15。热力图只作门控，不再作为未来发现真值。训练空闲时以 0.25 概率触发，评测用确定性额度调度；训练触发后有 0.15 概率随机选片段收集监督，评测无此项随机探索。

每个事件头默认累计使用至少 128 个有效窗口、1 个正例、1 个负例后才参与评分。这是优化累计使用次数，可能重复采样，并非独立样本数量或校准保证。未就绪头权重为 0；全部未就绪时评测保留 Actor。冻结 A 学习或关闭事件损失时，不再执行为采集事件监督而设的 epsilon 随机片段。

0.25 的触发概率不代表 25% 的最终动作占比，一次选择可持续六步。RSSM 保留原采样行为，整个评测并非完全确定性。

## 3. 新增数据与标签边界

训练端使用 45 条视线射线（水平约 ±40°、垂直约 ±25°、距离 32 格）读取实际命中，并处理实体/方块遮挡。支持原木、沙、水、铁矿、羊毛任务；羊毛对应羊实体，水桶任务对应水源证据。

| 字段 | 含义 |
| --- | --- |
| `search_visible` | 当前采样射线命中目标；不等于完整图像中的目标可见性 |
| `search_discovery` | 上一帧无目标证据、当前有证据；不是采集成功，也不是全回合只计一次的首次发现 |
| `search_new` | 当前命中表面按 2 格区域划分，至少 25% 区域此前未见 |
| `search_revisit` | 当前命中表面至少 80% 区域在至少 H 步前已见；包含站着反复看，不要求先离开再返回 |
| 四项各自的 `*_valid` | 标签是否有依据；缺失射线、无表面、未知前帧等按对应规则置无效 |
| `search_label_version` | 本版为 2；旧数据缺失时按未知处理 |

共 **9 个每帧标量字段**。visible 用于审计及生成 discovery；A 直接预测另外三项。new/discovery 监督为“未来 H 步内任一发生”：有效正例即可判定发生，负例要求 H 步全部已知且为负。revisit 使用第 H 步终点标签。

表面集合只在训练环境进程内标注时存在，每回合重置，**不输入策略、不作为 B 的记忆、不在评测维护**。评测及 off 关闭新增射线传感器，所有有效掩码为 0。策略入口剔除 search_*，Encoder/Decoder 构建也排除这些字段，放宽输入正则不会自动引入特权标签。

## 4. 保存内容

输出目录默认位于服务器的 `/root/rivermind-data/mine/projects/tb_logs/<项目名>/<full|off|loss_only>/minedojo_<task>/seed_<seed>/<timestamp>/`，可用 LOG_ROOT 覆盖。

| 位置/字段 | 内容 |
| --- | --- |
| `train_eps/*.npz`、`eval_eps/*.npz` | 原有图像、热力图、实际动作、奖励、终止等，加 9 个标签标量；评测标签未知 |
| `log_search_override` | 当前动作是否来自固定探索片段 |
| `log_search_decision` | 当前步是否计算六个候选评分 |
| `log_search_macro_id` | -1 无选择，0–4 固定片段，5 Actor；历史补零帧不能单独当作新选择 |
| `log_search_scores` | 六个分数，只在 decision=1 时解释 |
| `log_search_predictions` | 六个候选 × 三个概率，顺序 new/discovery/revisit；只在决策帧解释 |
| `latest.pt` | 原模型、A 参数、监督累计数、优化器状态、实际步数、配置、A 版本 2 |
| `metrics.jsonl` 及原日志 | 始终保存原 LS-Imagine 与 A 的全部已产生指标；终端默认只显示四项核心指标 |

NPZ 动作和决策日志对应产生当前观察的实际动作，因此监督使用 a[t+1:t+H]。采样忽略 log_*，不将其作为模型输入。回合先保存元数据，再从内存回放的训练内容中清除这些日志。

不保存新增传感器原始射线、物体名称、坐标、表面集合；不保存未执行候选的真实未来，也不逐帧保存预测/目标 RSSM 向量。原图像和热力图仍保存。off/loss_only 不创建控制器，因此不会新增候选评分日志字段。

沿用完成回合才保存 NPZ 的机制，截止时未结束尾段可能尚未保存；checkpoint 保存当前模型。dataset_size 限制内存回放，不自动清理磁盘文件。每次启动创建新时间戳目录，不自动续训上次目录。

## 5. 按服务器习惯启动及做消融

主入口已改为 `scripts/train.sh`，保持你的 `sh ./scripts/train.sh 任务名 [额外参数]` 风格。默认开启 A v2，训练 1M 步，沿用原生默认训练随机种子 0，不额外固定模拟器种子。

在服务器项目根目录、已激活 ls_imagine 的终端中，仍用原来的命令：

```bash
TASK=harvest_log_in_plains
mkdir -p logs/train/$TASK
nohup bash -c "sh ./scripts/train.sh $TASK && shutdown -h now" \
  > logs/train/$TASK/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

脚本兼容 sh 和 bash。训练失败返回非零状态，所以后面的 && shutdown -h now 只在命令成功结束后执行；本次检查没有运行关机命令，入口自身也不调用关机。

模型、Replay、TensorBoard、JSON 指标默认保存到：

`/root/rivermind-data/mine/projects/tb_logs/<项目文件夹名>/<full|off|loss_only>/minedojo_<任务名>/seed_<种子>/<时间戳>/`

保留你的 LOG_ROOT 和 PROJECT_NAME 结构，多一层模式目录便于消融。终端输出仍在原来的 logs/train/$TASK/ 下。可通过环境变量 LOG_ROOT 覆盖根目录，本地 WSL 可使用 `LOG_ROOT="$PWD/logs/experiments"`。

| A_MODE | 含义 |
| --- | --- |
| `full`（默认） | 训练 A，允许 A 干预动作 |
| `off` | 完全关闭 A，使用当前修复后的基础流程 |
| `loss_only` | 训练 A 辅助预测，不干预动作 |

关闭 A 时只需在 nohup 前加环境变量：

```bash
A_MODE=off nohup bash -c "sh ./scripts/train.sh $TASK && shutdown -h now" \
  > logs/train/$TASK/off_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

只加预测损失时把 off 改为 loss_only。这些是分别选择的实验命令，一次训练结束会关机。

目标自动匹配：原木→log、沙→sand、水桶→water_bucket、铁矿→iron_ore、羊毛→wool。额外参数仍可直接追加，覆盖入口默认值：

```bash
sh scripts/train.sh harvest_log_in_plains --seed 1
sh scripts/train.sh harvest_log_in_plains --log_every 2000
sh scripts/train.sh harvest_log_in_plains --log_all_metrics True
```

### SEED 是什么意思

原来的 `SEED="${3:-0}"` 表示“取第三个位置参数；没有则取 0”，再传给 Python 的 --seed。**每次只运行一个种子，不会循环训练多个种子。** 原生 configs.yaml 默认 seed 也是 0，没有因这一行变成多次训练。

当前主入口不再要求第三个位置参数。直接传任务名就是 seed=0，需要换种子才追加 --seed 1。它控制代码中的随机数初始化，不等于自动更换 Minecraft 世界。模拟器/世界种子是另一个参数 env_seed，默认 -1 保留未额外固定的行为，需要固定时才主动传 --env_seed 0。

旧的 train_a.sh 兼容转发入口已删除。统一使用 scripts/train.sh，种子按 --seed 参数传入。

对比 A 和 off 时保持种子、模型大小、预算及评测条件一致。单次 seed=0 是一个实验样本，要确认稳定提升仍需后续多种子对照。

### 同一 A 检查点关闭动作干预

保留 A 架构，通过 search_a_control=False 只关闭控制器：

```bash
CKPT=/绝对路径/包含latest.pt的目录
python test.py --configs minedojo apm_a \
  --task minedojo_harvest_log_in_plains --target_item log \
  --agent_checkpoint_dir "$CKPT" --search_a_control False \
  --compile False --video_pred_log False --logdir ./logs/eval_a_control_off
```

对照开启时用 True，其余参数相同。模型尺寸及非默认 A 参数须与训练一致；test.py 不自动用 checkpoint 配置替换 CLI。小模型测试检查点须额外带 apm_a_smoke，只有 96 步，默认 10000 步预热下不会接管。

不要对 A checkpoint 使用 search_a_enabled=False，那会删除 A 参数并使架构不匹配。同检查点开关不替代独立训练 off，因为表征和采样历史已受 A 影响。

## 6. 所有指标完整保存，终端只显示重点

**修正：之前将“精简显示”错误地实现为“过滤保存”，会丢掉原 LS-Imagine 的模型、Actor/Critic 等指标。这个行为已删除。**

当前规则：
- `metrics.jsonl` 以及当前启用的 TensorBoard/W&B 后端始终保存训练日志器收到的全部标量，包含原 LS-Imagine 指标、全部 A 指标以及以后新加的指标。
- 原有 model_loss、actor_loss、value_loss、image_loss、heatmap_loss、KL、梯度范数、train/eval return、回合长度等照常保存；原有 TensorBoard 路径保持 scalars/指标名。
- 六项 A 诊断仍分组放在 diagnostics/ 下；其他 A 指标保持 A/ 下的路径。

- 已核对服务器旧无 A 实验实际出现的原指标名称；它们都会通过新版 Logger 原样保存。部分 zoomed/expl 指标只在相应分支运行时产生，日志器不会凭空生成没有计算的指标。

| 指标类别 | 新版保留的主要原指标 |
| --- | --- |
| 任务效果 | `train_return`、`train_length`、`train_success`、`train_first_success_step`、`eval_return`、`eval_length`、`eval_success`、`eval_first_success_step`、train/eval_episodes |
| 世界模型 | `model_loss`、image/heatmap/reward/intrinsic/jump/end_loss、zoomed_*_loss、`kl`、dyn/rep_loss 及 img/jmp 分项、prior/post_ent 及 zoomed 分项 |
| 策略与价值 | `actor_loss`、`value_loss`、`actor_entropy`、model/actor/value_grad_norm、value/target/imag_reward/imag_action/imagination_num 的统计量 |
| 训练运行 | `fps`、`dataset_size`、`update_count`、`EMA_005`、`EMA_095`、normed_target_*，以及实际启用模块产生的其他指标 |
| A 全部诊断 | `A/prediction_loss`、`A/future_loss`、`A/valid_windows`，以及 new/discovery/revisit 各自的 loss、valid、positive_rate、brier，train/eval_search_rate 与 search_decision_rate |

原无斜线名称在 TensorBoard 的 `scalars/` 分组；A 六项重点诊断在 `diagnostics/`，其余 A 损失在 `A/`。`model_loss` 保持原世界模型损失的口径，A 预测损失单独记录。这里恢复的是以后运行修复版时的记录机制，不是让缺失的历史数据自动出现。

- 默认终端只显示四项核心结果。`log_all_metrics=False` 不再影响任何指标的保存；True 仅让终端也显示全部指标。
- fps=True 时 FPS 也始终保存。离线评测的 offline_scalar 同样不再过滤，沿用原来向 TensorBoard/W&B 写入的方式。
- 按原记录频率保存，不意味着把每次梯度更新都逐条存盘；train.sh 默认每约 1000 步汇总记录训练指标。

日常先看：

| 核心指标 | 解释 |
| --- | --- |
| `eval_success` | 同预算、同评测条件下，相比 off 是否更高 |
| `eval_first_success_step` | 越小通常越好；失败记上限，须结合成功率看 |
| `eval_search_rate` | A 固定片段实际占多少动作，不是越高越好 |
| `A/prediction_loss` | A 总预测损失，下降不能单独证明任务效果提升 |

需要分析时，重点看下面六项，**这只是阅读建议，不是保存白名单**：

| 诊断指标 | 能帮助判断什么 |
| --- | --- |
| `eval_search_decision_rate` | 是否比较了候选；大于 0 而接管为 0，表示比较后保留 Actor |
| `A/future_loss` | 未来 RSSM 特征预测是否改善 |
| `A/discovery_loss` | 目标发现预测是否改善，须结合正例比例 |
| `A/valid_windows` | 是否有可学习的有效六步真实轨迹窗口 |
| `A/discovery_valid` | 有多少窗口具备已知目标发现标签 |
| `A/discovery_positive_rate` | 正例比例，长期为 0 时低损失可能只是在预测未发现 |

有效数是记录期间优化批次的平均值，可能重复采样，不是独立轨迹数。正例比例按批次统计，无有效标签的批次为 0，须与有效数共同解释。new/revisit 分项损失、Brier、原模型损失和梯度等其他指标也都会保存，不需要另开参数。

终端的成功率/接管率显示百分比，文件保留 0～1 原值，小损失保留有效数字。视频和报错照常输出。只改变日志持久化，不改变训练损失、动作选择、Replay 和模型权重。

### 已经训练到 160k 的实验如何处理

- 日志过滤本身不影响已经完成的参数训练；不能因此认定模型训练作废。
- 正在运行的 Python 进程已加载旧日志代码，仅同步修复文件或编辑配置不会让它自动生效。
- 应先确认服务器输出目录、最新有效 checkpoint 和现有回放，再安排应用修复；本次未停止、重启或启动服务器训练。
- 当前 train.sh/expr.py 每次启动创建新时间戳目录，直接重新执行原命令不会自动恢复到 160k。不能把重新启动误当作续训。
- 若历史指标当时没有写入 TensorBoard、JSON、W&B 或终端文件，就无法从 latest.pt 精确恢复早期模型损失、Actor/Critic 损失和梯度曲线。
- 已保存的完整回合 NPZ 可能允许重新计算部分回合回报、长度、成功率等统计，但要检查文件是否完整及能否匹配原训练步数；这不等于恢复历史优化损失，不应伪装成原始训练曲线。

效果仍需在同设置下对比 full/off，再通过 loss_only 和同检查点控制开关分析来源。日志齐全也不能替代消融实验。

### 2026-10-23 服务器只读核查与可恢复范围

实验目录：`/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/full/minedojo_harvest_sand/seed_0/20261022T191156`。以下是运行中的快照，不表示训练停在这些步数。

- 通过 SSH 密钥只读核查，训练进程 PID 6042 仍运行。服务器 Git HEAD 为 `bfbaa95`，检查时工作区干净；没有修改服务器代码、日志、回放或检查点，也没有停止/重启训练。
- 审计时 JSON 最新训练指标到 **159,500 步**；TensorBoard 同样只有 10 个标量标签，原模型、Actor/Critic 损失与梯度没有副本。终端文件也没有这些指标，配置中 W&B 关闭。
- 只读取 checkpoint 内的小型元数据，未加载约 3.08 GB 的权重张量到 CPU/GPU。检查点 `step=152500`，`search_a_version=2`；任务为采沙，seed=0，env_seed=-1，评测间隔 10,000 步。
- 旧日志过滤发生在记录端，未取消损失计算、反向传播或优化器更新；已有训练不能因为缺日志而判定作废。
- 只解压回放中的小型标量数组，没有读取图像张量。快照包含 **150 个训练回合、48 个评测回合**，所读字段等长且与文件名长度一致。
- 按原 tools.simulate 口径重算了训练回合的回报、长度、成功、首次成功步数及 A 比例；未给训练回合强行添加未经验证的全局步数。
- 48 个评测回合组成 **16 轮评测（2,500～152,500 步）**，每轮 3 个回合。依次分组后，成功率、首次成功步数、接管率和决策率四项均与已有日志匹配，因而可以补算每轮 `eval_return`、`eval_length` 和 `eval_episodes`。
- 恢复结果独立存为本地 `logs/test_apm_a_v2/remote_metrics_recovery_20261023.json`，带来源文件及核对依据，**没有回填原始日志或伪造原始曲线**。该文件位于忽略的日志目录，不随 Git 代码提交同步。
- 无法精确补回历史 model/actor/value loss、KL、梯度范数、A 未记录分项损失或 FPS。用现在的权重重算会是新的测量，不能代表当时的数值。

本轮本地修改仅涉及 `tools.py`、`configs.yaml`、`tests/test_search_a.py`、`docs/README_A_CN.md`，没有改变训练算法。由用户提交 Git 后同步服务器。运行中的旧进程不会热加载修复；如果要让本次实验后续记录完整，必须另行安排从有效检查点续训。**当前入口直接重跑会新建时间戳目录，不应拿原命令当作续训命令。** 本次未修改续训入口、未替用户提交 Git。



### 172.5k 步效果分析（2026-10-23 12:45，北京时间）

**当前结论：A 已接入训练和行动，但尚未观察到明确的任务收益；现有证据也不足以判定它在 1M 步最终无效。** 最新日志和检查点均到 172,500 步；这只是读数时的快照，服务器训练继续运行。

本次额外找到旧无 A 采沙实验：
`/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref-copy/minedojo_harvest_sand/seed_0/20261007T194422`。检查点确实没有 A 参数。其配置没有保存在 checkpoint 中，项目代码版本与当前不同，当前工作区 task_specs.yaml 也存在未提交修改，故只能作为历史参考，不能冒充严格的 full/off 消融实验。

两组都取相同的 18 个评测点（2,500～172,500 步），每点评测 3 个回合：

| 对比项 | 当前 A | 历史无 A |
| --- | --- | --- |
| 18 个检查点累计成功回合 | 3/54（5.6%） | 3/54（5.6%） |
| 最近 5 个检查点累计成功回合 | 1/15（6.7%） | 0/15（0%） |
| 172,500 步单次评测 | 0/3 | 0/3 |
| 18 轮首次成功步数平均值，失败记 1000 | 971.1 | 979.3 |

这是**跨不同检查点的学习过程描述**，不能把累计 5.6% 当作当前固定模型的真实成功率。单轮只测 3 次，1 次成功就会跳动 33.3 个百分点。近期多成功 1 次、首次成功均值略低，都不足以证明 A 稳定优于对照。

历史无 A 实验在 912,500～1,002,500 步的最后 10 轮平均成功率为 60%（18/30）。这里只说明该旧实验后期学出了效果；不能拿这个 1M 成绩直接压过当前 172.5k 的 A，也不能保证 A 后续一定追上。

**A 的动作机制确实在工作，但接管强度较低。**

- 最近 5 轮 eval_search_rate 的平均值约 **4.1%**；最后一轮为 **1.4%**。
- 最近 15 个评测回合共发生 **273 次候选比较**；171 次（62.6%）选择继续 Actor，102 次选择固定动作片段。
- 102 次固定片段中 81 次选择左转（79.4%），其余为右转 9 次、前进 11 次、前进后左转 1 次；这种偏向值得后续核查，不能仅凭这些次数断言策略塌缩。
- 接管率低不能单独判坏：A 的设计本来就只在低目标证据等门槛满足时比较候选。现有记录能证明执行过比较/接管，无法单独区分未触发时是热力图门槛、节奏门槛还是其他条件导致，也不应直接调高接管率来“制造效果”。
- 训练阶段还包含 epsilon=0.15 的随机宏探索，评测没有；训练中的成功和接管不能直接等同于评测收益。

**预测损失没有给出持续改善的强证据。** 下表按区间内已记录的批次均值再取平均；不是固定验证集上的误差：

| 区间 | A 总预测损失 | 未来特征损失 | 发现目标损失 | 发现目标正例比例 |
| --- | --- | --- | --- | --- |
| 0～25k | 0.5015 | 0.1580 | 0.0471 | 1.07% |
| 25k～50k | 0.5039 | 0.1933 | 0.0687 | 1.90% |
| 50k～100k | 0.4794 | 0.1994 | 0.0461 | 1.41% |
| 100k～150k | 0.4870 | 0.2082 | 0.0391 | 1.38% |
| 150k～172.5k | 0.4857 | 0.2105 | 0.0394 | 1.46% |

- 总预测损失主要在约 0.48～0.50 附近波动，不能只截取第一条较高读数来宣称显著收敛。
- 未来特征损失没有持续下降。RSSM 表征和回放分布同时变化，因此也不能把该上升直接解释成固定任务上的预测能力退化。
- 发现目标损失略降，但正例仅约 1.46%，类别不平衡很强；低损失可能受大量负例影响，不能据此判定模型善于识别发现机会。
- 每批平均有约 337～350 个有效六步窗口（按这些阶段均值），说明监督数据持续进入；这不是独立样本数。此次没有看到保存下来的 A 指标出现 NaN/Inf，但原模型/Actor/Critic 历史损失缺失，无法检查它们的早期变化。
- 缺失的 A/new、revisit、Brier 等历史标量仍无法补回；新版会完整记录它们。当前证据不足以判断哪个预测头造成候选选择偏差。

**是否重跑：不建议仅因缺少损失曲线就立刻把这次训练作废。**

1. 如果要判断任务成功率是否改善：现有 eval_success 仍然可用；本次也从 54 个评测回合恢复了 18 轮 eval_return、eval_length、eval_episodes，并逐轮用原有四项评测指标核对一致。可以把当前运行保留为探索实验。
2. 如果必须获得从第 0 步开始完整的世界模型/Actor/Critic/A 分项曲线：历史值无法重建，需用修复版另开实验。
3. 如果要给出正式 A 消融结论：建议之后用**同一修复版代码**分别运行 off/full，统一任务、模型、训练预算和评测条件；增加评测回合，并用多种子判断稳定性。旧实验只能作参考，重新跑 A 一组仍然不能解决严格对照的问题。
4. 当前服务器进程仍使用旧日志器。用户提交 Git 并同步后，旧进程也不会热更新。要让本次后半程补上完整日志，需要明确的检查点续训流程；**现有 train.sh 直接重跑会新建实验，不是续训**。本轮没有改启动方式、停止训练或在服务器修改代码。

图表与核对结果均保存在本地忽略目录 `logs/test_apm_a_v2/`：
- `server_a_effect_20261023.png`：四幅训练趋势图。
- `server_a_effect_analysis_20261023.json`：比较统计及最新 18 轮恢复的评测数据。
- `server_a_effect_snapshot_20261023.json`、`server_a_effect_details_20261023.json`、`server_sand_baseline_20261023.json`：只读分析快照。

本轮只补充分析文档及本地结果文件，没有新增训练代码变更；日志修复的 22 项测试结果仍见下节。

## 7. 本机验证记录

沿用包内 tests/test_search_a.py 与 scripts/test_a.sh，未另建测试启动脚本：

```bash
bash scripts/test_a.sh
bash scripts/test_a.sh --minedojo
bash scripts/test_a.sh --minedojo-off
```

- 此前 A 功能的 CPU 19 项测试通过：标签/遮挡/未知掩码、动作时序、边界、梯度及真实 WorldModel/Actor 更新、未训练头门槛、开关、特权隔离、旧回放、元数据、checkpoint 往返。冻结学习时停止随机探索的补充断言通过。
- Python 编译和 Bash 语法通过；已有 assert(tuple) 语法警告不影响本次测试。
- 开启 A 的真实 MineDojo + GPU 通过：小模型 96 训练步（含 32 预填充），前后各 32 步评测。checkpoint.step=96，参数及指标有限，4 个已保存回合的 NPZ 字段等长。
- 测试放宽门控和监督门槛，训练 epsilon=1，以覆盖动作接入；最终 eval_search_rate=0.96875、eval_search_decision_rate=0.21875，不代表默认设置效果。
- new/discovery/revisit 有效监督累计使用数 71/90/67，正例 52/0/20。真实短测没有采到发现目标正例；该正例与遮挡分支通过构造观测测试，真实正例质量仍需后续采样检查。
- 最后一次已记录 A/future_loss≈0.3031、A/prediction_loss≈0.7577，eval_success=0；不能据此判断 A 有效或无效。
- 关闭 A 的真实 MineDojo + GPU 测试通过：同样 96 步训练（含 32 预填充）、两次 32 步评测。checkpoint 无 A 参数、无 A 损失，新增标签掩码及接管率均为 0。
- 同一 v2 checkpoint 关闭控制器的独立评测通过：恢复训练步数 96，完成 32 步评测；接管率和决策率均为 0，评测标签掩码全为 0，NPZ 字段等长。
- 三种正式训练入口的参数展开已检查，模式配置、1M 预算、种子、日志目录及附加参数均正确；检查时没有启动正式训练。

此前的 diagnostics_cleanup_checks.log 验证的是旧版“只保存十项”行为，不能作为日志完整性的通过依据。本次删除保存过滤，并在现有 tests/test_search_a.py 增加持久化回归：无论终端开关如何，原模型、A、新增未知名称指标和 FPS 均写入 JSON/真实 TensorBoard；只含非展示指标的记录也保存；W&B 与离线接口同样不再过滤。W&B 使用模拟接口，无网络上传。本轮 CPU 测试共 22 项通过，Python/Shell 语法检查通过；未重跑真实环境或正式训练。测试日志为 logs/test_apm_a_v2/restore_all_metrics.log。

日志和检查点：

- `logs/test_apm_a_v2/cpu_final.log`
- `logs/test_apm_a_v2/integration.log`
- `logs/test_apm_a_v2/integration_off.log`
- `logs/test_apm_a_v2/standalone_ablation.log`
- `logs/test_apm_a_v2/full_20261023T005826_654996/minedojo_harvest_log_in_plains/seed_0/20261023T005828/latest.pt`

GPU 为 6 GB RTX 2060，短测用 apm_a_smoke 小模型。正式大模型显存、长时间稳定性、1M 成功率和五任务完整表现未验证。真实集成只跑原木任务，其他目标映射通过构造观测测试。

## 8. 兼容性、修复与后续边界

v2 模型和标签已变化，旧 RGB/热力图版 A checkpoint 不直接兼容，评测会在启动环境前明确拒绝。应开始新的 v2 实验，没有把旧 A 权重转换成已训练 v2 头的迁移工具。旧 Replay 标签缺失按未知补齐，仍可做未来特征学习。

运行修复包括：真实缩放观察误过滤、实际动作反馈、NPZ 等长及旧数组采样、长程想象空批次/单样本维度、空日志 NaN、训练超预算、独立评测恢复步数、任务配置深拷贝防止 train/eval 相互污染。训练入口按任务匹配目标，日志按模式归档。

仍需区分：

1. 稀疏射线没命中不等于整幅图像绝对无目标，尚无完整视觉实例分割/可见区域标签。
2. new/revisit 是可见表面几何近似，不是 B 的可学习持久探索记忆，动态目标也无长期有效性更新。
3. 固定片段/评分尚未成为 C 的学习式切换或探索价值策略；没有额外的所有候选 RSSM 一致性约束或不确定性模型。
4. 未执行分支正确性依赖泛化，未在同一真实状态重放所有候选获取真值，也未做独立验证集预测校准。

A 的标签、损失、控制器、开关保持独立，以后 B/C 仍需维护接口与消融条件，不能保证任意新增耦合自动可拆。

## 9. 文件改动索引

| 文件 | 职责 |
| --- | --- |
| search_a.py | 特征/事件预测、掩码、候选控制、版本验证 |
| search_evidence.py | 射线证据、遮挡、表面标注、旧数据兼容 |
| models.py | 世界模型联合优化、特权字段隔离 |
| expr.py | 策略接入、训练/评测环境、种子、预算及保存 |
| test.py | v2 加载、恢复步数、关闭控制器 |
| tools.py | 回放、元数据、接管率/决策率、运行修复 |
| envs/tasks/__init__.py、envs/tasks/minedojo/__init__.py | 配置隔离和传感器/标签包装器 |
| envs/tasks/base/ls_imagine_wrapper.py | 输出标量标签，剔除原始射线/坐标 |
| configs.yaml | A 参数、三模式、小模型测试配置 |
| tests/test_search_a.py、scripts/test_a.sh | 扩展原测试入口 |
| scripts/train.sh | 唯一的正式训练主入口，兼容你的 sh/nohup 风格 |
| docs/README_A_CN.md | 本报告，替换旧说明 |

### 清理结果与调用关系

- 已删除 `scripts/train_a.sh`：仅转发到 train.sh 的旧入口，无独立训练能力；测试脚本中相关语法检查也已移除。
- 保留 `search_a.py`：由 models.py、expr.py、test.py 调用，负责预测、控制和检查点检查。
- 保留 `search_evidence.py`：由环境包装器和 tools.py 调用，负责训练标签和旧 Replay 字段兼容。
- 保留 `scripts/test_a.sh` 与 `tests/test_search_a.py`：仍是 CPU/短程真实环境测试入口和测试代码；不在正式训练中自动运行，但不是废弃代码。
- 保留 `apm_a/apm_a_off/apm_a_loss_only/apm_a_smoke` 配置：分别由正式训练和短测使用。
- 历史实验日志、checkpoint 和用户原有文件不属于废弃代码，本次未清理。


## 10. 原版与 A 的同地图交替评测（2026-10-26）

旧的两组 20 回合评测配置中，`fast_reset=0`。项目的半快速重置包装器此时每次都会调用完整重置，不会走随机传送。但 `env.seed()` 只重置 Python 侧随机数，不能独自保证 Java 任务和 Minecraft 地形使用相同种子；旧评测也没有逐回合明确设置 `world_seed`，所以后续同编号回合不能作为严格的同地图比较。

现有 `evaluate_checkpoints.py` 新增配对模式，不新增重复启动脚本：

- 每张地图运行顺序固定为原版一次、A 一次，再进入下一张地图；20 张地图共 40 个计分回合。
- 每个计分回合使用新的 Python 进程、Minecraft 客户端及新生成的地图。关闭快速重置，同时固定世界种子、Java 启动种子和任务种子，避免前一模型挖掘或移动留下影响。
- 两个模型均固定晴天、正午且停止昼夜流转。任务目标仍是获得 1 个沙子，最多 1000 步，动作重复为 1；这是新评测条件，旧结果仅作历史参考。
- A 行动前核验世界/任务种子、出生位置、朝向、初始背包、生命状态和周围方块及元数据摘要；不匹配立即停止，不计入有效配对。
- 核验保留方块种类及元数据；仅忽略 Minecraft 1.11.2 树叶的 CHECK_DECAY 第 8 位动态检查标志，树种等其他位仍严格比较。原始元数据和原始摘要同时保存，可复核。
- 周围方块和坐标只用于脚本核验，不传给模型。模型仍接收原来的图像、热力图等观测。
- 开始计分前另用 2 个预检种子，分别启动两个模型检查起始场景，共 4 次重置，不计入 40 个计分回合。
- 本次正式地图种子为 `2026102600–2026102619`，预检为 `2026103600–2026103601`，与旧评测的配置种子范围不同。每张地图两个模型的策略随机种子也一致。
- 新模型使用已训练完成的 A 1M 检查点，自动恢复其 A 配置及训练步数，未启用 B。

在 `ls_imagine` 环境中先预检：

```bash
python evaluate_checkpoints.py \
  --checkpoint-root ckpt/latest \
  --paired-checkpoint-root ckpt/apm_a_1m_20261022T191156 \
  --tasks harvest_sand --runs 1 --episodes 20 \
  --seed 2026102600 --world-seed-start 2026102600 \
  --output evaluation_results/paired_sand_20maps_20261026 \
  --probe-only
```

确认通过后，使用相同命令去掉 `--probe-only` 即开始正式配对评测；已通过的预检会复用。直接不加该参数启动时，脚本也会先预检，通过后自动开始正式评测。中断后同命令续跑会保留已完成回合，重试未完成回合；修改评测代码或配置后应使用新的输出目录。

输出重点：

- `status.json`：预检/评测阶段、当前模型及地图种子，错误时明确标记失败。
- `preflight.json`：两张预检地图的条件一致性结果。
- `paired_summary.json`：已完成地图对、各模型成功率、双方成功/仅原版成功/仅 A 成功/双方失败。
- `order.jsonl`：真实执行次序。
- `original/`、`a_model/`：分别保留每回合视频、动作轨迹、热力图、起始图片/方块核验数据、任务 XML 和配置。

相同起始地图并不表示后续像素完全一致：动画纹理、实体运动，以及两个模型采取不同动作后的环境变化仍可能不同。比较以相同地形和出生条件为基础，不能要求两种策略的整段轨迹相同。

本地验证记录（2026-10-26）：两个预检种子各运行原版/A 的独立真实游戏重置，4 次均成功；两对的地形及出生条件全部匹配，两张地图彼此不同。起始画面对照保存在本次输出目录的 preflight/scene_pairs.png。场景差异拒绝检查、树叶状态规范化检查、任务 XML 检查和 Python 语法检查通过。正式 40 回合已在本地后台启动，其完成状态以 status.json 为准。


## 11. 同一 A 检查点关闭接管的定位测试（2026-10-26）

新增评测参数 --a-control off，仅将 search_a_control 改为 False，保留 search_a_enabled=True、A 网络及同一份检查点的全部参数，继续严格加载。没有重新训练，也不是切换回原版权重。该模式强制使用已有 A 开启结果为参考，检查相同检查点路径、其他搜索参数、任务与环境条件一致。

本次只新增 20 个 A 关闭回合，复用刚完成的 20 个 A 开启回合及原版结果。地图/策略种子仍为 2026102600–2026102619。每回合使用新游戏客户端，在行动前再次比较出生条件和地形；若不一致立即停止。

示例：

```bash
python evaluate_checkpoints.py \
  --checkpoint-root ckpt/apm_a_1m_20261022T191156 \
  --output evaluation_results/a_control_off_sand_20maps_20261026 \
  --tasks harvest_sand --runs 1 --episodes 20 \
  --seed 2026102600 --world-seed-start 2026102600 \
  --reference-output evaluation_results/paired_sand_20maps_20261026/a_model \
  --a-control off
```

本次为尽早观察明显失败案例，实际命令另传 --episode-order，顺序为：
18 6 10 4 11 16 19 13 17 0 1 9 2 3 5 7 8 12 14 15。
该参数只是回合索引的排列，不改变种子与地图的对应关系。原命令续跑时应保持相同顺序。

输出目录新增：

- a_control_comparison.json：已完成地图的开启/关闭配对比较。
- diagnosis_progress_CN.md：自动更新的中文进度表，明确区分部分结果和完整结果。
- checkpoint_identity_verified.json：检查点 SHA256 与此前已验证下载文件一致的记录。
- configuration_isolation_verified.json：实际加载后的策略配置只有接管开关变化。
- fixed_run_manifest.json：固定本次代码、检查点文件状态及运行顺序，防止续跑混用设置。

现有 scripts/test_a.sh 已包含相关开关和参考配置检查；26 项 CPU 测试及语法检查通过。真实游戏中的首个 A 关闭回合已确认保留 A 网络、严格加载同一份 1M 权重并进入交互。完整结果以输出目录 status.json 为准。

离线代码/轨迹检查另发现两个控制机制限制：宏动作启动后不再依据新热力图证据提前结束；候选宏动作没有抬头/低头纠正。旧 A 开启轨迹中，572 个延续执行步的当前热力图证据已经达到 0.15 阈值，另有 174 个接管步在朝上超过 45° 的视角下执行。热力图只是代理线索，不能把这些步数解释为已经真实看到沙子。原始统计保存在原配对评测输出的 analysis/control_mechanism_audit.json。

这些是已确认的机制限制，尚不能代替同检查点开关测试证明其对成功率的实际影响；本次定位期间未修改控制器行为，避免改变对照条件。

定位测试启动后，种子 2026102618 关闭接管仍 1000 步失败，攻击动作 0 次、朝上超过 45° 的步数占 43.7%。这是单张地图的阶段性结果，不能代表完整 20 图。种子 2026102606 的首次重载因一块草方块在加载时变为泥土而被起始核验拒绝，未执行策略动作；已保留原始审计文件。评测仅对这种零动作、纯地形差异的启动失败最多进行 3 次同种子新客户端重载，始终要求严格通过相同条件检查；真实超时、死亡和已执行过动作的错误均不因此重试。启动检查修复及已完成回合保留记录见 scene_retry_code_revision.json。
