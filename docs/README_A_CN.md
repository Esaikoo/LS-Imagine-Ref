# A v2 实现、数据流与消融实验报告

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
| `metrics.jsonl` 及原日志 | 默认四项核心 + 六项诊断；详细模式恢复其余原有指标 |

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

## 6. 四项核心结果 + 六项必要诊断

**只存四项不够解释原因。** 成功率告诉我们总体效果，接管率和总损失无法区分“缺少发现正例”“未来特征学得不好”“候选比较后总选 Actor”等情况。因此现在把显示与保存分开：终端仍只显示四项核心指标，JSON 和 TensorBoard 额外保存六项诊断。

日常先看：

| 核心指标 | 解释 |
| --- | --- |
| `eval_success` | 最重要：同预算、同评测条件下，比 off 对照更高 |
| `eval_first_success_step` | 越小通常越好；失败记上限，须结合成功率看 |
| `eval_search_rate` | A 固定片段实际占多少动作，不是越高越好 |
| `A/prediction_loss` | A 总预测损失，下降不能单独证明任务效果提升 |

出现异常或需要分析时，在 TensorBoard 的 `diagnostics/` 分组看：

| 额外保存的指标 | 能帮助判断什么 |
| --- | --- |
| `eval_search_decision_rate` | 是否实际比较了候选；它大于 0 而接管为 0，表示比较后保留了 Actor |
| `A/future_loss` | 未来 RSSM 特征预测是否在改善 |
| `A/discovery_loss` | 目标发现事件预测是否在改善，必须结合正例比例 |
| `A/valid_windows` | 是否存在可学习的有效六步真实轨迹窗口 |
| `A/discovery_valid` | 其中有多少窗口具备已知的目标发现标签 |
| `A/discovery_positive_rate` | 有效发现标签中正例比例；长期为 0 时，低损失可能只是一直预测未发现 |

例如：discovery_valid 有值而 positive_rate 长期为 0，说明有标签但没采到发现正例；valid_windows 为 0，说明真实预测窗口不可用；两个有效数都有值且正例存在，再观察 discovery_loss 才更有意义。这两个有效数是记录期间优化批次的平均值，可能重复采样，不是独立轨迹数量。正例比例按批次统计，无有效标签的批次为 0，须与有效数一起解释。

终端默认四项；metrics.jsonl 使用原指标名保存全部十项（只有实际产生的指标才写入），TensorBoard 保持原四项路径，额外六项放在 diagnostics/ 下，例如 diagnostics/A/discovery_loss。W&B 启用时采用相同诊断分组。日志分组不改变损失、动作、Replay 和 checkpoint。

train.sh 每约 1000 个训练步记录训练指标，评测沿用每 10000 步设置。成功率和接管率在终端显示百分比，文件中保持 0～1；小损失保留有效数字。视频和报错正常输出。

预热期 A 可学习但接管为 0。预热后仍不接管，也可能是门控或监督就绪条件未满足。若需追查 new/revisit 分项损失、各头标签分布、Brier、原模型梯度等更多细节，下一次启动加 `--log_all_metrics True`。该开关恢复全部现有指标，不改变算法，也不能补回过去未保存的详细指标。

**指标只能提供证据，不能代替消融。** 判断 A 效果仍要在同设置下比较 full/off；再用 loss_only 和同检查点控制开关分析收益来源。成功率稳定提高、完成更快才支持 A 有用，接管率提高或训练损失下降本身都不足以证明。

## 7. 本机验证记录

沿用包内 tests/test_search_a.py 与 scripts/test_a.sh，未另建测试启动脚本：

```bash
bash scripts/test_a.sh
bash scripts/test_a.sh --minedojo
bash scripts/test_a.sh --minedojo-off
```

- CPU 19 项测试通过：标签/遮挡/未知掩码、动作时序、边界、梯度及真实 WorldModel/Actor 更新、未训练头门槛、开关、特权隔离、旧回放、元数据、checkpoint 往返。冻结学习时停止随机探索的补充断言通过。
- Python 编译和 Bash 语法通过；已有 assert(tuple) 语法警告不影响本次测试。
- 开启 A 的真实 MineDojo + GPU 通过：小模型 96 训练步（含 32 预填充），前后各 32 步评测。checkpoint.step=96，参数及指标有限，4 个已保存回合的 NPZ 字段等长。
- 测试放宽门控和监督门槛，训练 epsilon=1，以覆盖动作接入；最终 eval_search_rate=0.96875、eval_search_decision_rate=0.21875，不代表默认设置效果。
- new/discovery/revisit 有效监督累计使用数 71/90/67，正例 52/0/20。真实短测没有采到发现目标正例；该正例与遮挡分支通过构造观测测试，真实正例质量仍需后续采样检查。
- 最后一次已记录 A/future_loss≈0.3031、A/prediction_loss≈0.7577，eval_success=0；不能据此判断 A 有效或无效。
- 关闭 A 的真实 MineDojo + GPU 测试通过：同样 96 步训练（含 32 预填充）、两次 32 步评测。checkpoint 无 A 参数、无 A 损失，新增标签掩码及接管率均为 0。
- 同一 v2 checkpoint 关闭控制器的独立评测通过：恢复训练步数 96，完成 32 步评测；接管率和决策率均为 0，评测标签掩码全为 0，NPZ 字段等长。
- 三种正式训练入口的参数展开已检查，模式配置、1M 预算、种子、日志目录及附加参数均正确；检查时没有启动正式训练。

此前已用临时替身检查主入口的三种模式、五个任务、种子、路径和失败退出码，没有启动正式训练。本轮用真实 TensorBoard 文件及 JSON 验证“终端四项、保存十项”、仅含诊断数据时仍写入、详细模式及数值精度；W&B 用模拟接口验证分组，没有网络上传。检查记录为 logs/test_apm_a_v2/diagnostics_cleanup_checks.log，CPU 复测为 logs/test_apm_a_v2/cpu_diagnostics_cleanup.log。

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
