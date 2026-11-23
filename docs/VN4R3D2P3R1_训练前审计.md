# VN4R3D2P3 训练前代码审计

审计对象为本地 esaiko-pscv 分支的实际工作树。HEAD 为 17b6a0e，另有未提交的 D 接口、Replay、启动脚本、P3 绘图与测试修改，以及未跟踪的 apm/figure_style.py。审计后的工作树代号为 **VN4R3D2P3R1**，D 权重协议仍为 2。代码代号和 Git 提交号需分别记录。部署前应包含本地完整差异，尤其是新绘图文件。

**启动判断：采沙在两项工程修复后可以启动 1M。** 本轮在缩小模型上验证了采样、实际动作、联合更新、绘图、存档和继续训练。成功率和正式尺寸的长期资源峰值仍需正式实验给出。

## 对照范围

核对了 6fad743、861fe463、D 精简提交 b990fbd、17b6a0e 和工作树。6fad743 是历史 1M 权重参考，后续已有四任务图库与 C 消融接口差异。以 861fe463 为算法对照，检查了 git diff 及实际调用方。networks.py、envs/、assets/ 与 861fe463 一致。models.py 的差异集中在导入和 ImagBehavior.save_state_sequence；RSSM 转移、长短时程切换、Actor/Critic 目标和 World Model 优化公式没有变化。D 精简与此前的图表提交并非完全按功能拆开的提交，报告按代码作用归类。

## D 精简与决策影响

| 核对对象 | 实现与结论 |
|---|---|
| 两个旧头 | apm/goal_reference.py 的 GoalReference 只有区域编码和 match 分支。前向返回区域 logits、概率、集成分歧等。train_batch 通过独立 AdamW 优化区域网络。温度、可靠性均为长度 1 的区域缓冲。两个旧头的输出、独立损失、校准和推理调用均已移除。 |
| 正负参考与监督 | _match 使用归一化区域特征与正参考的相似度。负参考在训练中作为干扰物与困难峰值监督；_reference_crop_loss 分开参考来源世界，_paired_loss 仅比较满足相邻、相同样本序列、有效实收控制与同实例条件的正区域。区域保持 mask，未知区不参加正负损失。区域校准使用与图库世界分离的 holdout 记录，重复帧不重复计数。短训练不保证产生有效校准。 |
| 合法保留的字段 | apm/evidence.py 的 goal_visible_gt/valid 表示训练射线实际看见目标的证据，稀疏未命中保持未知。apm/outcome.py 的探索发现标签和 apm/diagnostics.py 的观测首现统计仍有用途。goal_interaction_valid/outcome 是真实交互反馈，在 apm/control.py 监督控制后果。独立 D 预测头已经不再读取这两类监督。 |
| 目标上下文 | apm/goal_guidance.py::build_policy_observation 将区域 peak、置信、不确定度、中心、冲突、状态及有效性打包。第一个分量由旧的区域峰值与整帧预测混合量改为区域峰值。_track 根据相邻真实观测一致性确认可信状态。gate 受区域校准、证据强度和不确定度约束，gate=0 回到原热图。训练批与在线决策共用这一变换；从回合中段开始的训练窗口不具备完整的前序跟踪状态。 |
| C 的预期行为变化 | VNextController.choose 移除旧 D 交互头否决、显式整帧 absent 对 high/low 和交互的影响，以及 absent 连续计数触发的预算提升和探测冷却缩短。保留区域证据/天空冲突门控、A 可交互条件、适用的中心和俯仰条件、工具与动作约束、探测次数和恢复。长期 unresolved 的有限预算调整仍保留。各条资格约束适用于其各自的控制器分支。原 Actor 自主提出交互与 C 额外发起探测应分别解释。 |
| A/B/C 其余路径 | apm/outcome.py::train_step 仍训练 H1/H2/H3，以及按原间隔运行的辅助 H8。verified_discovery、reobserve、new_area 均有独立的已知性掩码及输出；短时预测通过原有支持/评分或目标预测输入参与排序。H8 主要作辅助训练。B 以同一次视觉历史检索返回可选的 detached 权重，无第二次在线查询。C 额外记录规则前选择和模板，未重排候选。冻结评测的 learned_selection 与 rule_probes 分支及训练期限制保留。 |
| 旧数据与权重 | apm/replay.py 只将 v2 上下文标为新 D 有效输入，避免将 v1 的首分量当作新区域峰值。旧记录仍可离线分析。training_state.py::validate_goal_checkpoint、restore_training_state 校验协议、目标、图库指纹并严格加载。已用真实 VN4R2T1 的 1M v1 文件验证训练恢复与评测两条验证路径均明确拒绝。 |

**五个任务。** 实际执行了五次正式脚本的参数解析，截获 main 之前的配置，并分别实例化 GoalReference。五库各 12 正、8 负，指纹互异。脚本的库存任务名依次为 sand、log、water_bucket、iron_ore、wool；对应的视觉类别为 sand、log、water、iron_ore、sheep。水面和羊是可见目标，水桶和羊毛是交互结果。图库加载和归属正确。原有射线监督按树木/羊的类别识别，橡木图库及白羊图库的选图范围更窄。task_specs 的成功项使用 log、wool，单看这些字段无法验证亚型或颜色。铁矿图库含矿洞/石质上下文，射线正标签仍为铁矿。要报告严格亚型识别或矿洞探索收益，需另行验证。此轮未改共享监督或任务规则。

## 可视化旁路与发现的缺陷

1. **P1，零更新日志步会推进训练 Replay。** expr.py::LS_Imagine.__call__ 旧逻辑仅在本步确有更新时复用 batch；若此时要记录 train_openl，会再读一次训练迭代器，改变后续样本顺序。在 vNext4 改为记下视频请求，下一次梯度更新后用现成 batch 生成视频。额外内存仅一个布尔标志。视频时间戳可能移至下一次实际更新，标量日志仍按原时钟。此问题在 P3 排版以前已经存在。原 LS-Imagine 单独运行路径沿用此前的视频采样策略，此处的等价性结论限定于新 D 系统。
2. **P0 条件性故障，论文图导出异常会停止正式训练。** 正式 apm_vnext4 曾继承 vnext_visual_strict=True，SearchVisualizer._export 确实会重新抛出渲染/写图异常。正式预设改为 False，继续打印错误、计数并保留临时目录的原子发布。严格短测显式使用 True。该开关作为运行时选项可在恢复时覆盖。保护适用于已捕获的导出异常，不能兜住设备故障、磁盘写满导致的 checkpoint/Replay 保存失败或进程被终止。
3. **图像生成有额外计算。** 想象样本解码一次已算出的潜状态；结束回合的地图导出可能对终止观测额外运行 D/B，在复制的记忆状态上执行；TensorBoard 开环视频也调用世界模型。这些调用不额外推进环境。apm/imagination_figures.py::capture 使用 no_grad、分设备 RNG 隔离及确定性选样，保留原始时间轴与终止标记。tools.simulate 过滤 log_visual_，保留真实执行所需及原有 Replay 字段。绘图开关可以改变观察者输出、计算时间和文件大小。
4. **资源边界。** 配置每 10000 步最多一套阈值想象输出；恢复初始化使用当前步整除阈值，已有目录跳过。完整回合图在选中评测轮末局导出，其余两局可有单独 B_map。可视化默认每回合最多 4096 条低维记录，最多 8 个活跃回合，地图后台队列最多 4 个快照，满时有背压。原始 RGB/热图只保留少量关键帧。已写到磁盘的正式实验产物随训练继续累积。

## 本轮执行的验证

在本地 WSL Ubuntu-20.04 的 `ls_imagine` 环境运行。GPU 为 RTX 2060，6 GB，PyTorch 2.1.1+cu121。下表中 `$AUDIT` 指仓库下的 `logs/final_audit_1123`。此目录存放本次诊断脚本、实际日志与短测产物，按项目惯例不提交到 Git。

| 检查 | 命令 / 执行入口 | 实测结果和证据 |
|---|---|---|
| 静态差异、语法与回归 | `git diff 861fe463`、`git diff --check`、`bash scripts/test_a.sh` | 最后一次完整回归 17 组、460 项通过。包括 D v2 梯度及优化器、区域 holdout 和恢复、无有效标注、局部拒绝、有效融合、C 正式交互和两种消融、A 各时距、B 地图及视觉历史、可视化错误注入和 RNG 测试。见[完整测试输出](../logs/final_audit_1123/regression_final.log)。 |
| 启动参数和五图库 | 五个任务分别经 `A_MODE=vnext4 BANK=assets/goal_references sh scripts/train.sh <task>`，用本地检查器截取最终参数，在进入 main 前退出。另由 `python "$AUDIT/paired.py"` 逐库初始化 | 正式入口解析出 1,000,000 步，想象时距 15，评测频率 10,000，每轮 3 局，视频和阈值想象启用，D 协议 2，论文图错误记录后继续。C 两个评测消融默认开启正常路径。[五任务生效配置](../logs/final_audit_1123/argv.log) |
| CPU 多步开关对照 | `OMP_NUM_THREADS=2 python "$AUDIT/paired.py"` | 从同一 D v2 小模型检查点及真实 MineDojo Replay 开始。7 次在线调用中有 5 次执行完整更新，D 更新计数均从 20 增至 25。逐次比较 267 个参数张量、最终 86 个带梯度的张量、缓冲、7 套优化器状态、训练样本、数值指标（计时除外）、非绘图动作输出、各子模块模式和随机状态。全部逐字节相同。开图侧另调用终止观测地图旁路，写出 3 条训练视频和 4 个压缩间隔的想象导出。[CPU 比对结果](../logs/final_audit_1123/paired_final.log) |
| CUDA 多步开关与重复性 | `python "$AUDIT/cuda_control.py"`，另用 `paired_cuda.py` 验证严格模式，并运行 `cuda_warn.py` | 三组顺序为关、关、开。所有 7 次实际选择及控制命令、CPU/CUDA 随机状态、C 内部 RNG、采样批次和模型模式一致。两次关图最大绝对参数差为 2.98×10⁻⁸，关图对开图为 5.96×10⁻⁸。严格逐位数值对照未通过。强制确定性时 CUDA median 不受支持，警告模式还标出自适应平均池化反传。未改变算法以满足测试开关。[三组实测](../logs/final_audit_1123/cuda_control.log) · [严格模式报错](../logs/final_audit_1123/paired_cuda_strict.log) · [警告模式](../logs/final_audit_1123/cuda_warn.log) |
| 首轮真实采沙 | `MINEDOJO_GRADLE_OFFLINE=True APM_SMOKE_ROOT="$AUDIT" bash scripts/test_a.sh --minedojo-vnext4` | 首轮启动时使用修复前的 P3。随机预填充、真实训练、96 步检查点、20 次 D 更新和三类 TensorBoard 视频已落盘。结束阶段本地 WSL 会话中断，旧 metrics.jsonl 尾行损坏，原命令未通过完整验收。没有足够系统证据将该中断归因于 OOM。[首轮日志](../logs/final_audit_1123/suite_game.log) |
| 修复版全新启动 | `python expr.py --configs minedojo apm_vnext4 apm_a_smoke … --steps 48`，采用独立目录、40 步预填充、严格绘图、Java 8 与真实 GPU 环境。[完整实参](../logs/final_audit_1123/fresh_command.txt) | 进程正常退出。第 0 和 48 步各完成 3 局采沙评测。终点有 D 的 4 次更新、7 套优化器、4 条有限指标及三类视频。1 个训练 NPZ 含 534 个已知区域单元；6 个评测 NPZ 的目标和位姿监督均无有效标记。48 步末次阈值没有待解码想象状态，清单记录 skipped；有状态时的真实导出另见 CPU 多步和离线重绘。[程序输出](../logs/final_audit_1123/fresh_final.log) · [产物复核](../logs/final_audit_1123/verify_fresh.log) |
| 真实恢复及末轮评测 | `A_MODE=vnext4 LOG_ROOT="$AUDIT/resume" NTFY_TOPIC='' sh scripts/train.sh harvest_sand --resume_from <96步检查点> --steps 104 --log_every 8` | 修复版正式入口在独立目录成功退出。读取旧回放 96 步并跳过重新预填充和预训练，在 96 和 104 步各跑 3 局真实评测。新检查点记录 P3R1、104 步，7 套优化器仍在，D 更新从 20 增至 22。4 条新指标行可解析且数值有限，三类视频标签均存在。恢复会重建 Minecraft 环境与采样游标。[续训日志](../logs/final_audit_1123/resume.log) · [产物复核](../logs/final_audit_1123/verify_resume.log) |
| Replay 隔离、执行回执与旧权重 | `python "$AUDIT/replay_check.py" <原 run> <恢复 run>`；`python "$AUDIT/legacy.py"` | 分别检查 3 个训练 + 9 个评测 NPZ 和 1 个训练 + 6 个评测 NPZ。字段等长，实际控制与回执一致，未写入绘图专用字段。训练文件分别有 1,546 和 102 个已知区域单元。评测目标/位姿真值有效位全零；合法 RGB 有效位单独处理。真实 1M D v1 权重在训练和评测验证入口均明确拒绝。[原回放](../logs/final_audit_1123/replay.log) · [恢复回放](../logs/final_audit_1123/replay_resume.log) · [旧权重](../logs/final_audit_1123/legacy.log) |
| 图像格式与阈值 | `python scripts/redraw_search.py <回合归档> <新目录> --format <格式> --focus 0 --radius 6`；想象归档使用 `--style both` | 用真实第 48 步评测第 3 局与真实模型解码缓存重绘。PNG、SVG、PDF 各 3 个，分别解码、解析或实际渲染成功。单测还覆盖阈值去重、恢复时不补图、缺失预测、定位断裂、部分目录清理。[重绘记录](../logs/final_audit_1123/redraw.log) · [产物](../logs/final_audit_1123/artifacts) |

### 如何理解精度结果

CPU 对照每次从相同权重、优化器、采样状态开始，覆盖有训练批和没有训练批的日志时刻。它可以检出多取一次 Replay、推进随机状态、意外写入参数或改变后续选择。GPU 的关图重复基线本身存在浮点差异，关开对照没有出现不同的离散动作。现有实验足以排查本次修复针对的旁路干扰，无法保证 CUDA 长轨迹逐位一致。

### 修复位置与长期风险

- `expr.py::LS_Imagine.__call__` 增加延迟视频请求。对应新增用例为 `tests/test_goal_search_integration.py::test_open_loop_video_waits_for_real_update_without_consuming_another_batch`。
- `configs.yaml::apm_vnext4` 和 `training_state.py::_RUNTIME_KEYS` 调整正式训练与恢复的导出错误处理。回归覆盖模拟写图失败、无半成品成功目录，以及恢复时运行参数的优先级。版本登记于 `training_state.py` 和 `docs/README_A_CN.md`。
- 正式规模的吞吐、数日内的 CPU/GPU 内存峰值、磁盘余量和任意点断电后的日志完整性仍未验证。中断样本说明检查点可读与最后一条日志完整是两个独立条件。进程外强制退出仍可能留下不完整的最后一条记录。
- 真实游戏短测的 D 区域校准仍为零，未形成可用于判断实际融合收益的校准样本和跨视角对。有效融合与正样本配对经过受控回归。采沙以外四项只核对了参数和图库，尚缺各任务的真实交互及长期校准证据。
- 同步服务器前请一并包含 P3 尚未跟踪的 `apm/figure_style.py`。单独部署 HEAD 或遗漏该文件，不能复现本报告审计的工作树。此次验证均使用本地隔离的短测目录。

**最终判断。** 使用 `A_MODE=vnext4` 和此次完整本地工作树，采沙可从头启动正式 1M。新实验使用内置区域版 D 与现有图库，旧 D v1 的 1M 检查点不能作为续训起点。首轮中断、GPU 精确复现和四个其他任务的真实环境覆盖范围如上。未启动正式训练。
