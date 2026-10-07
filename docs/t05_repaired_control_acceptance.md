# T05修复worker的自主确认

2026-10-08：新增 `goal_repaired_control.py`、`scripts/t05_repaired_control.py check/evaluate`。本地仅静态阅读、编辑与差异核对；没有运行Python、测试、模型或环境，服务器待验收。已通过的参考修复check/train/verify、旧诊断和标定不用重跑，不再训练或选择中途快照。

## 本次固定的实验

只接受 `reference_repair_train_20261007T164732/latest.pt` 对应的独立修复verify及同一缓存：源分支200步加修复100步，worker version=300。专用修复加载器校验格式、内容和共享依赖；只构造冻结底座/两分支/因果状态编码器/目标库，不恢复优化器、不使用候选模型。旧残差和小探针格式、验收快照均不能代替新模型。

三组为 `goal`、独立修复 `no_goal` 和 `swapped_goal`。三组同用 **mode**；两个原固定视觉目标，每目标/条件五次，共30次。确认随机种子固定为 **1**，不同于旧试跑的0；每个重复内随机排列六个条件，各局动作随机数独立绑定计划索引（mode不靠随机数选动作）。不暴露改执行方式、重复数、场景、预热或门槛的命令参数。

场景只限已标定的case seed=0/world seed=1。每次fresh reset，保存真实reset、32步noop、32步原前缀，然后执行最多16步自主动作；全计划最多2400次真实env.step，包含预热/前缀，最多480次worker动作。目标仍是原两条完整参考的真实RGB/heatmap终点，位置/朝向只作为诊断，不能输入策略。完整历史从真实reset累计，不复制另一局状态或伪reset。

不要求本轮起点与参考或另一局位置、RGB、RSSM相同；跨局差异和各组起点平衡都报告。拒绝真实历史/动作/遥测缺失、终止后动作和计划篡改。计划在启动环境前保存；各trial独立目录在reset前创建并检查原子JSON读写。普通trial接口失败原样保留并继续固定顺序，输出/存储故障或中断则停止，保留已有文件；没有resume、重试、替换目标、补样或根据终点追加运行。

## check与evaluate做什么

`check`不导入MineDojo/MineCLIP、不启动环境。绑定专用修复verify、旧真实目标来源与标定计划、当前代码/环境指纹；复用已通过的增量/完整真实状态、reset/incoming/结束/remaining与部署分布检查。保存目标及设计。用明确的内存标记检查保存接口和行为门槛：真实下一动作必须等于记录的mode、缺测/未完成/与对照持平不能通过、目标0不能掩盖目标1、门槛修改不能通过。标记不会写成真实轨迹。

`evaluate`先核对完整check、固定计划、目标和代码，再做视频与空间预检，执行30次新自主控制。每局保存完整因果历史、原生动作、遥测、动作概率、剩余预算、录像以及自身起点/终点。沿用已验收部署分布与真实环境接口，不改旧脚本及其指纹。

执行后再从该局保存的真实历史核对：所有真实incoming/奖励/终止/遥测，记录的mode与真实动作、每一步部署概率、起点至最后观测的两目标距离；录像须完整编码并解码。逐帧保存对原两目标的视觉距离、RGB/heatmap差和位置/yaw/pitch误差，方便核对改善来自实际姿态变化还是渲染。主结果只取真实第16步终点；若真实提前结束，用真实最后观测并单列，不能取最佳中间帧替代。每帧不是独立样本。

## T06数值条件与人工核对

判定随check写入 `design.json`，复用 [既定准入条件](t05_to_t06_gate.md)，不得看结果后修改：

1. 全部30次已按计划尝试，真实执行、保存、测量和视频全部有效；缺测不能判通过。
2. 对每个目标，正确目标组至少4/5次同时有正进展 `d_start - d_end > 0`，且终点更近被评价的目标 `d_target < d_other`。
3. 对每个目标，正确目标组五次平均自身进展为正，并严格高于no_goal和swapped_goal各自五次均值。缺一条有效测量，该组五次均值记为不可判定；不能只平均剩余成功条目。
4. 数值达标后还须人工核对全部视频、位置/朝向和初始平衡，确认目标方向一致。没有新的经验性位置阈值，不把两个近距离视觉姿态目标解释成通用导航。

`gate.json`的 `passed` 仅代表**数值条件**。`human_review_pending=true`、`t06_approved=false`始终保留；完整接口PASS与数值失败分别报告，不为了最终PASS而隐藏效果失败。此入口不启动T06。满足两目标数值条件并完成行为核对后，才能进入当前场景的小规模宏转移采集；不能据此批准1M训练、跨世界泛化或论文级成功率结论。原26局退步仍保留，当前模型仅用于这个局部确认。

## 服务器命令：仅两步

提交/推送并在服务器拉取更新后，从项目根目录执行。第一步最终PASS才运行第二步；若check失败，保留结果反馈。第二步不重复已通过的训练、标定或诊断。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_CONFIRM_TAG="$(date +%Y%m%dT%H%M%S)"
T05_CONFIRM_CHECK="$PWD/relevance_map/t05_outputs/repaired_control_check_$T05_CONFIRM_TAG"
T05_CONFIRM_EVAL="$PWD/relevance_map/t05_outputs/repaired_control_evaluate_$T05_CONFIRM_TAG"

T05_CONFIRM_ARGS=(
  --cache-dir "$PWD/relevance_map/t04_outputs/cache_20261006T124827"
  --checkpoint "$PWD/relevance_map/t05_outputs/reference_repair_train_20261007T164732/latest.pt"
  --repair-verify-dir "$PWD/relevance_map/t05_outputs/reference_repair_verify_20261007T164732"
  --source-benchmark-dir "$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
  --device cuda:0
)

python scripts/t05_repaired_control.py check \
  "${T05_CONFIRM_ARGS[@]}" --output-dir "$T05_CONFIRM_CHECK"
```

check最终PASS后，在同一个bash会话执行：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_repaired_control.py evaluate \
  "${T05_CONFIRM_ARGS[@]}" --check-dir "$T05_CONFIRM_CHECK" \
  --output-dir "$T05_CONFIRM_EVAL"
```

## 反馈哪些文件

反馈两个最终PASS/FAIL、`[NUMERICAL GATE]`及两个`[TARGET]`日志。下载评估目录里的 `report.json`、`diagnostics.json`、`gate.json`、`trials.csv`、`frame_metrics.csv`、`evaluation_manifest.json`，以及五张 `outcomes_repeat_0..4.png`、五张 `starts_and_outcomes_repeat_0..4.png`。check报告如有失败请区分文件名。无需下载模型、缓存或完整轨迹；模型共享依赖、check和视频均保留在服务器，后续按结果选择需核对的视频。

每trial含 `trajectory.npz`、`events.json`、`video.mp4`、`control_trace.json`、`start.json`、`history_check.json`、`frame_metrics.json`、`metrics.json`及起点/终点图片。异常目录也保留；主manifest对trial文件及主要汇总做SHA256绑定，后续解释只能针对同一批真实数据。

中文Git提交备注：`feat(T05): 接入修复worker的30次固定自主确认与逐目标T06数值门槛`
