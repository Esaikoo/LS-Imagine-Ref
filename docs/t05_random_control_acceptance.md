# T05：独立起点、随机分组的真实目标控制

2026-10-07。本地已实现 `goal_random_control.py` 和 `scripts/t05_random_control.py`；用户服务器check/evaluate工程验收已通过。助手只做静态阅读、编辑和差异核对，未执行Python、测试、训练、模型或环境，未提交/推送。仍为T05。

最新结果：`random_control_check_sample_20261007T114359 / random_control_evaluate_sample_20261007T114359`通过，12/12次真实控制有效、192个控制动作、960个环境步，模型不更新。采样已增加动作种类，但goal/no_goal/swapped平均视觉进展为0.061381/0.098416/0.125034；所有终点按当前目标编码仍更近target0。sample工程验收通过，真实目标控制未验收，T06未开始。

用户服务器已通过 [同一真实状态的动作选择离线诊断](t05_action_choice_diagnosis.md)：192个查询原概率重现误差0，换目标只改变4个mode，12个起点均选择左转。随后已按下方两步完成三组统一sample的有限试跑。现有评估代码不修改，旧mode产物、失败和原check继续保留。下方命令作为已完成实验的运行记录，本轮不再重跑或追加重复。

## 采样试跑结果与下一步判断

| 条件 | 目标0平均进展 | 目标1平均进展 | 总体平均进展 |
| --- | ---: | ---: | ---: |
| goal | 0.010388 | 0.112375 | 0.061381 |
| no_goal | 0.116372 | 0.080459 | 0.098416 |
| swapped_goal | 0.044402 | 0.205666 | 0.125034 |

进展是初始减最终视觉距离，越大越好。交换组仍针对原指定目标计算，实际输入另一个目标。goal目标0两次进展0.015582/0.005193，均弱于独立无目标0.165904/0.066840；目标1第一次goal弱于无目标，第二次较好，平均多0.031916，但交换组反而多0.093291。goal两重复总体平均进展0.030908/0.091854，无目标为0.113886/0.082945，优势没有稳定复现。少量结果不能推断策略总体显著更差或目标信息不存在。

所有12个终点均更近target0。直接输入目标1的两次goal终点到目标0/1的距离分别是0.180760/0.662116、0.140095/0.510043。目标1绝对距离下降不等于出现目标1专属控制；从目标0方向变化也可能同时缩小两距离。第二次goal/目标1的相对偏好确有正向改善0.074425，但第一次为-0.028479，尚不稳定。

本轮12个控制起点记录的x/y/z/yaw/pitch均完全相同。各目标组间初始视觉距离均值差小：目标0最大约0.0073，目标1约0.0108；完整视觉/RSSM历史仍不声明相同。此次缺少目标优势不能主要归因于上一轮约0.712格的位置偏移。

真实动作直方图表明，192个控制动作中22个超出旧mode实际只执行的turn_left/turn_right/jump三种，包含forward、上下转向等；goal/no_goal/swapped各为9/6/7个。86次turn_left加39次turn_right仍占125/192，不能将动作更丰富当作方向正确。附件不含逐步control_trace，故本轮没有再次证明采样动作的逐值重现或完整序列差异；这些记录仍保留在服务器。任务成功均为假、外部回报均为0，不把16步试跑的任务成功作为单独工程门槛。

结束mode/sample执行方式的试探。该轮只分析附件与更新状态，不训练、改温度或延长预算。用户随后授权的目标可达性与视觉度量标定已本地实现：在当前32步预热加32步前缀流程下，固定执行两个已有真实参考动作脚本各两次，记录自身完整历史和真实终点，重点检查目标1能否由这些已知动作在16步内接近，以及视觉距离是否区分两个终点。不同局不要求隐藏状态相等，不挑选有利起点或替换脚本。按 [标定两步命令](t05_reference_calibration.md) 运行，最多320步；服务器待验收，尚无新采集结果。

若参考动作可复现两种目标方向，而worker不能，再定位训练覆盖与目标条件泛化；若参考动作也无法支持旧固定目标，则先建立与当前初始化流程相容的真实目标基准。现有附件无法区分这两种情况，不能直接宣布模型或目标编码器已损坏。真实目标控制通过前不推进T06。

历史失败：`random_control_check_20261007T070049`已通过；同次evaluate在首个trial写`start.json`时报`FileNotFoundError`。这是本入口遗漏目录创建的程序bug：原子JSON写入要在目标目录创建临时文件，原来trial目录只在保存轨迹时才创建；关闭截图/日志输出的环境不会主动创建它。首个trial已完成32步noop+32步前缀，尚未执行worker控制；不能从此失败判断模型效果。

本地修复：每个trial在启动环境前显式创建新目录，并原子写入`output_preflight.json`确认可写；已有trial目录拒绝覆盖。离线check用同一路径检查起点、历史检查、指标和动作记录的JSON读写，新增`trial_output_preflight`；不增加环境预热、训练或新协议。源模型和旧验收工具未改动。

修复后用户已用新目录重新执行并验收下列check/evaluate两步。check绑定脚本内容哈希，`random_control_check_20261007T070049`不能作为修复后代码的check输入；通过的`output_fixed`新check继续有效。旧失败目录保留，不从中自动续跑；修复重跑中固定目标、12次顺序、模型和预算均保持原样。

本次采样试跑保留残差worker第200步、无目标底座、WM和目标库。入口只有两步：`check`离线固定sample目标/计划；`evaluate`执行完整12次。已有动作诊断无需重跑；本轮没有旧prepare、新参考采集、双次probe、信息探针或BC训练。

已静态核对：`goal_control.select_action`在sample条件下按原动作分布的累计概率和均匀随机数选择动作；每局的动作随机数独立于环境初始化，并在控制前固定16个值。三组共用同一种采样方法，使用各自原有部署分布。`control_trace.json`保留执行方式、动作种子、每步概率、随机数、实际动作和remaining，后续可以用现有离线诊断复核采样动作。本轮不增加温度、不放大残差、不更换unimix，也不延长horizon；无新代码或checkpoint。

## 为什么改评估

早期worker基本忽略目标，独立无目标BC对照证实收益很小；归一化残差分支已带来有限的离线收益，实际控制仍待验证。最近卡住的是环境重放：相同原生动作仍产生约0.86格位置差，worker尚未开始控制。先前的强配对要求及反复预检增加了不必要的迭代。

新问题是：同一种环境初始化过程下，有目标策略的平均目标进展是否优于独立无目标策略，交换目标是否改变去向。每局保留自己的真实历史，比较独立运行的结果。位置/RGB/RSSM差异用于描述环境波动与起点平衡，不再用于拒绝一个本来有效的真实起点。

这是一份独立的新实验，不能将旧FAIL改成PASS。也没有随机瞬移或人为清理动物：仍使用明确world_seed=1和已有初始化设置，条件的运行顺序提前随机化，环境自然波动原样保留。

## 固定内容

- 主模型：`residual_train_20261007T031803/latest.pt`，第200步，两套修正分支均冻结；不用候选模型/高层。
- 两个目标：来自完整旧 `residual_control_adopt_20261007T040055` 的真实RGB/heatmap终点，重编码并核对内容。只代表提前固定的视觉条件，不保证从每个实际起点可达，也不声称是每局起点的两个实际分支。
- 三组：`goal`给指定目标；`no_goal`使用独立训练的无目标修正；`swapped_goal`给另一目标，但指标仍针对原指定目标。交换组的`issued_target_margin`另行保存。
- 两目标×三组×两重复=12次；每重复六种组合随机排序。每次fresh reset，32步noop＋32步原前缀＋最多16步worker控制。完整运行最多960个`env.step`，控制动作最多192次。真实结束立即停；提前结束/环境错误保留，不补跑。
- sample执行，种子和顺序在check固化；evaluate必须提供相同参数。每局从真实reset增量恢复RSSM，并重算自身完整历史核对。12局的每局采样流各自固定，组间不声明共享随机数或相同状态；离线8.64%的共享随机数分歧不是本轮收益预测。
- 整轮不重试、不从旧FAIL续跑、不复制隐藏状态；程序中断后保留部分产物，本入口不提供自动恢复，先反馈失败原因。

## 已完成的服务器运行：采样试跑两步

本次只改运行文档，沿用已通过的服务器评估代码即可。下列命令在服务器Bash运行，所有输入变量均明确指定；已有采样目录拒绝覆盖，因此使用新时间标签。只在第一步PASS后执行第二步。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_TARGET_SOURCE="$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
T05_SAMPLE_TAG="sample_$(date +%Y%m%dT%H%M%S)"
T05_SAMPLE_CHECK="$PWD/relevance_map/t05_outputs/random_control_check_$T05_SAMPLE_TAG"
T05_SAMPLE_EVAL="$PWD/relevance_map/t05_outputs/random_control_evaluate_$T05_SAMPLE_TAG"

python scripts/t05_random_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_TARGET_SOURCE" \
  --output-dir "$T05_SAMPLE_CHECK" --device cuda:0 \
  --case-seed 0 --warmup-steps 32 --repetitions 2 \
  --randomization-seed 0 --execution-policy sample
```

第一步出现 `[PASS] T05_RANDOM_CONTROL_CHECK` 后再执行第二步。如果FAIL，先反馈日志，不运行后续、不反复重跑。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_random_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_TARGET_SOURCE" \
  --check-dir "$T05_SAMPLE_CHECK" \
  --output-dir "$T05_SAMPLE_EVAL" --device cuda:0 \
  --case-seed 0 --warmup-steps 32 --repetitions 2 \
  --randomization-seed 0 --execution-policy sample
```

无需再次跑旧的warmed prepare/evaluate，旧剩余24次也不执行。本轮不新增大型checkpoint，不复制缓存/权重/旧视频；开始前预留轨迹视频空间，保留磁盘余量。

## 工程验收应该看到什么

check中应看到 `fixed_visual_targets`、`trial_output_preflight`、`causal_execution_interface`、`independent_start_contract`、`predeclared_design`、`no_training`、`source_inputs_unchanged` PASS；保存 `design.json / targets.npz / targets.png`。`trial_output_probe/`中的JSON明确标为输出预检，不含真实轨迹或训练样本。设计应为12次，最大960步，新增真实环境步0。

本轮 `design.json`中的`design.execution_policy`必须为`sample`，`horizon`为16，`control_start_frame`为64；每局`control_trace.json`的`execution_policy`也必须为`sample`。控制台`mode=goal/no_goal/swapped_goal`表示实验组别，采样方式由`execution_policy`决定。sample不保证选出非最大概率动作，也不以“动作一定变化”作为工程门槛。

evaluate正常完成时，12条 `[TRIAL n/12]` 应实际执行控制，不再出现 `START MISMATCH; block excluded`。最后应看到：

```text
[PASS] planned_attempts: ...12/12...
[PASS] real_control_interface: ...12/12...
[WARN] behavior_acceptance: ...描述性试跑...
[PASS] no_training: optimizer_updates=0...env.step=960
[PASS] source_inputs_unchanged: ...旧失败不改写
[PASS] T05_RANDOM_CONTROL_EVALUATE
```

960是所有局均完整执行时的总数；真实结束则更少，原事件和步数会解释差异。`attempted_env_steps`另报尝试调用次数，环境调用抛错时可能大于已返回步数。任务成功和奖励单独保存，16步视觉控制不以必须完成harvest任务为工程门槛。

有效动作、原生事件、真实reset/incoming、因果状态、遥测、视觉配置或环境执行错误仍会FAIL。错误尝试保留，正常的其它尝试不整块删除；汇总同时报告计划数、尝试数、有效测量数和错误数。存在错误时不能凭正常子集宣布控制成功。

## 行为分析和需要反馈的文件

第二步目录中请反馈：

- `report.json`、`diagnostics.json`、`trials.csv`、`evaluation_manifest.json`。
- `outcomes_repeat_0.png`、`outcomes_repeat_1.png`；必要时选取三组视频。
- `starts_and_outcomes.png`按实际执行顺序，每个trial左侧为自己的控制起点、右侧为终点；顺序见manifest。
- 如有错误，相关 `trial_*/error.json`；若起点偏差很大，相关 `trial_*/start.json / history_check.json`。

每trial保存全部真实RGB/heatmap/incoming/5120维状态、原生事件和MP4，不插入训练回放；`control_trace.json`记录目标、每步分布、动作、remaining与两个目标距离。有效trial的`history_check.json`检查它自己的完整reset历史；不与其它trial隐藏状态配对。`output_preflight.json`在启动该trial环境前落盘，只说明输出目录创建与原子写入成功，不是控制效果验收。

主要看：有目标是否比独立无目标更有进展；交换目标是否转向另一个目标；两目标都是否有相应方向；初始距离/位置/画面是否存在明显组间不平衡。`diagnostics.contrasts`是独立起点均值差：`progress_advantage`、`evaluated_target_margin_advantage`越大越好，`end_distance_difference`越小越好；同时看`initial_distance_difference`。

本轮主比较仍是sample三组内部的随机条件对照，保留两个重复和两个目标的结果，重点确认是否出现相应目标方向。与旧mode运行比较只能描述变化，因为两次运行的环境历史并不相同；不能将它们当作相同起点的配对，也不能只看总体平均进展。概率采样可能增加动作差异，却未产生正确控制方向；若仍全部偏向target0，继续定位监督/可达性/策略学习，不自动追加重复或改温度、预算。请先反馈这次完整结果。

这12次仅涉及一个世界，两个重复，不能给出稳健显著性或成功率提升结论。可能得到“接口通过但控制弱”或“初始波动太大，效果不确定”；两者都比worker尚未执行就被配对门槛拒绝提供更多信息。T06仍等待真实行为分析，程序不会自动推进。

## Git中文提交备注

用户授权后已本地实现 [固定参考动作标定入口与两步命令](t05_reference_calibration.md)：沿用完整sample的初始化、两旧目标和真实脚本，独立执行四次重放、最多320步；只作可达性和度量定位，不再执行worker或训练。服务器待验收，旧结果及旧代码指纹保持。下列备注对应之前的采样结果分析；本次代码交付备注见新文档。

```text
docs: 记录T05采样试跑验收结果与目标标定方向

- 记录192个真实状态动作选择诊断通过及目标影响偏弱
- 记录sample全部12次有效及三组进展与目标0偏向
- 区分采样动作变化、视觉进展与真实目标控制验收
- 停止执行方式试探，列出目标可达性和度量标定的后续范围
- 沿用现有评估代码，不新增模型训练或推进T06
```
