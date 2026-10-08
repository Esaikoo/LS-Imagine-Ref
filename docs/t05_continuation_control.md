# T05 worker350独立自主确认

2026-10-09本地实现，服务器尚未执行。入口是`scripts/t05_continuation_control.py check/evaluate`，协议在`goal_continuation_control.py`。本轮只新增这两个代码文件，保留续段修复已验收的38个代码文件及旧结果；只做静态阅读、编辑和差异核对，没有本地Python、测试、模型、训练或环境运行，没有提交或推送。

## 要回答的问题

固定50步续段混合修复的保持开发留出改善、六项退步检查通过，值得确认worker350能否在原16步预算内自主实现目标1并保留目标0及对照优势。唯一纠偏只有一个训练状态、开发留出没有纠偏，目标1固定参考mode由14/16降为13/16；离线收益不能代替这次真实执行。

新入口自动从本轮独立verify定位生产`latest.pt`、修复check、真实缓存及完整来源。要求生产源300加固定50步、两分支计数与verify一致，重新核对六项最终retention、原check的确认计划和真实来源SHA256；推理内容和后端与verify一致。只用生产latest，不接受best、旧worker300、原参考修复格式或验收副本。verify内存副本的额外更新不计入生产版本。

## 已提前固定的设计

原修复check的`confirmation_plan.json`原样嵌入新design；其中`implementation_pending=true`保留其当时的预声明状态，新入口自己的格式、代码身份、生产模型身份另存，不改写原文件。

- worker350，mode，随机顺序种子4；goal/no_goal/swapped_goal三条件×原两个固定视觉目标×5重复，共30次独立fresh reset。
- 每局32noop+32真实前缀+最多16自主控制；frame64–79查询、incoming65–80、remaining16–1。完整局实际终点为frame80。
- 原世界、初始化和RGB/heatmap预处理不变；无跨局相同位置/RGB/RSSM要求。坐标和朝向仅诊断，不能输入策略或选动作。
- 不加4步、不使用视觉手工规则、不覆盖末步动作、不选择最佳中间帧；不筛选起点、不重试、补样或失败续跑。
- 最多2400次新环境步，包括预热和前缀。真实环境提前结束立即停止，保存实际最后观测并单列；缺测和执行错误全保留且不能判通过。

每目标正确目标组仍须至少4/5同时满足自身正进展`d_start-d_end>0`及终点正确偏好`d_target<d_other`；全部五次平均自身进展为正，并严格高于no_goal和swapped_goal的五次均值。缺测不以少数有效局的均值补足，目标0不能替代目标1。门槛与原worker300完全相同，不根据这轮结果调整。

即使数值通过，`behavior_accepted=false`、`human_review_pending=true`、`t06_approved=false`仍保留，须人工核对全部视频、位置/朝向和起点平衡后再决定限定场景T06。

## 服务器运行

只同步新增的`goal_continuation_control.py`和`scripts/t05_continuation_control.py`，可同时同步这份说明和TODO文档。不要用本地已有38个验收代码覆盖服务器：有三个本地文件只有换行差异，覆盖也会破坏原SHA256。服务器模型、checkpoint、缓存、轨迹和视频继续保留原路径，无需下载。

在原`ls`环境中执行以下准备和离线check：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_350_TAG="$(date +%Y%m%dT%H%M%S)"
T05_350_VERIFY="$PWD/relevance_map/t05_outputs/continuation_repair_verify_20261008T143457"
T05_350_CHECK="$PWD/relevance_map/t05_outputs/continuation_control_check_$T05_350_TAG"
T05_350_EVAL="$PWD/relevance_map/t05_outputs/continuation_control_evaluate_$T05_350_TAG"

python scripts/t05_continuation_control.py check \
  --repair-verify-dir "$T05_350_VERIFY" \
  --output-dir "$T05_350_CHECK" --device cuda:0
```

最后出现`[PASS] T05_CONTINUATION_CONTROL_CHECK`后，在同一终端执行真实确认：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_continuation_control.py evaluate \
  --repair-verify-dir "$T05_350_VERIFY" --check-dir "$T05_350_CHECK" \
  --output-dir "$T05_350_EVAL" --device cuda:0
```

无需再传checkpoint、缓存、种子、重复次数或控制预算；这些由已验收来源和预声明计划绑定。两步须使用原verify的`cuda:0`和相同后端。输出目录必须全新且独立，不允许覆盖、与输入重叠或resume。遇到FAIL保留目录和日志；如果发生磁盘/保存故障立即停止，已执行尝试保留，不自行补样或另换种子。

check只读真实历史并核对增量因果状态、动作分布和保存/门槛守卫，不启动MineDojo/MineCLIP或构造优化器。内存里的正/负标记不保存为轨迹。evaluate仅构造冻结推理和真实环境，三个条件都使用worker350对应分支。

## 产物与验收

每局保存完整自身真实历史、原生事件、incoming action、建议概率和mode、固定种子/remaining、因果历史校验、全帧解码视频、逐帧两目标视觉和位姿误差。所有模型/轨迹记录都写实际worker350、源300及追加50，不能误写旧300/追加100。

汇总包括：

- `design.json`、`evaluation_manifest.json`：原预声明计划、新模型/代码/verify身份，以及真实产物和汇总SHA256。
- `trials.json/jsonl/csv`：全部30次尝试、终点进展/偏好、真实预算/提前终止、末两步动作与最后一步进展、失败原因。
- `frame_metrics.csv`：真实frame、incoming、下一实际动作/概率和remaining，以及两目标RGB/heatmap距离及位置/朝向误差；真实终点不查询下一动作。
- `initial_balance.json`：起点RGB、heatmap、RSSM和位置/朝向差异，缺失trial单列，只作诊断、不影响分组。
- `diagnostics.json`、`gate.json`：各目标各条件全部单次与五次统计、提前终止/缺测及原门槛；行为结论与工程PASS分开。
- 每个repeat的`outcomes_repeat_0…4.png`和`starts_and_outcomes_repeat_0…4.png`，全部视频留在服务器用于人工核对。

完整无提前终止时，预期30/30有效、worker查询/执行480、保存概率重现480、逐帧测量510、env.step2400；干预、优化器、底座/WM更新均0。真实提前终止的计数按实际长度减少，不补零或假装16步。普通执行/保存接口错误保留本局并继续固定顺序，磁盘写入故障停止；总体门槛不通过。

反馈两步完整日志。可下载check的`report.json/design.json`和evaluate的`report.json/diagnostics.json/gate.json/trials.csv/frame_metrics.csv/initial_balance.json/evaluation_manifest.json`及十张结果图，同名文件加check/evaluate前缀；不要下载模型、checkpoint、缓存、trajectory.npz或视频。新确认结果到来后再分析，原worker300失败不改写；工程PASS或数值PASS都不会自动启动T06或1M训练。

中文Git提交备注：`feat: 增加T05续段修复worker350的独立自主确认`
