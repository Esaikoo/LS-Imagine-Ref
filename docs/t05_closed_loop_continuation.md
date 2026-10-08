# T05真实纠偏与保持续段采集

2026-10-08。本地实现及用户服务器`closed_loop_continuation_check/evaluate_20261008T093057`工程验收已完成：10局/840步、查询200、worker执行180、手工20、更新0。手工终点偏好5/5、worker3/5，真实下转纠偏1局、noop保持4局；唯一纠偏在train，开发留出没有纠偏，位置仍移动。见 [结果与下一步修复分析](t05_closed_loop_continuation_result_analysis.md)。本页命令保留供复现，本次无需重跑。助手仅静态阅读/编辑和附件复算，没有本地运行Python、测试、模型或环境，未提交/推送；原已验收代码及旧失败不改写。

上轮固定forward真正改变建议的两局都没有纠正视角；成功三局本来就建议forward。原worker三次在正确视角下最后turn_up离开。此次取得真实纠偏和保持动作的执行证据，先采集再决定训练。来源见[上轮分析](t05_terminal_intervention_result_analysis.md)。

## 固定设计

同一worker300、目标编码器、两个原RGB/heatmap目标、世界和初始化流程。两组各5次独立fresh reset，随机顺序种子3，全10次预先写入check，不按画面或历史选择起点、补样、重试或续跑。

每局32noop+32真实前缀，再执行完整16步目标1/worker mode，保留第80帧自主终点。未真实终止时追加固定4步，保留第84帧实际终点，最多840个真实环境步。追加预算属于本开发采集，原30局的16步行为门槛不变；这不是延长horizon已有效的结论。任何真实终止都停止，最后真实观测与不足预算单列，不虚构第84帧。

| 条件 | 前16步 | 后4步 |
| --- | --- | --- |
| `worker_continuation` | worker300、目标1、mode，remaining16–1 | 同一worker、目标1、mode，新remaining4–1 |
| `visual_correct_hold` | 相同自主流程 | 每步仍保存worker建议，但实际执行预声明视觉反馈动作 |

视觉反馈只读取当前真实RGB/heatmap的冻结编码，到两个固定目标的cosine距离；坐标、俯仰和朝向仅作诊断，不进入反馈或worker目标输入。如果当前`d1 > d0 + 1e-7`且续段尚未纠偏，实际执行一次`turn_down`；否则执行`noop`。最多一次纠偏，之后全部noop；数值持平时noop。成功也不提前停，不反复下转追逐距离。该规则是待验证的手工采集假设，不是专家最优策略或已学会的worker；更近目标1也不等于任务成功，noop可能仍存在惯性/渲染变化。

第一阶段的turn_down不消耗手工纠偏额度；只统计追加4步中的手工动作。每次观测的incoming始终使用实际执行动作，不能用worker建议代替。每个动作记录阶段、真实remaining、建议概率/mode、实际动作、来源、是否手工干预、是否与建议不同、纠偏前状态及次数；私有动作种子取自design，不要求schedule条目包含种子。

所有尝试全保留。按整局固定repeat0–2为train、3–4为development_holdout，同一局不能跨split；两组都遵循相同划分。该留出仍属于同世界开发数据，不是独立自主确认。当前入口不构造优化器或训练修复，训练预算及新确认要在后续训练的独立check前固定，不允许从此次结果事后挑best模型或改旧门槛。

## 服务器命令

在服务器项目根目录同步两个新增Python文件及文档，保留旧代码指纹和完整旧产物。无需重跑旧修复、30局确认或已完成的10局干预。入口从下列来源报告自动找回原诊断、check、模型、verify、缓存和目标库依赖；这些路径及产物必须仍存在。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_CONT_SOURCE=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t05_outputs/terminal_intervention_evaluate_20261008T081058
T05_CONT_TAG=$(date -u +%Y%m%dT%H%M%S)
T05_CONT_CHECK="$PWD/relevance_map/t05_outputs/closed_loop_continuation_check_$T05_CONT_TAG"
T05_CONT_EVAL="$PWD/relevance_map/t05_outputs/closed_loop_continuation_evaluate_$T05_CONT_TAG"

python scripts/t05_closed_loop_continuation.py check \
  --intervention-dir "$T05_CONT_SOURCE" \
  --output-dir "$T05_CONT_CHECK" --device cuda:0
```

只有上述check完整PASS后才运行下一条。check不开环境，检查原完整十局/诊断/30局/模型和代码SHA256、冻结加载、真实历史增量接口、预算、反馈和持平处理、最多一次纠偏、真实结束、实际incoming、种子、整局划分、重复/缺测等拒绝守卫。内存守卫样例不会写成真实轨迹。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_closed_loop_continuation.py evaluate \
  --intervention-dir "$T05_CONT_SOURCE" --check-dir "$T05_CONT_CHECK" \
  --output-dir "$T05_CONT_EVAL" --device cuda:0
```

如果换终端，重新设置这四个变量，并使用已通过check的实际目录；不要重新生成同名目录覆盖已有尝试。运行报错保留原目录并反馈，不自动续跑或补样。修改新入口后必须使用新的check与输出目录，旧失败保留。纯check无需headless，真实evaluate必须显式headless。

## 产物与验收

check输出`report.json`、`design.json`、`targets.npz`与`targets.png`。预计最后为`T05_CLOSED_LOOP_CONTINUATION_CHECK`；状态允许scope WARN，但不能有FAIL。

真实evaluate逐局预先建立目录，保存完整0–84帧轨迹/RSSM状态、实际onehot incoming、原生动作事件/终止/遥测、全程视频、控制记录、逐帧测量与真实增量/完整前缀核对。失败和提前结束照样保留。写入失败或环境关闭失败停止后续启动，避免继续执行却无法留存证据。汇总包含：

- `report.json`、`evaluation_manifest.json`：工程验收、来源身份、SHA256、动作与环境成本；`behavior_accepted`和`t06_approved`始终false。
- `diagnostics.json`、`trials.csv`、`frame_metrics.csv`：自身第64→80帧进展、第80→84帧续段进展、两目标视觉距离、真实位姿、建议与执行来源。独立历史不能逐repeat当同状态反事实。
- `coverage.json`：手工纠偏分支、已更近目标1的保持分支各有多少真实执行，以及纠偏后/保持中是否每个后续观测都保留目标1视觉偏好。缺覆盖标WARN，禁止补样。全过程偏好维持也不自动批准训练或T06。
- `initial_balance.json`：各局第64帧RGB、位姿、RSSM差异只做诊断，不作筛选。
- `continuation_data.json`及各完整局的`continuation_endpoint.npz`：保存实际终点RGB/heatmap/目标编码，绑定本局state80–83→incoming81–84、remaining4–1、整局split及原始轨迹SHA256。每个完整局都索引，包括失败；未完成/接口失败仍保留记录，但不生成虚假的完整四标签。
- `outcomes_repeat_0.png`至`outcomes_repeat_4.png`及各局`video.mp4`：原目标1、第64帧、第80帧和真实结束画面；图集明确标出失败、提前结束和未执行。

完整10局的预计工程计数是env.step840、worker查询200、worker执行180、手工干预20、最多5次turn_down纠偏、概率重现200条、测量210帧。手工干预次数与真正改变worker建议的次数分开；noop恰等于worker建议也仍是手工来源。完整续段10条、事实动作索引40个，整局train24个/development_holdout16个；最多20个来自手工条件。提前终止不强求这些完整计数，缺测/错误不判工程有效，不以不足五个完整终点计算该组完整均值。

工程PASS不证明反馈规则有效。下一步须看是否实际覆盖纠偏与保持两类边界，纠偏之后是否维持目标1偏好，视觉距离/位姿/全视频是否一致，以及是否出现错误下转或noop漂移。若数据足以修复，再独立设计混合真实监督与原整局留出回退检查。未来监督目标只能是各局真正实现的终点，不能把失败动作指定在旧目标1下模仿；`approved_for_training=false`表示此次只交付待检查的事实索引，无专家最优标签。

T06仍需修复后的新自主worker按[原门槛](t05_to_t06_gate.md)通过两个目标各≥4/5正确方向、平均进展为正且胜no_goal/swapped_goal的新30局确认，并核对视频。手工反馈数据不能替代确认，不能改原16步终点或旧gate。

请反馈evaluate的完整日志，以及`report.json`、`diagnostics.json`、`coverage.json`、`continuation_data.json`、`trials.csv`、`frame_metrics.csv`、`initial_balance.json`、`evaluation_manifest.json`和五张结果图；保留所有视频供核对。中文Git提交备注：`feat: 增加T05真实视觉纠偏与保持续段采集`
