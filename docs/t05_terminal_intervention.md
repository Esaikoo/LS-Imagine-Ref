# T05自主轨迹的真实末步干预

2026-10-08。本地新增`goal_terminal_intervention.py`与`scripts/t05_terminal_intervention.py check/evaluate`，验证在新的自主目标1轨迹上，最后执行固定forward是否改善真实终点。只完成静态阅读、修改和差异核对，未运行Python、测试、模型、训练或环境，未提交/推送；服务器验收与效果待用户运行。旧验收代码、原30局、worker300和既定T06门槛保持。

后续用户反馈`terminal_intervention_check/evaluate_20261008T081058`工程完整通过：10局/800步、控制查询160、worker执行155、固定干预5，概率重现误差0。原mode方向2/5、固定forward3/5；真正改变建议的两局均失败，成功三局本来就建议forward，不能声称干预救回失败。固定forward失败两局在前15步未调对目标1角度，mode失败三局则在末步离开正确朝向。详见 [结果分析](t05_terminal_intervention_result_analysis.md)。本页命令保留供复现，本次无需重跑；下一步应取得真实闭环纠偏证据，T06未批准。

下一步独立 [闭环纠偏与保持续段入口](t05_closed_loop_continuation.md) 已本地完成，运行该页check/evaluate即可；服务器待验收。原十局及本页入口保持，不重跑或改写旧结果。

## 固定实验

依据是已完整通过的`terminal_action_diagnose_seed_fixed_20261008T070142`：目标1失败三局最后实际选turn_up，成功两局选forward，但失败状态上的forward未真实执行。新实验只验证真实续段，不能把成功参考动作直接转为失败自主状态的专家标签。见 [诊断分析](t05_terminal_action_result_analysis.md)。

| 项目 | 预声明内容 |
| --- | --- |
| 模型与目标 | 同一冻结worker300、原目标1真实RGB/heatmap终点；另保留目标0作终点方向测量，不输入坐标控制。 |
| 每局初始化 | 独立fresh reset，32noop+32原真实前缀，保留本局完整因果历史。 |
| 前15步 | worker300以目标1、mode执行，remaining16到2。 |
| 第16步 | `worker_mode`组执行原mode；`fixed_forward`组执行预声明forward；两组都先查询worker，remaining1。 |
| 顺序与预算 | 两组各5次，共10次；新随机种子2，顺序在check保存；最多800个真实env.step，包含初始化。 |
| 保留规则 | 不筛第15步状态、画面或距离，不重试、补样、替换目标、选最佳中间帧或续跑失败目录。 |
| 解释范围 | 一个世界的开发实验；组间独立reset，没有共享环境/RSSM快照，不宣称同状态配对反事实收益。 |

control_trace同时保存worker建议、实际动作、原分布、remaining和干预来源。即使末步worker本来就建议forward，固定forward组仍记`intervention_applied=true`，而`action_differs_from_worker=false`；两种计数分开。下一状态的incoming始终使用真正执行的动作。

第15步画面固定为frame79，只用于末步分析，主终点仍为实际第16步后的frame80。真实提前结束立即停止动作并保留其最后观测，单列为提前结束；接口异常与缺测保留为失败，不能生成完整终点或五次均值。存储或环境关闭失败后停止启动后续环境，保留可写的记录，不支持续跑。

## 服务器运行

用户提交/推送、服务器拉取后，在原`ls`环境执行下面两步。此前的修复训练、独立verify、30局确认和末步诊断均无需重跑。新入口从成功诊断报告推导原模型、cache、check、verify和标定路径；须保留这些原目录及完整真实历史。小附件不能代替服务器轨迹或checkpoint。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_LAST_TAG="$(date +%Y%m%dT%H%M%S)"
T05_LAST_DIAG="$PWD/relevance_map/t05_outputs/terminal_action_diagnose_seed_fixed_20261008T070142"
T05_LAST_CHECK="$PWD/relevance_map/t05_outputs/terminal_intervention_check_$T05_LAST_TAG"
T05_LAST_EVAL="$PWD/relevance_map/t05_outputs/terminal_intervention_evaluate_$T05_LAST_TAG"

python scripts/t05_terminal_intervention.py check \
  --diagnosis-dir "$T05_LAST_DIAG" \
  --output-dir "$T05_LAST_CHECK" --device cuda:0
```

check通过后，再运行真实实验：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_terminal_intervention.py evaluate \
  --diagnosis-dir "$T05_LAST_DIAG" --check-dir "$T05_LAST_CHECK" \
  --output-dir "$T05_LAST_EVAL" --device cuda:0
```

输出目录必须全新，拒绝覆盖源目录或已有输出。入口不提供更换worker/目标/预算/执行方式/种子的参数；执行设计与新check逐项核对。check不启动环境，先验证原SHA256身份、冻结推理、因果接口、目录原子写入、真实字段结构的内存守卫以及两种干预记录。旧验收指纹对应的代码没有修改。

## 工程验收与反馈

正常完整运行应看到：

- `[PASS] T05_TERMINAL_INTERVENTION_CHECK`和`[PASS] T05_TERMINAL_INTERVENTION_EVALUATE`。
- 计划尝试10/10、真实保存接口10/10、完整16步终点10/10；两组各5次，没有重试或补样。
- `new_env_steps=attempted_env_steps=800`，`worker_action_queries=160`，`worker_control_actions=155`，`intervention_actions=5`，`control_actions=160`，训练更新0。`worker_action_queries`计真实控制前的查询，不包含保存后的只读重现。
- 保存后的160次概率/建议/实际动作重现，以及170帧目标测量核对；报告最大概率误差，完整因果历史、原生事件、遥测和视频均有效。
- 固定forward实际改变worker建议的次数为0到5，不能强制算为5。scope WARN是预期边界，不代表控制效果通过。

提前结束可形成有效真实历史，但不能记为完整16步，也不能用少于5个完整终点输出该组的全五次均值。任何执行/保存异常都进入失败记录；工程PASS不使用进展数值作门槛，行为结果仍需分析。

反馈新evaluate目录的`report.json`、`diagnostics.json`、`trials.csv`、`frame_metrics.csv`、`initial_balance.json`、`evaluation_manifest.json`及`outcomes_repeat_0.png`至`outcomes_repeat_4.png`。若失败，另反馈日志和`error.txt`，不覆盖或直接续跑。

每个trial还保存`start.json`、`control_trace.json`、`history_check.json`、`metrics.json`、`frame_metrics.json`、`trajectory.npz`、`events.json`和`video.mp4`。根目录保留`design.json`、`targets.npz`、全部尝试日志与trial汇总；manifest逐项绑定真实文件、CSV和图集SHA256。无需把模型或整批轨迹下载到本地。

结果优先比较每局自己的最后一步距离变化、俯仰/位置变化、完整16步进展和终点是否更近目标1，并检查两组初始/第15步位姿。报告两组全五次描述性均值，不把不同历史相减当同状态因果效果，不声称统计显著或语义任务成功提升。

## 后续与T06

本轮不训练、不生成专家标签，也不批准T06。即使固定forward有效，它仍是预声明干预，不能说明worker自己会选对末步。有效真实轨迹可为后续闭环监督提供证据，但训练标签只能用实际执行动作，目标只能来自该局实际终点；训练预算、整局划分及新自主确认需另行固定。

T06仍要求修复后的自主worker在两个目标上分别至少4/5正进展且终点偏好正确，各自平均进展为正并胜no_goal与swapped_goal，且通过视频/位置/朝向核对。旧目标1门槛失败保持；详见 [T06准入规则](t05_to_t06_gate.md)。

中文Git提交备注：`feat: 增加T05自主轨迹末步mode与forward真实干预实验`
