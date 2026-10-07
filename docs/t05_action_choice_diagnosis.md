# T05：在现有真实状态上检查目标怎样影响动作

2026-10-07。新增 `goal_action_choice_diagnose.py` 和 `scripts/t05_action_choice_diagnose.py`，只读已经完整通过的12次随机控制轨迹及原冻结模型。助手仅做本地静态阅读、编辑与差异核对，未执行Python、测试、模型或环境，未提交/推送；新诊断服务器待验收。仍为T05。

## “目标”具体是什么

当前两个目标来自 `residual_control_adopt_20261007T040055` 的两条完整真实参考轨迹。每个目标是最后一帧的RGB和对应heatmap，经固定T02视觉编码器投影/归一化得到的128维连续向量。它们不是目标类别编号，也不是地图坐标、树的位置或任务成功标签。

参考执行确实曾在某个真实位置、朝向看到这个画面；但发送给worker的是视觉向量，没有把参考坐标作为导航指令。不同位置/视角可能具有接近的向量。当前指标是 `1 - 当前视觉向量·目标向量`，改善只表示视觉编码更接近，不能证明到达同一物理位置或完成harvest。当前评估也没有确认两个目标从每个实际起点都能在16步内到达。

## 已完成的试跑说明了什么

`random_control_evaluate_output_fixed_20261007T071508`完成12/12次、192次控制和960个总环境步；全部接口有效，参数不更新，目录创建修复已验收。旧失败不改写。

| 条件 | 第一次重复平均进展 | 第二次重复平均进展 | 总体平均进展 |
| --- | ---: | ---: | ---: |
| goal | 0.189649 | -0.008350 | 0.090650 |
| no_goal | -0.007157 | 0.029607 | 0.011225 |
| swapped_goal | -0.008384 | 0.000497 | -0.003944 |

总体进展有积极信号，但优势未稳定复现。全部终点按当前编码更接近target0，初始状态本身也更接近它。第一次goal/target1终点到target1距离0.498087，到target0只有0.016491；第二次两个goal的动作计数和最终位置/朝向相同，不能据此声称完整动作序列相同。起点存在约0.712格位置差；一个世界、两次重复仅供描述性分析。任务成功字段均为假。真实目标控制尚未验收，T06未开始。

## 本次只做一个离线入口

不修改随机控制评估器、原模型或旧验收代码，不重新跑12次环境、不追加BC或信息探针，也不因为新文件加入而要求重跑旧check。新增入口先校验原评估/check/manifest、全部真实产物SHA256、旧推理代码、模型及独立verify。

对每个原真实控制动作之前的状态，读取已按SHA256固定且已通过完整历史重算的5120维因果状态。严格保持该状态和原remaining，仅分别输入两个真实目标；另计算独立无目标、零目标、冻结底座分布。独立无目标分支换目标必须逐值不变。

采用与部署完全相同的原onehot混合分布；不把混合后的log概率再当原始偏好叠加。逐步重现原记录的动作概率及动作，核对 `features[t] → action[t+1]`、真实incoming、预算、终止边界和原动作随机数。读取目标编码器用于核对固定目标与原距离，不构造RSSM或重新制造状态。

保存的信息包括：

- 完整五组动作概率和混合前偏好，两个目标的总变差TV、最大概率差、JS距离。
- 每组最高/第二高概率、差距、动作编号/名称；换目标是否改变最大概率动作；相对于独立无目标、零目标和底座的变化。
- 两目标偏好差去除共同常数后的RMS，避免把不影响softmax的整体平移当作目标作用。
- 两分布使用同一个均匀随机数、同一动作编号顺序时的解析采样分歧概率。这只计算CDF区间，不抽动作、不执行新轨迹；它不等于采样控制会更好。
- 每局实际完整动作序列及起止遥测，便于核对之前“计数相同”是否也是“序列相同”。

所有原真实控制状态都纳入，完整执行应为192个查询。每条轨迹的16个状态相关，不能当作192个独立实验；分别保存按帧、等权按局、条件、remaining、重复和原指定目标的汇总。图中只展示原执行顺序的12个控制起点，避免挑选效果好看的状态。

## 服务器运行：一条命令

用户提交/推送并在服务器拉取新增文件后执行。已有check、模型、缓存和完整12局评估继续使用；无需启动MineDojo，所以本命令不需要headless前缀。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_RANDOM_EVAL="$PWD/relevance_map/t05_outputs/random_control_evaluate_output_fixed_20261007T071508"
T05_ACTION_DIAG="$PWD/relevance_map/t05_outputs/action_choice_diagnose_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_action_choice_diagnose.py \
  --eval-dir "$T05_RANDOM_EVAL" \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --output-dir "$T05_ACTION_DIAG" --device cuda:0
```

优先使用原评估的CUDA后端。程序允许CPU分析，但后端差异会WARN，仍必须逐步通过原概率和动作重现；不为通过检查而调大数值容差。已有轨迹/模型/缓存留在服务器，不需要下载或复制，不保存新checkpoint，新增环境步0。

正常应看到 `historical_identity`、`diagnostic_contracts`、`strict_frozen_load`、`fixed_visual_targets`、`real_query_alignment`、`recorded_forward_roundtrip`、`same_state_goal_comparison`、`no_training`、`source_inputs_unchanged` PASS，最终 `[PASS] T05_ACTION_CHOICE_DIAGNOSE`。记录的工程容差是atol/rtol各3e-5，只用于重现原前向，不是目标效果通过阈值。

如果概率、动作、状态索引、模型/代码身份或原文件哈希不匹配，保留日志先定位，不修改原评估结果或自动重跑环境。诊断出现接近零的目标作用或极少mode改变也可以工程PASS，因为如实报告这些值正是本工具的目的。

请反馈新目录的 `report.json`、`diagnostics.json`、`frame_metrics.csv`、`start_action_probabilities.png`；如需逐步核对，另附 `counterfactual_frames.json`。文件小，不包含模型/训练缓存/完整5120维状态。

## 怎样判断下一步

概率和去常数偏好差都很小、mode几乎不变，支持“当前worker的目标作用弱”，但不能推出数据中没有目标信息。概率变化明显而mode不变，说明当前最大概率动作没有切换；结合top2差距再判断是否值得设计新的采样试跑，不能直接用解析分歧当行为收益。mode切换明显但当前真实结果没有对应方向，则需要另外检验目标可达性、动作方向或预算，不能只延长horizon就宣称修复。

本诊断没有执行换目标后的反事实轨迹，无法判断会到哪里；没有证明目标可达、sample更好、harvest成功或T06可以开始。先分析本次输出，再选择下一项干预，保持现有评估入口。

## Git中文提交备注

```text
feat: 增加T05真实状态上的目标动作选择离线诊断

- 复用已验收12局轨迹并核对模型、目标、产物及旧代码身份
- 在同一状态和预算上比较两目标及三种无目标对照的概率和动作
- 重现原真实动作，记录top2差距、目标敏感度和解析采样分歧
- 更新T05试跑状态与单步服务器运行说明，不修改原评估协议
```
