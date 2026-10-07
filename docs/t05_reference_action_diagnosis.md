# T05：成功参考脚本历史上的worker动作诊断

2026-10-07。本地新增 `goal_reference_action_diagnose.py / scripts/t05_reference_action_diagnose.py`，只做静态阅读、编辑及差异核对；未运行Python、测试、模型或环境，服务器验收待反馈。原标定、sample及既有推理代码不修改，不重跑已通过的标定。

## 本步检查什么

已通过的`reference_calibration_probe_20261007T140544`包含四条真实完整历史，两脚本都能接近自己的固定目标。本步逐条读取第64–79帧的原因果状态，在remaining=16–1时查询同一第200步残差worker，并用第65–80帧的真实incoming动作作已知脚本标签。共64个动作前查询，不查询第80帧之后的动作。

每个状态分别查询goal0、goal1、独立no_goal、zero_goal、base五种部署分布，再按该局脚本目标映射为correct/swapped。保持自身状态和预算固定，保存：

- 对实际下一脚本动作的概率、NLL、排序、mode匹配与top3匹配；相同概率按动作编号稳定排序，与原argmax规则一致。
- `no_goal_minus_correct_nll / swapped_minus_correct_nll`：正值表示正确目标更支持该局已知脚本动作；它们不能解释为另一目标的反事实正确率。
- 两目标分布TV、mode变化及原动作偏好；共同前四步和第5步后的分叉分开汇总。
- 按四条run、目标、脚本步和实际动作分别汇总；报告第一次轨迹内mode不匹配，不能称为实际自主轨迹的分歧时刻。

一个已知有效脚本不等于每一步都是唯一最优动作。相同起点下两个正确脚本前四步都是左转，不能以“所有状态必须换mode”为门槛。无目标分支按两个目标查询应逐值不变。

工具校验原`calibration_manifest.json`和完整sample/check的内容身份、全部产物SHA256及原推理代码，绑定同一模型、冻结依赖与独立verify；复核真实incoming/原生事件、保存历史验收及68个逐帧视觉/物理测量。直接使用各局原5120维状态，不构造RSSM，不重算或拼接另一局历史。只构造冻结worker和目标编码器，无优化器、MineDojo或MineCLIP；不训练脚本标签、不执行反事实动作。

64个查询来自4条历史、同一个世界；按条目和按run汇总都保留，不能当作64个独立环境实验。本步工程PASS不要求目标NLL收益为正，也不批准T06。

## 服务器运行：一条离线命令

提交/同步本地新增代码后，在服务器项目根目录执行。无需`MINEDOJO_HEADLESS=1`，该入口不会启动环境。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_CAL_PROBE="$PWD/relevance_map/t05_outputs/reference_calibration_probe_20261007T140544"
T05_SCRIPT_DIAG="$PWD/relevance_map/t05_outputs/reference_action_diagnose_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_reference_action_diagnose.py \
  --calibration-dir "$T05_CAL_PROBE" \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --output-dir "$T05_SCRIPT_DIAG" --device cuda:0
```

该命令从原标定清单自动找到原check、目标和sample。保留服务器的完整NPZ/事件/视频、check、缓存、残差模型及共享依赖；无需下载大型模型或轨迹到本地。缺失或旧内容身份不同会拒绝，已有输出目录拒绝覆盖；不要修改旧manifest绕过校验。

## 正常产物与反馈

应出现四条`[SCRIPT ACTIONS]`，并有`historical_identity / diagnostic_contracts / strict_frozen_load / fixed_visual_targets / real_query_alignment / recorded_measurement_roundtrip / same_state_script_support / no_training / source_inputs_unchanged` PASS，最后`T05_REFERENCE_ACTION_DIAGNOSE` PASS。`scope` WARN是正常的解释边界。

应记录`query_frames=64 / diagnosed_runs=4 / first_script_divergence_step=5`，新增环境步、worker执行动作和优化器更新均为0。输出：

- `report.json`：工程身份、源输入未变和冻结检查。
- `diagnostics.json / runs.csv`：全量、按run/目标/动作/阶段的支持与mode汇总。
- `frame_metrics.csv`：64行，状态帧、真实下一动作帧、remaining和各条件标量指标。
- `action_queries.json`：五种完整概率与原动作偏好、真实状态内容哈希；不是训练数据或反事实轨迹。
- `script_action_probabilities.png`：按原四条轨迹顺序展示全部64个实际脚本动作的概率/排名。

反馈终端日志、report、diagnostics、runs.csv、frame_metrics.csv及图即可。若失败，反馈error.txt；先定位该入口，不重跑原四次标定。

## 如何决定修复方向

如果已知有效轨迹上的跳跃/俯仰/前进等动作概率低、正确目标也没有比交换或无目标更支持真实脚本，优先定位监督覆盖和目标利用；仅凭本步不能确定两者谁是主因。如果真实脚本历史上的预测较合理，而自主执行仍不能稳定接近目标，才更支持闭环偏移解释。替代可行脚本可能存在，所以mode不匹配不是逐步控制失败真值。

下一次真实确认需另行提前固定完整计划，不能把本步64个查询或旧试跑拼成通过结果。T06推进门槛见 [T05到T06的原型准入条件](t05_to_t06_gate.md)。本步只交付诊断入口，尚未修改训练方案或实现该确认评估。

## Git中文提交备注

```text
feat: 增加T05成功参考历史上的worker动作诊断

- 绑定完整标定和sample产物身份，查询64个真实动作前状态
- 比较脚本动作概率、排名和NLL，分别汇总共同前缀与后续分叉
- 保持原模型和历史指纹，不新增训练或环境交互
- 明确T06小规模采集的行为门槛与待服务器验收状态
```
