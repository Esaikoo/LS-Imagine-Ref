# T05真实参考监督的固定预算修复

2026-10-08：本地实现完成，用户服务器`reference_repair_check/train/verify_20261007T164732`全部工程通过；局部开发留出支持改善，两目标NLL均优于目标消融，但原26局退步，真实自主控制未确认，见 [结果分析](t05_reference_repair_result_analysis.md)。已验收三步不需重跑或追加更新。下一步专用自主确认入口已本地实现，服务器待验收，见 [check/evaluate两步命令](t05_repaired_control_acceptance.md)。新增 `goal_reference_repair.py`、`scripts/t05_reference_repair.py check/train/verify`，不修改旧模型、评估脚本或历史指纹。修复针对已知可行脚本状态上的动作支持不足；它不是新的控制评估，也不批准T06。

## 固定实验

源模型是已验收并完成动作诊断的第200步残差worker。goal/no_goal分别继承各自源权重，使用两个全新的Adam优化器；默认学习率`1e-4`，各追加固定100次更新。保留原冻结底座、WM和目标编码器、原12个宏动作、归一化、raw偏好相加以及仅一次onehot unimix。两分支共享每次真实batch、标签、更新数和采样序列，no_goal始终用零目标。旧优化器不恢复；只有新修复快照支持本实验精确resume。

四条已完整验收的标定历史按局声明新用途：repeat 0的两局训练，repeat 1的两局留出，各32个动作标签。四局都已被用于诊断且来自同一个世界，属于开发数据；留出仅检查新训练是否泛化到另一完整重放，不是未接触测试或独立控制证据。此前标定仍是原脚本的历史事实，不能用这些训练历史证明修复worker成功。

每条监督使用该局保存的真实因果状态第64–79帧，标签为真实incoming第65–80帧，remaining为16–1，目标由**该局真实第80帧RGB/heatmap终点**重编码。它是事后实现的视觉目标，没有替换成另一局目标或地图坐标。完整reset/预热/前缀保持为来源依据，不复制、拼接或重置RSSM状态。新训练不重算WM。

每批256条：192条从原55488训练expanded rows均匀有放回抽取；64条从新32个训练条目有放回抽取，两脚本目标各32条。每个目标内均匀抽真实条目，不重新给动作设权重。100步每分支总25600标签，其中原训练19200、参考训练6400（每目标3200）。比例和预算先固定，不根据留出结果继续调参、增加更新或选best。两个分支的原200次更新和新100次更新分开记录；底座更早的BC预算见其原报告。

## 检查和指标

`check`绑定原完整sample、标定、动作诊断、源残差及验收记录的SHA256和代码身份；检查真实监督、整局划分、warm start、新优化器、分布、训练采样及RNG恢复。保存小型`reference_supervision.npz`和`coverage.json`。覆盖包括原池动作/remaining计数、去重真实动作状态计数，以及每个参考状态在**同remaining的原训练局**中32个近邻的真实动作支持；近邻仅按当前冻结状态余弦距离排序，不看未来目标、不用于训练筛选、不宣称同一物理状态。覆盖不足或相似度高都只是描述。

`train`在步0/25/50/75/100评价并保存完整`latest.pt`，没有best快照或提前择优。每次报告两种参考输入：本局真实终点（与新监督一致）和原两个固定视觉目标（与后续控制问题一致）。两者分别统计goal/no_goal/zero_goal/base/swapped的真实下一动作NLL、概率、mode匹配、目标TV；整局、两目标和第5步后分别汇总。交换输入只作预测敏感性检查，不能把原标签当另一目标的反事实正确动作。

同时评价原缓存的完整26局、14144条留出及固定去重查询，检查旧能力是否退步。原缓存和新增参考没有混合成一个好看的平均数。评价不推进训练随机数。正确目标相对独立no_goal的收益不能全部归功目标输入，必须一起看zero_goal和swapped；概率或mode变化本身也不是正确控制。

`verify`检查新格式/共享依赖/来源/计数/混合采样/优化器/RNG，保存恢复逐值一致，专用推理加载与训练分布一致，以及下一次真实混合BC更新的一致性。仅两个验收内存副本额外各更新一次，不写回训练latest。`roundtrip_verification.pt`禁止resume/控制。新格式由旧残差加载器明确拒绝，后续控制入口需专门接入新加载器；本轮不将新模型直接塞进旧评估，也不重新启动MineDojo。

三个命令都只写新目录，均无环境交互；无需`MINEDOJO_HEADLESS=1`。不上传/复制冻结bundle、原模型或大型轨迹，须在服务器保留共享依赖和check目录。

## 服务器命令

先提交、推送并在服务器拉取本次代码，然后在项目根目录逐步执行；上一条最终PASS后才执行下一条。如果出现FAIL，保留目录并反馈，不重试覆盖或增加训练预算。旧诊断和标定不用重跑。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_REPAIR_TAG="$(date +%Y%m%dT%H%M%S)"
T05_REPAIR_CHECK="$PWD/relevance_map/t05_outputs/reference_repair_check_$T05_REPAIR_TAG"
T05_REPAIR_TRAIN="$PWD/relevance_map/t05_outputs/reference_repair_train_$T05_REPAIR_TAG"
T05_REPAIR_VERIFY="$PWD/relevance_map/t05_outputs/reference_repair_verify_$T05_REPAIR_TAG"

python scripts/t05_reference_repair.py check \
  --cache-dir "$PWD/relevance_map/t04_outputs/cache_20261006T124827" \
  --source-checkpoint "$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt" \
  --residual-verify-dir "$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803" \
  --calibration-dir "$PWD/relevance_map/t05_outputs/reference_calibration_probe_20261007T140544" \
  --diagnosis-dir "$PWD/relevance_map/t05_outputs/reference_action_diagnose_20261007T155501" \
  --output-dir "$T05_REPAIR_CHECK" --device cuda:0

python scripts/t05_reference_repair.py train \
  --check-dir "$T05_REPAIR_CHECK" \
  --output-dir "$T05_REPAIR_TRAIN" --device cuda:0

python scripts/t05_reference_repair.py verify \
  --check-dir "$T05_REPAIR_CHECK" --checkpoint "$T05_REPAIR_TRAIN/latest.pt" \
  --output-dir "$T05_REPAIR_VERIFY" --device cuda:0
```

## 反馈与后续

反馈三个最终PASS/FAIL和`[REPAIR] step=0/100`日志；附件只需check的`coverage.json`、train的`initial_metrics.json`、`metrics.json`、`diagnostics.json`、`sampling.json`、`evaluation.jsonl`、`frame_metrics.csv`，以及train/verify的`report.json`（区分文件名）。不要下载checkpoint、缓存、轨迹或共享模型。

工程通过要求真实来源、整局划分/动作/剩余步数、冻结内容、双分支恢复、混合采样计数和下一更新全部通过，新环境步0。损失有有限值即可完成工程验收；收益不通过也原样报告，不能修改门槛让训练结果变PASS。

分析固定100步相对步0：两个目标的留出支持、第5步后及forward/俯仰是否改善；正确目标是否分别优于no_goal、zero_goal、交换输入；原26局留出退步程度是否可接受。32个新留出标签不能当32个独立样本，无训练改善保证，不用任意离线NLL阈值批准T06。

若修复值得继续，再冻结此版本及执行方式，接入**新**自主确认入口和新随机计划，按 [T06准入条件](t05_to_t06_gate.md) 完成30次、两目标分别至少4/5正确方向，并分别优于无目标和交换组，之后才进入限定场景小规模T06。若修复未显示目标额外作用，先分析这次覆盖和固定预算结果，不再次切换mode/sample或立即扩大控制试跑。

中文Git提交备注：`feat(T05): 接入整局真实参考监督并实现同预算双分支残差短训练、覆盖审计与恢复验收`
