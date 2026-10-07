# T05：当前初始化流程下的固定参考动作标定

本地实现完成，用户服务器`reference_calibration_check / reference_calibration_probe_20261007T140544`已通过工程验收：4次完整重放、320步，四次对应目标距离0.016–0.019，记录位置/yaw/pitch误差均为0。助手只静态阅读、核对附件及编辑文档，没有运行Python、测试、模型或环境，也没有提交/推送。仍在T05，T06未开始。

详细分析见 [本次标定结果](t05_reference_calibration_result_analysis.md)。两脚本在该流程下能分别接近对应目标，后续 [worker动作支持诊断](t05_reference_action_diagnosis.md) 也已用户服务器通过，当前转向 [真实监督和训练修复](t05_reference_action_result_analysis.md)。下列两步命令是已完成运行的记录，无需重新执行。

sample试跑已完成12次有效执行，但goal/no_goal/swapped平均视觉进展为0.061381/0.098416/0.125034，全部终点仍更近target0。本步用已知真实动作定位目标基准和视觉度量问题，停止继续调整mode/sample或延长训练。

## 本步测量什么

目标仍是完整`residual_control_adopt_20261007T040055`保存的两个真实RGB/heatmap终点，经冻结目标编码器得到的128维向量；位置/朝向仅是附加诊断，不是目标输入。

两个动作脚本来自该来源各分支的`action[33:49]`，即原第32帧起点之后的16个真实动作。这次每局重新执行32步noop和32步前缀，再从第64帧起点执行同一脚本。额外预热会改变真实环境历史，所以不能假设脚本仍到达原终点。

- 两脚本各两次，共4次独立fresh reset；每重复的脚本顺序提前随机并固化。最多320个`env.step`，其中固定脚本动作最多64次，worker控制动作0。
- 沿用完整sample试跑绑定的世界种子、前缀、视觉处理、模型、缓存和环境。已有check的代码指纹保持；新入口有自己的身份。
- 不以不同局的位置、RGB、heatmap或RSSM相等作为筛选条件。每局保留自己的完整reset历史、真实incoming和原生事件，重算自身因果状态。
- 实际动作必须按脚本顺序，与`obs[t] -> action[t+1]`一致；没有策略采样或模型动作选择。原生动作的sticky效果受真实历史影响，其与旧参考的差异单独记录，不冒充原生执行完全相同。
- 无重试、替换脚本、补样或自动恢复。真实结束立即停，提前结束单列；环境/动作/历史错误保留并FAIL，跨局漂移或效果差作为测量，不强求正收益。
- 不产生新训练样本、不替换旧目标、不重新批准旧失败或重建benchmark。完整16步终点为主；途中最小距离仅诊断，不挑选中间帧充当成功终点。

## 服务器两步命令

以下为已完成服务器运行的命令，保留供查阅；本次结果分析不新增运行命令，也无需重跑旧sample/check、稳定性probe、warmed prepare或训练。

第一步离线check，不启动MineDojo：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_SAMPLE_EVAL="$PWD/relevance_map/t05_outputs/random_control_evaluate_sample_20261007T114359"
T05_CAL_TAG="$(date +%Y%m%dT%H%M%S)"
T05_CAL_CHECK="$PWD/relevance_map/t05_outputs/reference_calibration_check_$T05_CAL_TAG"
T05_CAL_PROBE="$PWD/relevance_map/t05_outputs/reference_calibration_probe_$T05_CAL_TAG"

python scripts/t05_reference_calibration.py check \
  --eval-dir "$T05_SAMPLE_EVAL" \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --output-dir "$T05_CAL_CHECK" --device cuda:0 --randomization-seed 0
```

出现`[PASS] T05_REFERENCE_CALIBRATION_CHECK`后，第二步执行固定4次重放：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_reference_calibration.py probe \
  --eval-dir "$T05_SAMPLE_EVAL" \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --check-dir "$T05_CAL_CHECK" --output-dir "$T05_CAL_PROBE" \
  --device cuda:0 --randomization-seed 0
```

输入来源从已通过sample的manifest读取，不另手填失败prepare目录。check/probe参数和代码身份必须一致；已有输出目录拒绝覆盖。保留服务器已有缓存、底座、verify、原目标及sample的完整产物，工具在服务器只读并验真；无需下载大型轨迹或模型到本地。

## 应看到的工程验收

check应有`historical_identity`、`fixed_reference_scripts`、`fixed_visual_targets`、`trial_output_preflight`、`causal_history_interface`、`calibration_contracts`、`predeclared_design`、`no_training`和`source_inputs_unchanged` PASS。保存`design.json / targets.npz`；输出预检JSON明确标记，不是实际轨迹。

probe完整执行应有四条`[REPLAY n/4]`，以及：

```text
[PASS] planned_attempts: ...4/4...
[PASS] real_script_interface: ...4/4...
[PASS] fixed_horizon_endpoints: ...4/4...
[WARN] behavior_acceptance: ...需结合视频分析...
[PASS] no_training: optimizer_updates=0...worker控制动作0...env.step=320...
[PASS] source_inputs_unchanged: ...
[PASS] T05_REFERENCE_CALIBRATION_PROBE
```

320仅适用于4次全部完整执行。提前终止会少于320，`fixed_horizon_endpoints`发出WARN且单列实际长度；历史仍须有效。环境、动作、遥测或文件错误会FAIL；中断或文件错误时保留已有产物并停止，先反馈，不反复重跑。每次真实输出验证数组/JSON事件保存往返及视频完整解码。

工程PASS不要求到达目标，也不表示worker控制成功。两重复、一个世界只能用于定位，不能作为总体可达率或显著性结论。

## 产物与解释

请反馈第二步目录的`report.json`、`diagnostics.json`、`runs.csv`、`frame_metrics.csv`、`calibration_manifest.json`、`endpoint_matrix.png`和`starts_and_outcomes.png`。必要时查看`run_*/video.mp4`，失败时附`error.txt / run_*/error.json`。

每run保存完整81帧真实历史（提前结束则更短）、原生事件、MP4、`start.json`、`history_check.json`、`metrics.json`和`script_trace.json`。trace含固定脚本、实际原生事件，以及从自己第64帧起点至终点的两目标距离和编码。没有候选模型、高层或worker动作概率查询。

`frame_metrics.csv`同时记录两个目标的cosine距离、RGB MAE/P99、heatmap MAE、位置/朝向误差。`endpoint_matrix`始终表示“脚本行 × 原目标列”，不是训练标签或任务成功率。`diagnostics`保留每次结果、两脚本各自平均进展、原目标之间及新脚本终点之间的差异；提前结束不混入完整16步矩阵。

优先看脚本1是否两次都更接近target1，以及脚本0是否两次更接近target0，再结合自己的起点进展和视频。如果脚本也普遍偏target0，当前原目标与预热初始化可能不匹配；这不证明目标1永不可达。如果画面和位置/朝向接近原target1而编码仍偏target0，需要进一步定位视觉度量/heatmap。如果两个脚本都能分别接近固定目标而worker不能，支持继续定位worker的目标利用和泛化。位置可能相近而视角不同，RGB像素误差也受渲染影响，任何单项都不能自动作为真值。

## Git中文提交备注

```text
feat: 增加T05固定参考动作的目标可达性标定

- 绑定已完成sample试跑及原真实目标和16步脚本，固定四次新重放
- 保留自身完整历史并核对下一动作、原生事件、因果状态和输出往返
- 同时记录两目标视觉距离、RGB和物理误差，区分工程检查与控制效果
- 更新两步服务器命令和T05状态，不训练或推进T06
```
