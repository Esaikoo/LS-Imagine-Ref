# T05：固定预热后的新参考与六组真实控制评估

当前状态（2026-10-07）：`check_event_fixed_20261007T061744`通过，事件类型修复已验收；同次prepare因真实位置/RGB超限失败，起点位置差0.858020，65帧原生动作全部一致。当前停止重跑以下历史命令，不执行此入口evaluate；独立起点随机分组评估已本地实现、服务器待验收，见 [新两步入口](t05_random_control_acceptance.md)。原失败分析见 [本次失败分析与后续设计](t05_warmed_start_failure_analysis.md)。

2026-10-07。本地已实现 `goal_warmed_control.py / scripts/t05_warmed_control.py check/prepare/evaluate`，服务器验收待运行。助手仅静态阅读、编辑和核对差异，未运行Python、测试、模型或环境，未提交/推送。旧工具和其代码指纹保持不变。当前仍为T05，T06未开始。

历史反馈：`warmed_control_check_20261007T060441`通过，但prepare第二条参考起点比较发生实时NumPy动作与JSON列表的类型错误。本地已修复并增加`live_saved_event_contract`；随后服务器check已验收该修复，当前停止修复说明中的重跑命令，旧不完整prepare不能用于evaluate。

## 本轮具体做什么

沿用已通过的 `start_probe_20261007T051726`，不重跑audit/check/probe。冻结残差worker第200步及所有原模型；不再训练BC、不重建T04缓存。

1. 离线check核对已通过probe的两段真实历史、当前模型/环境以及新接口。
2. prepare启动两个新环境，每个执行32步noop＋原32步共同前缀＋一个固定16步参考脚本。两个参考脚本来自旧真实执行，目标画面和目标向量必须重新采集。成功时每条81帧，共160次env.step。
3. evaluate按提前保存的随机顺序运行首个12-trial块：六组各执行两个新目标。每个trial是32步noop＋32步共同前缀＋最多16步控制；全段最多960次env.step。先反馈结果，不自动运行剩余24次。

六组为有目标残差、独立无目标残差、冻结BC底座、零目标、交换目标、固定参考动作重放。`mode`执行方式、3次重复和随机顺序种子0在check前固定，三步必须一致。候选模型不参与选目标。

新参考和每个trial的起点都检查完整65帧的物理/原生动作历史以及最后固定4帧（61–64）的RGB。阈值沿用位置0.05、朝向0.25、奖励1e-6、RGB MAE 1和P99 8，并要求物品/健康/结束标志/宏动作/原生动作一致。完整RGB、heatmap和RSSM严格比较仍保存为诊断，不宣称两次历史的隐藏状态相同。

不丢弃预热历史，不在第32或64帧伪造reset。`obs[64]`产生第一条控制动作，该动作记录在`action[65]`，随后remaining由16递减到1。每次RSSM均从本次真实reset逐帧恢复，并与本次完整历史重算核对。

## 服务器三步命令

先拉取新代码，在原`ls`环境执行。下列变量完整定义，不依赖此前终端变量；旧adopt只提供场景/固定脚本和来源记录，不能代替新prepare。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_SOURCE_BENCHMARK="$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
T05_STABILITY_PROBE="$PWD/relevance_map/t05_outputs/start_probe_20261007T051726"

T05_WARM_TAG="$(date +%Y%m%dT%H%M%S)"
T05_WARM_CHECK="$PWD/relevance_map/t05_outputs/warmed_control_check_$T05_WARM_TAG"
T05_WARM_BENCHMARK="$PWD/relevance_map/t05_outputs/warmed_control_prepare_$T05_WARM_TAG"
T05_WARM_PILOT="$PWD/relevance_map/t05_outputs/warmed_control_pilot_$T05_WARM_TAG"

python scripts/t05_warmed_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_SOURCE_BENCHMARK" \
  --stability-probe-dir "$T05_STABILITY_PROBE" \
  --output-dir "$T05_WARM_CHECK" --device cuda:0 \
  --repetitions 3 --randomization-seed 0 --execution-policy mode
```

第一步应以`[PASS] T05_WARMED_CONTROL_CHECK`结束；`accepted_stability_probe / fixed_reference_scripts / residual_execution_contract / live_saved_event_contract / new_reference_contract / no_training / source_inputs_unchanged`应为PASS。`accepted_probe_history_0/1`重算误差应在既有数值容差内。guard的PASS表示错误输入被正确拒绝；混合类型检查覆盖实时NumPy与保存JSON，不能只检查两侧都从文件读取。内存marker不会写入真实数据，新增环境步为0。

只有第一步通过才执行第二步：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_warmed_control.py prepare \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_SOURCE_BENCHMARK" \
  --stability-probe-dir "$T05_STABILITY_PROBE" \
  --check-dir "$T05_WARM_CHECK" \
  --output-dir "$T05_WARM_BENCHMARK" --device cuda:0 \
  --repetitions 3 --randomization-seed 0 --execution-policy mode
```

第二步应以`[PASS] T05_WARMED_CONTROL_PREPARE`结束；`new_reference_history_0/1 / new_real_endpoints / reference_start_acceptance / predeclared_design / no_training / source_inputs_unchanged`应为PASS。成功时`completed_references=2 / new_env_steps=160 / worker_version=200 / optimizer_updates=0`。新目标间距离需要达到旧设计中提前声明的最小值；距离变小或脚本提前结束会失败，不能挑选终点或自动补跑。

查看新目录的`targets.png`：左为预热后的新控制起点，中/右为两个新真实终点。它们无需与旧targets相同。`case_0/goals.npz`里的图像、heatmap、向量必须来自这两条新轨迹最后一帧；evaluate会独立重编码并核对。`benchmark.json / reference_plan.json / schedule.json`保存新内容身份、完整动作和六组36次顺序。

只有第二步通过才执行第三步；若失败，先反馈该步产物，停止后续命令：

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_warmed_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_SOURCE_BENCHMARK" \
  --stability-probe-dir "$T05_STABILITY_PROBE" \
  --check-dir "$T05_WARM_CHECK" --benchmark-dir "$T05_WARM_BENCHMARK" \
  --output-dir "$T05_WARM_PILOT" --device cuda:0 --blocks 1 \
  --repetitions 3 --randomization-seed 0 --execution-policy mode
```

第三步先完成12次。工程验收应为`stage_execution / comparable_blocks / no_training / source_inputs_unchanged`均PASS，结尾`[PASS] T05_WARMED_CONTROL_EVALUATE`；`completed_trials=12 / eligible_trials=12 / planned_trials=36`。若每次都完整执行16步控制，本段env.step=960；起点拒绝或真实提前结束会少于960，不能补满步数。首段的`design_completion / behavior_acceptance`及scope WARN正常，表示尚待行为分析。

`START MISMATCH`会显示实际失败项和最后4帧最大RGB误差。该trial保存真实历史但不执行控制、不重试；整块12次不进入效果比较，最终报告FAIL。失败不能靠重复运行至成功或续跑旧24次解决，应先分析失败明细。

## 反馈什么、怎样看效果

prepare反馈`report.json / targets.png / case_0/reference_diagnostics.json`。起点失败时另附`case_0/start_comparison.json / case_0/start_diagnostics/diagnostics.json / case_0/start_diagnostics/prefix_pairs.png`；部分轨迹和视频仍保存。

pilot反馈`report.json / diagnostics.json / trials.csv / outcomes_seed_0_repeat_0.png`。起点失败时另附对应trial的`start_comparison.json / start_frame_statistics.csv`；完整轨迹、原生事件和逐步控制概率在trial子目录保留。

工程PASS只说明可以比较这12次。行为上检查：有目标是否比独立无目标/底座更靠近指定终点，交换目标是否改变去向，参考重放自身的终点波动有多大。目标距离下降和任务success分别记录；本次16步视觉控制不要求完成harvest任务。一个世界的首块不构成统计结论，不自动批准T06。

若后续决定完成剩余重复，入口支持`--previous-eval-dir`，但只接受同一新benchmark、同模型/代码/环境、完整且工程通过的块，并核对每条轨迹/报告/视频SHA256。旧pilot、失败块、不完整块或重复trial均拒绝。本轮先不运行续跑。

三个命令只写各自独立目录，不复制大型checkpoint或缓存；保留源缓存、冻结依赖、残差verify、旧脚本来源和已通过probe。启动环境前检查MP4编码和磁盘余量；不会清理旧结果。

## Git中文提交备注

```text
feat: 增加T05固定预热新参考采集与六组真实评估

- 复用已通过稳定性probe，固定32步noop和32步共同前缀
- 重采两条真实参考及终点目标，不复用旧目标向量
- 按完整物理历史和最后4帧RGB验收起点，保留严格状态诊断
- 预声明六组随机对照，支持首块12次和完整块内容校验续跑
- 补充服务器命令和验收说明，保持第200步模型及T06暂停
```
