# T05：独立起点、随机分组的真实目标控制

2026-10-07。本地已实现 `goal_random_control.py` 和 `scripts/t05_random_control.py`；服务器待验收。助手只做静态阅读、编辑和差异核对，未执行Python、测试、训练、模型或环境，未提交/推送。仍为T05。

这次保留残差worker第200步、无目标底座、WM和目标库，结束同起点重放门槛循环。入口只有两步：`check`离线固定目标/计划；`evaluate`执行完整12次。无需再运行旧prepare、新参考采集、双次probe、信息探针或BC训练。

## 为什么改评估

早期worker基本忽略目标，独立无目标BC对照证实收益很小；归一化残差分支已带来有限的离线收益，实际控制仍待验证。最近卡住的是环境重放：相同原生动作仍产生约0.86格位置差，worker尚未开始控制。先前的强配对要求及反复预检增加了不必要的迭代。

新问题是：同一种环境初始化过程下，有目标策略的平均目标进展是否优于独立无目标策略，交换目标是否改变去向。每局保留自己的真实历史，比较独立运行的结果。位置/RGB/RSSM差异用于描述环境波动与起点平衡，不再用于拒绝一个本来有效的真实起点。

这是一份独立的新实验，不能将旧FAIL改成PASS。也没有随机瞬移或人为清理动物：仍使用明确world_seed=1和已有初始化设置，条件的运行顺序提前随机化，环境自然波动原样保留。

## 固定内容

- 主模型：`residual_train_20261007T031803/latest.pt`，第200步，两套修正分支均冻结；不用候选模型/高层。
- 两个目标：来自完整旧 `residual_control_adopt_20261007T040055` 的真实RGB/heatmap终点，重编码并核对内容。只代表提前固定的视觉条件，不保证从每个实际起点可达，也不声称是每局起点的两个实际分支。
- 三组：`goal`给指定目标；`no_goal`使用独立训练的无目标修正；`swapped_goal`给另一目标，但指标仍针对原指定目标。交换组的`issued_target_margin`另行保存。
- 两目标×三组×两重复=12次；每重复六种组合随机排序。每次fresh reset，32步noop＋32步原前缀＋最多16步worker控制。完整运行最多960个`env.step`，控制动作最多192次。真实结束立即停；提前结束/环境错误保留，不补跑。
- mode执行，种子和顺序在check固化；evaluate必须提供相同参数。每局从真实reset增量恢复RSSM，并重算自身完整历史核对。
- 整轮不重试、不从旧FAIL续跑、不复制隐藏状态；程序中断后保留部分产物，本入口不提供自动恢复，先反馈失败原因。

## 服务器运行：只需两步

先提交/推送本地修改，在服务器拉取。以下变量全部明确指定，不依赖上一轮终端残留。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_TARGET_SOURCE="$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
T05_RANDOM_TAG="$(date +%Y%m%dT%H%M%S)"
T05_RANDOM_CHECK="$PWD/relevance_map/t05_outputs/random_control_check_$T05_RANDOM_TAG"
T05_RANDOM_EVAL="$PWD/relevance_map/t05_outputs/random_control_evaluate_$T05_RANDOM_TAG"

python scripts/t05_random_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_TARGET_SOURCE" \
  --output-dir "$T05_RANDOM_CHECK" --device cuda:0 \
  --case-seed 0 --warmup-steps 32 --repetitions 2 \
  --randomization-seed 0 --execution-policy mode
```

第一步出现 `[PASS] T05_RANDOM_CONTROL_CHECK` 后再执行第二步。如果FAIL，先反馈日志，不运行后续、不反复重跑。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_random_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --source-benchmark-dir "$T05_TARGET_SOURCE" \
  --check-dir "$T05_RANDOM_CHECK" \
  --output-dir "$T05_RANDOM_EVAL" --device cuda:0 \
  --case-seed 0 --warmup-steps 32 --repetitions 2 \
  --randomization-seed 0 --execution-policy mode
```

无需再次跑旧的warmed prepare/evaluate，旧剩余24次也不执行。本轮不新增大型checkpoint，不复制缓存/权重/旧视频；开始前预留轨迹视频空间，保留磁盘余量。

## 工程验收应该看到什么

check中应看到 `fixed_visual_targets`、`causal_execution_interface`、`independent_start_contract`、`predeclared_design`、`no_training`、`source_inputs_unchanged` PASS；保存 `design.json / targets.npz / targets.png`。设计应为12次，最大960步，新增真实环境步0。

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

每trial保存全部真实RGB/heatmap/incoming/5120维状态、原生事件和MP4，不插入训练回放；`control_trace.json`记录目标、每步分布、动作、remaining与两个目标距离。有效trial的`history_check.json`检查它自己的完整reset历史；不与其它trial隐藏状态配对。

主要看：有目标是否比独立无目标更有进展；交换目标是否转向另一个目标；两目标都是否有相应方向；初始距离/位置/画面是否存在明显组间不平衡。`diagnostics.contrasts`是独立起点均值差：`progress_advantage`、`evaluated_target_margin_advantage`越大越好，`end_distance_difference`越小越好；同时看`initial_distance_difference`。

这12次仅涉及一个世界，两个重复，不能给出稳健显著性或成功率提升结论。可能得到“接口通过但控制弱”或“初始波动太大，效果不确定”；两者都比worker尚未执行就被配对门槛拒绝提供更多信息。T06仍等待真实行为分析，程序不会自动推进。

## Git中文提交备注

```text
feat: 增加T05独立起点随机分组目标控制评估

- 复用已验收第200步残差worker与固定真实视觉目标
- 两步完成离线随机计划与三组12次真实控制
- 按每局完整真实历史恢复状态，起点差异只用于平衡诊断
- 保留全部尝试和错误，核对部署动作、冻结参数及输入身份
- 停止旧重放门槛循环，补充运行验收说明和相关论文参考
```
