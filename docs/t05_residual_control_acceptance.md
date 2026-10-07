# T05：残差 worker 六组真实重复评估

2026-10-07。本地已实现，服务器待验收。助手仅静态阅读、修改和核对差异；未运行 Python、测试、训练、模型或环境，未提交/推送。本轮仍为 T05，T06 未开始。

## 本次范围

新增 `goal_residual_control.py` 和 `scripts/t05_residual_control.py check/adopt/evaluate`。固定已经验收的残差第200步 `latest.pt`，同时加载它的有目标、独立无目标修正分支及冻结BC底座。没有优化器/候选模型、额外训练或缓存重建。

原始参考来自 `prepare_visual_fixed_20261006T165045`，其两个真实终点和参考动作离线复用。参考生成策略与新残差策略不同，不要求它们模型ID相同；要求冻结WM/目标库/缓存相同，完整真实历史重算一致，任务RGB/HUD设置正确，物理/RGB/宏动作/原生动作前缀可比。原严格heatmap/RSSM失败完整保留，不宣称相同隐藏状态。旧报告缺少环境指纹时明确WARN，当前环境及新评估代码指纹在新计划中固定。

每个重复块六组×两个目标，共12次trial。默认三个随机重复块，总36次；先固定全部顺序和执行方式，再分段运行。首段结束后只允许从同一计划的完整、已通过接口验收的前缀接着执行；不重试失败trial、不按终点表现补样。旧五组30-trial计划不能换模型直接复用。

| 条件名 | 执行策略 |
| --- | --- |
| `goal` | 冻结底座＋有目标修正，输入指定真实终点 |
| `no_goal` | 冻结底座＋独立训练的无目标修正 |
| `base` | 冻结无目标BC底座，不加修正 |
| `zero_goal` | 有目标修正分支输入置零 |
| `swapped_goal` | 有目标修正分支输入另一个真实终点 |
| `reference_replay` | 重放该真实参考段的记录动作 |

## 服务器先执行三步

拉取代码，在原 `ls` 环境执行。变量重新完整定义，不依赖旧T05变量。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_SOURCE="$PWD/relevance_map/t05_outputs/prepare_visual_fixed_20261006T165045"
T05_CONTROL_TAG="$(date +%Y%m%dT%H%M%S)"
T05_CONTROL_CHECK="$PWD/relevance_map/t05_outputs/residual_control_check_$T05_CONTROL_TAG"
T05_CONTROL_BENCHMARK="$PWD/relevance_map/t05_outputs/residual_control_adopt_$T05_CONTROL_TAG"
T05_CONTROL_PILOT="$PWD/relevance_map/t05_outputs/residual_control_pilot_$T05_CONTROL_TAG"

python scripts/t05_residual_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --output-dir "$T05_CONTROL_CHECK" --device cuda:0 --prefix-steps 32
```

第一步纯离线，无MineDojo/MineCLIP。末尾应为 `[PASS] T05_RESIDUAL_CONTROL_CHECK`；`strict_residual_inference_load / residual_execution_contract / randomized_block_contract / no_training / source_inputs_unchanged`通过。guard的PASS表示拒绝无历史、终止后动作、越界预算、重复trial或失败块续跑；合成标记只在内存。原观测逐帧状态应与真实完整前缀一致。

```bash
python scripts/t05_residual_control.py adopt \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --check-dir "$T05_CONTROL_CHECK" --source-prepare-dir "$T05_SOURCE" \
  --output-dir "$T05_CONTROL_BENCHMARK" --device cuda:0 \
  --repetitions 3 --randomization-seed 0 --execution-policy mode
```

第二步纯离线，无新环境交互。末尾应为 `[PASS] T05_RESIDUAL_CONTROL_ADOPT`；`real_reference_adoption / predeclared_design / no_training / source_inputs_unchanged`通过，`planned_trials=36`。保存 `benchmark.json / schedule.json / targets.png` 和两个原始参考。`strict_pair=0`及`reference_provenance`的WARN可以存在，但物理/RGB/前缀原生动作不可比会失败，不放宽其阈值。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_residual_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --check-dir "$T05_CONTROL_CHECK" --benchmark-dir "$T05_CONTROL_BENCHMARK" \
  --output-dir "$T05_CONTROL_PILOT" --device cuda:0 --blocks 1
```

第三步启动真实MineDojo，执行首个完整12-trial块。每次fresh reset、重放相同32步前缀，随后最多16步真实控制；以实际新观测更新状态。环境提前结束立即停止，不按视觉距离阈值提前宣布成功。执行方式已在adopt固定为mode，evaluate不能临时切换。

预期最后 `[PASS] T05_RESIDUAL_CONTROL_EVALUATE`，且：

- `stage_execution`累计12/36、`comparable_blocks`为12/12、优化器更新0，所有权重和原输入文件未改动。
- `design_completion`为WARN：预声明计划尚未全部执行，这是正常的首段状态。
- 六组各2个trial；`controlled_reachability_verified=false`、`full_design_completed=false`。行为验收仍需分析，正收益不是程序PASS条件。
- 不结束时本段最多 `12×(32+16)=576` 个新 `env.step`；统计包含预热，不将缓存或离线更新计入新步数。

先反馈三步日志和首段 `report.json / diagnostics.json / trials.csv / evaluation_manifest.json / outcomes_seed_0_repeat_0.png`，再根据接口是否正常完成剩余计划。不要因为首段控制收益为负而改模型、选目标、执行方式或顺序；接口失败则保留原数据，先定位错误。

## 首段接口通过后的剩余两块

以下命令已实现；首段还未运行时不要提前执行。继续使用相同check、benchmark、模型及代码，输出另一个新目录，并保留首段全部文件。若终端变量丢失，重新设置以上变量为已经存在的实际目录，不用新的时间戳替换check/adopt/pilot路径。

```bash
T05_CONTROL_FULL="$PWD/relevance_map/t05_outputs/residual_control_full_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_residual_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --check-dir "$T05_CONTROL_CHECK" --benchmark-dir "$T05_CONTROL_BENCHMARK" \
  --previous-eval-dir "$T05_CONTROL_PILOT" \
  --output-dir "$T05_CONTROL_FULL" --device cuda:0 --blocks 2
```

预期 `complete_block_continuation`核对已有12次内容SHA256，新执行24次，累计36/36；`design_completion`为PASS，`full_design_completed=true`。完整新步数最多1728，其中本段最多1152。前段录像/状态缓存/大型checkpoint不复制，只保留已校验的外部路径引用；不要删除或移动首段目录。

如果前段未完整结束、起点/执行接口失败、发生重复trial、模型/代码/环境/计划身份变更，续跑会拒绝。它不是任意断点恢复器；失败的数据始终保留。不要修改report为PASS或只提取效果好的trial来继续。

## 产物和分析

每段根目录：`report.json / schedule.json / stage_schedule.json / diagnostics.json / trials.jsonl / trials.csv / cumulative_trials.json / evaluation_manifest.json / outcomes_seed_*_repeat_*.png`。`trials.jsonl`只记本段新执行；其余累计表包含以前完整块，避免重复计算。

每trial目录：`trajectory.npz / events.json / video.mp4 / control_trace.json / metrics.json / prefix_comparison.json`。完整轨迹包含真实RGB/heatmap、incoming action和因果状态；事件保存原生执行动作、位置/朝向、物品/健康、奖励/成功；trace保存每步剩余预算、动作分布、共同动作随机数和到两个目标的距离。

核心指标：

- `control_advantages.no_goal / base`：有目标相对独立无目标修正/底座的终点距离、相对起点改善和目标偏好优势。
- `target_switches`：两个目标各自的偏好及切换方向；`switch_change`还扣除真实起点的偏好，避免只把起点漂移当作控制。
- `reference_replay_variability`：参考终点距离、位置误差、原生动作差异及严格重放通过次数。参考未来结果只用于波动诊断，不据此删目标组。
- `per_seed_target`和图集/录像：检查两个目标分别是否改善，避免平均值隐藏一个目标退化。

物理/RGB/原生动作起点不可比或接口错误的整个重复块不纳入效果比较；原错误记录、轨迹和视频保存。`heatmap/RSSM`严格差异不会被删除，也不会当作相同隐藏状态的证据。一个世界的三次重复只报描述统计，不报显著性，不自动验收任务成功提升或批准T06。

工具不保存新模型；adopt只复制两段短参考。启动前检查剩余空间，evaluate为每个新trial预留16MiB，额外保留日志余量；这是轨迹/普通视频预算，不能保证任意增长的模拟器日志空间。原子JSON写入和报告磁盘错误处理复用已有工具。

## 中文 Git 提交备注

```text
feat: 增加T05残差worker六组真实重复评估与分段执行

- 复用真实参考，绑定已验收残差模型、冻结依赖和视觉环境协议
- 加入独立无目标修正、冻结底座及目标输入消融对照
- 预声明36次随机计划，支持12次首段和24次内容校验续跑
- 保存真实视频、原生动作、逐步目标距离及完整块描述统计
- 更新服务器命令和验收标准，真实控制待验收，T06未开始
```
