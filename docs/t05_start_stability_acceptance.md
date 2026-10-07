# T05：起点离线审计、固定预热与两次重放预检

2026-10-07。本地已实现，服务器待验收。助手仅静态读取/编辑/核对差异，未运行Python、测试、模型或环境，未提交/推送。仍为T05，T06未开始。

## 为什么先做这一步

用户反馈 `residual_control_check / adopt_20261007T040055` 通过；同次pilot完整执行12个trial，但11次起点拒绝、只有交换目标组执行16步控制。400次env.step中384次为前缀，16次为控制，完整可比块0/12。附件显示11次都包含RGB超限，其中1次还包含位置超限，12次原生动作一致。第一条失败的第32帧MAE=0.51245、P99=8、物理差异0，但全历史最大误差拒绝了它；该帧RSSM相对差异0.355仍需保留。

这些结果不能确定渲染差异来源，也不能证明worker控制失败。旧结果仍为失败，旧24次不继续执行。此工具不会修改原报告、图像、模型输入或训练代码。

新增 `goal_start_stability.py / scripts/t05_start_stability.py audit/check/probe`：

- `audit`：CPU读取已保存的全部12条轨迹，与原参考逐帧比较，核对SHA256和原报告；无torch/模型、GPU、MineCLIP或MineDojo。
- `check`：严格加载已独立验收的残差第200步模型，核对原场景/前缀和审计身份，固定新预热计划，检查历史/窗口接口；无环境。
- `probe`：只启动两次fresh reset，固定32步noop，再重放原32步共同前缀，共最多128次新env.step；没有16步worker控制、策略训练或新的目标采集。

## 新预检协议

参数在新check里提前固定，probe必须相同：`warmup_steps=32`、`tail_frames=4`、两次重放、实验seed=0/world_seed=1、原固定起点和共同前缀。

- 数值门槛沿用原值：RGB MAE≤1、P99≤8、位置≤0.05、角度≤0.25、奖励差≤1e-6。
- 物理状态、物品/健康、边界、宏动作及原生动作比较覆盖**完整真实reset、预热和前缀历史**。
- RGB比较取**拟控制起点结束的固定最后4帧**，不再由reset后最早画面的最大像素误差单独否决预检。
- 原全历史RGB/heatmap/RSSM严格比较完整保存为诊断。近期窗口通过不意味着隐藏状态相同。
- 预热每步实际env.step及真实incoming onehot都保留；RSSM从真实reset持续更新，不丢弃预热，不在预热后伪造reset。每条历史再独立完整rollout重算并核对。

这是新的稳定性预检协议，**不是把旧失败trial重新判为成功**。audit的当前帧/近期窗口通过数量仅用于定位，不批准旧结果。32步noop可能减少初始画面波动，但效果未知；不根据结果自动加步数、缩窗口、放宽阈值或重试。

## 服务器顺序运行三步

拉取代码后，在原 `ls` 环境执行。audit/check最后PASS才执行下一步。以下变量独立定义，不依赖旧终端变量。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_OLD_PILOT="$PWD/relevance_map/t05_outputs/residual_control_pilot_20261007T040055"
T05_OLD_BENCHMARK="$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"

T05_STABILITY_TAG="$(date +%Y%m%dT%H%M%S)"
T05_START_AUDIT="$PWD/relevance_map/t05_outputs/start_audit_$T05_STABILITY_TAG"
T05_START_CHECK="$PWD/relevance_map/t05_outputs/start_check_$T05_STABILITY_TAG"
T05_START_PROBE="$PWD/relevance_map/t05_outputs/start_probe_$T05_STABILITY_TAG"

python scripts/t05_start_stability.py audit \
  --eval-dir "$T05_OLD_PILOT" --tail-frames 4 \
  --output-dir "$T05_START_AUDIT"
```

预期 `[PASS] T05_START_STABILITY_AUDIT`，`historical_identity / recorded_statistics / artifacts / no_training / source_inputs_unchanged`通过。`historical_status=failed`、`historical_eligible_trials=0`保持原值；失败计数应为RGB MAE/P99各11次，position 1次。当前/近期通过数量未知，以实际输出为准，不要求它们增加才能通过工程审计。

保存根目录 `summary.csv / diagnostics.json / physical_failures.json / audit_overview.png`。每个trial子目录保存 `frame_statistics.csv / diagnostics.json / prefix_pairs.png`，包括实际控制起点、误差峰值、近期窗口、颜色偏差和画面分区。位置失败明细标出具体帧、x/y/z差异，不笼统归因于渲染。

```bash
python scripts/t05_start_stability.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --benchmark-dir "$T05_OLD_BENCHMARK" --audit-dir "$T05_START_AUDIT" \
  --output-dir "$T05_START_CHECK" --device cuda:0 \
  --case-seed 0 --warmup-steps 32 --tail-frames 4
```

预期 `[PASS] T05_START_STABILITY_CHECK`，严格加载/benchmark身份和 `warmup_contracts / no_training / source_inputs_unchanged`通过。guard的PASS表示真实历史/计划边界拒绝正常：丢弃预热、伪reset、缺失incoming、修改计划或非法窗口均拒绝。内存marker不写成真实轨迹。输出 `probe_plan.json` 固定两次序列和128步预算；实际稳定性尚未验收。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_start_stability.py probe \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_RESIDUAL" \
  --residual-verify-dir "$T05_RESIDUAL_VERIFY" \
  --benchmark-dir "$T05_OLD_BENCHMARK" --audit-dir "$T05_START_AUDIT" \
  --check-dir "$T05_START_CHECK" --output-dir "$T05_START_PROBE" \
  --device cuda:0 --case-seed 0 --warmup-steps 32 --tail-frames 4
```

正常接口完成时应保存 `run_0 / run_1` 两条65帧真实历史及视频，`completed_runs=2`、`new_env_steps=128`、优化器更新0、`worker_control_actions=0`。`full_causal_history_run_0 / run_1` 应通过：核对自己的完整真实历史，不要求两次RSSM状态相同。

稳定性预检是否通过取决于服务器结果：

- `physical_history=PASS`、`recent_rgb=PASS`：末尾 `[PASS] T05_START_STABILITY_PROBE`。原全历史严格比较仍可能失败，scope的WARN正常；这是新预检通过，目标控制仍未验收。
- 任一为FAIL：末尾FAIL，保留两条真实历史、具体失败帧和图集。先分析它们；不自动补跑或变更参数，不启动12/36次。
- 提前终止、环境错误、因果历史重算不一致或磁盘问题：保存已取得的真实数据及错误，不能把不完整结果当作预检通过。

无论PASS还是FAIL，请反馈三步日志、audit根目录的 `summary.csv / diagnostics.json / physical_failures.json / audit_overview.png`，以及probe根目录的 `report.json / diagnostics.json / prefix_pairs.png`。若输出较多，先发两个根目录diagnostics与probe图集。

## 后续边界

原数值门槛不变，但时间窗口与预热序列发生改变，因此旧benchmark的真实目标/参考不能直接拿来继续跑。即使两次预检通过，也只能据此准备新的参考和重复计划；新真实评估需重新采集该预热协议下的目标，提前声明代码、参数和随机顺序。此交付不自动生成或执行新36次，不推进T06，不重训BC，不跑1M。

工具不保存新checkpoint、不复制大型缓存/模型或旧录像。probe检查轨迹/普通视频及额外磁盘余量，任意增长的模拟器日志仍需服务器留有空间。所有输出使用新目录，保留旧失败评估和原参考。

## 中文 Git 提交备注

```text
feat: 增加T05起点离线审计与固定预热双次重放预检

- 审计全部旧trial的当前、近期及全历史差异，保留原失败判定
- 导出颜色/分区差分图及具体位置偏离帧
- 固定32步noop预热和两次真实前缀重放，不执行worker控制
- 保留完整因果历史及原物理门槛，独立记录近期RGB与严格历史结果
- 更新服务器三步命令和验收标准，真实控制及T06仍待验证
```
