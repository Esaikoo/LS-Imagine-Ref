# T05：从严格状态复现转到重复行为诊断

2026-10-07 本地实现；用户反馈 check/adopt 服务器工程验收已通过。助手只做静态阅读、修改和差异核对，没有执行测试、训练或环境，也未提交/推送。

当前执行状态：`repeated_check_20261006T174205 / repeated_adopt_20261006T174237` 已通过。32 个同状态查询的目标切换平均 L1=0.011754、mode 变化 1/32，目标控制尚未验收。**后续 30 trial 的 evaluate 暂缓，不需要现在继续；先按 [目标学习离线诊断说明](t05_goal_learning_diagnosis.md) 检查已有模型/数据。T06 未开始。** 已通过的 check/adopt 和真实参考保留，不重做。

## 当前定位

`prepare_visual_fixed_20261006T165045` 日志确认三个环境都保留了 ScreenshotWrapper/HUD 预处理。最大 RGB MAE 从前次 6.0387 降至 0.4034，当前只因 heatmap/RSSM 差异未通过。新图集第 0、1 帧显示 RSSM 相对 L2 接近 0，第 2 帧开始出现离散状态分歧，最大约 0.5553；所展示帧的 RGB P99 多为 1，最大为 2。

用户本次附的本地 `prefix_comparison.json` 仍记录旧的 6.0387/112/0.7086，与新日志/图集不对应。因此这里只记录新日志和图中可见数值，不使用旧 JSON 推断本次 heatmap 的具体误差。

HUD 输入问题已修复。严格配对仍未通过，不应宣称模型已经实现同隐藏状态控制。另一方面，几乎相同的 RGB 也可能使 posterior mode 改变离散类别，并随循环状态传播；严格要求 heatmap/RSSM 几乎逐位相同，将阻止当前真实环境的行为诊断。仅放宽两个阈值、继续称作同状态配对并不能解决这个实验设计问题。

本次保留原严格入口及阈值，新增明确不同的评估协议：固定可比的实测历史、随机执行顺序、每条件至少 3 次重复、参考动作重放波动，并增加同缓存状态的目标敏感度探针。

## 实现与边界

- `goal_control_stats.py`：起点可比检查、预先随机顺序、完整重复块与描述统计。
- `scripts/t05_repeated_control.py check`：沿用冻结模型/视觉预处理验收，并检查新协议边界、重复覆盖及整块排除。内存探针不写入真实数据。
- `adopt`：读取本次已有失败 prepare 的完整真实参考，核对当前模型/输入记录、视觉协议、原依赖未变、底层动作，离线重算因果 WM 状态和真实目标；将参考复制到新的独立小型产物目录。没有环境交互或模型更新。
- 两个参考起点、每个真实 trial 的完整前缀，继续使用原 RGB MAE/P99、位置、朝向、奖励、物品、生命、动作及结束标志边界；额外核对底层动作。heatmap/RSSM 保留原严格报告，在新协议中作为波动诊断，而非同状态通过条件。
- `same_state_goal_probe.json`：在两段参考的 32 个缓存因果状态上，用完全相同的状态及剩余步数分别输入两个目标和零目标，输出动作概率 L1 和 mode 动作变化率。这个探针确实比较同状态，但只能证明动作敏感度，不能证明成功到达。
- `evaluate`：同一世界、两个目标、goal/zero_goal/shuffled_goal/original/reference_replay 五条件，各 3 次重复，预先随机顺序共 30 trial。每个 trial 新建环境，从自己的真实观测逐步更新 RSSM，不替换图像或借用参考状态。
- 同一 repeat 内各条件共用预先固定的动作采样随机数。环境种子不靠重复编号更换，随机执行顺序独立固定并保存。
- 不重试失败，不根据终点表现选样。某个 trial 的起点/真实动作执行不符合记录时，该 seed/repeat 的整个块排除；不完整块也不纳入汇总。自然提前终止保留实际步数，不按失败结局删除。
- 参考动作重放保留实际终点距离、位姿及严格重放报告，用来估计波动，不因终点不完全一样而删除整个目标组。
- 控制阶段核对参考宏动作按记录执行，提前结束保留已执行前缀；其底层动作差异单独记录，避免按未来物品/状态导致的包装器变化选择结果。起点共同前缀的底层动作一致性仍是必须条件。
- 同世界重复不算新的独立世界种子；只输出均值、标准差、范围和随机块内对照差值，不报告 p 值、置信区间或自动验收控制有效。
- 当前没有独立 no_goal BC 对照，零目标只是推理消融，尚不能排除继续 BC 的影响。

旧失败 prepare 未保存完整环境文件指纹，adopt 会明确 WARN：原参考的视觉协议、模型、依赖记录和真实前缀可核对，但不能补造当时的完整环境指纹。当前环境与新评估代码指纹固定后用于所有新重复执行；原参考作为真实视觉目标及动作重放来源。后续严格 prepare 已增加该指纹记录。

## 先做两个离线步骤

建议先运行 check/adopt，反馈目标敏感度再决定是否执行 30 个真实 trial。它们需要读取服务器现有模型，但不启动 MineDojo、不训练，也不重建缓存。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_WORKER="$PWD/relevance_map/t04_outputs/train_early_fixed_20261006T155121/best_worker.pt"
T04_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_SOURCE="$PWD/relevance_map/t05_outputs/prepare_visual_fixed_20261006T165045"
T05_REPEAT_CHECK="$PWD/relevance_map/t05_outputs/repeated_check_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_repeated_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_VERIFY" --output-dir "$T05_REPEAT_CHECK" \
  --device cuda:0 --prefix-steps 32
```

应看到 `repeated_protocol / probe_scope` 的 PASS，最终 `[PASS] T05_REPEATED_CHECK`。通过后执行：

```bash
T05_REPEAT_BENCHMARK="$PWD/relevance_map/t05_outputs/repeated_adopt_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_repeated_control.py adopt \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_REPEAT_CHECK" --source-prepare-dir "$T05_SOURCE" \
  --output-dir "$T05_REPEAT_BENCHMARK" --device cuda:0 \
  --repetitions 3 --randomization-seed 0
```

预期输出：`real_state_reencode_0` PASS、`[GOAL PROBE] ... mean_l1=... mode_change_fraction=...`、`[ADOPT] ... start_comparable=1 strict_pair=0 ...`，最终 `[PASS] T05_REPEATED_ADOPT`。`strict_pair=0` 被原样保留；adopt 通过表示新设计的输入/工程验收，没有批准旧 prepare。

反馈新的终端日志和 `case_0/same_state_goal_probe.json`。若两目标在这些状态上几乎不改变动作分布，优先分析 T04 的目标学习，暂缓大量真实 trial；若有可检查的动作差异，再用真实重复试验核对是否带来对应目标进展。

如果 adopt 报“目标过近”或“起点已接近目标”，反馈新目录的 `case_0/goal_diagnostics.json`；不能通过修改目标距离阈值或挑选另一组结果来直接验收。若物理/RGB/底层动作检查失败，反馈错误和原始新配对报告，不能套用重复协议跳过这些差异。

## 后续真实重复执行命令

check/adopt 通过且需要进一步检查行为时，在同一终端会话使用上述变量：

```bash
T05_REPEAT_EVAL="$PWD/relevance_map/t05_outputs/repeated_evaluate_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_repeated_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_REPEAT_CHECK" --benchmark-dir "$T05_REPEAT_BENCHMARK" \
  --output-dir "$T05_REPEAT_EVAL" --device cuda:0 --execution-policy mode
```

单世界、32 步前缀、16 步控制、30 trial，最多新增 1440 个真实 `env.step`，可能因结束或前缀拒绝少于此数。每个 trial 都需要新建模拟器，墙钟时间不能只按环境步估计。这些试验不进行优化器更新；不运行 1M。

输出 `schedule.json / trials.jsonl / trials.csv / diagnostics.json`、每个 trial 的真实视频、状态、动作和原严格配对报告，以及 `outcomes_seed_0_repeat_0.png` 等三个终点图集。反馈 `diagnostics.json / trials.csv` 和终点图集，必要时检查视频。

最终 `[PASS] T05_REPEATED_EVAL` 仅说明全部随机重复执行和完整可比块检查完成。重点查看目标条件的对照优势是否稳定、参考动作重放自身波动有多大、实际行为是否对应目标。控制收益和 T06 仍需结果分析，不由代码自动批准。

## 中文 Git 提交备注

```text
feat: 增加T05可比起点的随机重复行为诊断

- 保留严格配对报告，新增带完整重复块的独立评估协议
- 离线复用已有真实参考，核对因果状态并检查同状态目标敏感度
- 预先固定随机执行顺序，保留参考动作重放波动和底层动作检查
- 输出描述统计和真实终点图集，不宣称隐藏状态相同或控制有效
- 更新HUD修复验收记录和无需重训的服务器运行步骤
```
