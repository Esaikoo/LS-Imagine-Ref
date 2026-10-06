# T05 相同位置与动作，却出现不同画面的修复

2026-10-07 本地实现，服务器待验收。助手只做静态阅读与差异核对，未运行测试、训练或环境；不需要重新训练 T04，也不需要重建缓存。

后续服务器结果：`check_seed_fixed_20261006T161453` 通过，但 `prepare_seed_fixed_20261006T161526` 使用 `world_seed=1` 后仍在 reset 第 0 帧发生画面差异，环境配对未通过。当前先按 [离线画面诊断说明](t05_render_diagnosis.md) 导出已有失败轨迹的对应图像，再决定下一项修复；下文保留种子修复的背景及首次重跑命令，不要求再次重跑 prepare。

## 诊断与已确认的问题

用户反馈：`train_early_fixed_20261006T155121` 已完成 100→400 次更新；`verify_early_fixed_20261006T155213` 的共享冻结依赖、完整恢复、下一次更新一致性与冻结约束全部通过。T04 本次修复验收通过，最佳 worker 为 400 次。

`prepare_20261006T155535/case_0/prefix_comparison.json` 中位置与朝向差为 0，动作、物品、生命、标志一致，奖励差为 0；但 RGB MAE=14.3503、P99=67、heatmap MAE=0.9456、RSSM 相对 L2=0.8067。这说明失败不是动作编号或当前位置不同；同一坐标也可能位于不同的世界中。该数据不能用于配对控制结论。

静态核对发现，T05 把 case `seed=0` 直接传为 `world_seed="0"`。MineDojo 的 [BiomeGeneratorImplementation.java](https://github.com/MineDojo/MineDojo/blob/main/minedojo/sim/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/MissionHandlers/BiomeGeneratorImplementation.java#L95) 先生成随机世界种子，只有解析出的整数非零时才用指定值替换；[DefaultWorldGeneratorImplementation.java](https://github.com/MineDojo/MineDojo/blob/main/minedojo/sim/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/MissionHandlers/DefaultWorldGeneratorImplementation.java#L47) 同样处理。Python `seed=0` 和 `world_seed="0"` 因而不是等价的确定性设置。

这是已确认的复现实现缺陷，符合本次“同坐标、同动作、不同画面”的表现；是否仍有其他渲染或环境随机性，须由服务器重跑确认。初始光照在诊断过程中只是候选原因，本次不修改光照/任务条件，也不放宽配对阈值。

## 修改

- Python/NumPy/Torch/任务内部的实验种子仍为 `seed=s`；实际世界种子统一为字符串 `str(s+1)`。例如 case 0→世界 1，case 1→世界 2。这是预先固定、无冲突的编号映射，不是失败后反复选择“容易通过”的种子。
- discovery、两个参考分支及全部 evaluate 条件共用持久化的 `world_seed / world_seed_policy`。`case_0/scenario.json` 和 `benchmark.json` 保存实际设置；终端同时显示两个种子。
- 拒绝零世界种子、缺少映射的旧 scenario、旧 benchmark 协议；真实环境指纹加入 T05 控制实现，避免修改控制协议后复用旧参考数据。
- 配对检查继续使用原阈值，同时保存 `failed_checks / first_mismatch_frame / per_frame`，区分 reset 画面就不同还是执行之后才分歧。
- offline check 增加种子映射/拒绝旧协议检查，以及图像/热力图/状态/动作/遥测变化的首次分歧帧检查；内存探针不写入训练或真实目标。

## 服务器重跑命令

拉取新代码后，在原 `ls` 环境运行。固定使用已经验收的 400 次 worker；以下命令不依赖之前终端会话留下的变量，也不会继续做 BC 更新。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_WORKER="$PWD/relevance_map/t04_outputs/train_early_fixed_20261006T155121/best_worker.pt"
T04_EARLY_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_CHECK="$PWD/relevance_map/t05_outputs/check_seed_fixed_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_EARLY_VERIFY" \
  --output-dir "$T05_CHECK" --device cuda:0 --prefix-steps 32
```

看到 `zero_world_seed_guard / legacy_scenario_guard / world_seed_contract / execution_contract` 和最终 `[PASS] T05_OFFLINE_CHECK` 后，再运行真实目标构建：

```bash
T05_BENCHMARK="$PWD/relevance_map/t05_outputs/prepare_seed_fixed_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py prepare \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --output-dir "$T05_BENCHMARK" \
  --device cuda:0 --seeds 0 --prefix-steps 32 --horizon 16 \
  --reference-mode original_sample
```

三个环境日志应全部显示 `seed=0 world_seed=1`。正常应出现 `[CASE] seed=0 world_seed=1 matched_prefix=1`，最终 `[PASS] T05_BENCHMARK_PREPARE`。`scenario.json` 的 `world_seed` 应为 `"1"`、策略为 `"case_seed_plus_one_v1"`；`prefix_comparison.json` 应为 `passed=true / failed_checks=[] / first_mismatch_frame=null`。

如果仍不匹配，保留新目录并反馈 `[PAIR MISMATCH]`、`prefix_comparison.json` 与 `case_0/scenario.json`。不要反复重跑来挑选匹配结果，不要提高阈值。首次不一致帧为 0 时优先检查 reset 世界/渲染；之后才出现时再检查执行、动态世界或视觉数值。这次修复只确认消除了零世界种子缺陷，不保证所有动态场景逐帧完全复现。

如果配对通过后报“两目标过近”或参考提前结束，则是下一项诊断，不等同于世界种子修复失败；反馈对应 `goal_diagnostics.json` 或错误日志，先分析再决定目标构建方式。

## Prepare 通过后才运行行为评估

```bash
T05_EVAL="$PWD/relevance_map/t05_outputs/evaluate_seed_fixed_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py evaluate \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --benchmark-dir "$T05_BENCHMARK" \
  --output-dir "$T05_EVAL" --device cuda:0 \
  --execution-policy mode --action-seed 0
```

按 [T05 验收说明](t05_goal_control_acceptance.md) 查看配对及参考动作重放、视频与对照指标。这次 prepare 通过仅证明参考目标构建成功，仍不能代替目标控制效果验收。

## 中文 Git 提交备注

```text
fix: 修复T05零世界种子导致参考环境不一致

- 分离实验种子与非零世界种子，统一持久化并复用seed+1映射
- 校验新环境协议和种子记录，拒绝旧随机零种子参考数据
- 输出逐帧配对差异及首次分歧位置，补充离线种子与诊断检查
- 记录T04早期模型验收结果，提供无需重训练的T05重跑命令
```

以上仅为提交备注文本，助手未执行 Git 提交或推送。
