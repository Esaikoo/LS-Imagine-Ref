# T05 重置画面不一致：离线定位

2026-10-07 本地交付。助手只做静态阅读、代码修改和差异核对，未运行工具、测试、训练或 MineDojo。新工具的运行验收由用户在服务器完成。

## 当前结论

用户反馈 `check_seed_fixed_20261006T161453` 最终为 `[PASS] T05_OFFLINE_CHECK`，但 `prepare_seed_fixed_20261006T161526` 仍在配对参考历史时失败。三个环境均记录 `seed=0 world_seed=1`，零世界种子修复已生效，但不能据此认定环境已完全复现。

新 `prefix_comparison.json` 的 33 帧中，位置、朝向和奖励差为 0，动作、物品、生命与结束标志一致。第一处差异在 reset 的第 0 帧：RGB MAE=6.0387，P99=26，RSSM 相对 L2=0.3230；整段最大 P99=112、RSSM 相对 L2=0.7086。RGB MAE 在前几帧降到约 1.70，之后又出现较大的局部像素差异。

这些数字证明观测历史不同，不能确定具体来源。初始化渲染、光照、动态对象等仍只是待检查的候选原因。RSSM 的离散 posterior mode 也可能放大小幅输入差异；较大的状态差不能单独证明编码器数值出错。先查看两段已有轨迹的对应画面，再决定是否修改环境设置。

T04 的 400 次最佳 worker 及其恢复验收仍有效，不需要重训，也不需要重建 T04 缓存。当前不继续 evaluate 或 T06，不放宽配对阈值，不通过反复重跑挑选恰好匹配的结果。

## 服务器命令

拉取新代码后，在现有 `ls` 环境运行：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

python scripts/t05_prefix_diagnose.py \
  --prepare-dir "$PWD/relevance_map/t05_outputs/prepare_seed_fixed_20261006T161526" \
  --case-seed 0
```

这是纯离线工具，无需 `MINEDOJO_HEADLESS=1`，无需指定 checkpoint 或 GPU。它只读取失败目录中两段 `reference_0 / reference_1` 的 `trajectory.npz / events.json`、原配对报告和场景设置，写入新的 `relevance_map/t05_outputs/diagnose_*` 目录；不会改写原参考数据。

## 预期输出与反馈

终端应出现 `saved_references / recorded_statistics / artifacts / source_inputs_unchanged` 的 PASS，以及最终 `[PASS] T05_PREFIX_DIAGNOSE`。`scope` 的 WARN 是说明用途，不代表工具失败。

输出文件：

- `prefix_pairs.png`：第 0/1/2/4/7/8 帧、最大 MAE/P99 帧及前缀最后一帧的图集；每行依次为两张原始 RGB、绝对差异放大 4 倍、两张原始 heatmap。原图只按最近邻放大显示，不改变训练或评估输入。
- `prefix_diagnostics.json`：原配对结果、实际场景、逐帧分区/颜色/差异面积统计、动作及遥测差异。
- `frame_statistics.csv`：便于逐帧检查的同一组统计。
- `frames/frame_000.png` 等：全部共同前缀帧的并排图。
- `report.json`：导出范围及源文件大小/修改时间检查。

请反馈 `prefix_pairs.png` 和 `prefix_diagnostics.json`。若工具失败，反馈终端错误和新目录中的 `error.txt`。大型 checkpoint、缓存和轨迹继续留在服务器。

查看图集时，先判断地形与物体是否一致，再看差异集中在全图颜色、天空、局部移动对象还是手部。统计中的 `channel_bias_removed_mae` 只用于判断差异是否近似整体颜色偏移，不会修正图像，也不用于配对验收。`native_actions_equal` 用来补充核对底层实际动作；宏动作相同不自动保证其底层执行列表相同。

诊断工具通过只表示现有数据读取和图像导出成功。报告明确保存 `approved_benchmark=false / new_env_steps=0 / optimizer_updates=0 / model_loaded=false`，T05 prepare 与目标控制仍待验收。

## 中文 Git 提交备注

```text
feat: 增加T05失败参考轨迹的离线画面诊断

- 从已保存参考轨迹导出并排RGB、差异图和完整前缀帧图集
- 输出分区、颜色偏移、底层动作和遥测统计，核对原逐帧误差
- 只读取已有产物，不加载模型或启动环境，不改变配对验收条件
- 记录非零世界种子修复后仍在reset帧分歧的结果和反馈步骤
```

以上为提交备注文本，助手未执行 Git 提交或推送。
