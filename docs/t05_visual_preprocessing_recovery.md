# T05 恢复训练时的视觉预处理

2026-10-07 本地交付。仅静态阅读、代码修改和差异核对；助手没有运行工具、测试、训练或 MineDojo，也没有提交/推送。

## 图集与代码定位

用户反馈 `diagnose_20261006T163845_292526` 的参考读取、原误差核对、图像导出和源文件检查全部通过。离线诊断工具验收通过；T05 环境配对与目标控制仍未通过。

图集两边的主要地形轮廓、相机视角和 HUD 位置一致，不能再把不同世界当作主要解释。第 0–4 帧仍有地面/颜色差异；第 8–9 帧的较大差异集中在右下手部以及边缘局部对象，第 32 帧也有类似动物对象的局部差异。不能仅凭图集确认每一处差异的物理或渲染原因。

静态核对确认一个输入错误：`goal_control.make_environment()` 为关闭截图而将 `screenshot_specs` 整个置为 `None`。但当前任务的 ScreenshotWrapper 在 `HUD=False` 时，除截图功能外，还使用 `HUD_mask.png` 和 `cv2.inpaint` 处理 HUD/手部区域；其默认 HUD 设置为 False。原 `harvest_log_in_plains` 任务保留该包装器，即使 reset/step 的截图开关都是 False，也仍进行视觉预处理。T05 跳过它后，原始血条、物品栏和手部进入 MineCLIP、目标编码器及 WM，和 T04 真实回放输入不一致。图集里的 HUD 直接支持这一定位。

这能解释一部分视觉差异和输入分布变化，不能证明它解释全部配对误差。RSSM 使用离散 posterior mode，相似画面可能产生不同类别和后续循环状态；0.7086 的状态距离不能直接解释为世界或任务进度相差 70%。

## 修复范围

- 保留任务 ScreenshotWrapper 的存在性、HUD 设置和其他选项，只关闭 `reset_flag / step_flag` 的文件输出。
- 实际环境启动前核对包装器和 HUD mask，终端记录是否启用去 HUD。没有新增裁剪、模糊、归一化或替换图像处理。
- 在 offline check 检查视觉配置保留；记录新的环境/预处理协议，拒绝旧 check 和旧 benchmark。
- 将 HUD mask 加入环境内容指纹，保存两段真实参考的 `case_0/visual_preprocessing.json`。
- 原 T04 模型、目标库和缓存不变，配对阈值不变。此次修复不宣称动态物体或初始化渲染已确定性复现。

## 服务器命令

拉取新代码后，先重新运行 T05 offline check，不重跑 T04：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_WORKER="$PWD/relevance_map/t04_outputs/train_early_fixed_20261006T155121/best_worker.pt"
T04_EARLY_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_20261006T155213"
T05_CHECK="$PWD/relevance_map/t05_outputs/check_visual_fixed_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_goal_control.py check \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --t04-verify-dir "$T04_EARLY_VERIFY" \
  --output-dir "$T05_CHECK" --device cuda:0 --prefix-steps 32
```

应新增 `[PASS] visual_preprocessing_contract`，最终 `[PASS] T05_OFFLINE_CHECK`。通过后再做一次相同预算的真实 prepare：

```bash
T05_BENCHMARK="$PWD/relevance_map/t05_outputs/prepare_visual_fixed_$(date +%Y%m%dT%H%M%S)"

MINEDOJO_HEADLESS=1 python scripts/t05_goal_control.py prepare \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --check-dir "$T05_CHECK" --output-dir "$T05_BENCHMARK" \
  --device cuda:0 --seeds 0 --prefix-steps 32 --horizon 16 \
  --reference-mode original_sample
```

三个环境应显示 `seed=0 world_seed=1`，并新增：

```text
[ENV] visual_preprocessing screenshot_wrapper=1 remove_hud=1 screenshot_file_output=0
```

正常完成时应看到 `[PASS] T05_BENCHMARK_PREPARE`，配对报告 `passed=true`。若仍失败，恢复预处理的工程核对通过也不能替代环境配对。不要直接 evaluate，反馈新的终端结果、`case_0/prefix_comparison.json / visual_preprocessing.json`，并导出新图集：

```bash
python scripts/t05_prefix_diagnose.py \
  --prepare-dir "$T05_BENCHMARK" --case-seed 0
```

修复后的 RGB 图不应再有原始血条与物品栏，HUD/手部区域应呈现与原任务相同的填补效果。局部对象或初始颜色差异可能仍然存在；给出结果后再决定下一步。

## 后续评估设计

当前严格配对方案要求整段观测历史、RSSM 状态几乎一致，才能解释为“只换目标的同起点比较”。这是该评估的前提，不是所有目标控制方法都必须满足的条件。如果恢复正确预处理后，剩余差异仍来自无法复现的动态对象/渲染，下一步应设计预先固定、随机分配条件、多个重复的统计对照：核对物理起点和预算可比，记录真实观测与历史，通过相同目标的重复波动和目标切换效果评估。不能将新设计的结果仍称为原方案的同状态配对，也不能简单放大旧阈值作为验收。

本次只实现已确认的视觉输入修复，尚未实现上述统计评估。修复验收后再根据剩余误差决定是否需要切换设计。

## 中文 Git 提交备注

```text
fix: 恢复T05与训练一致的HUD视觉预处理

- 保留ScreenshotWrapper和任务HUD设置，仅关闭截图文件输出
- 增加视觉预处理配置检查、实际包装器检查和HUD mask内容指纹
- 更新T05环境协议，记录参考预处理设置并拒绝旧验收产物
- 记录诊断图集验收结果和无需重训的服务器验证步骤
```
