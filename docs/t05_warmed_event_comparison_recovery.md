# T05：修复实时NumPy事件与保存JSON事件的比较

2026-10-07。用户服务器`warmed_control_check_20261007T060441`通过；同次prepare完成参考0的81帧，在参考1的65帧起点比较时报`The truth value of an array with more than one element is ambiguous`。这是程序异常，尚未得到这次起点是否通过的结果，不能据此判断目标控制失败。

助手已静态修复并核对差异，未执行Python、测试、模型或环境，未提交/推送。服务器修复验收待运行；仍为T05，T06未开始。

## 原因和修复范围

prepare的第一条参考从JSON读取，原生动作里的camera等字段已是Python列表；第二条参考来自实时Session，相应字段仍可能是NumPy数组。旧统计函数直接比较动作列表，嵌套数组的`==`得到布尔数组，Python无法将其当作单一真假值。

新入口在比较前复制两侧事件，并使用与`save_session`相同的值表示：数组变成列表、NumPy标量变成Python标量、嵌套结构递归处理。原值、顺序和缺失字段保留；不修改原事件、RGB/heatmap、策略输入、动作或门槛。

只修改`goal_warmed_control.py / scripts/t05_warmed_control.py`，不修改旧`prefix_diagnose / start_stability / control_stats`。因此已通过的旧probe和旧动作来源仍可读取，无需重跑旧预检。prepare/evaluate共用修复后的比较，真实trial同类错误一起修复。

新版check增加`live_saved_event_contract`：在内存中覆盖JSON对实时、实时对JSON、实时对实时、JSON对JSON；同值比较通过，camera/标量/动作长度/物品差异仍拒绝，缺失原生事件由guard拒绝。探针调用真实部署的compare入口，确认原事件保持NumPy类型且未改值；不写合成轨迹或训练标签。

## 为什么需要重新check和prepare

修复改变新入口的代码指纹，并新增了上次check没有覆盖的混合类型检查，所以需要运行新版离线check。原模型、残差verify、旧probe无需重新验收或训练。

旧prepare的参考0已完整保存，参考1在执行16步分支前异常退出；按日志应保留81帧和65帧，共144次env.step，具体以旧report和文件为准。finally会保存轨迹并关闭环境，第二个真实终点尚不存在。旧文件可用于诊断，但不能拿65帧的起点当作终点，不能虚构缺失16步，也不能直接运行evaluate。新prepare使用独立目录重新采集完整两条参考，旧失败记录保留。

## 服务器重跑命令

拉取修复后在原`ls`环境执行，先重新定义变量：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_RESIDUAL="$PWD/relevance_map/t05_outputs/residual_train_20261007T031803/latest.pt"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_20261007T031803"
T05_SOURCE_BENCHMARK="$PWD/relevance_map/t05_outputs/residual_control_adopt_20261007T040055"
T05_STABILITY_PROBE="$PWD/relevance_map/t05_outputs/start_probe_20261007T051726"

T05_WARM_TAG="event_fixed_$(date +%Y%m%dT%H%M%S)"
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

应新增`[PASS] live_saved_event_contract`及`[PASS] missing_native_event_guard`，末尾仍为`[PASS] T05_WARMED_CONTROL_CHECK`。新增环境步为0。

check通过后再运行：

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

程序不应再出现数组真假值异常。若起点实际不符合门槛，仍会FAIL并保存`case_0/start_comparison.json / case_0/start_diagnostics/diagnostics.json / prefix_pairs.png`，这与本次类型错误不同；先反馈这些文件，不补跑或放宽门槛。准备成功应为两条81帧、160步，末尾`[PASS] T05_WARMED_CONTROL_PREPARE`。

准备通过后，使用新变量按[原三步说明](t05_warmed_control_acceptance.md)的evaluate命令运行首块12次。旧`060441`目录不用于evaluate；不要删除或修改它。首块仍最多960步，不运行旧剩余24次。

## Git中文提交备注

```text
fix: 修复T05实时NumPy事件与JSON参考的比较异常

- 在新预热入口统一比较事件的值表示，保留原动作和观测
- 覆盖prepare与evaluate共用比较，不修改旧预检代码指纹
- 增加混合类型、真实差异拒绝及原事件不变的离线检查
- 补充新目录重跑与不完整参考处理说明
```
