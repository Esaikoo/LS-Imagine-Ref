# T05 原16步内双目标视觉纠偏与保持采集

新增 `goal_within_horizon_feedback.py` 与 `scripts/t05_within_horizon_feedback.py`，只进行独立离线check和真实开发采集。复用原严格生产worker350推理加载器和真实历史接口，不修改40个旧验收Python文件、三个已通过的诊断文件或任何旧产物。代码仅在本地编辑，服务器通过Git同步；本次未连接或改动服务器。

来源必须是完整通过的 `continuation_action_diagnose_20261009T045733_081902`：重新核对其30局/480状态/4800分布、480原概率误差0、510测量、诊断代码和全部产物SHA，并沿来源核对原350确认/check、固定50步生产latest、独立verify、全部旧轨迹/缓存/目标。只使用生产350推理，不构造优化器、训练器、候选模型或新目标。旧数值失败、人工核对待完成及T06未批准保持。

## 固定采集设计

| 项目 | 设计 |
| --- | --- |
| 条件 | 两目标×`worker_mode` / `visual_correct_hold`×5重复，共20局 |
| 环境与初始化 | 原world1、原两真实RGB/heatmap目标、独立fresh reset、32noop+32真实固定前缀 |
| 预算 | 每局原16控制，remaining16–1连续递减；不追加第17–20步 |
| 自主段 | 前12步worker350/正确固定目标/mode，保持本局真实因果历史 |
| 末4步 | 原frame76–79→incoming77–80，remaining4–1；worker组继续mode，手工组按下述视觉规则 |
| 随机顺序 | 新种子5固定20个trial；动作记录种子来自design，mode仍记录原均匀数 |
| 整局划分 | repeat0–2为训练开发来源，3–4为开发留出；自主worker事实单列，不用于训练 |
| 完整计数 | env.step1600、worker查询320、worker实际280、手工实际40；动作改变另计，纠偏最多10次，更新0 |

手工规则只读当前真实RGB/heatmap的编码距离：若更近另一目标且本局尚未纠偏，目标0执行一次`turn_up`，目标1执行一次`turn_down`；其余执行`noop`。偏好差的绝对值≤1e-7视为持平并执行noop。即使纠偏失败，本局仍不再次转向。坐标/朝向只用于测量，不能输入动作规则。环境真实结束立即停止；不因视觉接近提前停止，不按frame76状态筛选，不重试/补样/续跑失败目录，不挑最佳中间帧。

该规则需要通过真实前后测量和最终视频验证效果；noop不能保证物理状态或画面不动，反馈规则不能当专家最优策略。

## 服务器运行

先在本地完成Git提交/推送，再在服务器按已有流程拉取同一提交。不要把本地工作目录直接覆盖服务器目录。建议一句话提交备注：

```text
feat: 增加T05原预算内双目标纠偏保持采集与结果汇总
```

在服务器项目根目录、现有`(ls)`环境运行以下两步。先check成功，再evaluate；两步目录使用同一时间戳便于定位，已存在的目录拒绝覆盖。

```bash
T05_FEEDBACK_DIAG="$PWD/relevance_map/t05_outputs/continuation_action_diagnose_20261009T045733_081902"
T05_FEEDBACK_STAMP="$(date +%Y%m%dT%H%M%S)"
T05_FEEDBACK_CHECK="$PWD/relevance_map/t05_outputs/within_horizon_feedback_check_$T05_FEEDBACK_STAMP"
T05_FEEDBACK_EVAL="$PWD/relevance_map/t05_outputs/within_horizon_feedback_evaluate_$T05_FEEDBACK_STAMP"

python scripts/t05_within_horizon_feedback.py check \
  --diagnosis-dir "$T05_FEEDBACK_DIAG" \
  --output-dir "$T05_FEEDBACK_CHECK" --device cuda:0
```

check检查真实因果增量/完整前缀、reset/incoming/结束、两目标分布、20个计划条目、提前干预/二次纠偏/种子/整局划分/缺测，以及本局终点/事实索引。负守卫只使用内存标记，不保存成真实轨迹；check环境步、执行和更新均为0。

```bash
MINEDOJO_HEADLESS=1 python scripts/t05_within_horizon_feedback.py evaluate \
  --diagnosis-dir "$T05_FEEDBACK_DIAG" --check-dir "$T05_FEEDBACK_CHECK" \
  --output-dir "$T05_FEEDBACK_EVAL" --device cuda:0
```

evaluate必须绑定同一check、worker350、诊断、代码、目标、规则及种子5的完整计划。返回的真实动作始终作为下一状态的incoming；同时保留worker建议、实际执行动作和干预来源，保存/复查原生事件、遥测、全历史、视频、概率和逐帧测量。第76帧边界与真实第80帧终点分别保留；真实提前结束保存其实际最后观测并单列，不伪造满16步。

## 产物与判断

`report.json`记录工程状态、冻结参数、计数和来源未变；`design.json`锁定计划；`evaluation_manifest.json`绑定所有trial真实文件及汇总SHA。每局保存完整轨迹、实际动作/worker建议、因果历史检查、原生事件/遥测、视频、逐帧指标及`actual_endpoint.npz`。

- `trials.csv`、`frame_metrics.csv`：全部局和帧、实际原预算终点、纠偏实际一步收益、窗口/整局自身进展、俯仰与位置诊断。
- `diagnostics.json`：两目标各两条件的全部尝试/错误/提前结束，五局均完整时才输出全部五次均值。frame76边界分层只供诊断，不作配对反事实或筛选。
- `coverage.json`：两目标×train/development_holdout分别计实际纠偏和保持的独立局数、事实行数、实际纠偏收益和后续偏好。一次纠偏之后可能还有保持，同一局两类覆盖可重叠。缺效果统计为不可用，覆盖计数为0不能充当失败率0%；重采样不增加独立证据，缺覆盖不补跑。
- `endpoint_data.json`：全部完整末4步事实索引，绑定76–79→incoming77–80/remaining4–1、本局真实第80帧RGB/heatmap目标及相关SHA。完整20局共80条，其中手工训练开发24、手工开发留出16、worker事实40。固定意向目标与本局实际终点分开，失败不标成意向目标成功，worker事实不训练；本轮不批准任何训练。
- `initial_balance.json`和10张`target{0,1}_outcomes_repeat_{0..4}.png`：自身起点、边界、真实终点全部保留，跨局状态无需相等，位置/朝向仅诊断。

工程PASS表示采集接口和保存有效。纠偏/保持的真实效果、两目标独立开发覆盖必须结合完整数据与视频判断，不能由脚本名称、手工条件或离线NLL代替自主控制证据。之后是否修复须另行设计和验收；T06仍需固定生产latest、保持检查和原两目标三组完整30局自主确认达到既定门槛并人工核对。

## 自动反馈与本地接收

每一步在最终报告写完后自动生成独立结果包，包含本轮、check（evaluate时）、已通过诊断、原350确认和repair verify的报告/表格/图片，保留失败状态。模型、缓存、完整轨迹和视频继续留在服务器，不复制到电脑。包中的来源原名及SHA完整保留，不需逐文件下载或重命名。

把最后的 `FEEDBACK_INDEX=.../bundle_index.json` 发到本对话即可复用T2-3090的只读下载方式保存到本机 `C:\Users\28620\Downloads\T05Results`。这里只下载结果，不执行服务器模型/环境、不上传或修改服务器代码。一次check包和一次evaluate包互不覆盖；导出失败会独立报错，原运行产物保留。

初次交付仅本地静态编辑、阅读接口与差异核对；没有运行本地Python、项目测试、模型、训练或环境，也没有连接/执行服务器、提交或推送。服务器check/evaluate待用户运行验收。

## 首次check故障与修复

用户服务器 `within_horizon_feedback_check_20261009T055139` 通过历史SHA及冻结模型来源核对，但在新 `validate_plan` 中误拒绝正常继承的计划，尚未进入接口check或启动环境。原因是原已验收MineDojo场景保存 `scenario.world_seed` 为字符串 `"1"`，新入口误用整数 `1` 比较，两个值在Python中不相等。原350确认和check的已下载结果包均明确保存字符串；原计划/场景没有被用户修改。

本地修复只改变两个新入口代码：保持原字符串 `"1"` 的严格身份校验，将世界种子错误单独提示；在服务器check中增加整数 `1`、其他世界字符串 `"2"`、无效空值的内存拒绝守卫。不转换原计划字段、不放宽世界身份、不改变种子5/20局/原16步/反馈规则/划分，不修改43个旧验收Python文件或旧报告。守卫仅在用户服务器运行check时执行，本地没有运行Python、项目测试、模型或环境。

中文Git提交备注：`fix: 修复T05预算内反馈计划的世界种子类型校验`

先在本地提交/推送，再由用户在服务器拉取。失败目录保留，不能复用或覆盖；在原服务器终端重新设置新目录并重跑check：

```bash
T05_FEEDBACK_DIAG="$PWD/relevance_map/t05_outputs/continuation_action_diagnose_20261009T045733_081902"
T05_FEEDBACK_STAMP="$(date +%Y%m%dT%H%M%S)"
T05_FEEDBACK_CHECK="$PWD/relevance_map/t05_outputs/within_horizon_feedback_check_seed_fixed_$T05_FEEDBACK_STAMP"
T05_FEEDBACK_EVAL="$PWD/relevance_map/t05_outputs/within_horizon_feedback_evaluate_seed_fixed_$T05_FEEDBACK_STAMP"

python scripts/t05_within_horizon_feedback.py check \
  --diagnosis-dir "$T05_FEEDBACK_DIAG" \
  --output-dir "$T05_FEEDBACK_CHECK" --device cuda:0
```

新check通过后，同一终端使用本页evaluate命令和新变量继续。原诊断不重跑，模型不重训；旧失败结果包的FAIL原样保留，自动汇总/只读接收方式不变，实际采集及T06仍未验收。

失败结果已按既有授权只读接收至 `C:\Users\28620\Downloads\T05Results\received_20261009T140139_6042862\T05_within_horizon_feedback_check_20261009T055139_results.json`，4阶段36文件及整包SHA通过（整包 `d9b3877e839d79670dda1c4e2efc1f9abb34fe18c57760a92a205ca745d9da71`）。报告确认状态failed、环境步/worker查询/动作/优化器更新均0、源文件签名前后相同；原350计划warmup32/前缀32/起点frame64/预算16和初始化动作逐项一致，仅新校验把字符串世界种子误作整数。此次SSH只读取结果文件，没有修改服务器代码/原产物或执行项目脚本。修复仅完成本地静态核对及Git差异检查，服务器新check仍待用户验收。
