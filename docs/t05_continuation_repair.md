# T05真实续段的小预算混合修复

2026-10-08。本地实现完成，服务器check/train/verify待用户验收。新增独立的`goal_continuation_repair.py`和`scripts/t05_continuation_repair.py`，不修改已验收Python、环境wrapper、源worker300或旧失败结果。助手只静态阅读、编辑与差异核对，未运行Python、测试、模型、训练或环境，未提交/推送。

服务器首次`continuation_repair_check_20261008T115256`在读取旧标定报告时失败：本入口误把标定的`probe`命令按`evaluate`校验。已只修正新入口的命令名，仍要求报告状态通过；历史SHA256与后续真实参考校验保留。失败发生在输入加载阶段，尚未开始训练。保留该失败目录，同步修正后的`scripts/t05_continuation_repair.py`，重新设置下面的时间戳和三个输出目录后从check开始；服务器修复验收仍待反馈。

来源为完整通过的`closed_loop_continuation_evaluate_20261008T093057`：十条真实16+4步历史，手工五局中只有一条实际下转纠偏、四条保持，worker事实全部保留。见 [来源分析](t05_closed_loop_continuation_result_analysis.md)。本轮不重跑采集，不执行控制，不批准T06。

## 固定训练与留出

分别继承同一worker300的goal/no_goal权重，两个新的Adam从步0开始；不恢复上一修复优化器。两分支各追加固定50步，学习率`5e-5`，batch256、原归一化/raw偏好相加/一次onehot unimix不变，底座/WM/目标编码器继续冻结。没有best选择或看结果延长预算，主产物固定`latest.pt`，新版本为350。

| 每批来源 | 条目数 | 50步每分支标签数 | 监督和划分 |
| --- | ---: | ---: | --- |
| 原真实训练expanded rows | 192 | 9600 | 原本局真实终点/动作/remaining、原整局split |
| 原两目标参考脚本训练 | 48，每目标24 | 2400，每目标1200 | repeat0训练，repeat1留出；原本局第80帧终点 |
| 手工真实续段训练 | 16 | 800 | 三条repeat0–2完整局、12条动作，均匀有放回 |
| 总计 | 256 | 12800 | 双分支同批、同预算 |

手工训练池为11个noop和1个turn_down，12条真实动作全部均匀抽取，没有动作加权、纠偏过采样或按成功筛选。记录实际每条抽样次数及真正见到的唯一行数；重复抽一条纠偏不是新增独立证据。手工repeat3–4的两条完整局、8个noop只作开发留出，不拆局，不转入训练。原参考的两目标支持和原26局完整留出继续保留。

每条续段使用本局保存的因果feature[80–83]、真实incoming action[81–84]、remaining4–1，以及本局**实际第84帧**RGB/heatmap的冻结编码。保留原frame80和原前16步失败事实，不把下转移植成frame79/remaining1标签。各局实际终点均重编码核对，不把它们统一替换成旧固定目标1；坐标/朝向不进训练输入。宏动作noop也可能受原sticky jump影响，不能把训练解释为已经物理停止，wrapper和环境身份不改。

另外五条worker事实续段的20个动作全部保留并单独评价，含失败实际终点；本轮不训练这些worker事实，也不指定在旧目标1下模仿失败动作。两组历史不配对，不生成专家或反事实标签。

## check、报告与退步标准

入口由续段report追溯原10局干预、末步诊断、30局确认、四条参考、源worker300/check/独立verify、T04缓存和T03真实来源。完整核对旧代码和真实产物SHA256，包括视频；读取已验收的5120维因果状态，不构造RSSM/候选模型或MineDojo/MineCLIP。重现原worker300的200个查询概率与mode；目标重编码只使用冻结视觉目标库。不会把内存守卫写成真实轨迹。

`check`验证严格warm start、新优化器步0、三池抽样和RNG恢复、真实动作/remaining/手工来源/整局划分/本局终点守卫，并保存小型`supervision.npz`、`coverage.json`和未更新的`initial_metrics.json`。原参考的同预算32近邻支持仍为描述性覆盖，不能用于训练选择。单条训练纠偏、开发留出无纠偏会明确WARN；无覆盖组用`available=false, rows=0`，不会输出伪造的零NLL。

`train`在步0/10/20/30/40/50评价，10步保存一次完整latest。原26局留出、原两目标参考、手工保持/纠偏、worker事实分别报告，不能合并成单个总平均。续段每个状态保存真实动作NLL、概率、排名、mode匹配，以及goal/no_goal/zero/base/交换目标分布；本局真实终点与旧固定目标1是两套评价。交换输入是旧固定目标0，仅检查预测敏感性，不代表原动作在该目标下也正确。

训练前固定退步检查：相对check步0，原26局完整留出的goal/no_goal NLL，以及目标0参考留出在本局终点/旧固定目标输入下的goal/no_goal NLL，六项各自绝对增加不得超过`0.02`。这是本次离线开发修复的保留标准，不是目标可达或T06阈值。逐项存`retention.json`，超限仍保留固定50步结果并WARN，不选best、不追加更新，不自动开始自主确认。

保持的开发留出仅两个完整局/8个noop，不能当8个独立导航试验。唯一纠偏在训练侧，报告其动作支持变化，但任何离线改善都不证明纠偏泛化。原目标1的30局失败、原26局已有退步和全视频核对缺口不改写。

## 新快照与verify

独立新格式保存两个修正头、新优化器、三池计数/每条续段抽样次数、采样与训练RNG、源300和追加50预算；共享依赖用相对路径和SHA256引用，不复制底座、缓存或轨迹。旧残差加载器与旧参考修复加载器明确拒绝新格式，后续控制要专门接入新加载器，不能直接替换旧命令的checkpoint。

`verify`只接受本次固定50步latest：检查来源/格式/身份/形状/计数/训练池污染拒绝、保存恢复往返、独立无优化器推理分布与训练分布一致。两份内存副本额外各做一次真实混合更新以核对恢复等价，不回写latest，不计入生产训练追加50步；`roundtrip_verification.pt`带验收标志，禁止resume和控制。新环境步始终0。

新check同时锁定后续自主确认计划：worker350/mode、种子4、两目标×goal/no_goal/swapped_goal×5重复，共30局，原32noop+32前缀+16步，最多2400环境步，沿用原行为门槛和两固定目标。实际确认入口及新checkpoint/verify绑定待本轮结果分析后实现；`confirmation_plan.json`是预声明计划，不是已经执行的确认。

## 服务器命令

同步两个新增Python文件及文档，在服务器项目根目录依次执行；上一条最终PASS后再执行下一条。无需重新采集或运行旧修复，也无需`MINEDOJO_HEADLESS=1`，这三个入口都不开环境。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_MIX_TAG="$(date +%Y%m%dT%H%M%S)"
T05_MIX_SOURCE="$PWD/relevance_map/t05_outputs/closed_loop_continuation_evaluate_20261008T093057"
T05_MIX_CHECK="$PWD/relevance_map/t05_outputs/continuation_repair_check_$T05_MIX_TAG"
T05_MIX_TRAIN="$PWD/relevance_map/t05_outputs/continuation_repair_train_$T05_MIX_TAG"
T05_MIX_VERIFY="$PWD/relevance_map/t05_outputs/continuation_repair_verify_$T05_MIX_TAG"

python scripts/t05_continuation_repair.py check \
  --continuation-dir "$T05_MIX_SOURCE" \
  --output-dir "$T05_MIX_CHECK" --device cuda:0

python scripts/t05_continuation_repair.py train \
  --check-dir "$T05_MIX_CHECK" \
  --output-dir "$T05_MIX_TRAIN" --device cuda:0

python scripts/t05_continuation_repair.py verify \
  --check-dir "$T05_MIX_CHECK" --checkpoint "$T05_MIX_TRAIN/latest.pt" \
  --output-dir "$T05_MIX_VERIFY" --device cuda:0
```

输出目录必须是新的独立目录，拒绝覆盖或与输入目录重叠。遇到FAIL保留目录和完整日志；不重跑采集、修改旧代码指纹或延长预算。若训练中断且已有未完成的本实验快照，可在另一个新目录用同一check的`train --resume <该快照>`恢复，累计仍50步；步50latest不能再次resume。不要下载checkpoint、缓存、模型、trajectory.npz或视频。

## 反馈与后续验收

请反馈三步完整日志，尤其`[CONT REPAIR] step=0/50`。下载时给同名文件加check/train/verify前缀，便于区分：

- check：`report.json`、`coverage.json`、`initial_metrics.json`、`retention_plan.json`、`confirmation_plan.json`。
- train：`report.json`、`initial_metrics.json`、`metrics.json`、`diagnostics.json`、`sampling.json`、`retention.json`、`evaluation.jsonl`、`frame_metrics.csv`。
- verify：`report.json`、`metrics.json`、`retention.json`。

工程验收要求来源/实际动作/真实终点/整局split、冻结依赖、两分支同批训练、固定预算、严格恢复和下一更新等价全部通过。预期train两分支各50更新、累计worker350、每分支12800标签，原/参考/续段9600/2400/800，worker实际执行与新环境步0。verify额外更新只存在于验收副本。

分析重点是手工保持留出是否改善、目标消融是否仍提供额外作用、唯一训练纠偏支持是否提升、原26局与目标0是否超过预声明退步限度。若修复值得继续，再按已固定新计划实现自主确认，不把离线PASS或手工5/5当作T06准入。两个目标均须达到原4/5正进展且偏好正确、五次均值为正并胜两对照，且完整视频/位姿核对通过后，才考虑限定场景T06。

中文Git提交备注：`fix: 修复T05续段修复入口的标定probe报告校验`
