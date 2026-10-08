# T05自主轨迹最后两步的离线动作诊断

2026-10-08。新增`goal_terminal_action_diagnose.py`和`scripts/t05_terminal_action_diagnose.py`。本地仅静态阅读、编辑和差异核对，未运行Python、测试、模型或环境；服务器待验收。旧推理代码、worker300、原轨迹、固定16步及数值门槛不改。

## 本次要定位什么

原30次确认工程有效，但目标1只通过2/5方向，平均进展低于独立no_goal，T06未批准。其失败repeat1、2、4在第15步d1约0.037，第16步d1约0.493，pitch误差0°→10°，终点回到目标0姿态。详见 [自主确认结果分析](t05_repaired_control_result_analysis.md)。这说明末步是明确的定位边界，尚不说明网络内部原因。

新入口直接读取完整`repaired_control_evaluate_20261007T180059`，从原报告推导checkpoint、cache、repair verify和check路径，无需手动换模型参数。接受完整工程通过而行为门槛失败的原评估；不要求`gate.passed=true`，也不筛掉失败局。

1. 核对原manifest、check、repair verify、worker300与旧代码身份；全部30局真实产物逐项校验原SHA256，重新核对固定数值门槛。模型/轨迹不下载，留在服务器。
2. 使用各局保存的完整5120维因果状态，核对实际incoming、原生事件、遥测、mode和remaining。原480个动作概率与动作全部重现；原510个视觉/物理测量重编码核对。原RSSM状态由SHA256和已通过的history_check固定，不新建或重算RSSM。
3. 对全部30局的frame78/79分别用真实remaining2/1查询目标0、目标1、独立no_goal、zero_goal和冻结底座，共60个自主动作前状态。概率变化、mode变化、top2差距、实际动作概率/排名及事实动作前后变化分开保存。只查询实际预算，不把remaining改成2来覆盖真实的1。
4. 读取修复依赖中的四条完整参考历史及其真实脚本末尾动作，使用同一worker300/固定目标做8次同预算查询。自主状态与两目标各两条参考状态的余弦距离只用于定位；没有分布外阈值，参考动作也不是新自主状态的正确标签。repeat0参考已用于修复训练，repeat1已被检查，均属于开发资料。

真实自主动作可以失败，不能把其高概率或低排名当控制质量。改变目标后的动作没有执行；实际后续收益只属于原事实动作，不能赋给未执行的反事实动作。参考脚本也不是唯一最优动作。不新增环境步、优化器或训练标签，不选中间最佳帧，不批准T06。

## 服务器命令：一次离线运行

提交/推送后在服务器拉取更新，从项目根目录运行。原训练、30次确认和标定均不用重跑。本入口不启动MineDojo/MineCLIP，无需`MINEDOJO_HEADLESS=1`；使用原CUDA环境，以严格重现保存概率。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T05_TERMINAL_DIAG="$PWD/relevance_map/t05_outputs/terminal_action_diagnose_$(date +%Y%m%dT%H%M%S)"

python scripts/t05_terminal_action_diagnose.py \
  --eval-dir "$PWD/relevance_map/t05_outputs/repaired_control_evaluate_20261007T180059" \
  --output-dir "$T05_TERMINAL_DIAG" --device cuda:0
```

须保留原评估目录内的`evaluation_manifest.json`和30个trial目录，以及原checkpoint/共享依赖/cache/check/verify/四条标定历史。现有小附件用于分析，不能代替服务器真实状态。输出目录须新建，拒绝覆盖或写入源目录。入口先做输入身份/原产物哈希核对，随后输出`OUTPUT_DIR`；这些检查只读取文件。

## 预计产物与验收

| 文件 | 内容 |
| --- | --- |
| `report.json` | 身份、守卫、冻结、原概率/测量重现及输入不变检查；新环境步和更新均为0。 |
| `diagnostics.json` | 60次末尾查询汇总、目标1五次的末步完整对照、原数值门槛原样保留；不产生新到达判定。 |
| `trials.csv` | 全30局的最后实际动作、正确/交换目标mode、top2差距、末步实际进展及姿态变化。 |
| `frame_metrics.csv` | 60个末尾状态的动作支持、目标差异、实际前后测量和同预算参考状态距离。 |
| `action_queries.json` | 原状态SHA256、完整五种概率/偏好、实际前后位姿；不写RSSM或训练样本副本。 |
| `reference_context.json` | 四参考历史的8个末尾查询、真实脚本动作及当前worker支持，不移植标签。 |
| `target1_last_action_probabilities.png` | 正确目标1五个repeat的最后一步概率图，成功和失败全保留。 |

预期最终`[PASS] T05_TERMINAL_ACTION_DIAGNOSE`，并有`historical_identity`、`real_query_alignment`、`recorded_forward_roundtrip`、`recorded_measurement_roundtrip`、`reference_context_comparison`、`no_training`、`source_inputs_unchanged`的PASS。`scope`的WARN保留T05边界；guard的错误文字表示主动拒绝错误输入。概率须在原容差内且mode动作逐个重现，不能为运行通过放宽原容差。若接口/身份/重现失败，保留报告反馈，暂不解释概率或训练。

回传`report.json`、`diagnostics.json`、`trials.csv`、`frame_metrics.csv`、`reference_context.json`及概率图。完整`action_queries.json`已保存，初次无需上传；若失败另回传终端错误。无需下载模型、缓存、轨迹或视频。

分析时优先回答：失败三局最后实际选了什么，换目标是否改变mode，目标1下哪个动作领先以及差距多大；成功局是否保持目标1姿态；在同预算参考状态上是否也选同一末步动作。状态相似度本身不能证明泛化失败或为自主状态提供动作标签。

中文Git提交备注：`feat(T05): 增加自主轨迹末两步动作与同预算参考状态离线诊断`
