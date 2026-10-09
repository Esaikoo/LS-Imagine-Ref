# T05 原16步内事实监督检查

新增 `goal_within_horizon_supervision.py / scripts/t05_within_horizon_supervision.py` 的独立check。绑定已通过的 `within_horizon_feedback_evaluate_seed_fixed_20261009T084226`，读取全部20条真实历史，检查本局实际第80帧目标与原固定目标下的事实动作拟合。只有check入口，没有train/evaluate或策略执行入口。

2026-10-09服务器验收已完成：`within_horizon_supervision_check_20261009T112719_064845`完整PASS/WARN，20局/80事实查询/480分布，320原概率重现误差0、340测量和实际终点重编码通过，更新/环境步/动作执行0。6阶段68文件已只读接收到电脑，整包与各文件SHA、3440标量及全部分组独立复算一致。40个手工事实动作在六条件下mode均不匹配；目标0保持尤其弱，目标1训练纠偏仍缺失。见 [完整结果与有限修复方案](t05_within_horizon_supervision_result_analysis.md)。无需重复本check或原20局采集；用户后续授权的独立固定修复check/train/verify已本地实现，见 [新运行说明](t05_within_horizon_repair.md)，服务器训练/验收待执行。此源check的只读范围和训练批准false不改写，T06仍未批准。

实现时只在本地静态编辑、阅读接口、核对已有结果包字段和Git差异，没有运行本地Python、项目测试、模型、训练或环境，没有连接/修改服务器、提交或推送。服务器通过用户Git同步，45个已验收Python文件与原产物保持不动。随后本轮结果接收仅为读取报告包，没有上传/修改服务器代码或执行服务器项目命令。

## 检查范围

| 项目 | 固定范围 |
| --- | --- |
| 来源 | 原双目标反馈完整20局/check/诊断/350自主确认/固定50步latest/独立verify及全部SHA |
| 推理 | 生产worker350及冻结目标编码器，同设备/后端；无优化器、RSSM构造或环境 |
| 状态 | 每局已保存的自身真实因果features，不重算或拼接另一局RSSM历史 |
| 原建议重现 | 全部320概率及worker mode；手工实际动作与建议可以不同，按真实incoming另核对 |
| 测量重现 | 全部340帧视觉/位姿测量，以及20个本局实际80帧目标重编码 |
| 监督查询 | 每局frame76–79→incoming77–80/remaining4–1，共80事实状态 |
| 六种条件 | actual_endpoint、fixed_goal、swapped_goal、no_goal、zero_goal、base，保存480分布及raw偏好 |
| 整局划分 | repeat0–2 train、3–4 development_holdout，不重新划分 |
| 事实池 | 手工训练24、手工开发16、worker事实40；本轮全部只评价，不训练 |
| 参数/环境 | 更新0、新环境步0、worker实际执行0、手工实际执行0 |

`actual_endpoint`只使用本局真实第80帧RGB/heatmap及经重编码复核的目标特征。它是未来事后目标，仅用于事实拟合诊断，不是部署时可取得的输入。`fixed_goal`为原指定目标，`swapped_goal`为另一个原固定目标。不同目标的比较保持同一个真实状态和原remaining预算；没有反事实rollout，没有试执行新的动作。

目标0的两条偏航终点仍以自身实际终点评价，不能替换为原固定目标。目标1开发留出的下转保持留出；旧20步续段的80–83→84帧标签也不能移植到此次76–79→80帧。

## 服务器运行

先按已有流程在本地提交/推送，再由用户在服务器拉取同一提交。建议中文Git备注：

```text
feat: 增加T05原预算内事实监督检查与结果汇总
```

在服务器项目根目录的现有`(ls)`环境执行一次：

```bash
python scripts/t05_within_horizon_supervision.py check \
  --eval-dir "$PWD/relevance_map/t05_outputs/within_horizon_feedback_evaluate_seed_fixed_20261009T084226" \
  --device cuda:0
```

自动生成新时间戳目录 `relevance_map/t05_outputs/within_horizon_supervision_check_<时间戳>`。不需要重新跑旧check、诊断或20局采集，也不需要手工找checkpoint。入口从已验收来源推导check/verify/生产latest，拒绝不同设备/后端、best、验收副本、旧版本、原SHA改变或不完整20局。

该检查不启动MineDojo，无需无界面环境前缀。可用`--output-dir`指定独立新目录，已存在目录拒绝覆盖，输出不能位于旧模型/轨迹/验收目录内或覆盖其父目录。

## 产物与阅读顺序

- `report.json`：身份、冻结参数/版本/梯度、RNG、来源未变、320/340重现、80/480诊断计数，以及工程状态。
- `queries.json`：80条真实状态的实际动作、worker建议、目标/整局划分/事实池/分支、状态及实际终点SHA、六种分布/raw偏好与事实指标。
- `frame_metrics.csv`：80行同样的标量指标，供整包分析；不包含原图、未来RSSM或新训练样本。
- `trials.json`：20局分别汇总，保留实际终点到原固定目标的距离、偏航/俯仰/位置误差和原纠偏数。
- `diagnostics.json`：按池、两目标×train/development_holdout×纠偏/保持等分支、remaining及逐局报告事实NLL/动作排名/mode匹配和目标消融。另报等局权重均值，避免某局更多保持行掩盖唯一纠偏。
- `training_review.json`：只说明可检验的范围、独立分支覆盖、手工训练动作计数和残留偏航的局号，不批准训练，不给训练预算，不过滤原记录。
- `supervision_manifest.json`：绑定原完整来源身份、新代码SHA、全部本轮JSON/CSV、计数和只读范围。

动作NLL评价保存的事实动作，其匹配不代表动作最优。worker事实的原固定目标mode匹配用于保存概率的自洽检查，不能当作专家正确率。事实排名、目标消融与实际一步进展分别报告；不能由未来实际终点目标的拟合优势直接推断自主控制改善。

纠偏和保持的独立局数与原`coverage.json`交叉核对。同一纠偏局后续保持可重叠，重采样不增加独立纠偏。空组行数/局数为0，效果统计为null，不填成0损失、0失败率或满分。`residual_yaw_episodes`仅定位残差，不用于筛选训练历史。

## 验收与缺覆盖

服务器成功时应完成全部20局、320原概率/worker建议与340测量重现，输出80条真实末4步查询、480六条件分布，保持24/16/40事实池和原整局划分。概率误差打印实际值，超过原推理公差则失败；归一化/一次unimix必须与真实部署分布逐值相同，no_goal/zero_goal/base在实际/固定/交换目标下严格不变。

入口在服务器check中执行真实数据内存副本上的错误划分、worker池混入、终点替换、动作/incoming不一致、错误remaining、终点后查询、伪reset验收标志、专家/训练批准标志，以及80行遗漏/重复/顺序异常的拒绝检查。内存副本不写成轨迹或训练数据。本地未执行这些检查。

已知本轮目标1训练纠偏缺失应继续报告WARN；这是数据事实，不是程序失败。`training_review.json`应保留`target1_train_correction`缺覆盖，手工训练23noop/1上转/0下转、目标0偏航残留两局。其他分支和拟合结果以此次真实前向为准，不能提前填结论。

最终为`[PASS] T05_WITHIN_HORIZON_SUPERVISION_CHECK`，本次实际状态`passed_with_warnings`。所有`approved_for_training / behavior_accepted / t06_approved`仍为false。工程PASS不会生成新的训练批准或自主行为验收；根据完整指标再设计有限修复或另一个独立预声明开发实验，不能重划本批留出、补跑本计划或盲目加更新步数。原 [T06门槛](t05_to_t06_gate.md) 与旧失败保持。

## 自动结果包

最终报告写完后自动汇总本轮及反馈evaluate/check、同状态诊断、原350确认和repair verify，共6个关联阶段。只有报告/表格/原图进入单文件包，模型、缓存、原轨迹和视频留在服务器；运行中失败的报告也会在结束时尝试打包。

反馈末尾的`FEEDBACK_INDEX`即可按既有授权从T2-3090只读接收到电脑，无需逐个下载或改名。只读接收与代码同步分开：助手不上传/修改服务器代码、不执行服务器项目脚本。若来源身份在输出目录创建前已无效，入口打印`input_or_output`错误并拒绝加载，需反馈该错误；不会为无效来源编造检查结果。
