# T05 原16步末4步有限混合修复

2026-10-09新增 `goal_within_horizon_repair.py / scripts/t05_within_horizon_repair.py`，独立check/train/verify三个入口。绑定已通过的 `within_horizon_supervision_check_20261009T112719_064845`，从生产350双头权重开始，固定追加50步；目标是改善原预算内真实保持事实的拟合，不能声称双目标纠偏训练覆盖完整。

最新服务器验收：`within_horizon_repair_check/train/verify_20261009T150348` 已全部工程通过，生产350+50=400、严格恢复与下一更新等价通过；开发实际终点NLL3.737709→1.335897，mode0/16→11/16，固定目标仅8/16。预声明保留检查8/10超限；目标1下转事实排序退步，因此未来确认的保留前提未满足，旧gate/T06不改。9阶段99文件已只读保存到电脑并独立核对，见 [完整结果与下一步范围](t05_within_horizon_repair_result_analysis.md)。无需重跑以下三步、改预算或选best；下一步只读350/400冲突诊断已本地实现，服务器尚未运行，见 [单步入口](t05_within_horizon_conflict_diagnosis.md)。

以下为本地实现时的工作记录和原运行协议：只做静态阅读、实现、字段/来源/保存接口及Git差异核对，没有执行项目Python、测试、模型、训练或环境，没有连接/修改服务器，没有提交或推送。47个已验收Python文件保持不动；实际服务器执行由用户完成。`.idea/`不动。

## 固定协议

| 项目 | 预声明范围 |
| --- | --- |
| 生产来源 | 已独立verify的worker350固定latest，原300+50来源链完整SHA绑定 |
| 新监督 | 原20局/80条frame76–79→incoming77–80/remaining4–1，以本局真实80帧RGB/heatmap为事后目标 |
| 新训练池 | repeat0–2原6个完整手工局的全部24事实动作：23noop、1上转、0下转 |
| 只评价 | 手工开发16、worker事实40，原84帧续段40动作，原整局留出与参考validation |
| 每批 | 192原训练+48原参考（每目标24）+16本轮手工训练，共256 |
| 采样/损失 | 各池均匀有放回，两分支同批部署分布NLL，无动作加权/纠偏过采样/交换目标训练 |
| 更新 | 双头各固定50步、学习率5e-5，新Adam步0，350权重分别继承 |
| 生产版本 | 完成后350+50=400；源与追加计数分开，验收副本额外更新不进入生产计数 |
| 模型/环境 | 底座、WM、目标冻结；无优化器接管它们，无RSSM/候选构造，无新环境或动作执行 |
| 保存 | 新独立格式、固定latest、双头/优化器/三池计数/RNG及相对共享依赖SHA；不选best、不延预算 |

本局80帧是未来事后监督目标，不是部署输入或最优专家标签。完整20局全部保留：不按loss、终点距离、偏航或mode筛选，不移入开发下转，不把旧80–83/84帧标签改成此次76–79/80帧，不过滤两条目标0偏航历史。旧续段只评价，其保持/纠偏能力变化在独立指标中报告。

旧源报告的`approved_for_training=false`等只读范围标志不改写；新入口是用户授权的独立固定开发训练协议，不把旧check或事实数据重写成专家验收。check核对协议和step0，不自动启动train；行为/T06标志始终false。

## 先保护原能力，再看有限拟合收益

step0相对原生产350预声明最大NLL增加0.02，共10项：原26局完整留出×goal/no_goal两项，参考目标0/1×本局真实终点/原固定目标×goal/no_goal八项。目标1参考也必须单独保护。各项单独保存，不能用总均值抵消退步。超限报告WARN，保留实际固定latest；不追加更新、不选较早step或best来掩盖它。

原84帧续段作为单独保留能力描述，不加入新训练池，也不拿它的下转填补本轮目标1训练纠偏。新80帧全部80状态按两目标、整局划分、保持/纠偏、remaining、逐局以及24/16/40池分别输出。六种分布仍是实际终点、固定目标、交换目标、no_goal、zero_goal、base。空纠偏统计为null，不填零；重采样次数与独立纠偏行/局数分开。

仅离线NLL/mode改善不放行T06。check同时预声明未来worker400的mode/新随机种子6/原16步/30局goal-no_goal-swapped_goal确认计划，原两个目标分别方向≥4/5且五次均值胜两对照的gate保持。**此次只交付离线修复，新的400自主确认入口尚未实现**；先看train/verify与10项能力保留结果，再实现独立确认。完整视频人工验收仍待完成。

## 用户在服务器执行

先在本地提交/推送，再由用户在服务器拉取同一提交。中文Git备注：

```text
feat: 增加T05原预算末4步固定混合修复与独立验收
```

在服务器项目根目录现有`(ls)`环境复制执行。三个阶段使用同一组变量和独立目录；`&&`确保前一步失败时不继续后面的训练/验收。

```bash
T05_WH_SOURCE="$PWD/relevance_map/t05_outputs/within_horizon_supervision_check_20261009T112719_064845"
T05_WH_STAMP="$(date +%Y%m%dT%H%M%S)"
T05_WH_CHECK="$PWD/relevance_map/t05_outputs/within_horizon_repair_check_$T05_WH_STAMP"
T05_WH_TRAIN="$PWD/relevance_map/t05_outputs/within_horizon_repair_train_$T05_WH_STAMP"
T05_WH_VERIFY="$PWD/relevance_map/t05_outputs/within_horizon_repair_verify_$T05_WH_STAMP"

python scripts/t05_within_horizon_repair.py check \
  --supervision-dir "$T05_WH_SOURCE" \
  --output-dir "$T05_WH_CHECK" --device cuda:0 &&
python scripts/t05_within_horizon_repair.py train \
  --check-dir "$T05_WH_CHECK" \
  --output-dir "$T05_WH_TRAIN" --device cuda:0 &&
python scripts/t05_within_horizon_repair.py verify \
  --check-dir "$T05_WH_CHECK" --checkpoint "$T05_WH_TRAIN/latest.pt" \
  --output-dir "$T05_WH_VERIFY" --device cuda:0
```

三个入口都不启动MineDojo，无需`MINEDOJO_HEADLESS`前缀。没有可调整训练步数、学习率或采样比例的参数，也没有resume入口；失败目录原样保留，不能从失败后的状态追分或覆盖。输出目录已存在即拒绝。可省略`--output-dir`使用自动唯一时间戳，但后两步仍需填写上一阶段输出路径；以上固定变量方式最方便。

如果check失败，反馈它的日志及`FEEDBACK_INDEX`即可，不运行train。训练完成即使能力保留WARN，也照常verify其固定latest以确认格式/恢复工程；WARN不会被verify清除，后续自主确认仍须另审保留结果。

## 应看到的检查与产物

check应完成完整来源SHA、真实incoming/本局80帧编码、原84帧/参考/原缓存复核，warm start350、新Adam步0，192/48/16采样，开发/worker/子集/替换终点/错误动作预算等负守卫。未更新模型在同80状态重现原480分布/raw/mode，实际误差按实打印。输出step0所有池指标、10项保留计划与未来确认计划；`target1_train_correction`仍WARN，空组null。check新训练更新0。

train按step10/20/30/40/50报告实际/固定目标开发NLL和mode，写固定latest，不根据留出挑选模型。终点为各分支追加50更新、优化器总更新100、每分支12800标签；采样计数原训练9600/参考2400（每目标1200）/本轮800，开发/worker/旧84帧采样0。800次有放回事实抽样不增加24个训练行或那一条上转的独立证据。

verify严格恢复本轮格式/来源350/双头/新优化器步50/三池计数/RNG与check，拒绝旧格式、验收副本、版本/身份/计数/划分篡改与形状异常。生产latest、内存恢复副本和独立无优化器推理的五条件分布一致；最终指标逐值复现train。两内存副本各执行一次实际混合更新验证下一步等价，总4次验收优化器调用另计；不保存更新后的副本，不写回生产latest，不把版本改成401。源依赖、冻结底座/WM/目标及旧47代码不变。

主要产物：

- `report.json`：工程状态、固定协议/来源、计数、SHA/后端/输入未变与范围。
- check的`supervision.npz`：完整原参考64行、新80事实与旧84帧40行的已核对数值；只有规定24新事实进入训练。
- `initial_metrics.json / metrics.json / evaluation.jsonl`：step0及固定进度的原26局、双目标参考、新80帧和旧84帧评价。
- `queries.json / frame_metrics.csv`：新80状态的六种完整概率/raw及标量，含原事实worker350身份与当前评价版本两个字段，避免混淆历史动作和新预测。
- `retained_frame_metrics.json`：原参考与84帧续段的各自事实预算，另存JSON，不与新80帧CSV混列。
- `coverage.json / retention_plan.json / confirmation_plan.json`：check覆盖、10项能力保留规则与未来原16步确认计划。
- `sampling.json / retention.json / diagnostics.json`：实际抽样、逐项退步和初末变化，缺覆盖与旧gate保持。
- `latest.pt`：固定新格式生产快照；verify的`roundtrip_verification.pt`仅作恢复检查，禁止训练和控制。

结束或运行中失败后，自动按原字节打包关联阶段，check/train/verify分别7/8/9阶段，打印`FEEDBACK_FILE / FEEDBACK_INDEX / FEEDBACK_ZIP`。模型、NPZ缓存/轨迹/视频和验收快照不进入结果包；JSON/CSV/原图及报告完整保留。把最后一步的日志或`FEEDBACK_INDEX`反馈给助手即可按既有授权只读接收，无需逐文件下载/改名。助手不上传/修改服务器代码或自行执行服务器训练。

本地实现完成不等于服务器实际PASS，更不等于保持或纠偏泛化通过。服务器效果以本轮固定train/verify结果及后续新的自主确认判断；[原T06门槛](t05_to_t06_gate.md) 和旧行为失败不改写。
