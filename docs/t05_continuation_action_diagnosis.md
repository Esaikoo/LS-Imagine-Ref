# T05 worker350同状态目标与版本动作诊断

最新状态：首次服务器运行因自主历史字段校验错误失败；本地修复已完成，尚未提交、推送或同步服务器，模型诊断待重新验收。代码只在本地修改，服务器通过Git同步；SSH仅读取和接收结果。故障与重跑说明见下方。

2026-10-09。新增`goal_continuation_action_diagnose.py`、`scripts/t05_continuation_action_diagnose.py`及通用汇总工具`scripts/t05_result_bundle.py`，40个旧验收代码不改。本机只静态阅读/编辑，没有运行项目Python、测试、模型、训练或环境。服务器已同步三个新文件，通过AST语法检查及全部30局/480查询的真实来源、动作索引和内存负守卫检查；该检查未构造policy、encoder、RSSM、优化器或环境，模型前向诊断仍待执行。没有提交或推送，`.idea/`未改。

用户另授权从T2-3090接收结果。本轮旧确认已自动整理并保存为单一JSON：5个关联阶段、46个元数据/图片文件，原始文件共13379157字节。接收端校验整个传输文件SHA256及全部46个内嵌文件，原报告、CSV、PNG字节全部保留。没有下载模型、checkpoint、缓存、trajectory.npz或视频，也没有修改登录配置或保存密码。见 [汇总接收说明](t05_result_bundle.md)。

## 固定范围

输入为完整`continuation_control_evaluate_20261008T165222`。自动绑定其check、固定50步生产latest/独立verify，以及该latest的原worker300来源。只接受全部30局工程有效完整16步；源数值gate失败仍保留，不过滤失败局。

- 读取原保存5120维因果状态，frame64–79→incoming65–80、remaining16–1，全部480个真实动作前状态。真实终点frame80仅测量，不再查询动作。
- 两个版本共用严格相同的底座、WM/缓存、目标编码器及原两个固定RGB/heatmap目标；生产300/350和各自独立verify、后端严格核对。不构造OnlineRuntime、RSSM、优化器、候选模型或环境。
- 同一状态、同一预算下，各版本比较goal0/goal1/zero_goal/独立no_goal/base。保存4800个完整分布及raw偏好；两个版本base逐值一致，no_goal在两目标下严格相同。
- 原350的全部480个真实概率/mode与510个视觉/物理测量重现；真实incoming、原生事件、完整因果历史、真实起点及视频SHA核对。
- 保存实际动作的概率/排名/NLL、top2差距、目标mode变化与版本变化，以及实际前后观测的距离/偏好/姿态。按全部30局、16种预算、六个目标条件组和正确组成功/失败分别汇总，缺覆盖不能填零。
- 重点检查目标1repeat0/2末步上转、成功三次forward；目标0repeat0末步下转、成功四次上转；全部交换/无目标局也保留。

300的概率来自350实际生成的状态，不是300的新自主执行。失败实际动作也不是专家标签；概率差不能绑定未执行替代动作的真实收益，不移植frame80–83手工续段标签，不判分布外、不增加预算或改门槛。T06未批准。

## 服务器只需这一条命令

首次交付版本已在T2-3090，最新本地修复需先提交/推送，再在服务器通过`git pull --ff-only origin esaiko-elh`同步。不要直接复制文件或覆盖40个旧验收代码。随后在原`ls`环境执行：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref
python scripts/t05_continuation_action_diagnose.py \
  --eval-dir "$PWD/relevance_map/t05_outputs/continuation_control_evaluate_20261008T165222" \
  --device cuda:0
```

输出目录及汇总目录自动使用新时间戳，拒绝覆盖或与原输入重叠。不启动MineDojo，没有新环境步或训练。固定读取两份生产worker，不能传另一checkpoint、选best、改目标或预算。真实输入/输出初始化前失败时只打印错误；进入分析后的失败报告也自动汇总。

最终输出：

```text
[PASS/FAIL] T05_CONTINUATION_ACTION_DIAGNOSE; report=...
FEEDBACK_FILE=.../T05_continuation_action_diagnose_..._results.json
FEEDBACK_INDEX=.../bundle_index.json
FEEDBACK_ZIP=.../T05_continuation_action_diagnose_..._results.zip
```

运行后只需发`FEEDBACK_INDEX=...`这一行，或者发`OUTPUT_DIR=...`。Codex可从已确认的T2-3090整理/接收汇总文件到电脑，再读取本地文件分析；无需逐个下载、重命名或上传图片。若SSH登录需要用户操作，保留单文件下载方式，不改认证设置。没有安排定时任务或后台监控。

## 产物与验收

诊断目录保存`report.json`、`queries.json`、`frame_metrics.csv`、`trials.json/csv`、`diagnostics.json`、`diagnosis_manifest.json`及两个目标各5次末步的概率图。汇总JSON包含诊断和关联确认/check/verify的原文与PNG；ZIP另保留按阶段目录组织的原文件，二者任选其一反馈即可。

预期：30局/480状态/4800保存分布、末两步60状态、原概率重现480、测量重现510，优化器/环境/执行动作0。内存副本负守卫覆盖真实incoming、终止、预算、worker版本、种子、保存mode、终点后查询、遗漏/重复和底座变化，不写为真实轨迹。

服务器语法、真实来源及动作索引守卫已通过；两版本模型前向、原概率/测量重现与新诊断自动汇总仍待运行验收。新诊断全部通过后再分析正确目标是否改变错误转向的排序、50步更新的概率变化和实际状态覆盖；不能凭工程PASS或原离线NLL继续追加训练或批准T06。

中文Git提交备注：`feat: 增加T05同状态版本诊断与结果自动汇总接收`

## 首次运行的历史格式修复

2026-10-09，用户运行`continuation_action_diagnose_20261009T041904_344242`失败。完整来源、冻结300/350与固定目标编码均通过，尚未完成动作概率诊断。错误来自诊断入口误要求`history_check.json.saved_history_roundtrip`；该字段属于另一种历史格式，原自主记录代码只保存自身因果历史通过、真实reset、是否比较其他历史、起点/总帧数及最大绝对误差。原保存记录合法，没有缺失真实轨迹或视频。

失败结果包已自动接收到本机，4阶段31文件，全部字节SHA校验通过：

```text
C:\Users\28620\Downloads\T05Results\received_20261009T121949_5709748\T05_continuation_action_diagnose_20261009T041904_344242_results.json
```

另只读核对服务器全部30局：全部没有被误要求的字段；30/30自身历史字段有效，最大误差均与逐局指标一致，30/30逐局指标等于原manifest，30/30执行/视频验收标志通过且视频均81帧。本轮服务器只读取元数据，没有修改项目代码、旧记录或认证配置，没有运行模型/训练/环境。

本地仅修改`goal_continuation_action_diagnose.py`与`scripts/t05_continuation_action_diagnose.py`：按原自主保存格式验收，不补造字段；最大历史误差须有限、非负并与原指标相等；逐局指标、因果历史/reset/自身边界、执行与完整解码视频仍严格检查。全部30局元数据检查移到模型加载前；实际分析再绑定真实数组长度检查。不同失败条件单独报错，避免一个笼统错误遮盖具体原因。

首次trial新增10个内存负守卫：失败历史、伪reset、跨局历史、错误起点/帧数、缺失/不一致/非有限误差、缺帧视频及不同指标；即使加入`saved_history_roundtrip=True`也不能绕过这些检查。守卫只在内存中使用真实记录副本，不写成历史或训练数据。本机仅静态阅读、修改及差异核对，未运行项目Python或测试；新增守卫与完整模型诊断待用户服务器验收。

本地提交/推送后，在服务器运行：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref
git pull --ff-only origin esaiko-elh
python scripts/t05_continuation_action_diagnose.py \
  --eval-dir "$PWD/relevance_map/t05_outputs/continuation_control_evaluate_20261008T165222" \
  --device cuda:0
```

旧失败目录保留，新目录自动生成，无需重跑30局自主评估、训练或补采样。输出`source_saved_history_schema`及`saved_history_schema_contract`通过后继续原480概率/510测量诊断；最终仍自动汇总。只需反馈新的`FEEDBACK_INDEX`或`OUTPUT_DIR`，无需逐个下载或改名。行为gate失败与T06未批准不变。

本次中文Git提交备注：`fix: 修复T05同状态诊断的自主历史格式校验`
