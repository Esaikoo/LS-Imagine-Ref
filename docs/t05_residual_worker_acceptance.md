# T05：冻结底座的目标残差 worker

2026-10-07。本地已实现；用户反馈 `residual_check / residual_train / residual_verify_20261007T031803` 三步离线工程验收通过，两分支各更新 200 次。完整留出有目标/无目标修正/底座 NLL 为 1.687366/1.709115/1.752972；长预算动作匹配仍弱，真实控制尚未验收。详见 [本轮结果与真实评估计划](t05_residual_worker_result_analysis.md)。助手仅静态阅读、编辑和核对差异，未运行 Python、测试、训练、模型或环境，未提交/推送。仍属于 T05；T06 未开始，旧 30 trial 暂缓。

后续专用六组真实评估的服务器check/adopt已通过，首个12-trial块因起点不一致未通过；旧剩余24次暂停。起点审计和固定预热双次重放预检已通过，新参考/评估入口已本地实现，当前使用 [新预热三步命令](t05_warmed_control_acceptance.md)，服务器验收待运行。无需重跑本文件的训练或已完成预检，仍未进入T06。原六组方案见 [残差真实重复评估说明](t05_residual_control_acceptance.md)。

## 这次改变什么

已有三种子信息探针显示目标中有动作预测线索，但小探针压缩了当前状态。新实验固定已验收的独立无目标 BC，保留它对完整 5120 维状态及 remaining 的处理；只训练两个小修正分支。

```text
完整当前状态 + remaining → 冻结无目标BC底座 → 12维原始动作偏好
当前状态投影/归一化 + 目标归一化 + remaining → 小修正分支
底座动作偏好 + 修正量 → 原onehot分布（只做一次原unimix）
```

有目标分支使用真实终点目标；独立无目标分支使用恒零目标。两分支容量、初始化、真实批次与更新预算相同，优化器独立。修正最后一层权重/偏置为零，初始动作分布和 NLL 应与底座逐值一致。

底座是 `train_no_goal_uniform_20261007T012127/latest.pt`，其对应 T04 verify 是 `verify_no_goal_uniform_20261007T012218`。工具匹配该具体文件的验收记录、缓存/目标库身份、计数及共享依赖 SHA256。默认采样固定 `uniform_rows`，没有候选模型训练。

训练损失是最终部署动作分布的真实动作 NLL。目标置零/跨局替换只用于评价，不把替换目标配上伪造动作标签，也不强制不同目标产生不同动作。不新增轨迹或人工阶段标签。

## 本地交付

- `goal_residual_worker.py`：冻结底座、零初始化修正分支、成对 BC、完整保存恢复、共享依赖与专用推理加载器。
- `scripts/t05_residual_worker.py`：`check / train / verify` 三个离线入口。
- `OnlineRuntime`：原真实观测和 incoming action 的逐帧因果状态接口、目标条件动作分布；禁止结束后发动作。此次 verify 用已有真实回放检查它，不启动环境。
- 独立格式 `ls_imagine_goal_residual_worker_v1`；不改原 flat_ls、T04 worker、小探针和旧 T05 评估入口。旧入口不接受这份新模型，需要后续匹配新模型身份/条件的真实评估设计。

训练仅使用缓存状态，不实例化 WM/目标编码器；verify 的在线接口检查才严格加载冻结 encoder/RSSM 和目标库。新优化器只拥有两个修正分支。

## 服务器运行：按顺序执行三步

本轮三步已经通过，以下保留为运行说明，无需为进入真实评估重新训练或重复验收。残差真实评估入口已另行实现，使用上方的新验收说明；不能把新 `latest.pt` 直接交给旧 T05 入口。

拉取本地代码后，在原 `ls` 环境运行。三步均离线，**不启动 MineDojo，因此无需 `MINEDOJO_HEADLESS=1`**。每步最后出现 PASS 再执行下一步。以下所有变量均重新定义，不依赖旧终端中的 T05 变量。

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T05_BASE="$PWD/relevance_map/t04_outputs/train_no_goal_uniform_20261007T012127/latest.pt"
T05_BASE_VERIFY="$PWD/relevance_map/t04_outputs/verify_no_goal_uniform_20261007T012218"
T05_RESIDUAL_TAG="$(date +%Y%m%dT%H%M%S)"
T05_RESIDUAL_CHECK="$PWD/relevance_map/t05_outputs/residual_check_$T05_RESIDUAL_TAG"
T05_RESIDUAL_TRAIN="$PWD/relevance_map/t05_outputs/residual_train_$T05_RESIDUAL_TAG"
T05_RESIDUAL_VERIFY="$PWD/relevance_map/t05_outputs/residual_verify_$T05_RESIDUAL_TAG"

python scripts/t05_residual_worker.py check \
  --cache-dir "$T04_CACHE" \
  --base-checkpoint "$T05_BASE" --base-verify-dir "$T05_BASE_VERIFY" \
  --output-dir "$T05_RESIDUAL_CHECK" --device cuda:0 --seed 0
```

预期结尾：`[PASS] T05_RESIDUAL_WORKER_CHECK`。关键检查 `accepted_no_goal_base / zero_residual_initialization / distribution_contract / frozen_ownership / no_training` 为 PASS。`initialization_max_error=0`，每分支 724620 参数，优化器状态为空；格式/形状/remaining/artifact guard 的 PASS 表示成功拒绝错误输入。

```bash
python scripts/t05_residual_worker.py train \
  --cache-dir "$T04_CACHE" --check-dir "$T05_RESIDUAL_CHECK" \
  --output-dir "$T05_RESIDUAL_TRAIN" --device cuda:0 \
  --steps 200 --eval-every 50 --log-every 50 --save-every 50
```

预期结尾：`[PASS] T05_RESIDUAL_WORKER_TRAIN`。默认 batch=256、state projection=128、hidden=256、学习率=3e-4、grad clip=10；从原信息探针沿用小分支的配置，只改变底座和零初始化残差结构。先保持这些值完成一轮，不同时调采样或重建目标表示。

应记录 0/50/100/150/200 五个评价点；每分支 200 次更新、51200 个抽取标签，总优化器更新 400；`base_updates=wm_updates=new_env_steps=0`。三组指标名称分别 `base / no_goal / goal`；step=0 三组完全一致。当前缓存的完整留出底座 NLL 预计约 **1.75297**，每次完整评价应保持同一个值；若明显不符，先检查加载和统计协议。

```bash
python scripts/t05_residual_worker.py verify \
  --cache-dir "$T04_CACHE" --check-dir "$T05_RESIDUAL_CHECK" \
  --checkpoint "$T05_RESIDUAL_TRAIN/latest.pt" \
  --output-dir "$T05_RESIDUAL_VERIFY" --device cuda:0 --prefix-steps 32
```

预期结尾：`[PASS] T05_RESIDUAL_WORKER_VERIFY`。应通过 `strict_load / roundtrip / strict_inference_load / incremental_causal_state / next_update_equivalence / frozen_dependencies`。推理接口不构造优化器，与 BC 分布逐值一致；无目标修正不依赖目标。两局真实前缀逐帧更新与完整恢复一致，reset 清空旧历史，拒绝无历史/错误incoming/越界步数/终止后动作。恢复检查仅在两份内存模型各做一次真实批次更新，不修改训练 latest。

## 产物与磁盘

训练目录保留 `latest.pt / best_pair.pt / best_pair.json / metrics.json / evaluation.jsonl / training.jsonl / sampling.json / diagnostics.json / query_metrics.csv / queries_predictions.npz / validation_full_predictions.npz / report.json`。

每份成熟双分支完整快照预计约 18 MiB，包含两分支及其 Adam、计数、抽样器、RNG 与共享依赖引用；不重复保存底座、WM 或约 819 MiB 的状态缓存。写入前检查 latest、best 和一次临时写入，加 64 MiB 日志余量；原子保存失败保留上一份文件。verify 另存一份约 18 MiB 的 `roundtrip_verification.pt`，明确禁止真实 resume/控制。

请保留底座 checkpoint、其 T04 verify 报告和缓存 `frozen_bundle.pt`；新快照通过相对路径与 SHA256 引用这些依赖，不能只搬走/上传一份新 latest。服务器模型仍留在服务器。

`best_pair` 按完整留出两修正分支 NLL 平均值选取同更新数的配对快照，只作为次要诊断；可能选中第 0 步，这种文件不能用于真实控制。主比较固定第 200 步 `latest.pt`，不分别选择对各组最有利的步数。

## 反馈与判定

先反馈三步日志，以及训练目录 `metrics.json / diagnostics.json / evaluation.jsonl / sampling.json` 和 verify 目录 `report.json`。

工程验收要求三个入口 PASS、冻结/分布/恢复/在线因果接口通过；不把额外收益达到某阈值作为程序 PASS 条件。

科学判断以**同一完整留出数据**为主：

- `no_goal_minus_goal_nll`：同容量/同预算下目标的额外收益。
- `base_minus_goal_nll` 和 `base_minus_no_goal_nll`：两修正分支分别是否改善强底座；仅有目标比无目标好而两者都退化时，不能宣称改进了底座。
- `goal_accuracy / no_goal_accuracy / base_accuracy`、逐局、逐remaining及完整 `remaining_13_plus`：是否只在短步或概率上改善。
- 查询子集的 `zero_goal / swapped_goal` 敏感度与 centered correction RMS：诊断是否使用目标、修正是否明显偏离底座，不能当作可达性证明。
- 0–200 步的绝对 NLL 与增益曲线：不能只看 gap 增长，忽略模型整体退化。

若离线结果支持，再实施匹配新 worker 身份的 T05 真实重复执行比较；旧30trial不直接替换 checkpoint 继续跑。真实目标控制验收前不进入T06、不运行1M。若目标没有额外收益，或只在临近终点有效，保留该结果后分析表示/覆盖，不继续盲目延长训练。

## 中断恢复（只有需要时运行）

输出必须是新目录，保留同一个已通过 check 和底座依赖；`--steps` 是累计更新数。

```bash
T05_RESIDUAL_RESUME="$PWD/relevance_map/t05_outputs/residual_resume_$(date +%Y%m%dT%H%M%S)"
python scripts/t05_residual_worker.py train \
  --cache-dir "$T04_CACHE" --check-dir "$T05_RESIDUAL_CHECK" \
  --resume "$T05_RESIDUAL_TRAIN/latest.pt" \
  --output-dir "$T05_RESIDUAL_RESUME" --device cuda:0 \
  --steps 200 --eval-every 50 --log-every 50 --save-every 50
```

已到200步的文件无需再次resume；若从中途恢复，最终验收用新输出 latest。`sampling.json` 的本轮抽取数只计恢复后的更新，checkpoint总计数保留全部历史。

## 中文 Git 提交备注

```text
feat: 实现T05冻结底座的目标残差worker及独立无目标对照

- 保留完整状态BC底座，只训练零初始化动作修正分支
- 有无目标共享真实批次，动作偏好修正后沿用原onehot混合分布
- 增加200次短预算训练、完整留出三组评价和逐预算诊断
- 共享冻结依赖，校验分支优化器/RNG恢复与推理因果接口
- 更新服务器三步验收说明和TODO，真实控制及T06仍待验收
```
