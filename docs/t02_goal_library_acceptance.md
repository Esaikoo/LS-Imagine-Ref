# T02：真实观测目标库构建与验收

状态：本地实现完成；2026-10-06 用户服务器程序验收通过，图集初步检查允许推进 T03。助手只做静态阅读与差异核对，没有运行脚本、测试、采集或训练。

T01 服务器验收已通过：`check_20261006T095512_337705` 的三种模式严格迁移、保存/恢复、计数/RNG 和保护规则全部通过；两个 worker 的初始分布最大误差均为 `6.98e-09`。

2026-10-06 首次 T02 服务器反馈：`build` 在保存后的代表帧重编码检查失败，报“真实代表帧编码与已保存目标向量不同”。原实现从 GPU 提取缓存、默认切到 CPU 验收，也没有固定 TF32/AMP 数值设置；日志缺少实际误差，不能仅凭 traceback 断定差异大小或排除其他原因。该次未通过；修复与后续结果如下。

修复后用户反馈：`build_20261006T110428` / `verify_20261006T110657_826654` 全部工程检查通过，原始 CNN 代表帧重编码误差为 0，最终目标向量及 64 帧回放最大误差约 `1.19e-07`。128 局 6606 帧，PCA 保留方差 `0.9740`，训练 102 局/检查 26 局，两侧覆盖全部 16 类，最大训练类占 `0.1337`。图集中的树木远近、树干/树冠视角与地形差异有一定一致性，也含纯天空及暗场景类别；当前可推进 T03，实际目标可控性仍由 T05 验证，不能把图集当作任务阶段或成功标签。

## T02 做什么

从已有训练 replay 自动抽取真实观测，生成固定的目标表示和目标编号。使用流程为：

```text
真实 RGB + 原 spatial heatmap
    → 原 WM 视觉 CNN 的冻结副本
    → 2×2 空间池化
    → 训练 episode 上拟合一次 PCA，降到 128 维
    → L2 归一化
    → 训练 episode 上做 16 类球面聚类
    → 每类选一个真实成员帧作为目标代表
```

目标编码只读取 `image/heatmap`；不读取 `obs_reward`、动作、奖励、成功标志、未来 RSSM 状态或辅助 zoom 图。CNN 权重从原 checkpoint 严格加载，构建过程中保持冻结，没有新 CNN 微调。降维和聚类使用自动抽样数据，不需要人工标注或另外准备 trajectory。

在当前 64×64、CNN depth=96 配置中，2×2 池化后的维度为 3072，再降到 128。池化保留四个粗空间位置；这只是当前原型的表示选择，是否足以区分有用状态需要图集和后续真实控制检验。

聚类中心负责分配类别；发给 worker 的目标向量取自该类的一张真实代表帧。两者分别保存，避免把平均特征假装成真实观测。重新加载时直接使用已保存的 CNN、PCA、中心、目标向量及编号。

冻结目标编码和类别分配显式使用 FP32，禁用外层 AMP 和 CUDA 的 TF32；结束后恢复调用方精度设置，不更改原训练路径的精度配置。这个数值约定存入编码器结构及内容 ID。修复后的构建会在特征提取设备上重新加载库，比较序列化前后的结果，不把 GPU→CPU 切换混进保存恢复检查。容差仍为 `atol=rtol=3e-5`，没有放宽。不同设备的算子仍可能有小量浮点差异，不能把这些规则当作跨硬件逐位一致的保证。

## 数据规则

- 默认读取原 checkpoint 同级的 `train_eps`，不使用原 `eval_eps` 或 T00 评估轨迹拟合目标库。
- 固定种子抽取最多 128 个合规 episode；每局最多 64 帧，默认候选间隔为 8 步。每局先抽样，限制长 episode 对库的支配。
- 按完整 episode 留出约 20% 检查集。PCA 均值、投影和聚类中心仅在其余训练 episode 上拟合，检查集只用于覆盖与距离统计。
- 根据文件名中的 episode 标识排除重复副本/多次保存的同一 episode，防止它们分到两侧。
- 使用首次 `is_last/is_terminal` 或任务最大步数界限以内的真实帧，含实际终点帧；截去后续 collector 尾部。不读取训练采样器生成的 padding。
- 触发 zoom 的 `image/heatmap` 仍是实际观测，因此保留；`zoomed_image/heatmap_on_zoomed` 不进入编码器。显式标为虚拟/合成的回放会被排除。
- 不合规文件的路径与原因记录在 `replay_selection.json`，不以零图替代。

## 服务器命令

拉取代码后，在原 `ls` 环境中运行构建：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

T00_CHECKPOINT=/root/rivermind-data/mine/projects/tb_logs/LS-Imagine-Ref/minedojo_harvest_log_in_plains/seed_0/20260922T091826/latest.pt
T00_BASELINE=/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t00_outputs/evaluate_20261006T085822_184404
T02_BUILD_DIR="/root/rivermind-data/mine/projects/LS-Imagine-Ref/relevance_map/t02_outputs/build_$(date +%Y%m%dT%H%M%S)"

python scripts/t02_goal_library.py build \
  --checkpoint "$T00_CHECKPOINT" \
  --baseline-dir "$T00_BASELINE" \
  --output-dir "$T02_BUILD_DIR" \
  --device cuda:0 \
  --max-episodes 128 --frames-per-episode 64 \
  --goal-dim 128 --goal-count 16 --seed 0
```

构建通过后，在同一终端运行独立验收：

```bash
T02_LIBRARY="$T02_BUILD_DIR/goal_library.pt"

python scripts/t02_goal_library.py verify \
  --library "$T02_LIBRARY" \
  --device cuda:0 --refit --replay-samples 64
```

若换了终端，将 `T02_BUILD_DIR` 设置成构建时打印的实际 `OUTPUT_DIR`，再执行这段验收命令。

收到首次重编码报错后，拉取修复并重新执行完整 `build`，使用上面命令产生的新输出目录；不要指向旧目录覆盖，也不要对失败目录中的旧 `goal_library.pt` 或特征缓存继续执行验收。旧文件没有新的 FP32 数值约定，加载时会提示重新构建。重跑仍不需要采集或训练。

两条命令都不启动 MineDojo、不执行策略、不加载外部 MineCLIP，也不需要 `MINEDOJO_HEADLESS=1`。未来启动 MineDojo 的命令仍按约定显式附带该前缀。当前目标 CNN 工具也支持 `--device cpu`，但特征提取会更慢。

GPU 用于冻结 CNN 特征提取；PCA 与聚类在 CPU 上拟合，直接使用现有 PyTorch/NumPy，不新增 sklearn 依赖。默认最多 8192 帧，3072 维原始特征缓存约 96 MiB，目标库约几十 MiB。加载原 `latest.pt` 时仍需要容纳其约 2.9 GiB 内容，随后释放非视觉权重和旧优化器。PCA 需要额外 CPU 内存和计算，未在服务器测量运行时间。

这里只需构建一次并验收这份库，暂不进行 1M 训练。若想先检查读取和编码接口，可把构建命令的 `--max-episodes 128` 改为 `16`，输出到另一份新目录；这份小库只用于程序检查，正式语义检查以较完整数据为准。

## 工程验收结果

构建应输出：

```text
[PASS] baseline_input: ...
[PASS] real_replay: ...
[FEATURES] episodes=.../...
[PASS] frozen_encoder: ...
[PASS] pca_train_only: ...
[PASS] real_anchors: ...
[NUMERICS] anchor_raw_features: 最大绝对误差 ...
[NUMERICS] anchor_cached_projection: 最大绝对误差 ...
[PASS] roundtrip_features: 最大绝对误差 ...
[PASS] roundtrip: ...
[WARN] manual_review: 程序构建完成；请检查 ...
[PASS] source_file_unchanged: ...
[PASS] T02_GOAL_LIBRARY_BUILD; report=...
```

`manual_review` 是预期提示：构建完成后还需要检查视觉类别是否有用。类别不均衡、留出集未覆盖某些类别也会报告 `[WARN]`，供分析；它们不自动等同于程序错误。

独立验收应输出：

```text
[PASS] content_identity: ...
[PASS] representative_features: ...
[PASS] numeric_policy: ...
[PASS] zoom_only_guard: ...
[PASS] goal_id_guard: ...
[PASS] content_guard: ...
[PASS] observation_only_frozen: ...
[PASS] library_id_guard: ...
[PASS] agent_metadata: ...
[PASS] episode_split: ...
[PASS] refit_reproducibility: ...
[PASS] real_replay_reencode: ...
[PASS] scope: ...
[PASS] source_file_unchanged: ...
[PASS] T02_GOAL_LIBRARY_VERIFY; report=...
```

检查内容包括：独立加载后的向量/编号一致；代表图确实编码为其已保存目标；额外动作/奖励/未来字段不影响输出；无法只用 zoom 图生成目标；错误编号与被篡改的权重被拒绝；训练/检查 episode 不重叠；相同缓存和种子重新拟合后投影、中心和代表帧编号一致；抽查源真实帧，重编码与缓存一致。

`numeric_policy` 会主动在外层开启 AMP，并在 CUDA 上开启 TF32，确认冻结编码的向量及目标编号与正常前向一致，并检查调用方 TF32 设置得到恢复。这是服务器脚本中的回归验收，助手未在本地执行。

重拟合比较使用当前运行环境和浮点容差，不承诺不同硬件/软件上的新建库逐位相同。后续使用同一份已保存目标库，可通过内容 ID 核对编号语义，而不重新拟合。

如果出现 `[FAIL]`，提供 `error.txt`。样本不足、PCA 有效维度不足或空类别等情况会明确报错；先根据数据和图集修复，不以任意特征填充目标维度。

若仍是特征一致性失败，同时反馈 `numerical_checks.json` 和终端中的 `[NUMERICS]` 行。构建诊断分别比较：重编码原始 CNN 特征与原始缓存、已缓存 CNN 特征通过当前 PCA 后与目标向量、完整重编码与目标向量。每项保存最大/平均绝对误差、逐帧误差、cosine 距离和向量范数；这样可以区分 CNN/backend 差异与投影/归一化差异，而不是只报告一个布尔失败。

## 输出与语义验收

构建目录包含：

```text
goal_library.pt              # CNN、PCA、聚类中心、真实目标向量和代表图
library_manifest.json        # 结构、内容 ID、代表帧来源和构建配置
replay_selection.json        # episode 划分、抽样索引、排除原因和源文件记录
raw_features.npy             # 重拟合检查用的原始特征缓存
features.npz                 # 128 维特征、类别、训练行和代表行
coverage.csv                 # 每类训练/检查帧数、比例、episode 数与平均距离
diagnostics.json             # PCA 方差、覆盖、不均衡程度和有效类别数
numerical_checks.json        # 重编码/投影误差与逐帧数值诊断；失败前保留
representatives.png          # 16 类代表 RGB + 原 spatial heatmap 图集
examples.png                 # 每类最多 3 个不同训练 episode 的相近成员
representatives/             # 单独保存的真实代表 RGB/heatmap
gallery_references.json      # 图集中各成员的 replay 路径与帧索引
report.json
error.txt                    # 出错时生成
```

独立 `verify` 使用另一输出目录保存自己的报告，不覆盖库或构建结果。后续加载 `goal_library.pt` 进行目标编码只需要该文件；此处完整验收还会读取同级缓存和记录中的源 replay。

工程通过后，请检查两张图：

1. 类别是否区分与目标控制有关的视觉差异，例如树木远近、朝向、遮挡或接近目标区域的不同情况。
2. 同一类的不同 episode 成员是否有一致的视觉含义，还是只有亮度/背景相似。
3. 是否大部分类别都由重复的天空、地面或无关景色占据；少数与树有关的状态是否只出现于一个 episode。
4. `coverage.csv` 中的重要类别是否在留出 episode 也出现，是否有非常稀少或被单局支配的类别。

这些类别由数据自动生成，不需要给它们人工阶段标签。先反馈图集和统计，再判断是否适合进入 T03；即使图集合理，目标可执行性仍要到 T05 用真实行为检验。

请反馈：构建与验收终端日志、两个 `report.json`、`diagnostics.json`、`coverage.csv`、`representatives.png` 和 `examples.png`。无需发送 checkpoint、目标库 `.pt` 或大型特征缓存。

## 与 T01 的衔接

`configs.yaml` 新增 `goal_library_path`，默认空，原 `flat_ls` 不加载目标库。目标模式可加载正式目标库；初始化时核对其视觉 CNN 指纹与原 checkpoint 是否匹配。目标库必须与任务、`goal_dim/goal_count` 一致。

目标库以纯 CPU Tensor 元数据嵌入新 checkpoint，不加入原 WM 参数或优化器。保存/恢复同时核对编码器内容 ID 和完整目标库内容 ID，防止“同一编码器、不同聚类编号”被误用。重新加载运行时目标编码器可从 checkpoint 内的完整 payload 构造，无需重新训练或依赖外部库路径。

本阶段独立验收检查目标库元数据挂载/恢复接口；没有重新构造完整 235M agent 做正式目标库的整机恢复。完整策略训练/恢复验收随 T04 接入实际训练入口进行。T01 已验证的默认模式检查仍可使用。

真实目标执行入口仍按 TODO 留在后续步骤；T02 库本身没有训练 worker。其来源使用了原训练模型和 replay，后续实验需对各组公平计入这部分初始化与数据成本。

## 中文 Git 提交备注

```text
实现T02：构建冻结视觉观测目标库并增加可重复验收

- 严格复用原视觉CNN，只以真实RGB和spatial heatmap编码目标
- 从已有replay自动抽样，按episode留出检查集并截去终止后尾部
- 固定PCA降维和球面聚类，为每类保存真实代表帧与目标向量
- 持久化完整编码器、目标编号和内容指纹，接入T01目标库元数据
- 输出覆盖统计、代表图集和独立保存恢复/重拟合/源帧重编码检查
- 记录T01服务器验收通过并更新T02运行说明和TODO状态
```

首次重编码报错的修复提交备注：

```text
修复T02：统一冻结目标编码精度并补充重编码误差诊断

- 冻结CNN、PCA和目标编号分配禁用AMP与TF32，恢复调用方设置
- 在特征提取设备上进行保存恢复验收，保留原有浮点容差
- 保存原始CNN、缓存投影及最终目标向量的分阶段误差
- 增加外层AMP/TF32不改变目标向量与编号的回归验收
- 拒绝复用缺少新数值约定的旧目标库，更新服务器重跑说明
```
