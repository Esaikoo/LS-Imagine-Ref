# GRACE 工程命名、训练模式与历史兼容

GRACE（Goal-Referenced Adaptive Control and Exploration）是这份 LS-Imagine 扩展的工程名。工程版本 **GRACE-1.0.0** 继承 **VN4R3D2P3R1**，基准提交为 `f6ee70d9b5abbfd93595e7b80c3c6f789570dcb7`（分支 `esaiko-pscv`）。下文区分名称迁移和实验处理。原 LS-Imagine 的署名、环境和长短程世界模型沿用原项目。

## 训练实验的定义

| 训练配置及 EXPERIMENT_MODE | A 后果预测 | B 历史与局部地图 | C 选项控制 | D 区域验证 | 策略热图 | 参数数¹ |
|---|---|---|---|---|---|---:|
| `grace_full` | 学习、参与排序 | 学习、读写 | 学习与规则 | 学习、校准 | 按 D 门控融合 | 266,697,295 |
| `grace_d_only` | 无实例 | 无实例 | 无实例，原 Actor 执行动作 | 学习、校准 | 按 D 门控融合 | 258,507,390 |
| `grace_dac` | 学习、参与排序 | 无实例，输入为零 | 学习与规则 | 学习、校准 | 按 D 门控融合 | 265,852,043 |
| `grace_no_fusion` | 学习、参与排序 | 学习、读写 | 学习与规则 | 学习、校准，提供非热图证据 | 原 U-Net | 266,697,295 |

¹ 正式采沙配置，观测编码器使用 `image`、`heatmap`、`obs_reward`。计数包含慢速价值网络、C 的目标网络等参数副本，共享参数仅计一次。分别实例化正式尺寸 R1 和四种 GRACE 模式，逐个统计实际注册的参数。公共主体 258,457,228；A 1,618,349；B 845,252；C 5,726,304（其中目标副本 2,863,152）；D 50,162。运行时按本次实际实例重新打印并写入 checkpoint。网络参数数量与优化器更新的参数数量口径不同。

- **Full。** 依次建立原世界模型/Actor-Critic、D、B/A 和 C。保留原有初始化的随机数隔离、模型调用顺序、固定采样协议、计数和支持门槛。H8 仍按原间隔作为辅助训练。
- **D-only。** 在原有合法图像输入上计算 D 与融合热图，经 RSSM 到原 Actor。训练时仍有原有随机预填充、Actor 采样和 LS 长短程想象。正式动作交给同一执行器。保存 D 时间跟踪和 Actor 动作来源，不产生 C 决策、B 轨迹或 A 排名。训练优化器为世界模型、Actor、Value 和 D。
- **D+A+C。** 保留 A/C 的完整输入宽度。B 模型、优化器和持久状态均不创建。在线不读写记忆、不运行 PoseNet，提供全零记忆、地图摘要、方向及方向新颖性。A 的真实决策帧由时间对齐的 D 上下文记录识别，继续使用真实训练标签与原训练/留出划分。A/C 仍使用任务已有的训练期动作后果标签（例如新区域、重访）。这些标签与 B 预测出的地图分开。C 自身的动作计数、短时视觉停滞检测、动作回执和 D 的短时目标确认均保留。C 学习只使用本模式实际产生的选项，排除来源不符的旧决策掩码和转移。`log_grace_dac` 是新模式独有的采集标记。新实验从头采集。B 省略后 A/C 的初始随机权重不承诺与 Full 对应子网逐元素相同。
- **Full w/o D fusion。** 在线与训练期策略/世界模型使用原 U-Net 热图，D 仍优化区域损失、做校准，输出区域证据、冲突、目标状态，供其余模块使用。冻结评测的临时动作覆写与上述训练干预分别报告。

## 运行入口

仓库根目录，先 `conda activate ls_imagine`。四行命令分别对应独立的从头训练。使用前选择一行。

```sh
EXPERIMENT_MODE=grace_full      BANK=assets/goal_references sh scripts/train.sh harvest_sand
EXPERIMENT_MODE=grace_d_only    BANK=assets/goal_references sh scripts/train.sh harvest_sand
EXPERIMENT_MODE=grace_dac       BANK=assets/goal_references sh scripts/train.sh harvest_sand
EXPERIMENT_MODE=grace_no_fusion BANK=assets/goal_references sh scripts/train.sh harvest_sand
```

后台启动示例。更换 `MODE` 选择实验，输出目录按模式分开。脚本默认 1M 环境步，正常结束或 Python 失败均沿用现有可选 ntfy 通知。

```sh
TASK=harvest_sand
MODE=grace_full
mkdir -p "logs/train/$TASK"
LOG_FILE="logs/train/$TASK/${MODE}_$(date +%Y%m%d_%H%M%S).log"
nohup env EXPERIMENT_MODE="$MODE" BANK=assets/goal_references \
  NTFY_TOPIC=rl-training-esaiko TRAIN_LOG_FILE="$LOG_FILE" \
  sh ./scripts/train.sh "$TASK" > "$LOG_FILE" 2>&1 &
```

旧用法 `A_MODE=vnext4 sh scripts/train.sh harvest_sand` 仍解析 `minedojo apm_vnext4`。对应的新完整训练使用 `EXPERIMENT_MODE=grace_full`。两个变量同时提供时要求值一致。均未设置时沿用历史 `vnext` 默认。模式名、配置和训练模块开关冲突会在进入训练前报错。图库沿用各任务已提交的资源，四组共用同一任务指纹。

## 名称迁移表

| 旧名称 | 新名称 | 类别 | 兼容方式与作用 |
|---|---|---|---|
| `apm/` | `grace/` | Python 包 | 实现集中在新包；旧叶模块通过 sys.modules 指向同一个模块对象，可解析历史完整类名 |
| `apm/legacy/` | `grace/legacy/` | 历史包 | 保留历史算法和旧导入，未合并进当前训练模式 |
| `SearchVNext` | `PredictiveMemorySystem` | outcome.py 类 | 旧名字为同一类对象；A/B 保持一个封装 |
| `VNextController` | `OptionController` | control.py 类 | 旧名为别名 |
| `SearchVisualizer` | `EpisodeVisualizer` | visualization.py 类 | 旧名为别名 |
| `LS_Imagine._vnext_policy` | `LS_Imagine._grace_policy` | 策略方法 | 保留旧方法别名 |
| `LS_Imagine.observe_vnext_results` | `LS_Imagine.observe_control_results` | 环境回执 | 保留旧方法别名；模拟器对只有旧方法的调用方仍可交付回执 |
| `sample_episodes_vnext` | `sample_predictive_episodes` | 历史槽位 Replay 采样 | 同函数别名；当前 Full 仍走 sample_spatial_episodes |
| `check_vnext_checkpoint` | `check_prediction_checkpoint` | 模型协议 | 同函数别名；增加无 B 架构严格校验 |
| `restore_vnext_config` | `restore_prediction_config` | 评测恢复 | 同函数别名；识别真实训练模式 |
| `VNEXT_VERSION` | `PREDICTION_PROTOCOL_VERSION` | 历史基础预测常量 | 保留数值 1 与旧常量名；空间预测历史版本仍为 2 |
| `apm_vnext4` | `grace_full` | 配置 | 共用同一 YAML 基础配置；新增命名实验身份 |
| `A_MODE` | `EXPERIMENT_MODE` | 启动变量 | 旧变量可用，冲突显式失败 |
| `VN4R3D2P3R1` | `GRACE-1.0.0` | 工程版本 | 见 training_state.CODE_VERSION 与 grace.experiments；独立于权重、Replay 和评测协议 |
| `scripts/test_a.sh` | `scripts/test_all.sh` | 回归命令 | 旧命令转发全部参数，历史 MineDojo 短测选项保留 |
| `scripts/redraw_search.py` | `scripts/render_figures.py` | 论文重绘 | 旧 CLI 及 main 转发到同一渲染器 |
| `tests/test_search_vnext_models.py` | `tests/test_predictive_memory.py` | A/B 测试 | 旧脚本可单独执行 |
| `tests/test_search_vnext_control.py` | `tests/test_option_control.py` | C 测试 | 同上 |
| `tests/test_search_vnext_evidence.py` | `tests/test_search_evidence.py` | 证据测试 | 同上 |
| `tests/test_search_vnext_integration.py` | `tests/test_predictive_integration.py` | 联调测试 | 同上 |

**逐文件包迁移。** 对以下每一个文件，旧目录 `apm/` 和新目录 `grace/` 的相对路径相同。叶模块保留导入转发；包级 `__init__.py` 提供命名空间。

| 范围 | 相对路径 |
|---|---|
| 名称空间及执行器 | `__init__.py`、`actuator.py` |
| 预测、记忆、控制 | `outcome.py`、`spatial_memory.py`、`odometry.py`、`control.py` |
| 数据、监督、验证 | `replay.py`、`evidence.py`、`goal_reference.py`、`goal_guidance.py` |
| 归档和展示 | `diagnostics.py`、`visualization.py`、`paper_archive.py`、`paper_figures.py`、`figure_style.py`、`imagination_figures.py` |
| 历史研究实现 | `legacy/__init__.py`、`legacy/observation.py`、`legacy/memory.py`、`legacy/control.py`、`legacy/feedback.py`、`legacy/branches.py` |

**配置和模块开关。** 唯一新增的训练配置字段为 `experiment_mode`。默认空值用于历史实验，命名预设从同一 `apm_vnext4` YAML 锚点继承。保留全部已有 `vnext_*`、`goal_*`、`search_*` 字段名、CLI 类型和数值，不引入第二套名称。公共参数及 A/B/C/D、可视化、实验身份在 configs.yaml 分区。`grace.experiments.MODULES` 定义模式与模块成员；`validate_training_mode` 校验开关，`module_manifest` 与 `checkpoint_mode` 写读模式协议，`parameter_counts` 按实际实例计数。

| 新预设 | 相对基础配置的字段覆盖（另加同名 experiment_mode） |
|---|---|
| grace_full | 无 |
| grace_d_only | vnext_memory_enabled=False、vnext_control=False、goal_search_enabled=False |
| grace_dac | vnext_memory_enabled=False；按实验身份省略 B 实例及优化 |
| grace_no_fusion | goal_fusion_enabled=False |

`search_vnext_enabled`、`vnext_spatial_enabled` 在四组中仍作为同一驱动、执行器和 fix3 数据协议的标识。命名消融应读取 checkpoint 的 `experiment_modules` 判断 A/B/C/D 是否参与。旧 `full` 继续指向 `apm_a`；`vnext4_d_only` 继续关闭原控制/目标搜索并保留辅助网络；`vnext_ac` 是历史的记忆输出开关；`vnext4_search_only` 沿用历史关闭融合的参数与存档身份。历史 `off/loss_only/ab/abc/abc_v1/a5/ab5/ac5/bc2/b3/ac/bc/b/vnext/vnext_no_search/vnext_no_interaction` 均保持原启动映射。历史曲线不能仅靠更换显示名称归入新训练组。

## 持久化与恢复

| 数据层 | 固定名称或规则 |
|---|---|
| 权重与优化器 | 仍为 `_wm.*`、`_task_behavior.*`、`_goal_reference.*`、`_vnext.a.*`、`_vnext.b.*`、`_vnext_control.*`。优化器路径、`agent_state_dict`、`optims_state_dict`、`vnext_rng_state`、`rng_state` 和 `runtime_state` 沿用。关闭模块无对应参数和优化器。 |
| 实验身份 | 命名训练新增 checkpoint 顶层 `experiment_mode`、`experiment_modules` 和 `experiment_parameter_counts`。校验 config、顶层身份、模块参数前缀、优化器成员及旧协议，再严格加载。Full 和 no_fusion 权重形状相同也会拒绝跨模式恢复。 |
| 旧权重 | 原有模式按原协议恢复。显式 grace_full 可读取结构、融合和关键修订符合 R1 的旧完整 v2 权重；从旧权重开始的运行要按续训报告。命名消融拒绝以无相应训练身份的旧权重初始化。D v1 的拒绝策略保留。独立评测未点名模式时从存档恢复；显式点名须匹配。 |
| 协议 | Full 继承空间预测/记忆版本 2、局部地图 2、Pose 1、空间修订 3、执行器 1、A 标签 3、D 区域验证 2、目标控制 3、评测 7。无 B 时记忆、Pose 和地图模型版本为空；D-only 也不声明 A 模型/标签、候选或目标控制版本。执行器和评测协议仍可单独使用。 |
| Replay | 原 `log_vnext_*`、`vnext_memory_*`、`log_spatial_*`、`vnext_spatial_*`、`vnext_goal_*`、`log_goal_*`、`search_*`、`goal_*`、动作回执及窗口字段均保留原定义。新增 `log_grace_dac` 只标记无 B 的真实控制决策。无 B 的训练采样器不构造历史槽、局部地图和 Pose 输入。D-only 继续保存 Actor 来源与 D 日志。 |
| 日志与图片 | `A/*`、`B/*`、`C/*`、`D/*`、历史优化器打印名、`train_openl`、`eval_openl`、`eval_policy`、原回合 NPZ/JSON/图片名及想象缓存协议保留。新运行使用模式子目录。无 B 的运行省略 B 图；D-only 的回合图只输出有记录的 D 验证。冻结 C 评测的 `learned_selection`、`rule_probes` 仍独立于训练消融。 |

历史图可以用 `python scripts/render_figures.py <归档目录> <新输出目录> --format png` 重绘。训练恢复仍重建环境与 Replay 采样游标，旧实验应按原项目的恢复语义解读。

## 其他实际改动文件

- 训练与评测接线为 `expr.py`、`training_state.py`、`tools.py`、`models.py`、`test.py`、`evaluate_checkpoints.py`、`configs.yaml`、`scripts/train.sh`。模型属性保留历史拼写。`models.py` 只更新扩展模块引用。
- 环境与工具引用迁移涉及 `envs/tasks/base/ls_imagine_wrapper.py`、`envs/tasks/minedojo/__init__.py`、`envs/tasks/minedojo/wrappers.py`、`scripts/build_goal_references.py`、`scripts/test_b.sh`。原环境动作、奖励、图库字节保持原样。
- 新增 `grace/experiments.py` 和 `tests/test_grace_experiments.py`。其他更新导入/入口的测试为 `test_env_recovery`、`test_goal_guidance`、`test_goal_reference`、`test_goal_search_integration`、`test_paper_exports`、`test_search_a`、`test_search_b`、`test_search_visualization`、`test_spatial_evidence`、`test_spatial_integration`、`test_spatial_memory`（均在 tests/）。四个迁移测试见上表。
- 文档为 `README.md`、`docs/README_A_CN.md` 和本文。`.gitignore` 修正 IDE/gitignore 注释及桌面元数据规则。移除 Git 已跟踪的根目录 `.DS_Store`、`DS_Store`，以及 `assets/`、`envs/`、`envs/tasks/`、`envs/tasks/base/`、`envs/tasks/minedojo/` 中的 `.DS_Store`。

## 函数级索引

下表补充同名函数的接线变化。重命名且提供别名的接口见上方迁移表。模块迁移涉及的局部 import、自引用及错误信息也列入索引，便于比对基准源码。

| 文件 | 函数或方法 | 本轮涉及的内容 |
|---|---|---|
| expr.py | LS_Imagine.__init__、_policy、_grace_policy、_goal_diagnostics、observe_control_results、visualization_map、_goal_model_batch、_train_search_c、make_dataset、main、CLI 配置解析 | 实例与动作分派、D 日志抽取、无 B 采样、命名校验，原有方法迁移引用 |
| grace/outcome.py | PredictiveMemorySystem.__init__、memory_batch、online_memory、matched_option_windows、training_loss、_record、train_step，sample_predictive_episodes、check_prediction_checkpoint、restore_prediction_config | B 可省略，DAC 训练与留出掩码不依赖空间快照，模型协议校验 |
| grace/control.py | OptionController.initial、input、controls、choose、train_batch | 类型自引用迁移，DAC 训练剔除其他控制器轨迹。候选排序和规则计算沿用 R1 |
| grace/replay.py、experiments.py | sample_spatial_episodes、module_manifest、validate_training_mode、checkpoint_mode、parameter_counts | 可选记忆采样、实验身份和参数清点 |
| training_state.py | validate_goal_checkpoint、_validate_search_versions、_upgrade_b_optimizer、prepare_resume、save_training_checkpoint | 校验与兼容引用、旧模式续训身份、序列化 |
| test.py、evaluate_checkpoints.py | configure_evaluation、create_search_visualizer、configure_search_checkpoint、apply_a_control_mode、validate_reference、worker | 实验与环境身份区分，原评测入口迁移引用 |
| tools.py | simulate、_diagnostic_export、Logger.__init__、write、_write_backend、offline_scalar、offline_video、finish | 回执、DAC 采集标记、评测身份及输出错误隔离 |
| grace/visualization.py、scripts/render_figures.py | EpisodeVisualizer._open、observe、episode_end、_export、_render、_names，重绘 main | 跨版本归档，按有效模块输出，不给已关闭的模块绘图 |
| grace/spatial_memory.py、grace/evidence.py、envs/tasks/base/ls_imagine_wrapper.py | SpatialMapMemory.online、complete_episode_labels、LSImagineWrapper._obs | 局部 import 迁移 |
| grace/legacy/ | collect_real_branches、sample_episodes_with_history、SearchMemory.transition_loss、imagined_search_transitions、PrefixSearchController.choose | 历史调用所用 import 迁移 |

命名实验的 logger 对 JSONL、TensorBoard 或 W&B 输出故障按目的地报告步数和错误。有一条输出链故障时，仍向其他可用目的地写入，当前间隔的数据缓存随后释放。历史入口保留原错误处理。Replay 与 checkpoint 写入仍要求成功，磁盘已满时需释放空间并检查最后一个完整存档。vnext_visual_strict 在四个正式预设均为 False。显式开启 strict 用于测试导出错误。

## GRACE-1.0.0 本地验收记录（2026-11-24）

环境为 WSL Ubuntu-20.04、ls_imagine、RTX 2060 6 GiB。R1 代码独立从上述 Git 提交导出。采沙使用本地正式图库，12 张正例、8 张负例。

| 检查 | 本次实测 |
|---|---|
| 正式配置初始化 | 分别独立实例化 R1 和四种命名模型。R1 与 Full 均为 342 个参数张量、7 个优化器，初始参数、缓冲区（含名称与形状）和初始化后 CPU Torch RNG 摘要相同。no_fusion 的初始化也相同。 |
| CPU 逐步等价 | 在相同真实 Replay 上，对从头初始化和旧 D v2 存档恢复分别执行 7 次策略调用、5 次更新，并分别打开和关闭可视化。缩小模型含 267 个参数张量。R1 与 Full 每次调用的动作、候选和回执指令、隐状态、采样批次、网络输出记录、计时以外的指标、梯度、权重、缓冲区、优化器及 RNG 摘要一致。开启绘图会增加观察用的诊断输出。环境实际回执另外在真实短测保存。 |
| GPU 重复性 | 真实 Replay，7 次调用、5 次更新。4 次旧 R1 关闭绘图的重复运行，两两参数最大绝对差为 2.98e-8。R1 对 GRACE 最大为 5.96e-8，最大差发生在约 0.99916 的 float32 参数上，相差一个可表示间隔。最终参数相对 L2 差为 3.59e-9（同一基准的 R1 自身重复为 3.73e-9 至 3.83e-9）。动作、RNG、Replay 及模型模式签名完全一致。额外配对的显存已分配/峰值序列逐步相同，峰值 293193216 字节。 |
| 真实采沙训练及在线评测 | Full 完成 24 环境步和 2 局在线评测。三组消融各从头完成 96 环境步，在 0/48/96 步各评测 3 局，每局 32 步，共 27 局。均为缩小网络。D-only 的 288 次评测动作全部等于 Actor 提议。DAC 的 288 次决策中有 43 次最终命令与 Actor 不同，A 的未来预测梯度非零，C 有完整的优化器状态，B 无参数和优化器。no_fusion 的 288 个决策帧全部与时间对齐的原 U-Net 热图逐像素相同，七个优化器均有更新状态。 |
| 监督与融合的证据边界 | 三组消融的所有真实评测 NPZ 中，23 个可用的搜索/目标监督有效性字段均未启用。短测期间 D 尚未通过区域校准，实际融合门值均为零。另用同一已知合格区域输入验证开启融合能改变热图，关闭融合只清零 gate，区域、候选框和目标上下文一致。 |
| 恢复与旧数据 | 三组消融及新 Full 均通过两个评测配置入口，严格载入各自实训 checkpoint、优化器和真实 Replay，再做一次 D/世界模型更新、保存和读回。对每种新存档均拒绝其余三种训练身份。R1 的 104 步旧完整 checkpoint 也实测可按 Full 恢复、更新并保存新身份。 |
| 历史重绘 | 同一 R1 归档使用新、旧 CLI 导出的 PNG 字节一致。真实归档另成功导出 PDF、SVG。以带实验身份的 DAC 与 D-only 归档重绘，均省略 B 图，D-only 只输出 D 验证图。 |
| 完整回归 | bash scripts/test_all.sh 完成 18 套共 468 项测试及语法检查。覆盖 C 原有冻结消融、选项资格、历史 Replay、严格恢复、输出故障隔离。历史 19 组 YAML 配置的原有字段及值逐组相同。 |

本地检查记录位于 logs/grace_1_audit/，不随 Git 提交。test_all_release.log 记录完整回归，init_release.json 与 resume_release.json 可分别与 init_r1.json、resume_r1.json 按字节对照；formal_*、gpu_final.log、rollouts_final.log、resume_release.log、restore_grace_full.log 与 check_real.log 对应上表各项。games/ 中留有本次真实采沙短测的配置、checkpoint、NPZ 与图像。

这些检查支持冻结采沙实验的工程版本。正式尺寸只验证了初始化；训练与恢复的数值对照使用缩小模型。其他四个任务验证了图库读取和类别指纹，未逐任务进行本次真实联调。短测不能确定长期吞吐、场景难度、1M 校准质量或最终成功率。
