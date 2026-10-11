# 本地 esaiko-nlh 分支 Codex 工作提示词

以下正文可直接复制到新的 Codex 环境。本版更新本地运行授权，取代之前“只静态检查、由用户执行全部运行”的分工。

---

请在我的本地 LS-Imagine 项目直接开展工作，先完成硬件与模型运行测试，测试成功后继续实施改进，不要停在建议或反复询问是否继续。

## 工作目录、分支与授权

Windows 路径：
\\wsl.localhost\Ubuntu-20.04\home\user1\dl\projects\LS-Imagine-Ref

对应 WSL Linux 路径：
/home/user1/dl/projects/LS-Imagine-Ref

我已切换到干净分支 esaiko-nlh。开始时读取 AGENTS.md、检查实际分支及工作区，确认源码与官方 LS-Imagine 的差异。保留现有改动，不自行切换分支、reset、覆盖或修改 .idea/。

代码和文档由你在这个本地目录修改；Git 提交、推送及服务器同步由我负责。不要自行 commit/push，也不要连接服务器修改代码。

这次我明确允许你直接在本地运行项目脚本、数据检查、测试、模型加载、前向和验证改进所必需的小规模实验。WSL Ubuntu-20.04 中的 conda 环境已确定为 ls_imagine。所有项目运行、模型测试与实验都使用这个环境，在 WSL 项目目录下执行；可使用 conda run -n ls_imagine python ...，或激活该环境后执行。先核对实际解释器、PyTorch、CUDA 和依赖版本，不自行换用其他环境或 Windows 全局 Python。若 conda 未进入 PATH，查找现有 conda 安装，不擅自重建环境。此前禁止本地运行的规则不再适用于本轮。

我的 GPU 是 RTX 2060 6GB。原版 harvest_log_in_plains 已训练的 latest.pt 已保存到本地，固定使用以下文件：

Windows 路径：
\\wsl.localhost\Ubuntu-20.04\home\user1\dl\projects\LS-Imagine-Ref\ckpt\minedojo_harvest_log_in_plains\seed_0\20260517T103120\latest.pt

WSL Linux 路径：
/home/user1/dl/projects/LS-Imagine-Ref/ckpt/minedojo_harvest_log_in_plains/seed_0/20260517T103120/latest.pt

我已补充本轮需要的文件；先清点本地实际内容，不重复要求下载已有文件。该 checkpoint 文件约 2.84 GiB；文件大小不能直接当作推理显存需求。先检查真实保存格式、原训练配置与依赖，再实际测试。源 latest.pt 只读，测试产物写入 research/goal_horizon/outputs/ 的独立运行目录，新权重写入 research/goal_horizon/checkpoints/。如果依赖缺失，继续完成不依赖它的工具，并一次性列出具体缺项。不要把随机初始化或部分加载当作模型恢复成功。

## 本轮统一目录：Goal Horizon

本轮研究名为 Goal Horizon（目标与长程规划）。除项目根目录的 .gitignore 外，本轮新增的代码、配置、测试代码、文档和运行产物统一放在项目内 research/goal_horizon/，不分散到根目录的 scripts/、docs/ 或其他历史研究目录。

固定研究根目录：
/home/user1/dl/projects/LS-Imagine-Ref/research/goal_horizon

目录分工：
- src/：新增研究模块、适配器、局部控制和规划实现。
- scripts/：模型检查、数据检查、训练、评价及结果整理入口。
- configs/：可复现的静态实验配置与开关。
- tests/：可提交的测试代码；运行生成的数据另写 outputs/，不写入测试源码目录。
- docs/：提示词、clean_todo.md、experiment_log.md、方案和接口说明；这些文档跟随代码提交。
- outputs/：所有测试、训练与评价的运行产物。每次运行写 outputs/<阶段>/<时间戳或唯一ID>/，含实际报告、CSV、图像、视频、终端日志、配置快照、错误信息与反馈 ZIP；拒绝覆盖已有运行。
- data/：确实需要新建的抽样数据或派生数据；cache/：编码缓存和临时缓存。
- checkpoints/：本轮新增模型权重；logs/：独立工具日志；artifacts/：其他运行生成的打包产物。

原版权重、补充的真实回放和外部 CPP 论文保持原路径，通过配置引用，不复制全部模型或数据到研究目录。优先通过本目录的适配器、封装和配置调用原版模块，保持原版源码与运行入口作为基线；不要把整套原仓库复制进研究目录。

在研究入口统一解析研究根目录和输出路径。默认输出、缓存、临时文件、新权重、pytest 临时目录及覆盖率产物都写在上述研究目录内；不能因当前工作目录变化而落到项目根目录。新 Python 模块的 __pycache__ 可保留在模块旁，但必须被 Git 忽略。

Git 跟踪 src/、scripts/、configs/、tests/、docs/ 和 README.md；Git 忽略 outputs/、data/、cache/、checkpoints/、logs/、artifacts/、运行缓存与二进制模型/回放。实际测试报告和配置快照也是运行产物，放 outputs/ 并忽略；人为维护的方案文档和静态配置仍须可提交。不要全局忽略 *.json、*.yaml 或 *.md，以免把源码配置与文档一起屏蔽。

维护项目根 .gitignore 的研究专用规则，保留已有忽略内容。提交说明前用 git check-ignore 核对代表性的报告、回放、权重与缓存会被忽略，同时用 git status 确认代码、配置、测试脚本和 docs 仍可跟踪；检查已跟踪产物，不能假定加规则就能取消已有跟踪，不自行提交或重写 Git 历史。

## 最小文件清单：不要求复制整个训练目录

第一阶段模型加载与显存测试使用上述 latest.pt、匹配的配置，以及原版外部权重。本地已发现以下文件，开始时核对实际可读性及任务匹配：

- MineCLIP：weights/mineclip_attn.pth。
- harvest_log_in_plains U-Net：affordance_map/finetune_unet/finetune_checkpoints/harvest_log_in_plains/swin_unet_checkpoint.pth。
- 项目配置：configs.yaml、任务配置及模型配置；同时核对 U-Net/MineCLIP 加载代码是否声明其他依赖。原训练若有自定义参数，需要原配置或原启动命令，不能只按当前默认配置猜测。

仅加载与合成输入 smoke test 不需要 train_eps、eval_eps、check_obs 或 TensorBoard events 文件。请实现独立离线推理入口，避免为恢复模型调用会初始化 MineDojo 或进入训练流程的主入口。按原保存代码核对 agent_state_dict 与 optims_state_dict 等实际键，推理不恢复无关优化器到 GPU。

真实输入和历史验证需要少量完整 episode，checkpoint 不能替代这些数据。先检查本地已有文件，输出最小清单；不要要求我传十几 GB 的全部回放，也不要让我逐个重命名文件。起步可用 eval_eps 中 2–3 条完整 episode 做评价、train_eps 中 3–5 条完整 episode 做数据接口及小规模训练验证。数量只是起步建议，实际按序列长度、字段和显存决定；不能据此声称泛化或成功率改善。固定文件清单和选择方式，保留原文件名、完整历史与所属划分，不按表现挑选主评价样本。

- metrics.jsonl 建议一并保留，便于了解原训练与评估趋势；它不是恢复模型的必需输入。
- 原训练配置、启动命令及代码版本若可取得，作为小体积来源信息保留。
- events.out.tfevents.* 暂不需要，只在必须排查 TensorBoard 历史时再取。
- check_obs 暂不需要；若用于辅助查看，只取少量样例。静态图像不能替代含真实动作与 reset/terminal 的完整 episode。
- 保持 eval_eps 仅用于评价；新增训练用 train_eps，并先按完整 episode 固定开发留出。缺数据时先完成模型测试和采样工具，再一次性说明需要哪些文件。

后续正式训练或成功率比较所需数据另按预声明预算获取；小样本验证通过后再扩大，不把最小测试数据当作完整研究证据。

## 目标和范围

基于官方原版 LS-Imagine 与 harvest_log_in_plains 真实数据，研究可执行子目标、真实跨步动力学和短时闭环能否提高长程控制的成功率与样本效率。

首轮保持原任务热力图、MineCLIP 奖励、U-Net、RSSM 和训练主干，暂缓通用热力图、NACLIP、P85 和大规模架构迁移。本轮不安排剪羊毛任务或提示词排查。

官方仓库：https://github.com/qiwang067/LS-Imagine
原论文：https://arxiv.org/html/2410.03618v2

研究参考（含追加论文）：
- CPP（用户提供的 NeurIPS 2025 参考论文）：Windows 文件 D:/CODE/ls/research_20261006/CPP_NeurIPS2025.pdf；在 WSL 中对应 /mnt/d/CODE/ls/research_20261006/CPP_NeurIPS2025.pdf。把这篇论文加入研究与方案比较清单，方案阶段根据原文核对其具体方法、实验和适用前提，再判断哪些机制值得做最小实验。不要根据 CPP 缩写或文件名猜测论文内容，也不要未经原版基线验证就照搬整套方法。
- SAGE：https://arxiv.org/html/2607.17973v2 —— 子目标指导动作候选与模型评价。
- HWM：https://arxiv.org/abs/2604.03208 —— 真实动作片段与多时间尺度预测。
- H-JEPA：https://arxiv.org/html/2610.06805v1 —— 动作敏感性与表征退化诊断。
- TTGS：https://ktolnos.github.io/ttgs/ —— 简单子目标检索/搜索对照。
- OGBench：https://github.com/seohongpark/ogbench —— GCBC/GCIQL 等局部控制参考。

读取必要原文与代码，选最小适用机制，不把全部论文模块堆入项目。原版回放是事实数据，不自动成为专家数据。

## 第一步：实际验证 6GB 显卡能否运行

请在 research/goal_horizon/scripts/ 实现并亲自执行可重复的本地测试入口，将 local_runtime_report.json、Markdown 报告及其他运行产物写入 research/goal_horizon/outputs/runtime/<唯一运行ID>/，这些报告不提交到 Git。初始只做离线推理，不启动大规模训练或 MineDojo。

1. 检查 GPU 实际型号、可用显存、WSL CUDA 可见性、Python/PyTorch/CUDA 版本及依赖，区分缺文件、依赖错误、权重不兼容、CUDA 错误和显存不足。
2. 查明原版 U-Net、MineCLIP 权重与 latest.pt 的实际保存格式、配置、尺寸和推理依赖。latest.pt 不一定包含外部 U-Net/MineCLIP 权重，需要明确分别检查。
3. checkpoint 先加载到 CPU 检查，再构造推理所需模型。推理使用 eval 和 inference_mode，不恢复不需要的优化器到 GPU。严格核对权重键、形状和配置，不以 strict=False 掩盖不兼容。
4. 分别测试 U-Net 热力图前向，以及 latest.pt 所需 encoder/RSSM/actor 前向。优先 batch=1、原版输入尺寸和原版精度；核对输出形状、有限数值、动作空间及 reset/真实 incoming action 接口。
5. 单模块通过后，测试实际处理链路：RGB 与所需文本/历史 -> 原版热力图/奖励处理 -> encoder/RSSM -> actor。联合驻留和真实流程的显存峰值要测量，不能以“两个模型单独能跑”代替整体可运行。
6. 记录实际 allocated/reserved 显存峰值、GPU 可用内存、耗时和多次推理结果。最终区分：原版联合推理通过、仅单模块通过、分阶段加载通过、CPU 回退通过、失败/未测试。
7. 出现 OOM 时可以尝试小批量、分块或分阶段加载、CPU 回退、受当前 GPU/软件支持的混合精度。每个替代方案单独记录；精度变化需与原版输出和动作选择比较。不要默默降低输入分辨率或修改模型结构来宣布原版通过。
8. 有真实 replay/RGB 时做真实输入与完整历史测试；只有合成输入时明确标记 smoke test，不能声称真实数据/任务效果已验证。latest.pt 本身不能替代 replay；尚无 replay 时先完成模型测试和兼容工具。
9. 推理通过不等于训练能跑。进入新模块前，对新增小头实际做最小 batch 的前向/反向/更新与显存测试，保留原 checkpoint 不变。完整规模训练和环境评估的成本另行报告，超出本机能力的部分提供服务器命令由我运行。

不要凭显卡型号猜测结论。若 6GB 足以支持实际离线流程，直接继续下一阶段；若只支持部分流程，继续实现能在本地验证的部分，并给出明确的运行限制和服务器执行方式。

## 干净 Todo 与后续实施

维护 research/goal_horizon/docs/clean_todo.md 和 research/goal_horizon/docs/experiment_log.md，分别记录“本地实现”“本地实际验证”“服务器待验证”。文档引用 outputs/ 中的实际产物，不把数据、图表、日志或权重复制进 docs。保留官方原版路径；改进代码、开关和静态配置都在本轮研究目录，新的 checkpoint/output 按统一目录规则保存，不覆盖源 latest.pt。

C00：本地环境、U-Net、latest.pt 单模块及联合推理测试，输出实际显存和能力结论。

C01：原版数据清点与基线。检查完整 episode、真实动作、奖励、reset/terminal/truncation 和来源。确认状态/动作索引、原版表示与缓存身份，固定训练/开发/测试划分；复现可在本地完成的原策略和预测基线。

C02：局部目标条件控制。使用真实短片段与事后目标，GCBC 为最低对照；数据覆盖充分再考虑优势加权/目标条件 RL。新低层显式输入目标与剩余预算，保留原 actor，先冻结表示/WM。测试 goal/no-goal/shuffled-goal，离线拟合与真实可执行性分别报告。

C03：真实跨步转移。用真实动作片段监督终点、时长、折扣累计回报及终止，比较复制当前状态、迭代原短程和原 jump。原回放无 issued goal 时先学动作片段条件模型，不能把 achieved goal 冒充 commanded goal；目标条件宏模型使用新 worker 的真实执行日志。

C04：最小长短程闭环。原热图提名子目标，可达性/任务价值筛选，少量候选和 1–2 个宏段，短时执行后重规划。不可信时回退原 actor并报告比例。分别消融子目标、宏预测与门控。

C05：真实任务验证。以 harvest_log_in_plains 为首个任务，固定预算、多种子比较原版、局部控制无新宏模型、简单规划、完整方法。小规模本地测试不足以证明任务成功率改善，完整实验由合适硬件完成。

C06：核心方案有效后，才讨论通用热力图、NACLIP、P85 和更多任务。

完成 C00 后直接开展 C01/C02 的代码实现与可运行本地验证，不再问我“是否继续”。后续按必要前提推进；局部控制不可靠时先修复，不用增加高层模型掩盖问题。

## 实验约束与反馈

- 先按完整 episode/世界划分，再切片；同源视频、相邻帧、zoom 不跨划分，测试不参与超参/阈值选择或候选库。
- 先核对原版 replay 动作存储约定，再定义状态到实际下一动作的索引，不盲用旧 T05 schema。
- zoom 只提名视觉候选，不能当真实宏终点；未来帧可作离线事后目标，在线部署不能读本局未来。
- 不要求我额外手工标注每任务轨迹。统计独立 episode 和关键动作覆盖；重复采同一条记录不增加独立证据。
- 编码器、预处理、热图与 checkpoint 身份绑定；RSSM 输入本局真实动作，reset 清历史，真实终止后停止查询。
- 核对 env.step/action-repeat 的单位，正确使用 gamma^duration 和实际累计奖励。
- NLL、预测 loss、热图和距离都是辅助指标，任务成功率与样本效率是主证据。不能以工程 PASS 或本地加载成功宣布控制有效或 T06 通过。
- 预算计入初始化模型/原数据、新环境交互、离线更新、wall-clock、显存和推理时延。失败如实保留，不筛起点、不补样到通过、不选中间最佳帧。
- 结果自动整合为主 JSON、Markdown 和一个 ZIP，统一保存到 research/goal_horizon/outputs/<阶段>/<唯一运行ID>/，含配置快照、模型/代码身份、指标、CSV、少量图与完整视频索引；不打包模型和全量 replay。失败也保存已完成结果，输出本地绝对路径。这些实际测试数据和报告不进入 Git。
- 需要启动 MineDojo 时使用 MINEDOJO_HEADLESS=1；长时间或完整规模运行提供清楚的预算与运行位置，不能把硬件限制当作算法失效。

每次改进交付都说明：改了什么、为什么、实际本地运行了哪些检查、结果和未验证项、后续命令与产物，并给一句可直接复制的中文 Git 提交备注。

Git 提交备注沿用以下格式：feat: 增加……；fix: 修复……；docs: 整理……。不使用 feat(T05): 等格式。给出备注不等于你执行提交。

现在直接从本地分支与权重/环境检查开始。模型文件到位后亲自运行测试，成功后继续改进；只有缺少必须的信息或遇到不能独立解决的阻塞时才询问我。
