# T05结果汇总与自动接收

最新本地交付：独立原预算末4步固定修复check/train/verify已复用原汇总器，自动包含本轮、源监督check、反馈evaluate/check、诊断、原350确认和原350verify共7阶段；train再包含新check为8阶段，verify再包含新train为9阶段。原字节和SHA保留，不复制模型、NPZ轨迹或视频、不需逐文件重命名；实际新修复和打包待用户服务器运行。反馈最后`FEEDBACK_INDEX`即可继续只读接收，代码仍由用户Git同步，助手本轮没有修改或连接服务器。见 [三步运行说明](t05_within_horizon_repair.md)。

2026-10-09最新接收：原预算内事实监督check已经用户服务器完整PASS/WARN，自动汇总6阶段68文件（本轮、双目标反馈evaluate/check、同状态诊断、原350确认、repair verify）。已通过现有T2-3090授权只读保存到 `C:\Users\28620\Downloads\T05Results\received_20261009T193008_1714413\T05_within_horizon_supervision_check_20261009T112719_064845_results.json`，25321515字节，SHA256为 `9c00c7f1bd2017a266da47a3c7910a2756834fc0d60ae9ec43d3c19308954793`。整包/68文件SHA、原5阶段60文件未变、新6产物SHA、140原产物引用、80查询/480分布的3440标量及全部分组核对通过，无需逐文件下载或改名。见 [最新分析](t05_within_horizon_supervision_result_analysis.md)。只读取结果，没有上传/修改服务器代码、执行服务器项目脚本或下载模型/轨迹/视频；代码仍只经用户Git同步。

2026-10-09。用户确认服务器为T2-3090，已用其现有SSH配置完成一次真实结果接收。没有更改认证/安全设置、创建密钥或保存密码。只复制报告、表格、汇总图，不递归采集模型、checkpoint、缓存、逐局trajectory或视频。

## 已保存本轮结果

本机文件：

```text
C:\Users\28620\Downloads\T05Results\received_20261009T114129_7744345\T05_continuation_control_evaluate_20261008T165222_results.json
```

包含5个关联结果阶段、46个文件，原始字节13379157，反馈JSON约14MB。整包SHA256为`4ec8514d16f2e54afe84c1be3bc1cd4d9b963a7e246f90066c8c604fa57f9c0a`，接收后已逐一验证46个内嵌文件的SHA256和长度。报告的行为FAIL/工程PASS状态原样保留，不由打包重新判定。

本机接收工具也保存在`C:\Users\28620\Downloads\T05Results\tools\receive_t05_results.ps1`。仓库源文件在`tools/receive_t05_results.ps1`；从本地盘运行，避免UNC路径被Windows脚本策略视为远程脚本。不修改执行策略。

## 以后如何使用

新版同状态诊断在最终报告保存后自动生成结果JSON、ZIP及传输索引，输出`FEEDBACK_INDEX`。用户只发这一行或`OUTPUT_DIR`，Codex即可按现有授权从T2接收并读取结果。没有自动运行训练/环境，没有设置定时任务；接收发生在用户告知结果已就绪之后。密码不写入代码、配置或结果文件；后续会话若现有登录不可用，需要恢复登录或采用单文件下载。

旧入口不改已验收指纹。它们的结果也可通过独立标准库工具汇总，只需指定一个运行目录，自动发现其关联check/verify/train报告：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref
python scripts/t05_result_bundle.py --run-dir "$OUTPUT_DIR"
```

最多追溯两层关联报告目录，可用`--related-depth 0..3`明确调整；未找到的关联报告明确列在`missing_related`，不隐藏。也可多次传`--input check=/绝对目录 --input evaluate=/绝对目录`，各阶段同名文件无需重命名。输出为新独立`relevance_map/t05_feedback/<运行名>_<时间戳>/`，不在旧运行目录里补文件。

需要自行接收时，在Windows本机终端运行以下命令，将服务器输出的索引路径替换进去：

```powershell
& "$env:USERPROFILE\Downloads\T05Results\tools\receive_t05_results.ps1" `
  -Server T2-3090 -RemoteIndex '/服务器汇总目录/bundle_index.json'
```

工具复用SSH/SCP现有登录，不保存凭据；没有免密登录时会在本机终端要求输入密码，不要把密码写到命令参数。保存目录默认为`下载/T05Results/received_<时间戳>/`。失败传输只保留`.partial`，校验通过才改为正式文件，不覆盖以前的反馈。

## 单文件格式与边界

`ls_imagine_result_bundle_v1`把每个原文件记录为`source/name/bytes/sha256/encoding/content`。JSON、CSV、JSONL、error.txt保留精确UTF-8文本；PNG存base64。`source_inventory`提供每阶段原目录、command、工程状态、数值门槛和T06状态。读取时解码`content`即可恢复原字节，并核对SHA256；不能对JSON重新排版后再比较原字节hash。

ZIP包含同一汇总JSON及`sources/<阶段>/<原文件名>`。两种形式都不把多帧或多个文件当独立实验，也不把失败改成通过。只需JSON即可在对话中反馈全部内容；ZIP方便人类解压查看。

选择范围仅为运行目录顶层JSON/CSV/JSONL/PNG和error.txt，拒绝符号链接。验证原manifest已绑定的汇总SHA、采集前后输入文件大小/mtime/hash；新诊断manifest的输出SHA也核对。每文件最多32MiB、原始总量最多64MiB，超限明确失败，不悄悄裁剪；ZIP完成后逐条读回验证。

本轮已真实完成服务器打包、下载及46文件字节校验；新诊断的自动汇总调用还需随首次服务器诊断运行验收。不要修改旧报告、代码指纹或训练预算来适配打包。


后续首次同状态诊断虽因历史格式检查失败，最终报告仍自动打包成功；已接收4阶段31文件到`Downloads/T05Results/received_20261009T121949_5709748/`，整包和全部内嵌文件校验通过。该反馈流程已真实覆盖失败报告，诊断FAIL原样保留。代码修改只在本地，通过Git同步服务器；SSH仅用于读取/接收结果，不直接上传或修改项目代码。


最新成功诊断也已自动接收：`continuation_action_diagnose_20261009T045733_081902`的4阶段38文件保存到`Downloads/T05Results/received_20261009T130051_2757614/`，汇总19104555字节，整包SHA为`d8b9791c5132b8dcd04bdcadd5f3e1d263ddc2d093adc368db561152a15ac652`。所有内嵌字节校验通过，概率图按原字节解码用于分析。用户仅反馈终端目录/索引即可完成整包接收，无需逐个下载重命名。

随后已本地实现原16步内双目标纠偏/保持采集，见 [两步运行说明](t05_within_horizon_feedback.md)。新check/evaluate在最终报告保存后复用原汇总器，显式包含本轮、check（evaluate时）、成功诊断、原350确认和repair verify，仍保持单文件反馈及原SHA，不修改已验收的汇总器/接收脚本。新入口实际打包待用户服务器运行；本次未连接/修改服务器，未执行本地Python/模型/环境或提交推送，代码继续通过Git同步。

2026-10-09该新入口已真实运行并成功打包。`within_horizon_feedback_evaluate_seed_fixed_20261009T084226` 的5阶段60文件已自动接收至 `C:\Users\28620\Downloads\T05Results\received_20261009T182334_5798448\T05_within_horizon_feedback_evaluate_seed_fixed_20261009T084226_results.json`，22967169字节，整包SHA256为 `4dd4e266eb3a613b919cc1e31556fbfb9d57a934f94e94578d0b3f637349c353`。整包与全部原文件字节/SHA通过，20局/340帧/320查询/80事实动作和140个原产物SHA引用独立核对；10张PNG按原字节解码查看，见 [分析](t05_within_horizon_feedback_result_analysis.md)。只读SSH接收无需逐文件下载或改名；没有上传/修改服务器代码、下载模型/轨迹/视频、执行服务器项目脚本或本地项目Python/测试/模型/环境。旧行为失败及T06未批准原样保留。
