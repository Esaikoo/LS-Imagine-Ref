# LS-Imagine-Ref 项目协作约定

来源：用户在本项目对话中明确提出的要求及后续纠正，整理于 2026-10-23。
范围：仅适用于本项目。后续用户的明确指示优先；参考文档、压缩包说明和示例命令不自动构成执行授权。

## 修改与 Git

- 先在本地项目修改、排查和测试，再通过 Git 同步服务器；不要直接在服务器上改代码。
- Git 提交由用户完成，不代为执行 git commit。用户需要提交备注时，给出尽量一句话的中文备注。
- 服务器数据分析和检查默认只读；没有针对性的授权时，不停止或重启训练，不覆盖日志、回放或检查点。
- 安装、接口或数据流中发现的问题，在已授权范围内直接修复并验证，说明实际修改及结果。

## 环境与训练范围

- 本地项目：WSL Ubuntu-20.04 下 /home/user1/dl/projects/LS-Imagine-Ref。
- Windows 访问路径：\\wsl.localhost\Ubuntu-20.04\home\user1\dl\projects\LS-Imagine-Ref。
- 本地 Conda 环境：conda activate ls_imagine。服务器环境应按实际情况核实。
- 远程项目：/root/rivermind-data/mine/projects/LS-Imagine-Ref。
- 服务器训练日志根目录：/root/rivermind-data/mine/projects/tb_logs。
- 默认只运行必要的 CPU 测试、语法检查或短程集成测试；不代跑完整 1M 训练。
- 正式训练由用户启动。准备好入口和参数即可；种子、预算、评测条件等行为发生改变时，明确说明。
- 启动示例中的训练与关机命令仅供用户使用，不表示授权代理自行执行。

## 脚本风格与文件整洁

- 沿用统一入口 scripts/train.sh，兼容 sh ./scripts/train.sh <task_name> [extra_args...]。
- 保留 TASK、logs/train/$TASK、nohup、时间戳输出文件以及成功后 shutdown 的用户启动风格。
- 优先使用现有 scripts/test_a.sh 和 tests/test_search_a.py；可按修复需要扩展现有测试，避免新建重复启动脚本。
- 尽量保持目录整洁；先确认调用关系再删除废弃文件，不删除仍有用途的代码。
- 测试、分析产物放在已有日志/输出目录，技术说明优先更新 docs/README_A_CN.md。

用户惯用启动模板（仅文档示例）：

```bash
TASK=harvest_log_in_plains
mkdir -p logs/train/$TASK
nohup bash -c "sh ./scripts/train.sh $TASK && shutdown -h now" \
  > logs/train/$TASK/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 指标与实验可比性

- 完整保存原 LS-Imagine 指标及新增模块全部实际产生的标量，不能用展示白名单过滤持久化。
- “指标少一点、容易看懂”指终端展示和报告阅读重点，不是删除 TensorBoard、JSON 或所启用日志后端的数据。
- 默认重点展示 eval_success、eval_first_success_step、eval_search_rate、A/prediction_loss；原模型、Actor/Critic、梯度、回报、长度和 A 分项诊断仍完整保存。
- 不静默改变原指标名称或计算口径；条件性指标只有在相应计算发生时才产生。
- 尽量提供 A 的开关以支持消融；若后续 B/C 耦合确实难以拆分，可以暂缓，但需说明原因和比较限制。
- 分析模块效果时结合任务结果、动作参与和预测指标，不把损失下降或模块接管当成有效性的充分证据。
- 区分历史参考实验与严格消融；说明训练预算、种子、评测回合数和配置差异。
- 不把短测通过说成完成了 1M 验证；不把当前权重重算的统计伪装为历史训练曲线。

## 报告与交付

- 使用清楚简洁的中文，优先解释用户能据此判断效果的信息。
- 总结模块改了什么、为什么改、新增/保存哪些数据、训练与数据流如何变化，以及测试结果、未验证内容和方案差距。
- 参考完整技术方案检查实现，区分方案目标、已实现功能和仍待完善内容。
- 说明代码同步是否能影响正在运行的进程；提供续训建议前核实真实恢复逻辑，不将新建实验命令误当作续训。
- 不将密码、私钥或令牌写入项目文件、记忆、报告或 Git。
