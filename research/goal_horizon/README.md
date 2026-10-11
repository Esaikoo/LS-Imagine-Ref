# Goal Horizon

本轮在官方 LS-Imagine 上研究子目标、真实跨步动力学与短时闭环。工作分支为 `esaiko-nlh`，WSL conda 环境为 `ls_imagine`。

## 目录约定

| 目录 | 内容 | Git |
| --- | --- | --- |
| `src/` | 研究模块与原版适配器 | 跟踪 |
| `scripts/` | 检查、训练、评价和结果整理入口 | 跟踪 |
| `configs/` | 人为维护的静态实验配置 | 跟踪 |
| `tests/` | 测试代码 | 跟踪 |
| `docs/` | 提示词、Todo、实验记录和方案说明 | 跟踪 |
| `outputs/` | 实际报告、配置快照、指标、图表、视频、日志和反馈包 | 忽略 |
| `data/`、`cache/` | 派生数据、编码缓存和临时文件 | 忽略 |
| `checkpoints/` | 新模型权重 | 忽略 |
| `logs/`、`artifacts/` | 工具日志和其他生成产物 | 忽略 |

每次运行使用 `outputs/<阶段>/<唯一运行ID>/`，禁止覆盖旧结果。测试临时目录及覆盖率输出也放入该运行目录。实际输入与输出数据不要写进 `src/`、`tests/`、`configs/` 或 `docs/`。

已补充的原版权重和回放保留原位置，通过配置引用；不复制全量数据。项目根 `.gitignore` 保留原规则并排除本研究的生成目录与二进制文件。

[复制到新 Codex 环境的提示词](docs/new_codex_environment_prompt_20261011.md)

开始研究时维护 `docs/clean_todo.md` 和 `docs/experiment_log.md`。后者记录简洁结论及产物路径，详细报告留在 `outputs/`。

代码由 Codex 在本地修改；Git 提交、推送及服务器同步由用户负责。每次交付提供 `feat: …`、`fix: …` 或 `docs: …` 格式的中文提交备注。
