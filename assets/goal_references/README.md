# vNext4 固定目标参考库

`harvest_sand/` 是采沙任务的默认库。包含 24 张经中心射线和人工复核的沙子正图，覆盖 3 个训练世界组，并含 9 张人工复核的天空、水面难例。来源是 vNext 旧空间版和 fix2 的训练回放。manifest 记录每张图的世界组、源 NPZ 文件名与 SHA256、帧号、ROI、角色和图片 SHA256。

新训练按任务目录读取 `manifest.json` 和 PNG。一次实验从训练到评测使用同一参考库。完整 vNext4 检查点内嵌参考图和清单，因此跨机器测评或续训时可直接使用检查点内的库。

原木、取水、铁矿和羊目前需要各自的训练源。可用 `scripts/build_goal_references.py --collect-world-seeds ...` 启动短程真实环境采集，生成带证据的训练 NPZ。再用同一脚本的 `--train-replay ... --output ...` 模式建库。采集模式支持 `--capture-commands`，样本不足时可更换训练世界、增加视角或手工复核裁剪。只将最终 PNG 和 manifest 作为项目资产。新的库可以先输出到独立目录，再通过 `--goal_ref_bank_root` 显式选择。

建库和采集命令见 [vNext4 运行说明](../../docs/README_A_CN.md#27-apm-vnext4-可靠目标验证与主动空间搜索2026-11-07)。
