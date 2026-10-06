# T04 保存时磁盘写满：修复与服务器恢复

状态：本地代码已修改，只做静态阅读及差异核对，未运行任何测试、训练或评估；以下命令由用户在服务器执行。

本次失败的根因是 `torch.save` 写入临时快照时磁盘空间不足。后续 `error.txt`、`report.json` 和验收目录创建也失败；日志里的 `source_inputs_unchanged` 失败来自报告写入，不能据此认定源文件发生变化。新代码不会替用户清理已有服务器文件。

用户随后反馈已释放约 10 GB 空间：足够本次补跑和验收，可以从第二节存储检查开始，无需继续清理。

## 修复内容

- 新 `latest.pt / best_worker.pt / best_candidate.pt` 和验收快照不再重复内嵌冻结的 encoder/RSSM、原 actor 和目标库；引用现有 T04 缓存的 `frozen_bundle.pt`，并检查相对路径、文件 SHA256 和内容 ID。
- 每份快照仍保留两组新模块权重、两组优化器、计数、采样器及 RNG，按同一次更新完整恢复。旧内嵌格式仍可读取。迁移新快照时必须同时保留冻结依赖及相对目录关系，不能单独复制一个 `.pt` 就使用。
- 更新前检查三份快照与一次临时写入所需的预计空间，每次实际保存再检查，并保留 64 MiB 日志余量。估计不能阻止其他进程同时占用空间，但可提前发现空间明显不足。
- 保存失败清理本次创建的 `.tmp`，原文件只在新文件写完并同步后替换。已有不明来源的 `.tmp` 仍拒绝覆盖，不会自动删除。
- JSON 报告先写临时文件再替换；报告和错误日志写不进去时，错误仍输出到终端并返回失败，不递归报错，不误报源文件被改变。已有 `report.json` 可能是失败之前的状态，应以这次退出码和终端日志为准。
- T05 加载模式字段改为 T04 实际保存的 `experiment_mode`；支持新旧 T04 快照，仍严格检查模式和训练计数。

## 1. 先确认磁盘和 inode，释放空间

暂停重复启动失败的训练或 verify。先执行只读检查：

```bash
cd /root/rivermind-data/mine/projects/LS-Imagine-Ref

df -h . /tmp
df -i . /tmp
du -h --max-depth=1 relevance_map/t04_outputs
```

如需查看某次失败输出，使用日志中 `OUTPUT_DIR` 的**完整真实路径**运行 `du -ah /实际失败的输出目录`，确认大文件及残留 `.tmp`。本次错误没有给出失败训练目录的完整名称，因此这里不给自动删除命令。

请保留原 1M `latest.pt`、原 replay、已验收的 T00/T02/T03 产物、T04 `cache_20261006T124827`（包括 `frozen_bundle.pt`），以及已验收的 100/2000 次模型。确认失败进程已经结束后，可自行清理失败运行留下的未完成临时文件及确实不再需要的重复产物；不能把缓存依赖当作重复文件删除。

更换同一文件系统内的子目录不会增加可用容量。如果指定另一块盘上的 `--output-dir`，先用 `df -h /另一块盘的已存在父目录` 确认确实是不同文件系统且有空间。训练会输出实际空间估计；建议至少留出 **1 GiB** 用于这次补跑和后续验收，T05 视频还需要额外余量。旧的大快照不会被自动压缩或清除。

## 2. 拉取更新后，先跑 CPU 存储检查

```bash
python scripts/t04_storage_check.py
```

检查只创建很小的合成临时文件，用故障注入模拟空间不足，不填满磁盘、不读真实 checkpoint/回放、不启动环境、不需要 GPU。覆盖提前拒绝写入、PyTorch 部分写入失败、替换失败、已有临时文件保护、旧/共享格式读写、冻结依赖缺失/篡改、JSON 原报告保护及错误日志失败时的退出状态。

应看到 `Ran 8 tests`、`OK` 和 `[PASS] T04_STORAGE_CHECKS`。这个检查不证明真实恢复训练正确，后面仍需要 T04 verify。

## 3. 从已验收的 100 次模型补跑到 400 次

复用缓存，不重做 T02/T03/prepare；新增 300 次离线更新，环境步数仍为 0。以下明确从已验收的 100 次文件恢复，不使用这次写到一半的文件。失败目录中若有完整 `latest.pt`，可先另行 verify 再决定使用；不要把 `.tmp` 改名充当 checkpoint。

```bash
T04_CACHE="$PWD/relevance_map/t04_outputs/cache_20261006T124827"
T04_100="$PWD/relevance_map/t04_outputs/train_20261006T125314/latest.pt"
T04_EARLY="$PWD/relevance_map/t04_outputs/train_early_fixed_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py train \
  --cache-dir "$T04_CACHE" --resume "$T04_100" \
  --output-dir "$T04_EARLY" --device cuda:0 \
  --steps 400 --eval-every 100 --log-every 100 --save-every 100
```

如果要放到另一块盘，只修改 `T04_EARLY` 为该盘上的**新目录**。空间不足时应在任何新更新之前清楚提示可用与所需 MiB，并退出失败。

正常应出现 `[PASS] resume ... step=100`、`[PASS] storage_preflight`、最终 `[PASS] T04_BC_TRAIN`。查看实际大小：

```bash
ls -lh "$T04_EARLY/latest.pt" "$T04_EARLY/best_worker.pt" "$T04_EARLY/best_candidate.pt"
cat "$T04_EARLY/best_checkpoints.json"
```

快照不会包含重复冻结权重，实际大小以服务器结果为准；最佳 worker 的步数按留出误差选择，不强制为 400。

## 4. 验收实际用于 T05 的文件

```bash
T05_WORKER="$T04_EARLY/best_worker.pt"
T04_EARLY_VERIFY="$PWD/relevance_map/t04_outputs/verify_early_fixed_$(date +%Y%m%dT%H%M%S)"

python scripts/t04_goal_bc.py verify \
  --cache-dir "$T04_CACHE" --checkpoint "$T05_WORKER" \
  --output-dir "$T04_EARLY_VERIFY" --device cuda:0
```

验收也可选择另一块盘上的新输出目录。应通过新增 `[PASS] shared_frozen_bundle`，以及原有 `strict_load / roundtrip / next_update_equivalence / frozen_ownership / source_inputs_unchanged`，最终 `[PASS] T04_BC_VERIFY`。真实的恢复后下一次更新一致性，是判断新保存方式有没有影响训练状态的关键。

通过后按 [T05 验收说明](t05_goal_control_acceptance.md) 从第二步 `check` 继续；凡 `prepare/evaluate` 启动 MineDojo，仍显式使用 `MINEDOJO_HEADLESS=1`。本次修复不要求重跑 1M，也不代表目标控制效果已通过。

## 中文 Git 提交备注

```text
fix: 修复T04快照重复存储及磁盘写满时的连锁报错

- 新快照共享冻结模型并校验依赖，兼容旧格式且保留完整训练恢复状态
- 增加训练前与保存前空间检查，写入失败清理自有临时文件并保护原快照
- 原子写入JSON报告，磁盘不足时保留终端错误并避免误报源文件改变
- 修正T05读取T04模式字段，补充CPU故障注入检查及服务器恢复说明
```

以上仅为提交备注文本，助手未执行 Git 提交或推送。
