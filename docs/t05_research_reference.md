# T05问题与可复用研究参考

2026-10-07查询论文原文、作者项目页与官方仓库。以下是方法/实现参考，不是已经在本项目跑通的替换方案。

## 为什么T05多次迭代

T00–T04的加载、数据对齐、因果状态、训练/恢复工程检查已通过。T05承担的是实际低层目标控制验收；工程PASS与行为有效是两个结果。

先前克隆actor的目标路径弱，独立无目标对照收益接近零。短期采样均衡未解决；归一化小头证实有限的目标动作信息，随后冻结BC底座的残差worker在全留出上取得额外目标NLL收益0.021749。后续重点应是实际控制，而不是继续延长BC或堆探针。

真实评估又建立在“相同世界/动作可以重现足够相同的历史”上。固定世界种子、正确截图/HUD、预热改善了部分问题，但最近65帧原生动作全部一致，位置仍相差0.858020；两次预检通过不足以保证下一次重放稳定。环境完整状态未记录，实体接触/模拟时序只能列为可能原因，尚不能确定。

另有磁盘存储、HUD入口遗漏、实时NumPy事件比较等可避免的工程问题，不能全归因于环境；过度依赖逐历史配对门槛也造成不必要迭代。现在保持模型冻结，采用独立起点随机条件试跑，参见 [两步运行说明](t05_random_control_acceptance.md)。这不是论文算法复现，也不将旧FAIL转为PASS。

## 最相关的近期工作

**Instruct-to-Act / Decoupling Planning and Control for Instructable Agents，COLM 2026**。高层VLM给稀疏语言指令，Dreamer式RSSM控制器快速执行；对自己的策略片段事后标注指令，联合BC、奖励优化和WM学习。论文/代码与当前路线的相近之处是“已有控制行为→条件监督→持续任务学习”，可用于后续T09联合训练设计；不会仅靠给actor拼一个目标就保证控制。来源：[论文](https://arxiv.org/abs/2608.26788)、[方法正文](https://arxiv.org/html/2608.26788v1)、[官方代码](https://github.com/zinengtang/instruct-to-act-code)。

它提供完整项目，但README要求Python3.11、JAX、官方DreamerV3和MineRL0.4.4，本项目是PyTorch/Python3.9/MineDojo；指令编码/完成头、模型结构、动作和checkpoint格式也需要适配。可独立复现或移植训练机制，不能直接加载现有latest.pt。当前不安装、不切换框架、不引入LLM调用。兼容性判断依据：[官方README](https://raw.githubusercontent.com/zinengtang/instruct-to-act-code/master/README.md)。

**R2-Dreamer，ICLR 2026**。用冗余约减的自监督目标训练不含图像decoder的世界模型；作者提供PyTorch DreamerV3及R2实现。可参考表示学习或实现效率，未直接解决目标条件低层控制，也不解决Minecraft重放位置漂移。仓库主要列DMC、Meta-World、Atari、Crafter等，测试环境Python3.11/Ubuntu24.04；替换表示目标需要重新训练WM/缓存，当前不做。来源：[论文](https://arxiv.org/abs/2603.18202)、[官方代码](https://github.com/NM512/r2dreamer)。

**Director，NeurIPS 2022，年代较早但结构更贴近层级Dreamer**。manager选潜在子目标，worker通过低层动作达到目标，两者在WM想象轨迹中训练，目标压缩模块支持选取现实目标，各模块同时训练。可参考未来高层/低层/目标空间的衔接，不宜把它作为本方案新贡献。官方实现是TensorFlow2，与现有PyTorch模型不兼容，需要移植。来源：[官方实现与说明](https://github.com/danijar/director)、[论文](https://arxiv.org/abs/2206.04114)。

## 判断

有可以独立运行的开源研究项目，目前未找到可直接接入本项目现有checkpoint并解决当前验收问题的版本。最优先参考Instruct-to-Act的条件训练＋任务学习联合机制；未来层级设计参考Director。当前真实评估阻塞应通过适合环境随机性的实验设计处理，换WM论文并不会自动恢复相同模拟状态。

本轮只修改评估，未复现或移植上述论文。独立起点试跑使用预声明目标/随机条件、所有真实历史和显式错误计数；它是基于当前环境证据的工程选择，不宣称上述论文采用完全相同协议。
