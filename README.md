# ViewMend3D
Geometry-Aware Next-Best-View Selection for Active 3D Reconstruction

当前处于探索阶段：ActiveGS + Replica office0 的原始 confidence 方法已完成单场景 GPU 验证（含两行启动兼容修复）。ViewMend3D v1 已完成 3 个短实验、15 个固定观测分支和9个时间预算分支的完整GPU验收。三种子公平对照表明几何缺陷评分未取得稳定质量提升，全部退化与消融结果见[完整实验报告](docs/reproduction/optimization-v1-results.md)。

队员可先阅读 [当前项目说明](docs/project-overview.md)，了解目标、数据形式、baseline 实测结果、运行方式与下一阶段任务。

v1协议已归档。后续优化严格按预注册的 [v2框架](docs/optimization-v2.md) 执行：有界几何奖励、批量几何计算，开发种子和留出种子分别验证。当前源码冻结时仅完成CPU实现与回放，尚无v2 GPU质量结论；[v1机制诊断](docs/reproduction/v1-geometry-diagnosis.md)不替代新的重建实验。运行参数、公共前缀与审计见 [受控实验说明](docs/controlled-experiments.md)。

v2的164项Linux CPU检查、真实状态读写与共享启动锁验证、固定β历史候选回放见[源码冻结前验收](docs/reproduction/optimization-v2-cpu.md)。

第一阶段的49项CPU检查记录见[优化实现记录](docs/reproduction/optimization-v1.md)，后续118项CPU准备检查见下方记录；这些历史检查与真实GPU实验分开。

现已继续完成评分中的冗余复制优化、场景纹理身份校验、当前产物审计及空闲后的顺序实验推进工具，见 [CPU 准备记录](docs/reproduction/cpu-preparation-v2.md)。这些实现检查不代表新方法质量提升。

## 可视化实验界面

已有中文浏览器界面：24份v1正式结果的真实三维网格、分阶段回放、采集轨迹、热图、指标曲线与历史汇总；支持原始、公平和v2固定评分方法的新实验入口，以及v1/v2计划状态。使用已配置的服务器和 SSH 转发访问，见 [界面使用说明](docs/demo-interface.md)。数据与三维产物留在服务器，不随仓库发布。

## 技术调研

- [实现基础与数据集筛选](docs/research/baseline-dataset-screening.md)：第一轮验证 ActiveGS + Replica office0，备选 NBV-Gym + DTU；记录源码版本、验证协议和评估边界。
- [ActiveGS office0 复现记录](docs/reproduction/activegs-office0.md)：独立 GPU 环境、执行入口与实际验收结果。
- [主动重建系统对比与复用建议](docs/research/active-reconstruction-systems.md)：ActiveGS、NBV-Gym、NARUTO、ActiveGAMER 的源码闭环、复用边界、评估风险与候选技术路线。
