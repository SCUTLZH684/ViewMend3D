# ViewMend3D
Geometry-Aware Next-Best-View Selection for Active 3D Reconstruction

当前处于探索阶段：ActiveGS + Replica office0 的原始 confidence 方法已完成单场景 GPU 验证（含两行启动兼容修复）。ViewMend3D 第一版几何缺陷评分及公平对照入口已实现，CPU 检查通过，完整 GPU 闭环与质量提升仍待验证。

队员可先阅读 [当前项目说明](docs/project-overview.md)，了解目标、数据形式、baseline 实测结果、运行方式与下一阶段任务。

本轮按 [优化框架](docs/optimization-framework.md) 执行，运行参数、公共前缀、对照消融和结果审计见 [受控实验说明](docs/controlled-experiments.md)。

第一版扩展的 46 项 CPU 检查与真实旧结果展示验证已通过，范围和 GPU 待验收项见 [优化验证记录](docs/reproduction/optimization-v1.md)。

## 可视化实验界面

已有中文浏览器界面：真实三维网格、分阶段回放、采集轨迹和指标曲线；可在空闲 GPU 上启动新的 office0 原始重建实验，并查看阶段与日志。使用已配置的服务器和 SSH 转发访问，见 [界面使用说明](docs/demo-interface.md)。数据与三维产物留在服务器，不随仓库发布。

## 技术调研

- [实现基础与数据集筛选](docs/research/baseline-dataset-screening.md)：第一轮验证 ActiveGS + Replica office0，备选 NBV-Gym + DTU；记录源码版本、验证协议和评估边界。
- [ActiveGS office0 复现记录](docs/reproduction/activegs-office0.md)：独立 GPU 环境、执行入口与实际验收结果。
- [主动重建系统对比与复用建议](docs/research/active-reconstruction-systems.md)：ActiveGS、NBV-Gym、NARUTO、ActiveGAMER 的源码闭环、复用边界、评估风险与候选技术路线。
