# ViewMend3D
Geometry-Aware Next-Best-View Selection for Active 3D Reconstruction

当前处于探索阶段：ActiveGS + Replica office0 的原始 confidence 方法已在 GPU 上完成单场景闭环验证（含两行启动兼容修复），ViewMend3D 自研视角评分尚未实现。

## 可视化实验界面

已有中文浏览器界面：真实三维网格、分阶段回放、采集轨迹和指标曲线；可在空闲 GPU 上启动新的 office0 原始重建实验，并查看阶段与日志。使用已配置的服务器和 SSH 转发访问，见 [界面使用说明](docs/demo-interface.md)。数据与三维产物留在服务器，不随仓库发布。

## 技术调研

- [实现基础与数据集筛选](docs/research/baseline-dataset-screening.md)：第一轮验证 ActiveGS + Replica office0，备选 NBV-Gym + DTU；记录源码版本、验证协议和评估边界。
- [ActiveGS office0 复现记录](docs/reproduction/activegs-office0.md)：独立 GPU 环境、执行入口与实际验收结果。
- [主动重建系统对比与复用建议](docs/research/active-reconstruction-systems.md)：ActiveGS、NBV-Gym、NARUTO、ActiveGAMER 的源码闭环、复用边界、评估风险与候选技术路线。
