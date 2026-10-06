# ViewMend3D
Geometry-Aware Next-Best-View Selection for Active 3D Reconstruction

项目已完成原方法复现和两轮单场景优化验证，仍在研究探索阶段。ActiveGS + Replica office0 的原始 confidence 方法已完成 GPU 验证（含两行启动兼容修复）。ViewMend3D v1 已完成 3 个短实验、15 个固定观测分支和9个时间预算分支；三种子公平对照未取得稳定质量提升，质量、成本和消融结果见[v1完整报告](docs/reproduction/optimization-v1-results.md)。

队员可先阅读 [当前项目说明](docs/project-overview.md)，了解目标、数据形式、baseline 实测结果、运行方式与下一阶段任务。

v2严格按[预注册框架](docs/optimization-v2.md)完成五阶段27条GPU实验、全部导出及当前原始产物fresh审计，其中24条正式质量分支分为开发/留出种子、观测/时间预算四组。固定主方法为有界几何奖励 `defect_guarded`。留出 Completion 和覆盖率均值部分改善，但种子表现与 Accuracy 等指标存在权衡，未获得稳定提升；同次候选集的 S0/S2 改选不能解释独立分支的全部质量差异。详见[v2完整报告](docs/reproduction/optimization-v2-results.md)。v1结果和[v1机制诊断](docs/reproduction/v1-geometry-diagnosis.md)保留，运行参数、公共前缀与审计见[受控实验说明](docs/controlled-experiments.md)。

v2的164项Linux CPU检查、真实状态读写与共享启动锁验证、固定β历史候选回放见[源码冻结前验收](docs/reproduction/optimization-v2-cpu.md)。

第一阶段的49项CPU检查记录见[优化实现记录](docs/reproduction/optimization-v1.md)，后续118项CPU准备检查见下方记录；这些历史检查与真实GPU实验分开。

现已继续完成评分中的冗余复制优化、场景纹理身份校验、当前产物审计及空闲后的顺序实验推进工具，见 [CPU 准备记录](docs/reproduction/cpu-preparation-v2.md)。这些实现检查不代表新方法质量提升。

## 可视化实验界面

已有中文浏览器界面：保留24份v1正式结果，新增v2四组独立汇总及真实网格入口，展示分阶段回放、采集轨迹、热图、指标与历史结果。支持原始、公平和v2固定评分方法的新实验入口，以及v1/v2计划状态；已完成一次真实HTTP请求→GPU重建→60次观测→导出的完整启动验收。使用服务器和SSH转发访问，见[界面说明](docs/demo-interface.md)。场景、地图和三维产物留在服务器；公开指标与审计元数据随仓库提供。

## clone 后在 CPU 复现报告与图表

可信的[v2完整最终分析](docs/reproduction/evidence/optimization-v2-final-analysis.json)只含指标、成本、诊断信号和来源元数据，原字节SHA256为 `541b0effba87ae82928801a0ec2cfb26f4a5189674165c378a5ac7affadb61fd`。无需下载场景或占用GPU即可复现统计与图表；图表需要Matplotlib，去掉 `--plots` 时发布工具仅依赖Python标准库。输出目录必须从未存在。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/activegs/publish_v2_results.py \
  docs/reproduction/evidence/optimization-v2-final-analysis.json \
  --expected-sha256 541b0effba87ae82928801a0ec2cfb26f4a5189674165c378a5ac7affadb61fd \
  --output-dir runs/v2-cpu-reproduction-new --plots
```

工具要求完整五阶段27分支、固定配方与源码身份，并重算统计/配对和诊断门控。公开JSON与SHA支持复现已审阅结果，不能仅凭一个自行标记complete的文件证明真实GPU运行。完整重建及交互式网格浏览仍需数据、依赖环境或服务器产物。

## 技术调研

- [实现基础与数据集筛选](docs/research/baseline-dataset-screening.md)：第一轮验证 ActiveGS + Replica office0，备选 NBV-Gym + DTU；记录源码版本、验证协议和评估边界。
- [ActiveGS office0 复现记录](docs/reproduction/activegs-office0.md)：独立 GPU 环境、执行入口与实际验收结果。
- [主动重建系统对比与复用建议](docs/research/active-reconstruction-systems.md)：ActiveGS、NBV-Gym、NARUTO、ActiveGAMER 的源码闭环、复用边界、评估风险与候选技术路线。
