# 实现基础与数据集筛选

更新日期：2026-10-05。项目仍处于从零探索阶段，本文确定第一轮复现顺序，最终架构和改进点尚未确定。

## 当前选择

第一轮实测选用 **ActiveGS 原始方法 + Replica `office0`**。当前地图 → 下一视角 → RGB-D 新观测 → 地图更新 → 网格和几何指标，构成可供 A1“根据几何缺陷主动补拍”复用的闭环。先验证原实现，再讨论替换视角评分。

本轮 GPU 闭环已通过：原始 confidence 方法经两行启动兼容修复后，完成了重建、5 个网格导出和原几何评估。实际证据及解释边界见[复现记录](../reproduction/activegs-office0.md)。这仍是实现基础筛选，尚未确定项目最终架构或算法改进。

**NBV-Gym + DTU** 保留为有限候选照片池路线的备选。它接近“从已有照片中选择补充视角”的形式，但 DTU 数据转换和独立几何评估需要额外实现，尚未进行 GPU 实测。

本轮验收目标是原始方法能够执行、导出三维结果并计算几何指标。复现论文表格、验证优于其他方法、证明我们的改进有效，需要之后的对照实验。

## 实现基础比较

| 候选实现 | 输入与地图 | 原有闭环和成本 | 本轮决定 |
| --- | --- | --- | --- |
| [ActiveGS](https://github.com/dmar-bonn/active-gs) | RGB-D；2D Gaussian surfel 与体素地图 | 有规划、增量建图、TSDF 网格和几何指标；需要 Habitat、CUDA 扩展和 EGL | 优先验证完整原方法 |
| [NBV-Gym](https://github.com/chengine/nbv_gym) | RGB 照片池；Nerfstudio/Splatfacto | 有候选视角选择和训练集合更新；框架版本需固定，DTU 接入与几何评估另做 | 有限视角池备选 |
| [NARUTO](https://github.com/oppo-us-research/NARUTO) | RGB-D；神经隐式表面 | 有主动重建；依赖和采样逻辑复杂，需处理前期审查发现的评估/采样问题 | 暂缓 |
| [ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER) | RGB-D；SplaTAM/3D Gaussian | 有主动建图和精修；依赖较重，GT 参与部分过滤，公平评估需澄清 | 暂缓 |

以上为源码与依赖筛选，不能把“原仓库提供运行说明”写成“我们已复现成功”。详细边界见[前期调研](active-reconstruction-systems.md)。DA3 等通用重建模型可以成为后端，但还需我们编写主动选视角循环，不能直接当成已验证的原始主动 baseline。

## 数据集比较

| 数据 | 三维参考与观测 | 第一阶段规模 | 使用方式 |
| --- | --- | --- | --- |
| [Replica](https://github.com/facebookresearch/Replica-Dataset) `office0` | 场景 mesh、纹理、Habitat 配置；按选定相机生成 RGB-D | 实测 ZIP **744,494,396 字节**；解压文件合计 **1,000,725,289 字节** | 室内场景主动采集，用场景 mesh 评价几何；第一轮选用 |
| [DTU MVS](https://roboimagedata.compute.dtu.dk/?page_id=36) | 多视角图像、标定、结构光参考点云、观测掩码 | 官方 SampleSet 标注 **6.3 GB**，含 set1/set6 与评估材料 | 单场景、单光照候选照片池；转换相机坐标与尺度后接几何评估 |

Replica 完整场景资产与其他项目已渲染的固定 RGB-D 序列不同。ActiveGS 需要可从新相机渲染的完整场景与纹理。DTU 的 `Points` 是结构光参考；`Points_MVS` 与 `Surfaces` 是已有 MVS 方法预测，不能当作真实几何。初期无需下载 123/136 GB 的全量图像。[DTU 官方说明](https://roboimagedata.compute.dtu.dk/?page_id=36)

Replica 数据可用于非商业研究/教学，应遵守[数据许可](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE)并引用论文。数据许可与代码许可证分别记录。数据、权重和实验大文件存放 GPU 服务器，GitHub 保存代码、环境、配置、命令和小型结果。

## 固定版本

| 组件 | 提交 |
| --- | --- |
| ActiveGS | `558121a00b2eca84d9851a609e99ebb8e26ea2d8` |
| 作者 Habitat fork | `3023da6e8617a3b676b3e9f8845788631f8377d5` |
| 作者 2D rasterizer fork | `53be5e91b2d81081ae0829d5b5c16d47b98293fb` |
| NBV-Gym，仅源码审查 | `6e5e5ef48a59f369a821802842504668eac23325` |

Habitat fork 修复了 Replica PTex 几何着色器编译问题，本轮保留作者指定 fork 和对应子模块提交，不能直接假设官方二进制包等价。

## 原始方法验证协议

1. 在 NVIDIA GPU 上验证 PyTorch 运算，编译作者 CUDA 光栅器。
2. 下载示例脚本指向的 `office_0.zip`，核对大小、归档 CRC 和场景文件。
3. 用原初始相机实际渲染一帧，保存 RGB、原始深度与有效深度统计。
4. 运行原 `data_generation.py` 生成测试相机。
5. 运行 `main.py planner=confidence scene=replica/office0 use_gui=false`，保留默认 **300 秒任务预算、60 秒记录间隔、512×512 观测、100 个候选视角**。
6. 运行 `mesh_generation.py`，确认每个 checkpoint 对应非空三角网格。
7. 运行原 `eval.py` 的 `eval_mode=mesh`，获得几何精度、完整度、2 cm 完整率、Chamfer。
8. 用独立文件检查核对地图/相机/网格、记录数量、指标长度和数值。

任务预算是原程序累计的规划、建图和模拟移动时间，不等同于全部流程墙钟时间。运行入口为 [`scripts/activegs/run_original.sh`](../../scripts/activegs/run_original.sh)，实际结果另见复现记录。

## 环境与评估边界

- 原作者测试 Ubuntu 20 / CUDA 11.8；服务器为 Ubuntu 22.04、A100 80GB、驱动 580.173.02、CUDA 编译器 12.4。本轮独立环境采用 Python 3.9、PyTorch 2.1.2 cu121，属于环境兼容性调整，能否工作需由实测确认。
- NumPy 固定 1.26.4，`e3nn`/`timm` 添加约束，避免新依赖自动升级 PyTorch。实际依赖以运行记录的 `pip-freeze.txt` 为准。
- EGL 库与头文件放入项目私有目录，通过搜索路径编译和加载。
- 首次未改上游入口遇到多进程上下文重复设置错误，使用[两行启动兼容补丁](../../patches/activegs/multiprocessing-start-method.patch)再验证。兼容修复与算法改进分别记录。
- 原 `office0` 配置 `has_missing_surface=true` 会在候选评分时查询模拟器有效深度掩码。复现保留该设置；将来声称只依赖已采集信息时，需关闭/隔离未观测信息并重新对照。
- 原网格/评估加载器对单行记录存在维度问题，本轮保持默认时长以获得多条 checkpoint，检查脚本拒绝单条记录。
- 几何精度与完整度单位 **cm**，完整率 **%**，`mesh_chamfer_distance` 单位 **m**；Chamfer 定义为双向平均最近邻距离的平均值。
- 本轮只验证网格评估分支；GUI、渲染质量评估、多场景、多随机种子和对照方法另行验证。

## 后续顺序

原闭环通过后，先做相同初始化/预算下的 random 与原 confidence 对照，再设计局部几何缺陷评分。统一候选、观测预算、优化步数、随机种子与评估区域，并加“继续优化但不采集新视角”对照，区分补拍与额外优化的作用。最终采用室内模拟器还是 DTU 照片池路线，应依据真实运行成本、结果与改动复杂度决定。
