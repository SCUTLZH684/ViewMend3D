# ActiveGS + Replica office0 复现记录

**验收通过：原始 `confidence` 方法经两行启动兼容修复后，完成了真实 GPU 上的完整几何重建与评估闭环。** 首次未修改上游入口失败，修复后第二次实验退出码为 0。视角评分、地图更新、网格提取与评估算法均未修改。

## 实测结果

成功实验：`/workspace/ViewMend3D/runs/20261005-office0-original-02/`。任务累计时间 300.13 秒，采集 186 次观测，生成 1000 个测试相机，保存 5 个 Gaussian 地图和对应三角网格。最后网格为 291,273 顶点、578,175 三角面。

| 原作者几何指标 | 最后保存的 checkpoint |
| --- | ---: |
| Accuracy：重建表面到 GT 的平均距离 | 0.83084 cm |
| Completion：GT 到重建表面的平均距离 | 0.89985 cm |
| Completion ratio：GT 在 2 cm 内的比例 | 96.56680% |
| 双向平均 Chamfer 距离 | 0.00865345 m（约 8.65 mm） |

原评估器在重建/GT 表面各采样 500,000 点，使用最近邻距离。每个 checkpoint 的地图、相机和非空网格均存在；指标数量与记录一致，数值有限，Chamfer 与原定义/单位一致。

- [验收摘要](evidence/office0-run-summary.json)
- [原始指标 JSON](evidence/office0-final-result.json)与[checkpoint 记录](evidence/office0-record-info.txt)
- [独立产物检查](evidence/office0-artifact-check.json)与[实际实验配置](evidence/office0-exp-config.yaml)
- [模拟器实际渲染检查](evidence/office0-preflight.json)：512×512 RGB/深度，初始视角有效深度比例 100%
- [数据 CRC/大小/哈希](evidence/office0-download.json)、[Habitat 依赖提交](evidence/habitat-source-manifest.json)、[实际 Python 依赖](evidence/pip-freeze.txt)
- [首次未修复入口的错误记录](evidence/office0-unpatched-failure.txt)

![原始 confidence 方法单次实验的几何指标](evidence/office0-metrics.png)

曲线只描述本次原始方法的单次运行，不是我们改进方法的结果，也不是统计显著性验证。

## 环境与数据

- Ubuntu 22.04，单张 NVIDIA A100-SXM4-80GB（GPU 0），驱动 580.173.02。
- 项目目录 `/workspace/ViewMend3D`；Python 独立环境 `.envs/activegs`，版本 3.9.19。
- PyTorch 2.1.2+cu121、CUDA 编译器 12.4、NumPy 1.26.4、Open3D 0.17.0、CMake 3.14.0。
- ActiveGS、Habitat 与光栅器提交见[筛选记录](../research/baseline-dataset-screening.md)。Habitat 采用作者指定 fork 的源码与固定子模块，无窗口、Bullet、PTex 支持。
- EGL/GLVND 库与头文件从 Ubuntu 官方 jammy 包解包到项目 `.sysroot`，未通过 apt 安装或替换系统驱动。
- `office_0.zip` 完整归档 CRC 校验通过，SHA256：`0dc0be9445d441652fa76afe60cddbac964731573a4cc6a9506eef66ac995d86`。
- 原有服务器 Python/PyTorch 环境未用于依赖安装。数据、地图、网格、原始日志留在服务器，仓库只保存小型证据文件。

## 安装实录

使用 conda 创建独立 Python 3.9 / CMake 3.14.0 环境，安装 PyTorch 2.1.2 的 cu121 构建。安装 ActiveGS 与 Habitat Python 依赖时使用[兼容性约束](../../scripts/activegs/environment-constraints.txt)，以免依赖解析自动升级 PyTorch。

光栅器使用作者固定源码，设置 `CUDA_HOME=/usr/local/cuda-12.4`、`TORCH_CUDA_ARCH_LIST=8.0`、`MAX_JOBS=8`，通过 `pip install --no-build-isolation` 编译。CUDA 编译器与 PyTorch 的 CUDA 次版本不同，该组合已通过本次运行验证。

Habitat 的依赖按原 gitlink 获取源码；源代码归档不含子模块工作区时不能只下载主仓库 ZIP。编译时加入项目私有 EGL 搜索路径，再执行：

```bash
export PATH=/workspace/ViewMend3D/.envs/activegs/bin:$PATH
export CMAKE_PREFIX_PATH=/workspace/ViewMend3D/.sysroot/usr
export LD_LIBRARY_PATH=/workspace/ViewMend3D/.sysroot/usr/lib/x86_64-linux-gnu:/workspace/ViewMend3D/.envs/activegs/lib:${LD_LIBRARY_PATH:-}
cd /workspace/ViewMend3D/external/habitat-pinned
python setup.py build_ext --parallel 8 install --headless --bullet --no-update-submodules --force-cmake
```

构建还会获取 OpenEXR 所需 Imath v3.0.5，本次解析的提交为 `50c62962835bbab237c6f0c3ebbbb8c694dc5033`。服务器无法直连部分公开站点时，本轮通过临时 SSH 下载通道获取公开源码/数据；这属于传输环境问题，不涉及算法调整。

## 首次失败与兼容修复

未修改的上游入口在测试相机生成阶段退出：`RuntimeError: context has already been set`。模拟器和相关依赖完成初始化后，原程序再次调用 `mp.set_start_method("spawn")`，产生上下文冲突。

仅在 `main.py` 与 `data_generation.py` 的该调用中增加 `force=True`，见[两行启动补丁](../../patches/activegs/multiprocessing-start-method.patch)。这不改变视角评分、建图、网格提取或指标计算；它是环境兼容修复，不是项目算法改进。后续成功结果应表述为“原始方法经启动兼容修复后跑通”，不能表述为“未修改上游直接跑通”。

```bash
git -C /workspace/ViewMend3D/external/active-gs apply /workspace/ViewMend3D/patches/activegs/multiprocessing-start-method.patch
```

只需应用一次。未修复失败日志保留在 `/workspace/ViewMend3D/runs/20261005-office0-original-01/`。

## 再次运行

环境准备完毕后使用以下入口。每次 `RUN_DIR` 必须是新目录，避免把旧结果误当作本次输出：

```bash
export LD_LIBRARY_PATH=/workspace/ViewMend3D/.sysroot/usr/lib/x86_64-linux-gnu:/workspace/ViewMend3D/.envs/activegs/lib:${LD_LIBRARY_PATH:-}
export ACTIVEGS_ROOT=/workspace/ViewMend3D/external/active-gs
export ACTIVEGS_PYTHON=/workspace/ViewMend3D/.envs/activegs/bin/python
export RUN_DIR=/workspace/ViewMend3D/runs/office0-original-NEW_RUN
export GPU=0
bash /workspace/ViewMend3D/scripts/activegs/run_original.sh
```

入口按顺序执行实际模拟器渲染、测试视角生成、原 confidence 重建、网格导出、原 mesh 评估和独立产物检查。预算、记录间隔、相机分辨率、候选数和优化步数采用原默认值，仅关闭 GUI 并选择网格评估分支。

`logs/` 保存各阶段日志，`evidence/` 保存渲染检查、实际依赖和产物检查，`experiments/original/replica/office0/confidence/0/` 保存模型、相机、网格、配置与 `final_result.json`。

## 解释边界

首次单场景成功只能证明执行与结果链条可用，不能证明论文多场景结果已复现或我们的改进优于 baseline。原 `office0` 的候选有效深度掩码设置保持开启；今后做无未观测信息的对照时必须明确处理。精度/完整度单位 cm，完整率单位 %，Chamfer 单位 m。

原程序 `run_id` 是实验编号，不会设置随机种子。本次未固定随机状态，后续重跑的指标可以不同；正式对照需要统一种子并进行多次运行。原程序按间隔保存地图，没有无条件保存任务结束时最后一张地图，因此指标对应保存的 checkpoint。
