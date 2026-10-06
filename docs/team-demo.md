# 队员如何观看同一份演示

**观看已有结果不需要先在本地 GPU 重跑模型。** 推荐共享已经完成的演示产物：同一份包中的真实 RGB-D、相机位姿、选点记录、重建网格和指标按文件哈希保持一致。网页旋转、剖切和播放只读取这些结果，不运行 Habitat 或重建优化。

## 三种使用方式

| 用途 | 需要本地 NVIDIA GPU？ | 看到的结果 |
| --- | --- | --- |
| 阅读仓库中的报告和截图 | 不需要 | 已发布的静态说明、指标和截图 |
| 连接现有服务器，或在本地加载同一演示包 | 不需要 | 同一次运行的照片、网格、视角与指标一致；浏览器尺寸和渲染细节可能不同 |
| 重新启动采集、规划、重建与评估 | 需要已配置的 NVIDIA GPU 机器或服务器 | 新的一次运行，不保证与旧记录逐位相同 |

`git clone` 包含代码、前端依赖、公开统计与说明，**不包含数据集、网格和原始训练产物**。这些产物由独立 ZIP 或已有服务器提供；不要提交到 Git，也不要把 SSH 私钥放入演示包。只读回放仅需要 Python 3.10+ 和支持 WebGL 的现代浏览器，无需安装 PyTorch、CUDA、Habitat、Open3D 或 npm 包。

## 本地观看：clone + 演示包 + Python

当前演示包：`viewmend3d-office0-replay-v1.zip`，12,982,961 bytes（约 13 MB），包含 36 个显示文件和校验索引。对应运行 `web-replay-20261006T141303Z-ae2d2b`、Guarded v2、seed 0，8 次真实采集、8 个重建网格、7 次下一视角决策，以及同一评估源的 Ground Truth 简化预览。

包的 SHA-256：

```text
e18bd8d252b46a54f4b90650909742158cc2217a5f4036d3db984c0ca2c81404
```

从项目负责人接收该 ZIP，或由**已有个人服务器访问权限**的队员下载（SSH 别名应在队员自己的电脑配置，不共享私钥）：

```bash
scp JM-New-Outside:/workspace/ViewMend3D/runs/demo-packages/viewmend3d-office0-replay-v1.zip .
```

Windows PowerShell 示例：把 ZIP 放在待 clone 目录旁边，在全新克隆中解压，避免覆盖自己已有的运行产物。

```powershell
git clone https://github.com/SCUTLZH684/ViewMend3D.git
cd ViewMend3D
Get-FileHash -Algorithm SHA256 -LiteralPath ..\viewmend3d-office0-replay-v1.zip
Expand-Archive -LiteralPath ..\viewmend3d-office0-replay-v1.zip -DestinationPath .
python scripts/web/pack_replay.py runs/web-assets/web-replay-20261006T141303Z-ae2d2b-defect_guarded-s0 --verify
python scripts/web/server.py --read-only --port 8766
```

Linux/macOS 在全新克隆中用 `unzip ../viewmend3d-office0-replay-v1.zip` 解压，执行相同的验证和服务命令（按本机配置使用 `python3`）。然后打开 **http://127.0.0.1:8766/#capture-replay**，保持 Python 服务运行。

只读模式显示“本地只读回放已连接”，禁用新实验入口；服务端也拒绝创建作业，不查询 `nvidia-smi`、不申请启动锁、不运行模型。支持 Ground Truth → 首帧 → 算法选点 → 实际补拍的 16 步手动/自动回放，以及逐帧网格和指标。

**此包仅包含这条 8 帧演示。** 仓库内 v1/v2 正式统计仍可阅读，但其他历史运行的三维入口会因没有相应产物而禁用。若要浏览服务器上全部 55 条已完成运行，按[界面说明](demo-interface.md)通过 SSH 转发连接已有服务；同样无需在队员本机运行重建。

## 如果队员重新运行模型

重新运行需要[baseline 环境](reproduction/activegs-office0.md)、Replica 场景、NVIDIA GPU 和对应源码/配置。固定方法、种子、预算、数据和依赖版本有助于复现，但不保证跨硬件/平台/版本逐位一致；PyTorch 官方也明确说明这种[复现边界](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。本项目未宣称所有 CUDA 自定义算子跨机器完全确定。

主动视角选择还会让早期地图差异影响后续候选和轨迹；时间预算的终止位置受机器速度影响。因此展示同一次实验应共享已保存产物；验证方法是否有效应按既定协议重跑多个种子、比较质量与成本。包中的指标来自原完整网格评估，不因浏览器简化预览或回放速度而改变。

## 维护演示包

在服务器上对已验收的显示资产打包；新 ZIP 路径必须不存在。打包器只收集 manifest 引用的预览、实际观测 PNG 和必要 JSON/热图，核对观测、回放和参考预览哈希，不包含原始深度 NPY、模型 checkpoint、pickle、私钥或未被引用的旧网格。

```bash
python scripts/web/pack_replay.py \
  runs/web-assets/web-replay-20261006T141303Z-ae2d2b-defect_guarded-s0 \
  runs/demo-packages/a-new-demo-package.zip
```

更新 ZIP 时同步记录新文件大小与 SHA-256。代码版本和模型运行版本分别保存，不用新前端提交替换旧实验的冻结源码身份。
