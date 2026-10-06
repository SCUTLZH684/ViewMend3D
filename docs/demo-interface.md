# ViewMend3D 重建实验台

浏览器界面展示已完成的 **ActiveGS confidence + Replica office0**、24份v1正式公平实验及v2四组24份正式结果，支持启动原作者流程、v1公平方法或v2有界几何奖励方法的新实验。v1未取得稳定质量提升；v2五阶段27分支已完成导出与fresh审计，留出均值局部改善，但种子/Accuracy等权衡仍未稳定。全部结果见[v1报告](reproduction/optimization-v1-results.md)和[v2报告](reproduction/optimization-v2-results.md)，协议见[受控实验说明](controlled-experiments.md)。

## 可以操作什么

- 旋转、缩放和平移三维网格；剖切屋顶，查看室内结构；切换线框。
- 切换或播放真实保存阶段；原作者5阶段、v1固定观测3阶段、固定时间4阶段，运动路径、采集相机和指标随阶段同步变化。
- 查看 Accuracy / Completion（cm）、2 cm 真值覆盖率（%）、Chamfer（mm）及其变化曲线，点击曲线记录点定位阶段。
- 查看真实初始 RGB 观测、原始网格顶点数和面数。
- 选择空闲 GPU 和 60 / 180 / 300 秒任务预算，启动一次新实验；查看执行阶段和日志；完成后切换到新结果。
- 选择公平协议、种子 0/1/2 和单方法或三种主方法对照，执行 60 次更新。前 20 次为公共观测前缀；仅优化分支采集数始终为 20。
- 查看方法、种子、协议和已完成结果对比；有真实输出时显示选中候选的几何缺陷热力图。
- 查看自动实验计划的等待、阶段、已验收条目与进程状态。可选择v1历史或v2计划；v2顺序为短闭环、开发固定观测/时间、留出固定观测/时间，共5阶段27分支。这里只读显示，实际调度见[v2协议](optimization-v2.md)。
- v2诊断显示实际奖励、上限、回退原因和基线代理分数损失；“改选”指该方法自身当前地图上同次候选集的S0首选与S2首选不同。它不是与独立Confidence实际轨迹逐事件比较，局部零改选也不表示独立分支相同；分数约束不能证明几何质量改善。
- 查看v1历史汇总，两种预算分开显示三种子的均值±样本标准差、同种子配对差、成本和几何信号行为；该公开快照独立于当前任务状态，保留退化结论，±不是置信区间。
- 查看v2四组独立汇总：开发固定观测/固定时间（seeds0/1，n=2）、留出固定观测/固定时间（seeds3/4/5，n=3）。主方法固定Guarded；表格展示质量、配对差与成本，真实网格按钮关联具体方法/seed导出，不将四组合并为一个均值。

原作者入口沿用累计 mapping / planning / flight 时间，并非网页点击后的墙钟耗时；它包含初始观测检查、1,000 个评估视角生成、主动重建、网格生成、几何评估和网页导出。公平入口从公共前缀恢复，硬性禁止规划查询候选真值，使用统一的 500,000 点网格评估，无需生成未使用的测试相机。

## 在已配置的 GPU 服务器启动

以下路径对应当前已验证的安装布局；换机器时需先完成 [baseline 环境配置](reproduction/activegs-office0.md)。网页后端只依赖 Python 标准库；网格导出使用已安装的 ActiveGS Python 环境，不需要额外 Node 或 npm 安装。

```bash
cd /workspace/ViewMend3D
# 已有实验只需导出一次；数据与简化网格不提交 Git。
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .envs/activegs/bin/python \
  scripts/web/export_run.py \
  runs/20261005-office0-original-02 \
  runs/web-assets/20261005-office0-original-02

# 服务仅监听 127.0.0.1，保留此进程运行。
python scripts/web/server.py --port 8765
```

Windows 本机使用已配置密钥的 SSH 别名转发端口，在另一终端运行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 JM-New-Outside
```

然后在浏览器打开 **http://127.0.0.1:8765**。也可运行 `scripts/web/connect.ps1`；如果 SSH 配置未放在默认位置，传入 `-SshConfig '配置文件的完整路径'`。本机与服务器端口需一致。浏览器连接依赖服务器进程和 SSH 转发均保持运行。

## 数据与显示边界

- 首轮结果来自 2026-10-05 的 300 秒实测，5 个记录阶段，186 个采集视角，2,930 个实际运动路径位置。
- 浏览器默认将每阶段网格简化到约 80,000 个三角面。界面标注的原始顶点数、面数和全部评估指标来自完整网格；预览简化不会重算指标。
- 相机记录是 **OpenCV camera-to-world**，场景 **Z 轴向上**；每个阶段只显示当时已采集的相机及对应运动路径前缀，不显示未来路径或 1,000 个评估视角。
- 路径记录与采集相机数量必须匹配，否则导出拒绝生成 manifest。没有运动路径文件时，仅显示采集视角连线。
- 初始图像来自 Habitat RGB 传感器。当前流水线未保存完整 RGB 视频序列，页面不提供虚构视频或实时 RGB 画面。
- 当前是保存阶段浏览与实验管理，不提供逐帧实时 Gaussian Splatting 渲染。原作者结果含缺失表面候选掩码，公平入口关闭该查询；两类结果分开展示。

## 启动约束与验证

后端仅接受 office0 的固定协议：原作者三个时间预算，或公平 60 次更新、种子 0/1/2 和列出的方法。GPU 显存占用低于 1,024 MiB 且利用率不高于 5% 时才显示为空闲；接收请求和实际执行前各检查一次，不终止其他用户进程。每次使用独立 `runs/web-...` 目录。网页与自动实验推进器共享项目启动锁，核查彼此的启动记录和进程，只允许一个受管实验同时运行。此锁协调本项目的v1计划、v2计划与网页入口，无法锁住其他用户的程序或独立 CLI；服务器分配约定仍需遵循。

控制服务仅绑定回环地址，经 SSH 转发访问，检查 Host、Origin 和 JSON 请求头；不接受上传 pickle 或任意 shell 命令。已有实验导出会读取 **项目自己生成并信任的本地 pickle**，请勿替换为来源不明文件。此入口用于个人或课程组实验，不应直接发布为公网多用户服务。

已完成验证：

- 在真实服务器导出并加载首轮 5 个阶段的网格、实际相机/路径和指标。
- Edge 浏览器自动测试：WebGL 渲染无页面错误，首末阶段切换、曲线联动、剖切、线框及路径/相机开关通过；390 px 手机布局无横向溢出。
- 后端CPU检查覆盖原参数与固定公平协议、占用GPU拒绝、执行前复查、跨入口启动互斥、来源/路径限制、子进程登记、只读计划状态、失败与成功状态及多方法导出；v2还包含固定配方与双计划状态投影。历史验收记录见[CPU准备](reproduction/cpu-preparation-v2.md)及[v2冻结前记录](reproduction/optimization-v2-cpu.md)。
- 导出检查覆盖真实旧结果的浮点阶段编号、公共前缀诊断、采集与更新计数、路径端点和方法身份；接口及测试通过不代表新方法已完成 GPU 重建。
- 真实服务器启动请求在 GPU 全部占用时返回 409，未占用正在执行其他任务的 GPU。
- v1 CLI campaign已在空闲卡完成27条整链；24份正式预览、三种子缺陷热图、20观测/60事件Refine与时间预算的实际最终事件均已浏览器只读核对。v1历史汇总逐项匹配审计JSON，桌面和手机通过，零POST、零页面异常。
- v2五阶段27分支、27份真实导出已完成；最终fresh审计重新核查二进制/网格/实际相机、诊断与成本，正式24分支按四组分别汇总。页面识别v1/v2计划schema，完成状态来自真实state API；历史汇总与当前作业状态分开。
- **已完成一次真实HTTP启动→GPU重建→60次实际观测→3检查点→导出→completed及进程退出的完整验收**，未来候选观测调用为0。这条入口检查独立于正式27分支；后续启动器变更另立验收，不改变历史实验源与质量统计。不能据此声称其他硬件、所有参数组合或部署入口都已验证。

![v1历史汇总：固定观测与配对差](images/v1-results-interface.png)

公开统计随clone提供，可在没有大型地图时读取；[v2完整最终分析](reproduction/evidence/optimization-v2-final-analysis.json)仅含指标、成本、诊断与来源元数据，原SHA256为 `541b0effba87ae82928801a0ec2cfb26f4a5189674165c378a5ac7affadb61fd`。可用发布工具在CPU上复现报告与图，命令见[受控实验说明](controlled-experiments.md)。真实三维预览仍需服务器上的 `runs/web-assets/`。成本展开表包含任务、重建墙钟、规划、建图、观测、路径与Torch峰值，不能把更少观测伴随的低耗时写成等质量加速。留出质量局部均值改善与seed/Accuracy退化同时保留，不因果归为奖励收益。

后端测试不依赖 GPU：

```bash
python scripts/web/test_server.py
```

## 文件位置

| 路径 | 内容 |
| --- | --- |
| `web/` | 中文界面、三维交互与曲线 |
| `web/vendor/` | 固定版本 three.js 0.186.1、OrbitControls、PLYLoader 及 MIT 许可证；无运行时 CDN 依赖 |
| `scripts/web/server.py` | 回环 HTTP 服务、GPU 状态、受约束的实验启动与日志 |
| `scripts/web/export_run.py` | 原始实验产物 → 简化 PLY 与 JSON manifest |
| `scripts/web/connect.ps1` | Windows SSH 转发与浏览器入口 |
| `runs/web-assets/` | 本机实验预览，Git 忽略 |
| `runs/web-jobs/` | 作业状态、worker 日志，Git 忽略 |

可视化依赖使用 [three.js 官方发行包](https://registry.npmjs.org/three/0.186.1)，导出使用 [Open3D 网格简化](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-simplification)。Replica 数据和派生网格留在 GPU 服务器，遵循研究与教学使用范围。
