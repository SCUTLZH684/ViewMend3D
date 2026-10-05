# 不占用 GPU 的优化与后续实验推进

更新日期：2026-10-06。本轮遵循 [优化框架 v1](../optimization-framework.md)，保持评分公式、权重、候选规则、公共前缀、优化步数和评估定义不变。以下工作是实现效率、数据身份与验证流程的改进，不是新方法质量提升的证据。

## 1 评分计算

移除每候选三次 `nan_to_num` 前的冗余图像复制；该算子原本就返回新张量，原渲染图仍保持不变。相机量程在 CPU 元数据上验证并缓存，避免在每候选循环中重复读取设备标量。

隐藏全部 CUDA 后，用 100 个 128×128 合成候选检查旧版本与新版本：confidence、defect 和无深度门控三种方法的效用、有效比例、热图、可达性、得分和选择逐项一致。CPU profiler 中 `aten::clone` 调用 300→0，标量提取 711→311。共享服务器上的 CPU 夹具耗时有波动，不能据此推断 GPU 或完整重建加速，更不能推断质量提升。见 [评分检查证据](evidence/scoring-cpu-v2.json)。

替换邻域卷积和批处理的 CPU 探针没有证明稳定收益，未纳入实现。构造样例与实际新 GPU 重建结果在项目中始终分开记录。

## 2 数据与环境检查

新增 `scripts/activegs/check_environment_cpu.py`，只读取版本元数据、配置和文件，不导入 Torch 或 Habitat 运行时。检查固定 ActiveGS 提交、已记录的启动补丁、冻结参数、环境版本、磁盘余量以及 mesh、PTex 纹理和 stage 引用文件，见 [实际静态环境检查](evidence/environment-cpu-v2.json)。

公共前缀上下文现包含全部场景输入的文件指纹，纹理变化也会改变身份；不同数据不能复用同一个前缀或被归入同一个对照。场景指纹仅用于输入审计，不把真实语义或候选真值送入评分。

```bash
cd /workspace/ViewMend3D
CUDA_VISIBLE_DEVICES='' .envs/activegs/bin/python scripts/activegs/check_environment_cpu.py \
  --upstream external/active-gs --output runs/cpu-preparation/environment.json
```

通过静态检查仍不代表 CUDA、EGL、模拟器渲染或完整重建已通过。

## 3 产物和对照审计

标准库验收器重新检查当前文件，核对地图 ZIP 完整性、完整网格数据和面索引、相机参数与累积采集前缀、时间和路径对齐，并保存 SHA-256。汇总每次重新验收，避免旧的成功 JSON 掩盖当前文件缺失或被替换；还核对逐事件记录、检查点、成本、真实停止条件和种子唯一性。

此前真实原方法的 5 个阶段已重新通过检查，相机数为 40/78/115/153/186，见 [当前文件审计](evidence/office0-artifacts-cpu-v2.json)。标准库检查不解码地图张量语义或全局路径 Torch 位姿；这些限制保留在报告 `scope` 中。真实轨迹位置由已有 CPU 导出器进一步核验。旧结果仍只证明原方法的执行，不能作为新评分的质量证据。

## 4 空闲后顺序运行

`scripts/activegs/run_campaign.py` 在 Linux 服务器上使用标准库推进三个阶段：

| 阶段 | 实验范围 | 下一步条件 |
| --- | --- | --- |
| smoke | 6 次更新、2 次前缀，confidence/defect/refine 三方法、种子 0 | 完整评估、当前产物与事件记录验收通过 |
| observations | 60 次更新、20 次前缀，五方法 × 三种子，共 15 个分支 | 配对审计、汇总和网页导出通过 |
| time | 180 秒任务预算，三主方法 × 三种子，共 9 个分支 | 审计、汇总与网页导出通过 |

短实验不证明提升；正式结果尚待执行。阶段间重新检查空闲卡，启动 worker 和 benchmark 前再次准入。占用显存 ≥1,024 MiB 或利用率 >5% 的卡不会使用，不结束其他人的进程。

```bash
# plan/status 全只读，不创建状态文件、不读取 GPU、不导入重建库。
/opt/conda/bin/python scripts/activegs/run_campaign.py --plan
/opt/conda/bin/python scripts/activegs/run_campaign.py --status
# tick：有活 worker 就只检查；全忙就记录等待；空闲才启动一个阶段。
/opt/conda/bin/python scripts/activegs/run_campaign.py --tick
```

状态位于 `runs/campaigns/optimization-v1/state.json`，日志位于同目录 `logs/`。网页与推进器共用 `runs/launch.lock`，在同一启动事务中核查双方的记录、登记意图和保存进程句柄，避免在 CUDA 尚未分配显存的窗口同时启动。阶段状态另用文件锁保护，PID、进程启动时间与完整参数确认归属；进程身份在启动窗口暂时读不到时保留待确认状态，不因等待超时而重启。共享锁只协调本项目的网页和推进器，不能替代服务器跨用户调度。

开始实验后，主项目必须保持干净，源码身份整链冻结。实验中的日志和数据写入忽略目录；待全部阶段结束再提交结果说明。子程序失败、产物不完整或源码变化会停止后续阶段并保留现场，需要审查原因，不能自动重试或覆盖旧结果。GPU 在尚未启动重建的二次准入窗口变忙时，仅延期等待。

浏览器显示只读执行计划与实际进程状态。程序提供调度入口；定时调用和首次真实 GPU 闭环的完成状态，应以实际运行记录为准。

## 5 本轮 CPU 验收

在服务器隔离源码目录中隐藏全部 CUDA，执行当前完整检查集：**118 / 118 通过，没有跳过，CUDA 未初始化**。包括评分与规划接口、协议与随机域、当前文件审计、结果配对、输入环境、实验推进器、网页控制和导出；Linux 实际 `/proc` 和文件锁检查也已执行。见 [机器可读检查记录](evidence/cpu-preparation-checks-v2.json)。这些检查含合成数据和替代进程，不验证 CUDA 渲染、真实新方法闭环或重建提升。

网页控制新增检查覆盖 campaign 与网页启动互斥、worker 退出而 child 仍活、启动身份暂不可见、等待时释放启动锁、只读状态和内部字段过滤。新状态文件不存在时显示“尚未建立计划”；未知运行句柄显示待核查，不推断任务完成，也不重复启动。

独立 Linux 夹具另用真实 HTTP、`flock`、`Popen` 和 `/proc` 验证网页先启动、计划先启动和同时竞争三种顺序，每次都只有一个 CPU 替代 worker 启动。真实子进程等待期间能够取得启动锁，但已登记的活句柄继续阻挡新任务；退出码 7 被保留为失败。GPU 查询为替身，未导入重建库，见 [跨入口检查证据](evidence/cross-entry-cpu-v2.json)。
