# ViewMend3D 受控实验运行说明

本入口实现[v1归档框架](optimization-framework.md)及[v2预注册协议](optimization-v2.md)。它与 `run_original.sh` 分开，禁止规划阶段查询候选真值，使用持久化公共采集前缀，保存种子、配置、源码摘要、完整指标和成本。v1完整GPU闭环已通过：3短实验＋15固定观测＋9时间预算分支；正式结果未显示稳定质量提升，见[v1报告](reproduction/optimization-v1-results.md)。v2另完成五阶段27分支、全部导出和fresh审计，四组24正式分支见[v2报告](reproduction/optimization-v2-results.md)。

不占 GPU 的效率优化、当前文件审计和等待空闲后的顺序推进入口，见 [CPU 准备记录](reproduction/cpu-preparation-v2.md)。

## v2独立计划

本轮已按[v2预注册框架](optimization-v2.md)完成，新方法为`defect_guarded`及仅去除深度门控的`defect_guarded_no_gate`。旧方法和默认v1入口保留。v2完整recipe与campaign spec进入公共前缀、每个分支及聚合报告；β固定0.1，未按开发质量或消融结果更换主方法、参数或种子。

```bash
# 只读计划；不查询显卡、不创建run、不导入Torch/Habitat。
/opt/conda/bin/python scripts/activegs/run_campaign.py --campaign optimization-v2 --plan
# 每次至多推进一个阶段；活句柄/未知意图时只核查，本轮completed后不重跑。
/opt/conda/bin/python scripts/activegs/run_campaign.py --campaign optimization-v2 --tick
/opt/conda/bin/python scripts/activegs/run_campaign.py --campaign optimization-v2 --status
# 必须27分支与5阶段完成；未完成时返回pending并拒绝生成质量汇总。
/opt/conda/bin/python scripts/activegs/analyze_v2_campaign.py --root /workspace/ViewMend3D \
  --output /workspace/ViewMend3D/setup/optimization-v2-reanalysis-new.json
```

顺序为3短实验、8开发固定观测、4开发固定时间、6留出固定观测、6留出固定时间。所有阶段的新baseline在本轮冻结源码上重新生成，共享真实前缀；开发n=2与留出n=3分别统计。源码与配置在整链中冻结，失败保留、不自动重试或覆盖，源变化则停止并审查。统计只覆盖office0，留出种子不表示跨场景验证。下文保留v1运行与结果说明。

实际历史实验源为 `fafb552400903cd083a34f3966bf5068a95344d2`，五阶段27条与27份网页导出全部完成，worker/child已退出。最终分析重读原始二进制、完整网格、采集相机、成本与诊断，fresh聚合与保存paired结果一致。留出Completion/覆盖率均值局部改善，但部分种子和Accuracy等指标退化；四组小样本描述统计不支持稳定提升或显著性结论。奖励日志只比较该方法自身当前地图上同次候选集的S0首选与S2首选，不能把独立分支的全部质量差因果归于奖励。

### CPU复现公开汇总与图表

clone后可直接消费已审阅的[完整最终分析](reproduction/evidence/optimization-v2-final-analysis.json)，无需原数据集、地图或GPU。该JSON仅含指标、成本、标量诊断与来源元数据；原字节SHA256为 `541b0effba87ae82928801a0ec2cfb26f4a5189674165c378a5ac7affadb61fd`。图表另需Matplotlib；去掉 `--plots` 时发布工具只用标准库。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/activegs/publish_v2_results.py \
  docs/reproduction/evidence/optimization-v2-final-analysis.json \
  --expected-sha256 541b0effba87ae82928801a0ec2cfb26f4a5189674165c378a5ac7affadb61fd \
  --output-dir runs/v2-cpu-reproduction-new --plots
```

输出中文报告、compact快照、输入指纹及四组独立质量/成本图。两个绘图与发布入口都要求可信完整分析和独立SHA，核对五项源码身份、固定recipe、五阶段27分支、统计/配对/诊断链。新输出目录不能已存在；仓库内仅允许 `runs/` 或 `setup/`，不会自动写docs/web或覆盖历史结果。若单独重画图，可用 `scripts/activegs/plot_v2_results.py`，保持同一输入和 `--expected-sha256`，指定另一个新目录。

公开JSON与SHA是已审阅实验的复现输入，自行标记complete或自算摘要不能证明真实GPU运行。重新审计原始产物仍使用上方 `analyze_v2_campaign.py`，输出须是尚不存在的新路径。

## 1 运行环境

先完成 [已验证的 ActiveGS 环境](reproduction/activegs-office0.md)。当前服务器布局中的上游、Python 和私有 EGL 库如下；其他机器需替换路径。

```bash
cd /workspace/ViewMend3D
export ACTIVEGS_ROOT="$PWD/external/active-gs"
export ACTIVEGS_PYTHON="$PWD/.envs/activegs/bin/python"
export LD_LIBRARY_PATH="$PWD/.sysroot/usr/lib/x86_64-linux-gnu:$PWD/.envs/activegs/lib:${LD_LIBRARY_PATH:-}"
```

CLI 在导入 CUDA 建图模块之前检查 GPU 空闲状态；显存占用 ≥ 1,024 MiB 或利用率 > 5% 时拒绝启动。不得用修改门限绕过共享服务器分配约定，也不得结束他人进程。数据、地图与公共前缀缓存留在服务器。

## 2 先检查计划，再执行短闭环

`--dry-run` 不分配 GPU、不创建实验目录，可以先检查参数和阶段。

```bash
"$ACTIVEGS_PYTHON" scripts/activegs/run_benchmark.py \
  --upstream "$ACTIVEGS_ROOT" --run-dir "$PWD/runs/office0-smoke-v1" \
  --methods confidence_nooracle defect refine_only --seeds 0 \
  --frames 6 --prefix-frames 2 --dry-run
```

确认有空闲 GPU 后，选择其物理编号执行。以下 `2` 是示例，需要按真实状态替换；`RUN_DIR` 必须尚不存在。

```bash
export GPU=2
export RUN_DIR="$PWD/runs/office0-smoke-v1"
export BENCHMARK_METHODS=confidence_nooracle,defect,refine_only
export BENCHMARK_SEEDS=0
export BENCHMARK_FRAMES=6
export BENCHMARK_PREFIX=2
bash scripts/activegs/run_benchmark.sh
```

短闭环只检查接入和产物，不用于证明质量提升。正式运行应使用未创建的新目录。

## 3 正式对照与消融

固定观测主协议为 60 次更新、20 次公共采集前缀、种子 0/1/2。算法分支从同一保存的 Gaussian 参数、体素地图、原始训练 RGB-D、相机和运动路径恢复；每个事件仍使用 10 个优化步。refine_only 的观测数一直保持 20，后 40 次只是优化事件。

```bash
export RUN_DIR="$PWD/runs/office0-observations-v1"
export BENCHMARK_METHODS=confidence_nooracle,random_matched,defect,defect_no_gate,refine_only
export BENCHMARK_SEEDS=0,1,2
export BENCHMARK_FRAMES=60
export BENCHMARK_PREFIX=20
export BENCHMARK_PROTOCOL=observations
bash scripts/activegs/run_benchmark.sh
```

同一个批次保存每个种子的公共前缀摘要，各分支引用其 SHA256。不同闭环分支只共享候选采样规则，后续地图分叉后不会拥有完全相同的候选位姿。随机对照也采用相同 ROI 与路径成本，区别仅为效用评分；它不是作者原 random 配置。

refine_only 屏蔽原 `post_processing` 中的新增视角计数及剪枝，只对已有观测追加梯度优化，避免把重复训练伪记为新观测。该对照用于分析新增观测的作用，不保证与新增观测分支的实际计算耗时一致。

次级时间协议按任务时间 60/120/180 秒保存，并强制保存最终阶段；停止条件不受固定观测协议的 60 次更新限制，另设 10,000 次事件的安全上限。若提前到达安全上限则失败，不能冒充完整时间对照。

```bash
export RUN_DIR="$PWD/runs/office0-time-v1"
export BENCHMARK_METHODS=confidence_nooracle,random_matched,defect
export BENCHMARK_SEEDS=0,1,2
export BENCHMARK_PROTOCOL=time
export BENCHMARK_BUDGET=180
bash scripts/activegs/run_benchmark.sh
```

任务时间为同步规划＋建图＋估算移动时间；传感器采集与诊断写盘单列，另记录重建墙钟和完整网格/评估流水线墙钟。峰值显存统计为 Torch 分配量，不包含 Habitat/OpenGL，不能误写成整个进程的总显存。该时间协议的9个GPU分支已实测通过，实际最终观测数和负结果见[结果报告](reproduction/optimization-v1-results.md)。

## 4 审计和汇总

```bash
python scripts/activegs/aggregate_benchmark.py \
  runs/office0-observations-v1 \
  --methods confidence_nooracle random_matched defect defect_no_gate refine_only \
  --expected-seeds 0 1 2 \
  --output runs/office0-observations-v1/paired-results.json
```

汇总器拒绝种子缺失、重复方法、预算混用、候选真值掩码、不同来源或评估器、同种子不同前缀、指标单位不一致，以及只优化却增加观测数的记录。输出 JSON 保留逐种子数值、均值、样本标准差和与 confidence 的配对差值；同名 Markdown 用于报告。三种子只做描述统计，不宣称统计显著。

`final_result.json` 的距离单位沿用作者定义：Accuracy/Completion 为 cm，Chamfer 为 m，2 cm 覆盖率为 %。浏览器将 Chamfer 转成 mm。`step` 与 `update_event` 表示更新事件，实际采集数见 `observation_count`。

## 5 网页展示

网页支持原作者时间预算入口，以及固定 60 次更新的公平入口。可选择单一方法或三种主方法的同种子对照；完整三种子和两个消融建议用 CLI 批量执行。启动请求和实际执行前都检查 GPU 状态，失败时保留日志并显示失败。

对于 CLI 批次，逐个导出完成的分支：

```bash
"$ACTIVEGS_PYTHON" scripts/web/export_run.py \
  runs/office0-observations-v1 runs/web-assets/office0-defect-s0 \
  --experiment experiments/benchmark/replica/office0/defect/0
```

前端显示方法、种子、协议、更新次数与实际观测数，按真实协议和前缀摘要标记可配对结果。原作者含候选真值掩码的结果单列。只有真实导出的诊断才显示缺陷热力图；未提供诊断或尚未执行的实验不会生成示意性指标。v1历史汇总保留，v2四组分别显示n=2/3及seed列表，并可进入真实网格；均值±样本SD不作为置信区间。一次独立HTTP启动的60观测公平任务已完成GPU重建、3检查点与导出，未拼入正式27分支。

分支还记录场景名称、包含纹理的场景输入指纹、场景网格内容哈希和解析配置上下文哈希。汇总拒绝跨种子混入不同场景或配置，并要求当前产物重新验收且对应实际事件、阶段、预算和成本；网页也将场景与配置身份纳入配对分组。

## 6 不分配 GPU 的检查

```bash
python tests/test_protocol.py
python tests/test_aggregate.py
python scripts/web/test_server.py
python scripts/web/test_export_run.py
# 以下评分检查需 Torch，可在已配置的环境中禁用 CUDA。
CUDA_VISIBLE_DEVICES='' "$ACTIVEGS_PYTHON" tests/test_scoring.py
CUDA_VISIBLE_DEVICES='' "$ACTIVEGS_PYTHON" scripts/activegs/check_prefix_cpu.py --upstream "$ACTIVEGS_ROOT"
CUDA_VISIBLE_DEVICES='' "$ACTIVEGS_PYTHON" scripts/activegs/check_environment_cpu.py --upstream "$ACTIVEGS_ROOT"
python tests/test_artifacts.py
python tests/test_campaign.py
```

这些CPU检查验证评分边界、信息隔离、记录与接口。完整GPU证据与当前文件fresh审计见[v1结果](reproduction/optimization-v1-results.md)和[v2结果](reproduction/optimization-v2-results.md)。v1用 `scripts/activegs/analyze_campaign.py`、v2用 `scripts/activegs/analyze_v2_campaign.py` 重新审计；以新输出路径执行，不覆盖既有结果。

`check_prefix_cpu.py` 是独立进程中的实际类状态恢复检查，使用 AST 提取上游类体以避开 LPIPS 模块导入的 CUDA 副作用；它不是完整模块导入或实际重建检查。
