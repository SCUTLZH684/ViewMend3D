# 主动重建系统调研与复用决策

检查日期：2026-09-30。范围：四个官方仓库的 README、源码、配置、运行与评估脚本，以及必要的固定版本第三方接口。

这是静态源码调研，不是论文结果复现。没有安装或运行四套系统；数值反例仅用于验证评估逻辑，不代表实际场景表现。本文保存决策依据，不保存安装过程、普通目录结构或临时操作记录。

## 1. 项目目标与证据等级

ViewMend3D 是本科《3D视觉》课程项目，验证链路为：

**当前重建几何缺陷 → 下一最佳视角 → 新观测 → 更新重建 → 几何质量改善。**

优先复用成熟实现；机器人控制、RL、位姿估计 SLAM 和路径规划不是当前验证的必要条件。

- **事实**：由检查版本源码、配置或官方材料直接支持。
- **推断**：根据实现判断的适配成本、局限或选型建议。
- **待验证**：需要项目确认、运行或实验才能成立的判断。

本文路线是候选建议，尚未形成已批准的架构决策。后续实际选型及实验结果应另行记录，并链接回本文证据。

## 2. 官方身份与版本

源码证据链接固定到以下完整 commit，避免默认分支变化影响结论。

| 系统 | 官方仓库与对应论文 | 分支与 commit |
| --- | --- | --- |
| ActiveGS | [dmar-bonn/active-gs](https://github.com/dmar-bonn/active-gs)；Liren Jin 等，*ActiveGS: Active Scene Reconstruction Using Gaussian Splatting*，RA-L 2025。[论文](https://arxiv.org/html/2412.17769v2) | `main`：`558121a00b2eca84d9851a609e99ebb8e26ea2d8` |
| NBV-Gym | [chengine/nbv_gym](https://github.com/chengine/nbv_gym)；Stanford 的 Nerfstudio NBV 插件，配套 *Coverage Optimization for Camera View Selection*（COVER，CVPR 2026）。[官方项目页](https://chengine.github.io/nbv_gym/) | `master`：`6e5e5ef48a59f369a821802842504668eac23325` |
| NARUTO | [oppo-us-research/NARUTO](https://github.com/oppo-us-research/NARUTO)；Ziyue Feng、Huangying Zhan 等，*Neural Active Reconstruction from Uncertain Target Observations*，CVPR 2024。[论文](https://arxiv.org/html/2402.18771v2) | `release`：`1e5672832e8dd1172cba760e63514d3b92e2d803` |
| ActiveGAMER | [oppo-us-research/ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER)；Liyan Chen、Huangying Zhan 等，*Active GAussian Mapping through Efficient Rendering*，CVPR 2025。[论文](https://arxiv.org/html/2501.06897v1) | `release`：`2b6dfebc1048cc25961823e9a6ca441fe2d03db3` |

**事实：**NBV-Gym 不是 GenNBV，也不是 RL Gym 环境。ActiveGAMER 内部的 `ActiveGSPlanner` 是其自己的规划器，不能按名称认定它采用 Bonn ActiveGS 的置信度算法。

## 3. 实际闭环能力

下表均为事实；“缺陷信号”是决策代理量，不是真实几何误差。

| 能力 | ActiveGS | NBV-Gym / COVER | NARUTO | ActiveGAMER |
| --- | --- | --- | --- | --- |
| 输入 | 已知位姿 RGB-D | 已有图像、相机参数、可选初始点云 | 已知位姿 RGB-D | 已知位姿 RGB-D |
| 状态表示 | 2D Gaussian surfels + 粗体素地图 | Splatfacto 3DGS + 观察方向统计 | Hash-grid / 神经 SDF + 显式不确定性网格 | SplaTAM 各向同性 3DGS + 占据地图 |
| 决策信号 | 低观测置信度、未知空间 | 已有 Gaussian 的方向覆盖不足 | 学习的深度相关不确定性 | 当前地图渲染缺失像素 |
| 候选视角 | 自由空间随机采样 + 朝向 ROI 的采样 | 未加入训练集的现有相机 | 目标位置网格 + 朝向高不确定点 | 新增自由体素位置 × 采样方向 |
| 加入观测 | Habitat 在 NBV 生成 RGB-D | 将已有候选图像加入 active set | Habitat 随移动/旋转生成 RGB-D | Habitat 随规划执行生成 RGB-D |
| 重建更新 | 新增 surfels，RGB-D / 法线损失优化 | 持续优化同一 3DGS | 更新 SDF、不确定性和关键帧射线库 | 增补 Gaussian，局部/全局关键帧优化 |
| 几何评估 | 渲染深度 → TSDF mesh → 表面距离 | 主训练评估为图像质量；几何链需补充 | 提取/裁剪 mesh，距离指标 + MAD | Gaussian 中心点云距离指标，阶段评估 |

**推断：**四者均不能直接替 ViewMend3D 证明“特定几何缺陷被新视角修复”。覆盖、置信度、不确定性和缺失像素都需与更新前后的独立几何指标建立联系。渲染变好不能替代几何质量改善。

NBV-Gym 的“新观测”是揭示已有图像，另三者生成新的仿真观测。两类实验都可以形成重建反馈，但不能混称为同一种在线采集实验。

## 4. ActiveGS：低置信度表面与定向补观测

### 4.1 核心数据流（事实）

| 边界 | 接口、行为与源码 |
| --- | --- |
| 闭环 | [`mapping/mapper.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/mapping/mapper.py)：`IncrementalMapper.run()` 规划 → 获取路径终点 RGB-D → 更新 Gaussian map → 更新 voxel map |
| 观测 | [`simulator/habitat_simulator.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/simulator/habitat_simulator.py)：`simulate(c2w)` 返回 `rgb, depth, extrinsic, intrinsic, depth_range`；`extrinsic` 实际是 camera-to-world |
| 更新 | [`mapping/gaussian_map.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/mapping/gaussian_map.py)：`update()` → `add_gaussians()` → `train()`；深度反投影初始化表面 Gaussian，优化 RGB、深度和法线一致性 |
| ROI | [`mapping/voxel_map.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/mapping/voxel_map.py)：`update_utility()` 合并 frontier 与低置信度、高 opacity Gaussian 所在体素 |
| 候选/选择 | [`planning/plan_base.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/planning/plan_base.py)：ROI/随机候选 → utility → 路径长度 → `cal_view_scores()` → argmax |
| 几何输出 | [`mesh_generation.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/mesh_generation.py)：`generate_mesh()` 将模型渲染深度融合成 TSDF mesh，不是直接把 Gaussian 当作 mesh |

### 4.2 置信度与评分（事实）

每个 Gaussian 保存观察次数、平均观察方向及 `view_scores`。后者累积距离与法线夹角贡献。启用方向分布时：

`confidence = clip(exp(1 - norm(mean_view_direction)) * view_scores, 0, 1)`。

禁用方向分布时使用 `1 - exp(-view_supports)`。它是观察充分程度的显式模型，不是学习出的真实误差预测器。

[`planning/confidence.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/planning/confidence.py) 将可见未探索体素比例与距离加权的渲染置信度不足组合，再在选择阶段扣除路径代价。默认 [`config/planner/confidence.yaml`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/config/planner/confidence.yaml) 的探索权重为 `1000.0`，路径代价系数为 `0.5`，配置目标为 100 个候选、最多配置 30 个 ROI 候选。不能将默认系统理解为只做局部几何修补。

### 4.3 复用边界（推断）

- 优先移植观察次数、方向、距离和法线形成的置信度，以及朝向低置信度 ROI 的候选生成。可迁移到点云、surfel 或 mesh patch，但属于低成本改写。
- 已采用 2DGS 时，增量 mapper 和 confidence rendering 值得深入。
- voxel graph、A* 路径、估算飞行时间及 GUI 不属于课程初版必需部分。
- Open3D TSDF 融合可参考，但原仓库只将它用于离线 mesh 提取；直接融合新 RGB-D 的更新循环需自行接入。

### 4.4 复现风险（事实与待验证）

- **事实：**[`utils/operations.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/utils/operations.py) 的 renderer 传入 `confidences`，取回 importance/count 等输出；[`envs/requirements.txt`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/envs/requirements.txt) 指向特制 `liren-jin/diff-gaussian-rasterization_2d`。不能直接替换成标准 gsplat。
- **事实：**开启 `has_missing_surface` 时，评分会查询候选位姿的仿真深度有效 mask；office0 启用了该配置。严格未知观测实验需披露或去除这项先验。
- **事实：**mapper 只加入路径终点观测；GUI 中途显示的路径帧不进入重建。
- **事实：**论文包含 UAV 实验，但公开 [`simulator/__init__.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/simulator/__init__.py) 仅返回 Habitat；未提供对应 UAV 采集/控制入口。
- **待验证：**高置信度但几何错误的表面是否能被发现；置信度是否预测局部误差下降；关闭探索项后的补全效果。

## 5. NBV-Gym / COVER：可控选图闭环

### 5.1 模块边界（事实）

| 边界 | 接口、行为与源码 |
| --- | --- |
| 调度 | [`nbv_gym/pipeline.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/pipeline.py)：`get_train_loss_dict(step)` 定期扩充 active set，随后继续训练 |
| 观测状态 | [`nbv_gym/datamanager.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/datamanager.py)：`active_train_indices` 与剩余候选；`expand_active_set()` 加入选中图像 |
| 选择器 | [`nbv_gym/view_selector.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/view_selector.py)：`select_views(active_indices, remaining_indices, num_to_select, ...)`，按 score 升序选择 |
| 模型接口 | [`nbv_gym/model.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/model.py)：`setup_view_metric()`、`update_view_attributes()`、`view_metric_score_for_camera()` |
| COVER | [`nbv_gym/util/coverage.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/util/coverage.py)：Gaussian 投影、观察方向 bin 累积和 coverage 计算 |

每个 Gaussian 默认有 128 个球面方向 bin。选择时通常清零统计，再按当前模型和所有 active cameras 重算。coverage 更新只检查投影视锥，不计算遮挡 transmittance；候选 coverage 图仍通过渲染聚合。

### 5.2 实际评分（事实）

活跃代码对有观察记录的 bin 取 `max((1 + dot(candidate_direction, bin_direction)) / 2)`；不是注释所说的“超过中位数的 bin 的角距离”。然后对 coverage 图中 `accumulation > 0` 的像素求平均，越低越优先。完全无 Gaussian 支持的背景像素不直接贡献该平均值。

KD-tree 过滤仅保留离最后加入相机较近的候选，不包含碰撞、机器人动力学或路径执行。标准更新损失主要是 RGB L1 + SSIM，主路径没有深度监督损失。

### 5.3 复用与风险

- **推断：**采用 Nerfstudio / 3DGS 时，active-set、调度、random/basic/all 对比和 selector 接口可以直接复用，适合接入自定义几何分数。
- **推断：**不采用 Nerfstudio 时只移植 COVER 思想；不宜为一个分数引入整套模型/渲染封装。已有表面的方向不足有信号，模型完全不存在的缺失表面可能缺少信号。
- **事实：**主训练评估使用 Splatfacto 图像指标。[`scripts/eval.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/scripts/eval.py) 的可选点云距离默认关闭，仍引用 `shadow_splat` 和不在同目录的 `load_model`，不是完整可靠的当前几何评估入口。
- **事实：**[`nbv_gym/config.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/nbv_gym/config.py) 默认 `load_3D_points=True`；固定版本 [Nerfstudio 父类](https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/pipelines/base_pipeline.py#L257) 将 parser 点云直接传入模型初始化，此处不按 active views 过滤。
- **待验证：**初始点云是否由全部候选图像生成；如果是，严格采集实验需仅由初始观测生成，或固定并披露这项先验。
- **事实：**[`scripts/sweep_scenes.py`](https://github.com/chengine/nbv_gym/blob/6e5e5ef48a59f369a821802842504668eac23325/scripts/sweep_scenes.py) 默认是 `view_fig`、一个初始视角、开启 KD-tree，不是 README 的 COVER 示例设置。
- **待验证：**Fisher-RF 分支依赖修改过的 rasterizer，本次未验证运行或数值等价性，不作为课程第一条路线。

## 6. NARUTO：SDF 与学习的不确定性

### 6.1 核心数据流（事实）

| 边界 | 接口、行为与源码 |
| --- | --- |
| 闭环 | [`src/naruto/main.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/naruto/main.py)：RGB-D → 重建 → uncertainty/SDF volume → 下一步位姿 |
| 更新接口 | [`src/slam/coslam/coslam.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/slam/coslam/coslam.py)：`online_recon_step(i,color,depth,c2w)` 返回新体积或 `None`；关键帧保存射线用于后续优化 |
| 学习信号 | [`model/scene_rep.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/slam/coslam/model/scene_rep.py)：显式网格插值和深度残差相关的不确定性损失 |
| 规划输入 | [`coslam_utils.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/slam/coslam/coslam_utils.py)：`get_map_volumes()` 返回 SDF 与正不确定性，规划不确定性仅保留 `0 <= sdf < 0.5` 区域 |
| 目标选择 | [`src/planner/naruto_planner.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/planner/naruto_planner.py)：`uncertainty_aggregation_v2()`、`goal_search_v2()` 选择目标位置与 look-at targets，再由 RRT/旋转状态机执行 |

不是训练通用 NBV policy，而是每场景优化重建和不确定性网格。候选聚合检查感知距离、安全 SDF，并在线段上采样 30 个点检查遮挡。

标准 [`configs/Replica/replica_coslam.yaml`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/configs/Replica/replica_coslam.yaml) 关闭 tracking，使用输入位姿；地图/关键帧默认每 5 步更新。使用 mapper 不意味着必须实现位姿估计 SLAM。

### 6.2 复用边界（推断）

已经采用神经 SDF 时，空间不确定性 → 可见性 → 候选聚合有参考价值。可改为有限相机池评分，但不是现成独立 scorer。Co-SLAM、tiny-cuda-nn、体积查询及导航状态机使整套复用成本较高；RRT 和旋转执行应从课程最小闭环中剥离。

### 6.3 必须核验的实现问题

1. **事实：active-ray sampler 选择方向与描述相反。**[`active_ray_sampler.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/slam/coslam/active_ray_sampler.py#L110) 使用 `np.argpartition(pts_uncert, K)[:K]`，取较低不确定性；输入是 Softplus 后的不确定性，不是负分数。索引转换还写死乘 `10`，绑定 0.1 m 网格。**推断：**不能原样复用来“重点优化高不确定区域”；对论文结果的影响待验证。
2. **事实：几何评估参数传反。**[`src/evaluation/eval_recon.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/evaluation/eval_recon.py) 调用 `calc_3d_mesh_metric(mesh_gt, mesh_rec)`，固定 [neural_slam_eval 子模块](https://github.com/JingwenWang95/neural_slam_eval/blob/9f07a513f647291d5523b8202ac07e9471803d78/eval_recon.py) 的签名却是 `(mesh_rec, mesh_gt)`。accuracy/completion 方向交换，`comp%` 方向也改变。
3. **事实：**[`scripts/evaluation/eval_replica.sh`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/scripts/evaluation/eval_replica.sh) 先裁剪重建 mesh；[`eval_mad.py`](https://github.com/oppo-us-research/NARUTO/blob/1e5672832e8dd1172cba760e63514d3b92e2d803/src/evaluation/eval_mad.py) 在 GT 表面采样 200,000 点，不是 README 所述全部顶点，距离换算也应结合 SDF 尺度解释。
4. **事实：**部分碰撞检查查询下一位姿的仿真全景深度，尤其 MP3D/自定义场景分支不只使用当前 SDF。

参数反置的无依赖反例：GT 为位置 0、1 的两点，重建只有位置 0；正确 `(accuracy, completion, completion_ratio)` 为 `(0.0, 0.5, 0.5)`，反置后为 `(0.5, 0.0, 1.0)`。已执行该逻辑反例；未执行仓库评估程序，也不能据此认定论文表格来自错误版本。

**待验证：**不确定性能否区分模型缺陷和噪声；修正采样、评估后能否复现指标；替换为固定候选池后的效果。

## 7. ActiveGAMER：增量 GS 与历史关键帧

### 7.1 核心模块（事实）

| 边界 | 接口、行为与源码 |
| --- | --- |
| 闭环 | [`src/main/activegamer.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/src/main/activegamer.py)：Habitat RGB-D → SplaTAM → exploration map → 下一步位姿 |
| 更新 | [`src/slam/splatam/splatam.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/src/slam/splatam/splatam.py)：`online_recon_step(time_idx,color,depth,c2w,force_map_update,dont_add_kf,only_use_global_keyframe)` |
| 候选空间 | [`exploration_map.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/src/slam/splatam/exploration_map.py)：从新确认自由体素生成位置 |
| 评分/选择 | [`src/planner/active_gs_planner.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/src/planner/active_gs_planner.py)：`rendering_based_planning()` 维护候选池，用缺失像素与距离权重选择 |
| 历史重放 | `splatam.py` 的 global keyframe 判定与训练采样：按新覆盖或低 PSNR 保留重要帧，混合局部重叠帧和全局帧优化 |

[`configs/Replica/office0/ActiveGAMER.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/configs/Replica/office0/ActiveGAMER.py) 使用 GT 位姿和两阶段粗到细探索。post-refinement 主要复用全局关键帧优化；停止条件是超过 90% 的相关帧达到 PSNR 阈值，不是几何改善阈值。

### 7.2 复用边界（推断）

最值得参考全局关键帧保留：更新局部时重放重要历史观测，检查原本好的区域是否退化。已采用 SplaTAM 时可深入增量 RGB-D 更新封装；另起后端时无需整体移植。粗到细候选池应在候选规模确实成为瓶颈后再引入，RRT、移动/旋转状态机和面向 PSNR 的长后处理不是初版必要部分。

### 7.3 实现与评估风险

- **事实：**`splatam.py:render()` 计算 silhouette threshold mask，但返回的有效 mask 是 `rendered_depth > 0`；planner 据此数缺失像素。不能把论文 silhouette 描述直接当作当前阈值实现。
- **事实：**候选评分查询仿真 depth，将 `depth <= 0.2` 的无效区域排除。严格闭环需要披露这项候选观测先验。
- **事实：**独立 `refinement` 分支明确标为 `NOT USED`；默认探索结束进入 post-refinement，不能据分支存在声称默认会主动补拍低质量视角。
- **事实：**[`eval_splatam_recon.py`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/src/evaluation/eval_splatam_recon.py) 提取 `means3D`，进行 1 cm 体素采样，再通过 `evenly_sample_points()` 根据 GT 体素中心查询最近预测点，仅评估选出的预测子集；不是 mesh 表面评估。
- **推断，并经数值反例确认：**远离 GT 的伪点可能被完全排除，因而不能充分惩罚额外伪几何。反例 GT 为位置 0、1，预测为 0、1、100：完整预测的平均 accuracy 距离为 `33.0`，GT 条件筛选后为 `0.0`。实际场景偏差大小待验证。
- **事实：**README 的 MAD/SDF 描述未接入默认 GS 评估链；其独立评估命令参数与脚本实际参数也不一致，应依据 [`scripts/evaluation/eval_replica_activegamer_recon.sh`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/scripts/evaluation/eval_replica_activegamer_recon.sh)。
- **事实：**SplaTAM 不是未经修改的上游版本；[`scripts/activegamer/update_splatam.sh`](https://github.com/oppo-us-research/ActiveGAMER/blob/2b6dfebc1048cc25961823e9a6ca441fe2d03db3/scripts/activegamer/update_splatam.sh) 覆盖子模块中的修改文件，复现需保留这层边界。
- **待验证：**缺失像素分数能否修复已经覆盖但错误的表面；post-refinement 的独立几何收益；取消候选仿真查询后的效果。

## 8. 跨仓库选型与工程经验

### 8.1 不应忽略的设计差异

| 决策 | 事实 | 对项目的推断 |
| --- | --- | --- |
| 观测模态 | NBV-Gym 主路径为 RGB；其余依赖 RGB-D | 先决定输入，不可将后端/评分混用后直接比较 |
| 几何表示 | ActiveGS 为表面 surfels，NARUTO 为 SDF，其余为 3DGS | 几何为主时表面/SDF/TSDF 的输出边界更直接；3DGS 需另外定义几何提取 |
| 决策目标 | ActiveGS 兼顾探索和表面置信度，COVER 偏方向覆盖，NARUTO 偏不确定表面，ActiveGAMER 偏缺失像素 | 局部修补不等于全场景探索；已有几何误差可能无法被覆盖代理量识别 |
| 优化策略 | NARUTO 同时改变训练射线，ActiveGAMER 改变关键帧重放 | 做消融，避免将训练策略收益归因于 NBV |
| 预算 | 梯度步/选图数、估算飞行时间、移动/旋转步数各不相同 | 统一新增观测数与优化预算，再附运行时间 |
| 先验边界 | 部分规划会查询未选候选的仿真深度；初始点云可能预含几何 | 未选候选原则上只访问位姿和当前模型；额外先验单独披露 |

接口经验：ActiveGS 使用 OpenCV/RDF camera-to-world 与归一化内参；NARUTO mapper 使用 RUB；ActiveGAMER 主循环转换到相对首帧的 RDF，部分 docstring 仍写 RUB。适配时必须检查实际调用端的变换，不能凭 `c2w` 或 `extrinsic` 名称相接。固定深度单位、像素内参尺度、世界坐标及模型坐标的转换。

### 8.2 几何评估应独立统一

| 系统 | 事实口径 |
| --- | --- |
| ActiveGS | TSDF mesh，两侧各采样 500,000 表面点；[`utils/evaluation_tool.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/utils/evaluation_tool.py) 实际调用 2 cm 阈值；[`utils/operations.py`](https://github.com/dmar-bonn/active-gs/blob/558121a00b2eca84d9851a609e99ebb8e26ea2d8/utils/operations.py) 的 accuracy/completion 为 cm，而 Chamfer 保留 m |
| NBV-Gym | 主链 RGB 渲染指标，需额外几何评估 |
| NARUTO | 裁剪后的 mesh，5 cm 阈值；包装调用参数顺序反置 |
| ActiveGAMER | GT 条件筛选的 Gaussian 中心点子集，5 cm 阈值 |

**推断：**不能直接横比四套输出中的 accuracy 或 `comp%`。应统一待评估几何、采样、阈值、裁剪、坐标和单位。预测点不能先按 GT 位置挑选再用来评价整体 accuracy。用不对称的小反例核验 GT→预测和预测→GT 两个方向；用额外离群点核验伪几何是否被惩罚。

## 9. 候选路线与未决问题

### 9.1 最值得深入的三条路线（推断，尚未选定）

| 优先级 | 路线 | 成熟组件与项目需补部分 |
| --- | --- | --- |
| A | RGB-D 增量几何后端 + ActiveGS 式低置信度/缺陷 ROI + 有限候选视角 | 借鉴置信度与定向候选；复用成熟 TSDF/表面重建。自行建立 ROI 误差、可见性评分与逐轮评估，省掉路径规划 |
| B | 已有 RGB 图像池 + NBV-Gym + COVER/random/自定义几何分数 | 直接复用 active-set 与训练调度；补独立几何评估，控制初始点云先验，开展可重复消融 |
| C | NARUTO mapper/不确定性思想 + 固定候选池 | 仅在 A/B 信号确实不足且需要连续 SDF 时深入；先核验采样方向、体素尺度和评估调用，省掉 RRT |

ActiveGAMER 作为 A/B 的增量优化与历史关键帧保留参考，不另开完整导航系统路线。A 最贴近几何修补；B 最容易复用现成闭环；C 由前两者的实验不足触发。

### 9.2 必须确认与验证

- [ ] 观测是 RGB 还是 RGB-D？候选是已有图像池还是可任意生成仿真视角？位姿是否已知？
- [ ] 第一阶段修补对象是缺面、表面位置误差、法线不一致还是伪几何？低置信度仅作代理缺陷，不能当 GT。
- [ ] 重建输出如何转换为可评估几何？建立固定 GT 采样、坐标、阈值与 ROI；同时看全局和局部质量。
- [ ] 各方法固定相同初始观测、候选池、新增观测数、随机种子和优化预算；固定或披露初始点云先验。
- [ ] 至少比较 random、coverage/confidence、自定义缺陷分数，以及“只继续优化、不加新观测”。
- [ ] 固定 mapper 和训练采样后比较 NBV，随后单独消融历史重放或主动射线采样。
- [ ] 每轮保存缺陷位置、候选分数、选中位姿、新观测、更新前后几何指标；检查 ROI 改善及其他区域退化。
- [ ] 先在一个具有 GT、明确人为缺陷的简单场景验证闭环，再扩展场景复杂度。

只有上述实验支持“缺陷信号 → 有效观测 → 独立几何指标改善”，才能将代理分数的作用写成项目已验证结论。
