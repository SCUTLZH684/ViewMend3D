// Display-only descriptions. Budget, scoring and metric values remain in the recorded artifacts.
export const METHOD_EXPLANATIONS = {
  confidence: '原作者复现：按置信度等信息选视角，保留作者的候选真值有效深度掩码。用于验证原流程能运行，单独阅读。',
  confidence_nooracle: '公平基线：根据当前地图的未探索区域、置信度和移动代价选择下一视角。规划时不查询候选的真实 RGB-D。',
  random_matched: '随机对照：沿用相同候选生成与移动代价，在匹配的采样规则下选择视角，用来检查主动评分的作用。',
  defect: 'v1 几何评分：把当前地图的几何不一致信号加入候选评分。已有完整对照未显示稳定质量提升。',
  defect_no_gate: 'v1 门控消融：移除深度跳变过滤，检查边界附近的几何信号如何影响选点。',
  defect_guarded: 'v2 固定主方法：保留基线分数，再给当前地图中几何不一致的区域少量、有上限的奖励。检验能否补足重建；奖励不保证真实质量提升。',
  defect_guarded_no_gate: 'v2 门控消融：保留有界奖励，移除深度跳变过滤。它是机制对照，不替代预先固定的主方法。',
  refine_only: '仅优化对照：前 20 次采集后保持相机观测不变，继续优化已有数据。60 次更新仍只有 20 次观测。',
  suite: 'v1 三方法组合：分别运行 Confidence、Random 和旧 Defect，使用同一公共前缀；此组合不包含 v2 主方法。'
};

export function runCategory(id, method) {
  const v2 = [
    ['heldout_observations', 'v2 · 留出 · 60次观测'], ['heldout_time', 'v2 · 留出 · 180秒任务预算'],
    ['development_observations', 'v2 · 开发 · 60次观测'], ['development_time', 'v2 · 开发 · 180秒任务预算'],
    ['smoke', 'v2 · 短闭环验收']
  ];
  for (const [key, label] of v2) if (id.startsWith(`campaign-optimization-v2-${key}-`)) return label;
  if (id.startsWith('campaign-observations-') || id.startsWith('campaign-optimization-v1-observations-')) return 'v1 · 固定60次更新';
  if (id.startsWith('campaign-time-') || id.startsWith('campaign-optimization-v1-time-')) return 'v1 · 180秒任务预算';
  if (id.startsWith('campaign-smoke-') || id.startsWith('campaign-optimization-v1-smoke-')) return 'v1 · 短闭环验收';
  if (id.startsWith('web-')) return '手动启动 · 单次实验';
  return method === 'confidence' ? '原作者流程 · 独立复现' : '其他已完成实验';
}

export function budgetExplanation(data) {
  const definition = data.protocol?.protocol || data.protocol || {};
  if (!data.method_id || data.method_id === 'confidence') return '原作者任务时间预算；候选真值掩码开启，随机种子未固定。';
  const time = ['time', 'mission_time'].includes(definition.mode) || definition.budget_type === 'mission_time';
  if (time) return `累计任务时间达到 ${definition.seconds ?? '记录中指定的'} 秒后，在完整更新结束时停止；不同方法可能采集不同数量的图像。`;
  if (data.method_id === 'refine_only') return `${definition.observations ?? 60} 次建图更新，其中仅前 ${definition.prefix ?? 20} 次采集 RGB-D，其后反复优化已有观测。`;
  return `${definition.observations ?? '记录中指定的'} 次观测与建图更新；前 ${definition.prefix ?? '记录中指定的'} 次使用共同采集前缀。`;
}
