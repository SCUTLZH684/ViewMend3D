import { FROZEN } from './v2-results-schema.js';

// Independent results card: it neither imports nor changes the live 3D,
// historical-v1 or experiment-launch controller.
const BASELINE = 'confidence_nooracle', PRIMARY = 'defect_guarded';
export const GROUPS = {
  heldout_observations: '留出 · 固定60次观测', heldout_time: '留出 · 180秒任务预算',
  development_observations: '开发 · 固定60次观测', development_time: '开发 · 180秒任务预算'
};
const LABELS = { confidence_nooracle: 'Confidence（无候选真值）', defect_guarded: 'Guarded v2（固定主方法）',
  defect: 'Defect v1（本轮开发对照）', defect_guarded_no_gate: 'Guarded v2 no gate（消融）' };
const METRICS = [['accuracy_cm', '表面偏差 Accuracy / cm ↓', 3], ['completion_cm', '漏建距离 Completion / cm ↓', 3],
  ['coverage_percent', '2 cm 覆盖率 / % ↑', 3], ['chamfer_mm', '双向平均距离 Chamfer / mm ↓', 3]];
const METRIC_INFO = {
  accuracy_cm: { name: '表面偏差', term: 'Accuracy', unit: 'cm', direction: '越低越好',
    meaning: '从重建表面找最近的参考表面，衡量建出来的位置偏了多远。' },
  completion_cm: { name: '漏建距离', term: 'Completion', unit: 'cm', direction: '越低越好',
    meaning: '从参考表面找最近的重建表面，衡量真实区域离已建表面有多远。' },
  coverage_percent: { name: '2 cm 覆盖率', term: 'Coverage', unit: '%', direction: '越高越好',
    meaning: '参考表面采样点中，距重建表面小于 2 cm 的比例。' },
  chamfer_mm: { name: '双向平均距离', term: 'Chamfer', unit: 'mm', direction: '越低越好',
    meaning: '表面偏差与漏建距离的平均值，以毫米表示，兼顾偏移和遗漏。' }
};
const COSTS = [['mission_seconds', '计入任务预算 / s'], ['wall_seconds', '重建墙钟实测 / s'], ['planning_seconds', '选择下一视角 / s'],
  ['mapping_seconds', '更新三维地图 / s'], ['sensor_seconds', '取得观测 / s'], ['diagnostic_seconds', '记录诊断 / s'],
  ['simulated_flight_seconds', '估算移动时间 / s'], ['path_length_m', '移动路径长度 / m'], ['observations', '采集观测 / 次'],
  ['events', '建图更新 / 次'], ['optimizer_steps', '地图优化 / 步'], ['additional_optimizer_steps', '公共前缀后优化 / 步'],
  ['peak_torch_allocated_mb', 'PyTorch 显存峰值 / MiB'], ['meshing_evaluation_wall_seconds', '生成网格与评估 / s']];
const POST = ['planning_seconds', 'mapping_seconds', 'sensor_seconds', 'diagnostic_seconds', 'events', 'observations'];
const FALLBACKS = { weight_zero: 'β=0', base_all_zero: '基础效用全零', single_reachable: '仅一个可达候选',
  weak_geometry: '几何信号不足', constant_geometry: '几何项恒定' };
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const integer = value => Number.isSafeInteger(value) && value >= 0;
const digest = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const text = (value, limit = 4096) => typeof value === 'string' && value.length > 0 && value.length <= limit;
const require = (condition, reason) => { if (!condition) throw new Error(reason); };
const equal = (a, b) => {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => equal(v, b[i]));
  if (!record(a) || !record(b)) return false;
  const keys = Object.keys(a).sort();
  return equal(keys, Object.keys(b).sort()) && keys.every(key => equal(a[key], b[key]));
};
const keysEqual = (value, expected) => record(value) && equal(Object.keys(value).sort(), [...expected].sort());
const near = (a, b) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-9 + 1e-10 * Math.max(Math.abs(a), Math.abs(b));
const sum = values => values.reduce((total, value) => total + value, 0);

function validateStatistic(stat, values) {
  require(record(stat) && stat.n === values.length && finite(stat.mean), '统计样本数或均值缺失');
  const mean = sum(values) / values.length;
  require(near(stat.mean, mean), '均值未对应完整逐种子数据');
  if (values.length === 1) require(stat.sample_std === null, '单样本不得伪造标准差');
  else require(finite(stat.sample_std) && stat.sample_std >= 0
    && near(stat.sample_std, Math.sqrt(sum(values.map(value => (value - mean) ** 2)) / (values.length - 1))), '样本SD不对应逐种子数据');
}

function validatePaired(stat, values, reference, seeds) {
  const deltas = values.map((value, index) => value - reference[index]);
  validateStatistic(stat, deltas);
  require(keysEqual(stat.method_minus_baseline_by_seed, seeds.map(String)) && keysEqual(stat.relative_percent_by_seed, seeds.map(String)), '配对差缺失种子');
  seeds.forEach((seed, index) => {
    require(near(stat.method_minus_baseline_by_seed[seed], deltas[index]), '配对差不是同种子方法减Confidence');
    const relative = reference[index] ? 100 * deltas[index] / reference[index] : null;
    require(relative === null ? stat.relative_percent_by_seed[seed] === null : near(stat.relative_percent_by_seed[seed], relative), '相对差记录不一致');
  });
  require(stat.negative_difference_seeds === deltas.filter(value => value < 0).length
    && stat.positive_difference_seeds === deltas.filter(value => value > 0).length, '配对差符号计数不一致');
}

function validateGuarded(stage, method, seeds, byPair) {
  const diagnostics = stage.diagnostics_per_run.filter(item => item.method === method);
  let validated = 0, applied = 0, signal = 0, changed = 0, regretSum = 0, utility = 0, maximum = 0;
  const fallbacks = Object.fromEntries(Object.keys(FALLBACKS).map(key => [key, 0]));
  for (const item of diagnostics) {
    const events = byPair.get(`${method}:${item.seed}`).cost.events - stage.protocol.prefix;
    require(item.validated_events === events && integer(events), 'Guarded后缀诊断事件不完整');
    for (const field of ['signal_active_events', 'discriminative_events', 'reward_applied_events', 'selection_changed_events']) {
      require(integer(item[field]) && item[field] <= events, 'Guarded诊断计数无效');
    }
    // Weak but varying residuals may be discriminative without reaching the
    // minimum peak. Reward requires both flags; those flags are independent.
    require(item.selection_changed_events <= item.reward_applied_events && item.reward_applied_events <= item.discriminative_events
      && item.reward_applied_events <= item.signal_active_events, 'Guarded改选/奖励/信号计数矛盾');
    require(keysEqual(item.fallback_counts, Object.keys(FALLBACKS)) && Object.values(item.fallback_counts).every(integer)
      && sum(Object.values(item.fallback_counts)) === events - item.reward_applied_events, 'Guarded回退原因不完整');
    require(item.signal_active_events <= events - item.fallback_counts.weak_geometry
      && item.discriminative_events <= events - item.fallback_counts.constant_geometry, 'Guarded信号标志与回退原因矛盾');
    for (const field of ['maximum_baseline_regret', 'baseline_regret_sum', 'utility_seconds']) {
      require(finite(item[field]) && item[field] >= 0, 'Guarded regret/耗时无效');
    }
    require(item.maximum_baseline_regret <= 0.1 + 3e-6
      && item.baseline_regret_sum <= events * item.maximum_baseline_regret + 1e-9, 'Guarded regret超出有界方案或汇总矛盾');
    validated += events; applied += item.reward_applied_events; signal += item.signal_active_events;
    changed += item.selection_changed_events; regretSum += item.baseline_regret_sum; utility += item.utility_seconds;
    maximum = Math.max(maximum, item.maximum_baseline_regret);
    for (const key of Object.keys(FALLBACKS)) fallbacks[key] += item.fallback_counts[key];
  }
  require(diagnostics.length === seeds.length && keysEqual(stage.guarded_signal_statistics, Object.keys(stage.method_statistics).filter(key => key.startsWith('defect_guarded'))), 'Guarded诊断方法/种子不完整');
  const actual = stage.guarded_signal_statistics[method];
  require(record(actual) && actual.validated_events === validated && actual.reward_applied_events === applied
    && actual.signal_active_events === signal && actual.selection_changed_events === changed
    && equal(actual.fallback_counts, fallbacks), 'Guarded公开汇总与逐seed诊断不一致');
  require(near(actual.utility_seconds_total, utility), 'Guarded诊断耗时汇总不一致');
  if (!validated) {
    require(actual.event_weighted_mean_baseline_regret === null && actual.maximum_baseline_regret === null
      && equal(actual.bonus_cap_range, []) && actual.maximum_regret_over_cap === null, '无后缀事件时不可伪造regret/奖励上限统计');
    return;
  }
  require(near(actual.maximum_baseline_regret, maximum)
    && near(actual.event_weighted_mean_baseline_regret, regretSum / validated), 'Guarded事件加权regret不一致');
  require(Array.isArray(actual.bonus_cap_range) && actual.bonus_cap_range.length === 2
    && actual.bonus_cap_range.every(value => finite(value) && value >= 0 && value <= 0.1)
    && actual.bonus_cap_range[0] <= actual.bonus_cap_range[1]
    && maximum <= actual.bonus_cap_range[1] + 3e-6, 'Guarded奖励上限或regret无效');
  require(actual.maximum_regret_over_cap === null || (finite(actual.maximum_regret_over_cap) && actual.maximum_regret_over_cap >= 0), 'Guarded regret/cap无效');
  if (actual.bonus_cap_range[0] > 0 && actual.maximum_regret_over_cap !== null) {
    require(actual.maximum_regret_over_cap <= 1 + 3e-6 / actual.bonus_cap_range[0] + 1e-9, 'Guarded regret/cap超过实现容差');
  }
}

export function validateV2Summary(data) {
  require(record(data) && data.version === 'viewmend-v2-summary-v1' && data.status === 'complete'
    && data.campaign === 'optimization-v2' && data.scene === 'replica/office0', 'v2完整结果尚未发布');
  require(!('fixture_kind' in data), '合成CPU fixture不能作为实验结果发布');
  require(data.completed_campaign_stages === 5 && data.validated_campaign_experiments === 27 && data.quality_runs === 24
    && data.primary_method === PRIMARY && equal(data.development_seeds, [0, 1]) && equal(data.heldout_seeds, [3, 4, 5]), '必须是完整5阶段27分支和固定主方法/种子');
  require(data.campaign_spec_sha256 === FROZEN.campaign_spec_sha256
    && equal(data.campaign_project_source_identity, FROZEN.source_identity),
  'v2预注册spec或五项冻结源码指纹不匹配');
  require(data.process_observation_platform === 'linux', '实际进程退出门控缺失');
  require(text(data.generated_at, 128) && Number.isFinite(Date.parse(data.generated_at)) && /(?:Z|[+-]\d{2}:\d{2})$/.test(data.generated_at), '审计时间缺失时区');
  require(equal(data.units, { accuracy_cm: 'cm', completion_cm: 'cm', coverage_percent: '% / paired pp', chamfer_mm: 'mm' }), '指标单位不是本轮原始单位');
  require(record(data.evidence) && data.evidence.schema_sha256 === FROZEN.schema_sha256
    && ['analysis_sha256', 'generator_sha256', 'source_input_manifest_sha256'].every(key => digest(data.evidence[key]))
    && integer(data.evidence.source_input_count) && data.evidence.source_input_count > 0, 'fresh最终分析/发布证据不完整');
  require(!('source_input_sha256' in data), '公开快照仍含内部逐输入路径');
  require(Array.isArray(data.limitations) && data.limitations.length >= 6 && data.limitations.length <= 30
    && data.limitations.every(value => text(value)), '公开局限声明不完整');
  require(keysEqual(data.stages, FROZEN.profile.stages.map(stage => stage.name)), '五阶段缺失或混入其它轮实验');
  let total = 0, shared;
  for (const spec of FROZEN.profile.stages) {
    const stage = data.stages[spec.name], expectedProtocol = FROZEN.protocols[spec.name];
    require(record(stage) && stage.stage === spec.name && stage.partition === (spec.name === 'smoke' ? 'smoke' : spec.name.split('_')[0])
      && stage.status === 'completed_and_freshly_audited' && stage.fresh_artifact_audit === true
      && stage.saved_paired_report_verified_against_current_files === true, '阶段未通过fresh完整审计');
    require(equal(stage.seeds, spec.seeds) && equal(stage.protocol, expectedProtocol)
      && stage.scene === 'replica/office0' && stage.recipe === 'optimization-v2'
      && stage.recipe_sha256 === expectedProtocol.recipe_sha256 && stage.campaign_spec_sha256 === FROZEN.campaign_spec_sha256,
    '阶段种子/预算/完整recipe被替换');
    require(new RegExp(`^${spec.name}-[0-9a-f]{32}$`).test(stage.run_id) && finite(stage.pipeline_wall_seconds) && stage.pipeline_wall_seconds >= 0, '阶段标识/流水线耗时缺失');
    for (const key of ['scene_mesh_sha256', 'scene_assets_sha256', 'context_hash']) require(digest(stage[key]), '场景/配置指纹缺失');
    require(record(stage.source_versions) && record(stage.evaluation) && stage.evaluation.points_per_surface === 500000
      && stage.evaluation.threshold_m === 0.02, '固定评估身份缺失');
    const identity = Object.fromEntries(['source_versions', 'evaluation', 'scene_mesh_sha256', 'scene_assets_sha256'].map(key => [key, stage[key]]));
    if (shared) require(equal(identity, shared), '五阶段混合来源、评估或场景资产'); else shared = identity;
    require(keysEqual(stage.method_statistics, spec.methods) && keysEqual(stage.paired_difference_vs_confidence, spec.methods.filter(method => method !== BASELINE)), '阶段预注册方法缺失或被替换');
    require(Array.isArray(stage.per_run) && stage.per_run.length === spec.methods.length * spec.seeds.length, '阶段逐seed分支数不完整');
    const byPair = new Map();
    for (const row of stage.per_run) {
      require(record(row) && spec.methods.includes(row.method) && spec.seeds.includes(row.seed) && integer(row.seed), '未知方法或种子');
      const key = `${row.method}:${row.seed}`;
      require(!byPair.has(key), '重复method×seed'); byPair.set(key, row);
      for (const [family, fields] of [['metrics', METRICS.map(([field]) => field)], ['cost', COSTS.map(([field]) => field)], ['post_prefix_cost', POST]]) {
        require(keysEqual(row[family], fields) && fields.every(field => finite(row[family][field]) && row[family][field] >= 0), '逐seed质量/成本字段缺失或无效');
      }
      require(row.metrics.coverage_percent <= 100 && near(row.metrics.chamfer_mm, (row.metrics.accuracy_cm + row.metrics.completion_cm) * 5), '覆盖率或距离单位关系错误');
      const cost = row.cost, post = row.post_prefix_cost;
      for (const field of ['observations', 'events', 'optimizer_steps', 'additional_optimizer_steps']) require(integer(cost[field]), '采集或优化计数无效');
      require(cost.events === cost.observations && cost.events >= spec.prefix && cost.optimizer_steps === cost.events * 10
        && cost.additional_optimizer_steps === (cost.events - spec.prefix) * 10
        && post.events === cost.events - spec.prefix && post.observations === cost.observations - spec.prefix, '观测/事件/前缀后成本混淆');
      require(spec.mode === 'observations' ? cost.events === spec.frames : cost.mission_seconds >= spec.seconds, '分支未达到预注册预算');
      require(near(cost.mission_seconds, cost.planning_seconds + cost.mapping_seconds + cost.simulated_flight_seconds), '任务时间计量不一致');
      for (const field of POST.slice(0, 4)) require(post[field] <= cost[field] + 1e-7, '后缀成本大于完整成本');
      require(digest(row.prefix_sha256) && digest(row.prefix_camera_sha256) && text(row.wall_time_scope) && text(row.memory_scope), '真实前缀或成本适用范围缺失');
    }
    for (const seed of spec.seeds) {
      require(new Set(spec.methods.map(method => `${byPair.get(`${method}:${seed}`).prefix_sha256}:${byPair.get(`${method}:${seed}`).prefix_camera_sha256}`)).size === 1,
      '同seed的方法不是同一真实前缀/相机');
    }
    for (const method of spec.methods) {
      require(record(stage.method_statistics[method]) && (method === BASELINE || record(stage.paired_difference_vs_confidence[method])), '方法或配对统计对象缺失');
      for (const [family, fields] of [['metrics', METRICS.map(([field]) => field)], ['cost', COSTS.map(([field]) => field)], ['post_prefix_cost', POST]]) {
        require(keysEqual(stage.method_statistics[method][family], fields), '统计字段不完整');
        fields.forEach(field => {
          const values = spec.seeds.map(seed => byPair.get(`${method}:${seed}`)[family][field]);
          validateStatistic(stage.method_statistics[method][family][field], values);
          if (method !== BASELINE) {
            require(keysEqual(stage.paired_difference_vs_confidence[method][family], fields), '配对统计字段不完整');
            validatePaired(stage.paired_difference_vs_confidence[method][family][field], values,
              spec.seeds.map(seed => byPair.get(`${BASELINE}:${seed}`)[family][field]), spec.seeds);
          }
        });
      }
    }
    require(Array.isArray(stage.diagnostics_per_run) && stage.diagnostics_per_run.every(record) && stage.diagnostics_per_run.length === byPair.size
      && new Set(stage.diagnostics_per_run.map(item => `${item.method}:${item.seed}`)).size === byPair.size
      && stage.diagnostics_per_run.every(item => byPair.has(`${item.method}:${item.seed}`) && !('events' in item)), '公开诊断分支缺失或仍含逐事件rows');
    spec.methods.filter(method => method.startsWith('defect_guarded')).forEach(method => validateGuarded(stage, method, spec.seeds, byPair));
    const exports = spec.methods.flatMap(method => spec.seeds.map(seed => `campaign-optimization-v2-${stage.run_id}-${method}-s${seed}`));
    require(Array.isArray(stage.verified_exports) && equal([...stage.verified_exports].sort(), exports.sort()), '真实导出ID未通过完整审计');
    total += stage.per_run.length;
  }
  require(total === 27, '整链未验收27条');
  return data;
}

const el = (tag, className, content) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
};
const fmt = (value, digits = 3, signed = false) => {
  if (value === null) return '未定义';
  const clean = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
  return `${signed && clean > 0 ? '+' : ''}${clean.toFixed(digits)}`;
};
const statText = (value, digits = 3, signed = false) => `${fmt(value.mean, digits, signed)} ± ${fmt(value.sample_std, digits)}`;
function table(title, headers, rows) {
  const scroll = el('div', 'v2-scroll'); scroll.tabIndex = 0; scroll.setAttribute('role', 'region'); scroll.setAttribute('aria-label', title);
  const table = el('table', 'v2-table'), head = el('thead'), heading = el('tr'), body = el('tbody');
  headers.forEach(label => { const th = el('th', '', label); th.scope = 'col'; heading.append(th); }); head.append(heading);
  rows.forEach(values => {
    const row = el('tr');
    values.forEach((value, index) => { const cell = el(index ? 'td' : 'th', '', value); if (!index) cell.scope = 'row'; row.append(cell); });
    body.append(row);
  });
  table.append(head, body); scroll.append(table); return scroll;
}
function detail(title, content) {
  const node = el('details', 'v2-details'); node.append(el('summary', '', title), content); return node;
}
const note = content => el('p', 'v2-note', content);
const heading = content => el('h3', 'v2-subheading', content);

const improves = (key, difference) => key === 'coverage_percent' ? difference > 0 : difference < 0;
const worsens = (key, difference) => key === 'coverage_percent' ? difference < 0 : difference > 0;
const meanChange = (key, difference) => {
  const info = METRIC_INFO[key], unit = key === 'coverage_percent' ? '个百分点' : info.unit;
  return `${info.name}平均${difference === 0 ? '不变' : `${difference > 0 ? '增加' : '减少'} ${fmt(Math.abs(difference), 5)} ${unit}`}`;
};

// The conclusion is generated from the same audited, paired values shown in
// the tables. It cannot silently substitute an ablation or a selected seed.
export function describeV2Group(stage, group) {
  const paired = stage.paired_difference_vs_confidence[PRIMARY].metrics;
  const seeds = stage.seeds, n = seeds.length;
  const betterCoverage = seeds.filter(seed => improves('coverage_percent', paired.coverage_percent.method_minus_baseline_by_seed[seed]));
  const worseAccuracy = seeds.filter(seed => worsens('accuracy_cm', paired.accuracy_cm.method_minus_baseline_by_seed[seed]));
  const allWorse = seeds.filter(seed => METRICS.every(([key]) => worsens(key, paired[key].method_minus_baseline_by_seed[seed])));
  const betterCompletion = seeds.filter(seed => improves('completion_cm', paired.completion_cm.method_minus_baseline_by_seed[seed]));
  const lines = [
    `相对 Confidence 基线，${meanChange('completion_cm', paired.completion_cm.mean)}，${meanChange('coverage_percent', paired.coverage_percent.mean)}。`,
    `${meanChange('accuracy_cm', paired.accuracy_cm.mean)}；${meanChange('chamfer_mm', paired.chamfer_mm.mean)}。`,
    `漏建距离在 ${betterCompletion.length}/${n} 次运行中改善；覆盖率${betterCoverage.length < n ? '仅' : '在'} ${betterCoverage.length}/${n} 次运行中提高${betterCoverage.length ? `（seed ${betterCoverage.join('、')}）` : ''}。`
  ];
  if (worseAccuracy.length) lines.push(`表面偏差在 seed ${worseAccuracy.join('、')} 中退化，需要同时看“建得准”和“建得全”。`);
  if (allWorse.length) lines.push(`seed ${allWorse.join('、')} 的四项质量指标全部退化，平均值没有展示出这一单次风险。`);
  lines.push(group.startsWith('heldout')
    ? '这是 office0 同场景的新种子测试，样本量为 3；局部平均改善尚不足以证明稳定优化或跨场景泛化。'
    : '这是用于方案检查的开发种子，样本量为 2；开发结果不替换预先固定的主方法，也不证明稳定优化。');
  return lines;
}

function resultReading(stage, group) {
  const block = el('section', 'v2-result-reading');
  block.append(el('h3', '', '这组实验说明什么？'));
  const lines = describeV2Group(stage, group);
  lines.forEach((line, index) => block.append(el('p', index > 2 ? 'v2-reading-limit' : '', line)));
  return block;
}

function metricGuide(stage) {
  const cards = el('div', 'v2-metric-guide');
  METRICS.forEach(([key]) => {
    const info = METRIC_INFO[key], paired = stage.paired_difference_vs_confidence[PRIMARY].metrics[key];
    const base = stage.method_statistics[BASELINE].metrics[key], primary = stage.method_statistics[PRIMARY].metrics[key];
    const card = el('section', 'v2-metric-explanation'), unit = key === 'coverage_percent' ? 'pp' : info.unit;
    const favorable = improves(key, paired.mean), unfavorable = worsens(key, paired.mean);
    card.append(el('h4', '', info.name), el('p', 'v2-metric-term', `${info.term} · ${info.unit} · ${info.direction}`),
      el('p', 'v2-metric-meaning', info.meaning),
      el('p', `v2-metric-difference ${favorable ? 'improved' : unfavorable ? 'regressed' : ''}`, `${fmt(paired.mean, 5, true)} ${unit}`),
      el('p', 'v2-metric-direction', `平均配对差 · ${favorable ? '改善方向' : unfavorable ? '退化方向' : '没有变化'}`),
      el('p', 'v2-metric-means', `基线 ${fmt(base.mean, 5)} → Guarded ${fmt(primary.mean, 5)} ${info.unit}`));
    cards.append(card);
  });
  return cards;
}

function readingKey() {
  const content = el('div', 'v2-reading-key');
  content.append(el('p', '', '配对差 Δ = 同一个 seed 的 Guarded − Confidence。距离指标负值更好；覆盖率正值更好。'),
    el('p', '', 'pp 是百分点：95% → 96% = +1 pp。它与“相对提高 1%”含义不同。'),
    el('p', '', '均值 ± SD：SD 是不同随机种子结果的离散程度（样本标准差，n−1），不是误差界或置信区间。'),
    el('p', '', 'seed 是随机运行编号。每个编号对应一组配对实验；不要只挑表现最好的那次运行。'));
  return content;
}

function methodTables(stage, methods) {
  const block = el('div');
  block.append(heading('质量原值：各方法的均值 ± 样本 SD'), table('质量均值和样本SD', ['方法', ...METRICS.map(([, label]) => label)], methods.map(method =>
    [LABELS[method], ...METRICS.map(([key, , digits]) => statText(stage.method_statistics[method].metrics[key], digits))])));
  block.append(heading('配对差：相同随机种子下，方法 − Confidence'), note('以下对每个 seed 先求差，再算差的均值和 SD。距离负值更好；覆盖率正值更好，单位为百分点 pp。'),
    table('同种子配对质量差', ['方法', 'Δ表面偏差 / cm', 'Δ漏建距离 / cm', 'Δ覆盖率 / pp', 'Δ双向距离 / mm'],
      methods.filter(method => method !== BASELINE).map(method => [LABELS[method], ...METRICS.map(([key, , digits]) => statText(stage.paired_difference_vs_confidence[method].metrics[key], digits, true))])));
  const seedHeaders = ['方法', 'seed', '观测', ...METRICS.map(([, label]) => label)];
  const seedRows = methods.flatMap(method => stage.seeds.map(seed => {
    const row = stage.per_run.find(item => item.method === method && item.seed === seed);
    return [LABELS[method], String(seed), String(row.cost.observations), ...METRICS.map(([key, , digits]) => fmt(row.metrics[key], digits + 2))];
  }));
  block.append(detail('查看每个 seed 的质量原值与配对差', (() => {
    const group = el('div'); group.append(table('完整逐种子质量', seedHeaders, seedRows),
      table('完整逐种子配对质量差', ['方法 − Confidence', 'seed', 'Δ表面偏差 / cm', 'Δ漏建距离 / cm', 'Δ覆盖率 / pp', 'Δ双向距离 / mm'],
        methods.filter(method => method !== BASELINE).flatMap(method => stage.seeds.map(seed => [LABELS[method], String(seed),
          ...METRICS.map(([key, , digits]) => fmt(stage.paired_difference_vs_confidence[method].metrics[key].method_minus_baseline_by_seed[seed], digits + 2, true))])))); return group;
  })()));
  return block;
}

function costTables(stage, methods) {
  const content = el('div');
  const guide = el('div', 'v2-cost-guide');
  [
    ['任务预算时间', '选择下一视角＋更新地图＋按路径估算的移动时间。180 秒组在完成整次观测/更新后判断停止，因此最终值会略超过 180 秒。'],
    ['重建墙钟实测', '重建阶段实际耗时，包含公共前缀、采集、规划、建图及诊断/检查点写入；不含启动、前缀缓存保存/加载、网格生成与评估。它与任务预算不是同一个计时器。'],
    ['观测数与更新数', '本轮每次取得一个新视角观测，随后更新地图；每次更新执行 10 步地图优化。时间预算相同也可能得到不同观测数量。'],
    ['PyTorch 显存峰值', '只统计 PyTorch 分配，不含 Habitat/OpenGL 的占用；不是整张 GPU 的用量，也不是运行所需的最低显存。']
  ].forEach(([title, explanation]) => {
    const card = el('div'); card.append(el('strong', '', title), el('p', '', explanation)); guide.append(card);
  });
  content.append(guide, note('如果采集次数更少，时间和显存可能随之减少；这不能单独证明“同等质量、同等工作量”的加速。网格生成、质量评估与导出分别在建图之后执行。'));
  const fields = ['mission_seconds', 'wall_seconds', 'planning_seconds', 'mapping_seconds', 'observations', 'path_length_m', 'peak_torch_allocated_mb'];
  const costs = keys => table('实际成本均值和样本SD', ['方法', ...keys.map(key => COSTS.find(([field]) => field === key)[1])], methods.map(method =>
    [LABELS[method], ...keys.map(key => statText(stage.method_statistics[method].cost[key], key.includes('steps') ? 1 : 3))]));
  content.append(costs(fields), costs(COSTS.map(([key]) => key).filter(key => !fields.includes(key))));
  content.append(heading('成本配对差：相同 seed 的方法 − Confidence'), note('成本差保留各自单位。耗时更少要结合观测数和质量一起判断；观测数更多也不直接代表重建更好。'), table('实际方法成本配对差', ['方法 − Confidence', '成本指标', ...stage.seeds.map(seed => `seed ${seed} Δ`), '均值 ± SD'],
    methods.filter(method => method !== BASELINE).flatMap(method => COSTS.map(([key, label]) => {
      const value = stage.paired_difference_vs_confidence[method].cost[key]; return [LABELS[method], label,
        ...stage.seeds.map(seed => fmt(value.method_minus_baseline_by_seed[seed], 3, true)), statText(value, 3, true)]; }))));
  content.append(detail('查看每个 seed 的采集次数、耗时与显存', table('逐种子实际成本', ['方法', 'seed', '观测 / 次', '选择视角 / s', '任务预算计时 / s', '重建墙钟 / s', 'PyTorch 峰值 / MiB'],
    methods.flatMap(method => stage.seeds.map(seed => { const value = stage.per_run.find(row => row.method === method && row.seed === seed).cost;
      return [LABELS[method], String(seed), String(value.observations), fmt(value.planning_seconds), fmt(value.mission_seconds), fmt(value.wall_seconds), fmt(value.peak_torch_allocated_mb)]; })))));
  const suffix = el('div');
  suffix.append(note('“公共前缀”是各方法共享的前 20 次真实采集。下表只计其后的新观测与更新成本；与上方包含前缀的完整成本分开阅读。'),
    table('前缀后实际成本均值和样本SD', ['方法', ...POST.map(key => COSTS.find(([field]) => field === key)[1])],
      methods.map(method => [LABELS[method], ...POST.map(key => statText(stage.method_statistics[method].post_prefix_cost[key]))])),
    table('前缀后实际成本逐种子配对差', ['方法 − Confidence', '成本指标', ...stage.seeds.map(seed => `seed ${seed} Δ`), '均值 ± SD'],
      methods.filter(method => method !== BASELINE).flatMap(method => POST.map(key => {
        const value = stage.paired_difference_vs_confidence[method].post_prefix_cost[key];
        return [LABELS[method], COSTS.find(([field]) => field === key)[1],
          ...stage.seeds.map(seed => fmt(value.method_minus_baseline_by_seed[seed], 3, true)), statText(value, 3, true)]; }))));
  content.append(detail('查看公共前 20 次观测之后的成本', suffix));
  return content;
}

function signalTables(stage, methods) {
  const content = el('div');
  content.append(note('这里回答“几何奖励是否参与了视角选择”。S0 是基础评分，S2 是加入有界奖励后的评分；比较对象始终是 Guarded 自己当前地图上的同一组候选视角。局部首选改选不等于与独立 Confidence 实验的轨迹差，也不等于质量改善。'));
  const guarded = methods.filter(method => method.startsWith('defect_guarded'));
  for (const method of guarded) {
    const value = stage.guarded_signal_statistics[method];
    content.append(heading(LABELS[method]), note(`公共前缀后共 ${value.validated_events} 次诊断：几何信号有效 ${value.signal_active_events} 次，奖励参与评分 ${value.reward_applied_events} 次，同次候选集S0→S2首选改选 ${value.selection_changed_events} 次。`),
      note(`基础评分损失 regret（无单位）：事件加权平均 ${fmt(value.event_weighted_mean_baseline_regret, 6)}，最大 ${fmt(value.maximum_baseline_regret, 6)}；奖励上限 ${value.bonus_cap_range.length ? value.bonus_cap_range.map(number => fmt(number, 6)).join('–') : '无后缀事件，未定义'}；最大损失/上限 ${fmt(value.maximum_regret_over_cap, 5)}。`),
      note('regret 衡量改选后牺牲了多少基础代理分数，与上方以 cm/mm 计量的网格误差不同。奖励上限只能约束代理分数损失，不能保证真实质量提高。评分耗时包含渲染、可见性判断与计算。'),
      table('实际回退原因分布', ['回退原因', '事件数'], Object.entries(FALLBACKS).map(([key, label]) => [label, String(value.fallback_counts[key])])),
      table('逐种子Guarded信号', ['seed', '诊断 / 次', '奖励评分 / 次', '局部首选改选 / 次', '平均基础损失', '最大基础损失', '评分耗时 / s'],
        stage.diagnostics_per_run.filter(row => row.method === method).map(row => [String(row.seed), String(row.validated_events), String(row.reward_applied_events),
          String(row.selection_changed_events), fmt(row.validated_events ? row.baseline_regret_sum / row.validated_events : null, 6), fmt(row.maximum_baseline_regret, 6), fmt(row.utility_seconds)])));
  }
  return content;
}

export function mountV2Results(root, { fetcher = fetch, url = '/data/optimization-v2-summary.json' } = {}) {
  if (!root) return;
  const top = el('div', 'panel-heading'), title = el('div'); title.append(el('h2', '', 'v2 优化验证：是否建得更准、更全？')); top.append(title);
  const badge = el('span', 'badge', '读取中'); top.append(badge);
  const status = note('正在读取整链完成后发布的v2结果…'); status.setAttribute('aria-live', 'polite');
  const content = el('div'); content.hidden = true; root.replaceChildren(top, status, content);
  let observer;

  function render(data, group) {
    const stage = data.stages[group], n = stage.seeds.length;
    const select = el('select'); select.id = 'v2-results-group'; select.setAttribute('aria-label', '选择独立开发或留出预算组');
    Object.entries(GROUPS).forEach(([key, title]) => { const option = el('option', '', title); option.value = key; select.append(option); }); select.value = group;
    const toolbar = el('div', 'v2-toolbar'), label = el('label', '', '选择要比较的实验组'); label.htmlFor = select.id; toolbar.append(label, select);
    select.addEventListener('change', () => render(data, select.value));
    const source = el('div', 'v2-evidence');
    const audited = new Date(data.generated_at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
    [ `5/5阶段、27/27分支完整验收；质量表仅用24正式分支。审计 ${audited}（北京时间）。`,
      `本组真实批次：${stage.run_id}；实验源 ${FROZEN.source_commit}。`,
      `最终分析SHA256：${data.evidence.analysis_sha256}；fresh输入清单 ${data.evidence.source_input_count} 项。` ]
      .forEach(line => source.append(el('p', '', line)));
    source.append(heading('冻结实验源码与配置指纹'));
    Object.entries(FROZEN.source_identity).forEach(([key, value]) => source.append(el('p', '', `${key}：${value}`)));
    source.append(el('p', '', `本轮协议 SHA256：${data.campaign_spec_sha256}；配方 SHA256：${stage.recipe_sha256}。`));
    const link = el('a', 'v2-report-link', '查看完整 v2 报告与全部种子 ↗'); link.href = 'https://github.com/SCUTLZH684/ViewMend3D/blob/main/docs/reproduction/optimization-v2-results.md'; link.target = '_blank'; link.rel = 'noopener';
    const preview = el('div', 'v2-previews'), previewNote = note('真实网格入口尚未就绪；等实验列表加载后可选择对应导出。');
    const buttons = [];
    stage.seeds.forEach(seed => {
      const pair = el('div', 'v2-preview-pair'); pair.append(el('strong', '', `seed ${seed} · 同种子对照`));
      [BASELINE, PRIMARY].forEach(method => {
        const id = `campaign-optimization-v2-${stage.run_id}-${method}-s${seed}`;
        const methodLabel = method === BASELINE ? 'Confidence 基线' : 'Guarded v2 主方法';
        const button = el('button', `v2-preview-button ${method === PRIMARY ? 'primary-method' : ''}`, `${methodLabel} · 看网格`);
        require(stage.verified_exports.includes(id), '当前组未验证此真实导出'); button.dataset.exportId = id; button.disabled = true;
        button.setAttribute('aria-label', `${GROUPS[group]}，seed ${seed}，${methodLabel}，查看真实网格和采集轨迹`);
        button.addEventListener('click', () => {
          const picker = document.getElementById('run-select');
          if (!picker || ![...picker.options].some(option => option.value === id)) return;
          picker.value = id; picker.dispatchEvent(new Event('change', { bubbles: true }));
          document.getElementById('reconstruction')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }); pair.append(button); buttons.push(button);
      });
      preview.append(pair);
    });
    const refreshPreviews = () => {
      const picker = document.getElementById('run-select'), ids = new Set(picker ? [...picker.options].map(option => option.value) : []);
      buttons.forEach(button => { button.disabled = !ids.has(button.dataset.exportId); });
      previewNote.textContent = buttons.every(button => !button.disabled)
        ? `以上入口对应“${GROUPS[group]}”。点击后，页面上方切换为该方法/seed 的真实网格、采集轨迹与诊断。逐个切换可以对照建图范围和视角选择。`
        : '部分真实导出尚未出现在实验列表；对应入口保持禁用。';
    };
    observer?.disconnect();
    const picker = document.getElementById('run-select');
    if (picker && typeof MutationObserver !== 'undefined') { observer = new MutationObserver(refreshPreviews); observer.observe(picker, { childList: true }); }
    content.replaceChildren(toolbar,
      note('我们在检查：给 Confidence 的基础视角评分加上有上限的当前地图几何奖励后，Guarded v2 能否减少遗漏，并保持表面位置准确。本轮受控规划不读取候选真值 RGB-D、掩码或网格误差；选定视角后才取得新的仿真观测。'),
      note(`${GROUPS[group]} · seeds ${stage.seeds.join('/')} · n=${n}。同一 seed 的前 20 次真实观测与地图由两方法共享，之后各自规划与重建。开发 n=2 与留出 n=3 分开统计，两种预算也不合并。`),
      resultReading(stage, group), metricGuide(stage), detail('怎么看配对差、百分点 pp 和样本 SD？', readingKey()),
      methodTables(stage, [BASELINE, PRIMARY]), heading('打开本组的真实三维结果'), preview, previewNote,
      detail('质量改善花了多少成本？查看计时、观测数与显存', costTables(stage, [BASELINE, PRIMARY])),
      detail('几何奖励实际做了什么？查看局部改选与评分损失', signalTables(stage, [PRIMARY])));
    if (group === 'development_observations') {
      const extra = el('div'); extra.append(note('旧Defect和no gate只在开发观测阶段提供机制对照。完整保留两者，不按消融赢家改换固定主方法。'),
        methodTables(stage, ['defect', 'defect_guarded_no_gate']), costTables(stage, ['defect', 'defect_guarded_no_gate']), signalTables(stage, ['defect_guarded_no_gate']));
      content.append(detail('展开本轮开发对照与门控消融（两者均保留）', extra));
    }
    source.append(note('种子留出仍限 office0；小样本描述统计不证明显著或跨场景泛化。候选评分只用当前地图，共有已知包围盒先验；本轮受控规划不读取候选真值 RGB-D、掩码或网格误差。独立分支可能因地图/候选及原生渲染数值差异产生不同轨迹，具体成因尚未确定，不能把所有最终质量差异都归因于几何奖励。'));
    const limitations = el('ul', 'v2-limitations'); data.limitations.forEach(value => limitations.append(el('li', '', value)));
    source.append(detail('查看最终审计记录的全部局限', limitations));
    const publication = el('div', 'v2-publication'); publication.append(note('已完成 5/5 阶段、27/27 分支；其中 24 条正式分支用于四组质量对照，短闭环不加入质量结论。'), link);
    content.append(publication, detail('查看数据来源、冻结源码、完整审计与解释边界', source));
    refreshPreviews();
  }

  const ready = (async () => {
    try {
      const response = await fetcher(url, { signal: AbortSignal.timeout(20000), cache: 'no-cache' });
      if (response.status === 404) { badge.textContent = '尚未发布'; status.textContent = '尚未发布整链fresh审计后的v2汇总。已有单次真实网格与v1历史结果仍可查看。'; return { status: 'unpublished' }; }
      require(response.ok, `服务返回 ${response.status}`);
      const data = validateV2Summary(await response.json());
      render(data, 'heldout_observations'); badge.textContent = '27分支 · 已审计'; status.hidden = true; content.hidden = false;
      return { status: 'published', data };
    } catch (error) {
      observer?.disconnect(); content.replaceChildren(); content.hidden = true;
      badge.textContent = '汇总不可确认'; status.hidden = false; status.textContent = `无法确认v2整链结果：${error.message}。本次不显示汇总数值。`;
      return { status: 'unavailable', reason: error.message };
    }
  })();
  return { ready, destroy: () => observer?.disconnect() };
}

if (typeof document !== 'undefined') mountV2Results(document.getElementById('v2-results'));
