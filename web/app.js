import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PLYLoader } from './vendor/PLYLoader.js';
import { METHOD_EXPLANATIONS, runCategory, budgetExplanation } from './explanations.js';
import { CaptureReplay } from './capture-replay.js';

const $ = id => document.getElementById(id);
const activeStates = new Set(['starting', 'running', 'exporting']);
let manifest, assetBase, selected = 0, loadVersion = 0, playing = false;
let state, polling = false, submitting = false, seenJobStatus;
let runMenuKey;
const runSummaries = new Map();
const methodLabels = {
  suite: 'v1 三方法 · Confidence / Random / Defect',
  confidence_nooracle: 'Confidence · 无未来掩码', random_matched: 'Random · 匹配采样',
  defect: 'Defect · 几何缺陷评分', defect_no_gate: 'Defect · 移除跳变门控',
  defect_guarded: 'Defect v2 · 有界几何奖励', defect_guarded_no_gate: 'Defect v2 · 有界奖励，移除深度门控',
  refine_only: 'Refine only · 仅优化已有观测', confidence: 'ActiveGS · 原作者流程'
};
let historicalSummary;
const summaryMethods = {
  observations: ['confidence_nooracle', 'random_matched', 'defect', 'defect_no_gate', 'refine_only'],
  time: ['confidence_nooracle', 'random_matched', 'defect']
};
const summaryMetrics = [['accuracy_cm', 3], ['completion_cm', 3], ['coverage_percent', 2], ['chamfer_mm', 2]];
const summaryCosts = [['mission_seconds', 1], ['wall_seconds', 1], ['planning_seconds', 1], ['mapping_seconds', 1],
  ['observations', 1], ['path_length_m', 1], ['peak_torch_allocated_mb', 0]];
let renderer, scene, camera, controls, mesh, grid, pathLine, cameraLines;
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const viewer = $('viewer');
const loader = new PLYLoader();
const captureReplay = new CaptureReplay($('capture-replay'));
const replayOption = document.createElement('option');
replayOption.value = 'replay'; replayOption.textContent = '逐帧演示 · 8次采集';
$('protocol').prepend(replayOption);
$('protocol').setAttribute('aria-label', '实验协议');
const replayAnchor = document.createElement('a'); replayAnchor.href = '#capture-replay';
replayAnchor.textContent = '采集 → 选点 → 补拍';
document.querySelector('.read-nav').append(replayAnchor);

function showError(element, message) {
  element.textContent = message || '';
  element.hidden = !message;
}

async function fetchJSON(url, options = {}, timeout = 20000) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeout) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `服务返回 ${response.status}`);
  return result;
}

function setupViewer() {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.localClippingEnabled = true;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewer.prepend(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
  camera.up.set(0, 0, 1);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.4;
  controls.maxDistance = 40;
  scene.add(new THREE.HemisphereLight(0xf4f8ff, 0xc2c8d1, 2));
  const light = new THREE.DirectionalLight(0xffffff, 1.6);
  light.position.set(1, -3, 8);
  scene.add(light);
  const resize = () => {
    const { width, height } = viewer.getBoundingClientRect();
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(viewer);
  resize();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    $('viewer-loading').hidden = false;
    $('viewer-loading').textContent = '三维渲染连接中断，请刷新页面。';
  });
}

function resetView() {
  if (!manifest || !camera) return;
  const [min, max] = manifest.bounds;
  const span = Math.max(...max.map((value, i) => value - min[i]));
  const center = new THREE.Vector3(...min).add(new THREE.Vector3(...max)).multiplyScalar(0.5);
  center.z = min[2] + (max[2] - min[2]) * 0.3;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(span * 1.05, -span * 1.3, span * 1.15));
  camera.lookAt(center);
  controls.update();
}

function disposeObject(object) {
  if (!object) return;
  scene.remove(object);
  object.geometry?.dispose();
  if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
  else object.material?.dispose();
}

function updateClipping() {
  if (!manifest || !mesh) return;
  const [min, max] = manifest.bounds;
  clipPlane.constant = min[2] + (max[2] - min[2]) * Number($('cut-height').value) / 100;
  mesh.material.clippingPlanes = $('cutaway').checked ? [clipPlane] : [];
  mesh.material.needsUpdate = true;
  $('cut-height').disabled = !$('cutaway').checked;
}

function drawCameras(checkpoint) {
  disposeObject(pathLine);
  disposeObject(cameraLines);
  const count = checkpoint.camera_count;
  const stop = manifest.path_stops[count - 1] || 0;
  const positions = manifest.trajectory.slice(0, stop).flat();
  const pathGeometry = new THREE.BufferGeometry();
  pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  pathLine = new THREE.Line(pathGeometry, new THREE.LineBasicMaterial({ color: 0x4de0c3, transparent: true, opacity: 0.85, depthTest: false }));
  pathLine.renderOrder = 2;
  pathLine.visible = $('show-path').checked;
  scene.add(pathLine);
  const segments = [];
  for (const item of manifest.cameras.slice(0, count)) {
    const p = item.position, r = item.rotation;
    const depth = 0.11, halfX = depth * 0.5 / item.intrinsic[0], halfY = depth * 0.5 / item.intrinsic[4];
    const corners = [[-halfX, -halfY, depth], [halfX, -halfY, depth], [halfX, halfY, depth], [-halfX, halfY, depth]]
      .map(q => [p[0] + r[0] * q[0] + r[1] * q[1] + r[2] * q[2],
                 p[1] + r[3] * q[0] + r[4] * q[1] + r[5] * q[2],
                 p[2] + r[6] * q[0] + r[7] * q[1] + r[8] * q[2]]);
    for (let i = 0; i < 4; i++) segments.push(...p, ...corners[i], ...corners[i], ...corners[(i + 1) % 4]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
  cameraLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xf3c46e, transparent: true, opacity: 0.8, depthTest: false }));
  cameraLines.visible = $('show-cameras').checked;
  cameraLines.renderOrder = 3;
  scene.add(cameraLines);
}

function metricValue(id, value, unit, digits) {
  $(id).replaceChildren(document.createTextNode(value.toFixed(digits)));
  const label = document.createElement('em');
  label.textContent = unit;
  $(id).append(label);
}

const observationCount = checkpoint => checkpoint.observation_count ?? checkpoint.camera_count;
const updateEvent = checkpoint => checkpoint.update_event ?? checkpoint.step;
const seedLabel = data => Number.isInteger(data.seed) ? `种子 ${data.seed}` : '种子未固定';
const originalRun = data => !data.method_id || data.method_id === 'confidence';
function protocolLabel(data) {
  if (data.protocol?.replay_recording) return '独立演示 · 8帧过程回放';
  if (originalRun(data)) return '原作者流程 · 时间预算';
  const final = data.checkpoints?.at(-1) || data.final;
  const definition = data.protocol?.protocol || data.protocol || {};
  const fixedTime = definition.budget_type === 'mission_time' || definition.mode === 'mission_time' || definition.mode === 'time';
  return fixedTime ? '公平对照 · 时间预算' : `公平对照 · ${updateEvent(final)} 次更新`;
}

function renderMetadata(data) {
  $('result-method').textContent = data.method;
  const exportId = $('run-select').value;
  $('result-scope').textContent = `正在回放已完成结果 · ${runCategory(exportId, data.method_id)}；最近启动任务的状态在右侧单独显示。`;
  $('experiment-meta').textContent = `${runCategory(exportId, data.method_id)} · ${data.method} · ${seedLabel(data)}`;
  const explain = document.createElement('p');
  explain.textContent = METHOD_EXPLANATIONS[data.method_id] || '此结果的方法细节请查看记录中的协议与来源。';
  const budget = document.createElement('p');
  const budgetTitle = document.createElement('strong'); budgetTitle.textContent = '本次预算：';
  budget.append(budgetTitle, document.createTextNode(budgetExplanation(data)));
  $('run-explanation').replaceChildren(explain, budget);
  const original = originalRun(data);
  const definition = data.protocol?.protocol || data.protocol || {};
  const lines = original ? ['作者原配置；候选真值有效深度掩码开启。', '原记录中的 run_id 是目录编号，随机种子未固定。']
    : [`协议 ${data.protocol?.version || definition.version || '未记录'} · ${seedLabel(data)}`,
       `共享前缀 ${definition.prefix ?? '未记录'} 次观测 · 候选 ${definition.candidate_count ?? '未记录'} / ROI 最多 ${definition.roi_count ?? '未记录'}`,
       `每次更新 ${definition.mapping_optimizer_steps_per_event ?? '未记录'} 步建图优化 · 候选未来深度查询关闭。`,
       `共享场景包围盒先验 · 配对组 ${data.comparison_id || '未记录'}`];
  $('protocol-description').replaceChildren(...lines.map(text => {
    const line = document.createElement('div'); line.textContent = text; return line;
  }));
  $('boundary-description').textContent = original
    ? '原作者 office0 流程使用候选真值有效深度掩码；本结果是执行证明，应与公平对照分开展示。'
    : '候选评分只使用当前地图，共享场景包围盒先验。前缀后各策略的轨迹会分叉，候选生成规则保持一致。';
  $('initial-rgb').closest('section').querySelector('p').textContent = data.replay
    ? '来自 Habitat 的首帧。上方过程回放展示本次独立演示的全部8帧 RGB-D。'
    : '来自 Habitat 仿真器的首帧。此历史实验未保存完整 RGB-D 序列。';
  $('chart-note').textContent = `点击圆点切换保存的网格。${data.method_id === 'refine_only' ? '此对照只采集20次，横轴更新次数包含对已有观测的继续优化。' : '这条曲线属于所选的单次实验；多种子方法对照在 v2 汇总中。'}`;
  renderComparisonTable(state?.runs || []);
}

function renderDiagnostic(checkpoint) {
  const diagnostic = checkpoint.diagnostic;
  const image = $('diagnostic-heatmap');
  image.hidden = !diagnostic?.heatmap;
  if (diagnostic?.heatmap) image.src = `${assetBase}/${diagnostic.heatmap}`;
  else image.removeAttribute('src');
  $('diagnostic-event').textContent = diagnostic ? `更新 ${diagnostic.event}` : '未导出';
  if (!diagnostic) {
    $('diagnostic-content').textContent = '此阶段未导出选中候选的几何诊断。';
    return;
  }
  const candidate = diagnostic.candidate || {};
  const lines = [`候选 ${diagnostic.selected_index + 1} / ${diagnostic.candidate_count}`,
    diagnostic.geometry_measured === false ? '此策略未计算几何项' : `几何项${diagnostic.geometry_active ? '已激活' : '未激活'}`];
  for (const [key, label] of [['E', '未探索比例 E'], ['U', '不确定性 U'], ['D', '几何缺陷 D'],
    ['final_score', '最终得分'], ['path_length', '路径长度 / m'], ['valid_fraction', '稳定有效比例']]) {
    if (Number.isFinite(candidate[key])) lines.push(`${label}：${candidate[key].toPrecision(4)}`);
  }
  if (Number.isFinite(diagnostic.heatmap_max)) lines.push(`热图范围 0–1 · 当前最大值 ${diagnostic.heatmap_max.toPrecision(4)}`);
  if (diagnostic.scoring_version === 'bounded_geometry_v2') {
    const reasons = { none: '信号已应用', weight_zero: '奖励关闭', base_all_zero: '基础效用为零',
      single_reachable: '只有一个可达候选', weak_geometry: '几何信号过弱', constant_geometry: '候选几何信号相同' };
    lines.push(`v2 奖励${diagnostic.geometry_active ? '已应用' : '回退至基线'} · ${reasons[diagnostic.geometry_fallback_reason] || '原因待核查'}`);
    if (Number.isFinite(diagnostic.geometry_bonus_cap)) lines.push(`奖励上限：${diagnostic.geometry_bonus_cap.toPrecision(4)}`);
    if (Number.isFinite(candidate.geometry_bonus)) lines.push(`选中候选奖励：${candidate.geometry_bonus.toPrecision(4)}`);
    if (Number.isFinite(diagnostic.baseline_regret)) lines.push(`基线代理分数损失：${diagnostic.baseline_regret.toPrecision(4)}`);
    lines.push(`相对此批候选的基线选择${diagnostic.selection_changed ? '发生变化' : '保持一致'}；分数约束不保证重建质量。`);
  }
  $('diagnostic-content').replaceChildren(...lines.map(text => {
    const line = document.createElement('div'); line.textContent = text; return line;
  }));
}

function configureLaunch() {
  const original = $('protocol').value === 'original';
  const replay = $('protocol').value === 'replay';
  const current = $('method').value;
  const keys = original ? ['confidence'] : replay ? ['defect_guarded'] : ['suite', 'defect_guarded', 'defect_guarded_no_gate', 'defect', 'confidence_nooracle', 'random_matched', 'defect_no_gate', 'refine_only'];
  $('method').replaceChildren(...keys.map(key => {
    const option = document.createElement('option'); option.value = key; option.textContent = methodLabels[key]; return option;
  }));
  if (keys.includes(current)) $('method').value = current;
  else if (!original) $('method').value = 'defect_guarded';
  $('budget-row').hidden = !original;
  $('seed-row').hidden = original;
  $('frames-row').hidden = original;
  $('frames-row').querySelector('option').textContent = replay ? '8 次采集 · 初始1帧 · 每帧保存网格' : '60 次 · 前 20 次为共享前缀';
  $('method-launch-explanation').textContent = METHOD_EXPLANATIONS[$('method').value];
  $('launch-note').textContent = original
    ? '预算是原作者累计任务时间。此流程含候选真值掩码，保留为独立复现入口。网格与评估耗时另计。'
    : replay ? '独立教学演示：实际保存8次RGB-D与逐帧网格，初始化1帧后使用Guarded v2选点。额外记录和评估增加耗时，不纳入正式质量统计。完成后在上方回放过程。'
    : $('method').value.startsWith('defect_guarded')
      ? '60次观测 / 60次更新，前20次为共同confidence采集前缀。这里启动一次新运行，不会自动加入已发布的24条正式质量数据。'
      : $('method').value === 'refine_only'
        ? '20次观测 / 60次更新：后40次只优化已有图像，不新增拍摄。'
        : $('method').value === 'suite'
          ? '将运行v1的Confidence、Random和旧Defect三种方法；每种60次观测，使用同一个20次观测前缀。该组合不含v2主方法。'
          : '60次观测 / 60次更新，前20次为共同前缀。规划只使用已有地图，选定视角后才采集新图像。';
}

async function showCheckpoint(index) {
  if (!manifest) return;
  selected = Math.max(0, Math.min(manifest.checkpoints.length - 1, index));
  const checkpoint = manifest.checkpoints[selected];
  const version = ++loadVersion;
  $('checkpoint').value = String(selected);
  $('checkpoint-description').textContent = `${checkpoint.time.toFixed(1)} 秒任务时间 · ${observationCount(checkpoint)} 个采集视角 · 更新 ${updateEvent(checkpoint)}`;
  $('viewer-step').textContent = `采集 ${observationCount(checkpoint)} · 更新 ${updateEvent(checkpoint)} / ${updateEvent(manifest.checkpoints.at(-1))}`;
  $('current-stage-reading').textContent = `${manifest.method} · ${seedLabel(manifest)} · 更新 ${updateEvent(checkpoint)}：已采集 ${observationCount(checkpoint)} 次 RGB-D，累计任务时间 ${checkpoint.time.toFixed(1)} s。下方四项只评估这个保存阶段的完整网格。`;
  $('mesh-caption').textContent = `原始 ${checkpoint.original.vertices.toLocaleString()} 顶点 / ${checkpoint.original.faces.toLocaleString()} 面`;
  metricValue('accuracy', checkpoint.accuracy_cm, 'cm', 3);
  metricValue('completion', checkpoint.completion_cm, 'cm', 3);
  metricValue('coverage', checkpoint.coverage_percent, '%', 2);
  metricValue('chamfer', checkpoint.chamfer_mm, 'mm', 2);
  [...$('checkpoint-labels').children].forEach((button, i) => button.classList.toggle('selected', i === selected));
  drawChart();
  renderDiagnostic(checkpoint);
  if (!renderer) return;
  if (mesh) mesh.visible = false;
  if (pathLine) pathLine.visible = false;
  if (cameraLines) cameraLines.visible = false;
  $('viewer-loading').hidden = false;
  $('viewer-loading').textContent = `正在加载阶段 ${selected + 1} / ${manifest.checkpoints.length}…`;
  try {
    const geometry = await loader.loadAsync(`${assetBase}/${checkpoint.mesh}`);
    if (version !== loadVersion) { geometry.dispose(); return; }
    // Open3D writes double-precision PLY coordinates; WebGL accepts Float32.
    for (const [name, attribute] of Object.entries(geometry.attributes)) {
      if (attribute.array instanceof Float64Array) {
        geometry.setAttribute(name, new THREE.BufferAttribute(new Float32Array(attribute.array), attribute.itemSize, attribute.normalized));
      }
    }
    geometry.computeVertexNormals();
    disposeObject(mesh);
    mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: !!geometry.getAttribute('color'),
      color: geometry.getAttribute('color') ? 0xffffff : 0xbbc5cf, side: THREE.DoubleSide, wireframe: $('wireframe').checked }));
    scene.add(mesh);
    updateClipping();
    drawCameras(checkpoint);
    $('viewer-loading').hidden = true;
  } catch (error) {
    if (version !== loadVersion) return;
    $('viewer-loading').textContent = `网格加载失败：${error.message}。可以切换阶段重试。`;
    playing = false;
    $('play').textContent = '▶ 播放';
  }
}

async function loadRun(id) {
  captureReplay.clear();
  playing = false;
  $('play').textContent = '▶ 播放';
  ++loadVersion;
  const currentVersion = loadVersion;
  showError($('page-error'), '');
  try {
    const data = await fetchJSON(`/assets/${encodeURIComponent(id)}/manifest.json`, {}, 60000);
    if (currentVersion !== loadVersion) return;
    manifest = data;
    runSummaries.set(id, { ...data, final: data.checkpoints.at(-1) });
    $('run-select').value = id;
    renderMetadata(data);
    $('path-label').textContent = data.trajectory_kind === 'motion_path' ? '运动轨迹' : '视角连线';
    assetBase = `/assets/${encodeURIComponent(id)}`;
    void captureReplay.load(data, assetBase);
    $('checkpoint').max = String(data.checkpoints.length - 1);
    $('checkpoint-labels').replaceChildren(...data.checkpoints.map((checkpoint, i) => {
      const button = document.createElement('button');
      button.textContent = data.plot_axis === 'update_event' ? `更新 ${updateEvent(checkpoint)}` : `${Math.round(checkpoint.time)} s`;
      button.setAttribute('aria-label', `查看更新 ${updateEvent(checkpoint)}，采集 ${observationCount(checkpoint)} 次的重建`);
      button.addEventListener('click', () => showCheckpoint(i));
      return button;
    }));
    $('initial-rgb').hidden = !data.initial_rgb;
    if (data.initial_rgb) $('initial-rgb').src = `${assetBase}/${data.initial_rgb}`;
    if (scene) {
      disposeObject(grid);
      grid = new THREE.GridHelper(12, 24, 0x425168, 0x38485b);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = data.bounds[0][2] - 0.04;
      grid.material.transparent = true;
      grid.material.opacity = 0.25;
      scene.add(grid);
      resetView();
    }
    await showCheckpoint(data.checkpoints.length - 1);
  } catch (error) { showError($('page-error'), `实验结果加载失败：${error.message}`); }
}

function drawChart() {
  if (!manifest) return;
  const choice = $('chart-metric').value;
  const series = choice === 'distance' ? [{ key: 'accuracy_cm', label: '表面偏差 Accuracy ↓', color: '#7975e0' }, { key: 'completion_cm', label: '漏建距离 Completion ↓', color: '#54bba5' }]
    : [{ key: choice, label: choice === 'coverage_percent' ? '2cm覆盖率 ↑' : '双向平均距离 Chamfer ↓', color: '#7975e0' }];
  const unit = choice === 'distance' ? 'cm' : choice === 'coverage_percent' ? '%' : 'mm';
  const rows = manifest.checkpoints;
  const values = series.flatMap(item => rows.map(row => row[item.key]));
  let low = Math.min(...values), high = Math.max(...values);
  const padding = Math.max((high - low) * 0.3, high * 0.01, 0.02);
  low = Math.max(0, low - padding); high += padding;
  const width = 640, left = 53, right = 618, top = 20, bottom = 142;
  const updates = manifest.plot_axis === 'update_event';
  const axis = row => updates ? updateEvent(row) : row.time;
  const x = row => left + (right - left) * axis(row) / Math.max(1, axis(rows.at(-1)));
  const y = value => bottom - (bottom - top) * (value - low) / (high - low);
  const parts = [`<svg viewBox="0 0 ${width} 180" role="img" aria-label="各记录阶段的几何质量指标曲线">`];
  for (let i = 0; i < 4; i++) {
    const value = low + (high - low) * i / 3;
    parts.push(`<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" stroke="#edf0f6" stroke-dasharray="3 4"/><text x="43" y="${y(value) + 3}" text-anchor="end" fill="#64728a" font-size="12">${value.toFixed(choice === 'distance' ? 2 : 1)}</text>`);
  }
  parts.push(`<text x="${left}" y="11" fill="#64728a" font-size="12">${unit} · ${choice === 'coverage_percent' ? '越大越好' : '越小越好'}</text><line x1="${x(rows[selected])}" x2="${x(rows[selected])}" y1="${top}" y2="${bottom}" stroke="#dcdaf6"/>`);
  for (const item of series) {
    parts.push(`<polyline points="${rows.map(row => `${x(row)},${y(row[item.key])}`).join(' ')}" fill="none" stroke="${item.color}" stroke-width="2" stroke-linejoin="round"/>`);
    rows.forEach((row, i) => parts.push(`<circle data-index="${i}" tabindex="0" role="button" aria-label="更新 ${updateEvent(row)}，采集 ${observationCount(row)}，${item.label} ${row[item.key].toFixed(3)} ${unit}" cx="${x(row)}" cy="${y(row[item.key])}" r="${i === selected ? 5 : 3.5}" fill="${item.color}" stroke="white" stroke-width="2"><title>更新 ${updateEvent(row)} · 采集 ${observationCount(row)} · ${item.label}: ${row[item.key].toFixed(3)} ${unit}</title></circle>`));
  }
  rows.forEach(row => parts.push(`<text x="${x(row)}" y="161" text-anchor="middle" fill="#64728a" font-size="12">${Math.round(axis(row))}</text>`));
  parts.push(`<text x="${right}" y="177" text-anchor="end" fill="#64728a" font-size="12">${updates ? '建图更新次数（不一定等于拍摄次数）' : '累计任务时间 / s（不是实际墙钟）'}</text></svg>`);
  $('chart').innerHTML = parts.join('');
  $('chart-legend').replaceChildren(...series.map(item => {
    const span = document.createElement('span'), dot = document.createElement('i');
    dot.className = 'legend-dot'; dot.style.background = item.color;
    span.append(dot, document.createTextNode(item.label)); return span;
  }));
  $('chart').querySelectorAll('circle').forEach(dot => {
    const choose = () => showCheckpoint(Number(dot.dataset.index));
    dot.addEventListener('click', choose);
    dot.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); } });
  });
}

async function refreshComparison(runs) {
  await Promise.all(runs.filter(run => !runSummaries.has(run.id)).map(async run => {
    try {
      let data;
      try { data = await fetchJSON(`/assets/${encodeURIComponent(run.id)}/summary.json`, {}, 20000); }
      catch { data = await fetchJSON(run.manifest, {}, 60000); }
      if (data.status === 'completed') runSummaries.set(run.id, { ...data, final: data.final || data.checkpoints.at(-1) });
    } catch { /* Leave unreadable results out of the comparison, without inventing values. */ }
  }));
  renderRunOptions(runs);
  renderComparisonTable(runs);
}

function renderRunOptions(runs) {
  const key = runs.map(run => {
    const data = runSummaries.get(run.id);
    return `${run.id}:${data?.method_id}:${data?.seed}`;
  }).join('|');
  if (key === runMenuKey) return;
  runMenuKey = key;
  const selectedId = $('run-select').value;
  const groups = new Map();
  for (const run of runs) {
    const data = runSummaries.get(run.id), category = runCategory(run.id, data?.method_id);
    if (!groups.has(category)) {
      const group = document.createElement('optgroup'); group.label = category;
      groups.set(category, group);
    }
    const option = document.createElement('option'); option.value = run.id; option.title = run.id;
    const label = data ? `${methodLabels[data.method_id] || data.method} · ${seedLabel(data)}` : run.id;
    option.textContent = run.id.startsWith('web-') ? `${label} · ${run.id.slice(4)}` : label;
    groups.get(category).append(option);
  }
  $('run-select').replaceChildren(...groups.values());
  if (runs.some(run => run.id === selectedId)) $('run-select').value = selectedId;
}

function renderComparisonTable(runs) {
  const completed = runs.map(run => ({ id: run.id, data: runSummaries.get(run.id) })).filter(row => row.data?.final);
  const currentId = $('run-select').value, mode = $('comparison-filter').value;
  const category = runCategory(currentId, manifest?.method_id);
  const pairable = manifest && !originalRun(manifest) && Number.isInteger(manifest.seed) && manifest.comparison_id;
  const visible = completed.filter(row => mode === 'all' || (mode === 'stage'
    ? runCategory(row.id, row.data.method_id) === category
    : pairable ? !originalRun(row.data) && row.data.seed === manifest.seed && row.data.comparison_id === manifest.comparison_id : row.id === currentId));
  $('comparison-count').textContent = `${visible.length} / ${completed.length} 条`;
  $('comparison-context').textContent = mode === 'all'
    ? '跨阶段清单仅用于查找运行；预算、种子、前缀或源码不同的条目不能直接作为方法优劣证据。'
    : mode === 'stage'
      ? `${category}：同一类别的运行清单，各 seed 保留单独一行。手动类别可能包含不同预算，不构成质量对照；正式配对统计请看 v2 汇总。`
      : pairable
        ? '仅显示记录中 comparison_id 一致的运行：同种子、同预算协议、同公共前缀与实验来源。这里是单种子结果，不能替代多种子结论。'
        : manifest && originalRun(manifest)
          ? '原作者流程的随机种子未固定，属于独立复现，默认仅显示自身；不同预算的原作者运行不构成同种子配对。'
          : '当前结果没有登记可配对的种子与 comparison_id，只显示自身。它能说明本次重建完成，不能独立证明方法优于基线。';
  const rows = visible.map(({ id, data }) => {
    const tr = document.createElement('tr');
    tr.className = originalRun(data) ? 'original-row' : '';
    tr.dataset.runId = id;
    const method = document.createElement('td'), button = document.createElement('button');
    button.className = 'comparison-link'; button.textContent = `${methodLabels[data.method_id] || data.method} / ${seedLabel(data)}`;
    button.addEventListener('click', () => {
      void loadRun(id);
      $('reconstruction').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    method.append(button);
    const batch = document.createElement('small'); batch.textContent = runCategory(id, data.method_id); method.append(batch);
    method.title = `${id}\n配对组：${data.comparison_id || '未登记'}`;
    tr.append(method);
    const final = data.final;
    const values = [protocolLabel(data), `${observationCount(final)} / ${updateEvent(final)}`,
      `${final.accuracy_cm.toFixed(3)} cm`, `${final.completion_cm.toFixed(3)} cm`,
      `${final.coverage_percent.toFixed(2)} %`, `${final.chamfer_mm.toFixed(2)} mm`,
      `${final.time.toFixed(1)} s / ${Number.isFinite(data.cost?.wall_seconds) ? `${data.cost.wall_seconds.toFixed(1)} s` : '未记录'}`];
    for (const text of values) { const td = document.createElement('td'); td.textContent = text; tr.append(td); }
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement('tr'), td = document.createElement('td'); td.colSpan = 8; td.className = 'comparison-empty'; td.textContent = '当前范围尚无可读取的已完成结果；可选择其他实验或切换显示范围。'; tr.append(td); rows.push(tr);
  }
  $('comparison-body').replaceChildren(...rows);
}

function validateHistoricalSummary(data) {
  const record = value => value && typeof value === 'object' && !Array.isArray(value);
  const text = (value, limit = 4096) => typeof value === 'string' && value.length > 0 && value.length <= limit;
  const statistic = value => record(value) && value.n === 3 && Number.isFinite(value.mean)
    && Number.isFinite(value.sample_std) && value.sample_std >= 0;
  const count = value => Number.isInteger(value) && value >= 0;
  if (!record(data) || data.version !== 'viewmend-v1-summary-v1' || data.scene !== 'replica/office0'
      || !Array.isArray(data.seeds) || data.seeds.join(',') !== '0,1,2' || data.seeds.some(seed => !Number.isInteger(seed))
      || !text(data.generated_at, 128) || !Number.isFinite(Date.parse(data.generated_at))
      || !text(data.conclusion) || !record(data.units) || !record(data.evidence) || !record(data.stages)
      || data.evidence.validated_experiments !== 27 || data.evidence.quality_runs !== 24
      || !text(data.evidence.source_commit, 128) || !/^[0-9a-f]{64}$/.test(data.evidence.analysis_sha256 || '')
      || !text(data.evidence.report_url, 2048) || !Array.isArray(data.limitations)
      || !data.limitations.length || data.limitations.length > 20 || data.limitations.some(value => !text(value))) {
    throw new Error('历史汇总的版本、证据或适用范围不完整');
  }
  const report = new URL(data.evidence.report_url, location.origin);
  if (report.username || report.password || !(report.protocol === 'https:'
      || (report.protocol === 'http:' && report.origin === location.origin))) {
    throw new Error('历史报告链接格式不受支持');
  }
  for (const [budget, methods] of Object.entries(summaryMethods)) {
    const stage = data.stages[budget];
    if (!record(stage) || !text(stage.title, 128) || !text(stage.budget_text, 512)
        || !new RegExp(`^${budget}-[0-9a-f]{32}$`).test(stage.run_id || '')
        || !record(stage.protocol) || stage.protocol.mode !== budget || stage.protocol.prefix !== 20
        || stage.protocol.observations !== 60 || stage.protocol.seconds !== 180
        || !record(stage.method_statistics) || !record(stage.paired_difference_vs_confidence)
        || !record(stage.diagnostic_statistics)) throw new Error('历史预算协议不完整');
    if (Object.keys(stage.method_statistics).sort().join(',') !== [...methods].sort().join(',')) {
      throw new Error('历史汇总不是完整的方法与种子对照');
    }
    for (const method of methods) {
      const values = stage.method_statistics[method];
      if (!record(values) || !record(values.metrics) || !record(values.cost)) throw new Error('历史方法统计缺失');
      for (const [family, keys] of [['metrics', summaryMetrics], ['cost', summaryCosts]]) {
        for (const [key] of keys) {
          if (!statistic(values[family][key]) || values[family][key].mean < 0) throw new Error('历史三种子统计无效');
        }
      }
      if (method === 'confidence_nooracle') continue;
      const paired = stage.paired_difference_vs_confidence[method];
      if (!record(paired) || !record(paired.metrics)) throw new Error('历史配对差缺失');
      for (const [key] of summaryMetrics) {
        const value = paired.metrics[key];
        if (!statistic(value) || !record(value.method_minus_baseline_by_seed)
            || Object.keys(value.method_minus_baseline_by_seed).sort().join(',') !== '0,1,2'
            || Object.values(value.method_minus_baseline_by_seed).some(item => !Number.isFinite(item))) {
          throw new Error('历史配对差不是完整的三种子记录');
        }
      }
    }
    const diagnostic = stage.diagnostic_statistics.defect;
    if (!record(diagnostic) || ['geometry_measured_events', 'geometry_active_events', 'lambda_zero_robust_events',
      'lambda_zero_robust_changed_events'].some(key => !count(diagnostic[key]))
        || diagnostic.geometry_active_events > diagnostic.geometry_measured_events
        || diagnostic.lambda_zero_robust_changed_events > diagnostic.lambda_zero_robust_events) {
      throw new Error('历史评分行为记录无效');
    }
  }
  return data;
}

function summaryStatistic(value, digits, signed = false) {
  const mean = Math.abs(value.mean) < 0.5 * 10 ** -digits ? 0 : value.mean;
  const prefix = signed && mean > 0 ? '+' : '';
  return `${prefix}${mean.toFixed(digits)} ± ${value.sample_std.toFixed(digits)}`;
}

function renderHistoricalSummary() {
  if (!historicalSummary) return;
  const budget = $('summary-budget').value;
  const stage = historicalSummary.stages[budget];
  const methods = summaryMethods[budget];
  $('summary-budget-note').textContent = `${stage.title} · ${stage.budget_text}`;
  const rows = (family, keys, paired = false) => methods.filter(method => !paired || method !== 'confidence_nooracle').map(method => {
    const values = paired ? stage.paired_difference_vs_confidence[method][family] : stage.method_statistics[method][family];
    const tr = document.createElement('tr'); tr.dataset.method = method;
    const heading = document.createElement('th'); heading.scope = 'row';
    heading.textContent = methodLabels[method]; tr.append(heading);
    for (const [key, digits] of keys) {
      const td = document.createElement('td'); td.textContent = summaryStatistic(values[key], digits, paired);
      td.dataset.metric = key;
      if (paired) {
        td.title = [0, 1, 2].map(seed => `种子 ${seed}：${values[key].method_minus_baseline_by_seed[seed].toFixed(digits)}`).join('；');
      }
      tr.append(td);
    }
    return tr;
  });
  $('summary-quality').replaceChildren(...rows('metrics', summaryMetrics));
  $('summary-paired').replaceChildren(...rows('metrics', summaryMetrics, true));
  $('summary-cost').replaceChildren(...rows('cost', summaryCosts));
  const diagnostic = stage.diagnostic_statistics.defect;
  $('summary-diagnostics').textContent = `Defect 的前缀后评分：几何项激活 ${diagnostic.geometry_active_events} / ${diagnostic.geometry_measured_events} 次；`
    + `同一批候选移除几何项后，选中视角变化 ${diagnostic.lambda_zero_robust_changed_events} / ${diagnostic.lambda_zero_robust_events} 次（排除近似并列）。`
    + '这些记录说明评分行为，不能证明真值缺陷被修复。';
  const auditedAt = new Date(historicalSummary.generated_at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
  $('summary-source').replaceChildren(...[
    `历史 v1 · Replica office0 · 种子 0 / 1 / 2 · 审计 ${auditedAt}（北京时间）`,
    `整链验收 ${historicalSummary.evidence.validated_experiments} 条目，正式质量汇总 ${historicalSummary.evidence.quality_runs} 个分支；短实验与原作者真值掩码流程未纳入质量汇总。`,
    `当前预算批次：${stage.run_id}`,
    `实验源码：${historicalSummary.evidence.source_commit}`,
    `分析报告 SHA-256：${historicalSummary.evidence.analysis_sha256}`
  ].map(value => { const line = document.createElement('span'); line.textContent = value; return line; }));
}

async function loadHistoricalSummary() {
  try {
    const response = await fetch('/data/optimization-v1-summary.json', { signal: AbortSignal.timeout(20000) });
    if (response.status === 404) {
      $('summary-badge').textContent = '尚未发布';
      $('summary-state').textContent = '尚未发布已审计的 v1 历史汇总。单次实验预览仍可查看。';
      return;
    }
    if (!response.ok) throw new Error(`服务返回 ${response.status}`);
    historicalSummary = validateHistoricalSummary(await response.json());
    $('summary-conclusion').textContent = historicalSummary.conclusion;
    $('summary-report').href = new URL(historicalSummary.evidence.report_url, location.origin).href;
    $('summary-limitations').replaceChildren(...historicalSummary.limitations.map(value => {
      const item = document.createElement('li'); item.textContent = value; return item;
    }));
    renderHistoricalSummary();
    $('summary-badge').textContent = '3 种子 · 已审计';
    $('summary-state').hidden = true;
    $('summary-content').hidden = false;
  } catch (error) {
    historicalSummary = undefined;
    $('summary-badge').textContent = '汇总不可读取';
    $('summary-state').textContent = `无法确认 v1 历史汇总：${error.message}。本次不显示汇总数值。`;
    $('summary-content').hidden = true;
  }
}

function renderJob(job) {
  if (!job) return;
  const labels = { starting: '准备启动', running: '执行中', exporting: '导出中', completed: '已完成', failed: '失败', interrupted: '已中断' };
  $('job-badge').textContent = labels[job.status] || job.status;
  const detail = document.createElement('div'); detail.className = 'job-detail';
  const heading = document.createElement('strong');
  heading.textContent = job.status === 'completed' ? '三维结果已就绪' : job.status === 'failed' || job.status === 'interrupted' ? job.message : job.stage || job.message || '准备启动';
  const info = document.createElement('div');
  const elapsed = `${activeStates.has(job.status) ? '完整流水线已用时' : '完整流水线总耗时'} ${Number.isFinite(job.wall_seconds) ? job.wall_seconds : 0} s`;
  info.textContent = ['observations', 'replay'].includes(job.protocol)
    ? `GPU ${job.gpu} · ${job.protocol === 'replay' ? '独立逐帧演示 · ' : ''}${methodLabels[job.method] || job.method} · 种子 ${job.seed} · ${job.frames} 次更新 · ${elapsed}`
    : `GPU ${job.gpu} · 预算 ${job.budget} s · ${elapsed}`;
  const id = document.createElement('div'); id.className = 'small'; id.textContent = job.id;
  const timeNote = document.createElement('p'); timeNote.className = 'job-stage-note';
  timeNote.textContent = '从任务启动计时，包含环境准备、重建、网格评估与网页导出；与质量表中的“重建阶段墙钟”口径不同。';
  detail.append(heading, info, id, timeNote);
  if (job.status === 'completed') {
    const button = document.createElement('button');
    button.className = 'job-link'; button.style.cssText = 'border:0;background:none;padding:0;font-size:11px;';
    button.textContent = '查看这次实验的三维结果 →'; button.addEventListener('click', () => loadRun(job.result_id || job.id));
    detail.append(button);
  }
  $('job-content').replaceChildren(detail);
  $('job-log').textContent = job.log || job.message || '等待日志…';
}

function renderCampaign(campaign) {
  if(state?.read_only) {
    $('campaign-badge').textContent='只读回放';
    $('campaign-content').textContent='此电脑回放保存的结果，不读取服务器实时实验状态；正式质量结论见上方冻结汇总。';
    return;
  }
  const labels = { not_started: '尚未建立计划', waiting: '等待空闲设备', running: '执行中',
    failed: '失败', completed: '已完成', unavailable: '状态不可读取' };
  const stages = { smoke: '短实验验证', observations: '固定观测对照与消融', time: '固定时间对照',
    development_observations: '开发种子 · 固定观测与消融', development_time: '开发种子 · 固定时间',
    heldout_observations: '留出种子 · 固定观测', heldout_time: '留出种子 · 固定时间' };
  const badge = $('campaign-badge');
  const content = $('campaign-content');
  if (!campaign || !['viewmend-campaign-status-v1', 'viewmend-campaign-status-v2'].includes(campaign.schema)) {
    badge.textContent = '尚未建立计划';
    content.textContent = '服务端尚未提供优化实验计划记录。';
    return;
  }
  badge.textContent = labels[campaign.status] || '状态待核查';
  const lines = [];
  if (campaign.campaign) lines.push(`计划：${campaign.campaign}`);
  if (!campaign.configured) lines.push('尚未建立执行计划，没有启动计划内实验。');
  else if (campaign.status === 'running' && campaign.process_state !== 'live') {
    badge.textContent = campaign.process_state === 'starting' ? '身份确认中' : '状态待核查';
    lines.push(campaign.process_state === 'starting'
      ? '已登记的启动进程仍在执行，正在确认完整身份。'
      : '记录为执行中，当前无法确认已登记进程；需要核查服务端状态。');
  } else if (campaign.status === 'waiting') {
    if (campaign.waiting_reason === 'managed_web_job') {
      badge.textContent = '等待其他实验';
      lines.push('已有网页实验或启动记录待核查，计划等待该任务结束。');
    } else if (campaign.waiting_reason === 'managed_campaign') {
      badge.textContent = '等待其他计划';
      lines.push('另一优化计划仍有进程或启动记录，本计划暂不启动。');
    } else if (campaign.waiting_reason === 'gpu_busy') lines.push('显卡正被使用，等待空闲设备。');
    else {
      badge.textContent = '等待下一次检查';
      lines.push('等待下一次计划检查。');
    }
  }
  if (campaign.stage) lines.push(`当前阶段：${stages[campaign.stage] || '阶段待核查'}`);
  if (Number.isInteger(campaign.completed_stages) && Number.isInteger(campaign.completed_experiments)) {
    lines.push(`已验收 ${campaign.completed_stages} / ${campaign.total_stages} 阶段 · ${campaign.completed_experiments} / ${campaign.total_experiments} 个实验条目`);
  }
  if (Number.isInteger(campaign.gpu)) lines.push(`登记设备：GPU ${campaign.gpu}`);
  const formatTime = value => new Date(value).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
  if (campaign.checked_at) lines.push(`最近设备检查：${formatTime(campaign.checked_at)}（北京时间）`);
  if (campaign.updated_at) lines.push(`记录更新：${formatTime(campaign.updated_at)}（北京时间）`);
  if (campaign.error) lines.push(`错误：${campaign.error}`);
  content.replaceChildren(...lines.map(text => {
    const line = document.createElement('div'); line.textContent = text; return line;
  }));
}

async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    state = await fetchJSON('/api/status');
    const readOnly=state.read_only===true;
    const chosenCampaign = $('campaign-select').value;
    renderCampaign(state.campaigns?.[chosenCampaign] || (chosenCampaign === 'optimization-v1' ? state.campaign : null));
    $('connection-dot').className = 'online-dot connected'; $('connection-label').textContent = readOnly?'本地只读回放已连接':'实验服务已连接';
    $('new-experiment').disabled=readOnly;
    for(const id of ['protocol','method','seed','budget','gpu']) $(id).disabled=readOnly;
    const chosenGPU = $('gpu').value;
    const available = state.gpus.filter(gpu => gpu.available);
    $('gpu').replaceChildren(...state.gpus.map(gpu => {
      const option = document.createElement('option'); option.value = String(gpu.index);
      option.textContent = `GPU ${gpu.index} · ${gpu.available ? '空闲' : '占用'} · ${(gpu.used_mb / 1024).toFixed(1)} GiB`;
      option.disabled = !gpu.available;
      return option;
    }));
    if (!available.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = readOnly?'观看回放无需 NVIDIA GPU':'当前没有空闲 GPU'; option.selected = true;
      $('gpu').prepend(option);
    } else $('gpu').value = available.some(gpu => String(gpu.index) === chosenGPU) ? chosenGPU : String(available[0].index);
    $('gpu-summary').classList.toggle('available', !!available.length);
    $('gpu-summary').textContent = readOnly?'本地只读模式：直接读取演示包中的真实照片、网格与指标，不运行 Habitat 或模型。':state.gpu_error || `${available.length} / ${state.gpus.length} 张卡空闲。${available.length ? '可在空闲设备上启动。' : '已有任务正在使用显卡，等待空闲后可启动。'}`;
    const plannedBusy = Object.values(state.campaigns || { legacy: state.campaign }).some(campaign =>
      campaign?.configured && ['running', 'waiting', 'unavailable'].includes(campaign.status));
    const busy = state.jobs.some(job => activeStates.has(job.status)) || plannedBusy;
    $('launch').disabled = readOnly || submitting || busy || !available.length;
    $('launch').textContent = readOnly?'仅回放已有结果':busy ? '实验或优化计划正在执行…' : '▷ 启动实验';
    if(readOnly) $('launch-note').textContent='同一演示包中的照片、网格与指标一致。重新采集和重建需要配置 GPU 环境，结果可能有随机与硬件差异。';
    renderRunOptions(state.runs);
    if (!manifest && state.runs.length) await loadRun(state.runs[0].id);
    await refreshComparison(state.runs);
    if (!state.runs.length) showError($('page-error'), '尚无导出的三维结果。先完成实验，或按使用说明导出已有实验。');
    const job = state.jobs[0];
    renderJob(job);
    if(readOnly){$('job-badge').textContent='已完成结果回放';$('job-content').textContent='此入口不执行新实验。正在展示的内容来自已完成运行的保存产物。';}
    if (job && job.status === 'completed' && seenJobStatus === `${job.id}:running`) await loadRun(job.result_id || job.id);
    if (job) seenJobStatus = `${job.id}:${job.status === 'exporting' ? 'running' : job.status}`;
  } catch (error) {
    $('campaign-badge').textContent = '连接中断';
    $('campaign-content').textContent = '无法读取当前计划状态，检查实验服务与 SSH 转发。';
    $('connection-dot').className = 'online-dot error'; $('connection-label').textContent = '服务连接中断';
    $('launch').disabled = true;
    $('gpu-summary').textContent = `无法连接实验服务：${error.message}。检查服务与 SSH 转发。`;
    if (!manifest) showError($('page-error'), '无法连接实验服务，请先启动服务器与 SSH 转发。');
  } finally { polling = false; }
}

$('launch-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (submitting || $('launch').disabled) return;
  submitting = true; $('launch').disabled = true;
  showError($('launch-error'), '');
  try {
    const payload = $('protocol').value === 'original'
      ? { gpu: Number($('gpu').value), budget: Number($('budget').value) }
      : { gpu: Number($('gpu').value), protocol: $('protocol').value, method: $('method').value, seed: Number($('seed').value), frames: $('protocol').value === 'replay' ? 8 : 60 };
    const job = await fetchJSON('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-ViewMend3D': '1' },
      body: JSON.stringify(payload) });
    renderJob(job); $('logs-details').open = true;
  } catch (error) { showError($('launch-error'), error.message); }
  finally { submitting = false; await pollStatus(); }
});
$('protocol').addEventListener('change', configureLaunch);
$('method').addEventListener('change', configureLaunch);
$('campaign-select').addEventListener('change', () => renderCampaign(state?.campaigns?.[$('campaign-select').value]
  || ($('campaign-select').value === 'optimization-v1' ? state?.campaign : null)));
$('summary-budget').addEventListener('change', renderHistoricalSummary);
$('new-experiment').addEventListener('click', () => { $('launch-panel').scrollIntoView({ behavior: 'smooth', block: 'center' }); $('protocol').focus({ preventScroll: true }); });
$('run-select').addEventListener('change', event => loadRun(event.target.value));
$('comparison-filter').addEventListener('change', () => renderComparisonTable(state?.runs || []));
document.querySelectorAll('a[href="#metric-formulas"]').forEach(link => link.addEventListener('click', () => { $('metric-formulas').open = true; }));
$('checkpoint').addEventListener('input', event => showCheckpoint(Number(event.target.value)));
$('reset-view').addEventListener('click', resetView);
$('chart-metric').addEventListener('change', drawChart);
$('cutaway').addEventListener('change', updateClipping);
$('cut-height').addEventListener('input', updateClipping);
$('wireframe').addEventListener('change', () => { if (mesh) mesh.material.wireframe = $('wireframe').checked; });
$('show-path').addEventListener('change', () => { if (pathLine) pathLine.visible = $('show-path').checked; });
$('show-cameras').addEventListener('change', () => { if (cameraLines) cameraLines.visible = $('show-cameras').checked; });
$('play').addEventListener('click', async () => {
  if (!manifest) return;
  playing = !playing;
  $('play').textContent = playing ? 'Ⅱ 暂停' : '▶ 播放';
  if (!playing) return;
  if (selected === manifest.checkpoints.length - 1) await showCheckpoint(0);
  while (playing && selected < manifest.checkpoints.length - 1) {
    await new Promise(resolve => setTimeout(resolve, 1600));
    if (playing) await showCheckpoint(selected + 1);
  }
  playing = false; $('play').textContent = '▶ 播放';
});
try { setupViewer(); }
catch (error) { $('viewer-loading').textContent = `此浏览器无法初始化 WebGL：${error.message}。指标与实验启动仍可使用。`; }
configureLaunch();
void loadHistoricalSummary();
await pollStatus();
setInterval(pollStatus, 5000);
