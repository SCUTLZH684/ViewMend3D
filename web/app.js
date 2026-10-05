import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PLYLoader } from './vendor/PLYLoader.js';

const $ = id => document.getElementById(id);
const activeStates = new Set(['starting', 'running', 'exporting']);
let manifest, assetBase, selected = 0, loadVersion = 0, playing = false;
let state, polling = false, submitting = false, seenJobStatus;
let renderer, scene, camera, controls, mesh, grid, pathLine, cameraLines;
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const viewer = $('viewer');
const loader = new PLYLoader();

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

async function showCheckpoint(index) {
  if (!manifest) return;
  selected = Math.max(0, Math.min(manifest.checkpoints.length - 1, index));
  const checkpoint = manifest.checkpoints[selected];
  const version = ++loadVersion;
  $('checkpoint').value = String(selected);
  $('checkpoint-description').textContent = `${checkpoint.time.toFixed(1)} 秒任务时间 · ${checkpoint.camera_count} 个采集视角`;
  $('viewer-step').textContent = `观测 ${checkpoint.step} / ${manifest.checkpoints.at(-1).step}`;
  $('mesh-caption').textContent = `原始 ${checkpoint.original.vertices.toLocaleString()} 顶点 / ${checkpoint.original.faces.toLocaleString()} 面`;
  metricValue('accuracy', checkpoint.accuracy_cm, 'cm', 3);
  metricValue('completion', checkpoint.completion_cm, 'cm', 3);
  metricValue('coverage', checkpoint.coverage_percent, '%', 2);
  metricValue('chamfer', checkpoint.chamfer_mm, 'mm', 2);
  [...$('checkpoint-labels').children].forEach((button, i) => button.classList.toggle('selected', i === selected));
  drawChart();
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
  playing = false;
  $('play').textContent = '▶ 播放';
  ++loadVersion;
  const currentVersion = loadVersion;
  showError($('page-error'), '');
  try {
    const data = await fetchJSON(`/assets/${encodeURIComponent(id)}/manifest.json`, {}, 60000);
    if (currentVersion !== loadVersion) return;
    manifest = data;
    $('path-label').textContent = data.trajectory_kind === 'motion_path' ? '运动轨迹' : '视角连线';
    assetBase = `/assets/${encodeURIComponent(id)}`;
    $('run-select').value = id;
    $('checkpoint').max = String(data.checkpoints.length - 1);
    $('checkpoint-labels').replaceChildren(...data.checkpoints.map((checkpoint, i) => {
      const button = document.createElement('button');
      button.textContent = `${Math.round(checkpoint.time)} s`;
      button.setAttribute('aria-label', `查看 ${checkpoint.time.toFixed(1)} 秒的重建`);
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
  const series = choice === 'distance' ? [{ key: 'accuracy_cm', label: 'Accuracy', color: '#7975e0' }, { key: 'completion_cm', label: 'Completion', color: '#54bba5' }]
    : [{ key: choice, label: choice === 'coverage_percent' ? '覆盖率' : 'Chamfer', color: '#7975e0' }];
  const unit = choice === 'distance' ? 'cm' : choice === 'coverage_percent' ? '%' : 'mm';
  const rows = manifest.checkpoints;
  const values = series.flatMap(item => rows.map(row => row[item.key]));
  let low = Math.min(...values), high = Math.max(...values);
  const padding = Math.max((high - low) * 0.3, high * 0.01, 0.02);
  low = Math.max(0, low - padding); high += padding;
  const width = 640, left = 53, right = 618, top = 20, bottom = 142;
  const x = row => left + (right - left) * row.time / Math.max(1, rows.at(-1).time);
  const y = value => bottom - (bottom - top) * (value - low) / (high - low);
  const parts = [`<svg viewBox="0 0 ${width} 180" role="img" aria-label="各记录阶段的几何质量指标曲线">`];
  for (let i = 0; i < 4; i++) {
    const value = low + (high - low) * i / 3;
    parts.push(`<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" stroke="#edf0f6" stroke-dasharray="3 4"/><text x="43" y="${y(value) + 3}" text-anchor="end" fill="#a4adbd" font-size="9">${value.toFixed(choice === 'distance' ? 2 : 1)}</text>`);
  }
  parts.push(`<text x="${left}" y="10" fill="#a4adbd" font-size="9">${unit}</text><line x1="${x(rows[selected])}" x2="${x(rows[selected])}" y1="${top}" y2="${bottom}" stroke="#dcdaf6"/>`);
  for (const item of series) {
    parts.push(`<polyline points="${rows.map(row => `${x(row)},${y(row[item.key])}`).join(' ')}" fill="none" stroke="${item.color}" stroke-width="2" stroke-linejoin="round"/>`);
    rows.forEach((row, i) => parts.push(`<circle data-index="${i}" tabindex="0" role="button" aria-label="${Math.round(row.time)}秒，${item.label} ${row[item.key].toFixed(3)} ${unit}" cx="${x(row)}" cy="${y(row[item.key])}" r="${i === selected ? 5 : 3.5}" fill="${item.color}" stroke="white" stroke-width="2"><title>${row.time.toFixed(1)} s · ${item.label}: ${row[item.key].toFixed(3)} ${unit}</title></circle>`));
  }
  rows.forEach(row => parts.push(`<text x="${x(row)}" y="161" text-anchor="middle" fill="#a4adbd" font-size="9">${Math.round(row.time)}</text>`));
  parts.push(`<text x="${right}" y="177" text-anchor="end" fill="#a4adbd" font-size="9">任务时间 / s</text></svg>`);
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

function renderJob(job) {
  if (!job) return;
  const labels = { starting: '准备启动', running: '执行中', exporting: '导出中', completed: '已完成', failed: '失败', interrupted: '已中断' };
  $('job-badge').textContent = labels[job.status] || job.status;
  const detail = document.createElement('div'); detail.className = 'job-detail';
  const heading = document.createElement('strong');
  heading.textContent = job.status === 'completed' ? '三维结果已就绪' : job.status === 'failed' || job.status === 'interrupted' ? job.message : job.stage;
  const info = document.createElement('div'); info.textContent = `GPU ${job.gpu} · 预算 ${job.budget} s · 耗时 ${job.wall_seconds} s`;
  const id = document.createElement('div'); id.className = 'small'; id.textContent = job.id;
  detail.append(heading, info, id);
  if (job.status === 'completed') {
    const button = document.createElement('button');
    button.className = 'job-link'; button.style.cssText = 'border:0;background:none;padding:0;font-size:11px;';
    button.textContent = '查看这次实验的三维结果 →'; button.addEventListener('click', () => loadRun(job.id));
    detail.append(button);
  }
  $('job-content').replaceChildren(detail);
  $('job-log').textContent = job.log || job.message || '等待日志…';
}

async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    state = await fetchJSON('/api/status');
    $('connection-dot').className = 'online-dot connected'; $('connection-label').textContent = '实验服务已连接';
    const chosenGPU = $('gpu').value;
    const available = state.gpus.filter(gpu => gpu.available);
    $('gpu').replaceChildren(...state.gpus.map(gpu => {
      const option = document.createElement('option'); option.value = String(gpu.index);
      option.textContent = `GPU ${gpu.index} · ${gpu.available ? '空闲' : '占用'} · ${(gpu.used_mb / 1024).toFixed(1)} GB`;
      option.disabled = !gpu.available;
      return option;
    }));
    if (!available.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = '当前没有空闲 GPU'; option.selected = true;
      $('gpu').prepend(option);
    } else $('gpu').value = available.some(gpu => String(gpu.index) === chosenGPU) ? chosenGPU : String(available[0].index);
    $('gpu-summary').classList.toggle('available', !!available.length);
    $('gpu-summary').textContent = state.gpu_error || `${available.length} / ${state.gpus.length} 张卡空闲。${available.length ? '可在空闲设备上启动。' : '已有任务正在使用显卡，等待空闲后可启动。'}`;
    const busy = state.jobs.some(job => activeStates.has(job.status));
    $('launch').disabled = submitting || busy || !available.length;
    $('launch').textContent = busy ? '实验正在执行…' : '▷ 启动实验';
    const selectedRun = $('run-select').value;
    const oldRunIds = [...$('run-select').options].map(option => option.value).join(',');
    if (oldRunIds !== state.runs.map(run => run.id).join(',')) {
      $('run-select').replaceChildren(...state.runs.map(run => {
        const option = document.createElement('option'); option.value = run.id;
        option.textContent = run.id.startsWith('web-') ? `新实验 · ${run.id.slice(4, 20)}` : '首轮 baseline · 300 s';
        option.title = run.id; return option;
      }));
      if (state.runs.some(run => run.id === selectedRun)) $('run-select').value = selectedRun;
    }
    if (!manifest && state.runs.length) await loadRun(state.runs[0].id);
    if (!state.runs.length) showError($('page-error'), '尚无导出的三维结果。先完成实验，或按使用说明导出已有实验。');
    const job = state.jobs[0];
    renderJob(job);
    if (job && job.status === 'completed' && seenJobStatus === `${job.id}:running`) await loadRun(job.id);
    if (job) seenJobStatus = `${job.id}:${job.status === 'exporting' ? 'running' : job.status}`;
  } catch (error) {
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
    const job = await fetchJSON('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-ViewMend3D': '1' },
      body: JSON.stringify({ gpu: Number($('gpu').value), budget: Number($('budget').value) }) });
    renderJob(job); $('logs-details').open = true;
  } catch (error) { showError($('launch-error'), error.message); }
  finally { submitting = false; await pollStatus(); }
});
$('new-experiment').addEventListener('click', () => { $('launch-panel').scrollIntoView({ behavior: 'smooth', block: 'center' }); $('budget').focus({ preventScroll: true }); });
$('run-select').addEventListener('change', event => loadRun(event.target.value));
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
await pollStatus();
setInterval(pollStatus, 5000);
