import { ReplayScene } from './replay-scene.js';
import { replayStep } from './replay-state.js';

const $ = id => document.getElementById(id);
const num = (value, digits=3) => Number.isFinite(value) ? value.toFixed(digits) : '—';

export class CaptureReplay {
  constructor(host) {
    host.innerHTML = `<div class="replay-heading"><div><span class="eyebrow">看懂一次采集与补拍</span><h2>相机看到了什么，下一帧为什么去这里？</h2></div><span class="badge" id="replay-badge">已完成结果回放</span></div>
      <p id="replay-empty">正在读取所选实验的逐帧记录…</p>
      <div id="replay-content" hidden>
        <p id="replay-source" class="replay-small"></p><p class="replay-scope">独立 8 帧教学演示 · 使用实际采集的 RGB-D、位姿、评分日志和逐帧网格。额外保存与评估增加耗时，未纳入正式方法质量汇总。</p>
        <ol class="replay-stages"><li id="replay-stage-observed">① 查看已采集 RGB-D</li><li id="replay-stage-plan">② 根据当前地图选择补拍</li><li>③ 回放下一次采集与建图</li></ol>
        <div class="replay-grid"><div class="replay-sensors"><div class="replay-counter" id="replay-counter"></div><div class="replay-images"><figure><img id="replay-rgb" alt="当前实际采集的彩色图"><figcaption id="replay-rgb-caption"></figcaption></figure><figure><img id="replay-depth" alt="当前实际采集的深度图"><figcaption id="replay-depth-caption"></figcaption></figure></div><div class="depth-legend"><span id="replay-depth-near"></span><i></i><span id="replay-depth-far"></span></div><p class="replay-small">固定线性色标 · 灰色表示无效深度，不是近处表面</p><p id="replay-pose" class="replay-small"></p><p id="replay-image-error" role="alert" hidden></p></div>
          <div><div class="replay-viewport" id="replay-view"><span id="replay-map-label" class="replay-overlay"></span><span id="replay-map-loading" class="replay-loading">读取网格…</span><button id="replay-reset" type="button">重置视角</button></div><div class="replay-legend"><span class="current">● 当前相机与朝向</span><span class="chosen">● 选中的下一视角</span><span>● 可达候选位置</span><span class="baseline">● 基线选点（不同时显示）</span></div><p class="replay-small">箭头表示镜头朝向，虚线只连接当前与选中位置，不是实际行走路线。绿色线连接已采集位置。</p><div class="replay-map-controls"><button id="replay-before" type="button">本帧采集前的网格</button><button id="replay-after" type="button">本帧建图后的网格</button><label><input id="replay-cut" type="checkbox" checked>剖切屋顶</label></div></div></div>
        <div class="replay-actions"><button id="replay-start" type="button">从第一帧开始</button><button id="replay-prev" type="button">上一帧</button><button id="replay-plan" type="button" class="primary">准备补拍：查看选点</button><button id="replay-acquire" type="button" class="primary">回放采集下一帧</button></div>
        <p id="replay-phase" aria-live="polite" class="replay-phase"></p><p id="replay-map-reading" class="replay-small"></p>
        <div id="replay-decision" hidden><h3 id="replay-decision-title"></h3><p id="replay-reason"></p><div class="replay-table-scroll" tabindex="0" role="region" aria-label="下一帧候选评分"><table><thead><tr><th>候选</th><th>位置 / m</th><th>探索 E</th><th>不确定 U</th><th>几何 D</th><th>路径 / m</th><th>基线分数</th><th>几何奖励</th><th>最终得分 ↑</th></tr></thead><tbody id="replay-candidates"></tbody></table></div><p class="replay-small">显示可达候选中得分最高的 5 个，另保留选中项与基线项；编号沿用原始日志。E/U/D 来自当前地图，分数是选点依据，不是真值误差或成功概率。尚未拍摄的候选没有真实图像。</p></div>
      </div>`;
    this.version=0;
    $('replay-start').onclick=()=>this.render(0);
    $('replay-prev').onclick=()=>this.render(this.index-1);
    $('replay-plan').onclick=()=>this.render(this.index,true);
    $('replay-acquire').onclick=()=>this.render(this.index+1);
    $('replay-before').onclick=()=>this.render(this.index,false,true);
    $('replay-after').onclick=()=>this.render(this.index);
    $('replay-reset').onclick=()=>this.scene?.reset(this.manifest.bounds);
    $('replay-cut').onchange=()=>this.scene?.setCutaway($('replay-cut').checked);
    for(const id of ['replay-rgb','replay-depth']) $(id).onerror=()=>{
      $('replay-image-error').hidden=false; $('replay-image-error').textContent='观测图像加载失败；请检查服务连接或重新选择实验。';
    };
  }
  clear() {
    ++this.version; this.frames=null; this.scene?.clear();
    $('replay-content').hidden=true; $('replay-empty').hidden=false;
    for(const id of ['replay-rgb','replay-depth']) $(id).removeAttribute('src');
  }
  async load(manifest, base) {
    this.clear(); const version=this.version;
    this.manifest=manifest; this.base=base;
    if (!manifest.replay) {
      $('replay-badge').textContent='此实验无逐帧记录';
      $('replay-empty').textContent='所选历史实验未保存完整 RGB-D 序列与逐帧网格，不能逐帧展示。请在下方选择“独立演示 · 8帧过程回放”，或在启动面板选择“逐帧演示 · 8次采集”运行一条新演示。历史网格检查点仍可在下方浏览。';
      return;
    }
    $('replay-empty').textContent='读取实际采集记录…';
    try {
      const response=await fetch(`${base}/${manifest.replay.file}`,{signal:AbortSignal.timeout(20000)});
      if(!response.ok) throw new Error(`服务返回 ${response.status}`);
      const bytes=await response.arrayBuffer();
      const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
      const data=JSON.parse(new TextDecoder().decode(bytes));
      if(hash!==manifest.replay.sha256 || data.schema!=='viewmend-capture-replay-v1' || data.recorded_sensor_only!==true
        || data.frames?.length!==8 || data.frames.some((f,i)=>f.event!==i+1 || f.checkpoint_index!==i)) throw new Error('记录哈希或逐帧对应关系无效');
      if(version!==this.version) return;
      this.frames=data.frames; $('replay-empty').hidden=true; $('replay-content').hidden=false;
      $('replay-source').textContent=`${manifest.scene} · ${manifest.method} · 种子 ${manifest.seed} · 已完成结果回放`;
      $('replay-badge').textContent='8 次真实采集 · 可回放';
      if(!this.scene) {
        try{this.scene=new ReplayScene($('replay-view'));}
        catch(error){$('replay-map-loading').textContent=`WebGL 不可用：${error.message}`;}
      }
      this.scene?.reset(manifest.bounds);
      await this.render(0);
    }catch(error){if(version===this.version){$('replay-content').hidden=true; $('replay-empty').hidden=false;
      $('replay-empty').textContent=`无法确认逐帧记录：${error.message}`;}}
  }
  renderDecision(decision, event) {
    $('replay-decision').hidden=!decision;
    if(!decision) return;
    const selected=decision.selected_index, baseline=decision.baseline_selected_index;
    const reachable=decision.reachable.map((value,i)=>value ? i : null).filter(v=>v!==null);
    const indices=[...new Set([...reachable.sort((a,b)=>decision.final_scores[b]-decision.final_scores[a]).slice(0,5),selected,baseline])];
    $('replay-decision-title').textContent=`补拍第 ${event} 帧：选中候选 #${selected+1} / ${decision.candidate_poses.length}（${reachable.length} 个可达）`;
    const reasons={weak_geometry:'几何信号过弱',constant_geometry:'候选几何信号相同',base_all_zero:'基础效用为零',single_reachable:'只有一个可达候选',weight_zero:'奖励关闭',none:'几何奖励已应用'};
    $('replay-reason').textContent=decision.all_zero_utility_fallback
      ? '本轮基础效用为零，日志记录了随机回退评分，再考虑移动代价。不能把这个选点解释为“几何缺陷最大”。'
      : `在可达候选中选择最终得分最高项。${decision.geometry_active ? `本轮已加入有界几何奖励，选中项奖励为 ${num(decision.geometry_bonus[selected],6)}` : `本轮回退至基线：${reasons[decision.geometry_fallback_reason] || decision.geometry_fallback_reason}`}；相对此批候选的 Confidence 选点${selected===baseline ? '保持一致' : '发生变化'}。`;
    $('replay-candidates').replaceChildren(...indices.map(i=>{
      const row=document.createElement('tr'); row.classList.toggle('selected-candidate',i===selected);
      const pose=decision.candidate_poses[i];
      const values=[`#${i+1}${i===selected?' · 选中':''}${i===baseline?' · 基线':''}`, [0,1,2].map(j=>num(pose[j][3],2)).join(', '),
        ...['exploration','uncertainty','defect','path_lengths','baseline_scores','geometry_bonus','final_scores'].map(k=>num(decision[k]?.[i], k==='path_lengths'?2:['baseline_scores','geometry_bonus','final_scores'].includes(k)?6:4))];
      row.append(...values.map(value=>{const cell=document.createElement('td');cell.textContent=value;return cell;}));return row;
    }));
  }
  async render(index, preparing=false, before=false) {
    if(!this.frames || index<0 || index>=this.frames.length) return;
    this.index=index; const version=++this.version;
    const {frame,next,decision,mapFrame}=replayStep(this.frames,index,preparing,before);
    $('replay-image-error').hidden=true;
    $('replay-rgb').src=`${this.base}/${frame.rgb}`; $('replay-depth').src=`${this.base}/${frame.depth}`;
    $('replay-rgb-caption').textContent=`第 ${frame.event} 帧 · 实际 RGB`;
    $('replay-depth-caption').textContent=`第 ${frame.event} 帧 · 实际深度（m）`;
    $('replay-counter').textContent=`当前已采集 ${frame.event} / ${this.frames.length} 帧 · ${frame.width} × ${frame.height}`;
    $('replay-depth-near').textContent=`近 ${num(frame.depth_range_m[0],1)} m`;
    $('replay-depth-far').textContent=`远 ${num(frame.depth_range_m[1],1)} m`;
    $('replay-pose').textContent=`相机位置 x / y / z = ${[0,1,2].map(i=>num(frame.pose[i][3],2)).join(' / ')} m；朝向见右侧箭头。`;
    $('replay-phase').textContent=preparing && next
      ? `准备第 ${next.event} 帧：用第 ${frame.event} 帧建图后的地图评分，当前仍显示第 ${frame.event} 帧真实观测。点击“回放采集下一帧”查看选中视角随后实际拍到的画面。`
      : next ? `已回放第 ${frame.event} 帧采集与建图。点击“准备补拍”查看第 ${next.event} 帧的实际选点依据。`
        : '8 次采集已全部回放。可返回第一帧查看整个过程；这条短演示不证明方法质量提升。';
    $('replay-stage-plan').classList.toggle('active',preparing);
    $('replay-stage-observed').classList.toggle('active',!preparing);
    $('replay-prev').disabled=index===0; $('replay-plan').disabled=!next || preparing;
    $('replay-acquire').disabled=!next || !preparing;
    $('replay-before').disabled=index===0; $('replay-after').disabled=!before;
    $('replay-before').classList.toggle('active',before); $('replay-after').classList.toggle('active',!before);
    $('replay-map-label').textContent=before ? `第 ${frame.event} 帧采集前 · ${mapFrame.event} 次观测的地图` : `第 ${frame.event} 帧建图后 · ${frame.event} 次观测的地图`;
    const checkpoint=this.manifest.checkpoints[mapFrame.checkpoint_index];
    $('replay-map-reading').textContent=`当前网格：Accuracy ${num(checkpoint.accuracy_cm)} cm ↓ · Completion ${num(checkpoint.completion_cm)} cm ↓ · 2 cm覆盖率 ${num(checkpoint.coverage_percent,2)}% ↑ · Chamfer ${num(checkpoint.chamfer_mm,2)} mm ↓。切换前/后查看保存网格；左侧始终是第 ${frame.event} 帧观测。指标不保证逐帧改善。本区与下方检查点回放分别浏览。`;
    this.renderDecision(decision,next?.event);
    this.scene?.show(frame,decision,this.frames.slice(0,index+1));
    if(!this.scene) return;
    $('replay-map-loading').hidden=false; $('replay-map-loading').textContent='加载实际保存网格…';
    try{await this.scene.load(`${this.base}/${checkpoint.mesh}`);
      if(version===this.version) $('replay-map-loading').hidden=true;
    }catch(error){if(version===this.version) $('replay-map-loading').textContent=`网格读取失败：${error.message}`;}
  }
}
