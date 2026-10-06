import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PLYLoader } from './vendor/PLYLoader.js';

// A separate viewport keeps checkpoint browsing independent of decision replay.
export class ReplayScene {
  constructor(host) {
    this.host = host;
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x192636);
    this.renderer.localClippingEnabled = true;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.prepend(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 2.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(1, -2, 7); this.scene.add(sun);
    this.markers = new THREE.Group(); this.scene.add(this.markers);
    this.loader = new PLYLoader(); this.version = 0;
    this.clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.cutaway = true;
    new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      if (!width || !height) return;
      this.renderer.setSize(width, height); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    }).observe(host);
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }
  dispose(object) {
    object?.traverse(item => {
      item.geometry?.dispose();
      if (Array.isArray(item.material)) item.material.forEach(value => value.dispose());
      else item.material?.dispose();
    });
    object?.removeFromParent();
  }
  clear() {
    ++this.version; this.dispose(this.mesh); this.mesh = null;
    [...this.markers.children].forEach(item => this.dispose(item));
  }
  reset(bounds) {
    this.bounds = bounds;
    const [min, max] = bounds;
    const span = Math.max(...max.map((value, i) => value - min[i]), 2);
    const center = new THREE.Vector3(...min).add(new THREE.Vector3(...max)).multiplyScalar(0.5);
    center.z = min[2] + (max[2] - min[2]) * 0.3;
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(span, -span * 1.3, span));
    this.controls.update();
    this.clip.constant = min[2] + (max[2] - min[2]) * 0.63;
  }
  setCutaway(value) {
    this.cutaway = value;
    if (this.mesh) { this.mesh.material.clippingPlanes = value ? [this.clip] : []; this.mesh.material.needsUpdate = true; }
  }
  async load(url, expectedHash=null) {
    const version = ++this.version;
    if (this.mesh) this.mesh.visible = false;
    let geometry;
    if(expectedHash) {
      const response=await fetch(url,{signal:AbortSignal.timeout(30000)});
      if(!response.ok) throw new Error(`服务返回 ${response.status}`);
      const bytes=await response.arrayBuffer();
      const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
      if(hash!==expectedHash) throw new Error('Ground Truth 预览哈希不一致');
      geometry=this.loader.parse(bytes);
    } else geometry = await this.loader.loadAsync(url);
    if (version !== this.version) { geometry.dispose(); return false; }
    for (const [name, attribute] of Object.entries(geometry.attributes)) {
      if (attribute.array instanceof Float64Array) geometry.setAttribute(name, new THREE.BufferAttribute(new Float32Array(attribute.array), attribute.itemSize));
    }
    geometry.computeVertexNormals(); this.dispose(this.mesh);
    this.mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({
      vertexColors: !!geometry.getAttribute('color'), color: geometry.getAttribute('color') ? 0xffffff : 0xc0cad6,
      side: THREE.DoubleSide, clippingPlanes: this.cutaway ? [this.clip] : []
    }));
    this.scene.add(this.mesh); return true;
  }
  cameraMarker(pose, color) {
    const origin = new THREE.Vector3(pose[0][3], pose[1][3], pose[2][3]);
    const direction = new THREE.Vector3(pose[0][2], pose[1][2], pose[2][2]).normalize();
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: false }));
    sphere.position.copy(origin); sphere.renderOrder = 5; this.markers.add(sphere);
    const arrow = new THREE.ArrowHelper(direction, origin, 0.65, color, 0.16, 0.12);
    arrow.traverse(item => { if (item.material) { item.material.depthTest = false; item.renderOrder = 6; } });
    this.markers.add(arrow);
    // Recorded OpenCV camera-to-world: +Z is the optical axis.
    const corners = [[-.15,-.15,.3],[.15,-.15,.3],[.15,.15,.3],[-.15,.15,.3]].map(p =>
      [0,1,2].map(i => pose[i][3] + pose[i][0]*p[0] + pose[i][1]*p[1] + pose[i][2]*p[2]));
    const positions = [];
    for (let i=0;i<4;i++) positions.push(...origin.toArray(), ...corners[i], ...corners[i], ...corners[(i+1)%4]);
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, depthTest: false }));
    lines.renderOrder=5; this.markers.add(lines);
  }
  show(frame, decision, history) {
    [...this.markers.children].forEach(item => this.dispose(item));
    const path = history.flatMap(item => [item.pose[0][3],item.pose[1][3],item.pose[2][3]]);
    const line = new THREE.Line(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(path,3)),
      new THREE.LineBasicMaterial({color:0x4be0c3,depthTest:false}));
    line.renderOrder=3; this.markers.add(line);
    this.cameraMarker(frame.pose, 0x5de6f5);
    if (!decision?.candidate_poses?.length) return;
    const positions = decision.candidate_poses.filter((_,i)=>decision.reachable[i]).flatMap(p=>[p[0][3],p[1][3],p[2][3]]);
    const points = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(positions,3)),
      new THREE.PointsMaterial({ color:0xd5ddea,size:0.06,transparent:true,opacity:0.65,depthTest:false }));
    points.renderOrder=4; this.markers.add(points);
    if (decision.baseline_selected_index !== decision.selected_index) this.cameraMarker(decision.candidate_poses[decision.baseline_selected_index],0xb8a0ff);
    this.cameraMarker(decision.candidate_poses[decision.selected_index],0xffbd53);
    const next = decision.candidate_poses[decision.selected_index];
    const link = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(frame.pose[0][3],frame.pose[1][3],frame.pose[2][3]), new THREE.Vector3(next[0][3],next[1][3],next[2][3])]),
      new THREE.LineDashedMaterial({color:0xffbd53,dashSize:0.15,gapSize:0.1,depthTest:false}));
    link.computeLineDistances(); link.renderOrder=4; this.markers.add(link);
  }
}
