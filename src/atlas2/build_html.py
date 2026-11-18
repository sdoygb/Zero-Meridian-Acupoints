#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4：生成自包含 3D 交互图（three.js 内联）。"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ROOT, load_obj

OUT = ROOT / 'data' / 'atlas2'
D12 = json.load(open(OUT / 'paths12.json'))
D3 = json.load(open(OUT / 'vessels_channels.json'))
S1 = json.load(open(OUT / 'stage1.json'))
D5 = json.load(open(OUT / 'depth_paths.json'))
ZC = json.load(open(ROOT / 'data' / 'zc013' / 'depth_theorem.json'))
THM = json.load(open(OUT / 'theorem_atlas.json'))
try:
    ACU = json.load(open(ROOT / 'data' / 'zc014' / 'acupoints_derived.json'))
except Exception:
    ACU = {'meridians': {}}
try:
    ACUN = json.load(open(ROOT / 'data' / 'zc014' / 'acupoints_named.json'))
except Exception:
    ACUN = {'meridians': {}}
ACU_NAMES = {}
for _m, _v in ACUN.get('meridians', {}).items():
    ACU_NAMES[_m] = {int(k): '/'.join(x['hanzi'] for x in val) for k, val in _v['assign'].items()}

TIS = json.load(open(OUT / 'tissue_paths.json'))
THREE = open('/tmp/three_core.js', encoding='utf-8').read()
ORBIT = open('/tmp/orbit.js', encoding='utf-8').read()


def rl(P, nd=4):
    return [[round(float(x), nd) for x in p] for p in P]


def main():
    V, F = load_obj()
    body = {'pos': rl(V, 4), 'idx': F.ravel().tolist()}
    mer = []
    for code in D12['order']:
        m = D12['meridians'][code]
        use_thm = ((sys.argv[1] if len(sys.argv)>1 else 'v19') in ('v20','v21')) and code in THM['meridians']
        mer.append({'code': code, 'name': m['name'], 'color': m['color'],
                    'right': rl(THM['meridians'][code]['right']) if use_thm else rl(m['right']),
                    'left': rl(THM['meridians'][code]['left']) if use_thm else rl(m['left']),
                    'surface': rl(m['right']) if use_thm else None,
                    'surface_left': rl(m['left']) if use_thm else None,
                    'deep': rl(D5['meridians'][code]['pts']) if (code in D5['meridians'] and (sys.argv[1] if len(sys.argv)>1 else 'v19') == 'v19') else None,
                    'theorem': rl(ZC['meridians'][code]['pts']) if code in ZC['meridians'] else None,
                    'vox': rl(TIS['guided'][code]['pts']) if code in TIS.get('guided', {}) else None})
    use_thm_tr = ('v2' in (sys.argv[1] if len(sys.argv) > 1 else 'v19'))
    trans = []
    for t in D12['transitions']:
        key = t['a'] + '-' + t['b']
        if use_thm_tr and key in THM.get('transitions', {}):
            tr = THM['transitions'][key]
            trans.append({'a': t['a'], 'b': t['b'], 'kind': t['kind'],
                          'right': rl(tr['right']), 'left': rl(tr['left'])})
        else:
            trans.append({'a': t['a'], 'b': t['b'], 'kind': t['kind'],
                          'right': rl(t['right']), 'left': rl(t['left'])})
    ves = []
    for k, v in D3['vessels'].items():
        use_thm = ((sys.argv[1] if len(sys.argv)>1 else 'v19') in ('v20','v21')) and k in THM['vessels'] and not THM['vessels'][k].get('keep')
        ves.append({'code': k, 'name': v['name'], 'color': v['color'], 'kind': v.get('kind', ''),
                    'closed': bool(v.get('closed')),
                    'right': rl(THM['vessels'][k]['right']) if use_thm else rl(v['right']),
                    'left': rl(THM['vessels'][k]['left']) if use_thm else rl(v.get('left', v['right'])),
                    'surface': rl(v['right']) if use_thm else None,
                    'surface_left': rl(v.get('left', v['right'])) if use_thm else None})
    cha = [{'code': k, 'name': v['name'], 'color': v['color'],
            'right': rl(v['right']), 'left': rl(v.get('left', v['right']))}
           for k, v in D3['channels'].items()]
    chak = D3['chakras']

    data = dict(body=body, meridians=mer, transitions=trans, vessels=ves,
                channels=cha, chakras=chak, order=D12['order'],
                hasDeep=any(m.get('deep') for m in mer),
                acu={k: v['pts'] for k, v in ACU.get('meridians', {}).items()},
                acu_frac={k: v.get('frac', []) for k, v in ACU.get('meridians', {}).items()},
                acu_names=ACU_NAMES,
                acu_total=ACU.get('total', 0))
    blob = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

    html = TEMPLATE.replace('__THREE__', THREE).replace('__ORBIT__', ORBIT).replace('__DATA__', blob)
    html = html.replace('（326 点）', f'（{data["acu_total"]} 点）')
    name = sys.argv[1] if len(sys.argv) > 1 else 'v19'
    html = html.replace('v19', name)
    if name in ('v20', 'v21'):
        html = re.sub(r'  <label class="row"><input id="tthm"[^\n]*</label>\n', '', html)
        html = re.sub(r'  <label class="row"><input id="tdeep"[^\n]*</label>\n', '', html)
        html = html.replace('groups.thm.visible=false; groups.vox.visible=false; groups.surf.visible=false;',
                            'groups.thm.visible=false; groups.vox.visible=false; groups.surf.visible=false; groups.deep.visible=false; groups.acu.visible=false; groups.labAcu.visible=false;')
    if name == 'v18':
        html = html.replace(' · 深筋膜层', '').replace(' · 双侧 · w≠常数', ' · 双侧')
        for _lab in ('tdeep', 'tthm', 'tvox'):
            html = re.sub(r'  <label class="row"><input id="%s"[^\n]*</label>\n' % _lab, '', html)
        html = html.replace('  <div class="slider"><span>埋深放大（1×–8×）</span><input id="dh" type="range" min="10" max="80" value="10"></div>\n', '')
    p = OUT / f'zero_atlas_{name}.html'
    p.write_text(html, encoding='utf-8')
    print('wrote', p, round(p.stat().st_size / 1e6, 2), 'MB')


TEMPLATE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zero 经络图谱 v21 · 定理层深 + 推导穴位</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;background:#070b14;color:#dfe7f3;
  font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
#app{position:fixed;inset:0;background:radial-gradient(circle at 50% 35%,#16203a 0%,#0a0f1b 45%,#05070d 100%)}
canvas{display:block}
#ui{position:fixed;top:14px;left:14px;width:272px;padding:14px 14px 10px;border-radius:14px;
  background:rgba(12,18,32,.82);border:1px solid rgba(110,160,255,.18);
  box-shadow:0 10px 40px rgba(0,0,0,.5);backdrop-filter:blur(10px);font-size:12.5px;z-index:5;
  max-height:calc(100vh - 28px);overflow:auto}
#ui h1{margin:0 0 2px;font-size:15px;letter-spacing:.5px;font-weight:600}
#ui .sub{color:#7f8ea8;font-size:11px;margin-bottom:10px}
.sec{margin:9px 0 3px;color:#8aa0c4;font-size:11px;letter-spacing:1px}
label.row{display:flex;align-items:center;gap:7px;margin:3px 0;cursor:pointer;color:#c9d6ea}
label.row input{accent-color:#5aa9ff}
.dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;box-shadow:0 0 6px currentColor}
input[type=range]{width:100%;accent-color:#5aa9ff;height:16px}
.slider{margin:6px 0 2px}
.slider span{color:#8497b5;font-size:11px}
.scbox{margin:4px 0 2px;padding:7px 8px 6px;border-radius:9px;background:rgba(30,50,90,.30);
  border:1px solid rgba(110,160,255,.14)}
.scnow{font-size:12px;color:#e8f0ff;line-height:1.4}
.scnow b{color:#ffd166;font-weight:600}
.scnow .tag{float:right;color:#7f8ea8;font-size:10px;border:1px solid rgba(110,160,255,.2);
  border-radius:6px;padding:0 4px;margin-left:4px}
.scrange{color:#8aa0c4;font-size:10.5px;margin-top:2px}
.scnote{color:#62738f;font-size:10px;margin-top:3px;line-height:1.35}
.scprog{height:4px;background:rgba(110,160,255,.18);border-radius:3px;margin-top:6px;overflow:hidden}
.scprog i{display:block;height:100%;width:0%;background:linear-gradient(90deg,#5aa9ff,#ffd166);
  border-radius:3px;transition:width .2s linear}
.scrow{display:flex;align-items:center;gap:6px;margin-top:5px}
.scrow button{flex:0 0 auto;padding:3px 7px;border-radius:6px;border:1px solid rgba(110,160,255,.22);
  background:rgba(60,110,200,.14);color:#cfe0ff;font-size:10.5px;cursor:pointer}
.scrow label{margin:0;font-size:10.5px}
.btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.btns button{flex:1 1 auto;padding:5px 6px;border-radius:8px;border:1px solid rgba(110,160,255,.25);
  background:rgba(60,110,200,.16);color:#cfe0ff;font-size:11.5px;cursor:pointer}
.btns button:hover{background:rgba(80,140,240,.30)}
.dpad{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px}
.dpad button{padding:4px 0;border-radius:8px;border:1px solid rgba(110,160,255,.22);
  background:rgba(60,110,200,.14);color:#cfe0ff;font-size:12px;cursor:pointer;line-height:1.2}
.dpad button:hover{background:rgba(80,140,240,.28)}
.dpad span{display:block}
#legend{position:fixed;right:14px;bottom:40px;width:186px;padding:10px 12px;border-radius:12px;
  background:rgba(12,18,32,.72);border:1px solid rgba(110,160,255,.14);font-size:11.5px;
  color:#b9c8de;line-height:1.55;z-index:5;max-height:52vh;overflow:auto;font-size:11px}
#legend b{color:#e8f0ff;font-weight:600;display:block;margin:6px 0 2px}
#legend i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.ext{display:block;box-sizing:border-box;text-align:center;color:#9ec7ff;font-size:12px;text-decoration:none;border:1px solid #2c3f5a;border-radius:8px;padding:7px 0;background:rgba(30,50,80,.25)}
.ext:hover{background:rgba(40,70,110,.45)}
#hint{position:fixed;left:50%;bottom:12px;transform:translateX(-50%);color:#63748f;font-size:11.5px;z-index:5}
#title{position:fixed;right:14px;bottom:12px;color:#55637a;font-size:11px;text-align:right;z-index:5}
</style></head><body>
<div id="app"></div>
<div id="ui">
  <h1>Zero 经络图谱 v21</h1>
  <div class="sub">层深定理 d*=ST+SAT+t_f/2 · 穴位=φ 极值（326 点）</div>
  <div class="sec">图层</div>
  <label class="row"><input id="t12" type="checkbox" checked> 十二正经</label>
  <label class="row"><input id="tflow" type="checkbox" checked> 流注过渡（线）</label>
  <label class="row"><input id="tv" type="checkbox" checked> 奇经八脉</label>
  <label class="row"><input id="tc" type="checkbox" checked> 三脉七轮</label>
  <label class="row"><input id="tb" type="checkbox" checked> 人体</label>
  <label class="row"><input id="tl" type="checkbox" checked> 名称标签</label>
  <label class="row"><input id="tdeep" type="checkbox" checked> 深筋膜层（w≠常数）</label>
  <label class="row"><input id="tthm" type="checkbox"> 层深定理（皮+脂+筋膜）</label>
  <label class="row"><input id="tvox" type="checkbox"> 体素路由（对照）</label>
  <label class="row"><input id="tsurf" type="checkbox"> 体表投影（原 v18）</label>
  <label class="row"><input id="tacu" type="checkbox"> 推导穴位（φ 极值）</label>
  <label class="row"><input id="tacuname" type="checkbox"> 穴位名标签</label>
  <div class="sec">十二时辰</div>
  <label class="row"><input id="tshichen" type="checkbox" checked> 十二时辰游走（严格）</label>
  <div class="scbox">
    <div id="scnow" class="scnow"><span id="scname">当前：--</span><span class="tag" id="sctag">实时</span></div>
    <div id="scrange" class="scrange">--</div>
    <div class="scprog"><i id="scbar"></i></div>
    <div class="scnote">每经 2 h；光点位置＝本时辰已过时间 / 2 h</div>
    <div class="scrow">
      <button id="scnowbtn">回到现在</button>
      <label class="row"><input id="tdemo" type="checkbox"> 演示 ×600</label>
    </div>
  </div>
  <div class="slider"><span>穴位大小</span><input id="asize" type="range" min="30" max="300" value="100"></div>
  <div class="slider"><span>身体透明度</span><input id="ob" type="range" min="4" max="60" value="20"></div>
  <div class="slider"><span>线宽</span><input id="lw" type="range" min="25" max="320" value="100"></div>
  <div class="slider"><span>标签大小</span><input id="ls" type="range" min="50" max="200" value="100"></div>
  <div class="btns">
    <button data-view="front">正视</button><button data-view="back">背视</button>
    <button data-view="side">侧视</button><button data-view="reset">复位</button>
  </div>
  <div class="dpad">
    <span></span><button data-pan="up" title="上移视图">▲</button><span></span>
    <button data-pan="left" title="左移视图">◀</button><button data-pan="center" title="回到中心">●</button><button data-pan="right" title="右移视图">▶</button>
    <span></span><button data-pan="down" title="下移视图">▼</button><span></span>
  </div>
  <div class="btns"><button id="rot">自动旋转：关</button></div>
  <a class="ext" style="margin-top:6px" href="https://github.com/sdoygb/Zero-Meridian-Acupoints#theory-articles" target="_blank" rel="noopener">理论文章 · Theory</a>
</div>
<div id="legend"></div>
<div id="hint">拖动旋转 · 滚轮缩放 · 右键/双指/方向键/⌘+拖动平移</div>
<div id="title">Zero 理论生成 · 非解剖测量</div>
<script>__THREE__</script>
<script>__ORBIT__</script>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
'use strict';
const D = JSON.parse(document.getElementById('data').textContent);
const app = document.getElementById('app');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070b14, 0.012);
const camera = new THREE.PerspectiveCamera(30, innerWidth/innerHeight, 0.1, 400);
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.92;
app.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.enablePan = true; controls.screenSpacePanning = true;
controls.panSpeed = 0.9;
controls.target.set(0, 0.35, 0.35);
camera.position.set(0, 0.8, 37);

// ---------- lights
scene.add(new THREE.HemisphereLight(0xbcd4ff, 0x141a26, 0.85));
const key = new THREE.DirectionalLight(0xffffff, 0.85); key.position.set(6, 10, 12); scene.add(key);
const fill = new THREE.DirectionalLight(0x6fa8ff, 0.55); fill.position.set(-10, 2, -8); scene.add(fill);
const rim = new THREE.DirectionalLight(0xffb26b, 0.35); rim.position.set(0, -6, -10); scene.add(rim);

// ---------- body
function makeBody(){
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(D.body.pos.length*3);
  D.body.pos.forEach((p,i)=>{pos[i*3]=p[0];pos[i*3+1]=p[1];pos[i*3+2]=p[2];});
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.setIndex(D.body.idx);
  g.computeVertexNormals();
  const m = new THREE.MeshPhysicalMaterial({color:0xbfcde3, roughness:0.62, metalness:0.0,
    clearcoat:0.55, clearcoatRoughness:0.4, transparent:true, opacity:0.24,
    side:THREE.DoubleSide, depthWrite:false});
  return new THREE.Mesh(g, m);
}
const body = makeBody(); scene.add(body);
const bodyIn = new THREE.Mesh(body.geometry, new THREE.MeshBasicMaterial({
  color:0x28405f, transparent:true, opacity:0.10, side:THREE.BackSide, depthWrite:false}));
scene.add(bodyIn);

// ---------- curves
const groups = {m12:new THREE.Group(), flow:new THREE.Group(), qi:new THREE.Group(), ves:new THREE.Group(), cha:new THREE.Group(), lab:new THREE.Group(), deep:new THREE.Group(), thm:new THREE.Group(), vox:new THREE.Group(), surf:new THREE.Group(), acu:new THREE.Group(), labAcu:new THREE.Group()};
Object.values(groups).forEach(g=>scene.add(g));
groups.thm.visible=false; groups.vox.visible=false; groups.surf.visible=false;
const allTubes = [];
function curve(pts, closed){
  const v = pts.map(p=>new THREE.Vector3(p[0],p[1],p[2]));
  const c = new THREE.CatmullRomCurve3(v, !!closed, 'centripetal', 0.5);
  return c;
}
function tube(pts, color, radius, opts){
  opts = opts||{};
  const c = curve(pts, opts.closed);
  const geo = new THREE.TubeGeometry(c, Math.max(64, Math.min(420, Math.round(pts.length*1.4))),
                                     radius, opts.radial||8, !!opts.closed);
  const mat = new THREE.MeshBasicMaterial({color:color, transparent:true,
      opacity: opts.opacity===undefined?0.98:opts.opacity, depthWrite:true,
      toneMapped:false});
  // 线宽只改半径：沿管壁法向做顶点位移，不动中心线与端点
  mat.userData.radUniforms = [];
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uRadScale = {value: 0.0};
    mat.userData.radUniforms.push(shader.uniforms.uRadScale);
    shader.vertexShader = 'uniform float uRadScale;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\n\ttransformed += normal * uRadScale;'
    );
  };
  mat.customProgramCacheKey = function(){ return 'tubeRadial'; };
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData.baseRadius = radius;
  mesh.userData.kind = opts.kind||'tube';
  return mesh;
}
function addCurve(pts, color, radius, opts){
  const grp = (opts&&opts.group)||groups.m12;
  const t = tube(pts, color, radius, opts);
  grp.add(t); allTubes.push(t);
  if (!opts || opts.halo !== false){
    const h = tube(pts, color, radius*1.9, {closed:opts&&opts.closed, radial:6, glow:0.0,
      opacity:0.05, kind:'halo'});
    h.material.blending = THREE.AdditiveBlending; h.material.depthWrite = false;
    grp.add(h); allTubes.push(h);
  }
  return t;
}

// 十二经（双侧）
const merCurves = [];
D.meridians.forEach(m=>{
  ['right','left'].forEach(side=>{
    const t = addCurve(m[side], m.color, 0.030, {group:groups.m12});
    merCurves.push({mesh:t, pts:m[side], color:m.color});
  });
});
// 深筋膜层（v19 遗留，仅 v19 构建）
const deepCurves = [];
D.meridians.forEach(m=>{
  if (!m.deep) return;
  ['right','left'].forEach(side=>{
    const S = side==='right' ? m.right : m.left;
    const Dp = side==='right' ? m.deep : m.deep.map(p=>[-p[0],p[1],p[2]]);
    const mesh = addCurve(Dp, m.color, 0.020, {group:groups.deep, glow:0.9, halo:false});
    deepCurves.push({mesh, S, Dp, closed:false});
  });
});
function applyDeepScale(k){
  deepCurves.forEach(o=>{
    const P = o.S.map((s,i)=>{const d=o.Dp[i];return [s[0]+k*(d[0]-s[0]), s[1]+k*(d[1]-s[1]), s[2]+k*(d[2]-s[2])];});
    const c = curve(P, false);
    const geo = new THREE.TubeGeometry(c, Math.max(64,Math.min(420,Math.round(P.length*1.4))), 0.020, 8, false);
    o.mesh.geometry.dispose(); o.mesh.geometry = geo;
  });
}
// 推导穴位：φ 极值点（Zc014），可调大小 + 名称标签
const acuMeshes = [];
const acuLabels = [];
if (D.acu && D.acu_total){
  const geo = new THREE.SphereGeometry(0.045, 12, 12);
  const mat = new THREE.MeshBasicMaterial({color:0xffd166, toneMapped:false});
  const short = {LU:'肺',LI:'大肠',ST:'胃',SP:'脾',HT:'心',SI:'小肠',BL:'膀胱',KI:'肾',PC:'心包',TE:'三焦',GB:'胆',LV:'肝'};
  const byCode = {};
  D.meridians.forEach(m=>{ byCode[m.code]=m; });
  Object.entries(D.acu).forEach(([code,list])=>{
    const fr = (D.acu_frac && D.acu_frac[code]) || [];
    const LP = byCode[code] ? byCode[code].left : null;
    list.forEach((p,i)=>{
      const s1 = new THREE.Mesh(geo, mat); s1.position.set(p[0],p[1],p[2]); groups.acu.add(s1); acuMeshes.push(s1);
      let lp = [ -p[0], p[1], p[2] ];
      if (LP && fr.length === list.length){
        const j = Math.max(0, Math.min(LP.length-1, Math.round(fr[i]*(LP.length-1))));
        lp = LP[j];
      }
      const s2 = new THREE.Mesh(geo, mat); s2.position.set(lp[0],lp[1],lp[2]); groups.acu.add(s2); acuMeshes.push(s2);
      {
        const nm = (D.acu_names && D.acu_names[code] && D.acu_names[code][i]) ? D.acu_names[code][i] : ((short[code]||code) + (i+1));
        const l1 = label(nm, '#ffd166', [p[0]+0.09, p[1]+0.09, p[2]+0.04], 0.30);
        const l2 = label(nm, '#ffd166', [lp[0]-0.09, lp[1]+0.09, lp[2]+0.04], 0.30);
        groups.labAcu.add(l1); groups.labAcu.add(l2); acuLabels.push(l1,l2);
      }
    });
  });
}
// 体表投影（原 v18 几何）
D.meridians.forEach(m=>{
  if (!m.surface) return;
  ['right','left'].forEach(side=>{
    const pts = side==='right' ? m.surface : m.surface_left;
    addCurve(pts, m.color, 0.008, {group:groups.surf, halo:false, opacity:0.35});
  });
});
// 奇经体表投影
D.vessels.forEach(v=>{
  if (!v.surface) return;
  const MID = (v.code==='REN'||v.code==='DU'||v.code==='CHONG');
  (MID?['right']:['right','left']).forEach(side=>{
    const pts = side==='right' ? v.surface : v.surface_left;
    addCurve(pts, v.color, 0.010, {group:groups.surf, halo:false, opacity:0.35});
  });
});
// 层深定理（d* = ST+SAT+t_f/2）
D.meridians.forEach(m=>{
  if (!m.theorem) return;
  ['right','left'].forEach(side=>{
    const pts = side==='right' ? m.theorem : m.theorem.map(p=>[-p[0],p[1],p[2]]);
    addCurve(pts, m.color, 0.016, {group:groups.thm, halo:false, opacity:0.95});
  });
});
// 体素路由（对照）
D.meridians.forEach(m=>{
  if (!m.vox) return;
  ['right','left'].forEach(side=>{
    const pts = side==='right' ? m.vox : m.vox.map(p=>[-p[0],p[1],p[2]]);
    addCurve(pts, 0xff6b6b, 0.014, {group:groups.vox, halo:false, opacity:0.85});
  });
});
// 流注过渡
D.transitions.forEach(t=>{
  const col = t.kind==='deep' ? 0x6f7f96 : 0x8090a8;
  ['right','left'].forEach(side=>{
    addCurve(t[side], col, 0.013, {group:groups.flow, glow:0.25, opacity:0.75, halo:false});
  });
});
// 奇经八脉
D.vessels.forEach(v=>{
  const MID = (v.code==='REN'||v.code==='DU'||v.code==='CHONG');
  (MID?['right']:['right','left']).forEach(side=>{
    const pts = v.closed ? v[side].concat([v[side][0]]) : v[side];
    if (v.code==='REN' || v.code==='DU'){
      addCurve(pts, v.color, 0.040, {group:groups.ves, glow:0.85});
    } else if (v.code==='CHONG'){
      addCurve(pts, v.color, 0.022, {group:groups.ves, glow:0.7, opacity:0.95});
    } else {
      addCurve(pts, v.color, 0.026, {group:groups.ves, glow:0.6});
    }
  });
});
// 三轮
D.channels.forEach(c=>{
  const r = c.code==='SUSHUMA'?0.028:0.016;
  const MID = (c.code==='SUSHUMA');
  (MID?['right']:['right','left']).forEach(side=>addCurve(c[side], c.color, r, {group:groups.cha, glow:0.9}));
});
// 七轮
const chakraMats = [];
function glowSprite(color, size){
  const s = 128, cv = document.createElement('canvas'); cv.width=cv.height=s;
  const ctx = cv.getContext('2d');
  const g = ctx.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
  g.addColorStop(0,'rgba(255,255,255,0.95)');
  g.addColorStop(0.25, color);
  g.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=g; ctx.fillRect(0,0,s,s);
  const tex = new THREE.CanvasTexture(cv);
  const mat = new THREE.SpriteMaterial({map:tex, blending:THREE.AdditiveBlending,
    depthWrite:false, transparent:true, opacity:0.9});
  const sp = new THREE.Sprite(mat); sp.scale.set(size,size,1); return sp;
}
D.chakras.forEach(c=>{
  const geo = new THREE.TorusGeometry(c.rr, 0.022, 10, 72);
  const mat = new THREE.MeshBasicMaterial({color:0xffd166, transparent:true, opacity:0.95, toneMapped:false});
  const ring = new THREE.Mesh(geo, mat);
  ring.rotation.x = Math.PI/2; ring.position.set(0, c.y, c.z);
  groups.cha.add(ring); chakraMats.push(mat);
  const s = glowSprite('rgba(255,196,80,0.85)', c.rr*3.2);
  s.position.set(0, c.y, c.z); groups.cha.add(s);
});

// ---------- labels
function label(text, color, pos, size){
  const pad=8, fs=44, cv=document.createElement('canvas');
  const ctx=cv.getContext('2d');
  ctx.font = '600 '+fs+'px "PingFang SC","Microsoft YaHei",sans-serif';
  const w = Math.ceil(ctx.measureText(text).width)+pad*2;
  cv.width = w; cv.height = fs+pad*2;
  const c2 = cv.getContext('2d');
  c2.font = '600 '+fs+'px "PingFang SC","Microsoft YaHei",sans-serif';
  c2.fillStyle='rgba(8,12,20,0.55)'; c2.fillRect(0,0,cv.width,cv.height);
  c2.fillStyle='#ffffff'; c2.textBaseline='middle';
  c2.fillText(text,pad,cv.height/2);
  const tex=new THREE.CanvasTexture(cv); tex.minFilter=THREE.LinearFilter;
  const mat=new THREE.SpriteMaterial({map:tex, transparent:true, depthWrite:false, depthTest:false});
  const sp=new THREE.Sprite(mat);
  const k = 0.0023*(size||1);
  sp.scale.set(cv.width*k, cv.height*k, 1);
  sp.position.set(pos[0],pos[1],pos[2]);
  sp.userData.baseScale = sp.scale.clone();
  return sp;
}
const labelSprites = [];
const ARM = ['LU','LI','HT','SI','PC','TE'];
D.meridians.forEach((m,idx)=>{
  const side = (idx%2===0) ? 'right' : 'left';
  const pts = m[side], isArm = ARM.indexOf(m.code)>=0;
  const yT = isArm ? 2.5 : -3.2;
  let best = 0, bd = 1e9;
  pts.forEach((p,i)=>{ const d = Math.abs(p[1]-yT); if (d<bd){bd=d; best=i;} });
  const q = pts[best];
  const sgn = side==='right' ? 1 : -1;
  const k = Math.floor(idx/2)%3;
  const p = [q[0] + sgn*(0.55 + 0.20*(idx%2)), q[1] + 0.7*(k-1), q[2] + 0.25];
  const sp = label(m.name, m.color, p, 0.95);
  groups.lab.add(sp); labelSprites.push(sp);
});
D.vessels.forEach(v=>{
  if (v.code==='REN'||v.code==='DU'){
    const pts=v.left, i=Math.floor(pts.length*0.55);
    const sp=label(v.name, v.color, [pts[i][0]-0.55, pts[i][1], pts[i][2]+0.15], 0.9);
    groups.lab.add(sp); labelSprites.push(sp);
  }
});
D.chakras.forEach(c=>{
  const sp=label(c.name, '#ffd166', [0.72, c.y, c.z+0.35], 0.9);
  groups.lab.add(sp); labelSprites.push(sp);
});

// ---------- 十二时辰游走光点
// 十二时辰经序：子胆、丑肝、寅肺、卯大肠、辰胃、巳脾、午心、未小肠、申膀胱、酉肾、戌心包、亥三焦。
const SHICHEN = [
  {name:'子', code:'GB', label:'胆经', range:'23:00–01:00'},
  {name:'丑', code:'LV', label:'肝经', range:'01:00–03:00'},
  {name:'寅', code:'LU', label:'肺经', range:'03:00–05:00'},
  {name:'卯', code:'LI', label:'大肠经', range:'05:00–07:00'},
  {name:'辰', code:'ST', label:'胃经', range:'07:00–09:00'},
  {name:'巳', code:'SP', label:'脾经', range:'09:00–11:00'},
  {name:'午', code:'HT', label:'心经', range:'11:00–13:00'},
  {name:'未', code:'SI', label:'小肠经', range:'13:00–15:00'},
  {name:'申', code:'BL', label:'膀胱经', range:'15:00–17:00'},
  {name:'酉', code:'KI', label:'肾经', range:'17:00–19:00'},
  {name:'戌', code:'PC', label:'心包经', range:'19:00–21:00'},
  {name:'亥', code:'TE', label:'三焦经', range:'21:00–23:00'}
];
const codeToMer = {};
D.meridians.forEach(m=>{ codeToMer[m.code] = m; });
const dotGeo = new THREE.SphereGeometry(0.085, 12, 12);
const qiDots = [];
D.meridians.forEach(m=>{
  ['right','left'].forEach(side=>{
    const c = curve(m[side]);
    const mat = new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:1.0,
      blending:THREE.AdditiveBlending, depthWrite:false});
    const sphere = new THREE.Mesh(dotGeo, mat);
    const halo = glowSprite(m.color, 0.60);
    halo.material.opacity = 0.90;
    const g = new THREE.Group();
    g.add(sphere); g.add(halo); g.visible = false;
    groups.qi.add(g);
    qiDots.push({code:m.code, side, group:g, curve:c, mat, halo, color:m.color});
  });
});

// 十二时辰时钟：位置是本地时钟的纯函数，不做随机累积。
function shichenAt(ms){
  const d = new Date(ms);
  const h = d.getHours() + d.getMinutes()/60 + d.getSeconds()/3600 + d.getMilliseconds()/3600000;
  const idx = Math.floor(((h + 1) % 24) / 2) % 12;
  const start = (2*idx - 1 + 24) % 24;
  const elapsed = (h - start + 24) % 24;
  const frac = Math.max(0, Math.min(1, elapsed/2));
  return {idx, sc:SHICHEN[idx], frac, h, date:d};
}
let scOn = true;
let demoOn = false;
const DEMO_SPEED = 600;
let demoReal0 = 0, demoSim0 = 0;
function simNow(){
  if (!demoOn) return Date.now();
  return demoSim0 + (performance.now() - demoReal0) * DEMO_SPEED;
}
function resetDemoBase(){
  demoReal0 = performance.now();
  demoSim0 = Date.now();
}
function fmtClock(d){
  const p = n=>String(n).padStart(2,'0');
  return p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
}
let scLastKey = '';
function updateShichenUI(sc){
  const pct = Math.round(sc.frac*100);
  const tag = demoOn ? ('演示 ×'+DEMO_SPEED) : '实时';
  const key = sc.sc.name+'|'+pct+'|'+tag+'|'+scOn+'|'+fmtClock(sc.date);
  if (key === scLastKey) return;
  scLastKey = key;
  const nm = document.getElementById('scname');
  const rg = document.getElementById('scrange');
  const bar = document.getElementById('scbar');
  const tg = document.getElementById('sctag');
  if (!scOn){
    nm.innerHTML = '当前：<b>十二时辰已关闭</b>';
    rg.textContent = '光点隐藏'; bar.style.width = '0%'; if (tg) tg.textContent = '';
    return;
  }
  const m = codeToMer[sc.sc.code];
  const css = m ? m.color : '#ffd166';
  nm.innerHTML = '当前：<span class="dot" style="background:'+css+';color:'+css+'"></span><b>'+sc.sc.name+'时 · '+sc.sc.label+'</b>';
  if (tg) tg.textContent = tag;
  rg.textContent = sc.sc.range+' · '+fmtClock(sc.date)+' · '+pct+'%';
  bar.style.width = pct+'%';
}

// ---------- UI
const legend = document.getElementById('legend');
function buildLegend(){
  let h = '<b>十二正经</b>';
  D.meridians.forEach(m=>{h+=`<div><i style="background:${m.color};color:${m.color}"></i>${m.name}</div>`;});
  h += '<b>奇经八脉</b>';
  D.vessels.forEach(v=>{h+=`<div><i style="background:${v.color};color:${v.color}"></i>${v.name}</div>`;});
  h += '<b>三脉七轮</b>';
  D.channels.forEach(c=>{h+=`<div><i style="background:${c.color};color:${c.color}"></i>${c.name}</div>`;});
  D.chakras.forEach(c=>{h+=`<div><i style="background:#ffd166;color:#ffd166"></i>${c.name}</div>`;});
  legend.innerHTML = h;
}
buildLegend();

const cb = id=>document.getElementById(id);
cb('t12').onchange = e=>{groups.m12.visible=e.target.checked;};
cb('tflow').onchange = e=>{groups.flow.visible=e.target.checked;};
cb('tv').onchange = e=>{groups.ves.visible=e.target.checked;};
cb('tc').onchange = e=>{groups.cha.visible=e.target.checked;};
cb('tb').onchange = e=>{body.visible=e.target.checked; bodyIn.visible=e.target.checked;};
cb('tl').onchange = e=>{groups.lab.visible=e.target.checked;};
const _tdeep = cb('tdeep'); if (_tdeep) _tdeep.onchange = e=>{groups.deep.visible=e.target.checked;};
const _tthm = cb('tthm'); if (_tthm) _tthm.onchange = e=>{groups.thm.visible=e.target.checked;};
const _tvox = cb('tvox'); if (_tvox) _tvox.onchange = e=>{groups.vox.visible=e.target.checked;};
const _tsurf = cb('tsurf'); if (_tsurf) _tsurf.onchange = e=>{groups.surf.visible=e.target.checked;};
const _tacu = cb('tacu'); if (_tacu) _tacu.onchange = e=>{groups.acu.visible=e.target.checked;};
let labAcuOn = false;
const _tacuname = cb('tacuname'); if (_tacuname) _tacuname.onchange = e=>{labAcuOn = e.target.checked;};
const _tshichen = cb('tshichen'); if (_tshichen) _tshichen.onchange = e=>{
  scOn = e.target.checked; groups.qi.visible = scOn;
  const dm = cb('tdemo');
  if (!scOn){ demoOn = false; if (dm){ dm.checked = false; dm.disabled = true; } }
  else if (dm){ dm.disabled = false; }
  updateShichenUI(shichenAt(simNow()));
};
const _tdemo = cb('tdemo'); if (_tdemo) _tdemo.onchange = e=>{
  demoOn = e.target.checked; if (demoOn) resetDemoBase();
  updateShichenUI(shichenAt(simNow()));
};
const _scnowbtn = cb('scnowbtn'); if (_scnowbtn) _scnowbtn.onclick = ()=>{
  resetDemoBase(); updateShichenUI(shichenAt(simNow()));
};
const _asize = cb('asize'); if (_asize) _asize.oninput = e=>{
  const k = +e.target.value/100;
  acuMeshes.forEach(m=>m.scale.setScalar(k));
};

cb('ob').oninput = e=>{
  const o = e.target.value/100; body.material.opacity = o; bodyIn.material.opacity = o*0.6;
};
let lwValue = 100;
function applyLineWidth(v){
  lwValue = v;
  const k = v/100;
  allTubes.forEach(t=>{
    const base = t.userData.baseRadius;
    const s = t.userData.kind==='halo' ? k*1.15 : k;
    (t.material.userData.radUniforms||[]).forEach(u=>{ u.value = (s-1)*base; });
  });
}
cb('lw').oninput = e=>applyLineWidth(+e.target.value);
cb('ls').oninput = e=>{
  const k = e.target.value/100;
  labelSprites.forEach(s=>{const b=s.userData.baseScale; s.scale.set(b.x*k,b.y*k,1);});
};
function panBy(dx, dy){
  const dist = camera.position.distanceTo(controls.target);
  const k = dist * 0.0050;
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
  const off = right.multiplyScalar(dx * k).add(up.multiplyScalar(dy * k));
  camera.position.add(off); controls.target.add(off);
}
document.querySelectorAll('[data-pan]').forEach(b=>{
  b.onclick = ()=>{
    const d = b.dataset.pan;
    if (d==='up') panBy(0, 5);
    else if (d==='down') panBy(0, -5);
    else if (d==='left') panBy(-5, 0);
    else if (d==='right') panBy(5, 0);
    else { const v = views.reset || [0,0.8,37]; camera.position.set(v[0],v[1],v[2]); controls.target.set(0,0.35,0.35); }
  };
});
addEventListener('keydown', e=>{
  if (e.metaKey || e.ctrlKey) controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
  const step = e.shiftKey ? 12 : 4;
  if (e.key==='ArrowLeft'){ panBy(-step,0); e.preventDefault(); }
  else if (e.key==='ArrowRight'){ panBy(step,0); e.preventDefault(); }
  else if (e.key==='ArrowUp'){ panBy(0,step); e.preventDefault(); }
  else if (e.key==='ArrowDown'){ panBy(0,-step); e.preventDefault(); }
});
addEventListener('keyup', e=>{
  if (e.key==='Meta' || e.key==='Control') controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
});
const views = {
  front:[0,0.8,37], back:[0,0.8,-37], side:[37,0.8,0.5], reset:[0,0.8,37]
};
document.querySelectorAll('.btns button[data-view]').forEach(b=>{
  b.onclick = ()=>{
    const v = views[b.dataset.view];
    camera.position.set(v[0],v[1],v[2]);
    controls.target.set(0,0.4,0.35);
  };
});
function applyHash(){
  const h = (location.hash||'').replace('#','');
  if (views[h]){ const v=views[h]; camera.position.set(v[0],v[1],v[2]); controls.target.set(0,0.35,0.35); }
}
let autoRot = false;
cb('rot').onclick = ()=>{
  autoRot = !autoRot;
  cb('rot').textContent = '自动旋转：'+(autoRot?'开':'关');
};

addEventListener('resize', ()=>{
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- render loop
let t0 = performance.now();
function animate(t){
  requestAnimationFrame(animate);
  try{
  const dt = Math.max(0, Math.min(0.05, (t-t0)/1000)); t0 = t;
  if (autoRot) controls.autoRotate = true, controls.autoRotateSpeed = 0.55;
  else controls.autoRotate = false;
  controls.update();
  applyLineWidth(lwValue);
  groups.labAcu.visible = labAcuOn && camera.position.distanceTo(controls.target) < 42;
  const sc = shichenAt(simNow());
  updateShichenUI(sc);
  groups.qi.visible = scOn;
  qiDots.forEach(d=>{
    const on = scOn && d.code === sc.sc.code;
    d.group.visible = on;
    if (on){
      d.group.position.copy(d.curve.getPointAt(sc.frac));
      const s = 0.88 + 0.12*Math.sin(t/260);
      d.group.scale.setScalar(s);
    }
  });
  body.material.opacity = groups.m12.visible ? body.material.opacity : body.material.opacity;
  chakraMats.forEach((m,i)=>{ m.opacity = 0.72 + 0.26*Math.sin(t/650 + i); });
  renderer.render(scene, camera);
  }catch(e){ if(!window.__err){window.__err=1; console.error('ANIM '+e.stack);} }
}
applyHash();
try{
  const q = new URLSearchParams(location.search||'');
  if (q.get('lw')){ const v = +q.get('lw'); cb('lw').value = v; applyLineWidth(v); }
  if (q.get('rotate')==='1'){ autoRot = true; cb('rot').textContent='自动旋转：开'; }
  if (q.get('shichen')==='0'){ cb('tshichen').checked=false; scOn=false; groups.qi.visible=false; demoOn=false;
    const dm=cb('tdemo'); if(dm){ dm.checked=false; dm.disabled=true; } }
  if (q.get('demo')==='1'){ cb('tdemo').checked=true; demoOn=true; resetDemoBase(); }

  if (q.get('onlydeep')==='1'){ cb('t12').checked=false; groups.m12.visible=false; }
  if (q.get('panx')||q.get('pany')){ panBy(+(q.get('panx')||0), +(q.get('pany')||0)); }
  if (q.get('on')){
    const ON = q.get('on').split(',');
    if (ON.includes('thm')){ groups.thm.visible=true; if(cb('tthm')) cb('tthm').checked=true; }
    if (ON.includes('vox')){ groups.vox.visible=true; if(cb('tvox')) cb('tvox').checked=true; }
    if (ON.includes('acu')){ groups.acu.visible=true; if(cb('tacu')) cb('tacu').checked=true; }
    if (ON.includes('acuname')){ labAcuOn=true; if(cb('tacuname')) cb('tacuname').checked=true; }
  }
  if (q.get('hide')){
    const H = q.get('hide').split(',');
    if (H.includes('m12')){ groups.m12.visible=false; cb('t12').checked=false; }
    if (H.includes('ves')){ groups.ves.visible=false; cb('tv').checked=false; }
    if (H.includes('cha')){ groups.cha.visible=false; cb('tc').checked=false; }
    if (H.includes('flow')){ groups.flow.visible=false; groups.qi.visible=false; cb('tflow').checked=false;
      if(cb('tshichen')) cb('tshichen').checked=false; scOn=false; demoOn=false;
      const dm=cb('tdemo'); if(dm){ dm.checked=false; dm.disabled=true; } }
    if (H.includes('qi')){ groups.qi.visible=false; if(cb('tshichen')) cb('tshichen').checked=false; scOn=false; demoOn=false;
      const dm=cb('tdemo'); if(dm){ dm.checked=false; dm.disabled=true; } }
    if (H.includes('body')){ body.visible=false; bodyIn.visible=false; cb('tb').checked=false; }
    if (H.includes('deep')){ groups.deep.visible=false; if(cb('tdeep')) cb('tdeep').checked=false; }
  }
  if (q.get('cam')){ const c=q.get('cam').split(',').map(Number); camera.position.set(c[0],c[1],c[2]); }
  if (q.get('tgt')){ const t=q.get('tgt').split(',').map(Number); controls.target.set(t[0],t[1],t[2]); }
  if (q.get('pan4')==='1'){ for(let i=0;i<4;i++) panBy(1,0); }
}catch(e){}
requestAnimationFrame(animate);
})();
</script>
</body></html>
"""

if __name__ == '__main__':
    main()
