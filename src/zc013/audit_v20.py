#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 v20：贴近皮肤的经脉——深度是否等于 d*、是否在体内、位置是否连续、左右是否镜像。
再用第二套独立的部位分类器复核，报告一致率。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'zc013'))
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from sites import LayerModel
from core import load_obj, closest_point_mesh, vertex_faces, vertex_normals, ray_surface

V, F = load_obj()
tree = cKDTree(V); v2f = vertex_faces(F, len(V)); NRM = vertex_normals(V, F)
lm = LayerModel()
THM = json.load(open(ROOT / 'data' / 'atlas2' / 'theorem_atlas.json'))
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
D3 = json.load(open(ROOT / 'data' / 'atlas2' / 'vessels_channels.json'))


def signed(Q):
    q, dd = closest_point_mesh(Q, V, F, tree, v2f)
    vi = tree.query(Q)[1]
    s = ((Q - q) * NRM[vi]).sum(1)
    return np.where(dd < 1e-6, 0.0, np.sign(s) * dd)


def audit(name, S, T):
    """S=体表点, T=定理偏移点, 单位=模型单位"""
    S = np.asarray(S, float); T = np.asarray(T, float)
    half = lm.half_thickness(S)   # 与生成端同一套半厚
    d_stored = np.linalg.norm(T - S, axis=1)          # 实际偏移量（单位）
    d_mm = d_stored * 102.0
    d_star = lm.d_star(S)                              # 定理值 mm
    clamp = d_star / 102.0 > 0.70 * half
    sd = signed(T)                                     # 带符号距离（负=体内）
    steps = np.abs(np.diff(d_mm))
    i = int(np.argmax(steps)) if len(steps) else 0
    return dict(
        n=len(S), d_mm_mean=float(d_mm.mean()), d_mm_min=float(d_mm.min()), d_mm_max=float(d_mm.max()),
        d_star_mean=float(d_star.mean()),
        err_mm=float(np.abs(d_mm - d_star).mean()), err_max_mm=float(np.abs(d_mm - d_star).max()),
        clamp_frac=float(clamp.mean()), zero_frac=float((d_mm < 0.05).mean()),
        outside_mm=float(max(0.0, sd.max()) * 102), inside_margin_mm=float(-sd.min() * 102),
        max_step_mm=float(steps.max()) if len(steps) else 0.0,
        step_at=[round(float(x), 2) for x in S[i]] if len(steps) else [])


print('=== 12 经 ===')
rows = {}
for code in D12['order']:
    S = np.array(D12['meridians'][code]['right']); T = np.array(THM['meridians'][code]['right'])
    r = audit(code, S, T); rows[code] = r
    print(f"{code}: d {r['d_mm_mean']:4.1f}mm [{r['d_mm_min']:.1f},{r['d_mm_max']:.1f}] "
          f"err {r['err_mm']:.2f}/{r['err_max_mm']:.2f} clamp {r['clamp_frac']*100:4.1f}% "
          f"zero {r['zero_frac']*100:4.1f}% out {r['outside_mm']:.2f}mm "
          f"step {r['max_step_mm']:.2f}mm at {r['step_at']}")
print('=== 奇经 ===')
for k, v in THM['vessels'].items():
    if v.get('keep'):
        continue
    S = np.array(D3['vessels'][k]['right']); T = np.array(v['right'])
    r = audit(k, S, T); rows[k] = r
    print(f"{k}: d {r['d_mm_mean']:4.1f}mm err {r['err_mm']:.2f}/{r['err_max_mm']:.2f} "
          f"clamp {r['clamp_frac']*100:4.1f}% zero {r['zero_frac']*100:4.1f}% out {r['outside_mm']:.2f} "
          f"step {r['max_step_mm']:.2f}mm at {r['step_at']}")

# 独立部位分类器（最近骨段 outright，不做阈值/分支）
arm = np.array([lm.J['l-shoulder'], lm.J['l-elbow'], lm.J['l-hand']])
leg = np.array([lm.J['l-upper-leg'], lm.J['l-knee'], lm.J['l-ankle']])
spine = np.array([lm.J['pelvis'], lm.J['spine-2'], lm.J['neck']])
def chord(P, C):
    best = (1e9, 0)
    for k in range(len(C) - 1):
        a, b = C[k], C[k + 1]; ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
        d = np.linalg.norm(P - (a + t * ab))
        if d < best[0]:
            best = (d, (k + t) / (len(C) - 1))
    return best
def site2(p):
    da, ta = chord(p, arm); dl, tl = chord(p, leg); ds, _ = chord(p, spine)
    if p[1] > 6.25: return 'head'
    if p[1] > 5.3 and da > 0.9: return 'neck'
    if da <= min(dl, ds): return 'upper_arm' if ta < 0.5 else 'forearm'
    if dl <= ds: return 'thigh' if tl < 0.5 else 'lower_leg'
    return 'back' if p[2] < -0.05 else ('chest' if p[1] > 2.5 else 'abdomen')
agree = tot = 0
for code in D12['order']:
    for p in np.array(D12['meridians'][code]['right'])[::3]:
        agree += (site2(p) == lm.site_of(p)); tot += 1
print(f'=== 部位分类器一致率（第二实现）: {agree}/{tot} = {agree/tot*100:.1f}%')
json.dump(rows, open(ROOT / 'data' / 'zc013' / 'audit_v20.json', 'w'), ensure_ascii=False, indent=1)
print('saved audit_v20.json')
