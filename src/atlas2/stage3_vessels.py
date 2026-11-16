#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3：奇经八脉 + 三轮七脉。

奇经：
  任/督 = 对称面切片的前/后支（几何唯一）
  带    = 水平截面环
  冲    = 任脉沿内法向的深部偏移
  维/跷 = 内/外侧走廊上的体表最短路
三轮七脉：
  中脉 = 体轴（骨架关节 + 头顶几何中心）
  左右脉 = 绕中脉的两条对映螺旋（手性对，Zb092 读法）
  七轮 = 体轴上的 7 个解剖水平（数量 7 = 粗粒化 M）
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (ROOT, load_obj, build_graph, slice_mesh, ray_surface, route,
                  adj_list, flow_smooth, resample, smooth_project, spline_smooth,
                  vertex_faces, closest_point_mesh, vertex_normals)
from stage2_paths import Frame, read_joints

OUT = ROOT / 'data' / 'atlas2'
S1 = json.load(open(OUT / 'stage1.json'))
S2 = json.load(open(OUT / 'paths12.json'))

VESSEL_ROUTES = {
    'YINWEI': dict(name='阴维脉', color='#59c3c3', st=[
        ('L', 0.97, -110), ('L', 0.70, -105), ('L', 0.45, -100), ('L', 0.20, -95),
        ('F', 0.50, -0.20), ('F', 0.45, 1.00), ('F', 0.50, 2.50), ('F', 0.50, 3.80),
        ('F', 0.35, 5.00), ('F', 0.20, 5.90)]),
    'YANGWEI': dict(name='阳维脉', color='#9b8cff', st=[
        ('L', 0.97, 110), ('L', 0.70, 108), ('L', 0.45, 105), ('L', 0.20, 100),
        ('S', -0.20), ('S', 2.00), ('S', 3.40), ('S', 4.60), ('S', 6.10), ('S', 7.20)]),
    'YINQIAO': dict(name='阴跷脉', color='#5ec8e5', st=[
        ('P', 1.98, -8.15, 1.05), ('L', 0.97, -150), ('L', 0.70, -145), ('L', 0.45, -138),
        ('L', 0.20, -130), ('F', 0.30, -0.30), ('F', 0.22, 1.00), ('F', 0.18, 2.50),
        ('F', 0.16, 3.80), ('F', 0.15, 5.20), ('F', 0.12, 6.10), ('F', 0.12, 7.28)]),
    'YANGQIAO': dict(name='阳跷脉', color='#d98cff', st=[
        ('P', 2.62, -8.10, 0.90), ('L', 0.97, 145), ('L', 0.70, 142), ('L', 0.45, 138),
        ('L', 0.20, 132), ('S', -0.20), ('S', 1.60), ('S', 3.20), ('S', 4.60),
        ('S', 6.10), ('F', 0.55, 7.30)]),
}


def densify(P, step=0.05):
    P = np.asarray(P, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for i in range(1, k + 1):
            out.append(a + (b - a) * i / k)
    return np.asarray(out)


def main():
    V, F = load_obj()
    J = read_joints()
    fr = Frame(V, F, J)
    E, L, G = build_graph(V, F)
    adj = adj_list(E, len(V))
    tree = cKDTree(V)
    v2f = vertex_faces(F, len(V))
    nrm = vertex_normals(V, F)
    out = {'vessels': {}, 'channels': {}, 'chakras': []}

    # ---------- 任脉 / 督脉（对称面切片）
    for key, nm, col in (('REN', '任脉', '#ff5d8f'), ('DU', '督脉', '#ff8c42')):
        P = np.array(S1['rendu']['ren' if key == 'REN' else 'du'])
        S = resample(P, 240)
        out['vessels'][key] = dict(name=nm, color=col, kind='midline',
                                   right=S.round(4).tolist(), length=float(
                                       np.linalg.norm(np.diff(S, axis=0), axis=1).sum()))
        print(key, 'n', len(S), 'len', round(out['vessels'][key]['length'], 2))

    # ---------- 带脉（水平截面环）
    best = None
    for yy in (1.20, 1.00, 0.80, 1.40, 0.60):
        loops = [l for l in slice_mesh(V, F, 1, yy) if abs(l[:, 0].mean()) < 1.0 and len(l) > 20]
        if loops:
            best = max(loops, key=lambda l: len(l)) if best is None else best
            if len(best) < 40:
                best = max(loops, key=lambda l: len(l))
            break
    if best is not None:
        S = resample(np.vstack([best, best[:1]]), 220)[:-1]
        out['vessels']['DAI'] = dict(name='带脉', color='#43d9ad', kind='ring',
                                     right=S.round(4).tolist(), closed=True,
                                     length=float(np.linalg.norm(np.diff(np.vstack([S, S[:1]]), axis=0), axis=1).sum()))
        print('DAI loop', len(best), 'len', round(out['vessels']['DAI']['length'], 2))

    # ---------- 冲脉：任脉沿内法向偏移
    ren = np.array(out['vessels']['REN']['right'])
    iv = tree.query(ren)[1]
    nv = nrm[iv]
    # 按局部体厚限制内偏深度：从体表向内打射线，取对侧壁距离的一半，最大 3.2 mm×10
    depth = np.zeros(len(ren))
    for i, (p, n) in enumerate(zip(ren, nv)):
        q, t = ray_surface(V, F, p - n * 1e-3, -n, maxdist=2.0, mode='first')
        depth[i] = t if q is not None else 0.0
    d_use = np.clip(0.5 * depth, 0.0, 0.32)
    # 逐点法向/体厚会抖：沿曲线平滑方向和深度（21 点），避免冲脉在腹部出现锯齿
    k = 21
    nvs = np.vstack([np.convolve(np.pad(nv[:, j], (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
                     for j in range(3)]).T
    nvs /= (np.linalg.norm(nvs, axis=1, keepdims=True) + 1e-9)
    d_use = np.convolve(np.pad(d_use, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
    inner = ren - nvs * d_use[:, None]
    # 口/颌凹处法向翻转，冲脉止于咽喉（y<=6.0），不在面部绕圈
    keep = ren[:, 1] <= 6.0
    if keep.sum() > 20:
        inner = inner[keep]
    Q, dd = closest_point_mesh(inner, V, F, tree, v2f)
    S = resample(inner, 200)
    out['vessels']['CHONG'] = dict(name='冲脉', color='#c77dff', kind='deep',
                                   right=S.round(4).tolist(),
                                   length=float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum()))
    print('CHONG len', round(out['vessels']['CHONG']['length'], 2),
          'depth med', round(float(np.median(dd)), 3))

    # ---------- 维 / 跷
    for key, m in VESSEL_ROUTES.items():
        A = [fr.pt(s) for s in m['st']]
        idx = [int(tree.query(p)[1]) for p in A]
        ded = [idx[0]]
        for i in idx[1:]:
            if i != ded[-1]:
                ded.append(i)
        P, tot = route(G, V, ded)
        Q = flow_smooth(V, adj, P, iters=35)
        S0 = spline_smooth(V[Q], n=220, s=len(Q) * 0.006)
        S = smooth_project(S0, V, F, tree, v2f, iters=14, lam=0.25, maxstep=0.25)
        S[0], S[-1] = V[Q[0]], V[Q[-1]]
        out['vessels'][key] = dict(name=m['name'], color=m['color'], kind='surface',
                                   right=S.round(4).tolist(),
                                   length=float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum()))
        print(key, m['name'], 'len', round(out['vessels'][key]['length'], 2))

    # ---------- 中脉（体轴）
    prof = S1['profile']
    ay = np.array([p['y'] for p in prof]); az = np.array([p['z'] for p in prof])
    ar = np.array([p['r'] for p in prof])
    yy = np.linspace(ay[0], ay[-1], 220)
    axis = np.c_[np.zeros_like(yy), yy, np.interp(yy, ay, az)]
    out['channels']['SUSHUMA'] = dict(name='中脉', color='#ffe066', kind='central',
                                      right=axis.round(4).tolist(),
                                      length=float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum()))
    print('SUSHUMA len', round(out['channels']['SUSHUMA']['length'], 2))

    # ---------- 左脉/右脉（对映螺旋）
    r_of = lambda y: float(np.interp(y, ay, ar))
    for key, sgn, nm, col in (('LALA', +1, '左脉', '#7ec8ff'), ('RASA', -1, '右脉', '#ff9ecb')):
        t = np.linspace(0, 1, len(yy))
        phi = sgn * (2 * np.pi * 3.0 * t)
        rad = np.clip(0.34 * np.array([r_of(y) for y in yy]), 0.06, None)
        P = np.c_[rad * np.sin(phi), yy, np.interp(yy, ay, az) + rad * np.cos(phi)]
        out['channels'][key] = dict(name=nm, color=col, kind='helix',
                                    right=P.round(4).tolist(),
                                    length=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()))
        print(key, nm, 'len', round(out['channels'][key]['length'], 2), 'rmax', round(float(rad.max()), 2))

    # ---------- 七轮
    for c in S1['chakras']:
        y = c['y']
        z = float(np.interp(y, ay, az))
        r = float(np.interp(y, ay, ar))
        out['chakras'].append(dict(key=c['key'], name=c['name'], y=float(y), z=z,
                                   rr=round(0.55 * r, 3), radius=round(r, 3)))
    print('chakras', [(c['name'], round(c['y'], 2), c['rr']) for c in out['chakras']])

    # 左侧镜像
    def mirror(P):
        return (np.asarray(P) * [-1, 1, 1]).round(4).tolist()
    for d in (out['vessels'], out['channels']):
        for k, v in d.items():
            if 'right' in v:
                v['left'] = mirror(v['right'])

    json.dump(out, open(OUT / 'vessels_channels.json', 'w'), ensure_ascii=False)
    print('saved', OUT / 'vessels_channels.json')


if __name__ == '__main__':
    main()
