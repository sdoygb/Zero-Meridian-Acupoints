#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2：12 经（体表最短路）＋ 流注过渡。

路线 = 走廊站之间在体表网格上的最短测地线（Zc012 §10 的 C 内 min-cost 路径）。
站只有两类信息：**哪一段体表**（前/后/侧 x 偏移；四肢角向）＋**端口**（模型骨架关节）。
不取任何传统经络图的坐标；12 条互不重合由端口与走廊分担保证。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (ROOT, load_obj, build_graph, ray_surface, route,
                  flow_smooth, densify, adj_list, resample, smooth_project,
                  vertex_faces, relax_separation, spline_smooth)
from stage1_prep_joints import read_joints

OUT = ROOT / 'data' / 'atlas2'
S1 = json.load(open(OUT / 'stage1.json'))


class Frame:
    def __init__(self, V, F, J):
        self.V, self.F, self.J = V, F, J
        prof = S1['profile']
        self.py = np.array([p['y'] for p in prof])
        self.pz = np.array([p['z'] for p in prof])
        self.pr = np.array([p['r'] for p in prof])
        ks = ['pelvis', 'spine-4', 'spine-3', 'spine-2', 'spine-1', 'neck', 'head', 'head-2']
        fz = PchipInterpolator([-0.42] + [J[k][1] for k in ks] + [8.45],
                               [0.36] + [J[k][2] for k in ks] + [J['head-2'][2]])
        self.fz = fz
        self.arm = np.array([J['l-shoulder'], J['l-elbow'], J['l-hand']])
        self.leg = np.array([J['l-upper-leg'], J['l-knee'], J['l-ankle']])

    def zc(self, y):
        return float(np.interp(y, self.py, self.pz))

    def radius(self, y):
        return float(np.interp(y, self.py, self.pr))

    @staticmethod
    def _chain_pt(chain, t):
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        d = t * s[-1]
        i = min(int(np.searchsorted(s, d, side='right') - 1), len(seg) - 1)
        f = (d - s[i]) / seg[i]
        return chain[i] + f * (chain[i + 1] - chain[i]), (chain[i + 1] - chain[i]) / seg[i]

    @staticmethod
    def _frame(a):
        a = a / np.linalg.norm(a)
        ref = np.array([0, 0, 1.0])
        if abs(a @ ref) > 0.95:
            ref = np.array([0, 1.0, 0])
        u = ref - a * (ref @ a)
        u /= np.linalg.norm(u)
        return u, np.cross(u, a)

    def pt(self, spec):
        k = spec[0]
        if k in ('F', 'B'):
            x, y = spec[1], spec[2]
            o = np.array([x, y, self.zc(y)])
            d = np.array([0.0, 0.0, 1.0 if k == 'F' else -1.0])
            q, _ = ray_surface(self.V, self.F, o, d, maxdist=2.2, mode='first')
            return q if q is not None else o
        if k == 'S':
            y = spec[1]
            o = np.array([0.0, y, self.zc(y)])
            sgn = 1.0 if (len(spec) < 3 or spec[2] > 0) else -1.0
            q, _ = ray_surface(self.V, self.F, o, np.array([sgn, 0, 0]), maxdist=3.2, mode='first')
            return q if q is not None else o
        if k == 'A':
            p, a = self._chain_pt(self.arm, spec[1])
            u, v = self._frame(a)
            th = np.radians(spec[2])
            q, _ = ray_surface(self.V, self.F, p, np.cos(th) * u + np.sin(th) * v,
                               maxdist=1.4, mode='first')
            return q if q is not None else p
        if k == 'L':
            p, a = self._chain_pt(self.leg, spec[1])
            u, v = self._frame(a)
            th = np.radians(spec[2])
            q, _ = ray_surface(self.V, self.F, p, np.cos(th) * u + np.sin(th) * v,
                               maxdist=1.4, mode='first')
            return q if q is not None else p
        if k == 'J':
            return self.J[spec[1]]
        if k == 'P':
            return np.array(spec[1:4], float)
        raise ValueError(k)


MER = {
    'LU': dict(name='手太阴肺经', color='#6ec6ff', kind='yin', st=[
        ('F', 0.95, 4.60), ('F', 1.15, 4.95), ('A', 0.04, 40), ('A', 0.25, 38),
        ('A', 0.50, 34), ('A', 0.75, 30), ('A', 0.92, 25), ('J', 'l-finger-1-4')]),
    'LI': dict(name='手阳明大肠经', color='#ffb020', kind='yang', st=[
        ('J', 'l-finger-2-4'), ('A', 0.92, 150), ('A', 0.75, 148), ('A', 0.50, 145),
        ('A', 0.25, 142), ('A', 0.05, 134), ('B', 1.10, 4.95), ('B', 0.55, 5.7),
        ('S', 6.05), ('F', 0.46, 6.62), ('F', 0.24, 6.84)]),
    'ST': dict(name='足阳明胃经', color='#ffd166', kind='yang', st=[
        ('F', 0.32, 7.26), ('F', 0.34, 6.90), ('F', 0.32, 6.55), ('F', 0.30, 6.05),
        ('F', 0.42, 5.35), ('F', 0.52, 4.40), ('F', 0.50, 3.20), ('F', 0.48, 2.20),
        ('F', 0.46, 1.20), ('F', 0.44, 0.20), ('F', 0.42, -0.32),
        ('L', 0.06, 5), ('L', 0.30, 2), ('L', 0.50, 0), ('L', 0.70, 0),
        ('L', 0.90, 2), ('J', 'l-toe-2-4')]),
    'SP': dict(name='足太阴脾经', color='#f4a261', kind='yin', st=[
        ('P', 1.80, -8.00, 1.90), ('L', 0.95, -25), ('L', 0.70, -30),
        ('L', 0.50, -35), ('L', 0.25, -45), ('L', 0.08, -62),
        ('F', 1.05, -0.30), ('F', 1.15, 0.60), ('F', 1.18, 1.60),
        ('F', 1.22, 2.60), ('F', 1.10, 3.40)]),
    'HT': dict(name='手少阴心经', color='#ff6b6b', kind='yin', st=[
        ('F', 1.30, 4.90), ('S', 5.02), ('A', 0.03, -45), ('A', 0.25, -42),
        ('A', 0.50, -40), ('A', 0.75, -35), ('A', 0.92, -30), ('P', 4.83, 1.36, 2.30)]),
    'SI': dict(name='手太阳小肠经', color='#f72585', kind='yang', st=[
        ('P', 4.91, 1.24, 2.42), ('A', 0.92, 215), ('A', 0.75, 210), ('A', 0.50, 205),
        ('A', 0.25, 200), ('A', 0.05, 190), ('B', 1.20, 5.00), ('B', 0.55, 5.75),
        ('B', 0.30, 6.60), ('B', 0.40, 7.05)]),
    'BL': dict(name='足太阳膀胱经', color='#3f8efc', kind='yang', st=[
        ('F', 0.10, 7.28), ('F', 0.18, 7.75), ('F', 0.20, 8.15), ('B', 0.20, 8.15),
        ('B', 0.30, 7.70), ('B', 0.30, 6.20), ('B', 0.35, 4.80), ('B', 0.35, 3.00),
        ('B', 0.35, 1.20), ('B', 0.35, -0.20), ('L', 0.06, 180), ('L', 0.30, 180),
        ('L', 0.50, 178), ('L', 0.70, 175), ('L', 0.90, 172), ('J', 'l-toe-5-4')]),
    'KI': dict(name='足少阴肾经', color='#118ab2', kind='yin', st=[
        ('P', 2.05, -8.17, 1.45), ('L', 0.95, -140), ('L', 0.70, -135),
        ('L', 0.50, -130), ('L', 0.25, -125), ('L', 0.08, -120),
        ('F', 0.30, -0.30), ('F', 0.22, 0.60), ('F', 0.18, 1.60),
        ('F', 0.16, 2.60), ('F', 0.15, 3.50)]),
    'PC': dict(name='手厥阴心包经', color='#b5179e', kind='yin', st=[
        ('F', 0.72, 3.90), ('F', 0.95, 4.55), ('A', 0.05, 2), ('A', 0.25, 0),
        ('A', 0.50, 0), ('A', 0.75, 0), ('A', 0.92, 0), ('J', 'l-finger-3-4')]),
    'TE': dict(name='手少阳三焦经', color='#8f5cf7', kind='yang', st=[
        ('J', 'l-finger-4-4'), ('A', 0.92, 180), ('A', 0.75, 180), ('A', 0.50, 180),
        ('A', 0.25, 180), ('A', 0.05, 177), ('B', 1.15, 5.00), ('B', 0.45, 5.8),
        ('S', 6.15), ('S', 7.05), ('S', 7.35)]),
    'GB': dict(name='足少阳胆经', color='#06d6a0', kind='yang', st=[
        ('F', 0.48, 7.52), ('S', 7.32), ('S', 6.10), ('S', 5.00), ('S', 3.20),
        ('S', 1.40), ('S', -0.30), ('L', 0.06, 68), ('L', 0.30, 70),
        ('L', 0.50, 72), ('L', 0.70, 75), ('L', 0.90, 78), ('J', 'l-toe-4-4')]),
    'LV': dict(name='足厥阴肝经', color='#9bf6ff', kind='yin', st=[
        ('P', 2.06, -8.00, 1.86), ('L', 0.95, -55), ('L', 0.70, -65),
        ('L', 0.50, -75), ('L', 0.25, -85), ('L', 0.08, -96),
        ('F', 0.62, -0.30), ('F', 0.70, 0.60), ('F', 0.78, 1.60),
        ('F', 0.85, 2.60), ('F', 0.80, 3.00)]),
}
ORDER = ['LU', 'LI', 'ST', 'SP', 'HT', 'SI', 'BL', 'KI', 'PC', 'TE', 'GB', 'LV']

# 流注：体表过渡（手/足/头）与胸腔内部过渡
SURF_TRANS = [('LU', 'LI'), ('ST', 'SP'), ('HT', 'SI'), ('PC', 'TE'), ('GB', 'LV'),
              ('LI', 'ST'), ('SI', 'BL'), ('TE', 'GB')]
DEEP_TRANS = [('SP', 'HT'), ('KI', 'PC'), ('LV', 'LU')]


def main():
    V, F = load_obj()
    J = read_joints()
    fr = Frame(V, F, J)
    E, L, G = build_graph(V, F)
    adj = adj_list(E, len(V))
    tree = cKDTree(V)
    v2f = vertex_faces(F, len(V))

    paths, anchors = {}, {}
    for code in ORDER:
        m = MER[code]
        A = [fr.pt(s) for s in m['st']]
        idx = [int(tree.query(p)[1]) for p in A]
        ded = [idx[0]]
        for i in idx[1:]:
            if i != ded[-1]:
                ded.append(i)
        P, tot = route(G, V, ded)
        if P is None:
            print('FAIL', code); continue
        Q = flow_smooth(V, adj, P, iters=60)
        S0 = spline_smooth(V[Q], n=260, s=len(Q) * 0.006)
        S = smooth_project(S0, V, F, tree, v2f, iters=18, lam=0.25, maxstep=0.25)
        S[0], S[-1] = V[Q[0]], V[Q[-1]]
        S = resample(S, 240)
        Pm = S.copy(); Pm[:, 0] *= -1
        paths[code] = dict(name=m['name'], color=m['color'], kind=m['kind'],
                           right=S.round(4).tolist(), left=Pm.round(4).tolist(),
                           n=len(S), length=float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum()))
        anchors[code] = [a.round(3).tolist() for a in A]
        print(f"{code} {m['name']}: {paths[code]['length']:6.2f}  n {len(S):4d}")

    # ---- 可分辨性松弛（通道之间保持最小间距）
    C = {c: np.array(paths[c]['right']) for c in ORDER}
    C = relax_separation(C, V, F, tree, v2f, dmin=0.065, iters=10, step=0.35)
    for c in ORDER:
        P = C[c]
        paths[c]['right'] = P.round(4).tolist()
        paths[c]['left'] = (P * [-1, 1, 1]).round(4).tolist()
        paths[c]['length'] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        paths[c]['n'] = len(P)

    trans = []
    for a, b in SURF_TRANS:
        pa = np.array(paths[a]['right'][-1]); pb = np.array(paths[b]['right'][0])
        ia, ib = int(tree.query(pa)[1]), int(tree.query(pb)[1])
        P, tot = route(G, V, [ia, ib])
        if P is not None and len(P) > 4:
            Q = flow_smooth(V, adj, P, iters=12)
            S = smooth_project(resample(V[Q], max(20, min(60, len(Q)))), V, F, tree, v2f,
                               iters=10, lam=0.3, maxstep=0.2)
        else:
            S = V[P]
        trans.append(dict(a=a, b=b, kind='surface', right=S.round(4).tolist(),
                          left=(S * [-1, 1, 1]).round(4).tolist(), length=float(tot)))
        print(f"trans {a}->{b} {tot:.2f}")
    for a, b in DEEP_TRANS:
        pa = np.array(paths[a]['right'][-1]); pb = np.array(paths[b]['right'][0])
        c = 0.5 * (pa + pb)
        n = 24
        t = np.linspace(0, 1, n)[:, None]
        S = pa * (1 - t) + pb * t
        # 向体内压 0.35（沿 -法向近似：取胸口中心方向）
        S[:, 2] -= 0.35 * np.sin(np.pi * t[:, 0])[:, None][:, 0]
        trans.append(dict(a=a, b=b, kind='deep', right=S.round(4).tolist(),
                          left=(S * [-1, 1, 1]).round(4).tolist(), length=float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum())))
        print(f"deep {a}->{b}")

    # ---- 质量指标
    def pairwise(side='right'):
        out = {}
        for a, b in itertools.combinations(ORDER, 2):
            A = np.array(paths[a][side]); B = np.array(paths[b][side])
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            out[f'{a}-{b}'] = dict(min=float(d.min()), p10=float(np.percentile(d, 10)),
                                   mean=float(d.mean()))
        return out
    pw = pairwise()
    json.dump({'meridians': paths, 'transitions': trans, 'anchors': anchors,
               'pairwise': pw, 'order': ORDER,
               'meta': {'unit_cm': 10.0, 'model': 'MakeHuman hm08 body (CC0)',
                        'source': 'data/model/human_neutral_makehuman.obj'}},
              open(OUT / 'paths12.json', 'w'), ensure_ascii=False)
    close = sorted(pw.items(), key=lambda kv: kv[1]['min'])[:10]
    print('\nclosest:', [(k, round(v['min'], 2), round(v['p10'], 2)) for k, v in close])
    # 对称性（左=镜像）
    err = max(np.abs(np.array(paths[c]['left']) - np.array(paths[c]['right']) * [-1, 1, 1]).max()
              for c in ORDER)
    print('mirror err', err)


if __name__ == '__main__':
    main()
