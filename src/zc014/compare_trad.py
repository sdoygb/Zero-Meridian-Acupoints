#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计：v20/v21 的 12 经与 326 推导穴位 vs 传统 361 穴，在 2D 视图空间里的近似对应。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
TH = json.load(open(ROOT / 'data' / 'atlas2' / 'theorem_atlas.json'))
ACU = json.load(open(ROOT / 'data' / 'zc014' / 'acupoints_derived.json'))
TRAD = json.load(open(ROOT / 'data' / 'acupoints.json'))['points']
V = np.array([[0, 0, 0]])
# 用网格 bbox 做归一化（与旧项目一致：x 为横、y 为高、z 为深）
import sys
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj
Vm, Fm = load_obj()
xmin, xmax = Vm[:, 0].min(), Vm[:, 0].max()
ymin, ymax = Vm[:, 1].min(), Vm[:, 1].max()
zmin, zmax = Vm[:, 2].min(), Vm[:, 2].max()
W = (xmax - xmin); H = (ymax - ymin); D = (zmax - zmin)
CM = 10.2  # 单位→cm


def proj(P, view):
    P = np.asarray(P, float)
    u = (P[:, 0] - xmin) / W          # 横
    v = (ymax - P[:, 1]) / H          # 纵（0=头顶）
    if view == 'lateral':
        u = (P[:, 2] - zmin) / D
    return np.c_[u, v]


def dist_cm(a, b):
    return np.hypot((a[0] - b[0]) * W * CM, (a[1] - b[1]) * H * CM)


rows = []
chan = {}
for c in D12['order']:
    for side in ('right', 'left'):
        P = np.array(TH['meridians'][c][side])
        for view in ('anterior', 'posterior', 'lateral'):
            chan[(c, side, view)] = proj(P, view)

T = [p for p in TRAD if p['meridian'] in D12['order']]
print('traditional points (12经):', len(T))
d_chan, d_acu = [], []
for p in T:
    mer, view = p['meridian'], p.get('view', 'anterior')
    if view not in ('anterior', 'posterior', 'lateral'):
        view = 'anterior'
    q = np.array([float(p['x']), float(p['y'])])
    best = 1e9; besta = 1e9
    for side in ('right', 'left'):
        C = chan[(mer, side, view)]
        dd = np.hypot((C[:, 0] - q[0]) * W * CM, (C[:, 1] - q[1]) * H * CM)
        best = min(best, float(dd.min()))
        # 推导穴位
        A = np.array(ACU['meridians'][mer]['pts'])
        if side == 'left':
            A = A * np.array([-1, 1, 1])
        AP = proj(A, view)
        da = np.hypot((AP[:, 0] - q[0]) * W * CM, (AP[:, 1] - q[1]) * H * CM)
        besta = min(besta, float(da.min()))
    d_chan.append(best); d_acu.append(besta)

d_chan = np.array(d_chan); d_acu = np.array(d_acu)
def rep(name, d):
    print(f"{name}: n={len(d)} median {np.median(d):.1f} cm  <=1cm {np.mean(d<=1)*100:.0f}%  "
          f"<=3cm {np.mean(d<=3)*100:.0f}%  <=5cm {np.mean(d<=5)*100:.0f}%  p90 {np.percentile(d,90):.1f}")
rep('传统点 → 最近推导经络', d_chan)
rep('传统点 → 最近推导穴位', d_acu)

# 反方向：推导穴位 → 最近传统点（同经）
rev = []
for mer in D12['order']:
    A = np.array(ACU['meridians'][mer]['pts'])
    Ts = [p for p in T if p['meridian'] == mer]
    if not Ts: continue
    for side in (1, -1):
        AA = A * np.array([side, 1, 1])
        for view in ('anterior', 'posterior', 'lateral'):
            AP = proj(AA, view)
            for p in Ts:
                pv = p.get('view', 'anterior')
                if pv not in ('anterior', 'posterior', 'lateral'): pv = 'anterior'
                if pv != view: continue
                q = np.array([float(p['x']), float(p['y'])])
                dd = np.hypot((AP[:, 0] - q[0]) * W * CM, (AP[:, 1] - q[1]) * H * CM)
                rev.append(float(dd.min()))
rev = np.array(rev)
rep('推导穴位 → 最近传统点', rev)
json.dump({'trad_to_chan': d_chan.tolist(), 'trad_to_acu': d_acu.tolist(), 'acu_to_trad': rev.tolist()},
          open(ROOT / 'data' / 'zc014' / 'compare_trad.json', 'w'))
print('saved compare_trad.json')
