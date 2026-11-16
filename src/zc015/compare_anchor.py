#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""锚定点 vs 推导经络/穴位 vs 2D 图表点：把参照误差与 φ 误差分开。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj

V, F = load_obj()
AN = json.load(open(ROOT / 'data' / 'zc015' / 'anchors.json'))['points']
TH = json.load(open(ROOT / 'data' / 'atlas2' / 'theorem_atlas.json'))
ACU = json.load(open(ROOT / 'data' / 'zc014' / 'acupoints_derived.json'))
TRAD = json.load(open(ROOT / 'data' / 'acupoints.json'))['points']
CM = 10.2
MER = {'PC6': 'PC', 'LU9': 'LU', 'HT7': 'HT', 'LI4': 'LI', 'LI11': 'LI',
       'ST36': 'ST', 'SP6': 'SP', 'GB34': 'GB', 'GB39': 'GB', 'KI3': 'KI'}
xmin, xmax = V[:, 0].min(), V[:, 0].max()
ymin, ymax = V[:, 1].min(), V[:, 1].max()
zmin, zmax = V[:, 2].min(), V[:, 2].max()
Wb, Hb, Db = xmax - xmin, ymax - ymin, zmax - zmin
views = ('anterior', 'posterior', 'lateral')


def proj1(P, v):
    u = (P[0] - xmin) / Wb; w = (ymax - P[1]) / Hb
    if v == 'lateral':
        u = (P[2] - zmin) / Db
    return np.array([u, w])


rows = []
for name, q in AN.items():
    mer = MER[name]
    q = np.array(q, float)
    # 1) 到推导经络（同经、双侧）
    best_c = 1e9
    for side in ('right', 'left'):
        C = np.array(TH['meridians'][mer][side])
        best_c = min(best_c, float(np.min(np.linalg.norm(C - q, axis=1))))
    # 2) 到推导穴位（同经）
    A = np.array(ACU['meridians'][mer]['pts'])
    best_a = min(float(np.min(np.linalg.norm(A - q, axis=1))),
                 float(np.min(np.linalg.norm(A * np.array([-1, 1, 1]) - q, axis=1))))
    # 3) 到传统 2D 图表点（同经，任一视图）
    best_t = 1e9
    for p in TRAD:
        if p['meridian'] != mer:
            continue
        v = p.get('view', 'anterior')
        if v not in views:
            v = 'anterior'
        u = proj1(q, v)
        d = np.hypot((u[0] - float(p['x'])) * Wb * CM, (u[1] - float(p['y'])) * Hb * CM)
        best_t = min(best_t, float(d))
    rows.append((name, mer, best_c * CM, best_a * CM, best_t))
    print(f'{name:5s} {mer:3s} 锚点→经络 {best_c*CM:5.1f} cm  锚点→推导穴 {best_a*CM:5.1f} cm  锚点→图表点 {best_t:5.1f} cm')

arr = np.array([[r[2], r[3], r[4]] for r in rows])
print('\nmedian: 经络 %.1f  推导穴 %.1f  图表点 %.1f cm' % tuple(np.median(arr, axis=0)))
print('max:    经络 %.1f  推导穴 %.1f  图表点 %.1f cm' % tuple(arr.max(axis=0)))
json.dump([{'point': r[0], 'mer': r[1], 'to_chan_cm': round(r[2], 1),
            'to_acu_cm': round(r[3], 1), 'to_chart_cm': round(r[4], 1)} for r in rows],
          open(ROOT / 'data' / 'zc015' / 'anchor_compare.json', 'w'), ensure_ascii=False, indent=1)
