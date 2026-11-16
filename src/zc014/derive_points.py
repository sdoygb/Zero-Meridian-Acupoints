#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定稿：φ₁（端口调和场，win=9, prom=0.05）的极值 → 穴位候选 3D 点。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'zc014'
D = json.load(open(OUT / 'fields.json'))
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
TH = json.load(open(ROOT / 'data' / 'atlas2' / 'theorem_atlas.json'))


def extrema(s, win, prom):
    k = win
    sm = np.convolve(np.pad(s, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
    d = np.diff(sm); sgn = np.sign(d)
    idx = np.where(np.abs(np.diff(sgn)) > 0)[0] + 1
    rng = s.max() - s.min()
    keep = []
    for i in idx:
        left = sm[max(0, i - 12):i + 1]; right = sm[i:min(len(sm), i + 13)]
        if sm[i] == left.max() and (sm[i] - min(left.min(), right.min())) > prom * rng:
            keep.append(int(i))
        elif sm[i] == left.min() and (max(left.max(), right.max()) - sm[i]) > prom * rng:
            keep.append(int(i))
    return keep, sm


out = {'meta': {'field': 'phi1 (port Dirichlet harmonic)', 'win': 9, 'prom': 0.05},
       'meridians': {}}
tot = 0
for c in D12['order']:
    idx = np.array(D['channels'][c]['idx'])
    s = np.array(D['phi1'])[idx]
    ex, sm = extrema(s, 9, 0.05)
    P = np.array(TH['meridians'][c]['right'])
    # 通道 240 点，索引按比例映射
    pos = [P[min(len(P) - 1, int(round(i * len(P) / len(idx))))] for i in ex]
    out['meridians'][c] = dict(n=len(ex), idx=ex,
                               pts=[[round(float(x), 3) for x in p] for p in pos],
                               frac=[round(i / max(1, len(idx) - 1), 4) for i in ex],
                               phi=[round(float(sm[i]), 3) for i in ex])
    tot += len(ex)
out['total'] = tot
json.dump(out, open(OUT / 'acupoints_derived.json', 'w'), ensure_ascii=False)
print('total', tot)
for c in D12['order']:
    print(c, out['meridians'][c]['n'])
