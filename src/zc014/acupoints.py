#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 φ 场提取穴位候选（沿通道极值），核 1D index 定理，与传统 361 的每经计数对照。"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'zc014'
D = json.load(open(OUT / 'fields.json'))
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
TRAD = {'LU': 11, 'LI': 20, 'ST': 45, 'SP': 21, 'HT': 9, 'SI': 19,
        'BL': 67, 'KI': 27, 'PC': 9, 'TE': 23, 'GB': 44, 'LV': 14}


def extrema(s, win, prom):
    k = win
    sm = np.convolve(np.pad(s, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
    d = np.diff(sm)
    sgn = np.sign(d)
    idx = np.where(np.abs(np.diff(sgn)) > 0)[0] + 1
    # prominence 过滤
    rng = s.max() - s.min()
    keep = []
    for i in idx:
        left = sm[max(0, i - 12):i + 1]; right = sm[i:min(len(sm), i + 13)]
        if sm[i] == left.max() and (sm[i] - min(left.min(), right.min())) > prom * rng:
            keep.append(i)
        elif sm[i] == left.min() and (max(left.max(), right.max()) - sm[i]) > prom * rng:
            keep.append(i)
    return np.array(keep, int), sm


def run(phi, win, prom, tag):
    rows = {}
    for c in D12['order']:
        s = np.array(phi)[np.array(D['channels'][c]['idx'])]
        ex, sm = extrema(s, win, prom)
        # index 定理：极大-极小∈{-1,0,1}
        d = np.diff(sm); sgn = np.sign(d); sgn = sgn[sgn != 0]
        nmax = int(((sgn[:-1] > 0) & (sgn[1:] < 0)).sum()) if len(sgn) > 1 else 0
        nmin = int(((sgn[:-1] < 0) & (sgn[1:] > 0)).sum()) if len(sgn) > 1 else 0
        rows[c] = dict(n=len(ex), nmax=nmax, nmin=nmin, idx=ex.tolist())
    tot = sum(r['n'] for r in rows.values())
    keys = list(TRAD)
    a = [rows[k]['n'] for k in keys]; b = [TRAD[k] for k in keys]
    rho, p = spearmanr(a, b)
    ok = all(abs(r['nmax'] - r['nmin']) <= 1 for r in rows.values())
    print(f'{tag}: total {tot}, per-meridian {[rows[k]["n"] for k in keys]}, trad {b}, spearman {rho:.2f} (p={p:.3f}), index 1D {ok}')
    return rows, dict(total=tot, spearman=float(rho), p=float(p), index_ok=bool(ok),
                      counts=[int(rows[k]['n']) for k in keys], trad=b)


res = {}
for phi_name, phi in (('phi1', D['phi1']), ('phi2', D['phi2'])):
    for win in (5, 9, 13):
        for prom in (0.05, 0.10, 0.15):
            rows, summ = run(phi, win, prom, f'{phi_name} win{win} prom{prom}')
            res[f'{phi_name}_w{win}_p{int(prom*100)}'] = dict(rows=rows, summary=summ)
json.dump(res, open(OUT / 'acupoints_scan.json', 'w'))
print('saved')
