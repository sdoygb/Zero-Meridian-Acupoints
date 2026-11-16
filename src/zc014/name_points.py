#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按经内序号比例给推导穴位配传统名（ordinal 对应，不做空间最近——2D 投影姿态差异太大）。
每个传统穴名挂在对应相对位置的那个推导点上；其余点保留系统编号。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
trad = json.load(open(ROOT / 'data' / 'acupoints.json'))['points']
der = json.load(open(ROOT / 'data' / 'zc014' / 'acupoints_derived.json'))

by = {}
for p in trad:
    by.setdefault(p['meridian'], []).append(p)

out = {'meta': {'method': 'ordinal within meridian (relative index)', 'note': 'not spatial nearest'},
       'meridians': {}}
for mer, pts in by.items():
    if mer not in der['meridians']:
        continue
    m = der['meridians'][mer]['n']; n = len(pts)
    assign = {}
    for k in range(n):
        idx = int(round((k + 0.5) / n * m - 0.5))
        idx = max(0, min(m - 1, idx))
        assign.setdefault(idx, []).append(dict(hanzi=pts[k]['hanzi'], id=pts[k]['id']))
    out['meridians'][mer] = {'n_derived': m, 'n_trad': n, 'assign': assign}
    print(mer, 'derived', m, 'trad', n, 'first', assign.get(0), 'last', assign.get(m - 1))
json.dump(out, open(ROOT / 'data' / 'zc014' / 'acupoints_named.json', 'w'), ensure_ascii=False)
print('saved')
