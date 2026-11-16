#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""敏感度：把 12 经的躯干走廊站横向平移 0.08 单位（≈0.8 cm），重新算路径，量位移。
这是“横向位置里有多少是结构选择(I)”的直接估计。"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
import stage2_paths as S2
from core import load_obj, build_graph, route, flow_smooth, spline_smooth, smooth_project, vertex_faces, resample

V, F = load_obj()
E, L, G = build_graph(V, F)
tree = cKDTree(V)
v2f = vertex_faces(F, len(V))
fr = S2.Frame(V, F, S2.read_joints())
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))


def run(mer):
    out = {}
    for code in S2.ORDER:
        A = [fr.pt(s) for s in mer[code]['st']]
        idx = [int(tree.query(p)[1]) for p in A]
        ded = [idx[0]]
        for i in idx[1:]:
            if i != ded[-1]:
                ded.append(i)
        P, tot = route(G, V, ded)
        Q = flow_smooth(V, adj, P, iters=60)
        S = spline_smooth(V[Q], n=260, s=len(Q) * 0.006)
        S = smooth_project(S, V, F, tree, v2f, iters=18, lam=0.25, maxstep=0.25)
        S[0], S[-1] = V[Q[0]], V[Q[-1]]
        out[code] = resample(S, 240)
    return out


from core import adj_list
adj = adj_list(E, len(V))
base = run(S2.MER)
mer2 = copy.deepcopy(S2.MER)
for c in mer2:
    st = []
    for s in mer2[c]['st']:
        if s[0] == 'F':
            st.append(('F', s[1] + 0.08, s[2]))
        else:
            st.append(s)
    mer2[c]['st'] = st
pert = run(mer2)
res = {}
for c in S2.ORDER:
    A = np.array(base[c]); B = np.array(pert[c])
    D = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    d = np.array([D[i, max(0, i - 4):min(len(B), i + 5)].min() for i in range(len(A))])
    res[c] = dict(mean=float(d.mean()) * 10.2, p90=float(np.percentile(d, 90)) * 10.2)
    print(c, 'mean %.2f cm  p90 %.2f cm' % (d.mean() * 10.2, np.percentile(d, 90) * 10.2))
allm = np.mean([v['mean'] for v in res.values()])
print('ALL mean displacement %.2f cm' % allm)
json.dump(res, open(ROOT / 'data' / 'zc014' / 'sensitivity.json', 'w'))
