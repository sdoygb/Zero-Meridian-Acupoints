#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc014：穴位＝φ 沿通道的极值。两个 φ 代理：
 φ₁ = 24 端口 Dirichlet 的调和场（框架的边界通量读法）
 φ₂ = 骨距场（结构代理，负=离骨近）
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj, build_graph
from stage2_paths import read_joints

OUT = ROOT / 'data' / 'zc014'
OUT.mkdir(parents=True, exist_ok=True)
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
CM = 0.102  # 1 cm 单位


def bone_segments(J):
    return [
        (J['pelvis'], J['spine-4'], 0.12), (J['spine-4'], J['spine-3'], 0.12),
        (J['spine-3'], J['spine-2'], 0.12), (J['spine-2'], J['spine-1'], 0.12),
        (J['spine-1'], J['neck'], 0.10), (J['neck'], J['head'], 0.10),
        (J['spine-1'], J['l-shoulder'], 0.07), (J['spine-1'], J['r-shoulder'], 0.07),
        (J['l-shoulder'], J['l-elbow'], 0.09), (J['r-shoulder'], J['r-elbow'], 0.09),
        (J['l-elbow'], J['l-hand'], 0.055), (J['r-elbow'], J['r-hand'], 0.055),
        (J['l-upper-leg'], J['l-knee'], 0.11), (J['r-upper-leg'], J['r-knee'], 0.11),
        (J['l-knee'], J['l-ankle'], 0.09), (J['r-knee'], J['r-ankle'], 0.09),
        (J['l-upper-leg'], J['pelvis'], 0.10), (J['r-upper-leg'], J['pelvis'], 0.10),
    ]


def bone_dist(P, segs):
    d = np.full(len(P), 1e9)
    for a, b, r in segs:
        a = np.asarray(a, float); b = np.asarray(b, float)
        ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
        q = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(P - q, axis=1) - r)
    c = np.array([0.0, 7.55, 0.45])
    return np.minimum(d, np.linalg.norm(P - c, axis=1) - 0.52)


def main():
    V, F = load_obj()
    E, L, G = build_graph(V, F)
    n = len(V)
    tree = cKDTree(V)
    J = read_joints()
    segs = bone_segments(J)
    # 端口：每条经的起止点
    fixed = {}
    for c in D12['order']:
        P = np.array(D12['meridians'][c]['right'])
        ia = int(tree.query(P[0])[1]); ib = int(tree.query(P[-1])[1])
        fixed[ia] = +1.0; fixed[ib] = -1.0
    free = np.ones(n, bool); free[list(fixed)] = False
    A = G.tocsr()
    phi = np.zeros(n)
    phi[free] = spsolve(A[free][:, free].tocsc(), -A[free][:, list(fixed)] @ np.array(list(fixed.values())))
    phi[list(fixed)] = np.array(list(fixed.values()))
    # 骨距场（负=近骨）
    db = bone_dist(V, segs)
    phi2 = -db
    out = {'phi1': phi.round(4).tolist(), 'phi2': phi2.round(4).tolist(),
           'ports': {str(k): v for k, v in fixed.items()}}
    # 沿每条通道采样
    for c in D12['order']:
        P = np.array(D12['meridians'][c]['right'])
        idx = tree.query(P)[1]
        out.setdefault('channels', {})[c] = dict(idx=idx.tolist())
    json.dump(out, open(OUT / 'fields.json', 'w'))
    print('phi1 range', round(float(phi.min()), 2), round(float(phi.max()), 2),
          'phi2 range', round(float(phi2.min()), 2), round(float(phi2.max()), 2), 'saved')


if __name__ == '__main__':
    main()
