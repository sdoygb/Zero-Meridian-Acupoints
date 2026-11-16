#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 MakeHuman OBJ 的 joint-* 组读出骨架关节位置（模型自带解剖数据）。"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'data' / 'model' / 'human_neutral_makehuman.obj'


def read_joints(path=MODEL):
    g, dat, V = None, {}, []
    for line in open(path, encoding='utf-8', errors='ignore'):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('g '):
            g = line.split()[1]
        elif line.startswith('f ') and g and g.startswith('joint'):
            dat.setdefault(g, []).extend(int(t.split('/')[0]) - 1 for t in line.split()[1:])
    V = np.asarray(V, float)
    return {k[6:]: V[sorted(set(v))].mean(0) for k, v in dat.items() if v}


if __name__ == '__main__':
    J = read_joints()
    for k in sorted(J):
        print(f'{k:20s} {np.round(J[k], 3)}')
