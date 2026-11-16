#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TARA 点云配准（有序对应版）：每条经的 TARA 穴按序号、我们的通道按弧长同分位，
拟合相似变换（s,R,t）后量残差。这样尺度不会退化成 0。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj
V, F = load_obj()
TARA = json.load(open(ROOT / 'data' / 'zc015' / 'tara_points.json'))['points']
TH = json.load(open(ROOT / 'data' / 'atlas2' / 'theorem_atlas.json'))
CM = 10.2
MAP = {'LR': 'LV'}


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / ((Xc ** 2).sum() / len(X))
    t = my - s * R @ mx
    return s, R, t


def resample_curve(P, n):
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1]), np.interp(u, s, P[:, 2])]


bys = {}
for name, p in TARA.items():
    mer = MAP.get(name.split()[0], name.split()[0])
    if mer in TH['meridians']:
        num = int(name.split()[1])
        bys.setdefault(mer, []).append((num, p))

rows = []
for mer, lst in bys.items():
    lst.sort(key=lambda x: x[0])
    X = np.array([p for _, p in lst], float)
    n = len(X)
    Y = resample_curve(TH['meridians'][mer]['right'], n)   # 同分位
    s, R, t = umeyama(X, Y)
    Xt = (s * (R @ X.T)).T + t
    d = np.linalg.norm(Xt - Y, axis=1) * CM
    rms = float(np.sqrt((d ** 2).mean()))
    rows.append((mer, n, rms, float(np.median(d)), s))
    print(f'{mer:3s} n={n:2d}  RMS {rms:5.1f} cm  中位 {np.median(d):4.1f} cm  scale {s:.3f}')

allr = np.array([r[2] for r in rows])
print('\n全经 RMS 中位 %.1f cm（%s 最好 %.1f，%s 最差 %.1f）' % (
    np.median(allr), min(rows, key=lambda r: r[2])[0], min(r[2] for r in rows),
    max(rows, key=lambda r: r[2])[0], max(r[2] for r in rows)))
json.dump([{'mer': r[0], 'n': r[1], 'rms_cm': round(r[2], 1), 'median_cm': round(r[3], 1), 'scale': round(r[4], 3)} for r in rows],
          open(ROOT / 'data' / 'zc015' / 'tara_registration.json', 'w'), ensure_ascii=False, indent=1)
