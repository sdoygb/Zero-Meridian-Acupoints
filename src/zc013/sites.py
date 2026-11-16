#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc013 部位→层深 d*（可被 atlas 复用）。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj, vertex_normals, ray_surface
from stage2_paths import Frame, read_joints
from scipy.spatial import cKDTree as _T


class LayerModel:
    def __init__(self):
        self.V, self.F = load_obj()
        self.J = read_joints()
        self.fr = Frame(self.V, self.F, self.J)
        self.N = vertex_normals(self.V, self.F)
        self.tree = cKDTree(self.V)
        self.data = json.load(open(ROOT / 'data' / 'zc013' / 'layers.json'))
        self.arm = np.array([self.J['l-shoulder'], self.J['l-elbow'], self.J['l-hand']])
        self.leg = np.array([self.J['l-upper-leg'], self.J['l-knee'], self.J['l-ankle']])
        self.spine = np.array([self.J['pelvis'], self.J['spine-2'], self.J['neck']])
        area = {'head_neck': 0.09, 'trunk_front': 0.13, 'trunk_back': 0.18,
                'upper_limb': 0.19, 'lower_limb': 0.40}
        fac0 = {'head_neck': 0.55, 'trunk_front': 1.45, 'trunk_back': 1.45,
                'upper_limb': 0.78, 'lower_limb': 0.82}
        norm = sum(area[k] * fac0[k] for k in area)
        self.fac = {k: v / norm for k, v in fac0.items()}
        self.sat_mean = self.data['sat']['whole_body_mean_mm']
        self.tf = self.data['fascia']['t_mm']

    @staticmethod
    def _seg_d(p, a, b):
        ab = b - a
        t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab)), t

    def _chain_d(self, p, chain):
        best = (1e9, 0.0)
        for k in range(len(chain) - 1):
            d, t = self._seg_d(p, chain[k], chain[k + 1])
            if d < best[0]:
                best = (d, (k + t) / (len(chain) - 1))
        return best

    def site_of(self, p):
        y, z = p[1], p[2]
        da, ta = self._chain_d(p, self.arm)
        dl, tl = self._chain_d(p, self.leg)
        ds, ts = self._chain_d(p, self.spine)
        if y > 6.25:
            return 'head'
        if y > 5.25 and da > 0.9:
            return 'neck'
        if da < dl * 0.95 and da < ds * 0.95 and abs(p[0]) > 1.0:
            return 'upper_arm' if ta < 0.5 else 'forearm'
        if dl < ds * 0.95 and y < -1.0:
            if y < -7.5:
                return 'foot'
            return 'thigh' if tl < 0.5 else 'lower_leg'
        if y < -7.5:
            return 'foot'
        if y > 5.3:
            return 'neck'
        if y < -1.35:
            return 'lower_leg'
        return 'back' if z < -0.05 else ('chest' if y > 2.5 else 'abdomen')

    def d_star(self, P):
        P = np.atleast_2d(P)
        D = []
        for p in P:
            s = self.site_of(p)
            front = p[2] > (self.fr.zc(p[1]) if -0.4 < p[1] < 8.4 else 0)
            m = {'head': 'forehead' if front else 'back',
                 'neck': 'chest' if front else 'back',
                 'chest': 'chest', 'back': 'back', 'abdomen': 'abdomen',
                 'upper_arm': 'upper_arm_flex' if front else 'upper_arm_ext',
                 'forearm': 'forearm_flex' if front else 'forearm_ext',
                 'thigh': 'flex_thigh' if front else 'outer_thigh',
                 'lower_leg': 'lateral_leg' if p[0] > 0 else 'medial_leg',
                 'foot': 'lateral_leg'}[s]
            if s in ('head', 'neck'):
                k = 'head_neck'
            elif s in ('chest', 'abdomen'):
                k = 'trunk_front'
            elif s == 'back':
                k = 'trunk_back'
            elif s in ('upper_arm', 'forearm'):
                k = 'upper_limb'
            else:
                k = 'lower_limb'
            D.append(self.data['skin_st_mm'][m] + self.sat_mean * self.fac[k] + self.tf / 2)
        return np.array(D)

    def _axis_dir(self, p):
        """到最近骨段（脊柱/臂/腿骨线）的方向与距离——稳健的“向内”方向。"""
        best = (1e9, None)
        for C in (self.arm, self.leg, self.spine):
            for k in range(len(C) - 1):
                a, b = C[k], C[k + 1]
                ab = b - a
                t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
                q = a + t * ab
                d = np.linalg.norm(p - q)
                if d < best[0]:
                    best = (d, q)
        d, q = best
        v = q - p
        nv = np.linalg.norm(v)
        return (v / nv if nv > 1e-9 else np.array([0.0, 0.0, -1.0])), float(d)

    def half_thickness(self, P):
        """每点的半厚（模型单位）：优先内法向射线，失败则用骨轴距离。"""
        P = np.asarray(P, float)
        n = self.N[self.tree.query(P)[1]]
        half = np.zeros(len(P))
        for i, (p, nn) in enumerate(zip(P, n)):
            q, t = ray_surface(self.V, self.F, p - nn * 1e-3, -nn, maxdist=2.0, mode='first')
            if q is not None and 0.02 < t < 1.2:
                half[i] = 0.5 * t
            else:
                half[i] = self._axis_dir(p)[1]
        return half

    def offset(self, P, smooth_win=15):
        """沿内法向（失败时沿骨轴方向）偏移 d*，按半厚限深；再沿曲线平滑埋深。"""
        P = np.asarray(P, float)
        n = self.N[self.tree.query(P)[1]].copy()
        half = self.half_thickness(P)
        d_mm = self.d_star(P)
        d = np.minimum(d_mm / 102.0, 0.70 * half)   # 不超过半厚的 70%
        # 射线失败点的方向改用骨轴
        for i in range(len(P)):
            q, t = ray_surface(self.V, self.F, P[i] - n[i] * 1e-3, -n[i], maxdist=2.0, mode='first')
            if q is None or not (0.02 < t < 1.2):
                # 用骨轴方向作为“向内”方向；n 保持“向外”，所以取负
                n[i] = -self._axis_dir(P[i])[0]
        # 沿曲线平滑埋深，消掉部位边界的台阶
        if smooth_win and len(d) > smooth_win:
            k = smooth_win | 1
            d = np.convolve(np.pad(d, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
            d = np.minimum(d, 0.70 * half)
        Q = P - n * d[:, None]
        # 体外点回退：沿“最近体表点”方向拉回，直到进入体内（≤2 mm 余量）
        for i in range(len(Q)):
            for _ in range(8):
                vi = int(self.tree.query(Q[i])[1])
                nv = self.N[vi]
                over = (Q[i] - self.V[vi]) @ nv
                if over <= 0.003:   # ≤0.3 mm 视为在体内
                    break
                Q[i] = Q[i] - nv * (over + 0.002)
        return Q, d_mm, d

if __name__ == '__main__':
    lm = LayerModel()
    for y in (6.9, 4.4, 1.0, -3.0):
        p = np.array([0.5, y, 1.2])
        print(y, lm.site_of(p), round(float(lm.d_star(p)[0]), 2), 'mm')
