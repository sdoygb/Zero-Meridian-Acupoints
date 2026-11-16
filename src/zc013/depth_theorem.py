#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc013：经络层深的定理计算。

d*(s) = ST(site) + SAT(site) + t_f/2      （界面深度，由定理 1+2+推论给出）
λ(s)  = 1 - κ(s) d*                        （偏移长度比）
判据  = w_i / (λ max(w1,w2))               （界面定理，Zc012 §10）
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'atlas2'))
from core import load_obj, vertex_normals
from stage2_paths import Frame, read_joints

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'zc013'
OUT.mkdir(parents=True, exist_ok=True)
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
DATA = json.load(open(OUT / 'layers.json'))
CM = 0.098  # 1 cm in model units (1 unit ≈ 10.2 cm)


def main():
    V, F = load_obj()
    N = vertex_normals(V, F)
    tree = cKDTree(V)
    J = read_joints()
    fr = Frame(V, F, J)
    st = DATA['skin_st_mm']; sat = DATA['sat']; tf = DATA['fascia']['t_mm']; W = DATA['w']
    # 区域因子按 Lund–Browder 面积权重归一，使面积加权均值 = 实测全身均值
    area = {'head_neck': 0.09, 'trunk_front': 0.13, 'trunk_back': 0.18,
            'upper_limb': 0.19, 'lower_limb': 0.40}
    fac0 = {'head_neck': 0.55, 'trunk_front': 1.45, 'trunk_back': 1.45,
            'upper_limb': 0.78, 'lower_limb': 0.82}
    norm = sum(area[k] * fac0[k] for k in area)
    fac = {k: v / norm for k, v in fac0.items()}
    sat_mean = sat['whole_body_mean_mm']

    arm_chain = np.array([J['l-shoulder'], J['l-elbow'], J['l-hand']])
    leg_chain = np.array([J['l-upper-leg'], J['l-knee'], J['l-ankle']])
    spine = np.array([J['pelvis'], J['spine-2'], J['neck']])

    def seg_d(p, a, b):
        ab = b - a
        t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab)), t

    def chain_d(p, chain):
        best = (1e9, 0, 0)
        for k in range(len(chain) - 1):
            d, t = seg_d(p, chain[k], chain[k + 1])
            if d < best[0]:
                best = (d, (k + t) / (len(chain) - 1), k)
        return best

    def site_of(p):
        y, z = p[1], p[2]
        da, ta, _ = chain_d(p, arm_chain)
        dl, tl, _ = chain_d(p, leg_chain)
        ds, ts, _ = chain_d(p, spine)
        # 头/颈优先（y 高）
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

    def st_for(site, p):
        front = p[2] > (fr.zc(p[1]) if -0.4 < p[1] < 8.4 else 0)
        m = {
            'head': 'forehead' if front else 'back',
            'neck': 'chest' if front else 'back',
            'chest': 'chest', 'back': 'back', 'abdomen': 'abdomen',
            'upper_arm': 'upper_arm_flex' if front else 'upper_arm_ext',
            'forearm': 'forearm_flex' if front else 'forearm_ext',
            'thigh': 'flex_thigh' if front else 'outer_thigh',
            'lower_leg': 'lateral_leg' if p[0] > 0 else 'medial_leg',
            'foot': 'lateral_leg',
        }[site]
        return st[m], m

    def sat_for(site, p):
        if site in ('head', 'neck'):
            k = 'head_neck'
        elif site in ('chest', 'abdomen'):
            k = 'trunk_front'
        elif site == 'back':
            k = 'trunk_back'
        elif site in ('upper_arm', 'forearm'):
            k = 'upper_limb'
        else:
            k = 'lower_limb'
        return sat_mean * fac[k], k

    out = {'sites': {}, 'meridians': {}, 'params': {
        'sat_factors_normalized': fac, 'sat_mean_mm': sat_mean, 'tf_mm': tf,
        'w': W, 'area_weights': area}}
    for code in D12['order']:
        P = np.array(D12['meridians'][code]['right'])
        n = N[tree.query(P)[1]].copy()
        t = np.gradient(P, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        dn = np.gradient(n, axis=0) / CM
        kn = np.clip(-(dn * t).sum(1), -1, 1)
        kn = np.convolve(np.pad(kn, 10, mode='edge'), np.ones(21) / 21, 'valid')
        depths, sites, sts, sats = [], [], [], []
        for p in P:
            s = site_of(p)
            stv, _ = st_for(s, p)
            sav, _ = sat_for(s, p)
            d = stv + sav + tf / 2
            depths.append(d); sites.append(s); sts.append(stv); sats.append(sav)
        depths = np.array(depths)
        lam = 1 - kn * (depths / 10.0)  # cm
        # 脂-筋膜界面：w_i / (λ max(w_fat, w_muscle))
        ratio = W['interface_fat_fascia'] / (lam * max(W['fat'], W['muscle']))
        Q = P - n * (depths[:, None] / 10.0 * CM)
        out['meridians'][code] = dict(
            depth_mean_mm=float(depths.mean()), depth_min_mm=float(depths.min()),
            depth_max_mm=float(depths.max()), depth_p90_mm=float(np.percentile(depths, 90)),
            lam_mean=float(lam.mean()), ratio_mean=float(ratio.mean()),
            ratio_min=float(ratio.min()), pass_frac=float((ratio > 1).mean()),
            sites={s: int(sites.count(s)) for s in set(sites)},
            pts=[[round(float(x), 4) for x in p] for p in Q],
            depth_mm=[round(float(d), 2) for d in depths])
        r = out['meridians'][code]
        print(f"{code}: d* {r['depth_mean_mm']:4.1f} mm [{r['depth_min_mm']:.1f},{r['depth_max_mm']:.1f}] "
              f"λ {r['lam_mean']:.3f} ratio {r['ratio_mean']:.2f} pass {r['pass_frac']*100:.0f}% "
              f"sites {r['sites']}")

    # 部位表
    allpts = [(code, i) for code in D12['order'] for i in range(240)]
    rows = {}
    for code, i in allpts:
        P = np.array(D12['meridians'][code]['right'])[i]
        s = site_of(P); stv, sn = st_for(s, P); sav, rk = sat_for(s, P)
        rows.setdefault(s, []).append((stv, sav, stv + sav + tf / 2))
    out['sites'] = {k: dict(n=len(v), st_mm=float(np.mean([x[0] for x in v])),
                            sat_mm=float(np.mean([x[1] for x in v])),
                            depth_mm=float(np.mean([x[2] for x in v]))) for k, v in rows.items()}
    for k, v in sorted(out['sites'].items()):
        print(f"  site {k:10s} n={v['n']:4d} ST {v['st_mm']:.2f} SAT {v['sat_mm']:.2f} d* {v['depth_mm']:.2f} mm")
    # BMI 依赖（SAT 线性于 BMI：3.7 @23.6 → 16.75 @29.86）
    bmi0, sat0 = 23.6, 3.7
    bmi1, sat1 = DATA['sat']['obese_anchor']['bmi'], DATA['sat']['obese_anchor']['sat_mm']
    out['bmi_model'] = dict(bmi0=bmi0, sat0=sat0, bmi1=bmi1, sat1=sat1,
                            slope=(sat1 - sat0) / (bmi1 - bmi0))
    print('BMI slope mm/BMI', round(out['bmi_model']['slope'], 2))
    json.dump(out, open(OUT / 'depth_theorem.json', 'w'), ensure_ascii=False)
    print('saved', OUT / 'depth_theorem.json')


if __name__ == '__main__':
    main()
