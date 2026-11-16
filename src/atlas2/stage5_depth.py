#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5：法向深度优化（mm 尺度）——w ≠ 常数下的最优埋深。

体素网格 8 mm 分辨不出 2–8 mm 的深筋膜，所以把「埋深」从体积路由里分出来：
对 v18 每条通道的每个点，沿体表内法向偏移 d∈[0,0.8] cm，
单位长度代价 c(d) = (1 - d·κ_n) / w(d)   （κ_n = 沿路径的法曲率，凹处为负）
w(d)：0–2 mm 皮/脂 0.75 ｜ 2–8 mm 深筋膜 1.80 ｜ 8–25 mm 肌腹 0.60 ｜ 更深 0.90
骨：若落在骨 capsule 内（+4 mm 余量）代价 ×6。
取 argmin 得每个点的最优埋深，平滑后沿法向偏移；再算界面定理的 λ = L(d*)/L(0)。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_obj, vertex_normals, resample
from stage2_paths import read_joints

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'atlas2'
D12 = json.load(open(OUT / 'paths12.json'))
CM = 0.098  # 1 cm 对应的模型单位（1 单位 ≈ 10.2 cm）


def w_of_depth(depth_cm):
    w = np.full_like(depth_cm, 0.90, dtype=float)
    w[depth_cm <= 2.5] = 0.60
    w[depth_cm <= 0.8] = 1.80      # 深筋膜
    w[depth_cm <= 0.2] = 0.75      # 皮/皮下
    return w


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
        (J['l-upper-leg'], J['r-upper-leg'], 0.10),
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
    d = np.minimum(d, np.linalg.norm(P - c, axis=1) - 0.52)
    return d


def main():
    V, F = load_obj()
    J = read_joints()
    N = vertex_normals(V, F)
    tree = cKDTree(V)
    segs = bone_segments(J)
    depths = np.arange(0, 0.81, 0.05)  # cm
    out = {'meridians': {}, 'depths_cm': depths.tolist(),
           'w_of_depth': dict(zip([round(d, 2) for d in depths], w_of_depth(depths).round(3).tolist()))}
    for code in D12['order']:
        P = np.array(D12['meridians'][code]['right'])
        n = N[tree.query(P)[1]].copy()
        t = np.gradient(P, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        dn = np.gradient(n, axis=0) / CM  # per cm
        kn = -(dn * t).sum(1)             # 法曲率（1/cm），凹处为负
        kn = np.convolve(np.pad(kn, 20, mode='edge'), np.ones(41) / 41, 'valid')
        kn = np.clip(kn, -1.0, 1.0)
        # 每个深度候选的代价
        C = np.zeros((len(P), len(depths)))
        for j, d in enumerate(depths):
            Xd = P - n * (d * CM)
            bd = bone_dist(Xd, segs) / CM  # cm
            c = (1 - d * kn) / w_of_depth(np.full(len(P), d))
            # 筋膜带内取带中心（弱正则，避免 argmin 在带两端跳）
            c = c + 0.06 * ((d - 0.50) / 0.30) ** 2
            c = np.where(bd < 0.4, c * 6.0, c)
            C[:, j] = c
        # 平滑代价后再取 argmin（避免逐点跳变）
        Cs = np.vstack([np.convolve(np.pad(C[:, j], 10, mode='edge'), np.ones(21) / 21, 'valid')
                        for j in range(len(depths))]).T
        # 动态规划：Σ c_i(d_i) + λ Σ (d_i - d_{i+1})²
        lam_dp = 2.5
        D = len(depths)
        acc = Cs[0].copy()
        back = np.zeros((len(P), D), int)
        for i in range(1, len(P)):
            trans = lam_dp * (depths[:, None] - depths[None, :]) ** 2
            M = acc[:, None] + trans
            back[i] = np.argmin(M, axis=0)
            acc = Cs[i] + M[back[i], np.arange(D)]
        j = int(np.argmin(acc))
        path = [j]
        for i in range(len(P) - 1, 0, -1):
            j = back[i, j]
            path.append(j)
        jstar = np.array(path[::-1])
        dstar = depths[jstar]
        dsm = np.convolve(np.pad(dstar, 6, mode='edge'), np.ones(13) / 13, 'valid')
        Q = P - n * (dsm[:, None] * CM)
        L0 = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        L1 = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
        lam = L1 / L0
        w_i = 1.80
        w_bulk = 0.75  # max(表皮 0.75, 肌 0.60)
        out['meridians'][code] = dict(
            depth_mean_cm=float(dsm.mean()), depth_p90_cm=float(np.percentile(dsm, 90)),
            depth_min_cm=float(dsm.min()), depth_max_cm=float(dsm.max()),
            fascia_frac=float(((dsm >= 0.2) & (dsm <= 0.8)).mean()),
            surface_frac=float((dsm <= 0.2).mean()),
            length_surface=L0, length_deep=L1, lam=lam,
            w_i=w_i, w_bulk=w_bulk, ratio=w_i / (lam * w_bulk),
            theorem_pass=bool(w_i > lam * w_bulk),
            bone_clear_cm=float(bone_dist(Q[int(0.08*len(Q)):int(0.92*len(Q))], segs).min() / CM),
            pts=[[round(float(x), 4) for x in p] for p in Q],
            depth_cm=[round(float(d), 2) for d in dsm])
        r = out['meridians'][code]
        print(f"{code}: depth mean {r['depth_mean_cm']:.2f} cm "
              f"(min {r['depth_min_cm']:.2f} max {r['depth_max_cm']:.2f}) "
              f"fascia {r['fascia_frac']*100:4.1f}% surface {r['surface_frac']*100:4.1f}% "
              f"λ {lam:.3f} ratio {r['ratio']:.2f} {'PASS' if r['theorem_pass'] else 'FAIL'} "
              f"bone {r['bone_clear_cm']:.2f} cm")
    allr = list(out['meridians'].values())
    out['summary'] = dict(depth_mean=float(np.mean([r['depth_mean_cm'] for r in allr])),
                          fascia_mean=float(np.mean([r['fascia_frac'] for r in allr])),
                          theorem_pass=int(sum(r['theorem_pass'] for r in allr)),
                          n=len(allr),
                          lam_range=[float(min(r['lam'] for r in allr)), float(max(r['lam'] for r in allr))])
    json.dump(out, open(OUT / 'depth_paths.json', 'w'), ensure_ascii=False)
    print('summary', out['summary'])
    print('saved', OUT / 'depth_paths.json')


if __name__ == '__main__':
    main()
