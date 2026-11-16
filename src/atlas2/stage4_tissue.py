#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4：组织权重版（w ≠ 常数）——把 v18 的 12 条体表通道放进体积里重算。

w 的定义（结构选择 I，数值见 §1）：
  · 深筋膜壳层（皮下 1.2–8.5 mm）  w = 1.8   ← 界面（Zc012 §10）
  · 皮下/皮肤（<1.2 mm）           w = 0.75
  · 肌腹（8.5 mm 以下）           w = 0.60
  · 深部（> 25 mm 且离骨远）       w = 0.85
  · 骨 capsule（<8 mm）           w = 0.12；8–20 mm 过渡到 0.30
通道代价 = Σ 边长 × (1/w_a + 1/w_b)/2 + 0.08 × 离 v18 体表线的距离（弱先验，保住 12 条的身份）
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_obj, vertex_normals
from stage2_paths import read_joints

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'atlas2'
D12 = json.load(open(OUT / 'paths12.json'))
RES = 0.08  # 单位（≈8 mm）


def voxelize(V, F, res=RES):
    lo = V.min(0) - 2 * res
    hi = V.max(0) + 2 * res
    nx, ny, nz = [int(np.ceil((hi[i] - lo[i]) / res)) + 1 for i in range(3)]
    xs = lo[0] + np.arange(nx) * res
    ys = lo[1] + np.arange(ny) * res
    zs = lo[2] + np.arange(nz) * res
    surf = np.zeros((nx, ny, nz), bool)
    bary = []
    for wv in ([1/3, 1/3, 1/3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]):
        bary.append(wv[0] * V[F[:, 0]] + wv[1] * V[F[:, 1]] + wv[2] * V[F[:, 2]])
    edges = []
    for a, b in ((F[:, 0], F[:, 1]), (F[:, 1], F[:, 2]), (F[:, 2], F[:, 0])):
        for t in (0.25, 0.5, 0.75):
            edges.append((1 - t) * V[a] + t * V[b])
    P = np.vstack([V] + bary + edges)
    i = np.clip(np.round((P[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
    j = np.clip(np.round((P[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
    k = np.clip(np.round((P[:, 2] - lo[2]) / res).astype(int), 0, nz - 1)
    surf[i, j, k] = True
    from scipy.ndimage import binary_dilation
    surf = binary_dilation(surf, iterations=1)
    free = ~surf
    lab, n = label(free)
    edge = set(np.unique(np.r_[lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                               lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    edge.discard(0)
    outside = np.isin(lab, list(edge))
    inside = free & ~outside
    inside = binary_fill_holes(inside)
    lab2, n2 = label(inside)
    if n2 > 1:
        sizes = np.bincount(lab2.ravel())
        inside = lab2 == (np.argmax(sizes[1:]) + 1)
    return inside, xs, ys, zs, (lo, res)


def bone_distance(pos, J):
    segs = [
        (J['pelvis'], J['spine-4'], 0.12), (J['spine-4'], J['spine-3'], 0.12),
        (J['spine-3'], J['spine-2'], 0.12), (J['spine-2'], J['spine-1'], 0.12),
        (J['spine-1'], J['neck'], 0.10), (J['neck'], J['head'], 0.10),
        (J['spine-1'], J['l-shoulder'], 0.07), (J['spine-1'], J['r-shoulder'], 0.07),
        (J['l-shoulder'], J['l-elbow'], 0.09), (J['r-shoulder'], J['r-elbow'], 0.09),
        (J['l-elbow'], J['l-hand'], 0.06), (J['r-elbow'], J['r-hand'], 0.06),
        (J['l-upper-leg'], J['l-knee'], 0.11), (J['r-upper-leg'], J['r-knee'], 0.11),
        (J['l-knee'], J['l-ankle'], 0.09), (J['r-knee'], J['r-ankle'], 0.09),
        (J['l-upper-leg'], J['pelvis'], 0.10), (J['r-upper-leg'], J['pelvis'], 0.10),
        (J['l-upper-leg'], J['r-upper-leg'], 0.10),
    ]
    d = np.full(len(pos), 1e9)
    for a, b, r in segs:
        a = np.asarray(a, float); b = np.asarray(b, float)
        ab = b - a
        t = np.clip((pos - a) @ ab / (ab @ ab), 0, 1)
        q = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(pos - q, axis=1) - r)
    # 颅骨球
    c = np.array([0.0, 7.55, 0.45]); d = np.minimum(d, np.linalg.norm(pos - c, axis=1) - 0.52)
    return d


def weight(depth, dbone):
    w = np.where(depth < 0.12, 0.75, 0.60)          # 皮/脂 vs 肌腹
    w = np.where(depth > 2.5, 0.85, w)               # 深部
    fascia = (depth >= 0.12) & (depth <= 0.85)
    w = np.where(fascia, 1.80, w)                    # 深筋膜壳层
    w = np.where(dbone < 0.20, 0.30, w)
    w = np.where(dbone < 0.08, 0.12, w)              # 骨
    return w


def main():
    V, F = load_obj()
    J = read_joints()
    mask, xs, ys, zs, (lo, res) = voxelize(V, F)
    idx = -np.ones(mask.shape, int)
    ii = np.argwhere(mask)
    idx[ii[:, 0], ii[:, 1], ii[:, 2]] = np.arange(len(ii))
    pos = np.c_[xs[ii[:, 0]], ys[ii[:, 1]], zs[ii[:, 2]]]
    depth = distance_transform_edt(mask, sampling=res)[ii[:, 0], ii[:, 1], ii[:, 2]]
    dbone = bone_distance(pos, J)
    w = weight(depth, dbone)
    print('voxels', len(pos), 'res', res, 'depth med', round(float(np.median(depth)), 3),
          'fascia%', round(100 * float(((depth >= 0.12) & (depth <= 0.85)).mean()), 1))

    # ---- 图（26 邻域）
    E, L = [], []
    nbr = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) > (0, 0, 0)]
    for dx, dy, dz in nbr:
        a = ii
        b = a + np.array([dx, dy, dz])
        ok = ((b[:, 0] >= 0) & (b[:, 0] < mask.shape[0]) & (b[:, 1] >= 0) &
              (b[:, 1] < mask.shape[1]) & (b[:, 2] >= 0) & (b[:, 2] < mask.shape[2]))
        a = a[ok]; b = b[ok]
        ok2 = mask[b[:, 0], b[:, 1], b[:, 2]]
        a = a[ok2]; b = b[ok2]
        E.append(np.c_[idx[a[:, 0], a[:, 1], a[:, 2]], idx[b[:, 0], b[:, 1], b[:, 2]]])
        L.append(np.linalg.norm(b - a, axis=1) * res)
    E = np.vstack(E); L = np.concatenate(L)
    print('edges', len(E))

    tree_v = cKDTree(V)
    NRM = vertex_normals(V, F)
    results = {'guided': {}, 'free': {}}
    for code in D12['order']:
        P = np.array(D12['meridians'][code]['right'])
        a = P[0]; b = P[-1]
        ia = int(np.argmin(np.linalg.norm(pos - a, axis=1)))
        ib = int(np.argmin(np.linalg.norm(pos - b, axis=1)))
        tp = cKDTree(P)
        dpath, ipath = tp.query(pos)
        npath = NRM[tree_v.query(P)[1]][ipath]
        vec = pos - P[ipath]
        dn = (vec * npath).sum(1)
        dlat = np.linalg.norm(vec - dn[:, None] * npath, axis=1)
        for mode in ('guided', 'free'):
            if mode == 'guided':
                keep = dlat < 0.45
                pen = 1.10 * dlat
            else:
                keep = np.ones(len(pos), bool)
                pen = np.zeros(len(pos))
            # 端点必须保留
            keep[ia] = True; keep[ib] = True
            m = keep[E[:, 0]] & keep[E[:, 1]]
            Ee = E[m]; Ll = L[m]
            wa = w[Ee[:, 0]]; wb = w[Ee[:, 1]]
            cost = Ll * 0.5 * (1.0 / wa + 1.0 / wb) + 0.5 * (pen[Ee[:, 0]] + pen[Ee[:, 1]]) * Ll
            # 端点不在 keep 里时，索引仍指向全部节点；用 n 节点图 + 大代价禁用其它节点
            G = coo_matrix((np.r_[cost, cost], (np.r_[Ee[:, 0], Ee[:, 1]], np.r_[Ee[:, 1], Ee[:, 0]])),
                           shape=(len(pos), len(pos))).tocsr()
            d, pred = dijkstra(G, indices=ia, return_predecessors=True)
            if not np.isfinite(d[ib]):
                print('  no path', code, mode); continue
            k = ib; path = [k]
            while k != ia and k >= 0:
                k = int(pred[k])
                if k < 0: break
                path.append(k)
            path = path[::-1]
            Q = pos[path]
            dep = depth[path]
            # 与体表 v18 的比较
            dv, _ = cKDTree(P).query(Q)
            results[mode][code] = dict(
                n=len(Q), length=float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()),
                depth_mean=float(dep.mean()), depth_p90=float(np.percentile(dep, 90)),
                depth_max=float(dep.max()),
                fascia_frac=float(((dep >= 0.12) & (dep <= 0.85)).mean()),
                bone_min=float(dbone[path[int(0.08*len(path)):int(0.92*len(path))]].min()),
                bone_mean=float(dbone[path[int(0.08*len(path)):int(0.92*len(path))]].mean()),
                dev_to_v18_mean=float(dv.mean()), dev_to_v18_max=float(dv.max()),
                lat_mean=float(dlat[path].mean()), lat_p90=float(np.percentile(dlat[path], 90)),
                lam=float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum() /
                          max(np.linalg.norm(np.diff(P, axis=0), axis=1).sum(), 1e-9)),
                pts=[[round(float(x), 3) for x in p] for p in Q[::2]],
                depth=dep.round(3).tolist())
            print(f"{code} {mode:6s} len {results[mode][code]['length']:6.2f} "
                  f"depth mean {dep.mean():.3f} p90 {np.percentile(dep,90):.3f} "
                  f"fascia {results[mode][code]['fascia_frac']*100:4.1f}% "
                  f"lat {dlat[path].mean():.3f} λ {results[mode][code]['lam']:.3f} "
                  f"bone_min {results[mode][code]['bone_min']:.3f}")

    # 自由路由的塌缩程度
    import itertools
    results['meridians'] = results['guided']
    fk = list(results['free'])
    col = {}
    for a, b in itertools.combinations(fk, 2):
        A = np.array(results['free'][a]['pts']); B = np.array(results['free'][b]['pts'])
        D = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        col[f'{a}-{b}'] = dict(min=float(D.min()), mean=float(D.mean()))
    near = [k for k, v in col.items() if v['min'] < 0.05]
    print('free-routing collapsed pairs (<5mm):', len(near), near[:8])
    results['free_collapse'] = dict(pairs=near, all=col)
    results['meta'] = dict(res=res, w={'fascia': 1.8, 'skin': 0.75, 'muscle': 0.6,
                                       'deep': 0.85, 'bone': 0.12, 'bone_near': 0.3},
                           n_vox=int(len(pos)), depth_shell=[0.12, 0.85])
    json.dump(results, open(OUT / 'tissue_paths.json', 'w'), ensure_ascii=False)
    print('saved', OUT / 'tissue_paths.json')


if __name__ == '__main__':
    main()
