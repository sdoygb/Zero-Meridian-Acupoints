#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复核：v19 经络路径离皮肤多深、有没有出体。

与 stage5_depth.py 相互独立：
  · 不读 stage5 自报的 depth_cm 作为判据，只在最后做对照；
  · 自写三角面最近点 + 两套内外判据（最近面法线符号 / 射线奇偶）；
  · 体表路径 P 取自 paths12.json，深层路径 Q 取自 depth_paths.json。

单位：1 模型单位 ≈ 10.2 cm（README 标定）。
埋深符号：体内为正（cm），体外为负（出体距离，cm）。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'atlas2'
MODEL = ROOT / 'data' / 'model' / 'human_neutral_body.obj'
CM_PER_UNIT = 10.2


# ---------------------------------------------------------------- 网格解析（独立实现）
def load_mesh(path):
    V, F, cur = [], [], None
    for line in open(path, encoding='utf-8', errors='ignore'):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('g '):
            cur = line.split()[1]
        elif line.startswith('f '):
            F.append([int(t.split('/')[0]) - 1 for t in line.split()[1:]])
    V = np.asarray(V, float)
    used = sorted({i for f in F for i in f})
    remap = {o: k for k, o in enumerate(used)}
    V = V[used]
    F = [[remap[i] for i in f] for f in F]
    tri = []
    for f in F:
        if len(f) == 4:
            a, b, c, d = f
            if np.linalg.norm(V[a] - V[c]) <= np.linalg.norm(V[b] - V[d]):
                tri += [[a, b, c], [a, c, d]]
            else:
                tri += [[a, b, d], [b, c, d]]
        else:
            tri.append(f)
    return V, np.asarray(tri, np.int32)


def tri_closest(p, a, b, c):
    """p 到三角形 abc 的最近点（独立实现，重心坐标裁剪）。返回 q。"""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    den = 1.0 / (va + vb + vc)
    v, w = vb * den, vc * den
    return a + ab * v + ac * w


def ray_inside(p, V, F, dirs=(0, 1, 0)):
    """射线奇偶判据：沿 dir 打射线，交点奇=体内。Möller–Trumbore，向量化分块。"""
    d = np.asarray(dirs, float); d /= np.linalg.norm(d)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > 1e-9
    inv = np.zeros_like(det); inv[ok] = 1.0 / det[ok]
    s = p - a
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * (d * q).sum(1)
    t = inv * np.einsum('ij,ij->i', e2, q)
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
    return int(hit.sum()) % 2 == 1


def main():
    V, F = load_mesh(MODEL)
    print(f"mesh: {len(V)} verts, {len(F)} tris")

    # 顶点邻接面
    v2f = [[] for _ in range(len(V))]
    for k, f in enumerate(F):
        for i in f:
            v2f[i].append(k)

    # 面法线（朝外）
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    paths = json.load(open(OUT / 'paths12.json', encoding='utf-8'))
    depth = json.load(open(OUT / 'depth_paths.json', encoding='utf-8'))

    def nearest_mesh(P):
        """返回 dist(单位), q(最近点), face 法线符号 s。"""
        P = np.atleast_2d(np.asarray(P, float))
        # 最近顶点（分块暴力）
        vi = np.empty(len(P), int)
        for st in range(0, len(P), 4096):
            blk = P[st:st + 4096]
            d2 = ((blk[:, None, :] - V[None, :, :]) ** 2).sum(-1)
            vi[st:st + 4096] = d2.argmin(1)
        q = np.zeros_like(P); dist = np.zeros(len(P)); sgn = np.zeros(len(P))
        for i, p in enumerate(P):
            best, bd, bs = None, 1e18, 0.0
            for k in v2f[vi[i]]:
                a, b, c = V[F[k, 0]], V[F[k, 1]], V[F[k, 2]]
                qq = tri_closest(p, a, b, c)
                dd = np.linalg.norm(qq - p)
                if dd < bd:
                    best, bd, bn = qq, dd, fn[k]
                    bs = float((p - qq) @ bn)
            q[i] = best; dist[i] = bd; sgn[i] = bs
        return dist, q, sgn

    def region(y, x):
        if y > 5.6:
            return 'head'
        if y < -0.6:
            return 'leg'
        if 3.5 < y <= 5.6 and abs(x) > 1.2:
            return 'arm'
        return 'trunk'

    report = {'mesh': {'verts': len(V), 'tris': len(F)}, 'meridians': {}, 'regions': {}}
    order = paths['order']

    print(f"\n{'经':4} {'层':4} {'出体%':>6} {'埋深中位':>8} {'埋深均值':>8} {'p90':>7} {'最大':>7}  (cm)")
    for code in order:
        mer = paths['meridians'][code]
        P = np.array(mer['right'])
        dP, _, sP = nearest_mesh(P)
        insideP = sP < 0
        depthP = np.where(insideP, dP, -dP) * CM_PER_UNIT

        Q = np.array(depth['meridians'][code]['pts'])
        dQ, _, sQ = nearest_mesh(Q)
        insideQ = sQ < 0
        depthQ = np.where(insideQ, dQ, -dQ) * CM_PER_UNIT
        reported = np.array(depth['meridians'][code]['depth_cm'])

        # 射线法抽查 1/4 点，核对法线符号判据一致性
        step = max(1, len(P) // 60)
        ray_in = np.array([ray_inside(p, V, F) for p in Q[::step]])
        norm_in = sQ[::step] < 0
        agree = float((ray_in == norm_in).mean())

        def stat(x):
            return dict(out_pct=float((x < 0).mean() * 100),
                        median=float(np.median(x[x >= 0])) if (x >= 0).any() else None,
                        mean=float(x[x >= 0].mean()) if (x >= 0).any() else None,
                        p90=float(np.percentile(x[x >= 0], 90)) if (x >= 0).any() else None,
                        max=float(x.max()))

        report['meridians'][code] = dict(
            name=mer['name'], surface=stat(depthP), deep=stat(depthQ),
            reported_depth_cm=dict(median=float(np.median(reported)),
                                   mean=float(reported.mean()),
                                   min=float(reported.min()), max=float(reported.max())),
            sign_ray_agreement=agree,
            deep_outside_pts=int((depthQ < 0).sum()))
        r = report['meridians'][code]
        print(f"{code:4} {'P体表':4} {r['surface']['out_pct']:6.1f} "
              f"{(r['surface']['median'] or 0):8.2f} {(r['surface']['mean'] or 0):8.2f} "
              f"{(r['surface']['p90'] or 0):7.2f} {r['surface']['max']:7.2f}")
        print(f"{'':4} {'Q深层':4} {r['deep']['out_pct']:6.1f} "
              f"{(r['deep']['median'] or 0):8.2f} {(r['deep']['mean'] or 0):8.2f} "
              f"{(r['deep']['p90'] or 0):7.2f} {r['deep']['max']:7.2f}  "
              f"自报中位 {r['reported_depth_cm']['median']:.2f} 判据一致 {agree:.2f}")

    # 部位分组（深层 Q）
    reg = {k: [] for k in ('head', 'trunk', 'arm', 'leg')}
    for code in order:
        Q = np.array(depth['meridians'][code]['pts'])
        dQ, _, sQ = nearest_mesh(Q)
        depthQ = np.where(sQ < 0, dQ, -dQ) * CM_PER_UNIT
        for p, dd in zip(Q, depthQ):
            reg[region(p[1], p[0])].append(dd)
    print('\n== 深层 Q 按部位（cm）==')
    for k, vals in reg.items():
        vals = np.array(vals)
        inn = vals[vals >= 0]
        report['regions'][k] = dict(n=len(vals), out_pct=float((vals < 0).mean() * 100),
                                    median=float(np.median(inn)) if len(inn) else None,
                                    mean=float(inn.mean()) if len(inn) else None,
                                    p90=float(np.percentile(inn, 90)) if len(inn) else None)
        print(f"  {k:6} n={len(vals):4} 出体 {report['regions'][k]['out_pct']:5.1f}% "
              f"中位 {report['regions'][k]['median']:.2f} 均值 {report['regions'][k]['mean']:.2f} "
              f"p90 {report['regions'][k]['p90']:.2f}")

    # 总体
    allQ = []
    for code in order:
        Q = np.array(depth['meridians'][code]['pts'])
        dQ, _, sQ = nearest_mesh(Q)
        allQ.append(np.where(sQ < 0, dQ, -dQ) * CM_PER_UNIT)
    allQ = np.concatenate(allQ)
    inn = allQ[allQ >= 0]
    report['overall'] = dict(n=len(allQ), out_pct=float((allQ < 0).mean() * 100),
                             median=float(np.median(inn)), mean=float(inn.mean()),
                             p90=float(np.percentile(inn, 90)), max=float(allQ.max()),
                             frac_gt_1cm=float((inn > 1.0).mean() * 100),
                             frac_gt_2cm=float((inn > 2.0).mean() * 100))
    print('\n== 深层 Q 总体 ==')
    print(f"  n={report['overall']['n']} 出体 {report['overall']['out_pct']:.1f}% "
          f"中位 {report['overall']['median']:.2f} 均值 {report['overall']['mean']:.2f} "
          f"p90 {report['overall']['p90']:.2f} | >1cm {report['overall']['frac_gt_1cm']:.1f}% "
          f">2cm {report['overall']['frac_gt_2cm']:.1f}%")

    json.dump(report, open(OUT / 'depth_recheck.json', 'w'), ensure_ascii=False, indent=1)
    print('\nsaved', OUT / 'depth_recheck.json')


if __name__ == '__main__':
    main()
