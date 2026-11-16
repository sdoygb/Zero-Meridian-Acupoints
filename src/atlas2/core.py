#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 经络图谱 v18 · 核心几何库

设计原则（与 Zc011/Zc012 对齐）：
  · 端口：体表上的可作用点（Zc011 C1/C3）
  · 通道：因果允许集合 C 中的加权最短路（Zc012 §10）
  · 特殊点：通道上的极值（Zc011 C2）
本文件只提供**几何与图论**原语，不含任何传统经络坐标。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'data' / 'model' / 'human_neutral_makehuman.obj'


# ---------------------------------------------------------------- 网格
def load_obj(path=MODEL, groups=('body',)):
    """只取指定 group（MakeHuman 的 body；helper-* 是衣服/头发/眼球等附件）。"""
    V, Fq = [], []
    cur = None
    for line in open(path, encoding='utf-8', errors='ignore'):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('g '):
            cur = line.split()[1]
        elif line.startswith('f ') and cur in groups:
            Fq.append([int(t.split('/')[0]) - 1 for t in line.split()[1:]])
    V = np.asarray(V, float)
    used = sorted({i for f in Fq for i in f})
    remap = {o: k for k, o in enumerate(used)}
    V = V[used]
    Fq = [[remap[i] for i in f] for f in Fq]
    F = []
    for f in Fq:
        if len(f) == 4:
            a, b, c, d = f
            if np.linalg.norm(V[a] - V[c]) <= np.linalg.norm(V[b] - V[d]):
                F += [[a, b, c], [a, c, d]]
            else:
                F += [[a, b, d], [b, c, d]]
        else:
            F.append(f)
    return V, np.asarray(F, np.int32)


def build_graph(V, F):
    """返回 边表/长度/邻接矩阵（欧氏边长作代价）。"""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                   shape=(len(V), len(V))).tocsr()
    return E, L, G


def vertex_normals(V, F):
    n = np.zeros_like(V)
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def closest_point_on_mesh(P, V, F, tree=None):
    """顶点级投影（足够 1 cm 精度），返回最近顶点索引。"""
    tree = tree or cKDTree(V)
    return tree.query(np.atleast_2d(P))[1]


# ---------------------------------------------------------------- 测地线
def geodesic(G, src, weights=None):
    if weights is None:
        d, pred = dijkstra(G, indices=src, return_predecessors=True)
    else:
        Gw = G.copy()
        Gw.data = weights
        d, pred = dijkstra(Gw, indices=src, return_predecessors=True)
    return d, pred


def trace(pred, i0, i1):
    out = [i1]
    cur = i1
    guard = 0
    while cur != i0 and cur >= 0 and guard < 10 ** 6:
        cur = int(pred[cur])
        if cur < 0:
            break
        out.append(cur)
        guard += 1
    return out[::-1]


def route(G, V, anchors, weightfun=None):
    """依次最短路穿过锚点；返回点列与总长。锚点索引。"""
    pts, tot = [], 0.0
    for a, b in zip(anchors, anchors[1:]):
        d, pred = geodesic(G, int(a), weightfun)
        if not np.isfinite(d[int(b)]):
            return None, np.inf
        idx = trace(pred, int(a), int(b))
        tot += d[int(b)]
        pts += idx if not pts else idx[1:]
    return np.asarray(pts, int), tot


# ---------------------------------------------------------------- 平滑
def chaikin(P, it=4):
    P = np.asarray(P, float)
    for _ in range(it):
        Q = [P[0]]
        for i in range(len(P) - 1):
            Q.append(0.75 * P[i] + 0.25 * P[i + 1])
            Q.append(0.25 * P[i] + 0.75 * P[i + 1])
        Q.append(P[-1])
        P = np.asarray(Q)
    return P


def resample(P, n=200):
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] <= 0:
        return np.repeat(P[:1], n, 0)
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1]), np.interp(u, s, P[:, 2])]


def smooth_surface(P, V, tree=None, it=40, lam=0.5, n=None):
    """Laplacian 平滑 + 每步投影回网格顶点（贴体）。"""
    tree = tree or cKDTree(V)
    Q = resample(P, n) if n else np.asarray(P, float)
    for _ in range(it):
        R = Q.copy()
        R[1:-1] = (1 - lam) * Q[1:-1] + 0.5 * lam * (Q[:-2] + Q[2:])
        idx = tree.query(R)[1]
        Q = V[idx]
        Q = 0.5 * R + 0.5 * Q
    idx = tree.query(Q)[1]
    return V[idx]


# ---------------------------------------------------------------- 切片
def slice_mesh(V, F, axis, val, eps=1e-6):
    """平面 axis=val 与三角网格交线，返回 [Nx3] 环列表（未排序）。"""
    d = V[:, axis] - val
    segs = []
    for tri in F:
        ds = d[tri]
        if (ds > eps).all() or (ds < -eps).all():
            continue
        pts = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            da, db = d[a], d[b]
            if abs(da) < eps:
                pts.append(V[a])
            elif da * db < 0:
                t = da / (da - db)
                pts.append(V[a] + t * (V[b] - V[a]))
        if len(pts) >= 2:
            segs.append((np.asarray(pts[0]), np.asarray(pts[-1])))
        elif len(pts) == 1:
            pass
    if not segs:
        return []
    # 端点量化建图 → 走环
    key = lambda p: (round(p[0], 4), round(p[1], 4), round(p[2], 4))
    adj = {}
    for a, b in segs:
        ka, kb = key(a), key(b)
        adj.setdefault(ka, []).append((kb, a, b))
        adj.setdefault(kb, []).append((ka, b, a))
    seen, loops = set(), []
    for k0 in adj:
        if k0 in seen:
            continue
        loop = [np.asarray(k0)]
        prev, cur = None, k0
        while True:
            seen.add(cur)
            nxt = None
            for (kb, pa, pb) in adj[cur]:
                if kb != prev and kb not in seen:
                    nxt = (kb, pb if np.allclose(pa, np.asarray(cur)) else pa)
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt[0]
            loop.append(nxt[1])
        if len(loop) > 3:
            loops.append(np.asarray(loop))
    loops.sort(key=len, reverse=True)
    return loops


def loop_stats(loop, axis, center=None, res=0.03):
    """把环栅格化，返回 (质心, 最大内切半径, 内切圆心)。"""
    o = [0, 1, 2]
    o.remove(axis)
    A, B = o
    P = loop[:, o]
    lo = P.min(0) - 0.2
    hi = P.max(0) + 0.2
    nx = max(int((hi[0] - lo[0]) / res), 8)
    ny = max(int((hi[1] - lo[1]) / res), 8)
    img = Image.new('1', (nx, ny), 0)
    dr = ImageDraw.Draw(img)
    pts = [((p[0] - lo[0]) / res, (p[1] - lo[1]) / res) for p in P]
    dr.polygon(pts, fill=1)
    m = np.asarray(img, bool)
    if not m.any():
        return P.mean(0), 0.0, P.mean(0)
    dt = distance_transform_edt(m) * res
    j = np.unravel_index(np.argmax(dt), dt.shape)
    c = lo + np.array([j[0], j[1]]) * res
    r = dt[j]
    return c, r, c


def rot_frame(axis_dir):
    """给定轴向，构造 (u,v) 正交基，u 尽量指向 +Z（前）。"""
    a = axis_dir / np.linalg.norm(axis_dir)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - a * (ref @ a)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def ray_surface(V, F, origin, direction, maxdist=10.0, mode='first'):
    """从体内向 direction 打射线，返回交点（三角法，向量化）。mode: first/last。"""
    d0 = direction / np.linalg.norm(direction)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    d = np.broadcast_to(d0, a.shape)
    origin_pt = np.asarray(origin, float)
    origin = np.broadcast_to(origin_pt, a.shape)
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > 1e-9
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = origin - a
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum('ij,ij->i', d, q)
    t = inv * np.einsum('ij,ij->i', e2, q)
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < maxdist)
    if not hit.any():
        return None, None
    tt = np.where(hit, t, 1e9 if mode == 'first' else -1)
    k = int(np.argmin(tt)) if mode == 'first' else int(np.argmax(tt))
    return np.asarray(origin_pt) + d0 * tt[k], float(tt[k])


def angle_of(p, origin, u, v):
    q = np.asarray(p) - np.asarray(origin)
    return np.arctan2(q @ v, q @ u)


def adj_list(E, n):
    adj = [[] for _ in range(n)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def flow_smooth(V, adj, path, iters=30):
    """离散曲线缩短流：每个内点移到「两邻点中点」最近的网格邻居上（严格贴面）。"""
    P = list(path)
    P = [p for i, p in enumerate(P) if i == 0 or p != P[i - 1]]
    for _ in range(iters):
        if len(P) < 3:
            break
        Q = P[:]
        for i in range(1, len(P) - 1):
            tgt = 0.5 * (V[P[i - 1]] + V[P[i + 1]])
            cand = adj[P[i]] + [P[i]]
            j = min(cand, key=lambda k: np.linalg.norm(V[k] - tgt))
            Q[i] = j
        # 去重
        R = [Q[0]]
        for p in Q[1:]:
            if p != R[-1]:
                R.append(p)
        if len(R) == len(P) and all(a == b for a, b in zip(R, P)):
            break
        P = R
    return np.array(P)


def densify(P, maxstep=0.06):
    """沿折线等距加密（仅用于渲染，点仍在原折线上）。"""
    P = np.asarray(P, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        d = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(d / maxstep)))
        for i in range(1, k + 1):
            out.append(a + (b - a) * i / k)
    return np.asarray(out)


def vertex_faces(F, n):
    v2f = [[] for _ in range(n)]
    for k, f in enumerate(F):
        for v in f:
            v2f[v].append(k)
    return v2f


def _closest_on_tri(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    den = 1.0 / (va + vb + vc)
    v = vb * den
    w = vc * den
    return a + ab * v + ac * w


def closest_point_mesh(P, V, F, tree, v2f):
    """连续投影到三角面（不是顶点）：返回 (Q, dist)。"""
    P = np.atleast_2d(P)
    _, vi = tree.query(P)
    out = np.zeros_like(P)
    dd = np.zeros(len(P))
    for i, (p, v) in enumerate(zip(P, vi)):
        best, bd = None, 1e9
        for k in v2f[v]:
            q = _closest_on_tri(p, V[F[k, 0]], V[F[k, 1]], V[F[k, 2]])
            d = np.linalg.norm(q - p)
            if d < bd:
                best, bd = q, d
        out[i] = best
        dd[i] = bd
    return out, dd


def smooth_project(P, V, F, tree, v2f, iters=40, lam=0.35, maxstep=0.25):
    Q = np.asarray(P, float).copy()
    for _ in range(iters):
        R = Q.copy()
        R[1:-1] = (1 - lam) * Q[1:-1] + 0.5 * lam * (Q[:-2] + Q[2:])
        Qp, dd = closest_point_mesh(R, V, F, tree, v2f)
        ok = dd < maxstep
        Q[ok] = Qp[ok]
    return Q


def relax_separation(curves, V, F, tree, v2f, dmin=0.075, iters=25, step=0.35, smooth_n=8):
    """通道可分辨性约束：把过近的两条通道沿体表推开（C6 粗粒化的反面：12 条必须 12 条）。"""
    C = {k: np.asarray(v, float).copy() for k, v in curves.items()}
    keys = list(C)
    for it in range(iters):
        allp = np.vstack([C[k] for k in keys])
        lab = np.concatenate([[k] * len(C[k]) for k in keys])
        t = cKDTree(allp)
        for k in keys:
            P = C[k]
            if len(P) < 5:
                continue
            dist, idx = t.query(P, k=8)
            for i in range(1, len(P) - 1):
                best_d, best_q = 1e9, None
                for dd, jj in zip(np.atleast_1d(dist[i]), np.atleast_1d(idx[i])):
                    if lab[jj] == k:
                        continue
                    if dd < best_d:
                        best_d, best_q = dd, allp[jj]
                if best_q is None or best_d >= dmin:
                    continue
                n = P[i] - best_q
                nl = np.linalg.norm(n)
                if nl < 1e-6:
                    tang = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
                    n = np.cross(tang, np.array([0, 0, 1.0]))
                    nl = np.linalg.norm(n) + 1e-9
                n /= nl
                move = step * (dmin - best_d) * n
                q, d = closest_point_mesh((P[i] + move)[None], V, F, tree, v2f)
                if d[0] < 0.12:
                    P[i] = q[0]
            # 轻度平滑
            if smooth_n:
                R = P.copy()
                R[1:-1] = 0.75 * P[1:-1] + 0.25 * 0.5 * (P[:-2] + P[2:])
                q, d = closest_point_mesh(R, V, F, tree, v2f)
                ok = d < 0.12
                P[ok] = q[ok]
            C[k] = P
    return C


def spline_smooth(P, n=220, s=None, k=3):
    from scipy.interpolate import splprep, splev
    P = np.asarray(P, float)
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-7:
            keep.append(i)
    P = P[keep]
    if len(P) < 8:
        return P
    if s is None:
        s = len(P) * 0.004
    tck, u = splprep(P.T, s=s, k=min(k, len(P) - 1))
    uu = np.linspace(0, 1, n)
    Q = np.array(splev(uu, tck)).T
    Q[0], Q[-1] = P[0], P[-1]
    return Q
