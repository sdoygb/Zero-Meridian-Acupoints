#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc022：把深部三条过渡约束在肺/心表面（胸膜/心包代理）上的最短路。

与 chest_interface.py 的体素高权不同：这里直接在器官表面点云上建 kNN 图，
路径只能走器官表面；两端用短连接器接到表面。结果 100% 在界面上（连接器除外）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "atlas2"))
sys.path.insert(0, str(ROOT / "src" / "zc022"))
import chest_interface as CI
from core import load_obj

OUT = ROOT / "data" / "zc022"
D12 = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
PAIRS = [("SP", "HT"), ("KI", "PC"), ("LV", "LU")]
VIA_TARGET = {
    "SP-HT": ("heart", [0.30, 4.05, 0.85]),
    "KI-PC": ("heart", [0.25, 3.70, 1.05]),
    "LV-LU": ("left lung", [0.90, 3.85, 1.15]),
}


def main():
    by = CI.load_map()
    V, F = load_obj(ROOT / "data" / "model" / "human_neutral_makehuman.obj")
    B = CI.points_for(by, CI.BONES)
    O = CI.points_for(by, CI.ORGANS)
    A = np.vstack([B, O])
    a = (V[:, 1].max() - V[:, 1].min()) / (A[:, 2].max() - A[:, 2].min())
    tx = -a * (A[:, 0].min() + A[:, 0].max()) / 2
    ty = (V[:, 1].min() + V[:, 1].max()) / 2 - a * (A[:, 2].min() + A[:, 2].max()) / 2
    tz = (V[:, 2].min() + V[:, 2].max()) / 2 - a * (A[:, 1].min() + A[:, 1].max()) / 2

    def tr(P):
        return np.c_[a * P[:, 0] + tx, a * P[:, 2] + ty, a * P[:, 1] + tz]

    heart = tr(CI.points_for(by, ["heart"]))
    llung = tr(CI.points_for(by, ["left lung"]))
    rlung = tr(CI.points_for(by, ["right lung"]))
    rng = np.random.default_rng(0)

    def sub(P, n):
        return P[rng.choice(len(P), min(n, len(P)), replace=False)]

    heart = sub(heart, 4000); llung = sub(llung, 4000); rlung = sub(rlung, 4000)
    faces = np.hstack([np.full((len(F), 1), 3), F]).astype(np.int64).ravel()
    body = pv.PolyData(V, faces)

    def inside_mask(P):
        pd = pv.PolyData(np.asarray(P, float))
        out = pd.select_interior_points(surface=body, check_surface=False)
        return out.point_data["selected_points"].astype(bool)

    def fix_inside(P, name):
        center = P.mean(0)
        Q = P.copy()
        n0 = int((~inside_mask(Q)).sum())
        for _ in range(14):
            out = ~inside_mask(Q)
            if not out.any():
                break
            Q[out] = 0.5 * (Q[out] + center)
        n1 = int((~inside_mask(Q)).sum())
        print(f"  {name}: outside {n0} -> {n1}")
        return Q

    heart = fix_inside(heart, "heart")
    llung = fix_inside(llung, "left lung")
    rlung = fix_inside(rlung, "right lung")
    P = np.vstack([heart, llung, rlung])
    label = np.r_[np.zeros(len(heart), int), np.ones(len(llung), int), 2 * np.ones(len(rlung), int)]
    print("surface points", len(P), "heart/lungL/lungR", len(heart), len(llung), len(rlung))

    # kNN 表面图
    tree = cKDTree(P)
    k = 14
    dists, idxs = tree.query(P, k=k)
    edges = set()
    for i in range(len(P)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if d <= 0:
                continue
            u, v = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((u, v, float(d)))
    # 保证连通：不同分量之间连最近点
    E = np.array([[u, v] for u, v, _ in edges])
    G0 = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(P), len(P)))
    ncomp, comp = connected_components(G0, directed=False)
    if ncomp > 1:
        for c in range(1, ncomp):
            i = np.where(comp == c)[0][0]
            j = np.where(comp == 0)[0][0]
            d = float(np.linalg.norm(P[i] - P[j]))
            edges.add((min(i, j), max(i, j), d))
        print("connected components", ncomp, "-> bridged")
    edges = sorted(edges)

    skin = cKDTree(V)

    def build_and_route(pa, pb, via):
        # 端点 + via 作为额外节点，各接最近 5 个表面点
        n = len(P)
        nodes = np.vstack([P, pa[None, :], pb[None, :], via[None, :]])
        ee = list(edges)
        for end_idx, end_pt in [(n, pa), (n + 1, pb), (n + 2, via)]:
            dd, jj = tree.query(end_pt, k=5)
            for d, j in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                ee.append((end_idx, int(j), float(d) / 0.35))
        ee = sorted(set(ee))
        E2 = np.array([[u, v] for u, v, _ in ee])
        W = np.array([w for _, _, w in ee])
        G = coo_matrix((np.r_[W, W], (np.r_[E2[:, 0], E2[:, 1]], np.r_[E2[:, 1], E2[:, 0]])),
                       shape=(n + 3, n + 3)).tocsr()

        def one(src, dst):
            d, pred = dijkstra(G, indices=src, return_predecessors=True)
            if not np.isfinite(d[dst]):
                return None
            kk = dst; path = [kk]
            while kk != src and kk >= 0:
                kk = int(pred[kk])
                if kk < 0:
                    return None
                path.append(kk)
            return path[::-1]

        p1 = one(n, n + 2); p2 = one(n + 2, n + 1)
        if p1 is None or p2 is None:
            return None
        return p1 + p2[1:], nodes, G

    out = {}
    for aa, bb in PAIRS:
        key = f"{aa}-{bb}"
        pa = np.array(D12["meridians"][aa]["right"][-1], float)
        pb = np.array(D12["meridians"][bb]["right"][0], float)
        organ, target = VIA_TARGET[key]
        surf = {"heart": heart, "left lung": llung, "right lung": rlung}[organ]
        via = surf[int(np.argmin(np.linalg.norm(surf - np.array(target), axis=1)))]
        rr = build_and_route(pa, pb, via)
        if rr is None:
            print("no path", key); continue
        path, nodes, G = rr
        Q = nodes[path]
        # 分类：表面节点 vs 端点连接器
        surf_mask = np.array([p < len(P) for p in path])
        surface_len = float(np.linalg.norm(np.diff(Q[surf_mask], axis=0), axis=1).sum()) if surf_mask.sum() > 1 else 0.0
        total_len = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
        # 深度：表面点到皮肤的距离
        dep = np.array([skin.query(q)[0] for q in Q])
        lab = np.array([label[p] if p < len(P) else -1 for p in path])
        cur = next(t for t in D12["transitions"] if t["a"] == aa and t["b"] == bb)
        out[key] = {
            "n": len(path),
            "length": total_len,
            "surface_length": surface_len,
            "straight_length": float(cur["length"]),
            "depth_mean": float(dep.mean()),
            "depth_p90": float(np.percentile(dep, 90)),
            "surface_frac": float(surf_mask.mean()),
            "heart_frac": float((lab == 0).mean()),
            "lung_frac": float(((lab == 1) | (lab == 2)).mean()),
            "pts": Q.round(4).tolist(),
            "depth": dep.round(3).tolist(),
            "label": lab.tolist(),
        }
        print(f"{key} n {len(path):4d} len {total_len:6.2f} surface {surface_len:5.2f} "
              f"straight {cur['length']:5.2f} depth {dep.mean():.3f} "
              f"surface_frac {surf_mask.mean()*100:5.1f}% heart {(lab==0).mean()*100:4.1f}% "
              f"lung {(((lab==1)|(lab==2)).mean())*100:4.1f}%")

    out["meta"] = {
        "surface_points": {"heart": len(heart), "left_lung": len(llung), "right_lung": len(rlung)},
        "k": k,
        "connector_weight": 0.35,
        "organ_proxy": {"pleura": "left/right lung surface", "pericardium": "heart surface"},
        "note": "surface-only least-cost proxy; endpoints connected by short connectors; lung=pleura proxy, heart=pericardium proxy",
    }
    with open(OUT / "chest_surface_paths.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("saved", OUT / "chest_surface_paths.json")


if __name__ == "__main__":
    main()
