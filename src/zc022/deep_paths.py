#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc022：深部三条流注过渡的体积最短路（软组织实施，非实测）。

复用 stage4 的体素化/层权；把深部桥的直线占位换成体素图上的 least-cost 路径：
  · 端点：SP/HT、KI/PC、LV/LU 的模型路径端点；
  · 经由点：原直线中点再向体内压 0.35（保留“胸腔交接”的拓扑）；
  · 代价：Σ L·(1/w_a+1/w_b)/2；骨权重压低（避免穿骨），筋膜权重保持高。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "atlas2"))
import stage4_tissue as S4
from core import load_obj
from stage2_paths import read_joints

OUT = ROOT / "data" / "zc022"
OUT.mkdir(parents=True, exist_ok=True)
D12 = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
PAIRS = [("SP", "HT"), ("KI", "PC"), ("LV", "LU")]
VIA_X = {"SP-HT": 0.30, "KI-PC": 0.25, "LV-LU": -0.30}


def main():
    V, F = load_obj()
    J = read_joints()
    mask, xs, ys, zs, (lo, res) = S4.voxelize(V, F)
    idx = -np.ones(mask.shape, int)
    ii = np.argwhere(mask)
    idx[ii[:, 0], ii[:, 1], ii[:, 2]] = np.arange(len(ii))
    pos = np.c_[xs[ii[:, 0]], ys[ii[:, 1]], zs[ii[:, 2]]]
    depth = S4.distance_transform_edt(mask, sampling=res)[ii[:, 0], ii[:, 1], ii[:, 2]]
    dbone = S4.bone_distance(pos, J)
    w = S4.weight(depth, dbone)
    # 深部桥：骨太“快”会穿骨；压低骨、保留筋膜高权
    w = w.copy()
    w[dbone < 0.08] = 0.005
    w[(dbone >= 0.08) & (dbone < 0.20)] = 0.03

    # 26 邻域图
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
    wa = w[E[:, 0]]; wb = w[E[:, 1]]
    cost = L * 0.5 * (1.0 / wa + 1.0 / wb)
    G = coo_matrix((np.r_[cost, cost], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                   shape=(len(pos), len(pos))).tocsr()

    def nearest(p):
        return int(np.argmin(np.linalg.norm(pos - np.asarray(p, float), axis=1)))

    def path_between(ia, ib):
        d, pred = dijkstra(G, indices=ia, return_predecessors=True)
        if not np.isfinite(d[ib]):
            return None
        k = int(ib); out = [k]
        while k != ia and k >= 0:
            k = int(pred[k])
            if k < 0:
                return None
            out.append(k)
        return out[::-1]

    out = {}
    for a, b in PAIRS:
        pa = np.array(D12["meridians"][a]["right"][-1], float)
        pb = np.array(D12["meridians"][b]["right"][0], float)
        via = 0.5 * (pa + pb)
        via[0] = VIA_X[f"{a}-{b}"]  # 胸侧：心左、心包中、肺右（粗粒化）
        via[2] = 0.10  # 胸腔中心深度
        ia, iv, ib = nearest(pa), nearest(via), nearest(pb)
        print(f"{a}->{b} via depth {depth[iv]:.3f} via bone {dbone[iv]:.3f} via pos {pos[iv].round(2)}")
        p1 = path_between(ia, iv)
        p2 = path_between(iv, ib)
        if p1 is None or p2 is None:
            print("no path", a, b); continue
        path = p1 + p2[1:]
        Q = pos[path]
        dep = depth[path]
        dbo = dbone[path]
        cur = next(t for t in D12["transitions"] if t["a"] == a and t["b"] == b)
        out[f"{a}-{b}"] = {
            "n": len(path),
            "length": float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()),
            "straight_length": float(cur["length"]),
            "straight_gap": float(np.linalg.norm(pa - pb)),
            "depth_mean": float(dep.mean()),
            "depth_p90": float(np.percentile(dep, 90)),
            "depth_max": float(dep.max()),
            "fascia_frac": float(((dep >= 0.12) & (dep <= 0.85)).mean()),
            "bone_min": float(dbo.min()),
            "bone_mean": float(dbo.mean()),
            "depth": dep.round(3).tolist(),
            "pts": Q.round(4).tolist(),
        }
        print(f"{a}->{b} n {len(path):4d} len {out[f'{a}-{b}']['length']:6.2f} "
              f"straight {cur['length']:5.2f} depth mean {dep.mean():.3f} "
              f"p90 {np.percentile(dep,90):.3f} fascia {out[f'{a}-{b}']['fascia_frac']*100:4.1f}% "
              f"bone_min {dbo.min():.3f}")

    out["meta"] = {
        "res": float(res), "n_vox": int(len(pos)),
        "weights": {"fascia": 1.8, "skin": 0.75, "muscle": 0.6, "deep": 0.85,
                    "bone": 0.005, "bone_near": 0.03},
        "note": "voxel least-cost soft-tissue proxy; via = side-specific chest point (x=+0.30/+0.25/-0.30, y=endpoint midpoint, z=0.10); not a measured pleural path",
    }
    with open(OUT / "deep_paths.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("saved", OUT / "deep_paths.json")


if __name__ == "__main__":
    main()
