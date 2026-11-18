#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc022：用肺/心表面（胸膜/心包代理）做深部三条过渡的界面最短路。

BodyParts3D 骨+器官做 bbox 相似变换（同 register_anatomy.py）→
体素化（stage4）→ 器官表面距离场 → 表面高权 + 骨低压 → via 到心/肺表面 → Dijkstra。
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "atlas2"))
import stage4_tissue as S4
from core import load_obj
from stage2_paths import read_joints

OUT = ROOT / "data" / "zc022"
OUT.mkdir(parents=True, exist_ok=True)
D12 = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
PAIRS = [("SP", "HT"), ("KI", "PC"), ("LV", "LU")]
VIA_TARGET = {
    "SP-HT": ("heart", [0.30, 4.05, 0.85]),
    "KI-PC": ("heart", [0.25, 3.70, 1.05]),
    "LV-LU": ("left lung", [0.90, 3.85, 1.15]),
}
BONES = ['skull', 'mandible', 'sternum', 'right clavicle', 'left clavicle', 'right scapula',
         'left scapula', 'right humerus', 'left humerus', 'right radius', 'left radius',
         'right hip bone', 'left hip bone', 'right femur', 'left femur', 'right tibia',
         'left tibia', 'right fibula', 'left fibula', 'right patella', 'left patella',
         'rib cage', 'thoracic vertebral column']
ORGANS = ['heart', 'right lung', 'left lung', 'liver', 'stomach', 'right kidney',
          'left kidney', 'small intestine', 'large intestine', 'brain']


def raw_obj(path):
    V = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
    return np.array(V)


def load_map():
    rows = []
    for line in open(ROOT / "data" / "partof_element_parts.txt", encoding="utf-8", errors="ignore"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].startswith("FJ"):
            rows.append(p)
    by = collections.defaultdict(set)
    for cid, n, fid in rows:
        by[n].add(fid)
    return by


def points_for(by, names):
    P = []
    for n in names:
        for fid in sorted(by.get(n, [])):
            try:
                P.append(raw_obj(ROOT / "data" / "bodyparts3d" / "partof_BP3D_4.0_obj_99" / f"{fid}.obj"))
            except Exception:
                pass
    return np.vstack(P) if P else np.zeros((0, 3))


def main():
    by = load_map()
    V = load_obj(ROOT / "data" / "model" / "human_neutral_makehuman.obj")[0]
    B = points_for(by, BONES)
    O = points_for(by, ORGANS)
    A = np.vstack([B, O])
    a = (V[:, 1].max() - V[:, 1].min()) / (A[:, 2].max() - A[:, 2].min())
    tx = -a * (A[:, 0].min() + A[:, 0].max()) / 2
    ty = (V[:, 1].min() + V[:, 1].max()) / 2 - a * (A[:, 2].min() + A[:, 2].max()) / 2
    tz = (V[:, 2].min() + V[:, 2].max()) / 2 - a * (A[:, 1].min() + A[:, 1].max()) / 2

    def tr(P):
        return np.c_[a * P[:, 0] + tx, a * P[:, 2] + ty, a * P[:, 1] + tz]

    heart = tr(points_for(by, ["heart"]))
    llung = tr(points_for(by, ["left lung"]))
    rlung = tr(points_for(by, ["right lung"]))
    rng = np.random.default_rng(0)

    def sub(P, n):
        return P[rng.choice(len(P), min(n, len(P)), replace=False)] if len(P) else P

    heart = sub(heart, 12000); llung = sub(llung, 12000); rlung = sub(rlung, 12000)
    print("organs", len(heart), len(llung), len(rlung), "transform a", round(a, 6))

    J = read_joints()
    mask, xs, ys, zs, (lo, res) = S4.voxelize(V, load_obj(ROOT / "data" / "model" / "human_neutral_makehuman.obj")[1])
    idx = -np.ones(mask.shape, int)
    ii = np.argwhere(mask)
    idx[ii[:, 0], ii[:, 1], ii[:, 2]] = np.arange(len(ii))
    pos = np.c_[xs[ii[:, 0]], ys[ii[:, 1]], zs[ii[:, 2]]]
    depth = S4.distance_transform_edt(mask, sampling=res)[ii[:, 0], ii[:, 1], ii[:, 2]]
    dbone = S4.bone_distance(pos, J)
    w = S4.weight(depth, dbone)
    w = w.copy()
    w[dbone < 0.08] = 0.005
    w[(dbone >= 0.08) & (dbone < 0.20)] = 0.03

    # 器官表面距离场：肺表面=胸膜代理，心表面=心包代理
    dh = cKDTree(heart).query(pos)[0]
    dl = cKDTree(llung).query(pos)[0]
    dr = cKDTree(rlung).query(pos)[0]
    d_iface = np.minimum(np.minimum(dh, dl), dr)
    w[d_iface < 0.12] = 3.0
    w[(d_iface >= 0.12) & (d_iface < 0.22)] = 1.6

    # 26 邻域图
    E, L = [], []
    nbr = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) > (0, 0, 0)]
    for dx, dy, dz in nbr:
        aa = ii
        bb = aa + np.array([dx, dy, dz])
        ok = ((bb[:, 0] >= 0) & (bb[:, 0] < mask.shape[0]) & (bb[:, 1] >= 0) &
              (bb[:, 1] < mask.shape[1]) & (bb[:, 2] >= 0) & (bb[:, 2] < mask.shape[2]))
        aa = aa[ok]; bb = bb[ok]
        ok2 = mask[bb[:, 0], bb[:, 1], bb[:, 2]]
        aa = aa[ok2]; bb = bb[ok2]
        E.append(np.c_[idx[aa[:, 0], aa[:, 1], aa[:, 2]], idx[bb[:, 0], bb[:, 1], bb[:, 2]]])
        L.append(np.linalg.norm(bb - aa, axis=1) * res)
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
    for aa, bb in PAIRS:
        key = f"{aa}-{bb}"
        pa = np.array(D12["meridians"][aa]["right"][-1], float)
        pb = np.array(D12["meridians"][bb]["right"][0], float)
        organ, target = VIA_TARGET[key]
        surf = {"heart": heart, "left lung": llung, "right lung": rlung}[organ]
        via = surf[int(np.argmin(np.linalg.norm(surf - np.array(target), axis=1)))]
        ia, iv, ib = nearest(pa), nearest(via), nearest(pb)
        p1 = path_between(ia, iv); p2 = path_between(iv, ib)
        if p1 is None or p2 is None:
            print("no path", key); continue
        path = p1 + p2[1:]
        Q = pos[path]; dep = depth[path]; dbo = dbone[path]
        cur = next(t for t in D12["transitions"] if t["a"] == aa and t["b"] == bb)
        out[key] = {
            "n": len(path),
            "length": float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()),
            "straight_length": float(cur["length"]),
            "depth_mean": float(dep.mean()),
            "depth_p90": float(np.percentile(dep, 90)),
            "interface_frac": float((d_iface[path] < 0.12).mean()),
            "heart_frac": float((dh[path] < 0.12).mean()),
            "lung_frac": float(((dl[path] < 0.12) | (dr[path] < 0.12)).mean()),
            "bone_min": float(dbo.min()),
            "via": [float(x) for x in via],
            "organ": organ,
            "pts": Q.round(4).tolist(),
            "depth": dep.round(3).tolist(),
        }
        print(f"{key} organ {organ:9s} n {len(path):4d} len {out[key]['length']:6.2f} "
              f"straight {cur['length']:5.2f} depth {dep.mean():.3f} "
              f"iface {out[key]['interface_frac']*100:5.1f}% heart {out[key]['heart_frac']*100:4.1f}% "
              f"lung {out[key]['lung_frac']*100:4.1f}% bone_min {dbo.min():.3f}")

    out["meta"] = {
        "res": float(res), "n_vox": int(len(pos)),
        "organ_proxy": {"pleura": "left/right lung surface", "pericardium": "heart surface"},
        "weights": {"fascia": 1.8, "skin": 0.75, "muscle": 0.6, "deep": 0.85,
                    "interface": 3.0, "interface_near": 1.6,
                    "bone": 0.005, "bone_near": 0.03},
        "note": "organ-surface least-cost proxy; lung=pleura proxy, heart=pericardium proxy; not a measured pleural path",
    }
    with open(OUT / "chest_interface_paths.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("saved", OUT / "chest_interface_paths.json")


if __name__ == "__main__":
    main()
