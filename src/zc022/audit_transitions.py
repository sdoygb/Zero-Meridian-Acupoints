#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc022：流注过渡的数据审计（12 时辰边、已有/缺失、长度、端点距离）。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas2"))
from core import load_obj, build_graph, route

ROOT = Path(__file__).resolve().parents[2]
D = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
TR = {(t["a"], t["b"]): t for t in D["transitions"]}
MER = D["meridians"]

# 12 时辰顺序（子胆 … 亥三焦）；边 = 上一经终点→下一经起点
SHI = [("子", "GB"), ("丑", "LV"), ("寅", "LU"), ("卯", "LI"),
       ("辰", "ST"), ("巳", "SP"), ("午", "HT"), ("未", "SI"),
       ("申", "BL"), ("酉", "KI"), ("戌", "PC"), ("亥", "TE")]

rows = []
for i in range(len(SHI)):
    a = SHI[i][1]
    b = SHI[(i + 1) % len(SHI)][1]
    name_a, name_b = SHI[i][0], SHI[(i + 1) % len(SHI)][0]
    pa = np.array(MER[a]["right"][-1], float)
    pb = np.array(MER[b]["right"][0], float)
    gap = float(np.linalg.norm(pa - pb))
    t = TR.get((a, b))
    rows.append({
        "from_shichen": name_a, "to_shichen": name_b,
        "a": a, "b": b,
        "present": t is not None,
        "kind": t["kind"] if t else "missing",
        "length": float(t["length"]) if t else None,
        "endpoint_gap": gap,
        "n_points": len(t["right"]) if t else 0,
    })

present = [r for r in rows if r["present"]]
missing = [r for r in rows if not r["present"]]
# 缺失的体表边：用同一体表网格补算最短路（只补 surface；深部过渡仍是示意）
computed = []
if missing:
    V, F = load_obj()
    E, L, G = build_graph(V, F)
    tree = cKDTree(V)
    for r in missing:
        pa = np.array(MER[r["a"]]["right"][-1], float)
        pb = np.array(MER[r["b"]]["right"][0], float)
        ia = int(tree.query(pa)[1]); ib = int(tree.query(pb)[1])
        P, tot = route(G, V, [ia, ib])
        if P is None:
            r["computed"] = False
            continue
        pts = V[P]
        r["computed"] = True
        r["kind"] = "surface"
        r["length"] = float(tot)
        r["n_points"] = len(P)
        r["right"] = pts.round(4).tolist()
        r["left"] = (pts * [-1, 1, 1]).round(4).tolist()
        computed.append(r)

out = {
    "pairs": rows,
    "n_pairs": len(rows),
    "n_present": len(present),
    "n_missing": len(missing),
    "n_surface": sum(1 for r in present if r["kind"] == "surface"),
    "n_deep": sum(1 for r in present if r["kind"] == "deep"),
    "n_computed": len(computed),
    "n_surface_all": sum(1 for r in rows if r.get("kind") == "surface"),
    "n_deep_all": sum(1 for r in rows if r.get("kind") == "deep"),
    "missing": missing,
    "unit_cm": D["meta"]["unit_cm"],
}
with open(ROOT / "data" / "zc022" / "transitions_audit.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"pairs {len(rows)} present {len(present)} missing {len(missing)} "
      f"surface {out['n_surface']} deep {out['n_deep']}")
for r in rows:
    mark = "OK" if r["present"] else ("COMP" if r.get("computed") else "MISS")
    ln = f"{r['length']:.2f}" if r.get("length") is not None else "  - "
    print(f"{mark} {r['from_shichen']}->{r['to_shichen']} {r['a']:>3}->{r['b']:<3} "
          f"{r['kind']:<7} len {ln} gap {r['endpoint_gap']:.2f}")
print("saved", ROOT / "data" / "zc022" / "transitions_audit.json")
