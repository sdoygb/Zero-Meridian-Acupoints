#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 12/12 流注过渡：8 体表 + BL→KI + 3 器官表面（胸膜/心包代理）。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D12 = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
AUD = json.load(open(ROOT / "data" / "zc022" / "transitions_audit.json"))
SURF = json.load(open(ROOT / "data" / "zc022" / "chest_surface_paths.json"))

old = {(t["a"], t["b"]): t for t in D12["transitions"]}
order = [("GB", "LV"), ("LV", "LU"), ("LU", "LI"), ("LI", "ST"),
         ("ST", "SP"), ("SP", "HT"), ("HT", "SI"), ("SI", "BL"),
         ("BL", "KI"), ("KI", "PC"), ("PC", "TE"), ("TE", "GB")]
deep_keys = {"SP-HT", "KI-PC", "LV-LU"}

# BL→KI 从审计补算
blki = next(r for r in AUD["pairs"] if r["a"] == "BL" and r["b"] == "KI")

out = []
for a, b in order:
    key = f"{a}-{b}"
    if key in deep_keys:
        s = SURF[key]
        t = {
            "a": a, "b": b, "kind": "deep", "computed": True,
            "source": "organ_surface", "length": s["length"],
            "right": s["pts"], "left": [[-p[0], p[1], p[2]] for p in s["pts"]],
        }
    elif key == "BL-KI":
        t = {
            "a": a, "b": b, "kind": "surface", "computed": True,
            "source": "surface_computed", "length": blki["length"],
            "right": blki["right"], "left": blki["left"],
        }
    else:
        t = dict(old[(a, b)])
    out.append(t)

meta = {
    "count": len(out),
    "surface": sum(1 for t in out if t["kind"] == "surface"),
    "deep": sum(1 for t in out if t["kind"] == "deep"),
    "deep_source": "chest_surface_paths.json (lung=pleura proxy, heart=pericardium proxy)",
    "blki_source": "transitions_audit.json (surface shortest path)",
}
json.dump({"transitions": out, "meta": meta},
          open(ROOT / "data" / "atlas2" / "transitions_updated.json", "w", encoding="utf-8"),
          ensure_ascii=False, separators=(",", ":"))
print("saved", ROOT / "data" / "atlas2" / "transitions_updated.json")
for t in out:
    print(t["a"], "->", t["b"], t["kind"], "computed" if t.get("computed") else "data",
          round(t["length"], 2), len(t["right"]))
