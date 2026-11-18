#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Zc021 的联合图响应导出给 V21 HTML（data/zc021/web_dynamics.json）。"""
import json
from pathlib import Path

import numpy as np
import unified as U

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "zc021"
OUT.mkdir(parents=True, exist_ok=True)

D12 = json.load(open(ROOT / "data" / "atlas2" / "paths12.json"))
V = json.load(open(ROOT / "data" / "atlas2" / "vessels_channels.json"))
CH = V["chakras"]

# H name -> HTML data code
H2CODE = {"CV": "REN", "GV": "DU", "CHONG": "CHONG", "DAI": "DAI",
          "YINWEI": "YINWEI", "YANGWEI": "YANGWEI",
          "YINQIAO": "YINQIAO", "YANGQIAO": "YANGQIAO"}


def h_code(name):
    return H2CODE.get(name, name)


def h_curve(name, side):
    code = h_code(name)
    if code in D12["meridians"]:
        pts = D12["meridians"][code][side]
    else:
        pts = V["vessels"][code][side]
    return np.asarray(pts, float)


def nearest(pts, p):
    p = np.asarray(p, float)
    d = np.linalg.norm(pts - p, axis=1)
    return pts[int(np.argmin(d))].tolist()


def chakra_pos(k):
    c = CH[k]
    return [0.0, float(c["y"]), float(c["z"])]


def v_point(side, k):
    code = {"L": "LALA", "C": "SUSHUMA", "R": "RASA"}[side]
    curve_side = "left" if side == "L" else "right"
    pts = np.asarray(V["channels"][code][curve_side], float)
    return nearest(pts, chakra_pos(k))


def v_display(side, k):
    """源汇标记的显示点：左右各偏 0.28（仅显示，不是通道坐标）。"""
    x, y, z = chakra_pos(k)
    off = 0.28
    return [x + (off if side == "R" else (-off if side == "L" else 0.0)), y, z]


# H representative points (midpoint of right curve)
h_points = {}
for name in U.H_names:
    pts = h_curve(name, "right")
    h_points[h_code(name)] = pts[len(pts) // 2].round(4).tolist()

# V bead points
v_points = {s: [v_point(s, k) for k in range(len(CH))] for s in ("L", "C", "R")}

# interface links
links = {}
for mode in ("sym", "asym"):
    links[mode] = []
    for k in range(len(CH)):
        p0 = chakra_pos(k)
        for side in ("L", "C", "R"):
            if mode == "sym":
                hs = U.IFACE[k]
            else:
                hs = U.ASYM[k][side]
            for name in hs:
                p1 = nearest(h_curve(name, "left" if side == "L" else "right"), p0)
                links[mode].append({"k": k, "side": side, "h": h_code(name),
                                    "p0": p0, "p1": p1})

# responses
CASES = {
    "H_to_V": ("H", "ST", "SP"),
    "V_Caxis": ("V", ("C", 3), ("C", 6)),
    "V_LC": ("V", ("L", 3), ("C", 3)),
    "V_LR": ("V", ("L", 3), ("R", 3)),
}
G_LIST = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
responses = {}
for mode in ("sym", "asym"):
    responses[mode] = {}
    for g in G_LIST:
        responses[mode][f"{g:g}"] = {}
        for cname, (typ, a, b) in CASES.items():
            if typ == "H":
                src, sink = U.hidx(a), U.hidx(b)
            else:
                src, sink = U.vidx(*a), U.vidx(*b)
            u, s = U.solve(g, src, sink, mode=mode)
            uh, uv = U.norms(u)
            scale = float(np.max(np.abs(u)))
            responses[mode][f"{g:g}"][cname] = {
                "uH": [float(x) for x in u[:U.NH]],
                "uV": [float(x) for x in u[U.NH:]],
                "ratio": float((uv if typ == "H" else uh) / max((uh if typ == "H" else uv), 1e-30)),
                "zero": float(abs(u.sum())),
                "scale": scale,
            }

# source/sink points
case_points = {
    "H_to_V": {"src": h_points["ST"], "sink": h_points["SP"], "src_label": "ST 胃", "sink_label": "SP 脾"},
    "V_Caxis": {"src": v_display("C", 3), "sink": v_display("C", 6), "src_label": "C 心轮", "sink_label": "C 海底"},
    "V_LC": {"src": v_display("L", 3), "sink": v_display("C", 3), "src_label": "L 心轮", "sink_label": "C 心轮"},
    "V_LR": {"src": v_display("L", 3), "sink": v_display("R", 3), "src_label": "L 心轮", "sink_label": "R 心轮"},
}

out = {
    "h_order": U.H_names,
    "h_codes": {n: h_code(n) for n in U.H_names},
    "h_points": h_points,
    "chakras": CH,
    "v_points": v_points,
    "links": links,
    "cases": {k: {"typ": v[0], "src_label": v_info["src_label"], "sink_label": v_info["sink_label"],
                   "src": v_info["src"], "sink": v_info["sink"]}
              for k, v, v_info in [(k, v, case_points[k]) for k, v in CASES.items()]},
    "g_list": G_LIST,
    "responses": responses,
}
with open(OUT / "web_dynamics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("saved", OUT / "web_dynamics.json",
      "bytes", (OUT / "web_dynamics.json").stat().st_size)
