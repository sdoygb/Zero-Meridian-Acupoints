#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc021：十二经（H：12+8）与三脉七轮（V：3×7）的联合动力学。

把两套传统系统读成同一张图的两个投影：
  H = 12 经 + 8 奇经（横向/环-水库图，Zc020）；
  V = 左/中/右 + 7 轮（纵向/轴-轮桥图，Zc015/Zc020）；
  界面 = 7 轮 ↔ 横向节点（会阴/密/脐/心/喉/眉间/顶；M/I）；
  界面电导 g⊥ = “隔板”的穿透度：0=两套分开，>0=耦合，大=打通。

输出 data/zc021/unified.json、data/zc021/unified.png。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "zc021"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- H: 12 经 + 8 奇经
mer = ["LU", "LI", "ST", "SP", "HT", "SI", "BL", "KI", "PC", "TE", "GB", "LV"]
vessels = {
    "CV": ["LU", "SP", "HT", "KI", "PC"],
    "GV": ["LI", "SI", "BL", "GB", "TE"],
    "CHONG": ["ST", "KI", "PC", "LU"],
    "DAI": ["GB", "LV", "SP", "KI"],
    "YINWEI": ["KI", "SP", "LV", "PC"],
    "YANGWEI": ["GB", "TE", "SI", "BL"],
    "YINQIAO": ["KI", "SP"],
    "YANGQIAO": ["BL", "GB"],
}
H_names = mer + list(vessels)
NH = len(H_names)


def hidx(x):
    return H_names.index(x)


def h_edges():
    edges = []
    for i in range(12):
        edges.append((mer[i], mer[(i + 1) % 12], 1.0))
    for v, conns in vessels.items():
        for m in conns:
            edges.append((v, m, 0.5))
    for a, b, g in [("CV", "GV", 0.4), ("CV", "CHONG", 0.4), ("DAI", "GV", 0.3),
                    ("YINWEI", "YINQIAO", 0.3), ("YANGWEI", "YANGQIAO", 0.3),
                    ("CHONG", "YINWEI", 0.3), ("GV", "YANGWEI", 0.3)]:
        edges.append((a, b, g))
    return edges


# ---------------------------------------------------------------- V: 3×7
V = json.load(open(ROOT / "data" / "atlas2" / "vessels_channels.json"))
chakras = V["chakras"]  # crown .. perineum
ys = np.array([c["y"] for c in chakras], float)
n = len(ys)
dh = np.abs(np.diff(ys))
gvert = np.clip(np.mean(dh) / np.maximum(dh, 1e-9), 0.2, 5.0)
sides = ["L", "C", "R"]
NV = 3 * n


def vidx(s, k):
    return NH + sides.index(s) * n + k


def v_edges():
    edges = []
    for k in range(n - 1):
        for s in sides:
            edges.append((vidx(s, k), vidx(s, k + 1), float(gvert[k])))
    for k in range(n):
        edges.append((vidx("L", k), vidx("C", k), 1.0))
        edges.append((vidx("C", k), vidx("R", k), 1.0))
    return edges


# 7 轮 ↔ 横向节点（M/I 粗粒化；不是解剖同一）
IFACE = {
    0: ["GV", "BL", "GB"],       # 顶
    1: ["GV", "GB", "BL"],       # 眉间
    2: ["CV", "GV", "LU", "ST"], # 喉
    3: ["CV", "PC", "HT", "BL"], # 心
    4: ["CV", "ST", "SP", "DAI"],# 脐
    5: ["CV", "CHONG", "KI"],    # 密
    6: ["CV", "GV", "CHONG"],    # 海底/会阴
}
# 不对称界面（对照）：左→偏阴/左，右→偏阳/右，中→中线；这是两个投影的差模能互通的关键
ASYM = {
    0: {"L": ["GV", "BL"], "C": ["GV"], "R": ["GV", "GB"]},
    1: {"L": ["GV", "BL"], "C": ["GV"], "R": ["GV", "GB"]},
    2: {"L": ["CV", "LU"], "C": ["CV", "GV"], "R": ["GV", "ST"]},
    3: {"L": ["CV", "PC", "HT"], "C": ["CV"], "R": ["BL", "SI"]},
    4: {"L": ["CV", "SP"], "C": ["CV"], "R": ["ST", "DAI"]},
    5: {"L": ["CV", "KI", "CHONG"], "C": ["CV", "CHONG"], "R": ["GV", "BL"]},
    6: {"L": ["CV", "CHONG"], "C": ["CV", "GV", "CHONG"], "R": ["GV"]},
}
N = NH + NV


def laplacian(gperp, mode="sym"):
    L = np.zeros((N, N))

    def add(i, j, g):
        L[i, i] += g
        L[j, j] += g
        L[i, j] -= g
        L[j, i] -= g

    for a, b, g in h_edges():
        add(hidx(a), hidx(b), g)
    for i, j, g in v_edges():
        add(i, j, g)
    if mode == "sym":
        for k, hnodes in IFACE.items():
            for s in sides:
                for hn in hnodes:
                    add(vidx(s, k), hidx(hn), gperp)
    else:
        for k, m in ASYM.items():
            for s in sides:
                for hn in m[s]:
                    add(vidx(s, k), hidx(hn), gperp)
    return L


def solve(gperp, source, sink, mode="sym"):
    L = laplacian(gperp, mode=mode)
    s = np.zeros(N)
    s[source] += 1.0
    s[sink] -= 1.0
    u = np.linalg.pinv(L) @ s
    u += -(u.sum()) / N
    return u, s


def norms(u):
    uh = np.linalg.norm(u[:NH])
    uv = np.linalg.norm(u[NH:])
    return uh, uv


# ---------------------------------------------------------------- 扰动
# H→V: ST→SP（横向偶极）
h_src, h_sink = hidx("ST"), hidx("SP")
# V→H 的三种：纯左右差模、中央轴共模、左→中（不对称）
v_cases = {
    "L_R_diff": (vidx("L", 3), vidx("R", 3)),
    "C_axis": (vidx("C", 3), vidx("C", 6)),
    "L_C": (vidx("L", 3), vidx("C", 3)),
}

g_list = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
sweep = []
profiles = {}
for mode in ("sym", "asym"):
    for g in g_list:
        u_hv, _ = solve(g, h_src, h_sink, mode=mode)
        uh_hv, uv_hv = norms(u_hv)
        vh = {}
        for name, (a, b) in v_cases.items():
            u_vh, _ = solve(g, a, b, mode=mode)
            uh_vh, uv_vh = norms(u_vh)
            vh[name] = {
                "within_V": float(uv_vh), "cross_H": float(uh_vh),
                "ratio_cross_within": float(uh_vh / max(uv_vh, 1e-30)),
                "zero_sum": float(abs(u_vh.sum())),
            }
            if g == 1.0:
                profiles.setdefault(mode, {}).setdefault(g, {})[name + "_H_nodes"] = {
                    hname: float(u_vh[hidx(hname)]) for hname in H_names
                }
        sweep.append({
            "mode": mode, "g_perp": g,
            "H_to_V": {"within_H": float(uh_hv), "cross_V": float(uv_hv),
                        "ratio_cross_within": float(uv_hv / max(uh_hv, 1e-30)),
                        "zero_sum": float(abs(u_hv.sum()))},
            "V_to_H": vh,
        })
        if g == 1.0:
            profiles[mode][g].update({
                "H_to_V_chakra_mean": [
                    float(np.mean([u_hv[vidx(s, k)] for s in sides])) for k in range(n)
                ],
                "H_to_V_chakra_L2": [
                    float(np.linalg.norm([u_hv[vidx(s, k)] for s in sides])) for k in range(n)
                ],
            })

res = {
    "interface": {str(k): v for k, v in IFACE.items()},
    "chakra_heights": ys.tolist(),
    "sweep": sweep,
    "profiles": profiles,
    "graph_sizes": {"H": NH, "V": NV, "N": N},
}
with open(OUT / "unified.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

ax = axes[0]
gs = [r["g_perp"] for r in sweep if r["mode"] == "sym"]
hv_s = [r["H_to_V"]["ratio_cross_within"] for r in sweep if r["mode"] == "sym"]
hv_a = [r["H_to_V"]["ratio_cross_within"] for r in sweep if r["mode"] == "asym"]
c_s = [r["V_to_H"]["C_axis"]["ratio_cross_within"] for r in sweep if r["mode"] == "sym"]
c_a = [r["V_to_H"]["C_axis"]["ratio_cross_within"] for r in sweep if r["mode"] == "asym"]
lr_a = [r["V_to_H"]["L_R_diff"]["ratio_cross_within"] for r in sweep if r["mode"] == "asym"]
ax.plot(gs, hv_s, "o-", color="#ff9f43", label="H→V（对称界面）")
ax.plot(gs, c_s, "o-", color="#5aa9ff", label="V→H C轴（对称）")
ax.plot(gs, hv_a, "s--", color="#ff9f43", alpha=0.6, label="H→V（不对称）")
ax.plot(gs, c_a, "s--", color="#5aa9ff", alpha=0.6, label="V→H C轴（不对称）")
ax.plot(gs, lr_a, "^-", color="#59b36b", label="V→H 左右差（不对称）")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlabel("interface conductance g⊥")
ax.set_ylabel("cross / within")
ax.set_title("(a) 隔板穿透：共模耦合，差模被对称界面挡住", fontsize=11, loc="left")
ax.grid(alpha=0.2)
ax.legend(fontsize=7)

ax = axes[1]
k = np.arange(n)
vals = np.array(profiles["sym"][1.0]["H_to_V_chakra_mean"])
ax.bar(k, vals, color=["#59b36b" if v >= 0 else "#ff9f43" for v in vals])
ax.axhline(0, color="#888", lw=0.8)
ax.set_xticks(k)
ax.set_xticklabels([c["name"] for c in chakras], fontsize=8)
ax.set_title("(b) g⊥=1，对称界面：ST/SP → 七轮响应（共模）", fontsize=11, loc="left")
ax.set_ylabel("mean u (V)")
ax.grid(axis="y", alpha=0.2)

ax = axes[2]
names = H_names
vals = np.array([profiles["asym"][1.0]["L_R_diff_H_nodes"][x] for x in names])
order = np.argsort(-np.abs(vals))[:10]
ax.bar([names[i] for i in order], vals[order], color="#5aa9ff")
ax.axhline(0, color="#888", lw=0.8)
ax.set_title("(c) g⊥=1，不对称界面：左右差 → 横向节点（前 10）", fontsize=11, loc="left")
ax.set_ylabel("u (H)")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "unified.png", dpi=160)
plt.close(fig)

print("sizes", NH, NV, N)
for r in sweep:
    vh = r["V_to_H"]
    print(f"{r['mode']:4s} g={r['g_perp']:6g}  H→V {r['H_to_V']['ratio_cross_within']:.4f}  "
          f"V→H Caxis {vh['C_axis']['ratio_cross_within']:.4f}  "
          f"L→C {vh['L_C']['ratio_cross_within']:.4f}  "
          f"L→R {vh['L_R_diff']['ratio_cross_within']:.4f}")
print("g=1 asym H→V chakra means:", [round(x, 4) for x in profiles["asym"][1.0]["H_to_V_chakra_mean"]])
print("g=1 asym L/R top:", sorted([(names[i], round(profiles["asym"][1.0]["L_R_diff_H_nodes"][names[i]], 4)) for i in order], key=lambda x: -abs(x[1])))
print("saved", OUT / "unified.json", OUT / "unified.png")
