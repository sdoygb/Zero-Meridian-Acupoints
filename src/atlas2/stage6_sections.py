#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6：截面图——体表通道 vs 深筋膜层通道（w≠常数后的最优埋深）。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_obj, slice_mesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'atlas2'
D12 = json.load(open(OUT / 'paths12.json'))
D5 = json.load(open(OUT / 'depth_paths.json'))


def near(P, y, tol=0.18):
    P = np.asarray(P)
    return P[np.abs(P[:, 1] - y) < tol]


def main():
    V, F = load_obj()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), facecolor='#0d1117')

    # ---- 左：大腿截面
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    yy, xlo = -3.6, 0.8
    loops = [l for l in slice_mesh(V, F, 1, yy) if l[:, 0].mean() > xlo and len(l) > 8]
    L = max(loops, key=len)
    o = [0, 2]
    ax.fill(L[:, o[0]], L[:, o[1]], color='#1b2434', zorder=1)
    ax.plot(L[:, o[0]], L[:, o[1]], color='#9fb3d1', lw=1.2, zorder=2)
    c = L[:, o].mean(0)
    d = L[:, o] - c
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    for off in (0.02, 0.08):
        ax.plot(L[:, o[0]] - d[:, 0] * off, L[:, o[1]] - d[:, 1] * off,
                color='#ffd166', lw=0.6, alpha=0.35, zorder=2)
    for code in D12['order']:
        Ps = near(D12['meridians'][code]['right'], yy)
        Pd = near(D5['meridians'][code]['pts'], yy)
        Ps = Ps[Ps[:, 0] > xlo] if len(Ps) else Ps
        Pd = Pd[Pd[:, 0] > xlo] if len(Pd) else Pd
        col = D12['meridians'][code]['color']
        if len(Ps):
            ax.scatter(Ps[:, o[0]], Ps[:, o[1]], s=26, facecolors='none',
                       edgecolors=col, linewidths=1.4, zorder=4)
        if len(Pd):
            ax.scatter(Pd[:, o[0]], Pd[:, o[1]], s=20, c=col, zorder=5)
    ax.set_aspect('equal')
    ax.set_title('大腿截面 y = -3.6（右腿）：空心＝体表，实心＝深筋膜层', color='#dfe7f3', fontsize=11)
    ax.tick_params(colors='#7f8ea8', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#2b3648')
    ax.grid(alpha=0.12, color='#5a6b85')
    ax.legend(loc='lower left', facecolor='#141c2b', edgecolor='#2b3648',
              labelcolor='#cfd8e3', fontsize=9, framealpha=0.9)

    # ---- 右：12 条通道的埋深沿程
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    ax.axhspan(0.2, 0.8, color='#ffd166', alpha=0.13, zorder=1)
    ax.axhline(0.2, color='#ffd166', lw=0.6, alpha=0.5)
    ax.axhline(0.8, color='#ffd166', lw=0.6, alpha=0.5)
    for code in D12['order']:
        dep = np.array(D5['meridians'][code]['depth_cm'])
        u = np.linspace(0, 1, len(dep))
        ax.plot(u, dep, color=D12['meridians'][code]['color'], lw=1.0, alpha=0.85)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('沿通道弧长（归一化）', color='#9fb3d1', fontsize=10)
    ax.set_ylabel('埋深（cm，沿内法向）', color='#9fb3d1', fontsize=10)
    ax.set_title('十二条通道的埋深：全部落在深筋膜带 2–8 mm 内', color='#dfe7f3', fontsize=11)
    ax.tick_params(colors='#7f8ea8', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#2b3648')
    ax.grid(alpha=0.12, color='#5a6b85')

    fig.suptitle('Zero 经络 v19：w≠常数后的通道埋深（φ 未实测，w 取结构定义）',
                 color='#dfe7f3', fontsize=13)
    fig.tight_layout()
    p = OUT / 'zero_atlas_v19_sections.png'
    fig.savefig(p, dpi=150, facecolor='#0d1117')
    print('saved', p)


if __name__ == '__main__':
    main()
