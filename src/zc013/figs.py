#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zc013 图：层深分解（皮/脂/筋膜）与 12 经的 d* 对照。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'zc013'
D = json.load(open(OUT / 'depth_theorem.json'))
D12 = json.load(open(ROOT / 'data' / 'atlas2' / 'paths12.json'))
TIS = json.load(open(ROOT / 'data' / 'atlas2' / 'tissue_paths.json'))
V19 = json.load(open(ROOT / 'data' / 'atlas2' / 'depth_paths.json'))

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
BG = '#0d1117'; FG = '#dfe7f3'; GRID = '#2b3648'


def fig_layers():
    sites = D['sites']
    order = ['upper_arm', 'forearm', 'head', 'thigh', 'lower_leg', 'foot', 'neck', 'abdomen', 'chest', 'back']
    label = {'upper_arm': '上臂', 'forearm': '前臂', 'head': '头面', 'thigh': '大腿', 'lower_leg': '小腿',
             'foot': '足', 'neck': '颈', 'abdomen': '腹', 'chest': '胸', 'back': '背'}
    st = [sites[k]['st_mm'] for k in order]
    sa = [sites[k]['sat_mm'] for k in order]
    tf = D['params']['tf_mm'] / 2
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(order))
    ax.barh(y, st, color='#ffd166', label='皮肤 ST（实测，部位）')
    ax.barh(y, sa, left=st, color='#f4a261', alpha=0.85, label='皮下脂肪 SAT（全身均值锚定）')
    ax.barh(y, [tf] * len(order), left=np.array(st) + np.array(sa), color='#c77dff', label='深筋膜 t_f/2')
    for i, k in enumerate(order):
        d = sites[k]['depth_mm']
        ax.text(d + 0.15, i, f'{d:.1f} mm', va='center', color=FG, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([label[k] for k in order], color=FG)
    ax.set_xlabel('深度 d*（mm）：d* = ST + SAT + t_f/2', color=FG)
    ax.tick_params(colors=FG)
    ax.grid(axis='x', color=GRID, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(facecolor='#141c2b', edgecolor=GRID, labelcolor=FG, fontsize=9, loc='lower right')
    ax.set_title('Zc013 · 层深定理：按实测层厚的界面深度', color=FG, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'zc013_layers.png', dpi=150, facecolor=BG)
    print('saved', OUT / 'zc013_layers.png')


def fig_meridians():
    order = D12['order']
    names = [D12['meridians'][c]['name'] for c in order]
    dstar = [D['meridians'][c]['depth_mean_mm'] for c in order]
    v19 = [V19['meridians'][c]['depth_mean_cm'] * 10 for c in order]
    tis = [TIS['guided'][c]['depth_mean'] * 10.2 * 10 for c in order]  # units→mm
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - 0.22, dstar, width=0.22, color='#5ec8e5', label='定理 d*（皮+脂+筋膜，实测层厚）')
    ax.bar(x, v19, width=0.22, color='#ffd166', label='v19 法向优化（w 结构定义，封顶 8 mm）')
    ax.bar(x + 0.22, tis, width=0.22, color='#ff6b6b', alpha=0.85, label='体素路由（8 mm 网格，走廊内）')
    ax.set_xticks(x); ax.set_xticklabels([n[2:] for n in names], rotation=30, ha='right', color=FG, fontsize=9)
    ax.set_ylabel('埋深（mm）', color=FG)
    ax.tick_params(colors=FG)
    ax.grid(axis='y', color=GRID, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(facecolor='#141c2b', edgecolor=GRID, labelcolor=FG, fontsize=9)
    ax.set_title('Zc013 · 三种算法给出的 12 经埋深', color=FG, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'zc013_meridians.png', dpi=150, facecolor=BG)
    print('saved', OUT / 'zc013_meridians.png')


if __name__ == '__main__':
    fig_layers(); fig_meridians()
