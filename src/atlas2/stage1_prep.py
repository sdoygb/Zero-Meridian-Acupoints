#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1：网格体检 · 体轴（中脉）· 七轮 · 任督矢状轮廓 · 四肢轴。

中轴不手选：用模型自带骨架关节（pelvis/spine-1..4/neck/head/head-2）作锚，
PCHIP 插值成体轴；任督用对称面 x=0 与网格的交线（几何唯一）。
七轮：模型骨架给出的 7 个解剖水平（数量 7 是粗粒化 M，见 Zc011 C6）。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ROOT, load_obj, slice_mesh, ray_surface
from stage1_prep_joints import read_joints

OUT = ROOT / 'data' / 'atlas2'
OUT.mkdir(parents=True, exist_ok=True)


def fit_axis(P):
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    a = vt[0]
    t = (P - c) @ a
    if t.max() < -t.min():
        a, t = -a, -t
    return c + a * t.min(), c + a * t.max(), float(s[0] / max(s[1], 1e-9))


def main():
    V, F = load_obj()
    J = read_joints()
    x, y, z = V.T
    info = {'n_vert': int(len(V)), 'n_tri': int(len(F)),
            'bbox': [V.min(0).round(4).tolist(), V.max(0).round(4).tolist()]}

    # ---- 体轴：骨架关节 + 端点到顶/会阴
    ks = ['pelvis', 'spine-4', 'spine-3', 'spine-2', 'spine-1', 'neck', 'head', 'head-2']
    ax_y = [J[k][1] for k in ks]
    ax_z = [J[k][2] for k in ks]
    ax_y = [-0.42] + ax_y + [8.45]
    ax_z = [0.36] + ax_z + [J['head-2'][2]]
    fz = PchipInterpolator(ax_y, ax_z)
    info['spine_axis'] = [{'y': float(a), 'z': float(fz(a))} for a in ax_y]

    # ---- 体轴（中脉）：骨架关节 + 头顶几何中心
    def geo_center(yy, seed):
        p = np.array(seed, float); p[1] = yy
        for _ in range(6):
            hs = []
            for d in ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]):
                q, t = ray_surface(V, F, p, np.array(d, float), maxdist=3.0, mode='first')
                hs.append(q)
            if any(h is None for h in hs):
                break
            p = np.array([(hs[0][0] + hs[1][0]) / 2, yy, (hs[2][2] + hs[3][2]) / 2])
        return p
    head_c = geo_center(7.9, (0, 7.9, 0.5))
    ax_pts = [(-0.42, 0.36)] + [(J[k][1], J[k][2]) for k in ks[:-2]] + \
             [(J['head'][1], 0.5 * (J['head'][2] + head_c[2])), (J['head-2'][1], head_c[2]), (8.45, head_c[2])]
    fz2 = PchipInterpolator([a for a, b in ax_pts], [b for a, b in ax_pts])
    info['axis_pts'] = [[float(a), float(b)] for a, b in ax_pts]

    rows = []
    for yy in np.arange(-0.40, 8.42, 0.05):
        o = np.array([0.0, yy, float(fz2(yy))])
        rr = []
        for d in ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]):
            q, t = ray_surface(V, F, o, np.array(d, float), maxdist=3.2, mode='first')
            rr.append(t if q is not None else 0.0)
        rows.append(dict(y=float(yy), z=float(o[2]), rx=float(min(rr[0], rr[1])),
                         rz=float(min(rr[2], rr[3])), r=float(min(rr))))
    # 清理射线失败的坏行（r=0），线性插值 + 轻度平滑
    rr = np.array([r['r'] for r in rows], float)
    good = rr > 0.02
    if good.sum() > 3:
        rr = np.interp(np.arange(len(rr)), np.where(good)[0], rr[good])
        k = 5
        rr = np.convolve(np.pad(rr, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')
    for q, v in zip(rows, rr):
        q['r'] = float(v)
    info['profile'] = rows
    py = np.array([r['y'] for r in rows]); pzc = np.array([r['z'] for r in rows])
    info['axis'] = [{'y': float(a), 'z': float(b)} for a, b in zip(py[::5], pzc[::5])]

    # ---- 七轮（骨架水平 + 几何端点）
    chak = [
        dict(key='crown', name='顶轮', y=float(J['head-2'][1])),
        dict(key='brow', name='眉间轮', y=float(J['l-eye'][1])),
        dict(key='throat', name='喉轮', y=float(J['neck'][1])),
        dict(key='heart', name='心轮', y=float(0.5 * (J['spine-1'][1] + J['spine-2'][1]))),
        dict(key='navel', name='脐轮', y=float(J['spine-3'][1])),
        dict(key='sacral', name='密轮', y=float(J['pelvis'][1])),
        dict(key='perineum', name='海底轮', y=-0.42),
    ]
    for c in chak:
        c['z'] = float(fz2(c['y']))
        c['r'] = float(np.interp(c['y'], py, np.array([r['r'] for r in rows])))
    info['chakras'] = chak

    # ---- 任督：对称面 x=0 的交线
    loops = slice_mesh(V, F, 0, 0.0)
    info['n_sag_loops'] = len(loops)
    loop = max(loops, key=len)
    info['sag_loop_n'] = int(len(loop))
    info['sag_y'] = [float(loop[:, 1].min()), float(loop[:, 1].max())]
    i_per = int(np.argmin(loop[:, 1]))
    # 鹊桥：离舌尖最近的点
    tgt = np.array([0.0, 6.55, 1.35])
    i_mouth = int(np.argmin(np.linalg.norm(loop - tgt, axis=1)))
    n = len(loop)
    def arc(i, j):
        if i <= j:
            return loop[i:j + 1]
        return np.vstack([loop[i:], loop[:j + 1]])
    a1 = arc(i_per, i_mouth)
    a2 = arc(i_mouth, i_per)
    # 前支（胸腹）z 平均更大
    ren, du = (a1, a2) if np.median(a1[:, 2]) > np.median(a2[:, 2]) else (a2, a1)
    info['rendu'] = dict(ren=[[float(a), float(b), float(c)] for a, b, c in ren],
                         du=[[float(a), float(b), float(c)] for a, b, c in du],
                         perineum=[float(q) for q in loop[i_per]],
                         qiao=[float(q) for q in loop[i_mouth]],
                         ren_len=float(np.linalg.norm(np.diff(ren, axis=0), axis=1).sum()),
                         du_len=float(np.linalg.norm(np.diff(du, axis=0), axis=1).sum()))

    # ---- 四肢轴
    arm = (np.abs(x) > 2.7) & (y > 0.9)
    lp = (x > 0.35) & (x < 2.7) & (y < -2.2)
    axes = {}
    for side, m in (('L', arm & (x > 0)), ('R', arm & (x < 0))):
        p0, p1, ar = fit_axis(V[m])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        axes['arm' + side] = dict(p0=(p0 - d * 0.45).tolist(), p1=p1.tolist(), aspect=ar,
                                  joints=[J['l-shoulder'].tolist(), J['l-elbow'].tolist(), J['l-hand'].tolist()])
    for side in ('L', 'R'):
        m = lp if side == 'L' else (x < -0.35) & (x > -2.7) & (y < -2.2)
        p0, p1, ar = fit_axis(V[m])
        axes['leg' + side] = dict(
            p0=J['l-upper-leg'].tolist(), p1=J['l-ankle'].tolist(), aspect=ar,
            joints=[J['l-upper-leg'].tolist(), J['l-knee'].tolist(), J['l-ankle'].tolist()])
    info['axes'] = axes

    json.dump(info, open(OUT / 'stage1.json', 'w'), ensure_ascii=False, indent=1)
    print('mesh', info['n_vert'], 'verts', info['n_tri'], 'tris; sag loops', len(loops), 'loop n', len(loop))
    print('chakras', [(c['name'], round(c['y'], 2), round(c['z'], 2), round(c['r'], 2)) for c in chak])
    print('rendu: ren_len', round(info['rendu']['ren_len'], 2), 'du_len', round(info['rendu']['du_len'], 2),
          'perineum', np.round(info['rendu']['perineum'], 2), 'qiao', np.round(info['rendu']['qiao'], 2))
    print('profile r:', [(round(r['y'], 1), round(r['r'], 2)) for r in rows[::24]])


if __name__ == '__main__':
    main()
