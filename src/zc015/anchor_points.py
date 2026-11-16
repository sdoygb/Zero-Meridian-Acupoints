#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体表标记锚定：用关节+比例寸（WHO 定位描述）在模型上重建 10 个四肢穴位。
舌寸：前臂肘横纹—腕横纹 12 寸；小腿膝横纹—踝 16 寸（比例寸）。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'atlas2'))
from core import load_obj, ray_surface
from stage2_paths import read_joints

V, F = load_obj()
J = read_joints()
tree = cKDTree(V)


def surf(target):
    return V[int(tree.query(np.asarray(target, float))[1])]


def along(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


S = np.array(J['l-shoulder']); E = np.array(J['l-elbow']); W = np.array(J['l-hand'])
U = np.array(J['l-upper-leg']); K = np.array(J['l-knee']); A = np.array(J['l-ankle'])
cun_a = np.linalg.norm(W - E) / 12.0
cun_l = np.linalg.norm(A - K) / 16.0
fa = (W - E) / np.linalg.norm(W - E)          # 前臂轴（肘→腕）
sa = (A - K) / np.linalg.norm(A - K)          # 小腿轴（膝→踝）
# 手掌法向（拇-食-小指 平面）
thumb = np.array(J['l-finger-1-4']); index = np.array(J['l-finger-2-4']); pinky = np.array(J['l-finger-5-4'])
pn = np.cross(index - thumb, pinky - thumb); pn /= np.linalg.norm(pn)
# 让 pn 指向掌心一侧：掌心朝向身体内侧（x 小的一侧）
if pn[0] > 0:
    pn = -pn
dorsal = -pn
lateral = np.array([1.0, 0, 0])

anchors = {
    'PC6': surf(W - fa * 2 * cun_a + pn * 0.12),
    'LU9': surf(W + (thumb - W) / np.linalg.norm(thumb - W) * 0.10),
    'HT7': surf(W + (pinky - W) / np.linalg.norm(pinky - W) * 0.10),
    'LI4': surf(along(thumb, index, 0.45) + dorsal * 0.10),
    'LI11': surf(E + lateral * 0.16 + dorsal * 0.05),
    'ST36': surf(K + sa * 3 * cun_l + lateral * 0.14),
    'SP6': surf(A - sa * 3 * cun_l + np.array([-1.0, 0, 0]) * 0.13),
    'GB34': surf(K + lateral * 0.18 + np.array([0, -0.12, 0])),
    'GB39': surf(A - sa * 3 * cun_l + lateral * 0.13),
    'KI3': surf(A + np.array([-0.10, 0, -0.10])),
}
out = {'meta': {'num': len(anchors), 'cun_arm': float(cun_a), 'cun_leg': float(cun_l)},
       'points': {k: [round(float(x), 3) for x in v] for k, v in anchors.items()}}
json.dump(out, open(ROOT / 'data' / 'zc015' / 'anchors.json', 'w'), ensure_ascii=False)
for k, v in anchors.items():
    print(f'{k:5s} {np.round(v,2)}')
print('cun_arm %.3f  cun_leg %.3f (units)' % (cun_a, cun_l))
