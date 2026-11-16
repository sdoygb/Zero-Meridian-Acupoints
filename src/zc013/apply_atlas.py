#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Zc013 的层深 d* 应用到 3D 图谱：12 经 + 奇经（任督带维跷）沿内法向偏移 d*，按体厚限深。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'zc013'))
from sites import LayerModel

OUT = ROOT / 'data' / 'atlas2'
D12 = json.load(open(OUT / 'paths12.json'))
D3 = json.load(open(OUT / 'vessels_channels.json'))

lm = LayerModel()
res = {'meridians': {}, 'vessels': {}, 'meta': {'depth_rule': 'd*=ST+SAT+t_f/2, clamp 0.45*thickness'}}
for code in D12['order']:
    for side in ('right', 'left'):
        P = np.array(D12['meridians'][code][side])
        Q, dmm, d = lm.offset(P)
        res['meridians'].setdefault(code, {})[side] = Q.round(4).tolist()
        res['meridians'][code][side + '_mm'] = [round(float(x), 2) for x in dmm]
for k, v in D3['vessels'].items():
    if k == 'CHONG':
        # 冲脉本来就是深部通道，保留
        res['vessels'][k] = {'right': v['right'], 'left': v.get('left', v['right']),
                             'keep': True}
        continue
    for side in ('right', 'left'):
        P = np.array(v[side])
        Q, dmm, d = lm.offset(P)
        res['vessels'].setdefault(k, {})[side] = Q.round(4).tolist()
        res['vessels'][k][side + '_mm'] = [round(float(x), 2) for x in dmm]
# 流注过渡也按 d* 偏移（否则会在皮肤上“回表面”）
res['transitions'] = {}
for t in D12['transitions']:
    key = t['a'] + '-' + t['b']
    for side in ('right', 'left'):
        P = np.array(t[side])
        Q, dmm, d = lm.offset(P)
        res['transitions'].setdefault(key, {})[side] = Q.round(4).tolist()
        res['transitions'][key][side + '_mm'] = [round(float(x), 2) for x in dmm]

for c, m in res['meridians'].items():
    print(c, 'd* mean', round(float(np.mean(m['right_mm'])), 2), 'mm')
for k, v in res['vessels'].items():
    if not v.get('keep'):
        print('ves', k, 'd* mean', round(float(np.mean(v['right_mm'])), 2))
json.dump(res, open(OUT / 'theorem_atlas.json', 'w'), ensure_ascii=False)
print('saved', OUT / 'theorem_atlas.json')
