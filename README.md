# Zero Meridian & Acupoint Atlas

A 3D meridian and acupoint atlas **computed from the Zero framework** on a neutral human model — the channels are derived from a variational principle, not traced from traditional charts.

### ▶ [Open the interactive 3D atlas](https://sdoygb.github.io/Zero-Meridian-Acupoints/)

![Atlas front view](data/atlas2/zero_atlas_v21_front.png)

The page is a single self-contained HTML file (`index.html`, ~1.9 MB, no network needed). Drag to rotate, scroll to zoom, right-drag to pan; the left panel toggles layers.

## What it shows

| Layer | Content |
|:--|:--|
| 12 primary meridians | port-to-port surface geodesics, then lowered to the computed channel depth |
| 流注 transitions | 8 limb/head transitions + 3 deep transitions, at the same depth |
| 8 extraordinary vessels | Ren, Du, Dai, Chong, Wei (yin/yang), Qiao (yin/yang) |
| 3 channels, 7 chakras | central / left / right channel, 7 wheel rings along the axis |
| **326 acupoints** | local extrema of the field φ along each channel (Zc014), labelled with approximate traditional names |
| 3 channels / 7 chakras | central channel = max-clearance axis; left/right = equal-winding helices on the core shell; chakra = φ extremum along the axis (Zc015): 5 of 7 traditional wheels within 5 cm |
| Depth | theorem value d\* = ST + SAT + t_f/2 (Zc013), 4.5–8.9 mm by body region |

## Theory in brief

The Zero framework describes a meridian as the set of **ports–channels–extrema**:

- a channel is `argmin ∫ ds / w` over a causal allowed set C, where w = propagation rate (tissue-dependent);
- **interface theorem**: a channel runs at a tissue interface when `w_i > λ·max(w1, w2)`, `λ = L_i / L_b`;
- **depth**: `d* = ST + SAT + t_f/2` — subcutaneous thickness + superficial adipose + half the fascia thickness (Zc013);
- **acupoints**: a point is a port; a *special* point is a local extremum of φ along the channel; the 1D index theorem gives `#max − #min ∈ {−1, 0, 1}` (Zc011).

The formal derivations are in the five articles at the repository root — see **[Theory articles](#theory-articles)** below for the full list with links.


## Theory articles

The five derivations behind the atlas — formal statements, proofs and computation tables (Chinese text, as in the Zc series):

| No. | Article | Content |
|:--|:--|:--|
| Zc011 | [经络与穴位：端口、通道与极值](https://github.com/sdoygb/Zero-Meridian-Acupoints/blob/main/Zc011_%E7%BB%8F%E7%BB%9C%E4%B8%8E%E7%A9%B4%E4%BD%8D_%E7%AB%AF%E5%8F%A3%E9%80%9A%E9%81%93%E4%B8%8E%E6%9E%81%E5%80%BC.md) | channel = variational extremum; point = boundary port; special point = φ extremum; 1D index theorem |
| Zc012 | [经络与穴位的计算对照：361 穴、14 经、8 奇经](https://github.com/sdoygb/Zero-Meridian-Acupoints/blob/main/Zc012_%E7%BB%8F%E7%BB%9C%E4%B8%8E%E7%A9%B4%E4%BD%8D%E7%9A%84%E8%AE%A1%E7%AE%97%E5%AF%B9%E7%85%A7_361%E7%A9%B414%E7%BB%8F8%E5%A5%87%E7%BB%8F.md) | comparison with the traditional catalogue; counts are coarse-graining |
| Zc013 | [经络的层深：界面判据与实测层厚](https://github.com/sdoygb/Zero-Meridian-Acupoints/blob/main/Zc013_%E7%BB%8F%E7%BB%9C%E7%9A%84%E5%B1%82%E6%B7%B1_%E7%95%8C%E9%9D%A2%E5%88%A4%E6%8D%AE%E4%B8%8E%E5%AE%9E%E6%B5%8B%E5%B1%82%E5%8E%9A.md) | interface theorem; d* = ST + SAT + t_f/2; measured layer thicknesses |
| Zc014 | [穴位位置的推导：φ 极值与计数对照](https://github.com/sdoygb/Zero-Meridian-Acupoints/blob/main/Zc014_%E7%A9%B4%E4%BD%8D%E4%BD%8D%E7%BD%AE%E7%9A%84%E6%8E%A8%E5%AF%BC_%CF%86%E6%9E%81%E5%80%BC%E4%B8%8E%E8%AE%A1%E6%95%B0%E5%AF%B9%E7%85%A7.md) | 326 acupoints as field extrema; naming; comparison statistics |
| Zc015 | [三脉七轮的推导：轴心通道与轮的判据](https://github.com/sdoygb/Zero-Meridian-Acupoints/blob/main/Zc015_%E4%B8%89%E8%84%89%E4%B8%83%E8%BD%AE%E7%9A%84%E6%8E%A8%E5%AF%BC_%E8%BD%B4%E5%BF%83%E9%80%9A%E9%81%93%E4%B8%8E%E8%BD%AE%E7%9A%84%E5%88%A4%E6%8D%AE.md) | max-clearance axis; equal-winding helices; chakra = φ extremum along the axis |

The wider Zero series (causality, self, matter) lives in [sdoygb/Zero](https://github.com/sdoygb/Zero).

## Repository layout

```
index.html                  # the 3D atlas (GitHub Pages entry)
data/model/                 # neutral body mesh (MakeHuman base, CC0)
data/atlas2/                # channel geometry + build inputs (JSON)
data/zc013/                 # layer measurements (ST/SAT/t_f), audit outputs
data/zc014/                 # field φ, derived acupoints, naming, comparison
data/zc015/                 # WHO landmark anchors, TARA coordinates, depth anchors
src/atlas2/                 # mesh loading, surface routing, stage 1–5, HTML builder
src/zc013/                  # layer model, theorem application, audit
src/zc014/                  # φ fields, extremum extraction, naming
src/zc015/                  # anchor construction, TARA extraction, registration
Zc011–Zc015*.md             # theory articles
```

## Rebuild

```bash
pip install numpy scipy matplotlib
python3 src/atlas2/build_html.py v21     # writes data/atlas2/zero_atlas_v21.html
python3 src/zc014/phi_fields.py          # φ fields (port harmonic + bone distance)
python3 src/zc014/derive_points.py       # 326 acupoints
```

The HTML builder is self-contained (three.js r128 inlined); no bundler required.

## Validation (measured, not fitted)

Comparison against independent references (details in `Zc014` §5 and `data/zc014/compare_*.json`):

| Comparison | Median distance | ≤ 3 cm |
|:--|--:|--:|
| Channels vs 10 WHO-landmark anchors (no charts involved) | **2.0 cm** | — |
| Channels vs traditional 2D charts, per-view calibrated | 2.5 cm | 58 % |
| Acupoints vs WHO-landmark anchors | 2.7 cm | — |
| Acupoints vs traditional 2D charts, calibrated | 3.4 cm | 42 % |

![Distance CDF](data/zc014/compare_cdf.png)

Channel depth vs **measured needling depths** (literature: 5.5–21.2 mm ideal insertion, ultrasonic GB21/ST36 targets 22–26 mm): the channel layer (fascia, ~5 mm) and the stimulation target (muscle, 10–30 mm) are two different layers — see `data/zc015/depth_anchors.json`.

## Data provenance

- Body mesh: MakeHuman `hm08` base (CC0), body-only export.
- Traditional point catalogue and WHO location texts: `data/` (compare-only, never used to place the computed points).
- TARA repository public Unity build: acupoint coordinates extracted for anchor-registration experiments (`data/zc015/tara_points.json`); these are scaffold positions, not skin.
- Layer thicknesses: published ST/SAT/fascia measurements recorded in `data/zc013/layers.json` with URLs.

## Limitations

- φ is not measured; two proxies (port harmonic field, bone distance) give the same total order of points but different per-meridian distributions.
- Lateral channel position is a structural choice (corridor stations), not a theorem output; sensitivity to a 0.8 cm corridor shift is 0.34 cm.
- Comparison references are 2D chart projections; individual anatomical variation (≈1–2 cm) and the absence of a unified 3D standard bound further accuracy claims.
- Only the 12 primary meridians have computed acupoints so far.

## License

Code: MIT. Text (`Zc011`–`Zc014`): CC BY 4.0. Body mesh: CC0 (MakeHuman).
