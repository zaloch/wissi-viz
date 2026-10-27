"""Synthetic multiplex cell table registered to tests' synthetic_he.svs (for the analysis demo).

Writes, next to the slide:
  synthetic_he_cells.tsv   QuPath-style measurement export (µm centroids, marker means, DAB OD, Ki67)
  synthetic_he_rois.geojson tumour ROI + two "hot" regions (level-0 px)

Biology (made up but realistic in shape): tumour nests inside the ROI; an immune-rich invasive
margin (CD8/CD4 T cells, macrophages) 0–150 µm outside it; T-cell clusters (TLS-like) in stroma;
Ki67 high at the tumour edge; DAB (PD-L1-like) graded in tumour cells.

    python tests/make_analysis_data.py DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(d: Path, mpp: float = 0.5, seed: int = 7) -> None:
    import shapely
    from shapely.geometry import Polygon

    rng = np.random.default_rng(seed)
    gj = json.loads((d / 'synthetic_he.geojson').read_text(encoding='utf-8'))
    xy, rois = [], []
    for f in gj['features']:
        g = f['geometry']
        if f['properties']['classification']['name'] == 'ROI':
            rois.append(f)
            continue
        c = np.asarray(g['coordinates'][0])[:-1]
        xy.append(c.mean(0))
    xy = np.asarray(xy)
    W, H = 16000, 12000
    tumor = Polygon([[W * .1, H * .1], [W * .45, H * .12], [W * .5, H * .5], [W * .2, H * .45]])
    d_px = shapely.distance(tumor.boundary, shapely.points(xy[:, 0], xy[:, 1]))
    inside = shapely.contains_xy(tumor, xy[:, 0], xy[:, 1])
    sd = np.where(inside, -d_px, d_px) * mpp  # signed µm
    n = len(xy)
    cls = np.empty(n, object)
    u = rng.random(n)
    # tumour interior
    m = inside
    cls[m] = np.select([u[m] < .78, u[m] < .86, u[m] < .91], ['Tumor', 'CD8 T cell', 'CD4 T cell'], 'Macrophage')
    # invasive margin 0–150 µm outside
    m = (~inside) & (sd < 150)
    cls[m] = np.select([u[m] < .30, u[m] < .55, u[m] < .72], ['CD8 T cell', 'CD4 T cell', 'Macrophage'], 'Stroma')
    # distant stroma
    m = (~inside) & (sd >= 150)
    cls[m] = np.select([u[m] < .86, u[m] < .91, u[m] < .95], ['Stroma', 'CD8 T cell', 'CD4 T cell'], 'Macrophage')
    # TLS-like T/B clusters
    for cx, cy in ((12500, 8500), (13500, 3500), (6500, 9500)):
        near = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < 500
        cls[near] = rng.choice(['CD4 T cell', 'CD8 T cell', 'B cell'], near.sum(), p=[.45, .25, .30])

    def marker(pos, hi=900, lo=40):
        return np.where(pos, rng.normal(hi, hi * .25, n), rng.gamma(2, lo / 2, n)).clip(0)
    cd3 = marker(np.isin(cls, ['CD8 T cell', 'CD4 T cell']))
    cd8 = marker(cls == 'CD8 T cell', 800)
    cd68 = marker(cls == 'Macrophage', 700)
    ck = marker(cls == 'Tumor', 1200)
    cd20 = marker(cls == 'B cell', 650)
    edge = np.exp(-np.abs(sd) / 120)
    ki67 = np.where(cls == 'Tumor', rng.random(n) < (.15 + .5 * edge), rng.random(n) < .03) * rng.normal(700, 150, n).clip(0)
    dab = np.where(cls == 'Tumor', rng.beta(2, 3, n) * (0.4 + 0.6 * edge), rng.beta(1.2, 12, n)) * 1.0
    area = np.where(cls == 'Tumor', rng.normal(55, 12, n), rng.normal(28, 6, n)).clip(8)
    import pandas as pd
    df = pd.DataFrame({
        'Image': 'synthetic_he.svs', 'Object ID': np.arange(n), 'Class': cls,
        'Centroid X µm': xy[:, 0] * mpp, 'Centroid Y µm': xy[:, 1] * mpp,
        'Nucleus: Area µm^2': area.round(2),
        'Cell: CD3 mean': cd3.round(1), 'Cell: CD8 mean': cd8.round(1), 'Cell: CD68 mean': cd68.round(1),
        'Cell: PanCK mean': ck.round(1), 'Cell: CD20 mean': cd20.round(1), 'Nucleus: Ki67 mean': ki67.round(1),
        'DAB: Membrane: Mean OD': dab.round(4),
    })
    df.to_csv(d / 'synthetic_he_cells.tsv', sep='\t', index=False)
    feats = [{'type': 'Feature', 'properties': {'name': 'Tumor', 'classification': {'name': 'Tumor ROI', 'color': [255, 60, 60]}},
              'geometry': {'type': 'Polygon', 'coordinates': [list(map(list, tumor.exterior.coords))]}}]
    for name, (cx, cy, r) in {'Stroma A': (12800, 7800, 1800), 'Stroma B': (3800, 9600, 1500)}.items():
        t = np.linspace(0, 2 * np.pi, 40)
        ring = np.stack([cx + r * np.cos(t), cy + r * 0.8 * np.sin(t)], 1).tolist()
        feats.append({'type': 'Feature', 'properties': {'name': name, 'classification': {'name': name}},
                      'geometry': {'type': 'Polygon', 'coordinates': [ring]}})
    (d / 'synthetic_he_rois.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')
    print(n, 'cells;', pd.Series(cls).value_counts().to_dict())


if __name__ == '__main__':
    main(Path(sys.argv[1]))
