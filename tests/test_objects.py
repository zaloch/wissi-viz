"""Object importers (QuPath / HALO / CellProfiler / GeoJSON / label mask / Parquet / AnnData) and colour-by."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from wissi_viz.annotations import AnnotationLayer
from wissi_viz.objects import from_label_mask, from_table, read_objects

pd = pytest.importorskip('pandas')


def test_qupath_tsv_microns(tmp_path):
    df = pd.DataFrame({'Image': ['s'] * 3, 'Object ID': [1, 2, 3], 'Class': ['Tumor', 'CD8', None],
                       'Centroid X µm': [10, 20, 30.], 'Centroid Y µm': [5, 6, 7.], 'Cell: CD8 mean': [1, 500, 2.]})
    p = tmp_path / 'q.tsv'
    df.to_csv(p, sep='\t', index=False)
    t = read_objects(p, mpp=0.5)
    assert np.allclose(t.x, [20, 40, 60]) and np.allclose(t.y, [10, 12, 14])
    assert list(t.classes) == ['Tumor', 'CD8', ''] and 'Cell: CD8 mean' in t.columns
    with pytest.raises(ValueError, match='mpp'):
        read_objects(p)  # µm without mpp is an error, never a silent px/µm mix-up


def test_halo_object_data_phenotypes(tmp_path):
    df = pd.DataFrame({'Analysis Region': ['L1'] * 3, 'XMin': [0, 10, 20], 'XMax': [4, 16, 26], 'YMin': [0, 10, 20],
                       'YMax': [4, 14, 30], 'CD3 Positive Classification': [1, 1, 0], 'CD8 Positive Classification': [1, 0, 0],
                       'Cell Area (µm²)': [10, 12, 9]})
    p = tmp_path / 'halo.csv'
    df.to_csv(p, index=False)
    t = read_objects(p)
    assert list(t.classes) == ['CD3+CD8+', 'CD3+', 'Negative']
    assert np.allclose(t.x, [2, 13, 23]) and t.bbox.shape == (3, 4)
    L = t.to_layer()
    assert len(L) == 3 and L.properties(1)['Cell Area (µm²)'] == 12


def test_cellprofiler_and_parquet(tmp_path):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'ImageNumber': [1, 1], 'ObjectNumber': [1, 2], 'Location_Center_X': [5.5, 9.0],
                       'Location_Center_Y': [1.0, 2.0], 'Intensity_MeanIntensity_DAPI': [0.2, 0.4]})
    p = tmp_path / 'cp.parquet'
    df.to_parquet(p)
    t = read_objects(p)
    assert np.allclose(t.x, [5.5, 9]) and t.classes is None and 'Intensity_MeanIntensity_DAPI' in t.columns


def test_qupath_geojson_detections(tmp_path):
    feats = [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
              'properties': {'objectType': 'cell', 'classification': {'name': 'Tumor'},
                             'measurements': {'Nucleus: Area': 50.0, 'CD3 mean': 3.0}}},
             {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[20, 20], [30, 20], [25, 30], [20, 20]]]},
              'properties': {'classification': {'name': 'CD3'}, 'measurements': [{'name': 'CD3 mean', 'value': 900}]}}]
    p = tmp_path / 'cells.geojson'
    p.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}))
    t = read_objects(p, mpp=0.25)
    assert list(t.classes) == ['Tumor', 'CD3'] and len(t.outlines) == 2
    assert t.columns['CD3 mean'][1] == 900 and np.isnan(t.columns['Nucleus: Area'][1])


def test_label_mask():
    pytest.importorskip('scipy')
    m = np.zeros((60, 60), np.int32)
    m[5:15, 5:15] = 3
    m[30:40, 20:45] = 7
    img = np.zeros((60, 60))
    img[30:40, 20:45] = 100
    t = from_label_mask(m, mpp=0.5, downsample=2, offset=(1000, 0), intensities={'CD8': img})
    assert len(t) == 2 and list(t.columns['label']) == [3, 7]
    assert np.allclose(t.x, [1000 + 9.5 * 2, 1000 + 32 * 2]) and np.allclose(t.columns['area_px'], [400, 1000])
    assert np.allclose(t.columns['CD8 mean'], [0, 100])
    if t.outlines is not None:
        o = t.outlines[1]
        assert o[:, 0].min() >= 1000 + 19 * 2 and o[:, 0].max() <= 1000 + 45 * 2


def test_anndata(tmp_path):
    ad = pytest.importorskip('anndata')
    a = ad.AnnData(X=np.array([[1., 2.], [3., 4.], [5., 6.]], np.float32),
                   obs=pd.DataFrame({'cell_type': ['T', 'B', 'T'], 'area': [1., 2., 3.]}, index=['a', 'b', 'c']))
    a.var_names = ['CD3', 'CD20']
    a.obsm['spatial'] = np.array([[10, 20], [30, 40], [50, 60]], float)
    p = tmp_path / 'x.h5ad'
    a.write_h5ad(p)
    t = read_objects(p, layer_genes=['CD3'])
    assert list(t.classes) == ['T', 'B', 'T'] and np.allclose(t.y, [20, 40, 60]) and np.allclose(t.columns['CD3'], [1, 3, 5])


def test_color_by_encoding():
    L = AnnotationLayer.from_points(np.arange(20).reshape(10, 2) * 10.0, classes=['a', 'b'] * 5)
    v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan])
    L.color_by(v, 'magma', vmin=0, vmax=8, label='score')
    c = L.compiled()
    nc = len(L._class_index)
    assert c.col[0] == nc + 0 and c.col[8] == nc + 255 and c.col[9] == nc + 256  # NaN → last texel
    assert c.col[4] == nc + 128
    ov = L.overview_bytes()
    jl = struct.unpack('<I', ov[4:8])[0]
    meta = json.loads(ov[8:8 + jl])
    assert meta['colormap']['label'] == 'score' and len(meta['colormap']['lut']) == 256
    L.color_by(None)
    assert np.array_equal(L.compiled().col, L.compiled().cls)
    with pytest.raises(ValueError):
        L.color_by(np.arange(3))


def test_from_table_numeric_columns_kept_columnar():
    cols = {'x': np.arange(5.), 'y': np.arange(5.), 'class': list('ababa'), 'CD3': np.arange(5) * 10.}
    t = from_table(cols)
    L = t.to_layer(color_by='CD3', cmap='viridis')
    assert L.properties(3)['CD3'] == 30 and L.colormap['label'] == 'CD3'
