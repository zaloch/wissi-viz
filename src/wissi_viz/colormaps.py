"""Colormaps without matplotlib. Control points sampled from the originals:
viridis/magma/inferno/plasma/cividis (CC0, van der Walt & Smith / Nuñez et al.),
turbo (Apache-2.0, Google), coolwarm (Moreland, public domain), plus simple ramps."""

from __future__ import annotations

import colorsys

import numpy as np

_CP = {
    'viridis': [
        '#440154',
        '#482878',
        '#3e4989',
        '#31688e',
        '#26828e',
        '#1f9e89',
        '#35b779',
        '#6ece58',
        '#b5de2b',
        '#fde725',
    ],
    'magma': [
        '#000004',
        '#180f3d',
        '#440f76',
        '#721f81',
        '#9e2f7f',
        '#cd4071',
        '#f1605d',
        '#fd9668',
        '#feca8d',
        '#fcfdbf',
    ],
    'inferno': [
        '#000004',
        '#1b0c41',
        '#4a0c6b',
        '#781c6d',
        '#a52c60',
        '#cf4446',
        '#ed6925',
        '#fb9b06',
        '#f7d13d',
        '#fcffa4',
    ],
    'plasma': [
        '#0d0887',
        '#46039f',
        '#7201a8',
        '#9c179e',
        '#bd3786',
        '#d8576b',
        '#ed7953',
        '#fb9f3a',
        '#fdca26',
        '#f0f921',
    ],
    'cividis': [
        '#00224e',
        '#123570',
        '#3b496c',
        '#575d6d',
        '#707173',
        '#8a8678',
        '#a59c74',
        '#c3b369',
        '#e1cc55',
        '#fee838',
    ],
    'turbo': [
        '#30123b',
        '#4662d7',
        '#36aaf9',
        '#1ae4b6',
        '#72fe5e',
        '#c8ef34',
        '#faba39',
        '#f66b19',
        '#ca2a04',
        '#7a0403',
    ],
    'coolwarm': [
        '#3b4cc0',
        '#5977e3',
        '#7b9ff9',
        '#9ebeff',
        '#c0d4f5',
        '#dddcdc',
        '#f2cbb7',
        '#f7ac8e',
        '#ee8468',
        '#d65244',
        '#b40426',
    ],
    'rdbu': [
        '#053061',
        '#2166ac',
        '#4393c3',
        '#92c5de',
        '#d1e5f0',
        '#f7f7f7',
        '#fddbc7',
        '#f4a582',
        '#d6604d',
        '#b2182b',
        '#67001f',
    ],
    'gray': ['#000000', '#ffffff'],
    'reds': ['#fff5f0', '#fcbba1', '#fb6a4a', '#cb181d', '#67000d'],
    'blues': ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
    'hot': ['#0b0000', '#ff0000', '#ffff00', '#ffffff'],
}
DIVERGING = {'coolwarm', 'rdbu'}

#: categorical palette (Okabe-Ito first: colour-blind safe), then extras
CATEGORICAL = [
    '#e69f00',
    '#56b4e9',
    '#009e73',
    '#f0e442',
    '#0072b2',
    '#d55e00',
    '#cc79a7',
    '#999999',
    '#00e5ff',
    '#ff3d71',
    '#76ff03',
    '#d500f9',
    '#ffd600',
    '#651fff',
    '#1de9b6',
    '#ff9100',
]


def _hex(c: str) -> np.ndarray:
    c = c.lstrip('#')
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], np.float32)


def names() -> list[str]:
    return sorted(_CP)


def lut(name: str = 'viridis', n: int = 256) -> list[str]:
    """``n`` hex colours, linearly interpolated between the control points."""
    key = name.lower().removesuffix('_r')
    if key not in _CP:
        raise ValueError(f'unknown colormap {name!r}; available: {names()}')
    cp = np.stack([_hex(c) for c in _CP[key]])
    t = np.linspace(0, 1, n)
    xs = np.linspace(0, 1, len(cp))
    rgb = np.stack([np.interp(t, xs, cp[:, k]) for k in range(3)], 1)
    if name.lower().endswith('_r'):
        rgb = rgb[::-1]
    return ['#{:02x}{:02x}{:02x}'.format(*tuple(int(round(v)) for v in c)) for c in rgb]


def categorical(i: int) -> str:
    if i < len(CATEGORICAL):
        return CATEGORICAL[i]
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.7, 1.0)
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
