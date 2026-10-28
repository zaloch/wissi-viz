"""Minimal CZI writer for tests (uncompressed subblocks, optional pyramid subblocks).

Written from the published CZI file-format spec (ZISRAW segments). Lets the tests build
mosaics with overlapping tiles, pyramid levels (stored size < logical size) and RGB,
without the LGPL libCZI writer.
"""

from __future__ import annotations

import struct

import numpy as np

PIXEL = {(np.dtype('uint8'), 1): 0, (np.dtype('uint16'), 1): 1, (np.dtype('uint8'), 3): 3}


def _seg(sid: str, data: bytes) -> bytes:
    pad = (-len(data)) % 32
    data = data + b'\0' * pad
    return sid.encode().ljust(16, b'\0') + struct.pack('<qq', len(data), len(data)) + data


def _dir_entry(ptype, pos, dims, pyramid, compression=0):
    out = b'DV' + struct.pack('<iqii', ptype, pos, 0, compression) + struct.pack('<B', pyramid) + b'\0' * 5
    out += struct.pack('<i', len(dims))
    for name, start, size, stored in dims:
        out += name.encode().ljust(4, b'\0') + struct.pack('<iifi', start, size, float(start), stored)
    return out


def write_czi(path, blocks, channel_names=None, mpp=None, compression=None):
    """blocks: list of dict(c, x, y, data(ndarray stored), w, h (logical), m (mosaic index), pyramid(bool))."""
    body = bytearray()
    entries = []
    base = 32 + 512
    for b in blocks:
        a = np.ascontiguousarray(b['data'])
        spp = a.shape[2] if a.ndim == 3 else 1
        if spp == 3:
            a = np.ascontiguousarray(a[..., ::-1])  # RGB → BGR
        ptype = PIXEL[(a.dtype, spp)]
        sh, sw = a.shape[:2]
        dims = [('X', b['x'], b['w'], sw), ('Y', b['y'], b['h'], sh), ('C', b['c'], 1, 1), ('M', b.get('m', 0), 1, 1)]
        if 's' in b:  # scene
            dims.append(('S', b['s'], 1, 1))
        pos = base + len(body)
        de = _dir_entry(ptype, pos, dims, 2 if b.get('pyramid') else 0, 5 if compression == 'zstd' else 0)
        payload = a.tobytes()
        if compression == 'zstd':
            import imagecodecs

            payload = imagecodecs.zstd_encode(payload, level=1)
        head = struct.pack('<iiq', 0, 0, len(payload)) + de
        head = head.ljust(max(256, len(head)), b'\0')
        body += _seg('ZISRAWSUBBLOCK', head + payload)
        entries.append(de)
    dir_pos = base + len(body)
    body += _seg('ZISRAWDIRECTORY', struct.pack('<i', len(entries)) + b'\0' * 124 + b''.join(entries))
    meta_pos = base + len(body)
    chans = ''.join(f'<Channel Id="Channel:{i}" Name="{n}"/>' for i, n in enumerate(channel_names or []))
    scal = (
        (
            f'<Scaling><Items><Distance Id="X"><Value>{mpp * 1e-6}</Value></Distance>'
            f'<Distance Id="Y"><Value>{mpp * 1e-6}</Value></Distance></Items></Scaling>'
        )
        if mpp
        else ''
    )
    xml = (
        f'<ImageDocument><Metadata><Information><Image><Dimensions><Channels>{chans}</Channels>'
        f'</Dimensions></Image></Information>{scal}</Metadata></ImageDocument>'
    ).encode()
    body += _seg('ZISRAWMETADATA', struct.pack('<ii', len(xml), 0) + b'\0' * 248 + xml)
    hdr = (
        struct.pack('<iiii', 1, 0, 0, 0)
        + b'\x11' * 16
        + b'\x11' * 16
        + struct.pack('<iqqiq', 0, dir_pos, meta_pos, 0, 0)
    )
    with open(path, 'wb') as f:
        f.write(_seg('ZISRAWFILE', hdr.ljust(512, b'\0')))
        f.write(body)


def mosaic_with_pyramid(path, img: np.ndarray, tile=300, overlap=20, levels=(1, 2, 4), channel_names=None, mpp=0.5):
    """Write ``img`` (H, W) or (H, W, C=1) per channel list, as an overlapping mosaic + pyramid subblocks.
    ``img``: (C, H, W) uint16 or (H, W, 3) uint8 RGB. Returns the list of blocks written."""
    rgb = img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    chans = [img] if rgb else list(img)
    H, W = chans[0].shape[:2]
    blocks, m = [], 0
    for c, plane in enumerate(chans):
        for s in levels:
            step = tile * s
            for y in range(0, H, step - overlap * s if s == 1 else step):
                for x in range(0, W, step - overlap * s if s == 1 else step):
                    w, h = min(step, W - x), min(step, H - y)
                    w -= w % s
                    h -= h % s
                    if w <= 0 or h <= 0:
                        continue
                    region = plane[y : y + h, x : x + w]
                    if s > 1:
                        sh = (h // s, s, w // s, s) + region.shape[2:]
                        region = region.reshape(sh).mean((1, 3)).astype(plane.dtype)
                    blocks.append({'c': c, 'x': x, 'y': y, 'w': w, 'h': h, 'data': region, 'm': m, 'pyramid': s > 1})
                    m += 1
    write_czi(path, blocks, channel_names, mpp)
    return blocks


def write_czi_stream(path, block_iter, channel_names=None, mpp=None, compression='zstd'):
    """Like write_czi but consumes an iterator of blocks and writes each subblock immediately."""
    import imagecodecs

    entries = []
    with open(path, 'wb') as f:
        f.write(b'\0' * (32 + 512))  # header placeholder
        for b in block_iter:
            a = np.ascontiguousarray(b['data'])
            spp = a.shape[2] if a.ndim == 3 else 1
            if spp == 3:
                a = np.ascontiguousarray(a[..., ::-1])
            ptype = PIXEL[(a.dtype, spp)]
            sh, sw = a.shape[:2]
            dims = [
                ('X', b['x'], b['w'], sw),
                ('Y', b['y'], b['h'], sh),
                ('C', b['c'], 1, 1),
                ('M', b.get('m', 0), 1, 1),
            ]
            pos = f.tell()
            de = _dir_entry(ptype, pos, dims, 2 if b.get('pyramid') else 0, 5 if compression == 'zstd' else 0)
            payload = imagecodecs.zstd_encode(a.tobytes(), level=1) if compression == 'zstd' else a.tobytes()
            head = struct.pack('<iiq', 0, 0, len(payload)) + de
            f.write(_seg('ZISRAWSUBBLOCK', head.ljust(max(256, len(head)), b'\0') + payload))
            entries.append(de)
        dir_pos = f.tell()
        f.write(_seg('ZISRAWDIRECTORY', struct.pack('<i', len(entries)) + b'\0' * 124 + b''.join(entries)))
        meta_pos = f.tell()
        chans = ''.join(f'<Channel Id="Channel:{i}" Name="{n}"/>' for i, n in enumerate(channel_names or []))
        scal = (
            (
                f'<Scaling><Items><Distance Id="X"><Value>{mpp * 1e-6}</Value></Distance>'
                f'<Distance Id="Y"><Value>{mpp * 1e-6}</Value></Distance></Items></Scaling>'
            )
            if mpp
            else ''
        )
        xml = (
            f'<ImageDocument><Metadata><Information><Image><Dimensions><Channels>{chans}</Channels>'
            f'</Dimensions></Image></Information>{scal}</Metadata></ImageDocument>'
        ).encode()
        f.write(_seg('ZISRAWMETADATA', struct.pack('<ii', len(xml), 0) + b'\0' * 248 + xml))
        hdr = struct.pack('<iiii', 1, 0, 0, 0) + b'\x11' * 32 + struct.pack('<iqqiq', 0, dir_pos, meta_pos, 0, 0)
        f.seek(0)
        f.write(_seg('ZISRAWFILE', hdr.ljust(512, b'\0')))
