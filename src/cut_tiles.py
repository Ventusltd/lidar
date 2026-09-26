"""Cut a south-up 1 m height grid into .ght tiles, and decode them again.

.ght v1 ("GGH1"), all little-endian, 32-byte header then body:

  off  type      field
    0  char[4]   magic "GGH1"
    4  u16       version = 1
    6  u16       samples = 257 (per side; edges shared with neighbours)
    8  u16       spacing_mm = 1000
   10  u16       flags = 0
   12  i32       origin_e_m  (SW corner, EPSG:27700)
   16  i32       origin_n_m
   20  i32       base_cm
   24  u16       min_q       (over valid samples)
   26  u16       max_q
   28  u32       nodata_count
   32  u16[257*257] body, rows SOUTH to NORTH, each row WEST to EAST

height_m = (base_cm + q) / 100; q == 0xFFFF means no data.
"""
import hashlib
import json
import os
import struct
from datetime import datetime, timezone

import numpy as np

MAGIC = b"GGH1"
VERSION = 1
TILE_M = 256
SAMPLES = TILE_M + 1
SPACING_MM = 1000
NODATA_Q = 0xFFFF
HEADER = struct.Struct("<4sHHHHiiiHHI")
HEADER_BYTES = HEADER.size  # 32
BODY_BYTES = SAMPLES * SAMPLES * 2
ATTRIBUTION = ("© Environment Agency copyright and/or database right "
               "2022. All rights reserved.")
assert HEADER_BYTES == 32


def encode_tile(heights, origin_e, origin_n):
    """Encode a (257, 257) south-up float array (NaN = nodata) to bytes.

    Returns (blob, info) with info = dict(min_m, max_m, nodata).
    """
    h = np.asarray(heights, dtype=np.float64)
    if h.shape != (SAMPLES, SAMPLES):
        raise ValueError(f"tile must be {SAMPLES}x{SAMPLES}, got {h.shape}")
    valid = np.isfinite(h)
    nodata = int(h.size - np.count_nonzero(valid))
    q = np.full(h.shape, NODATA_Q, dtype=np.uint16)
    if nodata < h.size:
        cm = np.rint(h[valid] * 100.0).astype(np.int64)
        base_cm = int(cm.min())
        rel = cm - base_cm
        if rel.max() >= NODATA_Q:
            raise ValueError("tile relief exceeds 655.34 m; cannot quantise")
        q[valid] = rel.astype(np.uint16)
        min_q, max_q = 0, int(rel.max())
        min_m = base_cm / 100.0
        max_m = (base_cm + max_q) / 100.0
    else:
        base_cm, min_q, max_q, min_m, max_m = 0, 0, 0, None, None
    head = HEADER.pack(MAGIC, VERSION, SAMPLES, SPACING_MM, 0,
                       int(origin_e), int(origin_n), base_cm,
                       min_q, max_q, nodata)
    blob = head + q.astype("<u2").tobytes()
    return blob, dict(min_m=min_m, max_m=max_m, nodata=nodata)


def decode_tile(blob):
    """Decode .ght bytes -> (header dict, (257, 257) south-up float64, NaN)."""
    if len(blob) < HEADER_BYTES:
        raise ValueError("short .ght blob")
    (magic, ver, n, sp, flags, oe, on, base, mn, mx,
     nd) = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    need = HEADER_BYTES + n * n * 2
    if len(blob) != need:
        raise ValueError(f".ght size {len(blob)} != {need}")
    q = np.frombuffer(blob, dtype="<u2", count=n * n,
                      offset=HEADER_BYTES).reshape(n, n)
    h = (base + q.astype(np.float64)) / 100.0
    h[q == NODATA_Q] = np.nan
    head = dict(version=ver, samples=n, spacing_mm=sp, flags=flags,
                origin_e_m=oe, origin_n_m=on, base_cm=base, min_q=mn,
                max_q=mx, nodata_count=nd)
    return head, h


def cut_grid(grid, origin_e, origin_n, out_dir, site_name, source=""):
    """Cut a south-up grid of shape (k*256+1, m*256+1) into tiles.

    grid[r, c] is the height at node (origin_e + c, origin_n + r).
    Writes <out_dir>/tiles/<key>.ght and <out_dir>/tiles.json; returns the
    tiles.json dict.
    """
    rows, cols = grid.shape
    if (rows - 1) % TILE_M or (cols - 1) % TILE_M or rows < SAMPLES:
        raise ValueError(f"grid {grid.shape} is not k*256+1 per side")
    ny, nx = (rows - 1) // TILE_M, (cols - 1) // TILE_M
    tile_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    entries = []
    for iy in range(ny):
        for ix in range(nx):
            r0, c0 = iy * TILE_M, ix * TILE_M
            sub = grid[r0:r0 + SAMPLES, c0:c0 + SAMPLES]
            e0, n0 = origin_e + c0, origin_n + r0
            blob, info = encode_tile(sub, e0, n0)
            key = f"{ix}_{iy}"
            rel = f"tiles/{key}.ght"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            entries.append(dict(key=key, file=rel,
                                sha256=hashlib.sha256(blob).hexdigest(),
                                e0=int(e0), n0=int(n0), bytes=len(blob),
                                min_m=info["min_m"], max_m=info["max_m"],
                                nodata=info["nodata"]))
    manifest = dict(
        format="ght1", crs="EPSG:27700",
        site=dict(name=site_name, origin_e=int(origin_e),
                  origin_n=int(origin_n)),
        tile_m=TILE_M, spacing_m=1, tiles=entries,
        attribution=ATTRIBUTION, source=source,
        generated_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(out_dir, "tiles.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    return manifest


def load_tiles(out_dir):
    """Read tiles.json and every tile back into one south-up grid (NaN)."""
    with open(os.path.join(out_dir, "tiles.json"), encoding="utf-8") as f:
        man = json.load(f)
    oe, on = man["site"]["origin_e"], man["site"]["origin_n"]
    t = man["tile_m"]
    e_max = max(x["e0"] for x in man["tiles"]) + t
    n_max = max(x["n0"] for x in man["tiles"]) + t
    grid = np.full((n_max - on + 1, e_max - oe + 1), np.nan)
    for x in man["tiles"]:
        with open(os.path.join(out_dir, x["file"]), "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != x["sha256"]:
            raise ValueError(f"sha256 mismatch for {x['file']}")
        head, h = decode_tile(blob)
        r0, c0 = head["origin_n_m"] - on, head["origin_e_m"] - oe
        grid[r0:r0 + SAMPLES, c0:c0 + SAMPLES] = h
    return man, grid
