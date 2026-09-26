"""Synthetic grid -> .ght tiles -> decode -> heights within 5 mm.

Run: python tests/test_roundtrip.py   (or pytest tests/)
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cut_tiles import (cut_grid, decode_tile, encode_tile, load_tiles,  # noqa
                       HEADER_BYTES, SAMPLES, NODATA_Q)
from geotiff_read import mask_nodata  # noqa: E402

TOL_M = 0.005 + 1e-9  # 1 cm quantisation: worst case is exactly 5 mm


def synthetic(size=512, seed=7):
    """Rolling hills with a valley, fine noise and a few nodata holes."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size + 1, 0:size + 1].astype(np.float64)
    z = (120.0 + 25.0 * np.sin(x / 90.0) * np.cos(y / 70.0)
         - 18.0 * np.exp(-((x - size / 2) ** 2) / (2 * 40.0 ** 2))
         + 0.004 * x + rng.normal(0, 0.03, x.shape))
    z[10:14, 300:305] = np.nan
    z[400, 0:50] = -3.4028234663852886e38  # EA float32 nodata
    z[200, 200] = np.inf
    return mask_nodata(z, -3.4028234663852886e38)


def test_header_layout():
    h = np.full((SAMPLES, SAMPLES), 42.0)
    h[0, 0] = np.nan
    blob, info = encode_tile(h, 400000, 210000)
    assert HEADER_BYTES == 32
    assert len(blob) == 32 + SAMPLES * SAMPLES * 2
    assert blob[:4] == b"GGH1"
    ver, n, sp, fl = struct.unpack_from("<HHHH", blob, 4)
    assert (ver, n, sp, fl) == (1, 257, 1000, 0)
    oe, on, base = struct.unpack_from("<iii", blob, 12)
    assert (oe, on, base) == (400000, 210000, 4200)
    mn, mx, nd = struct.unpack_from("<HHI", blob, 24)
    assert (mn, mx, nd) == (0, 0, 1) and info["nodata"] == 1
    # first body sample is SW corner and is nodata
    assert struct.unpack_from("<H", blob, 32)[0] == NODATA_Q


def test_row_order_south_to_north():
    h = np.zeros((SAMPLES, SAMPLES))
    h[0, :] = 1.0      # south row
    h[:, -1] = 2.0     # east column
    blob, _ = encode_tile(h, 0, 0)
    q = np.frombuffer(blob, "<u2", offset=32).reshape(SAMPLES, SAMPLES)
    assert q[0, 0] == 100 and q[1, 0] == 0 and q[5, -1] == 200


def test_roundtrip_within_5mm():
    grid = synthetic(512)
    with tempfile.TemporaryDirectory() as d:
        man = cut_grid(grid, 400000, 210000, d, "synthetic", source="test")
        assert len(man["tiles"]) == 4
        with open(os.path.join(d, "tiles.json"), encoding="utf-8") as f:
            disk = json.load(f)
        assert disk["format"] == "ght1" and disk["crs"] == "EPSG:27700"
        assert disk["tile_m"] == 256 and disk["spacing_m"] == 1
        for t in disk["tiles"]:
            assert os.path.getsize(os.path.join(d, t["file"])) == t["bytes"]
            with open(os.path.join(d, t["file"]), "rb") as f:
                head, h = decode_tile(f.read())
            assert (head["origin_e_m"], head["origin_n_m"]) == (t["e0"],
                                                                t["n0"])
        _, back = load_tiles(d)
    ok = np.isfinite(grid)
    assert np.array_equal(ok, np.isfinite(back)), "nodata mask changed"
    err = np.abs(back[ok] - grid[ok])
    assert err.max() <= TOL_M, f"max error {err.max() * 1000:.3f} mm"
    return err.max(), int((~ok).sum()), len(man["tiles"])


def test_shared_edges_match():
    grid = synthetic(512, seed=3)
    with tempfile.TemporaryDirectory() as d:
        man = cut_grid(grid, 0, 0, d, "edges")
        tiles = {}
        for t in man["tiles"]:
            with open(os.path.join(d, t["file"]), "rb") as f:
                tiles[t["key"]] = decode_tile(f.read())[1]
    a, b = tiles["0_0"], tiles["1_0"]
    assert np.allclose(a[:, -1], b[:, 0], equal_nan=True, atol=TOL_M)
    a, b = tiles["0_0"], tiles["0_1"]
    assert np.allclose(a[-1, :], b[0, :], equal_nan=True, atol=TOL_M)


if __name__ == "__main__":
    test_header_layout()
    test_row_order_south_to_north()
    test_shared_edges_match()
    mx, holes, n = test_roundtrip_within_5mm()
    print(f"PASS: {n} tiles, max error {mx * 1000:.3f} mm, "
          f"{holes} nodata samples preserved")
