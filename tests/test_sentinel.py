# SPDX-License-Identifier: Apache-2.0
"""Tests for src/sentinel_site.py and src/s2_fetch.py. No network: a local tiled TIFF stands in for a COG.

    E:/swarm/gpu-bench/venv/Scripts/python.exe tests/test_sentinel.py
"""
import datetime as dt, inspect, io, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import s2_fetch  # noqa: E402
import sentinel_site as ss  # noqa: E402

XPS = [np] + ([ss.cp] if ss.cp is not None else [])


def test_last_seasons():
    s = ss.last_seasons(dt.date(2026, 9, 26))
    assert [x[0] for x in s] == ["autumn-2025", "winter-2025", "spring-2026", "summer-2026"], s
    assert s[1][1] == dt.date(2025, 12, 1) and s[1][2] == dt.date(2026, 2, 28)
    assert s[-1][2] == dt.date(2026, 8, 31)
    j = ss.last_seasons(dt.date(2026, 1, 5))           # January sits in winter-2025: last complete is autumn
    assert j[-1][0] == "autumn-2025" and j[-1][2] == dt.date(2025, 11, 30), j


def test_classify_and_mask():
    scl = np.array([0, 1, 3, 4, 4, 5, 8, 9, 10, 11, -1, 6])
    c = ss.classify(scl)
    assert c["box_nodata"] == round(3 / 12, 4)          # 0, 1 and outside
    assert c["box_cloud"] == round(4 / 9, 4)            # 3, 8, 9, 10 over nine valid
    assert c["box_snow"] == round(1 / 9, 4)
    assert ss.clear_mask(scl).tolist() == [False, False, False, True, True, True, False, False, False, True, False, True]


def test_reflectance_offset():
    assert np.allclose(ss.reflectance([1000, 2000], "05.11"), [0.0, 0.1])
    assert np.allclose(ss.reflectance([1000, 2000], "02.14"), [0.1, 0.2])


def test_median_and_central_mean():
    for xp in XPS:
        rng = np.random.default_rng(1)
        for n in range(1, 25):
            v = rng.integers(0, 50, (n, 64)).astype(np.float32)       # many ties on purpose
            m = np.ones_like(v, bool)
            e = ss.to_host(ss.median_electron(xp.asarray(v), xp.asarray(m), xp))
            assert np.allclose(e, np.median(v, axis=0)), n
            p = ss.to_host(ss.central_positron(xp.asarray(v), xp.asarray(m), xp))
            if n <= 10:                                                 # band holds at most the middle ranks
                assert np.allclose(p, e), n
            s = np.sort(v, 0)                                           # sorted-position trimmed mean
            lo, hi = np.ceil(0.4 * (n - 1) - 1e-9), np.floor(0.6 * (n - 1) + 1e-9)
            if lo > hi:
                lo, hi = np.floor((n - 1) / 2), np.ceil((n - 1) / 2)
            assert np.allclose(p, s[int(lo):int(hi) + 1].mean(0)), n
        v = np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
        p = ss.to_host(ss.central_positron(xp.asarray(v), xp.asarray(np.ones_like(v, bool)), xp))
        assert np.allclose(p, np.mean([8, 9, 10, 11])), p                 # ranks 7.6..11.4 -> 8..11


def test_mask_and_empty():
    for xp in XPS:
        v = np.array([[1, 5, 9], [100, 6, 9], [3, 7, 9]], np.float32)
        m = np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], bool)
        e = ss.to_host(ss.median_electron(xp.asarray(v), xp.asarray(m), xp))
        p = ss.to_host(ss.central_positron(xp.asarray(v), xp.asarray(m), xp))
        assert np.allclose(e[:2], [2, 6]) and np.isnan(e[2])            # the masked 100 never counts
        assert np.allclose(p[:2], [2, 6]) and np.isnan(p[2])


def test_photons_and_witness():
    rng = np.random.default_rng(7)
    n, P = 30, 500
    stack = rng.normal(0.1, 0.002, (n, 4, P)).astype(np.float32)
    stack[:, :, :50] = rng.uniform(0, 0.6, (n, 4, 50)).astype(np.float32)  # wide spread: estimators part
    mask = rng.random((n, P)) > 0.2
    for xp in XPS:
        e, p, ph, _, _ = ss.composite(stack, mask, xp)
        ph = ss.to_host(ph)
        assert ph[:, 50:].sum() == 0 and ph[:, :50].sum() > 0, ph.sum()
        w = ss.witness(stack, mask, e, p)
        assert w["pass"] and w["pixels"] == P, w


def test_images(tmp_path):
    tmp = tmp_path
    n = 5
    bgrn = np.full((4, n * n), 0.1, np.float32)
    bgrn[:, 0] = np.nan
    img = ss.true_colour(bgrn, n)
    assert img.shape == (5, 5, 4) and img[0, 0, 3] == 0 and img[0, 1, 3] == 255
    assert img[0, 1, 0] == int((0.4 ** (1 / ss.GAMMA)) * 255 + 0.5)
    r = ss.save(img, str(tmp / "a.webp"), "WEBP")
    assert r["bytes"] == os.path.getsize(tmp / "a.webp") and len(r["sha256"]) == 64
    g = ss.index_grey(np.array([0.3]), np.array([0.1]), 1)
    assert g[0, 0, 0] == int((0.5 + 1) * 127.5 + 0.5)
    ss.write_json(str(tmp / "i.json"), {"a": 1})
    assert b"\r" not in (tmp / "i.json").read_bytes()


def test_box_grid():
    g = s2_fetch.BoxGrid(399104, 208896, 2048, 205)
    assert g.e.shape == (205, 205) and abs(g.px - 2048 / 205) < 1e-12
    assert abs(g.e[0, 0] - (399104 + g.px / 2)) < 1e-9 and abs(g.nn[0, 0] - (208896 + 2048 - g.px / 2)) < 1e-9
    x, y = g.utm(32630)
    assert 5.7e6 < y.min() < y.max() < 5.8e6 and 5.6e5 < x.min() < 6.1e5
    dx = x[0, -1] - x[0, 0]                               # BNG and UTM 30N are nearly parallel here
    assert abs(dx - 204 * g.px) < 5, dx
    lon0, lat0, lon1, lat1 = g.lonlat_bbox()
    assert -2.1 < lon0 < lon1 < -1.9 and 51.7 < lat0 < lat1 < 51.9


def test_window_tiles():
    t = s2_fetch.window_tiles(1000, 1030, 2040, 2050, 1024, 1024, 11)
    assert [k for _, _, k in t] == [1, 2, 12, 13]


class LocalRange(s2_fetch.RangeFile):
    def __init__(self, data):
        self.url, self.pos, self.blocks, self.fetched, self.data, self.size = "mem", 0, {}, 0, data, len(data)

    def get(self, a, n):
        self.fetched += n
        return self.data[a:a + n]


def test_read_window_from_tiled_tiff():
    import tifffile
    H = W = 300
    arr = (np.arange(H)[:, None] * 1000 + np.arange(W)[None, :]).astype(np.uint32)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, tile=(64, 64), compression="zlib", predictor=True)
    rf = LocalRange(buf.getvalue())
    tr = [10.0, 0, 500000.0, 0, -10.0, 5800000.0]
    rows, cols = np.array([[0, 70], [299, 5]]), np.array([[0, 130], [299, 64]])
    x, y = 500000 + cols * 10 + 5, 5800000 - rows * 10 - 5
    v, n = s2_fetch.read_window("mem", tr, np.append(x, 400000), np.append(y, 0), fh=rf)
    assert v[:-1].tolist() == (rows * 1000 + cols).ravel().tolist() and v[-1] == -1, v
    assert n > 0
    big = io.BytesIO()                                     # one corner pixel touches one tile only
    tifffile.imwrite(big, np.random.default_rng(0).integers(0, 1 << 30, (1024, 1024), dtype=np.uint32),
                     tile=(256, 256), compression="zlib")
    rf2 = LocalRange(big.getvalue())
    v2, _ = s2_fetch.read_window("mem", tr, np.array([500005.0]), np.array([5799995.0]), fh=rf2)
    assert rf2.fetched < len(big.getvalue()) / 8, (rf2.fetched, len(big.getvalue()))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(*([pathlib.Path(d)] if inspect.signature(fn).parameters else []))
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name); traceback.print_exc()
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
