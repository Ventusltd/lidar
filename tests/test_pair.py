"""Tests for src/pair_gpu.py against a synthetic hilly grid.

A clean set of tiles must annihilate (zero photons); each kind of damage must show up as
photons in the right channel and never be suppressed.

    E:/swarm/gpu-bench/venv/Scripts/python.exe -m pytest lidar/tests/test_pair.py -q
"""
import hashlib, inspect, json, os, pathlib, struct, sys, tempfile, traceback
import numpy as np

try:
    import pytest
except ImportError:  # no pytest in the GPU venv: run this file directly, see the bottom
    pytest = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pair_gpu as pg  # noqa: E402

GPU = [False] + ([True] if pg.cp is not None else [])
PTS = 200_000


def make_tiles(tmp_path):
    pg.synth(str(tmp_path), tiles_e=2, tiles_n=2)
    return str(tmp_path)


if pytest:
    tiles = pytest.fixture(make_tiles, name="tiles")


def run(d, gpu=False, **kw):
    return pg.run(d, os.path.join(d, "source.npy"), points=kw.pop("points", PTS), batch=1 << 16, use_gpu=gpu, **kw)


def poke(path, row, col, delta_q=0, header=None):
    raw = bytearray(open(path, "rb").read())
    off = len(raw) - pg.N * pg.N * 2 + 2 * (row * pg.N + col)
    q, = struct.unpack_from("<H", raw, off)
    struct.pack_into("<H", raw, off, q + delta_q)
    if header:
        struct.pack_into("<I", raw, 28, header)
    open(path, "wb").write(bytes(raw))


def test_header_layout(tiles):
    t = pg.read_ght(os.path.join(tiles, "t_400000_300000.ght"))
    assert (t["samples"], t["spacing_mm"], t["oe"], t["on"], t["header_bytes"]) == (257, 1000, 400000, 300000, 32)
    assert t["q"].shape == (257, 257) and t["nodata"] == 0


def test_clean_tiles_annihilate(tiles):
    for gpu in GPU:
        check_clean(run(tiles, gpu))


def check_clean(r):
    assert r["photons_total"] == 0
    assert r["electron"]["samples"] == 4 * 257 * 257
    assert r["electron"]["max_err_m"] <= pg.TOL_Q + pg.SLACK
    assert r["positron"]["points"] == PTS and r["positron"]["pair_max_m"] < 1e-9
    assert r["positron"]["src_max_m"] <= pg.TOL_Q + pg.SLACK
    assert r["edges"]["pairs"] == 4 and r["edges"]["photons"] == 0
    assert r["witness"]["points"] == pg.WITNESS and r["witness"]["photons"] == 0
    assert len(r["check_points"]) == 16


def test_gpu_and_cpu_agree(tiles):
    if pg.cp is None:
        return                                        # no CuPy device: nothing to compare
    a, b = run(tiles, False), run(tiles, True)
    for k in ("pair_photons", "src_photons", "nodata_cells"):
        assert a["positron"][k] == b["positron"][k]
    assert abs(a["positron"]["src_max_m"] - b["positron"]["src_max_m"]) < 1e-12
    assert a["check_points"] == b["check_points"]


def test_seeded_repeatable(tiles):
    a, b, c = run(tiles), run(tiles), run(tiles, seed=7)
    assert a["check_points"] == b["check_points"] and a["positron"] == b["positron"]
    assert a["check_points"] != c["check_points"]


def test_check_points_retest(tiles):
    """What the browser does: find the tile, two lerps, compare."""
    r = run(tiles)
    for p in r["check_points"]:
        t = pg.read_ght(os.path.join(tiles, p["tile"]))
        h = (t["base"] + t["q"].astype(np.int64)) / 100.0
        u, v = (p["x"] - t["oe"]) / (t["spacing_mm"] / 1000), (p["y"] - t["on"]) / (t["spacing_mm"] / 1000)
        j, i = min(int(u), 255), min(int(v), 255)
        fx, fy = u - j, v - i
        a = h[i, j] + (h[i, j + 1] - h[i, j]) * fx
        b = h[i + 1, j] + (h[i + 1, j + 1] - h[i + 1, j]) * fx
        assert abs(a + (b - a) * fy - p["h"]) < 1e-6


def test_bad_sample_is_a_photon(tiles):
    poke(os.path.join(tiles, "t_400000_300000.ght"), 100, 100, delta_q=2)
    r = run(tiles)
    assert r["electron"]["photons"] == 1
    w = r["electron"]["worst"][0]
    assert (w["x"], w["y"]) == (400100.0, 300100.0) and w["err_m"] > 0.01
    assert r["positron"]["src_photons"] > 0          # interpolated near it, never suppressed
    assert r["positron"]["pair_photons"] == 0        # the two formulas still agree with each other
    assert r["photons_total"] > 0


def test_edge_mismatch_is_a_photon(tiles):
    poke(os.path.join(tiles, "t_400000_300000.ght"), 50, 256, delta_q=1)   # east column
    r = run(tiles)
    assert r["edges"]["photons"] == 1
    c = r["edges"]["cases"][0]
    assert c["side"] == "east" and c["sample"] == 50 and c["a_cm"] - c["b_cm"] == 1


def test_header_lie_is_a_photon(tiles):
    poke(os.path.join(tiles, "t_400256_300256.ght"), 0, 0, header=5)
    r = run(tiles)
    assert r["header"]["photons"] == 1 and r["header"]["cases"][0]["header"][2] == 5


def test_nodata_counted_not_hidden(tiles):
    p = os.path.join(tiles, "t_400000_300000.ght")
    raw = bytearray(open(p, "rb").read())
    struct.pack_into("<H", raw, len(raw) - pg.N * pg.N * 2 + 2 * (10 * pg.N + 10), pg.NODATA)
    struct.pack_into("<I", raw, 28, 1)
    open(p, "wb").write(bytes(raw))
    r = run(tiles)
    assert r["electron"]["nodata_mismatch"] == 1 and r["electron"]["photons"] == 1
    assert r["positron"]["nodata_cells"] > 0


def test_receipt_and_tiles_json(tiles):
    r = run(tiles)
    path, sha = pg.write_receipt(tiles, r)
    assert os.path.dirname(path) == tiles
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == sha
    assert json.load(open(os.path.join(tiles, "tiles.json")))["receipt"] == sha
    rec = json.load(open(path))
    assert rec["seed"] == pg.SEED and len(rec["script_sha256"]) == 64 and rec["device"]
    assert all(set(p) >= {"x", "y", "h"} for p in rec["check_points"])


def test_bounded_by_seconds(tiles):
    r = run(tiles, points=1 << 30, seconds=0.0)
    assert r["positron"]["stopped_by"] == "seconds"


def test_main_synth(tmp_path):
    assert pg.main(["--tiles", str(tmp_path), "--synth", "--points", "65536", "--cpu"]) == 0
    assert os.path.exists(tmp_path / pg.RECEIPT)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        with tempfile.TemporaryDirectory() as d:
            arg = make_tiles(pathlib.Path(d)) if "tiles" in inspect.signature(fn).parameters else pathlib.Path(d)
            try:
                fn(arg)
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name); traceback.print_exc()
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
