"""Tests for src/horizon_tiles.py on synthetic ground.

A tilted plane has a known horizon in every azimuth (atan of the slope along it); a wall at a known
distance has a known horizon towards it; the witness must agree with the GPU; the .ghz tiles and
their index must say what the maths says.

    E:/swarm/gpu-bench/venv/Scripts/python.exe tests/test_horizon.py
"""
import hashlib, inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import horizon_tiles as hz  # noqa: E402

GPU = [False] + ([True] if hz.cp is not None else [])


def plane(pe, pn, n=129):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return 100.0 + pe * x + pn * y


def test_plane_horizon_is_the_slope_along_each_azimuth():
    pe, pn = 0.05, -0.08
    ux, uy = hz.directions()
    want = np.degrees(np.arctan(pe * ux + pn * uy))
    for gpu in GPU:
        rec, deg, reach = hz.run(plane(pe, pn), use_gpu=gpu, witness_cells=24, cell=8)
        mid = deg[8, 8]                                    # node (64, 64), rays 64 m or more each way
        assert np.abs(mid - want).max() < 1e-9, (gpu, np.abs(mid - want).max())
        assert rec["witness"]["photons"] == 0, rec["witness"]
        assert rec["witness"]["pyramid_mismatch"] == 0
        # pooling reads high far out, but on a gentle plane the channels stay within a degree
        assert rec["photons"] == 0 and rec["positron_high_share"] > 0.5, rec


def test_wall_to_the_north():
    n, H, D = 129, 6.0, 20
    g = np.zeros((n, n)); g[64 + D, :] = H                 # a 6 m wall, one node thick, 20 m north of row 64
    for gpu in GPU:
        rec, deg, reach = hz.run(g, use_gpu=gpu, witness_cells=16, cell=8)
        c = deg[8, 8]                                      # node (64, 64)
        assert abs(c[0] - np.degrees(np.arctan(H / D))) < 1e-9, c[0]
        assert abs(c[16]) < 1e-12                          # due south: flat
        assert rec["witness"]["photons"] == 0
        # the positron sees the same wall (it sits in its 1 m band)
        assert rec["photons"] == 0 or rec["photons_with_near_horizon_lt_32m"] <= rec["photons"]


def test_shaded_follows_the_sun():
    h = np.zeros(hz.NAZ); h[0] = 16.7; h[1] = 10.0
    assert hz.shaded(h, 0.0, 16.0) and not hz.shaded(h, 0.0, 17.0)
    assert not hz.shaded(h, 180.0, 1.0) and hz.shaded(h, 180.0, -1.0)
    # halfway between azimuths 0 and 1 (5.625 degrees): halfway between 16.7 and 10.0
    assert hz.shaded(h, 5.625, 13.3) and not hz.shaded(h, 5.625, 13.4)
    assert hz.shaded(h, 365.625, 13.3)
    assert not hz.shaded(np.full(hz.NAZ, np.nan), 0.0, -5.0)


def test_bands_cover_every_distance_once():
    prev = 1.0
    for k, ds in hz.bands(3000):
        assert ds[0] > prev and ds[0] - prev <= (1 << k)
        assert np.all(np.diff(ds) == (1 << k)) if len(ds) > 1 else True
        prev = ds[-1]
    assert prev >= 3000 - (1 << k)


def test_pyramids_agree():
    g = np.random.default_rng(1).normal(size=(37, 53)); g[5, 7] = np.nan
    a = hz.pyramid_halving(np, g, 6); b = hz.pyramid_blocks(g, 6)
    for x, y in zip(a, b):
        assert x.shape == y.shape and (x == y).all()


def test_quantise_and_nodata():
    deg = np.array([[[12.345] * hz.NAZ]]); reach = np.full((1, 1, hz.NAZ), 500); reach[0, 0, 3] = 99
    q = hz.quantise(deg, reach)
    assert q[0, 0, 0] == 1234 or q[0, 0, 0] == 1235
    assert q[0, 0, 3] == hz.NODATA
    assert hz.quantise(np.full((1, 1, hz.NAZ), np.nan), reach)[0, 0, 0] == hz.NODATA


def test_tiles_index_and_shared_edges(tmp):
    rows = cols = 2 * (hz.NS - 1) + 1
    rng = np.random.default_rng(3)
    q = rng.integers(-500, 3000, size=(rows, cols, hz.NAZ)).astype(np.int16); q[0, 0, 0] = hz.NODATA
    index = hz.write_tiles(str(tmp), q, 399104, 208896, "synthetic")
    raw = open(tmp / hz.INDEX, "rb").read()
    assert b"\r\n" not in raw and json.loads(raw)["tiles"] == index["tiles"]
    assert len(index["tiles"]) == 4
    tiles = {}
    for t in index["tiles"]:
        blob = open(tmp / t["file"], "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"] and len(blob) == t["bytes"]
        head, body = hz.decode(blob)
        assert head["samples"] == hz.NS and head["azimuths"] == hz.NAZ and head["unit"] == hz.UNIT
        assert (head["origin_e_m"], head["origin_n_m"]) == (t["e0"], t["n0"])
        tiles[t["key"]] = body
    assert tiles["0_0"][0, 0, 0] == hz.NODATA and index["tiles"][0]["nodata"] == 1
    assert (tiles["0_0"][:, -1] == tiles["1_0"][:, 0]).all() and (tiles["0_0"][-1] == tiles["0_1"][0]).all()
    assert (tiles["1_1"] == q[64:, 64:]).all()


def test_decode_refuses_damage():
    blob = hz.encode(np.zeros((hz.NS, hz.NS, hz.NAZ), np.int16), 0, 0)
    for bad in (b"XXXX" + blob[4:], blob[:-2], blob[:10]):
        try:
            hz.decode(bad)
        except ValueError:
            continue
        raise AssertionError("damaged tile accepted")


def test_file_rules():
    src = open(hz.__file__, encoding="utf-8").read()
    assert "\r\n" not in src and len(src.splitlines()) < 400 and max(map(len, src.splitlines())) <= 200


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
