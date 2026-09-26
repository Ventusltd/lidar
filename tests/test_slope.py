"""Tests for src/slope_tiles.py on synthetic ground.

A plane must annihilate (Horn and Zevenbergen-Thorne agree exactly, zero photons); a crease
(break of slope) must make photons on the crease and nowhere else; classes, aspect octants and
the .gst tiles must say what the maths says.

    python lidar/tests/test_slope.py
"""
import hashlib, inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import slope_tiles as st  # noqa: E402

GPU = [False] + ([True] if st.cp is not None else [])


def plane(pe, pn, n=65):
    """South-up grid with dz/de = pe, dz/dn = pn (1 m spacing)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return 100.0 + pe * x + pn * y


def test_plane_has_no_photons_and_right_slope():
    for gpu in GPU:
        rec, cls, asp, ph = st.run(plane(0.072, 0.096), use_gpu=gpu)   # 12 %
        assert rec["photons"] == 0 and rec["max_diff_pp"] < 1e-9
        assert abs(rec["electron"]["mean_pct"] - 12.0) < 1e-9
        assert rec["witness"]["photons"] == 0
        inner = cls[1:-1, 1:-1]
        assert (inner == 3).all()                                     # 12 % falls in 10-15
        assert (cls[0] == st.NODATA).all() and (cls[:, -1] == st.NODATA).all()


def test_class_edges():
    for s_pct, k in [(0.0, 0), (1.99, 0), (2.001, 1), (4.9, 1), (5.001, 2), (9.99, 2), (15.001, 4), (24.9, 4), (25.001, 5), (80, 5)]:
        _, cls, _, _ = st.run(plane(s_pct / 100, 0.0, 9), use_gpu=False, witness_on=False)
        assert cls[4, 4] == k, (s_pct, cls[4, 4], k)


def test_aspect_octants_point_downslope():
    # (dz/de, dz/dn) -> downslope octant. Rising to the east means the ground falls to the WEST (6).
    cases = [((0, -0.1), 0), ((-0.1, -0.1), 1), ((-0.1, 0), 2), ((-0.1, 0.1), 3),
             ((0, 0.1), 4), ((0.1, 0.1), 5), ((0.1, 0), 6), ((0.1, -0.1), 7), ((0.001, 0), st.FLAT)]
    for gpu in GPU:
        for (pe, pn), o in cases:
            _, _, asp, _ = st.run(plane(pe, pn, 9), use_gpu=gpu, witness_on=False)
            assert asp[4, 4] == o, ((pe, pn), asp[4, 4], o)
            assert asp[0, 0] == st.NODATA


def test_crease_makes_photons_beside_the_crease_only():
    n = 65
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    # flat to the south-west of the diagonal x + y = 64, a 20 % (per axis) bank rising beyond it.
    # Horn smooths across the crease and ZT does not: they differ on the nodes either side of it.
    g = 50.0 + 0.2 * np.maximum(x + y - 64, 0)
    for gpu in GPU:
        rec, _, _, ph = st.run(g, use_gpu=gpu)
        r, c = np.nonzero(ph)
        band = set(((r + 1) + (c + 1) - 64).tolist())
        assert rec["photons"] > 0 and band == {-1, 1}, band
        assert rec["breaks"]["share_on_breaks"] == 1.0 and rec["breaks"]["by_chance"] < 0.1
        assert rec["witness"]["photons"] == 0 and rec["witness"]["photons_cpu"] == rec["photons"]


def test_nan_marks_neighbours_nodata():
    g = plane(0.05, 0.0)
    g[20, 20] = np.nan
    _, cls, asp, _ = st.run(g, use_gpu=False)
    assert cls[20, 21] == st.NODATA and cls[19, 19] == st.NODATA and asp[21, 21] == st.NODATA
    assert cls[20, 23] != st.NODATA


def test_tiles_roundtrip_and_index(tmp):
    n = 2 * st.TILE_M + 1
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    g = 100 + 10 * np.sin(x / 40.0) * np.cos(y / 55.0)
    rec, cls, asp, _ = st.run(g, 400000, 300000, use_gpu=False, witness_on=False)
    idx = st.write_tiles(str(tmp), cls, asp, 400000, 300000, "synth")
    assert len(idx["tiles"]) == 4 and idx["tile_m"] == 256
    again = json.load(open(tmp / st.INDEX))
    assert again["tiles"] == idx["tiles"]
    for t in idx["tiles"]:
        blob = open(tmp / t["file"], "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"] and len(blob) == t["bytes"]
        head, c, a = st.decode(blob)
        r0, c0 = t["n0"] - 300000, t["e0"] - 400000
        assert (c == cls[r0:r0 + st.N, c0:c0 + st.N]).all() and (a == asp[r0:r0 + st.N, c0:c0 + st.N]).all()
        assert head["steep_count"] == t["steep"] == int(((c >= 3) & (c != 255)).sum())
        assert abs(sum(t["share"]) - 1) < 1e-4
    # shared edges agree between neighbours
    a = st.decode(open(tmp / "tiles/0_0.gst", "rb").read())[1]
    b = st.decode(open(tmp / "tiles/1_0.gst", "rb").read())[1]
    assert (a[:, -1] == b[:, 0]).all()


def test_decode_refuses_damage():
    blob = st.encode(np.zeros((st.N, st.N), np.uint8), np.zeros((st.N, st.N), np.uint8), 0, 0)
    for bad in (b"XXXX" + blob[4:], blob[:-1], blob[:10]):
        try:
            st.decode(bad)
        except ValueError:
            continue
        raise AssertionError("damaged tile accepted")


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
