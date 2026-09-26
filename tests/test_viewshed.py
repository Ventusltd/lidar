"""Tests for src/viewshed.py and src/viewshed_cpu.py on synthetic ground.

Flat ground must be seen everywhere by both methods (which also proves the R2 sweep reaches every cell);
a wall must cast the shadow its geometry says; flat ground on a curved Earth must hide the ground beyond the
horizon distance sqrt(eye / c) and a target beyond sqrt(eye / c) + sqrt(target / c); rough ground must make
photons, mostly on visibility edges; the GPU and CPU must agree to the bit; tiles must round-trip.

    python lidar/tests/test_viewshed.py
"""
import hashlib, inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import viewshed as vs  # noqa: E402
import viewshed_cpu as vc  # noqa: E402

GPU = [False] + ([True] if vs.cp is not None else [])


def test_flat_ground_is_all_visible_both_ways():
    g = np.full((41, 57), 120.0)
    for gpu in GPU:
        rec, ce, cp_ = vs.run(g, [(3, 5), (20, 50)], sp=2.0, use_gpu=gpu)
        assert (ce == 2).all() and (cp_ == 2).all(), (np.unique(ce), np.unique(cp_))
        assert rec["photons"] == 0 and rec["union_photons"] == 0
        if gpu:
            assert rec["witness"]["photons"] == 0


def test_wall_casts_the_shadow_its_geometry_says():
    # observer on node (10, 0) of flat ground, a 2.5 m wall across column 20 (40 m out at 2 m spacing).
    # On the observer's row a 3 m target at distance x is seen iff (3 - 1.7) / x >= (2.5 - 1.7) / 40,
    # i.e. x <= 65 m: columns 21..32 seen, 33 onwards hidden. A 1 m target is never seen behind the wall.
    g = np.zeros((21, 61)); g[:, 20] = 2.5
    for gpu in GPU:
        for fn in ("r3", "r2"):
            if gpu:
                e, p = vs.pair_one(vs.cp, vs.cp.asarray(g.astype(np.float32)), 2.0, 10, 0, 1.7, 3.0, 0.0,
                                   vs.cp.asarray(vc.perimeter(*g.shape)))
                row = vs.to_host(e if fn == "r3" else p)[10]
            else:
                row = getattr(vc, fn)(g, 2.0, 10, 0, 1.7, 3.0, 0.0)[10]
            assert row[:21].all() and row[21:33].all() and not row[33:].any(), (fn, gpu, row)
        low = vc.r3(g, 2.0, 10, 0, 1.7, 1.0, 0.0)[10]
        assert low[:21].all() and not low[21:].any()


def test_curvature_hides_beyond_the_horizon():
    c = vs.curvature_coeff()
    assert abs(c * 1e6 - 0.0683) < 1e-4                        # 6.8 cm at 1 km
    h_eye, h_tgt = np.sqrt(1.7 / c), np.sqrt(3.0 / c)           # about 4.99 km and 6.63 km
    g = np.zeros((3, 321))                                     # 50 m spacing, 16 km of flat ground
    x = np.arange(321) * 50.0
    for gpu in GPU:
        _, ce, cpn = vs.run(g, [(1, 0)], sp=50.0, tgt=0.0, use_gpu=gpu)
        for row in (ce[1], cpn[1]):
            assert row[x < h_eye - 200].all() and not row[x > h_eye + 200].any()
        _, ce, _ = vs.run(g, [(1, 0)], sp=50.0, tgt=3.0, use_gpu=gpu)
        assert ce[1][x < h_eye + h_tgt - 300].all() and not ce[1][x > h_eye + h_tgt + 300].any()
        _, ce, _ = vs.run(g, [(1, 0)], sp=50.0, tgt=0.0, curvature=False, use_gpu=gpu)
        assert ce[1].all()


def rough(n=129, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return 150 + 8 * np.sin(x / 13) * np.cos(y / 17) + 5 * np.sin((x + 2 * y) / 9) + rng.normal(0, 0.3, (n, n))


def test_rough_ground_photons_sit_on_edges_and_witness_agrees():
    g = rough()
    nodes = [(10, 10), (64, 64), (120, 30)]
    for gpu in GPU:
        rec, ce, cpn = vs.run(g, nodes, sp=2.0, use_gpu=gpu, witness_n=3)
        assert 0 < rec["photons"] < 0.05 * rec["observer_cell_pairs"], rec["photons"]
        assert rec["edges"]["share_on_edge"] > 2 * rec["edges"]["by_chance"], rec["edges"]
        assert 0 < rec["electron"]["share_any"] < 1
        if gpu:
            w = rec["witness"]
            assert w["observers"] == 3 and w["photons"] == 0, w
    # CPU and GPU paths give the same counts
    if len(GPU) == 2:
        _, a, b = vs.run(g, nodes, sp=2.0, use_gpu=False)
        _, c, d = vs.run(g, nodes, sp=2.0, use_gpu=True, witness_n=0)
        assert (a == c).all() and (b == d).all()


def test_nodata_is_marked_and_not_a_blocker():
    g = np.zeros((21, 21)); g[5, 5] = np.nan
    for gpu in GPU:
        _, ce, cpn = vs.run(g, [(10, 10)], use_gpu=gpu)
        assert ce[5, 5] == vs.NODATA and cpn[5, 5] == vs.NODATA
        assert (ce[np.isfinite(g)] == 1).all()


def test_observers_from_lines_points_and_geojson(tmp):
    pts = vs.along([[0, 0], [100, 0], [100, 30]], 50)
    assert [tuple(np.round(p, 6)) for p in pts] == [(0, 0), (50, 0), (100, 0), (100, 30)]
    (tmp / "w.json").write_text(json.dumps({"lines": [{"pts": [[0, 0], [0, 120]]}], "points": [[7, 8]]}), encoding="utf-8", newline="\n")
    (tmp / "g.json").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"geometry": {"type": "Point", "coordinates": [1, 2]}},
        {"geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [60, 0]]]}}]}), encoding="utf-8", newline="\n")
    obs = vs.load_observers([tmp / "w.json", tmp / "g.json"], 50, points=[(3, 4)])
    assert obs[0] == (3, 4) and (7, 8) in obs and (1, 2) in obs
    assert len(obs) == 1 + 1 + 4 + 1 + 3
    nodes, dropped = vs.snap([(0, 0), (1, 0.4), (10, 10), (-5, 0), (1e6, 0)], 0, 0, 2, 10, 10)
    assert nodes == [(0, 0), (5, 5)] and dropped == 2        # (1, 0.4) snaps onto (0, 0) and merges


def test_tiles_roundtrip_index_and_caveat(tmp):
    n = 2 * 128 + 1
    g = rough(n)
    _, ce, _ = vs.run(g, [(5, 5), (200, 100)], sp=2.0, use_gpu=False)
    idx = vs.write_tiles(str(tmp), ce, 400000, 300000, 2, "synth", 1.7, 3.0, [(400010, 300010)])
    assert len(idx["tiles"]) == 4 and idx["caveat"] == vs.CAVEAT and idx["canopy"] is False
    raw = open(tmp / vs.INDEX, "rb").read()
    assert b"\r\n" not in raw and json.loads(raw)["tiles"] == idx["tiles"]
    for t in idx["tiles"]:
        blob = open(tmp / t["file"], "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"] and len(blob) == t["bytes"]
        head, v = vs.decode(blob)
        r0, c0 = (t["n0"] - 300000) // 2, (t["e0"] - 400000) // 2
        assert (v == ce[r0:r0 + 129, c0:c0 + 129]).all()
        assert head["visible_count"] == t["visible"] and head["eye_mm"] == 1700 and head["target_mm"] == 3000
        assert head["flags"] == 1 and head["spacing_mm"] == 2000
    a = vs.decode(open(tmp / "tiles/0_0.gvs", "rb").read())[1]
    b = vs.decode(open(tmp / "tiles/1_0.gvs", "rb").read())[1]
    assert (a[:, -1] == b[:, 0]).all()
    canopy = vs.write_tiles(str(tmp / "c"), ce, 0, 0, 2, "synth", 1.7, 3.0, [], canopy=True)
    assert canopy["caveat"] == vs.CAVEAT_CANOPY and canopy["curvature"]["k"] == 0.13


def test_decode_refuses_damage():
    blob = vs.encode(np.zeros((129, 129), np.uint8), 0, 0, 2000, 1.7, 3.0, 1)
    for bad in (b"XXXX" + blob[4:], blob[:-1], blob[:10]):
        try:
            vs.decode(bad)
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
