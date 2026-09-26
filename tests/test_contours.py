"""Tests for src/contour_tiles.py and src/contour_pair.py on synthetic ground.

A plane gives straight contours exactly where the algebra puts them; a cone gives circles of the right
radius; a saddle (z = x y) exercises the ambiguous cells, where both channels and the witness must make the
same choice; no-data holes are skipped by every channel; the simplifier keeps its promise; tiles, index and
receipt are LF and carry true hashes.

    E:/swarm/gpu-bench/venv/Scripts/python.exe tests/test_contours.py
"""
import hashlib, inspect, json, os, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import contour_pair as cpair  # noqa: E402
import contour_tiles as ct  # noqa: E402

GPU = [False] + ([True] if ct.cp is not None else [])


def mesh(n):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return x, y


def all_lines(tiles):
    for (tx, ty), lv in tiles.items():
        for k, lines in lv.items():
            for q in lines:
                yield k, q


def test_plane_contours_are_straight_and_placed_exactly():
    x, y = mesh(41)
    g = 100.0 + 0.13 * x + 0.0 * y           # contours run south-north at x = (L - 100) / 0.13
    for gpu in GPU:
        rec, tiles = ct.run(g, use_gpu=gpu)
        assert rec["photons"] == 0 and rec["witness"]["photons"] == 0, rec
        seen = set()
        for k, q in all_lines(tiles):
            L = k * ct.STEP_M
            assert np.allclose(q[:, 0], (L - 100) / 0.13, atol=1e-9)
            assert len(q) == 2 and abs(q[0, 1] - q[1, 1]) == 40   # one straight line, simplified to its ends
            seen.add(L)
        assert seen == {100.5 + 0.5 * i for i in range(10)}      # 100.5 .. 105.0 (max 105.2)


def test_cone_gives_closed_circles_of_the_right_radius():
    x, y = mesh(81)
    r = np.hypot(x - 40.3, y - 39.7)
    g = 200.0 - 0.1 * r
    for gpu in GPU:
        rec, tiles = ct.run(g, use_gpu=gpu, tol=0.05)
        assert rec["photons"] == 0 and rec["witness"]["photons"] == 0
        for k, q in all_lines(tiles):
            rad = (200.0 - k * ct.STEP_M) / 0.1
            if rad < 5 or rad > 38:
                continue
            assert np.allclose(q[0], q[-1]), "ring must close"
            got = np.hypot(q[:, 0] - 40.3, q[:, 1] - 39.7)
            assert np.abs(got - rad).max() < 0.1, (rad, np.abs(got - rad).max())


def test_saddle_cells_agree_across_channels_and_witness():
    x, y = mesh(33)
    g = 50.0 + 0.02 * (x - 16.2) * (y - 15.8) + 1e-3 * np.sin(x * 1.7 + y)
    for gpu in GPU:
        rec, _ = ct.run(g, use_gpu=gpu)
        assert rec["electron"]["saddles"] > 0, "the surface must contain saddles"
        assert rec["electron"]["saddles"] == rec["positron"]["saddles"] == rec["witness"]["saddles"]
        assert rec["photons"] == 0 and rec["witness"]["photons"] == 0, rec


def test_saddle_decider_follows_the_centre():
    # corners a=SW b=SE c=NE d=NW; a and c above 10, centre mean 10.25 above -> a joins c, b and d cut off
    g = np.array([[11.0, 9.0], [9.0, 12.0]])   # rows south to north: a=11 b=9 / d=9 c=12
    for xp_gpu in GPU:
        xp = ct.cp if xp_gpu else np
        el, n = cpair.electron(xp, xp.asarray(g), 10.0)
        pairs = sorted(sorted((int(p), int(q))) for p, q in zip(ct.to_host(el[2]), ct.to_host(el[3])))
        R, C = g.shape
        S, E, N, W = 0, R * C + 1, C, R * C
        assert n == 1 and pairs == sorted([sorted((S, E)), sorted((N, W))])


def test_nodata_holes_are_skipped_by_every_channel():
    x, y = mesh(40)
    g = 10.0 + 0.07 * x + 0.05 * y
    g[10:14, 20:23] = np.nan
    for gpu in GPU:
        rec, tiles = ct.run(g, use_gpu=gpu)
        assert rec["photons"] == 0 and rec["witness"]["photons"] == 0, rec
        for k, q in all_lines(tiles):
            assert np.isfinite(q).all()
            inside = (q[:, 0] > 19.5) & (q[:, 0] < 22.5) & (q[:, 1] > 9.5) & (q[:, 1] < 13.5)
            assert not inside.any()


def test_a_disagreement_is_counted_not_hidden():
    x, y = mesh(20)
    g = 5.0 + 0.1 * x
    seg, tl = cpair.pair_level(np, g, 5.55)
    assert tl["only_e"] == tl["only_p"] == 0
    shifted = tuple(s.copy() for s in seg)
    shifted[4][0] += 1e-6                               # one electron crossing moved 1 micrometre
    w = cpair.witness_level(g, 5.55, shifted)
    assert w["photons"] == 1 and w["max_dxy"] > cpair.TOL_XY
    dropped = tuple(s[1:] for s in seg)                 # one segment lost
    assert cpair.witness_level(g, 5.55, dropped)["only_w"] == 1


def test_douglas_peucker_keeps_its_tolerance():
    t = np.linspace(0, 2 * np.pi, 400)
    p = np.stack([30 * np.cos(t), 30 * np.sin(t)], 1)
    for tol in (0.01, 0.25, 1.0):
        q, dev = ct.simplify(p, tol)
        assert dev <= tol and len(q) < len(p) and np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1])
    q, dev = ct.simplify(np.array([[0, 0], [1, 0.1], [2, 0]], float), 0.25)
    assert len(q) == 2 and abs(dev - 0.1) < 1e-12


def test_tiles_split_on_the_256_m_grid_and_meet_on_their_edges():
    x, y = mesh(513)
    g = 80.0 + 0.011 * x + 0.004 * y
    rec, tiles = ct.run(g, use_gpu=bool(GPU[-1]), witness_on=False)
    assert set(tiles) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    ends = {}
    for (tx, ty), lv in tiles.items():
        for k, lines in lv.items():
            for q in lines:
                assert (q[:, 0] >= tx * 256 - 1e-9).all() and (q[:, 0] <= (tx + 1) * 256 + 1e-9).all()
                for e in (q[0], q[-1]):
                    if abs(e[0] - 256) < 1e-9 or abs(e[1] - 256) < 1e-9:
                        ends.setdefault((k, round(e[0], 9), round(e[1], 9)), set()).add((tx, ty))
    assert ends and all(len(v) == 2 for v in ends.values()), "every edge crossing is shared by two tiles"


def test_written_tiles_index_and_receipt_are_lf_with_true_hashes():
    x, y = mesh(257)
    g = 150.0 + 0.03 * x + 0.02 * y
    with tempfile.TemporaryDirectory() as d:
        site = os.path.join(d, "site")
        os.makedirs(site)
        np.save(os.path.join(site, "source.npy"), g)
        json.dump({"origin_e_m": 400000, "origin_n_m": 200000, "spacing_m": 1, "rows": "south-to-north"},
                  open(os.path.join(site, "source.json"), "w"))
        assert ct.main(["--site", site, "--cpu"]) == 0
        out = os.path.join(site, "contours")
        raw = open(os.path.join(out, ct.INDEX), "rb").read()
        assert b"\r" not in raw and b"\r" not in open(os.path.join(out, ct.RECEIPT), "rb").read()
        idx = json.loads(raw)
        assert idx["dp_tol_m"] == ct.DP_TOL_M and idx["intervals_m"] == [0.5, 1.0, 5.0]
        rec = json.load(open(os.path.join(out, ct.RECEIPT)))
        assert rec["index_sha256"] == hashlib.sha256(raw).hexdigest()
        (t,) = idx["tiles"]
        blob = open(os.path.join(out, t["file"]), "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"] and b"\r" not in blob
        doc = json.loads(blob)
        assert doc["format"] == "ggc1" and doc["e0"] == 400000 and doc["unit_m"] == 0.01
        zs = [lv["z"] for lv in doc["levels"]]
        assert zs[0] == 150.5 and all(abs(b - a - 0.5) < 1e-9 for a, b in zip(zs, zs[1:]))
        assert t["by_interval"]["5.0"]["lines"] < t["by_interval"]["1.0"]["lines"] < t["by_interval"]["0.5"]["lines"]
        for lv in doc["levels"]:
            for line in lv["lines"]:
                pts = np.array(line).reshape(-1, 2) / 100
                zz = 150 + 0.03 * pts[:, 0] + 0.02 * pts[:, 1]
                assert np.abs(zz - lv["z"]).max() < 0.01 * 0.05 + 1e-9    # 1 cm rounding on a 5 % plane


def _brute_crossings(lines):
    """Proper crossings, every stored segment against every other in blocks (exact on integer cm)."""
    A = np.concatenate([q[:-1] for q in lines]); B = np.concatenate([q[1:] for q in lines])
    line = np.concatenate([np.full(len(q) - 1, n) for n, q in enumerate(lines)])
    idx = np.concatenate([np.arange(len(q) - 1) for q in lines])
    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    bad = 0
    for r0 in range(0, len(A), 256):
        a, b = A[r0:r0 + 256, None], B[r0:r0 + 256, None]
        hit = (orient(a, b, A[None]) * orient(a, b, B[None]) < 0) & (orient(A[None], B[None], a) * orient(A[None], B[None], b) < 0)
        hit &= ~((line[r0:r0 + 256, None] == line[None]) & (np.abs(idx[r0:r0 + 256, None] - idx[None]) <= 1))
        bad += int(hit.sum())
    return bad // 2


def test_simplified_lines_never_cross():
    """FEEDBACK fix 3: Douglas-Peucker alone makes these two lines cross; the tile check puts a vertex back."""
    import contour_topo as topo
    a = np.array([[0, 0], [2, 0.24], [10, 0.24]], float)       # higher level
    b = np.array([[0, -0.1], [2, 0.2], [4, -0.1]], float)      # lower level, below a everywhere
    lines = [(a, ct.simplify_mask(a, 0.25)), (b, ct.simplify_mask(b, 0.25))]
    assert not lines[0][1][1] and lines[1][1][1]                # a's corner dropped, b's kept: they cross
    found, left, touch, restored = topo.untangle(lines, 0.0, 0.0)
    assert found >= 1 and left == 0 and touch == 0 and restored >= 1 and lines[0][1][1]
    # rough ground: whatever Douglas-Peucker does, the stored lines of a tile never cross
    x, y = mesh(129)
    g = 50 + 0.08 * x + 2 * np.sin(x / 9) * np.cos(y / 7) + np.random.default_rng(0).normal(0, 0.15, size=x.shape)
    for gpu in GPU:
        rec, tiles = ct.run(g, use_gpu=gpu, witness_on=False)
        sim = rec["simplify"]
        assert sim["crossings_found"] > 0 and sim["vertices_restored"] > 0, sim   # Douglas-Peucker alone crosses here
        assert sim["crossings"] == 0 and sim["within_tol"], sim
        for (tx, ty), lv in tiles.items():
            cm = [np.rint((q - (tx * ct.TILE_M, ty * ct.TILE_M)) * 100).astype(np.int64) for ls in lv.values() for q in ls]
            assert _brute_crossings(cm) == 0
    assert "traced marching-squares line" in ct.TOL_BASIS


if __name__ == "__main__":
    fails = 0
    for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if name.startswith("test_"):
            try:
                fn(); print("ok  ", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    print("GPU channel:", "yes" if len(GPU) > 1 else "no (numpy only)")
    sys.exit(1 if fails else 0)
