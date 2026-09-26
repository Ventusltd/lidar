"""Tests for src/canopy_tiles.py and src/canopy_lines.py on a synthetic scene.

One flat-roofed building, one tree, one 50 m hedge, one rough object the pulses do not pass through,
on a gently sloping field. The building must be structure and never shown, the tree tree, the hedge a
hedge with one traced run at its real height, the rough solid uncertain; the witness must agree.

    python lidar/tests/test_canopy.py
"""
import hashlib, inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import canopy_tiles as ct  # noqa: E402
import canopy_lines as cl  # noqa: E402
import fetch_wcs  # noqa: E402

GPU = [False] + ([True] if ct.cp is not None else [])
rng = np.random.default_rng(7)


def scene(n=129):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    dtm = 100.0 + 0.01 * x + 0.02 * y
    chm = np.zeros((n, n)); lz_up = np.zeros((n, n))
    tree = (x - 30) ** 2 + (y - 30) ** 2 <= 36                    # crown radius 6 m, 12 m tall
    chm[tree] = 12 - 0.2 * ((x - 30) ** 2 + (y - 30) ** 2)[tree] + rng.uniform(-1.5, 1.5, tree.sum())
    lz_up[tree] = 0.2 * chm[tree]                                 # most pulses reach low
    bld = (x >= 80) & (x < 96) & (y >= 20) & (y < 30)              # flat roof, 6 m, solid
    chm[bld] = 6.0; lz_up[bld] = 6.0
    hedge = (y >= 90) & (y <= 91) & (x >= 20) & (x < 70)           # 50 m by 2 m, about 2.2 m tall
    chm[hedge] = 2.2 + rng.uniform(-0.4, 0.4, hedge.sum())
    odd = (x >= 100) & (x < 110) & (y >= 80) & (y < 90)            # rough but solid (a bale stack?)
    chm[odd] = np.where((x[odd] + y[odd]) % 2 == 0, 3.0, 5.0); lz_up[odd] = chm[odd]
    return dtm, dtm + chm, dtm + lz_up, dict(tree=tree, bld=bld, hedge=hedge, odd=odd)


def test_scene_classes_and_heights():
    dtm, fz, lz, m = scene()
    for gpu in GPU:
        rec, cls, hq, lines = ct.run(dtm, fz, lz, use_gpu=gpu)
        assert (cls[m["bld"]] == ct.STRUCT).mean() > 0.5, "building core is structure"
        assert not np.isin(cls[m["bld"]], (ct.HEDGE, ct.TREE)).any(), "no building cell is shown"
        assert (hq[m["bld"]] == 0).all(), "no height published for hidden cells"
        assert (cls[m["tree"]] == ct.TREE).mean() > 0.9
        assert (cls[m["hedge"]] == ct.HEDGE).mean() > 0.9
        assert (cls[m["odd"]] == ct.UNSURE).mean() > 0.5 and not np.isin(cls[m["odd"]], (1, 2)).any()
        far = np.ones_like(cls, bool); far[:, :] = True
        for k in m.values():
            far &= ~cl.dilate(k, 3)
        assert (cls[far] == ct.OPEN).all()
        h = hq[m["tree"]].astype(float) * ct.STEP_MM / 1000
        assert abs(h.max() - (fz - dtm)[m["tree"]].max()) <= 0.1 + 1e-9
        assert rec["photons"] > 0 and rec["witness"]["photons"] == 0, rec["witness"]
        runs = {l["run"] for l in lines}
        assert len(runs) == 1, lines
        span = max(l["length_m"] for l in lines)
        assert span > 40, span
        assert all(1.5 < p[2] < 3.0 for l in lines for p in l["pts"])
        assert all(88 <= p[1] <= 93 for l in lines for p in l["pts"])


def test_lattice_mast_and_specks_are_never_shown():
    n = 97
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    dtm = 80.0 + 0.0 * x
    chm = np.zeros((n, n))
    # a porous 40 m lattice tower: four legs thickening to the ground, a 21 m cross-arm, open between
    for cx, cy in ((44, 44), (52, 44), (44, 52), (52, 52)):
        chm[cy - 1:cy + 1, cx - 1:cx + 1] = rng.uniform(5, 30, (2, 2))
    chm[47:50, 47:50] = [[34, 40, 33], [39, 12, 38], [31, 37, 35]]
    chm[48, 38:59] = 30 + rng.uniform(-1, 1, 21)
    chm[20, 20] = 10.0                                             # a lone return 10 m up (a wire, a bird)
    fz, lz = dtm + chm, dtm.copy()                                 # porous: every last return reaches the ground
    for gpu in GPU:
        rec, cls, hq, _ = ct.run(dtm, fz, lz, use_gpu=gpu)
        tall = chm >= ct.H_MIN
        assert not np.isin(cls[tall], (ct.HEDGE, ct.TREE)).any(), np.unique(cls[tall], return_counts=True)
        assert cls[48, 48 + 1] == ct.STRUCT and cls[20, 20] in (ct.UNSURE, ct.STRUCT)
        assert rec["mast_cells"] > 0 and (hq == 0).all()


def test_gate_gap_joins_two_short_runs():
    veg = np.zeros((30, 60), bool); chm = np.where(veg, 0.0, 2.0)
    veg[14:16, 5:17] = True; veg[14:16, 20:32] = True                # 12 m, a 3 m gate, 12 m
    _, _, lines, st = cl.hedgerows(veg, chm)
    assert st["runs_kept"] == 1 and len(lines) == 2, st
    veg[14:16, 20:32] = False
    assert cl.hedgerows(veg, chm)[3]["runs_kept"] == 0


def test_pair_agrees_between_gpu_and_cpu():
    if len(GPU) < 2:
        return
    dtm, fz, lz, _ = scene()
    a = ct.run(dtm, fz, lz, use_gpu=False, witness_on=False)
    b = ct.run(dtm, fz, lz, use_gpu=True, witness_on=False)
    assert (a[1] == b[1]).all() and (a[2] == b[2]).all() and a[3] == b[3]


def test_nodata_stays_nodata():
    dtm, fz, lz, _ = scene(65)
    lz[10, 10] = np.nan
    _, cls, _, _ = ct.run(dtm, fz, lz, use_gpu=False)
    assert cls[10, 10] == ct.NODATA and cls[10, 12] != ct.NODATA


def test_roughness_closed_form_matches_projection():
    z = rng.normal(0, 1, (20, 20)) + np.add.outer(np.arange(20) * 0.3, np.arange(20) * -0.2)
    a = ct.roughness(np, z)
    w = ct.witness(np.zeros_like(z), z, z)["rough"]
    assert np.abs(a - w).max() < 1e-9
    plane = np.add.outer(np.arange(9) * 0.5, np.arange(9) * 0.25)
    assert ct.roughness(np, plane)[1:-1, 1:-1].max() < 1e-9


def test_morphology():
    m = np.zeros((40, 40), bool); m[10:13, 5:35] = True; m[20:32, 20:32] = True
    assert (cl.opening(m, 4) == cl.opening_witness(m, 4)).all()
    op = cl.opening(m, 4)
    assert not op[10:13].any() and op[20:32, 20:32].all()
    sk = cl.thin(m[:15])
    assert sk[10:13].sum(0)[7:33].max() == 1 and sk.sum() >= 25
    lines = cl.trace(sk)
    assert len(cl.components(lines)) == 1
    assert cl.simplify([(0, 0), (0, 1), (0, 2), (0, 3)]) == [(0, 0), (0, 3)]
    d = cl.depth(m)
    assert d[11, 20] == 1 and d[26, 26] >= 5


def test_tiles_index_and_hedges(tmp_path):
    n = 2 * ct.TILE_M + 1
    dtm, fz, lz, _ = scene(n)
    rec, cls, hq, lines = ct.run(dtm, fz, lz, 400000, 300000, use_gpu=False, witness_on=False)
    idx = ct.write_tiles(str(tmp_path), cls, hq, lines, 400000, 300000, "synth")
    raw = open(tmp_path / ct.INDEX, "rb").read()
    assert b"\r\n" not in raw and b"\r\n" not in open(tmp_path / ct.HEDGES, "rb").read()
    assert idx["shown"] == [1, 2] and idx["licence"] == "OGL-3.0" and "Environment Agency" in idx["attribution"]
    assert hashlib.sha256(open(tmp_path / ct.HEDGES, "rb").read()).hexdigest() == idx["hedges"]["sha256"]
    hed = json.load(open(tmp_path / ct.HEDGES))
    assert hed["lines"] and hed["lines"][0]["pts"][0][0] >= 400000
    for t in idx["tiles"]:
        blob = open(tmp_path / t["file"], "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"]
        head, c, h = ct.decode(blob)
        r0, c0 = t["n0"] - 300000, t["e0"] - 400000
        assert (c == cls[r0:r0 + ct.N, c0:c0 + ct.N]).all() and (h == hq[r0:r0 + ct.N, c0:c0 + ct.N]).all()
        assert head["shown_count"] == t["hedge"] + t["tree"] and head["height_step_mm"] == 200
    a = ct.decode(open(tmp_path / "tiles/0_0.gcn", "rb").read())[1]
    b = ct.decode(open(tmp_path / "tiles/1_0.gcn", "rb").read())[1]
    assert (a[:, -1] == b[:, 0]).all()


def test_decode_refuses_damage():
    z = np.zeros((ct.N, ct.N), np.uint8)
    blob = ct.encode(z, z, 0, 0)
    for bad in (b"XXXX" + blob[4:], blob[:-1], blob[:10]):
        try:
            ct.decode(bad)
        except ValueError:
            continue
        raise AssertionError("damaged tile accepted")


def test_fetch_products():
    u = fetch_wcs.coverage_url(0, 1000, 0, 1000, "fzdsm1m")
    assert "first-return-dsm-1m/wcs" in u and "FZ_DSM_1m" in u
    assert "last-return-dsm-1m/wcs" in fetch_wcs.coverage_url(0, 1, 0, 1, "lzdsm1m")
    assert fetch_wcs.coverage_url(0, 1, 0, 1) == fetch_wcs.coverage_url(0, 1, 0, 1, "dtm1m")


def test_files_stay_small():
    for f in ("canopy_tiles.py", "canopy_lines.py"):
        p = os.path.join(os.path.dirname(__file__), "..", "src", f)
        assert len(open(p, encoding="utf-8").read().splitlines()) < 400, f


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
