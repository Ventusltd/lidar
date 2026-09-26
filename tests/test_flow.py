"""Tests for src/flow_tiles.py on synthetic valleys, pits and planes.

A V-valley must gather its whole catchment onto the thalweg (D8 exactly, D-inf nearly); a square
pit in a slope must pond to the known spill level with the known volume; a plane must split D-inf
flow by its angle; the GPU and the CPU witness (heap priority-flood, NumPy, sorted sweep) must say
the same thing; the .gfl tiles must say what the maths says.

    E:/swarm/gpu-bench/venv/Scripts/python.exe lidar/tests/test_flow.py
"""
import hashlib, inspect, json, math, os, pathlib, sys, tempfile, time, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import flow_tiles as ft  # noqa: E402

GPU = [False] + ([True] if ft.cp is not None else [])


def grid_xy(n):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)      # row 0 is the SOUTH edge
    return x, y


def v_valley(n=65):
    x, y = grid_xy(n)
    return 50.0 + 0.2 * np.abs(x - n // 2) + 0.05 * y     # thalweg down the middle, draining south


def test_v_valley_gathers_on_the_thalweg():
    n, mid = 65, 32
    for gpu in GPU:
        rec, o = ft.run(v_valley(n), use_gpu=gpu)
        acc, d8 = o["accE"], o["dir"]
        # every interior row sends its 63 cells to the thalweg, which runs south to the edge
        assert acc[1, mid] == 63 * 63, acc[1, mid]
        assert (d8[1:-1, mid] == 4).all()                          # S
        assert (d8[1:-1, 1:mid] == 2).all() and (d8[1:-1, mid + 1:-1] == 6).all()   # E / W into it
        chan = (acc >= ft.CHANNEL_M2)
        assert set(np.nonzero(chan)[1].tolist()) == {mid}
        assert np.nonzero(chan)[0].max() == 32                    # 63 * (64 - r) >= 2000 up to r = 32
        # D-inf splits the side slopes 14 degrees off the contour (76 % across, 24 % down-valley), so
        # the bottom row leaks a little straight to the edge: most, not all, of it on the thalweg
        assert 0.9 * 63 * 63 < o["accP"][1, mid] < 63 * 63
        assert set(np.nonzero(o["accP"] >= ft.CHANNEL_M2)[1].tolist()) == {mid}
        assert rec["electron"]["mass_balance"] == 1.0 and abs(rec["positron"]["mass_balance"] - 1) < 1e-12
        assert rec["electron"]["sinks"] == rec["positron"]["sinks"] == 0
        assert rec["fill"]["fixpoint_violations"] == 0 and rec["ponding"]["depressions"] == 0
        assert rec["witness"]["photons"] == 0, rec["witness"]


def test_square_pit_ponds_to_its_spill_level():
    n = 65
    x, y = grid_xy(n)
    g = 10.0 + 0.01 * y
    g[30:35, 30:35] -= 1.0
    want = sum(5 * (10.0 + 0.01 * 29 - (10.0 + 0.01 * r - 1.0)) for r in range(30, 35))   # 24.25 m3
    for gpu in GPU:
        rec, o = ft.run(g, use_gpu=gpu)
        p = rec["ponding"]
        assert p["depressions"] == 1 and p["wet_cells"] == 25
        assert abs(p["total_m3"] - want) < 1e-3 and abs(p["biggest"][0]["volume_m3"] - want) < 1e-3
        assert abs(p["max_depth_m"] - 0.99) < 1e-9 and abs(p["biggest"][0]["level_m"] - 10.29) < 1e-9
        assert abs(p["biggest"][0]["centroid_e"] - 32) < 1e-9 and abs(p["biggest"][0]["centroid_n"] - 32) < 1e-9
        # the pit drains over its lowest rim once filled: no sinks, all area reaches the edge
        assert rec["electron"]["sinks"] == 0 and rec["electron"]["mass_balance"] == 1.0
        assert (o["We"][30:35, 30:35] > 10.29).all() and (o["We"][30:35, 30:35] < 10.29 + 1e-3).all()
        assert rec["witness"]["photons"] == 0, rec["witness"]


def test_dinf_splits_a_plane_by_angle():
    n = 9
    x, y = grid_xy(n)
    th = math.radians(20)
    g = 100.0 - 0.1 * (math.cos(th) * x + math.sin(th) * y)       # falls 20 degrees north of east
    nod, fixed = ft.masks(g)
    i = 4 * n + 4
    for gpu in GPU:
        if gpu:
            W = ft.cp.asarray(g)
            p = ft.gpu_route(W, nod, fixed, 1.0, "dinf"); e = ft.gpu_route(W, nod, fixed, 1.0, "d8")
            tgt, frac, d8 = ft.to_host(p["tgt"]), ft.to_host(p["frac"]), ft.to_host(e["dir"])
        else:
            p = ft.cpu_route(g, nod, fixed, 1.0, "dinf"); e = ft.cpu_route(g, nod, fixed, 1.0, "d8")
            tgt, frac, d8 = p["tgt"], p["frac"], e["dir"]
        assert tgt[2 * i] == i + 1 and tgt[2 * i + 1] == i + n + 1          # E and NE
        assert abs(frac[2 * i + 1] - 20 / 45) < 1e-12 and abs(frac[2 * i] - 25 / 45) < 1e-12
        assert d8[4, 4] == 2                                                   # D8: all east


def test_longest_path_on_a_south_slope():
    n = 33
    _, y = grid_xy(n)
    for gpu in GPU:
        rec, _ = ft.run(0.1 * y + 5.0, 1000, 2000, use_gpu=gpu)
        lp = rec["electron"]["longest_path"]
        assert lp["length_m"] == 31.0 and lp["cells"] == 32 and lp["straight_m"] == 31.0
        assert lp["source_n"] == 2031 and lp["outlet_n"] == 2000
        # D8 and D-inf agree exactly on a cardinal plane: no photons
        assert rec["photons"] == 0


def test_witness_agrees_on_rough_terrain():
    rng = np.random.default_rng(7)
    n = 129
    x, y = grid_xy(n)
    g = 60 + 0.03 * y - 0.02 * x + 3 * np.sin(x / 11) * np.cos(y / 17) + rng.normal(0, 0.08, (n, n))
    g[40:44, 70:90] -= 0.6                                                     # a ditch that ponds
    for gpu in GPU:
        rec, o = ft.run(g, use_gpu=gpu)
        w = rec["witness"]
        assert w["photons"] == 0, w
        assert rec["fill"]["fixpoint_violations"] == 0
        assert rec["ponding"]["depressions"] > 10 and rec["ponding"]["total_m3"] > 5
        assert (o["Wf"] >= g).all() and (o["We"] >= o["Wf"]).all()
        assert rec["photons"] > 0 and rec["share_near_channel"] > rec["near_by_chance"]
    if ft.cp is not None:                                                      # the two paths, arrays
        _, a = ft.run(g, use_gpu=True, witness_on=False); _, b = ft.run(g, use_gpu=False, witness_on=False)
        assert (a["dir"] == b["dir"]).all() and (a["accE"] == b["accE"]).all()
        assert np.allclose(a["accP"], b["accP"], rtol=1e-12, atol=0)


def test_nodata_hole_is_an_outlet():
    g = v_valley(65)
    g[20:23, 10:13] = np.nan
    for gpu in GPU:
        rec, o = ft.run(g, use_gpu=gpu)
        assert (o["dir"][20:23, 10:13] == ft.NODATA).all()
        assert o["dir"][19, 9] == ft.OUTLET and o["dir"][21, 13] == ft.OUTLET
        assert rec["valid"] == 65 * 65 - 9
        assert rec["electron"]["mass_balance"] == 1.0 and abs(rec["positron"]["mass_balance"] - 1) < 1e-12
        assert rec["witness"]["photons"] == 0, rec["witness"]


def test_classes():
    acc = np.array([[1.0, 9.99, 10.0, 1995.0, 1996.0, 4.2e6]])
    k = ft.classes(acc, np.zeros(acc.shape, bool))
    assert k.tolist() == [[0, 9, 10, 32, 33, 66]]
    assert ft.classes(acc, np.ones(acc.shape, bool)).tolist() == [[255] * 6]


def test_tiles_roundtrip_and_index(tmp):
    n = 2 * ft.TILE_M + 1
    x, y = grid_xy(n)
    g = 100 + 8 * np.sin(x / 40.0) * np.cos(y / 55.0) + 0.02 * y
    rec, o = ft.run(g, 400000, 300000, use_gpu=bool(ft.cp), witness_on=False)
    idx = ft.write_tiles(str(tmp), o, 400000, 300000, "synth")
    raw = open(tmp / ft.INDEX, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert len(idx["tiles"]) == 4 and idx["tile_m"] == 256 and json.loads(raw)["tiles"] == idx["tiles"]
    acls = ft.classes(o["accE"], o["nod"])
    for t in idx["tiles"]:
        blob = open(tmp / t["file"], "rb").read()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"] and len(blob) == t["bytes"]
        head, a, d = ft.decode(blob)
        r0, c0 = t["n0"] - 300000, t["e0"] - 400000
        assert (a == acls[r0:r0 + ft.N, c0:c0 + ft.N]).all() and (d == o["dir"][r0:r0 + ft.N, c0:c0 + ft.N]).all()
        assert head["channel_count"] == t["channel"] and head["channel_m2"] == 2000 and head["log_scale"] == 10
        assert head["origin_e_m"] == t["e0"] and head["nodata_count"] == 0
    a = ft.decode(open(tmp / "tiles/0_0.gfl", "rb").read())[2]
    b = ft.decode(open(tmp / "tiles/1_0.gfl", "rb").read())[2]
    assert (a[:, -1] == b[:, 0]).all()


def test_decode_refuses_damage():
    z = np.zeros((ft.N, ft.N), np.uint8)
    blob = ft.encode(z, z, 0, 0)
    for bad in (b"XXXX" + blob[4:], blob[:-1], blob[:10]):
        try:
            ft.decode(bad)
        except ValueError:
            continue
        raise AssertionError("damaged tile accepted")


def test_direction_encoding_is_explicit(tmp):
    """FEEDBACK fix 4: header flags and index say COMPASS; a cell draining north decodes as N, not an ESRI code."""
    x, y = grid_xy(ft.N)
    g = 100.0 - 0.05 * y                                   # falls to the north
    rec, o = ft.run(g, 0, 0, use_gpu=bool(ft.cp), witness_on=False)
    idx = ft.write_tiles(str(tmp), o, 0, 0, "synth")
    head, a, d = ft.decode(open(tmp / "tiles/0_0.gfl", "rb").read())
    assert head["flags"] & ft.FLAG_COMPASS and head["dir_encoding"] == "compass" and head["method"] == 1
    assert d[100, 100] == 0 and idx["dir_codes"][str(d[100, 100])] == "N"
    assert idx["dir_encoding"] == "compass" and "NOT a direction code" in idx["header_layout"]
    assert idx["default_class"] == 33 and "filled surface" in idx["label"]
    assert idx["attribution"].startswith("© Environment Agency") and idx["licence"] == "Open Government Licence v3.0"


def test_ponding_hollows_are_recorded(tmp):
    """FEEDBACK fix 4: the real hollow is written (level, volume, cells); tiny puddles are left out."""
    import flow_hollows as fh
    x, y = grid_xy(ft.N)
    g = 10.0 + 0.01 * y
    g[30:35, 30:35] -= 1.0                                  # the square pit: 24.25 m3 to its spill level
    g[100, 100] -= 0.02                                     # a 1 cm, 0.01 m3 puddle: too small to keep
    rec, o = ft.run(g, 1000, 2000, use_gpu=bool(ft.cp), witness_on=False)
    idx = ft.write_tiles(str(tmp), o, 1000, 2000, "synth")
    raw = open(tmp / fh.INDEX, "rb").read()
    hol = json.loads(raw.decode("utf-8"))
    assert bytes([13]) not in raw and idx["hollows"]["count"] == 1 and len(hol["hollows"]) == 1
    assert rec["ponding"]["depressions"] == 2
    h = hol["hollows"][0]
    assert abs(h["volume_m3"] - 24.25) < 1e-6 and h["area_m2"] == 25 and abs(h["level_m"] - 10.29) < 1e-9
    assert h["bbox"] == [1030, 2030, 1034, 2034] and h["tiles"] == ["0_0"]
    t = hol["tiles"][0]
    blob = open(tmp / t["file"], "rb").read()
    assert hashlib.sha256(blob).hexdigest() == t["sha256"]
    head, depth = fh.decode(blob)
    assert head["wet_count"] == 25 and head["hollows"] == 1 and depth[100, 100] == 0
    assert depth[30, 30] == 99 and depth[34, 34] == 95      # 10.29 - (10.30 - 1) m and 10.29 - (10.34 - 1) m
    assert (o["dir"][30:35, 30:35] != ft.OUTLET).all()


if __name__ == "__main__":
    fails = 0; t0 = time.perf_counter()
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(*([pathlib.Path(d)] if inspect.signature(fn).parameters else []))
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name); traceback.print_exc()
    print(f"{fails} failed in {time.perf_counter() - t0:.1f} s")
    sys.exit(1 if fails else 0)
