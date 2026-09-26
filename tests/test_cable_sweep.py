# SPDX-License-Identifier: Apache-2.0
"""Tests for src/cable_sweep.py on synthetic ground with known lengths.

Flat ground: 3D = plan = legs - 2 R tan(theta/2) + R theta a bend. A plane of gradient g along a
straight: L sqrt(1 + g^2). A ridge |x - c| kinked on a grid line: each flank exact. Both channels
must hit the known length and each other; short legs must shrink the fillet and be counted.

    python lidar/tests/test_cable_sweep.py
"""
import inspect, json, math, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cable_sweep as cs  # noqa: E402
Ground = cs.Ground

XPS = [np] + ([cs.cp] if cs.cp is not None else [])
OE, ON = 400000.0, 200000.0
CABLES = {"rules": [
    {"voltage_kv": 33, "construction": "a", "when": "installation", "multiple_of_od": 20, "source": "t"},
    {"voltage_kv": 33, "construction": "a", "when": "final", "multiple_of_od": 40, "source": "t"},
    {"voltage_kv": 132, "construction": "b", "when": "installation", "multiple_of_od": 35, "source": "t"},
    {"voltage_kv": 132, "construction": "b", "when": "installation", "multiple_of_od": 30, "source": "t"},
    {"voltage_kv": 132, "construction": "c", "when": "installation", "multiple_of_od": None, "source": "t"}],
    "cables": [
    {"id": "33-1000", "voltage_kv": 33, "csa_mm2": 1000, "od_mm": 70.0, "od_status": "estimated"},
    {"id": "132-1000-spec", "voltage_kv": 132, "csa_mm2": 1000, "od_mm": 115.0, "od_status": "estimated"},
    {"id": "132-1000", "voltage_kv": 132, "csa_mm2": 1000, "od_mm": 91.0, "od_status": "verified"}]}


def ground(xp, z):
    return Ground(np.asarray(z, float), OE, ON, 1.0, xp)


def grid(n=401):
    return np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))


def routes(rows, R=3.0, depth=1.0):
    """rows of (x0, y0, heading, legs, turns) in local metres."""
    n = len(rows)
    D = dict(x0=np.array([r[0] for r in rows], float), y0=np.array([r[1] for r in rows], float),
             h0=np.array([r[2] for r in rows], float), legs=np.zeros((n, cs.NB + 1)), turns=np.zeros((n, cs.NB)))
    for k, r in enumerate(rows):
        D["legs"][k, :len(r[3])] = r[3]; D["turns"][k, :len(r[4])] = r[4]
    D["R"], D["depth"], D["kv"] = np.full(n, R), np.full(n, depth), np.full(n, 33)
    return D


def close(v, want, rel=1e-6):
    assert abs(v - want) <= rel * abs(want), (v, want, abs(v - want) / abs(want))


def test_min_radius_rule():
    r33, r132 = cs.min_radius(CABLES, 33), cs.min_radius(CABLES, 132)
    assert r33["multiple_of_od"] == 20 and abs(r33["radius_m"] - 1.4) < 1e-12          # final rule not used
    assert r132["cable_id"] == "132-1000" and r132["multiple_of_od"] == 35             # verified OD, largest rule
    assert abs(r132["radius_m"] - 3.185) < 1e-12 and r132["depth_m"] == 1.05
    if os.path.exists(cs.CABLES):                                                        # the real file, if present
        real = cs.load_cables(cs.CABLES)
        assert abs(cs.min_radius(real, 33)["radius_m"] - 1.4) < 1e-9
        assert abs(cs.min_radius(real, 132)["radius_m"] - 3.185) < 1e-9


def test_flat_plan_lengths():
    R = 3.185
    th = [math.radians(90), math.radians(-40), math.radians(135), math.radians(-150)]
    rows = [(50.3, 60.7, 0.3, [120.0], []),
            (50.3, 60.7, 0.3, [80.0, 70.0], th[:1]),
            (150.0, 150.0, 2.0, [40.0, 30.0, 25.0, 60.0, 35.0], th)]
    want = [120.0, 150.0 - 2 * R + R * math.pi / 2,
            190.0 + sum(R * abs(t) - 2 * R * math.tan(abs(t) / 2) for t in th)]
    for xp in XPS:
        r = cs.pair(routes(rows, R), ground(xp, np.full((401, 401), 80.0)))
        assert not r["viol"].any()
        for k in range(3):
            close(r["P"]["L2d"][k], want[k], 1e-12)
            close(r["p3"][k], want[k], 1e-12)
            close(r["e3"][k], want[k], 1e-5)       # chords of arcs, 0.05 m apart
        assert r["rel"].max() < cs.TOL


def test_plane_straight_and_arc():
    x, y = grid()
    gx, gy = 0.12, -0.05
    z = 100 + gx * x + gy * y
    h = 0.7
    g = gx * math.cos(h) + gy * math.sin(h)
    for xp in XPS:
        r = cs.pair(routes([(40.2, 200.1, h, [150.0], []), (100.5, 100.5, h, [60.0, 60.0], [math.radians(120)])], R=20.0),
                    ground(xp, z))
        close(r["p3"][0], 150.0 * math.sqrt(1 + g * g), 1e-12)
        close(r["e3"][0], 150.0 * math.sqrt(1 + g * g), 1e-10)
        # the bend: straights exact, arc by a fine midpoint sum of sqrt(1 + (grad . t)^2)
        t = 20.0 * math.tan(math.radians(60))
        phi = h + (np.arange(200000) + 0.5) / 200000 * math.radians(120)
        arc = 20.0 * math.radians(120) * np.mean(np.sqrt(1 + (gx * np.cos(phi) + gy * np.sin(phi)) ** 2))
        g2 = gx * math.cos(h + math.radians(120)) + gy * math.sin(h + math.radians(120))
        want = (60 - t) * math.sqrt(1 + g * g) + arc + (60 - t) * math.sqrt(1 + g2 * g2)
        close(r["p3"][1], want, 1e-9)
        close(r["e3"][1], want, 1e-5)


def test_ridge_kink_on_grid_line():
    x, y = grid()
    z = 50 + 0.3 * np.abs(x - 200) + 0.02 * y                 # kink on x = 200, a grid line
    for xp in XPS:
        h = 0.2
        r = cs.pair(routes([(120.37, 100.2, h, [170.0], [])]), ground(xp, z))
        ux, uy = math.cos(h), math.sin(h)
        a = (200 - 120.37) / ux                                # chainage at the kink
        want = a * math.hypot(1, -0.3 * ux + 0.02 * uy) + (170 - a) * math.hypot(1, 0.3 * ux + 0.02 * uy)
        close(r["p3"][0], want, 1e-12)
        close(r["e3"][0], want, 1e-6)


def test_short_legs_shrink_fillet():
    R = 3.185
    for xp in XPS:
        r = cs.pair(routes([(100, 100, 0, [2.0, 50.0], [math.radians(90)]),          # leg 2 m: r = 2
                            (100, 100, 0, [40.0, 3.0, 40.0], [math.radians(90), math.radians(90)]),  # shared 1.5 m
                            (100, 100, 0, [40.0, 40.0], [math.radians(90)])], R), ground(xp, np.full((301, 301), 7.0)))
        close(r["r"][0, 0], 2.0, 1e-12)
        close(r["r"][1, 0], 1.5, 1e-12); close(r["r"][1, 1], 1.5, 1e-12)
        assert r["viol"][0, 0] and r["viol"][1, :2].all() and not r["viol"][2].any()
        close(r["p3"][0], 52.0 - 4.0 + math.pi, 1e-12)
        assert r["rel"].max() < cs.TOL


def hilly(tmp_path, n=601, seed=5):
    rng = np.random.default_rng(seed)
    x, y = grid(n)
    z = 120 + 0.03 * x + 4 * np.sin(x / 37) * np.cos(y / 23) + rng.normal(0, 0.05, x.shape)
    np.save(tmp_path / "source.npy", z)
    json.dump({"origin_e_m": OE, "origin_n_m": ON, "spacing_m": 1}, open(tmp_path / "source.json", "w"))


def test_sweep_hilly(tmp_path):
    hilly(tmp_path)
    for gpu in [False] + ([True] if cs.cp is not None else []):
        r = cs.sweep(str(tmp_path), routes=300, seconds=60, batch=120, use_gpu=gpu, witness=16, cables=CABLES)
        c = r["counts"]
        assert c["routes"] == 300 and all(c["by_bends"][str(b)] > 0 for b in range(5))
        assert r["photons_total"] == 0 and r["witness"]["photons"] == 0, r["disagreement"]["worst"][:2]
        assert r["plan_check"]["max_rel_chords_vs_analytic"] < 1e-4
        for kv in ("33kV", "132kV"):
            e = r["extra_3d_over_plan"][kv]
            assert e["routes"] > 0 and e["cable_3d_km"] >= e["plan_km"] and e["extra_pct_percentiles_0_5_25_50_75_95_100"][0] >= 0
        assert r["bend_radius"]["bends_below_min"] > 0 and len(r["script_sha256"]) == 64
        path, _ = cs.write_receipt(str(tmp_path), r)
        raw = open(path, "rb").read()
        assert b"\r\n" not in raw and raw.endswith(b"\n") and json.loads(raw)["seed"] == cs.SEED


def test_bounded_by_seconds(tmp_path):
    np.save(tmp_path / "source.npy", np.full((401, 401), 50.0))
    json.dump({"origin_e_m": OE, "origin_n_m": ON, "spacing_m": 1}, open(tmp_path / "source.json", "w"))
    r = cs.sweep(str(tmp_path), routes=10 ** 6, seconds=0.0, batch=7, use_gpu=False, witness=4, cables=CABLES)
    assert r["counts"]["stopped_by"] == "seconds" and r["counts"]["routes"] == 7
    assert r["disagreement"]["max_rel"] < 1e-5          # flat: 3D is the plan


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
