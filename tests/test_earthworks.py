"""Tests for src/earthworks_pair.py on synthetic planes with known volumes.

On a plane the floor (centreline ground less d) makes cut = d + c t across the trench, so a
straight trench holds exactly L w d, and a bend of curvature kappa holds, by Pappus,
L (w d - kappa c w^3 / 12). Both channels must hit the known volume and each other.

    python lidar/tests/test_earthworks.py
"""
import inspect, json, math, os, pathlib, sys, tempfile, traceback
import numpy as np

try:
    import pytest
except ImportError:  # no pytest in the GPU venv: run this file directly, see the bottom
    pytest = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import earthworks_pair as ew  # noqa: E402

XPS = [np] + ([ew.cp] if ew.cp is not None else [])
OE, ON = 400000.0, 200000.0


def plane(xp, a=100.0, gx=0.0, gy=0.0, n=301):
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    return ew.Ground(a + gx * x + gy * y, OE, ON, 1.0, xp)


def close(v, want, rel=2e-3):
    assert abs(v - want) <= rel * want, (v, want)


def one(G, *a):
    e, p, d, _ = ew.trench_pair(ew.routes(*a, oe=OE, on=ON), G)
    return e[0], p[0], d[0]


def test_flat_straight():
    for xp in XPS:
        e, p, d = one(plane(xp), OE + 50.3, ON + 60.7, 0.61, 57.0, 0, 0, 0, 0.6, 1.5)
        close(e, 57 * 0.6 * 1.5); close(p, 57 * 0.6 * 1.5, 1e-9); assert d < ew.TOL


def test_sloped_along_and_across():
    for xp in XPS:
        G = plane(xp, gx=0.08, gy=-0.05)          # 8 % east, 5 % south: along and across any heading
        for h in (0.0, 0.4, 1.9, 4.0):
            e, p, d = one(G, OE + 150, ON + 150, h, 70.0, 0, 0, 0, 0.9, 1.2)
            close(e, 70 * 0.9 * 1.2); close(p, 70 * 0.9 * 1.2, 1e-9); assert d < ew.TOL


def test_bent_flat_and_pappus():
    for xp in XPS:
        for th in (math.radians(90), -math.radians(60)):
            L = 30 + 20 * abs(th) + 25
            e, p, d = one(plane(xp), OE + 120, ON + 100, 0.3, 30, th, 20, 25, 1.2, 1.0)
            close(e, L * 1.2); close(p, L * 1.2, 1e-9); assert d < ew.TOL
        # ground rising north (c = +0.3 on the left of an eastward trench); the bend is left, radius 8
        G = plane(xp, gy=0.3); w, dd, R, th = 1.2, 1.0, 8.0, math.pi / 2
        P = ew.routes(OE + 100, ON + 100, 0.0, 0.0, th, R, 0.0, w, dd, oe=OE, on=ON)
        e, p, d, _ = ew.trench_pair(P, G)
        # c varies round the bend: c(phi) = 0.3 cos(phi) on the left normal, integrate exactly
        want = R * th * w * dd - (1 / R) * 0.3 * w ** 3 / 12 * R * math.sin(th)
        close(p[0], want, 1e-5); close(e[0], want); assert d[0] < ew.TOL


def test_platform_on_plane():
    for xp in XPS:
        G = plane(xp, gx=0.1)                     # z = 100 + 0.1 x
        poly = [(OE + 100, ON + 100), (OE + 120, ON + 100), (OE + 120, ON + 130), (OE + 100, ON + 130)]
        (ec, ef), (pc, pf) = ew.platform_pair(poly, 111.0, G)   # level crosses at x = 110
        want = 0.5 * 10 * 1.0 * 30                # triangle wedge 10 m x 1 m x 30 m either side
        close(ec, want); close(ef, want); close(pc, want, 1e-6); close(pf, want, 1e-6)
        rot = [(OE + 150 + 20 * math.cos(a), ON + 150 + 20 * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1] + 0.2]
        (ec, ef), (pc, pf) = ew.platform_pair(rot, 115.0, G)   # a hexagon: the channels must agree
        close(pc, ec, ew.TOL); close(pf, ef, ew.TOL)


def test_sweep_receipt(tmp_path):
    rng = np.random.default_rng(3)
    x, y = np.meshgrid(np.arange(401.0), np.arange(401.0))
    z = 120 + 0.03 * x + 4 * np.sin(x / 37) * np.cos(y / 23) + rng.normal(0, 0.05, x.shape)
    np.save(tmp_path / "source.npy", z)
    json.dump({"origin_e_m": OE, "origin_n_m": ON, "spacing_m": 1}, open(tmp_path / "source.json", "w"))
    for gpu in [False] + ([True] if ew.cp is not None else []):
        r = ew.sweep(str(tmp_path), trenches=60, seconds=30, batch=25, use_gpu=gpu)
        assert r["counts"]["trenches"] == 60 and r["counts"]["bent"] > 0 and r["counts"]["straight"] > 0
        assert r["witness"]["photons"] == 0 and r["platform"]["photons"] == 0
        assert r["disagreement"]["max_rel"] < ew.TOL, r["disagreement"]["worst"][:2]
        assert len(r["script_sha256"]) == 64 and r["seed"] == ew.SEED


def test_bounded_by_seconds(tmp_path):
    np.save(tmp_path / "source.npy", np.full((201, 201), 50.0))
    json.dump({"origin_e_m": OE, "origin_n_m": ON, "spacing_m": 1}, open(tmp_path / "source.json", "w"))
    r = ew.sweep(str(tmp_path), trenches=10 ** 6, seconds=0.0, batch=5, use_gpu=False)
    assert r["counts"]["stopped_by"] == "seconds" and r["counts"]["trenches"] == 5


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
