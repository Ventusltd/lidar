# SPDX-License-Identifier: Apache-2.0
"""Tests for src/piles_sweep.py on synthetic ground with known reveals.

Flat ground: every reveal nominal. A plane inside the along-row limit: the table follows it. A plane over the
limit: slope held at the limit, spread = excess slope x span, centred on the window. The tile channel reads
the same ground as the source to within the 1 cm rounding of the .ght format, and on NumPy and CuPy alike.

    python tests/test_piles_sweep.py
"""
import inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import piles_sweep as ps  # noqa: E402
from cut_tiles import cut_grid  # noqa: E402

XPS = [np] + ([ps.cp] if ps.cp is not None else [])
OE, ON = 400000, 200000


def site(tmp, z):
    np.save(tmp / "source.npy", z)
    json.dump({"origin_e_m": OE, "origin_n_m": ON, "spacing_m": 1}, open(tmp / "source.json", "w", encoding="utf-8", newline="\n"))
    cut_grid(z, OE, ON, str(tmp), "synthetic")
    return str(tmp)


def one_row(sysid, x0, y0, ux, uy, L, sp):
    rng = np.random.default_rng(0)
    D = ps.draw(rng, 1, 513, 513)
    npile = int(np.floor(L / sp + 1e-9)) + 1
    k = np.arange(ps.K)[None]
    over = (L - (npile - 1) * sp) / 2
    name = ps.SYSTEMS[sysid]
    D.update(sys=np.array([sysid]), L=np.array([L]), sp=np.array([sp]), x0=np.array([x0]), y0=np.array([y0]),
             ux=np.array([ux]), uy=np.array([uy]), mask=k < npile, s=np.where(k < npile, over + k * sp, 0.0),
             rmin=np.array([ps.REVEAL[name]["min"]]), rnom=np.array([ps.REVEAL[name]["nominal"]]),
             rmax=np.array([ps.REVEAL[name]["max"]]), lim=np.array([ps.ALONG[name]]))
    return D


def grid(z0=100.0, gx=0.0, gy=0.0, n=513):
    X, Y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    return z0 + gx * X + gy * Y


def test_flat_reveal_nominal(tmp_path):
    t = site(tmp_path, grid())
    for xp in XPS:
        G = ps.ground_from(t, xp)
        E = ps.evaluate(one_row(0, 100, 50, 0, 1, 60, 7), G, xp)
        r = E["reveal"][0][~np.isnan(E["reveal"][0])]
        assert len(r) == 9 and np.allclose(r, 1.4, atol=1e-12) and E["feasible"][0]


def test_plane_inside_limit_followed(tmp_path):
    t = site(tmp_path, grid(gy=0.06))
    for xp in XPS:
        E = ps.evaluate(one_row(0, 100, 50, 0, 1, 90, 6), ps.ground_from(t, xp), xp)
        assert abs(E["b"][0] - 0.06) < 1e-12
        assert np.allclose(E["reveal"][0][:16], 1.4, atol=1e-9)


def test_plane_over_limit_held_and_centred(tmp_path):
    t = site(tmp_path, grid(gy=0.14))
    for xp in XPS:
        E = ps.evaluate(one_row(0, 100, 50, 0, 1, 60, 7), ps.ground_from(t, xp), xp)
        r = E["reveal"][0][:9]
        assert abs(E["b"][0] - 0.10) < 1e-12 and not E["feasible"][0]
        assert abs((r.max() - r.min()) - 0.04 * 56) < 1e-9 and abs((r.max() + r.min()) / 2 - 1.4) < 1e-9
        assert abs(E["cut"][0].sum() - E["fill"][0].sum()) < 1e-9 and E["cut"][0].sum() > 0


def test_fixed_row_east_west(tmp_path):
    t = site(tmp_path, grid(gx=0.12))
    for xp in XPS:
        E = ps.evaluate(one_row(1, 100, 50, 1, 0, 25, 5), ps.ground_from(t, xp), xp)
        assert abs(E["b"][0] - 0.12) < 1e-12 and np.allclose(E["reveal"][0][:6], 1.0, atol=1e-9)


def test_tiles_match_source_within_rounding(tmp_path):
    rng = np.random.default_rng(3)
    z = 150 + np.cumsum(rng.normal(0, 0.05, (513, 513)), 0) + 0.02 * np.arange(513)[None]
    t = site(tmp_path, z)
    for xp in XPS:
        Ge, Gp = ps.ground_from(t, xp), ps.TileGround(t, OE, ON, xp)
        x, y = xp.asarray(rng.uniform(0, 512, 5000)), xp.asarray(rng.uniform(0, 512, 5000))
        d = ps.to_host(xp.abs(Ge.h(x, y) - Gp.h(x, y)))
        assert d.max() <= 0.005 + 1e-9
        # a point on a tile seam reads the same from either side
        west = float(Gp.h(xp.asarray([256.0 - 1e-12]), xp.asarray([10.3]))[0])
        east = float(Gp.h(xp.asarray([256.0]), xp.asarray([10.3]))[0])
        assert abs(west - east) < 1e-9


def test_sweep_receipt_and_witness(tmp_path):
    rng = np.random.default_rng(5)
    z = 150 + 3 * np.sin(np.arange(513)[None] / 40.0) + 2 * np.cos(np.arange(513)[:, None] / 55.0) + rng.normal(0, 0.02, (513, 513))
    t = site(tmp_path, z)
    r, first, Ec = ps.sweep(t, rows=3000, seconds=60, batch=1000, use_gpu=ps.cp is not None, witness=32)
    assert r["counts"]["rows"] == 3000 and r["witness"]["photons"] == 0 and r["photons_total"] == 0
    assert r["disagreement"]["max_ground_diff_m"] <= 0.0051
    for name in ps.SYSTEMS:
        e = r["by_system"][name]
        assert e["rows"] > 0 and sum(e["histogram"]["direct"]) == e["piles"] == sum(e["histogram"]["tiles"])
        assert e["ks_max_cdf_diff"] < 0.05
    fx = ps.fixture(first, Ec, 3)
    assert len(fx["rows"]) == 6 and all(len(x["s"]) == len(x["reveal"]) for x in fx["rows"])
    path = tmp_path / "r.json"
    ps.write_json(str(path), r)
    raw = open(path, "rb").read()
    assert b"\r\n" not in raw and raw.endswith(b"\n") and json.loads(raw)["seed"] == ps.SEED


def test_bounded_by_seconds(tmp_path):
    t = site(tmp_path, grid())
    r, _, _ = ps.sweep(t, rows=10 ** 6, seconds=0.0, batch=7, use_gpu=False, witness=4)
    assert r["counts"]["stopped_by"] == "seconds" and r["counts"]["rows"] == 7


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
