"""Tests for src/sun_data.py and src/sun_cells.py on synthetic data (no network).

The two sun algorithms must agree and hit known noon elevations; a flat horizon shades nothing; a 90 degree
wall shades everything; a 30 degree ring leaves December dark and June lit; the witness must agree with the
GPU; tiles and the TMY file must round-trip; the PVGIS parser must count sunshine hours by the WMO rule.

    E:/swarm/gpu-bench/venv/Scripts/python.exe tests/test_sun.py
"""
import hashlib, inspect, json, os, pathlib, sys, tempfile, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sun_cells as sc  # noqa: E402
import sun_data as sd  # noqa: E402

GPU = [False] + ([True] if sc.cp is not None else [])
LAT, LON = 51.79, -2.0


def fake_tmy(dni=500.0, dfu=100.0):
    """8760 hours in PVGIS order; every daylight hour gets the same beam, so shares are easy to reason about."""
    doy, hr, _ = sc.hour_table(None, 0.0)
    _, el = sc.noaa(doy, hr, LAT, LON)
    s = np.clip(np.sin(np.radians(el)), 0, None)
    b = np.where(el > 0, dni, 0.0)
    return np.stack([b * s + dfu * (el > 0), b, dfu * (el > 0)], 1)


def fake_pvgis(tmy):
    rows = []
    for i in range(sd.HOURS):
        doy = i // 24; m = int(np.searchsorted(sc.MONTH_START, doy, side="right"))
        d = doy - sc.MONTH_START[m - 1] + 1
        rows.append({"time(UTC)": f"2015{m:02d}{d:02d}:{i % 24:02d}00", "G(h)": tmy[i, 0], "Gb(n)": tmy[i, 1],
                     "Gd(h)": tmy[i, 2]})
    return {"inputs": {"location": {"latitude": LAT, "longitude": LON, "elevation": 200.0, "irradiance_time_offset": 0.1834},
                       "meteo_data": {"radiation_db": "PVGIS-SARAH3", "meteo_db": "ERA5", "year_min": 2005,
                                      "year_max": 2023, "use_horizon": False, "horizon_db": None}},
            "outputs": {"months_selected": [{"month": m, "year": 2015} for m in range(1, 13)], "tmy_hourly": rows}}


def test_two_suns_agree_and_hit_known_noons():
    doy = np.arange(1, 366, dtype=float); hr = np.full(365, 12.0)
    a1, e1 = sc.noaa(doy, hr, LAT, 0.0); a2, e2 = sc.michalsky(doy, hr, LAT, 0.0)
    assert np.abs(e1 - e2).max() < 0.4, np.abs(e1 - e2).max()
    # noon elevation = 90 - lat + declination: June solstice about +23.44, December about -23.44
    assert abs(e2[171] - (90 - LAT + 23.44)) < 0.3 and abs(e2[354] - (90 - LAT - 23.44)) < 0.3
    assert 170 < a2[171] < 190 and 170 < a1[354] < 190          # due south at noon


def test_clear_dni_is_physical():
    el = np.array([-1.0, 1.0, 10.0, 30.0, 60.0, 90.0])
    d = sc.clear_dni(el, 172, 0.1)
    assert d[0] == 0 and np.all(np.diff(d[1:]) > 0) and 800 < d[-1] < 1100, d


def run_ring(deg, gpu, shape=(3, 3)):
    h = np.full(shape + (sc.NAZ,), float(deg))
    return sc.run(h, fake_tmy(), 0.1834, LAT, LON, 0.0, 0.2, use_gpu=gpu, witness_cells=4, seconds=120)


def test_flat_horizon_shades_nothing():
    for gpu in GPU:
        rec, e, eb, pb, nod = run_ring(-5.0, gpu)
        assert np.allclose(e, eb[None, None]), gpu
        assert rec["photons"] == 0 and rec["witness"]["photons"] == 0, rec
        assert abs(rec["base_relative_diff"][1]) < 0.01, rec["base_relative_diff"]


def test_wall_shades_everything():
    for gpu in GPU:
        rec, e, eb, pb, nod = run_ring(89.0, gpu)
        assert e.max() == 0 and rec["photons"] == 0 and rec["witness"]["photons"] == 0, rec


def test_thirty_degree_ring_december_dark_june_lit():
    for gpu in GPU:
        rec, e, eb, pb, nod = run_ring(30.0, gpu)
        assert e[1, 1, 11, 0] == 0 and e[1, 1, 0, 0] == 0          # noon sun about 15 degrees in Dec and Jan
        assert e[1, 1, 5, 0] > 200 and e[1, 1, 5, 1] > 0.5 * eb[5, 1]
        assert rec["witness"]["photons"] == 0


def test_nan_horizon_is_nodata_and_never_shades():
    h = np.full((3, 3, sc.NAZ), 10.0); h[0, 0, 16] = np.nan          # due south unknown at one cell
    for gpu in GPU:
        rec, e, eb, pb, nod = sc.run(h, fake_tmy(), 0.1834, LAT, LON, 0.0, 0.2, use_gpu=gpu, witness_cells=4)
        assert nod[0, 0] and nod.sum() == 1 and rec["witness"]["photons"] == 0
        assert e[0, 0, 5, 0] >= e[1, 1, 5, 0]


def test_gsc_tiles_round_trip(tmp):
    n = 2 * 64 + 1
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 1, (n, n, 12, sc.NQ)); base = np.ones((12, sc.NQ)); nod = np.zeros((n, n), bool); nod[5, 7] = True
    q = sc.quantise(vals, base[None, None], nod)
    idx = sc.write_tiles(str(tmp), q, 400000, 200000, dict(site=dict(name="t")))
    assert len(idx["tiles"]) == 4
    raw = (tmp / "sun-cells.json").read_bytes(); assert b"\r\n" not in raw
    for t in idx["tiles"]:
        blob = (tmp / t["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == t["sha256"]
        head, arr = sc.decode(blob)
        ix, iy = (int(v) for v in t["key"].split("_"))
        assert np.array_equal(arr, q[iy * 64:iy * 64 + 65, ix * 64:ix * 64 + 65])
        assert head["origin_e_m"] == 400000 + 256 * ix
    ok = q != sc.NODATA_U8
    assert np.abs(q[ok] / sc.SCALE - vals[ok]).max() <= 0.5 / sc.SCALE + 1e-12
    assert q[5, 7, 0, 0] == sc.NODATA_U8


def test_pvgis_parse_and_site_files(tmp):
    tmy = fake_tmy(dni=150.0)
    tmy[:, 1] = np.where(np.arange(sd.HOURS) % 2 == 0, tmy[:, 1], np.minimum(tmy[:, 1], 119.0))
    d = fake_pvgis(tmy)
    cache = tmp / "cache.json"; cache.write_text(json.dumps(d))
    vals, month, hour, off, meta = sd.parse_tmy(d)
    assert off == 0.1834 and month[0] == 1 and month[-1] == 12
    doc, path = sd.build(str(tmp / "site"), 400000, 210000, d, str(cache))
    want = [int(((month == m) & (tmy[:, 1] >= 120)).sum()) for m in range(1, 13)]
    assert doc["monthly"]["sunshine_hours"] == want, (doc["monthly"]["sunshine_hours"], want)
    raw = open(path, "rb").read(); assert b"\r\n" not in raw
    blob = open(os.path.join(os.path.dirname(path), "tmy-hourly.bin"), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == doc["hourly"]["sha256"]
    back, off2 = sd.decode_tmy(blob)
    assert abs(off2 - 0.1834) < 1e-4 and np.abs(back - tmy).max() <= 0.5
    assert abs(doc["grid_azimuth_of_true_north_deg"]) < 0.01            # 2 W is the central meridian
    assert "no restrictions" in doc["licence"]["text"] and "Joint Research Centre" in doc["attribution"]


def test_parse_rejects_short_tmy():
    d = fake_pvgis(fake_tmy()); d["outputs"]["tmy_hourly"] = d["outputs"]["tmy_hourly"][:100]
    try:
        sd.parse_tmy(d)
    except ValueError:
        return
    raise AssertionError("short TMY accepted")


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
    print("failures:", fails)
    sys.exit(1 if fails else 0)
