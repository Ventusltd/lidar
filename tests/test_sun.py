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


def write_fake_haduk(path, x0=398500.0, y0=208500.0, n=6, bad=None):
    """A tiny HadUK-shaped netCDF: sun = 10 month + i_x + 10 i_y, linear in e and n, so bilinear is exact."""
    import netCDF4
    d = netCDF4.Dataset(str(path), "w")
    d.createDimension("time", 12); d.createDimension("projection_y_coordinate", n)
    d.createDimension("projection_x_coordinate", n)
    d.setncattr("source", "HadUK-Grid_v1.3.2.0"); d.setncattr("version", "v20260512"); d.setncattr("lta_period", "1991-2020")
    gm = d.createVariable("transverse_mercator", "i4"); gm.grid_mapping_name = "transverse_mercator"
    for k, v in sd.HADUK_BNG.items():
        gm.setncattr(k, v)
    x = d.createVariable("projection_x_coordinate", "f8", ("projection_x_coordinate",)); x[:] = x0 + 1000 * np.arange(n)
    y = d.createVariable("projection_y_coordinate", "f8", ("projection_y_coordinate",)); y[:] = y0 + 1000 * np.arange(n)
    m = d.createVariable("month_number", "i8", ("time",)); m[:] = np.arange(1, 13)
    s = d.createVariable("sun", "f8", ("time", "projection_y_coordinate", "projection_x_coordinate"), fill_value=1e20)
    s.units = "hour"; s.grid_mapping = "transverse_mercator"
    val = np.arange(1, 13)[:, None, None] * 10.0 + np.arange(n)[None, None, :] + 10.0 * np.arange(n)[None, :, None]
    val = np.ma.masked_array(val, np.zeros(val.shape, bool))
    if bad:
        val[:, bad[0], bad[1]] = np.ma.masked
    s[:] = val
    d.close()


def test_haduk_bilinear_is_exact_on_a_linear_field(tmp):
    write_fake_haduk(tmp / "h.nc")
    with sd.HadUK(str(tmp / "h.nc")) as h:
        _haduk_linear(h)


def _haduk_linear(h):
    e, n = 400250.0, 210750.0                    # 1.75 cells east, 2.25 cells north of the first centre
    vals, rec = sd.bilinear_sample(h.x, h.y, h.window, e, n)
    want = np.arange(1, 13) * 10.0 + 1.75 + 22.5
    assert np.allclose(vals, want), (vals, want)
    assert not rec["renormalised"] and abs(sum(c["weight"] for c in rec["cells"]) - 1) < 1e-9
    assert sorted((c["e"], c["n"]) for c in rec["cells"]) == [(399500, 210500), (399500, 211500), (400500, 210500), (400500, 211500)]
    vals, _ = sd.bilinear_sample(h.x, h.y, h.window, 399500.0, 210500.0)     # on a centre: that cell's value
    assert np.allclose(vals, np.arange(1, 13) * 10.0 + 1 + 20)


def test_haduk_nodata_cell_renormalised_and_outside_rejected(tmp):
    write_fake_haduk(tmp / "h.nc", bad=(2, 1))                 # the (399500, 210500) cell is sea
    with sd.HadUK(str(tmp / "h.nc")) as h:
        _haduk_nodata(h)


def _haduk_nodata(h):
    vals, rec = sd.bilinear_sample(h.x, h.y, h.window, 400000.0, 211000.0)
    assert rec["renormalised"] and [c["valid"] for c in rec["cells"]].count(False) == 1
    assert np.allclose(vals, np.arange(1, 13) * 10.0 + (22 + 31 + 32) / 3), vals
    try:
        sd.bilinear_sample(h.x, h.y, h.window, 900000.0, 211000.0)
    except ValueError:
        return
    raise AssertionError("point outside the grid accepted")


def test_haduk_rejects_a_grid_that_is_not_bng(tmp):
    import netCDF4
    write_fake_haduk(tmp / "h.nc")
    d = netCDF4.Dataset(str(tmp / "h.nc"), "a"); d["transverse_mercator"].false_northing = 0.0; d.close()
    try:
        sd.HadUK(str(tmp / "h.nc")).close()
    except ValueError as e:
        assert "British National Grid" in str(e)
        return
    raise AssertionError("non-BNG grid accepted")


def test_site_file_carries_haduk_licence_ratio_and_preference(tmp):
    write_fake_haduk(tmp / "h.nc", x0=397500.0, y0=207500.0)
    d = fake_pvgis(fake_tmy()); cache = tmp / "cache.json"; cache.write_text(json.dumps(d))
    h = sd.HadUK(str(tmp / "h.nc"))
    doc, path = sd.build(str(tmp / "site"), 400000, 210000, d, str(cache), h)
    h.ds.close()
    hg = doc["haduk_grid"]
    want = np.arange(1, 13) * 10.0 + 2.5 + 25.0
    assert np.allclose(hg["monthly_sunshine_hours"], want, atol=0.05)
    pv = doc["monthly"]["sunshine_hours"]
    assert np.allclose(hg["compare_pvgis"]["ratio_monthly"], np.array(pv) / want, atol=1e-3)
    assert abs(hg["compare_pvgis"]["ratio_annual"] - sum(pv) / want.sum()) < 1e-3
    assert doc["preferred"]["sunshine_hours"] == "haduk_grid" and doc["preferred"]["irradiance"] == "pvgis"
    assert hg["licence"]["name"] == "Open Government Licence v3.0" and "open-government-licence/version/3" in hg["licence"]["url"]
    assert "Open Government Licence v3.0" in hg["attribution"] and "gridded by the Met Office" in hg["label"]
    assert hg["source"]["doi"] == "10.5285/789b3065d74a4c948ab05d33556c86d0" and "v1.3.2.ceda" in hg["source"]["citation"]
    assert "haduk_grid_sunshine" not in doc["not_used"]
    assert b"\r\n" not in open(path, "rb").read()


def test_real_haduk_file_is_bng_and_matches_owner_hash():
    if not os.path.exists(sd.HADUK_FILE):
        print("     (real HadUK file absent, skipped)"); return
    with sd.HadUK() as h:
        _haduk_real(h)


def _haduk_real(h):
    assert h.sha256.startswith("d1cda0d700368a29"), h.sha256
    assert h.x[0] == -199500 and h.y[0] == -199500 and h.attrs["source"] == "HadUK-Grid_v1.3.2.0"
    vals, rec = sd.bilinear_sample(h.x, h.y, h.window, 400000.0, 210000.0)
    assert not rec["renormalised"] and 1300 < vals.sum() < 1800 and vals[5] > vals[11] * 2, vals


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
