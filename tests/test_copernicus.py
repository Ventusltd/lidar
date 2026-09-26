"""Copernicus GLO-30 path: naming, OS grid maths, byte-range COG reads, BNG tiles, and the pair.

No network: synthetic COGs shaped like the real ones (3600 x 2400 float32, 1024 x 1024 DEFLATE
blocks, floating-point predictor, PixelIsPoint, EPSG:4326) are served from memory by a fake
range fetcher that counts every request.

Run: E:/swarm/gpu-bench/venv/Scripts/python.exe -m pytest tests/test_copernicus.py
"""
import io
import json
import os
import struct
import sys
import tempfile
import warnings

import numpy as np
import pytest
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
warnings.filterwarnings("ignore")
import copernicus as cop  # noqa: E402
import osgb  # noqa: E402
from cut_tiles import cut_grid, decode_tile  # noqa: E402

TOL_Q = 0.005 + 1e-9


def field(lat, lon):
    """A surface linear in lat/lon, so bilinear interpolation of it is exact."""
    return 150.0 + 900.0 * (np.asarray(lat) - 51.5) - 400.0 * (np.asarray(lon) + 2.0)


def synthetic_cog(lat_i, lon_i, f=field):
    rows, cols = 3600, 2400
    dlat, dlon = 1 / 3600, 1.5 / 3600
    lat = (lat_i + 1) - np.arange(rows) * dlat
    lon = lon_i + np.arange(cols) * dlon
    z = f(lat[:, None], lon[None, :]).astype(np.float32)
    geokeys = (1, 1, 0, 4, 1024, 0, 1, 2, 1025, 0, 1, 2, 2048, 0, 1, 4326, 2054, 0, 1, 9102)
    buf = io.BytesIO()
    tifffile.imwrite(buf, z, tile=(1024, 1024), compression="deflate", predictor=3,
                     extratags=[(33550, "d", 3, (dlon, dlat, 0.0)),
                                (33922, "d", 6, (0.0, 0.0, 0.0, float(lon_i), float(lat_i + 1), 0.0)),
                                (34735, "H", len(geokeys), geokeys)])
    return buf.getvalue()


class FakeBucket:
    def __init__(self, tiles):
        self.files = {cop.tile_url(cop.tile_name(a, b)): synthetic_cog(a, b) for a, b in tiles}
        self.calls = []

    def __call__(self, url, start, end):
        self.calls.append((url, start, end))
        blob = self.files.get(url)
        if blob is None:
            return None, None
        return blob[start:end], len(blob)


@pytest.fixture(scope="module")
def bucket():
    return FakeBucket([(51, -3), (51, -2)])


def test_tile_names():
    assert cop.tile_name(51, -3) == "Copernicus_DSM_COG_10_N51_00_W003_00_DEM"
    assert cop.tile_name(-1, 0) == "Copernicus_DSM_COG_10_S01_00_E000_00_DEM"
    assert cop.tile_name(0, 6) == "Copernicus_DSM_COG_10_N00_00_E006_00_DEM"
    assert cop.tile_url(cop.tile_name(55, -4)).endswith(
        "Copernicus_DSM_COG_10_N55_00_W004_00_DEM/Copernicus_DSM_COG_10_N55_00_W004_00_DEM.tif")
    assert cop.tiles_for_box(51.2, -2.1, 51.9, -1.9) == [(51, -3), (51, -2)]


def test_longitude_bands():
    assert cop.lon_step_arcsec(49) == 1.0 and cop.lon_step_arcsec(50) == 1.5
    assert cop.lon_step_arcsec(59) == 1.5 and cop.lon_step_arcsec(60) == 2.0
    assert cop.lon_step_arcsec(-50) == 1.0 and cop.lon_step_arcsec(-51) == 1.5


def test_os_worked_example():
    # OS guide v3.6 Annex C.1: 52 39 27.2531 N, 1 43 4.5177 E (OSGB36) -> E 651409.903 N 313177.270
    lat = np.radians(52 + 39 / 60 + 27.2531 / 3600)
    lon = np.radians(1 + 43 / 60 + 4.5177 / 3600)
    e, n = osgb.tm_forward(lat, lon)
    assert abs(e - 651409.903) < 1e-3 and abs(n - 313177.270) < 1e-3
    la, lo = osgb.tm_inverse(651409.903, 313177.270)
    assert abs(la - lat) < 1e-9 and abs(lo - lon) < 1e-9  # ~6 mm


def test_round_trip_and_witness():
    e, n = np.meshgrid(np.linspace(100000, 650000, 12), np.linspace(20000, 1200000, 12))
    lat, lon = osgb.bng_to_wgs84(e, n)
    e2, n2 = osgb.wgs84_to_bng(lat, lon)
    assert np.max(np.hypot(e2 - e, n2 - n)) < 1e-3
    try:
        from pyproj import Transformer
    except ImportError:
        pytest.skip("pyproj not installed: independent witness unavailable")
    wlon, wlat = Transformer.from_crs(27700, 4326, always_xy=True).transform(e, n)
    # pyproj may use OSTN15 where installed; the Helmert route is good to ~3.5 m, allow 5 m
    k = np.cos(np.radians(lat)) * 111320.0
    assert np.max(np.hypot((wlon - lon) * k, (wlat - lat) * 111320.0)) < 5.0


def test_range_read_matches_full_and_fetches_little(bucket, tmp_path):
    rc = cop.RangeCache(str(tmp_path), fetch=bucket)
    bucket.calls.clear()
    m = cop.read_box(51.80, -2.90, 51.82, -2.85, rc=rc)
    name = cop.tile_name(51, -3)
    full = tifffile.imread(io.BytesIO(bucket.files[cop.tile_url(name)])).astype(np.float64)
    i0 = round((52 - m["lat_top"]) * 3600)
    j0 = round((m["lon_left"] + 3) / (1.5 / 3600))
    h, w = m["z"].shape
    assert np.array_equal(m["z"], full[i0:i0 + h, j0:j0 + w])
    size = len(bucket.files[cop.tile_url(name)])
    got = sum(e - s for _, s, e in bucket.calls)
    assert len(bucket.calls) <= 3 and got < size / 3, (bucket.calls, size)
    n_before = len(bucket.calls)
    cop.read_box(51.80, -2.90, 51.82, -2.85, rc=rc)
    assert len(bucket.calls) == n_before  # all from the disk cache


def test_mosaic_joins_tiles_and_sea(tmp_path):
    fb = FakeBucket([(51, -3)])                     # W002 missing = open sea
    m = cop.read_box(51.5, -2.01, 51.51, -1.99, rc=cop.RangeCache(str(tmp_path), fetch=fb))
    assert m["tiles"] == [cop.tile_name(51, -3)]
    assert m["sea_tiles"] == [cop.tile_name(51, -2)]
    lon = m["lon_left"] + np.arange(m["z"].shape[1]) * m["dlon"]
    assert np.all(m["z"][:, lon >= -2.0 + 1e-9] == 0.0)
    assert not np.isnan(m["z"]).any()


def test_band_edge_refused(bucket, tmp_path):
    with pytest.raises(NotImplementedError):
        cop.read_box(49.9, -3.0, 50.1, -2.9, rc=cop.RangeCache(str(tmp_path), fetch=bucket))


def test_spacing_is_honest(tmp_path):
    for bad in (1, 10, 29, 30.5, 70):
        with pytest.raises(ValueError):
            cop.build("x", 400000, 210000, 2048, spacing=bad, out_dir=str(tmp_path))


@pytest.fixture(scope="module")
def built(bucket):
    d = tempfile.mkdtemp(prefix="cop-test-")
    rc = cop.RangeCache(os.path.join(d, "cache"), fetch=bucket)
    res = cop.build("synthetic-01", 400128, 209920, 2048, out_dir=os.path.join(d, "cop"), rc=rc,
                    log=lambda *a: None)
    return d, res


def test_build_tiles_are_the_field(built):
    d, res = built
    man = res["manifest"]
    assert man["tile_m"] == 8192 and man["spacing_m"] == 32 and len(man["tiles"]) == 1
    blob = open(os.path.join(d, "cop", man["tiles"][0]["file"]), "rb").read()
    assert struct.unpack_from("<H", blob, 8)[0] == 32000
    head, h = decode_tile(blob)
    e, n = np.meshgrid(head["origin_e_m"] + 32.0 * np.arange(257),
                       head["origin_n_m"] + 32.0 * np.arange(257))
    lat, lon = osgb.bng_to_wgs84(e, n)
    assert np.max(np.abs(h - field(lat, lon))) <= TOL_Q + 1e-4  # float32 source adds <0.1 mm


def test_index_carries_datum_accuracy_and_notice(built):
    d, _ = built
    raw = open(os.path.join(d, "cop", "tiles.json"), "rb").read()
    assert b"\r\n" not in raw
    man = json.loads(raw.decode("utf-8"))
    assert man["attribution"] == (
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
        "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all "
        "rights reserved")
    assert man["surface_model"] is True and "EGM2008" in man["datum"]["vertical"]
    assert man["accuracy"]["vertical_absolute_le90_m"] == 4.0
    assert man["citation"] == "https://doi.org/10.5270/ESA-c5d3d65"


def test_pair_finds_known_offset(built):
    """EA stand-in = the same field on the 1 m grid, 1.5 m lower: the pair must report +1.5 m."""
    import copernicus_pair as cpair
    d, res = built
    e0, n0 = 399104, 208896
    e, n = np.meshgrid(e0 + np.arange(2049.0), n0 + np.arange(2049.0))
    lat, lon = osgb.bng_to_wgs84(e, n)
    cut_grid(field(lat, lon) - 1.5, e0, n0, os.path.join(d, "ea"), "synthetic-01")
    for gpu in (True, False):
        rec = cpair.run(os.path.join(d, "ea"), os.path.join(d, "cop"), use_gpu=gpu,
                        cache_dir=os.path.join(d, "cache"), log=lambda *a: None)
        assert rec["photons_total"] == 0, rec["photons"]
        s = rec["difference_block_mean"]
        assert abs(s["percentiles_m"]["50"] - 1.5) < 0.02 and s["std_m"] < 0.02
        assert rec["nodes"] > 3000


if __name__ == "__main__":  # the repo runs tests as scripts: python tests/test_copernicus.py
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
