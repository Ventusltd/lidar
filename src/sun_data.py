"""Site sun climate: a typical year of hourly sunshine at each site centre, fetched once and kept small.

Source (checked 26 Sept 2026, see NOTICE.md):
  PVGIS 5.3 (European Commission, Joint Research Centre), typical meteorological year, radiation database
  PVGIS-SARAH3 (satellite-derived, so cloud is in it), meteo ERA5, horizon OFF (usehorizon=0) because our own
  1 m LiDAR horizon is applied cell by cell in sun_cells.py and PVGIS's coarse DEM horizon would count twice.
  API: https://re.jrc.ec.europa.eu/api/v5_3/tmy  (GET only, at most 30 calls/s per IP; we make one call per site,
  sequentially, with a named User-Agent, and cache the answer under E:/world-cache/sun/).
  Usage conditions: "The information provided by PVGIS is free and there are no restrictions on its use."

Also used (checked 26 Sept 2026, see NOTICE.md): Met Office HadUK-Grid v1.3.2.ceda, 1 km monthly 1991-2020
sunshine averages (station observations gridded by the Met Office), doi:10.5285/789b3065d74a4c948ab05d33556c86d0,
Open Government Licence v3.0. The owner downloads the file from CEDA (registered login) into
E:/world-cache/sun/haduk/; it is sampled bilinearly from the four nearest 1 km cell centres on British National Grid
(the file's own transverse_mercator grid mapping is checked against EPSG:27700). The web layer prefers HadUK-Grid for
sunshine hours and PVGIS for irradiance; the PVGIS/HadUK ratio is recorded per month.

Not used: Met Office Weather DataHub (forecasts and 48 h observations only, API key, own licence), Met Office UKV on
the AWS Open Data registry (forecast model output, rolling two years, CC BY-SA).
PVGIS sunshine hours are derived from the TMY by the WMO definition: direct normal irradiance of at least
120 W/m2 (WMO Guide to Instruments and Methods of Observation, WMO-No. 8, Vol. I, ch. 8).

Output per site (the page makes no live call; it reads these):
  <site>/sun/sun-climate.json   monthly totals, month x hour mean profiles, licence and attribution (LF)
  <site>/sun/tmy-hourly.bin     8760 hours x (G(h), Gb(n), Gd(h)) u16 W/m2, 16-byte header "GTM1"

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/sun_data.py            (all sites in sites-index.json)
    E:/swarm/gpu-bench/venv/Scripts/python.exe src/sun_data.py --site E:/lidar-out/open-land-01
"""
import argparse, hashlib, json, os, struct, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone
import numpy as np

API = "https://re.jrc.ec.europa.eu/api/v5_3/tmy"
RADDB = "PVGIS-SARAH3"
UA = "GlobalGrid2050-sun-data/1.0 (+https://globalgrid2050.com)"
CACHE = "E:/world-cache/sun"
SITES_INDEX = "E:/lidar-out/sites-index.json"
SUNSHINE_WM2 = 120.0
HOURS = 8760
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TMY_MAGIC = b"GTM1"
TMY_HEADER = struct.Struct("<4sHHHhHH")     # magic, version, hours, fields, offset (1/10000 h), unit, reserved
TMY_HEADER_BYTES = 16
FIELDS = ("G(h)", "Gb(n)", "Gd(h)")
LICENCE = dict(
    name="PVGIS usage conditions (European Commission, Joint Research Centre)",
    text="The information provided by PVGIS is free and there are no restrictions on its use.",
    url="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en",
    checked="2026-09-26")
ATTRIBUTION = ("Solar radiation: PVGIS 5.3 typical meteorological year, PVGIS-SARAH3 satellite radiation and ERA5 "
               "meteorology, European Commission Joint Research Centre. Not endorsed by the European Commission.")


HADUK_FILE = "E:/world-cache/sun/haduk/sun_hadukgrid_uk_1km_mon-30y_199101-202012.nc"
HADUK_BNG = dict(semi_major_axis=6377563.396, semi_minor_axis=6356256.909, longitude_of_central_meridian=-2.0,
                 latitude_of_projection_origin=49.0, false_easting=400000.0, false_northing=-100000.0,
                 scale_factor_at_central_meridian=0.9996012717)     # OSGB36 / British National Grid, EPSG:27700
HADUK_SOURCE = dict(
    dataset="HadUK-Grid Gridded Climate Observations on a 1km grid over the UK, v1.3.2.ceda (1836-2025)",
    product="monthly sunshine, 30-year average 1991-2020 (mon-30y), 1 km",
    kind="station-based observations (Met Office sunshine recorders) gridded by the Met Office; not satellite, not a model",
    doi="10.5285/789b3065d74a4c948ab05d33556c86d0",
    catalogue="https://catalogue.ceda.ac.uk/uuid/789b3065d74a4c948ab05d33556c86d0/",
    citation=("Met Office; Hollis, D.; Carlisle, E.; Kendon, M.; Packman, S.; Doherty, A. (2026): HadUK-Grid Gridded "
              "Climate Observations on a 1km grid over the UK, v1.3.2.ceda (1836-2025). NERC EDS Centre for "
              "Environmental Data Analysis, 23 June 2026. doi:10.5285/789b3065d74a4c948ab05d33556c86d0."),
    method_reference=("Hollis, D, McCarthy, M, Kendon, M, Legg, T and Simpson, I (2019), HadUK-Grid - A new UK dataset "
                      "of gridded climate observations, Geosci. Data J., 6(2), 151-159. doi:10.1002/gdj3.78"))
HADUK_LICENCE = dict(
    name="Open Government Licence v3.0",
    url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
    text=("CEDA record: access \"available to any registered CEDA user\", licence \"Open Government Licence\"; "
          "\"When using these data you must cite them correctly using the citation given on the CEDA Data Catalogue "
          "record.\" Met Office HadUK-Grid page: \"The HadUK-Grid datasets are freely available for use under Open "
          "Government Licence\" and users should \"acknowledge the source if the data are used in any report or "
          "product.\""),
    checked_urls=["https://catalogue.ceda.ac.uk/uuid/789b3065d74a4c948ab05d33556c86d0/",
                  "https://www.metoffice.gov.uk/hadobs/hadukgrid/",
                  "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"],
    checked="2026-09-26")
HADUK_ATTRIBUTION = ("Sunshine hours: Met Office HadUK-Grid v1.3.2.ceda, 1 km monthly averages 1991-2020, station "
                     "observations gridded by the Met Office (doi:10.5285/789b3065d74a4c948ab05d33556c86d0). Contains "
                     "public sector information licensed under the Open Government Licence v3.0.")
PREFERRED = dict(sunshine_hours="haduk_grid", irradiance="pvgis",
                 note=("web layer: show HadUK-Grid (station observations) for sunshine hours and PVGIS (satellite) for "
                       "every irradiance quantity; fall back to PVGIS sunshine hours only when haduk_grid is null"))


class HadUK:
    """The HadUK-Grid monthly 30-year sunshine file, opened once, read a 2 x 2 window per site."""

    def __init__(self, path=HADUK_FILE):
        import netCDF4
        self.path = path
        self.ds = netCDF4.Dataset(path)
        try:
            self._check()
        except Exception:
            self.ds.close()
            raise
        with open(path, "rb") as f:
            self.sha256 = hashlib.sha256(f.read()).hexdigest()

    def _check(self):
        v = self.ds["sun"]
        if v.getncattr("units") != "hour" or v.dimensions != ("time", "projection_y_coordinate", "projection_x_coordinate"):
            raise ValueError("unexpected HadUK sun variable")
        gm = self.ds[v.getncattr("grid_mapping")]
        if gm.getncattr("grid_mapping_name") != "transverse_mercator":
            raise ValueError("HadUK grid is not transverse mercator")
        for k, want in HADUK_BNG.items():
            if abs(float(gm.getncattr(k)) - want) > 1e-6 * max(1.0, abs(want)):
                raise ValueError(f"HadUK grid is not British National Grid: {k}={gm.getncattr(k)}")
        if [int(m) for m in self.ds["month_number"][:]] != list(range(1, 13)):
            raise ValueError("HadUK months are not Jan..Dec")
        self.x = np.asarray(self.ds["projection_x_coordinate"][:], float)
        self.y = np.asarray(self.ds["projection_y_coordinate"][:], float)
        for a in (self.x, self.y):
            if not np.allclose(np.diff(a), 1000.0):
                raise ValueError("HadUK grid is not a regular 1 km grid")
        self.attrs = {k: str(self.ds.getncattr(k)) for k in self.ds.ncattrs()}

    def close(self):
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def window(self, i0, j0):
        a = self.ds["sun"][:, i0:i0 + 2, j0:j0 + 2]
        return np.ma.filled(np.ma.asarray(a, float), np.nan)


def bilinear_sample(x, y, window_fn, e, n):
    """Bilinear from the four nearest cell centres. x, y ascending centre coordinates; window_fn(i0, j0) -> (12, 2, 2).
    Cells with no data are dropped and the remaining weights renormalised (recorded). Returns (12 values, record)."""
    j0 = int(np.searchsorted(x, e, side="right")) - 1
    i0 = int(np.searchsorted(y, n, side="right")) - 1
    if not (0 <= j0 < len(x) - 1 and 0 <= i0 < len(y) - 1):
        raise ValueError(f"({e}, {n}) is outside the HadUK grid")
    tx = (e - x[j0]) / (x[j0 + 1] - x[j0]); ty = (n - y[i0]) / (y[i0 + 1] - y[i0])
    w = np.array([[(1 - ty) * (1 - tx), (1 - ty) * tx], [ty * (1 - tx), ty * tx]])
    win = window_fn(i0, j0)
    ok = np.all(np.isfinite(win), axis=0)
    if not ok.any():
        raise ValueError(f"no HadUK data in the four cells around ({e}, {n})")
    wv = np.where(ok, w, 0.0); wv = wv / wv.sum()
    vals = np.einsum("mij,ij->m", np.where(ok[None], win, 0.0), wv)
    cells = [dict(e=float(x[j0 + dj]), n=float(y[i0 + di]), weight=round(float(wv[di, dj]), 6), valid=bool(ok[di, dj]),
                  sunshine_hours=[round(float(v), 2) for v in win[:, di, dj]] if ok[di, dj] else None)
             for di in (0, 1) for dj in (0, 1)]
    return vals, dict(method="bilinear from the 4 nearest 1 km cell centres (British National Grid, EPSG:27700)",
                      renormalised=not bool(ok.all()), cells=cells)


def haduk_block(h, e, n, pvgis_monthly):
    vals, rec = bilinear_sample(h.x, h.y, h.window, e, n)
    pv = pvgis_monthly["sunshine_hours"]
    return dict(
        label="Sunshine hours: station-based observations gridded by the Met Office (HadUK-Grid)",
        source=dict(HADUK_SOURCE, version=h.attrs.get("source"), file_version=h.attrs.get("version"),
                    period=h.attrs.get("lta_period"), cache_file=os.path.basename(h.path), cache_sha256=h.sha256),
        licence=HADUK_LICENCE, attribution=HADUK_ATTRIBUTION,
        sampling=rec, unit="hours per month (mean over 1991-2020)",
        monthly_sunshine_hours=[round(float(v), 1) for v in vals], annual_sunshine_hours=round(float(vals.sum()), 1),
        compare_pvgis=dict(
            note=("PVGIS sunshine hours are counted from one typical year (SARAH-3, hours with Gb(n) >= 120 W/m2); "
                  "ratio = PVGIS / HadUK-Grid; PVGIS was expected to read higher"),
            pvgis_monthly_sunshine_hours=pv,
            ratio_monthly=[round(p / float(hv), 3) if hv > 0 else None for p, hv in zip(pv, vals)],
            ratio_annual=round(sum(pv) / float(vals.sum()), 3)))


def bng_to_wgs84(e, n):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(e, n)
    return lat, lon


def grid_north_convergence(e, n):
    """Grid azimuth (degrees clockwise from grid north) of true north at (e, n), measured numerically."""
    from pyproj import Transformer
    lat, lon = bng_to_wgs84(e, n)
    fwd = Transformer.from_crs(4326, 27700, always_xy=True)
    e1, n1 = fwd.transform(lon, lat + 0.01)
    e0, n0 = fwd.transform(lon, lat)
    return float(np.degrees(np.arctan2(e1 - e0, n1 - n0)))


def cache_path(lat, lon, cache=CACHE):
    return os.path.join(cache, f"pvgis53-tmy-{RADDB.lower()}-nohorizon-{lat:.4f}_{lon:.4f}.json")


def fetch_tmy(lat, lon, cache=CACHE, pause_s=1.5, tries=4):
    """One polite GET per location, cached forever. Returns (json, cache file, fetched_now)."""
    path = cache_path(lat, lon, cache)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), path, False
    os.makedirs(cache, exist_ok=True)
    url = f"{API}?lat={lat:.4f}&lon={lon:.4f}&raddatabase={RADDB}&usehorizon=0&outputformat=json"
    for k in range(tries):
        time.sleep(pause_s * (2 ** k))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            d = json.loads(body)
            with open(path, "wb") as f:
                f.write(body)
            with open(path + ".url", "w", encoding="utf-8", newline="\n") as f:
                f.write(url + "\n" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
            return d, path, True
        except urllib.error.HTTPError as e:
            if e.code not in (429, 529, 500, 502, 503, 504) or k == tries - 1:
                raise
    raise RuntimeError("unreachable")


def parse_tmy(d):
    """-> (hourly (8760, 3) float W/m2 in FIELDS order, month (8760,) 1..12, hour (8760,) 0..23, offset_h, meta)."""
    rows = d["outputs"]["tmy_hourly"]
    if len(rows) != HOURS:
        raise ValueError(f"TMY has {len(rows)} hours, want {HOURS}")
    stamp = [r["time(UTC)"] for r in rows]
    month = np.array([int(s[4:6]) for s in stamp]); day = np.array([int(s[6:8]) for s in stamp])
    hour = np.array([int(s[9:11]) for s in stamp])
    want_m = np.repeat(np.arange(1, 13), [24 * n for n in MONTH_DAYS])
    want_d = np.concatenate([np.repeat(np.arange(1, n + 1), 24) for n in MONTH_DAYS])
    if not (np.array_equal(month, want_m) and np.array_equal(day, want_d) and np.array_equal(hour, np.tile(np.arange(24), 365))):
        raise ValueError("TMY rows are not Jan 1 00:00 .. Dec 31 23:00 in order")
    vals = np.array([[float(r[k]) for k in FIELDS] for r in rows])
    loc = d["inputs"]["location"]; met = d["inputs"]["meteo_data"]
    meta = dict(latitude=loc["latitude"], longitude=loc["longitude"], elevation_m=loc["elevation"],
                irradiance_time_offset_h=loc.get("irradiance_time_offset", 0.0), radiation_db=met["radiation_db"],
                meteo_db=met["meteo_db"], year_min=met["year_min"], year_max=met["year_max"],
                use_horizon=met["use_horizon"], months_selected=d["outputs"]["months_selected"])
    return vals, month, hour, float(meta["irradiance_time_offset_h"]), meta


def summarise(vals, month, hour):
    g, b, dfu = vals[:, 0], vals[:, 1], vals[:, 2]
    out = dict(ghi_kwh_m2=[], dhi_kwh_m2=[], bhi_kwh_m2=[], dni_kwh_m2=[], sunshine_hours=[])
    prof = {k: [] for k in FIELDS}
    for m in range(1, 13):
        s = month == m
        out["ghi_kwh_m2"].append(round(float(g[s].sum()) / 1000, 2))
        out["dhi_kwh_m2"].append(round(float(dfu[s].sum()) / 1000, 2))
        out["bhi_kwh_m2"].append(round(float((g[s] - dfu[s]).sum()) / 1000, 2))
        out["dni_kwh_m2"].append(round(float(b[s].sum()) / 1000, 2))
        out["sunshine_hours"].append(int((b[s] >= SUNSHINE_WM2).sum()))
        for i, k in enumerate(FIELDS):
            prof[k].append([int(round(float(vals[s & (hour == h), i].mean()))) for h in range(24)])
    out["annual"] = {k: round(sum(v), 2) for k, v in out.items()}
    return out, prof


def encode_tmy(vals, offset_h):
    q = np.clip(np.round(vals), 0, 65535).astype("<u2")
    head = TMY_HEADER.pack(TMY_MAGIC, 1, HOURS, len(FIELDS), int(round(offset_h * 10000)), 1, 0)
    return head.ljust(TMY_HEADER_BYTES, b"\0") + np.ascontiguousarray(q).tobytes()


def decode_tmy(blob):
    magic, ver, n, nf, off, unit, _ = TMY_HEADER.unpack_from(blob, 0)
    if magic != TMY_MAGIC or ver != 1 or len(blob) != TMY_HEADER_BYTES + 2 * n * nf:
        raise ValueError("bad tmy-hourly.bin")
    return np.frombuffer(blob, "<u2", offset=TMY_HEADER_BYTES).reshape(n, nf).astype(np.float64), off / 10000.0


def build(site_dir, centre_e, centre_n, d, cache_file, haduk=None):
    vals, month, hour, off, meta = parse_tmy(d)
    monthly, prof = summarise(vals, month, hour)
    out = os.path.join(site_dir, "sun"); os.makedirs(out, exist_ok=True)
    blob = encode_tmy(vals, off)
    with open(os.path.join(out, "tmy-hourly.bin"), "wb") as f:
        f.write(blob)
    lat, lon = bng_to_wgs84(centre_e, centre_n)
    doc = dict(
        format="sun-climate/1", site=os.path.basename(os.path.normpath(site_dir)), crs="EPSG:27700",
        centre_e=int(centre_e), centre_n=int(centre_n), latitude=round(lat, 6), longitude=round(lon, 6),
        grid_azimuth_of_true_north_deg=round(grid_north_convergence(centre_e, centre_n), 5),
        source=dict(service=API, radiation_db=meta["radiation_db"], meteo_db=meta["meteo_db"],
                    years=[meta["year_min"], meta["year_max"]], use_horizon=meta["use_horizon"],
                    query_lat=meta["latitude"], query_lon=meta["longitude"], pvgis_elevation_m=meta["elevation_m"],
                    months_selected=meta["months_selected"], cache_file=os.path.basename(cache_file),
                    cache_sha256=hashlib.sha256(open(cache_file, "rb").read()).hexdigest()),
        licence=LICENCE, attribution=ATTRIBUTION,
        time=dict(zone="UTC", irradiance_time_offset_h=off,
                  note="each hourly value stands for the hour centred on HH:00 + offset (PVGIS's own instant)"),
        sunshine_rule=f"hour counted as sunshine when Gb(n) >= {SUNSHINE_WM2:g} W/m2 (WMO-No. 8 definition)",
        monthly=monthly,
        profiles_w_m2=dict(note="mean W/m2 by month (12 rows) and UTC hour (24 columns)", **prof),
        hourly=dict(file="tmy-hourly.bin", fields=list(FIELDS), unit="W/m2", dtype="u16 little-endian",
                    header_bytes=TMY_HEADER_BYTES, hours=HOURS, sha256=hashlib.sha256(blob).hexdigest(),
                    bytes=len(blob)),
        not_used=dict(met_office_datahub="forecasts and 48 h observations only, API key, not climate",
                      met_office_aws_ukv="forecast model output, rolling 2 years, CC BY-SA; not a climate"),
        preferred=PREFERRED,
        haduk_grid=haduk_block(haduk, centre_e, centre_n, monthly) if haduk is not None else None,
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = os.path.join(out, "sun-climate.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    return doc, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", help="one site folder (default: every site in the sites index)")
    ap.add_argument("--index", default=SITES_INDEX)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--haduk", default=HADUK_FILE, help="HadUK-Grid monthly 30-year sunshine netCDF ('' to skip)")
    a = ap.parse_args(argv)
    haduk = HadUK(a.haduk) if a.haduk and os.path.exists(a.haduk) else None
    if haduk is None:
        print(f"HadUK-Grid file not found ({a.haduk!r}); writing PVGIS only")
    sites = json.load(open(a.index))["sites"]
    root = os.path.dirname(a.index)
    if a.site:
        name = os.path.basename(os.path.normpath(a.site))
        sites = [s for s in sites if s["name"] == name]
        if not sites:
            raise SystemExit(f"{name} not in {a.index}")
    for s in sites:
        site_dir = a.site or os.path.join(root, s["name"])
        lat, lon = bng_to_wgs84(s["centre_e"], s["centre_n"])
        d, cf, fresh = fetch_tmy(round(lat, 4), round(lon, 4), a.cache)
        doc, path = build(site_dir, s["centre_e"], s["centre_n"], d, cf, haduk)
        m = doc["monthly"]["annual"]
        print(f"{s['name']}: {'fetched' if fresh else 'cached'} lat {doc['latitude']} lon {doc['longitude']} "
              f"GHI {m['ghi_kwh_m2']} kWh/m2, DNI {m['dni_kwh_m2']} kWh/m2, sunshine {m['sunshine_hours']} h, "
              f"{os.path.getsize(path)} + {doc['hourly']['bytes']} bytes")
        if doc["haduk_grid"]:
            c = doc["haduk_grid"]["compare_pvgis"]
            print(f"  HadUK-Grid sunshine {doc['haduk_grid']['annual_sunshine_hours']} h/yr, PVGIS/HadUK "
                  f"{c['ratio_annual']}, monthly {c['ratio_monthly']}")
    if haduk is not None:
        haduk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
