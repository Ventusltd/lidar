"""Copernicus DEM GLO-30 -> .ght tiles on British National Grid, at an honest spacing.

  python src/copernicus.py --name open-land-01 --e 400128 --n 209920 --size 2048 [--spacing 32]

What the source is (Copernicus DEM Product Handbook, and the bucket's readme.html):
  * A SURFACE model (DSM): tree tops, roofs and hedges are in it. It is not bare earth.
  * 1 x 1 degree Cloud Optimised GeoTIFFs, float32, DEFLATE with floating-point predictor,
    1024 x 1024 internal tiles, public on https://copernicus-dem-30m.s3.amazonaws.com/ with no
    key. Browsers are blocked by CORS, so this PC fetches and the site serves the result.
  * Grid: 1 arc second in latitude everywhere; 1.5 arc seconds in longitude between 50 and 60
    degrees (about 31 m x 19-29 m on the ground in Great Britain). PixelIsPoint: pixel (0, 0)
    is the node at the tile's north-west corner. A missing tile is open sea.
  * Heights are metres above the EGM2008 geoid, NOT Ordnance Datum Newlyn. No vertical
    conversion is made here; the offset is measured, together with the surface-versus-bare-earth
    difference, by src/copernicus_pair.py.
  * Absolute vertical accuracy < 4 m (LE90), absolute horizontal < 6 m (CE90) (Handbook).

What this does:
  1. Converts the BNG box to a latitude/longitude box and lists the 1-degree tiles it touches.
  2. For each tile, fetches the TIFF header by HTTP byte range, works out which 1024 x 1024
     internal blocks the box needs, and fetches only those, each by one byte range. Every range
     is cached under $LIDAR_CACHE/copernicus/<tile>/ and never fetched twice.
  3. Places the blocks in one regular lat/lon mosaic (tiles in one longitude band share one
     lattice; a box that crosses a band edge at 50, 60 or 70 degrees is refused, not guessed).
  4. Reprojects to BNG by resampling: for every BNG node, the node's WGS84 position (osgb.py:
     OS Transverse Mercator inverse + 7-parameter Helmert, ~3.5 m) is bilinearly interpolated
     in the mosaic. No gdalwarp, no OSTN15; the method and its error are recorded.
  5. Cuts 257 x 257 .ght tiles at --spacing metres (default 32 m, never below 30 m, because the
     source holds nothing finer). Tiles sit on an absolute BNG lattice of 256 x spacing metres,
     so sets from different sites share tiles. tiles.json (LF) carries datum, accuracy and the
     Copernicus notice for modified data.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut_tiles import encode_tile, SAMPLES, TILE_M  # noqa: E402
from geotiff_read import mask_nodata  # noqa: E402
import osgb  # noqa: E402

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com/"
CACHE_ROOT = os.path.join(os.environ.get("LIDAR_CACHE", os.path.join(".local", "lidar-cache")), "copernicus")
OUT_ROOT = os.environ.get("LIDAR_OUT", os.path.join(".local", "lidar-out"))
USER_AGENT = "lidar-tiles/1 (copernicus glo-30; polite sequential fetch)"
HEAD_BYTES = 65536
MIN_SPACING_M = 30
DEFAULT_SPACING_M = 32
ATTRIBUTION = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
               "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
               "Union and ESA; all rights reserved")
CITATION = "https://doi.org/10.5270/ESA-c5d3d65"
LICENCE_URL = ("https://dataspace.copernicus.eu/explore-data/data-collections/"
               "copernicus-contributing-missions/collections-description/COP-DEM")
ACCURACY = dict(vertical_absolute_le90_m=4.0, horizontal_absolute_ce90_m=6.0,
                reference="Copernicus DEM Product Handbook (AIRBUS, for ESA), "
                          "GLO-30 accuracy specification",
                reprojection_helmert_m=osgb.HELMERT_ACCURACY_M)
DATUM = dict(horizontal_source="WGS84 (EPSG:4326)",
             horizontal_output="EPSG:27700 via OSGB36 Helmert + Transverse Mercator",
             vertical="EGM2008 geoid heights as delivered; NOT converted to ODN Newlyn",
             model="surface (DSM): includes vegetation and buildings; not bare earth")


def tile_name(lat_i, lon_i):
    """Bucket key stem for the 1-degree tile whose south-west corner is (lat_i, lon_i)."""
    ns = f"N{lat_i:02d}" if lat_i >= 0 else f"S{-lat_i:02d}"
    ew = f"E{lon_i:03d}" if lon_i >= 0 else f"W{-lon_i:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tile_url(name):
    return f"{BUCKET}{name}/{name}.tif"


def tiles_for_box(lat0, lon0, lat1, lon1):
    """(lat_i, lon_i) of every 1-degree tile overlapping the closed box, south-west first."""
    return [(a, b) for a in range(math.floor(lat0), math.floor(lat1) + 1)
            for b in range(math.floor(lon0), math.floor(lon1) + 1)]


def lon_step_arcsec(lat_i):
    """Longitude spacing (arc seconds) of the GLO-30 band holding tile row lat_i."""
    a = lat_i if lat_i >= 0 else -lat_i - 1  # distance of the equator-side edge
    for top, step in ((50, 1.0), (60, 1.5), (70, 2.0), (80, 3.0), (85, 5.0)):
        if a < top:
            return step
    return 10.0


# ---------------------------------------------------------------- byte ranges, cached
def http_range(url, start, end):
    """Bytes [start, end) of url, or None when the object does not exist (404: open sea)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Range": f"bytes={start}-{end - 1}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                total = r.headers.get("Content-Range", "").rpartition("/")[2]
            return data, (int(total) if total.isdigit() else None)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, None
            last = exc
        except OSError as exc:
            last = exc
        time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError(f"range fetch failed for {url} [{start},{end}): {last}")


class RangeCache:
    """Byte ranges of bucket objects, each kept on disk as <cache>/<tile>/<start>-<end>.bin."""

    def __init__(self, cache_dir=CACHE_ROOT, fetch=http_range, log=print):
        self.cache_dir, self.fetch, self.log = cache_dir, fetch, log
        self.fetched_bytes = self.hit_bytes = self.requests = 0

    def get(self, name, start, end):
        d = os.path.join(self.cache_dir, name)
        path = os.path.join(d, f"{start}-{end}.bin")
        missing = os.path.join(d, "MISSING")
        if os.path.exists(missing):
            return None, None
        if os.path.exists(path):
            data = open(path, "rb").read()
            total = open(os.path.join(d, "SIZE"), encoding="utf-8").read().strip()
            self.hit_bytes += len(data)
            return data, int(total)
        data, total = self.fetch(tile_url(name), start, end)
        self.requests += 1
        os.makedirs(d, exist_ok=True)
        if data is None:
            open(missing, "w", encoding="utf-8").close()
            return None, None
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        with open(os.path.join(d, "SIZE"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{total if total is not None else len(data)}\n")
        self.fetched_bytes += len(data)
        return data, total


class _HeadFile:
    """Just enough of a file for tifffile to parse the header: reads past HEAD_BYTES fail."""

    def __init__(self, data, size):
        self.data, self.size, self.pos = data, size, 0

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else self.pos + n
        if end > len(self.data) and self.pos < self.size:
            raise IOError("TIFF header runs past the fetched head bytes")
        out = self.data[self.pos:end]
        self.pos += len(out)
        return out

    def readinto(self, b):
        chunk = self.read(len(b))
        b[:len(chunk)] = chunk
        return len(chunk)


def open_tile(rc, name):
    """Parse a tile's header from one cached range. None for open sea."""
    data, total = rc.get(name, 0, HEAD_BYTES)
    if data is None:
        return None
    tf = tifffile.TiffFile(_HeadFile(data, total), name=name + ".tif")
    page = tf.pages[0]
    tags = page.tags
    sx, sy = tags[33550].value[:2]
    tp = tags[33922].value
    keys = list(tags[34735].value)
    geo = {keys[4 + 4 * i]: keys[7 + 4 * i] for i in range(keys[3]) if keys[5 + 4 * i] == 0}
    if geo.get(2048) != 4326:
        raise ValueError(f"{name}: geographic CRS {geo.get(2048)}, expected EPSG:4326")
    if geo.get(1025) != 2:
        raise ValueError(f"{name}: expected PixelIsPoint (1025=2), got {geo.get(1025)}")
    return dict(name=name, page=page, tf=tf, rows=page.shape[0], cols=page.shape[1],
                bh=page.tilelength, bw=page.tilewidth, dlon=float(sx), dlat=float(sy),
                lon_left=float(tp[3] - tp[0] * sx), lat_top=float(tp[4] + tp[1] * sy),
                offsets=page.dataoffsets, counts=page.databytecounts)


def read_rows_cols(rc, t, r0, r1, c0, c1):
    """Float64 array of tile rows [r0, r1) x cols [c0, c1), fetching only the blocks needed."""
    out = np.full((r1 - r0, c1 - c0), np.nan)
    nbx = -(-t["cols"] // t["bw"])
    for br in range(r0 // t["bh"], (r1 - 1) // t["bh"] + 1):
        for bc in range(c0 // t["bw"], (c1 - 1) // t["bw"] + 1):
            k = br * nbx + bc
            off, cnt = int(t["offsets"][k]), int(t["counts"][k])
            raw, _ = rc.get(t["name"], off, off + cnt)
            block, _idx, _shape = t["page"].decode(raw, k)
            block = np.asarray(block).reshape(t["bh"], t["bw"])
            y0, x0 = br * t["bh"], bc * t["bw"]
            ya, yb = max(r0, y0), min(r1, y0 + t["bh"], t["rows"])
            xa, xb = max(c0, x0), min(c1, x0 + t["bw"], t["cols"])
            out[ya - r0:yb - r0, xa - c0:xb - c0] = block[ya - y0:yb - y0, xa - x0:xb - x0]
    return mask_nodata(out)


def read_box(lat0, lon0, lat1, lon1, rc=None, margin=2):
    """Regular north-up mosaic covering the box plus `margin` nodes on every side.

    Returns dict(z, lat_top, lon_left, dlat, dlon, tiles, sea_tiles). Open sea is 0 m, as the
    bucket readme advises for tiles that do not exist.
    """
    rc = rc or RangeCache()
    steps = {lon_step_arcsec(a) for a, _ in tiles_for_box(lat0, lon0, lat1, lon1)}
    if len(steps) != 1:
        raise NotImplementedError("box crosses a GLO-30 longitude band edge (50/60/70 deg); "
                                  "split it into one box per band")
    dlat, dlon = 1.0 / 3600.0, steps.pop() / 3600.0
    # global node indices: row i at lat 90 - i*dlat, column j at lon -180 + j*dlon
    i0 = math.floor((90.0 - lat1) / dlat + 1e-9) - margin
    i1 = math.ceil((90.0 - lat0) / dlat - 1e-9) + margin + 1
    j0 = math.floor((lon0 + 180.0) / dlon + 1e-9) - margin
    j1 = math.ceil((lon1 + 180.0) / dlon - 1e-9) + margin + 1
    z = np.full((i1 - i0, j1 - j0), np.nan)
    lat_a, lat_b = 90.0 - (i1 - 1) * dlat, 90.0 - i0 * dlat
    lon_a, lon_b = -180.0 + j0 * dlon, -180.0 + (j1 - 1) * dlon
    used, sea = [], []
    # a node on a whole degree of latitude is row 0 of the tile to its south, hence lat_a - 1
    for la, lo in tiles_for_box(lat_a - 1.0, lon_a, lat_b, lon_b):
        name = tile_name(la, lo)
        ti0 = round((90.0 - (la + 1)) / dlat)          # global row of the tile's top row
        tj0 = round((lo + 180.0) / dlon)                # global column of its west column
        rows, cols = round(1.0 / dlat), round(1.0 / dlon)
        r0, r1 = max(i0, ti0), min(i1, ti0 + rows)
        c0, c1 = max(j0, tj0), min(j1, tj0 + cols)
        if r1 <= r0 or c1 <= c0:
            continue
        t = open_tile(rc, name)
        if t is None:
            z[r0 - i0:r1 - i0, c0 - j0:c1 - j0] = 0.0
            sea.append(name)
            continue
        if (t["rows"], t["cols"]) != (rows, cols) or abs(t["dlon"] - dlon) > 1e-12:
            raise ValueError(f"{name}: grid {t['rows']}x{t['cols']} dlon {t['dlon']} "
                             f"does not match the band lattice {rows}x{cols}")
        z[r0 - i0:r1 - i0, c0 - j0:c1 - j0] = read_rows_cols(
            rc, t, r0 - ti0, r1 - ti0, c0 - tj0, c1 - tj0)
        used.append(name)
    return dict(z=z, lat_top=lat_b, lon_left=lon_a, dlat=dlat, dlon=dlon,
                tiles=used, sea_tiles=sea)


def sample(m, lat, lon):
    """Bilinear height at (lat, lon) degrees in a read_box mosaic; NaN outside or near nodata."""
    z = m["z"]
    gy = (m["lat_top"] - np.asarray(lat)) / m["dlat"]
    gx = (np.asarray(lon) - m["lon_left"]) / m["dlon"]
    ny, nx = z.shape
    ok = (gx >= 0) & (gy >= 0) & (gx <= nx - 1) & (gy <= ny - 1)
    j = np.clip(np.floor(gx), 0, nx - 2).astype(np.int64)
    i = np.clip(np.floor(gy), 0, ny - 2).astype(np.int64)
    fx, fy = gx - j, gy - i
    top = z[i, j] + (z[i, j + 1] - z[i, j]) * fx
    bot = z[i + 1, j] + (z[i + 1, j + 1] - z[i + 1, j]) * fx
    return np.where(ok, top + (bot - top) * fy, np.nan)


# ---------------------------------------------------------------- BNG tiles
def lattice_box(e, n, size, spacing):
    """SW corner and tile counts of the tile-lattice box covering size x size centred on (e, n)."""
    tm = TILE_M * spacing
    e0 = math.floor((e - size / 2) / tm) * tm
    n0 = math.floor((n - size / 2) / tm) * tm
    nx = math.ceil((e + size / 2) / tm) - e0 // tm
    ny = math.ceil((n + size / 2) / tm) - n0 // tm
    return int(e0), int(n0), int(nx), int(ny)


def bng_grid(m, e0, n0, nx, ny, spacing):
    """South-up grid of heights at BNG nodes (e0 + c*spacing, n0 + r*spacing)."""
    rows, cols = ny * TILE_M + 1, nx * TILE_M + 1
    ee, nn = np.meshgrid(e0 + spacing * np.arange(cols, dtype=np.float64),
                         n0 + spacing * np.arange(rows, dtype=np.float64))
    lat, lon = osgb.bng_to_wgs84(ee, nn)
    return sample(m, lat, lon)


def bng_box_latlon(e0, n0, e1, n1, pad_deg=0.0):
    """Lat/lon box enclosing a BNG rectangle (edges sampled, since grid lines curve)."""
    s = np.linspace(0.0, 1.0, 33)
    e = np.concatenate([e0 + (e1 - e0) * s, np.full(33, e1), e0 + (e1 - e0) * s, np.full(33, e0)])
    n = np.concatenate([np.full(33, n0), n0 + (n1 - n0) * s, np.full(33, n1), n0 + (n1 - n0) * s])
    lat, lon = osgb.bng_to_wgs84(e, n)
    return (lat.min() - pad_deg, lon.min() - pad_deg, lat.max() + pad_deg, lon.max() + pad_deg)


def write_tiles(grid, e0, n0, spacing, out_dir, site, source):
    """Cut a south-up (k*256+1, m*256+1) grid into .ght tiles + tiles.json (LF)."""
    tm = TILE_M * spacing
    ny, nx = (grid.shape[0] - 1) // TILE_M, (grid.shape[1] - 1) // TILE_M
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    entries = []
    for iy in range(ny):
        for ix in range(nx):
            sub = grid[iy * TILE_M:iy * TILE_M + SAMPLES, ix * TILE_M:ix * TILE_M + SAMPLES]
            te, tn = e0 + ix * tm, n0 + iy * tm
            blob, info = encode_tile(sub, te, tn, spacing_mm=spacing * 1000)
            key = f"{te // tm}_{tn // tm}"
            rel = f"tiles/{key}.ght"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(),
                                e0=int(te), n0=int(tn), bytes=len(blob), min_m=info["min_m"],
                                max_m=info["max_m"], nodata=info["nodata"]))
    man = dict(format="ght1", crs="EPSG:27700", site=dict(name=site, origin_e=int(e0),
                                                             origin_n=int(n0)),
               tile_m=int(tm), spacing_m=int(spacing), tiles=entries,
               surface_model=True, datum=DATUM, accuracy=ACCURACY,
               attribution=ATTRIBUTION, citation=CITATION, licence_url=LICENCE_URL,
               access=("Copernicus Digital Elevation Model (DEM) was accessed on "
                       f"{datetime.now(timezone.utc):%Y-%m-%d} from "
                       "https://registry.opendata.aws/copernicus-dem"),
               source=source, generated_utc=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    with open(os.path.join(out_dir, "tiles.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    return man


def build(name, e, n, size, spacing=DEFAULT_SPACING_M, out_dir=None, rc=None, log=print):
    if int(spacing) != spacing or not MIN_SPACING_M <= spacing <= 65:
        raise ValueError(f"spacing must be a whole number of metres, {MIN_SPACING_M}-65: the "
                         "source holds nothing finer than ~30 m, and the header holds mm in u16")
    spacing = int(spacing)
    out_dir = out_dir or os.path.join(OUT_ROOT, name + "-copernicus")
    rc = rc or RangeCache(log=log)
    e0, n0, nx, ny = lattice_box(e, n, size, spacing)
    e1, n1 = e0 + nx * TILE_M * spacing, n0 + ny * TILE_M * spacing
    box = bng_box_latlon(e0, n0, e1, n1)
    t0 = time.time()
    m = read_box(*box, rc=rc)
    grid = bng_grid(m, e0, n0, nx, ny, spacing)
    log(f"{name}: BNG E{e0}-{e1} N{n0}-{n1} at {spacing} m from {len(m['tiles'])} tile(s) "
        f"{m['tiles']} (+{len(m['sea_tiles'])} sea); {rc.requests} range request(s), "
        f"{rc.fetched_bytes:,} B fetched, {rc.hit_bytes:,} B from cache")
    source = (f"{BUCKET} {', '.join(m['tiles'])}; byte ranges of 1024x1024 blocks; "
              f"bilinear at BNG nodes via osgb.py (Helmert + TM)")
    man = write_tiles(grid, e0, n0, spacing, out_dir, name, source)
    log(f"wrote {len(man['tiles'])} tile(s) to {out_dir} in {time.time() - t0:.1f} s; "
        f"nodata {int(np.isnan(grid).sum())}, {np.nanmin(grid):.2f}-{np.nanmax(grid):.2f} m EGM2008")
    return dict(manifest=man, grid=grid, mosaic=m, e0=e0, n0=n0, out_dir=out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--e", type=float, required=True, help="centre easting")
    ap.add_argument("--n", type=float, required=True, help="centre northing")
    ap.add_argument("--size", type=float, default=2048, help="box side to cover, m")
    ap.add_argument("--spacing", type=int, default=DEFAULT_SPACING_M, help="node spacing, m")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(a.name, a.e, a.n, a.size, a.spacing, a.out)


if __name__ == "__main__":
    main()
