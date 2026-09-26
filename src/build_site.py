"""Build a .ght tile set for one site from the EA LIDAR Composite DTM 1m.

  python src/build_site.py --name open-land-01 --e 400000 --n 210000 --size 2048

The box is centred on (E, N), snapped to a 256 m lattice, and size must be a
multiple of 256. Tiles share their edge samples, so the fetch covers size+1 m
per side. Raw GeoTIFFs are cached under $LIDAR_CACHE/<name>/ (default .local/lidar-cache) and never
land in the repository.

Sampling note: grid node (E, N) takes the 1 m cell whose SW corner is (E, N),
so node heights sit 0.5 m west/south of the cell centres they came from.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_wcs import fetch_box, WCS_BASE, COVERAGE_ID  # noqa: E402
from geotiff_read import mosaic  # noqa: E402
from cut_tiles import cut_grid, load_tiles, TILE_M  # noqa: E402

CACHE_ROOT = os.environ.get("LIDAR_CACHE", os.path.join(".local", "lidar-cache"))
OUT_ROOT = os.environ.get("LIDAR_OUT", os.path.join(".local", "lidar-out"))


def snapped_box(e, n, size):
    """SW corner of a size x size box centred on (e, n), snapped to 256 m."""
    if size <= 0 or size % TILE_M:
        raise ValueError(f"--size must be a positive multiple of {TILE_M}")
    e0 = int(math.floor((e - size / 2) / TILE_M + 0.5)) * TILE_M
    n0 = int(math.floor((n - size / 2) / TILE_M + 0.5)) * TILE_M
    return e0, n0


def build(name, e, n, size, out_dir=None, cache_dir=None, log=print):
    out_dir = out_dir or os.path.join(OUT_ROOT, name)
    cache_dir = cache_dir or os.path.join(CACHE_ROOT, name)
    e0, n0 = snapped_box(e, n, size)
    e1, n1 = e0 + size + 1, n0 + size + 1  # +1: shared north/east edge
    log(f"site {name}: box E{e0}-{e0 + size} N{n0}-{n0 + size} "
        f"(fetch E{e0}-{e1} N{n0}-{n1})")
    t0 = time.time()
    chunks = fetch_box(e0, e1, n0, n1, cache_dir, log=log)
    raw = sum(c["bytes"] for c in chunks)
    got = sum(c["bytes"] for c in chunks if c["fetched"])
    grid = mosaic([c["path"] for c in chunks], e0, n0, size + 1, size + 1)
    nan = int(np.isnan(grid).sum())
    log(f"mosaic {grid.shape}, nodata {nan}, "
        f"min {np.nanmin(grid):.2f} m, max {np.nanmax(grid):.2f} m")
    source = (f"{WCS_BASE} CoverageId={COVERAGE_ID} "
              f"subset E({e0},{e1}) N({n0},{n1}), WCS 2.0.1 GetCoverage")
    man = cut_grid(grid, e0, n0, out_dir, name, source=source)
    _, back = load_tiles(out_dir)
    both = ~np.isnan(grid)
    if not np.array_equal(both, ~np.isnan(back)):
        raise RuntimeError("nodata mask changed in round trip")
    err = float(np.max(np.abs(back[both] - grid[both]))) if both.any() else 0
    tile_bytes = sum(t["bytes"] for t in man["tiles"])
    log(f"wrote {len(man['tiles'])} tiles ({tile_bytes:,} B) to {out_dir}; "
        f"raw {raw:,} B in {len(chunks)} chunks ({got:,} B fetched now); "
        f"round-trip max err {err * 1000:.2f} mm; {time.time() - t0:.1f} s")
    return dict(manifest=man, raw_bytes=raw, fetched_bytes=got,
                chunks=len(chunks), max_err_m=err, nodata=nan,
                min_m=float(np.nanmin(grid)), max_m=float(np.nanmax(grid)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--e", type=float, required=True, help="centre easting")
    ap.add_argument("--n", type=float, required=True, help="centre northing")
    ap.add_argument("--size", type=int, default=2048, help="box side, m")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--cache", default=None, help="raw GeoTIFF cache folder")
    a = ap.parse_args(argv)
    build(a.name, a.e, a.n, a.size, a.out, a.cache)


if __name__ == "__main__":
    main()
