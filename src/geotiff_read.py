"""Read EA DTM GeoTIFF chunks with tifffile and mosaic them into one grid.

Georeferencing comes from ModelPixelScale (33550) + ModelTiepoint (33922), or
from ModelTransformation (34264), which is what the EA WCS actually returns.
The CRS must be EPSG:27700 (GeoKey 3072 ProjectedCSTypeGeoKey). Nodata is the
GDAL_NODATA tag (42113); non-finite values and values below -1000 m are also
treated as nodata.
"""
import logging

import numpy as np
import tifffile

# tifffile warns that the EA nodata (-FLT_MAX as text) is not castable to
# float32; the raw tag is still read below, so silence that one warning.
logging.getLogger("tifffile").setLevel(logging.ERROR)

TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_TRANSFORM = 34264
TAG_GEOKEYS = 34735
TAG_NODATA = 42113
KEY_PROJECTED_CRS = 3072
KEY_RASTER_TYPE = 1025
EPSG_BNG = 27700


def _geokeys(tags):
    t = tags.get(TAG_GEOKEYS)
    if t is None:
        return {}
    v = list(t.value)
    n = v[3]
    out = {}
    for i in range(n):
        key, loc, _count, val = v[4 + 4 * i: 8 + 4 * i]
        if loc == 0:
            out[key] = val
    return out


def _raw_nodata(tags):
    """Read tag 42113 as text; tifffile may fail to cast float32 min."""
    t = tags.get(TAG_NODATA)
    if t is None:
        return None
    v = t.value
    try:
        if isinstance(v, bytes):
            v = v.decode("ascii", "ignore")
        return float(str(v).strip().strip("\x00"))
    except ValueError:
        return None


def read_geotiff(path):
    """Return dict(z float64 north-up, west_e, north_n, dx, dy, epsg, nodata)."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags
        z = page.asarray().astype(np.float64)
        keys = _geokeys(tags)
        epsg = keys.get(KEY_PROJECTED_CRS)
        if epsg != EPSG_BNG:
            raise ValueError(f"{path}: CRS EPSG:{epsg}, expected EPSG:27700")
        if TAG_PIXEL_SCALE in tags and TAG_TIEPOINT in tags:
            sx, sy = tags[TAG_PIXEL_SCALE].value[:2]
            tp = tags[TAG_TIEPOINT].value
            i, j, x, y = tp[0], tp[1], tp[3], tp[4]
            west, north = x - i * sx, y + j * sy
        elif TAG_TRANSFORM in tags:
            m = tags[TAG_TRANSFORM].value
            if abs(m[1]) > 1e-12 or abs(m[4]) > 1e-12:
                raise ValueError(f"{path}: rotated raster not supported")
            sx, sy, west, north = m[0], -m[5], m[3], m[7]
        else:
            raise ValueError(f"{path}: no GeoTIFF georeferencing tags")
        if keys.get(KEY_RASTER_TYPE, 1) == 2:  # PixelIsPoint -> shift to area
            west -= sx / 2.0
            north += sy / 2.0
        nodata = _raw_nodata(tags)
    return dict(z=z, west_e=float(west), north_n=float(north), dx=float(sx),
                dy=float(sy), epsg=epsg, nodata=nodata)


def mask_nodata(z, nodata=None):
    """Copy of z with nodata (tag value, non-finite, < -1000) set to NaN."""
    z = np.array(z, dtype=np.float64, copy=True)
    bad = ~np.isfinite(z) | (z < -1000.0)
    if nodata is not None and np.isfinite(nodata):
        bad |= np.isclose(z, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6)
    z[bad] = np.nan
    return z


def mosaic(paths, e0, n0, width, height):
    """Mosaic chunk files into a south-up float64 grid of shape (height, width).

    Row r, column c holds the 1 m cell whose SW corner is (e0 + c, n0 + r).
    Cells not covered by any chunk (or nodata) are NaN.
    """
    grid = np.full((height, width), np.nan, dtype=np.float64)
    for p in paths:
        g = read_geotiff(p)
        if abs(g["dx"] - 1.0) > 1e-9 or abs(g["dy"] - 1.0) > 1e-9:
            raise ValueError(f"{p}: spacing {g['dx']}x{g['dy']}, expected 1 m")
        z = mask_nodata(g["z"], g["nodata"])[::-1]  # now south-up
        rows, cols = z.shape
        c0 = int(round(g["west_e"] - e0))
        r0 = int(round(g["north_n"] - rows - n0))
        gr0, gc0 = max(r0, 0), max(c0, 0)
        gr1, gc1 = min(r0 + rows, height), min(c0 + cols, width)
        if gr1 <= gr0 or gc1 <= gc0:
            continue
        sub = z[gr0 - r0:gr1 - r0, gc0 - c0:gc1 - c0]
        dst = grid[gr0:gr1, gc0:gc1]
        take = np.isnan(dst) & ~np.isnan(sub)
        dst[take] = sub[take]
    return grid
