# SPDX-License-Identifier: Apache-2.0
"""Sentinel-2 L2A over a British National Grid box, streamed from Microsoft Planetary Computer.

Nothing here needs an account: the STAC search is open and the blob SAS token endpoint is anonymous.
Reads are HTTP range requests into the Cloud-Optimised GeoTIFFs, so only the internal 1024 px tiles
that touch the box are fetched (a few MB a band a scene, never the 120 MB granule).

    search(bbox_wgs84, start, end)       STAC items, every page, oldest first
    BoxGrid(origin_e, origin_n, size, px) output pixel centres in BNG, and in any UTM zone
    read_window(href, transform, e, n)   nearest-neighbour samples of one band at those UTM points

Copernicus Sentinel data are free and open (EU Regulation 377/2014 and the Sentinel data legal notice);
anything built from them must say "Contains modified Copernicus Sentinel data [year]".
"""
import calendar, io, json, math, threading, time, urllib.error, urllib.request
import numpy as np
import tifffile

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
COLLECTION = "sentinel-2-l2a"
UA = {"User-Agent": "ventus-lidar-imagery/1 (+https://github.com/Ventusltd/lidar)"}


def _open(req, tries=6):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except Exception as e:
            if k == tries - 1:
                raise
            wait = 2 * (k + 1)
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:   # the host asks us to slow down
                wait = max(wait, float(e.headers.get("Retry-After") or 10))
            time.sleep(wait)


# ---------------------------------------------------------------- catalogue
def search(bbox, start, end, max_cloud=90, page=200):
    """Every sentinel-2-l2a item whose footprint touches bbox (lon/lat), start..end (YYYY-MM-DD)."""
    body = {"collections": [COLLECTION], "bbox": list(bbox), "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": page, "query": {"eo:cloud_cover": {"lte": max_cloud}}}
    items = []
    while body is not None:
        req = urllib.request.Request(STAC, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **UA})
        r = json.load(_open(req))
        items += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        body = {**body, **nxt[0]["body"]} if nxt and nxt[0].get("body") else None
    items.sort(key=lambda it: it["properties"]["datetime"])
    return items


class Signer:
    """Anonymous SAS token for the sentinel-2-l2a blob container, renewed five minutes before expiry."""
    def __init__(self):
        self.token, self.expiry, self.lock = None, 0.0, threading.Lock()

    def sign(self, href):
        with self.lock:
            return self._sign(href)

    def _sign(self, href):
        if self.token is None or time.time() > self.expiry - 300:
            t = json.load(_open(urllib.request.Request(SAS, headers=UA)))
            self.token = t["token"]
            ex = t.get("msft:expiry", "")
            try:
                self.expiry = calendar.timegm(time.strptime(ex[:19], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                self.expiry = time.time() + 1800
        return href + ("&" if "?" in href else "?") + self.token


# ---------------------------------------------------------------- range-request file for tifffile
class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file over HTTP Range, cached in 64 KB blocks."""
    BLOCK = 65536

    def __init__(self, url):
        self.url, self.pos, self.blocks, self.fetched = url, 0, {}, 0
        r = _open(urllib.request.Request(url, headers={"Range": "bytes=0-0", **UA}))
        cr = r.headers.get("Content-Range", "")
        self.size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("Content-Length", 0))
        r.read()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def get(self, a, n):
        """n bytes at offset a, one request (tile payloads bypass the block cache)."""
        if n <= 0:
            return b""
        r = _open(urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{a + n - 1}", **UA}))
        d = r.read()
        self.fetched += len(d)
        return d

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out, a = bytearray(), self.pos
        while len(out) < n:
            b = (a + len(out)) // self.BLOCK
            if b not in self.blocks:
                self.blocks[b] = self.get(b * self.BLOCK, min(self.BLOCK, self.size - b * self.BLOCK))
            blk, o = self.blocks[b], (a + len(out)) - b * self.BLOCK
            out += blk[o:o + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def window_tiles(r0, r1, c0, c1, th, tw, ncols):
    """Internal tile indices (row-major) covering rows r0..r1 and cols c0..c1 inclusive."""
    return [(ti, tj, ti * ncols + tj) for ti in range(r0 // th, r1 // th + 1) for tj in range(c0 // tw, c1 // tw + 1)]


def read_window(href, transform, x, y, fh=None):
    """Nearest-neighbour samples of band 1 at UTM points (x, y); -1 outside the raster.
    transform is the STAC proj:transform [a, 0, x0, 0, e, y0]. Returns (values int32, bytes fetched)."""
    a, _, x0, _, e, y0 = transform[:6]
    col = np.floor((np.asarray(x) - x0) / a).astype(np.int64)
    row = np.floor((np.asarray(y) - y0) / e).astype(np.int64)
    rf = fh or RangeFile(href)
    with tifffile.TiffFile(rf) as tf:
        page = tf.pages[0]
        H, W = page.shape[:2]
        inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        out = np.full(row.shape, -1, np.int32)
        if not inside.any():
            return out, rf.fetched
        r0, r1 = int(row[inside].min()), int(row[inside].max())
        c0, c1 = int(col[inside].min()), int(col[inside].max())
        th, tw = page.tilelength, page.tilewidth
        ncols = math.ceil(W / tw)
        sub = np.zeros(((r1 // th - r0 // th + 1) * th, (c1 // tw - c0 // tw + 1) * tw), np.int32)
        for ti, tj, k in window_tiles(r0, r1, c0, c1, th, tw, ncols):
            off, n = page.dataoffsets[k], page.databytecounts[k]
            if not n:
                continue
            tile, _, _ = page.decode(rf.get(off, n), k)
            tile = np.asarray(tile).reshape(th, tw)
            sub[(ti - r0 // th) * th:(ti - r0 // th + 1) * th, (tj - c0 // tw) * tw:(tj - c0 // tw + 1) * tw] = tile
        rr, cc = row[inside] - (r0 // th) * th, col[inside] - (c0 // tw) * tw
        out[inside] = sub[rr, cc]
    return out, rf.fetched


# ---------------------------------------------------------------- the box
class BoxGrid:
    """n x n output pixels covering the BNG square [origin_e, origin_e+size] x [origin_n, origin_n+size].
    Row 0 is the north edge (image order). Pixel size size/n (2048 m / 205 px = 9.99 m)."""
    def __init__(self, origin_e, origin_n, size, n=205):
        self.oe, self.on, self.size, self.n = float(origin_e), float(origin_n), float(size), int(n)
        self.px = self.size / self.n
        c = (np.arange(self.n) + 0.5) * self.px
        self.e = np.broadcast_to(self.oe + c[None, :], (self.n, self.n))
        self.nn = np.broadcast_to(self.on + self.size - c[:, None], (self.n, self.n))
        self._utm = {}

    def lonlat_bbox(self, pad_deg=0.0):
        from pyproj import Transformer
        t = Transformer.from_crs(27700, 4326, always_xy=True)
        es = [self.oe, self.oe + self.size]
        ns = [self.on, self.on + self.size]
        lon, lat = t.transform([es[0], es[1], es[0], es[1]], [ns[0], ns[0], ns[1], ns[1]])
        return [min(lon) - pad_deg, min(lat) - pad_deg, max(lon) + pad_deg, max(lat) + pad_deg]

    def utm(self, epsg):
        """Pixel centres in EPSG:epsg (x, y), cached per zone."""
        if epsg not in self._utm:
            from pyproj import Transformer
            t = Transformer.from_crs(27700, int(epsg), always_xy=True)
            self._utm[epsg] = t.transform(np.asarray(self.e, float), np.asarray(self.nn, float))
        return self._utm[epsg]
