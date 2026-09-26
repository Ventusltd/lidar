"""Fetch Environment Agency LIDAR Composite DTM 1m GeoTIFFs over WCS 2.0.1.

No API key. Requests are sequential, at most 1 km per side, with a short pause
between them. Raw GeoTIFFs are cached on disk and reused on later runs.
"""
import os
import sys
import time
import urllib.request

WCS_BASE = ("https://environment.data.gov.uk/spatialdata/"
            "lidar-composite-digital-terrain-model-dtm-1m/wcs")
COVERAGE_ID = ("13787b9a-26a4-4775-8523-806d13af58fc__"
               "Lidar_Composite_Elevation_DTM_1m")
CHUNK_M = 1000
PAUSE_S = 1.0
USER_AGENT = "lidar-tiles/1 (+https://github.com; polite sequential fetch)"


def coverage_url(e0, e1, n0, n1):
    """GetCoverage URL for the box E[e0,e1] x N[n0,n1] in EPSG:27700 metres."""
    return (f"{WCS_BASE}?service=WCS&version=2.0.1&request=GetCoverage"
            f"&CoverageId={COVERAGE_ID}&format=image/tiff"
            f"&subset=E({e0},{e1})&subset=N({n0},{n1})")


def chunk_ranges(lo, hi, step=CHUNK_M):
    """Split [lo, hi) into consecutive ranges no longer than step."""
    out = []
    a = lo
    while a < hi:
        b = min(a + step, hi)
        out.append((a, b))
        a = b
    return out


def fetch_one(e0, e1, n0, n1, cache_dir, retries=3):
    """Fetch one chunk into cache_dir; return (path, bytes, fetched_now)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"dtm1m_E{e0}-{e1}_N{n0}-{n1}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, os.path.getsize(path), False
    url = coverage_url(e0, e1, n0, n1)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read()
            if "tiff" not in ctype or data[:2] not in (b"II", b"MM"):
                raise RuntimeError(f"not a TIFF ({ctype}): {data[:200]!r}")
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return path, len(data), True
        except Exception as exc:  # network or service error: back off, retry
            last = exc
            time.sleep(PAUSE_S * (2 ** attempt) * 3)
    raise RuntimeError(f"WCS fetch failed for {url}: {last}")


def fetch_box(e0, e1, n0, n1, cache_dir, log=print):
    """Fetch the box E[e0,e1) x N[n0,n1) in <=1 km chunks.

    Returns a list of dicts {path, e0, e1, n0, n1, bytes, fetched}.
    """
    chunks = []
    for (ce0, ce1) in chunk_ranges(e0, e1):
        for (cn0, cn1) in chunk_ranges(n0, n1):
            path, nbytes, fresh = fetch_one(ce0, ce1, cn0, cn1, cache_dir)
            log(f"  {'GET ' if fresh else 'hit '} E{ce0}-{ce1} N{cn0}-{cn1}"
                f"  {nbytes:,} B")
            chunks.append(dict(path=path, e0=ce0, e1=ce1, n0=cn0, n1=cn1,
                               bytes=nbytes, fetched=fresh))
            if fresh:
                time.sleep(PAUSE_S)
    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: fetch_wcs.py E0 E1 N0 N1 CACHE_DIR")
        sys.exit(2)
    a = [int(x) for x in sys.argv[1:5]]
    fetch_box(a[0], a[1], a[2], a[3], sys.argv[5])
