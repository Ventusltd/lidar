# SPDX-License-Identifier: Apache-2.0
"""Sentinel-2 imagery for a test site: seasonal cloud-free composites and clear dates, two ways on the card.

For the site's 2048 m BNG box (sites-index.json), every Sentinel-2 L2A scene of the last four complete
seasons is found on Microsoft Planetary Computer. The scene classification layer (SCL, 20 m) is read over
the box first, so the box's own cloud is MEASURED, not taken from the scene-wide figure:

    cloud  = SCL 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus   (share of valid pixels)
    nodata = SCL 0 no data, 1 saturated or defective                           (share of all pixels)

Every scene's measured box cloud is kept in the index (Sun mode reads it). A scene joins its season's
composite when box cloud <= 40 % and nodata <= 10 %; its cloudy pixels are masked. A scene is a CLEAR
DATE when box cloud <= 1 % and nodata == 0; the best one a month goes to the time lapse.

Composite, per pixel over the clear observations of a season (reflectance, offset removed):
  electron   median: masked sort, middle value (mean of the two middles for an even count).
  positron   central-rank mean: each observation's rank found by pairwise counting (no sort), the mean
             of those whose rank lies in the 40th-60th percentile band (the two middle ranks when the band
             holds no whole rank). Equal to the median for small counts; a different estimator above ten.
  photon     a pixel-band where the two differ by more than 0.01 reflectance: COUNTED, never suppressed.
  witness    both estimators recomputed in NumPy on 4096 pixels, agreeing with the card within 1e-4.

Output in <site>/imagery/: composite-<season>.webp (205 x 205, 10 m, true colour, alpha 0 where no clear
look), optional ndvi-/ndwi-<season>.png, date-<YYYYMMDD>.webp, index.json (LF). Scene samples are cached
under imagery/cache (open data; E: only, never a repository).

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/sentinel_site.py --site E:/lidar-out/open-land-01
      [--index E:/lidar-out/sites-index.json] [--indices] [--cpu] [--today YYYY-MM-DD]
"""
import argparse, datetime as dt, hashlib, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_fetch  # noqa: E402

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:
    cp = None

BANDS = ["B02", "B03", "B04", "B08"]          # blue, green, red, near infrared (10 m)
CLOUD, NODATA = (3, 8, 9, 10), (0, 1)
COMPOSITE_MAX_CLOUD, COMPOSITE_MAX_NODATA = 0.40, 0.10
CLEAR_MAX_CLOUD = 0.01
PHOTON_TOL = 0.01                             # reflectance
WITNESS_PX, WITNESS_TOL = 4096, 1e-4
STRETCH_MAX, GAMMA = 0.25, 1.6                # true colour: reflectance 0..0.25 -> 0..255, gamma 1/1.6
LICENCE = {"name": "Copernicus Sentinel data: free, full and open (Regulation (EU) No 377/2014; Sentinel data legal notice)",
           "url": "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
           "host": "Microsoft Planetary Computer, collection sentinel-2-l2a",
           "host_url": "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a"}
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


# ---------------------------------------------------------------- pure pieces
def last_seasons(today, k=4):
    """The k most recent complete meteorological seasons before today, oldest first: (name, start, end)."""
    cur = today.year * 12 + (today.month // 3) * 3 - 1      # month count (y*12 + m-1) of the season start
    out = []
    for j in range(k, 0, -1):
        a, b = cur - 3 * j, cur - 3 * j + 3                  # season start, next season start
        s = dt.date(a // 12, a % 12 + 1, 1)
        e = dt.date(b // 12, b % 12 + 1, 1) - dt.timedelta(days=1)
        out.append((f"{SEASONS[s.month]}-{s.year}", s, e))
    return out


def classify(scl):
    """Measured cover over the box from SCL samples (-1 = outside the granule counts as nodata)."""
    scl = np.asarray(scl)
    total = scl.size
    nod = np.isin(scl, NODATA) | (scl < 0)
    valid = total - int(nod.sum())
    cloud = int(np.isin(scl, CLOUD).sum())
    return {"box_nodata": round(float(nod.sum()) / total, 4),
            "box_cloud": round(cloud / valid, 4) if valid else 1.0,
            "box_shadow": round(int((scl == 3).sum()) / valid, 4) if valid else 0.0,
            "box_cirrus": round(int((scl == 10).sum()) / valid, 4) if valid else 0.0,
            "box_snow": round(int((scl == 11).sum()) / valid, 4) if valid else 0.0}


def clear_mask(scl):
    scl = np.asarray(scl)
    return ~(np.isin(scl, CLOUD) | np.isin(scl, NODATA) | (scl < 0))


def reflectance(dn, baseline):
    """L2A digital numbers to reflectance; processing baseline 04.00 on adds 1000 (BOA_ADD_OFFSET)."""
    off = 1000.0 if float(baseline or 0) >= 4.0 else 0.0
    return (np.asarray(dn, np.float32) - off) / 10000.0


def median_electron(v, m, xp):
    """v (N, P) values, m (N, P) clear mask. Masked median over axis 0; NaN where no clear look."""
    s = xp.sort(xp.where(m, v, xp.inf), axis=0)
    n = m.sum(axis=0)
    lo = xp.clip((n - 1) // 2, 0, None)[None]
    hi = xp.clip(n // 2, 0, None)[None]
    med = (xp.take_along_axis(s, lo, 0)[0] + xp.take_along_axis(s, hi, 0)[0]) / 2
    return xp.where(n > 0, med, xp.nan)


def central_positron(v, m, xp, lo_q=0.4, hi_q=0.6):
    """Mean of observations whose rank (pairwise count, ties spanning a range) meets the 40-60 % band."""
    n = m.sum(axis=0).astype(xp.float64)                      # (P,)
    a, b = v[:, None, :], v[None, :, :]                       # k, j
    mj = m[None, :, :]
    below = ((b < a) & mj).sum(axis=1).astype(xp.float64)     # rank_lo of k
    equal = ((b == a) & mj).sum(axis=1).astype(xp.float64)
    rlo, rhi = below, below + equal - 1
    qlo, qhi = lo_q * (n - 1), hi_q * (n - 1)
    qlo, qhi = qlo - 1e-9, qhi + 1e-9                         # 0.4 * 10 must count as rank 4
    none = xp.ceil(qlo) > xp.floor(qhi)                       # band holds no whole rank
    mid = (n - 1) / 2
    qlo = xp.where(none, xp.floor(mid), qlo)
    qhi = xp.where(none, xp.ceil(mid), qhi)
    a0, a1 = xp.ceil(qlo)[None], xp.floor(qhi)[None]         # whole rank positions in the band
    share = xp.clip(xp.minimum(rhi, a1) - xp.maximum(rlo, a0) + 1, 0, None) / xp.maximum(equal, 1)
    w = xp.where(m, share, 0)                                 # a tie group shares its positions equally
    k = w.sum(axis=0)
    s = (w * v.astype(xp.float64)).sum(axis=0)
    return xp.where(n > 0, s / xp.maximum(k, 1e-12), xp.nan)


def composite(stack, mask, xp):
    """stack (N, B, P) float32, mask (N, P) -> electron (B, P), positron (B, P), photon bool (B, P), ms."""
    v, m = xp.asarray(stack), xp.asarray(mask)
    t0 = time.perf_counter()
    e = xp.stack([median_electron(v[:, b], m, xp) for b in range(v.shape[1])])
    if xp is not np:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    p = xp.stack([central_positron(v[:, b], m, xp) for b in range(v.shape[1])])
    if xp is not np:
        cp.cuda.Stream.null.synchronize()
    t2 = time.perf_counter()
    ph = xp.abs(e - p) > PHOTON_TOL
    return e, p, ph, (t1 - t0) * 1e3, (t2 - t1) * 1e3


def witness(stack, mask, e, p, seed=20260926):
    """Both estimators again in NumPy on WITNESS_PX pixels; max abs difference from the card."""
    P = stack.shape[2]
    idx = np.random.default_rng(seed).choice(P, min(WITNESS_PX, P), replace=False)
    we, wp, _, _, _ = composite(stack[:, :, idx], mask[:, idx], np)
    ee, pp = to_host(e)[:, idx], to_host(p)[:, idx]
    d = max(np.nanmax(np.abs(we - ee), initial=0), np.nanmax(np.abs(wp - pp), initial=0))
    nan_ok = bool((np.isnan(we) == np.isnan(ee)).all() and (np.isnan(wp) == np.isnan(pp)).all())
    return {"pixels": int(len(idx)), "max_err": float(d), "pass": bool(d <= WITNESS_TOL and nan_ok)}


def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def true_colour(bgrn, n):
    """(4, n*n) reflectance -> (n, n, 4) uint8 RGBA; alpha 0 where any of R, G, B is NaN."""
    rgb = np.stack([bgrn[2], bgrn[1], bgrn[0]], -1).reshape(n, n, 3)
    ok = ~np.isnan(rgb).any(-1)
    v = np.clip(np.nan_to_num(rgb) / STRETCH_MAX, 0, 1) ** (1 / GAMMA)
    return np.dstack([(v * 255 + 0.5).astype(np.uint8), (ok * 255).astype(np.uint8)])


def index_grey(a, b, n):
    """Normalised difference (a - b) / (a + b) -> (n, n, 2) uint8 LA; value = (x + 1) * 127.5."""
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (a - b) / (a + b)
    ok = np.isfinite(x)
    g = ((np.clip(np.nan_to_num(x), -1, 1) + 1) * 127.5 + 0.5).astype(np.uint8)
    return np.dstack([g.reshape(n, n), (ok * 255).astype(np.uint8).reshape(n, n)])


def save(img, path, fmt):
    from PIL import Image
    mode = {4: "RGBA", 2: "LA"}[img.shape[2]]
    buf = io.BytesIO()
    Image.fromarray(img, mode).save(buf, fmt, **({"quality": 88, "method": 6} if fmt == "WEBP" else {"optimize": True}))
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return {"file": os.path.basename(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


# ---------------------------------------------------------------- streaming
def sample_scene(it, grid, signer, cache, bands):
    """SCL (and bands if asked) at the box pixels; cached per item as .npz."""
    path = os.path.join(cache, it["id"] + ".npz")
    have = dict(np.load(path)) if os.path.exists(path) else {}
    epsg = it["properties"].get("proj:epsg") or int(str(it["properties"].get("proj:code", "0")).split(":")[-1])
    x, y = grid.utm(epsg)
    got = 0
    for b in ["SCL"] + list(bands):
        if b in have:
            continue
        a = it["assets"][b]
        have[b], n = s2_fetch.read_window(signer.sign(a["href"]), a["proj:transform"], x, y)
        got += n
    if got:
        np.savez_compressed(path, **have)
    return have, got


def run(site_dir, entry, today, indices=False, use_cpu=False, workers=8):
    xp = np if (use_cpu or cp is None) else cp
    out = os.path.join(site_dir, "imagery")
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    grid = s2_fetch.BoxGrid(entry["origin_e"], entry["origin_n"], entry["size_m"])
    seasons = last_seasons(today)
    items = s2_fetch.search(grid.lonlat_bbox(), seasons[0][1].isoformat(), seasons[-1][2].isoformat())
    signer, fetched = s2_fetch.Signer(), 0
    with ThreadPoolExecutor(workers) as ex:
        scl = list(ex.map(lambda it: sample_scene(it, grid, signer, cache, []), items))
    scenes = []
    for it, (h, n) in zip(items, scl):
        fetched += n
        pr = it["properties"]
        scenes.append({"id": it["id"], "datetime": pr["datetime"], "date": pr["datetime"][:10],
                       "platform": pr.get("platform"), "mgrs": pr.get("s2:mgrs_tile"), "baseline": pr.get("s2:processing_baseline"),
                       "scene_cloud": round(float(pr.get("eo:cloud_cover", -1)), 2),
                       "sun_zenith": round(float(pr.get("s2:mean_solar_zenith", 0)), 2),
                       "sun_azimuth": round(float(pr.get("s2:mean_solar_azimuth", 0)), 2), **classify(h["SCL"]), "use": []})
    # one look a date: the granule that covers the box best
    best = {}
    for i, s in enumerate(scenes):
        k = s["date"]
        if k not in best or (s["box_nodata"], s["box_cloud"]) < (scenes[best[k]]["box_nodata"], scenes[best[k]]["box_cloud"]):
            best[k] = i
    looks = sorted(best.values(), key=lambda i: scenes[i]["date"])
    comp = [i for i in looks if scenes[i]["box_cloud"] <= COMPOSITE_MAX_CLOUD and scenes[i]["box_nodata"] <= COMPOSITE_MAX_NODATA]
    clear = [i for i in comp if scenes[i]["box_cloud"] <= CLEAR_MAX_CLOUD and scenes[i]["box_nodata"] == 0]
    month = {}
    for i in clear:
        k = scenes[i]["date"][:7]
        if k not in month or scenes[i]["box_cloud"] < scenes[month[k]]["box_cloud"]:
            month[k] = i
    lapse = sorted(month.values(), key=lambda i: scenes[i]["date"])
    with ThreadPoolExecutor(workers) as ex:
        full = dict(zip(comp, ex.map(lambda i: sample_scene(items[i], grid, signer, cache, BANDS), comp)))
    fetched += sum(n for _, n in full.values())
    P, nn = grid.n * grid.n, grid.n
    composites, gpu = [], {"device": "cpu" if xp is np else cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                           "electron_ms": 0.0, "positron_ms": 0.0, "photons": 0, "pixel_bands": 0, "witness": []}
    for name, s0, s1 in seasons:
        ids = [i for i in comp if s0.isoformat() <= scenes[i]["date"] <= s1.isoformat()]
        rec = {"season": name, "start": s0.isoformat(), "end": s1.isoformat(), "looks": len(ids),
               "dates": [scenes[i]["date"] for i in ids]}
        if not ids:
            composites.append({**rec, "file": None})
            continue
        stack = np.stack([np.stack([reflectance(full[i][0][b], scenes[i]["baseline"]).ravel() for b in BANDS]) for i in ids])
        mask = np.stack([clear_mask(full[i][0]["SCL"]).ravel() for i in ids])
        for i in ids:
            scenes[i]["use"].append("composite")
        e, p, ph, te, tp = composite(stack, mask, xp)
        w = witness(stack, mask, e, p)
        eh, nobs = to_host(e), mask.sum(0)
        gpu["electron_ms"] += round(te, 2)
        gpu["positron_ms"] += round(tp, 2)
        gpu["photons"] += int(to_host(ph).sum())
        gpu["pixel_bands"] += int(eh.size)
        gpu["witness"].append({"season": name, **w})
        rec.update(save(true_colour(eh, nn), os.path.join(out, f"composite-{name}.webp"), "WEBP"))
        rec.update({"photons": int(to_host(ph).sum()), "max_diff": round(float(np.nanmax(np.abs(eh - to_host(p)), initial=0)), 5),
                    "filled": round(float((nobs > 0).mean()), 4), "obs_min": int(nobs.min()), "obs_median": float(np.median(nobs))})
        if indices:
            rec["ndvi"] = save(index_grey(eh[3], eh[2], nn), os.path.join(out, f"ndvi-{name}.png"), "PNG")
            rec["ndwi"] = save(index_grey(eh[1], eh[3], nn), os.path.join(out, f"ndwi-{name}.png"), "PNG")
        composites.append(rec)
    dates = []
    for i in lapse:
        s = scenes[i]
        s["use"].append("time-lapse")
        bgrn = np.stack([reflectance(full[i][0][b], s["baseline"]).ravel() for b in BANDS])
        bgrn[:, ~clear_mask(full[i][0]["SCL"]).ravel()] = np.nan
        dates.append({"date": s["date"], "id": s["id"], "box_cloud": s["box_cloud"],
                      **save(true_colour(bgrn, nn), os.path.join(out, f"date-{s['date'].replace('-', '')}.webp"), "WEBP")})
    years = sorted({s["date"][:4] for s in scenes if s["use"]})
    yr = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}" if years else str(today.year)
    index = {"schema": "lidar.imagery.v1", "site": entry["name"], "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "box": {"crs": "EPSG:27700", "origin_e": entry["origin_e"], "origin_n": entry["origin_n"], "size_m": entry["size_m"],
                     "pixels": nn, "pixel_m": round(grid.px, 4), "row0": "north", "sampling": "nearest 10 m sample at each pixel centre (pyproj BNG to UTM)"},
             "licence": {**LICENCE, "credit": f"Contains modified Copernicus Sentinel data {yr}"},
             "rules": {"cloud_scl": list(CLOUD), "nodata_scl": list(NODATA), "composite_max_box_cloud": COMPOSITE_MAX_CLOUD,
                       "composite_max_box_nodata": COMPOSITE_MAX_NODATA, "clear_max_box_cloud": CLEAR_MAX_CLOUD,
                       "true_colour": f"R,G,B = B04,B03,B02 reflectance 0..{STRETCH_MAX} to 0..255, gamma 1/{GAMMA}",
                       "index_png": "grey = (index + 1) * 127.5; NDVI (B08-B04)/(B08+B04), NDWI (B03-B08)/(B03+B08)",
                       "photon_tol_reflectance": PHOTON_TOL},
             "gpu": gpu, "fetched_bytes": fetched, "composites": composites, "dates": dates, "scenes": scenes}
    write_json(os.path.join(out, "index.json"), index)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True)
    ap.add_argument("--index", default="E:/lidar-out/sites-index.json")
    ap.add_argument("--indices", action="store_true", help="also write NDVI and NDWI greyscale PNGs")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--today", default=None)
    a = ap.parse_args(argv)
    name = os.path.basename(os.path.normpath(a.site))
    entry = next((s for s in json.load(open(a.index))["sites"] if s["name"] == name), None)
    if entry is None:
        sys.exit(f"{name} not in {a.index}")
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    ix = run(a.site, entry, today, indices=a.indices, use_cpu=a.cpu)
    g = ix["gpu"]
    ok = all(w["pass"] for w in g["witness"])
    print(f"{name}: {len(ix['scenes'])} scenes, {sum(c['looks'] for c in ix['composites'])} composite looks, "
          f"{len(ix['dates'])} clear dates; {g['device']} electron {g['electron_ms']:.1f} ms positron {g['positron_ms']:.1f} ms; "
          f"photons {g['photons']}/{g['pixel_bands']}; witness {'PASS' if ok else 'FAIL'}; {ix['fetched_bytes'] / 1e6:.1f} MB fetched")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
