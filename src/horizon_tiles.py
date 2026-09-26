"""Terrain horizon tiles: the horizon angle in 32 azimuths for every 4 m cell, computed as a pair on the GPU.

The horizon angle in an azimuth is the highest elevation angle at which the bare-earth terrain is seen
from the ground at the cell, looking that way: max over distance d of atan((z(p + d u) - z(p)) / d).
Ground is in terrain shadow when the sun's elevation is below the horizon angle in the sun's azimuth
(the method of Dozier and Frew, "Rapid calculation of terrain parameters for radiation modeling from
digital elevation data", IEEE Trans. Geosci. Remote Sens. 28(5), 1990, pp. 963-969).
The source is a bare-earth model (DTM): hedges, trees and buildings cast no shadow here.

Two channels, photons back (the electron/positron pattern of pair_gpu.py):

  electron   a ray march along each azimuth on the 1 m grid (horizon_march.py, a CUDA kernel, float64, no
             fused multiply-add): out to 32 m EXACT on the bilinear surface (cut at every grid line, the
             in-cell maximum of the quadratic taken), where a 1 m march missed knolls by up to 3.8
             degrees; then one bilinear sample every metre to the edge of the site.
  positron   a different sampling: a max pyramid (level k holds the highest node in each 2^k m block),
             distance bands d in [16 * 2^k, 32 * 2^k) sampled every 2^k m on level k, nearest block,
             and the first band staggered half a step (0.15, 0.25 .. 31.95 m, every 0.1 m) with the
             four-corner bilinear polynomial on the 1 m grid (CuPy array operations).
             Pooling takes the highest point of a block, so the positron reads a little high far out.
  photons    (cell, azimuth) pairs where |electron - positron| > 1 degree. COUNTED and located, never
             suppressed; the receipt says how many sit where the electron's horizon is within 32 m
             (staggered samples miss or catch a knoll) and how many further out (pyramid pooling).
  witness    a seeded sample of cells again on the CPU in NumPy by other code paths (bilinear written as
             lerps, pyramid built blockwise from level 0, not by repeated halving); both channels must
             agree with the GPU within 1e-9 degree and in reach.

Earth curvature is omitted: on a 2 km site it moves a horizon by under 0.01 degree (larger sites will
need it, as viewshed.py applies it). Azimuths are clockwise from GRID north, k * 11.25 degrees, k = 0..31. A value is no data when the
ray leaves the site within 100 m (its horizon is unknown); elsewhere near the edge it is a lower bound.

.ghz v1 ("GHZ1"), little-endian, 32-byte header then an i16 body:
    0 char[4] "GHZ1" | 4 u16 version=1 | 6 u16 samples=65 | 8 u16 spacing_mm=4000 | 10 u16 azimuths=32
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u16 unit=100 (value / unit = degrees)
   22 u16 min_reach_m=100 | 24 u32 nodata_count | 28 i16 max_value | 30 u16 reserved=0
   32 i16[65*65*32] horizon, rows SOUTH to NORTH, each WEST to EAST, 32 azimuths per cell; -32768 no data.
Tiles are 256 m (edges shared with neighbours), indexed in horizon-tiles.json with sha256s (LF);
the pair receipt lands beside it as horizon_receipt.json.

    python src/horizon_tiles.py --site $LIDAR_OUT/open-land-01
"""
import argparse, hashlib, json, os, struct, sys, time
from datetime import datetime, timezone
import numpy as np
from cut_tiles import ea_notice, TERRAIN_ONLY  # noqa: F401
from horizon_march import electron_gpu, electron_cpu, NEAR_M  # noqa: F401  (tests use horizon_tiles.*)

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and says so
    cp = None

TILE_M = 256
CELL_M = 4
NS = TILE_M // CELL_M + 1                     # 65 samples per tile side
NAZ = 32
UNIT = 100                                     # centidegrees
NODATA = -32768
MIN_REACH_M = 100
BAND0 = 32                                     # positron: the staggered bilinear band ends at 31.95 m
NEAR_DS_M = 0.1                                # positron: 0.1 m steps in its first band
NEAR_N = 320                                   # 32 m / NEAR_DS_M
TOL_DEG = 1.0
TOL_WIT = 1e-9
MAGIC = b"GHZ1"
HEADER = struct.Struct("<4sHHHHiiHHIhH")
assert HEADER.size == 32
INDEX = "horizon-tiles.json"
RECEIPT = "horizon_receipt.json"
AZ_DEG = np.arange(NAZ) * (360.0 / NAZ)


def directions(naz=NAZ):
    a = np.radians(np.arange(naz) * (360.0 / naz))
    return np.sin(a), np.cos(a)                # (east, north) per azimuth


# ---------------------------------------------------------------- electron: horizon_march.py


# ---------------------------------------------------------------- positron: max pyramid, distance bands
def pyramid_halving(xp, z, levels):
    """Level k by repeated 2x2 max of level k-1 (NaN counts as -inf)."""
    p = [xp.where(xp.isfinite(z), z, -xp.inf)]
    for _ in range(1, levels):
        a = p[-1]; h, w = a.shape
        a = xp.pad(a, ((0, h % 2), (0, w % 2)), constant_values=-xp.inf)
        p.append(a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).max(axis=(1, 3)))
    return p


def pyramid_blocks(z, levels):
    """Witness: level k straight from level 0 by 2^k x 2^k blocks."""
    base = np.where(np.isfinite(z), z, -np.inf); out = []
    for k in range(levels):
        b = 1 << k; h, w = base.shape
        hp, wp = -(-h // b) * b, -(-w // b) * b
        a = np.full((hp, wp), -np.inf); a[:h, :w] = base
        out.append(a.reshape(hp // b, b, wp // b, b).max(axis=(1, 3)))
    return out


def bands(dmax):
    """(level, distances) for the positron: 0.15..31.95 m every 0.1 m on level 0, then [16*2^k, 32*2^k)
    every 2^k m."""
    out = [(0, (np.arange(1, NEAR_N) + 0.5) * NEAR_DS_M)]
    k = 1
    while BAND0 // 2 * (1 << k) <= dmax:
        s = 1 << k
        out.append((k, np.arange(BAND0 // 2 * s, BAND0 * s, s, dtype=np.float64)))
        k += 1
    return out


def positron(xp, pyr, z0, cr, cc, H, W, naz=NAZ):
    """All cells at once. Returns best tangent (ncell, naz), reach (m) and the number of samples."""
    ux, uy = (xp.asarray(v)[None, :] for v in directions(naz))
    r = xp.asarray(cr, xp.float64)[:, None]; c = xp.asarray(cc, xp.float64)[:, None]
    z0 = xp.asarray(z0, xp.float64)[:, None]
    best = xp.full((len(cr), naz), -1.0e300); reach = xp.zeros((len(cr), naz), xp.int32)
    alive = xp.ones((len(cr), naz), bool); n = 0
    for k, ds in bands(int(np.hypot(H, W)) + 2):
        lev = pyr[k]
        for d in ds:
            x = c + d * ux; y = r + d * uy
            alive &= (x >= 0) & (y >= 0) & (x <= W - 1) & (y <= H - 1)
            if not bool(alive.any()):
                return best, reach, n
            if k == 0:                                   # four-corner polynomial a + b fx + c fy + d fx fy
                i = xp.clip(xp.floor(x).astype(xp.int64), 0, W - 2); j = xp.clip(xp.floor(y).astype(xp.int64), 0, H - 2)
                fx, fy = x - i, y - j
                a, b, c_, d_ = lev[j, i], lev[j, i + 1], lev[j + 1, i], lev[j + 1, i + 1]
                h = a + (b - a) * fx + (c_ - a) * fy + (a - b - c_ + d_) * fx * fy
            else:
                ix = xp.clip(xp.floor(x + 0.5).astype(xp.int64), 0, W - 1) >> k
                iy = xp.clip(xp.floor(y + 0.5).astype(xp.int64), 0, H - 1) >> k
                h = lev[iy, ix]
            s = xp.where(alive & xp.isfinite(h), (h - z0) / d, -1.0e300)
            best = xp.maximum(best, s)
            reach = xp.where(alive, xp.int32(int(d)), reach)
            n += int(alive.sum())
    return best, reach, n


# ---------------------------------------------------------------- tiles
def to_deg(t):
    return np.where(t > -1e299, np.degrees(np.arctan(t)), np.nan)


def quantise(deg, reach):
    q = np.clip(np.round(np.nan_to_num(deg) * UNIT), -9000, 9000).astype(np.int16)
    q[(reach < MIN_REACH_M) | ~np.isfinite(deg)] = NODATA
    return q


def encode(q, e0, n0):
    if q.shape != (NS, NS, NAZ):
        raise ValueError(f"horizon tile must be {NS}x{NS}x{NAZ}")
    nod = int((q == NODATA).sum()); valid = q[q != NODATA]
    mx = int(valid.max()) if valid.size else NODATA
    head = HEADER.pack(MAGIC, 1, NS, CELL_M * 1000, NAZ, int(e0), int(n0), UNIT, MIN_REACH_M, nod, mx, 0)
    return head + np.ascontiguousarray(q, "<i2").tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .ghz blob")
    magic, ver, ns, sp, naz, e0, n0, unit, reach, nod, mx, _ = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    if len(blob) != HEADER.size + 2 * ns * ns * naz:
        raise ValueError(f".ghz size {len(blob)} != {HEADER.size + 2 * ns * ns * naz}")
    head = dict(samples=ns, spacing_mm=sp, azimuths=naz, origin_e_m=e0, origin_n_m=n0, unit=unit,
                min_reach_m=reach, nodata_count=nod, max_value=mx)
    return head, np.frombuffer(blob, "<i2", offset=HEADER.size).reshape(ns, ns, naz)


def shaded(h_deg, sun_az, sun_el):
    """h_deg: (..., NAZ) horizon in degrees; the sun's grid azimuth and elevation in degrees. Linear
    between neighbouring azimuths. True where the sun is below the horizon (NaN never shaded)."""
    f = (sun_az % 360.0) / (360.0 / h_deg.shape[-1]); k = int(np.floor(f)) % h_deg.shape[-1]; w = f - np.floor(f)
    h = h_deg[..., k] * (1 - w) + h_deg[..., (k + 1) % h_deg.shape[-1]] * w
    return np.isfinite(h) & (sun_el < h)


def write_tiles(out_dir, q, oe, on, site_name):
    rows, cols, _ = q.shape
    step = TILE_M // CELL_M
    if (rows - 1) % step or (cols - 1) % step:
        raise ValueError(f"cell grid {q.shape[:2]} is not k*{step}+1 per side")
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    entries = []
    for iy in range((rows - 1) // step):
        for ix in range((cols - 1) // step):
            t = q[iy * step:iy * step + NS, ix * step:ix * step + NS]
            e0, n0 = oe + ix * TILE_M, on + iy * TILE_M
            blob = encode(t, e0, n0)
            key = f"{ix}_{iy}"; rel = f"tiles/{key}.ghz"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            v = t[t != NODATA]
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(e0), n0=int(n0),
                                bytes=len(blob), nodata=int((t == NODATA).sum()),
                                max_deg=round(float(v.max()) / UNIT, 2) if v.size else None))
    index = dict(format="ghz1", crs="EPSG:27700", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)),
                 tile_m=TILE_M, cell_m=CELL_M, azimuths=NAZ, azimuth="clockwise from grid north, k * 11.25 degrees",
                 unit="centidegrees", nodata=NODATA, min_reach_m=MIN_REACH_M, source="bare-earth DTM, 1 m",
                 caveat="computed from terrain only; hedges, trees and buildings cast no shadow",
                 near_field=f"exact maximum on the bilinear surface out to {NEAR_M} m, then a sample every 1 m",
                 **ea_notice(), tiles=entries, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    return index


# ---------------------------------------------------------------- the pair
def host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def cells(shape, cell=CELL_M):
    H, W = shape
    r, c = np.mgrid[0:H:cell, 0:W:cell]
    return r.ravel(), c.ravel(), (len(range(0, H, cell)), len(range(0, W, cell)))


def run(grid, oe=0, on=0, use_gpu=True, witness_cells=1024, seed=20260926, cell=CELL_M):
    """grid: south-up float64, 1 m. Returns (receipt, electron degrees (rows, cols, NAZ), reach)."""
    t0 = time.perf_counter()
    gpu = use_gpu and cp is not None
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if gpu else "cpu (numpy)"
    z = np.asarray(grid, np.float64); H, W = z.shape
    cr, cc, shape = cells(z.shape, cell)
    if gpu:
        e_best, e_reach, e_at, e_n = (host(v) if not isinstance(v, int) else v for v in electron_gpu(z, cr, cc))
    else:
        parts = [electron_cpu(z, r, c) for r, c in zip(cr, cc)]
        e_best, e_reach, e_at = (np.stack([p[i] for p in parts]) for i in range(3)); e_n = int(e_reach.sum())
    xp = cp if gpu else np
    levels = len(bands(int(np.hypot(H, W)) + 2))
    pyr = pyramid_halving(xp, xp.asarray(z), levels)
    p_best, p_reach, p_n = positron(xp, pyr, z[cr, cc], cr, cc, H, W)
    p_best, p_reach = host(p_best), host(p_reach)
    e_deg, p_deg = to_deg(e_best), to_deg(p_best)
    ok = (e_reach >= MIN_REACH_M) & np.isfinite(e_deg) & np.isfinite(p_deg)
    diff = np.where(ok, p_deg - e_deg, 0.0)
    photon = ok & (np.abs(diff) > TOL_DEG)
    nph = int(photon.sum()); nok = int(ok.sum())
    near = int((photon & (e_at < BAND0)).sum())   # electron horizon inside the positron's 1 m band
    counts = np.bincount((np.nonzero(photon)[0]), minlength=len(cr)).reshape(shape)
    hot = np.argsort(counts.ravel())[::-1][:8]
    rec = dict(
        device=device, cells=len(cr), cell_m=cell, azimuths=NAZ, pairs=int(photon.size), comparable=nok, tol_deg=TOL_DEG,
        electron=dict(method=f"ray march, exact in-cell maximum to {NEAR_M} m, then 1 m bilinear steps", samples=e_n,
                      mean_deg=round(float(e_deg[ok].mean()), 4) if nok else None,
                      max_deg=round(float(e_deg[ok].max()), 3) if nok else None),
        positron=dict(method="max pyramid, distance bands", samples=p_n,
                      mean_deg=round(float(p_deg[ok].mean()), 4) if nok else None,
                      max_deg=round(float(p_deg[ok].max()), 3) if nok else None),
        photons=nph, photon_share=round(nph / nok, 6) if nok else 0.0,
        mean_signed_diff_deg=round(float(diff[ok].mean()), 5) if nok else 0.0,
        max_abs_diff_deg=round(float(np.abs(diff).max()), 4),
        positron_high_share=round(float((diff[ok] > 0).mean()), 4) if nok else None,
        photons_with_near_horizon_lt_32m=near, photons_far=nph - near,
        hot_cells=[dict(e=int(oe + cc[k]), n=int(on + cr[k]), photons=int(counts.ravel()[k])) for k in hot if counts.ravel()[k]],
        nodata_pairs=int((e_reach < MIN_REACH_M).sum()))
    if witness_cells:
        rng = np.random.default_rng(seed)
        pick = np.unique(np.concatenate([[0, len(cr) - 1], rng.choice(len(cr), min(witness_cells, len(cr)), replace=False)]))
        wpyr = pyramid_blocks(z, levels)
        we = [electron_cpu(z, cr[k], cc[k]) for k in pick]
        wb, wr, wa = (np.stack([v[i] for v in we]) for i in range(3))
        pb, pr_, _ = positron(np, wpyr, z[cr[pick], cc[pick]], cr[pick], cc[pick], H, W)
        de = np.abs(to_deg(wb) - e_deg[pick]); dp = np.abs(to_deg(pb) - p_deg[pick])
        nan_mismatch = int((np.isnan(to_deg(wb)) != np.isnan(e_deg[pick])).sum() + (np.isnan(to_deg(pb)) != np.isnan(p_deg[pick])).sum())
        de, dp = np.nan_to_num(de), np.nan_to_num(dp)
        rec["witness"] = dict(cells=int(len(pick)), seed=seed, electron_max_diff_deg=float(de.max()),
                              positron_max_diff_deg=float(dp.max()), nan_mismatch=nan_mismatch,
                              reach_mismatch=int((wr != e_reach[pick]).sum() + (pr_ != p_reach[pick]).sum()),
                              horizon_distance_mismatch=int((wa != e_at[pick]).sum()),
                              pyramid_mismatch=int(sum(int((host(a) != b).sum()) for a, b in zip(pyr, wpyr))),
                              photons=int((de > TOL_WIT).sum() + (dp > TOL_WIT).sum() + nan_mismatch
                                          + (wr != e_reach[pick]).sum() + (pr_ != p_reach[pick]).sum()))
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, e_deg.reshape(shape + (NAZ,)), e_reach.reshape(shape + (NAZ,))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy and source.json")
    ap.add_argument("--out", help="default SITE/horizon")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--witness", type=int, default=1024, help="cells re-done on the CPU")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.site, "source.json"), encoding="utf-8"))
    grid = np.load(os.path.join(a.site, "source.npy"))
    if meta.get("rows", "south-to-north") != "south-to-north" or float(meta.get("spacing_m", 1)) != 1.0:
        raise SystemExit("source must be 1 m with rows running south to north")
    oe, on = meta["origin_e_m"], meta["origin_n_m"]
    rec, deg, reach = run(grid, oe, on, use_gpu=not a.cpu, witness_cells=a.witness)
    out = a.out or os.path.join(a.site, "horizon")
    index = write_tiles(out, quantise(deg, reach), oe, on, os.path.basename(os.path.normpath(a.site)))
    rec.update(tiles=len(index["tiles"]), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    with open(os.path.join(out, RECEIPT), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "hot_cells"}, indent=1))
    print("hot cells:", rec["hot_cells"][:5])
    return 0 if rec.get("witness", {}).get("photons", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
