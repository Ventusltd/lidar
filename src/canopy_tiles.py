"""Canopy tiles: hedges and trees from the EA LIDAR composites, classified as a pair on the GPU.

canopy height  chm = first-return DSM - DTM (both 1 m, same node grid). Cells below H_MIN are open.
Without building footprints, tall cells are told apart by two independent signals:

  electron  RETURNS. A laser pulse passes partly through leaves, so on vegetation the last return
            sits well below the first; on a roof, a wall, a vehicle or a bale stack they coincide.
            A tall cell is vegetation when at least FRAC_A of the tall cells in its 5x5 window have
            first - last >= PEN_M. (Last-return DSM; its surveys can differ in date from the first-
            return composite, so large negative differences are counted as date mismatches.)
  positron  SHAPE. A roof is made of planes; a crown is not. The RMS residual of a least-squares
            plane through the 3x3 first-return window is the roughness; a tall cell is vegetation
            when at least FRAC_B of the tall cells in its 5x5 window have roughness >= ROUGH_M.
  photons   tall cells where the two disagree. COUNTED and located, never suppressed: they become
            class 4 "uncertain" (with vegetation on the rim of a structure), which, like class 3 "structure", is NEVER displayed (rule: no
            buildings on screen). Only cells both formulations call vegetation are shown.
  witness   the whole grid again on the CPU in NumPy by another code path (sliding windows and a
            projection matrix instead of shifted slices, integral images and a closed form); the
            window counts must match exactly and the mean-square roughness within 1e-9 m^2.

Classes: 0 open (< H_MIN), 1 hedge, 2 tree canopy, 3 structure (hidden), 4 uncertain (hidden), 255 no data.
Hedge versus tree comes from canopy_lines.hedgerows (opening, thinning, tracing).

.gcn v1 ("GGC1"), little-endian, 32-byte header then two u8 bodies:
    0 char[4] "GGC1" | 4 u16 version=1 | 6 u16 samples=257 | 8 u16 spacing_mm | 10 u16 flags=0
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u16 height_step_mm=200 | 22 u16 reserved=0
   24 u32 nodata_count | 28 u32 shown_count (classes 1 and 2)
   32 u8[257*257] class, then u8[257*257] canopy height in 0.2 m steps (0..254, 255 = 50.8 m or more),
   written for classes 1 and 2 only (0 elsewhere: no heights are published for hidden cells).
   Rows SOUTH to NORTH, each WEST to EAST. Tiles sit on the .ght grid; canopy-tiles.json indexes them
   (sha256 each) and names hedges.json (polylines, BNG metres); canopy_receipt.json holds the pair.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/canopy_tiles.py --site E:/lidar-out/open-land-01
"""
import argparse, glob, hashlib, json, os, struct, sys, time
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canopy_lines  # noqa: E402

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and says so
    cp = None

TILE_M = 256
N = TILE_M + 1
H_MIN = 1.0          # m: lower than this is open ground (grass, crops, walls under a metre)
PEN_M = 0.5          # m: first minus last return that counts as a pulse getting through
FRAC_A = 0.35
ROUGH_M = 0.3        # m RMS about a 3x3 plane
FRAC_B = 0.5
DATE_MISMATCH_M = -0.5
EAVES_M = 2          # m: vegetation this close to a structure is hidden as uncertain
TOL_WIT = 1e-9
STEP_MM = 200
OPEN, HEDGE, TREE, STRUCT, UNSURE, NODATA = 0, 1, 2, 3, 4, 255
LABELS = ("open", "hedge", "tree", "structure (hidden)", "uncertain (hidden)")
MAGIC = b"GGC1"
HEADER = struct.Struct("<4sHHHHiiHHII")
assert HEADER.size == 32
INDEX, RECEIPT, HEDGES = "canopy-tiles.json", "canopy_receipt.json", "hedges.json"
ATTRIBUTION = "© Environment Agency copyright and/or database right 2022. All rights reserved."
SOURCES = {
    "dtm": "https://www.data.gov.uk/dataset/01b3ee39-da3f-47b6-83da-dc98e73a461f",
    "first_return_dsm": "https://www.data.gov.uk/dataset/92534f24-0b92-4b28-9986-347cf6678b39",
    "last_return_dsm": "https://www.data.gov.uk/dataset/cf3f1137-c12b-44a1-a835-e80fe4a60b92",
}


# ---------------------------------------------------------------- GPU (or NumPy) formulations
def box5(xp, m):
    """5x5 window sums of a 0/1 grid by an integral image (zero outside the grid)."""
    p = xp.pad(m.astype(xp.int32), ((3, 2), (3, 2)))
    s = p.cumsum(0).cumsum(1)
    return s[5:, 5:] - s[:-5, 5:] - s[5:, :-5] + s[:-5, :-5]


def roughness(xp, z):
    """RMS residual of the least-squares plane through each 3x3 window (edge-padded).

    With x, y in {-1, 0, 1}: p = sum(x z) / 6, q = sum(y z) / 6 and RSS = sum (z - mean)^2 - 6p^2 - 6q^2.
    """
    zp = xp.pad(z, 1, mode="edge")
    R, C = z.shape
    ctr = zp[1:1 + R, 1:1 + C]                   # heights relative to the centre: no cancellation at 100 m
    win = [(dr, dc, zp[1 + dr:1 + dr + R, 1 + dc:1 + dc + C] - ctr) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    mean = sum(w for _, _, w in win) / 9.0
    ss = sum((w - mean) ** 2 for _, _, w in win)
    p = sum(dc * w for _, dc, w in win) / 6.0
    q = sum(dr * w for dr, _, w in win) / 6.0
    return xp.sqrt(xp.maximum(ss - 6 * p * p - 6 * q * q, 0.0) / 9.0)


def formulations(xp, dtm, fz, lz):
    chm = fz - dtm
    pen = fz - lz
    valid = xp.isfinite(chm) & xp.isfinite(pen)
    chm0 = xp.where(valid, chm, 0.0)
    tall = valid & (chm0 >= H_MIN)
    nt = box5(xp, tall)
    na = box5(xp, tall & (xp.where(valid, pen, 0.0) >= PEN_M))
    rough = roughness(xp, xp.where(xp.isfinite(fz), fz, 0.0))
    nb = box5(xp, tall & (rough >= ROUGH_M))
    veg_a = tall & (na >= FRAC_A * nt)
    veg_b = tall & (nb >= FRAC_B * nt)
    return dict(chm=chm, pen=pen, valid=valid, tall=tall, nt=nt, na=na, nb=nb, rough=rough, veg_a=veg_a, veg_b=veg_b)


# ---------------------------------------------------------------- CPU witness: another code path
def witness(dtm, fz, lz):
    W = np.lib.stride_tricks.sliding_window_view
    chm, pen = fz - dtm, fz - lz
    valid = np.isfinite(chm) & np.isfinite(pen)
    tall = valid & (np.nan_to_num(chm, nan=-1e9) >= H_MIN)
    count = lambda m: W(np.pad(m, 2), (5, 5)).sum((-1, -2), dtype=np.int64)
    y, x = np.mgrid[-1:2, -1:2]
    A = np.c_[np.ones(9), x.ravel(), y.ravel()]            # window order: row dr=-1 (south) first
    resid = np.eye(9) - A @ np.linalg.pinv(A)
    w = W(np.pad(np.nan_to_num(fz, nan=0.0), 1, mode="edge"), (3, 3)).reshape(fz.shape + (9,))
    rough = np.sqrt(((w @ resid.T) ** 2).mean(-1))
    nt = count(tall)
    na = count(tall & (np.nan_to_num(pen, nan=-1e9) >= PEN_M))
    nb = count(tall & (rough >= ROUGH_M))
    return dict(tall=tall, nt=nt, na=na, nb=nb, rough=rough,
                veg_a=tall & (na >= FRAC_A * nt), veg_b=tall & (nb >= FRAC_B * nt))


def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def hot_cells(mask, oe, on, cell=32, top=8):
    r, c = np.nonzero(mask)
    if not len(r):
        return []
    keys, counts = np.unique((r // cell) * 100000 + (c // cell), return_counts=True)
    order = np.argsort(-counts, kind="stable")[:top]
    return [dict(e=int(oe + (keys[k] % 100000) * cell), n=int(on + (keys[k] // 100000) * cell), cells=int(counts[k]))
            for k in order]


def run(dtm, fz, lz, oe=0, on=0, use_gpu=True, witness_on=True):
    """South-up float grids (NaN = no data). Returns (receipt, class grid u8, height grid u8, hedge lines)."""
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    f = formulations(xp, *(xp.asarray(g, dtype=xp.float64) for g in (dtm, fz, lz)))
    h = {k: to_host(v) for k, v in f.items()}
    tall, va, vb, valid = h["tall"], h["veg_a"], h["veg_b"], h["valid"]
    photon = tall & (va != vb)
    struct = tall & ~va & ~vb
    # Eaves and roof edges return twice and look rough, so both methods can call a building's rim
    # vegetation. Vegetation within EAVES_M of a structure is demoted to uncertain unless it is part
    # of a wide canopy (a big tree beside a barn stays a tree).
    # A pitched roof is rough at its ridge and hips but returns once, so it reads "uncertain, returns
    # say structure": that counts as structure for the rim.
    solid = struct | (photon & ~va)
    rim = (va & vb) & canopy_lines.dilate(solid, EAVES_M) & ~canopy_lines.opening(va & vb)
    veg = va & vb & ~rim
    cls = np.full(tall.shape, NODATA, np.uint8)
    cls[valid] = OPEN
    cls[struct] = STRUCT
    cls[photon | rim] = UNSURE
    hedge, wide, lines, lstats = canopy_lines.hedgerows(veg, np.where(valid, h["chm"], np.nan), oe, on)
    cls[veg] = TREE
    cls[veg & hedge] = HEDGE
    shown = (cls == HEDGE) | (cls == TREE)
    hq = np.zeros(cls.shape, np.uint8)
    hq[shown] = np.clip(np.rint(h["chm"][shown] * 1000 / STEP_MM), 0, 255).astype(np.uint8)
    nt, nv = int(tall.sum()), int(valid.sum())
    share = {LABELS[k]: round(float((cls[valid] == k).mean()), 6) if nv else 0.0 for k in range(len(LABELS))}
    date_mm = valid & (h["pen"] < DATE_MISMATCH_M)
    rec = dict(
        device=device, nodes=int(cls.size), valid=nv, tall=nt,
        thresholds=dict(h_min_m=H_MIN, pen_m=PEN_M, frac_a=FRAC_A, rough_m=ROUGH_M, frac_b=FRAC_B),
        electron=dict(method="returns: first minus last >= pen_m over a 5x5 share", vegetation=int(va.sum())),
        positron=dict(method="shape: 3x3 plane residual >= rough_m over a 5x5 share", vegetation=int(vb.sum())),
        photons=int(photon.sum()), photon_share_of_tall=round(int(photon.sum()) / nt, 6) if nt else 0.0,
        rim_demoted=int(rim.sum()), photons_a_says_vegetation=int((photon & va).sum()), photons_b_says_vegetation=int((photon & vb).sum()),
        hot_32m_cells=hot_cells(photon, oe, on),
        structure_cells=int((cls == STRUCT).sum()), structure_hot_32m_cells=hot_cells(cls == STRUCT, oe, on),
        date_mismatch_cells=int(date_mm.sum()), share=share, hedgerows=lstats,
        canopy_height_m=dict(p50=round(float(np.median(h["chm"][shown])), 2) if shown.any() else None,
                             p95=round(float(np.percentile(h["chm"][shown], 95)), 2) if shown.any() else None,
                             max=round(float(h["chm"][shown].max()), 2) if shown.any() else None))
    if witness_on:
        w = witness(*(np.asarray(g, np.float64) for g in (dtm, fz, lz)))
        dr = np.abs(w["rough"] ** 2 - h["rough"] ** 2)                # compared as mean square (m^2):
        near = np.abs(w["rough"] ** 2 - ROUGH_M ** 2) <= TOL_WIT       # a root near 0 magnifies ulps
        flips = {k: int((w[k] != h[k]).sum()) for k in ("tall", "nt", "na", "veg_a")}
        tie = canopy_lines.dilate(near, 2)
        nb_flip = int((((w["nb"] != h["nb"]) | (w["veg_b"] != h["veg_b"])) & ~tie).sum())
        opened = canopy_lines.opening_witness(veg) & veg
        rec["witness"] = dict(rough_sq_max_diff_m2=float(dr.max()), mismatches=flips, nb_mismatch=nb_flip,
                              threshold_ties=int(tie.sum()),
                              opening_mismatch=int((opened != wide).sum()),
                              photons_cpu=int((w["tall"] & (w["veg_a"] != w["veg_b"])).sum()))
        rec["witness"]["photons"] = int(sum(flips.values()) + nb_flip + rec["witness"]["opening_mismatch"]
                                        + (dr.max() > TOL_WIT) + abs(rec["witness"]["photons_cpu"] - rec["photons"]))
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, cls, hq, lines


# ---------------------------------------------------------------- tiles
def encode(cls, hq, e0, n0, spacing_mm=1000):
    if cls.shape != (N, N) or hq.shape != (N, N):
        raise ValueError(f"canopy tile must be {N}x{N}")
    nod = int((cls == NODATA).sum())
    shown = int(((cls == HEDGE) | (cls == TREE)).sum())
    head = HEADER.pack(MAGIC, 1, N, spacing_mm, 0, int(e0), int(n0), STEP_MM, 0, nod, shown)
    return head + np.ascontiguousarray(cls, np.uint8).tobytes() + np.ascontiguousarray(hq, np.uint8).tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .gcn blob")
    magic, ver, n, sp, flags, e0, n0, step, _, nod, shown = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    if len(blob) != HEADER.size + 2 * n * n:
        raise ValueError(f".gcn size {len(blob)} != {HEADER.size + 2 * n * n}")
    body = np.frombuffer(blob, np.uint8, offset=HEADER.size)
    head = dict(samples=n, spacing_mm=sp, origin_e_m=e0, origin_n_m=n0, height_step_mm=step,
                nodata_count=nod, shown_count=shown)
    return head, body[:n * n].reshape(n, n), body[n * n:].reshape(n, n)


def dump(path, obj):
    """JSON with LF endings, UTF-8; returns its sha256."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_tiles(out_dir, cls, hq, lines, oe, on, site_name, spacing_mm=1000):
    rows, cols = cls.shape
    if (rows - 1) % TILE_M or (cols - 1) % TILE_M:
        raise ValueError(f"grid {cls.shape} is not k*256+1 per side")
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    common = dict(crs="EPSG:27700", licence="OGL-3.0",
                  licence_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
                  attribution=ATTRIBUTION, sources=SOURCES)
    hsha = dump(os.path.join(out_dir, HEDGES), dict(format="gcl1", site=site_name, units="m",
                                                    coords="easting, northing, hedge height above ground (m)",
                                                    **common, lines=lines))
    entries = []
    for iy in range((rows - 1) // TILE_M):
        for ix in range((cols - 1) // TILE_M):
            r0, c0 = iy * TILE_M, ix * TILE_M
            sc, sh = cls[r0:r0 + N, c0:c0 + N], hq[r0:r0 + N, c0:c0 + N]
            blob = encode(sc, sh, oe + c0, on + r0, spacing_mm)
            key = f"{ix}_{iy}"; rel = f"tiles/{key}.gcn"
            with open(os.path.join(out_dir, rel), "wb") as fh:
                fh.write(blob)
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + c0),
                                n0=int(on + r0), bytes=len(blob), hedge=int((sc == HEDGE).sum()),
                                tree=int((sc == TREE).sum()), hidden=int(((sc == STRUCT) | (sc == UNSURE)).sum())))
    index = dict(format="gcn1", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)), tile_m=TILE_M,
                 spacing_m=spacing_mm / 1000, height_step_m=STEP_MM / 1000, classes=list(LABELS), shown=[HEDGE, TREE],
                 note=("Vegetation is classified without building footprints: cells both methods call vegetation "
                       "are shown; structure and uncertain cells are never shown and carry no height."),
                 **common, hedges=dict(file=HEDGES, sha256=hsha, lines=len(lines)), tiles=entries,
                 generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    dump(os.path.join(out_dir, INDEX), index)
    return index


def load_surface(site, product, oe, on, rows, cols, cache=None):
    """Mosaic a cached DSM product onto the site grid (never fetches; fetch_wcs.fetch_box does)."""
    from geotiff_read import mosaic
    cache = cache or os.path.join(r"E:\lidar-cache", os.path.basename(os.path.normpath(site)))
    paths = sorted(glob.glob(os.path.join(cache, f"{product}_*.tif")))
    if not paths:
        raise SystemExit(f"no {product} GeoTIFFs in {cache}: fetch them with fetch_wcs.fetch_box(product=...)")
    return mosaic(paths, oe, on, cols, rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy (DTM) and source.json")
    ap.add_argument("--cache", help=r"raw GeoTIFF cache, default E:\lidar-cache\<site name>")
    ap.add_argument("--out", help="default SITE/canopy")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.site, "source.json")))
    dtm = np.load(os.path.join(a.site, "source.npy"))
    if meta.get("rows", "south-to-north") != "south-to-north":
        raise SystemExit("source rows must run south to north")
    oe, on = meta["origin_e_m"], meta["origin_n_m"]
    fz = load_surface(a.site, "fzdsm1m", oe, on, *dtm.shape, a.cache)
    lz = load_surface(a.site, "lzdsm1m", oe, on, *dtm.shape, a.cache)
    rec, cls, hq, lines = run(dtm, fz, lz, oe, on, use_gpu=not a.cpu)
    out = a.out or os.path.join(a.site, "canopy")
    index = write_tiles(out, cls, hq, lines, oe, on, os.path.basename(os.path.normpath(a.site)))
    rec.update(tiles=len(index["tiles"]), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               lines_sha256=hashlib.sha256(open(canopy_lines.__file__, "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    dump(os.path.join(out, RECEIPT), rec)
    print(json.dumps({k: rec[k] for k in ("device", "tall", "photons", "photon_share_of_tall", "structure_cells",
                                          "date_mismatch_cells", "share", "hedgerows", "canopy_height_m",
                                          "witness", "tiles", "wall_s", "index_sha256")}, indent=1))
    return 0 if rec.get("witness", {}).get("photons", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
