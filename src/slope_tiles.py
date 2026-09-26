"""Slope and aspect tiles from a south-up 1 m height grid, computed as a pair on the GPU.

Two channels, photons back (the electron/positron pattern of pair_gpu.py):

  electron   Horn's 3x3 method: dz/de = ((c + 2f + i) - (a + 2d + g)) / 8h, dz/dn likewise over
             the rows. It weights the whole window, so it smooths.
  positron   Zevenbergen-Thorne: dz/de = (f - d) / 2h, dz/dn = (b - h) / 2h. Only the four
             rook neighbours; sharper, noisier.
  photons    nodes where |slope_Horn - slope_ZT| > 1 percentage point. They are COUNTED and
             located, never suppressed. The two methods differ by a third-derivative term, so
             photons are expected at breaks of slope (banks, ditch edges, walls); the receipt says
             how many photons sit on the top 5 % of |Laplacian| (curvature) against 5 % by chance.
  witness    the whole grid again on the CPU in NumPy by a different code path (sliding 3x3
             windows and weight kernels, not shifted slices); both methods must agree with the
             GPU within 1e-9 percent, and the classes must be identical.

Slope in percent = 100 * sqrt(p^2 + q^2). Aspect is the DOWNSLOPE compass direction,
azimuth = atan2(-p, -q) clockwise from grid north, as an octant 0..7 = N NE E SE S SW W NW,
8 = flat (slope < 0.5 %), 255 = no data. Slope classes (Horn): 0: 0-2, 1: 2-5, 2: 5-10,
3: 10-15, 4: 15-25, 5: >25 percent; 255 = no data (site border, or a missing neighbour).

.gst v1 ("GGS1"), little-endian, 32-byte header then two u8 bodies:
    0 char[4] "GGS1" | 4 u16 version=1 | 6 u16 samples=257 | 8 u16 spacing_mm | 10 u16 flags=0
   12 i32 origin_e_m | 16 i32 origin_n_m (SW corner) | 20 u8 classes=6 | 21 u8 method=1 (Horn)
   22 u16 reserved=0 | 24 u32 nodata_count | 28 u32 steep_count (class >= 3, i.e. >= 10 %)
   32 u8[257*257] slope class, then u8[257*257] aspect; rows SOUTH to NORTH, each WEST to EAST.
Tiles sit on the .ght grid (256 m, edges shared with neighbours) and are indexed in
slope-tiles.json with sha256s; the pair receipt lands beside it as slope_receipt.json.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/slope_tiles.py --site E:/lidar-out/open-land-01
"""
import argparse, hashlib, json, os, struct, sys, time
from datetime import datetime, timezone
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and says so
    cp = None

TILE_M = 256
N = TILE_M + 1
EDGES = (2.0, 5.0, 10.0, 15.0, 25.0)          # percent; class k covers [EDGES[k-1], EDGES[k])
LABELS = ("0-2", "2-5", "5-10", "10-15", "15-25", ">25")
FLAT_PCT = 0.5
TOL_PP = 1.0                                   # percentage points: the photon bound
TOL_WIT = 1e-9
NODATA = 255
FLAT = 8
MAGIC = b"GGS1"
HEADER = struct.Struct("<4sHHHHiiBBHII")
assert HEADER.size == 32
INDEX = "slope-tiles.json"
RECEIPT = "slope_receipt.json"


# ---------------------------------------------------------------- the two methods (GPU or NumPy)
def neighbours(xp, z):
    """3x3 neighbours of every interior node, south-up: a b c is the NORTH row, g h i the south."""
    s = z[:-2, :]; m = z[1:-1, :]; n_ = z[2:, :]
    a, b, c = n_[:, :-2], n_[:, 1:-1], n_[:, 2:]
    d, f = m[:, :-2], m[:, 2:]
    g, h, i = s[:, :-2], s[:, 1:-1], s[:, 2:]
    return a, b, c, d, f, g, h, i


def horn(xp, z, sp):
    a, b, c, d, f, g, h, i = neighbours(xp, z)
    p = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * sp)
    q = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * sp)
    return p, q


def zevenbergen_thorne(xp, z, sp):
    a, b, c, d, f, g, h, i = neighbours(xp, z)
    return (f - d) / (2 * sp), (b - h) / (2 * sp)


def pad(xp, core):
    """Interior result -> full grid with NaN on the border ring."""
    out = xp.full((core.shape[0] + 2, core.shape[1] + 2), xp.nan)
    out[1:-1, 1:-1] = core
    return out


def slope_pct(xp, p, q):
    return 100.0 * xp.sqrt(p * p + q * q)


def classify(xp, s):
    cls = xp.searchsorted(xp.asarray(EDGES), s, side="right").astype(xp.uint8)
    cls[~xp.isfinite(s)] = NODATA
    return cls


def octant(xp, p, q, s):
    az = (xp.degrees(xp.arctan2(-p, -q)) + 360.0) % 360.0
    o = (xp.floor(az / 45.0 + 0.5) % 8).astype(xp.uint8)
    o[s < FLAT_PCT] = FLAT
    o[~xp.isfinite(s)] = NODATA
    return o


def laplacian(xp, z, sp):
    a, b, c, d, f, g, h, i = neighbours(xp, z)
    return (b + d + f + h - 4 * z[1:-1, 1:-1]) / (sp * sp)


# ---------------------------------------------------------------- CPU witness: another code path
def witness(z, sp):
    """Both methods by weight kernels over sliding windows (NumPy, float64)."""
    w = np.lib.stride_tricks.sliding_window_view(z, (3, 3))   # w[r, c, dr, dc], dr=0 is SOUTH
    kx_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * sp)
    ky_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * sp)
    kx_z = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2 * sp)
    ky_z = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / (2 * sp)
    k = lambda ker: np.einsum("rcij,ij->rc", w, ker)
    return (k(kx_h), k(ky_h)), (k(kx_z), k(ky_z))


# ---------------------------------------------------------------- tiles
def encode(cls, asp, e0, n0, spacing_mm=1000):
    if cls.shape != (N, N) or asp.shape != (N, N):
        raise ValueError(f"slope tile must be {N}x{N}")
    nod = int((cls == NODATA).sum()); steep = int(((cls >= 3) & (cls != NODATA)).sum())
    head = HEADER.pack(MAGIC, 1, N, spacing_mm, 0, int(e0), int(n0), len(LABELS), 1, 0, nod, steep)
    return head + np.ascontiguousarray(cls, np.uint8).tobytes() + np.ascontiguousarray(asp, np.uint8).tobytes()


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("short .gst blob")
    magic, ver, n, sp, flags, e0, n0, ncls, method, _, nod, steep = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    if len(blob) != HEADER.size + 2 * n * n:
        raise ValueError(f".gst size {len(blob)} != {HEADER.size + 2 * n * n}")
    body = np.frombuffer(blob, np.uint8, offset=HEADER.size)
    head = dict(samples=n, spacing_mm=sp, origin_e_m=e0, origin_n_m=n0, classes=ncls, method=method,
                nodata_count=nod, steep_count=steep)
    return head, body[:n * n].reshape(n, n), body[n * n:].reshape(n, n)


def write_tiles(out_dir, cls, asp, oe, on, site_name, spacing_mm=1000):
    rows, cols = cls.shape
    if (rows - 1) % TILE_M or (cols - 1) % TILE_M:
        raise ValueError(f"grid {cls.shape} is not k*256+1 per side")
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    entries = []
    for iy in range((rows - 1) // TILE_M):
        for ix in range((cols - 1) // TILE_M):
            r0, c0 = iy * TILE_M, ix * TILE_M
            sc, sa = cls[r0:r0 + N, c0:c0 + N], asp[r0:r0 + N, c0:c0 + N]
            blob = encode(sc, sa, oe + c0, on + r0, spacing_mm)
            key = f"{ix}_{iy}"; rel = f"tiles/{key}.gst"
            with open(os.path.join(out_dir, rel), "wb") as f:
                f.write(blob)
            valid = sc != NODATA
            share = [round(float((sc[valid] == k).mean()), 6) if valid.any() else 0.0 for k in range(len(LABELS))]
            entries.append(dict(key=key, file=rel, sha256=hashlib.sha256(blob).hexdigest(), e0=int(oe + c0),
                                n0=int(on + r0), bytes=len(blob), steep=int(((sc >= 3) & valid).sum()), share=share))
    index = dict(format="gst1", crs="EPSG:27700", site=dict(name=site_name, origin_e=int(oe), origin_n=int(on)),
                 tile_m=TILE_M, spacing_m=spacing_mm / 1000, method="horn", classes_pct=LABELS,
                 aspect="0..7 downslope N NE E SE S SW W NW, 8 flat (<0.5 %), 255 no data",
                 tiles=entries, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    return index


# ---------------------------------------------------------------- the pair
def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def run(grid, oe=0, on=0, sp=1.0, use_gpu=True, witness_on=True):
    """grid: south-up float64 (NaN = no data). Returns (receipt, slope classes, aspect octants, photon mask)."""
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    z = xp.asarray(grid, dtype=xp.float64)
    ph, qh = horn(xp, z, sp); pz, qz = zevenbergen_thorne(xp, z, sp)
    sh, sz = slope_pct(xp, ph, qh), slope_pct(xp, pz, qz)
    diff = xp.abs(sh - sz)
    valid = xp.isfinite(diff)
    photon = valid & (diff > TOL_PP)
    lap = xp.abs(laplacian(xp, z, sp))
    nvalid = int(valid.sum()); nph = int(photon.sum())

    # where the photons are: on breaks of slope (top 5 % of |Laplacian|) or not
    lv = lap[valid]
    cut = float(xp.percentile(lv, 95)) if nvalid else 0.0
    brk = xp.isfinite(lap) & (lap >= cut) & (lap > 1e-6)   # 1e-6 per m: above float noise
    near_brk = brk.copy()                                        # within one node of a break
    near_brk[1:, :] |= brk[:-1, :]; near_brk[:-1, :] |= brk[1:, :]
    near_brk[:, 1:] |= near_brk[:, :-1].copy(); near_brk[:, :-1] |= near_brk[:, 1:].copy()
    near_brk &= valid
    on_break = int((photon & near_brk).sum())
    chance = float(near_brk.sum()) / nvalid if nvalid else 0.0
    ph_host = to_host(photon)
    rr, cc = np.nonzero(ph_host)
    cells = {}
    for r, c in zip((rr + 1) // 16, (cc + 1) // 16):
        cells[(r, c)] = cells.get((r, c), 0) + 1
    hot = sorted(cells.items(), key=lambda kv: -kv[1])[:8]
    d_host = to_host(diff)
    worst = []
    if nph:
        top = np.argsort(np.where(ph_host, d_host, -1).ravel())[::-1][:8]
        for k in top:
            r, c = divmod(int(k), d_host.shape[1])
            worst.append(dict(e=oe + c + 1, n=on + r + 1, horn_pct=round(float(to_host(sh[r, c])), 3),
                              zt_pct=round(float(to_host(sz[r, c])), 3), diff_pp=round(float(d_host[r, c]), 3)))

    sfull = pad(xp, sh)
    cls = to_host(classify(xp, sfull)); asp = to_host(octant(xp, pad(xp, ph), pad(xp, qh), sfull))
    vc = cls != NODATA
    share = {LABELS[k]: round(float((cls[vc] == k).mean()), 6) for k in range(len(LABELS))}
    rec = dict(
        device=device, nodes=int(grid.size), valid=nvalid, tol_pp=TOL_PP,
        electron=dict(method="horn", mean_pct=round(float(sh[valid].mean()), 4) if nvalid else None,
                      max_pct=round(float(sh[valid].max()), 3) if nvalid else None),
        positron=dict(method="zevenbergen-thorne", mean_pct=round(float(sz[valid].mean()), 4) if nvalid else None,
                      max_pct=round(float(sz[valid].max()), 3) if nvalid else None),
        photons=nph, photon_share=round(nph / nvalid, 6) if nvalid else 0.0,
        max_diff_pp=round(float(diff[valid].max()), 4) if nvalid else 0.0,
        breaks=dict(laplacian_p95_per_m=round(cut, 5), photons_within_1m_of_top5pct_curvature=on_break,
                    share_on_breaks=round(on_break / nph, 4) if nph else None, by_chance=round(chance, 4)),
        hot_16m_cells=[dict(e=int(oe + c * 16), n=int(on + r * 16), photons=int(v)) for (r, c), v in hot],
        worst=worst, share=share)
    if witness_on:
        g = np.asarray(grid, np.float64)
        (hp, hq), (zp, zq) = witness(g, sp)
        wh, wz = 100 * np.hypot(hp, hq), 100 * np.hypot(zp, zq)
        both = np.isfinite(wh) & np.isfinite(to_host(sh))
        dh = np.abs(wh - to_host(sh))[both]; dz = np.abs(wz - to_host(sz))[both]
        wcls = np.full(g.shape, NODATA, np.uint8); wcls[1:-1, 1:-1] = np.where(
            np.isfinite(wh), np.digitize(np.nan_to_num(wh), EDGES), NODATA)
        wph = int((np.abs(wh - wz) > TOL_PP)[both].sum())
        # a class may flip only where the slope sits within TOL_WIT of a class edge (an ulp tie)
        near = np.zeros(g.shape, bool)
        near[1:-1, 1:-1] = np.isfinite(wh) & (np.abs(np.nan_to_num(wh)[..., None] - np.asarray(EDGES)).min(-1) <= TOL_WIT)
        flip = wcls != cls
        rec["witness"] = dict(nodes=int(both.sum()), horn_max_diff=float(dh.max()) if dh.size else 0.0,
                              zt_max_diff=float(dz.max()) if dz.size else 0.0,
                              class_mismatch=int((flip & ~near).sum()), edge_ties=int((flip & near).sum()),
                              photons_cpu=wph,
                              photons=int((dh > TOL_WIT).sum() + (dz > TOL_WIT).sum() + (flip & ~near).sum()
                                          + abs(wph - nph)))
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec, cls, asp, ph_host


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", required=True, help="folder with source.npy and source.json")
    ap.add_argument("--out", help="default SITE/slope")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.site, "source.json")))
    grid = np.load(os.path.join(a.site, "source.npy"))
    if meta.get("rows", "south-to-north") != "south-to-north":
        raise SystemExit("source rows must run south to north")
    oe, on, sp = meta["origin_e_m"], meta["origin_n_m"], float(meta.get("spacing_m", 1))
    rec, cls, asp, _ = run(grid, oe, on, sp, use_gpu=not a.cpu)
    out = a.out or os.path.join(a.site, "slope")
    index = write_tiles(out, cls, asp, oe, on, os.path.basename(os.path.normpath(a.site)), int(round(sp * 1000)))
    rec.update(tiles=len(index["tiles"]), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               index_sha256=hashlib.sha256(open(os.path.join(out, INDEX), "rb").read()).hexdigest())
    with open(os.path.join(out, RECEIPT), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("device", "valid", "photons", "photon_share", "max_diff_pp", "breaks",
                                          "share", "witness", "tiles", "wall_s", "index_sha256")}, indent=1))
    print("hot 16 m cells:", rec["hot_16m_cells"][:5])
    return 0 if rec.get("witness", {}).get("photons", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
