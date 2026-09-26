"""Copernicus GLO-30 tiles against the EA 1 m DTM: the real difference, surface minus bare earth.

  E:/swarm/gpu-bench/venv/Scripts/python.exe src/copernicus_pair.py \
      --ea E:/lidar-out/open-land-01 --cop E:/lidar-out/open-land-01-copernicus

Two checks, each a GPU pair with a CPU witness. Disagreements are counted, never suppressed.

  tiles     every Copernicus .ght node inside the EA site is recomputed from the cached source
            mosaic (copernicus.read_box, no network) at the node's WGS84 position: channel A by
            two lerps, channel B by the four-corner polynomial, on the GPU. A and B must agree
            within 1e-9 m; the stored tile must match within 0.005 m (centimetre rounding).
            Witness: the CPU recomputes every node by a third form (corner weights).
  compare   the EA 1 m DTM is brought to the Copernicus spacing by the mean of the
            (spacing+1) x (spacing+1) EA samples centred on each node (a 33 x 33 m block at
            32 m), because GLO-30 is itself an area average. Channel A takes block sums from a
            summed-area table; channel B gathers each block and sums it directly. They must agree
            within 1e-6 m. Witness: the CPU takes each block's mean from a NumPy slice, node by
            node, and recomputes every statistic; the GPU and CPU statistics must agree.

The difference d = Copernicus - EA mixes three things this script does not separate: vegetation
and buildings (the surface model sees them, the DTM does not), the EGM2008 - ODN Newlyn datum
offset, and the stated errors of both products (< 4 m LE90 against 0.15 m RMSE). The median of d
over open ground is the best single estimate of the datum offset; the upper tail is canopy.
The EA grid node sits 0.5 m south-west of its cell centre; at 32 m that shift is ignored.

The receipt (copernicus_pair_receipt.json, LF) lands next to the Copernicus tiles.json and its
sha256 is written into that tiles.json under "receipt".
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut_tiles import decode_tile, load_tiles  # noqa: E402
import copernicus  # noqa: E402
import osgb  # noqa: E402

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card: the same arithmetic runs in NumPy and the receipt says so
    cp = None

TOL_PAIR = 1e-9
TOL_Q = 0.005 + 1e-9
TOL_MEAN = 1e-6
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
BINS = np.arange(-10.0, 31.0, 1.0)
RECEIPT = "copernicus_pair_receipt.json"


def load_cop(cop_dir):
    """Copernicus tiles.json and one south-up grid at its own spacing."""
    man = json.load(open(os.path.join(cop_dir, "tiles.json"), encoding="utf-8"))
    sp, tm = man["spacing_m"], man["tile_m"]
    oe, on = man["site"]["origin_e"], man["site"]["origin_n"]
    nx = (max(t["e0"] for t in man["tiles"]) + tm - oe) // sp + 1
    ny = (max(t["n0"] for t in man["tiles"]) + tm - on) // sp + 1
    grid = np.full((ny, nx), np.nan)
    for t in man["tiles"]:
        blob = open(os.path.join(cop_dir, t["file"]), "rb").read()
        if hashlib.sha256(blob).hexdigest() != t["sha256"]:
            raise ValueError(f"sha256 mismatch for {t['file']}")
        head, h = decode_tile(blob)
        if head["spacing_mm"] != sp * 1000:
            raise ValueError(f"{t['file']}: spacing {head['spacing_mm']} mm, index says {sp} m")
        r0, c0 = (head["origin_n_m"] - on) // sp, (head["origin_e_m"] - oe) // sp
        grid[r0:r0 + h.shape[0], c0:c0 + h.shape[1]] = h
    return man, grid


def overlap_nodes(cop_man, ea_man, ea_shape, half):
    """Row/col indices of Copernicus nodes whose EA block lies wholly inside the EA grid."""
    sp = cop_man["spacing_m"]
    coe, con = cop_man["site"]["origin_e"], cop_man["site"]["origin_n"]
    eoe, eon = ea_man["site"]["origin_e"], ea_man["site"]["origin_n"]
    ny, nx = ea_shape
    k = np.arange(0, 1 << 16)
    ce = coe + k * sp - eoe          # EA column of each Copernicus column
    cn = con + k * sp - eon
    cols = k[(ce - half >= 0) & (ce + half <= nx - 1)]
    rows = k[(cn - half >= 0) & (cn + half <= ny - 1)]
    return rows, cols, (con + rows * sp - eon), (coe + cols * sp - eoe)


def stats(xp, d):
    d = d[~xp.isnan(d)]
    out = dict(n=int(d.size), mean_m=float(d.mean()), std_m=float(d.std()),
               rmse_m=float(xp.sqrt((d * d).mean())), min_m=float(d.min()), max_m=float(d.max()),
               abs_le90_m=float(xp.percentile(xp.abs(d), 90)),
               percentiles_m={str(p): float(v) for p, v in zip(PCTS, _host(xp.percentile(d, xp.asarray(PCTS, dtype=float))))},
               share_over_2m=float((d > 2).mean()), share_over_5m=float((d > 5).mean()),
               share_under_minus_2m=float((d < -2).mean()))
    h, _ = xp.histogram(xp.clip(d, BINS[0], BINS[-1] - 1e-9), bins=xp.asarray(BINS))
    out["histogram_1m"] = dict(edges_m=[float(b) for b in BINS],
                               counts=[int(c) for c in (cp.asnumpy(h) if xp is not np else h)],
                               note="values outside [-10, 30) are clipped into the end bins")
    return out


def run(ea_dir, cop_dir, use_gpu=True, cache_dir=copernicus.CACHE_ROOT, log=print):
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = (cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np
              else "cpu (numpy)")
    ea_man, ea = load_tiles(ea_dir)
    cop_man, cop = load_cop(cop_dir)
    sp = cop_man["spacing_m"]
    half = sp // 2
    rows, cols, er, ec = overlap_nodes(cop_man, ea_man, ea.shape, half)
    if rows.size == 0 or cols.size == 0:
        raise SystemExit("the Copernicus tiles do not overlap the EA site")
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    ER, EC = np.meshgrid(er, ec, indexing="ij")
    photons = {}

    # ---- tiles: stored nodes against a fresh bilinear from the cached source mosaic
    e = cop_man["site"]["origin_e"] + cc * sp
    n = cop_man["site"]["origin_n"] + rr * sp
    lat, lon = osgb.bng_to_wgs84(e.astype(float), n.astype(float))
    m = copernicus.read_box(lat.min(), lon.min(), lat.max(), lon.max(),
                            rc=copernicus.RangeCache(cache_dir, fetch=_no_network))
    gy = (m["lat_top"] - lat) / m["dlat"]
    gx = (lon - m["lon_left"]) / m["dlon"]
    i, j = np.floor(gy).astype(np.int64), np.floor(gx).astype(np.int64)
    fy, fx = gy - i, gx - j
    z = m["z"]
    c4 = [z[i, j], z[i, j + 1], z[i + 1, j], z[i + 1, j + 1]]
    g4 = [xp.asarray(a) for a in c4]
    gfx, gfy = xp.asarray(fx), xp.asarray(fy)
    a_lo = g4[0] + (g4[1] - g4[0]) * gfx
    a_hi = g4[2] + (g4[3] - g4[2]) * gfx
    A = a_lo + (a_hi - a_lo) * gfy
    B = g4[0] + (g4[1] - g4[0]) * gfx + (g4[2] - g4[0]) * gfy + (g4[3] - g4[1] - g4[2] + g4[0]) * gfx * gfy
    W = (c4[0] * (1 - fx) * (1 - fy) + c4[1] * fx * (1 - fy) + c4[2] * (1 - fx) * fy + c4[3] * fx * fy)
    stored = cop[rr, cc]
    photons["tiles_pair"] = int((xp.abs(A - B) > TOL_PAIR).sum())
    photons["tiles_vs_stored"] = int((xp.abs(A - xp.asarray(stored)) > TOL_Q).sum())
    photons["tiles_witness"] = int((np.abs(_host(A) - W) > TOL_PAIR).sum())
    tiles_max = float(xp.abs(A - xp.asarray(stored)).max())

    # ---- compare: EA block means two ways on the GPU, then the CPU witness
    g = xp.asarray(ea)
    sat = xp.zeros((g.shape[0] + 1, g.shape[1] + 1))
    sat[1:, 1:] = xp.cumsum(xp.cumsum(g, axis=0), axis=1)
    R0, R1 = xp.asarray(ER - half), xp.asarray(ER + half + 1)
    C0, C1 = xp.asarray(EC - half), xp.asarray(EC + half + 1)
    area = float((2 * half + 1) ** 2)
    mean_a = (sat[R1, C1] - sat[R0, C1] - sat[R1, C0] + sat[R0, C0]) / area
    off = xp.arange(-half, half + 1)
    blk = g[xp.asarray(ER)[..., None, None] + off[:, None], xp.asarray(EC)[..., None, None] + off[None, :]]
    mean_b = blk.sum(axis=(-2, -1)) / area
    photons["mean_pair"] = int((xp.abs(mean_a - mean_b) > TOL_MEAN).sum())
    d_gpu = xp.asarray(stored) - mean_b
    d_point = xp.asarray(stored) - g[xp.asarray(ER), xp.asarray(EC)]
    s_gpu = stats(xp, d_gpu.ravel())
    s_point = stats(xp, d_point.ravel())

    w_mean = np.array([[ea[a - half:a + half + 1, b - half:b + half + 1].mean() for b in ec] for a in er])
    photons["mean_witness"] = int((np.abs(_host(mean_b) - w_mean) > TOL_MEAN).sum())
    s_cpu = stats(np, (stored - w_mean).ravel())
    photons["stats_witness"] = sum(int(abs(s_cpu[k] - s_gpu[k]) > TOL_MEAN)
                                   for k in ("mean_m", "std_m", "rmse_m", "abs_le90_m"))
    photons["stats_witness"] += sum(int(abs(s_cpu["percentiles_m"][k] - s_gpu["percentiles_m"][k]) > TOL_MEAN)
                                    for k in s_cpu["percentiles_m"])
    photons["hist_witness"] = int(s_cpu["histogram_1m"]["counts"] != s_gpu["histogram_1m"]["counts"])

    script = open(os.path.abspath(__file__), "rb").read()
    cop_index = open(os.path.join(cop_dir, "tiles.json"), "rb").read()
    rec = dict(
        what="Copernicus GLO-30 (surface, EGM2008) minus EA LIDAR Composite DTM 1 m (bare earth, ODN)",
        device=device, spacing_m=sp, ea_block_m=2 * half + 1,
        nodes=int(rows.size * cols.size), node_rows=int(rows.size), node_cols=int(cols.size),
        ea_site=ea_man["site"], cop_site=cop_man["site"],
        tolerances=dict(pair_m=TOL_PAIR, stored_m=TOL_Q, block_mean_m=TOL_MEAN),
        photons=photons, photons_total=int(sum(photons.values())),
        tiles_max_stored_err_m=tiles_max,
        difference_block_mean=s_gpu, difference_point_sample=s_point,
        witness=dict(device="cpu (numpy)", nodes=int(w_mean.size),
                     formula="per-node slice mean; corner-weight bilinear"),
        reading=("median d estimates the EGM2008 - ODN offset plus bare-ground bias; the upper "
                 "tail is vegetation and structures the surface model sees and the DTM removes"),
        sources=dict(cop_tiles_json_sha256=hashlib.sha256(cop_index).hexdigest(),
                     cop_tiles=m["tiles"], script_sha256=hashlib.sha256(script).hexdigest()),
        seconds=round(time.perf_counter() - t0, 3))
    path = os.path.join(cop_dir, RECEIPT)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    body = open(path, "rb").read()
    cop_man["receipt"] = dict(file=RECEIPT, sha256=hashlib.sha256(body).hexdigest())
    with open(os.path.join(cop_dir, "tiles.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(cop_man, f, indent=1, ensure_ascii=False)
    p = s_gpu["percentiles_m"]
    log(f"{device}: {rec['nodes']} nodes; photons {photons}; d = Copernicus - EA block mean: "
        f"median {p['50']:+.2f} m, mean {s_gpu['mean_m']:+.2f}, std {s_gpu['std_m']:.2f}, "
        f"P5 {p['5']:+.2f}, P95 {p['95']:+.2f}, |d| LE90 {s_gpu['abs_le90_m']:.2f} m, "
        f">5 m {100 * s_gpu['share_over_5m']:.1f}%; receipt {path}")
    return rec


def _host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


def _no_network(url, start, end):
    raise RuntimeError(f"pair runs from the cache only; {url} [{start},{end}) is not cached")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ea", required=True, help="EA 1 m tile folder (tiles.json)")
    ap.add_argument("--cop", required=True, help="Copernicus tile folder (tiles.json)")
    ap.add_argument("--cpu", action="store_true", help="run both channels in NumPy")
    a = ap.parse_args(argv)
    rec = run(a.ea, a.cop, use_gpu=not a.cpu)
    sys.exit(1 if rec["photons_total"] else 0)


if __name__ == "__main__":
    main()
