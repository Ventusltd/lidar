"""Height tile pair: do the .ght tiles say what the float64 source said?

Two channels a case, photons back (the electron/positron pattern of worlds-/src/annihilate.py).
The channels are not copies; where they disagree beyond tolerance a photon is COUNTED, never
suppressed, and the worst cases are kept with their places.

  electron   every quantised sample decoded, h_q = (base_cm + q) / 100, against the float64
             source at the same place: |h_q - h_src| <= 0.005 m (half a centimetre, the
             rounding the format promises).
  positron   heightAt at millions of seeded points computed two ways in float64 from the tile:
             two lerps (x then y) and the four-corner polynomial a + bx + cy + dxy. They must
             agree within 1e-9 m. The same point interpolated in the source grid must agree
             with the tile within 0.005 m.
  edges      tiles are 257 samples over 256 intervals, so neighbours share a row or column.
             Shared samples must be identical, integer centimetre for integer centimetre.
  header     min_q, max_q and nodata_count must match the body.

A float slack of 1e-9 m is added to the 0.005 m bound, because (base+q)/100 and the source are
themselves binary floats; the slack is recorded in the receipt.

Seeded: points come from NumPy's default_rng(seed) on the host, so any run can be repeated.
Bounded: point batches stop at --points or --seconds, whichever comes first (thermal safety).
Witness: 4,096 of the points are recomputed on the CPU in NumPy by a third formula (the corner
weights (1-x)(1-y), x(1-y), (1-x)y, xy), an independent code path, and compared within 1e-9 m.

The receipt (pair_receipt.json) lands next to tiles.json with the seed, counts, worst cases,
16 check points (x, y, h) the browser can re-test, the script sha256 and the device name. If
tiles.json exists, the receipt's sha256 is written into it under "receipt".

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/pair_gpu.py --tiles DIR [--source NPY]
    ... --tiles DIR --synth        write a synthetic hilly source and its tiles first (dev only)
"""
import argparse, glob, hashlib, json, math, os, struct, sys, time
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no card, no CuPy: the same arithmetic runs in NumPy and says so
    cp = None

SEED = 20260926
N = 257                      # samples a side; 256 intervals
NODATA = 0xFFFF
TOL_Q = 0.005                # metres: half a centimetre
TOL_PAIR = 1e-9              # metres: two float64 formulas of one bilinear surface
SLACK = 1e-9                 # float slack on the quantisation bound
WITNESS = 4096
CHECKS = 16
KEEP = 8                     # worst cases kept a check
MAGIC = b"GGH1"
RECEIPT = "pair_receipt.json"


# ---------------------------------------------------------------- format
def read_ght(path):
    raw = open(path, "rb").read()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not GGH1")
    version, samples, spacing_mm, flags = struct.unpack_from("<4H", raw, 4)
    oe, on, base = struct.unpack_from("<3i", raw, 12)
    min_q, max_q = struct.unpack_from("<2H", raw, 24)
    nodata, = struct.unpack_from("<I", raw, 28)
    body = samples * samples * 2
    q = np.frombuffer(raw, "<u2", count=samples * samples, offset=len(raw) - body).reshape(samples, samples)
    return dict(path=path, version=version, samples=samples, spacing_mm=spacing_mm, flags=flags,
                oe=oe, on=on, base=base, min_q=min_q, max_q=max_q, nodata=nodata,
                header_bytes=len(raw) - body, q=q)


def write_ght(path, h, oe, on, spacing_mm):
    """Dev writer (channel E owns the real one): 32-byte header, rows south to north."""
    cm = np.round(h * 100.0)
    ok = np.isfinite(cm)
    base = int(cm[ok].min()) if ok.any() else 0
    q = np.where(ok, cm - base, NODATA).astype("<u2")
    good = q[ok]
    hdr = MAGIC + struct.pack("<4H3i2HI", 1, N, spacing_mm, 0, oe, on, base,
                              int(good.min()) if good.size else 0, int(good.max()) if good.size else 0,
                              int((~ok).sum()))
    with open(path, "wb") as f:
        f.write(hdr + q.tobytes())


def synth(out, tiles_e=2, tiles_n=2, spacing_m=1.0, oe=400000, on=300000, seed=SEED):
    """A hilly float64 grid and its tiles, to develop against before channel E lands."""
    rng = np.random.default_rng(seed)
    ne, nn = tiles_e * (N - 1) + 1, tiles_n * (N - 1) + 1
    x, y = np.meshgrid(np.arange(ne) * spacing_m, np.arange(nn) * spacing_m)
    z = 80.0 + 0.01 * x - 0.004 * y
    for _ in range(6):
        k, a, p = rng.uniform(0.005, 0.08, 2), rng.uniform(0.5, 6.0), rng.uniform(0, 6.3)
        z += a * np.sin(k[0] * x + p) * np.cos(k[1] * y - p)
    z += rng.normal(0, 0.03, z.shape)        # sub-decimetre roughness
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "source.npy"), z)
    json.dump({"origin_e_m": oe, "origin_n_m": on, "spacing_m": spacing_m, "rows": "south_to_north",
               "synthetic": True, "seed": seed}, open(os.path.join(out, "source.json"), "w"), indent=1)
    tiles, step = [], N - 1
    for tn in range(tiles_n):
        for te in range(tiles_e):
            e0, n0 = oe + int(te * step * spacing_m), on + int(tn * step * spacing_m)
            name = f"t_{e0}_{n0}.ght"
            write_ght(os.path.join(out, name), z[tn * step:tn * step + N, te * step:te * step + N],
                      e0, n0, int(round(spacing_m * 1000)))
            tiles.append({"file": name, "origin_e_m": e0, "origin_n_m": n0})
    json.dump({"format": "GGH1", "samples": N, "tiles": tiles}, open(os.path.join(out, "tiles.json"), "w"), indent=1)
    return z


def load_source(npy):
    grid = np.load(npy).astype(np.float64)
    meta_path = os.path.splitext(npy)[0] + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pick = lambda *ks: next((meta[k] for k in ks if k in meta), None)
    oe = pick("origin_e_m", "origin_e", "sw_e", "x0", "west")
    on = pick("origin_n_m", "origin_n", "sw_n", "y0", "south")
    sp = pick("spacing_m", "spacing", "cell_m", "res_m")
    if sp is None and "spacing_mm" in meta:
        sp = meta["spacing_mm"] / 1000.0
    if oe is None or on is None or sp is None:
        raise ValueError(f"{meta_path}: need SW origin (origin_e_m, origin_n_m) and spacing_m")
    if str(meta.get("rows", "south_to_north")).startswith("north"):
        grid = grid[::-1]
    return grid, float(oe), float(on), float(sp), meta


# ---------------------------------------------------------------- arithmetic
def bilinear(xp, g, gx, gy):
    """Source bilinear at fractional indices; NaN outside. Exact at nodes (no NaN bleed)."""
    ny, nx = g.shape
    inside = (gx >= 0) & (gy >= 0) & (gx <= nx - 1) & (gy <= ny - 1)
    j = xp.clip(xp.floor(gx), 0, nx - 2).astype(xp.int64)
    i = xp.clip(xp.floor(gy), 0, ny - 2).astype(xp.int64)
    fx, fy = xp.clip(gx - j, 0, 1), xp.clip(gy - i, 0, 1)
    h00, h10, h01, h11 = g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1]
    lo = xp.where(fx == 0, h00, xp.where(fx == 1, h10, h00 + (h10 - h00) * fx))
    hi = xp.where(fx == 0, h01, xp.where(fx == 1, h11, h01 + (h11 - h01) * fx))
    h = xp.where(fy == 0, lo, xp.where(fy == 1, hi, lo + (hi - lo) * fy))
    return xp.where(inside, h, xp.nan)


def corners(H, t, u, v, xp):
    j = xp.clip(xp.floor(u), 0, N - 2).astype(xp.int64)
    i = xp.clip(xp.floor(v), 0, N - 2).astype(xp.int64)
    return H[t, i, j], H[t, i, j + 1], H[t, i + 1, j], H[t, i + 1, j + 1], u - j, v - i


def two_lerps(h00, h10, h01, h11, fx, fy):
    a = h00 + (h10 - h00) * fx
    b = h01 + (h11 - h01) * fx
    return a + (b - a) * fy


def polynomial(h00, h10, h01, h11, fx, fy):
    return h00 + (h10 - h00) * fx + (h01 - h00) * fy + (h11 - h10 - h01 + h00) * fx * fy


def weights(h00, h10, h01, h11, fx, fy):
    """The witness's formula: a third form, run only on the CPU."""
    return h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy


def worst(xp, err, info, k=KEEP):
    """Top-k errors of a batch as host dicts; info maps name -> array of the same length."""
    e = xp.where(xp.isnan(err), -1.0, err)
    idx = xp.argsort(e)[-k:][::-1]
    idx = idx[e[idx] >= 0]
    host = {n: to_host(a[idx]) for n, a in info.items()}
    ev = to_host(err[idx])
    return [dict({"err_m": float(ev[r])}, **{n: float(host[n][r]) for n in host}) for r in range(len(ev))]


def merge(kept, new):
    return sorted(kept + new, key=lambda d: -d["err_m"])[:KEEP]


def to_host(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)


# ---------------------------------------------------------------- the pair
def run(tiles_dir, source_npy, seed=SEED, points=1 << 22, seconds=20.0, batch=1 << 20, use_gpu=True):
    t_start = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    files = sorted(glob.glob(os.path.join(tiles_dir, "**", "*.ght"), recursive=True))
    if not files:
        raise SystemExit(f"no .ght tiles under {tiles_dir} (use --synth to develop against a synthetic grid)")
    tiles = [read_ght(f) for f in files]
    T = len(tiles)
    grid, soe, son, ssp, smeta = load_source(source_npy)

    # header photons: the header must describe its own body
    header_ph = []
    for tl in tiles:
        q = tl["q"]; good = q[q != NODATA]
        real = (int(good.min()) if good.size else 0, int(good.max()) if good.size else 0, int((q == NODATA).sum()))
        if tl["samples"] != N or real != (tl["min_q"], tl["max_q"], tl["nodata"]):
            header_ph.append({"file": os.path.relpath(tl["path"], tiles_dir), "header": [tl["min_q"], tl["max_q"], tl["nodata"]],
                              "body": list(real), "samples": tl["samples"]})

    Q = np.stack([t["q"] for t in tiles]).astype(np.int64)
    base = np.array([t["base"] for t in tiles], np.int64)[:, None, None]
    CM = np.where(Q == NODATA, np.iinfo(np.int64).min, base + Q)          # integer centimetres
    Hh = np.where(Q == NODATA, np.nan, (base + Q) / 100.0)
    oe = np.array([t["oe"] for t in tiles], np.float64); on = np.array([t["on"] for t in tiles], np.float64)
    sp = np.array([t["spacing_mm"] for t in tiles], np.float64) / 1000.0

    # edges: shared rows and columns must be the same integer centimetres
    where = {(t["oe"], t["on"]): k for k, t in enumerate(tiles)}
    pairs = edge_samples = edge_ph = 0; edge_worst = []
    for k, t in enumerate(tiles):
        ext = int(round((N - 1) * t["spacing_mm"] / 1000.0))
        for east in (True, False):
            m = where.get((t["oe"] + ext, t["on"]) if east else (t["oe"], t["on"] + ext))
            if m is None:
                continue
            a, b = (CM[k][:, -1], CM[m][:, 0]) if east else (CM[k][-1, :], CM[m][0, :])
            bad = np.nonzero(a != b)[0]
            pairs += 1; edge_samples += N; edge_ph += int(bad.size)
            for s in bad[:KEEP]:
                edge_worst.append({"tile": os.path.relpath(t["path"], tiles_dir), "side": "east" if east else "north",
                                   "sample": int(s), "a_cm": int(a[s]), "b_cm": int(b[s])})

    G = xp.asarray(grid); H = xp.asarray(Hh)
    OE, ON, SP = xp.asarray(oe), xp.asarray(on), xp.asarray(sp)

    # electron: every sample against the source at the same place
    jj, ii = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64))
    JJ, II = xp.asarray(jj), xp.asarray(ii)
    el = dict(samples=0, photons=0, nodata_mismatch=0, nodata_both=0, outside_source=0, max_err_m=0.0)
    el_worst = []
    for k in range(T):
        e = OE[k] + JJ * SP[k]; n = ON[k] + II * SP[k]
        hs = bilinear(xp, G, (e - soe) / ssp, (n - son) / ssp)
        hq = H[k]
        tq, ts = xp.isnan(hq), xp.isnan(hs)
        el["samples"] += N * N
        el["nodata_both"] += int((tq & ts).sum())
        el["nodata_mismatch"] += int((tq ^ ts).sum())
        err = xp.abs(hq - hs)
        el["photons"] += int((err > TOL_Q + SLACK).sum())
        if bool((~xp.isnan(err)).any()):
            el["max_err_m"] = max(el["max_err_m"], float(xp.nanmax(err)))
        el_worst = merge(el_worst, worst(xp, err.ravel(), {"x": e.ravel(), "y": n.ravel(), "h_q": hq.ravel(), "h_src": hs.ravel()}))
    el["photons"] += el["nodata_mismatch"]

    # positron: seeded points, two tile formulas, and tile against source
    rng = np.random.default_rng(seed)
    po = dict(points=0, batches=0, pair_photons=0, pair_max_m=0.0, src_photons=0, src_max_m=0.0,
              nodata_cells=0, outside_source=0, stopped_by="points")
    pair_worst, src_worst, first = [], [], None
    while po["points"] < points:
        if po["batches"] and time.perf_counter() - t_start > seconds:
            po["stopped_by"] = "seconds"; break
        m = min(batch, points - po["points"])
        t_h = rng.integers(0, T, m); u_h = rng.uniform(0, N - 1, m); v_h = rng.uniform(0, N - 1, m)
        if first is None:
            first = (t_h[:WITNESS].copy(), u_h[:WITNESS].copy(), v_h[:WITNESS].copy())
        t, u, v = xp.asarray(t_h), xp.asarray(u_h), xp.asarray(v_h)
        c = corners(H, t, u, v, xp)
        hA, hB = two_lerps(*c), polynomial(*c)
        x, y = OE[t] + u * SP[t], ON[t] + v * SP[t]
        hs = bilinear(xp, G, (x - soe) / ssp, (y - son) / ssp)
        dp, ds = xp.abs(hA - hB), xp.abs(hA - hs)
        po["nodata_cells"] += int(xp.isnan(hA).sum())
        po["outside_source"] += int((xp.isnan(hs) & ~xp.isnan(hA)).sum())
        po["pair_photons"] += int((dp > TOL_PAIR).sum()); po["src_photons"] += int((ds > TOL_Q + SLACK).sum())
        if bool((~xp.isnan(dp)).any()):
            po["pair_max_m"] = max(po["pair_max_m"], float(xp.nanmax(dp)))
        if bool((~xp.isnan(ds)).any()):
            po["src_max_m"] = max(po["src_max_m"], float(xp.nanmax(ds)))
        info = {"x": x, "y": y, "h_lerp": hA, "h_poly": hB, "h_src": hs}
        pair_worst = merge(pair_worst, worst(xp, dp, info)); src_worst = merge(src_worst, worst(xp, ds, info))
        po["points"] += m; po["batches"] += 1
        if xp is not np:
            cp.cuda.Stream.null.synchronize()

    # CPU witness: the first 4,096 points, a third formula, NumPy only
    wt, wu, wv = first
    wc = corners(Hh, wt, wu, wv, np)
    w_h = weights(*wc)
    g_c = corners(H, xp.asarray(wt), xp.asarray(wu), xp.asarray(wv), xp)
    g_h = to_host(two_lerps(*g_c))
    wd = np.abs(w_h - g_h)
    wit = {"points": int(wt.size), "photons": int((wd > TOL_PAIR).sum()),
           "max_diff_m": float(np.nanmax(wd)) if np.isfinite(wd).any() else None,
           "nan_mismatch": int((np.isnan(w_h) ^ np.isnan(g_h)).sum()), "formula": "corner weights"}
    wit["photons"] += wit["nan_mismatch"]

    checks = check_points(tiles, Hh, oe, on, sp, np.random.default_rng(seed + 1))
    wall = time.perf_counter() - t_start
    receipt = {
        "kind": "lidar height tile pair", "seed": seed, "device": device,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "source": {"file": os.path.basename(source_npy), "shape": list(grid.shape), "origin_e_m": soe, "origin_n_m": son,
                   "spacing_m": ssp, "synthetic": bool(smeta.get("synthetic", False)),
                   "sha256": hashlib.sha256(open(source_npy, "rb").read()).hexdigest()},
        "tiles": T, "header_bytes": sorted({t["header_bytes"] for t in tiles}),
        "tolerance_m": {"quantised": TOL_Q, "float_slack": SLACK, "two_formulas": TOL_PAIR},
        "header": {"photons": len(header_ph), "cases": header_ph[:KEEP]},
        "edges": {"pairs": pairs, "samples": edge_samples, "photons": edge_ph, "cases": edge_worst[:KEEP]},
        "electron": dict(el, worst=el_worst),
        "positron": dict(po, pair_worst=pair_worst, src_worst=src_worst),
        "witness": wit,
        "photons_total": len(header_ph) + edge_ph + el["photons"] + po["pair_photons"] + po["src_photons"] + wit["photons"],
        "check_points": checks,
        "bounds": {"points": points, "seconds": seconds, "batch": batch}, "wall_s": round(wall, 3),
    }
    return receipt


def check_points(tiles, Hh, oe, on, sp, rng):
    """16 points at millimetre coordinates with the tile height there, for the browser to re-test."""
    out, tries = [], 0
    while len(out) < CHECKS and tries < 10000:
        tries += 1
        k = int(rng.integers(0, len(tiles)))
        x = round(float(oe[k] + rng.uniform(0, N - 1) * sp[k]), 3)
        y = round(float(on[k] + rng.uniform(0, N - 1) * sp[k]), 3)
        u, v = np.array([(x - oe[k]) / sp[k]]), np.array([(y - on[k]) / sp[k]])
        h = float(two_lerps(*corners(Hh, np.array([k]), u, v, np))[0])
        if math.isfinite(h):
            out.append({"x": x, "y": y, "h": round(h, 9), "tile": os.path.basename(tiles[k]["path"])})
    return out


def write_receipt(tiles_dir, receipt):
    body = json.dumps(receipt, indent=1, sort_keys=True).encode()
    path = os.path.join(tiles_dir, RECEIPT)
    with open(path, "wb") as f:
        f.write(body)
    sha = hashlib.sha256(body).hexdigest()
    tj = os.path.join(tiles_dir, "tiles.json")
    if os.path.exists(tj):
        meta = json.load(open(tj))
        meta["receipt"] = sha
        with open(tj, "w") as f:
            json.dump(meta, f, indent=1)
    return path, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--source", help="float64 source .npy (default TILES/source.npy); SW origin in the .json beside it")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--points", type=int, default=1 << 22)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cpu", action="store_true", help="run the pair in NumPy")
    ap.add_argument("--synth", action="store_true", help="write a synthetic hilly source and tiles into TILES first")
    a = ap.parse_args(argv)
    if a.synth:
        synth(a.tiles)
    src = a.source or os.path.join(a.tiles, "source.npy")
    r = run(a.tiles, src, seed=a.seed, points=a.points, seconds=a.seconds, use_gpu=not a.cpu)
    path, sha = write_receipt(a.tiles, r)
    el, po, w = r["electron"], r["positron"], r["witness"]
    print(f"{r['tiles']} tiles on {r['device']} in {r['wall_s']} s")
    print(f"electron  {el['samples']:,} samples, photons {el['photons']:,}, max {el['max_err_m']:.6f} m")
    print(f"positron  {po['points']:,} points, two-formula photons {po['pair_photons']:,} (max {po['pair_max_m']:.2e} m), "
          f"tile-vs-source photons {po['src_photons']:,} (max {po['src_max_m']:.6f} m), stopped by {po['stopped_by']}")
    print(f"edges     {r['edges']['pairs']} shared edges, photons {r['edges']['photons']}; header photons {r['header']['photons']}")
    print(f"witness   {w['points']} points, photons {w['photons']}, max diff {w['max_diff_m']}")
    print(f"receipt   {path} sha256 {sha}")
    return 0 if r["photons_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
