# SPDX-License-Identifier: Apache-2.0
"""Piles sweep: how far do solar table piles stand out of real ground, counted two ways?

Two channels a case, photons back (the electron/positron pattern of pair_gpu.py and cable_sweep.py). The
channels differ in where the ground comes from; the pile layout and the top-of-pile fit are shared and match
graphics-engines web/world/piles.mjs (fitTopLine) to 1e-9, checked by a fixture this script writes.

A row is one straight table on a straight placement line: single-axis tracker rows run north-south (0-3 deg
jitter), fixed-tilt rows east-west, half each. Piles are pileSpacing apart, centred on the table. The top of
the piles is one straight line: the slope that makes the spread of (ground - b s) smallest within the
along-row limit (least value at a pairwise slope or a limit, ties to the least-squares slope), then the level
as near the nominal reveal as the window [revealMin, revealMax] allows, or the window's middle if it is empty.

  electron   direct sampling: the float64 source array (source.npy) indexed as one grid, bilinear.
  positron   the published tiles: tiles.json, every .ght tile decoded (heights in whole centimetres), each pile
             found in its tile by key, bilinear inside that tile only.

The .ght heights are rounded to 1 cm, so the two ground values may differ by up to 0.005 m: a ground photon is
counted above 0.0051 m. A reveal photon is counted where the two reveals of one pile differ by more than
0.02 m (ASSUMED threshold: quantisation 0.005 m at most, times the fit's leverage). Photons are COUNTED, never
suppressed; the worst rows are kept with their places. Witness: the first rows recomputed in NumPy on the CPU.

Reveal range, slope limits, embedment and grading width are ASSUMED typical figures (see LIMITS, REVEAL); the
sources for the slope figures are listed in piles.mjs.

    python src/piles_sweep.py --tiles DIR
      [--rows 20000] [--seconds 60] [--seed N] [--cpu] [--fixture PATH]
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cable_sweep import cp, ground_from, to_host  # noqa: E402
from cut_tiles import decode_tile  # noqa: E402

SEED = 20260926
TOL_G = 0.0051        # ground photon (m): half a centimetre plus rounding
TOL_R = 0.02          # reveal photon (m): ASSUMED
WIT_TOL = 1e-9
KEEP = 8
RECEIPT = "piles_sweep_receipt.json"
SYSTEMS = ("tracker", "fixed")
# ASSUMED, as piles.mjs DEFAULTS and LIMITS
REVEAL = {"tracker": dict(min=1.0, nominal=1.4, max=1.8), "fixed": dict(min=0.6, nominal=1.0, max=1.6)}
LIMITS = {"tracker": dict(nsMax=0.10, ewMax=0.15, along="ns"), "fixed": dict(nsMax=0.20, ewMax=0.15, along="ew")}
ALONG = {k: v["nsMax"] if v["along"] == "ns" else v["ewMax"] for k, v in LIMITS.items()}
TABLES = {"tracker": ((45, 60, 90), (6, 7, 8)), "fixed": ((20, 25, 30), (4, 5, 6))}   # lengths, spacings (m)
EMBED_MIN, GRADING_W = 1.5, 2.0
K = 16                # most piles a row (90 / 6 + 1)
BINS = np.round(np.arange(-1.0, 5.0001, 0.05), 6)


class TileGround:
    """Bilinear ground from decoded .ght tiles, each pile read inside its own tile."""
    def __init__(self, tiles_dir, oe, on, xp):
        man = json.load(open(os.path.join(tiles_dir, "tiles.json"), encoding="utf-8"))
        self.xp, self.t = xp, int(man["tile_m"] / man["spacing_m"])
        self.sp = float(man["spacing_m"])
        oe, on = int(round(oe)), int(round(on))
        cols = sorted({(x["e0"] - oe) // man["tile_m"] for x in man["tiles"]})
        rows = sorted({(x["n0"] - on) // man["tile_m"] for x in man["tiles"]})
        self.ntx, self.nty = len(cols), len(rows)
        stack, idx = [], np.full((self.nty, self.ntx), -1, np.int64)
        for x in man["tiles"]:
            blob = open(os.path.join(tiles_dir, x["file"]), "rb").read()
            if hashlib.sha256(blob).hexdigest() != x["sha256"]:
                raise ValueError(f"sha256 mismatch for {x['file']}")
            head, h = decode_tile(blob)
            ix, iy = (head["origin_e_m"] - oe) // man["tile_m"], (head["origin_n_m"] - on) // man["tile_m"]
            idx[iy, ix] = len(stack); stack.append(h)
        if (idx < 0).any():
            raise ValueError("tiles.json: missing tiles in the site rectangle")
        self.stack, self.idx = xp.asarray(np.stack(stack)), xp.asarray(idx)
        self.nodata = int(sum(x["nodata"] for x in man["tiles"]))

    def h(self, x, y):
        xp, t = self.xp, self.t
        gx, gy = x / self.sp, y / self.sp
        ix = xp.clip(xp.floor(gx / t), 0, self.ntx - 1).astype(xp.int64)
        iy = xp.clip(xp.floor(gy / t), 0, self.nty - 1).astype(xp.int64)
        lx, ly = gx - ix * t, gy - iy * t
        j = xp.clip(xp.floor(lx), 0, t - 1).astype(xp.int64)
        i = xp.clip(xp.floor(ly), 0, t - 1).astype(xp.int64)
        fx, fy, k, g = lx - j, ly - i, self.idx[iy, ix], self.stack
        a = g[k, i, j] + (g[k, i, j + 1] - g[k, i, j]) * fx
        b = g[k, i + 1, j] + (g[k, i + 1, j + 1] - g[k, i + 1, j]) * fx
        return a + (b - a) * fy


def draw(rng, n, W, H, margin=3.0):
    """n seeded rows wholly inside the site; returns host arrays."""
    sysid = (rng.random(n) < 0.5).astype(np.int64)          # 0 tracker, 1 fixed
    pick = rng.integers(0, 3, (n, 2))
    L = np.where(sysid == 0, np.take(TABLES["tracker"][0], pick[:, 0]), np.take(TABLES["fixed"][0], pick[:, 0])).astype(float)
    sp = np.where(sysid == 0, np.take(TABLES["tracker"][1], pick[:, 1]), np.take(TABLES["fixed"][1], pick[:, 1])).astype(float)
    jit = np.radians(rng.uniform(-3, 3, n))
    head = np.where(sysid == 0, np.pi / 2, 0.0) + jit + np.where(rng.random(n) < 0.5, 0, np.pi)
    ux, uy = np.cos(head), np.sin(head)
    rx, ry = np.abs(ux) * L / 2 + margin, np.abs(uy) * L / 2 + margin
    cx, cy = rng.uniform(rx, W - rx), rng.uniform(ry, H - ry)
    x0, y0 = cx - ux * L / 2, cy - uy * L / 2
    npile = np.floor(L / sp + 1e-9).astype(np.int64) + 1
    over = (L - (npile - 1) * sp) / 2
    k = np.arange(K)[None]
    mask = k < npile[:, None]
    s = np.where(mask, over[:, None] + k * sp[:, None], 0.0)
    rmin = np.where(sysid == 0, REVEAL["tracker"]["min"], REVEAL["fixed"]["min"])
    rnom = np.where(sysid == 0, REVEAL["tracker"]["nominal"], REVEAL["fixed"]["nominal"])
    rmax = np.where(sysid == 0, REVEAL["tracker"]["max"], REVEAL["fixed"]["max"])
    lim = np.where(sysid == 0, ALONG["tracker"], ALONG["fixed"])
    return dict(sys=sysid, L=L, sp=sp, x0=x0, y0=y0, ux=ux, uy=uy, s=s, mask=mask, rmin=rmin, rnom=rnom, rmax=rmax, lim=lim)


PI, PJ = np.triu_indices(K, 1)


def fit(xp, s, g, m, rmin, rnom, rmax, lim):
    """Vectorised piles.mjs fitTopLine over rows: returns a, b, spread, feasible."""
    n_ = m.sum(1)
    ms = (s * m).sum(1) / n_
    mg = xp.where(m, g, 0).sum(1) / n_
    ds, dg = xp.where(m, s - ms[:, None], 0), xp.where(m, g - mg[:, None], 0)
    sxx = (ds * ds).sum(1)
    bls = xp.where(sxx > 0, (ds * dg).sum(1) / xp.where(sxx > 0, sxx, 1), 0.0)
    clip = lambda b: xp.clip(b, -lim[:, None], lim[:, None])
    pi, pj = xp.asarray(PI), xp.asarray(PJ)
    ok = m[:, pi] & m[:, pj]
    dsij = s[:, pj] - s[:, pi]
    bij = xp.where(ok, (xp.where(ok, g[:, pj], 0) - xp.where(ok, g[:, pi], 0)) / xp.where(ok, dsij, 1), bls[:, None])
    cand = xp.concatenate([clip(bls[:, None]), -lim[:, None], lim[:, None], clip(bij)], 1)
    d = xp.where(m[:, None, :], g[:, None, :] - cand[:, :, None] * s[:, None, :], xp.nan)
    hi, lo = xp.nanmax(d, 2), xp.nanmin(d, 2)
    w = hi - lo
    wmin = w.min(1)
    score = xp.where(w <= wmin[:, None] + 1e-9, xp.abs(cand - bls[:, None]), xp.inf)
    c = xp.argmin(score, 1)
    r = xp.arange(len(c))
    b, hi, lo, w = cand[r, c], hi[r, c], lo[r, c], w[r, c]
    mean = xp.where(m, g - b[:, None] * s, 0).sum(1) / n_
    wl, wh = hi + rmin, lo + rmax
    feas = wl <= wh + 1e-9
    a = xp.where(feas, xp.clip(mean + rnom, wl, wh), (wl + wh) / 2)
    return a, b, w, feas


def evaluate(D, G, xp):
    """One channel: ground under every pile, the fit, reveals, cut and fill."""
    A = {k: xp.asarray(v) for k, v in D.items()}
    x = A["x0"][:, None] + A["ux"][:, None] * A["s"]
    y = A["y0"][:, None] + A["uy"][:, None] * A["s"]
    g = xp.where(A["mask"], G.h(x, y), xp.nan)
    a, b, w, feas = fit(xp, A["s"], g, A["mask"], A["rmin"], A["rnom"], A["rmax"], A["lim"])
    rev = xp.where(A["mask"], a[:, None] + b[:, None] * A["s"] - g, xp.nan)
    fill = xp.where(A["mask"], xp.maximum(rev - A["rmax"][:, None], 0), 0)
    cut = xp.where(A["mask"], xp.maximum(A["rmin"][:, None] - rev, 0), 0)
    return {k: to_host(v) for k, v in dict(g=g, a=a, b=b, spread=w, feasible=feas, reveal=rev, fill=fill, cut=cut).items()}


def slopes(D, G, xp, h=1.0):
    """Largest |north-south| and |east-west| ground gradient at a row's piles (central differences)."""
    A = {k: xp.asarray(D[k]) for k in ("x0", "y0", "ux", "uy", "s", "mask")}
    x = A["x0"][:, None] + A["ux"][:, None] * A["s"]
    y = A["y0"][:, None] + A["uy"][:, None] * A["s"]
    gx = (G.h(x + h, y) - G.h(x - h, y)) / (2 * h)
    gy = (G.h(x, y + h) - G.h(x, y - h)) / (2 * h)
    return to_host(xp.where(A["mask"], xp.abs(gy), 0).max(1)), to_host(xp.where(A["mask"], xp.abs(gx), 0).max(1))


def pair(D, Ge, Gp):
    E, P = evaluate(D, Ge, Ge.xp), evaluate(D, Gp, Gp.xp)
    m = D["mask"]
    dg = np.where(m, np.abs(E["g"] - P["g"]), 0).max(1)
    dr = np.where(m, np.abs(E["reveal"] - P["reveal"]), 0).max(1)
    return E, P, dg, dr


def pct(v):
    return [round(float(x), 6) for x in np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])] if len(v) else []


def out_of_range(rev, rmin, rmax, m):
    return ((rev > rmax[:, None] + 1e-9) | (rev < rmin[:, None] - 1e-9)) & m


def summarise(C, si, name):
    r = C["sys"] == si
    m = C["mask"][r]
    ve, vp = C["re"][r][m], C["rp"][r][m]
    he, _ = np.histogram(np.clip(ve, BINS[0], BINS[-1]), BINS)
    hp, _ = np.histogram(np.clip(vp, BINS[0], BINS[-1]), BINS)
    oe_ = out_of_range(C["re"][r], C["rmin"][r], C["rmax"][r], m)
    op_ = out_of_range(C["rp"][r], C["rmin"][r], C["rmax"][r], m)
    row_m = float(C["L"][r].sum())
    per100 = lambda k: round(float((C[k][r] * C["sp"][r][:, None]).sum() * GRADING_W) / row_m * 100, 4)
    return dict(
        rows=int(r.sum()), piles=int(m.sum()), reveal_limits_m=REVEAL[name], along_limit=ALONG[name],
        slope_limits=LIMITS[name],
        reveal_pct_0_1_5_25_50_75_95_99_100={"direct": pct(ve), "tiles": pct(vp)},
        histogram={"bin_edges_m": [float(BINS[0]), float(BINS[-1]), 0.05], "direct": he.tolist(), "tiles": hp.tolist()},
        ks_max_cdf_diff=float(np.abs(np.cumsum(he) - np.cumsum(hp)).max() / max(len(ve), 1)),
        piles_out_of_range={"direct": int(oe_.sum()), "tiles": int(op_.sum()), "classified_differently": int((oe_ != op_).sum())},
        rows_needing_grading={"direct": int((~C["fe_ok"][r]).sum()), "tiles": int((~C["fp_ok"][r]).sum())},
        grading_m3_per_100m_row={"direct_cut": per100("ce"), "direct_fill": per100("fe"),
                                 "tiles_cut": per100("cpt"), "tiles_fill": per100("fpt")},
        rows_at_along_limit=int((np.abs(C["b"][r]) >= C["lim"][r] - 1e-12).sum()),
        rows_ns_slope_over=int((C["ns"][r] > LIMITS[name]["nsMax"]).sum()),
        rows_ew_slope_over=int((C["ew"][r] > LIMITS[name]["ewMax"]).sum()))


def sweep(tiles_dir, seed=SEED, rows=20000, seconds=60.0, batch=4000, use_gpu=True, witness=64):
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    Ge = ground_from(tiles_dir, xp)
    Gp = TileGround(tiles_dir, Ge.oe, Ge.on, xp)
    W, H = Ge.sp * (Ge.nx - 1), Ge.sp * (Ge.ny - 1)
    rng = np.random.default_rng(seed)
    keep, worst, first, stopped, done = {}, [], None, "rows", 0
    while done < rows:
        if done and time.perf_counter() - t0 > seconds:
            stopped = "seconds"; break
        D = draw(rng, min(batch, rows - done), W, H)
        if first is None:
            first = {k: v[:witness].copy() for k, v in D.items()}
        E, P, dg, dr = pair(D, Ge, Gp)
        ns, ew = slopes(D, Ge, xp)
        got = dict(sys=D["sys"], L=D["L"], sp=D["sp"], mask=D["mask"], rmin=D["rmin"], rmax=D["rmax"], lim=D["lim"],
                   re=E["reveal"], rp=P["reveal"], dg=dg, dr=dr, ce=E["cut"], fe=E["fill"], cpt=P["cut"], fpt=P["fill"],
                   fe_ok=E["feasible"], fp_ok=P["feasible"], b=E["b"], ns=ns, ew=ew)
        for k, v in got.items():
            keep.setdefault(k, []).append(v)
        for i in np.argsort(-dr)[:KEEP]:
            worst.append(dict(start_e=round(float(D["x0"][i] + Ge.oe), 3), start_n=round(float(D["y0"][i] + Ge.on), 3),
                              system=SYSTEMS[D["sys"][i]], table_m=float(D["L"][i]), spacing_m=float(D["sp"][i]),
                              heading_deg=round(math.degrees(math.atan2(D["uy"][i], D["ux"][i])) % 360, 3),
                              max_ground_diff_m=round(float(dg[i]), 6), max_reveal_diff_m=round(float(dr[i]), 6),
                              slope_direct=round(float(E["b"][i]), 6), slope_tiles=round(float(P["b"][i]), 6)))
        done += len(D["L"])
        if xp is not np:
            cp.cuda.Stream.null.synchronize()
    C = {k: np.concatenate(v) for k, v in keep.items()}
    # CPU witness: the first rows, both channels, NumPy only
    Ec, Pc, _, _ = pair(first, ground_from(tiles_dir, np), TileGround(tiles_dir, Ge.oe, Ge.on, np))
    wm = first["mask"]
    nw = len(wm)
    wd = np.concatenate([np.abs(Ec["reveal"] - C["re"][:nw])[wm], np.abs(Pc["reveal"] - C["rp"][:nw])[wm]])
    wit = dict(rows=int(nw), photons=int((wd > WIT_TOL).sum()), max_abs_m=float(wd.max()), tol_m=WIT_TOL)
    ph_g, ph_r = int((C["dg"] > TOL_G).sum()), int((C["dr"] > TOL_R).sum())
    return {
        "kind": "pile reveal sweep pair", "seed": seed, "device": device,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "source": {"shape": [Ge.ny, Ge.nx], "origin_e_m": Ge.oe, "origin_n_m": Ge.on, "spacing_m": Ge.sp,
                   "tiles": int(Gp.ntx * Gp.nty), "tile_nodata": Gp.nodata},
        "method": {"electron": "direct sampling: bilinear on the float64 source array as one grid",
                   "positron": "bilinear inside each decoded .ght tile (1 cm heights), pile found in its tile by key",
                   "fit": "minimax slope within the along-row limit, ties to least squares; level nearest nominal in the window",
                   "tolerance_ground_m": TOL_G, "tolerance_reveal_m": TOL_R, "tolerance_reveal_status": "ASSUMED",
                   "rows_drawn": "half tracker (north-south, 0-3 deg jitter), half fixed (east-west); one straight table a row; "
                                 "tracker 45/60/90 m at 6/7/8 m spacing, fixed 20/25/30 m at 4/5/6 m",
                   "assumed": dict(embed_min_m=EMBED_MIN, grading_width_m=GRADING_W,
                                   note="reveal range, slope limits, embedment, grading width are ASSUMED typical figures")},
        "counts": {"rows": int(len(C["L"])), "piles": int(C["mask"].sum()), "stopped_by": stopped},
        "disagreement": {"ground_photons": ph_g, "reveal_photons": ph_r, "max_ground_diff_m": float(C["dg"].max()),
                         "max_reveal_diff_m": float(C["dr"].max()), "median_row_max_reveal_diff_m": float(np.median(C["dr"])),
                         "worst": sorted(worst, key=lambda d: -d["max_reveal_diff_m"])[:KEEP]},
        "by_system": {name: summarise(C, si, name) for si, name in enumerate(SYSTEMS)},
        "witness": wit, "photons_total": ph_g + ph_r + wit["photons"],
        "bounds": {"rows": rows, "seconds": seconds, "batch": batch}, "wall_s": round(time.perf_counter() - t0, 3),
    }, first, Ec


def write_json(path, obj):
    body = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    with open(path, "wb") as f:       # bytes: LF on every platform
        f.write(body)
    return hashlib.sha256(body).hexdigest()


def fixture(first, Ec, per_system=6):
    """Rows for piles.mjs to refit: s, ground (direct) and the reveals this script found."""
    out = []
    for si, name in enumerate(SYSTEMS):
        for i in np.nonzero(first["sys"] == si)[0][:per_system]:
            m = first["mask"][i]
            out.append(dict(id=int(i), system=name, s=[float(v) for v in first["s"][i][m]],
                            ground=[float(v) for v in Ec["g"][i][m]], reveal=[float(v) for v in Ec["reveal"][i][m]]))
    return {"kind": "piles fixture from piles_sweep.py", "reveal": REVEAL, "alongMax": ALONG, "rows": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    r, first, Ec = sweep(a.tiles, a.seed, a.rows, a.seconds, a.batch, use_gpu=not a.cpu)
    path = os.path.join(a.tiles, RECEIPT)
    sha = write_json(path, r)
    if a.fixture:
        print(f"fixture {a.fixture} sha256 {write_json(a.fixture, fixture(first, Ec))}")
    c, d = r["counts"], r["disagreement"]
    print(f"{c['rows']} rows, {c['piles']} piles on {r['device']} in {r['wall_s']} s, stopped by {c['stopped_by']}")
    print(f"photons: ground {d['ground_photons']} (> {TOL_G} m), reveal {d['reveal_photons']} (> {TOL_R} m); "
          f"max ground diff {d['max_ground_diff_m']:.4f} m, max reveal diff {d['max_reveal_diff_m']:.4f} m")
    for name, e in r["by_system"].items():
        q = e["reveal_pct_0_1_5_25_50_75_95_99_100"]
        print(f"  {name}: {e['rows']} rows, reveal p5/p50/p95 direct {q['direct'][2]}/{q['direct'][4]}/{q['direct'][6]} "
              f"tiles {q['tiles'][2]}/{q['tiles'][4]}/{q['tiles'][6]}; KS {e['ks_max_cdf_diff']:.2e}; "
              f"out of range {e['piles_out_of_range']}; grading rows {e['rows_needing_grading']}")
    print(f"witness {r['witness']['rows']} rows, photons {r['witness']['photons']} (max {r['witness']['max_abs_m']:.1e})")
    print(f"receipt {path} sha256 {sha}")
    return 0 if r["photons_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
