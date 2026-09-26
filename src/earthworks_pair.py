"""Earthworks pair: how much spoil does a trench (or a platform) make, counted two ways?

Two channels a case, photons back (the electron/positron pattern of pair_gpu.py and
worlds-/src/annihilate.py). The channels are not copies; where they disagree by more than
0.5 % a photon is COUNTED, never suppressed, and the worst cases are kept with their places.

The trench: a centreline in BNG metres (straight, then an optional circular bend of radius R
and signed angle theta, then straight), width w, depth d, flat square ends. The floor at
chainage s is the ground on the centreline there, less d; cut depth = max(ground - floor, 0).

  electron   world-aligned prisms. Every 0.1 m grid cell near the route is split into sub x sub
             points (default 4, so 0.025 m); a point belongs to the trench if it projects onto
             a piece of the centreline within w/2. Volume = sum of cut depth x point area.
  positron   cross sections every <= 0.1 m of chainage, each integrated across the width on 33
             offsets (trapezoid) with the curvature weight (1 - kappa t), then the section areas
             integrated along chainage by the trapezoid rule. No grid, no membership test.

The platform: a polygon and a target level. Electron = the same prisms with point-in-polygon;
positron = sections perpendicular to the polygon's first edge, each cut against every edge.
Cut and fill are each a pair.

Ground is the bilinear surface of the float64 source (source.npy, SW origin in source.json).
Seeded (default_rng), bounded (--trenches or --seconds, whichever first). Witness: the first
trenches and the platform recomputed in NumPy on the CPU, agreeing with the card within 1e-9.

    E:/swarm/gpu-bench/venv/Scripts/python.exe src/earthworks_pair.py --tiles DIR
      [--trenches 4000] [--seconds 20] [--seed N] [--cpu]
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair_gpu import cp, load_source, to_host  # noqa: E402

SEED = 20260926
TOL = 0.005           # relative disagreement that makes a photon
CELL, SUB = 0.1, 4    # electron grid cell (m) and sub-points a side
DS, NT = 0.1, 33      # positron section spacing (m) and offsets across
WIT_TOL = 1e-9
KEEP = 8
RECEIPT = "earthworks_receipt.json"
BUDGET = 1 << 23      # electron points per batch


class Ground:
    """Bilinear ground in local metres (x east, y north from the source's SW corner)."""
    def __init__(self, grid, oe, on, sp, xp):
        self.xp, self.g, self.oe, self.on, self.sp = xp, xp.asarray(grid), oe, on, sp
        self.ny, self.nx = grid.shape

    def h(self, x, y):
        xp, g = self.xp, self.g
        gx, gy = x / self.sp, y / self.sp
        j = xp.clip(xp.floor(gx), 0, self.nx - 2).astype(xp.int64)
        i = xp.clip(xp.floor(gy), 0, self.ny - 2).astype(xp.int64)
        fx, fy = gx - j, gy - i
        a = g[i, j] + (g[i, j + 1] - g[i, j]) * fx
        b = g[i + 1, j] + (g[i + 1, j + 1] - g[i + 1, j]) * fx
        return a + (b - a) * fy


def ground_from(tiles_dir, xp):
    grid, oe, on, sp, _ = load_source(os.path.join(tiles_dir, "source.npy"))
    return Ground(grid, oe, on, sp, xp)


# ---------------------------------------------------------------- trench geometry
def routes(e0, n0, heading, L1, theta, R, L2, w, d, oe=0.0, on=0.0):
    """Arrays of trench parameters (T,) in local metres. theta > 0 bends left."""
    f = lambda v: np.atleast_1d(np.asarray(v, np.float64))
    e0, n0, h, L1, th, R, L2, w, d = map(f, (e0, n0, heading, L1, theta, R, L2, w, d))
    sg = np.sign(th); La = R * np.abs(th) * (sg != 0)
    ax, ay = e0 - oe, n0 - on
    bx, by = ax + L1 * np.cos(h), ay + L1 * np.sin(h)
    cx, cy = bx - np.sin(h) * sg * R, by + np.cos(h) * sg * R
    h2 = h + th
    dx, dy = cx + np.sin(h2) * sg * R, cy - np.cos(h2) * sg * R
    dx, dy = np.where(sg == 0, bx, dx), np.where(sg == 0, by, dy)
    return dict(ax=ax, ay=ay, h=h, L1=L1, bx=bx, by=by, cx=cx, cy=cy, sg=sg, R=R, th=np.abs(th), La=La,
                dx=dx, dy=dy, h2=h2, L2=L2, L=L1 + La + L2, w=w, d=d)


def centre(P, k, s, xp, piece=None):
    """Centreline point, left normal and signed curvature at chainage s of trenches k.
    piece (0 straight, 1 bend, 2 straight) with s local to it pins a joint to one side."""
    g = {n: xp.asarray(P[n])[k] for n in ("ax", "ay", "h", "L1", "cx", "cy", "sg", "R", "La", "dx", "dy", "h2")}
    if piece is None:
        piece = xp.where(s <= g["L1"], 0, xp.where(s <= g["L1"] + g["La"], 1, 2))
        s = s - xp.where(piece >= 1, g["L1"], 0) - xp.where(piece == 2, g["La"], 0)
    Rs = xp.where(g["R"] > 0, g["R"], 1.0)
    hh = xp.where(piece == 0, g["h"], xp.where(piece == 1, g["h"] + g["sg"] * s / Rs, g["h2"]))
    nx, ny = -xp.sin(hh), xp.cos(hh)
    x = xp.where(piece == 0, g["ax"] + s * ny, xp.where(piece == 1, g["cx"] - nx * g["sg"] * g["R"], g["dx"] + s * ny))
    y = xp.where(piece == 0, g["ay"] - s * nx, xp.where(piece == 1, g["cy"] - ny * g["sg"] * g["R"], g["dy"] - s * nx))
    kap = xp.where(piece == 1, g["sg"] / Rs, 0.0)
    return x, y, nx, ny, kap


def positron(P, G, ks):
    """Section areas (1 - kappa t weighted) integrated along chainage, per trench in ks.
    Each piece gets its own sections, so no trapezoid straddles a change of curvature."""
    xp = G.xp
    K, S, W, PC = [], [], [], []
    for pc, key in enumerate(("L1", "La", "L2")):
        Lp = P[key][ks]; kk = ks[Lp > 0]; Lp = Lp[Lp > 0]
        ns = np.ceil(Lp / DS).astype(np.int64) + 1
        j = np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)
        step = np.repeat(Lp / (ns - 1), ns)
        K.append(np.repeat(kk, ns)); S.append(j * step); PC.append(np.full(j.size, pc))
        W.append(np.where((j == 0) | (j == np.repeat(ns - 1, ns)), step / 2, step))
    k, s, ws, pcs = map(np.concatenate, (K, S, W, PC))
    kd = xp.asarray(k)
    x, y, nx, ny, kap = centre(P, kd, xp.asarray(s), xp, xp.asarray(pcs))
    w, d = xp.asarray(P["w"])[kd], xp.asarray(P["d"])[kd]
    tau = xp.linspace(-0.5, 0.5, NT)
    wt = xp.full(NT, 1.0 / (NT - 1)); wt[0] = wt[-1] = 0.5 / (NT - 1)
    t = w[:, None] * tau[None, :]
    floor = G.h(x, y) - d
    cut = xp.maximum(G.h(x[:, None] + t * nx[:, None], y[:, None] + t * ny[:, None]) - floor[:, None], 0)
    area = (cut * (1 - kap[:, None] * t) * wt[None, :]).sum(1) * w
    idx = xp.asarray(np.searchsorted(ks, k))
    return to_host(xp.bincount(idx, weights=area * xp.asarray(ws), minlength=len(ks)))


def candidates(P, ks):
    """1 m cells near each route (sampled every 0.5 m, dilated one cell): a superset only."""
    out = []
    for k in ks:
        s = np.linspace(0, P["L"][k], int(P["L"][k] / 0.5) + 2)
        x, y, nx, ny, _ = centre(P, np.full(s.size, k), s, np)
        t = np.linspace(-P["w"][k] / 2, P["w"][k] / 2, max(2, int(P["w"][k] / 0.5) + 2))
        ix = np.floor(x[:, None] + t * nx[:, None]).astype(np.int64).ravel()
        iy = np.floor(y[:, None] + t * ny[:, None]).astype(np.int64).ravel()
        c = np.unique(np.stack([ix, iy], 1), axis=0)
        c = np.unique((c[:, None, :] + np.array([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)])[None]).reshape(-1, 2), axis=0)
        out.append(np.column_stack([np.full(len(c), k), c]))
    return np.concatenate(out)


def claim(P, k, X, Y, xp):
    """Which trench piece owns each point; returns (inside, foot x, foot y)."""
    g = {n: xp.asarray(P[n])[k] for n in P if n != "L"}
    hw = g["w"] / 2
    ins, fx, fy = xp.zeros(X.shape, bool), xp.zeros_like(X), xp.zeros_like(X)
    for (px, py, hh, LL) in (("ax", "ay", "h", "L1"), ("dx", "dy", "h2", "L2")):
        ux, uy = xp.cos(g[hh]), xp.sin(g[hh])
        rx, ry = X - g[px], Y - g[py]
        s, t = rx * ux + ry * uy, ux * ry - uy * rx
        m = ~ins & (s >= 0) & (s <= g[LL]) & (xp.abs(t) <= hw)
        fx, fy = xp.where(m, g[px] + s * ux, fx), xp.where(m, g[py] + s * uy, fy)
        ins |= m
    vx, vy = X - g["cx"], Y - g["cy"]
    b0x, b0y = g["bx"] - g["cx"], g["by"] - g["cy"]
    phi = xp.arctan2(b0x * vy - b0y * vx, b0x * vx + b0y * vy) * g["sg"]
    r = xp.sqrt(vx * vx + vy * vy); rs = xp.where(r > 0, r, 1.0)
    m = ~ins & (g["sg"] != 0) & (phi >= 0) & (phi <= g["th"]) & (xp.abs(g["sg"] * (g["R"] - r)) <= hw)
    fx = xp.where(m, g["cx"] + vx / rs * g["R"], fx); fy = xp.where(m, g["cy"] + vy / rs * g["R"], fy)
    return ins | m, fx, fy


def electron(P, G, ks):
    """Prism sum on the world grid, per trench in ks, batched to BUDGET points."""
    xp = G.xp
    m = int(round(1 / CELL)) * SUB
    q = xp.asarray((np.arange(m) + 0.5) / m)
    QX, QY = xp.meshgrid(q, q); QX, QY = QX.ravel(), QY.ravel()
    vol = np.zeros(len(ks)); pts = 0
    cand = candidates(P, ks)
    per = max(1, BUDGET // (m * m))
    for a in range(0, len(cand), per):
        c = xp.asarray(cand[a:a + per])
        k = xp.repeat(c[:, 0], m * m)
        X = (c[:, 1, None] + QX[None]).ravel(); Y = (c[:, 2, None] + QY[None]).ravel()
        ins, fx, fy = claim(P, k, X, Y, xp)
        cut = xp.where(ins, xp.maximum(G.h(X, Y) - G.h(fx, fy) + xp.asarray(P["d"])[k], 0), 0)
        idx = xp.asarray(np.searchsorted(ks, to_host(c[:, 0])))
        vol += to_host(xp.bincount(xp.repeat(idx, m * m), weights=cut, minlength=len(ks)))
        pts += int(X.size)
    return vol / (m * m), pts


def trench_pair(P, G):
    ks = np.arange(len(P["L"]))
    e, pts = electron(P, G, ks)
    p = positron(P, G, ks)
    return e, p, np.abs(e - p) / np.maximum(np.maximum(e, p), 1e-12), pts


# ---------------------------------------------------------------- platform
def inside_poly(X, Y, vx, vy, xp):
    ins = xp.zeros(X.shape, bool)
    for i in range(len(vx)):
        x1, y1, x2, y2 = vx[i], vy[i], vx[i - 1], vy[i - 1]
        if y1 == y2:
            continue
        cross = ((y1 > Y) != (y2 > Y)) & (X < x1 + (Y - y1) * (x2 - x1) / (y2 - y1))
        ins ^= cross
    return ins


def platform_pair(poly_bng, level, G):
    """(electron cut, fill), (positron cut, fill) for a polygon (BNG) and target level."""
    xp = G.xp
    v = np.asarray(poly_bng, np.float64) - [G.oe, G.on]
    vx, vy = v[:, 0], v[:, 1]
    m = int(round(1 / CELL)) * SUB; step = CELL / SUB
    x0, y0 = math.floor(vx.min()), math.floor(vy.min())
    nx, ny = int(math.ceil((vx.max() - x0) / step)), int(math.ceil((vy.max() - y0) / step))
    ec = ef = 0.0
    for r0 in range(0, ny, max(1, BUDGET // nx)):
        yy = y0 + (xp.arange(r0, min(ny, r0 + BUDGET // nx)) + 0.5) * step
        X, Y = xp.meshgrid(x0 + (xp.arange(nx) + 0.5) * step, yy)
        dz = xp.where(inside_poly(X, Y, vx, vy, xp), G.h(X, Y) - level, 0)
        ec += float(xp.maximum(dz, 0).sum()); ef += float(xp.maximum(-dz, 0).sum())
    ec, ef = ec * step * step, ef * step * step
    # positron: chainage along the first edge's direction, sections across it
    ux, uy = vx[1] - vx[0], vy[1] - vy[0]; n = math.hypot(ux, uy); ux, uy = ux / n, uy / n
    s_all, t_all = (vx - vx[0]) * ux + (vy - vy[0]) * uy, (vy - vy[0]) * ux - (vx - vx[0]) * uy
    smin, smax = s_all.min(), s_all.max()
    ns = int(math.ceil((smax - smin) / DS)) + 1
    s = xp.linspace(smin, smax, ns); h = (smax - smin) / (ns - 1)
    s[0] += 1e-9 * h; s[-1] -= 1e-9 * h       # end sections just inside, so they cut the polygon
    E = len(vx); c = xp.full((ns, E), xp.inf)
    for i in range(E):
        s1, t1, s2, t2 = s_all[i], t_all[i], s_all[i - 1], t_all[i - 1]
        if s1 == s2:
            continue
        hit = (s1 > s) != (s2 > s)
        c[:, i] = xp.where(hit, t1 + (s - s1) * (t2 - t1) / (s2 - s1), xp.inf)
    c = xp.sort(c, 1)
    a, b = c[:, 0:E - 1:2], c[:, 1:E:2]
    ok = xp.isfinite(a) & xp.isfinite(b)
    a, b = xp.where(ok, a, 0), xp.where(ok, b, 0)
    tau = xp.linspace(0, 1, 4 * NT + 1); wt = xp.full(tau.size, 1.0 / (tau.size - 1)); wt[0] = wt[-1] = wt[0] / 2
    t = a[..., None] + (b - a)[..., None] * tau
    X = vx[0] + s[:, None, None] * ux - t * uy; Y = vy[0] + s[:, None, None] * uy + t * ux
    dz = G.h(X, Y) - level
    L = (b - a)[..., None] * wt
    ws = xp.full(ns, h); ws[0] = ws[-1] = h / 2
    pc = float(((xp.maximum(dz, 0) * L).sum((1, 2)) * ws).sum())
    pf = float(((xp.maximum(-dz, 0) * L).sum((1, 2)) * ws).sum())
    return (ec, ef), (pc, pf)


# ---------------------------------------------------------------- sweep
def draw(rng, n, G, margin=5.0):
    """n seeded routes wholly inside the site: half straight, half bent."""
    W = G.sp * (G.nx - 1); H = G.sp * (G.ny - 1); rows = []
    while len(rows) < n:
        w = rng.choice([0.3, 0.45, 0.6, 0.9, 1.2]); d = rng.uniform(0.6, 2.0)
        bent = rng.random() < 0.5
        th = rng.uniform(np.radians(15), np.radians(120)) * rng.choice([-1, 1]) if bent else 0.0
        R = rng.uniform(max(8.0, 5 * w), 40.0) if bent else 0.0
        row = (G.oe + rng.uniform(0, W), G.on + rng.uniform(0, H), rng.uniform(0, 2 * np.pi),
               rng.uniform(5, 80), th, R, rng.uniform(5, 80) if bent else 0.0, w, d)
        P = routes(*row, oe=G.oe, on=G.on)
        s = np.linspace(0, P["L"][0], 64)
        x, y, *_ = centre(P, np.zeros(64, np.int64), s, np)
        if x.min() > margin and y.min() > margin and x.max() < W - margin and y.max() < H - margin:
            rows.append(row)
    return routes(*np.array(rows).T, oe=G.oe, on=G.on)


def sweep(tiles_dir, seed=SEED, trenches=4000, seconds=20.0, batch=250, use_gpu=True, witness=3):
    t0 = time.perf_counter()
    xp = cp if (use_gpu and cp is not None) else np
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not np else "cpu (numpy)"
    G = ground_from(tiles_dir, xp)
    rng = np.random.default_rng(seed)
    E, Pv, D, meta, pts, stopped, first = [], [], [], [], 0, "trenches", None
    while len(E) < trenches:
        if E and time.perf_counter() - t0 > seconds:
            stopped = "seconds"; break
        P = draw(rng, min(batch, trenches - len(E)), G)
        first = first or {k: v[:witness].copy() for k, v in P.items()}
        e, p, d, n = trench_pair(P, G)
        E += list(e); Pv += list(p); D += list(d); pts += n
        meta += [dict(start_e=round(P["ax"][i] + G.oe, 3), start_n=round(P["ay"][i] + G.on, 3),
                      heading_deg=round(math.degrees(P["h"][i]) % 360, 3), length_m=round(P["L"][i], 3), l1_m=round(P["L1"][i], 3), l2_m=round(P["L2"][i], 3),
                      bend_deg=round(math.degrees(P["th"][i] * P["sg"][i]), 3), radius_m=round(P["R"][i], 3),
                      width_m=P["w"][i], depth_m=round(P["d"][i], 4)) for i in range(len(e))]
        if xp is not np:
            cp.cuda.Stream.null.synchronize()
    E, Pv, D = map(np.array, (E, Pv, D))
    order = np.argsort(-D)[:KEEP]
    worst = [dict(meta[i], electron_m3=round(E[i], 4), positron_m3=round(Pv[i], 4), rel=float(D[i])) for i in order]
    # the demo platform: a 40 x 60 m pad at the site's middle, rotated 30 degrees, at its mean ground
    cx, cy = G.oe + G.sp * (G.nx - 1) / 2, G.on + G.sp * (G.ny - 1) / 2
    a = math.radians(30); ca, sa = math.cos(a), math.sin(a)
    poly = [(cx + ca * u - sa * v, cy + sa * u + ca * v) for u, v in ((-20, -30), (20, -30), (20, 30), (-20, 30))]
    level = float(G.h(xp.asarray([cx - G.oe]), xp.asarray([cy - G.on]))[0])
    (pe_c, pe_f), (pp_c, pp_f) = platform_pair(poly, level, G)
    rel = lambda a, b: abs(a - b) / max(a, b, 1e-12)
    plat = dict(polygon_bng=[[round(x, 3), round(y, 3)] for x, y in poly], level_m=round(level, 4),
                electron_cut_m3=pe_c, electron_fill_m3=pe_f, positron_cut_m3=pp_c, positron_fill_m3=pp_f,
                rel_cut=rel(pe_c, pp_c), rel_fill=rel(pe_f, pp_f))
    plat["photons"] = int(plat["rel_cut"] > TOL) + int(plat["rel_fill"] > TOL)
    # CPU witness: the first trenches and the platform, NumPy only
    Gc = ground_from(tiles_dir, np)
    we, wp, _, _ = trench_pair(first, Gc)
    (wc, wf), _ = platform_pair(poly, level, Gc)
    wd = np.concatenate([np.abs(we - E[:witness]) / E[:witness], np.abs(wp - Pv[:witness]) / Pv[:witness],
                         [rel(wc, pe_c), rel(wf, pe_f)]])
    wit = dict(trenches=int(witness), platform=1, photons=int((wd > WIT_TOL).sum()), max_rel=float(wd.max()))
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [0, 5, 25, 50, 75, 95, 100])]
    bent = np.array([m["bend_deg"] != 0 for m in meta])
    photons = int((D > TOL).sum())
    return {
        "kind": "earthworks trench pair", "seed": seed, "device": device,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "source": {"shape": [G.ny, G.nx], "origin_e_m": G.oe, "origin_n_m": G.on, "spacing_m": G.sp},
        "method": {"electron": f"prisms, {CELL} m cells x {SUB}x{SUB} points", "tolerance_rel": TOL,
                   "positron": f"sections every <= {DS} m, {NT} offsets, trapezoid, (1 - kappa t) weight"},
        "counts": {"trenches": len(E), "straight": int((~bent).sum()), "bent": int(bent.sum()),
                   "electron_points": pts, "stopped_by": stopped},
        "spoil_m3": {"total_electron": float(E.sum()), "total_positron": float(Pv.sum()),
                     "percentiles_0_5_25_50_75_95_100": q(E),
                     "per_metre_percentiles": q(E / np.array([m["length_m"] for m in meta]))},
        "disagreement": {"photons": photons, "max_rel": float(D.max()), "median_rel": float(np.median(D)),
                         "worst": worst},
        "platform": plat, "witness": wit, "photons_total": photons + plat["photons"] + wit["photons"],
        "bounds": {"trenches": trenches, "seconds": seconds, "batch": batch},
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--trenches", type=int, default=4000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    r = sweep(a.tiles, a.seed, a.trenches, a.seconds, use_gpu=not a.cpu)
    body = json.dumps(r, indent=1, sort_keys=True).encode()
    path = os.path.join(a.tiles, RECEIPT)
    open(path, "wb").write(body)
    c, s, dg, p = r["counts"], r["spoil_m3"], r["disagreement"], r["platform"]
    print(f"{c['trenches']} trenches ({c['straight']} straight, {c['bent']} bent) on {r['device']} in {r['wall_s']} s")
    print(f"spoil m3 total {s['total_electron']:.1f}; min/p5/p25/p50/p75/p95/max {s['percentiles_0_5_25_50_75_95_100']}")
    print(f"photons {dg['photons']} (> {TOL:.1%}), max rel {dg['max_rel']:.2e}, median {dg['median_rel']:.2e}")
    print(f"platform cut {p['electron_cut_m3']:.1f}/{p['positron_cut_m3']:.1f} fill {p['electron_fill_m3']:.1f}/"
          f"{p['positron_fill_m3']:.1f} m3; witness photons {r['witness']['photons']} (max {r['witness']['max_rel']:.1e})")
    print(f"receipt {path} sha256 {hashlib.sha256(body).hexdigest()}")
    return 0 if r["photons_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
